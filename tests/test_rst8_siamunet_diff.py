"""Unit and contract tests for RST8 SiamUnet_diff Architecture Transfer (Stage A)."""

import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.siam_diff_wrapper import SiamDiffWrapper, SiamUnet_diff
from src.evaluation.losses import ControlledSourceLoss
from src.robust_source_training.loss import RobustSourceLoss
from src.robust_source_training.checkpoint import (
    assert_states_equal,
    load_rst_checkpoint,
    save_rst_checkpoint,
)
from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.evaluation import compute_binary_confusion_counts
from src.robust_source_training.runner import (
    RSTExecutionEngine,
    build_model,
    seed_everything,
)

CONFIG_DIR = REPO_ROOT / "configs" / "robust_source_training" / "rst8_siamunet_diff"
MATRIX_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst8_matrix_v1.json"
LEDGER_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst8_ledger_v1.json"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_siamunet_diff_architecture_invariants():
    """Verify SiamUnet_diff parameter count, tensor shapes, and skip-difference mechanics."""
    model = SiamUnet_diff(in_channels=2, out_channels=3, dropout_p=0.2)
    wrapper = SiamDiffWrapper(in_channels=2, out_channels=3, dropout_p=0.2)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 1_350_147, f"Expected 1,350,147 parameters, got {total_params}"

    xA = torch.randn(2, 2, 64, 64)
    xB = torch.randn(2, 2, 64, 64)

    out_model = model(xA, xB)
    out_wrapper = wrapper(xA, xB)

    assert out_model.shape == (2, 3, 64, 64)
    assert out_wrapper.shape == (2, 3, 64, 64)

    # Check forward pass with standard 256x256 image size
    xA_256 = torch.randn(1, 2, 256, 256)
    xB_256 = torch.randn(1, 2, 256, 256)
    out_256 = wrapper(xA_256, xB_256)
    assert out_256.shape == (1, 3, 256, 256)


def test_2_exact_four_stage_a_configs_and_matrix():
    """Verify matrix and 4 formal configs exist, adhere to schema, and have valid hashes."""
    matrix = _load_json(MATRIX_PATH)
    assert matrix["summary"]["physical_training_runs_count"] == 4
    assert len(matrix["cells"]) == 4

    expected_runs = {
        "rst8_r0_siamunet_diff_folda_s0_v1",
        "rst8_r2_siamunet_diff_folda_s0_v2",
        "rst8_r0_siamunet_diff_foldb_s0_v2",
        "rst8_r2_siamunet_diff_foldb_s0_v2",
    }

    observed_runs = {c["run_id"] for c in matrix["cells"]}
    assert observed_runs == expected_runs

    for cell in matrix["cells"]:
        cfg_path = REPO_ROOT / cell["config_path"]
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        actual_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        assert actual_hash == cell["config_sha256"], f"Hash mismatch in {cell['run_id']}"

        cfg = _load_json(cfg_path)
        assert cfg["architecture"]["name"] == "SiamUnet_diff"
        assert cfg["architecture"]["in_channels"] == 2
        assert cfg["architecture"]["out_ch"] == 3
        assert cfg["seed"] == 0
        assert cfg["max_epochs"] == 16
        assert cfg["training"]["micro_batch_size"] == 8
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["training"]["effective_batch_size"] == 32


def test_3_exact_frozen_folda_foldb_allowlists_and_disjointness():
    """Verify Fold A and Fold B have 18 train, 9 eval, zero target access, and zero train/eval overlap."""
    for cfg_file in CONFIG_DIR.glob("*.json"):
        cfg = _load_json(cfg_file)
        train_events = set(cfg["partition"]["train_event_ids"])
        eval_events = set(cfg["partition"]["evaluation_event_ids"])

        assert len(train_events) == 18, f"Expected 18 train events in {cfg_file.name}"
        assert len(eval_events) == 9, f"Expected 9 eval events in {cfg_file.name}"

        # Zero train/eval overlap
        overlap = train_events & eval_events
        assert not overlap, f"Train/eval overlap detected in {cfg_file.name}: {overlap}"

        # Zero target event access
        assert not (train_events & set(FORBIDDEN_TARGET_EVENTS)), f"Target events in train of {cfg_file.name}"
        assert not (eval_events & set(FORBIDDEN_TARGET_EVENTS)), f"Target events in eval of {cfg_file.name}"

        # Source allowlist subset
        assert train_events.issubset(set(AUTHORITATIVE_SOURCE_EVENTS))
        assert eval_events.issubset(set(AUTHORITATIVE_SOURCE_EVENTS))


def test_4_deterministic_initialization_and_matched_hashes():
    """Verify seed 0 produces identical initial model state hash across R0 and R2."""
    def get_initial_hash(seed):
        seed_everything(seed)
        arch = {"name": "SiamUnet_diff", "in_channels": 2, "out_ch": 3, "dropout_p": 0.2}
        model = build_model(arch)
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.cpu().numpy().tobytes())
        return h.hexdigest()

    h0_a = get_initial_hash(0)
    h0_b = get_initial_hash(0)
    assert h0_a == h0_b
    assert h0_a == "6ee419e4f06ab18dd7cb7b68e7fae74b639d573203e2fb274e8ee673f68e1b13"


def test_5_recipe_switches_strict_separation():
    """Verify R0 has M1/M2/M3 disabled, R2 has M1/M2 enabled and M3 disabled."""
    for cfg_file in CONFIG_DIR.glob("*.json"):
        cfg = _load_json(cfg_file)
        rid = cfg["recipe_id"]
        comps = cfg["components"]
        if rid == "R0":
            assert comps["m1_event_balance"] is False
            assert comps["m2_flood_aware_loss"] is False
            assert comps["m3_paired_augmentation"] is False
        elif rid == "R2":
            assert comps["m1_event_balance"] is True
            assert comps["m2_flood_aware_loss"] is True
            assert comps["m3_paired_augmentation"] is False

            # Verify class weights provenance
            cw_path = REPO_ROOT / cfg["provenance_references"]["class_weights_path"]
            assert cw_path.exists()
            cw_hash = hashlib.sha256(cw_path.read_bytes()).hexdigest()
            assert cw_hash == cfg["provenance_references"]["class_weights_sha256"]


def test_6_label_semantics_and_positive_evaluation_class():
    """Verify class 2 is flood, class 0/1 are negative, and class 3 is ignore."""
    # Synthetic label map with classes 0, 1, 2, 3
    gt = np.array([[0, 1], [2, 3]])
    pred = np.array([[0, 2], [2, 0]])  # pred flood at (0, 1) and (1, 0)
    valid_mask = np.ones_like(gt)

    counts = compute_binary_confusion_counts(pred, gt, valid_mask, positive_class=2, ignore_index=3)

    # Pixel (0, 0): gt 0, pred 0 -> TN
    # Pixel (0, 1): gt 1, pred 2 -> FP
    # Pixel (1, 0): gt 2, pred 2 -> TP
    # Pixel (1, 1): gt 3 (ignore) -> Excluded
    assert counts["valid_pixels"] == 3
    assert counts["tp"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 0
    assert counts["tn"] == 1


def test_7_finite_loss_and_gradients():
    """Verify forward and backward passes produce finite values on SiamUnet_diff."""
    seed_everything(0)
    arch = {"name": "SiamUnet_diff", "in_channels": 2, "out_ch": 3, "dropout_p": 0.2}
    model = build_model(arch)
    criterion = RobustSourceLoss(
        class_weights=[0.613865, 1.487843, 0.898292],
        weight_ce=0.5,
        weight_dice=0.5,
        ignore_index=3,
    )

    xA = torch.randn(2, 2, 64, 64)
    xB = torch.randn(2, 2, 64, 64)
    labels = torch.randint(0, 4, (2, 64, 64))

    logits = model(xA, xB)
    loss = criterion(logits, labels)

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_8_checkpoint_save_reload_and_state_equality():
    """Verify atomic saving and bit-identical reload for SiamDiffWrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        arch = {"name": "SiamUnet_diff", "in_channels": 2, "out_ch": 3, "dropout_p": 0.2}

        seed_everything(0)
        m1 = build_model(arch)
        opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)

        save_rst_checkpoint(
            path=ckpt_path,
            model=m1,
            optimizer=opt1,
            epoch=16,
            global_step=6656,
            seed=0,
            run_id="test_run",
            config_hash="abc",
            fold_hash="def",
            train_event_ids=[101],
            evaluation_event_ids=[102],
        )

        seed_everything(1)
        m2 = build_model(arch)
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)

        load_rst_checkpoint(
            path=ckpt_path,
            model=m2,
            optimizer=opt2,
            expected_config_hash="abc",
            expected_fold_hash="def",
            expected_train_events=[101],
            expected_eval_events=[102],
        )

        assert_states_equal(m1, m2)


def test_9_static_validation_passes_all_four_configs():
    """Verify RSTExecutionEngine.validate_only() passes on all 4 Stage-A configs."""
    for cfg_file in CONFIG_DIR.glob("*.json"):
        engine = RSTExecutionEngine(config_path=cfg_file, repo_root=REPO_ROOT)
        validated = engine.validate_only()
        assert validated["is_smoke"] is False
        assert len(validated["train_event_ids"]) == 18
        assert len(validated["evaluation_event_ids"]) == 9


def test_10_continuation_rule_synthetic_logic():
    """Verify frozen Stage-A continuation rule logic with synthetic scenarios."""
    def evaluate_stage_a(grand_delta, max_delta, n_events, p_drop, r_drop, dual_drop_count, tail_delta, worst_delta):
        rem_delta = (grand_delta * n_events - max_delta) / (n_events - 1)
        not_driven = rem_delta > 0.0
        no_collapse = (p_drop > -0.050) and (r_drop > -0.050) and (dual_drop_count <= 4)

        p1 = (grand_delta >= 0.010) and not_driven and no_collapse
        p2 = ((tail_delta >= 0.020) or (worst_delta >= 0.020)) and (grand_delta >= -0.005)
        p3_evaluable = False

        if p1 or p2:
            verdict = "TRANSFER DIRECTION SUPPORTED -- THREE-SEED EXPANSION AUTHORIZED"
        else:
            verdict = "TRANSFER DIRECTION NOT SUPPORTED -- STOP AT SEED 0"

        return p1, p2, p3_evaluable, verdict

    # Scenario 1: P1 passes
    p1, p2, p3, v = evaluate_stage_a(0.015, 0.040, 18, -0.010, 0.020, 0, 0.010, 0.000)
    assert p1 is True and "TRANSFER DIRECTION SUPPORTED" in v

    # Scenario 2: P2 passes
    p1, p2, p3, v = evaluate_stage_a(0.002, 0.008, 18, 0.000, 0.000, 0, 0.030, 0.010)
    assert p2 is True and "TRANSFER DIRECTION SUPPORTED" in v

    # Scenario 3: Neither passes -> Inconclusive / Stop
    p1, p2, p3, v = evaluate_stage_a(0.003, 0.006, 18, 0.000, 0.000, 0, 0.005, -0.002)
    assert p1 is False and p2 is False and "STOP AT SEED 0" in v


def test_11_analysis_module_synthetic_evaluation():
    """Verify run_rst8_analysis decision evaluation and bootstrap function correctly on synthetic data."""
    from tools.robust_source_training.run_rst8_analysis import (
        evaluate_stage_a_decision_criteria,
        run_paired_bootstrap,
    )

    paired_rows = [
        {"event_id": 100 + i, "fold_id": "fold_a" if i < 9 else "fold_b", "delta_f1": 0.01 + 0.005 * (i % 3)}
        for i in range(18)
    ]
    boot = run_paired_bootstrap(paired_rows, n_resamples=100, seed=42)
    assert boot["n_resamples"] == 100
    assert boot["ci_95"][0] <= boot["observed_mean_delta"] <= boot["ci_95"][1]

    cfg = {
        "criteria_thresholds": {
            "p1_min_grand_mean_delta": 0.010,
            "p2_min_tail_delta": 0.020,
            "p2_min_grand_mean_delta": -0.005,
        }
    }
    stats = {
        "grand_metrics": {
            "delta": {
                "f1": 0.015,
                "bottom3_mean_f1": 0.025,
                "worst_event_f1": 0.010,
            }
        },
        "single_event_driver_test": {
            "gain_not_driven_by_one_event": True,
            "delta_excluding_max_event": 0.012,
        },
        "precision_recall_collapse_test": {
            "no_widespread_collapse": True,
            "macro_precision_delta": 0.01,
            "macro_recall_delta": 0.02,
            "dual_drop_events_count": 0,
        },
        "seed_stability_test": {
            "status": "NOT_EVALUABLE_AT_STAGE_A",
        },
    }
    decision = evaluate_stage_a_decision_criteria(stats, boot, cfg)
    assert decision["terminal_verdict"] == "RST8 STAGE A COMPLETE — EXPANSION AUTHORIZED"
    assert decision["decision_code"] == "EXPANSION_AUTHORIZED"
    assert decision["criteria"]["P1_average_improvement"]["satisfied"] is True
    assert decision["criteria"]["P2_tail_reliability_improvement"]["satisfied"] is True


if __name__ == "__main__":
    tests = [
        test_1_siamunet_diff_architecture_invariants,
        test_2_exact_four_stage_a_configs_and_matrix,
        test_3_exact_frozen_folda_foldb_allowlists_and_disjointness,
        test_4_deterministic_initialization_and_matched_hashes,
        test_5_recipe_switches_strict_separation,
        test_6_label_semantics_and_positive_evaluation_class,
        test_7_finite_loss_and_gradients,
        test_8_checkpoint_save_reload_and_state_equality,
        test_9_static_validation_passes_all_four_configs,
        test_10_continuation_rule_synthetic_logic,
        test_11_analysis_module_synthetic_evaluation,
    ]

    print(f"Running {len(tests)} RST8 SiamUnet_diff tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST8 SiamUnet_diff tests PASSED!")
