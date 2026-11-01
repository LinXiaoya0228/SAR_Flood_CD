"""Unit and contract tests for RST9 Final Deployable Source Model Training and Historical-Target Extension."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Safe numpy alias for older checkpoints if necessary
try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    sys.modules.setdefault("numpy._core.multiarray", _npc.multiarray)
except Exception:
    pass

from src.models.snunet_wrapper import SNUNetWrapper
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

CONFIG_DIR = REPO_ROOT / "configs" / "robust_source_training" / "rst9_all27"
MATRIX_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst9_matrix_v1.json"
LEDGER_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst9_ledger_v1.json"
CLASS_WEIGHTS_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst9_all27_class_weights_v1.json"

HISTORICAL_DEV_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
HISTORICAL_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
ALL_HISTORICAL_TARGET_EVENTS = HISTORICAL_DEV_EVENTS | HISTORICAL_TEST_EVENTS


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_snunet_architecture_invariants():
    """Verify SNUNet-ECAM parameter count and output dimensions."""
    model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 12034819, f"Expected 12,034,819 parameters, got {param_count}"

    x1 = torch.randn(2, 2, 64, 64)
    x2 = torch.randn(2, 2, 64, 64)
    out = model(x1, x2)
    assert out.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {out.shape}"


def test_2_exact_three_configs_and_matrix():
    """Verify exactly 3 training configs exist, match the matrix, and have exact sha256 sums."""
    matrix = _load_json(MATRIX_PATH)
    assert matrix["total_planned_training_runs"] == 3
    assert len(matrix["cells"]) == 3
    assert set(matrix["historical_target_events"]["dev_events"]) == HISTORICAL_DEV_EVENTS
    assert set(matrix["historical_target_events"]["test_events"]) == HISTORICAL_TEST_EVENTS

    for cell in matrix["cells"]:
        cfg_path = REPO_ROOT / cell["config_path"]
        assert cfg_path.exists(), f"Config does not exist: {cfg_path}"
        actual_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        assert actual_sha == cell["config_sha256"], f"SHA mismatch for {cfg_path}"

        cfg = _load_json(cfg_path)
        assert cfg["execution_class"] == "ALL_SOURCE_TRAINING"
        assert cfg["scientific_result"] is True
        assert cfg["launch_authorized"] is True
        assert cfg["recipe_id"] == "R2"
        assert cfg["components"] == {
            "m1_event_balance": True,
            "m2_flood_aware_loss": True,
            "m3_paired_augmentation": False,
        }
        assert cfg["max_epochs"] == 16
        assert cfg["architecture"]["name"] == "SNUNet_ECAM"
        assert cfg["training"]["effective_batch_size"] == 32


def test_3_exact_27_source_events_and_zero_target_leakage():
    """Verify training contains all 27 source events, 0 eval events, and zero target event leakage."""
    for s in (0, 1, 2):
        cfg_path = CONFIG_DIR / f"rst9_r2_snunet_all27_s{s}_v1.json"
        cfg = _load_json(cfg_path)

        train_events = set(cfg["partition"]["train_event_ids"])
        eval_events = set(cfg["partition"]["evaluation_event_ids"])

        assert train_events == AUTHORITATIVE_SOURCE_EVENTS, "Train events do not match 27 source events"
        assert len(eval_events) == 0, "Evaluation events must be empty during all-source training"

        # Check target isolation
        leakage = train_events & ALL_HISTORICAL_TARGET_EVENTS
        assert len(leakage) == 0, f"CRITICAL LEAKAGE: Target events in training set: {leakage}"


def test_4_class_weights_artifact_integrity():
    """Verify M2 frozen class weights artifact exact values and hash."""
    assert CLASS_WEIGHTS_PATH.exists()
    actual_sha = hashlib.sha256(CLASS_WEIGHTS_PATH.read_bytes()).hexdigest()
    expected_sha = "0a1f38bb285c40b91ce15d8fe2c024523ce9b640fd2351ef79da514a184bda85"
    assert actual_sha == expected_sha, f"SHA mismatch: {actual_sha} vs {expected_sha}"

    cw_data = _load_json(CLASS_WEIGHTS_PATH)
    assert cw_data["total_samples_counted"] == 21462
    assert set(cw_data["training_event_ids"]) == AUTHORITATIVE_SOURCE_EVENTS
    assert cw_data["aggregate_counts"]["total_valid"] == 1062845830
    assert cw_data["final_weights_list"] == [0.496316, 1.54455, 0.959134]


def test_5_deterministic_initialization_across_seeds():
    """Verify model initialization is deterministic per seed and distinct across seeds."""
    models = {}
    for s in (0, 1, 2):
        seed_everything(s)
        m = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        h = hashlib.sha256()
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.cpu().numpy().tobytes())
        models[s] = h.hexdigest()

    # Re-running seed 0 must match
    seed_everything(0)
    m0_repeat = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    h0_rep = hashlib.sha256()
    for k, v in sorted(m0_repeat.state_dict().items()):
        h0_rep.update(k.encode("utf-8"))
        h0_rep.update(v.cpu().numpy().tobytes())
    assert models[0] == h0_rep.hexdigest(), "Seed 0 initialization not deterministic!"
    assert models[0] != models[1], "Seed 0 and Seed 1 should have distinct initial states!"
    assert models[1] != models[2], "Seed 1 and Seed 2 should have distinct initial states!"


def test_6_static_validation_passes_all_three_configs():
    """Verify RSTExecutionEngine.validate_only() passes on all 3 all-27 configs."""
    for s in (0, 1, 2):
        cfg_path = CONFIG_DIR / f"rst9_r2_snunet_all27_s{s}_v1.json"
        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        val = engine.validate_only()
        assert val["is_valid"] is True
        assert val["train_event_count"] == 27
        assert val["evaluation_event_count"] == 0


def test_7_label_semantics_and_confusion_counts():
    """Verify evaluation positive class is Class 2 (inundated flood water)."""
    # 3 classes: 0=dry, 1=perm water, 2=flood, 3=ignore
    preds = np.array([[0, 1, 2], [2, 0, 1]])
    labels = np.array([[0, 1, 2], [0, 2, 1]])
    masks = np.ones((2, 3), dtype=np.uint8)

    counts = compute_binary_confusion_counts(preds, labels, masks)
    # Class 2 is positive:
    # (0, 2): pred=2, label=2 -> TP (1)
    # (1, 0): pred=2, label=0 -> FP (1)
    # (1, 1): pred=0, label=2 -> FN (1)
    # (0, 0): pred=0, label=0 -> TN (1)
    # (0, 1): pred=1, label=1 -> TN (1)
    # (1, 2): pred=1, label=1 -> TN (1)
    assert counts["tp"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 1
    assert counts["tn"] == 3


def test_8_historical_r0_checkpoints_reloadable():
    """Verify that all 3 historical R0 all-27 checkpoints can be loaded cleanly."""
    matrix = _load_json(MATRIX_PATH)
    for b in matrix["matched_r0_baselines"]:
        ckpt_path = REPO_ROOT / b["checkpoint_path"]
        assert ckpt_path.exists(), f"Baseline checkpoint missing: {ckpt_path}"
        actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert actual_sha == b["checkpoint_sha256"], f"SHA mismatch for baseline {ckpt_path}"

        # Load weights into fresh model
        model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw))
        model.load_state_dict(sd)

        # Check all params are finite
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite weights in {name}"


def test_9_completed_r2_checkpoints_and_step_reconciliation():
    """Verify that all 3 completed R2 all-27 checkpoints reload cleanly and have 10,736 steps."""
    ledger = _load_json(LEDGER_PATH)
    assert ledger["completed_training_runs_count"] == 3
    assert ledger["reconciled_step_budget"]["expected_total_optimizer_steps"] == 10736

    for run_entry in ledger["runs"]:
        run_id = run_entry["run_id"]
        out_root = REPO_ROOT / run_entry["output_root"]

        # Check terminal status
        term_status_file = out_root / "terminal_status.json"
        assert term_status_file.exists(), f"Terminal status missing for {run_id}"
        term_status = _load_json(term_status_file)
        assert term_status["status"] == "COMPLETED_SCIENTIFIC_RUN"
        assert term_status["exit_code"] == 0

        # Check final manifest and steps
        manifest_file = out_root / "final_manifest.json"
        assert manifest_file.exists(), f"Final manifest missing for {run_id}"
        manifest = _load_json(manifest_file)
        obs = manifest["observed_execution"]
        assert obs["optimizer_steps"] == 10736, f"Expected 10,736 steps, got {obs['optimizer_steps']}"
        assert len(obs["m1_summary"]["epoch_summaries"]) == 16, f"Expected 16 epochs"

        # Check checkpoint file and SHA-256
        ckpt_path = REPO_ROOT / run_entry["checkpoint_path"]
        assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
        actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert actual_sha == run_entry["checkpoint_sha256"], f"SHA mismatch for {run_id}"

        # Reload weights into model and verify finiteness
        model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw))
        model.load_state_dict(sd)
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite weights in {name} for {run_id}"


def test_10_target_inference_accounting_and_analysis():
    """Verify that historical target inference produced 48 R0 rows, 48 R2 rows, and valid analysis."""
    master_csv = REPO_ROOT / "outputs" / "robust_source_training" / "rst9_inference" / "all_historical_target_metrics.csv"
    assert master_csv.exists(), f"Master CSV missing: {master_csv}"

    with open(master_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 1 header + 48 R0 + 48 R2 = 97 lines
    assert len(lines) == 97, f"Expected 97 lines in master CSV, got {len(lines)}"

    # Check analysis JSON
    analysis_file = REPO_ROOT / "outputs" / "robust_source_training" / "rst9_analysis" / "rst9_comparative_analysis_v1.json"
    assert analysis_file.exists(), f"Analysis JSON missing: {analysis_file}"
    analysis = _load_json(analysis_file)

    assert analysis["decision_code"] == "POSITIVE_HISTORICAL_EXTENSION"
    assert analysis["criteria"]["P1_grand_macro_improvement"]["satisfied"] is True
    assert analysis["criteria"]["P2_tail_reliability_improvement"]["satisfied"] is True
    assert analysis["criteria"]["P3_seed_stability"]["satisfied"] is True
    assert analysis["bootstrap_statistics"]["ci_crosses_zero"] is False


if __name__ == "__main__":
    tests = [
        test_1_snunet_architecture_invariants,
        test_2_exact_three_configs_and_matrix,
        test_3_exact_27_source_events_and_zero_target_leakage,
        test_4_class_weights_artifact_integrity,
        test_5_deterministic_initialization_across_seeds,
        test_6_static_validation_passes_all_three_configs,
        test_7_label_semantics_and_confusion_counts,
        test_8_historical_r0_checkpoints_reloadable,
        test_9_completed_r2_checkpoints_and_step_reconciliation,
        test_10_target_inference_accounting_and_analysis,
    ]

    print(f"Running {len(tests)} RST9 All-27 tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST9 All-27 tests PASSED!")
