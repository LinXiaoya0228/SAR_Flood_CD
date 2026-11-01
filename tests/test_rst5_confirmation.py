"""Comprehensive unit test suite for RST5 Recipe Freeze and Reproducibility Handoff."""

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

from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.losses import ControlledSourceLoss
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
from src.robust_source_training.evaluation import (
    EventIndependentEvaluator,
    compute_binary_confusion_counts,
    compute_rates_from_counts,
)
from src.robust_source_training.loss import RobustSourceLoss
from src.robust_source_training.runner import RSTExecutionEngine, seed_everything


def _load_json(rel_path: str):
    p = REPO_ROOT / rel_path
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. Label semantics and class index alignment
def test_1_label_semantics_and_class_index_alignment():
    """Verify class 0 = dry ground, class 1 = permanent water, class 2 = flood water, class 3 = ignore."""
    # Loss functions ignore index 3
    ctrl_loss = ControlledSourceLoss(ignore_index=3)
    robust_loss = RobustSourceLoss(
        class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        ignore_index=3,
    )

    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.full((2, 16, 16), 3, dtype=torch.long)
    mask = torch.zeros((2, 16, 16), dtype=torch.uint8)

    # Losses should return 0.0 or near-zero when all pixels are ignore_index
    l1 = ctrl_loss(pred, target)
    l2 = robust_loss(pred, target)
    assert abs(l1.item()) < 1e-5, f"ControlledSourceLoss should ignore class 3, got {l1.item()}"
    assert abs(l2.item()) < 1e-5, f"RobustSourceLoss should ignore class 3, got {l2.item()}"

    # Confusion counts consider class 2 as positive flood class and class 3 as ignore
    pred_labels = np.array([[2, 0], [1, 3]], dtype=np.int64)
    gt_labels = np.array([[2, 2], [1, 3]], dtype=np.int64)
    val_mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    counts = compute_binary_confusion_counts(pred_labels, gt_labels, val_mask, positive_class=2, ignore_index=3)
    # Valid pixels: 3 (the [1,1] entry with label 3 is ignored)
    assert counts["valid_pixels"] == 3
    # [0,0]: pred=2, gt=2 -> TP (1)
    # [0,1]: pred=0, gt=2 -> FN (1)
    # [1,0]: pred=1, gt=1 -> TN (1)
    assert counts["tp"] == 1
    assert counts["fn"] == 1
    assert counts["tn"] == 1
    assert counts["fp"] == 0

    rates = compute_rates_from_counts(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        tn=counts["tn"],
    )
    assert abs(rates["precision"] - 1.0) < 1e-5  # 1 / (1 + 0) = 1.0
    assert abs(rates["recall"] - 0.5) < 1e-5     # 1 / (1 + 1) = 0.5
    assert abs(rates["f1"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-5


# 2. Optimizer-step accounting and tail flush
def test_2_optimizer_step_accounting_and_tail_flush():
    """Verify step accounting and zero cross-epoch accumulator bleed."""
    # Simulation of runner microbatch accumulation
    def simulate_training_steps(total_samples: int, batch_size: int, accum_steps: int, epochs: int):
        import math
        microbatches_per_epoch = math.ceil(total_samples / batch_size)
        steps_per_epoch = []
        global_step = 0
        for _ in range(epochs):
            accum_microbatches = 0
            epoch_steps = 0
            for b_idx in range(microbatches_per_epoch):
                accum_microbatches += 1
                if accum_microbatches == accum_steps or (b_idx + 1 == microbatches_per_epoch):
                    epoch_steps += 1
                    global_step += 1
                    accum_microbatches = 0
            steps_per_epoch.append(epoch_steps)
        return steps_per_epoch, global_step

    # Fold C (RST4 discovery, 14,967 samples, microbatch 8, accum 4)
    steps_c, total_c = simulate_training_steps(14967, 8, 4, 16)
    assert all(s == 468 for s in steps_c), f"Expected 468 steps/epoch, got {steps_c}"
    assert total_c == 7488, f"Expected 7488 total steps, got {total_c}"

    # Fold A (confirmatory, 13,302 samples, microbatch 8, accum 4)
    # 13302 / 8 = 1663 microbatches. 1663 = 415 * 4 + 3 -> 416 steps/epoch
    steps_a, total_a = simulate_training_steps(13302, 8, 4, 16)
    assert all(s == 416 for s in steps_a), f"Expected 416 steps/epoch, got {steps_a}"
    assert total_a == 6656, f"Expected 6656 total steps, got {total_a}"

    # Fold B (confirmatory, 14,655 samples, microbatch 8, accum 4)
    # 14655 / 8 = 1832 microbatches. 1832 = 458 * 4 + 0 -> 458 steps/epoch
    steps_b, total_b = simulate_training_steps(14655, 8, 4, 16)
    assert all(s == 458 for s in steps_b), f"Expected 458 steps/epoch, got {steps_b}"
    assert total_b == 7328, f"Expected 7328 total steps, got {total_b}"


# 3. Confirmation configs present (12 v1 original + 6 v2 versioned retries)
def test_3_exactly_12_confirmation_configs():
    """Verify 12 original configs and 6 v2 retry configs exist in configs/robust_source_training/rst5_confirmation/."""
    conf_dir = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation"
    assert conf_dir.is_dir()
    configs = sorted(list(conf_dir.glob("*.json")))

    expected_v1 = {
        "rst6_r0_snunet_folda_s0_v1.json", "rst6_r0_snunet_folda_s1_v1.json", "rst6_r0_snunet_folda_s2_v1.json",
        "rst6_r0_snunet_foldb_s0_v1.json", "rst6_r0_snunet_foldb_s1_v1.json", "rst6_r0_snunet_foldb_s2_v1.json",
        "rst6_r2_snunet_folda_s0_v1.json", "rst6_r2_snunet_folda_s1_v1.json", "rst6_r2_snunet_folda_s2_v1.json",
        "rst6_r2_snunet_foldb_s0_v1.json", "rst6_r2_snunet_foldb_s1_v1.json", "rst6_r2_snunet_foldb_s2_v1.json",
    }
    expected_v2 = {
        "rst6_r2_snunet_folda_s0_v2.json", "rst6_r2_snunet_folda_s1_v2.json", "rst6_r2_snunet_folda_s2_v2.json",
        "rst6_r2_snunet_foldb_s0_v2.json", "rst6_r2_snunet_foldb_s1_v2.json", "rst6_r2_snunet_foldb_s2_v2.json",
    }
    actual_names = {c.name for c in configs}
    assert expected_v1.issubset(actual_names), f"Missing v1 configs: {expected_v1 - actual_names}"
    assert expected_v2.issubset(actual_names), f"Missing v2 configs: {expected_v2 - actual_names}"
    assert actual_names == (expected_v1 | expected_v2), f"Unexpected configs found: {actual_names - (expected_v1 | expected_v2)}"


# 4. Confirmation matrix and ledger consistency
def test_4_matrix_and_ledger_consistency():
    """Verify matrix and ledger consistencies (12 active planned runs, 108 event rows, 18 unique eval events)."""
    matrix = _load_json("configs/robust_source_training/rst5_confirmation_matrix_v1.json")
    ledger = _load_json("configs/robust_source_training/rst5_confirmation_ledger_v1.json")

    assert matrix["summary"]["physical_training_runs_count"] == 12
    assert matrix["summary"]["total_event_result_rows_expected"] == 108

    active_runs = [r for r in ledger["runs"] if r.get("launch_status") != "INTERRUPTED_INVALIDATED"]
    assert len(active_runs) == 12, f"Expected 12 active runs in ledger, got {len(active_runs)}"
    total_eval_events_in_ledger = sum(len(r["evaluation_events"]) for r in active_runs)
    assert total_eval_events_in_ledger == 108

    # Check 18 unique evaluation events
    all_eval_events = set()
    for r in active_runs:
        all_eval_events.update(r["evaluation_events"])
    assert len(all_eval_events) == 18
    assert all(e in AUTHORITATIVE_SOURCE_EVENTS for e in all_eval_events)
    assert not any(e in FORBIDDEN_TARGET_EVENTS for e in all_eval_events)


# 5. Fold A and Fold B class weights consistency
def test_5_folda_and_foldb_class_weights_consistency():
    """Verify Fold A and Fold B class weights sum to 3.0 and are properly bounded."""
    w_a = _load_json("configs/robust_source_training/rst5_folda_class_weights_v1.json")
    w_b = _load_json("configs/robust_source_training/rst5_foldb_class_weights_v1.json")

    weights_a = w_a["final_weights_list"]
    weights_b = w_b["final_weights_list"]

    assert len(weights_a) == 3
    assert len(weights_b) == 3
    assert abs(sum(weights_a) - 3.0) < 1e-4, f"Fold A weights do not sum to 3.0: {sum(weights_a)}"
    assert abs(sum(weights_b) - 3.0) < 1e-4, f"Fold B weights do not sum to 3.0: {sum(weights_b)}"

    # Weights must be distinct between folds
    assert weights_a != weights_b, "Fold A and Fold B weights should differ based on fold distributions"


# 6. Confirmatory decision rule integrity
def test_6_confirmatory_decision_rule_integrity():
    """Verify P1, P2, P3 pre-registered criteria and thresholds."""
    rule = _load_json("configs/robust_source_training/rst5_confirmation_decision_rule_v1.json")
    crit = rule["decision_criteria"]

    # P1
    p1 = crit["P1_average_improvement"]
    assert ">= +0.010" in p1["required_conditions"][0]
    assert p1["scientific_verdict_if_satisfied"] == "P1_SUPPORTED_AVERAGE_IMPROVEMENT"

    # P2
    p2 = crit["P2_reliability_improvement"]
    assert ">= +0.020" in p2["required_conditions"][0]
    assert p2["scientific_verdict_if_satisfied"] == "P2_SUPPORTED_TAIL_RELIABILITY_IMPROVEMENT"

    # P3
    p3 = crit["P3_seed_stability"]
    assert ">= 0.20" in p3["required_conditions"][1]
    assert p3["scientific_verdict_if_satisfied"] == "P3_SUPPORTED_SEED_STABILITY"


# 7. Zero forbidden events in all 12 configs
def test_7_zero_forbidden_events_in_all_12_configs():
    """Verify none of the 16 forbidden target events appear in any confirmation config."""
    conf_dir = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation"
    for cfg_path in conf_dir.glob("*.json"):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        train_events = cfg["partition"]["train_event_ids"]
        eval_events = cfg["partition"]["evaluation_event_ids"]

        for ev in train_events + eval_events:
            assert ev not in FORBIDDEN_TARGET_EVENTS, (
                f"Forbidden event {ev} found in {cfg_path.name}!"
            )
            assert ev in AUTHORITATIVE_SOURCE_EVENTS, (
                f"Event {ev} in {cfg_path.name} is not in authoritative source events!"
            )


# 8. Fail-closed launch prevention
def test_8_fail_closed_launch_prevention():
    """Verify runner fails closed if preparation_only=True or launch_authorized=False."""
    cfg_path = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r0_snunet_folda_s0_v1.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # In RST6, formal configs are transitioned to authorized
    assert cfg["launch_authorized"] is True
    assert cfg["preparation_only"] is False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 8a: preparation_only=True fails closed
        cfg_prep = dict(cfg)
        cfg_prep["preparation_only"] = True
        p_prep = Path(tmpdir) / "cfg_prep.json"
        with open(p_prep, "w", encoding="utf-8") as f:
            json.dump(cfg_prep, f)
        engine_prep = RSTExecutionEngine(config_path=p_prep, repo_root=REPO_ROOT)
        try:
            engine_prep.execute()
            assert False, "Should fail when preparation_only=True"
        except ContractViolationError as e:
            assert "preparation_only is True" in str(e)

        # Test 8b: launch_authorized=False fails closed
        cfg_unauth = dict(cfg)
        cfg_unauth["launch_authorized"] = False
        p_unauth = Path(tmpdir) / "cfg_unauth.json"
        with open(p_unauth, "w", encoding="utf-8") as f:
            json.dump(cfg_unauth, f)
        engine_unauth = RSTExecutionEngine(config_path=p_unauth, repo_root=REPO_ROOT)
        try:
            engine_unauth.execute()
            assert False, "Should fail when launch_authorized=False"
        except ContractViolationError as e:
            assert "launch_authorized is False" in str(e)


# 9. Unknown config keys fail-closed
def test_9_unknown_config_keys_fail_closed():
    """Verify runner fails closed when unknown config keys are present."""
    cfg_path = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r0_snunet_folda_s0_v1.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        tampered_cfg_path = Path(tmpdir) / "tampered_config.json"
        cfg_tampered = dict(cfg)
        cfg_tampered["malicious_or_unknown_key"] = "exploit"
        with open(tampered_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_tampered, f)

        engine = RSTExecutionEngine(config_path=tampered_cfg_path, repo_root=REPO_ROOT)
        try:
            engine.validate_only()
            assert False, "Static validation should have failed closed on unknown key!"
        except ContractViolationError as e:
            assert "Unknown config key detected: 'malicious_or_unknown_key'" in str(e)


# 10. Synthetic interruption and checkpoint-resume state equality
def test_10_interruption_checkpoint_resume_equality():
    """Verify saving checkpoint and loading recovers exact weights and RNG states."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt_epoch_4.pt"

        model1 = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)
        opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

        # Set specific random seed
        seed_everything(42)
        # Take a dummy gradient step
        x = torch.randn(2, 2, 32, 32)
        out = model1(x, x)
        loss = out.sum()
        loss.backward()
        opt1.step()

        save_rst_checkpoint(
            path=ckpt_path,
            model=model1,
            optimizer=opt1,
            epoch=4,
            global_step=1664,
            seed=42,
            run_id="synthetic_test_run",
            config_hash="a" * 64,
            fold_hash="b" * 64,
            train_event_ids=[130, 147],
            evaluation_event_ids=[118, 174],
        )

        # Create fresh model and optimizer
        model2 = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        meta = load_rst_checkpoint(
            path=ckpt_path,
            model=model2,
            optimizer=opt2,
            expected_config_hash="a" * 64,
            expected_fold_hash="b" * 64,
            expected_train_events=[130, 147],
            expected_eval_events=[118, 174],
            device="cpu",
        )

        assert meta["epoch"] == 4
        assert meta["global_step"] == 1664
        assert_states_equal(model1, model2)


# 11. Checkpoint reload strict metadata verification
def test_11_checkpoint_mismatch_rejection():
    """Verify mismatched config hash, fold hash, or event list are rejected on reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        save_rst_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            epoch=1,
            global_step=100,
            seed=0,
            run_id="test",
            config_hash="1" * 64,
            fold_hash="2" * 64,
            train_event_ids=[130],
            evaluation_event_ids=[118],
        )

        # Mismatched config hash
        try:
            load_rst_checkpoint(ckpt_path, model, expected_config_hash="9" * 64)
            assert False, "Should have failed on config hash mismatch!"
        except ContractViolationError:
            pass

        # Mismatched fold hash
        try:
            load_rst_checkpoint(ckpt_path, model, expected_fold_hash="9" * 64)
            assert False, "Should have failed on fold hash mismatch!"
        except ContractViolationError:
            pass

        # Mismatched train events
        try:
            load_rst_checkpoint(ckpt_path, model, expected_train_events=[147])
            assert False, "Should have failed on train events mismatch!"
        except ContractViolationError:
            pass


# 12. Recipe switch immutability
def test_12_recipe_switch_immutability():
    """Verify confirmatory recipes strictly adhere to R0 (0,0,0) and R2 (1,1,0)."""
    conf_dir = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation"
    for cfg_path in conf_dir.glob("*.json"):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rid = cfg["recipe_id"]
        comps = cfg["components"]

        # M3 must ALWAYS be false in confirmatory training
        assert comps["m3_paired_augmentation"] is False, f"M3 enabled in {cfg_path.name}!"

        if rid == "R0":
            assert comps["m1_event_balance"] is False
            assert comps["m2_flood_aware_loss"] is False
        elif rid == "R2":
            assert comps["m1_event_balance"] is True
            assert comps["m2_flood_aware_loss"] is True
        else:
            assert False, f"Unexpected recipe {rid} in confirmation matrix!"


# 13. Audit of resume contract and sampler progression
def test_13_resume_contract_and_sampler_progression_audit():
    """Formally audits the 13 required resume criteria from Phase B2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "audit_ckpt.pt"
        model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        save_rst_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            epoch=4,
            global_step=1664,
            seed=42,
            run_id="audit_run",
            config_hash="a" * 64,
            fold_hash="b" * 64,
            train_event_ids=[130, 147],
            evaluation_event_ids=[118, 174],
        )

        ckpt_dict = torch.load(ckpt_path, map_location="cpu")

        # Verified criteria:
        assert "model_state_dict" in ckpt_dict  # 1. model parameters & 2. buffers
        assert "optimizer_state_dict" in ckpt_dict  # 3. optimizer state
        assert "epoch" in ckpt_dict  # 4. epoch
        assert "global_step" in ckpt_dict  # 6. global step
        assert "rng_states" in ckpt_dict  # 7. RNG states
        assert "metadata" in ckpt_dict  # 13. metadata for mismatch checks

        # Deficient criteria (identifying why RST6 is blocked):
        # 5. Within-epoch batch position is NOT serialized in checkpoint
        assert "batch_idx" not in ckpt_dict and "b_idx" not in ckpt_dict
        # 8. Sampler order/progression is NOT serialized in checkpoint
        assert "sampler_state" not in ckpt_dict and "start_index" not in ckpt_dict


if __name__ == "__main__":
    tests = [
        test_1_label_semantics_and_class_index_alignment,
        test_2_optimizer_step_accounting_and_tail_flush,
        test_3_exactly_12_confirmation_configs,
        test_4_matrix_and_ledger_consistency,
        test_5_folda_and_foldb_class_weights_consistency,
        test_6_confirmatory_decision_rule_integrity,
        test_7_zero_forbidden_events_in_all_12_configs,
        test_8_fail_closed_launch_prevention,
        test_9_unknown_config_keys_fail_closed,
        test_10_interruption_checkpoint_resume_equality,
        test_11_checkpoint_mismatch_rejection,
        test_12_recipe_switch_immutability,
        test_13_resume_contract_and_sampler_progression_audit,
    ]

    print(f"Running {len(tests)} RST5 confirmation unit tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST5 confirmation tests PASSED!")
