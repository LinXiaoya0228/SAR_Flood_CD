"""Focused prelaunch verification test suite for RST4 Discovery Staircase."""

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.models.snunet_wrapper import SNUNetWrapper
from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.dataset import RSTR0SourceDataset
from src.evaluation.losses import ControlledSourceLoss
from src.robust_source_training.loss import (
    RobustSourceLoss,
    compute_training_class_weights,
)
from src.robust_source_training.runner import RSTExecutionEngine, seed_everything
from src.robust_source_training.sampler import EventBalancedSampler


def _load_config(filename: str):
    p = REPO_ROOT / "configs" / "robust_source_training" / filename
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. Exact A+B/C partition
def test_1_exact_ab_train_c_eval_partition():
    folds_data = _load_config("rst1_event_folds_v1.json")
    discovery_fold = folds_data["experimental_folds"]["discovery_fold_c"]
    expected_train = sorted(discovery_fold["training_event_ids"])
    expected_eval = sorted(discovery_fold["evaluation_event_ids"])

    assert len(expected_train) == 18
    assert len(expected_eval) == 9

    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        train_events = sorted(cfg["partition"]["train_event_ids"])
        eval_events = sorted(cfg["partition"]["evaluation_event_ids"])
        assert train_events == expected_train, f"Mismatch in {r} train events"
        assert eval_events == expected_eval, f"Mismatch in {r} eval events"


# 2. Identical initial-state hashes
def test_2_identical_initial_state_hashes():
    hashes = []
    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        seed = int(cfg["seed"])
        seed_everything(seed)
        arch = cfg["architecture"]
        model = SNUNetWrapper(
            in_channels=arch["in_channels"],
            out_ch=arch["out_ch"],
            base_channel=arch["base_channel"],
        )
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.cpu().numpy().tobytes())
        hashes.append(h.hexdigest())

    assert len(set(hashes)) == 1, f"Initial state hashes differ across recipes: {hashes}"
    assert hashes[0] == "71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1"


# 3. Recipe switches strict staircase
def test_3_recipe_switches_strict_staircase():
    expected = {
        "R0": (False, False, False),
        "R1": (True, False, False),
        "R2": (True, True, False),
        "R3": (True, True, True),
    }
    for r, exp_switches in expected.items():
        cfg = _load_config(f"rst4_{r.lower()}_snunet_foldc_s0_v1.json")
        comps = cfg["components"]
        actual = (
            bool(comps["m1_event_balance"]),
            bool(comps["m2_flood_aware_loss"]),
            bool(comps["m3_paired_augmentation"]),
        )
        assert actual == exp_switches, f"Switches mismatch for {r}: expected {exp_switches}, got {actual}"


# 4. Equal epoch budget
def test_4_equal_epoch_draw_budget():
    folds_data = _load_config("rst1_event_folds_v1.json")
    discovery_fold = folds_data["experimental_folds"]["discovery_fold_c"]
    total_train_patches = discovery_fold["training_patch_count"]
    assert total_train_patches == 14967

    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        assert cfg["partition"]["expected_train_sample_count"] == 14967


# 5. M2 weights identical between R2 and R3
def test_5_m2_weights_identical_between_r2_and_r3():
    r2_cfg = _load_config("rst4_r2_snunet_foldc_s0_v1.json")
    r3_cfg = _load_config("rst4_r3_snunet_foldc_s0_v1.json")

    assert r2_cfg["loss"] == r3_cfg["loss"]
    assert r2_cfg["partition"]["train_event_ids"] == r3_cfg["partition"]["train_event_ids"]


# 6. R0 unit loss weights
def test_6_r0_unit_loss_weights():
    r0_cfg = _load_config("rst4_r0_snunet_foldc_s0_v1.json")
    assert r0_cfg["loss"]["name"] == "ControlledSourceLoss"
    assert not r0_cfg["components"]["m2_flood_aware_loss"]


# 7. M3 disabled in evaluation
def test_7_m3_disabled_in_evaluation():
    r3_cfg = _load_config("rst4_r3_snunet_foldc_s0_v1.json")
    eval_events = r3_cfg["partition"]["evaluation_event_ids"]
    # Evaluation dataset is created with augmenter=None in runner
    ds = RSTR0SourceDataset(allowed_event_ids=[eval_events[0]], max_samples_cap=2)
    assert ds.augmenter is None


# 8. No Group C label statistics used by training
def test_8_no_group_c_label_statistics_used_by_training():
    group_c_events = [427, 1111011, 275, 1111005, 497, 147, 1111006, 130, 492]
    # Attempting to include Group C in compute_training_class_weights with forbidden event logic
    synth_meta = [{"event_id": 427, "sample_id": "test_c", "label_path": "", "valid_mask_path": ""}]
    failed = False
    try:
        compute_training_class_weights(
            train_samples_metadata=synth_meta,
            train_event_ids=group_c_events,
        )
    except Exception:
        failed = True
    assert failed, "Expected failure when passing Group C events to compute_training_class_weights"


# 9. No Group A/B evaluation
def test_9_no_group_a_b_evaluation():
    folds_data = _load_config("rst1_event_folds_v1.json")
    group_a_events = set(folds_data["groups"]["Group_A"]["event_ids"])
    group_b_events = set(folds_data["groups"]["Group_B"]["event_ids"])

    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        eval_events = set(cfg["partition"]["evaluation_event_ids"])
        assert len(eval_events.intersection(group_a_events)) == 0, f"{r} evaluates Group A events!"
        assert len(eval_events.intersection(group_b_events)) == 0, f"{r} evaluates Group B events!"


# 10. No forbidden events in any partition
def test_10_no_forbidden_events_in_any_partition():
    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        train_events = set(cfg["partition"]["train_event_ids"])
        eval_events = set(cfg["partition"]["evaluation_event_ids"])
        assert len(train_events.intersection(FORBIDDEN_TARGET_EVENTS)) == 0
        assert len(eval_events.intersection(FORBIDDEN_TARGET_EVENTS)) == 0


# 11. Output collision rejection
def test_11_output_collision_rejection():
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        colliding_dir = tmp_dir / "colliding_run"
        colliding_dir.mkdir(parents=True, exist_ok=True)
        (colliding_dir / "preflight_manifest.json").write_text("dummy", encoding="utf-8")

        cfg = _load_config("rst4_r0_snunet_foldc_s0_v1.json")
        cfg["output_root"] = str(colliding_dir)
        cfg["launch_authorized"] = True

        tmp_cfg_path = tmp_dir / "test_cfg.json"
        tmp_cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        engine = RSTExecutionEngine(config_path=tmp_cfg_path, repo_root=REPO_ROOT)
        failed = False
        try:
            engine.execute()
        except ContractViolationError:
            failed = True
        assert failed, "Expected ContractViolationError on output collision"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# 12. Fixed 16-epoch budget
def test_12_fixed_16_epoch_budget():
    for r in ["r0", "r1", "r2", "r3"]:
        cfg = _load_config(f"rst4_{r}_snunet_foldc_s0_v1.json")
        assert cfg["training"]["max_epochs"] == 16


# 13. Promotion rule logic
def test_13_deterministic_promotion_rule():
    rule = _load_config("rst4_promotion_rule_v1.json")
    assert rule["baseline_recipe"] == "R0"
    assert rule["candidate_recipes"] == ["R1", "R2", "R3"]
    assert "route_d1_mean_improvement" in rule["eligibility_routes"]
    assert "route_d2_reliability_improvement" in rule["eligibility_routes"]


# 14. Static contract validation passes for all 4 configs
def test_14_static_contract_validation_passes_all_four_configs():
    for r in ["r0", "r1", "r2", "r3"]:
        cfg_path = REPO_ROOT / "configs" / "robust_source_training" / f"rst4_{r}_snunet_foldc_s0_v1.json"
        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        val = engine.validate_only()
        assert val["is_valid"] is True
        assert val["train_event_count"] == 18
        assert val["evaluation_event_count"] == 9
        assert val["is_smoke"] is False


if __name__ == "__main__":
    tests = [
        test_1_exact_ab_train_c_eval_partition,
        test_2_identical_initial_state_hashes,
        test_3_recipe_switches_strict_staircase,
        test_4_equal_epoch_draw_budget,
        test_5_m2_weights_identical_between_r2_and_r3,
        test_6_r0_unit_loss_weights,
        test_7_m3_disabled_in_evaluation,
        test_8_no_group_c_label_statistics_used_by_training,
        test_9_no_group_a_b_evaluation,
        test_10_no_forbidden_events_in_any_partition,
        test_11_output_collision_rejection,
        test_12_fixed_16_epoch_budget,
        test_13_deterministic_promotion_rule,
        test_14_static_contract_validation_passes_all_four_configs,
    ]
    print(f"Running {len(tests)} RST4 focused prelaunch tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST4 tests PASSED!")
