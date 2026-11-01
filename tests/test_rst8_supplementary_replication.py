"""Focused validation tests for RST8 SiamUnet_diff Supplementary Replication Study (Seeds 1 and 2)."""

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

from src.models.siam_diff_wrapper import SiamDiffWrapper, SiamUnet_diff
from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.runner import (
    RSTExecutionEngine,
    build_model,
    seed_everything,
)

CONFIG_DIR = REPO_ROOT / "configs" / "robust_source_training" / "rst8_siamunet_diff"
SUPP_MATRIX_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst8_supplementary_matrix_v1.json"
FULL_MATRIX_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst8_full_3seed_matrix_v1.json"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_supplementary_matrix_and_config_fidelity():
    """Verify supplementary matrix and all 8 new configs exist and match hashes."""
    supp_matrix = _load_json(SUPP_MATRIX_PATH)
    assert supp_matrix["summary"]["physical_training_runs_count"] == 8
    assert len(supp_matrix["cells"]) == 8

    expected_new_runs = {
        "rst8_r0_siamunet_diff_folda_s1_v1",
        "rst8_r0_siamunet_diff_folda_s2_v1",
        "rst8_r2_siamunet_diff_folda_s1_v1",
        "rst8_r2_siamunet_diff_folda_s2_v1",
        "rst8_r0_siamunet_diff_foldb_s1_v1",
        "rst8_r0_siamunet_diff_foldb_s2_v1",
        "rst8_r2_siamunet_diff_foldb_s1_v1",
        "rst8_r2_siamunet_diff_foldb_s2_v1",
    }
    observed_runs = {c["run_id"] for c in supp_matrix["cells"]}
    assert observed_runs == expected_new_runs

    for cell in supp_matrix["cells"]:
        cfg_path = REPO_ROOT / cell["config_path"]
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        actual_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        assert actual_hash == cell["config_sha256"], f"Hash mismatch in {cell['run_id']}"

        cfg = _load_json(cfg_path)
        assert cfg["architecture"]["name"] == "SiamUnet_diff"
        assert cfg["architecture"]["in_channels"] == 2
        assert cfg["architecture"]["out_ch"] == 3
        assert cfg["architecture"]["dropout_p"] == 0.2
        assert cfg["seed"] in (1, 2)
        assert cfg["max_epochs"] == 16
        assert cfg["training"]["micro_batch_size"] == 8
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["training"]["effective_batch_size"] == 32
        assert cfg["optimization"]["learning_rate"] == 0.001
        assert cfg["execution_class"] == "CONFIRMATORY_TRAINING"
        assert cfg["launch_authorized"] is True
        assert cfg["preparation_only"] is False


def test_2_full_12_run_matrix_coverage():
    """Verify full 3-seed matrix contains exactly 4 existing + 8 new runs = 12 total."""
    full_matrix = _load_json(FULL_MATRIX_PATH)
    assert full_matrix["summary"]["physical_training_runs_count"] == 12
    assert full_matrix["summary"]["existing_completed_runs_count"] == 4
    assert full_matrix["summary"]["new_planned_runs_count"] == 8
    assert len(full_matrix["cells"]) == 12

    # Check seed distribution
    seeds = [c["seed"] for c in full_matrix["cells"]]
    assert seeds.count(0) == 4
    assert seeds.count(1) == 4
    assert seeds.count(2) == 4

    # Check recipe distribution
    recipes = [c["recipe"] for c in full_matrix["cells"]]
    assert recipes.count("R0") == 6
    assert recipes.count("R2") == 6

    # Check fold distribution
    folds = [c["fold"] for c in full_matrix["cells"]]
    assert folds.count("confirmation_fold_a") == 6
    assert folds.count("confirmation_fold_b") == 6


def test_3_static_engine_validation_all_eight_runs():
    """Ensure all 8 new configs pass RSTExecutionEngine.validate_only() fail-closed."""
    supp_matrix = _load_json(SUPP_MATRIX_PATH)
    for cell in supp_matrix["cells"]:
        cfg_path = REPO_ROOT / cell["config_path"]
        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        val = engine.validate_only()
        assert val["is_valid"] is True
        assert engine.config["recipe_id"] in ("R0", "R2")
        assert len(val["train_event_ids"]) == 18
        assert len(val["evaluation_event_ids"]) == 9
        assert not (set(val["train_event_ids"]) & set(FORBIDDEN_TARGET_EVENTS))
        assert not (set(val["evaluation_event_ids"]) & set(FORBIDDEN_TARGET_EVENTS))
        assert set(val["train_event_ids"]).isdisjoint(set(val["evaluation_event_ids"]))


def test_4_class_weights_provenance_and_integrity():
    """Verify that class weights referenced in all R2 configs match frozen artifacts."""
    supp_matrix = _load_json(SUPP_MATRIX_PATH)
    for cell in supp_matrix["cells"]:
        if cell["recipe"] == "R2":
            cfg = _load_json(REPO_ROOT / cell["config_path"])
            prov = cfg["provenance_references"]
            cw_path = REPO_ROOT / prov["class_weights_path"]
            assert cw_path.exists()
            actual_sha = hashlib.sha256(cw_path.read_bytes()).hexdigest()
            assert actual_sha == prov["class_weights_sha256"]


def test_5_seed_determinism_siamunet_diff():
    """Verify initial parameter determinism for SiamUnet_diff across seeds 0, 1, 2."""
    hashes = {}
    for seed in [0, 1, 2]:
        seed_everything(seed)
        model1 = build_model({"name": "SiamUnet_diff", "in_channels": 2, "out_ch": 3, "dropout_p": 0.2})
        h1 = hashlib.sha256()
        for k, v in sorted(model1.state_dict().items()):
            h1.update(k.encode("utf-8"))
            h1.update(v.cpu().numpy().tobytes())
        hash1 = h1.hexdigest()

        # Re-initialize with same seed
        seed_everything(seed)
        model2 = build_model({"name": "SiamUnet_diff", "in_channels": 2, "out_ch": 3, "dropout_p": 0.2})
        h2 = hashlib.sha256()
        for k, v in sorted(model2.state_dict().items()):
            h2.update(k.encode("utf-8"))
            h2.update(v.cpu().numpy().tobytes())
        hash2 = h2.hexdigest()

        assert hash1 == hash2, f"Seed {seed} initialization is non-deterministic!"
        hashes[seed] = hash1

    # Ensure different seeds produce different initial weights
    assert hashes[0] != hashes[1]
    assert hashes[1] != hashes[2]
    assert hashes[0] != hashes[2]
    print(f"Seed 0 initial hash: {hashes[0][:16]}...")
    print(f"Seed 1 initial hash: {hashes[1][:16]}...")
    print(f"Seed 2 initial hash: {hashes[2][:16]}...")


if __name__ == "__main__":
    test_1_supplementary_matrix_and_config_fidelity()
    print("test_1_supplementary_matrix_and_config_fidelity PASSED")
    test_2_full_12_run_matrix_coverage()
    print("test_2_full_12_run_matrix_coverage PASSED")
    test_3_static_engine_validation_all_eight_runs()
    print("test_3_static_engine_validation_all_eight_runs PASSED")
    test_4_class_weights_provenance_and_integrity()
    print("test_4_class_weights_provenance_and_integrity PASSED")
    test_5_seed_determinism_siamunet_diff()
    print("test_5_seed_determinism_siamunet_diff PASSED")
    print("\nALL SUPPLEMENTARY VALIDATION TESTS PASSED SUCCESSFULLY!")
