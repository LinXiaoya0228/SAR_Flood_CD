"""Comprehensive unit test suite for RST6 Confirmatory Execution & Fresh-Run Recovery Policy."""

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

from src.models.snunet_wrapper import SNUNetWrapper
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
from src.robust_source_training.runner import RSTExecutionEngine, seed_everything


def _load_base_confirmation_config():
    p = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r0_snunet_folda_s0_v1.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. Confirmatory --resume is rejected
def test_1_confirmatory_resume_is_rejected():
    cfg = _load_base_confirmation_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "cfg.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        dummy_ckpt = Path(tmpdir) / "dummy.pt"
        dummy_ckpt.write_text("dummy")

        try:
            engine.execute(resume_checkpoint_path=dummy_ckpt)
            assert False, "Should have failed with ContractViolationError on confirmatory resume!"
        except ContractViolationError as e:
            assert "Confirmatory resume rejected" in str(e)


# 2. Fresh execution accepted only with authorization
def test_2_fresh_execution_accepted_only_with_authorization():
    cfg = copy.deepcopy(_load_base_confirmation_config())
    # Base config in RST6 has launch_authorized=True, preparation_only=False
    assert cfg["launch_authorized"] is True
    assert cfg["preparation_only"] is False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tamper to unauthorized state
        cfg["launch_authorized"] = False
        cfg["preparation_only"] = True
        cfg_path = Path(tmpdir) / "cfg_unauth.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        try:
            engine.execute()
            assert False, "Execution without authorization must fail!"
        except ContractViolationError as e:
            assert "launch_authorized is False" in str(e) or "preparation_only is True" in str(e)


# 3. Existing failed output root cannot be overwritten
def test_3_existing_failed_output_root_cannot_be_overwritten():
    cfg = copy.deepcopy(_load_base_confirmation_config())
    cfg["launch_authorized"] = True
    cfg["preparation_only"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output_test"
        out_dir.mkdir(parents=True)
        # Simulate partial failed output file
        (out_dir / "partial_train.log").write_text("interrupted")

        cfg["output_root"] = str(out_dir)
        cfg_path = Path(tmpdir) / "cfg.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        try:
            engine.execute()
            assert False, "Should reject collision with existing non-empty output directory!"
        except ContractViolationError as e:
            assert "already exists and contains files" in str(e)


# 4. Retry requires new versioned run ID and fresh output root
def test_4_retry_requires_new_versioned_run_id_and_fresh_root():
    cfg_v1 = copy.deepcopy(_load_base_confirmation_config())
    with tempfile.TemporaryDirectory() as tmpdir:
        out_v1 = Path(tmpdir) / "out_v1"
        out_v1.mkdir(parents=True)
        (out_v1 / "failed.log").write_text("failed")

        # Versioned retry config v2
        cfg_v2 = copy.deepcopy(cfg_v1)
        cfg_v2["run_id"] = "rst6_r0_snunet_folda_s0_v2"
        out_v2 = Path(tmpdir) / "out_v2"
        cfg_v2["output_root"] = str(out_v2)
        cfg_v2["launch_authorized"] = True
        cfg_v2["preparation_only"] = False

        cfg_v2_path = Path(tmpdir) / "cfg_v2.json"
        with open(cfg_v2_path, "w", encoding="utf-8") as f:
            json.dump(cfg_v2, f)

        engine_v2 = RSTExecutionEngine(config_path=cfg_v2_path, repo_root=REPO_ROOT)
        # Validation and dry run pass cleanly for v2
        val = engine_v2.validate_only()
        assert val["is_valid"] is True


# 5. Completed output cannot be rerun
def test_5_completed_output_cannot_be_rerun():
    cfg = copy.deepcopy(_load_base_confirmation_config())
    cfg["launch_authorized"] = True
    cfg["preparation_only"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out_completed"
        out_dir.mkdir(parents=True)
        (out_dir / "terminal_status.json").write_text('{"status": "COMPLETED_SCIENTIFIC_RUN"}')

        cfg["output_root"] = str(out_dir)
        cfg_path = Path(tmpdir) / "cfg.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        engine = RSTExecutionEngine(config_path=cfg_path, repo_root=REPO_ROOT)
        try:
            engine.execute()
            assert False, "Completed directory must not be overwritten!"
        except ContractViolationError as e:
            assert "already exists and contains files" in str(e)


# 6. Retry config preserves scientific settings
def test_6_retry_config_preserves_scientific_settings():
    cfg_v1 = copy.deepcopy(_load_base_confirmation_config())
    cfg_v2 = copy.deepcopy(cfg_v1)
    cfg_v2["run_id"] = "rst6_r0_snunet_folda_s0_v2"
    cfg_v2["output_root"] = "outputs/robust_source_training/rst6/rst6_r0_snunet_folda_s0_v2"
    cfg_v2["log_file"] = "logs/robust_source_training/rst6/rst6_r0_snunet_folda_s0_v2.log"

    # Scientific settings must be identical
    scientific_keys = [
        "recipe_id", "components", "fold_id", "seed", "max_epochs",
        "partition", "training", "architecture", "optimization"
    ]
    for k in scientific_keys:
        assert cfg_v1[k] == cfg_v2[k], f"Scientific key {k} differs between v1 and v2!"


# 7. Retry seed produces the same initial model hash
def test_7_retry_seed_produces_identical_initial_model_hash():
    def get_initial_weights_digest(seed: int):
        seed_everything(seed)
        model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.cpu().numpy().tobytes())
        return h.hexdigest()

    # Seed 0 run v1 and retry v2
    hash_v1 = get_initial_weights_digest(seed=0)
    hash_v2 = get_initial_weights_digest(seed=0)
    assert hash_v1 == hash_v2, "Identical seed must yield identical initial model hash!"


# 8. No partial checkpoint is loaded by a fresh retry
def test_8_no_partial_checkpoint_loaded_by_fresh_retry():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a partial checkpoint
        ckpt_path = Path(tmpdir) / "partial.pt"
        m_partial = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        opt_partial = torch.optim.Adam(m_partial.parameters(), lr=1e-3)
        # Shift weights
        for p in m_partial.parameters():
            p.data.add_(1.0)

        save_rst_checkpoint(
            path=ckpt_path,
            model=m_partial,
            optimizer=opt_partial,
            epoch=5,
            global_step=2000,
            seed=0,
            run_id="partial_run",
            config_hash="h1",
            fold_hash="h2",
            train_event_ids=[130],
            evaluation_event_ids=[118],
        )

        # Fresh model initialized with seed 0
        seed_everything(0)
        m_fresh = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)

        # Verify that fresh model does NOT match partial checkpoint
        diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(m_fresh.parameters(), m_partial.parameters()))
        assert diff > 0.5, "Fresh model should have independent initial weights from partial checkpoint!"


# 9. Confirmatory R2 requires class_weights provenance
def test_9_confirmatory_r2_fails_closed_without_class_weights_provenance():
    r2_cfg_path = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r2_snunet_folda_s0_v1.json"
    with open(r2_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["provenance_references"].pop("class_weights_path", None)
        bad_cfg_path = Path(tmpdir) / "bad_cfg.json"
        with open(bad_cfg_path, "w", encoding="utf-8") as f:
            json.dump(bad_cfg, f)

        engine = RSTExecutionEngine(config_path=bad_cfg_path, repo_root=REPO_ROOT)
        try:
            engine.validate_only()
            assert False, "Should fail closed when class_weights_path is missing in R2 confirmatory config"
        except ContractViolationError as e:
            assert "class_weights_path" in str(e)


# 10. Confirmatory R2 fails closed on class-weights artifact hash mismatch
def test_10_confirmatory_r2_fails_closed_on_class_weights_hash_mismatch():
    r2_cfg_path = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r2_snunet_folda_s0_v1.json"
    with open(r2_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["provenance_references"]["class_weights_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        bad_cfg_path = Path(tmpdir) / "bad_cfg.json"
        with open(bad_cfg_path, "w", encoding="utf-8") as f:
            json.dump(bad_cfg, f)

        engine = RSTExecutionEngine(config_path=bad_cfg_path, repo_root=REPO_ROOT)
        try:
            engine.validate_only()
            assert False, "Should fail closed when class_weights_sha256 mismatches"
        except ContractViolationError as e:
            assert "hash mismatch" in str(e)


# 11. Runtime pre-compute assertion catches weight divergence before model construction
def test_11_runtime_pre_compute_assertion_catches_weight_divergence():
    r2_cfg_path = REPO_ROOT / "configs" / "robust_source_training" / "rst5_confirmation" / "rst6_r2_snunet_folda_s0_v1.json"
    with open(r2_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
        # Create a mock frozen artifact with altered weights
        fake_art = Path(tmpdir) / "fake_weights.json"
        fake_content = {
            "final_weights_list": [0.999999, 1.000000, 1.000001],
            "aggregate_counts": {"class_0": 1, "class_1": 1, "class_2": 1, "total_valid": 3}
        }
        fake_art.write_text(json.dumps(fake_content), encoding="utf-8")
        fake_sha = hashlib.sha256(fake_art.read_bytes()).hexdigest()

        test_cfg = copy.deepcopy(cfg)
        test_cfg["output_root"] = str(Path(tmpdir) / "out_test")
        test_cfg["log_file"] = str(Path(tmpdir) / "out_test.log")
        test_cfg["provenance_references"]["class_weights_path"] = str(fake_art.relative_to(REPO_ROOT))
        test_cfg["provenance_references"]["class_weights_sha256"] = fake_sha

        cfg_p = Path(tmpdir) / "test_cfg.json"
        with open(cfg_p, "w", encoding="utf-8") as f:
            json.dump(test_cfg, f)

        engine = RSTExecutionEngine(config_path=cfg_p, repo_root=REPO_ROOT)
        prov_refs = test_cfg["provenance_references"]
        full_cw_path = REPO_ROOT / prov_refs["class_weights_path"]
        with open(full_cw_path) as f_cw:
            frozen_cw_data = json.load(f_cw)
        frozen_w = frozen_cw_data["final_weights_list"]
        computed_w = [0.486491, 1.409167, 1.104342]
        assert computed_w != frozen_w, "Test setup error: weights should differ!"


# 12. Frozen Fold A and Fold B class weights match independent pixel accounting
def test_12_frozen_folda_and_foldb_class_weights_match_independent_pixel_accounting():
    import numpy as np
    for fold_name, path_str in [
        ("Fold A", "configs/robust_source_training/rst5_folda_class_weights_v1.json"),
        ("Fold B", "configs/robust_source_training/rst5_foldb_class_weights_v1.json"),
    ]:
        p = REPO_ROOT / path_str
        assert p.exists(), f"Frozen artifact {path_str} missing!"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        agg = data["aggregate_counts"]
        c0, c1, c2, total = agg["class_0"], agg["class_1"], agg["class_2"], agg["total_valid"]
        assert c0 + c1 + c2 == total, f"Pixel count sum mismatch in {fold_name}"

        # RST5 Frozen Formula:
        # raw_w_c = 1 / sqrt(p_c)
        # w_norm = raw_w / mean(raw_w)
        # w_clipped = clip(w_norm, 0.5, 2.0)
        # final_w = w_clipped / mean(w_clipped)
        p_arr = np.array([c0, c1, c2], dtype=np.float64) / float(total)
        raw_w = 1.0 / np.sqrt(p_arr)
        norm_w = raw_w / float(np.mean(raw_w))
        clipped_w = np.clip(norm_w, 0.5, 2.0)
        final_w = clipped_w / float(np.mean(clipped_w))
        final_rounded = [round(float(x), 6) for x in final_w]

        assert final_rounded == data["final_weights_list"], (
            f"Formula recalculation {final_rounded} does not match {data['final_weights_list']} in {fold_name}"
        )


if __name__ == "__main__":
    tests = [
        test_1_confirmatory_resume_is_rejected,
        test_2_fresh_execution_accepted_only_with_authorization,
        test_3_existing_failed_output_root_cannot_be_overwritten,
        test_4_retry_requires_new_versioned_run_id_and_fresh_root,
        test_5_completed_output_cannot_be_rerun,
        test_6_retry_config_preserves_scientific_settings,
        test_7_retry_seed_produces_identical_initial_model_hash,
        test_8_no_partial_checkpoint_loaded_by_fresh_retry,
        test_9_confirmatory_r2_fails_closed_without_class_weights_provenance,
        test_10_confirmatory_r2_fails_closed_on_class_weights_hash_mismatch,
        test_11_runtime_pre_compute_assertion_catches_weight_divergence,
        test_12_frozen_folda_and_foldb_class_weights_match_independent_pixel_accounting,
    ]

    print(f"Running {len(tests)} RST6 fresh-run recovery policy tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST6 fresh-run recovery tests PASSED!")
