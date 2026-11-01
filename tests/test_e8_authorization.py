#!/usr/bin/env python3
"""Fail-closed authorization and smoke-dependency tests for E8 v2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.run_e8_confirmation_matrix import E8ConfirmationRunner, compute_file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v2.json"
LEDGER = REPO_ROOT / "configs/changeguard/e8_execution_ledger_v2.json"
SMOKE = REPO_ROOT / "configs/changeguard/e7_cotta_smoke_v3.json"


class TestE8Authorization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, output_dir=None):
        return E8ConfirmationRunner(CONFIG, LEDGER, output_dir=output_dir, device="cpu")

    def write_smoke_evidence(self, runner, *, status="COMPLETED", label_io_pass=True):
        smoke_root = self.tmp / "smoke"
        stream_dir = smoke_root / "cotta_reset_s0_oa"
        stream_dir.mkdir(parents=True)
        smoke = json.loads(SMOKE.read_text())
        smoke["output_directory"] = str(smoke_root)
        smoke_path = self.tmp / "smoke_config.json"
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        dependency = runner.ledger["mandatory_launch_dependency"]
        dependency["smoke_config_path"] = str(smoke_path)
        dependency["completion_artifact"] = str(stream_dir / "completion_status.json")
        dependency["initial_manifest_artifact"] = str(stream_dir / "initial_manifest.json")
        dependency["final_manifest_artifact"] = str(stream_dir / "manifest.json")

        sample_ids = [f"sample_{i:02d}" for i in range(32)]
        for batch_index in range(2):
            ids = sample_ids[batch_index * 16 : (batch_index + 1) * 16]
            shard_dir = stream_dir / "event_520/shards"
            audit_dir = stream_dir / "event_520/label_io_audits"
            shard_dir.mkdir(parents=True, exist_ok=True)
            audit_dir.mkdir(parents=True, exist_ok=True)
            shard = shard_dir / f"batch_{batch_index:04d}.npz"
            np.savez_compressed(shard, sample_ids=np.array(ids), pred_cand_pre=np.zeros((16, 2, 2)))
            audit = {
                "event_id": 520,
                "batch_index": batch_index,
                "sample_ids": ids,
                "prediction_shard": str(shard),
                "prediction_shard_sha256": compute_file_sha256(shard),
                "persistence_timestamp_utc": "2026-10-22T02:00:00+00:00",
                "label_read_timestamp_utc": "2026-10-22T02:00:01+00:00",
                "label_reads": 16,
                "sample_identity_match": True,
                "label_read_after_persistence": label_io_pass,
            }
            (audit_dir / f"batch_{batch_index:04d}.json").write_text(json.dumps(audit), encoding="utf-8")

        config = json.loads(CONFIG.read_text())
        initial = {
            "config_sha256": compute_file_sha256(CONFIG),
            "ledger_sha256": compute_file_sha256(LEDGER),
            "runner_sha256": compute_file_sha256(REPO_ROOT / "tools/run_e8_confirmation_matrix.py"),
            "adapter_sha256": compute_file_sha256(REPO_ROOT / "src/changeguard/cotta_adapter.py"),
            "checkpoint_sha256": config["source_checkpoints"]["seeds"]["0"]["expected_sha256"],
            "split_sha256": config["dataset"]["split_expected_sha256"],
            "dataset_manifest_sha256": config["dataset"]["manifest_expected_sha256"],
            "execution_contract_sha256": compute_file_sha256(smoke_path),
            "git_tracked_dirty": False,
            "start_timestamp_utc": "2026-10-22T01:59:59+00:00",
            "event_ids": [520],
            "seed": 0,
            "expected_total_samples": 32,
            "expected_total_batches": 2,
            "gt_or_gt_derived_mask_entered_adaptation": False,
            "sealed_test_rejection_status": "VERIFIED_ZERO_SEALED_EVENTS",
        }
        initial_path = stream_dir / "initial_manifest.json"
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        initial_sha = compute_file_sha256(initial_path)
        verification = {
            "zero_gt_reads_before_first_prediction_persistence": label_io_pass,
            "all_label_reads_after_matching_prediction_persistence": label_io_pass,
            "gt_or_gt_derived_mask_entered_adaptation": False,
            "sample_identity_verified": True,
            "sealed_test_access": 0,
            "finite_outputs_verified": True,
            "failed_persistence_count": 0,
        }
        terminal = {
            "status": status,
            "initial_manifest_sha256": initial_sha,
            "completed_events": [520],
            "total_samples": 32,
            "total_batches": 2,
            "deferred_gt_reads": 32,
            "label_io_verification": verification,
        }
        (stream_dir / "completion_status.json").write_text(json.dumps(terminal), encoding="utf-8")
        (stream_dir / "manifest.json").write_text(json.dumps(terminal), encoding="utf-8")
        (stream_dir / "stream_summary.json").write_text(
            json.dumps({"deferred_gt_reads": 32, "macro_f1": 0.5}), encoding="utf-8"
        )
        return smoke_path, stream_dir

    def test_authorized_contract_has_exact_frozen_scope(self):
        runner = self.runner()
        runner.validate_authorized_matrix_contract()
        reused, execute_new = runner.planned_cells()
        self.assertEqual((len(reused), len(execute_new), len(reused) + len(execute_new)), (2, 67, 69))
        self.assertEqual((len(reused) + len(execute_new)) * 6, 414)
        self.assertEqual({cell["status"] for cell in reused}, {"REUSE_CERTIFIED"})
        self.assertEqual({cell["status"] for cell in execute_new}, {"EXECUTE_NEW"})

        smoke = json.loads(SMOKE.read_text())
        self.assertTrue(smoke["launch_authorized"])
        self.assertEqual(smoke["authorization"]["scope"]["event_ids"], [520])
        self.assertEqual(smoke["authorization"]["scope"]["expected_samples"], 32)
        self.assertFalse(smoke["authorization"]["scope"]["scientific_evidence"])

    def test_scientific_settings_unchanged_from_v1(self):
        v1 = json.loads((REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v1.json").read_text())
        v2 = json.loads(CONFIG.read_text())
        for key in ("model", "source_checkpoints", "stream_orders"):
            self.assertEqual(v1[key], v2[key])
        for key in (
            "dataset_root", "manifest_path", "manifest_expected_sha256", "split_path",
            "split_expected_sha256", "sar_mean", "sar_std", "sar_clamp_max",
            "canonical_preevent", "development_events", "forbidden_sealed_test_events", "batch_size",
        ):
            self.assertEqual(v1["dataset"][key], v2["dataset"][key])
        for method in ("cotta_continual", "cotta_reset"):
            for key in (
                "adapter_class", "learning_rate", "optimizer", "ema_alpha",
                "restoration_prob", "confidence_threshold", "num_views", "mode",
            ):
                self.assertEqual(v1["evaluated_methods"][method][key], v2["evaluated_methods"][method][key])

    def test_formal_launch_waits_when_smoke_evidence_absent(self):
        runner = self.runner()
        smoke = json.loads(SMOKE.read_text())
        smoke["output_directory"] = str(self.tmp / "missing_smoke")
        smoke_path = self.tmp / "smoke_config.json"
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        dependency = runner.ledger["mandatory_launch_dependency"]
        dependency["smoke_config_path"] = str(smoke_path)
        stream_dir = self.tmp / "missing_smoke/cotta_reset_s0_oa"
        dependency["completion_artifact"] = str(stream_dir / "completion_status.json")
        dependency["initial_manifest_artifact"] = str(stream_dir / "initial_manifest.json")
        dependency["final_manifest_artifact"] = str(stream_dir / "manifest.json")
        with self.assertRaisesRegex(FileNotFoundError, "WAITING_FOR_SMOKE_V3"):
            runner.verify_smoke_completion()

    def test_formal_launch_rejects_failed_smoke(self):
        runner = self.runner()
        self.write_smoke_evidence(runner, status="FAILED")
        with self.assertRaisesRegex(RuntimeError, "terminal status"):
            runner.verify_smoke_completion()

    def test_formal_launch_rejects_hash_mismatch(self):
        runner = self.runner()
        _, stream_dir = self.write_smoke_evidence(runner)
        initial_path = stream_dir / "initial_manifest.json"
        initial = json.loads(initial_path.read_text())
        initial["runner_sha256"] = "0" * 64
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        new_sha = compute_file_sha256(initial_path)
        for name in ("completion_status.json", "manifest.json"):
            terminal = json.loads((stream_dir / name).read_text())
            terminal["initial_manifest_sha256"] = new_sha
            (stream_dir / name).write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runner_sha256"):
            runner.verify_smoke_completion()

    def test_formal_launch_rejects_failed_label_io_verification(self):
        runner = self.runner()
        self.write_smoke_evidence(runner, label_io_pass=False)
        with self.assertRaisesRegex(RuntimeError, "label-I/O verification"):
            runner.verify_smoke_completion()

    def test_smoke_launch_rejects_existing_output(self):
        smoke = json.loads(SMOKE.read_text())
        output = self.tmp / "existing_smoke"
        output.mkdir()
        smoke["output_directory"] = str(output)
        smoke["authorization"]["fresh_output_directory"] = str(output)
        smoke_path = self.tmp / "smoke_config.json"
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        runner = self.runner(output_dir=output)
        with self.assertRaisesRegex(FileExistsError, "must be fresh"):
            runner.validate_smoke_launch(smoke_path)

    def test_count_tampering_fails_closed(self):
        runner = self.runner()
        runner.cfg = copy.deepcopy(runner.cfg)
        runner.cfg["authorized_scope"]["execute_new_physical_streams"] = 66
        with self.assertRaisesRegex(ValueError, "execute_new_physical_streams"):
            runner.validate_authorized_matrix_contract()


if __name__ == "__main__":
    unittest.main()
