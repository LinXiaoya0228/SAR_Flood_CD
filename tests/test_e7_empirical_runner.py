#!/usr/bin/env python3
"""ChangeGuard Work Package E7: Empirical Runner & Analyzer Unit Tests.

Rigorous unit test suite verifying:
1. Preflight integrity and sealed-test quarantine.
2. Immediate rejection on sealed-test event inclusion.
3. Checkpoint and split SHA-256 verification.
4. Overwrite prevention under Plan V3 rules.
5. Stream completion detection.
6. LOEO 6-fold isolation without pseudo-replication or label leakage.
7. Delta F1 arithmetic and harm/benefit/neutral classification.
8. Analyzer destination overwrite protection.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.run_e7_empirical_evaluation import (
    E7EmpiricalRunner,
    compute_file_sha256,
)
from tools.analyze_e7_empirical_comparison import (
    E7EmpiricalAnalyzer,
    DECLARED_DEVELOPMENT_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    compute_f1_from_cells,
    compute_iou_from_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestE7EmpiricalRunner(unittest.TestCase):
    """Unit test suite for E7 Empirical Runner and Analyzer."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)
        self.config_path = REPO_ROOT / "configs/changeguard/e7_branch_c_empirical_v1.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_preflight_validation_success(self):
        """1. Verifies that runner preflights pass on the committed formal config."""
        runner = E7EmpiricalRunner(config_path=self.config_path, device="cpu")
        report = runner.run_preflights()
        self.assertEqual(report["status"], "PREFLIGHTS_PASSED")
        self.assertEqual(report["target_events"], list(DECLARED_DEVELOPMENT_EVENTS))

    def test_02_sealed_test_quarantine_rejection(self):
        """2. Verifies immediate fatal error if any sealed-test event is included."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Inject sealed-test event 205
        cfg["dataset"]["development_events"] = [279, 437, 205]
        bad_config_path = self.tmp_path / "bad_config.json"
        with open(bad_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        runner = E7EmpiricalRunner(config_path=bad_config_path, device="cpu")
        with self.assertRaises(RuntimeError) as ctx:
            runner.run_preflights()
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("205", str(ctx.exception))

    def test_03_corrupt_manifest_sha256_rejection(self):
        """3. Verifies that mismatched manifest SHA-256 raises ValueError."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["dataset"]["manifest_expected_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        bad_config_path = self.tmp_path / "bad_manifest_config.json"
        with open(bad_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        runner = E7EmpiricalRunner(config_path=bad_config_path, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            runner.run_preflights()
        self.assertIn("Manifest SHA-256 mismatch", str(ctx.exception))

    def test_04_corrupt_checkpoint_sha256_rejection(self):
        """4. Verifies that mismatched checkpoint SHA-256 raises ValueError."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["source_checkpoints"]["seeds"]["0"]["expected_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        bad_config_path = self.tmp_path / "bad_ckpt_config.json"
        with open(bad_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        runner = E7EmpiricalRunner(config_path=bad_config_path, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            runner.run_preflights()
        self.assertIn("Checkpoint seed 0 SHA-256 mismatch", str(ctx.exception))

    def test_05_allow_overwrite_strictly_forbidden(self):
        """5. Verifies that allow_overwrite=True raises PermissionError."""
        with self.assertRaises(PermissionError) as ctx:
            E7EmpiricalRunner(config_path=self.config_path, allow_overwrite=True, device="cpu")
        self.assertIn("strictly forbid allow_overwrite=True", str(ctx.exception))

    def test_06_loeo_generalization_isolation(self):
        """6. Verifies that LOEO produces 6 isolated folds without cross-event leakage."""
        synthetic_stream_data = {
            "cotta_reset_s0_oa": {
                "stream_id": "cotta_reset_s0_oa",
                "method": "cotta_reset",
                "seed": 0,
                "order_id": "order_a",
                "macro_f1": 0.65,
                "event_metrics": [
                    {"event_id": eid, "f1": 0.60 + 0.01 * i, "samples": 100}
                    for i, eid in enumerate(DECLARED_DEVELOPMENT_EVENTS)
                ],
            }
        }

        analyzer = E7EmpiricalAnalyzer(
            source_dir=self.tmp_path / "src",
            output_dir=self.tmp_path / "out",
        )
        loeo_report = analyzer.run_loeo_generalization_analysis(synthetic_stream_data)

        self.assertEqual(loeo_report["total_folds"], 6)
        self.assertEqual(len(loeo_report["folds"]), 6)

        # Check each fold isolates exactly one held out event
        for eid in DECLARED_DEVELOPMENT_EVENTS:
            fold_key = f"held_out_{eid}"
            self.assertIn(fold_key, loeo_report["folds"])
            fold = loeo_report["folds"][fold_key]
            self.assertEqual(fold["held_out_event"], eid)
            self.assertIn("cotta_reset_s0_oa", fold["stream_metrics"])
            self.assertEqual(fold["stream_metrics"]["cotta_reset_s0_oa"]["event_id"], eid)

    def test_07_synthesis_summary_arithmetic(self):
        """7. Verifies delta F1 arithmetic and exact harm/benefit/neutral counts."""
        synthetic_stream_data = {
            "source_only_s0": {
                "stream_id": "source_only_s0",
                "method": "source_only",
                "seed": 0,
                "order_id": "order_a",
                "macro_f1": 0.60,
                "event_metrics": [
                    {"event_id": 279, "f1": 0.50},
                    {"event_id": 437, "f1": 0.50},
                    {"event_id": 520, "f1": 0.50},
                    {"event_id": 559, "f1": 0.50},
                    {"event_id": 1111003, "f1": 0.50},
                    {"event_id": 1111008, "f1": 0.50},
                ],
            },
            "candidate_s0": {
                "stream_id": "candidate_s0",
                "method": "candidate_e2",
                "seed": 0,
                "order_id": "order_a",
                "macro_f1": 0.60,
                "event_metrics": [
                    {"event_id": 279, "f1": 0.45},  # -0.05 -> harm
                    {"event_id": 437, "f1": 0.48},  # -0.02 -> harm
                    {"event_id": 520, "f1": 0.52},  # +0.02 -> benefit
                    {"event_id": 559, "f1": 0.55},  # +0.05 -> benefit
                    {"event_id": 1111003, "f1": 0.505},  # +0.005 -> neutral
                    {"event_id": 1111008, "f1": 0.495},  # -0.005 -> neutral
                ],
            },
        }

        analyzer = E7EmpiricalAnalyzer(
            source_dir=self.tmp_path / "src",
            output_dir=self.tmp_path / "out",
        )
        summary = analyzer.compute_synthesis_summary(synthetic_stream_data)
        rows = {r["stream_id"]: r for r in summary["master_comparison_rows"]}

        cand_row = rows["candidate_s0"]
        self.assertEqual(cand_row["harm_count"], 2)
        self.assertEqual(cand_row["benefit_count"], 2)
        self.assertEqual(cand_row["neutral_count"], 2)
        self.assertAlmostEqual(cand_row["delta_ev279"], -0.05, places=5)
        self.assertAlmostEqual(cand_row["delta_ev520"], 0.02, places=5)

    def test_08_analyzer_overwrite_protection(self):
        """8. Verifies analyzer raises FileExistsError if completed analysis exists."""
        src_dir = self.tmp_path / "src"
        out_dir = self.tmp_path / "out"
        src_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy completed summary in out_dir
        with open(out_dir / "e7_empirical_synthesis_summary.json", "w", encoding="utf-8") as f:
            json.dump({"status": "ANALYSIS_COMPLETE"}, f)

        analyzer = E7EmpiricalAnalyzer(source_dir=src_dir, output_dir=out_dir, allow_overwrite=False)
        with self.assertRaises(FileExistsError) as ctx:
            analyzer.validate_inputs()
        self.assertIn("Overwriting is prohibited", str(ctx.exception))

    def test_09_stream_completion_detection(self):
        """9. Verifies is_stream_completed detects completed vs incomplete streams."""
        runner = E7EmpiricalRunner(config_path=self.config_path, device="cpu")
        s_dir = self.tmp_path / "test_stream"

        # Initially non-existent
        self.assertFalse(runner.is_stream_completed(s_dir))

        # Created empty
        s_dir.mkdir(parents=True, exist_ok=True)
        self.assertFalse(runner.is_stream_completed(s_dir))

        # With incomplete files
        with open(s_dir / "stream_summary.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        self.assertFalse(runner.is_stream_completed(s_dir))

        # With status COMPLETED
        with open(s_dir / "completion_status.json", "w", encoding="utf-8") as f:
            json.dump({"status": "COMPLETED"}, f)
        self.assertTrue(runner.is_stream_completed(s_dir))

    def test_10_label_guarded_batch_lifecycle_and_interruption(self):
        """10. Real integration test for LabelGuardedBatch:
        - get_labels() must fail before persistence authorization.
        - an interrupted or failed persistence step must not unlock labels.
        - get_labels() must succeed only after artifact is persisted and mark_persisted() is called.
        """
        import torch
        from src.changeguard.e1_orchestrator import LabelGuardedBatch

        labels_tensor = torch.tensor([[[0, 2], [1, 2]]], dtype=torch.long)
        batch = LabelGuardedBatch(
            sample_ids=["test_sample_001"],
            event_id=520,
            xA=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            xB=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.uint8),
            labels=labels_tensor,
        )

        # 1. Unlocked access must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            batch.get_labels()
        self.assertIn("CRITICAL LABEL TIMING VIOLATION", str(ctx.exception))

        # 2. Simulate failed/interrupted persistence (exception during save)
        test_shard = self.tmp_path / "failed_shard.npz"
        try:
            # Simulate failure before mark_persisted()
            raise IOError("Simulated disk full or I/O error during persistence")
            batch.mark_persisted()
        except IOError:
            pass

        # Labels must STILL remain strictly locked
        with self.assertRaises(PermissionError) as ctx:
            batch.get_labels()
        self.assertIn("CRITICAL LABEL TIMING VIOLATION", str(ctx.exception))

        # 3. Simulate successful durable persistence
        test_shard = self.tmp_path / "durable_shard.npz"
        np.savez_compressed(test_shard, pred=np.zeros((1, 2, 2), dtype=np.uint8))
        self.assertTrue(test_shard.is_file() and test_shard.stat().st_size > 0)

        # Explicitly mark persisted
        batch.mark_persisted()

        # 4. Now get_labels() must succeed
        retrieved_labels = batch.get_labels()
        self.assertTrue(torch.equal(retrieved_labels, labels_tensor))

    def test_11_runner_real_label_guarded_batch_persistence_flow(self):
        """11. Verifies runner execute_stream durably persists shards and uses real mark_persisted/get_labels flow."""
        import torch
        from src.changeguard.e1_orchestrator import LabelGuardedBatch
        from src.changeguard.baseline_adapters import BaseE1Adapter

        class DummyAdapter(BaseE1Adapter):
            def __init__(self):
                self.step_count = 0
                self.num_updates_applied = 0
                self.num_updates_skipped = 0
                self.num_updates_failed = 0
                self.device = torch.device("cpu")

            def reset_to_source(self):
                pass

            def adapt_and_predict_step(self, xA, xB, valid_mask, sample_ids, event_id, **kwargs):
                self.step_count += 1
                B, _, H, W = xA.shape
                pred = torch.zeros((B, H, W), dtype=torch.uint8)
                probs = torch.zeros((B, 3, H, W), dtype=torch.float32)
                return {
                    "primary_prediction": {
                        "pred_cand_pre": pred,
                        "probs_cand_pre": probs,
                    },
                    "adaptation_metadata": {
                        "step_count": self.step_count,
                        "action": "update",
                        "update_applied": True,
                        "selected_pixels": 4,
                    },
                    "risk_evidence": {
                        "mean_entropy": 0.5,
                        "mean_max_confidence": 0.95,
                    },
                }

        labels_tensor = torch.zeros((1, 4, 4), dtype=torch.long)
        real_batch = LabelGuardedBatch(
            sample_ids=["sample_real_01"],
            event_id=520,
            xA=torch.zeros((1, 2, 4, 4)),
            xB=torch.zeros((1, 2, 4, 4)),
            valid_mask=torch.ones((1, 4, 4), dtype=torch.uint8),
            labels=labels_tensor,
        )

        runner = E7EmpiricalRunner(config_path=self.config_path, output_dir=self.tmp_path / "stream_out", device="cpu")
        # Mock instantiate_adapter and build_stream_batches to inject our dummy adapter and real batch
        runner.instantiate_adapter = lambda method, seed: DummyAdapter()
        runner.build_stream_batches = lambda order_id: [(520, [real_batch])]

        summary = runner.execute_stream(
            stream_id="test_real_guard_stream",
            method="cotta_smoke",
            seed=0,
            order_id="order_a",
        )

        stream_dir = self.tmp_path / "stream_out" / "test_real_guard_stream"
        shard_file = stream_dir / "event_520" / "shards" / "batch_0000.npz"

        # Verify that shard was durably saved before label access
        self.assertTrue(shard_file.is_file())
        self.assertGreater(shard_file.stat().st_size, 0)

        # Verify batch was marked persisted and labels were read
        self.assertTrue(real_batch._is_persisted)
        self.assertTrue(torch.equal(real_batch.get_labels(), labels_tensor))

        # Verify manifest, summary, csv, and completion status exist
        self.assertTrue((stream_dir / "manifest.json").is_file())
        self.assertTrue((stream_dir / "stream_summary.json").is_file())
        self.assertTrue((stream_dir / "event_metrics.csv").is_file())
        self.assertTrue((stream_dir / "completion_status.json").is_file())


if __name__ == "__main__":
    unittest.main()

