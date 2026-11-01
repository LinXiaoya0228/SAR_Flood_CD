#!/usr/bin/env python3
"""ChangeGuard Work Package E8: Confirmation Matrix Test Suite.

Tests governing Plan V3 Block E8 requirements:
1. Exact planned cell enumeration (57 reused, 12 new = 69 physical streams, 414 event rows).
2. Zero duplicate cells in ledger.
3. Reuse vs execute-new classification logic.
4. Sealed-test rejection and fail-closed quarantine.
5. Prediction persistence lifecycle and label isolation.
6. Interruption and resumption semantics.
7. Terminal-status preservation (failed v1 smoke retained).
8. Method, config, split, and checkpoint SHA-256 validation.
9. Metrics, event-row counts, and harm/benefit accounting.
10. LOEO jackknife sensitivity and explicit 'NA under Branch C' for policy thresholds.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.changeguard.e1_orchestrator import LabelGuardedBatch
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)
from tools.analyze_e8_confirmation_matrix import (
    E8ConfirmationAnalyzer,
    compute_f1_from_cells,
    compute_iou_from_cells,
    compute_precision_from_cells,
    compute_recall_from_cells,
)
from tools.run_e8_confirmation_matrix import (
    E8ConfirmationRunner,
    compute_file_sha256,
)


class TestE8ConfirmationMatrix(unittest.TestCase):
    """Test suite for Work Package E8 confirmation matrix."""

    def setUp(self) -> None:
        self.config_path = REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v1.json"
        self.ledger_path = REPO_ROOT / "configs/changeguard/e8_execution_ledger_v1.json"

        self.assertTrue(self.config_path.is_file(), f"Missing config: {self.config_path}")
        self.assertTrue(self.ledger_path.is_file(), f"Missing ledger: {self.ledger_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

    def test_01_exact_cell_enumeration_and_ledger_counts(self) -> None:
        """Verifies exact planned stream and event-row enumeration."""
        reused = self.ledger.get("reused_cells", [])
        new_cells = self.ledger.get("execute_new_cells", [])
        na_cells = self.ledger.get("not_applicable_cells", [])
        failed_cells = self.ledger.get("failed_and_retained_cells", [])
        incompat = self.ledger.get("incompatible_methods", [])

        # 57 Reused streams (3 Source-Only + 6 BN-No-Update + 12 AdaBN + 12 Tent + 12 EATA + 12 SAR)
        self.assertEqual(len(reused), 57, f"Expected 57 reused streams, got {len(reused)}")
        # 12 Execute-New streams (6 CoTTA Continual + 6 CoTTA Reset)
        self.assertEqual(len(new_cells), 12, f"Expected 12 execute-new streams, got {len(new_cells)}")

        total_streams = len(reused) + len(new_cells)
        self.assertEqual(total_streams, 69, f"Expected 69 total physical streams, got {total_streams}")

        total_reused_rows = sum(len(c["events"]) for c in reused)
        total_new_rows = sum(len(c["events"]) for c in new_cells)
        self.assertEqual(total_reused_rows, 342, f"Expected 342 reused event rows, got {total_reused_rows}")
        self.assertEqual(total_new_rows, 72, f"Expected 72 new event rows, got {total_new_rows}")
        self.assertEqual(total_reused_rows + total_new_rows, 414, "Expected 414 total event rows")

        self.assertEqual(len(na_cells), 2, "Expected 2 exploratory cells (BS=1, BS=4)")
        self.assertEqual(len(failed_cells), 1, "Expected 1 failed smoke cell retained")
        self.assertEqual(len(incompat), 5, "Expected 5 incompatible literature methods")

    def test_02_zero_duplicate_cells_in_ledger(self) -> None:
        """Verifies zero duplicate stream IDs or factor combinations in ledger."""
        all_cells = self.ledger["reused_cells"] + self.ledger["execute_new_cells"]
        stream_ids = [c["stream_id"] for c in all_cells]
        self.assertEqual(len(stream_ids), len(set(stream_ids)), "Duplicate stream IDs detected in ledger!")

        # Verify uniqueness of (method, seed, order, reset_mode) tuples
        factor_tuples = []
        for c in all_cells:
            t = (c["method"], c["seed"], c["order"], c.get("reset_mode", "invariant"), c.get("batch_size", 16))
            factor_tuples.append(t)
        self.assertEqual(len(factor_tuples), len(set(factor_tuples)), "Duplicate factor configurations detected!")

    def test_03_reuse_vs_execute_new_decisions(self) -> None:
        """Verifies reuse vs execute-new classification logic."""
        reused = self.ledger["reused_cells"]
        for c in reused:
            self.assertEqual(c["status"], "REUSED")
            art_dir = REPO_ROOT / c["artifact_directory"]
            self.assertTrue(art_dir.is_dir(), f"Reused directory missing: {art_dir}")
            has_record = (art_dir / "manifest.json").is_file() or (art_dir / "condition_summary.json").is_file()
            self.assertTrue(has_record, f"Missing manifest/summary in {art_dir}")

        new_cells = self.ledger["execute_new_cells"]
        for c in new_cells:
            self.assertEqual(c["status"], "EXECUTE_NEW")
            self.assertIn(c["method"], {"cotta_continual", "cotta_reset"})
            self.assertEqual(c["completion_status"], "PENDING_EXECUTION")

        # Confirm PA-TTA is NOT in execute-new cells
        new_methods = [c["method"] for c in new_cells]
        self.assertNotIn("pa_tta", new_methods)

    def test_04_sealed_test_rejection_and_quarantine(self) -> None:
        """Verifies that attempting to target sealed-test events fails closed immediately."""
        runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )

        # Inject a sealed-test event into target_events
        original_targets = list(runner.target_events)
        forbidden_event = sorted(list(FORBIDDEN_SEALED_TEST_EVENTS))[0]
        runner.target_events = [original_targets[0], forbidden_event]

        with self.assertRaises(RuntimeError) as ctx:
            runner.run_preflights()
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

    def test_05_label_isolation_and_lifecycle_guard(self) -> None:
        """Verifies that LabelGuardedBatch enforces persistence before label access."""
        x_pre = torch.zeros((2, 2, 64, 64), dtype=torch.float32)
        x_post = torch.zeros((2, 2, 64, 64), dtype=torch.float32)
        labels = torch.ones((2, 64, 64), dtype=torch.long)
        mask = torch.ones((2, 64, 64), dtype=torch.bool)
        sample_ids = ["s0", "s1"]

        batch = LabelGuardedBatch(
            sample_ids=sample_ids,
            event_id=520,
            xA=x_pre,
            xB=x_post,
            valid_mask=mask,
            labels=labels,
        )

        # Access before mark_persisted must fail closed
        with self.assertRaises(PermissionError):
            batch.get_labels()

        # Unlocking without persistence is prevented by runner flow; here we test mark_persisted() unlock
        batch.mark_persisted()
        y_out = batch.get_labels()
        self.assertEqual(y_out.shape, labels.shape)
        self.assertEqual(batch.valid_mask.shape, mask.shape)

    def test_06_resumption_and_interruption_behavior(self) -> None:
        """Verifies that runner detects completed vs interrupted stream directories."""
        runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Empty directory: not completed
            self.assertFalse(runner.is_stream_completed(tmp_path))

            # 2. Incomplete files: missing completion_status.json
            (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
            (tmp_path / "stream_summary.json").write_text("{}", encoding="utf-8")
            self.assertFalse(runner.is_stream_completed(tmp_path))

            # 3. Interrupted status: status != "COMPLETED"
            status_file = tmp_path / "completion_status.json"
            status_file.write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
            self.assertFalse(runner.is_stream_completed(tmp_path))

            # 4. Valid completed status
            status_file.write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
            self.assertTrue(runner.is_stream_completed(tmp_path))

    def test_07_terminal_status_and_failed_v1_preservation(self) -> None:
        """Verifies that failed smoke v1 artifacts remain preserved and intact."""
        v1_log = REPO_ROOT / "logs/e7_cotta_smoke/e7_cotta_smoke_v1.log"
        self.assertTrue(v1_log.is_file(), "Preserved v1 smoke log missing!")
        content = v1_log.read_text(encoding="utf-8")
        self.assertIn("AttributeError", content)
        self.assertIn("Terminal Status:  FAILURE", content)

        v1_out = REPO_ROOT / "outputs/e7_cotta_smoke_v1"
        self.assertTrue(v1_out.is_dir(), "Preserved v1 smoke output directory missing!")

    def test_08_checkpoint_manifest_and_split_hash_validation(self) -> None:
        """Verifies SHA-256 integrity of dataset manifest, split, and source checkpoints."""
        runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )
        preflights = runner.run_preflights()
        self.assertEqual(preflights["status"], "PREFLIGHTS_PASSED")

        # Verify checkpoint hashes match bit-for-bit
        for seed_idx, ckpt in preflights["checkpoints"].items():
            exp = self.config["source_checkpoints"]["seeds"][str(seed_idx)]["expected_sha256"]
            self.assertEqual(ckpt["sha256"], exp)

    def test_09_metrics_and_event_row_counts(self) -> None:
        """Verifies loading of reused streams and correctness of offline metric calculations."""
        analyzer = E8ConfirmationAnalyzer(
            ledger_path=self.ledger_path,
        )
        analyzer.load_stream_data()

        # The completed v1 output root now contributes all 12 execute-new CoTTA streams.
        self.assertEqual(len(analyzer.stream_records), 69)
        self.assertEqual(len(analyzer.event_records), 414)

        # Test metric calculation functions
        self.assertAlmostEqual(compute_f1_from_cells(tp=100, fp=20, fn=30), 200 / 250)
        self.assertAlmostEqual(compute_iou_from_cells(tp=100, fp=20, fn=30), 100 / 150)
        self.assertAlmostEqual(compute_precision_from_cells(tp=100, fp=20), 100 / 120)
        self.assertAlmostEqual(compute_recall_from_cells(tp=100, fn=30), 100 / 130)

        # Division by zero safety
        self.assertEqual(compute_f1_from_cells(0, 0, 0), 0.0)
        self.assertEqual(compute_iou_from_cells(0, 0, 0), 0.0)

    def test_10_loeo_jackknife_and_branch_c_na(self) -> None:
        """Verifies LOEO jackknife sensitivity and explicit NA under Branch C for policy thresholds."""
        analyzer = E8ConfirmationAnalyzer(
            ledger_path=self.ledger_path,
        )
        analyzer.load_stream_data()

        loeo_res = analyzer.compute_loeo_jackknife_sensitivity()
        self.assertEqual(loeo_res["policy_threshold_loeo"], "NA under Branch C")
        self.assertIn("Gate E proved unsupervised pre-update risk scores invert", loeo_res["policy_threshold_rationale"])

        jackknife = loeo_res["held_out_event_jackknife"]
        self.assertEqual(len(jackknife), 6)
        for eid in DEFAULT_TTA_DEV_EVENTS:
            self.assertIn(str(eid), jackknife)
            entry = jackknife[str(eid)]
            self.assertEqual(entry["held_out_event"], eid)
            self.assertEqual(len(entry["retained_events"]), 5)
            self.assertNotIn(eid, entry["retained_events"])

    def test_11_harm_benefit_breakdown_accounting(self) -> None:
        """Verifies harm/benefit/neutral accounting against source and bn_no_update controls."""
        analyzer = E8ConfirmationAnalyzer(
            ledger_path=self.ledger_path,
        )
        analyzer.load_stream_data()

        hb = analyzer.compute_harm_benefit_breakdown(delta_threshold=0.0100)
        vs_src = hb["vs_source_only"]
        self.assertGreater(len(vs_src), 0)

        for sid, counts in vs_src.items():
            self.assertEqual(
                counts["harms"] + counts["benefits"] + counts["neutrals"],
                counts["total_events"],
                f"Harm/benefit/neutral sum does not equal total events for {sid}!",
            )
            self.assertEqual(counts["total_events"], 6)

    def test_12_instantiate_model_and_cotta_adapter(self) -> None:
        """Verifies instantiate_model and instantiate_adapter work on seed 0."""
        runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )
        model = runner.instantiate_model(seed=0)
        self.assertIsNotNone(model)
        adapter = runner.instantiate_adapter("cotta_continual", model, seed=0)
        self.assertEqual(adapter.mode, "continual")
        self.assertAlmostEqual(adapter.learning_rate, 1e-4)
        self.assertAlmostEqual(adapter.ema_alpha, 0.999)
        self.assertAlmostEqual(adapter.restoration_prob, 0.01)
        self.assertAlmostEqual(adapter.confidence_threshold, 0.90)

        # Verify adapt_and_predict_step contract
        xA = torch.zeros(1, 2, 64, 64)
        xB = torch.zeros(1, 2, 64, 64)
        vmask = torch.ones(1, 64, 64, dtype=torch.uint8)
        res = adapter.adapt_and_predict_step(
            xA=xA,
            xB=xB,
            valid_mask=vmask,
            sample_ids=["dummy_sample_0"],
            event_id=520,
        )
        self.assertIn("primary_prediction", res)
        self.assertIn("pred_cand_pre", res["primary_prediction"])
        self.assertIn("adaptation_metadata", res)

    def test_13_canonical_v2_ledger_and_exported_artifacts(self) -> None:
        """Verifies canonical v2 matrix loading, artifact schema, and Branch C confirmation."""
        v2_ledger_path = REPO_ROOT / "configs/changeguard/e8_execution_ledger_v2.json"
        v2_out_dir = REPO_ROOT / "outputs/e8_confirmation_v2"
        self.assertTrue(v2_ledger_path.is_file(), f"Missing v2 ledger: {v2_ledger_path}")
        self.assertTrue(v2_out_dir.is_dir(), f"Missing v2 outputs: {v2_out_dir}")

        analyzer = E8ConfirmationAnalyzer(ledger_path=v2_ledger_path, output_dir=v2_out_dir)
        analyzer.load_stream_data()
        self.assertEqual(len(analyzer.stream_records), 69)
        self.assertEqual(len(analyzer.event_records), 414)

        # Verify exported files exist and are complete
        matrix_csv = v2_out_dir / "e8_consolidated_matrix_v2.csv"
        matrix_json = v2_out_dir / "e8_consolidated_matrix_v2.json"
        summary_csv = v2_out_dir / "e8_method_summary_v2.csv"
        summary_json = v2_out_dir / "e8_method_summary_v2.json"
        paired_csv = v2_out_dir / "e8_paired_event_comparisons_v2.csv"
        uncertainty_json = v2_out_dir / "e8_uncertainty_loeo_v2.json"
        branch_json = v2_out_dir / "e8_branch_rule_evaluation_v2.json"

        for p in [matrix_csv, matrix_json, summary_csv, summary_json, paired_csv, uncertainty_json, branch_json]:
            self.assertTrue(p.is_file(), f"Missing exported artifact: {p}")

        with open(branch_json, "r", encoding="utf-8") as f:
            b_eval = json.load(f)

        self.assertEqual(b_eval["branch_decision"], "BRANCH_C_CONFIRMED")
        self.assertFalse(b_eval["branch_b_promoted"])
        self.assertEqual(b_eval["terminal_verdict"], "E8 CLOSED — BRANCH C CONFIRMATION COMPLETE")
        rules = b_eval["branch_b_promotion_criteria"]
        self.assertEqual(rules["rule_1_statistically_significant_improvement_over_source_only"]["status"], "FAILED")
        self.assertEqual(rules["rule_2_consistent_improvement_over_matched_bn_control"]["status"], "FAILED")
        self.assertEqual(rules["rule_3_adaptive_controller_feasibility"]["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
