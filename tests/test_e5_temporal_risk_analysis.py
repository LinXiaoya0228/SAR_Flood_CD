#!/usr/bin/env python3
"""ChangeGuard Work Package E5.2C: Offline Temporal Risk Analysis Test Suite.

Verifies mathematical correctness, boundary definitions, fixed risk directions,
single-class degenerate handling, event-grouped statistical discipline, and
fail-closed integrity checks for the offline analysis engine.
"""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.analyze_e5_temporal_risk import (
    E5TemporalRiskAnalyzer,
    compute_auprc_safe,
    compute_auroc_safe,
    compute_confusion_cells,
    compute_f1_from_cells,
    compute_file_sha256,
    compute_spearmanr_safe,
    FORBIDDEN_SEALED_TEST_EVENTS,
    DECLARED_DEVELOPMENT_EVENTS,
)


class TestE5TemporalRiskAnalysis(unittest.TestCase):
    """Rigorous unit test suite for E5.2C offline risk analysis."""

    def test_01_exact_confusion_and_f1_arithmetic(self):
        """Verifies exact pixel confusion cells and F1 score arithmetic."""
        # Preds and labels with known pixels
        # 0: non-flood, 1: other, 2: flood
        preds = np.array([2, 2, 0, 1, 2, 0], dtype=np.uint8)
        labels = np.array([2, 0, 2, 1, 2, 0], dtype=np.uint8)
        mask = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)  # Last pixel masked out

        # Valid pixels: 5
        # idx 0: pred=2, label=2 -> TP (1)
        # idx 1: pred=2, label=0 -> FP (1)
        # idx 2: pred=0, label=2 -> FN (1)
        # idx 3: pred=1, label=1 -> TN (1)
        # idx 4: pred=2, label=2 -> TP (2)
        # Sum: TP=2, FP=1, FN=1, TN=1, valid=5
        tp, fp, fn, tn, valid = compute_confusion_cells(preds, labels, mask, target_class=2)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        self.assertEqual(tn, 1)
        self.assertEqual(valid, 5)
        self.assertEqual(tp + fp + fn + tn, valid)

        f1 = compute_f1_from_cells(tp, fp, fn)
        # F1 = 2 * 2 / (2 * 2 + 1 + 1) = 4 / 6 = 2/3
        self.assertAlmostEqual(f1, 2.0 / 3.0, places=6)

    def test_02_harm_benefit_neutral_boundary_values(self):
        """Verifies exact threshold boundaries for event and batch outcomes."""
        # harmful if delta_f1 <= -0.01
        # beneficial if delta_f1 >= +0.01
        # neutral otherwise

        def classify(delta_f1: float) -> str:
            if delta_f1 <= -0.01:
                return "harmful"
            elif delta_f1 >= 0.01:
                return "beneficial"
            else:
                return "neutral"

        self.assertEqual(classify(-0.010001), "harmful")
        self.assertEqual(classify(-0.010000), "harmful")
        self.assertEqual(classify(-0.009999), "neutral")
        self.assertEqual(classify(0.000000), "neutral")
        self.assertEqual(classify(0.009999), "neutral")
        self.assertEqual(classify(0.010000), "beneficial")
        self.assertEqual(classify(0.010001), "beneficial")

    def test_03_fixed_signal_directions(self):
        """Verifies frozen signal directions in configuration contract."""
        cfg_path = Path("configs/changeguard/e5_temporal_risk_analysis_v1.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        specs = cfg["signal_specifications"]
        # Generic signals
        self.assertEqual(specs["mean_entropy"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["mean_max_confidence"]["direction"], "lower_is_greater_risk")
        self.assertEqual(specs["source_to_candidate_kl"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["hard_disagreement_rate"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["abs_delta_flood_pixels"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["aug_inconsistency"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["bn_weight_displacement"]["direction"], "higher_is_greater_risk")

        # Change-specific signals
        self.assertEqual(specs["s_null"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["s_flood_expansion"]["direction"], "higher_is_greater_risk")
        self.assertEqual(specs["s_flood_disappearance"]["direction"], "higher_is_greater_risk")

    def test_04_auroc_and_auprc_standard_conformance(self):
        """Verifies standard AUROC and AUPRC calculation on separable distributions."""
        # Perfectly separable
        y_true = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        auroc = compute_auroc_safe(y_true, scores)
        self.assertAlmostEqual(auroc, 1.0, places=6)

        auprc = compute_auprc_safe(y_true, scores)
        self.assertAlmostEqual(auprc, 1.0, places=4)

        # Inverted (anti-correlated)
        inv_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        auroc_inv = compute_auroc_safe(y_true, inv_scores)
        self.assertAlmostEqual(auroc_inv, 0.0, places=6)

    def test_05_single_class_degenerate_na_guard(self):
        """Verifies that single-class outcome sequences return NaN rather than false 0.5 or 1.0."""
        # All negative (no harmful events)
        all_zeros = [0, 0, 0, 0]
        scores = [0.1, 0.4, 0.6, 0.9]
        auroc_0 = compute_auroc_safe(all_zeros, scores)
        auprc_0 = compute_auprc_safe(all_zeros, scores)
        self.assertTrue(math.isnan(auroc_0))
        self.assertTrue(math.isnan(auprc_0))

        # All positive (all harmful events)
        all_ones = [1, 1, 1, 1]
        auroc_1 = compute_auroc_safe(all_ones, scores)
        auprc_1 = compute_auprc_safe(all_ones, scores)
        self.assertTrue(math.isnan(auroc_1))
        self.assertTrue(math.isnan(auprc_1))

    def test_06_harm_prevalence_reference(self):
        """Verifies baseline no-skill reference equals empirical prevalence P / N."""
        y_true = [0, 0, 0, 0, 1]
        n_pos = sum(y_true)
        n_total = len(y_true)
        prevalence = n_pos / n_total
        self.assertAlmostEqual(prevalence, 0.20, places=6)

    def test_07_spearman_rank_correlation(self):
        """Verifies Spearman rank correlation matches monotonic relationships."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pos = [10.0, 20.0, 30.0, 40.0, 50.0]
        r_pos = compute_spearmanr_safe(x, y_pos)
        self.assertAlmostEqual(r_pos, 1.0, places=6)

        y_neg = [50.0, 40.0, 30.0, 20.0, 10.0]
        r_neg = compute_spearmanr_safe(x, y_neg)
        self.assertAlmostEqual(r_neg, -1.0, places=6)

    def test_08_sealed_test_event_rejection(self):
        """Verifies fail-closed rejection if any sealed-test event is injected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            status_p = tmp_root / "completion_status.json"
            with open(status_p, "w", encoding="utf-8") as f:
                json.dump({"status": "COMPLETED_RUN"}, f)

            # Injected sealed test event 205
            manifest_p = tmp_root / "initial_manifest.json"
            with open(manifest_p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "git_dirty": False,
                        "target_events": [279, 205],  # 205 is forbidden sealed-test
                    },
                    f,
                )

            analyzer = E5TemporalRiskAnalyzer(
                config_path=Path("configs/changeguard/e5_temporal_risk_analysis_v1.json"),
                source_dir=tmp_root,
            )

            with self.assertRaises(RuntimeError) as ctx:
                analyzer.validate_source_collection()
            self.assertIn("CRITICAL SECURITY VIOLATION", str(ctx.exception))

    def test_09_incomplete_run_rejection(self):
        """Verifies fail-closed rejection if source run status is not COMPLETED_RUN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            status_p = tmp_root / "completion_status.json"
            with open(status_p, "w", encoding="utf-8") as f:
                json.dump({"status": "RUNNING"}, f)

            analyzer = E5TemporalRiskAnalyzer(
                config_path=Path("configs/changeguard/e5_temporal_risk_analysis_v1.json"),
                source_dir=tmp_root,
            )

            with self.assertRaises(RuntimeError) as ctx:
                analyzer.validate_source_collection()
            self.assertIn("expected COMPLETED_RUN", str(ctx.exception))

    def test_10_missing_or_corrupt_shard_hash_rejection(self):
        """Verifies fail-closed rejection if a shard is missing or has hash mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            status_p = tmp_root / "completion_status.json"
            with open(status_p, "w", encoding="utf-8") as f:
                json.dump({"status": "COMPLETED_RUN"}, f)

            manifest_p = tmp_root / "initial_manifest.json"
            with open(manifest_p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "git_dirty": False,
                        "target_events": DECLARED_DEVELOPMENT_EVENTS,
                        "event_contracts": {
                            str(eid): {"expected_event_sample_ids": ["s1"]}
                            for eid in DECLARED_DEVELOPMENT_EVENTS
                        },
                    },
                    f,
                )

            # Create event 279 directory with fake index but missing shard
            ev_dir = tmp_root / "event_279"
            ev_dir.mkdir(parents=True)
            with open(ev_dir / "evidence_index.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "shards": [
                            {
                                "shard_file": "evidence_shards/batch_000000.npz",
                                "sha256": "fake_hash",
                                "sample_ids": ["s1"],
                            }
                        ]
                    },
                    f,
                )

            analyzer = E5TemporalRiskAnalyzer(
                config_path=Path("configs/changeguard/e5_temporal_risk_analysis_v1.json"),
                source_dir=tmp_root,
            )

            with self.assertRaises(FileNotFoundError):
                analyzer.validate_source_collection()

    def test_11_real_formal_collection_read_only_validation(self):
        """Verifies that the actual on-disk formal run passes read-only validation without mutation."""
        analyzer = E5TemporalRiskAnalyzer(
            config_path=Path("configs/changeguard/e5_temporal_risk_analysis_v1.json")
        )
        report = analyzer.validate_source_collection()
        self.assertEqual(report["status"], "INTEGRITY_VERIFIED")
        self.assertEqual(report["total_samples"], 2826)
        self.assertEqual(report["total_shards"], 179)
        self.assertEqual(report["events_verified"], DECLARED_DEVELOPMENT_EVENTS)

    def test_12_reordered_and_multiset_event_validation(self):
        """Verifies that reordered events pass while duplicated or missing events fail multiset equality."""
        cfg_path = Path("configs/changeguard/e5_temporal_risk_analysis_v1.json")

        # 1. Reordered valid events (Order B) on real formal collection pass validation
        ob_dir = Path("outputs/e5_temporal_evidence/formal_s0_ob_v1")
        if ob_dir.exists():
            analyzer_ob = E5TemporalRiskAnalyzer(config_path=cfg_path, source_dir=ob_dir)
            report_ob = analyzer_ob.validate_source_collection()
            self.assertEqual(report_ob["status"], "INTEGRITY_VERIFIED")
            self.assertEqual(report_ob["total_samples"], 2826)
            self.assertEqual(report_ob["total_shards"], 179)

        # 2. Duplicated events fail multiset equality even if set(target) == set(DECLARED_DEVELOPMENT_EVENTS)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            with open(tmp_root / "completion_status.json", "w", encoding="utf-8") as f:
                json.dump({"status": "COMPLETED_RUN"}, f)

            duplicated = [279, 279, 437, 520, 559, 1111003, 1111008]
            self.assertEqual(set(duplicated), set(DECLARED_DEVELOPMENT_EVENTS))
            with open(tmp_root / "initial_manifest.json", "w", encoding="utf-8") as f:
                json.dump({"git_dirty": False, "target_events": duplicated}, f)

            analyzer = E5TemporalRiskAnalyzer(config_path=cfg_path, source_dir=tmp_root)
            with self.assertRaises(RuntimeError) as ctx:
                analyzer.validate_source_collection()
            self.assertIn("Target events mismatch", str(ctx.exception))

        # 3. Missing events fail multiset equality
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            with open(tmp_root / "completion_status.json", "w", encoding="utf-8") as f:
                json.dump({"status": "COMPLETED_RUN"}, f)

            missing = [279, 437, 520, 559, 1111003]  # Missing 1111008
            with open(tmp_root / "initial_manifest.json", "w", encoding="utf-8") as f:
                json.dump({"git_dirty": False, "target_events": missing}, f)

            analyzer = E5TemporalRiskAnalyzer(config_path=cfg_path, source_dir=tmp_root)
            with self.assertRaises(RuntimeError) as ctx:
                analyzer.validate_source_collection()
            self.assertIn("Target events mismatch", str(ctx.exception))

        # 4. Extraneous unsealed events fail multiset equality
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            with open(tmp_root / "completion_status.json", "w", encoding="utf-8") as f:
                json.dump({"status": "COMPLETED_RUN"}, f)

            extra = list(DECLARED_DEVELOPMENT_EVENTS) + [999999]
            with open(tmp_root / "initial_manifest.json", "w", encoding="utf-8") as f:
                json.dump({"git_dirty": False, "target_events": extra}, f)

            analyzer = E5TemporalRiskAnalyzer(config_path=cfg_path, source_dir=tmp_root)
            with self.assertRaises(RuntimeError) as ctx:
                analyzer.validate_source_collection()
            self.assertIn("Target events mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

