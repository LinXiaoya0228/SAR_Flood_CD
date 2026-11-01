"""ChangeGuard Week 12 Matched BN No-Update Diagnostic Tests.

Governing Config: configs/changeguard/week12_bn_diagnostic_dev6_v1.json

Focused CPU verification covering:
1. BN prediction context matches the adapted conditions (train mode, track_running_stats=False, running_mean/var=None).
2. Zero optimizer/backward calls (no grad, no updates applied, step count increments).
3. Exact parameter immutability (parameters before == parameters after).
4. Identical coverage, sample ordering, batch sizes, and tails as Week 12.
5. Predictions, step records, and progress persisted strictly before GT access.
6. Offline double-masking evaluation and confusion conservation (TP + FP + FN + TN == valid_pixels).
7. Preflight validation, device regex, and authorization status enforcement.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_week12_ablation import MockSNUNet
from tools.run_week12_bn_diagnostic import (
    MatchedBNNoUpdateAdapter,
    Week12BNDiagnosticRunner,
)


class TestWeek12BNDiagnostic(unittest.TestCase):
    """Focused synthetic test suite for Week 12 Matched BN No-Update diagnostic."""

    def setUp(self):
        self.config_path = REPO_ROOT / "configs/changeguard/week12_bn_diagnostic_dev6_v1.json"
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_cfg = json.load(f)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="w12_bn_test_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _generate_mock_events(self, num_samples_per_event: int = 18) -> dict:
        mock_data = {}
        for ev in [279, 437, 520, 559, 1111003, 1111008]:
            recs = []
            for i in range(num_samples_per_event):
                label = torch.zeros((32, 32), dtype=torch.int64)
                valid_mask = torch.ones((32, 32), dtype=torch.uint8)
                valid_mask[0, 0:5] = 0
                label[0, 5:10] = 3
                label[10:15, 10:15] = 2

                recs.append({
                    "sample_id": f"s{ev}_{i:04d}",
                    "event_id": ev,
                    "xA": torch.randn((2, 32, 32), dtype=torch.float32),
                    "xB": torch.randn((2, 32, 32), dtype=torch.float32),
                    "valid_mask": valid_mask,
                    "label": label,
                })
            mock_data[ev] = recs
        return mock_data

    # -------------------------------------------------------------------------
    # 1. BN Context & Module State
    # -------------------------------------------------------------------------
    def test_01_bn_prediction_context_matches_adapted_conditions(self):
        """Candidate BN modules are in train() mode with track_running_stats=False and None buffers."""
        model = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        adapter = MatchedBNNoUpdateAdapter(model=model, device="cpu")

        # Global model in eval mode
        self.assertFalse(adapter.model.training)

        # BN modules in train mode with tracking disabled
        bn_count = 0
        for name, module in adapter.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_count += 1
                self.assertTrue(module.training)
                self.assertFalse(module.track_running_stats)
                self.assertIsNone(module.running_mean)
                self.assertIsNone(module.running_var)

        self.assertGreater(bn_count, 0)

    # -------------------------------------------------------------------------
    # 2. Zero Optimizer/Backward Calls & Exact Parameter Immutability
    # -------------------------------------------------------------------------
    def test_02_zero_backward_and_exact_parameter_immutability(self):
        """Adapter performs zero backward passes, zero updates, and parameters remain bit-identical."""
        model = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

        adapter = MatchedBNNoUpdateAdapter(model=model, device="cpu")

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Run 5 consecutive steps
        for step in range(5):
            res = adapter.adapt_and_predict(xA, xB, sar_valid_mask=mask)
            meta = res["adaptation_metadata"]
            self.assertEqual(meta["action"], "bn_eval_no_update")
            self.assertFalse(meta["update_applied"])
            self.assertIsNone(meta["loss"])
            self.assertEqual(meta["selected_pixels"], 0)
            self.assertEqual(meta["selected_pixel_fraction"], 0.0)

        self.assertEqual(adapter.step_count, 5)
        self.assertEqual(adapter.num_updates_applied, 0)
        self.assertEqual(adapter.num_updates_skipped, 0)
        self.assertEqual(adapter.num_updates_failed, 0)

        # Verify bit-identical parameters
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, params_before[n]))

    # -------------------------------------------------------------------------
    # 3. Coverage, Batch Boundaries, and Tails
    # -------------------------------------------------------------------------
    def test_03_coverage_and_batch_boundaries_matching_week12(self):
        """Processes all 6 events with exact batch sizes [16, 2] on 18-sample events."""
        out_dir = self.temp_dir / "out_coverage"
        mock_data = self._generate_mock_events(num_samples_per_event=18)

        runner = Week12BNDiagnosticRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )

        res = runner.run_diagnostic(mock_records_by_event=mock_data)
        self.assertEqual(res["condition_id"], "matched_bn_no_update_v1")
        self.assertEqual(res["events_evaluated"], [279, 437, 520, 559, 1111003, 1111008])
        self.assertEqual(res["total_samples"], 18 * 6)
        self.assertEqual(res["total_batches"], 2 * 6)

        for ev_row in res["event_rows"]:
            self.assertEqual(ev_row["sample_count"], 18)
            self.assertEqual(ev_row["batch_sizes"], [16, 2])
            self.assertEqual(ev_row["updates_applied"], 0)

    # -------------------------------------------------------------------------
    # 4. Pre-GT Persistence Verification
    # -------------------------------------------------------------------------
    def test_04_pre_gt_persistence(self):
        """Frozen predictions and step records are saved before offline GT scoring."""
        out_dir = self.temp_dir / "out_persistence"
        mock_data = self._generate_mock_events(num_samples_per_event=6)

        runner = Week12BNDiagnosticRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )

        from src.data.evaluation_dataset import KuroEvaluationDataset
        orig_getitem = KuroEvaluationDataset.__getitem__

        gt_accessed = False
        def spying_getitem(ds_self, idx):
            nonlocal gt_accessed
            if getattr(ds_self, "load_labels", False):
                gt_accessed = True
                pred_file = out_dir / "matched_bn_no_update_v1" / "event_279" / "frozen_predictions.npz"
                step_file = out_dir / "matched_bn_no_update_v1" / "event_279" / "step_records.json"
                assert pred_file.is_file(), "GT accessed before predictions saved!"
                assert step_file.is_file(), "GT accessed before step records saved!"
            return orig_getitem(ds_self, idx)

        with patch.object(KuroEvaluationDataset, "__getitem__", spying_getitem):
            runner.run_diagnostic(mock_records_by_event=mock_data)

        self.assertTrue(gt_accessed)

    # -------------------------------------------------------------------------
    # 5. Confusion Conservation & Valid-Only Pixels
    # -------------------------------------------------------------------------
    def test_05_confusion_conservation(self):
        """TP + FP + FN + TN == valid_pixels for all events and pooled summary."""
        out_dir = self.temp_dir / "out_conservation"
        mock_data = self._generate_mock_events(num_samples_per_event=8)

        runner = Week12BNDiagnosticRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )
        res = runner.run_diagnostic(mock_records_by_event=mock_data)

        pm = res["pooled_metrics"]
        self.assertEqual(pm["tp"] + pm["fp"] + pm["fn"] + pm["tn"], pm["valid_pixels"])

        for ev in res["event_rows"]:
            self.assertEqual(ev["tp"] + ev["fp"] + ev["fn"] + ev["tn"], ev["valid_pixels"])

    # -------------------------------------------------------------------------
    # 6. Preflight & Authorization
    # -------------------------------------------------------------------------
    def test_06_preflight_and_device_syntax(self):
        """Validates device syntax, preflight, and status authorization."""
        cfg_bad = copy.deepcopy(self.base_cfg)
        cfg_bad["status"] = "UNAUTHORIZED_STATUS"
        p_bad = self.temp_dir / "bad_config.json"
        with open(p_bad, "w", encoding="utf-8") as f:
            json.dump(cfg_bad, f)

        with self.assertRaises(PermissionError) as ctx:
            Week12BNDiagnosticRunner(config_path=p_bad, output_dir=self.temp_dir / "out", device="cuda:0", synthetic_mock=False)
        self.assertIn("Real-data execution rejected", str(ctx.exception))

        for bad_dev in ["cpu", "mps", "cuda", "cuda:-1"]:
            with self.assertRaises(ValueError):
                Week12BNDiagnosticRunner(config_path=self.config_path, output_dir=self.temp_dir / "out", device=bad_dev, synthetic_mock=False)


if __name__ == "__main__":
    unittest.main()

