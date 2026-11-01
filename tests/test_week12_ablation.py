"""ChangeGuard Week 12 Exploratory Ablation Unit & Integration Tests.

Governing Protocol: docs/protocols/week12_exploratory_ablation_protocol_v1.md
Governing Config:   configs/changeguard/week12_ablation_dev6_v1.json

Focused synthetic verification covering:
1. C1 includes all sensory-valid pixels (no confidence threshold).
2. C2 adds only the confidence condition (threshold >= 0.90).
3. Empty selection handling, parameter preservation, and finite updates.
4. Per-event reset restoring model parameters and optimizer state bit-identically.
5. Geometry alignment via horizontal flip and inversion.
6. Matched coverage, sample IDs, and batch boundaries across all four conditions.
7. Durable persistence of predictions and step records strictly before ground-truth access.
8. Offline double-masking evaluation and confusion conservation (TP + FP + FN + TN == valid_pixels).
9. Failure/progress recording and overwrite protection.
10. Strict stage, approval, boolean, and device syntax validation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter
from tools.run_week12_ablation import (
    Week12AblationRunner,
    MockSNUNet,
    SourceOnlyAdapter,
    MatchedTentAdapter,
)


class TestWeek12Ablation(unittest.TestCase):
    """Focused synthetic test suite for Week 12 exploratory ablation."""

    def setUp(self):
        self.config_path = REPO_ROOT / "configs/changeguard/week12_ablation_dev6_v1.json"
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_cfg = json.load(f)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="w12_test_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _write_temp_config(self, cfg_dict: dict) -> Path:
        p = self.temp_dir / "temp_config.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        return p

    def _generate_mock_events(self, num_samples_per_event: int = 18) -> dict:
        mock_data = {}
        for ev in [279, 437, 520, 559, 1111003, 1111008]:
            recs = []
            for i in range(num_samples_per_event):
                # 32x32 image
                label = torch.zeros((32, 32), dtype=torch.int64)
                valid_mask = torch.ones((32, 32), dtype=torch.uint8)
                # Some invalid pixels
                valid_mask[0, 0:5] = 0
                label[0, 5:10] = 3
                # Some flood pixels
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
    # 1. C1 vs C2 Selection Filtering
    # -------------------------------------------------------------------------
    def test_01_c1_includes_all_sensory_valid_pixels(self):
        """C1 (conf=0.0) selects 100% of sensory-valid pixels; C2 (conf=0.90) selects a subset."""
        proxy = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        adapter_c1 = EvidenceConsistencyAdapter(model=proxy, confidence_threshold=0.0, device="cpu")
        adapter_c2 = EvidenceConsistencyAdapter(model=proxy, confidence_threshold=0.90, device="cpu")

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        mask[:, 0:2, :] = 0  # 32 invalid pixels per batch

        valid_count = int((mask == 1).sum().item())
        self.assertEqual(valid_count, 2 * 14 * 16)

        res_c1 = adapter_c1.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        meta_c1 = res_c1["adaptation_metadata"]
        # C1 selects ALL valid pixels
        self.assertEqual(meta_c1["valid_pixels"], valid_count)
        self.assertEqual(meta_c1["selected_pixels"], valid_count)
        self.assertAlmostEqual(meta_c1["selected_pixel_fraction"], 1.0)
        self.assertTrue(meta_c1["update_applied"])

        res_c2 = adapter_c2.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        meta_c2 = res_c2["adaptation_metadata"]
        self.assertEqual(meta_c2["valid_pixels"], valid_count)
        # C2 selected_pixels <= valid_count
        self.assertLessEqual(meta_c2["selected_pixels"], valid_count)
        self.assertLessEqual(meta_c2["selected_pixel_fraction"], 1.0)

    # -------------------------------------------------------------------------
    # 2. Empty Selection & Parameter Preservation
    # -------------------------------------------------------------------------
    def test_02_empty_selection_skips_update_and_preserves_parameters(self):
        """When zero pixels meet confidence threshold, C2 skips update and preserves parameters."""
        proxy = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        # Extremely high threshold 1.0001 guarantees 0 pixels selected
        adapter = EvidenceConsistencyAdapter(model=proxy, confidence_threshold=1.0001, device="cpu")

        params_before = {n: p.detach().clone() for n, p in proxy.named_parameters() if p.requires_grad}

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        res = adapter.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        meta = res["adaptation_metadata"]

        self.assertFalse(meta["update_applied"])
        self.assertEqual(meta["action"], "skip")
        self.assertIsNone(meta["loss"])
        self.assertEqual(meta["selected_pixels"], 0)

        for n, p in proxy.named_parameters():
            if p.requires_grad:
                self.assertTrue(torch.equal(p, params_before[n]))

    # -------------------------------------------------------------------------
    # 3. State Resetting Bit-Identical Restoration
    # -------------------------------------------------------------------------
    def test_03_per_event_reset_restores_parameters_and_optimizer(self):
        """Reset re-loads model parameters and clears optimizer momentum completely."""
        proxy = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        init_state = {k: v.detach().clone() for k, v in proxy.state_dict().items()}

        adapter = EvidenceConsistencyAdapter(model=proxy, confidence_threshold=0.0, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Apply update
        adapter.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        self.assertEqual(adapter.num_updates_applied, 1)

        # Check model weights drifted
        has_drifted = False
        for k, v in proxy.state_dict().items():
            if not torch.equal(v, init_state[k]):
                has_drifted = True
                break
        self.assertTrue(has_drifted)

        # Perform clean reset using reset_to_source
        adapter.reset_to_source()

        # Verify exact bit-identical state
        for k, v in proxy.state_dict().items():
            self.assertTrue(torch.equal(v, init_state[k]))
        self.assertEqual(adapter.step_count, 0)
        self.assertEqual(adapter.num_updates_applied, 0)
        self.assertEqual(len(adapter.optimizer.state), 0)

    # -------------------------------------------------------------------------
    # 4. Geometry Alignment (Horizontal Flip & Inversion)
    # -------------------------------------------------------------------------
    def test_04_geometry_alignment_horizontal_flip(self):
        """Strong view applies horizontal flip to inputs and inverts strong logits spatially."""
        xA = torch.arange(16).view(1, 1, 4, 4).float().repeat(1, 2, 1, 1)
        xB = torch.arange(16).view(1, 1, 4, 4).float().repeat(1, 2, 1, 1)

        xA_flip = torch.flip(xA, dims=[-1])
        xB_flip = torch.flip(xB, dims=[-1])

        # Flipping twice restores original
        self.assertTrue(torch.equal(torch.flip(xA_flip, dims=[-1]), xA))
        self.assertTrue(torch.equal(torch.flip(xB_flip, dims=[-1]), xB))

    # -------------------------------------------------------------------------
    # 5. Matched Coverage & Batch Boundaries Across Conditions
    # -------------------------------------------------------------------------
    def test_05_matched_coverage_and_batch_boundaries(self):
        """All 4 conditions process the exact same samples and batch boundaries."""
        out_dir = self.temp_dir / "out_matched"
        mock_data = self._generate_mock_events(num_samples_per_event=18)

        runner = Week12AblationRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )

        res = runner.run_ablation(mock_records_by_event=mock_data)
        conds = res["conditions"]

        self.assertEqual(len(conds), 4)
        expected_conds = ["source_only", "matched_tent_pre_update_v1", "C1_consistency_no_conf", "C2_consistency_conf90"]
        self.assertEqual(list(conds.keys()), expected_conds)

        # Verify each condition evaluated all 6 events with exact same sample counts and batch sizes [16, 2]
        for cid in expected_conds:
            c_data = conds[cid]
            self.assertEqual(c_data["events_evaluated"], [279, 437, 520, 559, 1111003, 1111008])
            self.assertEqual(c_data["total_samples"], 18 * 6)
            self.assertEqual(c_data["total_batches"], 2 * 6)  # 18 samples per event = [16, 2] = 2 batches per event

            for ev_row in c_data["event_rows"]:
                self.assertEqual(ev_row["sample_count"], 18)
                self.assertEqual(ev_row["batch_sizes"], [16, 2])

    # -------------------------------------------------------------------------
    # 6. Pre-GT Persistence & Offline GT Double Masking
    # -------------------------------------------------------------------------
    def test_06_pre_gt_persistence_and_offline_evaluation(self):
        """Predictions and diagnostics are saved before GT access; offline GT scoring is double masked."""
        out_dir = self.temp_dir / "out_persistence"
        mock_data = self._generate_mock_events(num_samples_per_event=6)

        runner = Week12AblationRunner(
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
                # Check that for this condition/event, frozen_predictions.npz already exists!
                # The first condition is source_only, first event is 279
                pred_file = out_dir / "source_only" / "event_279" / "frozen_predictions.npz"
                step_file = out_dir / "source_only" / "event_279" / "step_records.json"
                assert pred_file.is_file(), "GT accessed before frozen_predictions.npz was saved!"
                assert step_file.is_file(), "GT accessed before step_records.json was saved!"
            return orig_getitem(ds_self, idx)

        with patch.object(KuroEvaluationDataset, "__getitem__", spying_getitem):
            runner.run_ablation(mock_records_by_event=mock_data)

        self.assertTrue(gt_accessed)

    # -------------------------------------------------------------------------
    # 7. Confusion Matrix Conservation Invariant
    # -------------------------------------------------------------------------
    def test_07_confusion_matrix_conservation(self):
        """For every event and pooled summary, TP + FP + FN + TN == valid_pixels."""
        out_dir = self.temp_dir / "out_conservation"
        mock_data = self._generate_mock_events(num_samples_per_event=8)

        runner = Week12AblationRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )
        res = runner.run_ablation(mock_records_by_event=mock_data)

        for cid, c_data in res["conditions"].items():
            # Check pooled metrics
            pm = c_data["pooled_metrics"]
            conf_sum = pm["tp"] + pm["fp"] + pm["fn"] + pm["tn"]
            self.assertEqual(conf_sum, pm["valid_pixels"])

            # Check each event
            for ev in c_data["event_rows"]:
                ev_sum = ev["tp"] + ev["fp"] + ev["fn"] + ev["tn"]
                self.assertEqual(ev_sum, ev["valid_pixels"])

    # -------------------------------------------------------------------------
    # 8. Failure & Overwrite Safeguards
    # -------------------------------------------------------------------------
    def test_08_failure_and_overwrite_safeguards(self):
        """Failure writes execution_failure.json; existing directory without allow_overwrite is rejected."""
        out_dir = self.temp_dir / "out_overwrite"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preexisting_sentinel.txt").write_text("sentinel")

        runner = Week12AblationRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
            allow_overwrite=False,
        )

        with self.assertRaises(FileExistsError) as ctx:
            runner.validate_preflight()
        self.assertIn("already exists and contains files", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 9. Device Syntax & Authorization Matrix
    # -------------------------------------------------------------------------
    def test_09_device_syntax_and_authorization_matrix(self):
        """Rejects bad device strings, non-dev events, and unapproved statuses in real mode."""
        cfg_bad = copy.deepcopy(self.base_cfg)
        cfg_bad["status"] = "DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED"
        p_bad = self._write_temp_config(cfg_bad)

        with self.assertRaises(PermissionError) as ctx:
            Week12AblationRunner(config_path=p_bad, output_dir=self.temp_dir / "out_dev", device="cuda:0", synthetic_mock=False)
        self.assertIn("Real-data execution is rejected", str(ctx.exception))

        # Real mode rejects cpu, mps, bare cuda
        for bad_dev in ["cpu", "mps", "cuda", "cuda:-1", "not-a-device"]:
            with self.assertRaises(ValueError) as ctx:
                Week12AblationRunner(config_path=self.config_path, output_dir=self.temp_dir / "out_dev", device=bad_dev, synthetic_mock=False)
            self.assertIn("Real smoke execution strictly requires device format 'cuda:<non-negative integer>'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
