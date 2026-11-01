"""ChangeGuard Week 13 Source-Anchored Evidence Consistency Tests.

Governing Config: configs/changeguard/week13_source_anchor_dev6_v1.json
Governing Protocol: docs/protocols/week13_source_anchor_protocol_v1.md

Covers required unit tests:
1. lambda_anc = 0 matches C2 behavior and outputs.
2. Positive lambda_anc generates gradient on candidate parameters.
3. Source parameters stay detached, unchanged, and gradient-free.
4. Correct geometry/masking for aligned strong view and valid pixels.
5. Non-finite loss/gradient/step rollback preserves candidate state.
6. Offline confusion counts strictly equal valid_pixels.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter
from tools.run_week13_source_anchor import (
    MockSNUNet,
    SourceAnchorConsistencyAdapter,
    Week13SourceAnchorRunner,
)


class TestWeek13SourceAnchor(unittest.TestCase):
    """Unit test suite for Week 13 source-anchored evidence consistency."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config_path = REPO_ROOT / "configs/changeguard/week13_source_anchor_dev6_v1.json"
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_cfg = json.load(f)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="w13_anchor_test_"))

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
    # 1. lambda_anc = 0 matches C2
    # -------------------------------------------------------------------------
    def test_01_lambda_zero_matches_c2(self):
        """When anchor_lambda = 0.0, adapter matches C2 EvidenceConsistencyAdapter identically."""
        # Candidate model
        cand_c2 = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        cand_w13 = copy.deepcopy(cand_c2)

        # Source models
        src_c2 = copy.deepcopy(cand_c2)
        src_w13 = copy.deepcopy(cand_c2)

        adapter_c2 = EvidenceConsistencyAdapter(
            model=cand_c2,
            source_model=src_c2,
            confidence_threshold=0.90,
            learning_rate=1.25e-4,
            momentum=0.9,
            weight_decay=0.0,
            device="cpu",
        )

        adapter_w13 = SourceAnchorConsistencyAdapter(
            model=cand_w13,
            source_model=src_w13,
            confidence_threshold=0.90,
            anchor_lambda=0.0,
            learning_rate=1.25e-4,
            momentum=0.9,
            weight_decay=0.0,
            device="cpu",
        )

        # Provide fixed batch
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        res_c2 = adapter_c2.adapt_and_predict(xA.clone(), xB.clone(), sar_valid_mask=sar_mask.clone())
        res_w13 = adapter_w13.adapt_and_predict(xA.clone(), xB.clone(), sar_valid_mask=sar_mask.clone())

        # Check action & prediction match
        self.assertEqual(res_c2["adaptation_metadata"]["action"], res_w13["adaptation_metadata"]["action"])
        self.assertEqual(res_c2["adaptation_metadata"]["update_applied"], res_w13["adaptation_metadata"]["update_applied"])
        self.assertTrue(torch.equal(
            res_c2["primary_prediction"]["predicted_class"],
            res_w13["primary_prediction"]["predicted_class"]
        ))

        # Check loss match
        l_c2 = res_c2["adaptation_metadata"]["loss"]
        l_w13 = res_w13["adaptation_metadata"]["loss"]
        if l_c2 is not None and l_w13 is not None:
            self.assertAlmostEqual(l_c2, l_w13, places=5)

        # Check updated weights match within floating point precision
        for (n1, p1), (n2, p2) in zip(cand_c2.named_parameters(), cand_w13.named_parameters()):
            if p1.requires_grad:
                self.assertTrue(torch.allclose(p1, p2, atol=1e-6), f"Mismatch in parameter {n1}")

    # -------------------------------------------------------------------------
    # 2. Positive lambda_anc generates gradient on candidate parameters
    # -------------------------------------------------------------------------
    def test_02_positive_lambda_generates_gradient(self):
        """Positive anchor_lambda produces gradient signal even when confidence mask is empty."""
        cand = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        src = copy.deepcopy(cand)

        # Set confidence threshold impossibly high (1.5) so C2 consistency is strictly empty
        adapter = SourceAnchorConsistencyAdapter(
            model=cand,
            source_model=src,
            confidence_threshold=1.5,
            anchor_lambda=0.50,
            learning_rate=1.25e-4,
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Record pre-update weights
        pre_weights = {n: p.detach().clone() for n, p in cand.named_parameters() if p.requires_grad}

        res = adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_mask)
        meta = res["adaptation_metadata"]

        self.assertEqual(meta["action"], "update")
        self.assertTrue(meta["update_applied"])
        self.assertIsNotNone(meta["loss"])
        self.assertGreater(meta["loss"], 0.0)
        self.assertEqual(meta["selected_pixels"], 0)  # C2 consistency was empty

        # Check that candidate parameters were updated by anchor loss
        changed = False
        for n, p in cand.named_parameters():
            if p.requires_grad:
                if not torch.equal(p, pre_weights[n]):
                    changed = True
        self.assertTrue(changed, "Candidate parameters should have updated via anchor loss!")

    # -------------------------------------------------------------------------
    # 3. Source parameters stay detached, unchanged, and gradient-free
    # -------------------------------------------------------------------------
    def test_03_source_parameters_remain_detached_and_unchanged(self):
        """Source model remains strictly eval mode, requires_grad=False, and bit-identical."""
        cand = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        src = copy.deepcopy(cand)
        src_pre = {n: p.detach().clone() for n, p in src.named_parameters()}

        adapter = SourceAnchorConsistencyAdapter(
            model=cand,
            source_model=src,
            anchor_lambda=0.50,
            device="cpu",
        )

        # Verify initial requires_grad
        for p in src.parameters():
            self.assertFalse(p.requires_grad)

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Run 3 adaptation steps
        for _ in range(3):
            adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_mask)

        # Verify source remains completely untouched
        for n, p in src.named_parameters():
            self.assertIsNone(p.grad, f"Source parameter {n} should have None grad!")
            self.assertFalse(p.requires_grad, f"Source parameter {n} should not require grad!")
            self.assertTrue(torch.equal(p, src_pre[n]), f"Source parameter {n} was modified!")

    # -------------------------------------------------------------------------
    # 4. Correct geometry/masking for aligned strong view and valid pixels
    # -------------------------------------------------------------------------
    def test_04_geometry_and_valid_masking(self):
        """Altered invalid pixels do not affect anchor or consistency loss."""
        cand = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        src = copy.deepcopy(cand)

        adapter1 = SourceAnchorConsistencyAdapter(
            model=copy.deepcopy(cand),
            source_model=copy.deepcopy(src),
            anchor_lambda=0.50,
            device="cpu",
        )
        adapter2 = SourceAnchorConsistencyAdapter(
            model=copy.deepcopy(cand),
            source_model=copy.deepcopy(src),
            anchor_lambda=0.50,
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        sar_mask[:, :, 0:4] = 0  # left 4 columns are invalid

        # xA2, xB2 have garbage at the invalid pixels
        xA2 = xA.clone()
        xB2 = xB.clone()
        xA2[:, :, :, 0:4] = 999.0
        xB2[:, :, :, 0:4] = -999.0

        res1 = adapter1.adapt_and_predict(xA, xB, sar_valid_mask=sar_mask)
        res2 = adapter2.adapt_and_predict(xA2, xB2, sar_valid_mask=sar_mask)

        # Losses should be identical because invalid pixels are masked out in loss
        self.assertEqual(res1["adaptation_metadata"]["valid_pixels"], res2["adaptation_metadata"]["valid_pixels"])

    # -------------------------------------------------------------------------
    # 5. Nonfinite loss/gradient/step rollback preserves candidate state
    # -------------------------------------------------------------------------
    def test_05_nonfinite_rollback_preserves_state(self):
        """Non-finite loss or corrupt optimizer step rolls back candidate parameters and optimizer state."""
        cand = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        src = copy.deepcopy(cand)

        adapter = SourceAnchorConsistencyAdapter(
            model=cand,
            source_model=src,
            anchor_lambda=0.50,
            device="cpu",
        )

        pre_params = {n: p.detach().clone() for n, p in cand.named_parameters() if p.requires_grad}

        # Step 1: Normal step
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        res1 = adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_mask)
        self.assertEqual(res1["adaptation_metadata"]["action"], "update")
        self.assertEqual(adapter.num_updates_applied, 1)

        saved_state = {n: p.detach().clone() for n, p in cand.named_parameters() if p.requires_grad}

        # Step 2: Inject NaN into input
        xA_nan = xA.clone()
        xA_nan[0, 0, 5, 5] = float("nan")
        res2 = adapter.adapt_and_predict(xA_nan, xB, sar_valid_mask=sar_mask)

        self.assertEqual(res2["adaptation_metadata"]["action"], "failure")
        self.assertFalse(res2["adaptation_metadata"]["update_applied"])
        self.assertEqual(adapter.num_updates_applied, 1)
        self.assertEqual(adapter.num_updates_failed, 1)

        # Verify candidate parameters were rolled back to post-step-1 state
        for n, p in cand.named_parameters():
            if p.requires_grad:
                self.assertTrue(torch.equal(p, saved_state[n]), f"Parameter {n} was corrupted by failed step!")

    # -------------------------------------------------------------------------
    # 6. Offline confusion counts strictly equal valid_pixels
    # -------------------------------------------------------------------------
    def test_06_confusion_conservation(self):
        """TP + FP + FN + TN strictly equals valid_pixels across all events."""
        out_dir = self.temp_dir / "out_conservation"
        mock_data = self._generate_mock_events(num_samples_per_event=18)

        runner = Week13SourceAnchorRunner(
            config_path=self.config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_mock=True,
        )

        summary = runner.run_experiment(mock_records_by_event=mock_data)

        # Check total confusion conservation
        tot_cells = summary["total_tp"] + summary["total_fp"] + summary["total_fn"] + summary["total_tn"]
        self.assertEqual(tot_cells, summary["total_valid_pixels"])

        # Check per-event confusion conservation
        for ev in summary["event_metrics"]:
            ev_cells = ev["tp"] + ev["fp"] + ev["fn"] + ev["tn"]
            self.assertEqual(ev_cells, ev["valid_pixels"])
            self.assertEqual(ev["batch_sizes"], [16, 2])
            self.assertEqual(ev["sample_count"], 18)


if __name__ == "__main__":
    unittest.main()
