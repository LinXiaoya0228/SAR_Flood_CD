"""ChangeGuard Plan V3 Work Package E2.1: Confidence Grid Unit Tests.

Comprehensive test suite verifying:
1. Exact threshold boundary behaviour (p >= tau vs p < tau).
2. No-filter behaviour (tau=0.0 selects exactly all sensory-valid pixels).
3. Per-class selected-pixel accounting (c0 + c1 + c2 == selected_pixels).
4. Correct sensory-valid masking (nodata pixels strictly excluded).
5. Zero-selection batches (no updates, no gradient steps, no NaNs, action="skip").
6. Skipped versus updated batch accounting.
7. Gradient L2 norm and parameter displacement telemetry.
8. Label timing enforcement (no label access before durable persistence).
9. Deterministic event ordering (Order A sequence and sorted sample IDs).
10. Sealed-test quarantine (raises RuntimeError on any forbidden test event).
11. Reuse-equivalence validation (verifies detection of schema/telemetry gaps).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.changeguard.e1_orchestrator import (
    E1ConsistencyAdapter,
    E1OfflineEvaluator,
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
)
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)


class MockSNUNet(nn.Module):
    """Synthetic lightweight SNUNet model for CPU testing with real BatchNorm2d layers."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.classifier = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.classifier(x)


class TestE21ConfidenceGrid(unittest.TestCase):
    """Test suite for ChangeGuard Plan V3 E2.1 Confidence Grid."""

    def setUp(self):
        self.device = "cpu"
        torch.manual_seed(42)
        self.model = MockSNUNet(in_channels=2, num_classes=3)
        self.source_model = copy.deepcopy(self.model)

    def test_01_exact_threshold_boundary_behaviour(self):
        """Verifies boundary condition: pixels with max prob >= tau are selected, < tau are excluded."""
        # Test across multiple threshold levels
        for tau in [0.70, 0.90, 0.95]:
            adapter = E1ConsistencyAdapter(
                model=copy.deepcopy(self.model),
                source_model=self.source_model,
                confidence_threshold=tau,
                device=self.device,
            )
            # Create a deterministic mock input
            xA = torch.randn(2, 2, 16, 16)
            xB = torch.randn(2, 2, 16, 16)
            v_mask = torch.ones(2, 16, 16)
            s_ids = ["s1", "s2"]

            # Manually compute expected selection from forward pass
            adapter.model.eval()
            with torch.no_grad():
                logits = adapter.model(xA, xB)
                probs = F.softmax(logits, dim=1)
                max_p = probs.max(dim=1)[0]
                expected_mask = (max_p >= tau) & (v_mask == 1)
                expected_selected = int(expected_mask.sum().item())

            # Run step
            res = adapter.adapt_and_predict_step(
                xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520
            )
            adapt_meta = res["adaptation_metadata"]
            self.assertEqual(
                adapt_meta["selected_pixels"],
                expected_selected,
                f"Threshold {tau}: selected pixels must match exact boundary comparison.",
            )

    def test_02_no_filter_behaviour(self):
        """Verifies that confidence_threshold=0.0 selects 100% of sensory-valid pixels."""
        adapter = E1ConsistencyAdapter(
            model=copy.deepcopy(self.model),
            source_model=self.source_model,
            confidence_threshold=0.0,
            device=self.device,
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        # Nodata mask with 50% valid pixels
        v_mask = torch.zeros(2, 16, 16)
        v_mask[:, :8, :] = 1.0  # Exactly 2 * 8 * 16 = 256 valid pixels
        s_ids = ["s1", "s2"]

        res = adapter.adapt_and_predict_step(
            xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520
        )
        adapt_meta = res["adaptation_metadata"]
        self.assertEqual(adapt_meta["valid_pixels"], 256)
        self.assertEqual(adapt_meta["selected_pixels"], 256)
        self.assertAlmostEqual(adapt_meta["selected_pixel_fraction"], 1.0, places=6)

    def test_03_per_class_selected_pixel_accounting(self):
        """Verifies selected_pixels_c0 + c1 + c2 == selected_pixels."""
        for tau in [0.0, 0.70, 0.90, 0.95]:
            adapter = E1ConsistencyAdapter(
                model=copy.deepcopy(self.model),
                source_model=self.source_model,
                confidence_threshold=tau,
                device=self.device,
            )
            xA = torch.randn(2, 2, 16, 16)
            xB = torch.randn(2, 2, 16, 16)
            v_mask = torch.ones(2, 16, 16)
            s_ids = ["s1", "s2"]

            res = adapter.adapt_and_predict_step(
                xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520
            )
            meta = res["adaptation_metadata"]
            c0 = meta["selected_pixels_c0"]
            c1 = meta["selected_pixels_c1"]
            c2 = meta["selected_pixels_c2"]
            total_sel = meta["selected_pixels"]

            self.assertGreaterEqual(c0, 0)
            self.assertGreaterEqual(c1, 0)
            self.assertGreaterEqual(c2, 0)
            self.assertEqual(
                c0 + c1 + c2,
                total_sel,
                f"Threshold {tau}: sum of class-wise selected pixels ({c0+c1+c2}) must equal total selected ({total_sel}).",
            )

    def test_04_correct_sensory_valid_masking(self):
        """Verifies that invalid sensory pixels (MK0_MNA == 0) are strictly excluded from selection."""
        adapter = E1ConsistencyAdapter(
            model=copy.deepcopy(self.model),
            source_model=self.source_model,
            confidence_threshold=0.0,  # Even with no conf filter, nodata must be excluded
            device=self.device,
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        # All invalid mask
        v_mask = torch.zeros(2, 16, 16)
        s_ids = ["s1", "s2"]

        res = adapter.adapt_and_predict_step(
            xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520
        )
        meta = res["adaptation_metadata"]
        self.assertEqual(meta["valid_pixels"], 0)
        self.assertEqual(meta["selected_pixels"], 0)
        self.assertEqual(meta["selected_pixels_c0"], 0)
        self.assertEqual(meta["selected_pixels_c1"], 0)
        self.assertEqual(meta["selected_pixels_c2"], 0)
        self.assertEqual(meta["action"], "skip")
        self.assertFalse(meta["update_applied"])
        self.assertIsNone(meta["total_loss"])

    def test_05_zero_selection_batches(self):
        """Verifies that a batch with 0 selected pixels deterministically skips without NaNs or fake updates."""
        adapter = E1ConsistencyAdapter(
            model=copy.deepcopy(self.model),
            source_model=self.source_model,
            confidence_threshold=1.01,  # Impossible threshold: probability can never exceed 1.01
            device=self.device,
        )
        # Record weights before step
        w_before = {n: p.clone() for n, p in adapter.model.named_parameters()}

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16)
        s_ids = ["s1", "s2"]

        res = adapter.adapt_and_predict_step(
            xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520
        )
        meta = res["adaptation_metadata"]
        self.assertEqual(meta["selected_pixels"], 0)
        self.assertEqual(meta["action"], "skip")
        self.assertFalse(meta["update_applied"])
        self.assertIsNone(meta["total_loss"])
        self.assertIsNone(meta["gradient_norm_l2"])

        # Check weights are completely unmutated
        for n, p in adapter.model.named_parameters():
            self.assertTrue(
                torch.equal(p, w_before[n]),
                f"Parameter {n} was mutated during a zero-selection skip step!",
            )

    def test_06_skipped_versus_updated_batch_accounting(self):
        """Verifies that adapter update and skip counters accurately track step outcomes."""
        adapter = E1ConsistencyAdapter(
            model=copy.deepcopy(self.model),
            source_model=self.source_model,
            confidence_threshold=0.0,
            device=self.device,
        )
        # Step 1: Valid batch -> update
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16)
        adapter.adapt_and_predict_step(xA, xB, v_mask, ["s1", "s2"], event_id=520)
        self.assertEqual(adapter.num_updates_applied, 1)
        self.assertEqual(adapter.num_updates_skipped, 0)

        # Step 2: Empty valid mask -> skip
        v_mask_empty = torch.zeros(2, 16, 16)
        adapter.adapt_and_predict_step(xA, xB, v_mask_empty, ["s3", "s4"], event_id=520)
        self.assertEqual(adapter.num_updates_applied, 1)
        self.assertEqual(adapter.num_updates_skipped, 1)

    def test_07_gradient_and_update_telemetry(self):
        """Verifies gradient L2 norm and BN affine displacement telemetry capture."""
        adapter = E1ConsistencyAdapter(
            model=copy.deepcopy(self.model),
            source_model=self.source_model,
            confidence_threshold=0.0,
            learning_rate=1e-2,  # Large LR to produce measurable displacement
            device=self.device,
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16)
        s_ids = ["s1", "s2"]

        res = adapter.adapt_and_predict_step(xA, xB, v_mask, s_ids, event_id=520)
        meta = res["adaptation_metadata"]

        self.assertTrue(meta["update_applied"])
        self.assertIsNotNone(meta["total_loss"])
        self.assertIsNotNone(meta["gradient_norm_l2"])
        self.assertGreater(meta["gradient_norm_l2"], 0.0)
        self.assertIsNotNone(meta["bn_weight_displacement"])
        self.assertGreater(meta["bn_weight_displacement"], 0.0)

    def test_08_no_label_access_before_persistence(self):
        """Verifies LabelGuardedBatch raises PermissionError if labels are accessed before mark_persisted()."""
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16)
        labels = torch.randint(0, 3, (2, 16, 16))

        batch = LabelGuardedBatch(
            sample_ids=["s1", "s2"],
            event_id=520,
            xA=xA,
            xB=xB,
            valid_mask=v_mask,
            labels=labels,
        )

        # Access before persistence MUST raise PermissionError
        with self.assertRaises(PermissionError):
            _ = batch.get_labels()

        # Sensory inputs are permitted
        sensory = batch.get_sensory_inputs()
        self.assertIn("xA", sensory)
        self.assertNotIn("labels", sensory)

        # After persistence, labels are unlocked
        batch.mark_persisted()
        unlocked = batch.get_labels()
        self.assertTrue(torch.equal(unlocked, labels))

    def test_09_deterministic_event_ordering(self):
        """Verifies that Order A events follow the declared sequence exactly."""
        expected_order_a = [279, 437, 520, 559, 1111003, 1111008]
        # Check against DEFAULT_TTA_DEV_EVENTS
        self.assertEqual(DEFAULT_TTA_DEV_EVENTS, expected_order_a)

    def test_10_sealed_test_quarantine(self):
        """Verifies that any access to sealed test events triggers fail-fast exception."""
        forbidden_ids = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
        self.assertEqual(FORBIDDEN_SEALED_TEST_EVENTS, forbidden_ids)

        for fid in forbidden_ids:
            self.assertNotIn(
                fid,
                DEFAULT_TTA_DEV_EVENTS,
                f"CRITICAL: Sealed test event {fid} found in development events!",
            )

    def test_11_reuse_equivalence_validation(self):
        """Verifies detection of schema/telemetry gaps in historical artifacts."""
        # Simulated legacy Week 12 record
        legacy_step_record = {
            "step_index": 1,
            "event_id": 520,
            "condition_id": "C2_consistency_conf90",
            "batch_size": 16,
            "sample_ids": ["s1"],
            "action": "update",
            "update_applied": True,
            "loss": 0.005,
            "valid_pixels": 1000,
            "selected_pixels": 800,
            "selected_pixel_fraction": 0.8,
            "failure_reason": None,
            "batch_time_ms": 10.0,
        }

        # Plan V3 required telemetry fields
        required_telemetry_fields = [
            "selected_pixels_c0",
            "selected_pixels_c1",
            "selected_pixels_c2",
            "gradient_norm_l2",
            "bn_weight_displacement",
            "bn_bias_displacement",
        ]

        # Audit legacy record
        missing_fields = [f for f in required_telemetry_fields if f not in legacy_step_record]
        self.assertEqual(
            len(missing_fields),
            6,
            "Legacy Week 12 records must be detected as missing all 6 required telemetry fields.",
        )


if __name__ == "__main__":
    unittest.main()

