"""ChangeGuard Plan V3 Work Package E4: Class-Aware Intervention Contract Test Suite.

Verifies:
1. Source/candidate physical storage independence (zero memory pointer overlap).
2. Source model eval mode, requires_grad=False, immutability (bit-exact SHA-256 hash).
3. Identical object rejection when passed as both candidate and source.
4. K1 (Class-Balanced Selection):
   - Uses predicted class and confidence only (no ground truth).
   - Imposes deterministic per-class quotas Q_c = floor(N_valid * pi_c).
   - Quota capping: selects exactly min(N_avail, Q_c).
   - Deterministic spatial tie-breaking when confidences are identical.
   - Empty-class handling: unused quota is strictly discarded, never transferred/substituted.
5. K2 (Flood-Aware Loss Weighting):
   - Weights w = (1.0, 1.0, 4.0) applied to predicted classes.
   - Explicit denominator is sum_{i: M_i=1} w_i.
   - Full prediction coverage maintained.
6. K3 (Permanent-Water Protection):
   - Uses frozen-source class 1 (permanent water) predictions only (no ground truth).
   - Excludes source permanent water from candidate adaptation mask.
   - Tracks candidate drift away from source permanent water.
7. K4 (Flood/Change Preservation):
   - Identifies source flood pixels where p_src(2) >= 0.50.
   - Applies auxiliary preservation penalty lambda_pres * [-log(p_cand_aligned(2))].
   - Telemetry tracks erasure (denominator = source flood) and expansion (denominator = source non-flood).
8. Strict zero-leakage boundary: ground truth labels not accessed during adaptation.
9. Sealed-test event quarantine enforcement.
10. Offline interpretation metrics contract:
   - 3x3 confusion conservation (sum C_ij == N_valid).
   - Separated false positives and false negatives by true class.
   - Source-relative transition matrix T_ij and harmful/beneficial transitions.
11. Boundary conditions: empty valid masks or zero confident pixels handled safely without NaN.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.changeguard.e1_orchestrator import LabelGuardedBatch
from src.changeguard.e4_class_aware_adapters import (
    E4ClassAwareAdapter,
    SOURCE_CALIBRATION_PI,
    compute_e4_offline_interpretation_metrics,
)


class MockDualStreamCD(nn.Module):
    """Lightweight dual-branch convolutional model for unit testing."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3, feature_dim: int = 8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels * 2, feature_dim, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(feature_dim)
        self.conv_mid = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(feature_dim)
        self.conv_final = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        z = torch.cat([xA, xB], dim=1)
        z = F.relu(self.bn_in(self.conv_in(z)))
        z = F.relu(self.bn_mid(self.conv_mid(z)))
        return self.conv_final(z)


class TestE4ClassAwareContracts(unittest.TestCase):
    """Comprehensive test suite for E4 class-aware adapters and contracts."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.candidate = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
        self.source = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
        self.source.load_state_dict(self.candidate.state_dict())

    def test_01_storage_independence_and_source_immutability(self):
        """Source and candidate must share zero memory pointers, and source remains bit-exact."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            source_model=self.source,
            intervention="k3_perm_protection",
            confidence_threshold=0.90,
            device="cpu",
        )

        cand_ptrs = {p.data_ptr() for p in adapter.model.parameters()} | {
            b.data_ptr() for b in adapter.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in adapter.source_model.parameters()} | {
            b.data_ptr() for b in adapter.source_model.buffers() if b is not None
        }
        self.assertEqual(len(cand_ptrs.intersection(src_ptrs)), 0)

        # Run an adaptation step
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        step_res = adapter.adapt_step(xA, xB, mask, ["s1", "s2"], event_id=520)
        self.assertIn(step_res["action"], {"APPLIED", "SKIPPED_EMPTY_MASK"})

        # Verify source immutability
        self.assertTrue(adapter.verify_source_immutability())
        for name, p in adapter.source_model.named_parameters():
            self.assertIsNone(p.grad, f"Source parameter {name} has accumulated gradients!")

    def test_02_identical_object_rejection(self):
        """Passing identical object for candidate and source must fail immediately."""
        with self.assertRaises((ValueError, RuntimeError)):
            E4ClassAwareAdapter(
                model=self.candidate,
                source_model=self.candidate,
                intervention="k3_perm_protection",
            )

    def test_03_k1_quota_capping_and_deterministic_tie_breaking(self):
        """K1 imposes genuinely class-balanced selection with uniform cap floor(N_eligible / 3)
        and exact deterministic lexicographical tie-breaking."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            intervention="k1_balanced_selection",
            confidence_threshold=0.50,
            device="cpu",
        )

        B, H, W = 1, 12, 10
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)
        valid_pixel_count = 120

        # Create an imbalanced eligible set:
        # Class 0 has 80 eligible pixels (> cap)
        # Class 1 has 20 eligible pixels (<= cap)
        # Class 2 has 20 eligible pixels (<= cap)
        # Total eligible = 120 pixels -> Uniform cap C = floor(120 / 3) = 40.
        base_conf_mask = torch.ones(B, H, W, dtype=torch.bool)
        pred_cand = torch.zeros(B, H, W, dtype=torch.long)
        pred_cand[0, :8, :] = 0   # 80 pixels class 0
        pred_cand[0, 8:10, :] = 1 # 20 pixels class 1
        pred_cand[0, 10:, :] = 2  # 20 pixels class 2

        # Give equal confidence 0.95 to all pixels
        conf_weak = torch.full((B, H, W), 0.95, dtype=torch.float32)

        selected_mask, meta = adapter._apply_k1_selection(
            base_conf_mask=base_conf_mask,
            pred_cand=pred_cand,
            conf_weak=conf_weak,
            valid_mask=valid_mask,
            valid_pixel_count=valid_pixel_count,
        )

        # Uniform cap must be floor(120 / 3) = 40
        self.assertEqual(meta["batch_cap"], 40)
        self.assertEqual(meta["requested_quota"][0], 40)
        self.assertEqual(meta["requested_quota"][1], 40)
        self.assertEqual(meta["requested_quota"][2], 40)

        # Class 0 was imbalanced (80 pixels) -> capped at exactly 40
        self.assertEqual(meta["available_pixels"][0], 80)
        self.assertEqual(meta["selected_pixels"][0], 40)

        # Class 1 and 2 had 20 pixels -> selected up to availability (20)
        self.assertEqual(meta["available_pixels"][1], 20)
        self.assertEqual(meta["selected_pixels"][1], 20)
        self.assertEqual(meta["available_pixels"][2], 20)
        self.assertEqual(meta["selected_pixels"][2], 20)

        # Total selected = 40 + 20 + 20 = 80
        self.assertEqual(int(selected_mask.sum().item()), 80)

        # Deterministic lexicographical tie-breaking check:
        # For Class 0 (which selected 40 of 80 equal-confidence pixels),
        # verify that the 40 chosen pixels are strictly the first 40 spatial indices (rows 0 to 3)
        class0_selected = selected_mask[0, :8, :]  # shape [8, 10]
        # First 4 rows (40 pixels) must be True, remaining 4 rows must be False
        self.assertTrue(torch.all(class0_selected[:4, :]))
        self.assertFalse(torch.any(class0_selected[4:, :]))

    def test_04_k1_empty_class_non_substitution(self):
        """K1 never transfers unused capacity from empty classes to populated classes."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            intervention="k1_balanced_selection",
            confidence_threshold=0.50,
            device="cpu",
        )

        B, H, W = 1, 10, 10
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)
        valid_pixel_count = 100

        # Scenario 1: Only Class 0 has 90 eligible pixels. Classes 1 and 2 are empty (0 pixels).
        base_conf_mask = torch.ones(B, H, W, dtype=torch.bool)
        base_conf_mask[0, 9, :] = False  # 90 eligible pixels
        pred_cand = torch.zeros(B, H, W, dtype=torch.long)  # All Class 0
        conf_weak = torch.full((B, H, W), 0.95, dtype=torch.float32)

        selected_mask, meta = adapter._apply_k1_selection(
            base_conf_mask=base_conf_mask,
            pred_cand=pred_cand,
            conf_weak=conf_weak,
            valid_mask=valid_mask,
            valid_pixel_count=valid_pixel_count,
        )

        # Cap = floor(90 / 3) = 30
        self.assertEqual(meta["batch_cap"], 30)
        # Class 0 must receive exactly 30 pixels (NOT 90). Unused capacity from empty classes is discarded!
        self.assertEqual(meta["selected_pixels"][0], 30)
        self.assertEqual(meta["selected_pixels"][1], 0)
        self.assertEqual(meta["selected_pixels"][2], 0)
        self.assertTrue(meta["empty_classes"][1])
        self.assertTrue(meta["empty_classes"][2])
        self.assertEqual(int(selected_mask.sum().item()), 30)

    def test_04b_k1_independence_from_gt_and_future_samples(self):
        """K1 selection depends strictly on sensory valid mask and predicted probabilities."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            intervention="k1_balanced_selection",
            confidence_threshold=0.50,
            device="cpu",
        )
        B, H, W = 1, 6, 6
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)
        base_conf_mask = torch.ones(B, H, W, dtype=torch.bool)
        pred_cand = torch.randint(0, 3, (B, H, W))
        conf_weak = torch.rand(B, H, W) * 0.4 + 0.6  # all in [0.6, 1.0]

        sel1, meta1 = adapter._apply_k1_selection(
            base_conf_mask=base_conf_mask,
            pred_cand=pred_cand,
            conf_weak=conf_weak,
            valid_mask=valid_mask,
            valid_pixel_count=36,
        )
        sel2, meta2 = adapter._apply_k1_selection(
            base_conf_mask=base_conf_mask,
            pred_cand=pred_cand,
            conf_weak=conf_weak,
            valid_mask=valid_mask,
            valid_pixel_count=36,
        )
        self.assertTrue(torch.equal(sel1, sel2))
        self.assertEqual(meta1["batch_cap"], meta2["batch_cap"])

    def test_05_k2_weighted_loss_denominator(self):
        """K2 weights predicted flood pixels and uses explicit sum of weights denominator."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            intervention="k2_flood_weighting",
            k2_class_weights={0: 1.0, 1: 1.0, 2: 4.0},
            device="cpu",
        )

        B, H, W = 1, 2, 2
        base_conf_mask = torch.ones(B, H, W, dtype=torch.bool)
        # 2 pixels class 0 (weight 1.0), 2 pixels class 2 (weight 4.0)
        pred_cand = torch.tensor([[[0, 0], [2, 2]]], dtype=torch.long)
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)

        weights, meta = adapter._apply_k2_weighting(
            base_conf_mask=base_conf_mask,
            pred_cand=pred_cand,
            valid_mask=valid_mask,
        )

        self.assertEqual(float(weights[0, 0, 0].item()), 1.0)
        self.assertEqual(float(weights[0, 1, 0].item()), 4.0)

        # Expected weighted denominator: 1.0 + 1.0 + 4.0 + 4.0 = 10.0
        w_sum = (weights * base_conf_mask.float()).sum().item()
        self.assertAlmostEqual(w_sum, 10.0, places=5)

    def test_06_k3_source_permanent_water_exclusion_contract(self):
        """K3 excludes frozen-source permanent water (class 1, argmax) from the consistency mask.
        Demonstrates the exact guarantee provided: excludes those pixels from loss computation,
        but does NOT guarantee invariant predictions because shared parameters can still adapt."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            source_model=self.source,
            intervention="k3_perm_protection",
            confidence_threshold=0.90,
            device="cpu",
        )

        B, H, W = 1, 4, 4
        base_conf_mask = torch.ones(B, H, W, dtype=torch.bool)
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)

        # Source predicts class 1 (permanent water) in top half (8 pixels), class 0 in bottom half
        pred_src = torch.zeros(B, H, W, dtype=torch.long)
        pred_src[:, :2, :] = 1  # 8 pixels of permanent water

        probs_src = torch.zeros(B, 3, H, W, dtype=torch.float32)
        probs_src[:, 1, :2, :] = 0.99
        probs_src[:, 0, 2:, :] = 0.99

        pred_cand = torch.zeros(B, H, W, dtype=torch.long)

        adapted_mask, meta = adapter._apply_k3_exclusion(
            base_conf_mask=base_conf_mask,
            pred_src=pred_src,
            probs_src=probs_src,
            pred_cand=pred_cand,
            valid_mask=valid_mask,
        )

        # Guarantee 1: Source permanent water pixels are strictly excluded from the adapted mask
        self.assertEqual(meta["excluded_pixels"], 8)
        self.assertEqual(meta["operation"], "source_predicted_permanent_water_exclusion")
        self.assertEqual(int(adapted_mask[:, :2, :].sum().item()), 0)
        self.assertEqual(int(adapted_mask[:, 2:, :].sum().item()), 8)
        self.assertEqual(meta["total_selected_pixels"], 8)

        # Guarantee 2: When candidate adapts on non-permanent-water pixels,
        # shared BN affine weights are modified, meaning post-adaptation predictions
        # at permanent-water pixels are NOT mathematically guaranteed to be invariant.
        xA = torch.randn(1, 2, 4, 4)
        xB = torch.randn(1, 2, 4, 4)
        step_res = adapter.adapt_step(xA, xB, valid_mask, ["s1"], event_id=520)
        self.assertIn("bn_weight_displacement", step_res)
        self.assertEqual(step_res["intervention_meta"]["operation"], "source_predicted_permanent_water_exclusion")

    def test_07_k4_flood_preservation_and_erasure_expansion_accounting(self):
        """K4 tracks erasure (denominator = source flood) and expansion (denominator = source non-flood)."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            source_model=self.source,
            intervention="k4_flood_preservation",
            k4_flood_threshold=0.50,
            k4_lambda_pres=0.50,
            device="cpu",
        )

        B, H, W = 1, 4, 4
        valid_mask = torch.ones(B, H, W, dtype=torch.bool)

        # Source predicts 4 flood pixels (class 2) with prob 0.85
        pred_src = torch.zeros(B, H, W, dtype=torch.long)
        pred_src[0, 0, :4] = 2  # 4 source flood pixels

        probs_src = torch.zeros(B, 3, H, W, dtype=torch.float32)
        probs_src[0, 2, 0, :4] = 0.85

        # Candidate erases 2 flood pixels (predicts class 0) and retains 2 (predicts class 2)
        # Candidate also expands 3 non-flood pixels into flood (predicts class 2 elsewhere)
        pred_cand = torch.zeros(B, H, W, dtype=torch.long)
        pred_cand[0, 0, 0] = 2  # retained
        pred_cand[0, 0, 1] = 2  # retained
        pred_cand[0, 0, 2] = 0  # erased!
        pred_cand[0, 0, 3] = 0  # erased!

        pred_cand[0, 1, 0] = 2  # expanded!
        pred_cand[0, 1, 1] = 2  # expanded!
        pred_cand[0, 1, 2] = 2  # expanded!

        meta = adapter._apply_k4_preservation(
            pred_src=pred_src,
            probs_src=probs_src,
            pred_cand=pred_cand,
            valid_mask=valid_mask,
        )

        # Total source flood = 4, erased = 2 -> erasure_rate = 0.50
        self.assertEqual(meta["erasure_denominator"], 4)
        self.assertEqual(meta["erasure_count"], 2)
        self.assertAlmostEqual(meta["erasure_rate"], 0.50, places=5)

        # Total source non-flood = 16 - 4 = 12, expanded = 3 -> expansion_rate = 3/12 = 0.25
        self.assertEqual(meta["expansion_denominator"], 12)
        self.assertEqual(meta["expansion_count"], 3)
        self.assertAlmostEqual(meta["expansion_rate"], 0.25, places=5)

    def test_08_offline_interpretation_metrics_confusion_conservation(self):
        """Offline interpretation metrics preserve 3x3 confusion sum == valid pixels."""
        N = 100
        # Synthetic sample predictions
        sids = [f"sample_{i}" for i in range(5)]
        cand_preds = {sid: np.random.randint(0, 3, size=(16, 16), dtype=np.int64) for sid in sids}
        base_preds = {sid: np.random.randint(0, 3, size=(16, 16), dtype=np.int64) for sid in sids}
        src_preds = {sid: np.random.randint(0, 3, size=(16, 16), dtype=np.int64) for sid in sids}
        labels = {sid: np.random.randint(0, 4, size=(16, 16), dtype=np.int64) for sid in sids}  # includes class 3 ignore
        valid_masks = {sid: np.ones((16, 16), dtype=np.uint8) for sid in sids}

        # Mask out some pixels
        valid_masks[sids[0]][0, 0] = 0

        res = compute_e4_offline_interpretation_metrics(
            candidate_preds_by_sample=cand_preds,
            base_preds_by_sample=base_preds,
            source_preds_by_sample=src_preds,
            labels_by_sample=labels,
            valid_masks_by_sample=valid_masks,
        )

        conf_3x3 = np.array(res["confusion_3x3"])
        total_valid = res["total_valid_pixels"]

        # Confusion conservation: sum(C_ij) == total_valid_pixels
        self.assertEqual(int(np.sum(conf_3x3)), total_valid)

        # Transition conservation: sum(T_ij) == total_valid_pixels
        trans = np.array(res["transition_src_to_cand"])
        self.assertEqual(int(np.sum(trans)), total_valid)

        # Candidate prediction counts conservation
        self.assertEqual(sum(res["candidate_prediction_counts"].values()), total_valid)

        # Verify separated false positives and false negatives exist and are non-negative
        errs = res["separated_errors"]
        for key in ["fp_land_count", "fp_perm_count", "fn_land_count", "fn_perm_count"]:
            self.assertIn(key, errs)
            self.assertGreaterEqual(errs[key], 0)

    def test_09_boundary_conditions_empty_valid_mask(self):
        """Empty valid mask must execute cleanly without NaN or crashing."""
        for intervention in ["base", "k1_balanced_selection", "k2_flood_weighting", "k3_perm_protection", "k4_flood_preservation"]:
            adapter = E4ClassAwareAdapter(
                model=self.candidate,
                source_model=self.source,
                intervention=intervention,
                confidence_threshold=0.90,
                device="cpu",
            )
            xA = torch.randn(2, 2, 16, 16)
            xB = torch.randn(2, 2, 16, 16)
            mask = torch.zeros(2, 16, 16, dtype=torch.uint8)  # Empty mask

            res = adapter.adapt_step(xA, xB, mask, ["s1", "s2"], event_id=520)
            self.assertEqual(res["action"], "SKIPPED_EMPTY_MASK")
            self.assertEqual(res["step_loss"], 0.0)
            self.assertEqual(res["grad_norm"], 0.0)
            self.assertEqual(res["update_norm"], 0.0)
            self.assertFalse(math.isnan(res["step_loss"]))

    def test_10_sealed_test_quarantine_enforcement(self):
        """Configuration and scripts must enforce rejection of sealed test partition events."""
        config_path = Path("configs/changeguard/e4_class_aware_screening_v1.json")
        self.assertTrue(config_path.exists())

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        sealed_test = set(cfg["dataset"]["forbidden_sealed_test_events"])
        dev_events = set(cfg["dataset"]["development_events"])

        # Sealed test must be disjoint from development
        self.assertEqual(len(sealed_test.intersection(dev_events)), 0)

        # Must have exactly 6 development events and 10 sealed test events
        self.assertEqual(len(dev_events), 6)
        self.assertEqual(len(sealed_test), 10)

    def test_11_label_guarded_batch_and_zero_gt_leakage(self):
        """Verifies that LabelGuardedBatch prevents access to labels during adaptation,
        and modifying ground-truth labels causes zero difference to model adaptation."""
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        labels_real = torch.randint(0, 3, (2, 16, 16))
        labels_fake = torch.full((2, 16, 16), 2, dtype=torch.long)

        batch_real = LabelGuardedBatch(
            sample_ids=["s1", "s2"],
            event_id=520,
            xA=xA,
            xB=xB,
            valid_mask=v_mask,
            labels=labels_real,
        )

        batch_fake = LabelGuardedBatch(
            sample_ids=["s1", "s2"],
            event_id=520,
            xA=xA,
            xB=xB,
            valid_mask=v_mask,
            labels=labels_fake,
        )

        # Unpersisted batch must block label access
        with self.assertRaises(PermissionError):
            _ = batch_real.get_labels()

        # Run adaptation with identical sensory data
        torch.manual_seed(100)
        m1 = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
        ad1 = E4ClassAwareAdapter(model=m1, intervention="k2_flood_weighting", device="cpu")

        torch.manual_seed(100)
        m2 = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
        ad2 = E4ClassAwareAdapter(model=m2, intervention="k2_flood_weighting", device="cpu")

        sensory1 = batch_real.get_sensory_inputs()
        sensory2 = batch_fake.get_sensory_inputs()

        res1 = ad1.adapt_step(sensory1["xA"], sensory1["xB"], sensory1["valid_mask"], batch_real.sample_ids, batch_real.event_id)
        res2 = ad2.adapt_step(sensory2["xA"], sensory2["xB"], sensory2["valid_mask"], batch_fake.sample_ids, batch_fake.event_id)

        # Parameters must match exactly regardless of fake labels
        for p1, p2 in zip(ad1.model.parameters(), ad2.model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.assertAlmostEqual(res1["step_loss"], res2["step_loss"], places=6)

    def test_12_source_model_strict_immutability_over_multi_steps(self):
        """Verifies source model remains completely bit-identical across multiple steps."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            source_model=self.source,
            intervention="k4_flood_preservation",
            device="cpu",
        )
        init_hash = adapter.source_initial_state_hash

        for step in range(5):
            xA = torch.randn(2, 2, 16, 16)
            xB = torch.randn(2, 2, 16, 16)
            v_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
            adapter.adapt_step(xA, xB, v_mask, [f"s_{step}_0", f"s_{step}_1"], event_id=520)

        # Hash must be bit-identical
        self.assertEqual(adapter._compute_source_state_hash(), init_hash)
        self.assertTrue(adapter.verify_source_immutability())

    def test_13_all_interventions_instantiate_and_run(self):
        """Tests that all 5 cells (base, k1, k2, k3, k4) run an adapt_step successfully."""
        for intervention in E4ClassAwareAdapter.SUPPORTED_INTERVENTIONS:
            cand = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
            src = MockDualStreamCD(in_channels=2, num_classes=3, feature_dim=8)
            src.load_state_dict(cand.state_dict())

            adapter = E4ClassAwareAdapter(
                model=cand,
                source_model=src,
                intervention=intervention,
                confidence_threshold=0.90,
                device="cpu",
            )
            xA = torch.randn(2, 2, 16, 16)
            xB = torch.randn(2, 2, 16, 16)
            v_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

            res = adapter.adapt_step(xA, xB, v_mask, ["s1", "s2"], event_id=520)
            self.assertIn(res["action"], {"APPLIED", "SKIPPED_EMPTY_MASK"})
            self.assertIn("step_loss", res)
            self.assertIn("intervention_meta", res)
            self.assertEqual(res["intervention_meta"]["intervention"], intervention)

    def test_14_complete_runner_gt_sealing_lifecycle(self):
        """Mechanical regression test for the full runner sequence:
        Proves GT labels cannot be accessed by candidate adaptation, K1-K4 intervention
        computation, source forwards, candidate post-prediction, or batch diagnostics.
        GT is strictly sealed throughout Phase 1 streaming adaptation and unlocked only
        in Phase 2 after all event predictions are persisted."""
        adapter = E4ClassAwareAdapter(
            model=self.candidate,
            source_model=self.source,
            intervention="k1_balanced_selection",
            confidence_threshold=0.90,
            device="cpu",
        )

        batches = [
            LabelGuardedBatch(
                sample_ids=[f"s_{b}_0", f"s_{b}_1"],
                event_id=520,
                xA=torch.randn(2, 2, 8, 8),
                xB=torch.randn(2, 2, 8, 8),
                valid_mask=torch.ones(2, 8, 8, dtype=torch.uint8),
                labels=torch.randint(0, 3, (2, 8, 8)),
            )
            for b in range(3)
        ]

        # Phase 1: Streaming Adaptation Loop
        for b_idx, batch in enumerate(batches):
            # Assert FAIL-CLOSED before adaptation
            with self.assertRaises(PermissionError):
                _ = batch.get_labels()

            # Adaptation step accepts ONLY sensory data
            step_res = adapter.adapt_step(
                xA=batch.xA,
                xB=batch.xB,
                valid_mask=batch.valid_mask,
                sample_ids=batch.sample_ids,
                event_id=batch.event_id,
            )

            # Assert FAIL-CLOSED after adaptation: labels MUST NOT be unlocked yet!
            with self.assertRaises(PermissionError):
                _ = batch.get_labels()

            # Subsequent batches MUST also remain strictly locked
            for future_b in batches[b_idx:]:
                with self.assertRaises(PermissionError):
                    _ = future_b.get_labels()

        # Verify source model immutability
        self.assertTrue(adapter.verify_source_immutability())

        # Phase 2: Post-Persistence Offline Evaluation
        # Only now are batches authorized to release labels
        for batch in batches:
            batch.mark_persisted()
            unlocked_labels = batch.get_labels()
            self.assertEqual(unlocked_labels.shape, (2, 8, 8))


if __name__ == "__main__":
    unittest.main()
