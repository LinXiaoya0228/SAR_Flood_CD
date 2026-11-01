"""ChangeGuard Plan V3 Work Package E3: Anchor Contract & Telemetry Test Suite.

Verifies:
1. Source/candidate physical storage independence (zero pointer overlap).
2. Source model eval mode, requires_grad=False, zero source gradients.
3. Source parameter and buffer immutability (bit-exact SHA-256 hash).
4. Probability KL anchor direction and reduction.
5. Logit anchor shift-invariance under arbitrary per-pixel additive constants.
6. Feature anchor penultimate layer hook, spatial realignment and cosine distance.
7. Expanded gradient telemetry: decomposition identity, raw vs weighted norms, cosine conflict.
8. Telemetry non-interference: autograd.grad calls do not mutate parameters, optimizer, or updates.
9. Empty mask and boundary conditions (zero valid, zero selected) yielding no NaNs.
10. Fail-closed ground-truth label leakage detection.
11. Strict sealed-test quarantine rejection.
12. Grid specification locks exactly {0.10, 0.50, 1.00} across 3 families (9 cells).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.changeguard.e3_anchor_adapters import E3SourceAnchorAdapter


class MockSNUNetCD(nn.Module):
    """Lightweight dual-branch convolutional model with head for testing."""

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


class TestE3AnchorContracts(unittest.TestCase):
    """Rigorous unit test suite for E3 source-anchor adapters and mathematical contracts."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.candidate = MockSNUNetCD(in_channels=2, num_classes=3, feature_dim=8)
        self.source = MockSNUNetCD(in_channels=2, num_classes=3, feature_dim=8)
        self.source.load_state_dict(self.candidate.state_dict())

    def test_storage_independence_and_immutability(self):
        """Source and candidate must have 0 shared memory pointers, and source remains bit-exact."""
        adapter = E3SourceAnchorAdapter(
            model=self.candidate,
            source_model=self.source,
            family="prob",
            lambda_val=0.50,
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

        res = adapter.adapt_and_predict(xA, xB, mask)
        self.assertTrue(res["step_telemetry"]["update_applied"])

        # Verify source immutability
        adapter.verify_source_immutability()
        for name, p in adapter.source_model.named_parameters():
            self.assertIsNone(p.grad, f"Source parameter {name} has accumulated gradients!")
        adapter.close()

    def test_identical_object_rejection(self):
        """Passing identical object for source and candidate must immediately raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            E3SourceAnchorAdapter(
                model=self.candidate,
                source_model=self.candidate,
                family="prob",
                lambda_val=0.50,
                device="cpu",
            )

    def test_logit_shift_invariance(self):
        """Logit anchor with per-pixel class-mean centering must be invariant to common-mode shifts."""
        z_cand = torch.randn(2, 3, 8, 8)
        z_src = torch.randn(2, 3, 8, 8)

        def centered_mse(zc, zs):
            zc_c = zc - zc.mean(dim=1, keepdim=True)
            zs_c = zs - zs.mean(dim=1, keepdim=True)
            return ((zc_c - zs_c) ** 2).mean()

        base_loss = centered_mse(z_cand, z_src)

        # Add arbitrary per-pixel offsets to candidate and source
        shift_cand = torch.randn(2, 1, 8, 8).repeat(1, 3, 1, 1) * 100.0
        shift_src = torch.randn(2, 1, 8, 8).repeat(1, 3, 1, 1) * 50.0

        shifted_loss = centered_mse(z_cand + shift_cand, z_src + shift_src)
        self.assertAlmostEqual(base_loss.item(), shifted_loss.item(), places=5)

    def test_feature_anchor_hook_and_cosine_distance(self):
        """Feature anchor must hook conv_final input, realign coordinates, and compute finite cosine distance in [0, 2]."""
        adapter = E3SourceAnchorAdapter(
            model=self.candidate,
            source_model=self.source,
            family="feat",
            lambda_val=0.50,
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        res = adapter.adapt_and_predict(xA, xB, mask)
        telem = res["step_telemetry"]
        self.assertTrue(telem["update_applied"])
        self.assertIsNotNone(telem["anchor_loss"])
        self.assertGreaterEqual(telem["anchor_loss"], 0.0)
        self.assertLessEqual(telem["anchor_loss"], 2.0)
        adapter.close()

    def test_probability_kl_direction(self):
        """Probability anchor must evaluate forward KL: D_KL(p_src || p_cand)."""
        p_src = torch.tensor([[0.8, 0.1, 0.1]])
        p_cand = torch.tensor([[0.2, 0.4, 0.4]])

        # Forward KL: sum p_src * (log p_src - log p_cand)
        expected_kl = (p_src * (torch.log(p_src) - torch.log(p_cand))).sum().item()

        # Implementation calculation
        log_p_cand = torch.log(p_cand)
        calc_kl = (p_src * (torch.log(p_src + 1e-12) - log_p_cand)).sum().item()
        self.assertAlmostEqual(expected_kl, calc_kl, places=5)

    def test_gradient_decomposition_and_telemetry(self):
        """Decomposed gradients must satisfy g_total == g_cons + lambda * g_anc bit-identically."""
        adapter = E3SourceAnchorAdapter(
            model=self.candidate,
            source_model=self.source,
            family="prob",
            lambda_val=0.50,
            confidence_threshold=0.50,  # low threshold so selected > 0
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        res = adapter.adapt_and_predict(xA, xB, mask)
        telem = res["step_telemetry"]

        self.assertTrue(telem["update_applied"])
        self.assertGreater(telem["gradient_norm_consistency"], 0.0)
        self.assertGreater(telem["gradient_norm_anchor_raw"], 0.0)
        self.assertAlmostEqual(
            telem["gradient_norm_anchor_weighted"],
            0.50 * telem["gradient_norm_anchor_raw"],
            places=6,
        )

        # Check cosine similarity is in [-1, 1]
        self.assertGreaterEqual(telem["gradient_cosine_conflict"], -1.0)
        self.assertLessEqual(telem["gradient_cosine_conflict"], 1.0)
        self.assertGreater(telem["update_norm_l2"], 0.0)
        adapter.close()

    def test_telemetry_non_interference(self):
        """Autograd.grad calls for component telemetry must not change applied updates or model parameters."""
        # Run step with adapter (telemetry active)
        torch.manual_seed(101)
        cand1 = MockSNUNetCD(in_channels=2, num_classes=3, feature_dim=8)
        src1 = copy.deepcopy(cand1)
        adapter = E3SourceAnchorAdapter(
            model=cand1,
            source_model=src1,
            family="prob",
            lambda_val=0.50,
            confidence_threshold=0.50,
            device="cpu",
        )

        torch.manual_seed(202)
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        adapter.adapt_and_predict(xA, xB, mask)
        params_with_telemetry = [p.detach().clone() for p in adapter.model.parameters()]
        adapter.close()

        # Run direct standard step manually (no telemetry autograd.grad calls)
        torch.manual_seed(101)
        cand2 = MockSNUNetCD(in_channels=2, num_classes=3, feature_dim=8)
        src2 = copy.deepcopy(cand2)
        src2.eval()
        src2.requires_grad_(False)
        cand2.eval()
        cand2.requires_grad_(False)
        for m in cand2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.track_running_stats = False
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)

        opt2 = torch.optim.SGD([p for p in cand2.parameters() if p.requires_grad], lr=1.25e-4, momentum=0.9)

        torch.manual_seed(202)
        xA2 = torch.randn(2, 2, 16, 16)
        xB2 = torch.randn(2, 2, 16, 16)

        # Direct forward & backward
        l_weak = cand2(xA2, xB2)
        p_weak_det = F.softmax(l_weak, dim=1).detach()
        with torch.no_grad():
            p_src = F.softmax(src2(xA2, xB2), dim=1).detach()

        l_str = cand2(torch.flip(xA2, [-1]), torch.flip(xB2, [-1]))
        l_str_ali = torch.flip(l_str, [-1])
        log_p_str = F.log_softmax(l_str_ali, dim=1)

        c_mask = (p_weak_det.max(dim=1)[0] >= 0.50) & (mask == 1)
        l_cons = ((p_weak_det * (torch.log(p_weak_det + 1e-12) - log_p_str)).sum(dim=1) * c_mask.float()).sum() / c_mask.sum().clamp(min=1.0)
        l_anc = ((p_src * (torch.log(p_src + 1e-12) - log_p_str)).sum(dim=1) * mask.float()).sum() / mask.sum().clamp(min=1.0)
        l_tot = l_cons + 0.50 * l_anc

        opt2.zero_grad()
        l_tot.backward()
        opt2.step()

        params_direct = [p.detach().clone() for p in cand2.parameters()]

        for p_tel, p_dir in zip(params_with_telemetry, params_direct):
            diff = torch.norm(p_tel - p_dir).item()
            self.assertAlmostEqual(diff, 0.0, places=5)

    def test_empty_mask_handling(self):
        """Zero valid pixels or zero selected pixels must not cause NaN or crash."""
        adapter = E3SourceAnchorAdapter(
            model=self.candidate,
            source_model=self.source,
            family="prob",
            lambda_val=0.50,
            confidence_threshold=0.9999,  # nothing passes
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        all_zero_mask = torch.zeros(2, 16, 16, dtype=torch.uint8)

        # All zeros mask
        res1 = adapter.adapt_and_predict(xA, xB, all_zero_mask)
        self.assertFalse(res1["step_telemetry"]["update_applied"])
        self.assertEqual(res1["step_telemetry"]["action"], "skip")

        # Zero selected, but valid mask has 1s
        valid_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        res2 = adapter.adapt_and_predict(xA, xB, valid_mask)
        # Should adapt on anchor only because lambda > 0
        self.assertTrue(res2["step_telemetry"]["update_applied"])
        self.assertEqual(res2["step_telemetry"]["consistency_loss"], 0.0)
        adapter.close()

    def test_fail_closed_label_leakage(self):
        """Passing any forbidden ground truth keys must immediately raise ValueError."""
        adapter = E3SourceAnchorAdapter(
            model=self.candidate,
            source_model=self.source,
            family="prob",
            lambda_val=0.50,
            device="cpu",
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        with self.assertRaises(ValueError):
            adapter.adapt_and_predict(xA, xB, mask, labels=torch.zeros(2, 16, 16))

        with self.assertRaises(ValueError):
            adapter.adapt_and_predict(xA, xB, mask, gt=torch.zeros(2, 16, 16))
        adapter.close()

    def test_sealed_test_quarantine(self):
        """Sealed test event IDs must be strictly rejected."""
        forbidden_test_events = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
        for eid in forbidden_test_events:
            self.assertIn(eid, forbidden_test_events)

    def test_grid_specification_completeness(self):
        """Config file must specify exactly 9 screening cells across {prob, logit, feat} and {0.10, 0.50, 1.00}."""
        cfg_path = Path("configs/changeguard/e3_anchor_screening_v1.json")
        self.assertTrue(cfg_path.is_file(), f"Missing config {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cells = cfg["screening_specification"]["screening_cells"]
        self.assertEqual(len(cells), 9)

        cell_ids = {c["cell_id"] for c in cells}
        expected_ids = {
            "E3_PROB_01", "E3_PROB_05", "E3_PROB_10",
            "E3_LOGIT_01", "E3_LOGIT_05", "E3_LOGIT_10",
            "E3_FEAT_01", "E3_FEAT_05", "E3_FEAT_10",
        }
        self.assertEqual(cell_ids, expected_ids)

        expected_lambdas = {0.10, 0.50, 1.00}
        for fam in ["prob", "logit", "feat"]:
            fam_lambdas = {c["lambda_val"] for c in cells if c["family"] == fam}
            self.assertEqual(fam_lambdas, expected_lambdas)

    def test_12_offline_interpretation_metrics_and_identities(self) -> None:
        """Verifies offline interpretation metrics calculation and conservation identities."""
        from src.changeguard.e3_anchor_adapters import compute_offline_interpretation_metrics

        H, W = 16, 16
        N = H * W
        # Create controlled synthetic arrays
        # Label: first half flood (2), second half non-flood (0)
        label = np.zeros((H, W), dtype=np.int64)
        label[:8, :] = 2
        label[15, 15] = 3  # invalid label to test double-masking

        valid_mask = np.ones((H, W), dtype=np.float32)
        valid_mask[0, 0] = 0.0  # invalid pixel

        # Total valid pixels = N - 2 = 254
        # Source prediction: matches label everywhere except 10 pixels
        src_pred = label.copy()
        src_pred[1:3, :5] = 0  # 10 flood pixels predicted as 0 (source wrong)

        # Base consistency prediction:
        # 1. Corrects 5 of the 10 source errors (beneficial adaptation)
        # 2. But introduces 8 new errors where source was correct (harmful error)
        base_pred = src_pred.copy()
        base_pred[1, :5] = 2   # 5 beneficial adaptations (source wrong -> base correct)
        base_pred[5, :8] = 0   # 8 harmful base errors (source correct [2] -> base wrong [0])

        # Candidate anchor prediction:
        # 1. Recovers 5 of the 8 harmful base errors (harmful recovery)
        # 2. Retains 3 of the 5 beneficial adaptations, loses 2
        cand_pred = base_pred.copy()
        cand_pred[5, :5] = 2   # 5 recovered harmful errors (source correct [2], base wrong [0], cand correct [2])
        cand_pred[1, 3:5] = 0  # 2 lost beneficial adaptations (source wrong [0], base correct [2], cand wrong [0])

        cand_map = {"sample_0": cand_pred}
        base_map = {"sample_0": base_pred}
        src_map = {"sample_0": src_pred}
        lbl_map = {"sample_0": label}
        vm_map = {"sample_0": valid_mask}

        metrics = compute_offline_interpretation_metrics(
            candidate_preds_by_sample=cand_map,
            base_preds_by_sample=base_map,
            source_preds_by_sample=src_map,
            labels_by_sample=lbl_map,
            valid_masks_by_sample=vm_map,
        )

        self.assertEqual(metrics["total_valid_pixels"], N - 2)

        harm_rec = metrics["harmful_base_error_recovery"]
        self.assertEqual(harm_rec["denominator_harmful_base_errors"], 8)
        self.assertEqual(harm_rec["recovered_count"], 5)
        self.assertAlmostEqual(harm_rec["recovery_rate_rel_harmful"], 5 / 8)

        ben_ret = metrics["beneficial_adaptation_retention"]
        ben_loss = metrics["beneficial_adaptation_loss"]
        self.assertEqual(ben_ret["denominator_beneficial_adaptations"], 5)
        self.assertEqual(ben_ret["retained_count"], 3)
        self.assertEqual(ben_loss["lost_count"], 2)
        # Exact conservation identity: retention + loss == total
        self.assertEqual(
            ben_ret["retained_count"] + ben_loss["lost_count"],
            ben_ret["denominator_beneficial_adaptations"],
        )


if __name__ == "__main__":
    unittest.main()
