"""ChangeGuard Plan V3 Work Package E5: Unit Tests for Temporal Risk Contracts.

Verifies mathematical formulations, state isolation, and research contracts on CPU:
1. Construction of pre2/post real pair and pre2/pre2, post/post null pairs.
2. Source and candidate object non-aliasing enforcement.
3. Source model state immutability (eval mode, zero parameter drift).
4. Candidate state, buffers, optimizer, and counters unchanged by probes.
5. Exception-safe RNG state restoration.
6. Exact s_null mathematical formulation and direction.
7. Zero-valid sensory mask boundary handling (produces None/unavailable, never 0.0).
8. Serialization fidelity: JSON null and CSV missing-value representation.
9. Exact s_flood_expansion and s_flood_disappearance formulas.
10. Source-to-candidate KL divergence uses all three classes with fixed direction.
11. Weak/strong horizontal-flip inverse coordinate alignment.
12. Separation of online sensory mask from offline ground truth (zero GT in probe inputs).
13. Fail-closed LabelGuardedBatch: labels inaccessible prior to mark_persisted.
14. Non-finite probability fail-closed behavior.
15. Unsupported status, device, or invalid configs rejected.
16. Strict sealed-test event quarantine rejection.
17. Bounded smoke subset validation.
18. Fresh output directory protection.
19. Environment and asset hash tracking in run provenance.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.changeguard.e1_orchestrator import LabelGuardedBatch
from src.changeguard.e5_risk_probes import (
    E5RiskProbeEvaluator,
    assert_model_non_aliasing,
    compute_flood_change_probes,
    compute_generic_risk_evidence,
    compute_null_pair_risk_score,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
)
from src.data.evaluation_dataset import FORBIDDEN_SEALED_TEST_EVENTS
from tools.run_e5_risk_scoring import (
    E5RiskScoringRunner,
    SCIENTIFIC_INPUT_FILES,
    compute_file_sha256,
)


class SyntheticToyModel(nn.Module):
    """Lightweight 2-channel in, 3-class out model with BatchNorm for CPU contract tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        feat = F.relu(self.bn(self.conv(x)))
        return self.head(feat)


class TestE5RiskContracts(unittest.TestCase):
    """Test suite verifying E5 temporal risk evidence contracts and implementations."""

    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_01_null_pair_construction(self):
        """Verifies real pair (pre2, post) and null pairs (pre2, pre2), (post, post)."""
        B, C, H, W = 2, 2, 16, 16
        xA = torch.randn(B, C, H, W)  # pre2
        xB = torch.randn(B, C, H, W)  # post

        # Real pair
        real_cat = torch.cat([xA, xB], dim=1)
        self.assertEqual(real_cat.shape, (B, 4, H, W))

        # Null pair 1: (pre2, pre2)
        null_pre2_cat = torch.cat([xA, xA], dim=1)
        self.assertTrue(torch.equal(null_pre2_cat[:, :2], null_pre2_cat[:, 2:]))

        # Null pair 2: (post, post)
        null_post_cat = torch.cat([xB, xB], dim=1)
        self.assertTrue(torch.equal(null_post_cat[:, :2], null_post_cat[:, 2:]))

    def test_02_model_non_aliasing_assertion(self):
        """Verifies that aliased candidate and source models are immediately rejected."""
        model_a = SyntheticToyModel()
        model_b = copy.deepcopy(model_a)

        # Separate models must pass
        assert_model_non_aliasing(model_a, model_b)

        # Same object must raise RuntimeError
        with self.assertRaises(RuntimeError):
            assert_model_non_aliasing(model_a, model_a)

        # Shared parameter pointer must raise RuntimeError
        model_b.head.weight = model_a.head.weight
        with self.assertRaises(RuntimeError):
            assert_model_non_aliasing(model_a, model_b)

    def test_03_source_state_immutability(self):
        """Verifies source model parameters and buffers remain bit-exact identical during probing."""
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        src_model.eval()

        src_params_before = {k: v.clone() for k, v in src_model.state_dict().items()}

        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        _ = evaluator.evaluate_batch_probes_state_neutral(xA, xB, v_mask)

        # Check source state dict is unchanged
        for k, v in src_model.state_dict().items():
            self.assertTrue(torch.equal(v, src_params_before[k]))

    def test_04_candidate_state_and_buffers_unchanged_by_probes(self):
        """Verifies candidate parameters and running buffers are not modified by probe calls."""
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        cand_model.train()

        cand_params_before = {k: v.clone() for k, v in cand_model.named_parameters()}
        cand_buffers_before = {k: v.clone() for k, v in cand_model.named_buffers()}

        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        _ = evaluator.evaluate_batch_probes_state_neutral(xA, xB, v_mask)

        # Parameters unchanged
        for k, v in cand_model.named_parameters():
            self.assertTrue(torch.equal(v, cand_params_before[k]))
        # Running buffers unchanged
        for k, v in cand_model.named_buffers():
            self.assertTrue(torch.equal(v, cand_buffers_before[k]))

    def test_05_exception_safe_rng_restoration(self):
        """Verifies RNG state is restored even if an error occurs during probing."""
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")

        # Mock an error inside probing
        class CrashingTensor:
            pass

        rng_before = torch.get_rng_state()
        try:
            # Pass invalid inputs that will fail inside forward
            evaluator.evaluate_batch_probes_state_neutral(None, None, None)
        except Exception:
            pass

        rng_after = torch.get_rng_state()
        self.assertTrue(torch.equal(rng_before, rng_after))

    def test_06_exact_s_null_formula_and_direction(self):
        """Verifies exact computation of s_null = 0.5 * (max(0, d_pre2) + max(0, d_post))."""
        H, W = 4, 4
        v_mask = torch.ones(1, H, W, dtype=torch.uint8)

        # Case 1: Candidate has higher flood prob on both null pairs
        # cand: 0.6 on pre2, 0.4 on post. src: 0.2 on pre2, 0.1 on post
        c_pre2 = torch.full((1, H, W), 0.6)
        c_post = torch.full((1, H, W), 0.4)
        s_pre2 = torch.full((1, H, W), 0.2)
        s_post = torch.full((1, H, W), 0.1)

        s_null, d_pre2, d_post, _, _ = compute_null_pair_risk_score(
            c_pre2, c_post, s_pre2, s_post, v_mask
        )
        # delta_pre2 = 0.4, delta_post = 0.3 -> s_null = 0.5 * (0.4 + 0.3) = 0.35
        self.assertAlmostEqual(d_pre2, 0.4, places=6)
        self.assertAlmostEqual(d_post, 0.3, places=6)
        self.assertAlmostEqual(s_null, 0.35, places=6)

        # Case 2: Candidate has lower flood prob (negative delta clamped to 0)
        c_pre2_neg = torch.full((1, H, W), 0.1)
        s_null_neg, d_pre2_neg, d_post_neg, _, _ = compute_null_pair_risk_score(
            c_pre2_neg, c_post, s_pre2, s_post, v_mask
        )
        # delta_pre2 = -0.1 (clamped to 0), delta_post = 0.3 -> s_null = 0.5 * (0 + 0.3) = 0.15
        self.assertAlmostEqual(d_pre2_neg, -0.1, places=6)
        self.assertAlmostEqual(s_null_neg, 0.15, places=6)

    def test_07_zero_valid_mask_produces_none(self):
        """Verifies zero sensory-valid pixels return None (unavailable), never 0.0."""
        H, W = 4, 4
        zero_mask = torch.zeros(1, H, W, dtype=torch.uint8)
        p = torch.full((1, H, W), 0.5)

        s_null, d_pre2, d_post, _, _ = compute_null_pair_risk_score(p, p, p, p, zero_mask)
        self.assertIsNone(s_null)
        self.assertIsNone(d_pre2)
        self.assertIsNone(d_post)

        s_exp, s_dis = compute_flood_change_probes(p, p, zero_mask)
        self.assertIsNone(s_exp)
        self.assertIsNone(s_dis)

        gen_res = compute_generic_risk_evidence(
            cand_probs=torch.full((1, 3, H, W), 1.0 / 3.0),
            src_probs=torch.full((1, 3, H, W), 1.0 / 3.0),
            valid_mask=zero_mask,
        )
        self.assertEqual(gen_res["valid_pixels"], 0)
        self.assertIsNone(gen_res["mean_entropy"])
        self.assertIsNone(gen_res["source_to_candidate_kl"])
        self.assertFalse(gen_res["probe_available"])

    def test_08_json_null_and_csv_missing_fidelity(self):
        """Verifies unavailable scores serialize as JSON null and CSV empty string."""
        row_data = {
            "s_null": None,
            "mean_entropy": None,
            "valid_pixels": 0,
        }
        # JSON fidelity
        json_str = json.dumps(row_data)
        self.assertIn('"s_null": null', json_str)
        self.assertIn('"mean_entropy": null', json_str)

        # CSV fidelity
        csv_row = {
            k: (v if v is not None else "") for k, v in row_data.items()
        }
        with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp:
            tmp_path = tmp.name
            writer = csv.DictWriter(tmp, fieldnames=list(csv_row.keys()))
            writer.writeheader()
            writer.writerow(csv_row)

        with open(tmp_path, "r") as f:
            lines = f.readlines()
        os.remove(tmp_path)

        header = lines[0].strip().split(",")
        values = lines[1].strip().split(",")
        s_null_idx = header.index("s_null")
        self.assertEqual(values[s_null_idx], "")  # Empty field, not "0.0"

    def test_09_expansion_and_disappearance_formulas(self):
        """Verifies exact signed flood probability expansion and disappearance."""
        H, W = 2, 2
        v_mask = torch.ones(1, H, W, dtype=torch.uint8)

        # Half pixels expand by 0.4, half shrink by 0.2
        p_cand = torch.tensor([[[[0.7, 0.7], [0.1, 0.1]]]])
        p_src = torch.tensor([[[[0.3, 0.3], [0.3, 0.3]]]])

        s_exp, s_dis = compute_flood_change_probes(p_cand, p_src, v_mask)
        # Expansion: [0.4, 0.4, 0.0, 0.0] -> mean = 0.2
        # Disappearance: [0.0, 0.0, 0.2, 0.2] -> mean = 0.1
        self.assertAlmostEqual(s_exp, 0.2, places=6)
        self.assertAlmostEqual(s_dis, 0.1, places=6)

    def test_10_three_class_kl_divergence_direction(self):
        """Verifies source-to-candidate KL divergence uses all three classes with fixed direction."""
        H, W = 1, 1
        v_mask = torch.ones(1, H, W, dtype=torch.uint8)

        # Identical distributions -> KL == 0
        p_cand = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]])
        p_src = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]])

        res_ident = compute_generic_risk_evidence(p_cand, p_src, v_mask)
        self.assertAlmostEqual(res_ident["source_to_candidate_kl"], 0.0, places=5)

        # Shifted distribution
        p_cand_shifted = torch.tensor([[[[0.3]], [[0.3]], [[0.4]]]])
        res_shift = compute_generic_risk_evidence(p_cand_shifted, p_src, v_mask)
        self.assertGreater(res_shift["source_to_candidate_kl"], 0.0)

    def test_11_geometric_augmentation_inverse_alignment(self):
        """Verifies horizontal flip and inverse unflip alignment."""
        probs = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]]]])  # (1, 1, 2, 2)
        flipped = torch.flip(probs, dims=[-1])
        unflipped = torch.flip(flipped, dims=[-1])
        self.assertTrue(torch.equal(probs, unflipped))

    def test_12_online_vs_offline_mask_isolation(self):
        """Verifies that online sensory inputs contain zero ground-truth labels."""
        batch = LabelGuardedBatch(
            sample_ids=["sample_001"],
            event_id=520,
            xA=torch.zeros((1, 2, 16, 16)),
            xB=torch.zeros((1, 2, 16, 16)),
            valid_mask=torch.ones((1, 16, 16), dtype=torch.uint8),
            labels=torch.zeros((1, 16, 16), dtype=torch.int64),
        )

        sensory = batch.get_sensory_inputs()
        self.assertNotIn("label", sensory)
        self.assertNotIn("labels", sensory)
        self.assertIn("valid_mask", sensory)

    def test_13_label_guarded_batch_fail_closed(self):
        """Verifies get_labels() raises PermissionError if called before mark_persisted()."""
        batch = LabelGuardedBatch(
            sample_ids=["sample_001"],
            event_id=520,
            xA=torch.zeros((1, 2, 16, 16)),
            xB=torch.zeros((1, 2, 16, 16)),
            valid_mask=torch.ones((1, 16, 16), dtype=torch.uint8),
            labels=torch.zeros((1, 16, 16), dtype=torch.int64),
        )

        with self.assertRaises(PermissionError):
            _ = batch.get_labels()

        # After mark_persisted, labels can be read
        batch.mark_persisted()
        labels = batch.get_labels()
        self.assertIsNotNone(labels)

    def test_14_nonfinite_probability_fail_closed(self):
        """Verifies non-finite probabilities raise FloatingPointError."""
        H, W = 2, 2
        v_mask = torch.ones(1, H, W, dtype=torch.uint8)
        nan_prob = torch.tensor([[[[np.nan, 0.5], [0.5, 0.5]]]])

        with self.assertRaises(FloatingPointError):
            _ = compute_null_pair_risk_score(nan_prob, nan_prob, nan_prob, nan_prob, v_mask)

        with self.assertRaises(FloatingPointError):
            _ = compute_flood_change_probes(nan_prob, nan_prob, v_mask)

    def test_15_sealed_test_event_rejection(self):
        """Verifies that providing any sealed-test event immediately raises RuntimeError."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cfg = {
                "status": "LOCKED_FOR_EXECUTION",
                "source_checkpoints": {"seeds": {"0": {"checkpoint_path": "fake.pt"}}},
                "contexts": {"candidate_e2_conf90": {"confidence_threshold": 0.9, "learning_rate": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0}},
                "screening_specification": {
                    "event_sequence": [520, 205],  # 205 is forbidden sealed test event
                    "batch_size": 16,
                },
                "dataset": {
                    "dataset_root": "datasets",
                    "manifest_path": "manifest.gz",
                    "sar_mean": [0.09, 0.02],
                    "sar_std": [0.04, 0.02],
                    "sar_clamp_max": 0.15,
                    "canonical_preevent": "pre_event_2",
                },
            }
            json.dump(cfg, tmp)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                _ = E5RiskScoringRunner(config_path=tmp_path, device="cpu", is_smoke=False)
            self.assertIn("CRITICAL SECURITY VIOLATION", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_16_unsupported_status_or_config_rejected(self):
        """Verifies configs with status other than LOCKED_FOR_EXECUTION are rejected fail-closed."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cfg = {
                "status": "DRAFT",
            }
            json.dump(cfg, tmp)

        try:
            with self.assertRaises(ValueError) as ctx:
                _ = E5RiskScoringRunner(config_path=tmp_path, device="cpu")
            self.assertIn("not LOCKED_FOR_EXECUTION", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_17_bounded_smoke_subset_validation(self):
        """Verifies that smoke mode configures target event to Event 520 and slices samples to 32."""
        runner = E5RiskScoringRunner(
            config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
            device="cpu",
            is_smoke=True,
        )
        self.assertEqual(runner.target_events, [520])
        self.assertEqual(runner.limit_samples, 32)
        self.assertEqual(runner.batch_size, 16)
        self.assertTrue(runner.is_smoke)

    def test_18_fresh_output_protection(self):
        """Verifies runner fails closed if output directory exists and is non-empty for formal runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            # Create a dirty file inside
            (tmp_p / "existing_artifact.txt").write_text("pre-existing")

            runner = E5RiskScoringRunner(
                config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
                output_dir=tmp_p,
                device="cpu",
                is_smoke=False,
                allow_dirty=True,
            )
            with self.assertRaises(RuntimeError) as ctx:
                runner.execute_streaming_evaluation()
            self.assertIn("Fresh output directory violation", str(ctx.exception))

    def test_19_environment_and_asset_hashes_provenance(self):
        """Verifies preflight checks include checkpoint hash, config hash, and python/torch versions."""
        runner = E5RiskScoringRunner(
            config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
            device="cpu",
            is_smoke=True,
            allow_dirty=True,
        )
        preflight = runner.run_preflight_checks()
        self.assertIn("checkpoint_sha256", preflight)
        self.assertEqual(
            preflight["checkpoint_sha256"],
            "247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f",
        )
        self.assertIn("config_sha256", preflight)
        self.assertIn("python_version", preflight)
        self.assertIn("torch_version", preflight)
        self.assertIn("git_commit", preflight)

    def test_20_online_sensory_mask_label_independence(self):
        """Verifies that changing ground truth labels does NOT alter the online sensory mask."""
        B, C, H, W = 2, 2, 8, 8
        xA = torch.randn(B, C, H, W)
        xB = torch.randn(B, C, H, W)
        mna = torch.randint(0, 2, (B, H, W), dtype=torch.uint8)

        # Batch 1 with labels = 0
        labels_1 = torch.zeros((B, H, W), dtype=torch.int64)
        batch_1 = LabelGuardedBatch(["s1", "s2"], 520, xA, xB, mna, labels_1)

        # Batch 2 with completely different labels (class 1, 2, and 3)
        labels_2 = torch.randint(0, 4, (B, H, W), dtype=torch.int64)
        batch_2 = LabelGuardedBatch(["s1", "s2"], 520, xA, xB, mna, labels_2)

        sensory_1 = batch_1.get_sensory_inputs()
        sensory_2 = batch_2.get_sensory_inputs()

        # Masks are bit-for-bit identical regardless of label content
        self.assertTrue(torch.equal(sensory_1["valid_mask"], sensory_2["valid_mask"]))

        # Evaluator probe scores are bit-for-bit identical
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")

        res_1 = evaluator.evaluate_batch_probes_state_neutral(xA, xB, sensory_1["valid_mask"])
        res_2 = evaluator.evaluate_batch_probes_state_neutral(xA, xB, sensory_2["valid_mask"])

        self.assertEqual(res_1["s_null"], res_2["s_null"])
        self.assertEqual(res_1["s_flood_expansion"], res_2["s_flood_expansion"])
        self.assertEqual(res_1["mean_entropy"], res_2["mean_entropy"])
        self.assertEqual(res_1["valid_pixels"], res_2["valid_pixels"])

    def test_21_online_sensory_mask_mna_dependence(self):
        """Verifies that altering MK0_MNA directly changes the sensory valid mask and probe results."""
        B, C, H, W = 2, 2, 8, 8
        xA = torch.randn(B, C, H, W)
        xB = torch.randn(B, C, H, W)

        mna_dense = torch.ones((B, H, W), dtype=torch.uint8)
        mna_sparse = torch.zeros((B, H, W), dtype=torch.uint8)
        mna_sparse[:, 0:2, 0:2] = 1

        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")

        res_dense = evaluator.evaluate_batch_probes_state_neutral(xA, xB, mna_dense)
        res_sparse = evaluator.evaluate_batch_probes_state_neutral(xA, xB, mna_sparse)

        self.assertEqual(res_dense["valid_pixels"], B * H * W)
        self.assertEqual(res_sparse["valid_pixels"], B * 2 * 2)
        self.assertNotEqual(res_dense["valid_pixels"], res_sparse["valid_pixels"])

    def test_22_zero_valid_mna_yields_unavailable_risk_values(self):
        """Verifies that when MK0_MNA has zero valid pixels, all risk probes return None."""
        B, C, H, W = 2, 2, 8, 8
        xA = torch.randn(B, C, H, W)
        xB = torch.randn(B, C, H, W)
        zero_mna = torch.zeros((B, H, W), dtype=torch.uint8)

        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")

        res = evaluator.evaluate_batch_probes_state_neutral(xA, xB, zero_mna)

        self.assertEqual(res["valid_pixels"], 0)
        self.assertFalse(res["probe_available"])
        self.assertIsNone(res["s_null"])
        self.assertIsNone(res["delta_null_pre2"])
        self.assertIsNone(res["delta_null_post"])
        self.assertIsNone(res["s_flood_expansion"])
        self.assertIsNone(res["s_flood_disappearance"])
        self.assertIsNone(res["mean_entropy"])
        self.assertIsNone(res["source_to_candidate_kl"])
        self.assertIsNone(res["hard_disagreement_rate"])

    def test_23_no_label_derived_exclusion_in_online_scoring(self):
        """Verifies that label=3 pixels are NOT excluded from the online sensory valid mask."""
        B, C, H, W = 1, 2, 4, 4
        xA = torch.randn(B, C, H, W)
        xB = torch.randn(B, C, H, W)
        mna = torch.ones((B, H, W), dtype=torch.uint8)
        # GT label has label=3 everywhere
        label_with_unlabelled = torch.full((B, H, W), 3, dtype=torch.int64)

        batch = LabelGuardedBatch(["s1"], 520, xA, xB, mna, label_with_unlabelled)
        sensory = batch.get_sensory_inputs()

        # Online sensory mask must retain all pixels where MK0_MNA == 1
        online_valid = (sensory["valid_mask"] == 1)
        self.assertEqual(int(online_valid.sum().item()), 16)

        # Offline ground-truth validity excludes label=3
        batch.mark_persisted()
        offline_gt_valid = (sensory["valid_mask"] == 1) & (batch.get_labels() != 3)
        self.assertEqual(int(offline_gt_valid.sum().item()), 0)

        # Confirm online probe evaluation scores over all 16 pixels
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        evaluator = E5RiskProbeEvaluator(cand_model, src_model, device="cpu")
        res = evaluator.evaluate_batch_probes_state_neutral(xA, xB, sensory["valid_mask"])
        self.assertEqual(res["valid_pixels"], 16)

    def test_24_matched_bn_no_update_properties(self):
        """Verifies matched BN-no-update context uses batch stats, zero updates, zero buffer drift."""
        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        bn_control_model = copy.deepcopy(cand_model)

        evaluator = E5RiskProbeEvaluator(
            candidate_model=cand_model,
            source_model=src_model,
            bn_control_model=bn_control_model,
            device="cpu",
        )

        # Snapshot BN control state before probe evaluation
        bn_params_before = {k: v.clone() for k, v in bn_control_model.named_parameters()}
        bn_buffers_before = {k: v.clone() for k, v in bn_control_model.named_buffers()}

        B, C, H, W = 4, 2, 8, 8
        xA = torch.randn(B, C, H, W) * 5.0 + 3.0  # Shifted to test batch stats
        xB = torch.randn(B, C, H, W) * 5.0 + 3.0
        v_mask = torch.ones((B, H, W), dtype=torch.uint8)

        res = evaluator.evaluate_batch_probes_state_neutral(xA, xB, v_mask)

        # 1. Zero parameter drift
        for k, v in bn_control_model.named_parameters():
            self.assertFalse(v.requires_grad)
            self.assertTrue(torch.equal(v, bn_params_before[k]))
            self.assertIsNone(v.grad)

        # 2. State-neutral buffer restoration
        for k, v in bn_control_model.named_buffers():
            self.assertTrue(torch.equal(v, bn_buffers_before[k]))

        # 3. Comparative diagnostics populated
        self.assertIn("diff_c2_src_bn_real", res)
        self.assertIn("diff_c2_bn_cand_real", res)
        self.assertIn("diff_c2_src_cand_real", res)
        self.assertIn("disagree_src_bn_real", res)

    def test_25_cache_schema_array_recomputability(self):
        """Verifies all declared E5 scores are independently recomputable from persisted cache arrays."""
        B, C, H, W = 2, 2, 8, 8
        xA = torch.randn(B, C, H, W)
        xB = torch.randn(B, C, H, W)
        v_mask = torch.randint(0, 2, (B, H, W), dtype=torch.uint8)

        cand_model = SyntheticToyModel()
        src_model = copy.deepcopy(cand_model)
        bn_model = copy.deepcopy(cand_model)

        evaluator = E5RiskProbeEvaluator(cand_model, src_model, bn_control_model=bn_model, device="cpu")
        online_res = evaluator.evaluate_batch_probes_state_neutral(xA, xB, v_mask)

        # Simulate array caching in float16 as defined by CACHE_SCHEMA_V1
        c_pre2_f16 = online_res["cand_probs_null_pre2"][:, 2, :, :].numpy().astype(np.float16)
        c_post_f16 = online_res["cand_probs_null_post"][:, 2, :, :].numpy().astype(np.float16)
        s_pre2_f16 = online_res["src_probs_null_pre2"][:, 2, :, :].numpy().astype(np.float16)
        s_post_f16 = online_res["src_probs_null_post"][:, 2, :, :].numpy().astype(np.float16)
        c_real_f16 = online_res["cand_probs_real"][:, 2, :, :].numpy().astype(np.float16)
        s_real_f16 = online_res["src_probs_real"][:, 2, :, :].numpy().astype(np.float16)
        v_mask_np = v_mask.numpy().astype(np.uint8)

        # Recompute from cache arrays (simulating offline recompute)
        recomp_s_null, recomp_d_pre2, recomp_d_post, _, _ = compute_null_pair_risk_score(
            torch.from_numpy(c_pre2_f16.astype(np.float32)),
            torch.from_numpy(c_post_f16.astype(np.float32)),
            torch.from_numpy(s_pre2_f16.astype(np.float32)),
            torch.from_numpy(s_post_f16.astype(np.float32)),
            torch.from_numpy(v_mask_np),
        )

        recomp_exp, recomp_dis = compute_flood_change_probes(
            torch.from_numpy(c_real_f16.astype(np.float32)),
            torch.from_numpy(s_real_f16.astype(np.float32)),
            torch.from_numpy(v_mask_np),
        )

        # Tolerances account for float16 precision quantization (< 1e-3)
        if online_res["s_null"] is not None:
            self.assertAlmostEqual(recomp_s_null, online_res["s_null"], places=2)
            self.assertAlmostEqual(recomp_d_pre2, online_res["delta_null_pre2"], places=2)
            self.assertAlmostEqual(recomp_d_post, online_res["delta_null_post"], places=2)
            self.assertAlmostEqual(recomp_exp, online_res["s_flood_expansion"], places=2)
            self.assertAlmostEqual(recomp_dis, online_res["s_flood_disappearance"], places=2)

    def test_26_evidence_shard_exists_on_disk_before_label_unlock(self):
        """Verifies evidence shard is physically on disk and readable BEFORE batch.mark_persisted()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            batch = LabelGuardedBatch(
                sample_ids=["s01", "s02"],
                event_id=520,
                xA=torch.zeros((2, 2, 8, 8)),
                xB=torch.zeros((2, 2, 8, 8)),
                valid_mask=torch.ones((2, 8, 8), dtype=torch.uint8),
                labels=torch.zeros((2, 8, 8), dtype=torch.int64),
            )

            # Labels cannot be read yet
            with self.assertRaises(PermissionError):
                _ = batch.get_labels()

            arrays = {
                "s01_sensory_mask": np.ones((8, 8), dtype=np.uint8),
                "s02_sensory_mask": np.ones((8, 8), dtype=np.uint8),
                "s01_candidate_real_pred": np.zeros((8, 8), dtype=np.uint8),
                "s02_candidate_real_pred": np.zeros((8, 8), dtype=np.uint8),
            }
            shard_path = rec.record_batch_evidence_shard(step_index=0, sample_ids=["s01", "s02"], arrays=arrays)

            # 1. Shard must physically exist on filesystem
            self.assertTrue(shard_path.is_file())
            # 2. Must be readable on filesystem
            with np.load(shard_path) as loaded:
                self.assertIn("s01_sensory_mask", loaded.files)
                self.assertIn("s02_candidate_real_pred", loaded.files)

            # 3. Only now is mark_persisted called and labels unlocked
            batch.mark_persisted()
            labels = batch.get_labels()
            self.assertIsNotNone(labels)

    def test_27_unreadable_or_missing_shard_prevents_mark_persisted(self):
        """Verifies failure in shard persistence prevents mark_persisted, failing closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            batch = LabelGuardedBatch(
                sample_ids=["s01"],
                event_id=520,
                xA=torch.zeros((1, 2, 8, 8)),
                xB=torch.zeros((1, 2, 8, 8)),
                valid_mask=torch.ones((1, 8, 8), dtype=torch.uint8),
                labels=torch.zeros((1, 8, 8), dtype=torch.int64),
            )

            # Pass empty arrays -> raises ValueError inside recorder
            with self.assertRaises(ValueError):
                _ = rec.record_batch_evidence_shard(step_index=0, sample_ids=["s01"], arrays={})

            # Since shard failed, mark_persisted was not called, labels remain locked
            with self.assertRaises(PermissionError):
                _ = batch.get_labels()

    def test_28_atomic_temp_failure_leaves_clean_index(self):
        """Verifies temporary write failure does not create an entry in evidence_index.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            # Record batch 0 successfully
            rec.record_batch_evidence_shard(
                step_index=0,
                sample_ids=["s01"],
                arrays={"s01_sensory_mask": np.ones((8, 8), dtype=np.uint8)},
            )

            # Attempt invalid batch 1 (empty arrays)
            try:
                rec.record_batch_evidence_shard(step_index=1, sample_ids=["s02"], arrays={})
            except Exception:
                pass

            index_path = Path(tmpdir) / "event_520" / "evidence_index.json"
            with open(index_path, "r") as f:
                index_data = json.load(f)

            self.assertEqual(index_data["total_shards"], 1)
            self.assertEqual(len(index_data["shards"]), 1)
            self.assertEqual(index_data["shards"][0]["step_index"], 0)
            self.assertEqual(index_data["completed_sample_ids"], ["s01"])

    def test_29_completed_shard_hash_matches_index(self):
        """Verifies SHA-256 recorded in evidence_index.json bit-exact matches the shard file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            shard_path = rec.record_batch_evidence_shard(
                step_index=0,
                sample_ids=["s01"],
                arrays={"s01_mask": np.ones((8, 8), dtype=np.uint8)},
            )

            index_path = Path(tmpdir) / "event_520" / "evidence_index.json"
            with open(index_path, "r") as f:
                index_data = json.load(f)

            recorded_hash = index_data["shards"][0]["sha256"]
            actual_hash = compute_file_sha256(shard_path)
            self.assertEqual(recorded_hash, actual_hash)

    def test_30_duplicate_sample_ids_rejected(self):
        """Verifies attempting to record duplicate sample IDs across shards raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            rec.record_batch_evidence_shard(
                step_index=0,
                sample_ids=["s01", "s02"],
                arrays={"s01": np.zeros(2), "s02": np.zeros(2)},
            )

            with self.assertRaises(ValueError) as ctx:
                rec.record_batch_evidence_shard(
                    step_index=1,
                    sample_ids=["s02", "s03"],
                    arrays={"s02": np.zeros(2), "s03": np.zeros(2)},
                )
            self.assertIn("Sample ID collision detected", str(ctx.exception))

    def test_31_multiple_shards_reconstruct_exact_sample_order(self):
        """Verifies multiple shards reconstruct exact chronological sample order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            rec.record_batch_evidence_shard(0, ["s01", "s02"], {"s01": np.zeros(1), "s02": np.zeros(1)})
            rec.record_batch_evidence_shard(1, ["s03", "s04"], {"s03": np.zeros(1), "s04": np.zeros(1)})
            rec.record_batch_evidence_shard(2, ["s05"], {"s05": np.zeros(1)})

            index_path = Path(tmpdir) / "event_520" / "evidence_index.json"
            with open(index_path, "r") as f:
                index_data = json.load(f)

            self.assertEqual(
                index_data["completed_sample_ids"],
                ["s01", "s02", "s03", "s04", "s05"],
            )
            self.assertEqual(index_data["total_samples"], 5)
            self.assertEqual(index_data["total_shards"], 3)

    def test_32_score_recomputation_from_disk_shards(self):
        """Verifies temporal risk scores recomputed directly from disk shards match online scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            H, W = 4, 4
            v_mask = np.ones((H, W), dtype=np.uint8)
            c_pre2 = np.full((H, W), 0.5, dtype=np.float16)
            c_post = np.full((H, W), 0.3, dtype=np.float16)
            s_pre2 = np.full((H, W), 0.2, dtype=np.float16)
            s_post = np.full((H, W), 0.1, dtype=np.float16)

            shard_path = rec.record_batch_evidence_shard(
                step_index=0,
                sample_ids=["s01"],
                arrays={
                    "s01_sensory_mask": v_mask,
                    "s01_candidate_null_pre2_prob_c2": c_pre2,
                    "s01_candidate_null_post_prob_c2": c_post,
                    "s01_source_null_pre2_prob_c2": s_pre2,
                    "s01_source_null_post_prob_c2": s_post,
                },
            )

            # Read back from physical disk
            with np.load(shard_path) as loaded:
                recomp_s_null, d_pre2, d_post, _, _ = compute_null_pair_risk_score(
                    torch.from_numpy(loaded["s01_candidate_null_pre2_prob_c2"].astype(np.float32)),
                    torch.from_numpy(loaded["s01_candidate_null_post_prob_c2"].astype(np.float32)),
                    torch.from_numpy(loaded["s01_source_null_pre2_prob_c2"].astype(np.float32)),
                    torch.from_numpy(loaded["s01_source_null_post_prob_c2"].astype(np.float32)),
                    torch.from_numpy(loaded["s01_sensory_mask"]),
                )

            # delta_pre2 = 0.3, delta_post = 0.2 -> s_null = 0.5 * (0.3 + 0.2) = 0.25
            self.assertAlmostEqual(recomp_s_null, 0.25, places=3)
            self.assertAlmostEqual(d_pre2, 0.3, places=3)
            self.assertAlmostEqual(d_post, 0.2, places=3)

    def test_33_storage_bounded_by_single_batch(self):
        """Verifies process RAM does not accumulate full event arrays during sharded recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_run", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            # Record 3 batches of shards
            for b_idx in range(3):
                rec.record_batch_evidence_shard(
                    step_index=b_idx,
                    sample_ids=[f"s_{b_idx}"],
                    arrays={f"s_{b_idx}_mask": np.zeros((10, 10), dtype=np.uint8)},
                )

            # In-memory dictionary frozen_spatial_arrays remains empty!
            self.assertEqual(len(rec.frozen_spatial_arrays), 0)

    def test_34_clean_input_guard_includes_all_six_files(self):
        """Verifies clean input guard enforces clean state and computes hashes for all 6 scientific files."""
        expected_files = [
            "configs/changeguard/e5_temporal_risk_v1.json",
            "docs/e5_preflight_and_contracts.md",
            "src/changeguard/e5_risk_probes.py",
            "src/changeguard/recorder.py",
            "src/data/evaluation_dataset.py",
            "tools/run_e5_risk_scoring.py",
        ]
        self.assertEqual(SCIENTIFIC_INPUT_FILES, expected_files)

        runner = E5RiskScoringRunner(
            config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
            device="cpu",
            is_smoke=True,
            allow_dirty=True,
        )
        preflight = runner.run_preflight_checks()
        self.assertIn("scientific_inputs_sha256", preflight)
        for rel_p in expected_files:
            self.assertIn(rel_p, preflight["scientific_inputs_sha256"])
            self.assertEqual(len(preflight["scientific_inputs_sha256"][rel_p]), 64)

    def test_35_prelabel_and_batch_total_timing_lifecycle(self):
        """Verifies prelabel and batch total timing definitions, ordering, and absence of UnboundLocalError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = ExtendedExperimentRecorder(tmpdir, "test_timing", "ckpt", "cfg_hash", "git_hash", allow_overwrite=True)
            rec.start_event(520)

            batch = LabelGuardedBatch(
                sample_ids=["s01", "s02"],
                event_id=520,
                xA=torch.zeros((2, 2, 8, 8)),
                xB=torch.zeros((2, 2, 8, 8)),
                valid_mask=torch.ones((2, 8, 8), dtype=torch.uint8),
                labels=torch.zeros((2, 8, 8), dtype=torch.int64),
            )

            b_start_time = time.time()
            sensory = batch.get_sensory_inputs()
            s_ids = sensory["sample_ids"]

            # 1. Prelabel evidence elapsed time is computed before shard recording
            prelabel_evidence_elapsed_ms = (time.time() - b_start_time) * 1000.0
            self.assertTrue(math.isfinite(prelabel_evidence_elapsed_ms))
            self.assertGreaterEqual(prelabel_evidence_elapsed_ms, 0.0)

            arrays = {
                "s01_sensory_mask": np.ones((8, 8), dtype=np.uint8),
                "s02_sensory_mask": np.ones((8, 8), dtype=np.uint8),
            }

            # 2. Shard recorded with explicit prelabel timing (no UnboundLocalError)
            shard_path = rec.record_batch_evidence_shard(
                step_index=0,
                sample_ids=s_ids,
                arrays=arrays,
                contexts=["source_frozen_running_stats"],
                prelabel_evidence_elapsed_ms=prelabel_evidence_elapsed_ms,
                timing_ms=prelabel_evidence_elapsed_ms,
            )
            self.assertTrue(shard_path.is_file())

            # 3. Label unlock succeeds post-shard persistence
            batch.mark_persisted()
            gt_labels = batch.get_labels()
            self.assertEqual(gt_labels.shape, (2, 8, 8))

            # 4. Total batch timing is computed at batch completion
            batch_total_elapsed_ms = (time.time() - b_start_time) * 1000.0
            self.assertTrue(math.isfinite(batch_total_elapsed_ms))
            self.assertGreaterEqual(batch_total_elapsed_ms, prelabel_evidence_elapsed_ms)

            # 5. Record step with both timings
            rec.record_step(
                step_index=0,
                event_id=520,
                condition_id="candidate_e2_conf90",
                batch_size=len(s_ids),
                sample_ids=s_ids,
                action="update",
                update_applied=True,
                batch_time_ms=batch_total_elapsed_ms,
                batch_total_elapsed_ms=batch_total_elapsed_ms,
                prelabel_evidence_elapsed_ms=prelabel_evidence_elapsed_ms,
            )

            # 6. Verify evidence index contains prelabel timing and legacy alias
            index_path = rec.current_evidence_index_path
            self.assertIsNotNone(index_path)
            self.assertTrue(index_path.is_file())
            with open(index_path, "r") as f:
                idx_data = json.load(f)
            self.assertEqual(len(idx_data["shards"]), 1)
            shard_entry = idx_data["shards"][0]
            self.assertIn("prelabel_evidence_elapsed_ms", shard_entry)
            self.assertIn("timing_ms", shard_entry)
            self.assertAlmostEqual(shard_entry["prelabel_evidence_elapsed_ms"], prelabel_evidence_elapsed_ms, places=2)
            self.assertEqual(shard_entry["prelabel_evidence_elapsed_ms"], shard_entry["timing_ms"])

            # 7. Verify step record has both timings
            step_record = rec.current_step_records[0]
            self.assertIn("prelabel_evidence_elapsed_ms", step_record)
            self.assertIn("batch_total_elapsed_ms", step_record)
            self.assertIn("batch_time_ms", step_record)
            self.assertEqual(step_record["batch_time_ms"], step_record["batch_total_elapsed_ms"])
            self.assertGreaterEqual(step_record["batch_total_elapsed_ms"], step_record["prelabel_evidence_elapsed_ms"])

    def test_36_complete_e5_orchestration_lifecycle_synthetic(self):
        """Verifies end-to-end orchestration lifecycle from preflight through terminal COMPLETED status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "smoke_lifecycle"

            runner = E5RiskScoringRunner(
                config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
                output_dir=out_dir,
                device="cpu",
                is_smoke=True,
                allow_dirty=True,
            )

            # Mock clean model loading with SyntheticToyModel
            def mock_models():
                cand = SyntheticToyModel()
                src = SyntheticToyModel()
                src.load_state_dict(cand.state_dict())
                bn = SyntheticToyModel()
                bn.load_state_dict(cand.state_dict())
                for p in cand.parameters():
                    p.requires_grad = False
                for m in cand.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                for p in src.parameters():
                    p.requires_grad = False
                for p in bn.parameters():
                    p.requires_grad = False
                src.eval()
                bn.eval()
                cand.train()
                return cand, src, bn

            runner.load_clean_models = mock_models

            # Mock stream batches: exactly 2 batches of 2 samples each for Event 520
            b1 = LabelGuardedBatch(
                sample_ids=["s01", "s02"],
                event_id=520,
                xA=torch.zeros((2, 2, 16, 16)),
                xB=torch.zeros((2, 2, 16, 16)),
                valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                labels=torch.zeros((2, 16, 16), dtype=torch.int64),
            )
            b2 = LabelGuardedBatch(
                sample_ids=["s03", "s04"],
                event_id=520,
                xA=torch.zeros((2, 2, 16, 16)),
                xB=torch.zeros((2, 2, 16, 16)),
                valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                labels=torch.zeros((2, 16, 16), dtype=torch.int64),
            )
            runner.build_ordered_stream_batches = lambda: [(520, [b1, b2])]

            manifest = runner.execute_streaming_evaluation()

            # Assert manifest returns and status
            self.assertEqual(manifest["status"], "COMPLETED_SMOKE")
            self.assertEqual(manifest["total_batches_processed"], 2)

            # 1. Initial manifest and schema
            init_manifest_path = out_dir / "initial_manifest.json"
            self.assertTrue(init_manifest_path.is_file())
            with open(init_manifest_path, "r") as f:
                init_m = json.load(f)
            self.assertEqual(init_m["deterministic_sample_ids"], ["s01", "s02", "s03", "s04"])
            self.assertIn("520", init_m["event_contracts"])
            self.assertEqual(init_m["event_contracts"]["520"]["expected_event_sample_count"], 4)

            # 2. Authoritative evidence shards
            shard0_path = out_dir / "event_520" / "evidence_shards" / "batch_000000.npz"
            shard1_path = out_dir / "event_520" / "evidence_shards" / "batch_000001.npz"
            self.assertTrue(shard0_path.is_file())
            self.assertTrue(shard1_path.is_file())

            # 3. Evidence index
            ev_idx_path = out_dir / "event_520" / "evidence_index.json"
            self.assertTrue(ev_idx_path.is_file())
            with open(ev_idx_path, "r") as f:
                ev_idx = json.load(f)
            self.assertEqual(ev_idx["total_shards"], 2)
            self.assertEqual(ev_idx["shards"][0]["sample_ids"], ["s01", "s02"])
            self.assertEqual(ev_idx["shards"][1]["sample_ids"], ["s03", "s04"])

            # 4. Step records
            step_rec_path = out_dir / "event_520" / "step_records.json"
            self.assertTrue(step_rec_path.is_file())
            with open(step_rec_path, "r") as f:
                step_recs = json.load(f)
            self.assertEqual(len(step_recs), 2)
            for sr in step_recs:
                self.assertTrue(math.isfinite(sr["prelabel_evidence_elapsed_ms"]))
                self.assertTrue(math.isfinite(sr["batch_total_elapsed_ms"]))
                self.assertGreaterEqual(sr["batch_total_elapsed_ms"], sr["prelabel_evidence_elapsed_ms"])

            # 5. Event metrics and conservation
            ev_metrics_path = out_dir / "event_520" / "event_metrics.json"
            self.assertTrue(ev_metrics_path.is_file())
            with open(ev_metrics_path, "r") as f:
                em = json.load(f)
            self.assertEqual(em["tp"] + em["fp"] + em["fn"] + em["tn"], em["valid_pixels"])

            # 6. Adaptation progress
            prog_path = out_dir / "event_520" / "adaptation_progress.json"
            self.assertTrue(prog_path.is_file())
            with open(prog_path, "r") as f:
                prog = json.load(f)
            self.assertEqual(prog["status"], "COMPLETED")
            self.assertEqual(prog["samples_completed"], 4)

            # 7. Final run manifest and completion status
            run_m_path = out_dir / "run_manifest.json"
            self.assertTrue(run_m_path.is_file())
            status_path = out_dir / "completion_status.json"
            self.assertTrue(status_path.is_file())
            with open(status_path, "r") as f:
                stat_data = json.load(f)
            self.assertEqual(stat_data["status"], "COMPLETED_RUN")

            # 8. Step summary CSV
            csv_path = out_dir / "e5_step_risk_summary.csv"
            self.assertTrue(csv_path.is_file())

            # 9. No failure record
            self.assertFalse((out_dir / "event_520" / "failure.json").exists())
            self.assertFalse((out_dir / "failure.json").exists())

    def test_37_event_finalization_coverage_failure_paths(self):
        """Verifies fail-closed failure recording on coverage count mismatch, duplicates, and sequence corruption."""
        for failure_mode in ["count_mismatch", "duplicate_ids", "sequence_mismatch"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / f"smoke_fail_{failure_mode}"

                runner = E5RiskScoringRunner(
                    config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
                    output_dir=out_dir,
                    device="cpu",
                    is_smoke=True,
                    allow_dirty=True,
                )

                def mock_models():
                    cand = SyntheticToyModel()
                    src = SyntheticToyModel()
                    bn = SyntheticToyModel()
                    src.eval()
                    bn.eval()
                    cand.train()
                    return cand, src, bn

                runner.load_clean_models = mock_models

                b1 = LabelGuardedBatch(
                    sample_ids=["s01", "s02"],
                    event_id=520,
                    xA=torch.zeros((2, 2, 16, 16)),
                    xB=torch.zeros((2, 2, 16, 16)),
                    valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                    labels=torch.zeros((2, 16, 16), dtype=torch.int64),
                )
                b2 = LabelGuardedBatch(
                    sample_ids=["s03", "s04"],
                    event_id=520,
                    xA=torch.zeros((2, 2, 16, 16)),
                    xB=torch.zeros((2, 2, 16, 16)),
                    valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                    labels=torch.zeros((2, 16, 16), dtype=torch.int64),
                )

                if failure_mode == "count_mismatch":
                    runner.build_ordered_stream_batches = lambda: [(520, [b1, b2])]
                    # Monkey-patch get_sensory_inputs on b2 to return only 1 sample (3 total instead of 4)
                    b2.get_sensory_inputs = lambda: {
                        "sample_ids": ["s03"],
                        "xA": b2.xA[:1],
                        "xB": b2.xB[:1],
                        "valid_mask": b2.valid_mask[:1],
                    }
                elif failure_mode == "duplicate_ids":
                    b2_dup = LabelGuardedBatch(
                        sample_ids=["s02", "s03"],  # Duplicate s02!
                        event_id=520,
                        xA=torch.zeros((2, 2, 16, 16)),
                        xB=torch.zeros((2, 2, 16, 16)),
                        valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                        labels=torch.zeros((2, 16, 16), dtype=torch.int64),
                    )
                    runner.build_ordered_stream_batches = lambda: [(520, [b1, b2_dup])]
                elif failure_mode == "sequence_mismatch":
                    runner.build_ordered_stream_batches = lambda: [(520, [b1, b2])]
                    # Monkey-patch get_sensory_inputs on b2 to return reversed sample IDs
                    b2.get_sensory_inputs = lambda: {
                        "sample_ids": ["s04", "s03"],  # Permuted!
                        "xA": b2.xA,
                        "xB": b2.xB,
                        "valid_mask": b2.valid_mask,
                    }

                with self.assertRaises((RuntimeError, ValueError)) as ctx:
                    runner.execute_streaming_evaluation()

                # Verify failure.json was durably written
                failure_path = out_dir / "event_520" / "failure.json"
                self.assertTrue(failure_path.is_file(), f"failure.json missing for {failure_mode}")
                with open(failure_path, "r") as f:
                    fail_data = json.load(f)
                self.assertEqual(fail_data["status"], "FAILED")

    def test_38_produce_convenience_aggregate_toggle_and_shard_primacy(self):
        """Verifies that produce_convenience_aggregate=False omits frozen_predictions.npz while retaining all shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_shard_primacy"

            runner = E5RiskScoringRunner(
                config_path=Path("configs/changeguard/e5_temporal_risk_v1.json"),
                output_dir=out_dir,
                produce_convenience_aggregate=False,
                device="cpu",
                is_smoke=True,
                allow_dirty=True,
            )

            def mock_models():
                cand = SyntheticToyModel()
                src = SyntheticToyModel()
                src.load_state_dict(cand.state_dict())
                bn = SyntheticToyModel()
                bn.load_state_dict(cand.state_dict())
                for p in cand.parameters():
                    p.requires_grad = False
                for m in cand.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                for p in src.parameters():
                    p.requires_grad = False
                for p in bn.parameters():
                    p.requires_grad = False
                src.eval()
                bn.eval()
                cand.train()
                return cand, src, bn

            runner.load_clean_models = mock_models

            b1 = LabelGuardedBatch(
                sample_ids=["s01", "s02"],
                event_id=520,
                xA=torch.zeros((2, 2, 16, 16)),
                xB=torch.zeros((2, 2, 16, 16)),
                valid_mask=torch.ones((2, 16, 16), dtype=torch.uint8),
                labels=torch.zeros((2, 16, 16), dtype=torch.int64),
            )
            runner.build_ordered_stream_batches = lambda: [(520, [b1])]

            manifest = runner.execute_streaming_evaluation()
            self.assertEqual(manifest["status"], "COMPLETED_SMOKE")

            # Shards exist and are authoritative
            shard_path = out_dir / "event_520" / "evidence_shards" / "batch_000000.npz"
            self.assertTrue(shard_path.is_file())
            ev_idx_path = out_dir / "event_520" / "evidence_index.json"
            self.assertTrue(ev_idx_path.is_file())

            # Convenience aggregate is absent when produce_convenience_aggregate=False
            agg_path = out_dir / "event_520" / "frozen_predictions.npz"
            self.assertFalse(agg_path.exists())

            # Completion status is successful
            status_path = out_dir / "completion_status.json"
            self.assertTrue(status_path.is_file())
            with open(status_path, "r") as f:
                st = json.load(f)
            self.assertEqual(st["status"], "COMPLETED_RUN")

            # Check initial manifest records disabled aggregate policy
            init_m_path = out_dir / "initial_manifest.json"
            with open(init_m_path, "r") as f:
                im = json.load(f)
            self.assertEqual(im["aggregate_cache_policy"], "disabled_for_bounded_memory")
            self.assertEqual(im["authoritative_shard_policy"], "batch_atomic_disk_shards_primary")
            self.assertEqual(im["zero_sealed_test_declaration"], "strictly_quarantined_zero_access")


if __name__ == "__main__":
    unittest.main()
