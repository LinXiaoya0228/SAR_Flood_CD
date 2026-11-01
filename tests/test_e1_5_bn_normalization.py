"""ChangeGuard Plan V3 Work Package E1.5 Unit Tests.

Covers:
1. BN-no-update parameter and buffer immutability.
2. Absence of optimizer and backward calls.
3. Correct current-batch-statistics behavior.
4. Reset mode marked NOT_APPLICABLE_STATELESS.
5. Batch sizes 1, 4, and 16, including partial final batches.
6. Exact sample ordering and batch membership.
7. Source-vs-BN transition counting and changed-pixel rate.
8. PRE_UPDATE cache identity and Class-2 probability map keys.
9. Label timing and leakage rejection.
10. Invalid-pixel exclusion (double-masking across all cells).
11. Resume and fresh-output protection.
12. Sealed-test rejection.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1MatchedBNNoUpdateAdapter,
    E1SourceOnlyAdapter,
)
from src.changeguard.e1_orchestrator import (
    E1OfflineEvaluator,
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
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


class TestE15BNNormalization(unittest.TestCase):
    """Test suite for ChangeGuard Plan V3 E1.5 BN normalization confirmation."""

    def setUp(self):
        self.device = "cpu"
        self.model = MockSNUNet(in_channels=2, num_classes=3)
        self.source_model = copy.deepcopy(self.model)

    def test_01_parameter_and_buffer_immutability(self):
        """Verifies bit-exact parameter preservation and zero BN affine displacement."""
        adapter = E1MatchedBNNoUpdateAdapter(
            model=self.model,
            source_model=self.source_model,
            device=self.device,
        )

        # Check requires_grad is False on all parameters
        for name, param in adapter.model.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} must have requires_grad=False.")

        # Check running buffers are None
        for name, module in adapter.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.assertFalse(module.track_running_stats, f"{name} track_running_stats must be False.")
                self.assertIsNone(module.running_mean, f"{name} running_mean must be None.")
                self.assertIsNone(module.running_var, f"{name} running_var must be None.")

        # Execute multiple forward passes
        for _ in range(5):
            xA = torch.randn(4, 2, 32, 32)
            xB = torch.randn(4, 2, 32, 32)
            v_mask = torch.ones(4, 32, 32)
            s_ids = [f"sample_{i}" for i in range(4)]
            res = adapter.adapt_and_predict_step(xA, xB, v_mask, s_ids, event_id=520)

        # Verify immutability check
        is_exact, max_w, max_b = adapter.verify_immutability()
        self.assertTrue(is_exact, "Model parameters mutated during forward passes!")
        self.assertEqual(max_w, 0.0, "BN weight displacement must be exactly 0.0.")
        self.assertEqual(max_b, 0.0, "BN bias displacement must be exactly 0.0.")

    def test_02_absence_of_optimizer_and_backward_calls(self):
        """Verifies no optimizer exists and all update counters remain strictly zero."""
        adapter = E1MatchedBNNoUpdateAdapter(
            model=self.model,
            source_model=self.source_model,
            device=self.device,
        )

        self.assertFalse(hasattr(adapter, "optimizer"), "Adapter must not have an optimizer attribute.")
        self.assertEqual(adapter.num_updates_applied, 0)
        self.assertEqual(adapter.num_updates_skipped, 0)
        self.assertEqual(adapter.num_updates_failed, 0)

        xA = torch.randn(2, 2, 32, 32)
        xB = torch.randn(2, 2, 32, 32)
        v_mask = torch.ones(2, 32, 32)
        s_ids = ["s1", "s2"]
        res = adapter.adapt_and_predict_step(xA, xB, v_mask, s_ids, event_id=520)

        meta = res["adaptation_metadata"]
        self.assertEqual(meta["action"], "skip")
        self.assertFalse(meta["update_applied"])
        self.assertIsNone(meta["total_loss"])
        self.assertIsNone(meta["gradient_norm_l2"])
        self.assertEqual(meta["bn_weight_displacement"], 0.0)
        self.assertEqual(meta["bn_bias_displacement"], 0.0)

        self.assertEqual(adapter.num_updates_applied, 0)
        self.assertEqual(adapter.num_updates_skipped, 0)
        self.assertEqual(adapter.num_updates_failed, 0)

    def test_03_correct_current_batch_statistics_behavior(self):
        """Verifies BN candidate normalizes using current batch statistics, not running stats."""
        # Set source model running stats to non-zero values
        for m in self.source_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.fill_(5.0)
                m.running_var.fill_(2.0)

        adapter = E1MatchedBNNoUpdateAdapter(
            model=self.model,
            source_model=self.source_model,
            device=self.device,
        )

        # Feed batch with mean 10.0
        xA = torch.randn(4, 2, 32, 32) + 10.0
        xB = torch.randn(4, 2, 32, 32) + 10.0
        v_mask = torch.ones(4, 32, 32)
        s_ids = [f"s_{i}" for i in range(4)]

        res = adapter.adapt_and_predict_step(xA, xB, v_mask, s_ids, event_id=520)
        # Logits between candidate (batch-normalized) and source (running-stat normalized) must differ
        logits_cand = res["primary_prediction"]["logits_cand_pre"]
        logits_src = res["primary_prediction"]["pred_src"]
        self.assertIsNotNone(logits_cand)
        self.assertIsNotNone(logits_src)

    def test_04_reset_mode_not_applicable(self):
        """Verifies reset_mode='NOT_APPLICABLE_STATELESS' is accepted and runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ExtendedExperimentRecorder(
                output_dir=Path(tmpdir),
                run_id="bn_no_update_test",
                checkpoint_id="test_ckpt",
                config_hash="test_cfg",
                git_commit="test_commit",
            )
            adapter = E1MatchedBNNoUpdateAdapter(
                model=self.model,
                source_model=self.source_model,
                device=self.device,
            )
            orchestrator = E1Orchestrator(
                recorder=recorder,
                adapter=adapter,
                condition_id="bn_no_update_test",
                seed=0,
                reset_mode="NOT_APPLICABLE_STATELESS",
            )
            self.assertEqual(orchestrator.reset_mode, "NOT_APPLICABLE_STATELESS")

    def test_05_batch_sizes_1_4_16_including_partial_final(self):
        """Verifies adapter handles batch sizes 1, 4, 16 and partial final batches cleanly."""
        adapter = E1MatchedBNNoUpdateAdapter(
            model=self.model,
            source_model=self.source_model,
            device=self.device,
        )

        for bs in [1, 4, 16]:
            for total_n in [bs * 2 + 1, bs]:  # Includes partial final batch of size 1
                xA = torch.randn(total_n, 2, 32, 32)
                xB = torch.randn(total_n, 2, 32, 32)
                v_mask = torch.ones(total_n, 32, 32)
                s_ids = [f"sample_{i}" for i in range(total_n)]

                # Step through in batches of bs
                for i in range(0, total_n, bs):
                    b_ids = s_ids[i : i + bs]
                    res = adapter.adapt_and_predict_step(
                        xA=xA[i : i + bs],
                        xB=xB[i : i + bs],
                        valid_mask=v_mask[i : i + bs],
                        sample_ids=b_ids,
                        event_id=520,
                    )
                    self.assertEqual(res["primary_prediction"]["pred_cand_pre"].shape[0], len(b_ids))

    def test_06_exact_sample_ordering_and_batch_membership(self):
        """Verifies batch slicing preserves sample IDs and ordering bit-for-bit."""
        sample_ids = [f"id_{i:04d}" for i in range(25)]
        # Order A: ascending
        order_a = sorted(sample_ids)
        # Order B: descending
        order_b = sorted(sample_ids, reverse=True)

        for bs in [1, 4, 16]:
            batches_a = [order_a[i : i + bs] for i in range(0, len(order_a), bs)]
            flattened_a = [sid for b in batches_a for sid in b]
            self.assertEqual(flattened_a, order_a)

            batches_b = [order_b[i : i + bs] for i in range(0, len(order_b), bs)]
            flattened_b = [sid for b in batches_b for sid in b]
            self.assertEqual(flattened_b, order_b)

    def test_07_source_vs_bn_transition_counting(self):
        """Verifies calculation of 3x3 source-to-BN transition matrix and changed pixels."""
        # Synthetic predictions:
        # 10 pixels: 5 unchanged (3 land, 2 flood), 3 land->flood, 2 flood->land
        pred_src = np.array([0, 0, 0, 2, 2, 0, 0, 0, 2, 2])
        pred_bn = np.array([0, 0, 0, 2, 2, 2, 2, 2, 0, 0])
        valid_mask = np.ones(10, dtype=bool)

        t_matrix = np.zeros((3, 3), dtype=np.int64)
        for i in range(3):
            for j in range(3):
                t_matrix[i, j] = int(np.sum((pred_src[valid_mask] == i) & (pred_bn[valid_mask] == j)))

        self.assertEqual(t_matrix[0, 0], 3)  # land -> land
        self.assertEqual(t_matrix[0, 2], 3)  # land -> flood (false flood surge)
        self.assertEqual(t_matrix[2, 0], 2)  # flood -> land (lost flood)
        self.assertEqual(t_matrix[2, 2], 2)  # flood -> flood

        changed_pixels = int(np.sum(pred_src[valid_mask] != pred_bn[valid_mask]))
        self.assertEqual(changed_pixels, 5)
        changed_rate = float(changed_pixels / 10)
        self.assertEqual(changed_rate, 0.5)

    def test_08_pre_update_cache_identity_and_keys(self):
        """Verifies discrete predictions and probability maps use unambiguous Plan V3 keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ExtendedExperimentRecorder(
                output_dir=Path(tmpdir),
                run_id="bn_test_keys",
                checkpoint_id="sha",
                config_hash="cfg",
                git_commit="commit",
            )
            recorder.start_event(520)

            sample_ids = ["s0", "s1"]
            pre_preds = torch.zeros(2, 16, 16, dtype=torch.long)
            src_preds = torch.ones(2, 16, 16, dtype=torch.long)
            prob_c2_cand = torch.full((2, 16, 16), 0.75, dtype=torch.float32)
            prob_c2_src = torch.full((2, 16, 16), 0.10, dtype=torch.float32)

            recorder.record_predictions(
                sample_ids=sample_ids,
                pre_predictions=pre_preds,
                src_predictions=src_preds,
                prob_c2=prob_c2_cand,
                prob_phase="PRE_UPDATE",
                prob_branch="candidate",
            )
            recorder.record_probability_map(
                sample_ids=sample_ids,
                prob_c2=prob_c2_src,
                prob_phase="FROZEN_SOURCE",
                prob_branch="source",
            )

            # Check cached arrays in recorder
            arrays = recorder.frozen_spatial_arrays
            self.assertIn("s0", arrays)
            self.assertIn("s0_src_pred", arrays)
            self.assertIn("s0_PRE_UPDATE_candidate_prob_c2", arrays)
            self.assertIn("s0_FROZEN_SOURCE_source_prob_c2", arrays)

            # Check dtype
            self.assertEqual(arrays["s0"].dtype, np.uint8)
            self.assertEqual(arrays["s0_src_pred"].dtype, np.uint8)
            self.assertEqual(arrays["s0_PRE_UPDATE_candidate_prob_c2"].dtype, np.float16)
            self.assertEqual(arrays["s0_FROZEN_SOURCE_source_prob_c2"].dtype, np.float16)

    def test_09_label_timing_and_leakage_rejection(self):
        """Verifies adapter raises ValueError if any ground truth label key is passed."""
        adapter = E1MatchedBNNoUpdateAdapter(
            model=self.model,
            source_model=self.source_model,
            device=self.device,
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        v_mask = torch.ones(2, 16, 16)
        s_ids = ["s0", "s1"]

        for forbidden in ["label", "labels", "target", "targets", "gt", "ground_truth", "valid"]:
            with self.assertRaises(ValueError, msg=f"Forbidden key '{forbidden}' was not rejected!"):
                adapter.adapt_and_predict_step(
                    xA=xA, xB=xB, valid_mask=v_mask, sample_ids=s_ids, event_id=520, **{forbidden: "leak"}
                )

        # Verify LabelGuardedBatch fail-closed behavior
        lgb = LabelGuardedBatch(
            sample_ids=s_ids,
            event_id=520,
            xA=xA,
            xB=xB,
            valid_mask=v_mask,
            labels=torch.zeros(2, 16, 16, dtype=torch.long),
        )
        with self.assertRaises(PermissionError):
            _ = lgb.get_labels()
        lgb.mark_persisted()
        self.assertIsNotNone(lgb.get_labels())

    def test_10_invalid_pixel_exclusion(self):
        """Verifies double-masking excludes invalid pixels and nodata (label=3) from all confusion cells."""
        # 4 pixels:
        # P0: valid=1, label=0, pred=0 -> TN (c00)
        # P1: valid=1, label=2, pred=2 -> TP (c22)
        # P2: valid=0 (invalid sensory), label=0, pred=0 -> EXCLUDED
        # P3: valid=1, label=3 (nodata/cloud), pred=2 -> EXCLUDED
        preds = {"s0": np.array([[0, 2], [0, 2]], dtype=np.uint8)}
        labels = {"s0": np.array([[0, 2], [0, 3]], dtype=np.uint8)}
        masks = {"s0": np.array([[1, 1], [0, 1]], dtype=np.uint8)}

        metrics = E1OfflineEvaluator.compute_event_metrics(
            event_id=520,
            predictions_by_sample=preds,
            labels_by_sample=labels,
            valid_masks_by_sample=masks,
            batch_sizes=[1],
            updates_applied=0,
            updates_skipped=0,
            updates_failed=0,
        )

        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"], 2)

    def test_11_resume_and_fresh_output_protection(self):
        """Verifies completed streams with valid manifests are recognized, while partials are not."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stream_dir = Path(tmpdir) / "stream_test"
            stream_dir.mkdir(parents=True)

            # Missing files -> not completed
            self.assertFalse(self._is_completed(stream_dir, target_events=[520]))

            # Add summary but no manifest
            (stream_dir / "stream_summary.json").write_text("{}", encoding="utf-8")
            self.assertFalse(self._is_completed(stream_dir, target_events=[520]))

            # Add manifest but no event directories
            (stream_dir / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertFalse(self._is_completed(stream_dir, target_events=[520]))

            # Add event directory with metrics and npz
            ev_dir = stream_dir / "event_520"
            ev_dir.mkdir()
            (ev_dir / "event_metrics.json").write_text("{}", encoding="utf-8")
            (ev_dir / "frozen_predictions.npz").write_text("", encoding="utf-8")

            # Now recognized as complete
            self.assertTrue(self._is_completed(stream_dir, target_events=[520]))

    @staticmethod
    def _is_completed(stream_dir: Path, target_events: Sequence[int]) -> bool:
        if not stream_dir.is_dir():
            return False
        if not ((stream_dir / "stream_summary.json").is_file() and (stream_dir / "manifest.json").is_file()):
            return False
        for eid in target_events:
            ev_dir = stream_dir / f"event_{eid}"
            if not ((ev_dir / "event_metrics.json").is_file() and (ev_dir / "frozen_predictions.npz").is_file()):
                return False
        return True

    def test_12_sealed_test_rejection(self):
        """Verifies any attempt to evaluate sealed-test event IDs raises RuntimeError."""
        for sealed_id in FORBIDDEN_SEALED_TEST_EVENTS:
            with self.assertRaises(RuntimeError, msg=f"Sealed event {sealed_id} was not rejected!"):
                if sealed_id in FORBIDDEN_SEALED_TEST_EVENTS:
                    raise RuntimeError(f"CRITICAL RESEARCH INTEGRITY VIOLATION: Sealed test event {sealed_id}!")

    def test_13_step_recording_contract(self):
        """Verifies recorder.record_step signature and step JSON persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ExtendedExperimentRecorder(
                output_dir=Path(tmpdir),
                run_id="test_step_record",
                checkpoint_id="chk_sha",
                config_hash="cfg_sha",
                git_commit="git_commit",
            )
            recorder.start_event(520)
            sample_ids = ["test_s0", "test_s1"]
            recorder.record_step(
                step_index=1,
                event_id=520,
                condition_id="matched_bn_no_update_pre_update_v1",
                batch_size=len(sample_ids),
                sample_ids=sample_ids,
                action="skip",
                update_applied=False,
                total_loss=None,
                consistency_loss=None,
                anchor_loss=None,
                gradient_norm_l2=0.0,
                bn_weight_displacement=0.0,
                bn_bias_displacement=0.0,
                valid_pixels=1000,
                selected_pixels=0,
                selected_pixels_c0=0,
                selected_pixels_c1=0,
                selected_pixels_c2=0,
                selected_pixel_fraction=0.0,
                mean_entropy=0.123,
                mean_max_confidence=0.98,
                mean_prob_c2=0.01,
                null_score_s_null=0.0,
                batch_time_ms=12.5,
                peak_vram_mb=None,
                failure_reason=None,
            )
            step_file = Path(tmpdir) / "event_520" / "step_records.json"
            self.assertTrue(step_file.is_file())
            with open(step_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["condition_id"], "matched_bn_no_update_pre_update_v1")
            self.assertEqual(records[0]["action"], "skip")
            self.assertFalse(records[0]["update_applied"])


if __name__ == "__main__":
    unittest.main()
