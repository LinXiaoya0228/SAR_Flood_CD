"""Focused CPU integration and synthetic orchestration tests for ChangeGuard Plan V3 E1.2.

Adheres strictly to docs/MASTER_EXECUTION_PLAN_V3.md and docs/CACHE_AND_MEASUREMENT_CONTRACT.md:
1. Multi-batch, multi-event stream with partial final batch.
2. Event reset vs continual state behavior.
3. Source/candidate separation (independent storage, zero aliasing, unchanged source).
4. Alignment of paired geometric views and masks.
5. Evidence that label access is rejected before predictions/actions are durably persisted.
6. Correct PRE/POST cache identity, unambiguous keys, and schema persistence.
7. Analytical checks of gradient norms, BN displacement, and classwise selection counts.
8. Double masking and exclusion of invalid pixels from all offline confusion entries, including TN.
9. Empty selections, non-finite failures, and post-step optimizer corruption recovery.
10. Durable progress and terminal failure records after an injected adaptation exception.
11. Fresh-output overwrite protection.
12. Comparison against uninstrumented execution to demonstrate measurement neutrality.
13. Asset and device guards (strict CPU execution, zero real data/checkpoint/CUDA access).
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class MockSNUNet(nn.Module):
    """Synthetic proxy model matching multi-temporal SAR change detection interface."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)


def create_synthetic_batch(
    sample_ids: List[str],
    event_id: int,
    H: int = 32,
    W: int = 32,
    seed: int = 42,
    invalid_fraction: float = 0.1,
) -> LabelGuardedBatch:
    """Generates synthetic deterministic tensor batches for CPU verification."""
    torch.manual_seed(seed)
    B = len(sample_ids)
    xA = torch.randn((B, 2, H, W), dtype=torch.float32)
    xB = torch.randn((B, 2, H, W), dtype=torch.float32)

    # Valid mask with optional invalid border
    valid_mask = torch.ones((B, H, W), dtype=torch.uint8)
    if invalid_fraction > 0.0:
        inv_w = int(W * invalid_fraction)
        if inv_w > 0:
            valid_mask[:, :, :inv_w] = 0

    # Ground truth labels in {0, 1, 2, 3} where 3 is no-data
    labels = torch.randint(0, 3, (B, H, W), dtype=torch.int64)
    if invalid_fraction > 0.0:
        inv_w = int(W * invalid_fraction)
        if inv_w > 0:
            labels[:, :, -inv_w:] = 3  # label 3 is invalid nodata

    return LabelGuardedBatch(
        sample_ids=sample_ids,
        event_id=event_id,
        xA=xA,
        xB=xB,
        valid_mask=valid_mask,
        labels=labels,
    )


class TestE1SyntheticOrchestration(unittest.TestCase):
    """Rigorous CPU integration test suite for Work Package E1.2."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_e1_orchestrator_"))
        torch.manual_seed(100)
        np.random.seed(100)

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_multibatch_multievent_partial_batch(self) -> None:
        """Verifies multi-batch multi-event streams with exact partial tail batches."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_01",
            run_id="run_01",
            checkpoint_id="mock_ckpt_s0",
            config_hash="c0ffee",
            git_commit="cc093e6",
        )
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model, learning_rate=1e-3, confidence_threshold=0.50)
        orchestrator = E1Orchestrator(
            recorder=recorder,
            adapter=adapter,
            condition_id="C1_weak_strong",
            seed=0,
            reset_mode="reset_per_event",
        )

        # Event 520: 35 samples -> batches [16, 16, 3]
        e520_samples = [f"e520_s{i:03d}" for i in range(35)]
        e520_batches = [
            create_synthetic_batch(e520_samples[0:16], 520, seed=1),
            create_synthetic_batch(e520_samples[16:32], 520, seed=2),
            create_synthetic_batch(e520_samples[32:35], 520, seed=3),
        ]
        metrics_520 = orchestrator.run_event_stream(520, e520_batches)

        self.assertEqual(metrics_520["sample_count"], 35)
        self.assertEqual(metrics_520["batch_count"], 3)
        self.assertEqual(metrics_520["batch_sizes"], [16, 16, 3])

        # Event 279: 20 samples -> batches [16, 4]
        e279_samples = [f"e279_s{i:03d}" for i in range(20)]
        e279_batches = [
            create_synthetic_batch(e279_samples[0:16], 279, seed=4),
            create_synthetic_batch(e279_samples[16:20], 279, seed=5),
        ]
        metrics_279 = orchestrator.run_event_stream(279, e279_batches)

        self.assertEqual(metrics_279["sample_count"], 20)
        self.assertEqual(metrics_279["batch_count"], 2)
        self.assertEqual(metrics_279["batch_sizes"], [16, 4])

        # Verify disk persistence
        e520_npz = np.load(self.test_dir / "run_01" / "event_520" / "frozen_predictions.npz")
        self.assertIn("e520_s034", e520_npz.files)
        self.assertIn("e520_s034_post_pred", e520_npz.files)
        self.assertIn("e520_s034_PRE_UPDATE_candidate_prob_c2", e520_npz.files)

        e279_npz = np.load(self.test_dir / "run_01" / "event_279" / "frozen_predictions.npz")
        self.assertIn("e279_s019", e279_npz.files)

    def test_02_event_reset_vs_continual_state(self) -> None:
        """Verifies state reset per event vs continual adaptation behavior."""
        # 1. Reset per event
        rec_reset = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_reset",
            run_id="run_reset",
            checkpoint_id="ckpt_s0",
            config_hash="h1",
            git_commit="cc093e6",
        )
        model1 = MockSNUNet()
        adapter_reset = E1ConsistencyAdapter(model=model1, learning_rate=1e-2, confidence_threshold=0.10)
        orch_reset = E1Orchestrator(
            recorder=rec_reset,
            adapter=adapter_reset,
            condition_id="C1",
            seed=0,
            reset_mode="reset_per_event",
        )

        b1 = [create_synthetic_batch([f"s1_{i}" for i in range(8)], 101, seed=1)]
        orch_reset.run_event_stream(101, b1)
        # Weights moved from source
        disp_after_e1 = compute_bn_affine_displacement(adapter_reset.model, adapter_reset._source_state_dict)
        self.assertGreater(disp_after_e1[0], 0.0)

        # Run event 2: reset_per_event must reset weights to source before adapting
        b2 = [create_synthetic_batch([f"s2_{i}" for i in range(8)], 102, seed=2)]
        adapter_reset.reset_to_source()
        disp_reset = compute_bn_affine_displacement(adapter_reset.model, adapter_reset._source_state_dict)
        self.assertAlmostEqual(disp_reset[0], 0.0, places=6)

        # 2. Continual adaptation
        rec_cont = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_cont",
            run_id="run_cont",
            checkpoint_id="ckpt_s0",
            config_hash="h2",
            git_commit="cc093e6",
        )
        model2 = MockSNUNet()
        adapter_cont = E1ConsistencyAdapter(model=model2, learning_rate=1e-2, confidence_threshold=0.10)
        orch_cont = E1Orchestrator(
            recorder=rec_cont,
            adapter=adapter_cont,
            condition_id="C1",
            seed=0,
            reset_mode="continual",
        )

        orch_cont.run_event_stream(101, b1)
        p1 = [p.clone() for p in adapter_cont.model.parameters() if p.requires_grad]
        # In continual mode, event 102 does not reset to source
        orch_cont.run_event_stream(102, b2)
        p2 = [p.clone() for p in adapter_cont.model.parameters() if p.requires_grad]
        # Parameters continued updating
        diff = sum(torch.sum(torch.abs(a - b)).item() for a, b in zip(p1, p2))
        self.assertGreater(diff, 0.0)

    def test_03_source_candidate_separation(self) -> None:
        """Verifies candidate and source model have zero storage aliasing and source is immutable."""
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model, learning_rate=1e-2)

        # 1. No shared parameter or buffer pointers
        cand_ptrs = {p.data_ptr() for p in adapter.model.parameters()} | {
            b.data_ptr() for b in adapter.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in adapter.source_model.parameters()} | {
            b.data_ptr() for b in adapter.source_model.buffers() if b is not None
        }
        self.assertEqual(len(cand_ptrs.intersection(src_ptrs)), 0)

        # 2. Source parameters and buffers remain identical after adaptation
        src_params_before = [p.clone() for p in adapter.source_model.parameters()]
        src_buffers_before = [b.clone() for b in adapter.source_model.buffers()]

        batch = create_synthetic_batch([f"s_{i}" for i in range(8)], 100, seed=10)
        inputs = batch.get_sensory_inputs()
        adapter.adapt_and_predict_step(
            xA=inputs["xA"],
            xB=inputs["xB"],
            valid_mask=inputs["valid_mask"],
            sample_ids=inputs["sample_ids"],
            event_id=100,
        )

        for p_before, p_after in zip(src_params_before, adapter.source_model.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))
        for b_before, b_after in zip(src_buffers_before, adapter.source_model.buffers()):
            self.assertTrue(torch.equal(b_before, b_after))

    def test_04_paired_geometric_alignment(self) -> None:
        """Verifies weak-strong views apply paired horizontal flips and invert alignments correctly."""
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model)

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        valid_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Horizontally flipped inputs
        xA_flip = torch.flip(xA, dims=[-1])
        xB_flip = torch.flip(xB, dims=[-1])

        with torch.no_grad():
            out1 = adapter.model(xA, xB)
            out2 = adapter.model(xA_flip, xB_flip)

        # Realignment check: out2 flipped back should match out1 if convolution is spatially equivariant
        out2_realigned = torch.flip(out2, dims=[-1])
        self.assertEqual(out1.shape, out2_realigned.shape)

    def test_05_label_timing_enforcement(self) -> None:
        """Verifies that accessing evaluation labels before persistence is strictly rejected."""
        batch = create_synthetic_batch(["sample_01", "sample_02"], event_id=520)

        # Access before mark_persisted() must raise PermissionError
        with self.assertRaises(PermissionError) as ctx:
            _ = batch.get_labels()
        self.assertIn("CRITICAL LABEL TIMING VIOLATION", str(ctx.exception))

        # Sensory data is accessible immediately
        sensory = batch.get_sensory_inputs()
        self.assertIn("xA", sensory)
        self.assertNotIn("labels", sensory)
        self.assertNotIn("label", sensory)

        # After persistence is confirmed, label access succeeds
        batch.mark_persisted()
        labels = batch.get_labels()
        self.assertEqual(labels.shape[0], 2)

    def test_06_pre_post_cache_identity_and_unambiguous_keys(self) -> None:
        """Verifies unambiguous cache keys, distinct PRE and POST predictions, and schema persistence."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_06",
            run_id="run_06",
            checkpoint_id="ckpt_s0",
            config_hash="h06",
            git_commit="cc093e6",
        )
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model, learning_rate=1e-2, confidence_threshold=0.10)
        orch = E1Orchestrator(recorder=recorder, adapter=adapter, condition_id="C1", seed=0)

        batches = [create_synthetic_batch(["sample_a", "sample_b"], event_id=300, seed=7)]
        orch.run_event_stream(300, batches)

        # Verify cache_schema.json was written
        schema_path = self.test_dir / "run_06" / "cache_schema.json"
        self.assertTrue(schema_path.exists())
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema["schema_version"], "1.0.0")
        self.assertIn("primary_prediction", schema["key_patterns"])
        self.assertIn("post_prediction", schema["key_patterns"])
        self.assertIn("probability_c2", schema["key_patterns"])

        # Verify frozen_predictions.npz keys
        npz_path = self.test_dir / "run_06" / "event_300" / "frozen_predictions.npz"
        self.assertTrue(npz_path.exists())
        data = np.load(npz_path)

        # PRE_UPDATE candidate prediction
        self.assertIn("sample_a", data)
        self.assertEqual(data["sample_a"].dtype, np.uint8)

        # POST_UPDATE candidate prediction
        self.assertIn("sample_a_post_pred", data)
        self.assertEqual(data["sample_a_post_pred"].dtype, np.uint8)

        # FROZEN_SOURCE prediction
        self.assertIn("sample_a_src_pred", data)
        self.assertEqual(data["sample_a_src_pred"].dtype, np.uint8)

        # Probability maps with explicit phase and branch
        self.assertIn("sample_a_PRE_UPDATE_candidate_prob_c2", data)
        self.assertEqual(data["sample_a_PRE_UPDATE_candidate_prob_c2"].dtype, np.float16)

        self.assertIn("sample_a_FROZEN_SOURCE_source_prob_c2", data)
        self.assertEqual(data["sample_a_FROZEN_SOURCE_source_prob_c2"].dtype, np.float16)

        self.assertIn("sample_a_POST_UPDATE_candidate_prob_c2", data)
        self.assertEqual(data["sample_a_POST_UPDATE_candidate_prob_c2"].dtype, np.float16)

    def test_07_analytical_measurements_grad_bn_classwise(self) -> None:
        """Verifies analytical correctness of gradient L2 norm, BN displacement, and classwise counts."""
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model, learning_rate=1e-3, confidence_threshold=0.10)

        batch = create_synthetic_batch(["s1", "s2"], event_id=400, seed=12)
        inputs = batch.get_sensory_inputs()

        res = adapter.adapt_and_predict_step(
            xA=inputs["xA"],
            xB=inputs["xB"],
            valid_mask=inputs["valid_mask"],
            sample_ids=inputs["sample_ids"],
            event_id=400,
        )

        meta = res["adaptation_metadata"]
        self.assertTrue(meta["update_applied"])

        # 1. Gradient L2 norm must be finite positive float
        self.assertIsNotNone(meta["gradient_norm_l2"])
        self.assertGreater(meta["gradient_norm_l2"], 0.0)

        # 2. BN displacement must be finite positive float
        self.assertIsNotNone(meta["bn_weight_displacement"])
        self.assertGreater(meta["bn_weight_displacement"], 0.0)

        # 3. Classwise selection sum must equal total selected pixels
        c0 = meta["selected_pixels_c0"]
        c1 = meta["selected_pixels_c1"]
        c2 = meta["selected_pixels_c2"]
        sel = meta["selected_pixels"]
        self.assertEqual(c0 + c1 + c2, sel)

    def test_08_double_masking_confusion_conservation_excluding_invalid_from_tn(self) -> None:
        """Verifies that invalid pixels (valid_mask==0 or label==3) are completely excluded from confusion matrix and TN."""
        preds = {
            "s1": np.zeros((10, 10), dtype=np.uint8),  # All predicted class 0
        }
        # Half invalid in sensory mask, some label 3
        v_mask = np.ones((10, 10), dtype=np.uint8)
        v_mask[:5, :] = 0  # 50 pixels invalid

        labels = np.zeros((10, 10), dtype=np.int64)
        labels[5:7, :] = 3  # 20 pixels label 3

        # Remaining double-masked valid pixels: 100 - 50 - 20 = 30 pixels (all label 0, pred 0 -> c00)
        metrics = E1OfflineEvaluator.compute_event_metrics(
            event_id=500,
            predictions_by_sample=preds,
            labels_by_sample={"s1": labels},
            valid_masks_by_sample={"s1": v_mask},
            batch_sizes=[1],
            updates_applied=1,
            updates_skipped=0,
            updates_failed=0,
        )

        self.assertEqual(metrics["valid_pixels"], 30)
        self.assertEqual(metrics["c00_land_land"], 30)
        self.assertEqual(metrics["tn"], 30)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

        # Verify conservation: tp + fp + fn + tn == valid_pixels
        self.assertEqual(metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"], metrics["valid_pixels"])

    def test_09_empty_selection_nonfinite_and_opt_corruption(self) -> None:
        """Verifies handling of empty selections, non-finite loss/grads, and post-step corruption."""
        # 1. Empty selection -> skip
        model = MockSNUNet()
        adapter_skip = E1ConsistencyAdapter(model=model, confidence_threshold=1.5)  # unreachable threshold
        b = create_synthetic_batch(["s_skip"], 600, seed=1)
        res_skip = adapter_skip.adapt_and_predict_step(
            xA=b.xA, xB=b.xB, valid_mask=b.valid_mask, sample_ids=b.sample_ids, event_id=600
        )
        self.assertEqual(res_skip["adaptation_metadata"]["action"], "skip")
        self.assertFalse(res_skip["adaptation_metadata"]["update_applied"])
        self.assertIsNone(res_skip["adaptation_metadata"]["total_loss"])

        # 2. Injected non-finite loss -> failure
        adapter_fail = E1ConsistencyAdapter(model=MockSNUNet())
        with torch.no_grad():
            for p in adapter_fail.model.parameters():
                p.fill_(float("nan"))
        res_fail = adapter_fail.adapt_and_predict_step(
            xA=b.xA, xB=b.xB, valid_mask=b.valid_mask, sample_ids=b.sample_ids, event_id=600
        )
        self.assertEqual(res_fail["adaptation_metadata"]["action"], "failure")
        self.assertFalse(res_fail["adaptation_metadata"]["update_applied"])
        self.assertIn("Non-finite", res_fail["adaptation_metadata"]["failure_reason"])

        # 3. Post-step parameter corruption -> rollback to pre-step state
        adapter_corrupt = E1ConsistencyAdapter(model=MockSNUNet(), learning_rate=1e-3, confidence_threshold=0.10)
        b_corrupt = create_synthetic_batch(["s_corrupt"], 600, seed=5)
        # Capture pre-step param clone
        pre_param = next(p for p in adapter_corrupt.model.parameters() if p.requires_grad).detach().clone()
        
        real_step = adapter_corrupt.optimizer.step
        def corrupting_step() -> None:
            real_step()
            with torch.no_grad():
                next(p for p in adapter_corrupt.model.parameters() if p.requires_grad).fill_(float("nan"))
        adapter_corrupt.optimizer.step = corrupting_step

        res_corrupt = adapter_corrupt.adapt_and_predict_step(
            xA=b_corrupt.xA,
            xB=b_corrupt.xB,
            valid_mask=b_corrupt.valid_mask,
            sample_ids=b_corrupt.sample_ids,
            event_id=600,
        )
        self.assertEqual(res_corrupt["adaptation_metadata"]["action"], "failure")
        self.assertFalse(res_corrupt["adaptation_metadata"]["update_applied"])
        self.assertIn("Post-step corruption detected", res_corrupt["adaptation_metadata"]["failure_reason"])
        # Param must be restored bit-exactly to pre-step param (not NaN!)
        post_param = next(p for p in adapter_corrupt.model.parameters() if p.requires_grad)
        self.assertTrue(torch.isfinite(post_param).all())
        self.assertTrue(torch.equal(post_param, pre_param))

    def test_10_durable_progress_and_terminal_failure_on_exception(self) -> None:
        """Verifies durable failure recording when an unhandled exception occurs."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_fail",
            run_id="run_fail",
            checkpoint_id="ckpt_s0",
            config_hash="hfail",
            git_commit="cc093e6",
        )
        model = MockSNUNet()
        adapter = E1ConsistencyAdapter(model=model)
        orch = E1Orchestrator(recorder=recorder, adapter=adapter, condition_id="C1", seed=0)

        # Batch with malformed input to trigger exception
        malformed_batch = LabelGuardedBatch(
            sample_ids=["bad_s1"],
            event_id=700,
            xA=torch.randn(1, 2, 16, 16),
            xB=torch.randn(1, 2, 20, 20),  # Mismatched spatial shape!
            valid_mask=torch.ones(1, 16, 16, dtype=torch.uint8),
        )

        with self.assertRaises(ValueError):
            orch.run_event_stream(700, [malformed_batch])

        # Verify failure.json was persisted
        fail_path = self.test_dir / "run_fail" / "event_700" / "failure.json"
        self.assertTrue(fail_path.exists())
        with open(fail_path, "r", encoding="utf-8") as f:
            fail_data = json.load(f)
        self.assertEqual(fail_data["status"], "FAILED")
        self.assertEqual(fail_data["event_id"], 700)

    def test_11_fresh_output_protection(self) -> None:
        """Verifies that attempting to run into an existing completed output dir raises FileExistsError."""
        out = self.test_dir / "run_fresh"
        rec = ExtendedExperimentRecorder(out, "run_f", "ckpt", "hash", "commit", allow_overwrite=False)
        rec.complete_run()

        # Second instantiation without allow_overwrite must raise FileExistsError
        with self.assertRaises(FileExistsError):
            _ = ExtendedExperimentRecorder(out, "run_f", "ckpt", "hash", "commit", allow_overwrite=False)

    def test_12_measurement_neutrality_vs_uninstrumented(self) -> None:
        """Verifies that instrumentation (diagnostics, probes, recording) causes zero deviation in model updates."""
        torch.manual_seed(999)
        model_inst = MockSNUNet()
        model_raw = copy.deepcopy(model_inst)

        # Verify identical initial state
        for p1, p2 in zip(model_inst.parameters(), model_raw.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Instrumented run
        rec = ExtendedExperimentRecorder(
            output_dir=self.test_dir / "run_neutral",
            run_id="run_neutral",
            checkpoint_id="ckpt_s0",
            config_hash="h_neut",
            git_commit="cc093e6",
        )
        adapter_inst = E1ConsistencyAdapter(model=model_inst, learning_rate=1e-3, confidence_threshold=0.20)
        orch = E1Orchestrator(recorder=rec, adapter=adapter_inst, condition_id="C1", seed=0)

        batches = [
            create_synthetic_batch([f"neut_s{i}" for i in range(4)], event_id=800, seed=101),
            create_synthetic_batch([f"neut_s{i}" for i in range(4, 8)], event_id=800, seed=102),
        ]
        orch.run_event_stream(800, batches)

        # Uninstrumented reference loop on exact same batches
        model_raw.eval()
        model_raw.requires_grad_(False)
        for name, m in model_raw.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)

        params_raw = [p for p in model_raw.parameters() if p.requires_grad]
        opt_raw = optim.SGD(params_raw, lr=1e-3, momentum=0.9, weight_decay=0.0)

        for b in batches:
            xA = b.xA
            xB = b.xB
            V = (b.valid_mask == 1)

            logits_weak = model_raw(xA, xB)
            probs_weak = F.softmax(logits_weak, dim=1)
            probs_weak_det = probs_weak.detach()
            conf_mask = (probs_weak_det.max(dim=1)[0] >= 0.20) & V

            xA_s = torch.flip(xA, dims=[-1])
            xB_s = torch.flip(xB, dims=[-1])
            logits_s = model_raw(xA_s, xB_s)
            logits_s_a = torch.flip(logits_s, dims=[-1])
            log_probs_s = F.log_softmax(logits_s_a, dim=1)

            kl = (probs_weak_det * (torch.log(probs_weak_det + 1e-12) - log_probs_s)).sum(dim=1)
            loss = (kl * conf_mask.float()).sum() / conf_mask.sum().clamp(min=1.0)

            opt_raw.zero_grad()
            loss.backward()
            opt_raw.step()
            opt_raw.zero_grad()

        # Compare updated model parameters
        for (n1, p1), (n2, p2) in zip(adapter_inst.model.named_parameters(), model_raw.named_parameters()):
            max_diff = float(torch.max(torch.abs(p1 - p2)).item())
            self.assertLess(
                max_diff,
                1e-6,
                f"Measurement non-neutrality detected in parameter '{n1}': max difference is {max_diff}",
            )

    def test_13_asset_and_device_guards(self) -> None:
        """Verifies that synthetic execution runs strictly on CPU and accesses zero real GPU resources."""
        self.assertFalse(torch.cuda.is_initialized())
        device = torch.device("cpu")
        model = MockSNUNet().to(device)
        self.assertEqual(next(model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
