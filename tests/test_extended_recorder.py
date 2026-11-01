"""Unit tests for ChangeGuard Extended Experiment Recorder (Plan V3 E1.1).

Verifies the declared cache contract under CPU-only synthetic execution:
1. Disk round trips, dtype/shape and phase separation (PRE vs POST predictions, float16 probabilities).
2. Key collisions and mismatched count guards.
3. Known synthetic gradient and BN displacement calculations.
4. Explicit missing/failed-value serialization (None -> null, never fabricated zero).
5. Full-precision score summaries alongside float16 spatial probability maps.
6. Failure and progress serialization under simulated interruption.
7. Artifact overwrite protection for completed runs.
8. Confusion-cell sum conservation assertion.
"""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.changeguard.recorder import (
    ExtendedExperimentRecorder,
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
    verify_quantization_drift,
)


class TestExtendedExperimentRecorder(unittest.TestCase):
    """Focused unit tests for ExtendedExperimentRecorder."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_cg_recorder_"))
        self.run_id = "test_run_e1_1"
        self.checkpoint_id = "test_ckpt_all27_s0"
        self.config_hash = "abc123hash"
        self.git_commit = "24ffcc9"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_01_provenance_and_artifact_protection(self):
        """Verifies initial provenance write and fail-fast overwrite protection."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
            allow_overwrite=False,
        )
        prov_file = self.test_dir / "initial_provenance.json"
        self.assertTrue(prov_file.exists())
        with open(prov_file, "r") as f:
            prov_data = json.load(f)
        self.assertEqual(prov_data["run_id"], self.run_id)
        self.assertEqual(prov_data["checkpoint_id"], self.checkpoint_id)
        self.assertEqual(prov_data["config_hash"], self.config_hash)
        self.assertEqual(prov_data["git_commit"], self.git_commit)

        # Mark run as complete
        recorder.complete_run()
        self.assertTrue((self.test_dir / "completion_status.json").exists())

        # Attempting to re-initialize with allow_overwrite=False must raise FileExistsError
        with self.assertRaises(FileExistsError):
            ExtendedExperimentRecorder(
                output_dir=self.test_dir,
                run_id="another_run",
                checkpoint_id=self.checkpoint_id,
                config_hash=self.config_hash,
                git_commit=self.git_commit,
                allow_overwrite=False,
            )

    def test_02_disk_roundtrip_predictions_and_phase_separation(self):
        """Verifies PRE and POST predictions and float16 probability round trips."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        event_dir = recorder.start_event(520)
        self.assertTrue(event_dir.exists())

        sample_ids = ["samp_001", "samp_002"]
        pre_preds = np.array([
            np.zeros((224, 224), dtype=np.uint8),
            np.ones((224, 224), dtype=np.uint8) * 2,
        ])
        post_preds = np.array([
            np.ones((224, 224), dtype=np.uint8),
            np.ones((224, 224), dtype=np.uint8) * 2,
        ])
        prob_c2 = np.array([
            np.full((224, 224), 0.125, dtype=np.float32),
            np.full((224, 224), 0.875, dtype=np.float32),
        ])

        recorder.record_predictions(
            sample_ids=sample_ids,
            pre_predictions=pre_preds,
            post_predictions=post_preds,
            prob_c2=prob_c2,
            prob_phase="PRE_UPDATE",
            prob_branch="candidate",
        )

        metrics = {
            "tp": 100,
            "fp": 20,
            "fn": 30,
            "tn": 850,
            "valid_pixels": 1000,
            "precision": 100 / 120,
            "recall": 100 / 130,
            "f1": 200 / 250,
            "iou": 100 / 150,
        }
        recorder.complete_event(520, metrics)

        npz_file = event_dir / "frozen_predictions.npz"
        self.assertTrue(npz_file.exists())
        data = np.load(npz_file)

        # PRE_UPDATE primary predictions
        self.assertIn("samp_001", data)
        self.assertEqual(data["samp_001"].dtype, np.uint8)
        self.assertEqual(data["samp_001"].shape, (224, 224))
        self.assertEqual(data["samp_001"][0, 0], 0)

        self.assertIn("samp_002", data)
        self.assertEqual(data["samp_002"][0, 0], 2)

        # POST_UPDATE diagnostic predictions
        self.assertIn("samp_001_post_pred", data)
        self.assertEqual(data["samp_001_post_pred"].dtype, np.uint8)
        self.assertEqual(data["samp_001_post_pred"][0, 0], 1)

        # Float16 Class-2 probabilities with explicit branch/phase key
        prob_key_1 = "samp_001_PRE_UPDATE_candidate_prob_c2"
        self.assertIn(prob_key_1, data)
        self.assertEqual(data[prob_key_1].dtype, np.float16)
        self.assertAlmostEqual(float(data[prob_key_1][0, 0]), 0.125, places=3)

    def test_03_ambiguous_probability_phase_rejection(self):
        """Verifies that missing or ambiguous probability phases are rejected."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        recorder.start_event(279)
        sample_ids = ["s1"]
        pre_preds = np.zeros((1, 224, 224), dtype=np.uint8)
        prob_c2 = np.full((1, 224, 224), 0.5, dtype=np.float32)

        # Missing prob_phase
        with self.assertRaises(ValueError):
            recorder.record_predictions(
                sample_ids=sample_ids,
                pre_predictions=pre_preds,
                prob_c2=prob_c2,
                prob_phase=None,
                prob_branch="candidate",
            )

        # Ambiguous / invalid prob_phase
        with self.assertRaises(ValueError):
            recorder.record_predictions(
                sample_ids=sample_ids,
                pre_predictions=pre_preds,
                prob_c2=prob_c2,
                prob_phase="AMBIGUOUS_PHASE",
                prob_branch="candidate",
            )

    def test_04_sample_key_collision_and_mismatched_counts(self):
        """Verifies collision detection and batch size validation."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        recorder.start_event(437)

        # Step 1: batch_size mismatch with len(sample_ids)
        with self.assertRaises(ValueError):
            recorder.record_step(
                step_index=1,
                event_id=437,
                condition_id="test_cond",
                batch_size=2,
                sample_ids=["s1"],  # length 1 != 2
                action="update",
                update_applied=True,
            )

        # Record step 1 normally
        recorder.record_step(
            step_index=1,
            event_id=437,
            condition_id="test_cond",
            batch_size=2,
            sample_ids=["s1", "s2"],
            action="update",
            update_applied=True,
        )

        # Step 2: duplicate sample ID 's1' -> collision error
        with self.assertRaises(ValueError):
            recorder.record_step(
                step_index=2,
                event_id=437,
                condition_id="test_cond",
                batch_size=2,
                sample_ids=["s1", "s3"],  # 's1' already seen!
                action="update",
                update_applied=True,
            )

    def test_05_explicit_missing_value_serialization(self):
        """Verifies uncomputed metrics serialize as null, never fabricated zero."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        event_dir = recorder.start_event(559)

        # Skipped step with None values for optimization telemetry
        recorder.record_step(
            step_index=1,
            event_id=559,
            condition_id="test_skip",
            batch_size=1,
            sample_ids=["s_skip_1"],
            action="skip",
            update_applied=False,
            total_loss=None,
            gradient_norm_l2=None,
            bn_weight_displacement=None,
            null_score_s_null=None,
        )

        step_file = event_dir / "step_records.json"
        self.assertTrue(step_file.exists())
        with open(step_file, "r") as f:
            records = json.load(f)

        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertIsNone(r["total_loss"])
        self.assertIsNone(r["gradient_norm_l2"])
        self.assertIsNone(r["bn_weight_displacement"])
        self.assertIsNone(r["null_score_s_null"])
        self.assertFalse(r["update_applied"])
        self.assertEqual(r["action"], "skip")

    def test_06_synthetic_gradient_and_bn_displacement_calculations(self):
        """Verifies mathematical correctness of gradient L2 and BN displacement helpers."""
        # 1. Gradient L2 norm calculation
        p1 = nn.Parameter(torch.zeros(2, 2))
        p2 = nn.Parameter(torch.zeros(3))
        p1.grad = torch.tensor([[3.0, 0.0], [0.0, 4.0]])  # norm = sqrt(9+16) = 5.0
        p2.grad = torch.tensor([0.0, 12.0, 0.0])          # norm = 12.0
        # Total norm = sqrt(5^2 + 12^2) = sqrt(169) = 13.0
        norm = compute_gradient_l2_norm([p1, p2])
        self.assertIsNotNone(norm)
        self.assertAlmostEqual(norm, 13.0, places=5)

        # No gradients -> returns None
        p_no_grad = nn.Parameter(torch.zeros(2))
        self.assertIsNone(compute_gradient_l2_norm([p_no_grad]))

        # 2. BatchNorm displacement calculation
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(2)

        model = DummyModel()
        source_state = {
            "bn.weight": torch.tensor([1.0, 1.0]),
            "bn.bias": torch.tensor([0.0, 0.0]),
        }
        # Adapt model: shift weights by +3, +4 (displacement = 5.0), bias by +6, +8 (displacement = 10.0)
        with torch.no_grad():
            model.bn.weight.copy_(torch.tensor([4.0, 5.0]))
            model.bn.bias.copy_(torch.tensor([6.0, 8.0]))

        w_disp, b_disp = compute_bn_affine_displacement(model, source_state)
        self.assertIsNotNone(w_disp)
        self.assertIsNotNone(b_disp)
        self.assertAlmostEqual(w_disp, 5.0, places=5)
        self.assertAlmostEqual(b_disp, 10.0, places=5)

    def test_07_quantization_drift_and_full_precision_retention(self):
        """Verifies full-precision summary retention and float16 quantization bounds."""
        # Float32 probabilities
        orig_probs = np.array([0.123456789, 0.987654321], dtype=np.float32)
        quantized = orig_probs.astype(np.float16)
        drift = verify_quantization_drift(orig_probs, quantized)
        # In IEEE float16, relative precision is ~10^-3, max error on [0, 1] is < 0.001
        self.assertLess(drift["max_abs_diff"], 1e-3)

        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        event_dir = recorder.start_event(1111003)

        # Full-precision summary stored in step records
        exact_mean_p_c2 = float(np.mean(orig_probs))
        recorder.record_step(
            step_index=1,
            event_id=1111003,
            condition_id="test_fp_retention",
            batch_size=2,
            sample_ids=["s_fp1", "s_fp2"],
            action="update",
            update_applied=True,
            mean_prob_c2=exact_mean_p_c2,
        )

        with open(event_dir / "step_records.json", "r") as f:
            rec = json.load(f)[0]
        self.assertAlmostEqual(rec["mean_prob_c2"], exact_mean_p_c2, places=8)

    def test_08_failure_and_interrupted_progress_handling(self):
        """Verifies durable failure serialization on error."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        event_dir = recorder.start_event(1111008)

        # Record step 1
        recorder.record_step(
            step_index=1,
            event_id=1111008,
            condition_id="test_fail",
            batch_size=1,
            sample_ids=["s_pass"],
            action="update",
            update_applied=True,
        )

        # Injected failure
        fail_path = recorder.record_failure(
            error_message="Simulated CUDA Out Of Memory or Gradient NaN",
            phase="backward",
            details={"step_index": 2},
        )
        self.assertTrue(fail_path.exists())
        with open(fail_path, "r") as f:
            fail_data = json.load(f)

        self.assertEqual(fail_data["status"], "FAILED")
        self.assertEqual(fail_data["phase"], "backward")
        self.assertEqual(fail_data["event_id"], 1111008)
        self.assertIn("Simulated CUDA Out Of Memory", fail_data["error_message"])

        # Progress file confirms steps completed before failure
        with open(event_dir / "adaptation_progress.json", "r") as f:
            prog = json.load(f)
        self.assertEqual(prog["steps_completed"], 1)
        self.assertEqual(prog["samples_completed"], 1)

    def test_09_confusion_conservation_invariant(self):
        """Verifies complete_event enforces tp + fp + fn + tn == valid_pixels."""
        recorder = ExtendedExperimentRecorder(
            output_dir=self.test_dir,
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
        )
        recorder.start_event(279)

        # Invalid metrics where sum != valid_pixels
        corrupt_metrics = {
            "tp": 100,
            "fp": 50,
            "fn": 50,
            "tn": 700,  # sum = 900
            "valid_pixels": 1000,  # mismatch!
        }
        with self.assertRaises(ValueError):
            recorder.complete_event(279, corrupt_metrics)

    def test_10_per_class_pixel_counting(self):
        """Verifies compute_per_class_pixel_counts under selection and valid masks."""
        preds = np.array([
            [0, 1, 2],
            [0, 1, 2],
        ])
        valid_mask = np.array([
            [True, True, True],
            [True, True, False],  # last pixel invalid
        ])
        selection_mask = np.array([
            [True, False, True],
            [True, True, True],
        ])
        counts = compute_per_class_pixel_counts(
            preds,
            selection_mask=selection_mask,
            valid_mask=valid_mask,
        )
        # Pixel evaluations:
        # (0,0): pred=0, valid=T, sel=T -> C0
        # (0,1): pred=1, valid=T, sel=F -> skipped
        # (0,2): pred=2, valid=T, sel=T -> C2
        # (1,0): pred=0, valid=T, sel=T -> C0
        # (1,1): pred=1, valid=T, sel=T -> C1
        # (1,2): pred=2, valid=F, sel=T -> skipped
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 1)


if __name__ == "__main__":
    unittest.main()

