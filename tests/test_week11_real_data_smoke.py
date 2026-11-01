"""Focused regression test suite for Week 11 real-data smoke infrastructure.

Governing Review: docs/reviews/week11c_smoke_infrastructure_validation.md
Governing Draft Config: configs/changeguard/week11_smoke_event520_draft.json

Covers W11C-R01 through W11C-R07:
- R01: Dataset and manifest path metadata validation in preflight.
- R02: Durable pre-GT persistence of predictions, actions, progress, and failure recovery.
- R03: Offline true negatives restricted to valid mask; confusion cell sum equals valid pixels.
- R04: Stage, approval, boolean, and device authorization matrix.
- R05: Strict bounded smoke invariants: seed 0, Event 520, order_a, max_samples 50, batch 16, head 50, batches [16,16,16,2].
- R06: Pure synthetic mode asset-independence (zero real checkpoint/raster reads).
- R07: CUDA RNG state operations gated strictly on intended CUDA device context.
"""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter
from tools.run_week11_real_data_smoke import Week11RealDataSmokeRunner, MockSNUNet


class TestWeek11SmokeInfrastructure(unittest.TestCase):
    """Test suite resolving findings W11C-R01 through W11C-R07."""

    def setUp(self):
        self.draft_config_path = REPO_ROOT / "configs/changeguard/week11_smoke_event520_draft.json"
        with open(self.draft_config_path, "r", encoding="utf-8") as f:
            self.base_cfg = json.load(f)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="w11c_test_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _write_temp_config(self, cfg_dict: dict) -> Path:
        cfg_path = self.temp_dir / "temp_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        return cfg_path

    def _generate_synthetic_event520_records(self, count: int = 50) -> list:
        records = []
        for i in range(count):
            label = torch.zeros((224, 224), dtype=torch.int64)
            if i % 3 == 0:
                label[10:30, 10:30] = 2  # flood annotation
            records.append({
                "sample_id": f"s520_{i:04d}",
                "event_id": 520,
                "xA": torch.randn((2, 224, 224), dtype=torch.float32),
                "xB": torch.randn((2, 224, 224), dtype=torch.float32),
                "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
                "label": label,
            })
        return records

    # -------------------------------------------------------------------------
    # W11C-R01: Dataset & Manifest Path Metadata Validation
    # -------------------------------------------------------------------------
    def test_r01_dataset_path_metadata_validation(self):
        """Preflight in real mode must validate dataset/manifest path metadata before device/model work."""
        cfg_real = copy.deepcopy(self.base_cfg)
        cfg_real["status"] = "APPROVED_FOR_REAL_DATA_SMOKE"
        cfg_real["approval_status"] = "APPROVED_FOR_EXECUTION"
        cfg_real["execution_authorizations"]["real_data_execution_authorized"] = True
        cfg_real["execution_authorizations"]["gpu_execution_authorized"] = True

        # 1. Nonexistent dataset root
        cfg_real["dataset"]["dataset_root"] = str(self.temp_dir / "nonexistent_dataset_dir")
        cfg_path1 = self._write_temp_config(cfg_real)
        runner1 = Week11RealDataSmokeRunner(
            config_path=cfg_path1,
            output_dir=self.temp_dir / "out_r01_1",
            device="cuda:0",
            synthetic_mock=False,
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            runner1.validate_preflight()
        self.assertIn("Dataset root directory not found", str(ctx.exception))

        # 2. Nonexistent manifest file
        valid_dataset_dir = self.temp_dir / "dummy_dataset"
        valid_dataset_dir.mkdir(parents=True, exist_ok=True)
        cfg_real["dataset"]["dataset_root"] = str(valid_dataset_dir)
        cfg_real["dataset"]["manifest_path"] = str(self.temp_dir / "nonexistent_manifest.gz")
        cfg_path2 = self._write_temp_config(cfg_real)
        runner2 = Week11RealDataSmokeRunner(
            config_path=cfg_path2,
            output_dir=self.temp_dir / "out_r01_2",
            device="cuda:0",
            synthetic_mock=False,
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            runner2.validate_preflight()
        self.assertIn("Manifest file not found", str(ctx.exception))

    # -------------------------------------------------------------------------
    # W11C-R02: Durable Pre-GT Persistence & Failure Recovery
    # -------------------------------------------------------------------------
    def test_r02_durable_pre_gt_persistence_and_failure_recovery(self):
        """Predictions and diagnostics must be persisted before GT access; crashes preserve evidence."""
        out_dir = self.temp_dir / "out_r02"
        cfg_path = self._write_temp_config(self.base_cfg)
        mock_recs = self._generate_synthetic_event520_records(50)

        # 1. Inject a failure at ground truth access to simulate crash during offline scoring
        runner = Week11RealDataSmokeRunner(
            config_path=cfg_path,
            output_dir=out_dir,
            synthetic_mock=True,
        )

        from src.data.evaluation_dataset import KuroEvaluationDataset
        orig_getitem = KuroEvaluationDataset.__getitem__
        def selective_getitem(ds_self, idx):
            if getattr(ds_self, "load_labels", False):
                raise RuntimeError("Simulated ground-truth read failure!")
            return orig_getitem(ds_self, idx)

        with patch.object(KuroEvaluationDataset, "__getitem__", selective_getitem):
            with self.assertRaises(RuntimeError) as ctx:
                runner.run_smoke(mock_records=mock_recs)
            self.assertIn("Simulated ground-truth read failure", str(ctx.exception))

        # Output directory MUST NOT be empty! It must contain pre-GT predictions and failure record
        self.assertTrue((out_dir / "initial_manifest.json").is_file())
        self.assertTrue((out_dir / "frozen_predictions.npz").is_file())
        self.assertTrue((out_dir / "adaptation_step_records.json").is_file())
        self.assertTrue((out_dir / "adaptation_step_records.csv").is_file())
        self.assertTrue((out_dir / "adaptation_progress.json").is_file())
        self.assertTrue((out_dir / "execution_failure.json").is_file())

        # Verify predictions.npz contains all 50 keys
        loaded_npz = np.load(out_dir / "frozen_predictions.npz")
        self.assertEqual(len(loaded_npz.files), 50)

        # 2. On clean run, verify completion records
        out_clean = self.temp_dir / "out_r02_clean"
        runner_clean = Week11RealDataSmokeRunner(
            config_path=cfg_path,
            output_dir=out_clean,
            synthetic_mock=True,
        )
        runner_clean.run_smoke(mock_records=mock_recs)
        self.assertTrue((out_clean / "completion_status.json").is_file())
        self.assertTrue((out_clean / "smoke_manifest.json").is_file())
        with open(out_clean / "completion_status.json") as f:
            comp = json.load(f)
        self.assertEqual(comp["status"], "COMPLETED")

    def test_r02_adaptation_failure_preserves_durable_progress(self):
        """Adaptation crash after batch 1 must leave durable progress with batches_completed=1 and failure log."""
        out_dir = self.temp_dir / "out_r02_adapt_fail"
        cfg_path = self._write_temp_config(self.base_cfg)
        mock_recs = self._generate_synthetic_event520_records(50)

        runner = Week11RealDataSmokeRunner(
            config_path=cfg_path,
            output_dir=out_dir,
            synthetic_mock=True,
        )

        class FailingAdapter(EvidenceConsistencyAdapter):
            call_count = 0

            def adapt_and_predict(self, *a, **kw):
                FailingAdapter.call_count += 1
                if FailingAdapter.call_count > 1:
                    raise RuntimeError("Simulated crash during 2nd adaptation batch")
                return super().adapt_and_predict(*a, **kw)

        with patch("tools.run_week11_real_data_smoke.EvidenceConsistencyAdapter", FailingAdapter):
            with self.assertRaises(RuntimeError) as ctx:
                runner.run_smoke(mock_records=mock_recs)
            self.assertIn("Simulated crash during 2nd adaptation batch", str(ctx.exception))

        # Check durable progress persisted after batch 1
        prog_path = out_dir / "adaptation_progress.json"
        self.assertTrue(prog_path.is_file())
        with open(prog_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        self.assertEqual(prog["batches_completed"], 1)
        self.assertEqual(prog["samples_completed"], 16)
        self.assertEqual(len(prog["completed_sample_ids"]), 16)

        # Check execution_failure.json records ADAPTATION phase and last_durable_progress
        fail_path = out_dir / "execution_failure.json"
        self.assertTrue(fail_path.is_file())
        with open(fail_path, "r", encoding="utf-8") as f:
            fail_data = json.load(f)
        self.assertEqual(fail_data["phase"], "ADAPTATION")
        self.assertIn("Simulated crash during 2nd adaptation batch", fail_data["traceback"])
        self.assertIsNotNone(fail_data["last_durable_progress"])
        self.assertEqual(fail_data["last_durable_progress"]["batches_completed"], 1)

        # Completion status must NOT exist
        self.assertFalse((out_dir / "completion_status.json").exists())

    def test_r02_persistence_failure_records_phase(self):
        """Persistence failure must record phase PERSISTENCE in execution_failure.json."""
        out_dir = self.temp_dir / "out_r02_persist_fail"
        cfg_path = self._write_temp_config(self.base_cfg)
        mock_recs = self._generate_synthetic_event520_records(50)

        runner = Week11RealDataSmokeRunner(
            config_path=cfg_path,
            output_dir=out_dir,
            synthetic_mock=True,
        )

        with patch("numpy.savez_compressed", side_effect=IOError("Simulated disk write error")):
            with self.assertRaises(IOError) as ctx:
                runner.run_smoke(mock_records=mock_recs)
            self.assertIn("Simulated disk write error", str(ctx.exception))

        fail_path = out_dir / "execution_failure.json"
        self.assertTrue(fail_path.is_file())
        with open(fail_path, "r", encoding="utf-8") as f:
            fail_data = json.load(f)
        self.assertEqual(fail_data["phase"], "PERSISTENCE")
        self.assertIn("Simulated disk write error", fail_data["traceback"])
        self.assertFalse((out_dir / "completion_status.json").exists())

    # -------------------------------------------------------------------------
    # W11C-R03: Offline True Negatives Restricted to Valid Pixels
    # -------------------------------------------------------------------------
    def test_r03_confusion_matrix_validity_and_true_negatives(self):
        """Invalid and unannotated pixels must NOT be counted as True Negatives."""
        out_dir = self.temp_dir / "out_r03"
        cfg_path = self._write_temp_config(self.base_cfg)

        # Craft 50 small samples (4x6 = 24 pixels each) with specific invalid pixels
        mock_recs = []
        for i in range(50):
            # sensory mask: pixel (0,0) is invalid
            valid_mask = torch.ones((4, 6), dtype=torch.uint8)
            valid_mask[0, 0] = 0

            # ground truth: pixel (0,1) is unannotated (value 3), pixel (1,1) is flood (value 2)
            label = torch.zeros((4, 6), dtype=torch.int64)
            label[0, 1] = 3
            label[1, 1] = 2

            mock_recs.append({
                "sample_id": f"s520_{i:04d}",
                "event_id": 520,
                "xA": torch.randn((2, 4, 6), dtype=torch.float32),
                "xB": torch.randn((2, 4, 6), dtype=torch.float32),
                "valid_mask": valid_mask,
                "label": label,
            })

        runner = Week11RealDataSmokeRunner(
            config_path=cfg_path,
            output_dir=out_dir,
            synthetic_mock=True,
        )
        res = runner.run_smoke(mock_records=mock_recs)
        event_row = res["event_row"]

        # In each sample: 24 pixels total.
        # (0,0) has valid_mask=0 -> invalid.
        # (0,1) has valid_mask=1, label=3 -> invalid.
        # Valid pixels per sample = 24 - 2 = 22. Total valid across 50 samples = 22 * 50 = 1100.
        self.assertEqual(event_row["valid_pixels"], 1100)

        # Assert sum of confusion cells EXACTLY equals evaluated valid pixels (never 1200)
        conf_sum = event_row["tp"] + event_row["fp"] + event_row["fn"] + event_row["tn"]
        self.assertEqual(conf_sum, 1100)
        self.assertEqual(conf_sum, event_row["valid_pixels"])

    # -------------------------------------------------------------------------
    # W11C-R04: Stage, Approval, Boolean, and Device Combination Matrix
    # -------------------------------------------------------------------------
    def test_r04_stage_and_approval_matrix(self):
        """Rejects draft, missing, null, or unsupported stage combinations for real execution."""
        # 1. Draft status with true/true flags must fail closed
        cfg_draft_override = copy.deepcopy(self.base_cfg)
        cfg_draft_override["execution_authorizations"]["real_data_execution_authorized"] = True
        cfg_draft_override["execution_authorizations"]["gpu_execution_authorized"] = True
        cfg_path1 = self._write_temp_config(cfg_draft_override)
        with self.assertRaises(PermissionError) as ctx:
            Week11RealDataSmokeRunner(
                config_path=cfg_path1,
                output_dir=self.temp_dir / "out_r04_1",
                synthetic_mock=False,
            )
        self.assertIn("configuration status is 'DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED'", str(ctx.exception))

        # 2. Null status must fail closed
        cfg_null = copy.deepcopy(self.base_cfg)
        cfg_null["status"] = None
        cfg_path2 = self._write_temp_config(cfg_null)
        with self.assertRaises(ValueError) as ctx:
            Week11RealDataSmokeRunner(
                config_path=cfg_path2,
                output_dir=self.temp_dir / "out_r04_2",
                synthetic_mock=True,
            )
        self.assertIn("missing or null", str(ctx.exception))

        # 3. Unsupported status must fail closed
        cfg_unsupported = copy.deepcopy(self.base_cfg)
        cfg_unsupported["status"] = "UNSUPPORTED_STAGE"
        cfg_path3 = self._write_temp_config(cfg_unsupported)
        with self.assertRaises(ValueError) as ctx:
            Week11RealDataSmokeRunner(
                config_path=cfg_path3,
                output_dir=self.temp_dir / "out_r04_3",
                synthetic_mock=True,
            )
        self.assertIn("Unsupported configuration status", str(ctx.exception))

        # 4. Synthetic mock on non-CPU device must fail closed
        with self.assertRaises(PermissionError) as ctx:
            Week11RealDataSmokeRunner(
                config_path=self.draft_config_path,
                output_dir=self.temp_dir / "out_r04_4",
                device="cuda:0",
                synthetic_mock=True,
            )
        self.assertIn("Synthetic mock execution supports 'cpu' only", str(ctx.exception))

    def test_r04_device_syntax_validation(self):
        """Enforces regex ^cuda:(0|[1-9]\d*)$ for real execution, rejecting cpu, mps, bare cuda, negative indices, etc."""
        cfg_approved = copy.deepcopy(self.base_cfg)
        cfg_approved["status"] = "APPROVED_FOR_REAL_DATA_SMOKE"
        cfg_approved["approval_status"] = "APPROVED_FOR_EXECUTION"
        cfg_approved["execution_authorizations"]["real_data_execution_authorized"] = True
        cfg_approved["execution_authorizations"]["gpu_execution_authorized"] = True

        bad_devices = [
            "not-a-device",
            "mps",
            "cpu",
            "cuda",
            "cuda:",
            "cuda:abc",
            "cuda:-1",
            "cuda: 0",
            "gpu:0",
        ]
        cfg_path = self._write_temp_config(cfg_approved)
        for bad_dev in bad_devices:
            with self.assertRaises(ValueError) as ctx:
                Week11RealDataSmokeRunner(
                    config_path=cfg_path,
                    output_dir=self.temp_dir / f"out_bad_dev_{bad_dev.replace(':', '_').replace(' ', '_')}",
                    device=bad_dev,
                    synthetic_mock=False,
                )
            self.assertIn("Real smoke execution strictly requires device format 'cuda:<non-negative integer>'", str(ctx.exception))

        # Valid cuda device syntax passes _enforce_authorizations
        for good_dev in ["cuda:0", "cuda:1"]:
            runner = Week11RealDataSmokeRunner(
                config_path=cfg_path,
                output_dir=self.temp_dir / f"out_good_dev_{good_dev.replace(':', '_')}",
                device=good_dev,
                synthetic_mock=False,
            )
            self.assertEqual(runner.device_str, good_dev)

    # -------------------------------------------------------------------------
    # W11C-R05: Bounded Smoke Invariants Enforcement
    # -------------------------------------------------------------------------
    def test_r05_bounded_smoke_invariants(self):
        """Enforces seed 0, Event 520, order_a, max_samples 50, batch 16, and batches [16,16,16,2]."""
        # 1. max_samples <= 0 (unbounded) rejected
        cfg_unbounded = copy.deepcopy(self.base_cfg)
        cfg_unbounded["smoke_specification"]["max_samples"] = 0
        cfg_path1 = self._write_temp_config(cfg_unbounded)
        runner1 = Week11RealDataSmokeRunner(cfg_path1, self.temp_dir / "out_r05_1", synthetic_mock=True)
        with self.assertRaises(ValueError) as ctx:
            runner1.validate_preflight()
        self.assertIn("max_samples 50 (hard ceiling)", str(ctx.exception))

        # 2. max_samples = 51 (oversized) rejected
        cfg_oversized = copy.deepcopy(self.base_cfg)
        cfg_oversized["smoke_specification"]["max_samples"] = 51
        cfg_path2 = self._write_temp_config(cfg_oversized)
        runner2 = Week11RealDataSmokeRunner(cfg_path2, self.temp_dir / "out_r05_2", synthetic_mock=True)
        with self.assertRaises(ValueError) as ctx:
            runner2.validate_preflight()
        self.assertIn("max_samples 50 (hard ceiling)", str(ctx.exception))

        # 3. order_b (reversed ordering) rejected
        cfg_order_b = copy.deepcopy(self.base_cfg)
        cfg_order_b["smoke_specification"]["stream_order"] = "order_b"
        cfg_path3 = self._write_temp_config(cfg_order_b)
        runner3 = Week11RealDataSmokeRunner(cfg_path3, self.temp_dir / "out_r05_3", synthetic_mock=True)
        with self.assertRaises(ValueError) as ctx:
            runner3.validate_preflight()
        self.assertIn("stream_order 'order_a'", str(ctx.exception))

        # 4. seed != 0 rejected
        cfg_seed1 = copy.deepcopy(self.base_cfg)
        cfg_seed1["smoke_specification"]["checkpoint_seed"] = 1
        cfg_path4 = self._write_temp_config(cfg_seed1)
        runner4 = Week11RealDataSmokeRunner(cfg_path4, self.temp_dir / "out_r05_4", synthetic_mock=True)
        with self.assertRaises(ValueError) as ctx:
            runner4.validate_preflight()
        self.assertIn("checkpoint_seed 0", str(ctx.exception))

        # 5. Non-520 event rejected
        cfg_event = copy.deepcopy(self.base_cfg)
        cfg_event["dataset"]["target_events"] = [279]
        cfg_path5 = self._write_temp_config(cfg_event)
        runner5 = Week11RealDataSmokeRunner(cfg_path5, self.temp_dir / "out_r05_5", synthetic_mock=True)
        with self.assertRaises(ValueError) as ctx:
            runner5.validate_preflight()
        self.assertIn("target_events [520]", str(ctx.exception))

        # 6. Correct setup executes batches [16, 16, 16, 2]
        out_dir = self.temp_dir / "out_r05_batches"
        mock_recs = self._generate_synthetic_event520_records(50)
        runner_ok = Week11RealDataSmokeRunner(self.draft_config_path, out_dir, synthetic_mock=True)
        res = runner_ok.run_smoke(mock_records=mock_recs)
        self.assertEqual(res["event_row"]["batch_sizes"], [16, 16, 16, 2])
        self.assertEqual(res["event_row"]["sample_count"], 50)

    # -------------------------------------------------------------------------
    # W11C-R06: Pure Synthetic Mode Asset Independence
    # -------------------------------------------------------------------------
    # W11C-R06: Pure Synthetic Mode Asset Independence
    # -------------------------------------------------------------------------
    def test_r06_synthetic_mode_zero_asset_reads(self):
        """Synthetic preflight and execution must perform zero physical asset or checkpoint reads."""
        out_dir = self.temp_dir / "out_r06"
        runner = Week11RealDataSmokeRunner(self.draft_config_path, out_dir, synthetic_mock=True)

        from tools.run_week11_real_data_smoke import compute_file_sha256 as real_sha256
        def spy_compute_sha(p):
            p_str = str(p)
            if "checkpoint" in p_str or "final_checkpoint" in p_str or "outputs" in p_str:
                raise AssertionError(f"Checkpoint was hashed during synthetic mode: {p_str}")
            return real_sha256(p)

        with patch("tools.run_week11_real_data_smoke.compute_file_sha256", side_effect=spy_compute_sha):
            # Preflight in synthetic mock mode
            preflight = runner.validate_preflight()
            # Assert checkpoint is marked synthetic and not hashed
            self.assertEqual(preflight["checkpoint_path"], "MOCKED_SYNTHETIC")
            self.assertEqual(preflight["dataset_root"], "MOCKED_SYNTHETIC")
            self.assertEqual(preflight["manifest_path"], "MOCKED_SYNTHETIC")

            # Run smoke with synthetic mock records
            mock_recs = self._generate_synthetic_event520_records(50)
            runner.run_smoke(mock_records=mock_recs)

    # -------------------------------------------------------------------------
    # W11C-R07: CUDA RNG Device Context Guard
    # -------------------------------------------------------------------------
    def test_r07_cuda_rng_device_context_guard(self):
        """CPU execution must NOT touch CUDA RNG APIs even if torch.cuda.is_available() is True."""
        proxy = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)

        # 1. CPU adapter on host with CUDA available
        adapter_cpu = EvidenceConsistencyAdapter(
            model=proxy,
            device="cpu",
        )

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_rng_state_all") as mock_get_rng, \
             patch("torch.cuda.set_rng_state_all") as mock_set_rng:

            xA = torch.randn(2, 2, 32, 32)
            xB = torch.randn(2, 2, 32, 32)
            m = torch.ones(2, 32, 32, dtype=torch.uint8)

            adapter_cpu.adapt_and_predict(xA, xB, sar_valid_mask=m)

            # Neither get nor set should be called because device is cpu
            mock_get_rng.assert_not_called()
            mock_set_rng.assert_not_called()

        # 2. CUDA adapter context: verifies exception-safe restoration when running on CUDA
        adapter_cuda = EvidenceConsistencyAdapter(
            model=proxy,
            device="cpu",
        )
        # Emulate CUDA device context
        adapter_cuda.device = MagicMock()
        adapter_cuda.device.type = "cuda"

        dummy_out = torch.zeros(2, 3, 32, 32)
        adapter_cuda.model = MagicMock(return_value=dummy_out)
        adapter_cuda.source_model = MagicMock(return_value=dummy_out)

        dummy_cuda_rng = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_rng_state_all", return_value=dummy_cuda_rng) as mock_get_cuda, \
             patch("torch.cuda.set_rng_state_all") as mock_set_cuda:

            xA_mock = MagicMock()
            xA_mock.to.return_value = torch.randn(2, 2, 32, 32)
            xB_mock = MagicMock()
            xB_mock.to.return_value = torch.randn(2, 2, 32, 32)
            m_mock = MagicMock()
            m_mock.to.return_value = torch.ones(2, 32, 32, dtype=torch.uint8)

            adapter_cuda.adapt_and_predict(xA_mock, xB_mock, sar_valid_mask=m_mock)
            self.assertGreater(mock_get_cuda.call_count, 0)
            self.assertGreater(mock_set_cuda.call_count, 0)

    # -------------------------------------------------------------------------
    # Manifest Digest Propagation
    # -------------------------------------------------------------------------
    def test_manifest_digest_propagation(self):
        """Verifies that initial and completion manifests contain expected digest fields."""
        out_dir = self.temp_dir / "out_manifest_digests"
        mock_recs = self._generate_synthetic_event520_records(50)
        runner = Week11RealDataSmokeRunner(self.draft_config_path, out_dir, synthetic_mock=True)
        runner.run_smoke(mock_records=mock_recs)

        with open(out_dir / "initial_manifest.json", "r", encoding="utf-8") as f:
            init_m = json.load(f)
        self.assertIn("checkpoint_expected_sha256", init_m)
        self.assertIn("checkpoint_actual_sha256", init_m)
        self.assertIn("manifest_sha256", init_m)

        with open(out_dir / "smoke_manifest.json", "r", encoding="utf-8") as f:
            smoke_m = json.load(f)
        self.assertIn("checkpoint_expected_sha256", smoke_m)
        self.assertIn("checkpoint_actual_sha256", smoke_m)
        self.assertIn("manifest_sha256", smoke_m)


if __name__ == "__main__":
    unittest.main()
