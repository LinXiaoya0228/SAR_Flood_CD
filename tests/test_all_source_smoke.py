"""Unit and synthetic integration tests for tools/run_all27_smoke.py.

Verifies:
1. Smoke preflight safety guards:
   - Rejection of formal output directory (cannot overwrite outputs/week5_source_all27).
   - Rejection of approved formal run IDs (cannot fake formal completion).
   - Validation of config, protocol, split, and lock hashes.
2. Synthetic execution of diagnostic smoke workflow:
   - Epoch 1 training pass without validation.
   - Resumable checkpoint saving.
   - Reload and bit-exact parameter restoration verification.
   - Diagnostic Epoch 2 continuation steps.
   - Generation of smoke_manifest.json and smoke_summary.json.
   - Status marked SMOKE_TEST_PASSED and is_formal_run=False.
"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.run_all27_smoke import (
    validate_smoke_preflight,
    run_all27_smoke,
    FORMAL_OUTPUT_PREFIX,
    DEFAULT_SMOKE_RUN_ID,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 16):
        self.num_samples = num_samples
        self.allowed_events = set(range(1, 28))
        self.manifest_path = Path("/dummy/manifest.json")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "xA": torch.randn(2, 224, 224, dtype=torch.float32),
            "xB": torch.randn(2, 224, 224, dtype=torch.float32),
            "label": torch.zeros((224, 224), dtype=torch.long),
            "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
            "sample_id": f"syn_{idx}",
            "event_id": 118,
        }


class DummyModel(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, xA, xB):
        h = self.conv(xA) + self.conv(xB)
        return self.bn(h)


class TestSmokePreflight(unittest.TestCase):
    def setUp(self):
        self.config_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"

    def test_preflight_validates_successfully(self):
        """Standard smoke arguments pass preflight validation."""
        output_dir = REPO_ROOT / "outputs/week6_smoke/test_smoke_dir"
        res = validate_smoke_preflight(
            config_path=self.config_path,
            output_dir=output_dir,
            run_id="smoke_test_run_id_001",
        )
        self.assertIn("config", res)
        self.assertEqual(len(res["config_hash"]), 64)

    def test_rejection_formal_directory_reuse(self):
        """Smoke preflight strictly rejects output directory inside formal production path."""
        formal_dir = REPO_ROOT / FORMAL_OUTPUT_PREFIX / "source_snunet_vvvh_all27_seed0_v1_fp32"
        with self.assertRaises(ValueError) as ctx:
            validate_smoke_preflight(
                config_path=self.config_path,
                output_dir=formal_dir,
                run_id="smoke_test_run_id_001",
            )
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("formal production directory", str(ctx.exception))

    def test_rejection_formal_run_id_reuse(self):
        """Smoke preflight strictly rejects reusing an approved formal run ID."""
        output_dir = REPO_ROOT / "outputs/week6_smoke/test_smoke_dir"
        with self.assertRaises(ValueError) as ctx:
            validate_smoke_preflight(
                config_path=self.config_path,
                output_dir=output_dir,
                run_id="source_snunet_vvvh_all27_seed0_v1_fp32",  # Formal ID!
            )
        self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("matches approved formal run ID", str(ctx.exception))


class TestSyntheticSmokeExecution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "smoke_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthetic_smoke_workflow(self):
        """Execute end-to-end synthetic smoke test verifying reload and continuation."""
        syn_dataset = SyntheticDataset(num_samples=16)
        syn_loader = DataLoader(syn_dataset, batch_size=8, shuffle=False)

        model = DummyModel(in_channels=2)

        class MockArgs:
            config = str(REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json")
            output_dir = str(self.output_dir)
            run_id = "test_synthetic_smoke"
            device = "cpu"
            seed = 0
            num_workers = 0
            log_interval = 1
            resume_batches = 2
            max_train_batches = 2
            validate_only = False

        mock_args = MockArgs()

        with patch("tools.run_all27_smoke.KuroSourceDataset", return_value=syn_dataset), \
             patch("tools.run_all27_smoke.DataLoader", return_value=syn_loader), \
             patch("tools.run_all27_smoke.SNUNetWrapper", side_effect=lambda **kwargs: DummyModel(in_channels=2)):

            exit_code = run_all27_smoke(mock_args)
            self.assertEqual(exit_code, 0)

        # 1. Verify checkpoints created
        self.assertTrue((self.output_dir / "epoch_001_checkpoint.pt").exists())
        self.assertTrue((self.output_dir / "last_checkpoint.pt").exists())
        # final_checkpoint.pt must NOT be created by smoke test
        self.assertFalse((self.output_dir / "final_checkpoint.pt").exists())

        # 2. Verify manifest and summary
        manifest_path = self.output_dir / "smoke_manifest.json"
        summary_path = self.output_dir / "smoke_summary.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(summary_path.exists())

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        self.assertEqual(manifest_data["status"], "SMOKE_TEST_PASSED")
        self.assertFalse(manifest_data["is_formal_run"])
        self.assertEqual(summary_data["status"], "SMOKE_TEST_PASSED")
        self.assertTrue(summary_data["resumption_audit"]["parameter_bit_exact_match"])
        self.assertTrue(summary_data["resumption_audit"]["all_gradients_finite"])
        self.assertTrue(summary_data["resumption_audit"]["all_parameters_finite"])
        self.assertTrue(summary_data["resumption_audit"]["all_buffers_finite"])
        self.assertEqual(summary_data["epoch_1"]["validation_calls_made"], 0)


if __name__ == "__main__":
    unittest.main()

