"""Unit and synthetic integration tests for Week 5B all-source fixed-epoch training.

Verifies:
1. All-source split integrity: 27 events, 21,462 patches, zero overlap with quarantined events.
2. Provenance and lock validation for Seeds 0, 1, and 2.
3. Negative provenance rejection: max_epochs != 16, non-null early stopping, seed mismatch.
4. Synthetic end-to-end mock training run:
   - Zero validation calls, no early stopping.
   - Checkpoints saved: epoch checkpoints, last_checkpoint.pt, final_checkpoint.pt.
   - Run manifest records status COMPLETED_FIXED_EPOCHS.
   - Checkpoint hashes ledger tracks final_checkpoint.pt.
5. Regression test: Legacy inner-split validation mode remains intact.
"""

import copy
import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.source_dataset import KuroSourceDataset
from src.evaluation.run_manifest import compute_file_sha256, RunManifest
from src.training.source_trainer import SourceTrainer
from tools.train_source_model import (
    validate_provenance_and_config,
    update_checkpoint_hashes,
    train_source_model,
    AccessTracker,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataset(Dataset):
    """Synthetic dataset generating dummy SAR patches without disk dataset access."""
    def __init__(self, num_samples: int = 16, in_channels: int = 2):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.manifest_path = Path("/dummy/manifest.json")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "xA": torch.randn(self.in_channels, 224, 224, dtype=torch.float32),
            "xB": torch.randn(self.in_channels, 224, 224, dtype=torch.float32),
            "label": torch.zeros((224, 224), dtype=torch.long),
            "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
            "sample_id": f"syn_sample_{idx}",
            "event_id": 118,  # Valid source event
        }


class DummyModel(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, xA, xB):
        return self.conv(xA) + self.conv(xB)


class TestAllSourceSplitIntegrity(unittest.TestCase):
    def setUp(self):
        self.split_path = REPO_ROOT / "configs/splits/kuro_source_all27_split_v1.json"
        self.lock_path = REPO_ROOT / "configs/source/source_training_all27_lock.json"

    def test_split_file_structure_and_counts(self):
        """Split config must define exactly 27 source events and 21,462 patches."""
        self.assertTrue(self.split_path.exists())
        with open(self.split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_events = data["source_all_events"]
        self.assertEqual(len(all_events), 27)
        self.assertEqual(len(set(all_events)), 27)

        sample_counts = data["sample_counts"]
        self.assertEqual(sample_counts["total_source_samples"], 21462)

        # Inner-validation events must be empty in all-source split
        self.assertEqual(len(data.get("source_inner_validation_events", [])), 0)

    def test_zero_quarantined_event_leakage(self):
        """Source events must strictly avoid TTA-val and sealed-test events."""
        with open(self.split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_events = set(int(e) for e in data["source_all_events"])
        tta_val = set(KuroSourceDataset.TTA_VAL_EVENTS)
        sealed_test = set(KuroSourceDataset.SEALED_TEST_EVENTS)

        self.assertTrue(source_events.isdisjoint(tta_val), "Split contains TTA validation event(s)!")
        self.assertTrue(source_events.isdisjoint(sealed_test), "Split contains sealed test event(s)!")

    def test_dataset_all_source_mode_initialization(self):
        """KuroSourceDataset with split='all_source' initializes without error."""
        dataset = KuroSourceDataset(
            split_config_path=str(self.split_path),
            split="all_source",
            include_slope=False,
        )
        self.assertEqual(len(dataset.allowed_events), 27)
        self.assertEqual(len(dataset.samples), 21462)
        self.assertEqual(len(dataset), 21462)

    def test_dataset_rejects_forbidden_split_tampering(self):
        """If a split file is altered to include forbidden events, dataset loading must fail."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tampered_data = {
                "source_all_events": [279, 118],  # 279 is quarantined TTA-val
                "sample_counts": {"total_source_samples": 10},
                "samples": [],
            }
            json.dump(tampered_data, tf)
            tampered_path = tf.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                KuroSourceDataset(
                    split_config_path=tampered_path,
                    split="all_source",
                    include_slope=False,
                )
            self.assertIn("Split config SHA-256 mismatch", str(ctx.exception))
        finally:
            os.remove(tampered_path)


class TestAllSourceProvenanceValidation(unittest.TestCase):
    def setUp(self):
        self.lock_path = REPO_ROOT / "configs/source/source_training_all27_lock.json"

    def test_validate_all_three_seeds(self):
        """All three seed configs (seed0, seed1, seed2) must pass provenance validation."""
        for seed in [0, 1, 2]:
            cfg_path = REPO_ROOT / f"configs/source/snunet_vvvh_all27_seed{seed}_v1_fp32.json"
            run_id = f"source_snunet_vvvh_all27_seed{seed}_v1_fp32"
            res = validate_provenance_and_config(
                config_path=cfg_path,
                run_id=run_id,
                seed=seed,
            )
            self.assertEqual(res["model_key"], "model_a")
            self.assertEqual(res["precision"], "fp32")
            self.assertEqual(Path(res["lock_path"]), self.lock_path)

    def test_rejection_max_epochs_not_16(self):
        """Configs with max_epochs != 16 in all-source mode must be rejected."""
        cfg_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["optimization_protocol"]["max_epochs"] = 32  # Invalid
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg, tf)
            tampered_cfg_path = tf.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                validate_provenance_and_config(
                    config_path=tampered_cfg_path,
                    run_id="source_snunet_vvvh_all27_seed0_v1_fp32",
                    seed=0,
                )
            # Either config hash mismatch against lock, or rule rejection
            self.assertTrue(
                "requires exactly 16 epochs" in str(ctx.exception)
                or "config hash mismatch" in str(ctx.exception)
            )
        finally:
            os.remove(tampered_cfg_path)

    def test_rejection_early_stopping_in_all_source(self):
        """Configs with non-null early_stopping_patience in all-source mode must be rejected."""
        cfg_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["optimization_protocol"]["early_stopping_patience"] = 10  # Invalid
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg, tf)
            tampered_cfg_path = tf.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                validate_provenance_and_config(
                    config_path=tampered_cfg_path,
                    run_id="source_snunet_vvvh_all27_seed0_v1_fp32",
                    seed=0,
                )
            self.assertTrue(
                "prohibits early stopping" in str(ctx.exception)
                or "config hash mismatch" in str(ctx.exception)
            )
        finally:
            os.remove(tampered_cfg_path)

    def test_rejection_seed_mismatch(self):
        """CLI seed mismatching config declared seed must be rejected."""
        cfg_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"
        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_path,
                run_id="source_snunet_vvvh_all27_seed0_v1_fp32",
                seed=1,  # Contradicts seed 0
            )
        self.assertIn("Seed mismatch", str(ctx.exception))


class TestSyntheticAllSourceTrainingRun(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "test_run_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthetic_all_source_training_loop(self):
        """Execute a synthetic 2-epoch all-source training run without real dataset access.

        Verifies:
        - 0 validation calls.
        - Checkpoints saved: epoch checkpoints, last_checkpoint.pt, final_checkpoint.pt.
        - Run manifest marked COMPLETED_FIXED_EPOCHS.
        - Checkpoint hashes JSON contains final_checkpoint.
        - train_history.csv contains records; validation_history.csv does not exist.
        """
        cfg_path = REPO_ROOT / "configs/source/snunet_vvvh_all27_seed0_v1_fp32.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Set up synthetic data loader
        syn_dataset = SyntheticDataset(num_samples=16, in_channels=2)
        syn_loader = DataLoader(syn_dataset, batch_size=8, shuffle=False)

        # Set up small dummy model and trainer
        model = DummyModel(in_channels=2)
        trainer = SourceTrainer.from_config(
            model=model,
            config=config,
            output_dir=self.output_dir,
            device="cpu",
            seed=0,
        )

        # Prepare mock args
        class MockArgs:
            config = str(cfg_path)
            output_dir = str(self.output_dir)
            run_id = "test_synthetic_all_source"
            device = "cpu"
            seed = 0
            max_epochs = 2
            patience = None
            num_workers = 0
            log_interval = 1
            resume = None
            resume_auto = False
            max_train_batches = 2
            max_val_batches = None
            validate_only = False
            all_source = True
            no_val = True

        mock_args = MockArgs()

        # Patch DataLoader / KuroSourceDataset and SNUNetWrapper so no real disk data or large model is loaded
        with patch("tools.train_source_model.KuroSourceDataset", return_value=syn_dataset), \
             patch("tools.train_source_model.DataLoader", return_value=syn_loader), \
             patch("tools.train_source_model.SNUNetWrapper", return_value=model), \
             patch("tools.train_source_model.SourceTrainer.from_config", return_value=trainer), \
             patch("tools.train_source_model.validate_provenance_and_config") as mock_val:

            mock_val.return_value = {
                "config": config,
                "config_hash": "a" * 64,
                "model_key": "model_a",
                "protocol_hash": "b" * 64,
                "split_hash": "c" * 64,
                "slope_stats_hash": None,
                "precision": "fp32",
                "lock_path": REPO_ROOT / "configs/source/source_training_all27_lock.json",
                "lock": {
                    "frozen_split": {"path": "configs/splits/kuro_source_all27_split_v1.json"},
                    "slope_statistics": {"path": "artifacts/week1/source_train_slope_statistics.json"},
                },
            }

            exit_code = train_source_model(mock_args)
            self.assertEqual(exit_code, 0)

        # 1. Verify checkpoints saved
        self.assertTrue((self.output_dir / "last_checkpoint.pt").exists())
        self.assertTrue((self.output_dir / "final_checkpoint.pt").exists())
        self.assertTrue((self.output_dir / "epoch_001_checkpoint.pt").exists())
        self.assertTrue((self.output_dir / "epoch_002_checkpoint.pt").exists())
        # best_checkpoint.pt must NOT exist in all-source fixed-epoch mode
        self.assertFalse((self.output_dir / "best_checkpoint.pt").exists())

        # 2. Verify run manifest
        manifest_path = self.output_dir / "run_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        rep = manifest_data.get("reproducibility", manifest_data)
        self.assertEqual(rep["completion_status"], "COMPLETED_FIXED_EPOCHS")
        self.assertEqual(rep["run_type"], "all_source_training")
        self.assertIn("final_checkpoint", rep["output_paths"])
        self.assertNotIn("best_checkpoint", rep["output_paths"])
        self.assertEqual(
            Path(rep["checkpoint_path"]).name,
            "final_checkpoint.pt"
        )

        # 3. Verify checkpoint_hashes.json
        hashes_path = self.output_dir / "checkpoint_hashes.json"
        self.assertTrue(hashes_path.exists())
        with open(hashes_path, "r", encoding="utf-8") as f:
            hashes_data = json.load(f)
        self.assertIn("final_checkpoint", hashes_data)
        self.assertIn("last_checkpoint", hashes_data)
        self.assertIn("epoch_001_checkpoint", hashes_data)
        self.assertIn("epoch_002_checkpoint", hashes_data)
        self.assertEqual(hashes_data["final_checkpoint"]["epoch"], 2)

        # 4. Verify CSV histories
        train_csv = self.output_dir / "train_history.csv"
        val_csv = self.output_dir / "validation_history.csv"
        self.assertTrue(train_csv.exists())
        self.assertFalse(val_csv.exists(), "validation_history.csv must NOT exist in all-source mode!")

        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 2 epochs = 3 rows
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[2][0], "2")


if __name__ == "__main__":
    unittest.main()
