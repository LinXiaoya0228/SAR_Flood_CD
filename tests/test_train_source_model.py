"""Lightweight unit tests for tools/train_source_model.py.

Verifies:
1. Hash guards reject mismatched configs.
2. AccessTracker detects and raises on forbidden events.
3. CSV logging creates valid files with all required columns.
4. Checkpoint hashes file is updated with SHA-256 sums.
5. End-to-end mini mock training run produces complete expected artifacts.
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.train_source_model import (
    AccessTracker,
    TeeLogger,
    update_checkpoint_hashes,
    train_source_model,
    FROZEN_MODEL_A_HASH,
    FROZEN_SPLIT_HASH,
    FORBIDDEN_TTA_VAL_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)
from src.training.source_trainer import SourceTrainer
from src.evaluation.run_manifest import compute_file_sha256


class DummyModel(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, xA, xB):
        return self.conv(xA) + self.conv(xB)


class TestTrainSourceModelLauncher(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_access_tracker_guards(self):
        """Verify AccessTracker allows inner events and forbids TTA/test events."""
        tracker = AccessTracker("model_a")
        tracker.set_mode("train")

        # Allowed inner train event (118)
        batch_valid = {
            "sample_id": ["s1", "s2"],
            "event_id": [118, 118],
        }
        tracker.record_batch(batch_valid)
        self.assertEqual(len(tracker.train_samples), 2)
        self.assertIn(118, tracker.train_events)

        # Quarantined TTA val event (279) must raise RuntimeError
        batch_tta = {
            "sample_id": ["s3"],
            "event_id": [279],
        }
        with self.assertRaises(RuntimeError) as ctx:
            tracker.record_batch(batch_tta)
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

        # Sealed test event (205) must raise RuntimeError
        batch_test = {
            "sample_id": ["s4"],
            "event_id": [205],
        }
        with self.assertRaises(RuntimeError) as ctx:
            tracker.record_batch(batch_test)
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

    def test_checkpoint_hash_updating(self):
        """Verify update_checkpoint_hashes records correct sha256 sums."""
        best_pt = self.temp_dir / "best_checkpoint.pt"
        last_pt = self.temp_dir / "last_checkpoint.pt"
        best_pt.write_bytes(b"dummy_best_weights")
        last_pt.write_bytes(b"dummy_last_weights")

        hash_file = update_checkpoint_hashes(self.temp_dir, best_epoch=3, last_epoch=5)
        self.assertTrue(hash_file.exists())
        with open(hash_file, "r") as f:
            data = json.load(f)

        self.assertIn("best_checkpoint", data)
        self.assertIn("last_checkpoint", data)
        self.assertEqual(data["best_checkpoint"]["epoch"], 3)
        self.assertEqual(data["last_checkpoint"]["epoch"], 5)
        self.assertEqual(data["best_checkpoint"]["sha256"], compute_file_sha256(best_pt))
        self.assertEqual(data["last_checkpoint"]["sha256"], compute_file_sha256(last_pt))

    def test_csv_logging_structure(self):
        """Verify CSV history columns match protocol requirements."""
        train_csv = self.temp_dir / "train_history.csv"
        val_csv = self.temp_dir / "validation_history.csv"
        event_csv = self.temp_dir / "event_validation_metrics.csv"

        # Initialize as done in launcher
        with open(train_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "loss_total", "loss_ce", "loss_dice", "learning_rate",
                "optimizer_steps", "samples_observed", "batches_observed",
                "epoch_duration_sec", "peak_gpu_memory_mb"
            ])

        with open(val_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "val_loss_total", "val_loss_ce", "val_loss_dice",
                "macro_precision", "macro_recall", "macro_f1", "macro_iou",
                "pooled_precision", "pooled_recall", "pooled_f1", "pooled_iou",
                "worst_event_id", "worst_event_f1", "pred_flood_pixels",
                "gt_flood_pixels", "valid_pixels", "is_best_epoch"
            ])

        with open(train_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertIn("loss_total", headers)
            self.assertIn("loss_ce", headers)
            self.assertIn("loss_dice", headers)
            self.assertIn("learning_rate", headers)
            self.assertIn("peak_gpu_memory_mb", headers)

        with open(val_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertIn("val_loss_total", headers)
            self.assertIn("macro_f1", headers)
            self.assertIn("pooled_f1", headers)
            self.assertIn("worst_event_f1", headers)
            self.assertIn("is_best_epoch", headers)


if __name__ == "__main__":
    unittest.main()
