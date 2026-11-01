"""Unit tests for SourceTrainer and checkpoint tie-breaking."""

import shutil
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.source_trainer import SourceTrainer, is_better_checkpoint
from src.evaluation.losses import ControlledSourceLoss


class DummyModel(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, xA, xB):
        return self.conv(xA) + self.conv(xB)


class TestSourceTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tie_breaking_rules(self):
        """Test deterministic tie-breaking hierarchy (Section 7.2)."""
        # Case 1: Higher Macro F1 wins
        m1 = {"macro_f1": 0.65, "worst_event_f1": {"f1": 0.20}, "macro_recall": 0.50}
        m2 = {"macro_f1": 0.60, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.60}
        self.assertTrue(is_better_checkpoint(m1, 10, m2, 5))
        self.assertFalse(is_better_checkpoint(m2, 5, m1, 10))

        # Case 2: Tied Macro F1 -> Higher Worst Event F1 wins
        m1_tied = {"macro_f1": 0.600001, "worst_event_f1": {"f1": 0.35}, "macro_recall": 0.50}
        m2_tied = {"macro_f1": 0.600000, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.60}
        self.assertTrue(is_better_checkpoint(m1_tied, 10, m2_tied, 5))

        # Case 3: Tied Worst Event F1 -> Higher Macro Recall wins
        m1_rec = {"macro_f1": 0.60, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.58}
        m2_rec = {"macro_f1": 0.60, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.52}
        self.assertTrue(is_better_checkpoint(m1_rec, 10, m2_rec, 5))

        # Case 4: Tied Recall -> Earlier epoch wins
        m1_ep = {"macro_f1": 0.60, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.50}
        m2_ep = {"macro_f1": 0.60, "worst_event_f1": {"f1": 0.30}, "macro_recall": 0.50}
        self.assertTrue(is_better_checkpoint(m1_ep, 5, m2_ep, 10))
        self.assertFalse(is_better_checkpoint(m1_ep, 10, m2_ep, 5))

    def test_trainer_steps_and_weight_update(self):
        """Test that train_steps updates model weights."""
        model = DummyModel(in_channels=2)
        initial_weight = model.conv.weight.detach().clone()

        # Dummy dataset
        xA = torch.randn(16, 2, 16, 16)
        xB = torch.randn(16, 2, 16, 16)
        label = torch.randint(0, 3, (16, 16, 16))
        vmask = torch.ones(16, 16, 16, dtype=torch.uint8)
        eids = torch.full((16,), 518)

        dataset = [
            {"xA": xA[i], "xB": xB[i], "label": label[i], "valid_mask": vmask[i], "event_id": eids[i]}
            for i in range(16)
        ]
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        metrics = trainer.train_steps(loader, num_optimizer_steps=2)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(trainer.global_step, 2)
        # Weights must have changed
        self.assertFalse(torch.allclose(initial_weight, model.conv.weight))

    def test_checkpoint_save_and_resume(self):
        """Test checkpoint save and bit-exact state restoration."""
        model = DummyModel(in_channels=2)
        trainer1 = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        trainer1.current_epoch = 7
        trainer1.global_step = 42
        trainer1.best_epoch = 5
        trainer1.best_metrics = {"macro_f1": 0.62}

        ckpt_file = trainer1.save_checkpoint("test_ckpt.pt")
        self.assertTrue(ckpt_file.exists())

        # Second trainer resumes
        model2 = DummyModel(in_channels=2)
        trainer2 = SourceTrainer(
            model=model2,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        state = trainer2.resume_from_checkpoint(ckpt_file)
        self.assertEqual(trainer2.current_epoch, 7)
        self.assertEqual(trainer2.global_step, 42)
        self.assertEqual(trainer2.best_epoch, 5)
        self.assertEqual(trainer2.best_metrics["macro_f1"], 0.62)

        # Model weights must match exactly
        for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
