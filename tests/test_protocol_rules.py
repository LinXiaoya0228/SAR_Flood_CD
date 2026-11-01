"""Unit tests for mathematical loss formulation, absent-class Dice stability, tie-breaking, and protocol rules."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


def compute_unweighted_ce_plus_soft_dice(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 3, eps: float = 1.0) -> torch.Tensor:
    """Exact frozen loss formula:

    total_loss = 0.5 * CE + 0.5 * mean_soft_dice_over_classes_0_1_2
    """
    # 1. Cross entropy
    ce_loss = F.cross_entropy(logits, target, ignore_index=ignore_index)

    # 2. Soft Dice over valid classes [0, 1, 2]
    probs = F.softmax(logits, dim=1)
    valid_mask = (target != ignore_index)

    dice_losses = []
    for c in [0, 1, 2]:
        p_c = probs[:, c, :, :]
        t_c = (target == c).float()

        # Mask invalid pixels
        p_c_valid = p_c[valid_mask]
        t_c_valid = t_c[valid_mask]

        if p_c_valid.numel() == 0:
            continue

        intersection = torch.sum(p_c_valid * t_c_valid)
        cardinality = torch.sum(p_c_valid ** 2) + torch.sum(t_c_valid ** 2)

        dice_score = (2.0 * intersection + eps) / (cardinality + eps)
        dice_losses.append(1.0 - dice_score)

    mean_dice_loss = torch.mean(torch.stack(dice_losses)) if len(dice_losses) > 0 else torch.tensor(0.0)
    return 0.5 * ce_loss + 0.5 * mean_dice_loss


def select_best_epoch(epoch_records: list) -> int:
    """Deterministic tie-breaking rule:

    1. higher event_macro_flood_f1
    2. higher worst_event_flood_f1
    3. higher event_macro_flood_recall
    4. earlier epoch (smaller integer)
    """
    # Sort key: (-macro_f1, -worst_f1, -macro_recall, epoch)
    sorted_records = sorted(
        epoch_records,
        key=lambda r: (-round(r["macro_f1"], 5), -round(r["worst_f1"], 5), -round(r["macro_recall"], 5), r["epoch"])
    )
    return sorted_records[0]["epoch"]


class TestProtocolRules(unittest.TestCase):
    def test_loss_finite_and_differentiable(self):
        torch.manual_seed(42)
        # B=2, C=3, H=32, W=32
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))  # labels 0, 1, 2, 3

        loss = compute_unweighted_ce_plus_soft_dice(logits, target)
        self.assertTrue(torch.isfinite(loss), "Loss must be finite")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")

        loss.backward()
        self.assertIsNotNone(logits.grad, "Logits must receive gradients")
        self.assertTrue(torch.isfinite(logits.grad).all(), "Gradients must be finite")

    def test_absent_class_dice_behavior(self):
        # Create a batch where class 2 (flood) is completely absent in ground truth
        logits = torch.zeros(1, 3, 16, 16)
        # High confidence for class 0
        logits[:, 0, :, :] = 10.0
        logits[:, 1, :, :] = -10.0
        logits[:, 2, :, :] = -10.0  # very low prediction for flood

        target = torch.zeros(1, 16, 16, dtype=torch.long)  # All class 0 (no flood)

        loss = compute_unweighted_ce_plus_soft_dice(logits, target, eps=1.0)
        self.assertTrue(torch.isfinite(loss), "Loss must be finite when class is absent")
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN when class is absent")

    def test_absent_class_false_alarm_penalty(self):
        # Target has no flood (all class 0)
        target = torch.zeros(1, 16, 16, dtype=torch.long)

        # Case A: Correctly predicting no flood
        logits_correct = torch.zeros(1, 3, 16, 16)
        logits_correct[:, 0, :, :] = 5.0
        logits_correct[:, 2, :, :] = -5.0
        loss_correct = compute_unweighted_ce_plus_soft_dice(logits_correct, target, eps=1.0)

        # Case B: False alarms (predicting flood when there is none)
        logits_fa = torch.zeros(1, 3, 16, 16)
        logits_fa[:, 0, :, :] = -5.0
        logits_fa[:, 2, :, :] = 5.0
        loss_fa = compute_unweighted_ce_plus_soft_dice(logits_fa, target, eps=1.0)

        self.assertGreater(loss_fa.item(), loss_correct.item(), "False alarms on absent class must increase loss!")

    def test_deterministic_epoch_tie_breaking(self):
        # Case 1: Clear winner by macro F1
        records1 = [
            {"epoch": 10, "macro_f1": 0.55, "worst_f1": 0.40, "macro_recall": 0.50},
            {"epoch": 15, "macro_f1": 0.58, "worst_f1": 0.42, "macro_recall": 0.52},
            {"epoch": 20, "macro_f1": 0.57, "worst_f1": 0.45, "macro_recall": 0.55},
        ]
        self.assertEqual(select_best_epoch(records1), 15)

        # Case 2: Tie in macro F1 -> broken by worst-event F1
        records2 = [
            {"epoch": 12, "macro_f1": 0.60000, "worst_f1": 0.45, "macro_recall": 0.55},
            {"epoch": 18, "macro_f1": 0.60000, "worst_f1": 0.48, "macro_recall": 0.50},
        ]
        self.assertEqual(select_best_epoch(records2), 18)

        # Case 3: Tie in macro F1 and worst F1 -> broken by macro recall
        records3 = [
            {"epoch": 14, "macro_f1": 0.60000, "worst_f1": 0.45, "macro_recall": 0.52},
            {"epoch": 22, "macro_f1": 0.60000, "worst_f1": 0.45, "macro_recall": 0.57},
        ]
        self.assertEqual(select_best_epoch(records3), 22)

        # Case 4: Complete tie across all 3 metrics -> broken by earlier epoch
        records4 = [
            {"epoch": 25, "macro_f1": 0.60000, "worst_f1": 0.45, "macro_recall": 0.55},
            {"epoch": 19, "macro_f1": 0.60000, "worst_f1": 0.45, "macro_recall": 0.55},
        ]
        self.assertEqual(select_best_epoch(records4), 19)

    def test_median_epoch_retraining_rule(self):
        best_epochs = [34, 42, 28]
        median_epoch = int(np.median(best_epochs))
        self.assertEqual(median_epoch, 34)

        best_epochs_tied = [40, 40, 45]
        median_epoch_tied = int(np.median(best_epochs_tied))
        self.assertEqual(median_epoch_tied, 40)

    def test_effective_batch_divisibility(self):
        effective_b = 32
        for micro in [2, 4, 8, 16]:
            self.assertEqual(effective_b % micro, 0)
            accum_steps = effective_b // micro
            self.assertEqual(accum_steps * micro, effective_b)


if __name__ == "__main__":
    unittest.main()
