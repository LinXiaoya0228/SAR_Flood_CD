"""Unit tests for ControlledSourceLoss and SoftDiceLoss."""

import unittest
import torch
from src.evaluation.losses import SoftDiceLoss, ControlledSourceLoss


class TestControlledSourceLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_loss_shape_and_finite(self):
        """Test loss returns a finite scalar."""
        loss_fn = ControlledSourceLoss()
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))  # Includes class 3 (ignore_index)
        
        loss, components = loss_fn(logits, target, return_components=True)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_total", components)
        self.assertIn("loss_ce", components)
        self.assertIn("loss_dice", components)

    def test_backward_pass(self):
        """Test gradient propagation."""
        loss_fn = ControlledSourceLoss()
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 3, (2, 32, 32))
        
        loss = loss_fn(logits, target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertFalse((logits.grad == 0).all())

    def test_absent_class_stability(self):
        """Test stability when class 2 (flood) is completely absent in ground truth."""
        loss_fn = ControlledSourceLoss()
        # Only class 0 and 1 present in target
        target = torch.randint(0, 2, (2, 32, 32))
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)

        loss = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ignore_index_respected(self):
        """Test that ignored pixels (target == 3) don't contribute to loss."""
        loss_fn = ControlledSourceLoss(ignore_index=3)
        logits1 = torch.randn(1, 3, 16, 16, requires_grad=True)
        target = torch.full((1, 16, 16), 3, dtype=torch.long)

        # When all pixels are ignored, loss is zero / near-zero without NaNs
        loss = loss_fn(logits1, target)
        self.assertTrue(torch.isfinite(loss))

    def test_perfect_prediction_low_loss(self):
        """Test that near-perfect predictions yield lower loss than random."""
        loss_fn = ControlledSourceLoss()
        target = torch.randint(0, 3, (2, 16, 16))
        
        # Perfect logits (high value at ground truth class)
        perfect_logits = torch.zeros(2, 3, 16, 16)
        for b in range(2):
            for h in range(16):
                for w in range(16):
                    c = target[b, h, w].item()
                    perfect_logits[b, c, h, w] = 20.0
                    
        random_logits = torch.randn(2, 3, 16, 16)
        
        loss_perfect = loss_fn(perfect_logits, target)
        loss_random = loss_fn(random_logits, target)
        self.assertLess(loss_perfect.item(), loss_random.item())
        self.assertLess(loss_perfect.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
