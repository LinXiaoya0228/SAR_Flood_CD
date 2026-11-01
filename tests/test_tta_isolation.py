"""
Unit tests for TTA label isolation, parameter freeze integrity, and checkpoint immutability.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.source_only import SourceOnlyAdapter
from src.tta.adabn import AdaBNAdapter
from src.tta.tent import TentAdapter
from src.tta.entropy import compute_entropy_loss


CHECKPOINT_PATH = "/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/checkpoints/kuro_snunet_official.pt"
EXPECTED_CHECKPOINT_HASH = "6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestTTAIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.raw_model, _ = SNUNetWrapper.load_from_checkpoint(CHECKPOINT_PATH, device=cls.device)

    def setUp(self):
        # Create a fresh model instance for each test
        self.model = copy.deepcopy(self.raw_model)
        self.dummy_xA = torch.randn(2, 3, 64, 64, device=self.device)
        self.dummy_xB = torch.randn(2, 3, 64, 64, device=self.device)
        self.dummy_valid_mask = torch.ones(2, 64, 64, device=self.device)

    def test_adaptation_function_has_no_label_argument(self):
        """Proves the adapt_and_predict interface does not accept ground-truth labels."""
        adapters = [
            SourceOnlyAdapter(self.model, CHECKPOINT_PATH, device=self.device),
            AdaBNAdapter(self.model, CHECKPOINT_PATH, device=self.device),
            TentAdapter(self.model, CHECKPOINT_PATH, device=self.device),
        ]
        for adapter in adapters:
            sig = inspect.signature(adapter.adapt_and_predict)
            self.assertNotIn("label", sig.parameters)
            self.assertNotIn("labels", sig.parameters)
            self.assertNotIn("target", sig.parameters)
            self.assertNotIn("targets", sig.parameters)
            self.assertNotIn("gt", sig.parameters)
            self.assertNotIn("batch", sig.parameters)

            # Explicitly verify passing forbidden label keys raises ValueError
            with self.assertRaises(ValueError) as ctx:
                adapter.adapt_and_predict(
                    self.dummy_xA,
                    self.dummy_xB,
                    labels=torch.zeros(2, 64, 64),
                )
            self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

    def test_labels_cannot_enter_entropy_loss(self):
        """Proves entropy loss accepts only logits and non-label observation validity mask."""
        sig = inspect.signature(compute_entropy_loss)
        self.assertIn("logits", sig.parameters)
        self.assertIn("sar_valid_mask", sig.parameters)
        self.assertNotIn("label", sig.parameters)
        self.assertNotIn("labels", sig.parameters)

        # Loss executes purely unsupervised on logits
        dummy_logits = torch.randn(2, 3, 64, 64, requires_grad=True, device=self.device)
        loss = compute_entropy_loss(dummy_logits, self.dummy_valid_mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(loss.grad_fn)

    def test_only_intended_bn_parameters_change_under_tent(self):
        """Proves Tent updates ONLY BatchNorm affine parameters, leaving non-BN parameters strictly frozen."""
        adapter = TentAdapter(self.model, CHECKPOINT_PATH, learning_rate=1e-3, device=self.device)

        # Store pre-update snapshots
        pre_params = {name: p.detach().clone() for name, p in self.model.named_parameters()}

        # Verify requires_grad setup
        for name, p in self.model.named_parameters():
            if "bn" in name.lower() and ("weight" in name or "bias" in name):
                self.assertTrue(p.requires_grad, f"BN parameter {name} should require grad")
            else:
                self.assertFalse(p.requires_grad, f"Non-BN parameter {name} must NOT require grad")

        # Run 2 adaptation steps
        adapter.adapt_and_predict(self.dummy_xA, self.dummy_xB, self.dummy_valid_mask)
        adapter.adapt_and_predict(self.dummy_xA, self.dummy_xB, self.dummy_valid_mask)

        # Assert BN params changed and non-BN params did not change
        bn_changed = False
        for name, p in self.model.named_parameters():
            if "bn" in name.lower() and ("weight" in name or "bias" in name):
                if not torch.equal(p, pre_params[name]):
                    bn_changed = True
            else:
                # Non-BN must be strictly identical
                self.assertTrue(
                    torch.equal(p, pre_params[name]),
                    f"Non-BN parameter {name} was modified by Tent! Non-BN parameters must be frozen.",
                )
        self.assertTrue(bn_changed, "Tent failed to update any BN affine parameter.")

    def test_no_parameters_change_under_source_only(self):
        """Proves SourceOnlyAdapter modifies zero parameters."""
        adapter = SourceOnlyAdapter(self.model, CHECKPOINT_PATH, device=self.device)
        pre_params = {name: p.detach().clone() for name, p in self.model.named_parameters()}

        for _ in range(3):
            adapter.adapt_and_predict(self.dummy_xA, self.dummy_xB, self.dummy_valid_mask)

        for name, p in self.model.named_parameters():
            self.assertTrue(
                torch.equal(p, pre_params[name]),
                f"Parameter {name} was unexpectedly modified under source-only!",
            )
            self.assertFalse(p.requires_grad)

    def test_only_permitted_bn_state_changes_under_adabn(self):
        """Proves AdaBN modifies NO parameters (weights/biases frozen), updating only running stats."""
        adapter = AdaBNAdapter(self.model, CHECKPOINT_PATH, device=self.device)
        pre_params = {name: p.detach().clone() for name, p in self.model.named_parameters()}

        # Collect initial running stats
        pre_running_means = {}
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and m.running_mean is not None:
                pre_running_means[name] = m.running_mean.detach().clone()

        # Run 3 adaptation steps with distinct input distribution
        for _ in range(3):
            xA = self.dummy_xA * 2.0 + 1.0
            xB = self.dummy_xB * 2.0 + 1.0
            adapter.adapt_and_predict(xA, xB, self.dummy_valid_mask)

        # 1. Zero affine parameter changes
        for name, p in self.model.named_parameters():
            self.assertTrue(
                torch.equal(p, pre_params[name]),
                f"Parameter {name} was modified under AdaBN! AdaBN must not update affine parameters.",
            )
            self.assertFalse(p.requires_grad)

        # 2. Running statistics did update
        running_stats_changed = False
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and m.running_mean is not None:
                if not torch.equal(m.running_mean, pre_running_means[name]):
                    running_stats_changed = True
                    break
        self.assertTrue(running_stats_changed, "AdaBN failed to update running statistics.")

    def test_checkpoint_file_hash_unchanged(self):
        """Proves original checkpoint on disk is never modified by any adaptation routine."""
        hash_before = compute_sha256(CHECKPOINT_PATH)
        self.assertEqual(hash_before, EXPECTED_CHECKPOINT_HASH)

        # Run all three adapters heavily
        for AdapterCls in [SourceOnlyAdapter, AdaBNAdapter, TentAdapter]:
            model = copy.deepcopy(self.raw_model)
            adapter = AdapterCls(model, CHECKPOINT_PATH, device=self.device)
            for _ in range(5):
                adapter.adapt_and_predict(self.dummy_xA, self.dummy_xB, self.dummy_valid_mask)

        hash_after = compute_sha256(CHECKPOINT_PATH)
        self.assertEqual(hash_after, EXPECTED_CHECKPOINT_HASH)

    def test_resetting_event_restores_source_weights_and_optimizer_state(self):
        """Proves reset_to_source exactly restores source weights, running stats, and optimizer state."""
        adapter = TentAdapter(self.model, CHECKPOINT_PATH, learning_rate=1e-3, device=self.device)

        # Run adaptation to create parameter drift
        for _ in range(4):
            adapter.adapt_and_predict(self.dummy_xA, self.dummy_xB, self.dummy_valid_mask)

        drift = adapter.compute_bn_drift()["total_bn_drift_l2"]
        self.assertGreater(drift, 0.0, "BN parameters should have drifted.")
        self.assertEqual(adapter.step_count, 4)

        # Reset to source
        adapter.reset_to_source()

        # Verify drift is strictly 0.0
        drift_after = adapter.compute_bn_drift()["total_bn_drift_l2"]
        self.assertEqual(drift_after, 0.0)
        self.assertEqual(adapter.step_count, 0)

        # Verify all parameters exactly match source
        for name, p in adapter.model.named_parameters():
            p_orig = self.raw_model.state_dict()[name]
            self.assertTrue(
                torch.equal(p.cpu(), p_orig.cpu()),
                f"Parameter {name} did not match original after reset!",
            )


if __name__ == "__main__":
    unittest.main()
