#!/usr/bin/env python3
"""Test suite for SiamUnet_diff second backbone and wrapper (E10 Phase 1)."""

import tempfile
import unittest
from pathlib import Path
import torch

from src.models.siam_diff_wrapper import SiamDiffWrapper, SiamUnet_diff


class TestSiamDiff(unittest.TestCase):
    """Verifies SiamUnet_diff model and wrapper contracts."""

    def test_01_raw_model_forward_shape(self):
        model = SiamUnet_diff(in_channels=2, out_channels=3)
        xA = torch.randn(2, 2, 64, 64)
        xB = torch.randn(2, 2, 64, 64)
        out = model(xA, xB)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_02_wrapper_interface_and_prediction(self):
        wrapper = SiamDiffWrapper(in_channels=2, out_channels=3)
        xA = torch.randn(2, 2, 128, 128)
        xB = torch.randn(2, 2, 128, 128)
        preds = wrapper.predict(xA, xB)

        self.assertIn("logits", preds)
        self.assertIn("probabilities", preds)
        self.assertIn("predicted_class", preds)
        self.assertIn("flood_probability", preds)

        self.assertEqual(preds["logits"].shape, (2, 3, 128, 128))
        self.assertEqual(preds["probabilities"].shape, (2, 3, 128, 128))
        self.assertEqual(preds["predicted_class"].shape, (2, 128, 128))
        self.assertEqual(preds["flood_probability"].shape, (2, 128, 128))

        # Check probability bounds
        probs = preds["probabilities"]
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        prob_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

    def test_03_channel_truncation(self):
        """Ensures passing 3-channel input [VV, VH, DEM] safely slices first 2 channels."""
        wrapper = SiamDiffWrapper(in_channels=2, out_channels=3)
        xA3 = torch.randn(1, 3, 64, 64)
        xB3 = torch.randn(1, 3, 64, 64)
        out = wrapper(xA3, xB3)
        self.assertEqual(out.shape, (1, 3, 64, 64))

    def test_04_parameter_count_is_lightweight(self):
        """Verifies parameter count is approximately 1.35M (lightweight criterion)."""
        model = SiamUnet_diff(in_channels=2, out_channels=3)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Should be between 1.0M and 2.0M
        self.assertGreater(n_params, 1_000_000)
        self.assertLess(n_params, 2_000_000)

    def test_05_checkpoint_save_and_reload(self):
        wrapper = SiamDiffWrapper(in_channels=2, out_channels=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            torch.save(
                {
                    "model_state_dict": wrapper.state_dict(),
                    "epoch": 5,
                    "seed": 0,
                    "architecture": "SiamUnet_diff",
                },
                ckpt_path,
            )

            loaded_wrapper, meta = SiamDiffWrapper.load_from_checkpoint(ckpt_path)
            self.assertEqual(meta["epoch"], 5)
            self.assertEqual(meta["architecture"], "SiamUnet_diff")

            # Forward check with loaded model in eval mode (disables dropout)
            wrapper.eval()
            loaded_wrapper.eval()
            xA = torch.randn(1, 2, 64, 64)
            xB = torch.randn(1, 2, 64, 64)
            orig_out = wrapper(xA, xB)
            loaded_out = loaded_wrapper(xA, xB)
            self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
