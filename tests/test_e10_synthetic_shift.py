"""Unit tests for E10 synthetic domain-shift family."""

import unittest
import numpy as np
import torch

from src.changeguard.synthetic_shift import MultiplicativeBackscatterShift, SUPPORTED_SEVERITIES


class TestSyntheticShift(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.H, self.W = 224, 224
        self.raw_vv = np.clip(self.rng.uniform(0.01, 0.12, (self.H, self.W)), 0.0, 0.15).astype(np.float32)
        self.raw_vh = np.clip(self.rng.uniform(0.005, 0.05, (self.H, self.W)), 0.0, 0.15).astype(np.float32)

    def test_01_clean_identity(self):
        """Clean shift (s=0.0) must produce identical values to standard preprocessing."""
        shift_clean = MultiplicativeBackscatterShift(severity_factor=0.0)
        norm_sar = shift_clean.transform_raw_sar(self.raw_vv, self.raw_vh)

        # Baseline manual computation
        sar = np.stack([self.raw_vv, self.raw_vh], axis=0)
        sar = np.clip(sar, 0.0, 0.15)
        sar = np.nan_to_num(sar, nan=0.15, posinf=0.15, neginf=0.0)
        expected = (sar - np.array([0.0953, 0.0264])[:, None, None]) / np.array([0.0427, 0.0215])[:, None, None]

        np.testing.assert_allclose(norm_sar, expected, rtol=1e-5, atol=1e-6)

    def test_02_monotonic_severity_impact(self):
        """Higher severity must produce strictly higher pre-clamping values and higher mean."""
        sar_clean = MultiplicativeBackscatterShift(0.0).transform_raw_sar(self.raw_vv, self.raw_vh)
        sar_s1 = MultiplicativeBackscatterShift(0.05).transform_raw_sar(self.raw_vv, self.raw_vh)
        sar_s2 = MultiplicativeBackscatterShift(0.10).transform_raw_sar(self.raw_vv, self.raw_vh)
        sar_s3 = MultiplicativeBackscatterShift(0.20).transform_raw_sar(self.raw_vv, self.raw_vh)

        # Means should be monotonically strictly increasing
        m0 = float(sar_clean.mean())
        m1 = float(sar_s1.mean())
        m2 = float(sar_s2.mean())
        m3 = float(sar_s3.mean())

        self.assertLess(m0, m1)
        self.assertLess(m1, m2)
        self.assertLess(m2, m3)

    def test_03_tensor_inversion_roundtrip(self):
        """Applying shift via tensor inversion should match raw computation for non-clamped pixels."""
        # Unclamped values (well below 0.15 threshold)
        raw_vv_low = np.full((self.H, self.W), 0.05, dtype=np.float32)
        raw_vh_low = np.full((self.H, self.W), 0.02, dtype=np.float32)

        shifter = MultiplicativeBackscatterShift(severity_factor=0.10)
        from_raw = shifter.transform_raw_sar(raw_vv_low, raw_vh_low)

        clean_shifter = MultiplicativeBackscatterShift(severity_factor=0.0)
        clean_norm = clean_shifter.transform_raw_sar(raw_vv_low, raw_vh_low)
        t_clean = torch.from_numpy(clean_norm).unsqueeze(0)  # (1, 2, H, W)

        from_tensor = shifter.transform_normalized_tensor(t_clean).squeeze(0).numpy()

        np.testing.assert_allclose(from_raw, from_tensor, rtol=1e-5, atol=1e-5)

    def test_04_finite_and_clamped_boundaries(self):
        """Values must remain strictly finite and within physical bounds."""
        for name, s in SUPPORTED_SEVERITIES.items():
            shifter = MultiplicativeBackscatterShift(severity_factor=s)
            res = shifter.transform_raw_sar(self.raw_vv, self.raw_vh)
            self.assertTrue(np.isfinite(res).all(), f"Severity {name} produced non-finite values!")


if __name__ == "__main__":
    unittest.main()

