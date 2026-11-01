"""Unit tests for KuroSourceDataset and Policy S2 slope processing."""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.data.source_dataset import compute_policy_s2_slope, KuroSourceDataset
from src.evaluation.metrics import compute_confusion_counts


class MockCRS:
    def __init__(self, is_geographic=False):
        self.is_geographic = is_geographic


class TestSourceDataset(unittest.TestCase):
    def test_slope_flat_terrain(self):
        """Test that a flat elevation grid yields 0 degree slope."""
        dem = np.full((10, 10), 100.0, dtype=np.float32)
        crs = MockCRS(is_geographic=False)
        slope_deg, valid_mask = compute_policy_s2_slope(dem, dx=10.0, dy=10.0, crs=crs)
        self.assertTrue(valid_mask.all())
        np.testing.assert_allclose(slope_deg, 0.0, atol=1e-5)

    def test_slope_45_degree_ramp(self):
        """Test a 45 degree ramp (10m rise per 10m run)."""
        x = np.arange(10) * 10.0  # 10m per pixel
        dem = np.tile(x, (10, 1)).astype(np.float32)
        crs = MockCRS(is_geographic=False)
        slope_deg, valid_mask = compute_policy_s2_slope(dem, dx=10.0, dy=10.0, crs=crs)
        # Interior pixels (away from edge boundary differences) have slope 45 degrees
        interior = slope_deg[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 45.0, atol=1e-3)

    def test_slope_rejects_geographic_crs(self):
        """Test that geographic CRS (degrees) fails loudly."""
        dem = np.full((10, 10), 100.0, dtype=np.float32)
        crs = MockCRS(is_geographic=True)
        with self.assertRaises(ValueError):
            compute_policy_s2_slope(dem, dx=0.0001, dy=0.0001, crs=crs)

    def test_slope_nodata_boundary_cliff_mitigation(self):
        """Test that nodata boundary does not create 89-degree cliff artifacts."""
        dem = np.full((20, 20), 10.0, dtype=np.float32)
        # Introduce nodata corner
        dem[:5, :5] = -32768.0
        crs = MockCRS(is_geographic=False)
        slope_deg, valid_mask = compute_policy_s2_slope(dem, dx=10.0, dy=10.0, crs=crs)
        
        # Valid pixels should all have 0 degree slope (since ground is flat 10m)
        valid_slopes = slope_deg[valid_mask]
        self.assertGreater(len(valid_slopes), 0)
        self.assertLess(float(valid_slopes.max()), 1.0)  # No cliff artifact!

    def test_dataset_inner_val_manifest(self):
        """Test that KuroSourceDataset loads the 5 inner-val events correctly."""
        val_dataset = KuroSourceDataset(
            split="val",
            include_slope=False,
        )
        self.assertEqual(len(val_dataset), 2284)
        
        # Verify event IDs are strictly from the 5 approved inner-val events
        approved_val_events = {267, 275, 497, 518, 1111005}
        for s in val_dataset.samples:
            self.assertIn(s["event_id"], approved_val_events)

        # Check sample item shapes and double-masking rule
        item = val_dataset[0]
        self.assertEqual(item["xA"].shape, (2, 224, 224))
        self.assertEqual(item["xB"].shape, (2, 224, 224))
        self.assertEqual(item["label"].shape, (224, 224))
        self.assertEqual(item["valid_mask"].shape, (224, 224))
        
        # Double masking: where valid_mask == 0, label MUST be 3
        invalid_pixels = (item["valid_mask"] == 0)
        if invalid_pixels.any():
            self.assertTrue((item["label"][invalid_pixels] == 3).all())

    def test_dataset_model_b_slope_input(self):
        """Test that Model B loads 3-channel [VV, VH, slope] inputs with valid stats."""
        val_dataset_b = KuroSourceDataset(
            split="val",
            include_slope=True,
            slope_stats_path="artifacts/week1/source_train_slope_statistics.json",
        )
        self.assertEqual(len(val_dataset_b), 2284)
        item = val_dataset_b[0]
        self.assertEqual(item["xA"].shape, (3, 224, 224))
        self.assertEqual(item["xB"].shape, (3, 224, 224))
        self.assertEqual(item["label"].shape, (224, 224))
        self.assertTrue(torch.isfinite(item["xA"]).all())
        self.assertTrue(torch.isfinite(item["xB"]).all())

    def test_model_a_and_model_b_valid_mask_fairness(self):
        """Verify Model A [VV,VH] and Model B [VV,VH,slope] share identical labels, valid masks, and metric denominators."""
        val_dataset_a = KuroSourceDataset(
            split="val",
            include_slope=False,
        )
        val_dataset_b = KuroSourceDataset(
            split="val",
            include_slope=True,
            slope_stats_path="artifacts/week1/source_train_slope_statistics.json",
        )

        item_a = val_dataset_a[0]
        item_b = val_dataset_b[0]

        # 1. Exact equality of supervised target label and valid mask
        self.assertTrue(torch.equal(item_a["label"], item_b["label"]))
        self.assertTrue(torch.equal(item_a["valid_mask"], item_b["valid_mask"]))

        # 2. SAR channels in xA and xB are identical
        self.assertTrue(torch.equal(item_a["xA"][:2], item_b["xA"][:2]))
        self.assertTrue(torch.equal(item_a["xB"][:2], item_b["xB"][:2]))

        # 3. Model B slope is finite everywhere
        slope_xA = item_b["xA"][2]
        slope_xB = item_b["xB"][2]
        self.assertTrue(torch.isfinite(slope_xA).all())
        self.assertTrue(torch.isfinite(slope_xB).all())
        self.assertTrue(torch.equal(slope_xA, slope_xB))  # Static topographic feature

        # 4. If slope_valid_mask is 0, normalized slope must be 0.0 (neutral prior)
        if "slope_valid_mask" in item_b:
            invalid_slope = (item_b["slope_valid_mask"] == 0)
            if invalid_slope.any():
                self.assertTrue(torch.allclose(slope_xA[invalid_slope], torch.zeros(1), atol=1e-6))

        # 5. Metric denominators for the same sample are bit-identical
        dummy_pred = torch.zeros_like(item_a["label"]).numpy()
        counts_a = compute_confusion_counts(dummy_pred, item_a["label"].numpy(), item_a["valid_mask"].numpy())
        counts_b = compute_confusion_counts(dummy_pred, item_b["label"].numpy(), item_b["valid_mask"].numpy())
        self.assertEqual(counts_a["valid_pixels"], counts_b["valid_pixels"])
        self.assertEqual(counts_a["gt_flood_pixels"], counts_b["gt_flood_pixels"])


if __name__ == "__main__":
    unittest.main()
