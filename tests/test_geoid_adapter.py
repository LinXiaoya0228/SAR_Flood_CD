"""
Unit tests for GEOID-Flood adapter and evaluation pipeline.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.external_evaluation.geoid_adapter import (
    discover_geoid_heldout_tiles,
    GEOIDTileDataset,
    evaluate_model_on_geoid,
)


class TestGEOIDAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = REPO_ROOT / "data/external_geoid/geoid-flood-heldout"
        cls.catalog_path = REPO_ROOT / "data/geoid_metadata/heldout_tile_catalog.json"

    def test_01_discovery_all_tiles(self):
        """Verifies all 1,301 tiles are discovered and valid."""
        records = discover_geoid_heldout_tiles(self.base_dir, self.catalog_path)
        self.assertEqual(len(records), 1301)
        # Check first record has all fields populated
        rec0 = records[0]
        self.assertTrue(Path(rec0.pre_path).is_file())
        self.assertTrue(Path(rec0.post_path).is_file())
        self.assertTrue(Path(rec0.label_path).is_file())
        self.assertTrue(Path(rec0.validity_path).is_file())

    def test_02_discovery_filtered_event(self):
        """Verifies filtering by single event returns exact known tile count."""
        records = discover_geoid_heldout_tiles(self.base_dir, self.catalog_path, filter_events=["EMSR858"])
        self.assertEqual(len(records), 10)
        for r in records:
            self.assertEqual(r.event_id, "EMSR858")

    def test_03_tile_dataset_item(self):
        """Verifies dataset item shapes, types, value ranges, and mask invariants."""
        records = discover_geoid_heldout_tiles(self.base_dir, self.catalog_path, filter_events=["EMSR858"])
        dataset = GEOIDTileDataset(records[:1])
        item = dataset[0]

        # Shape assertions
        self.assertEqual(item["xA"].shape, torch.Size([16, 2, 256, 256]))
        self.assertEqual(item["xB"].shape, torch.Size([16, 2, 256, 256]))
        self.assertEqual(item["labels"].shape, torch.Size([16, 256, 256]))
        self.assertEqual(item["valid_mask"].shape, torch.Size([16, 256, 256]))

        # Finite checks
        self.assertTrue(torch.isfinite(item["xA"]).all().item())
        self.assertTrue(torch.isfinite(item["xB"]).all().item())

        # Mask invariant: where valid_mask == 0, label must be 3
        invalid_locs = (item["valid_mask"] == 0)
        if invalid_locs.any():
            self.assertTrue((item["labels"][invalid_locs] == 3).all().item())

        # Valid classes should only be 0, 1, 2
        valid_locs = (item["valid_mask"] == 1)
        if valid_locs.any():
            valid_labels = item["labels"][valid_locs].numpy()
            self.assertTrue(set(np.unique(valid_labels)).issubset({0, 1, 2}))

    def test_04_evaluate_mock_model(self):
        """Verifies evaluation harness with a lightweight mock model."""
        class MockModel(nn.Module):
            def forward(self, xA, xB):
                # Returns 3 class logits: B x 3 x H x W
                B, C, H, W = xA.shape
                logits = torch.zeros(B, 3, H, W, dtype=torch.float32)
                # Assign highest logit to class 0 (background)
                logits[:, 0, :, :] = 2.0
                return logits

        records = discover_geoid_heldout_tiles(self.base_dir, self.catalog_path, filter_events=["EMSR858"])
        dataset = GEOIDTileDataset(records[:2])
        model = MockModel()

        res = evaluate_model_on_geoid(
            model=model,
            dataset=dataset,
            device="cpu",
            batch_size_tiles=1,
            num_workers=0,
        )

        self.assertIn("tile_records", res)
        self.assertIn("event_metrics", res)
        self.assertIn("macro_metrics", res)
        self.assertIn("pooled_metrics", res)

        self.assertEqual(len(res["tile_records"]), 2)
        self.assertEqual(len(res["event_metrics"]), 1)
        ev0 = res["event_metrics"][0]
        self.assertEqual(ev0["event_id"], "EMSR858")
        self.assertEqual(ev0["tp"] + ev0["fp"] + ev0["fn"] + ev0["tn"], ev0["valid_pixels"])

    def test_05_nan_transform_values(self):
        """Verifies synthetic transform for both old-fill (0.0) and source-fill (0.15)."""
        records = discover_geoid_heldout_tiles(self.base_dir, self.catalog_path, filter_events=["EMSR858"])
        ds_old = GEOIDTileDataset(records[:1], sar_clamp_max=0.15, sar_nan_val=0.0)
        ds_src = GEOIDTileDataset(records[:1], sar_clamp_max=0.15, sar_nan_val=0.15)
        self.assertEqual(ds_old.sar_nan_val, 0.0)
        self.assertEqual(ds_src.sar_nan_val, 0.15)


if __name__ == "__main__":
    unittest.main()

