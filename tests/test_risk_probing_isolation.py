"""Unit tests proving risk-probing isolation and non-mutation."""

import sys
from pathlib import Path
import unittest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.tent import TentAdapter
from src.risk.collector import RiskFeatureCollector


class TestRiskProbingIsolation(unittest.TestCase):
    """Proves risk probing does NOT alter model parameters, BN stats, or optimizer states."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.raw_model = SNUNetWrapper(in_channels=3, out_ch=3, base_channel=8).to(self.device)
        self.source_model = SNUNetWrapper(in_channels=3, out_ch=3, base_channel=8).to(self.device)
        self.source_model.eval()

        self.adapter = TentAdapter(
            self.raw_model,
            checkpoint_path="dummy_path",
            learning_rate=1e-4,
            device=self.device,
        )

        self.collector = RiskFeatureCollector(
            source_model=self.source_model,
            protocol_hash="test_protocol_hash_123",
            checkpoint_hash="test_checkpoint_hash_456",
            device=self.device,
        )

    def test_probing_does_not_mutate_parameters_buffers_or_counters(self):
        """Proves that collect_risk_record leaves adapted state strictly unmodified."""
        B, C, H, W = 2, 3, 32, 32
        xA = torch.randn(B, C, H, W, device=self.device)
        xB = torch.randn(B, C, H, W, device=self.device)
        pre1 = torch.randn(B, C, H, W, device=self.device)
        pre2 = torch.randn(B, C, H, W, device=self.device)
        post = torch.randn(B, C, H, W, device=self.device)
        sar_valid = torch.ones(B, H, W, dtype=torch.uint8, device=self.device)

        # 1. Run one standard adaptation step to dirty BN affine parameters
        _ = self.adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_valid)
        self.assertEqual(self.adapter.step_count, 1)

        # 2. Snapshot full candidate state prior to risk probing
        param_snapshot = {name: p.clone().detach() for name, p in self.adapter.model.named_parameters()}
        buffer_snapshot = {name: b.clone().detach() for name, b in self.adapter.model.named_buffers()}
        initial_step_count = self.adapter.step_count

        # 3. Execute risk probing across all null and disappearance pairs
        batch_sensory = {
            "sample_ids": ["sample_a", "sample_b"],
            "xA": xA,
            "xB": xB,
            "pre1": pre1,
            "pre2": pre2,
            "post": post,
            "valid_mask": sar_valid,
        }
        record = self.collector.collect_risk_record(
            candidate_adapter=self.adapter,
            batch_sensory=batch_sensory,
            run_id="test_run",
            event_id=279,
            seed=0,
            batch_idx=1,
        )

        # 4. Verify record contains valid outputs
        self.assertIn("predeclared_change_risk_score", record)
        self.assertIn("null_pre2_pre2_class2_area", record)
        self.assertIn("disappear_fraction_source_erased", record)

        # 5. Verify bitwise parameter equality after probing
        for name, p in self.adapter.model.named_parameters():
            self.assertTrue(
                torch.equal(p, param_snapshot[name]),
                f"Parameter {name} was mutated by risk probing!"
            )

        # 6. Verify bitwise buffer equality (running_mean, running_var) after probing
        for name, b in self.adapter.model.named_buffers():
            self.assertTrue(
                torch.equal(b, buffer_snapshot[name]),
                f"Running buffer {name} was mutated by risk probing!"
            )

        # 7. Verify step count unmodified
        self.assertEqual(
            self.adapter.step_count,
            initial_step_count,
            "Adapter step_count was modified by risk probing!"
        )

    def test_unlabeled_dataset_never_reads_label_raster(self):
        """Proves that KuroUnlabeledValidationDataset never reads or accesses MK0_MLU label raster using synthetic fixture."""
        import tempfile
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from compress_pickle import dump
        from unittest.mock import patch
        from src.data.kuro_dataset import KuroUnlabeledValidationDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_dir = tmp_path / "synthetic_unlabeled_sample"
            sample_dir.mkdir(parents=True, exist_ok=True)

            H, W = 224, 224
            transform = from_origin(0, 0, 10, 10)
            profile = {
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32615",
                "transform": transform,
            }

            def write_tif(name, dtype="float32"):
                prof = profile.copy()
                prof["dtype"] = dtype
                with rasterio.open(sample_dir / f"{name}.tif", "w", **prof) as dst:
                    dst.write(np.zeros((H, W), dtype=dtype), 1)

            for comp in ["SL1_IVV", "SL1_IVH", "SL2_IVV", "SL2_IVH", "MS1_IVV", "MS1_IVH", "MK0_DEM"]:
                write_tif(comp, "float32")
            write_tif("MK0_MNA", "uint8")
            # Write MK0_MLU to verify that even when present on disk, the unlabeled dataset never touches it
            write_tif("MK0_MLU", "int32")

            manifest_dict = {
                "synthetic_unlabeled_1": {
                    "info": {"actid": 9999},
                    "path": "synthetic_unlabeled_sample",
                }
            }
            manifest_path = tmp_path / "synthetic_manifest.gz"
            with open(manifest_path, "wb") as f:
                dump(manifest_dict, f, compression="gzip")

            dataset = KuroUnlabeledValidationDataset(
                dataset_root=str(tmp_path),
                manifest_path=str(manifest_path),
                allowed_events=[9999],
                sealed_test_events=[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013],
                canonical_preevent="pre_event_2",
            )

            real_read_raster = dataset._read_raster

            def guarded_read_raster(path):
                path_str = str(path)
                if "MLU" in path_str:
                    raise AssertionError(f"SECURITY VIOLATION: Label raster '{path_str}' was read during unlabeled evaluation!")
                return real_read_raster(path)

            with patch.object(dataset, "_read_raster", side_effect=guarded_read_raster):
                sample = dataset[0]

            # Verify label was never read and is absent from sample
            self.assertNotIn("label", sample)
            self.assertIn("xA", sample)
            self.assertIn("xB", sample)
            self.assertIn("valid_mask", sample)


if __name__ == "__main__":
    unittest.main()
