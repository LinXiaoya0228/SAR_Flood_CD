"""Unit tests for DEM NoData sanitization and preprocessing correctness."""

import sys
import json
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import sanitize_and_normalize_dem, KuroValidationDataset


DEM_MEAN = 67.0293
DEM_STD = 1765.0062


def test_nan_becomes_zero():
    """Verify NaN values become normalized 0.0 and are flagged invalid."""
    raw = np.array([[np.nan, 100.0], [50.0, np.nan]], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(raw, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert norm[0, 0] == 0.0
    assert norm[1, 1] == 0.0
    assert inv[0, 0]
    assert inv[1, 1]
    assert not inv[0, 1]
    assert not inv[1, 0]
    print("PASS: test_nan_becomes_zero")


def test_infinities_become_zero():
    """Verify positive and negative infinity become normalized 0.0 and are flagged invalid."""
    raw = np.array([[np.inf, -np.inf], [0.0, 50.0]], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(raw, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert norm[0, 0] == 0.0
    assert norm[0, 1] == 0.0
    assert inv[0, 0]
    assert inv[0, 1]
    assert not inv[1, 0]
    assert not inv[1, 1]
    print("PASS: test_infinities_become_zero")


def test_sentinel_negative_32768_becomes_zero():
    """Verify standard SRTM/Copernicus NoData sentinel -32768 becomes normalized 0.0."""
    raw = np.array([[-32768.0, -32767.0], [100.0, 200.0]], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(raw, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert norm[0, 0] == 0.0
    assert norm[0, 1] == 0.0
    assert inv[0, 0]
    assert inv[0, 1]
    assert not inv[1, 0]
    assert not inv[1, 1]
    print("PASS: test_sentinel_negative_32768_becomes_zero")


def test_values_below_negative_1000_become_zero():
    """Verify arbitrary values strictly below -1000.0 become normalized 0.0."""
    raw = np.array([[-1000.01, -1500.0], [-9999.0, -999.9]], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(raw, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    # Below -1000.0
    assert norm[0, 0] == 0.0
    assert norm[0, 1] == 0.0
    assert norm[1, 0] == 0.0
    assert inv[0, 0]
    assert inv[0, 1]
    assert inv[1, 0]

    # -999.9 is above -1000.0 and should remain valid
    expected_valid = (-999.9 - DEM_MEAN) / DEM_STD
    assert np.isclose(norm[1, 1], expected_valid, atol=1e-5)
    assert not inv[1, 1]
    print("PASS: test_values_below_negative_1000_become_zero")


def test_dead_sea_negative_430_remains_valid():
    """Verify physically valid depressions (e.g. -430 m Dead Sea, -100 m) remain valid."""
    raw = np.array([[-430.0, -100.0], [-10.0, -1.0]], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(raw, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert not inv.any(), "Physically valid below-sea-level elevations were incorrectly flagged invalid!"
    expected_neg430 = (-430.0 - DEM_MEAN) / DEM_STD
    expected_neg100 = (-100.0 - DEM_MEAN) / DEM_STD
    assert np.isclose(norm[0, 0], expected_neg430, atol=1e-5)
    assert np.isclose(norm[0, 1], expected_neg100, atol=1e-5)
    assert norm[0, 0] < 0.0
    print("PASS: test_dead_sea_negative_430_remains_valid")


def test_normal_elevations_normalized_normally():
    """Verify 0, 67.0293 (mean), 1000, and 1800 are normalized normally."""
    elevations = np.array([0.0, 67.0293, 1000.0, 1800.0], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(elevations, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert not inv.any()
    expected_0 = (0.0 - DEM_MEAN) / DEM_STD
    expected_mean = 0.0
    expected_1000 = (1000.0 - DEM_MEAN) / DEM_STD
    expected_1800 = (1800.0 - DEM_MEAN) / DEM_STD

    assert np.isclose(norm[0], expected_0, atol=1e-5)
    assert np.isclose(norm[1], expected_mean, atol=1e-5)
    assert np.isclose(norm[2], expected_1000, atol=1e-5)
    assert np.isclose(norm[3], expected_1800, atol=1e-5)
    print("PASS: test_normal_elevations_normalized_normally")


def test_no_nan_or_inf_in_returned_dem():
    """Verify returned DEM array/tensor contains strictly finite numbers under hostile inputs."""
    hostile = np.array([
        [np.nan, np.inf, -np.inf],
        [-32768.0, -100000.0, -1000.5],
        [-430.0, 0.0, 2500.0]
    ], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(hostile, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert np.isfinite(norm).all(), "Non-finite values present in sanitized DEM output!"
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    print("PASS: test_no_nan_or_inf_in_returned_dem")


def test_high_elevation_not_clipped():
    """Verify valid high elevations (1800, 2500, 5000, 8848) are not clipped."""
    highs = np.array([1000.0, 1800.0, 2500.0, 5000.0, 8848.0], dtype=np.float32)
    norm, inv = sanitize_and_normalize_dem(highs, dem_mean=DEM_MEAN, dem_std=DEM_STD)

    assert not inv.any()
    for raw_val, norm_val in zip(highs, norm):
        expected = (raw_val - DEM_MEAN) / DEM_STD
        assert np.isclose(norm_val, expected, atol=1e-5), f"Mismatch for elevation {raw_val}"
    # Verify strictly monotonic
    assert (np.diff(norm) > 0).all(), "High elevations must preserve strict monotonicity without clipping!"
    print("PASS: test_high_elevation_not_clipped")


def test_sar_channels_and_labels_unchanged_by_dem_sanitization():
    """Verify that SAR channels, labels, and SAR valid masks are identical between raw and sanitized modes using synthetic fixture."""
    import tempfile
    import rasterio
    from rasterio.transform import from_origin
    from compress_pickle import dump

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_dir = tmp_path / "synthetic_sample_001"
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

        np.random.seed(42)
        sl1_vv = (np.random.rand(H, W) * 0.1).astype(np.float32)
        sl1_vh = (np.random.rand(H, W) * 0.05).astype(np.float32)
        sl2_vv = (np.random.rand(H, W) * 0.1).astype(np.float32)
        sl2_vh = (np.random.rand(H, W) * 0.05).astype(np.float32)
        ms1_vv = (np.random.rand(H, W) * 0.1).astype(np.float32)
        ms1_vh = (np.random.rand(H, W) * 0.05).astype(np.float32)
        dem = np.full((H, W), 100.0, dtype=np.float32)
        dem[0, 0] = -32768.0  # nodata sentinel
        mna = np.ones((H, W), dtype=np.uint8)
        mna[0, 0] = 0  # invalid mask
        mlu = np.zeros((H, W), dtype=np.int32)
        mlu[10:20, 10:20] = 2  # flood
        mlu[0, 0] = 3  # invalid label

        def write_tif(name, data, dtype="float32"):
            prof = profile.copy()
            prof["dtype"] = dtype
            with rasterio.open(sample_dir / f"{name}.tif", "w", **prof) as dst:
                dst.write(data.astype(dtype), 1)

        write_tif("SL1_IVV", sl1_vv)
        write_tif("SL1_IVH", sl1_vh)
        write_tif("SL2_IVV", sl2_vv)
        write_tif("SL2_IVH", sl2_vh)
        write_tif("MS1_IVV", ms1_vv)
        write_tif("MS1_IVH", ms1_vh)
        write_tif("MK0_DEM", dem)
        write_tif("MK0_MNA", mna, dtype="uint8")
        write_tif("MK0_MLU", mlu, dtype="int32")

        # Create synthetic manifest without touching any real dataset or manifest
        manifest_dict = {
            "synthetic_sample_1": {
                "info": {"actid": 9999},
                "path": "synthetic_sample_001",
            }
        }
        manifest_path = tmp_path / "synthetic_manifest.gz"
        with open(manifest_path, "wb") as f:
            dump(manifest_dict, f, compression="gzip")

        # Instantiate dataset with synthetic inputs (Event 9999 is isolated)
        ds_sanitized = KuroValidationDataset(
            dataset_root=str(tmp_path),
            manifest_path=str(manifest_path),
            allowed_events=[9999],
            sealed_test_events=[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013],
            canonical_preevent="pre_event_2",
        )

        assert len(ds_sanitized) == 1
        sample = ds_sanitized[0]
        xA = sample["xA"]
        xB = sample["xB"]
        lbl = sample["label"]
        vld = sample["valid_mask"]
        dem_inv = sample["dem_invalid_mask"]

        # Model input shapes
        assert xA.shape == (3, 224, 224), f"Unexpected shape {xA.shape}"
        assert xB.shape == (3, 224, 224), f"Unexpected shape {xB.shape}"
        assert lbl.shape == (224, 224)
        assert vld.shape == (224, 224)
        assert dem_inv.shape == (224, 224)

        # Check finite
        assert torch.isfinite(xA).all()
        assert torch.isfinite(xB).all()

        # Channels 0 and 1 are VV and VH
        # Check that SAR channels in xA match pre2 SAR exactly
        assert torch.equal(xA[:2], sample["pre2"][:2])
        assert torch.equal(xB[:2], sample["post"][:2])

        # Check that DEM channel is at index 2
        assert torch.equal(xA[2:3], xB[2:3])  # DEM is static across temporal pair
        assert torch.isfinite(xA[2:3]).all()

        # SAR valid mask must be unchanged and independent of DEM
        # Check that valid mask is uint8 and label is long
        assert vld.dtype == torch.uint8
        assert lbl.dtype == torch.int64
        assert dem_inv.dtype == torch.bool

    print("PASS: test_sar_channels_and_labels_unchanged_by_dem_sanitization")


class TestPreprocessing(unittest.TestCase):
    def test_nan_becomes_zero(self):
        test_nan_becomes_zero()

    def test_infinities_become_zero(self):
        test_infinities_become_zero()

    def test_sentinel_negative_32768_becomes_zero(self):
        test_sentinel_negative_32768_becomes_zero()

    def test_values_below_negative_1000_become_zero(self):
        test_values_below_negative_1000_become_zero()

    def test_dead_sea_negative_430_remains_valid(self):
        test_dead_sea_negative_430_remains_valid()

    def test_normal_elevations_normalized_normally(self):
        test_normal_elevations_normalized_normally()

    def test_no_nan_or_inf_in_returned_dem(self):
        test_no_nan_or_inf_in_returned_dem()

    def test_high_elevation_not_clipped(self):
        test_high_elevation_not_clipped()

    def test_sar_channels_and_labels_unchanged_by_dem_sanitization(self):
        test_sar_channels_and_labels_unchanged_by_dem_sanitization()


if __name__ == "__main__":
    unittest.main()
