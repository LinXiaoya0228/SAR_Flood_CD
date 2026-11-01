"""Generate streaming train-only slope statistics across the 22 approved source-inner training events.

Strictly adheres to Policy S2 (Physically Specified & Robust Metric Slope):
- Evaluates only the 22 approved inner-training events.
- Never accesses the 5 inner-validation events, the 6 TTA-validation events, or the 10 sealed test events.
- Reads DEM CRS and transform; asserts metric projected coordinates.
- Eliminates bilinear nodata fringe via morphological dilation.
- Inpaints nodata before gradient computation to avoid artificial boundary cliffs.
- Computes physical slope angle in degrees using metric pixel spacing.
- Accumulates streaming global statistics (mean, std, min, max, percentiles, nodata counts).
- Saves outputs/week1/source_train_slope_statistics.json.
"""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from compress_pickle import load
from scipy.ndimage import binary_dilation, distance_transform_edt


def compute_sample_slope(args):
    s_key, record, dataset_root = args
    rel_path = Path(record["path"])
    if rel_path.parts and rel_path.parts[0] == "data":
        sample_dir = dataset_root.joinpath(*rel_path.parts[1:])
    else:
        sample_dir = dataset_root / rel_path

    dem_files = list(sample_dir.glob("*MK0_DEM*.tif"))
    if not dem_files:
        raise FileNotFoundError(f"Missing DEM file in {sample_dir}")
    dem_file = dem_files[0]

    with rasterio.open(dem_file) as src:
        dem = src.read(1).astype(np.float32)
        crs = src.crs
        dx, dy = src.res
        meta_nodata = src.nodata

    if crs is None or crs.is_geographic:
        raise ValueError(f"Sample {s_key} DEM has unsupported geographic/missing CRS: {crs}")

    # Check metric units
    if hasattr(crs, "linear_units") and crs.linear_units not in ("metre", "meter", "m"):
        raise ValueError(f"Sample {s_key} DEM linear units not metres: {crs.linear_units}")

    # Identify nodata
    raw_nodata = (~np.isfinite(dem)) | (dem < -430.0)
    if meta_nodata is not None and np.isfinite(meta_nodata):
        raw_nodata = raw_nodata | (np.isclose(dem, meta_nodata))

    total_pixels = dem.size
    has_nodata = raw_nodata.any()

    if has_nodata:
        if raw_nodata.all():
            return {
                "sample_id": s_key,
                "valid_count": 0,
                "nodata_count": total_pixels,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": np.nan,
                "max": np.nan,
                "hist": np.zeros(9001, dtype=np.int64),
                "dx": dx,
                "dy": dy,
                "crs": str(crs),
            }

        # Dilate by 3 pixels to eliminate bilinear fringe
        nodata_mask = binary_dilation(raw_nodata, iterations=3)
        valid_mask = ~nodata_mask

        # Inpaint with nearest valid elevation to avoid gradient explosion at boundary
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        dem_clean = dem[tuple(indices)]
    else:
        nodata_mask = np.zeros_like(dem, dtype=bool)
        valid_mask = np.ones_like(dem, dtype=bool)
        dem_clean = dem

    gy, gx = np.gradient(dem_clean, dy, dx)
    rise_run = np.sqrt(gx**2 + gy**2)
    slope_deg = np.arctan(rise_run) * (180.0 / np.pi)

    valid_slopes = slope_deg[valid_mask]
    nodata_count = int(nodata_mask.sum())
    valid_count = int(valid_mask.sum())

    if valid_count > 0:
        s_min = float(valid_slopes.min())
        s_max = float(valid_slopes.max())
        s_sum = float(valid_slopes.sum())
        s_sum_sq = float((valid_slopes.astype(np.float64)**2).sum())

        # Histogram with 0.01 degree bins from 0 to 90 degrees
        hist, _ = np.histogram(valid_slopes, bins=9001, range=(0.0, 90.01))
    else:
        s_min = np.nan
        s_max = np.nan
        s_sum = 0.0
        s_sum_sq = 0.0
        hist = np.zeros(9001, dtype=np.int64)

    return {
        "sample_id": s_key,
        "valid_count": valid_count,
        "nodata_count": nodata_count,
        "sum": s_sum,
        "sum_sq": s_sum_sq,
        "min": s_min,
        "max": s_max,
        "hist": hist,
        "dx": dx,
        "dy": dy,
        "crs": str(crs),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    split_v1_path = repo_root / "configs" / "splits" / "kuro_source_inner_split_v1.json"
    dataset_root = Path("/usr1/home/s126mdg22_06/FYP_SARCD/datasets/KuroSiwo")
    pickle_path = Path("/usr1/home/s126mdg22_06/FYP_SARCD/third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz")
    out_dir = repo_root / "outputs" / "week1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "source_train_slope_statistics.json"

    with open(split_v1_path) as f:
        split_cfg = json.load(f)

    with open(split_v1_path, "rb") as f:
        split_hash = hashlib.sha256(f.read()).hexdigest()

    train_events = sorted(int(e) for e in split_cfg["source_inner_train_events"])
    val_events = sorted(int(e) for e in split_cfg["source_inner_validation_events"])
    tta_events = sorted(int(e) for e in split_cfg["tta_validation_events"])
    test_events = sorted(int(e) for e in split_cfg["sealed_test_events"])

    assert len(train_events) == 22, f"Expected 22 train events, got {len(train_events)}"
    assert set(train_events).isdisjoint(set(val_events)), "Security violation: Train overlaps with Val!"
    assert set(train_events).isdisjoint(set(tta_events)), "Security violation: Train overlaps with TTA-Val!"
    assert set(train_events).isdisjoint(set(test_events)), "Security violation: Train overlaps with Sealed Test!"

    print(f"Loading training manifest from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        all_records = load(f)

    train_records = [
        (k, r, dataset_root)
        for k, r in all_records.items()
        if int(r["info"]["actid"]) in train_events
    ]
    train_records.sort(key=lambda x: x[0])

    print(f"Total inner-train samples to process: {len(train_records)}")
    assert len(train_records) == 19178, f"Expected 19,178 samples, got {len(train_records)}"

    t0 = time.time()
    total_valid_pixels = 0
    total_nodata_pixels = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    global_min = 90.0
    global_max = 0.0
    global_hist = np.zeros(9001, dtype=np.int64)

    crs_seen = set()
    res_seen = set()

    max_workers = min(32, (os.cpu_count() or 4))
    print(f"Processing slope across {len(train_records)} samples using {max_workers} worker threads...")

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(compute_sample_slope, train_records, chunksize=128):
            completed += 1
            if completed % 2000 == 0 or completed == len(train_records):
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"  [{completed}/{len(train_records)}] ({rate:.1f} samples/s) - Valid Pixels: {total_valid_pixels:,}")

            if res["valid_count"] > 0:
                total_valid_pixels += res["valid_count"]
                total_nodata_pixels += res["nodata_count"]
                total_sum += res["sum"]
                total_sum_sq += res["sum_sq"]
                if res["min"] < global_min:
                    global_min = res["min"]
                if res["max"] > global_max:
                    global_max = res["max"]
                global_hist += res["hist"]
            else:
                total_nodata_pixels += res["nodata_count"]

            crs_seen.add(res["crs"])
            res_seen.add((round(res["dx"], 2), round(res["dy"], 2)))

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s. Calculating global moments and percentiles...")

    mean = total_sum / total_valid_pixels
    variance = (total_sum_sq / total_valid_pixels) - (mean ** 2)
    std = np.sqrt(max(0.0, variance))

    # Calculate percentiles from global histogram
    # Bin centers: 0.005, 0.015, ..., 89.995, 90.005
    bin_edges = np.linspace(0.0, 90.01, 9002)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cumulative_counts = np.cumsum(global_hist)

    def get_percentile(pct):
        target_count = (pct / 100.0) * total_valid_pixels
        idx = np.searchsorted(cumulative_counts, target_count)
        return float(bin_centers[min(idx, len(bin_centers) - 1)])

    percentiles = {
        "p1": round(get_percentile(1.0), 4),
        "p5": round(get_percentile(5.0), 4),
        "p10": round(get_percentile(10.0), 4),
        "p25": round(get_percentile(25.0), 4),
        "p50_median": round(get_percentile(50.0), 4),
        "p75": round(get_percentile(75.0), 4),
        "p90": round(get_percentile(90.0), 4),
        "p95": round(get_percentile(95.0), 4),
        "p99": round(get_percentile(99.0), 4),
    }

    stats = {
        "formula_version": "Policy_S2_metric_degrees_v1",
        "description": "Empirical train-only slope normalization statistics computed strictly over the 22 approved source-inner training events using physical metric resolution (Policy S2).",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_inner_train_events": train_events,
        "source_inner_train_event_count": len(train_events),
        "source_split_v1_sha256": split_hash,
        "code_commit": "af164ec69aff0105fc462d94417582be79ffd8a4",
        "creation_command": "python tools/generate_train_slope_statistics.py",
        "sample_count": len(train_records),
        "valid_slope_pixels": int(total_valid_pixels),
        "nodata_slope_pixels": int(total_nodata_pixels),
        "nodata_percentage": round((total_nodata_pixels / (total_valid_pixels + total_nodata_pixels)) * 100, 4),
        "crs_summary": sorted(list(crs_seen)),
        "resolution_summary_meters": [list(r) for r in sorted(list(res_seen))],
        "statistics": {
            "mean_degrees": round(float(mean), 6),
            "std_degrees": round(float(std), 6),
            "min_degrees": round(float(global_min), 6),
            "max_degrees": round(float(global_max), 6),
            "percentiles": percentiles,
        },
        "normalization_constants": {
            "slope_mean": round(float(mean), 6),
            "slope_std": round(float(std), 6),
            "slope_clamp_min": 0.0,
            "slope_clamp_max": 90.0,
            "invalid_replacement_value": 0.0,
        },
    }

    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)

    with open(out_file, "rb") as f:
        out_sha256 = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "=" * 60)
    print("SLOPE STATISTICS GENERATED SUCCESSFULLY (Policy S2)")
    print(f"Output File: {out_file}")
    print(f"SHA-256 Digest: {out_sha256}")
    print(f"Processed Samples: {len(train_records):,} across 22 inner-train events")
    print(f"Valid Slope Pixels: {total_valid_pixels:,}")
    print(f"Nodata Pixels: {total_nodata_pixels:,} ({stats['nodata_percentage']}%)")
    print(f"Mean Slope: {mean:.4f} degrees")
    print(f"Std Slope: {std:.4f} degrees")
    print(f"Median Slope: {percentiles['p50_median']:.4f} degrees")
    print(f"Min / Max: {global_min:.4f}° / {global_max:.4f}°")
    print(f"Percentiles: {percentiles}")
    print("=" * 60)


if __name__ == "__main__":
    main()
