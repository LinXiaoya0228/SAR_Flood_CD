"""Day 2 Supplemental Baseline Audit Script:
1. Exhaustive event-wise DEM statistics across all 27 training events and 6 validation events.
2. Elevation counterfactual sweep on Event 559 fixed GT-flood subset.
3. DEM NoData diagnostic policies evaluation across validation events.
"""

import sys
import json
import csv
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import compute_confusion_counts, compute_flood_metrics, EventMetricAccumulator


# =============================================================================
# PART 1: EXHAUSTIVE DEM STATISTICS ACROSS ALL TRAINING & VALIDATION EVENTS
# =============================================================================

def _process_single_dem(fpath_str):
    try:
        with rasterio.open(fpath_str) as src:
            arr = src.read(1)
        
        total = arr.size
        is_nan = np.isnan(arr)
        n_nan = int(np.count_nonzero(is_nan))
        
        is_32768 = (arr == -32768.0) | (arr == -32767.0)
        n_32768 = int(np.count_nonzero(is_32768))
        
        is_below_1000 = (arr < -1000.0)
        n_below_1000 = int(np.count_nonzero(is_below_1000))
        
        valid_mask = np.isfinite(arr) & (~is_below_1000)
        valid_vals = arr[valid_mask]
        
        if valid_vals.size == 0:
            return (total, n_nan, n_32768, n_below_1000, 0, 0.0, None, None, np.array([], dtype=np.float32))
        
        step = max(1, valid_vals.size // 500)
        sampled = valid_vals[::step][:500].astype(np.float32)
        
        return (
            total,
            n_nan,
            n_32768,
            n_below_1000,
            int(valid_vals.size),
            float(np.sum(valid_vals, dtype=np.float64)),
            float(np.min(valid_vals)),
            float(np.max(valid_vals)),
            sampled
        )
    except Exception as e:
        print(f"Error reading {fpath_str}: {e}")
        return (0, 0, 0, 0, 0, 0.0, None, None, np.array([], dtype=np.float32))


def compute_all_event_dem_statistics(dataset_root, output_csv_path):
    print("\n" + "=" * 80)
    print("PART 1: EXHAUSTIVE DEM STATISTICS ACROSS TRAINING AND VALIDATION EVENTS")
    print("=" * 80)

    train_events = [
        130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502,
        498, 497, 496, 492, 147, 267, 273, 275, 417, 567,
        1111011, 1111004, 1111009, 1111010, 1111006, 1111005
    ]
    val_events = [279, 437, 520, 559, 1111003, 1111008]

    all_target_events = [("train", eid) for eid in train_events] + [("val", eid) for eid in val_events]
    rows = []

    for split_tag, eid in all_target_events:
        edir = Path(dataset_root) / str(eid)
        tifs = sorted([str(p) for p in edir.glob("**/MK0_DEM*.tif")])
        num_rasters = len(tifs)
        if num_rasters == 0:
            print(f"Event {eid:8d} ({split_tag}): 0 DEM rasters found!")
            continue

        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(_process_single_dem, tifs, chunksize=64))

        total_pixels = sum(r[0] for r in results)
        total_nan = sum(r[1] for r in results)
        total_32768 = sum(r[2] for r in results)
        total_below_1000 = sum(r[3] for r in results)
        total_valid_count = sum(r[4] for r in results)
        total_valid_sum = sum(r[5] for r in results)

        valid_mins = [r[6] for r in results if r[6] is not None]
        valid_maxs = [r[7] for r in results if r[7] is not None]

        overall_min = float(np.min(valid_mins)) if valid_mins else np.nan
        overall_max = float(np.max(valid_maxs)) if valid_maxs else np.nan
        overall_mean = (total_valid_sum / total_valid_count) if total_valid_count > 0 else np.nan

        # Pool samples for percentiles
        pooled_samples = np.concatenate([r[8] for r in results if len(r[8]) > 0]) if total_valid_count > 0 else np.array([])
        if len(pooled_samples) > 0:
            p1 = float(np.percentile(pooled_samples, 1))
            p50 = float(np.percentile(pooled_samples, 50))
            p99 = float(np.percentile(pooled_samples, 99))
        else:
            p1, p50, p99 = np.nan, np.nan, np.nan

        pct_nan = (total_nan / total_pixels * 100) if total_pixels > 0 else 0.0
        pct_32768 = (total_32768 / total_pixels * 100) if total_pixels > 0 else 0.0
        pct_below_1000 = (total_below_1000 / total_pixels * 100) if total_pixels > 0 else 0.0

        row = {
            "split": split_tag,
            "event_id": eid,
            "rasters_examined": num_rasters,
            "valid_min": overall_min,
            "p1": p1,
            "median": p50,
            "mean": overall_mean,
            "p99": p99,
            "valid_max": overall_max,
            "pct_nan": pct_nan,
            "pct_32768": pct_32768,
            "pct_below_1000": pct_below_1000,
        }
        rows.append(row)
        print(
            f"Event {eid:8d} ({split_tag:5s}): Rasters={num_rasters:5d} | "
            f"Min={overall_min:7.1f}m, P1={p1:6.1f}m, Med={p50:6.1f}m, Mean={overall_mean:6.1f}m, "
            f"P99={p99:6.1f}m, Max={overall_max:7.1f}m | "
            f"NaN={pct_nan:5.2f}%, -32768={pct_32768:5.2f}%, <-1000={pct_below_1000:5.2f}%"
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved train & val event DEM statistics: {output_csv_path}")
    return rows


# =============================================================================
# PART 2: ELEVATION COUNTERFACTUAL SWEEP FOR EVENT 559
# =============================================================================

def run_event559_elevation_sweep(cfg, output_csv_path, device="cuda"):
    print("\n" + "=" * 80)
    print("PART 2: ELEVATION COUNTERFACTUAL SWEEP FOR EVENT 559")
    print("=" * 80)

    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=[559],
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
    )

    # Document fixed subset: samples containing GT flood
    flood_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        lbl = sample["label"].numpy()
        vld = sample["valid_mask"].numpy()
        d_vld = (vld == 1) & (lbl != 3)
        if ((lbl == 2) & d_vld).sum() > 0:
            flood_indices.append(i)

    print(f"Event 559 total samples: {len(dataset)}, samples with GT flood: {len(flood_indices)}")

    model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    model.freeze()

    dem_mean = cfg["dem_normalization"]["mean"]
    dem_std = cfg["dem_normalization"]["std"]

    elevations = [0.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]
    sweep_rows = []

    for elev in elevations:
        norm_val = (elev - dem_mean) / dem_std
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_pred_pixels = 0
        total_valid_pixels = 0
        total_gt_flood_pixels = 0
        gt_flood_probs = []

        with torch.no_grad():
            for idx in flood_indices:
                sample = dataset[idx]
                xA = sample["xA"].unsqueeze(0).clone().to(device)
                xB = sample["xB"].unsqueeze(0).clone().to(device)
                xA[:, 2, :, :] = norm_val
                xB[:, 2, :, :] = norm_val

                out = model.predict(xA, xB)
                pred = out["predicted_class"].squeeze(0).cpu().numpy()
                prob = out["flood_probability"].squeeze(0).cpu().numpy()

                lbl = sample["label"].numpy()
                vld = sample["valid_mask"].numpy()
                d_vld = (vld == 1) & (lbl != 3)

                gt_flood_mask = (lbl == 2) & d_vld
                if gt_flood_mask.any():
                    gt_flood_probs.extend(prob[gt_flood_mask].tolist())

                counts = compute_confusion_counts(pred, lbl, vld)
                total_tp += counts["tp"]
                total_fp += counts["fp"]
                total_fn += counts["fn"]
                total_tn += counts["tn"]
                total_pred_pixels += counts["pred_flood_pixels"]
                total_valid_pixels += counts["valid_pixels"]
                total_gt_flood_pixels += counts["gt_flood_pixels"]

        met = compute_flood_metrics({"tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn})
        mean_gt_prob = float(np.mean(gt_flood_probs)) if gt_flood_probs else 0.0

        row = {
            "elevation_meters": elev,
            "normalized_dem_input": norm_val,
            "gt_flood_pixels": total_gt_flood_pixels,
            "mean_flood_probability_gt": mean_gt_prob,
            "predicted_flood_pixels": total_pred_pixels,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "precision": met["precision"],
            "recall": met["recall"],
            "f1": met["f1"],
            "iou": met["iou"],
        }
        sweep_rows.append(row)
        print(
            f"Elevation {elev:5.0f}m (norm={norm_val:+.4f}) | "
            f"Mean GT Prob={mean_gt_prob:.4f} | Pred Pixels={total_pred_pixels:6d} | "
            f"TP={total_tp:5d}, FP={total_fp:4d}, FN={total_fn:5d} | "
            f"Prec={met['precision']:.4f}, Rec={met['recall']:.4f}, F1={met['f1']:.4f}, IoU={met['iou']:.4f}"
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        for r in sweep_rows:
            writer.writerow(r)
    print(f"\nSaved Event 559 elevation sweep: {output_csv_path}")
    return sweep_rows


# =============================================================================
# PART 3: DEM NODATA DIAGNOSTIC POLICIES ACROSS VALIDATION EVENTS
# =============================================================================

def run_dem_nodata_policy_evaluation(cfg, output_csv_path, device="cuda"):
    print("\n" + "=" * 80)
    print("PART 3: DEM NODATA DIAGNOSTIC POLICIES EVALUATION")
    print("=" * 80)

    val_events = cfg["validation_event_ids"]
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=val_events,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
    )

    model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    model.freeze()

    policies = [
        ("policy1_raw", "Existing raw behavior (NoData < -1000 untouched)"),
        ("policy2_replace_mean", "Replace NoData (< -1000 or NaN) with training DEM mean (normalized value = 0.0)"),
        ("policy3_exclude_dem_invalid", "Exclude DEM NoData pixels from evaluation mask"),
    ]

    policy_rows = []

    for pol_id, pol_desc in policies:
        print(f"\n--- Evaluating Policy: {pol_id} ({pol_desc}) ---")
        accum = EventMetricAccumulator()
        affected_pixel_counts = {eid: 0 for eid in val_events}
        total_eval_pixel_counts = {eid: 0 for eid in val_events}

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                eid = sample["event_id"]
                xA = sample["xA"].unsqueeze(0).clone().to(device)
                xB = sample["xB"].unsqueeze(0).clone().to(device)
                lbl = sample["label"].numpy()
                vld = sample["valid_mask"].numpy()

                # Read raw DEM to detect NoData
                with rasterio.open(sample["component_paths"]["MK0_DEM"]) as r:
                    raw_dem = r.read(1)

                is_nodata = np.isnan(raw_dem) | (raw_dem < -1000.0)
                affected_pixel_counts[eid] += int(np.count_nonzero(is_nodata & (vld == 1) & (lbl != 3)))
                total_eval_pixel_counts[eid] += int(np.count_nonzero((vld == 1) & (lbl != 3)))

                if pol_id == "policy2_replace_mean":
                    # Where raw DEM is nodata, normalized value becomes 0.0
                    nodata_tensor = torch.from_numpy(is_nodata).unsqueeze(0).to(device)
                    xA[:, 2, :, :][nodata_tensor] = 0.0
                    xB[:, 2, :, :][nodata_tensor] = 0.0

                out = model.predict(xA, xB)
                pred_c = out["predicted_class"].squeeze(0).cpu().numpy()

                eval_vld = vld.copy()
                if pol_id == "policy3_exclude_dem_invalid":
                    # Additionally exclude DEM-invalid pixels from evaluation
                    eval_vld[is_nodata] = 0

                counts = compute_confusion_counts(pred_c, lbl, eval_vld)
                accum.update([eid], torch.from_numpy(pred_c).unsqueeze(0), torch.from_numpy(lbl).unsqueeze(0), torch.from_numpy(eval_vld).unsqueeze(0))

        table = accum.compute_event_table()
        for row in table:
            eid = row["event_id"]
            aff_cnt = affected_pixel_counts[eid]
            tot_cnt = total_eval_pixel_counts[eid]
            aff_pct = (aff_cnt / tot_cnt * 100) if tot_cnt > 0 else 0.0
            row["policy"] = pol_id
            row["policy_description"] = pol_desc
            row["affected_eval_pixels"] = aff_cnt
            row["total_eval_pixels"] = tot_cnt
            row["affected_eval_pixel_pct"] = aff_pct
            policy_rows.append(row)
            print(
                f"  {pol_id:28s} | Event {eid:8d} | Affected: {aff_cnt:6d} ({aff_pct:5.2f}%) | "
                f"Prec={row['precision']:.4f}, Rec={row['recall']:.4f}, F1={row['f1']:.4f}, IoU={row['iou']:.4f}"
            )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(policy_rows[0].keys()))
        writer.writeheader()
        for r in policy_rows:
            writer.writerow(r)
    print(f"\nSaved NoData policy evaluation: {output_csv_path}")
    return policy_rows


def main():
    config_file = PROJECT_ROOT / "configs/baseline_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    audit_dir = PROJECT_ROOT / "outputs/day2_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exhaustive training DEM statistics
    train_dem_csv = audit_dir / "train_event_dem_statistics.csv"
    compute_all_event_dem_statistics(cfg["dataset_root"], train_dem_csv)

    # 2. Elevation counterfactual sweep for Event 559
    sweep_csv = audit_dir / "event559_elevation_sweep.csv"
    run_event559_elevation_sweep(cfg, sweep_csv)

    # 3. DEM NoData policy evaluation
    nodata_csv = audit_dir / "dem_nodata_policy_evaluation.csv"
    run_dem_nodata_policy_evaluation(cfg, nodata_csv)

    print("\n" + "=" * 80)
    print("ALL SUPPLEMENTAL AUDIT TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

