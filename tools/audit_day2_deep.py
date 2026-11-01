"""Comprehensive Day 2 Baseline Audit Script:
1. Complete data path trace & numpy hand-check for events 559, 1111003, and 1111008.
2. Preprocessing & statistical distribution audit across validation events.
3. Diagnostic comparisons on fixed subset:
   - Canonical pre2 vs pre1
   - DEM v1 stats vs DEM v2 stats vs No-DEM vs Unnormalized DEM
4. Failure pattern inspection and multi-panel diagnostic visualizations.
5. Saves CSV and figure artifacts under outputs/day2_audit/.
"""

import sys
import json
import csv
from pathlib import Path
from collections import Counter
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import compute_confusion_counts, compute_flood_metrics, EventMetricAccumulator


def run_comprehensive_audit():
    config_file = PROJECT_ROOT / "configs/baseline_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    audit_dir = PROJECT_ROOT / "outputs/day2_audit"
    fig_dir = audit_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running audit on device: {device}")

    # =========================================================================
    # STEP 2 & 3: AUDIT COMPLETE DATA PATH & INDEPENDENT NUMPY HAND-CHECK
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2 & 3: TRACING COMPLETE DATA PATH & NUMPY HAND-CHECK")
    print("=" * 80)

    # Load baseline dataset
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=cfg["validation_event_ids"],
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
    )

    # Load model
    model, _ = SNUNetWrapper.load_from_checkpoint(
        cfg["checkpoint_path"], in_channels=3, out_ch=3, base_channel=32, device=device
    )
    model.freeze()

    target_events = [559, 1111003, 1111008]
    data_path_records = []

    for eid in target_events:
        # Find sample that contains flood if possible, else first sample
        matching_indices = [i for i, s in enumerate(dataset.samples) if s["event_id"] == eid]
        selected_idx = None
        for i in matching_indices:
            sample = dataset[i]
            if (sample["label"].numpy() == 2).any():
                selected_idx = i
                break
        if selected_idx is None:
            selected_idx = matching_indices[0]

        sample = dataset[selected_idx]
        sid = sample["sample_id"]
        
        # Raw shapes and dtypes
        comp_paths = sample["component_paths"]
        with rasterio.open(comp_paths["SL1_IVV"]) as s:
            raw_sl1_vv = s.read(1)
            meta_sl1 = (raw_sl1_vv.shape, str(raw_sl1_vv.dtype))
        with rasterio.open(comp_paths["SL2_IVV"]) as s:
            raw_sl2_vv = s.read(1)
            meta_sl2 = (raw_sl2_vv.shape, str(raw_sl2_vv.dtype))
        with rasterio.open(comp_paths["MS1_IVV"]) as s:
            raw_ms1_vv = s.read(1)
            meta_ms1 = (raw_ms1_vv.shape, str(raw_ms1_vv.dtype))
        with rasterio.open(comp_paths["MK0_DEM"]) as s:
            raw_dem = s.read(1)
            meta_dem = (raw_dem.shape, str(raw_dem.dtype))
        with rasterio.open(comp_paths["MK0_MLU"]) as s:
            raw_mlu = s.read(1)
            meta_mlu = (raw_mlu.shape, str(raw_mlu.dtype))
        with rasterio.open(comp_paths["MK0_MNA"]) as s:
            raw_mna = s.read(1)
            meta_mna = (raw_mna.shape, str(raw_mna.dtype))

        # Forward pass
        xA = sample["xA"].unsqueeze(0).to(device)
        xB = sample["xB"].unsqueeze(0).to(device)
        out = model.predict(xA, xB)
        pred_class = out["predicted_class"].squeeze(0).cpu().numpy()
        label_np = sample["label"].numpy()
        valid_np = sample["valid_mask"].numpy()

        # Metrics via metric module
        conf_module = compute_confusion_counts(pred_class, label_np, valid_np)
        met_module = compute_flood_metrics(conf_module)

        # Independent pure NumPy hand-check
        hand_valid = (valid_np == 1) & (label_np != 3)
        hand_target = (label_np == 2) & hand_valid
        hand_pred = (pred_class == 2) & hand_valid

        hand_tp = int(np.count_nonzero(hand_pred & hand_target))
        hand_fp = int(np.count_nonzero(hand_pred & (~hand_target) & hand_valid))
        hand_fn = int(np.count_nonzero((~hand_pred) & hand_target))
        hand_tn = int(np.count_nonzero((~hand_pred) & (~hand_target) & hand_valid))

        # Verify exact hand-check equality
        assert conf_module["tp"] == hand_tp, f"Mismatch in TP: {conf_module['tp']} vs {hand_tp}"
        assert conf_module["fp"] == hand_fp, f"Mismatch in FP: {conf_module['fp']} vs {hand_fp}"
        assert conf_module["fn"] == hand_fn, f"Mismatch in FN: {conf_module['fn']} vs {hand_fn}"
        assert conf_module["tn"] == hand_tn, f"Mismatch in TN: {conf_module['tn']} vs {hand_tn}"

        record = {
            "event_id": eid,
            "sample_id": sid,
            "sample_dir": str(Path(comp_paths["MK0_MLU"]).parent),
            "pre2_file": Path(comp_paths["SL2_IVV"]).name,
            "pre1_file": Path(comp_paths["SL1_IVV"]).name,
            "post_file": Path(comp_paths["MS1_IVV"]).name,
            "dem_file": Path(comp_paths["MK0_DEM"]).name,
            "label_file": Path(comp_paths["MK0_MLU"]).name,
            "valid_file": Path(comp_paths["MK0_MNA"]).name,
            "raw_sar_shape_dtype": f"{meta_sl2[0]}_{meta_sl2[1]}",
            "raw_dem_shape_dtype": f"{meta_dem[0]}_{meta_dem[1]}",
            "tensor_xA_shape": list(sample["xA"].shape),
            "tensor_xB_shape": list(sample["xB"].shape),
            "label_values_present": sorted(list(np.unique(label_np))),
            "valid_mask_values_present": sorted(list(np.unique(valid_np))),
            "valid_pixels": conf_module["valid_pixels"],
            "gt_flood_pixels": conf_module["gt_flood_pixels"],
            "pred_flood_pixels": conf_module["pred_flood_pixels"],
            "tp": conf_module["tp"],
            "fp": conf_module["fp"],
            "fn": conf_module["fn"],
            "precision": met_module["precision"],
            "recall": met_module["recall"],
            "f1": met_module["f1"],
            "iou": met_module["iou"],
            "hand_check_passed": True,
        }
        data_path_records.append(record)
        print(f"\nTrace for Event {eid} Sample {sid[:12]}:")
        print(f"  Pre2: {record['pre2_file']} | Pre1: {record['pre1_file']}")
        print(f"  Post: {record['post_file']} | DEM: {record['dem_file']}")
        print(f"  GT Flood Pixels: {record['gt_flood_pixels']} | Pred Flood: {record['pred_flood_pixels']}")
        print(f"  TP={record['tp']}, FP={record['fp']}, FN={record['fn']}")
        print(f"  Precision={record['precision']:.4f}, Recall={record['recall']:.4f}, F1={record['f1']:.4f}, IoU={record['iou']:.4f}")
        print(f"  Hand-check assertion: MATCHED EXACTLY.")

    # =========================================================================
    # STEP 4: PREPROCESSING & DATASET STATISTICAL AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: PREPROCESSING & SENSOR STATISTICAL AUDIT ACROSS EVENTS")
    print("=" * 80)

    val_events = [279, 437, 520, 559, 1111003, 1111008]
    event_stats_rows = []

    for eid in val_events:
        event_indices = [i for i, s in enumerate(dataset.samples) if s["event_id"] == eid]
        
        # Accumulators
        all_sl1_vv = []
        all_sl1_vh = []
        all_sl2_vv = []
        all_sl2_vh = []
        all_ms1_vv = []
        all_ms1_vh = []
        all_dem = []
        label_counter = Counter()
        valid_counter = Counter()
        total_valid_pixels = 0
        total_gt_flood = 0
        total_pred_flood = 0

        # Diagnostic sample selection for descriptive statistics
        is_partial = len(event_indices) > 200
        selection_rule = "first_100_samples" if is_partial else "all_samples"
        eval_indices = event_indices[:100] if is_partial else event_indices

        if is_partial:
            print("\n" + "!" * 80)
            print("WARNING: PARTIAL DIAGNOSTIC ONLY — MUST NOT BE REPORTED AS FULL EVENT METRICS")
            print(f"Event {eid}: Processed {len(eval_indices)} of {len(event_indices)} samples for descriptive statistics.")
            print("Selection rule:", selection_rule, "| allow_partial=True")
            print("!" * 80 + "\n")

        for idx in eval_indices:
            s = dataset[idx]
            raw = s["raw_stats"]
            all_sl1_vv.append(raw["sl1_vv"])
            all_sl1_vh.append(raw["sl1_vh"])
            all_sl2_vv.append(raw["sl2_vv"])
            all_sl2_vh.append(raw["sl2_vh"])
            all_ms1_vv.append(raw["ms1_vv"])
            all_ms1_vh.append(raw["ms1_vh"])
            all_dem.append(raw["dem"])

            lbl = s["label"].numpy()
            vld = s["valid_mask"].numpy()
            for val, cnt in Counter(lbl.flatten()).items():
                label_counter[val] += cnt
            for val, cnt in Counter(vld.flatten()).items():
                valid_counter[val] += cnt

            d_val = (vld == 1) & (lbl != 3)
            total_valid_pixels += int(np.count_nonzero(d_val))
            total_gt_flood += int(np.count_nonzero((lbl == 2) & d_val))

        def agg_stats(stats_list):
            mins = [x[0] for x in stats_list]
            maxs = [x[1] for x in stats_list]
            means = [x[2] for x in stats_list]
            stds = [x[3] for x in stats_list]
            return float(np.min(mins)), float(np.max(maxs)), float(np.mean(means)), float(np.mean(stds))

        sl1_vv_stat = agg_stats(all_sl1_vv)
        sl1_vh_stat = agg_stats(all_sl1_vh)
        sl2_vv_stat = agg_stats(all_sl2_vv)
        sl2_vh_stat = agg_stats(all_sl2_vh)
        ms1_vv_stat = agg_stats(all_ms1_vv)
        ms1_vh_stat = agg_stats(all_ms1_vh)
        dem_stat = agg_stats(all_dem)

        row = {
            "event_id": eid,
            "expected_event_samples": len(event_indices),
            "processed_statistics_samples": len(eval_indices),
            "is_partial_statistics": is_partial,
            "selection_rule": selection_rule,
            "allow_partial": True,
            "table_description": "descriptive_partial_statistics_not_evaluation_metrics",
            "sl1_vv_mean": sl1_vv_stat[2],
            "sl1_vh_mean": sl1_vh_stat[2],
            "sl2_vv_mean": sl2_vv_stat[2],
            "sl2_vh_mean": sl2_vh_stat[2],
            "ms1_vv_mean": ms1_vv_stat[2],
            "ms1_vh_mean": ms1_vh_stat[2],
            "dem_min": dem_stat[0],
            "dem_max": dem_stat[1],
            "dem_mean": dem_stat[2],
            "dem_std": dem_stat[3],
            "raw_class_0_pixels": label_counter[0],
            "raw_class_1_pixels": label_counter[1],
            "raw_class_2_flood_pixels": label_counter[2],
            "raw_class_3_invalid_pixels": label_counter[3],
            "valid_mask_0_pixels": valid_counter[0],
            "valid_mask_1_pixels": valid_counter[1],
            "double_valid_pixels": total_valid_pixels,
            "gt_flood_prevalence_pct": (total_gt_flood / total_valid_pixels * 100) if total_valid_pixels > 0 else 0.0,
        }
        event_stats_rows.append(row)
        print(f"Event {eid:8d}: Expected={row['expected_event_samples']:4d} | Processed={row['processed_statistics_samples']:4d} | Partial={row['is_partial_statistics']} | DEM mean={row['dem_mean']:8.2f} | Flood Prev={row['gt_flood_prevalence_pct']:.2f}%")

    stats_csv_path = audit_dir / "descriptive_partial_statistics.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_stats_rows[0].keys()))
        writer.writeheader()
        for r in event_stats_rows:
            writer.writerow(r)
    print(f"Saved descriptive partial statistics CSV: {stats_csv_path}")

    # Also save to legacy path with updated schema to maintain backward compatibility
    legacy_csv_path = audit_dir / "preprocessing_statistics.csv"
    with open(legacy_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_stats_rows[0].keys()))
        writer.writeheader()
        for r in event_stats_rows:
            writer.writerow(r)

    # =========================================================================
    # STEP 5: LIMITED DIAGNOSTIC COMPARISONS ON FIXED VALIDATION SUBSET
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: LIMITED DIAGNOSTIC COMPARISONS ON FIXED VALIDATION SUBSET")
    print("=" * 80)

    # Fixed subset: all samples of 559 (42), 1111003 (19), 520 (92), 1111008 (123)
    # Total samples = 276
    diagnostic_events = [559, 1111003, 520, 1111008]
    subset_samples = [s for s in dataset.samples if s["event_id"] in diagnostic_events]
    print(f"Diagnostic fixed subset: {len(subset_samples)} samples across events {diagnostic_events}")

    diagnostic_configs = [
        ("baseline_pre2_dem_v1", "pre_event_2", 67.0293, 1765.0062, "dem"),
        ("diag_pre1_dem_v1", "pre_event_1", 67.0293, 1765.0062, "dem"),
        ("diag_pre2_dem_v2", "pre_event_2", 93.4313, 1410.8382, "dem"),
        ("diag_pre1_dem_v2", "pre_event_1", 93.4313, 1410.8382, "dem"),
        ("diag_pre2_no_dem", "pre_event_2", 0.0, 1.0, "zero"),
        ("diag_pre1_no_dem", "pre_event_1", 0.0, 1.0, "zero"),
    ]

    diagnostic_results_rows = []

    for name, pre_choice, d_mean, d_std, dem_mode in diagnostic_configs:
        print(f"\n--- Running diagnostic: {name} ---")
        
        diag_dataset = KuroValidationDataset(
            dataset_root=cfg["dataset_root"],
            manifest_path=cfg["manifest_path"],
            allowed_events=diagnostic_events,
            sealed_test_events=cfg["sealed_test_event_ids"],
            canonical_preevent=pre_choice,
            sar_clamp_max=cfg["sar_clamping"]["max"],
            sar_mean=cfg["sar_normalization"]["mean"],
            sar_std=cfg["sar_normalization"]["std"],
            dem_mean=d_mean,
            dem_std=d_std,
        )

        diag_loader = DataLoader(
            diag_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=(device == "cuda")
        )

        diag_accum = EventMetricAccumulator()

        with torch.no_grad():
            for batch in diag_loader:
                eids = batch["event_id"].tolist()
                xA = batch["xA"].to(device)
                xB = batch["xB"].to(device)
                lbls = batch["label"]
                vlds = batch["valid_mask"]

                if dem_mode == "zero":
                    xA[:, 2, :, :] = 0.0
                    xB[:, 2, :, :] = 0.0

                out = model.predict(xA, xB)
                diag_accum.update(eids, out["predicted_class"], lbls, vlds)

        diag_table = diag_accum.compute_event_table()
        for row in diag_table:
            row["diagnostic_config"] = name
            row["canonical_preevent"] = pre_choice
            row["dem_mode"] = dem_mode
            row["dem_mean"] = d_mean
            row["dem_std"] = d_std
            diagnostic_results_rows.append(row)
            print(f"  {name:22s} | Event {row['event_id']:8d} | Prec={row['precision']:.4f} | Rec={row['recall']:.4f} | F1={row['f1']:.4f} | IoU={row['iou']:.4f}")

    sample_diag_csv_path = audit_dir / "sample_diagnostics.csv"
    with open(sample_diag_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostic_results_rows[0].keys()))
        writer.writeheader()
        for r in diagnostic_results_rows:
            writer.writerow(r)
    print(f"\nSaved diagnostic comparisons CSV: {sample_diag_csv_path}")

    # =========================================================================
    # STEP 6: EVENT-LEVEL FAILURE PATTERNS & MULTI-PANEL VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EVENT-LEVEL FAILURE PATTERNS & MULTI-PANEL VISUALIZATIONS")
    print("=" * 80)

    # For events 559, 1111003, and 1111008
    # Collect sample-level metrics and find representative cases:
    # 1. Highest False Negatives (missed flood)
    # 2. Highest False Positives (hallucinated flood)
    # 3. Median F1 failure
    # 4. Successful sample (highest F1)
    # 5. Non-flood agreement
    cmap_label = ListedColormap(["black", "deepskyblue", "magenta", "gray"])

    def norm_disp(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr)
        p2, p98 = np.percentile(finite, [2, 98])
        if p98 <= p2:
            return np.zeros_like(arr)
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    event_summary_rows = []

    for eid in target_events:
        matching_indices = [i for i, s in enumerate(dataset.samples) if s["event_id"] == eid]
        sample_metrics = []

        for idx in matching_indices:
            s = dataset[idx]
            xA = s["xA"].unsqueeze(0).to(device)
            xB = s["xB"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.predict(xA, xB)
            p_class = out["predicted_class"].squeeze(0).cpu().numpy()
            f_prob = out["flood_probability"].squeeze(0).cpu().numpy()
            lbl = s["label"].numpy()
            vld = s["valid_mask"].numpy()

            conf = compute_confusion_counts(p_class, lbl, vld)
            met = compute_flood_metrics(conf)

            sample_metrics.append({
                "dataset_idx": idx,
                "sample_id": s["sample_id"],
                "event_id": eid,
                "tp": conf["tp"],
                "fp": conf["fp"],
                "fn": conf["fn"],
                "tn": conf["tn"],
                "valid_pixels": conf["valid_pixels"],
                "gt_flood_pixels": conf["gt_flood_pixels"],
                "pred_flood_pixels": conf["pred_flood_pixels"],
                "precision": met["precision"],
                "recall": met["recall"],
                "f1": met["f1"],
                "iou": met["iou"],
                "gt_has_flood": conf["gt_flood_pixels"] > 0,
                "pred_has_flood": conf["pred_flood_pixels"] > 0,
            })

        # Summary statistics
        total_samples = len(sample_metrics)
        samples_with_gt_flood = sum(1 for m in sample_metrics if m["gt_has_flood"])
        samples_with_pred_flood = sum(1 for m in sample_metrics if m["pred_has_flood"])
        gt_flood_no_pred = sum(1 for m in sample_metrics if m["gt_has_flood"] and not m["pred_has_flood"])
        pred_flood_no_gt = sum(1 for m in sample_metrics if m["pred_has_flood"] and not m["gt_has_flood"])
        both_flood = sum(1 for m in sample_metrics if m["gt_has_flood"] and m["pred_has_flood"])
        neither_flood = sum(1 for m in sample_metrics if not m["gt_has_flood"] and not m["pred_has_flood"])

        sum_tp = sum(m["tp"] for m in sample_metrics)
        sum_fp = sum(m["fp"] for m in sample_metrics)
        sum_fn = sum(m["fn"] for m in sample_metrics)
        sum_tn = sum(m["tn"] for m in sample_metrics)
        sum_valid = sum(m["valid_pixels"] for m in sample_metrics)
        sum_gt_flood = sum(m["gt_flood_pixels"] for m in sample_metrics)
        sum_pred_flood = sum(m["pred_flood_pixels"] for m in sample_metrics)

        ev_met = compute_flood_metrics({"tp": sum_tp, "fp": sum_fp, "fn": sum_fn, "tn": sum_tn})

        summary_row = {
            "event_id": eid,
            "total_samples": total_samples,
            "valid_pixels": sum_valid,
            "total_gt_flood_pixels": sum_gt_flood,
            "total_pred_flood_pixels": sum_pred_flood,
            "gt_prevalence_pct": (sum_gt_flood / sum_valid * 100) if sum_valid > 0 else 0.0,
            "pred_prevalence_pct": (sum_pred_flood / sum_valid * 100) if sum_valid > 0 else 0.0,
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "tn": sum_tn,
            "precision": ev_met["precision"],
            "recall": ev_met["recall"],
            "f1": ev_met["f1"],
            "iou": ev_met["iou"],
            "samples_with_gt_flood": samples_with_gt_flood,
            "samples_with_pred_flood": samples_with_pred_flood,
            "samples_gt_flood_zero_pred": gt_flood_no_pred,
            "samples_pred_flood_zero_gt": pred_flood_no_gt,
            "samples_both_flood": both_flood,
            "samples_neither_flood": neither_flood,
        }
        event_summary_rows.append(summary_row)

        print(f"\nFailure Analysis for Event {eid}:")
        print(f"  Samples: {total_samples} | GT Flood Pixels: {sum_gt_flood} | Pred Flood Pixels: {sum_pred_flood}")
        print(f"  Samples with GT Flood: {samples_with_gt_flood} | Samples with Pred Flood: {samples_with_pred_flood}")
        print(f"  GT Flood but Pred=0 (Complete Misses): {gt_flood_no_pred} / {samples_with_gt_flood}")
        print(f"  Pred Flood but GT=0 (Pure False Alarms): {pred_flood_no_gt}")
        print(f"  Precision: {ev_met['precision']:.4f} | Recall: {ev_met['recall']:.4f} | F1: {ev_met['f1']:.4f} | IoU: {ev_met['iou']:.4f}")

        # Select 5 representative samples using deterministic rules:
        # Case A: Highest FN (worst missed flood)
        sorted_by_fn = sorted(sample_metrics, key=lambda m: m["fn"], reverse=True)
        case_high_fn = sorted_by_fn[0]

        # Case B: Highest FP (worst hallucination)
        sorted_by_fp = sorted(sample_metrics, key=lambda m: m["fp"], reverse=True)
        case_high_fp = sorted_by_fp[0]

        # Case C: Median F1 among samples with GT flood
        gt_samples = [m for m in sample_metrics if m["gt_has_flood"]]
        if gt_samples:
            sorted_by_f1 = sorted(gt_samples, key=lambda m: m["f1"])
            case_median = sorted_by_f1[len(sorted_by_f1) // 2]
            case_best = sorted_by_f1[-1]
        else:
            case_median = sample_metrics[len(sample_metrics) // 2]
            case_best = sample_metrics[0]

        # Case D: Non-flood sample
        non_flood = [m for m in sample_metrics if not m["gt_has_flood"] and not m["pred_has_flood"]]
        case_non_flood = non_flood[0] if non_flood else sample_metrics[-1]

        cases = [
            ("worst_missed_flood", case_high_fn),
            ("worst_hallucination", case_high_fp),
            ("median_failure", case_median),
            ("best_sample", case_best),
            ("non_flood_reference", case_non_flood),
        ]

        # Render each case
        for label_tag, c in cases:
            s_obj = dataset[c["dataset_idx"]]
            xA = s_obj["xA"].unsqueeze(0).to(device)
            xB = s_obj["xB"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.predict(xA, xB)
            pred_c = out["predicted_class"].squeeze(0).cpu().numpy()
            flood_p = out["flood_probability"].squeeze(0).cpu().numpy()
            lbl = s_obj["label"].numpy()
            vld = s_obj["valid_mask"].numpy()
            d_vld = (vld == 1) & (lbl != 3)

            # Load raw rasters
            with rasterio.open(s_obj["component_paths"]["SL2_IVV"]) as r:
                disp_pre_vv = r.read(1)
            with rasterio.open(s_obj["component_paths"]["SL2_IVH"]) as r:
                disp_pre_vh = r.read(1)
            with rasterio.open(s_obj["component_paths"]["MS1_IVV"]) as r:
                disp_post_vv = r.read(1)
            with rasterio.open(s_obj["component_paths"]["MS1_IVH"]) as r:
                disp_post_vh = r.read(1)
            with rasterio.open(s_obj["component_paths"]["MK0_DEM"]) as r:
                disp_dem = r.read(1)

            # Binary targets
            bin_gt = ((lbl == 2) & d_vld).astype(np.uint8)
            bin_pred = ((pred_c == 2) & d_vld).astype(np.uint8)

            # Overlay: TP = Green, FP = Red, FN = Blue
            overlay = np.zeros((224, 224, 3), dtype=np.float32)
            bg_sar = norm_disp(disp_post_vv)
            for ch in range(3):
                overlay[:, :, ch] = bg_sar * 0.7
            tp_mask = (bin_pred == 1) & (bin_gt == 1)
            fp_mask = (bin_pred == 1) & (bin_gt == 0)
            fn_mask = (bin_pred == 0) & (bin_gt == 1)
            overlay[tp_mask] = [0.0, 1.0, 0.0]  # Green TP
            overlay[fp_mask] = [1.0, 0.0, 0.0]  # Red FP
            overlay[fn_mask] = [0.0, 0.4, 1.0]  # Blue FN

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))

            # Row 1
            axes[0, 0].imshow(norm_disp(disp_pre_vv), cmap="gray")
            axes[0, 0].set_title(f"Pre-event VV (SL2)\nEvent {eid}")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(norm_disp(disp_pre_vh), cmap="gray")
            axes[0, 1].set_title("Pre-event VH (SL2)")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(norm_disp(disp_post_vv), cmap="gray")
            axes[0, 2].set_title("Post-event VV (MS1)")
            axes[0, 2].axis("off")

            axes[0, 3].imshow(norm_disp(disp_post_vh), cmap="gray")
            axes[0, 3].set_title("Post-event VH (MS1)")
            axes[0, 3].axis("off")

            # Row 2
            axes[1, 0].imshow(lbl, cmap=cmap_label, vmin=0, vmax=3)
            axes[1, 0].set_title("GT 3-Class Label\n(0:NW, 1:PW, 2:Flood, 3:Inv)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(vld, cmap="gray", vmin=0, vmax=1)
            axes[1, 1].set_title("Valid Mask (MK0_MNA)")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(bin_gt, cmap="gray", vmin=0, vmax=1)
            axes[1, 2].set_title(f"Binary GT Flood\n(Pixels={c['gt_flood_pixels']})")
            axes[1, 2].axis("off")

            axes[1, 3].imshow(disp_dem, cmap="terrain")
            axes[1, 3].set_title(f"DEM Elevation\n(Mean={disp_dem.mean():.1f}m)")
            axes[1, 3].axis("off")

            # Row 3
            axes[2, 0].imshow(pred_c, cmap=cmap_label, vmin=0, vmax=3)
            axes[2, 0].set_title("Predicted 3-Class Label")
            axes[2, 0].axis("off")

            axes[2, 1].imshow(bin_pred, cmap="gray", vmin=0, vmax=1)
            axes[2, 1].set_title(f"Binary Pred Flood\n(Pixels={c['pred_flood_pixels']})")
            axes[2, 1].axis("off")

            im_p = axes[2, 2].imshow(flood_p, cmap="magma", vmin=0.0, vmax=1.0)
            axes[2, 2].set_title("Flood Probability")
            axes[2, 2].axis("off")
            fig.colorbar(im_p, ax=axes[2, 2], fraction=0.046, pad=0.04)

            axes[2, 3].imshow(overlay)
            axes[2, 3].set_title(f"Overlay Error\nGreen:TP, Red:FP, Blue:FN\n(F1={c['f1']:.3f}, IoU={c['iou']:.3f})")
            axes[2, 3].axis("off")

            fig.suptitle(f"Event {eid} Diagnostic: {label_tag} (Sample: {c['sample_id'][:12]}...)", fontsize=14, fontweight="bold")
            plt.tight_layout()

            save_fpath = fig_dir / f"event_{eid}_{label_tag}_{c['sample_id'][:8]}.png"
            plt.savefig(save_fpath, dpi=160, bbox_inches="tight")
            plt.close(fig)

    ev_sum_csv_path = audit_dir / "event_summary.csv"
    with open(ev_sum_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_summary_rows[0].keys()))
        writer.writeheader()
        for r in event_summary_rows:
            writer.writerow(r)
    print(f"\nSaved event summary CSV: {ev_sum_csv_path}")

    print("\n" + "=" * 80)
    print("ALL AUDIT CALCULATIONS AND DIAGNOSTICS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_audit()

