#!/usr/bin/env python3
"""ChangeGuard Week 14 Offline Class Error Diagnosis.

Protocol: Week 14 Offline Error Diagnosis
Scope: Strictly 6 KuroSiwo development events: [279, 437, 520, 559, 1111003, 1111008]
Quarantined: 10 sealed test events: [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]

Reuses saved primary predictions for:
1. source_only (Week 12)
2. matched_bn_no_update_v1 (Week 12 BN diagnostic)
3. matched_tent_pre_update_v1 (Week 12)
4. C1_consistency_no_conf (Week 12)
5. C2_consistency_conf90 (Week 12)
6. C3_anchored_consistency_v1 (Week 13)

For each event and condition:
- Computes full 3x3 confusion matrix over sensory-valid and label-valid pixels:
    valid = (MK0_MNA == 1) & (MK0_MLU != 3)
- Verifies sum(C_3x3) == valid_pixels
- Distinguishes Land -> Flood vs Permanent-Water -> Flood false positives
- Distinguishes Flood -> Land vs Flood -> Permanent-Water false negatives
- Reports full-precision counts, rates, and explicit denominators
- Compares each condition with source_only and with matched_bn_no_update_v1
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.evaluation_dataset import KuroEvaluationDataset

DEV_EVENTS = [279, 437, 520, 559, 1111003, 1111008]
FORBIDDEN_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}

WEEK12_DIR = REPO_ROOT / "outputs/week12_ablation/approved_dev6_s0_v1"
BN_DIAG_DIR = REPO_ROOT / "outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1"
WEEK13_DIR = REPO_ROOT / "outputs/week13_source_anchor/approved_dev6_s0_v1"
OUTPUT_DIR = REPO_ROOT / "outputs/week14_offline_diagnosis"

CONDITIONS = [
    "source_only",
    "matched_bn_no_update_v1",
    "matched_tent_pre_update_v1",
    "C1_consistency_no_conf",
    "C2_consistency_conf90",
    "C3_anchored_consistency_v1",
]

# Ground Truth & Prediction Class Definitions
CLASS_NAMES = {
    0: "land",
    1: "permanent_water",
    2: "flood",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_3x3_confusion(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Computes full 3x3 confusion matrix C[gt, pred] on valid pixels."""
    v_pred = pred[valid_mask]
    v_gt = gt[valid_mask]
    cm = np.zeros((3, 3), dtype=np.int64)
    for g in range(3):
        mask_g = (v_gt == g)
        if not np.any(mask_g):
            continue
        p_g = v_pred[mask_g]
        for p in range(3):
            cm[g, p] = int(np.sum(p_g == p))
    return cm


def run_week14_diagnosis():
    print("================================================================================")
    print(" ChangeGuard Week 14 Offline Class Error Diagnosis")
    print("================================================================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Prediction Directories
    pred_paths_by_cond: Dict[str, Dict[int, Path]] = {c: {} for c in CONDITIONS}
    for ev in DEV_EVENTS:
        pred_paths_by_cond["source_only"][ev] = WEEK12_DIR / f"source_only/event_{ev}/frozen_predictions.npz"
        pred_paths_by_cond["matched_bn_no_update_v1"][ev] = BN_DIAG_DIR / f"matched_bn_no_update_v1/event_{ev}/frozen_predictions.npz"
        pred_paths_by_cond["matched_tent_pre_update_v1"][ev] = WEEK12_DIR / f"matched_tent_pre_update_v1/event_{ev}/frozen_predictions.npz"
        pred_paths_by_cond["C1_consistency_no_conf"][ev] = WEEK12_DIR / f"C1_consistency_no_conf/event_{ev}/frozen_predictions.npz"
        pred_paths_by_cond["C2_consistency_conf90"][ev] = WEEK12_DIR / f"C2_consistency_conf90/event_{ev}/frozen_predictions.npz"
        pred_paths_by_cond["C3_anchored_consistency_v1"][ev] = WEEK13_DIR / f"C3_anchored_consistency_v1/event_{ev}/frozen_predictions.npz"

    # Verify all prediction files exist and load their sample keys
    sample_keys_by_cond_event: Dict[str, Dict[int, List[str]]] = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for ev in DEV_EVENTS:
            p = pred_paths_by_cond[c][ev]
            assert p.is_file(), f"Missing prediction file: {p}"
            with np.load(p) as npz:
                keys = sorted(list(npz.files))
                sample_keys_by_cond_event[c][ev] = keys

    # Verify identical sample keys across all conditions
    for ev in DEV_EVENTS:
        ref_keys = sample_keys_by_cond_event["source_only"][ev]
        for c in CONDITIONS:
            assert sample_keys_by_cond_event[c][ev] == ref_keys, f"Sample keys mismatch in {c} event {ev}"
    print(f"✓ Verified identical sample sets across all 6 conditions for all {len(DEV_EVENTS)} dev events.")

    # 2. Initialize Offline Dataset with GT
    # Strictly authorized development events only
    manifest_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"
    dataset_root = REPO_ROOT.parent / "datasets/KuroSiwo"

    dataset = KuroEvaluationDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        allowed_events=DEV_EVENTS,
        load_labels=True,
    )
    print(f"✓ KuroEvaluationDataset loaded: {len(dataset)} samples across allowed development events.")

    # Index dataset samples by sample_id
    dataset_idx_by_sid: Dict[str, int] = {}
    for idx, item in enumerate(dataset.samples):
        sid = item["sample_id"]
        dataset_idx_by_sid[sid] = idx

    # Verify that all cached prediction sample IDs exist in dataset
    all_cached_sids = set()
    for ev in DEV_EVENTS:
        for sid in sample_keys_by_cond_event["source_only"][ev]:
            assert sid in dataset_idx_by_sid, f"Cached sample ID {sid} not found in offline dataset!"
            all_cached_sids.add(sid)
    assert len(all_cached_sids) == 2826
    print(f"✓ All 2,826 cached sample IDs matched in offline ground-truth dataset.")

    # 3. Compute 3x3 Confusion Matrices
    print("\nProcessing full 3x3 confusion matrices across all conditions and events...")
    # cm_dict[cond][event] -> 3x3 np.ndarray
    cm_by_cond_event: Dict[str, Dict[int, np.ndarray]] = {c: {ev: np.zeros((3, 3), dtype=np.int64) for ev in DEV_EVENTS} for c in CONDITIONS}
    valid_pixels_by_event: Dict[int, int] = {ev: 0 for ev in DEV_EVENTS}

    # Cache predictions into memory by condition and event
    preds_cache: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for ev in DEV_EVENTS:
            with np.load(pred_paths_by_cond[c][ev]) as npz:
                preds_cache[c][ev] = {k: npz[k] for k in npz.files}

    # Iterate through each event and sample once to read GT and evaluate all conditions
    for ev in DEV_EVENTS:
        sids = sample_keys_by_cond_event["source_only"][ev]
        ev_valid_px = 0
        for sid in sids:
            d_idx = dataset_idx_by_sid[sid]
            rec = dataset[d_idx]
            gt = rec["label"].numpy()
            mna = rec["valid_mask"].numpy()

            valid = (mna == 1) & (gt != 3)
            v_count = int(np.sum(valid))
            ev_valid_px += v_count

            for c in CONDITIONS:
                pred = preds_cache[c][ev][sid]
                # Assert 3-class predictions
                u_pred = np.unique(pred)
                assert np.all(np.isin(u_pred, [0, 1, 2])), f"Non 3-class prediction in {c} {sid}: {u_pred}"

                cm_sample = compute_3x3_confusion(pred, gt, valid)
                cm_by_cond_event[c][ev] += cm_sample

        valid_pixels_by_event[ev] = ev_valid_px
        # Check conservation for each condition
        for c in CONDITIONS:
            cm_sum = int(np.sum(cm_by_cond_event[c][ev]))
            assert cm_sum == ev_valid_px, f"3x3 sum {cm_sum} != valid pixels {ev_valid_px} for {c} event {ev}"

        print(f"  Event {ev} ({len(sids)} samples): {ev_valid_px:,} valid pixels verified.")

    total_valid_pixels = sum(valid_pixels_by_event.values())
    assert total_valid_pixels == 135859080, f"Total valid pixels {total_valid_pixels} != 135,859,080"
    print(f"\n✓ Confusion matrix cell sums strictly equal valid pixels across all 135,859,080 pixels.")

    # 4. Generate Tables and Detailed Metrics
    # (A) Full 3x3 Table per event and condition
    cm_3x3_rows = []
    for ev in DEV_EVENTS:
        for c in CONDITIONS:
            cm = cm_by_cond_event[c][ev]
            cm_3x3_rows.append({
                "event_id": ev,
                "condition_id": c,
                "valid_pixels": valid_pixels_by_event[ev],
                "gt_land_total": int(np.sum(cm[0, :])),
                "gt_permwater_total": int(np.sum(cm[1, :])),
                "gt_flood_total": int(np.sum(cm[2, :])),
                "c00_land_to_land": int(cm[0, 0]),
                "c01_land_to_permwater": int(cm[0, 1]),
                "c02_land_to_flood_FP": int(cm[0, 2]),
                "c10_permwater_to_land": int(cm[1, 0]),
                "c11_permwater_to_permwater": int(cm[1, 1]),
                "c12_permwater_to_flood_FP": int(cm[1, 2]),
                "c20_flood_to_land_FN": int(cm[2, 0]),
                "c21_flood_to_permwater_FN": int(cm[2, 1]),
                "c22_flood_to_flood_TP": int(cm[2, 2]),
            })

    # Summary 3x3 table (pooled across all 6 events)
    for c in CONDITIONS:
        cm_tot = sum(cm_by_cond_event[c][ev] for ev in DEV_EVENTS)
        cm_3x3_rows.append({
            "event_id": "ALL",
            "condition_id": c,
            "valid_pixels": total_valid_pixels,
            "gt_land_total": int(np.sum(cm_tot[0, :])),
            "gt_permwater_total": int(np.sum(cm_tot[1, :])),
            "gt_flood_total": int(np.sum(cm_tot[2, :])),
            "c00_land_to_land": int(cm_tot[0, 0]),
            "c01_land_to_permwater": int(cm_tot[0, 1]),
            "c02_land_to_flood_FP": int(cm_tot[0, 2]),
            "c10_permwater_to_land": int(cm_tot[1, 0]),
            "c11_permwater_to_permwater": int(cm_tot[1, 1]),
            "c12_permwater_to_flood_FP": int(cm_tot[1, 2]),
            "c20_flood_to_land_FN": int(cm_tot[2, 0]),
            "c21_flood_to_permwater_FN": int(cm_tot[2, 1]),
            "c22_flood_to_flood_TP": int(cm_tot[2, 2]),
        })

    with open(OUTPUT_DIR / "class_confusion_3x3.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cm_3x3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cm_3x3_rows)

    # (B) Error Breakdown Table (False Positives and False Negatives Dissected)
    error_breakdown_rows = []
    events_with_all = DEV_EVENTS + ["ALL"]
    for ev in events_with_all:
        for c in CONDITIONS:
            if ev == "ALL":
                cm = sum(cm_by_cond_event[c][e] for e in DEV_EVENTS)
                n_valid = total_valid_pixels
            else:
                cm = cm_by_cond_event[c][ev]
                n_valid = valid_pixels_by_event[ev]

            gt_land = int(np.sum(cm[0, :]))
            gt_perm = int(np.sum(cm[1, :]))
            gt_flood = int(np.sum(cm[2, :]))

            # False Positives
            fp_land = int(cm[0, 2])
            fp_perm = int(cm[1, 2])
            fp_total = fp_land + fp_perm

            # False Negatives
            fn_land = int(cm[2, 0])
            fn_perm = int(cm[2, 1])
            fn_total = fn_land + fn_perm

            # True Positives & True Negatives
            tp = int(cm[2, 2])
            tn = int(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])

            # Assert binary consistency
            assert (tp + fp_total + fn_total + tn) == n_valid

            # Precision, Recall, F1
            prec = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
            rec = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            # Error Rates with Explicit Denominators
            # FP rates
            land_fa_rate = fp_land / gt_land if gt_land > 0 else 0.0  # Denominator: GT Land
            perm_fa_rate = fp_perm / gt_perm if gt_perm > 0 else 0.0  # Denominator: GT Permanent Water
            share_fp_from_land = fp_land / fp_total if fp_total > 0 else 0.0  # Denominator: Total FP
            share_fp_from_perm = fp_perm / fp_total if fp_total > 0 else 0.0  # Denominator: Total FP

            # FN rates
            flood_miss_to_land_rate = fn_land / gt_flood if gt_flood > 0 else 0.0  # Denominator: GT Flood
            flood_miss_to_perm_rate = fn_perm / gt_flood if gt_flood > 0 else 0.0  # Denominator: GT Flood
            share_fn_to_land = fn_land / fn_total if fn_total > 0 else 0.0  # Denominator: Total FN
            share_fn_to_perm = fn_perm / fn_total if fn_total > 0 else 0.0  # Denominator: Total FN

            error_breakdown_rows.append({
                "event_id": ev,
                "condition_id": c,
                "valid_pixels": n_valid,
                "gt_land": gt_land,
                "gt_permanent_water": gt_perm,
                "gt_flood": gt_flood,
                "tp": tp,
                "fp_total": fp_total,
                "fn_total": fn_total,
                "tn": tn,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "fp_from_land_count": fp_land,
                "fp_from_permwater_count": fp_perm,
                "fn_miss_to_land_count": fn_land,
                "fn_miss_to_permwater_count": fn_perm,
                "land_false_alarm_rate (denom=gt_land)": land_fa_rate,
                "permwater_false_alarm_rate (denom=gt_perm)": perm_fa_rate,
                "fp_share_from_land (denom=fp_total)": share_fp_from_land,
                "fp_share_from_permwater (denom=fp_total)": share_fp_from_perm,
                "flood_miss_rate_to_land (denom=gt_flood)": flood_miss_to_land_rate,
                "flood_miss_rate_to_permwater (denom=gt_flood)": flood_miss_to_perm_rate,
                "fn_share_to_land (denom=fn_total)": share_fn_to_land,
                "fn_share_to_permwater (denom=fn_total)": share_fn_to_perm,
            })

    with open(OUTPUT_DIR / "error_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_breakdown_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_breakdown_rows)

    # (C) Comparative Shifts Table: condition vs source_only and condition vs matched_bn_no_update_v1
    comparative_rows = []
    for ev in events_with_all:
        row_src = next(r for r in error_breakdown_rows if r["event_id"] == ev and r["condition_id"] == "source_only")
        row_bn = next(r for r in error_breakdown_rows if r["event_id"] == ev and r["condition_id"] == "matched_bn_no_update_v1")

        for c in CONDITIONS:
            r = next(item for item in error_breakdown_rows if item["event_id"] == ev and item["condition_id"] == c)

            # Versus Source
            diff_f1_src = r["f1"] - row_src["f1"]
            diff_fp_land_src = r["fp_from_land_count"] - row_src["fp_from_land_count"]
            diff_fp_perm_src = r["fp_from_permwater_count"] - row_src["fp_from_permwater_count"]
            diff_fn_land_src = r["fn_miss_to_land_count"] - row_src["fn_miss_to_land_count"]
            diff_fn_perm_src = r["fn_miss_to_permwater_count"] - row_src["fn_miss_to_permwater_count"]

            # Versus BN No-Update
            diff_f1_bn = r["f1"] - row_bn["f1"]
            diff_fp_land_bn = r["fp_from_land_count"] - row_bn["fp_from_land_count"]
            diff_fp_perm_bn = r["fp_from_permwater_count"] - row_bn["fp_from_permwater_count"]
            diff_fn_land_bn = r["fn_miss_to_land_count"] - row_bn["fn_miss_to_land_count"]
            diff_fn_perm_bn = r["fn_miss_to_permwater_count"] - row_bn["fn_miss_to_permwater_count"]

            comparative_rows.append({
                "event_id": ev,
                "condition_id": c,
                "f1": r["f1"],
                "delta_f1_vs_source": diff_f1_src,
                "delta_f1_vs_bn_no_update": diff_f1_bn,
                "delta_fp_land_vs_source": diff_fp_land_src,
                "delta_fp_permwater_vs_source": diff_fp_perm_src,
                "delta_fn_land_vs_source": diff_fn_land_src,
                "delta_fn_permwater_vs_source": diff_fn_perm_src,
                "delta_fp_land_vs_bn_no_update": diff_fp_land_bn,
                "delta_fp_permwater_vs_bn_no_update": diff_fp_perm_bn,
                "delta_fn_land_vs_bn_no_update": diff_fn_land_bn,
                "delta_fn_permwater_vs_bn_no_update": diff_fn_perm_bn,
            })

    with open(OUTPUT_DIR / "comparative_shifts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparative_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparative_rows)

    # 5. SHA-256 Manifest
    sha_lines = []
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file() and p.name != "artifact_sha256.txt":
            sha_lines.append(f"{sha256_file(p)}  {p.name}\n")
    with open(OUTPUT_DIR / "artifact_sha256.txt", "w", encoding="utf-8") as f:
        f.writelines(sha_lines)

    print("✓ Analysis complete. Output files written to", OUTPUT_DIR)
    return {
        "cm_3x3": cm_3x3_rows,
        "error_breakdown": error_breakdown_rows,
        "comparative": comparative_rows,
    }


if __name__ == "__main__":
    run_week14_diagnosis()

