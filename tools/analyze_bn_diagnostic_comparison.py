"""ChangeGuard Week 12 BN Diagnostic Analysis & Comparison Script.

Compares:
1. source_only
2. matched_bn_no_update_v1
3. matched_tent_pre_update_v1
4. C1_consistency_no_conf
5. C2_consistency_conf90

Quantifies:
- no-update BN vs source_only
- each updating condition vs no-update BN
- per-event, macro, pooled, and worst-event F1
- exact pixel prediction differences across frozen predictions
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WEEK12_DIR = REPO_ROOT / "outputs/week12_ablation/approved_dev6_s0_v1"
BN_DIAG_DIR = REPO_ROOT / "outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1"
ANALYSIS_DIR = REPO_ROOT / "outputs/week12_ablation/analysis_v1"

EVENTS = [279, 437, 520, 559, 1111003, 1111008]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_comparison():
    print("================================================================================")
    print(" ChangeGuard Week 12 BN Diagnostic Verification & Controlled Comparison")
    print("================================================================================")

    # 1. Integrity Verification of BN Diagnostic
    init_manifest_path = BN_DIAG_DIR / "diagnostic_initial_manifest.json"
    comp_status_path = BN_DIAG_DIR / "completion_status.json"
    cond_sum_path = BN_DIAG_DIR / "matched_bn_no_update_v1/condition_summary.json"

    assert init_manifest_path.is_file(), f"Missing {init_manifest_path}"
    assert comp_status_path.is_file(), f"Missing {comp_status_path}"
    assert cond_sum_path.is_file(), f"Missing {cond_sum_path}"

    with open(init_manifest_path, "r", encoding="utf-8") as f:
        init_man = json.load(f)
    with open(comp_status_path, "r", encoding="utf-8") as f:
        comp_stat = json.load(f)
    with open(cond_sum_path, "r", encoding="utf-8") as f:
        bn_cond_sum = json.load(f)

    print(f"Status:         {comp_stat.get('status')}")
    print(f"Elapsed Time:   {comp_stat.get('elapsed_seconds'):.2f}s")
    print(f"Updates:        {bn_cond_sum['adaptation_behavior']['updates_applied']} applied, "
          f"{bn_cond_sum['adaptation_behavior']['updates_skipped']} skipped, "
          f"{bn_cond_sum['adaptation_behavior']['updates_failed']} failed")

    # Load metrics from Week 12
    with open(WEEK12_DIR / "ablation_summary.json", "r", encoding="utf-8") as f:
        w12_abl = json.load(f)

    all_conditions = [
        "source_only",
        "matched_bn_no_update_v1",
        "matched_tent_pre_update_v1",
        "C1_consistency_no_conf",
        "C2_consistency_conf90",
    ]

    event_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {c: {} for c in all_conditions}
    preds_by_cond: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {c: {} for c in all_conditions}

    # Load week 12 conditions
    for c in ["source_only", "matched_tent_pre_update_v1", "C1_consistency_no_conf", "C2_consistency_conf90"]:
        for ev in EVENTS:
            ev_dir = WEEK12_DIR / c / f"event_{ev}"
            with open(ev_dir / "event_metrics.json", "r", encoding="utf-8") as f:
                event_metrics[c][ev] = json.load(f)
            npz = np.load(ev_dir / "frozen_predictions.npz")
            preds_by_cond[c][ev] = {k: npz[k] for k in npz.files}

    # Load BN diagnostic condition
    for ev in EVENTS:
        ev_dir = BN_DIAG_DIR / "matched_bn_no_update_v1" / f"event_{ev}"
        assert ev_dir.is_dir(), f"Missing {ev_dir}"
        with open(ev_dir / "event_metrics.json", "r", encoding="utf-8") as f:
            em = json.load(f)
        event_metrics["matched_bn_no_update_v1"][ev] = em

        # Confusion conservation
        conf_sum = em["tp"] + em["fp"] + em["fn"] + em["tn"]
        assert conf_sum == em["valid_pixels"], f"Conservation failed in BN diagnostic event {ev}"
        assert em["updates_applied"] == 0, f"Non-zero updates in BN diagnostic event {ev}"

        npz = np.load(ev_dir / "frozen_predictions.npz")
        preds_by_cond["matched_bn_no_update_v1"][ev] = {k: npz[k] for k in npz.files}

        # Check identical sample coverage and order
        ref_samples = list(preds_by_cond["source_only"][ev].keys())
        bn_samples = list(preds_by_cond["matched_bn_no_update_v1"][ev].keys())
        assert ref_samples == bn_samples, f"Sample mismatch in event {ev}"

    print("Sample coverage, order, and confusion conservation verified across all 6 events.")

    # 2. Pixel Prediction Differences
    print("\n--- Pixel Prediction Divergence Analysis ---")
    pixel_diffs_table = []
    for ev in EVENTS:
        sample_ids = list(preds_by_cond["source_only"][ev].keys())
        total_pixels = 0
        diff_bn_vs_src = 0
        diff_tent_vs_bn = 0
        diff_c1_vs_bn = 0
        diff_c2_vs_bn = 0
        diff_c2_vs_c1 = 0

        for sid in sample_ids:
            p_src = preds_by_cond["source_only"][ev][sid]
            p_bn = preds_by_cond["matched_bn_no_update_v1"][ev][sid]
            p_tent = preds_by_cond["matched_tent_pre_update_v1"][ev][sid]
            p_c1 = preds_by_cond["C1_consistency_no_conf"][ev][sid]
            p_c2 = preds_by_cond["C2_consistency_conf90"][ev][sid]

            n_px = p_src.size
            total_pixels += n_px

            diff_bn_vs_src += int(np.sum(p_bn != p_src))
            diff_tent_vs_bn += int(np.sum(p_tent != p_bn))
            diff_c1_vs_bn += int(np.sum(p_c1 != p_bn))
            diff_c2_vs_bn += int(np.sum(p_c2 != p_bn))
            diff_c2_vs_c1 += int(np.sum(p_c2 != p_c1))

        pixel_diffs_table.append({
            "event_id": ev,
            "samples": len(sample_ids),
            "total_pixels": total_pixels,
            "diff_bn_vs_src": diff_bn_vs_src,
            "frac_bn_vs_src": diff_bn_vs_src / total_pixels,
            "diff_tent_vs_bn": diff_tent_vs_bn,
            "frac_tent_vs_bn": diff_tent_vs_bn / total_pixels,
            "diff_c1_vs_bn": diff_c1_vs_bn,
            "frac_c1_vs_bn": diff_c1_vs_bn / total_pixels,
            "diff_c2_vs_bn": diff_c2_vs_bn,
            "frac_c2_vs_bn": diff_c2_vs_bn / total_pixels,
            "diff_c2_vs_c1": diff_c2_vs_c1,
            "frac_c2_vs_c1": diff_c2_vs_c1 / total_pixels,
        })

    print(f"{'Event':<8} | {'Samples':<7} | {'BN vs Source':<15} | {'Tent vs BN':<15} | {'C1 vs BN':<15} | {'C2 vs BN':<15} | {'C2 vs C1':<12}")
    print("-" * 95)
    for r in pixel_diffs_table:
        print(f"{r['event_id']:<8} | {r['samples']:<7} | {r['diff_bn_vs_src']:<8} ({r['frac_bn_vs_src']*100:.2f}%) | "
              f"{r['diff_tent_vs_bn']:<8} ({r['frac_tent_vs_bn']*100:.2f}%) | "
              f"{r['diff_c1_vs_bn']:<8} ({r['frac_c1_vs_bn']*100:.2f}%) | "
              f"{r['diff_c2_vs_bn']:<8} ({r['frac_c2_vs_bn']*100:.2f}%) | "
              f"{r['diff_c2_vs_c1']:<8} ({r['frac_c2_vs_c1']*100:.4f}%)")

    # 3. Macro, Pooled, Worst Summary Table
    print("\n--- Summary Performance Table Across 5 Conditions ---")
    summary_rows = []
    for c in all_conditions:
        f1s = [event_metrics[c][ev]["f1"] for ev in EVENTS]
        ious = [event_metrics[c][ev]["iou"] for ev in EVENTS]
        precs = [event_metrics[c][ev]["precision"] for ev in EVENTS]
        recs = [event_metrics[c][ev]["recall"] for ev in EVENTS]

        macro_f1 = float(np.mean(f1s))
        macro_iou = float(np.mean(ious))
        macro_prec = float(np.mean(precs))
        macro_rec = float(np.mean(recs))

        tot_tp = sum(event_metrics[c][ev]["tp"] for ev in EVENTS)
        tot_fp = sum(event_metrics[c][ev]["fp"] for ev in EVENTS)
        tot_fn = sum(event_metrics[c][ev]["fn"] for ev in EVENTS)
        tot_tn = sum(event_metrics[c][ev]["tn"] for ev in EVENTS)
        tot_vp = sum(event_metrics[c][ev]["valid_pixels"] for ev in EVENTS)

        pooled_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        pooled_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        pooled_f1 = (2 * pooled_prec * pooled_rec) / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = tot_tp / (tot_tp + tot_fp + tot_fn) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

        worst_f1 = min(f1s)
        worst_ev = EVENTS[f1s.index(worst_f1)]

        summary_rows.append({
            "condition_id": c,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_f1": worst_f1,
            "worst_event_id": worst_ev,
            "macro_iou": macro_iou,
            "pooled_iou": pooled_iou,
            "macro_precision": macro_prec,
            "pooled_precision": pooled_prec,
            "macro_recall": macro_rec,
            "pooled_recall": pooled_rec,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "tn": tot_tn,
            "valid_pixels": tot_vp,
        })

    print(f"{'Condition':<28} | {'Macro F1':<10} | {'Pooled F1':<10} | {'Worst F1':<10} (Ev) | {'Pooled IoU':<10} | {'Pooled Prec':<11} | {'Pooled Rec':<10}")
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['condition_id']:<28} | {r['macro_f1']:<10.4f} | {r['pooled_f1']:<10.4f} | {r['worst_event_f1']:<10.4f} ({r['worst_event_id']}) | {r['pooled_iou']:<10.4f} | {r['pooled_precision']:<11.4f} | {r['pooled_recall']:<10.4f}")

    # 4. Per-Event Controlled Contrasts
    print("\n--- Per-Event Controlled Contrasts Table ---")
    per_event_rows = []
    print(f"{'Event':<8} | {'Condition':<28} | {'F1':<8} | {'IoU':<8} | {'Diff vs Src':<12} | {'Diff vs BN No-Up':<16} | {'Pred Flood':<11} | {'GT Flood':<11}")
    print("-" * 115)

    for ev in EVENTS:
        src_f1 = event_metrics["source_only"][ev]["f1"]
        bn_f1 = event_metrics["matched_bn_no_update_v1"][ev]["f1"]

        for c in all_conditions:
            em = event_metrics[c][ev]
            f1 = em["f1"]
            iou = em["iou"]
            diff_src = f1 - src_f1
            diff_bn = f1 - bn_f1

            pred_fl = em["tp"] + em["fp"]
            gt_fl = em["tp"] + em["fn"]

            diff_src_str = f"{diff_src:+.4f}" if c != "source_only" else "0.0000"
            diff_bn_str = f"{diff_bn:+.4f}" if c != "matched_bn_no_update_v1" else "[ref]"

            print(f"{ev:<8} | {c:<28} | {f1:<8.4f} | {iou:<8.4f} | {diff_src_str:<12} | {diff_bn_str:<16} | {pred_fl:<11} | {gt_fl:<11}")

            per_event_rows.append({
                "event_id": ev,
                "condition_id": c,
                "f1": f1,
                "iou": iou,
                "precision": em["precision"],
                "recall": em["recall"],
                "tp": em["tp"],
                "fp": em["fp"],
                "fn": em["fn"],
                "tn": em["tn"],
                "valid_pixels": em["valid_pixels"],
                "pred_flood_pixels": pred_fl,
                "gt_flood_pixels": gt_fl,
                "diff_vs_source_f1": diff_src,
                "diff_vs_bn_no_update_f1": diff_bn,
            })

    # Save CSVs
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "bn_diagnostic_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(ANALYSIS_DIR / "bn_diagnostic_per_event.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_event_rows)

    with open(ANALYSIS_DIR / "bn_diagnostic_pixel_diffs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pixel_diffs_table[0].keys()))
        writer.writeheader()
        writer.writerows(pixel_diffs_table)

    print("\nCSV summaries written to outputs/week12_ablation/analysis_v1/")


if __name__ == "__main__":
    run_comparison()

