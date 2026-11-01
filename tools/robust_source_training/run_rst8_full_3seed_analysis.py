#!/usr/bin/env python3
"""RST8 Full Three-Seed Architecture Replication Analysis Tool (SiamUnet_diff).

Executes the comprehensive 3-seed statistical analysis for ChangeGuard
SiamUnet_diff replication study: R0 vs R2 on Fold A and Fold B across seeds 0, 1, 2 (12 runs).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
)

ANALYSIS_RUNS = {
    "fold_a": {
        "evaluation_events": [118, 174, 273, 421, 496, 498, 554, 567, 1111009],
        "r0": [
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_folda_s0_v1",
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_folda_s1_v1",
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_folda_s2_v1",
        ],
        "r2": [
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_folda_s0_v2",
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_folda_s1_v1",
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_folda_s2_v1",
        ],
    },
    "fold_b": {
        "evaluation_events": [267, 324, 417, 470, 502, 518, 555, 1111004, 1111010],
        "r0": [
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_foldb_s0_v2",
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_foldb_s1_v1",
            "outputs/robust_source_training/rst8/rst8_r0_siamunet_diff_foldb_s2_v1",
        ],
        "r2": [
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_foldb_s0_v2",
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_foldb_s1_v1",
            "outputs/robust_source_training/rst8/rst8_r2_siamunet_diff_foldb_s2_v1",
        ],
    },
}


def validate_contracts():
    events_a = set(ANALYSIS_RUNS["fold_a"]["evaluation_events"])
    events_b = set(ANALYSIS_RUNS["fold_b"]["evaluation_events"])
    forbidden = set(FORBIDDEN_TARGET_EVENTS)

    if events_a & forbidden or events_b & forbidden:
        raise ContractViolationError("Analysis configuration contains forbidden target events")
    if events_a & events_b:
        raise ContractViolationError("Evaluation sets of Fold A and Fold B are not disjoint")

    all_eval = events_a | events_b
    if len(all_eval) != 18:
        raise ContractViolationError(f"Expected exactly 18 evaluation events, got {len(all_eval)}")
    if not all_eval.issubset(set(AUTHORITATIVE_SOURCE_EVENTS)):
        raise ContractViolationError("Evaluation events contain unauthorized event IDs")


def ingest_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]], Dict[str, Dict[str, Any]]]:
    """Ingests all 12 runs and produces canonical records, nested lookup, and run metadata."""
    canonical_rows = []
    nested = {"fold_a": {"r0": {}, "r2": {}}, "fold_b": {"r0": {}, "r2": {}}}
    run_meta = {}

    for fold_name in ["fold_a", "fold_b"]:
        fold_cfg = ANALYSIS_RUNS[fold_name]
        expected_events = set(fold_cfg["evaluation_events"])

        for recipe in ["r0", "r2"]:
            for run_rel in fold_cfg[recipe]:
                run_dir = REPO_ROOT / run_rel
                with open(run_dir / "preflight_manifest.json", "r", encoding="utf-8") as f:
                    pre = json.load(f)
                seed = pre["partition"]["seed"]
                run_id = pre["run_id"]
                method = pre["recipe_id"]
                fold_id = pre["partition"]["fold_id"]

                with open(run_dir / "terminal_status.json", "r", encoding="utf-8") as f:
                    tstat = json.load(f)
                if tstat["status"] != "COMPLETED_SCIENTIFIC_RUN" or tstat["exit_code"] != 0:
                    raise ContractViolationError(f"Run {run_id} is not COMPLETED_SCIENTIFIC_RUN: status={tstat.get('status')}")

                with open(run_dir / "final_manifest.json", "r", encoding="utf-8") as f:
                    fman = json.load(f)

                # Parse runtime from log if final_manifest has 0.0
                runtime_sec = fman.get("runtime_seconds", 0.0)
                peak_vram = fman.get("peak_vram_mib", 0.0)
                if runtime_sec == 0.0:
                    log_file = REPO_ROOT / pre.get("log_file", f"logs/robust_source_training/rst8/{run_id}.log")
                    if log_file.exists():
                        lines = log_file.read_text(encoding="utf-8").splitlines()
                        for line in lines[-20:]:
                            if "EXECUTION: SUCCESS" in line and "Runtime:" in line:
                                try:
                                    part = line.split("Runtime:")[1].split("s")[0].strip()
                                    runtime_sec = float(part)
                                except Exception:
                                    pass

                run_meta[run_id] = {
                    "run_id": run_id,
                    "recipe": method,
                    "fold": fold_id,
                    "seed": seed,
                    "status": tstat["status"],
                    "optimizer_steps": fman.get("optimizer_steps", 6656 if "folda" in run_id else 7328),
                    "runtime_seconds": runtime_sec,
                    "peak_vram_mib": peak_vram,
                    "checkpoint_sha256": fman.get("checkpoints", {}).get("final_checkpoint", {}).get("sha256", ""),
                }

                with open(run_dir / "event_metrics.json", "r", encoding="utf-8") as f:
                    em = json.load(f)

                if em["n_events_evaluated"] != 9 or len(em["per_event"]) != 9:
                    raise ContractViolationError(f"Run {run_id} does not have exactly 9 evaluated events")

                observed_events = set()
                for e_rec in em["per_event"]:
                    eid = int(e_rec["event_id"])
                    observed_events.add(eid)

                    row = {
                        "run_id": run_id,
                        "method": method,
                        "fold_id": fold_id,
                        "seed": seed,
                        "event_id": eid,
                        "sample_count": e_rec["sample_count"],
                        "valid_pixel_count": e_rec["valid_pixel_count"],
                        "tp": e_rec["tp"],
                        "fp": e_rec["fp"],
                        "fn": e_rec["fn"],
                        "tn": e_rec["tn"],
                        "precision": float(e_rec["precision"]),
                        "recall": float(e_rec["recall"]),
                        "f1": float(e_rec["f1"]),
                        "iou": float(e_rec["iou"]),
                    }
                    canonical_rows.append(row)

                    if eid not in nested[fold_name][recipe]:
                        nested[fold_name][recipe][eid] = {}
                    nested[fold_name][recipe][eid][seed] = row

                if observed_events != expected_events:
                    raise ContractViolationError(f"Run {run_id} observed events do not match fold definition")

    if len(canonical_rows) != 108:
        raise ContractViolationError(f"Expected 108 canonical rows, got {len(canonical_rows)}")

    return canonical_rows, nested, run_meta


def compute_statistics(nested: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]]) -> Dict[str, Any]:
    """Computes all per-event paired metrics across seeds, fold macros, and grand metrics."""
    paired_rows = []
    events_all = []

    # 1. Per-event paired calculations (averaged across seeds 0, 1, 2)
    for fold_name in ["fold_a", "fold_b"]:
        eval_events = ANALYSIS_RUNS[fold_name]["evaluation_events"]
        for eid in eval_events:
            events_all.append(eid)
            r0_f1_list = [nested[fold_name]["r0"][eid][s]["f1"] for s in [0, 1, 2]]
            r0_p_list = [nested[fold_name]["r0"][eid][s]["precision"] for s in [0, 1, 2]]
            r0_r_list = [nested[fold_name]["r0"][eid][s]["recall"] for s in [0, 1, 2]]
            r0_iou_list = [nested[fold_name]["r0"][eid][s]["iou"] for s in [0, 1, 2]]

            r2_f1_list = [nested[fold_name]["r2"][eid][s]["f1"] for s in [0, 1, 2]]
            r2_p_list = [nested[fold_name]["r2"][eid][s]["precision"] for s in [0, 1, 2]]
            r2_r_list = [nested[fold_name]["r2"][eid][s]["recall"] for s in [0, 1, 2]]
            r2_iou_list = [nested[fold_name]["r2"][eid][s]["iou"] for s in [0, 1, 2]]

            r0_f1_mean = float(np.mean(r0_f1_list))
            r2_f1_mean = float(np.mean(r2_f1_list))
            delta_f1 = r2_f1_mean - r0_f1_mean

            r0_p_mean = float(np.mean(r0_p_list))
            r2_p_mean = float(np.mean(r2_p_list))
            delta_p = r2_p_mean - r0_p_mean

            r0_r_mean = float(np.mean(r0_r_list))
            r2_r_mean = float(np.mean(r2_r_list))
            delta_r = r2_r_mean - r0_r_mean

            r0_iou_mean = float(np.mean(r0_iou_list))
            r2_iou_mean = float(np.mean(r2_iou_list))
            delta_iou = r2_iou_mean - r0_iou_mean

            status = "Benefit" if delta_f1 > 0.010 else ("Harm" if delta_f1 < -0.010 else "Neutral")

            paired_rows.append({
                "event_id": eid,
                "fold_id": fold_name,
                "r0_f1_mean": r0_f1_mean,
                "r2_f1_mean": r2_f1_mean,
                "delta_f1": delta_f1,
                "status": status,
                "r0_precision_mean": r0_p_mean,
                "r2_precision_mean": r2_p_mean,
                "delta_precision": delta_p,
                "r0_recall_mean": r0_r_mean,
                "r2_recall_mean": r2_r_mean,
                "delta_recall": delta_r,
                "r0_iou_mean": r0_iou_mean,
                "r2_iou_mean": r2_iou_mean,
                "delta_iou": delta_iou,
                "r0_f1_seeds": r0_f1_list,
                "r2_f1_seeds": r2_f1_list,
            })

    # 2. Fold-level per-seed and macro metrics
    fold_summaries = {}
    for fold_name in ["fold_a", "fold_b"]:
        eval_events = ANALYSIS_RUNS[fold_name]["evaluation_events"]
        fold_summaries[fold_name] = {}
        for recipe in ["r0", "r2"]:
            seed_macros = []
            seed_precisions = []
            seed_recalls = []
            seed_ious = []
            for s in [0, 1, 2]:
                f1s = [nested[fold_name][recipe][eid][s]["f1"] for eid in eval_events]
                ps = [nested[fold_name][recipe][eid][s]["precision"] for eid in eval_events]
                rs = [nested[fold_name][recipe][eid][s]["recall"] for eid in eval_events]
                ious = [nested[fold_name][recipe][eid][s]["iou"] for eid in eval_events]
                seed_macros.append(float(np.mean(f1s)))
                seed_precisions.append(float(np.mean(ps)))
                seed_recalls.append(float(np.mean(rs)))
                seed_ious.append(float(np.mean(ious)))

            mean_macro = float(np.mean(seed_macros))
            sd_macro = float(np.std(seed_macros, ddof=1)) if len(seed_macros) > 1 else 0.0

            fold_summaries[fold_name][recipe] = {
                "seed_macro_f1": seed_macros,
                "mean_macro_f1": mean_macro,
                "sd_macro_f1": sd_macro,
                "seed_macro_precision": seed_precisions,
                "mean_macro_precision": float(np.mean(seed_precisions)),
                "seed_macro_recall": seed_recalls,
                "mean_macro_recall": float(np.mean(seed_recalls)),
                "seed_macro_iou": seed_ious,
                "mean_macro_iou": float(np.mean(seed_ious)),
            }

    # 3. Grand mean calculations (18 unique events)
    all_f1_r0 = [p["r0_f1_mean"] for p in paired_rows]
    all_f1_r2 = [p["r2_f1_mean"] for p in paired_rows]
    all_p_r0 = [p["r0_precision_mean"] for p in paired_rows]
    all_p_r2 = [p["r2_precision_mean"] for p in paired_rows]
    all_r_r0 = [p["r0_recall_mean"] for p in paired_rows]
    all_r_r2 = [p["r2_recall_mean"] for p in paired_rows]
    all_iou_r0 = [p["r0_iou_mean"] for p in paired_rows]
    all_iou_r2 = [p["r2_iou_mean"] for p in paired_rows]
    all_delta_f1 = [p["delta_f1"] for p in paired_rows]

    grand_mean_f1_r0 = float(np.mean(all_f1_r0))
    grand_mean_f1_r2 = float(np.mean(all_f1_r2))
    grand_mean_delta_f1 = grand_mean_f1_r2 - grand_mean_f1_r0

    grand_mean_p_r0 = float(np.mean(all_p_r0))
    grand_mean_p_r2 = float(np.mean(all_p_r2))
    grand_mean_delta_p = grand_mean_p_r2 - grand_mean_p_r0

    grand_mean_r_r0 = float(np.mean(all_r_r0))
    grand_mean_r_r2 = float(np.mean(all_r_r2))
    grand_mean_delta_r = grand_mean_r_r2 - grand_mean_r_r0

    grand_mean_iou_r0 = float(np.mean(all_iou_r0))
    grand_mean_iou_r2 = float(np.mean(all_iou_r2))
    grand_mean_delta_iou = grand_mean_iou_r2 - grand_mean_iou_r0

    # Tail metrics across 18 events
    worst_event_r0 = float(np.min(all_f1_r0))
    worst_event_r2 = float(np.min(all_f1_r2))
    delta_worst_event = worst_event_r2 - worst_event_r0

    bottom3_r0 = float(np.mean(sorted(all_f1_r0)[:3]))
    bottom3_r2 = float(np.mean(sorted(all_f1_r2)[:3]))
    delta_bottom3 = bottom3_r2 - bottom3_r0

    # Event counts
    n_benefit = sum(1 for p in paired_rows if p["status"] == "Benefit")
    n_harm = sum(1 for p in paired_rows if p["status"] == "Harm")
    n_neutral = sum(1 for p in paired_rows if p["status"] == "Neutral")

    stats = {
        "paired_rows": paired_rows,
        "fold_summaries": fold_summaries,
        "grand_metrics": {
            "r0": {
                "f1": grand_mean_f1_r0,
                "precision": grand_mean_p_r0,
                "recall": grand_mean_r_r0,
                "iou": grand_mean_iou_r0,
                "worst_event_f1": worst_event_r0,
                "bottom3_mean_f1": bottom3_r0,
            },
            "r2": {
                "f1": grand_mean_f1_r2,
                "precision": grand_mean_p_r2,
                "recall": grand_mean_r_r2,
                "iou": grand_mean_iou_r2,
                "worst_event_f1": worst_event_r2,
                "bottom3_mean_f1": bottom3_r2,
            },
            "delta": {
                "f1": grand_mean_delta_f1,
                "precision": grand_mean_delta_p,
                "recall": grand_mean_delta_r,
                "iou": grand_mean_delta_iou,
                "worst_event_f1": delta_worst_event,
                "bottom3_mean_f1": delta_bottom3,
            },
            "event_counts": {
                "benefit_count": n_benefit,
                "neutral_count": n_neutral,
                "harm_count": n_harm,
            },
        },
    }
    return stats


def run_paired_bootstrap(paired_rows: List[Dict[str, Any]], n_resamples: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Runs paired event-level bootstrap across the 18 unique held-out events."""
    rng = np.random.RandomState(seed)
    n = len(paired_rows)
    assert n == 18, f"Bootstrap requires exactly 18 events, found {n}"

    deltas = np.array([p["delta_f1"] for p in paired_rows], dtype=np.float64)
    resampled_means = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        resampled_means[b] = np.mean(deltas[indices])

    resampled_means.sort()
    ci_lower = float(np.percentile(resampled_means, 2.5))
    ci_upper = float(np.percentile(resampled_means, 97.5))
    p_value = float(np.mean(resampled_means <= 0.0))

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "18_unique_held_out_events",
        "observed_mean_delta": float(np.mean(deltas)),
        "ci_95": [ci_lower, ci_upper],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "one_sided_p_value_h0_delta_le_0": p_value,
    }


def main():
    validate_contracts()
    canonical_rows, nested, run_meta = ingest_data()
    stats = compute_statistics(nested)
    bootstrap = run_paired_bootstrap(stats["paired_rows"])

    out_dir = REPO_ROOT / "outputs/robust_source_training/rst8"

    # 1. Save canonical 108-row CSV
    csv_108_path = out_dir / "canonical_event_results_full_3seed_v1.csv"
    with open(csv_108_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(canonical_rows[0].keys()))
        writer.writeheader()
        writer.writerows(canonical_rows)
    print(f"Saved canonical 108-row CSV: {csv_108_path}")

    # 2. Save paired 18-row CSV
    csv_18_path = out_dir / "paired_event_deltas_full_3seed_v1.csv"
    paired_export = []
    for p in stats["paired_rows"]:
        rec = dict(p)
        rec["r0_f1_seeds"] = str(rec["r0_f1_seeds"])
        rec["r2_f1_seeds"] = str(rec["r2_f1_seeds"])
        paired_export.append(rec)
    with open(csv_18_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_export[0].keys()))
        writer.writeheader()
        writer.writerows(paired_export)
    print(f"Saved paired 18-event CSV: {csv_18_path}")

    # 3. Save bootstrap JSON
    boot_path = out_dir / "bootstrap_distribution_full_3seed_v1.json"
    with open(boot_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap, f, indent=2)
    print(f"Saved bootstrap distribution JSON: {boot_path}")

    # 4. Save comprehensive summary JSON
    # Load original seed-0 summary for direct side-by-side comparison
    orig_stage_a_path = out_dir / "stage_a_summary_v1.json"
    orig_summary = {}
    if orig_stage_a_path.exists():
        with open(orig_stage_a_path, "r", encoding="utf-8") as f:
            orig_summary = json.load(f)

    full_summary = {
        "study": "ChangeGuard RST8 Supplementary Architecture Replication Study (SiamUnet_diff)",
        "backbone": "SiamUnet_diff",
        "parameters_count": 1350147,
        "scope": "Three-seed evaluation (seeds 0, 1, 2) on Confirmation Folds A and B (12 total runs)",
        "run_metadata": run_meta,
        "fold_summaries": stats["fold_summaries"],
        "grand_metrics": stats["grand_metrics"],
        "bootstrap": bootstrap,
        "seed0_screen_comparison": {
            "seed0_grand_f1_r0": orig_summary.get("grand_metrics", {}).get("r0", {}).get("f1"),
            "seed0_grand_f1_r2": orig_summary.get("grand_metrics", {}).get("r2", {}).get("f1"),
            "seed0_delta_f1": orig_summary.get("grand_metrics", {}).get("delta", {}).get("f1"),
            "seed0_bootstrap_95ci": orig_summary.get("bootstrap", {}).get("ci_95"),
            "seed0_bottom3_delta": orig_summary.get("grand_metrics", {}).get("delta", {}).get("bottom3_mean_f1"),
            "seed0_worst_delta": orig_summary.get("grand_metrics", {}).get("delta", {}).get("worst_event_f1"),
            "three_seed_grand_f1_r0": stats["grand_metrics"]["r0"]["f1"],
            "three_seed_grand_f1_r2": stats["grand_metrics"]["r2"]["f1"],
            "three_seed_delta_f1": stats["grand_metrics"]["delta"]["f1"],
            "three_seed_bootstrap_95ci": bootstrap["ci_95"],
            "three_seed_bottom3_delta": stats["grand_metrics"]["delta"]["bottom3_mean_f1"],
            "three_seed_worst_delta": stats["grand_metrics"]["delta"]["worst_event_f1"],
        },
        "snunet_ecam_contrast": {
            "snunet_r0_grand_f1": 0.6923,
            "snunet_r2_grand_f1": 0.7135,
            "snunet_delta_f1": +0.0212,
            "snunet_bootstrap_95ci": [+0.0075, +0.0357],
            "snunet_benefited_events": 15,
            "snunet_harmed_events": 2,
        },
    }

    summary_path = out_dir / "rst8_full_3seed_summary_v1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved comprehensive summary JSON: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SIAMUNET_DIFF THREE-SEED REPLICATION RESULTS SUMMARY")
    print("=" * 70)
    g = stats["grand_metrics"]
    print(f"Grand Event-Macro F1 (18 Events):")
    print(f"  R0 (3-Seed Mean): {g['r0']['f1']:.4f}")
    print(f"  R2 (3-Seed Mean): {g['r2']['f1']:.4f}")
    print(f"  Δ(R2 - R0):       {g['delta']['f1']:+.4f} ({g['delta']['f1']*100:+.2f} pp)")
    print(f"  95% Bootstrap CI: [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}] (p = {bootstrap['one_sided_p_value_h0_delta_le_0']:.4f})")
    print(f"\nTail Reliability:")
    print(f"  Bottom-3 Mean F1: R0 = {g['r0']['bottom3_mean_f1']:.4f}, R2 = {g['r2']['bottom3_mean_f1']:.4f} (Δ = {g['delta']['bottom3_mean_f1']:+.4f})")
    print(f"  Worst-Event F1:   R0 = {g['r0']['worst_event_f1']:.4f}, R2 = {g['r2']['worst_event_f1']:.4f} (Δ = {g['delta']['worst_event_f1']:+.4f})")
    print(f"\nPrecision / Recall Trade-off:")
    print(f"  Macro Precision:  R0 = {g['r0']['precision']:.4f}, R2 = {g['r2']['precision']:.4f} (Δ = {g['delta']['precision']:+.4f})")
    print(f"  Macro Recall:     R0 = {g['r0']['recall']:.4f}, R2 = {g['r2']['recall']:.4f} (Δ = {g['delta']['recall']:+.4f})")
    print(f"  Macro IoU:        R0 = {g['r0']['iou']:.4f}, R2 = {g['r2']['iou']:.4f} (Δ = {g['delta']['iou']:+.4f})")
    print(f"\nFold-Level Breakdown:")
    fa = stats["fold_summaries"]["fold_a"]
    fb = stats["fold_summaries"]["fold_b"]
    print(f"  Fold A: R0 = {fa['r0']['mean_macro_f1']:.4f} ± {fa['r0']['sd_macro_f1']:.4f} | R2 = {fa['r2']['mean_macro_f1']:.4f} ± {fa['r2']['sd_macro_f1']:.4f} (Δ = {fa['r2']['mean_macro_f1'] - fa['r0']['mean_macro_f1']:+.4f})")
    print(f"  Fold B: R0 = {fb['r0']['mean_macro_f1']:.4f} ± {fb['r0']['sd_macro_f1']:.4f} | R2 = {fb['r2']['mean_macro_f1']:.4f} ± {fb['r2']['sd_macro_f1']:.4f} (Δ = {fb['r2']['mean_macro_f1'] - fb['r0']['mean_macro_f1']:+.4f})")
    ec = g["event_counts"]
    print(f"\nEvent Benefit / Neutral / Harm Counts (18 Events):")
    print(f"  Benefit (> +0.010): {ec['benefit_count']} events")
    print(f"  Neutral (±0.010):   {ec['neutral_count']} events")
    print(f"  Harm (< -0.010):    {ec['harm_count']} events")
    print("=" * 70)


if __name__ == "__main__":
    main()

