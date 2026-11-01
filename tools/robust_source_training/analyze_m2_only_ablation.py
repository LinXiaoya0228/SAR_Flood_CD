#!/usr/bin/env python3
"""Supplementary Ablation Analysis Tool: Class Weighting without Event-Balanced Sampling (M2_ONLY).

Compares:
1. R0: Ordinary baseline patch sampling, unweighted loss.
2. M2_ONLY: Ordinary baseline patch sampling + frozen flood-aware class-weighted loss (M2).
3. R2: Event-balanced patch sampling (M1) + frozen flood-aware class-weighted loss (M2).

Answers the central scientific question:
Does class weighting alone explain the R2 improvement, or does event-balanced sampling
add value when class weighting is enabled?
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
)

FOLD_EVAL_EVENTS = {
    "fold_a": [118, 174, 273, 421, 496, 498, 554, 567, 1111009],
    "fold_b": [267, 324, 417, 470, 502, 518, 555, 1111004, 1111010],
}

RUN_PATHS = {
    "fold_a": {
        "R0": [
            "outputs/robust_source_training/rst6/rst6_r0_snunet_folda_s0_v1",
            "outputs/robust_source_training/rst6/rst6_r0_snunet_folda_s1_v1",
            "outputs/robust_source_training/rst6/rst6_r0_snunet_folda_s2_v1",
        ],
        "R2": [
            "outputs/robust_source_training/rst6/rst6_r2_snunet_folda_s0_v2",
            "outputs/robust_source_training/rst6/rst6_r2_snunet_folda_s1_v2",
            "outputs/robust_source_training/rst6/rst6_r2_snunet_folda_s2_v2",
        ],
        "M2_ONLY": [
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_folda_s0_v1",
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_folda_s1_v1",
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_folda_s2_v1",
        ],
    },
    "fold_b": {
        "R0": [
            "outputs/robust_source_training/rst6/rst6_r0_snunet_foldb_s0_v1",
            "outputs/robust_source_training/rst6/rst6_r0_snunet_foldb_s1_v1",
            "outputs/robust_source_training/rst6/rst6_r0_snunet_foldb_s2_v1",
        ],
        "R2": [
            "outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s0_v2",
            "outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s1_v2",
            "outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s2_v2",
        ],
        "M2_ONLY": [
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_foldb_s0_v1",
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_foldb_s1_v1",
            "outputs/robust_source_training/ablation_m2_only/ablation_m2_only_snunet_foldb_s2_v1",
        ],
    },
}


def run_paired_bootstrap(
    delta_list: List[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs paired event-level bootstrap across the 18 unique held-out events."""
    rng = np.random.RandomState(seed)
    n = len(delta_list)
    assert n == 18, f"Bootstrap requires exactly 18 events, found {n}"

    deltas = np.array(delta_list, dtype=np.float64)
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
        "percentile_distribution": {
            "p1": float(np.percentile(resampled_means, 1.0)),
            "p5": float(np.percentile(resampled_means, 5.0)),
            "p10": float(np.percentile(resampled_means, 10.0)),
            "p25": float(np.percentile(resampled_means, 25.0)),
            "p50_median": float(np.percentile(resampled_means, 50.0)),
            "p75": float(np.percentile(resampled_means, 75.0)),
            "p90": float(np.percentile(resampled_means, 90.0)),
            "p95": float(np.percentile(resampled_means, 95.0)),
            "p99": float(np.percentile(resampled_means, 99.0)),
        },
    }


def classify_delta(delta: float, threshold: float = 0.010) -> str:
    if delta >= threshold:
        return "benefit"
    elif delta <= -threshold:
        return "harm"
    else:
        return "neutral"


def analyze():
    # 1. Ingest data
    canonical_rows = []
    # nested: fold -> recipe -> event_id -> seed -> record
    nested: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]] = {
        "fold_a": {"R0": {}, "R2": {}, "M2_ONLY": {}},
        "fold_b": {"R0": {}, "R2": {}, "M2_ONLY": {}},
    }

    print("Verifying run outputs and ingesting per-event metrics...")
    for fold_id, fold_dict in RUN_PATHS.items():
        expected_events = set(FOLD_EVAL_EVENTS[fold_id])
        for recipe_id, run_rel_paths in fold_dict.items():
            for run_rel in run_rel_paths:
                run_dir = REPO_ROOT / run_rel
                if not run_dir.exists():
                    raise FileNotFoundError(f"Run directory not found: {run_dir}")

                # Check terminal status
                status_path = run_dir / "terminal_status.json"
                if not status_path.exists():
                    raise FileNotFoundError(f"Missing terminal_status.json in {run_dir}")
                with open(status_path, "r", encoding="utf-8") as f:
                    tstat = json.load(f)
                if tstat.get("status") != "COMPLETED_SCIENTIFIC_RUN" or tstat.get("exit_code") != 0:
                    raise ContractViolationError(
                        f"Run {run_dir.name} is not COMPLETED_SCIENTIFIC_RUN with exit_code 0 (status: {tstat})"
                    )

                # Read preflight
                pref_path = run_dir / "preflight_manifest.json"
                with open(pref_path, "r", encoding="utf-8") as f:
                    pref = json.load(f)
                seed = pref["partition"]["seed"]
                run_id = pref["run_id"]

                # Read event metrics
                metrics_path = run_dir / "event_metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing event_metrics.json in {run_dir}")
                with open(metrics_path, "r", encoding="utf-8") as f:
                    em = json.load(f)

                if em.get("n_events_evaluated") != 9 or len(em.get("per_event", [])) != 9:
                    raise ContractViolationError(f"Run {run_id} does not have exactly 9 evaluated events")

                observed_events = set()
                for e_rec in em["per_event"]:
                    eid = int(e_rec["event_id"])
                    observed_events.add(eid)

                    row = {
                        "run_id": run_id,
                        "recipe_id": recipe_id,
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

                    if eid not in nested[fold_id][recipe_id]:
                        nested[fold_id][recipe_id][eid] = {}
                    nested[fold_id][recipe_id][eid][seed] = row

                if observed_events != expected_events:
                    raise ContractViolationError(
                        f"Run {run_id} observed events {observed_events} != expected {expected_events}"
                    )

    print(f"Ingested {len(canonical_rows)} canonical records across 18 runs (162 total event evaluations).")

    # 2. Write canonical records CSV
    out_dir = REPO_ROOT / "outputs" / "robust_source_training" / "ablation_m2_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_csv_path = out_dir / "m2_only_ablation_canonical_records.csv"
    fieldnames = [
        "run_id", "recipe_id", "fold_id", "seed", "event_id",
        "sample_count", "valid_pixel_count", "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "iou"
    ]
    with open(canonical_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(canonical_rows)
    print(f"Wrote canonical records to {canonical_csv_path}")

    # 3. Compute per-event paired averages across seeds
    paired_rows = []
    for fold_id in ["fold_a", "fold_b"]:
        for eid in FOLD_EVAL_EVENTS[fold_id]:
            # R0 averages
            r0_f1_list = [nested[fold_id]["R0"][eid][s]["f1"] for s in [0, 1, 2]]
            r0_p_list = [nested[fold_id]["R0"][eid][s]["precision"] for s in [0, 1, 2]]
            r0_r_list = [nested[fold_id]["R0"][eid][s]["recall"] for s in [0, 1, 2]]
            r0_iou_list = [nested[fold_id]["R0"][eid][s]["iou"] for s in [0, 1, 2]]

            r0_f1 = float(np.mean(r0_f1_list))
            r0_p = float(np.mean(r0_p_list))
            r0_r = float(np.mean(r0_r_list))
            r0_iou = float(np.mean(r0_iou_list))

            # M2_ONLY averages
            m2_f1_list = [nested[fold_id]["M2_ONLY"][eid][s]["f1"] for s in [0, 1, 2]]
            m2_p_list = [nested[fold_id]["M2_ONLY"][eid][s]["precision"] for s in [0, 1, 2]]
            m2_r_list = [nested[fold_id]["M2_ONLY"][eid][s]["recall"] for s in [0, 1, 2]]
            m2_iou_list = [nested[fold_id]["M2_ONLY"][eid][s]["iou"] for s in [0, 1, 2]]

            m2_f1 = float(np.mean(m2_f1_list))
            m2_p = float(np.mean(m2_p_list))
            m2_r = float(np.mean(m2_r_list))
            m2_iou = float(np.mean(m2_iou_list))

            # R2 averages
            r2_f1_list = [nested[fold_id]["R2"][eid][s]["f1"] for s in [0, 1, 2]]
            r2_p_list = [nested[fold_id]["R2"][eid][s]["precision"] for s in [0, 1, 2]]
            r2_r_list = [nested[fold_id]["R2"][eid][s]["recall"] for s in [0, 1, 2]]
            r2_iou_list = [nested[fold_id]["R2"][eid][s]["iou"] for s in [0, 1, 2]]

            r2_f1 = float(np.mean(r2_f1_list))
            r2_p = float(np.mean(r2_p_list))
            r2_r = float(np.mean(r2_r_list))
            r2_iou = float(np.mean(r2_iou_list))

            # Deltas
            delta_m2_vs_r0 = m2_f1 - r0_f1
            delta_r2_vs_m2 = r2_f1 - m2_f1
            delta_r2_vs_r0 = r2_f1 - r0_f1

            paired_rows.append({
                "event_id": eid,
                "fold_id": fold_id,
                "r0_f1_mean": r0_f1,
                "m2_only_f1_mean": m2_f1,
                "r2_f1_mean": r2_f1,
                "delta_f1_m2_only_vs_r0": delta_m2_vs_r0,
                "delta_f1_r2_vs_m2_only": delta_r2_vs_m2,
                "delta_f1_r2_vs_r0": delta_r2_vs_r0,
                "r0_precision_mean": r0_p,
                "m2_only_precision_mean": m2_p,
                "r2_precision_mean": r2_p,
                "r0_recall_mean": r0_r,
                "m2_only_recall_mean": m2_r,
                "r2_recall_mean": r2_r,
                "r0_iou_mean": r0_iou,
                "m2_only_iou_mean": m2_iou,
                "r2_iou_mean": r2_iou,
                "classification_m2_only_vs_r0": classify_delta(delta_m2_vs_r0),
                "classification_r2_vs_m2_only": classify_delta(delta_r2_vs_m2),
                "classification_r2_vs_r0": classify_delta(delta_r2_vs_r0),
                "r0_f1_seeds": r0_f1_list,
                "m2_only_f1_seeds": m2_f1_list,
                "r2_f1_seeds": r2_f1_list,
            })

    paired_csv_path = out_dir / "m2_only_ablation_per_event_paired.csv"
    paired_fieldnames = [
        "event_id", "fold_id",
        "r0_f1_mean", "m2_only_f1_mean", "r2_f1_mean",
        "delta_f1_m2_only_vs_r0", "delta_f1_r2_vs_m2_only", "delta_f1_r2_vs_r0",
        "r0_precision_mean", "m2_only_precision_mean", "r2_precision_mean",
        "r0_recall_mean", "m2_only_recall_mean", "r2_recall_mean",
        "r0_iou_mean", "m2_only_iou_mean", "r2_iou_mean",
        "classification_m2_only_vs_r0", "classification_r2_vs_m2_only", "classification_r2_vs_r0",
    ]
    with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paired_fieldnames)
        writer.writeheader()
        for r in paired_rows:
            row_dict = {k: r[k] for k in paired_fieldnames}
            writer.writerow(row_dict)
    print(f"Wrote paired per-event records to {paired_csv_path}")

    # 4. Fold-level Macro Metrics across Seeds
    fold_summaries = {}
    for fold_id in ["fold_a", "fold_b"]:
        eval_events = FOLD_EVAL_EVENTS[fold_id]
        fold_summaries[fold_id] = {}
        for recipe_id in ["R0", "M2_ONLY", "R2"]:
            seed_f1s = []
            seed_ps = []
            seed_rs = []
            seed_ious = []
            for s in [0, 1, 2]:
                f1s = [nested[fold_id][recipe_id][eid][s]["f1"] for eid in eval_events]
                ps = [nested[fold_id][recipe_id][eid][s]["precision"] for eid in eval_events]
                rs = [nested[fold_id][recipe_id][eid][s]["recall"] for eid in eval_events]
                ious = [nested[fold_id][recipe_id][eid][s]["iou"] for eid in eval_events]

                seed_f1s.append(float(np.mean(f1s)))
                seed_ps.append(float(np.mean(ps)))
                seed_rs.append(float(np.mean(rs)))
                seed_ious.append(float(np.mean(ious)))

            fold_summaries[fold_id][recipe_id] = {
                "seed_macro_f1": seed_f1s,
                "mean_macro_f1": float(np.mean(seed_f1s)),
                "sd_macro_f1": float(np.std(seed_f1s, ddof=1)),
                "seed_macro_precision": seed_ps,
                "mean_macro_precision": float(np.mean(seed_ps)),
                "seed_macro_recall": seed_rs,
                "mean_macro_recall": float(np.mean(seed_rs)),
                "seed_macro_iou": seed_ious,
                "mean_macro_iou": float(np.mean(seed_ious)),
            }

    # 5. Grand Macro Metrics across 18 unique held-out events
    all_r0_f1 = [p["r0_f1_mean"] for p in paired_rows]
    all_m2_f1 = [p["m2_only_f1_mean"] for p in paired_rows]
    all_r2_f1 = [p["r2_f1_mean"] for p in paired_rows]

    all_r0_p = [p["r0_precision_mean"] for p in paired_rows]
    all_m2_p = [p["m2_only_precision_mean"] for p in paired_rows]
    all_r2_p = [p["r2_precision_mean"] for p in paired_rows]

    all_r0_r = [p["r0_recall_mean"] for p in paired_rows]
    all_m2_r = [p["m2_only_recall_mean"] for p in paired_rows]
    all_r2_r = [p["r2_recall_mean"] for p in paired_rows]

    all_r0_iou = [p["r0_iou_mean"] for p in paired_rows]
    all_m2_iou = [p["m2_only_iou_mean"] for p in paired_rows]
    all_r2_iou = [p["r2_iou_mean"] for p in paired_rows]

    grand_metrics = {
        "R0": {
            "macro_f1": float(np.mean(all_r0_f1)),
            "macro_precision": float(np.mean(all_r0_p)),
            "macro_recall": float(np.mean(all_r0_r)),
            "macro_iou": float(np.mean(all_r0_iou)),
            "worst_event_f1": float(np.min(all_r0_f1)),
            "bottom3_mean_f1": float(np.mean(sorted(all_r0_f1)[:3])),
        },
        "M2_ONLY": {
            "macro_f1": float(np.mean(all_m2_f1)),
            "macro_precision": float(np.mean(all_m2_p)),
            "macro_recall": float(np.mean(all_m2_r)),
            "macro_iou": float(np.mean(all_m2_iou)),
            "worst_event_f1": float(np.min(all_m2_f1)),
            "bottom3_mean_f1": float(np.mean(sorted(all_m2_f1)[:3])),
        },
        "R2": {
            "macro_f1": float(np.mean(all_r2_f1)),
            "macro_precision": float(np.mean(all_r2_p)),
            "macro_recall": float(np.mean(all_r2_r)),
            "macro_iou": float(np.mean(all_r2_iou)),
            "worst_event_f1": float(np.min(all_r2_f1)),
            "bottom3_mean_f1": float(np.mean(sorted(all_r2_f1)[:3])),
        },
    }

    # 6. Comparative Statistics & Bootstrap
    deltas_m2_vs_r0 = [p["delta_f1_m2_only_vs_r0"] for p in paired_rows]
    deltas_r2_vs_m2 = [p["delta_f1_r2_vs_m2_only"] for p in paired_rows]
    deltas_r2_vs_r0 = [p["delta_f1_r2_vs_r0"] for p in paired_rows]

    boot_m2_vs_r0 = run_paired_bootstrap(deltas_m2_vs_r0)
    boot_r2_vs_m2 = run_paired_bootstrap(deltas_r2_vs_m2)
    boot_r2_vs_r0 = run_paired_bootstrap(deltas_r2_vs_r0)

    counts_m2_vs_r0 = {
        "benefit": sum(1 for p in paired_rows if p["classification_m2_only_vs_r0"] == "benefit"),
        "harm": sum(1 for p in paired_rows if p["classification_m2_only_vs_r0"] == "harm"),
        "neutral": sum(1 for p in paired_rows if p["classification_m2_only_vs_r0"] == "neutral"),
    }
    counts_r2_vs_m2 = {
        "benefit": sum(1 for p in paired_rows if p["classification_r2_vs_m2_only"] == "benefit"),
        "harm": sum(1 for p in paired_rows if p["classification_r2_vs_m2_only"] == "harm"),
        "neutral": sum(1 for p in paired_rows if p["classification_r2_vs_m2_only"] == "neutral"),
    }
    counts_r2_vs_r0 = {
        "benefit": sum(1 for p in paired_rows if p["classification_r2_vs_r0"] == "benefit"),
        "harm": sum(1 for p in paired_rows if p["classification_r2_vs_r0"] == "harm"),
        "neutral": sum(1 for p in paired_rows if p["classification_r2_vs_r0"] == "neutral"),
    }

    comparison_summary = {
        "M2_ONLY_vs_R0": {
            "description": "Effect of class weighting alone without event-balanced sampling",
            "delta_macro_f1": grand_metrics["M2_ONLY"]["macro_f1"] - grand_metrics["R0"]["macro_f1"],
            "delta_bottom3_f1": grand_metrics["M2_ONLY"]["bottom3_mean_f1"] - grand_metrics["R0"]["bottom3_mean_f1"],
            "delta_worst_f1": grand_metrics["M2_ONLY"]["worst_event_f1"] - grand_metrics["R0"]["worst_event_f1"],
            "delta_macro_precision": grand_metrics["M2_ONLY"]["macro_precision"] - grand_metrics["R0"]["macro_precision"],
            "delta_macro_recall": grand_metrics["M2_ONLY"]["macro_recall"] - grand_metrics["R0"]["macro_recall"],
            "delta_macro_iou": grand_metrics["M2_ONLY"]["macro_iou"] - grand_metrics["R0"]["macro_iou"],
            "event_counts_threshold_0_010": counts_m2_vs_r0,
            "bootstrap_ci_95": boot_m2_vs_r0["ci_95"],
            "bootstrap_p_value": boot_m2_vs_r0["one_sided_p_value_h0_delta_le_0"],
        },
        "R2_vs_M2_ONLY": {
            "description": "Added value of event-balanced sampling on top of class weighting",
            "delta_macro_f1": grand_metrics["R2"]["macro_f1"] - grand_metrics["M2_ONLY"]["macro_f1"],
            "delta_bottom3_f1": grand_metrics["R2"]["bottom3_mean_f1"] - grand_metrics["M2_ONLY"]["bottom3_mean_f1"],
            "delta_worst_f1": grand_metrics["R2"]["worst_event_f1"] - grand_metrics["M2_ONLY"]["worst_event_f1"],
            "delta_macro_precision": grand_metrics["R2"]["macro_precision"] - grand_metrics["M2_ONLY"]["macro_precision"],
            "delta_macro_recall": grand_metrics["R2"]["macro_recall"] - grand_metrics["M2_ONLY"]["macro_recall"],
            "delta_macro_iou": grand_metrics["R2"]["macro_iou"] - grand_metrics["M2_ONLY"]["macro_iou"],
            "event_counts_threshold_0_010": counts_r2_vs_m2,
            "bootstrap_ci_95": boot_r2_vs_m2["ci_95"],
            "bootstrap_p_value": boot_r2_vs_m2["one_sided_p_value_h0_delta_le_0"],
        },
        "R2_vs_R0": {
            "description": "Full combined system (M1 + M2) vs baseline",
            "delta_macro_f1": grand_metrics["R2"]["macro_f1"] - grand_metrics["R0"]["macro_f1"],
            "delta_bottom3_f1": grand_metrics["R2"]["bottom3_mean_f1"] - grand_metrics["R0"]["bottom3_mean_f1"],
            "delta_worst_f1": grand_metrics["R2"]["worst_event_f1"] - grand_metrics["R0"]["worst_event_f1"],
            "delta_macro_precision": grand_metrics["R2"]["macro_precision"] - grand_metrics["R0"]["macro_precision"],
            "delta_macro_recall": grand_metrics["R2"]["macro_recall"] - grand_metrics["R0"]["macro_recall"],
            "delta_macro_iou": grand_metrics["R2"]["macro_iou"] - grand_metrics["R0"]["macro_iou"],
            "event_counts_threshold_0_010": counts_r2_vs_r0,
            "bootstrap_ci_95": boot_r2_vs_r0["ci_95"],
            "bootstrap_p_value": boot_r2_vs_r0["one_sided_p_value_h0_delta_le_0"],
        },
    }

    full_summary = {
        "$schema_version": "1.0.0",
        "analysis_name": "ChangeGuard M2_ONLY Supplementary Ablation Analysis",
        "fold_summaries": fold_summaries,
        "grand_metrics": grand_metrics,
        "comparisons": comparison_summary,
        "bootstrap_details": {
            "M2_ONLY_vs_R0": boot_m2_vs_r0,
            "R2_vs_M2_ONLY": boot_r2_vs_m2,
            "R2_vs_R0": boot_r2_vs_r0,
        },
    }

    summary_json_path = out_dir / "m2_only_ablation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
        f.write("\n")
    print(f"Wrote full ablation summary to {summary_json_path}")

    # Generate high-resolution publication plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Per-event paired differences
    events = [str(r["event_id"]) for r in paired_rows]
    deltas_m2_r0 = [r["delta_f1_m2_only_vs_r0"] for r in paired_rows]
    deltas_r2_m2 = [r["delta_f1_r2_vs_m2_only"] for r in paired_rows]

    x = np.arange(len(events))
    width = 0.38

    ax1.bar(x - width/2, deltas_m2_r0, width, label="M2_ONLY - R0 (Weighting alone)", color="#d95f02", alpha=0.85)
    ax1.bar(x + width/2, deltas_r2_m2, width, label="R2 - M2_ONLY (+ Balancing)", color="#2b83ba", alpha=0.85)

    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.axhline(0.010, color="gray", linestyle="--", linewidth=0.7, label=r"Practical Band ($\pm 0.010$)")
    ax1.axhline(-0.010, color="gray", linestyle="--", linewidth=0.7)

    # Highlight fold boundary
    ax1.axvline(8.5, color="purple", linestyle=":", linewidth=1.2, label="Fold A / Fold B Boundary")

    ax1.set_xticks(x)
    ax1.set_xticklabels(events, rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel(r"$\Delta$ F1 (Seed-Averaged)", fontsize=11)
    ax1.set_title("Per-Event Paired Differences (18 Held-Out Events)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2: Contrasts with 95% bootstrap CIs
    contrasts = [
        "M2_ONLY vs R0\n(Weighting Alone)",
        "R2 vs M2_ONLY\n(+ Balancing)",
        "R2 vs R0\n(Combined M1+M2)",
    ]
    diff_means = [
        comparison_summary["M2_ONLY_vs_R0"]["delta_macro_f1"],
        comparison_summary["R2_vs_M2_ONLY"]["delta_macro_f1"],
        comparison_summary["R2_vs_R0"]["delta_macro_f1"],
    ]
    ci_lows = [
        comparison_summary["M2_ONLY_vs_R0"]["bootstrap_ci_95"][0],
        comparison_summary["R2_vs_M2_ONLY"]["bootstrap_ci_95"][0],
        comparison_summary["R2_vs_R0"]["bootstrap_ci_95"][0],
    ]
    ci_ups = [
        comparison_summary["M2_ONLY_vs_R0"]["bootstrap_ci_95"][1],
        comparison_summary["R2_vs_M2_ONLY"]["bootstrap_ci_95"][1],
        comparison_summary["R2_vs_R0"]["bootstrap_ci_95"][1],
    ]
    yerr = [
        [m - l for m, l in zip(diff_means, ci_lows)],
        [u - m for m, u in zip(diff_means, ci_ups)],
    ]

    colors = ["#d95f02", "#2b83ba", "#008837"]
    for i in range(len(contrasts)):
        ax2.errorbar(
            contrasts[i], diff_means[i],
            yerr=[[yerr[0][i]], [yerr[1][i]]],
            fmt="o", color=colors[i], ecolor=colors[i],
            elinewidth=2.5, capsize=6, markersize=8
        )
        ax2.annotate(
            f"{diff_means[i]:+.4f}\n[{ci_lows[i]:+.4f}, {ci_ups[i]:+.4f}]",
            (i, diff_means[i] + (0.005 if diff_means[i] >= 0 else -0.010)),
            ha="center", fontsize=8.5, fontweight="bold", color=colors[i]
        )

    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.axhline(0.010, color="gray", linestyle="--", linewidth=0.7)
    ax2.axhline(-0.010, color="gray", linestyle="--", linewidth=0.7)
    ax2.set_ylabel(r"Grand Event-Macro $\Delta$ F1", fontsize=11)
    ax2.set_title(r"Grand Mean $\Delta$ F1 $\pm$ 95% Bootstrap CI", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig_path = out_dir / "m2_only_ablation_figure.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Wrote ablation figure to {fig_path}")

    return full_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze M2_ONLY supplementary ablation runs")
    args = parser.parse_args()
    summary = analyze()
    print("\n=== ABLATION ANALYSIS COMPLETE ===")
    print(f"R0 Grand Macro F1:       {summary['grand_metrics']['R0']['macro_f1']:.4f}")
    print(f"M2_ONLY Grand Macro F1:  {summary['grand_metrics']['M2_ONLY']['macro_f1']:.4f}")
    print(f"R2 Grand Macro F1:       {summary['grand_metrics']['R2']['macro_f1']:.4f}")
    print("\n--- M2_ONLY vs R0 (Class Weighting Alone) ---")
    c1 = summary["comparisons"]["M2_ONLY_vs_R0"]
    print(f"Delta Grand Macro F1:    {c1['delta_macro_f1']:+.4f}")
    print(f"Delta Bottom-3 F1:       {c1['delta_bottom3_f1']:+.4f}")
    print(f"95% CI:                  [{c1['bootstrap_ci_95'][0]:+.4f}, {c1['bootstrap_ci_95'][1]:+.4f}]")
    print(f"Event Counts (B/H/N):    {c1['event_counts_threshold_0_010']['benefit']}/{c1['event_counts_threshold_0_010']['harm']}/{c1['event_counts_threshold_0_010']['neutral']}")
    print("\n--- R2 vs M2_ONLY (Value of Event-Balanced Sampling) ---")
    c2 = summary["comparisons"]["R2_vs_M2_ONLY"]
    print(f"Delta Grand Macro F1:    {c2['delta_macro_f1']:+.4f}")
    print(f"Delta Bottom-3 F1:       {c2['delta_bottom3_f1']:+.4f}")
    print(f"95% CI:                  [{c2['bootstrap_ci_95'][0]:+.4f}, {c2['bootstrap_ci_95'][1]:+.4f}]")
    print(f"Event Counts (B/H/N):    {c2['event_counts_threshold_0_010']['benefit']}/{c2['event_counts_threshold_0_010']['harm']}/{c2['event_counts_threshold_0_010']['neutral']}")


if __name__ == "__main__":
    main()

