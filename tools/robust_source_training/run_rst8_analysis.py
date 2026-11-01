#!/usr/bin/env python3
"""RST8 Stage-A Architecture Transfer Analysis Tool (SiamUnet_diff).

Executes the frozen confirmatory statistical analysis for ChangeGuard
robust source training Stage A: R0 vs R2 on Fold A and Fold B for seed 0.
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


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_contracts(cfg: Dict[str, Any]):
    forbidden = set(cfg.get("forbidden_target_events", []))
    assert forbidden == set(FORBIDDEN_TARGET_EVENTS), "Forbidden target events mismatch in config"

    events_a = set(cfg["runs"]["fold_a"]["evaluation_events"])
    events_b = set(cfg["runs"]["fold_b"]["evaluation_events"])

    if events_a & forbidden or events_b & forbidden:
        raise ContractViolationError("Analysis config contains forbidden target events")
    if events_a & events_b:
        raise ContractViolationError("Evaluation sets of Fold A and Fold B are not disjoint")

    all_eval = events_a | events_b
    if len(all_eval) != 18:
        raise ContractViolationError(f"Expected exactly 18 evaluation events, got {len(all_eval)}")
    if not all_eval.issubset(set(AUTHORITATIVE_SOURCE_EVENTS)):
        raise ContractViolationError("Evaluation events contain unauthorized event IDs")


def ingest_data(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[int, Dict[str, float]]]]]:
    """Ingests all 4 Stage-A runs and produces canonical records and nested lookup."""
    canonical_rows = []
    nested = {"fold_a": {"r0": {}, "r2": {}}, "fold_b": {"r0": {}, "r2": {}}}

    for fold_name in ["fold_a", "fold_b"]:
        fold_cfg = cfg["runs"][fold_name]
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

    if len(canonical_rows) != 36:
        raise ContractViolationError(f"Expected 36 canonical rows for Stage A, got {len(canonical_rows)}")

    return canonical_rows, nested


def compute_stage_a_statistics(
    cfg: Dict[str, Any],
    nested: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
) -> Dict[str, Any]:
    """Computes all paired per-event deltas, fold macro, grand macro, and tail stats for seed 0."""
    paired_rows = []
    events_all = []

    # Per-event paired calculations (seed 0)
    for fold_name in ["fold_a", "fold_b"]:
        eval_events = cfg["runs"][fold_name]["evaluation_events"]
        for eid in eval_events:
            events_all.append(eid)
            r0_rec = nested[fold_name]["r0"][eid][0]
            r2_rec = nested[fold_name]["r2"][eid][0]

            r0_f1 = r0_rec["f1"]
            r2_f1 = r2_rec["f1"]
            delta_f1 = r2_f1 - r0_f1

            r0_p = r0_rec["precision"]
            r2_p = r2_rec["precision"]
            delta_p = r2_p - r0_p

            r0_r = r0_rec["recall"]
            r2_r = r2_rec["recall"]
            delta_r = r2_r - r0_r

            r0_iou = r0_rec["iou"]
            r2_iou = r2_rec["iou"]
            delta_iou = r2_iou - r0_iou

            paired_rows.append({
                "event_id": eid,
                "fold_id": fold_name,
                "r0_f1_mean": r0_f1,
                "r2_f1_mean": r2_f1,
                "delta_f1": delta_f1,
                "r0_precision_mean": r0_p,
                "r2_precision_mean": r2_p,
                "delta_precision": delta_p,
                "r0_recall_mean": r0_r,
                "r2_recall_mean": r2_r,
                "delta_recall": delta_r,
                "r0_iou_mean": r0_iou,
                "r2_iou_mean": r2_iou,
                "delta_iou": delta_iou,
            })

    # Fold-level macro metrics
    fold_summaries = {}
    for fold_name in ["fold_a", "fold_b"]:
        eval_events = cfg["runs"][fold_name]["evaluation_events"]
        fold_summaries[fold_name] = {}
        for recipe in ["r0", "r2"]:
            f1s = [nested[fold_name][recipe][eid][0]["f1"] for eid in eval_events]
            ps = [nested[fold_name][recipe][eid][0]["precision"] for eid in eval_events]
            rs = [nested[fold_name][recipe][eid][0]["recall"] for eid in eval_events]
            ious = [nested[fold_name][recipe][eid][0]["iou"] for eid in eval_events]

            fold_summaries[fold_name][recipe] = {
                "macro_f1": float(np.mean(f1s)),
                "macro_precision": float(np.mean(ps)),
                "macro_recall": float(np.mean(rs)),
                "macro_iou": float(np.mean(ious)),
            }

    # Grand mean calculations (equal weight across the 18 unique held-out events)
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

    # Tail metrics (across 18 events)
    worst_event_r0 = float(np.min(all_f1_r0))
    worst_event_r2 = float(np.min(all_f1_r2))
    delta_worst_event = worst_event_r2 - worst_event_r0

    bottom3_r0 = float(np.mean(sorted(all_f1_r0)[:3]))
    bottom3_r2 = float(np.mean(sorted(all_f1_r2)[:3]))
    delta_bottom3 = bottom3_r2 - bottom3_r0

    # Single-event driver test (for P1)
    max_delta_idx = int(np.argmax(all_delta_f1))
    max_delta_event = paired_rows[max_delta_idx]["event_id"]
    max_delta_val = all_delta_f1[max_delta_idx]
    remaining_deltas = [d for i, d in enumerate(all_delta_f1) if i != max_delta_idx]
    delta_excluding_max = float(np.mean(remaining_deltas))
    gain_not_driven_by_one = bool(delta_excluding_max > 0.0)

    # Precision/recall collapse test (for P1)
    dual_drop_events = [
        p["event_id"] for p in paired_rows
        if p["delta_precision"] <= -0.030 and p["delta_recall"] <= -0.030
    ]
    no_pr_collapse = bool(
        grand_mean_delta_p > -0.050 and
        grand_mean_delta_r > -0.050 and
        len(dual_drop_events) <= 4
    )

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
        },
        "single_event_driver_test": {
            "max_delta_event_id": max_delta_event,
            "max_delta_value": float(max_delta_val),
            "delta_excluding_max_event": delta_excluding_max,
            "gain_not_driven_by_one_event": gain_not_driven_by_one,
        },
        "precision_recall_collapse_test": {
            "macro_precision_delta": grand_mean_delta_p,
            "macro_recall_delta": grand_mean_delta_r,
            "dual_drop_events_count": len(dual_drop_events),
            "dual_drop_event_ids": dual_drop_events,
            "no_widespread_collapse": no_pr_collapse,
        },
        "seed_stability_test": {
            "status": "NOT_EVALUABLE_AT_STAGE_A",
            "reason": "Stage A runs seed 0 only. Multi-seed stability evaluated in Stage B if authorized.",
        },
    }

    return stats


def run_paired_bootstrap(
    paired_rows: List[Dict[str, Any]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
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


def evaluate_stage_a_decision_criteria(
    stats: Dict[str, Any],
    bootstrap: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Applies the frozen Stage-A decision rules for P1 and P2."""
    thresholds = cfg["criteria_thresholds"]
    grand_delta = stats["grand_metrics"]["delta"]["f1"]
    bottom3_delta = stats["grand_metrics"]["delta"]["bottom3_mean_f1"]
    worst_delta = stats["grand_metrics"]["delta"]["worst_event_f1"]

    # P1: Average Improvement
    p1_cond1 = bool(grand_delta >= thresholds["p1_min_grand_mean_delta"])
    p1_cond2 = bool(stats["single_event_driver_test"]["gain_not_driven_by_one_event"])
    p1_cond3 = bool(stats["precision_recall_collapse_test"]["no_widespread_collapse"])
    p1_satisfied = bool(p1_cond1 and p1_cond2 and p1_cond3)

    # P2: Tail Reliability Improvement
    p2_cond1 = bool(
        bottom3_delta >= thresholds["p2_min_tail_delta"] or
        worst_delta >= thresholds["p2_min_tail_delta"]
    )
    p2_cond2 = bool(grand_delta >= thresholds["p2_min_grand_mean_delta"])
    p2_satisfied = bool(p2_cond1 and p2_cond2)

    # Stage-A Route Verdict (per Protocol Section 4.3 and continuation rules)
    if p1_satisfied or p2_satisfied:
        verdict = "RST8 STAGE A COMPLETE — EXPANSION AUTHORIZED"
        decision_code = "EXPANSION_AUTHORIZED"
    elif round(grand_delta, 3) <= -0.010 or (not p1_satisfied and not p2_satisfied and (bottom3_delta < 0.0 or grand_delta < 0.0)):
        verdict = "RST8 STAGE A COMPLETE — STOP AT SEED 0"
        decision_code = "STOP_AT_SEED_0"
    else:
        verdict = "RST8 STAGE A COMPLETE — INCONCLUSIVE"
        decision_code = "INCONCLUSIVE"

    return {
        "criteria": {
            "P1_average_improvement": {
                "satisfied": p1_satisfied,
                "conditions": {
                    "grand_mean_delta_ge_threshold": {
                        "satisfied": p1_cond1,
                        "observed": grand_delta,
                        "threshold": thresholds["p1_min_grand_mean_delta"],
                    },
                    "not_driven_by_one_event": {
                        "satisfied": p1_cond2,
                        "delta_excluding_max": stats["single_event_driver_test"]["delta_excluding_max_event"],
                    },
                    "no_precision_recall_collapse": {
                        "satisfied": p1_cond3,
                        "macro_precision_delta": stats["precision_recall_collapse_test"]["macro_precision_delta"],
                        "macro_recall_delta": stats["precision_recall_collapse_test"]["macro_recall_delta"],
                        "dual_drop_events_count": stats["precision_recall_collapse_test"]["dual_drop_events_count"],
                    },
                },
            },
            "P2_tail_reliability_improvement": {
                "satisfied": p2_satisfied,
                "conditions": {
                    "tail_gain_ge_threshold": {
                        "satisfied": p2_cond1,
                        "observed_bottom3_delta": bottom3_delta,
                        "observed_worst_delta": worst_delta,
                        "threshold": thresholds["p2_min_tail_delta"],
                    },
                    "average_performance_not_degraded": {
                        "satisfied": p2_cond2,
                        "observed_grand_delta": grand_delta,
                        "threshold": thresholds["p2_min_grand_mean_delta"],
                    },
                },
            },
            "P3_seed_stability": {
                "satisfied": False,
                "status": "NOT_EVALUABLE_AT_STAGE_A",
                "note": "Multi-seed stability evaluated in Stage B if authorized.",
            },
        },
        "terminal_verdict": verdict,
        "decision_code": decision_code,
    }


def export_artifacts(
    out_dir: Path,
    canonical_rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    bootstrap: Dict[str, Any],
    decision: Dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Canonical long-form table (36 rows)
    csv_path = out_dir / "canonical_event_results_v1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "method", "fold_id", "seed", "event_id",
            "sample_count", "valid_pixel_count", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "iou"
        ])
        writer.writeheader()
        writer.writerows(canonical_rows)

    with open(out_dir / "canonical_event_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(canonical_rows, f, indent=2)

    # 2. Paired event deltas table (18 rows)
    paired_csv = out_dir / "paired_event_deltas_v1.csv"
    with open(paired_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "fold_id",
            "r0_f1_mean", "r2_f1_mean", "delta_f1",
            "r0_precision_mean", "r2_precision_mean", "delta_precision",
            "r0_recall_mean", "r2_recall_mean", "delta_recall",
            "r0_iou_mean", "r2_iou_mean", "delta_iou",
        ])
        writer.writeheader()
        writer.writerows(stats["paired_rows"])

    with open(out_dir / "paired_event_deltas_v1.json", "w", encoding="utf-8") as f:
        json.dump(stats["paired_rows"], f, indent=2)

    # 3. Bootstrap distribution
    with open(out_dir / "bootstrap_distribution_v1.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap, f, indent=2)

    # 4. Summary and decision
    summary = {
        "$schema_version": "1.0.0",
        "created_at_utc": "2026-10-28T13:48:00Z",
        "backbone": "SiamUnet_diff",
        "stage": "STAGE_A_SEED_0",
        "canonical_rows_count": len(canonical_rows),
        "paired_events_count": len(stats["paired_rows"]),
        "grand_metrics": stats["grand_metrics"],
        "fold_summaries": stats["fold_summaries"],
        "single_event_driver_test": stats["single_event_driver_test"],
        "precision_recall_collapse_test": stats["precision_recall_collapse_test"],
        "seed_stability_test": stats["seed_stability_test"],
        "bootstrap": bootstrap,
        "decision": decision,
    }
    with open(out_dir / "stage_a_summary_v1.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Artifacts exported to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run ChangeGuard RST8 Stage-A analysis.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robust_source_training/rst8_stage_a_analysis_config_v1.json",
        help="Path to analysis configuration JSON.",
    )
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    cfg = load_config(cfg_path)
    validate_contracts(cfg)

    canonical_rows, nested = ingest_data(cfg)
    stats = compute_stage_a_statistics(cfg, nested)

    boot_cfg = cfg["statistical_parameters"]
    bootstrap = run_paired_bootstrap(
        stats["paired_rows"],
        n_resamples=boot_cfg.get("bootstrap_resamples", 10000),
        seed=boot_cfg.get("bootstrap_seed", 42),
    )

    decision = evaluate_stage_a_decision_criteria(stats, bootstrap, cfg)

    out_dir = REPO_ROOT / cfg.get("output_directory", "outputs/robust_source_training/rst8")
    export_artifacts(out_dir, canonical_rows, stats, bootstrap, decision)

    print(f"Stage A Verdict: {decision['terminal_verdict']}")


if __name__ == "__main__":
    main()

