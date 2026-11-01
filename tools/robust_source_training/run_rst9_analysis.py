#!/usr/bin/env python3
"""Statistical analysis engine for ChangeGuard RST9 Historical-Target Extension.

Compares deployed Recipe R2 models against matched baseline Recipe R0 models across
the 16 historical target events (6 dev + 10 former test events).
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_DEV_EVENTS: List[int] = [279, 437, 520, 559, 1111003, 1111008]
HISTORICAL_TEST_EVENTS: List[int] = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
ALL_HISTORICAL_TARGET_EVENTS: List[int] = HISTORICAL_DEV_EVENTS + HISTORICAL_TEST_EVENTS


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes two-sided bootstrap confidence interval on mean delta."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(deltas[indices])

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    mean_val = float(np.mean(boot_means))
    return mean_val, lower, upper


def run_rst9_comparative_analysis(
    inference_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Loads all R0 and R2 target event metrics and computes comprehensive statistical comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load run data
    runs_data: Dict[str, Dict[str, Any]] = {}
    expected_runs = [
        ("R0", 0, "r0_snunet_all27_s0"),
        ("R0", 1, "r0_snunet_all27_s1"),
        ("R0", 2, "r0_snunet_all27_s2"),
        ("R2", 0, "rst9_r2_snunet_all27_s0_v1"),
        ("R2", 1, "rst9_r2_snunet_all27_s1_v1"),
        ("R2", 2, "rst9_r2_snunet_all27_s2_v1"),
    ]

    for recipe, seed, run_id in expected_runs:
        run_file = inference_dir / run_id / "event_metrics.json"
        if not run_file.is_file():
            raise FileNotFoundError(f"Missing event metrics file: {run_file}")
        with open(run_file, "r", encoding="utf-8") as f:
            runs_data[run_id] = json.load(f)

    # 2. Extract per-seed and per-event summaries
    summary_by_seed: Dict[int, Dict[str, Any]] = {}
    for s in (0, 1, 2):
        r0_id = f"r0_snunet_all27_s{s}"
        r2_id = f"rst9_r2_snunet_all27_s{s}_v1"
        r0_run = runs_data[r0_id]
        r2_run = runs_data[r2_id]

        r0_grand = r0_run["grand_macro"]["f1"]
        r2_grand = r2_run["grand_macro"]["f1"]
        delta_grand = r2_grand - r0_grand

        r0_dev = r0_run["dev_macro"]["f1"]
        r2_dev = r2_run["dev_macro"]["f1"]
        delta_dev = r2_dev - r0_dev

        r0_test = r0_run["test_macro"]["f1"]
        r2_test = r2_run["test_macro"]["f1"]
        delta_test = r2_test - r0_test

        r0_b3 = r0_run["bottom3_tail"]["mean_f1"]
        r2_b3 = r2_run["bottom3_tail"]["mean_f1"]
        delta_b3 = r2_b3 - r0_b3

        summary_by_seed[s] = {
            "r0": {
                "grand_macro_f1": r0_grand,
                "dev_macro_f1": r0_dev,
                "test_macro_f1": r0_test,
                "bottom3_mean_f1": r0_b3,
                "worst_event": r0_run["worst_event"],
            },
            "r2": {
                "grand_macro_f1": r2_grand,
                "dev_macro_f1": r2_dev,
                "test_macro_f1": r2_test,
                "bottom3_mean_f1": r2_b3,
                "worst_event": r2_run["worst_event"],
            },
            "delta": {
                "grand_macro_f1": round(delta_grand, 6),
                "dev_macro_f1": round(delta_dev, 6),
                "test_macro_f1": round(delta_test, 6),
                "bottom3_mean_f1": round(delta_b3, 6),
            },
        }

    # 3. Aggregate grand macro means across seeds
    r0_grand_means = [summary_by_seed[s]["r0"]["grand_macro_f1"] for s in (0, 1, 2)]
    r2_grand_means = [summary_by_seed[s]["r2"]["grand_macro_f1"] for s in (0, 1, 2)]
    mean_r0_grand = float(np.mean(r0_grand_means))
    mean_r2_grand = float(np.mean(r2_grand_means))
    delta_grand_mean = mean_r2_grand - mean_r0_grand

    r0_dev_means = [summary_by_seed[s]["r0"]["dev_macro_f1"] for s in (0, 1, 2)]
    r2_dev_means = [summary_by_seed[s]["r2"]["dev_macro_f1"] for s in (0, 1, 2)]
    mean_r0_dev = float(np.mean(r0_dev_means))
    mean_r2_dev = float(np.mean(r2_dev_means))
    delta_dev_mean = mean_r2_dev - mean_r0_dev

    r0_test_means = [summary_by_seed[s]["r0"]["test_macro_f1"] for s in (0, 1, 2)]
    r2_test_means = [summary_by_seed[s]["r2"]["test_macro_f1"] for s in (0, 1, 2)]
    mean_r0_test = float(np.mean(r0_test_means))
    mean_r2_test = float(np.mean(r2_test_means))
    delta_test_mean = mean_r2_test - mean_r0_test

    r0_b3_means = [summary_by_seed[s]["r0"]["bottom3_mean_f1"] for s in (0, 1, 2)]
    r2_b3_means = [summary_by_seed[s]["r2"]["bottom3_mean_f1"] for s in (0, 1, 2)]
    mean_r0_b3 = float(np.mean(r0_b3_means))
    mean_r2_b3 = float(np.mean(r2_b3_means))
    delta_b3_mean = mean_r2_b3 - mean_r0_b3

    # 4. Per-event average performance across the 3 seeds
    event_table: List[Dict[str, Any]] = []
    event_deltas: List[float] = []

    for eid in ALL_HISTORICAL_TARGET_EVENTS:
        ev_type = "dev" if eid in HISTORICAL_DEV_EVENTS else "test"
        r0_f1s = []
        r2_f1s = []
        r0_precs = []
        r2_precs = []
        r0_recs = []
        r2_recs = []

        for s in (0, 1, 2):
            r0_ev = next(r for r in runs_data[f"r0_snunet_all27_s{s}"]["per_event"] if r["event_id"] == eid)
            r2_ev = next(r for r in runs_data[f"rst9_r2_snunet_all27_s{s}_v1"]["per_event"] if r["event_id"] == eid)
            r0_f1s.append(r0_ev["f1"])
            r2_f1s.append(r2_ev["f1"])
            r0_precs.append(r0_ev["precision"])
            r2_precs.append(r2_ev["precision"])
            r0_recs.append(r0_ev["recall"])
            r2_recs.append(r2_ev["recall"])

        m_r0_f1 = float(np.mean(r0_f1s))
        m_r2_f1 = float(np.mean(r2_f1s))
        d_f1 = m_r2_f1 - m_r0_f1
        event_deltas.append(d_f1)

        event_table.append({
            "event_id": eid,
            "event_type": ev_type,
            "r0_f1_mean": round(m_r0_f1, 6),
            "r2_f1_mean": round(m_r2_f1, 6),
            "delta_f1": round(d_f1, 6),
            "r0_precision_mean": round(float(np.mean(r0_precs)), 6),
            "r2_precision_mean": round(float(np.mean(r2_precs)), 6),
            "r0_recall_mean": round(float(np.mean(r0_recs)), 6),
            "r2_recall_mean": round(float(np.mean(r2_recs)), 6),
        })

    # 5. Bootstrap CI on mean event delta
    deltas_arr = np.array(event_deltas, dtype=np.float64)
    boot_mean, boot_lower, boot_upper = paired_bootstrap_ci(deltas_arr, n_resamples=10000, ci_level=0.95, seed=42)

    # 6. Evaluation criteria evaluation
    p1_satisfied = delta_grand_mean > 0.0
    p2_satisfied = delta_b3_mean > 0.0
    seed_stability_satisfied = all(summary_by_seed[s]["delta"]["grand_macro_f1"] > 0.0 for s in (0, 1, 2))

    # Verdict selection
    if p1_satisfied and p2_satisfied:
        verdict = "RST9 COMPLETE — DEPLOYABLE MODELS AND POSITIVE HISTORICAL EXTENSION"
        decision_code = "POSITIVE_HISTORICAL_EXTENSION"
    elif delta_grand_mean < -0.005 and delta_b3_mean < -0.005:
        verdict = "RST9 COMPLETE — DEPLOYABLE MODELS AND NEGATIVE HISTORICAL EXTENSION"
        decision_code = "NEGATIVE_HISTORICAL_EXTENSION"
    else:
        verdict = "RST9 COMPLETE — DEPLOYABLE MODELS AND INCONCLUSIVE HISTORICAL EXTENSION"
        decision_code = "INCONCLUSIVE_HISTORICAL_EXTENSION"

    analysis_result = {
        "work_package": "RST9",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "terminal_verdict": verdict,
        "decision_code": decision_code,
        "criteria": {
            "P1_grand_macro_improvement": {
                "metric": "grand_event_macro_f1",
                "r0_mean": round(mean_r0_grand, 6),
                "r2_mean": round(mean_r2_grand, 6),
                "delta": round(delta_grand_mean, 6),
                "satisfied": p1_satisfied,
            },
            "P2_tail_reliability_improvement": {
                "metric": "bottom3_mean_f1",
                "r0_mean": round(mean_r0_b3, 6),
                "r2_mean": round(mean_r2_b3, 6),
                "delta": round(delta_b3_mean, 6),
                "satisfied": p2_satisfied,
            },
            "P3_seed_stability": {
                "seed_deltas": {s: summary_by_seed[s]["delta"]["grand_macro_f1"] for s in (0, 1, 2)},
                "all_seeds_positive": seed_stability_satisfied,
                "satisfied": seed_stability_satisfied,
            },
        },
        "partition_summaries": {
            "dev_macro": {
                "r0_mean": round(mean_r0_dev, 6),
                "r2_mean": round(mean_r2_dev, 6),
                "delta": round(delta_dev_mean, 6),
            },
            "test_macro": {
                "r0_mean": round(mean_r0_test, 6),
                "r2_mean": round(mean_r2_test, 6),
                "delta": round(delta_test_mean, 6),
            },
        },
        "bootstrap_statistics": {
            "n_resamples": 10000,
            "ci_level": 0.95,
            "mean_delta": round(boot_mean, 6),
            "ci_lower": round(boot_lower, 6),
            "ci_upper": round(boot_upper, 6),
            "ci_crosses_zero": bool(boot_lower <= 0.0 <= boot_upper),
        },
        "per_seed_summary": summary_by_seed,
        "per_event_table": event_table,
    }

    # Save JSON result
    out_json = output_dir / "rst9_comparative_analysis_v1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)
    print(f"Saved comparative analysis result to {out_json}")

    return analysis_result


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard RST9 Historical Comparative Analyzer")
    parser.add_argument("--inference_dir", type=str, default="outputs/robust_source_training/rst9_inference", help="Directory containing run outputs")
    parser.add_argument("--output_dir", type=str, default="outputs/robust_source_training/rst9_analysis", help="Output directory")

    args = parser.parse_args()
    inf_dir = (REPO_ROOT / args.inference_dir).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()

    res = run_rst9_comparative_analysis(inf_dir, out_dir)
    print("\n================================================================================")
    print(f"Terminal Verdict: {res['terminal_verdict']}")
    print(f"Grand Macro F1: R0={res['criteria']['P1_grand_macro_improvement']['r0_mean']:.4f} -> R2={res['criteria']['P1_grand_macro_improvement']['r2_mean']:.4f} (Delta: {res['criteria']['P1_grand_macro_improvement']['delta']:+.4f})")
    print(f"Bottom-3 Mean:  R0={res['criteria']['P2_tail_reliability_improvement']['r0_mean']:.4f} -> R2={res['criteria']['P2_tail_reliability_improvement']['r2_mean']:.4f} (Delta: {res['criteria']['P2_tail_reliability_improvement']['delta']:+.4f})")
    print(f"Dev Macro F1:   R0={res['partition_summaries']['dev_macro']['r0_mean']:.4f} -> R2={res['partition_summaries']['dev_macro']['r2_mean']:.4f} (Delta: {res['partition_summaries']['dev_macro']['delta']:+.4f})")
    print(f"Test Macro F1:  R0={res['partition_summaries']['test_macro']['r0_mean']:.4f} -> R2={res['partition_summaries']['test_macro']['r2_mean']:.4f} (Delta: {res['partition_summaries']['test_macro']['delta']:+.4f})")
    print(f"Bootstrap 95% CI: [{res['bootstrap_statistics']['ci_lower']:+.4f}, {res['bootstrap_statistics']['ci_upper']:+.4f}]")
    print("================================================================================")


if __name__ == "__main__":
    main()

