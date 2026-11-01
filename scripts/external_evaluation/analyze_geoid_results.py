#!/usr/bin/env python3
"""
Statistical analysis and reporting script for ChangeGuard external evaluation on GEOID-Flood.
Aggregates R0 and R2 across 3 random seeds, computes paired differences, bootstrap CIs,
event breakdowns, sensitivity groupings, LaTeX table snippets, and publication figures.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs/external_evaluation"

MODEL_IDS = ["R0_s0", "R0_s1", "R0_s2", "R2_s0", "R2_s1", "R2_s2"]


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42, ci: float = 95.0) -> Dict[str, float]:
    """Computes empirical bootstrap confidence interval on mean paired difference."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (100.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha))
    high = float(np.percentile(boot_means, 100.0 - alpha))
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "ci_level": ci,
    }


def main():
    print("Loading GEOID-Flood evaluation results...")
    results = {}
    for m_id in MODEL_IDS:
        res_file = OUTPUT_DIR / f"{m_id}_geoid_eval_results.json"
        if not res_file.is_file():
            print(f"Error: missing result file {res_file}")
            sys.exit(1)
        with open(res_file, "r", encoding="utf-8") as f:
            results[m_id] = json.load(f)

    # 1. Seed-level metrics table
    seed_rows = []
    for m_id in MODEL_IDS:
        r = results[m_id]
        m9 = r["macro_metrics_9_events"]
        m8 = r["macro_metrics_8_events_grouped"]
        pooled = r["pooled_metrics"]
        seed_rows.append({
            "model_id": m_id,
            "recipe": r["recipe"],
            "seed": r["seed"],
            "macro_f1_9events": m9["macro_f1"],
            "macro_precision_9events": m9["macro_precision"],
            "macro_recall_9events": m9["macro_recall"],
            "macro_iou_9events": m9["macro_iou"],
            "macro_f1_8grouped": m8["macro_f1"],
            "macro_precision_8grouped": m8["macro_precision"],
            "macro_recall_8grouped": m8["macro_recall"],
            "macro_iou_8grouped": m8["macro_iou"],
            "pooled_f1": pooled["pooled_f1"],
            "pooled_precision": pooled["pooled_precision"],
            "pooled_recall": pooled["pooled_recall"],
            "pooled_iou": pooled["pooled_iou"],
            "eval_time_sec": r["evaluation_time_seconds"],
        })

    df_seeds = pd.DataFrame(seed_rows)
    df_seeds.to_csv(OUTPUT_DIR / "geoid_seed_metrics.csv", index=False)
    print(f"Saved seed metrics to {OUTPUT_DIR / 'geoid_seed_metrics.csv'}")

    # 2. Recipe-level summaries
    r0_df = df_seeds[df_seeds["recipe"] == "R0"].sort_values("seed")
    r2_df = df_seeds[df_seeds["recipe"] == "R2"].sort_values("seed")

    r0_f1_9 = r0_df["macro_f1_9events"].values
    r2_f1_9 = r2_df["macro_f1_9events"].values
    delta_f1_9 = r2_f1_9 - r0_f1_9

    r0_f1_8 = r0_df["macro_f1_8grouped"].values
    r2_f1_8 = r2_df["macro_f1_8grouped"].values
    delta_f1_8 = r2_f1_8 - r0_f1_8

    r0_pooled_f1 = r0_df["pooled_f1"].values
    r2_pooled_f1 = r2_df["pooled_f1"].values
    delta_pooled_f1 = r2_pooled_f1 - r0_pooled_f1

    ci_9 = bootstrap_ci(delta_f1_9)
    ci_8 = bootstrap_ci(delta_f1_8)
    ci_pooled = bootstrap_ci(delta_pooled_f1)

    # 3. Event-level breakdown
    event_ids = [e["event_id"] for e in results["R0_s0"]["event_metrics_9_events"]]
    event_rows = []

    for ev_id in event_ids:
        r0_ev_f1s = [
            next(e["f1"] for e in results[f"R0_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id)
            for s in [0, 1, 2]
        ]
        r2_ev_f1s = [
            next(e["f1"] for e in results[f"R2_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id)
            for s in [0, 1, 2]
        ]
        r0_mean = float(np.mean(r0_ev_f1s))
        r0_std = float(np.std(r0_ev_f1s, ddof=1))
        r2_mean = float(np.mean(r2_ev_f1s))
        r2_std = float(np.std(r2_ev_f1s, ddof=1))
        diff = r2_mean - r0_mean

        status = "Benefit" if diff > 0.0050 else ("Harm" if diff < -0.0050 else "Neutral")

        # Get metadata from first run
        em0 = next(e for e in results["R0_s0"]["event_metrics_9_events"] if e["event_id"] == ev_id)

        event_rows.append({
            "event_id": ev_id,
            "valid_pixels": em0["valid_pixels"],
            "flood_pixels": em0["gt_flood_pixels"],
            "flood_pct": (em0["gt_flood_pixels"] / em0["valid_pixels"] * 100) if em0["valid_pixels"] > 0 else 0.0,
            "r0_f1_mean": r0_mean,
            "r0_f1_std": r0_std,
            "r2_f1_mean": r2_mean,
            "r2_f1_std": r2_std,
            "delta_f1": diff,
            "status": status,
        })

    df_events = pd.DataFrame(event_rows)
    df_events.to_csv(OUTPUT_DIR / "geoid_event_breakdown.csv", index=False)
    print(f"Saved event breakdown to {OUTPUT_DIR / 'geoid_event_breakdown.csv'}")

    n_benefit = int((df_events["status"] == "Benefit").sum())
    n_harm = int((df_events["status"] == "Harm").sum())
    n_neutral = int((df_events["status"] == "Neutral").sum())

    # 4. Summary JSON
    summary_data = {
        "benchmark": "GEOID-Flood held-out (v1.1)",
        "evaluation_scope": "Zero-shot external evaluation of frozen all-27-source models",
        "models_evaluated": {
            "R0": ["outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32",
                   "outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32",
                   "outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32"],
            "R2": ["outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1",
                   "outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s1_v1",
                   "outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s2_v1"],
        },
        "canonical_9_events": {
            "r0_macro_f1_mean": float(r0_f1_9.mean()),
            "r0_macro_f1_std": float(r0_f1_9.std(ddof=1)),
            "r2_macro_f1_mean": float(r2_f1_9.mean()),
            "r2_macro_f1_std": float(r2_f1_9.std(ddof=1)),
            "delta_macro_f1_mean": float(delta_f1_9.mean()),
            "delta_macro_f1_std": float(delta_f1_9.std(ddof=1)),
            "delta_macro_f1_bootstrap_95ci": [ci_9["ci_low"], ci_9["ci_high"]],
            "benefit_events_count": n_benefit,
            "harm_events_count": n_harm,
            "neutral_events_count": n_neutral,
        },
        "grouped_8_events_sensitivity": {
            "r0_macro_f1_mean": float(r0_f1_8.mean()),
            "r0_macro_f1_std": float(r0_f1_8.std(ddof=1)),
            "r2_macro_f1_mean": float(r2_f1_8.mean()),
            "r2_macro_f1_std": float(r2_f1_8.std(ddof=1)),
            "delta_macro_f1_mean": float(delta_f1_8.mean()),
            "delta_macro_f1_std": float(delta_f1_8.std(ddof=1)),
            "delta_macro_f1_bootstrap_95ci": [ci_8["ci_low"], ci_8["ci_high"]],
        },
        "pooled_dataset_metrics": {
            "r0_pooled_f1_mean": float(r0_pooled_f1.mean()),
            "r2_pooled_f1_mean": float(r2_pooled_f1.mean()),
            "delta_pooled_f1_mean": float(delta_pooled_f1.mean()),
            "delta_pooled_f1_bootstrap_95ci": [ci_pooled["ci_low"], ci_pooled["ci_high"]],
        },
    }

    with open(OUTPUT_DIR / "geoid_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to {OUTPUT_DIR / 'geoid_evaluation_summary.json'}")

    # 5. Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})

    # Subplot 1: Macro F1 comparison across seeds
    x_r0 = [0, 0, 0]
    x_r2 = [1, 1, 1]
    axes[0].scatter(x_r0, r0_f1_9, color="#1f77b4", s=60, zorder=3, label="R0 Seeds")
    axes[0].scatter(x_r2, r2_f1_9, color="#2ca02c", s=60, zorder=3, label="R2 Seeds")
    for s in range(3):
        axes[0].plot([0, 1], [r0_f1_9[s], r2_f1_9[s]], color="gray", linestyle="--", alpha=0.6)

    axes[0].errorbar(
        [0], [r0_f1_9.mean()], yerr=[r0_f1_9.std(ddof=1)],
        fmt="o", color="#1f77b4", capsize=6, markersize=10, elinewidth=2.5, markeredgewidth=2, label="R0 Mean ± SD"
    )
    axes[0].errorbar(
        [1], [r2_f1_9.mean()], yerr=[r2_f1_9.std(ddof=1)],
        fmt="s", color="#2ca02c", capsize=6, markersize=10, elinewidth=2.5, markeredgewidth=2, label="R2 Mean ± SD"
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Standard (R0)", "Robust (R2)"], fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Held-Out Macro F1", fontsize=12)
    axes[0].set_title(
        f"GEOID-Flood Zero-Shot Macro F1 (9 Events)\n"
        f"R0: {r0_f1_9.mean():.4f} ± {r0_f1_9.std(ddof=1):.4f} | R2: {r2_f1_9.mean():.4f} ± {r2_f1_9.std(ddof=1):.4f}\n"
        f"$\Delta$: {delta_f1_9.mean():+.4f} (95% CI: [{ci_9['ci_low']:+.4f}, {ci_9['ci_high']:+.4f}])",
        fontsize=11
    )
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="best", fontsize=9)

    # Subplot 2: Per-event Delta(R2 - R0)
    ev_names = df_events["event_id"].tolist()
    ev_deltas = df_events["delta_f1"].tolist()
    colors = ["#2ca02c" if d > 0.0050 else ("#d62728" if d < -0.0050 else "#7f7f7f") for d in ev_deltas]

    y_pos = np.arange(len(ev_names))
    bars = axes[1].barh(y_pos, ev_deltas, color=colors, height=0.6, edgecolor="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1.0)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ev_names, fontsize=10)
    axes[1].set_xlabel("$\Delta$ F1 (R2 - R0)", fontsize=12)
    axes[1].set_title(
        f"Event-by-Event Paired Differences $\Delta(R2 - R0)$\n"
        f"Benefit: {n_benefit} | Neutral: {n_neutral} | Harm: {n_harm}",
        fontsize=11
    )
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        x_text = width + (0.002 if width >= 0 else -0.006)
        ha = "left" if width >= 0 else "right"
        axes[1].text(x_text, bar.get_y() + bar.get_height() / 2, f"{width:+.4f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "geoid_r2_vs_r0_comparison_figure.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {fig_path}")

    # 6. LaTeX table snippet
    tex_path = OUTPUT_DIR / "geoid_table_snippet.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% ChangeGuard External Evaluation on GEOID-Flood Held-Out Benchmark\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Zero-shot cross-benchmark evaluation on GEOID-Flood held-out partition (1,301 tiles across 9 independent Copernicus EMS activations). Models trained purely on Kuro Siwo without target adaptation.}\n")
        f.write("\\label{tab:geoid_external_eval}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Recipe & Macro F1 (9 Ev.) & Macro F1 (8 Grp.) & Pooled F1 & Macro IoU \\\\\n\\midrule\n")
        f.write(f"R0 (Standard Source) & {r0_f1_9.mean():.4f} $\\pm$ {r0_f1_9.std(ddof=1):.4f} & {r0_f1_8.mean():.4f} $\\pm$ {r0_f1_8.std(ddof=1):.4f} & {r0_pooled_f1.mean():.4f} $\\pm$ {r0_pooled_f1.std(ddof=1):.4f} & {df_seeds[df_seeds['recipe']=='R0']['macro_iou_9events'].mean():.4f} \\\\\n")
        f.write(f"R2 (Robust Source)   & \\textbf{{{r2_f1_9.mean():.4f}}} $\\pm$ {r2_f1_9.std(ddof=1):.4f} & \\textbf{{{r2_f1_8.mean():.4f}}} $\\pm$ {r2_f1_8.std(ddof=1):.4f} & \\textbf{{{r2_pooled_f1.mean():.4f}}} $\\pm$ {r2_pooled_f1.std(ddof=1):.4f} & \\textbf{{{df_seeds[df_seeds['recipe']=='R2']['macro_iou_9events'].mean():.4f}}} \\\\\n\\midrule\n")
        f.write(f"$\\Delta(\\text{{R2}} - \\text{{R0}})$ & {delta_f1_9.mean():+.4f} & {delta_f1_8.mean():+.4f} & {delta_pooled_f1.mean():+.4f} & {df_seeds[df_seeds['recipe']=='R2']['macro_iou_9events'].mean() - df_seeds[df_seeds['recipe']=='R0']['macro_iou_9events'].mean():+.4f} \\\\\n")
        f.write(f"95\\% Bootstrap CI & [{ci_9['ci_low']:+.4f}, {ci_9['ci_high']:+.4f}] & [{ci_8['ci_low']:+.4f}, {ci_8['ci_high']:+.4f}] & [{ci_pooled['ci_low']:+.4f}, {ci_pooled['ci_high']:+.4f}] & -- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved LaTeX table snippet to {tex_path}")

    # Console output of results
    print("\n" + "=" * 60)
    print("GEOID-FLOOD EXTERNAL EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Canonical 9-Event Macro F1:")
    print(f"  R0: {r0_f1_9.mean():.4f} ± {r0_f1_9.std(ddof=1):.4f}  (seeds: {[f'{x:.4f}' for x in r0_f1_9]})")
    print(f"  R2: {r2_f1_9.mean():.4f} ± {r2_f1_9.std(ddof=1):.4f}  (seeds: {[f'{x:.4f}' for x in r2_f1_9]})")
    print(f"  Δ(R2 - R0): {delta_f1_9.mean():+.4f} (95% CI: [{ci_9['ci_low']:+.4f}, {ci_9['ci_high']:+.4f}])")
    print(f"\nGrouped 8-Event Sensitivity Macro F1 (Iberia combined):")
    print(f"  R0: {r0_f1_8.mean():.4f} ± {r0_f1_8.std(ddof=1):.4f}")
    print(f"  R2: {r2_f1_8.mean():.4f} ± {r2_f1_8.std(ddof=1):.4f}")
    print(f"  Δ(R2 - R0): {delta_f1_8.mean():+.4f} (95% CI: [{ci_8['ci_low']:+.4f}, {ci_8['ci_high']:+.4f}])")
    print(f"\nPooled Dataset F1:")
    print(f"  R0: {r0_pooled_f1.mean():.4f} | R2: {r2_pooled_f1.mean():.4f} | Δ: {delta_pooled_f1.mean():+.4f}")
    print(f"\nEvent Benefit / Harm / Neutral (9 events):")
    print(f"  Benefit (> +0.0050): {n_benefit} events")
    print(f"  Neutral (±0.0050):   {n_neutral} events")
    print(f"  Harm (< -0.0050):    {n_harm} events")
    print("=" * 60)


if __name__ == "__main__":
    main()

