#!/usr/bin/env python3
"""
Statistical analysis and reporting script for ChangeGuard NaN-corrected external evaluation on GEOID-Flood.
Processes the 6 corrected evaluation JSONs:
- Aggregates R0 and R2 across seeds 0, 1, 2.
- Generates 54-row activation-level full precision confusion metrics table.
- Generates 48-row grouped full precision confusion metrics table (EMSR861 & EMSR864 grouped).
- Generates seed metrics summary CSV.
- Generates event breakdown CSV with historical +/-0.0100 and +/-0.0050 benefit/neutral/harm classification.
- Performs event-level paired bootstrap (N=10,000, seed 42) for 9-event and 8-group configurations.
- Compares original archived vs corrected metrics to quantify the exact mathematical shift caused by NaN handling.
- Generates publication figure and LaTeX snippet.
- Exports comprehensive summary JSON.
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
ORIG_DIR = REPO_ROOT / "outputs/external_evaluation"
CORR_DIR = REPO_ROOT / "outputs/external_evaluation/geoid_nan_corrected_v1"

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
    print("=" * 70)
    print("ChangeGuard GEOID-Flood NaN-Corrected Evaluation Analysis")
    print("=" * 70)

    # 1. Load original and corrected results
    orig_results = {}
    corr_results = {}

    for m_id in MODEL_IDS:
        orig_file = ORIG_DIR / f"{m_id}_geoid_eval_results.json"
        corr_file = CORR_DIR / f"{m_id}_geoid_eval_results_nan_corrected.json"

        if not orig_file.is_file():
            raise FileNotFoundError(f"Missing original result file: {orig_file}")
        if not corr_file.is_file():
            raise FileNotFoundError(f"Missing corrected result file: {corr_file}")

        with open(orig_file, "r", encoding="utf-8") as f:
            orig_results[m_id] = json.load(f)
        with open(corr_file, "r", encoding="utf-8") as f:
            corr_results[m_id] = json.load(f)

    # Verify pixel count invariants
    EXPECTED_VALID_PIXELS = 873090528
    EXPECTED_FLOOD_PIXELS = 45048201

    for m_id in MODEL_IDS:
        p_c = corr_results[m_id]["pooled_metrics"]
        val_pix = p_c["pooled_tp"] + p_c["pooled_fp"] + p_c["pooled_fn"] + p_c["pooled_tn"]
        flood_pix = p_c["pooled_tp"] + p_c["pooled_fn"]
        assert val_pix == EXPECTED_VALID_PIXELS, f"{m_id} valid pixels {val_pix} != {EXPECTED_VALID_PIXELS}"
        assert flood_pix == EXPECTED_FLOOD_PIXELS, f"{m_id} flood pixels {flood_pix} != {EXPECTED_FLOOD_PIXELS}"
    print(f"Verified dataset invariants across all 6 runs: {EXPECTED_VALID_PIXELS:,} valid pixels, {EXPECTED_FLOOD_PIXELS:,} flood pixels.")

    # 2. Build 54-row activation-level table (full precision)
    row_54_list = []
    for m_id in MODEL_IDS:
        r = corr_results[m_id]
        for ev in r["event_metrics_9_events"]:
            row_54_list.append({
                "model_id": m_id,
                "recipe": r["recipe"],
                "seed": r["seed"],
                "event_id": ev["event_id"],
                "tp": ev["tp"],
                "fp": ev["fp"],
                "fn": ev["fn"],
                "tn": ev["tn"],
                "valid_pixels": ev["valid_pixels"],
                "gt_flood_pixels": ev["gt_flood_pixels"],
                "flood_fraction": ev["gt_flood_pixels"] / ev["valid_pixels"] if ev["valid_pixels"] > 0 else 0.0,
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
                "iou": ev["iou"],
            })
    df_54 = pd.DataFrame(row_54_list)
    csv_54_path = CORR_DIR / "full_precision_54_runs_confusion_metrics_nan_corrected.csv"
    df_54.to_csv(csv_54_path, index=False)
    print(f"Saved 54-row activation metrics table: {csv_54_path}")

    # 3. Build 48-row grouped table (full precision)
    row_48_list = []
    for m_id in MODEL_IDS:
        r = corr_results[m_id]
        for grp in r["event_metrics_8_events_grouped"]:
            grp_id = grp.get("group_id", grp.get("event_id"))
            row_48_list.append({
                "model_id": m_id,
                "recipe": r["recipe"],
                "seed": r["seed"],
                "group_id": grp_id,
                "tp": grp["tp"],
                "fp": grp["fp"],
                "fn": grp["fn"],
                "tn": grp["tn"],
                "valid_pixels": grp["valid_pixels"],
                "gt_flood_pixels": grp["gt_flood_pixels"],
                "flood_fraction": grp["gt_flood_pixels"] / grp["valid_pixels"] if grp["valid_pixels"] > 0 else 0.0,
                "precision": grp["precision"],
                "recall": grp["recall"],
                "f1": grp["f1"],
                "iou": grp["iou"],
            })
    df_48 = pd.DataFrame(row_48_list)
    csv_48_path = CORR_DIR / "full_precision_48_grouped_confusion_metrics_nan_corrected.csv"
    df_48.to_csv(csv_48_path, index=False)
    print(f"Saved 48-row grouped metrics table: {csv_48_path}")

    # 4. Build seed-level summary table & compute shifts
    seed_rows = []
    for m_id in MODEL_IDS:
        rc = corr_results[m_id]
        ro = orig_results[m_id]

        m9_c = rc["macro_metrics_9_events"]
        m8_c = rc["macro_metrics_8_events_grouped"]
        pool_c = rc["pooled_metrics"]

        m9_o = ro["macro_metrics_9_events"]
        m8_o = ro["macro_metrics_8_events_grouped"]
        pool_o = ro["pooled_metrics"]

        seed_rows.append({
            "model_id": m_id,
            "recipe": rc["recipe"],
            "seed": rc["seed"],
            # Corrected metrics
            "macro_f1_9events": m9_c["macro_f1"],
            "macro_precision_9events": m9_c["macro_precision"],
            "macro_recall_9events": m9_c["macro_recall"],
            "macro_iou_9events": m9_c["macro_iou"],
            "macro_f1_8grouped": m8_c["macro_f1"],
            "macro_precision_8grouped": m8_c["macro_precision"],
            "macro_recall_8grouped": m8_c["macro_recall"],
            "macro_iou_8grouped": m8_c["macro_iou"],
            "pooled_f1": pool_c["pooled_f1"],
            "pooled_precision": pool_c["pooled_precision"],
            "pooled_recall": pool_c["pooled_recall"],
            "pooled_iou": pool_c["pooled_iou"],
            # Shift from original
            "shift_macro_f1_9events": m9_c["macro_f1"] - m9_o["macro_f1"],
            "shift_macro_precision_9events": m9_c["macro_precision"] - m9_o["macro_precision"],
            "shift_macro_recall_9events": m9_c["macro_recall"] - m9_o["macro_recall"],
            "shift_pooled_f1": pool_c["pooled_f1"] - pool_o["pooled_f1"],
            "shift_tp": pool_c["pooled_tp"] - pool_o["pooled_tp"],
            "shift_fp": pool_c["pooled_fp"] - pool_o["pooled_fp"],
            "shift_fn": pool_c["pooled_fn"] - pool_o["pooled_fn"],
            "shift_tn": pool_c["pooled_tn"] - pool_o["pooled_tn"],
        })

    df_seeds = pd.DataFrame(seed_rows)
    csv_seeds_path = CORR_DIR / "geoid_seed_metrics_nan_corrected.csv"
    df_seeds.to_csv(csv_seeds_path, index=False)
    print(f"Saved seed metrics table: {csv_seeds_path}")

    # Recipe-level arrays
    r0_seeds = df_seeds[df_seeds["recipe"] == "R0"].sort_values("seed")
    r2_seeds = df_seeds[df_seeds["recipe"] == "R2"].sort_values("seed")

    r0_f1_9 = r0_seeds["macro_f1_9events"].values
    r2_f1_9 = r2_seeds["macro_f1_9events"].values
    r0_f1_8 = r0_seeds["macro_f1_8grouped"].values
    r2_f1_8 = r2_seeds["macro_f1_8grouped"].values
    r0_pool_f1 = r0_seeds["pooled_f1"].values
    r2_pool_f1 = r2_seeds["pooled_f1"].values

    r0_prec_9 = r0_seeds["macro_precision_9events"].values
    r2_prec_9 = r2_seeds["macro_precision_9events"].values
    r0_rec_9 = r0_seeds["macro_recall_9events"].values
    r2_rec_9 = r2_seeds["macro_recall_9events"].values
    r0_iou_9 = r0_seeds["macro_iou_9events"].values
    r2_iou_9 = r2_seeds["macro_iou_9events"].values

    # 5. Event breakdown (9 events)
    event_ids = [e["event_id"] for e in corr_results["R0_s0"]["event_metrics_9_events"]]
    event_rows = []

    per_event_diffs_9 = []
    for ev_id in event_ids:
        r0_f1s = [next(e["f1"] for e in corr_results[f"R0_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id) for s in [0, 1, 2]]
        r2_f1s = [next(e["f1"] for e in corr_results[f"R2_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id) for s in [0, 1, 2]]

        # Original F1s for comparison
        r0_f1s_orig = [next(e["f1"] for e in orig_results[f"R0_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id) for s in [0, 1, 2]]
        r2_f1s_orig = [next(e["f1"] for e in orig_results[f"R2_s{s}"]["event_metrics_9_events"] if e["event_id"] == ev_id) for s in [0, 1, 2]]

        r0_m = float(np.mean(r0_f1s))
        r0_s = float(np.std(r0_f1s, ddof=1))
        r2_m = float(np.mean(r2_f1s))
        r2_s = float(np.std(r2_f1s, ddof=1))
        diff = r2_m - r0_m
        per_event_diffs_9.append(diff)

        r0_m_orig = float(np.mean(r0_f1s_orig))
        r2_m_orig = float(np.mean(r2_f1s_orig))
        diff_orig = r2_m_orig - r0_m_orig

        status_0010 = "Benefit" if diff > 0.0100 else ("Harm" if diff < -0.0100 else "Neutral")
        status_0005 = "Benefit" if diff > 0.0050 else ("Harm" if diff < -0.0050 else "Neutral")

        em0 = next(e for e in corr_results["R0_s0"]["event_metrics_9_events"] if e["event_id"] == ev_id)

        event_rows.append({
            "event_id": ev_id,
            "valid_pixels": em0["valid_pixels"],
            "flood_pixels": em0["gt_flood_pixels"],
            "flood_pct": (em0["gt_flood_pixels"] / em0["valid_pixels"] * 100) if em0["valid_pixels"] > 0 else 0.0,
            "r0_f1_mean": r0_m,
            "r0_f1_std": r0_s,
            "r2_f1_mean": r2_m,
            "r2_f1_std": r2_s,
            "delta_f1": diff,
            "status_0010": status_0010,
            "status_0005": status_0005,
            # Shifts
            "shift_r0_f1": r0_m - r0_m_orig,
            "shift_r2_f1": r2_m - r2_m_orig,
            "shift_delta_f1": diff - diff_orig,
        })

    df_events = pd.DataFrame(event_rows)
    csv_events_path = CORR_DIR / "geoid_event_breakdown_nan_corrected.csv"
    df_events.to_csv(csv_events_path, index=False)
    print(f"Saved event breakdown table: {csv_events_path}")

    # Grouped breakdown (8 groups)
    group_ids = [g.get("group_id", g.get("event_id")) for g in corr_results["R0_s0"]["event_metrics_8_events_grouped"]]
    per_group_diffs_8 = []
    for grp_id in group_ids:
        r0_f1s = [next(g["f1"] for g in corr_results[f"R0_s{s}"]["event_metrics_8_events_grouped"] if g.get("group_id", g.get("event_id")) == grp_id) for s in [0, 1, 2]]
        r2_f1s = [next(g["f1"] for g in corr_results[f"R2_s{s}"]["event_metrics_8_events_grouped"] if g.get("group_id", g.get("event_id")) == grp_id) for s in [0, 1, 2]]
        per_group_diffs_8.append(float(np.mean(r2_f1s) - np.mean(r0_f1s)))

    # 6. Bootstrap Confidence Intervals
    # Canonical event-level bootstrap on seed-averaged event differences
    diffs_9_arr = np.array(per_event_diffs_9, dtype=np.float64)
    ci_9_event = bootstrap_ci(diffs_9_arr, n_boot=10000, seed=42)

    diffs_8_arr = np.array(per_group_diffs_8, dtype=np.float64)
    ci_8_event = bootstrap_ci(diffs_8_arr, n_boot=10000, seed=42)

    # Seed-level difference bootstrap (for comparison with historical report)
    delta_f1_9_seeds = r2_f1_9 - r0_f1_9
    ci_9_seeds = bootstrap_ci(delta_f1_9_seeds, n_boot=10000, seed=42)

    delta_pool_f1_seeds = r2_pool_f1 - r0_pool_f1
    ci_pool_seeds = bootstrap_ci(delta_pool_f1_seeds, n_boot=10000, seed=42)

    # 7. Summary metrics & Shift quantification
    # Original numbers for comparison
    orig_r0_macro_f1 = float(np.mean([ro["macro_metrics_9_events"]["macro_f1"] for ro in [orig_results[f"R0_s{s}"] for s in [0, 1, 2]]]))
    orig_r2_macro_f1 = float(np.mean([ro["macro_metrics_9_events"]["macro_f1"] for ro in [orig_results[f"R2_s{s}"] for s in [0, 1, 2]]]))
    orig_delta_macro_f1 = orig_r2_macro_f1 - orig_r0_macro_f1

    corr_r0_macro_f1 = float(r0_f1_9.mean())
    corr_r2_macro_f1 = float(r2_f1_9.mean())
    corr_delta_macro_f1 = corr_r2_macro_f1 - corr_r0_macro_f1

    macro_f1_r0_shift = corr_r0_macro_f1 - orig_r0_macro_f1
    macro_f1_r2_shift = corr_r2_macro_f1 - orig_r2_macro_f1
    delta_macro_f1_shift = corr_delta_macro_f1 - orig_delta_macro_f1

    n_benefit_0010 = int((df_events["status_0010"] == "Benefit").sum())
    n_harm_0010 = int((df_events["status_0010"] == "Harm").sum())
    n_neutral_0010 = int((df_events["status_0010"] == "Neutral").sum())

    n_benefit_0005 = int((df_events["status_0005"] == "Benefit").sum())
    n_harm_0005 = int((df_events["status_0005"] == "Harm").sum())
    n_neutral_0005 = int((df_events["status_0005"] == "Neutral").sum())

    summary_json = {
        "benchmark": "GEOID-Flood held-out (v1.1, NaN-corrected nan=0.15)",
        "nan_correction_details": {
            "source_matched_fill_value": 0.15,
            "original_fill_value": 0.0,
            "total_tiles": 1301,
            "affected_tiles_count": 143,
            "affected_batches_count": 111,
            "unaffected_batches_count": 540,
            "replay_verification_mismatches_all_models": 0,
        },
        "canonical_9_events": {
            "r0_macro_f1_mean": corr_r0_macro_f1,
            "r0_macro_f1_std": float(r0_f1_9.std(ddof=1)),
            "r0_macro_f1_seeds": [float(x) for x in r0_f1_9],
            "r2_macro_f1_mean": corr_r2_macro_f1,
            "r2_macro_f1_std": float(r2_f1_9.std(ddof=1)),
            "r2_macro_f1_seeds": [float(x) for x in r2_f1_9],
            "delta_macro_f1_mean": corr_delta_macro_f1,
            "delta_macro_f1_event_bootstrap_95ci": [ci_9_event["ci_low"], ci_9_event["ci_high"]],
            "delta_macro_f1_seed_bootstrap_95ci": [ci_9_seeds["ci_low"], ci_9_seeds["ci_high"]],
            "macro_precision": {
                "r0_mean": float(r0_prec_9.mean()),
                "r2_mean": float(r2_prec_9.mean()),
                "delta": float(r2_prec_9.mean() - r0_prec_9.mean()),
            },
            "macro_recall": {
                "r0_mean": float(r0_rec_9.mean()),
                "r2_mean": float(r2_rec_9.mean()),
                "delta": float(r2_rec_9.mean() - r0_rec_9.mean()),
            },
            "macro_iou": {
                "r0_mean": float(r0_iou_9.mean()),
                "r2_mean": float(r2_iou_9.mean()),
                "delta": float(r2_iou_9.mean() - r0_iou_9.mean()),
            },
            "counts_band_0010": {
                "benefit_count": n_benefit_0010,
                "neutral_count": n_neutral_0010,
                "harm_count": n_harm_0010,
            },
            "counts_band_0005": {
                "benefit_count": n_benefit_0005,
                "neutral_count": n_neutral_0005,
                "harm_count": n_harm_0005,
            },
        },
        "grouped_8_events_sensitivity": {
            "r0_macro_f1_mean": float(r0_f1_8.mean()),
            "r0_macro_f1_std": float(r0_f1_8.std(ddof=1)),
            "r2_macro_f1_mean": float(r2_f1_8.mean()),
            "r2_macro_f1_std": float(r2_f1_8.std(ddof=1)),
            "delta_macro_f1_mean": float(r2_f1_8.mean() - r0_f1_8.mean()),
            "delta_macro_f1_event_bootstrap_95ci": [ci_8_event["ci_low"], ci_8_event["ci_high"]],
        },
        "pooled_dataset_metrics": {
            "total_valid_pixels": EXPECTED_VALID_PIXELS,
            "total_gt_flood_pixels": EXPECTED_FLOOD_PIXELS,
            "r0_pooled_f1_mean": float(r0_pool_f1.mean()),
            "r2_pooled_f1_mean": float(r2_pool_f1.mean()),
            "delta_pooled_f1_mean": float(r2_pool_f1.mean() - r0_pool_f1.mean()),
            "delta_pooled_f1_seed_bootstrap_95ci": [ci_pool_seeds["ci_low"], ci_pool_seeds["ci_high"]],
        },
        "nan_correction_shifts": {
            "r0_macro_f1_shift": macro_f1_r0_shift,
            "r2_macro_f1_shift": macro_f1_r2_shift,
            "delta_macro_f1_shift": delta_macro_f1_shift,
            "net_tp_shift_r0_mean": float(df_seeds[df_seeds["recipe"]=="R0"]["shift_tp"].mean()),
            "net_fp_shift_r0_mean": float(df_seeds[df_seeds["recipe"]=="R0"]["shift_fp"].mean()),
            "net_fn_shift_r0_mean": float(df_seeds[df_seeds["recipe"]=="R0"]["shift_fn"].mean()),
            "net_tn_shift_r0_mean": float(df_seeds[df_seeds["recipe"]=="R0"]["shift_tn"].mean()),
            "net_tp_shift_r2_mean": float(df_seeds[df_seeds["recipe"]=="R2"]["shift_tp"].mean()),
            "net_fp_shift_r2_mean": float(df_seeds[df_seeds["recipe"]=="R2"]["shift_fp"].mean()),
            "net_fn_shift_r2_mean": float(df_seeds[df_seeds["recipe"]=="R2"]["shift_fn"].mean()),
            "net_tn_shift_r2_mean": float(df_seeds[df_seeds["recipe"]=="R2"]["shift_tn"].mean()),
        },
    }

    summary_path = CORR_DIR / "geoid_nan_corrected_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    # 8. Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})

    # Subplot 1: Macro F1 comparison across seeds
    axes[0].scatter([0, 0, 0], r0_f1_9, color="#1f77b4", s=60, zorder=3, label="R0 Seeds")
    axes[0].scatter([1, 1, 1], r2_f1_9, color="#2ca02c", s=60, zorder=3, label="R2 Seeds")
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
        f"GEOID-Flood Zero-Shot Macro F1 (9 Events, NaN Corrected)\n"
        f"R0: {r0_f1_9.mean():.4f} ± {r0_f1_9.std(ddof=1):.4f} | R2: {r2_f1_9.mean():.4f} ± {r2_f1_9.std(ddof=1):.4f}\n"
        f"$\\Delta$: {corr_delta_macro_f1:+.4f} (95% Event CI: [{ci_9_event['ci_low']:+.4f}, {ci_9_event['ci_high']:+.4f}])",
        fontsize=11
    )
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="best", fontsize=9)

    # Subplot 2: Per-event Delta(R2 - R0)
    ev_names = df_events["event_id"].tolist()
    ev_deltas = df_events["delta_f1"].tolist()
    colors = ["#2ca02c" if d > 0.0100 else ("#d62728" if d < -0.0100 else "#7f7f7f") for d in ev_deltas]

    y_pos = np.arange(len(ev_names))
    bars = axes[1].barh(y_pos, ev_deltas, color=colors, height=0.6, edgecolor="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1.0)
    axes[1].axvline(0.0100, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[1].axvline(-0.0100, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ev_names, fontsize=10)
    axes[1].set_xlabel("$\\Delta$ F1 (R2 - R0)", fontsize=12)
    axes[1].set_title(
        f"Event-by-Event Paired Differences $\\Delta(R2 - R0)$\n"
        f"Benefit (>+0.01): {n_benefit_0010} | Neutral (±0.01): {n_neutral_0010} | Harm (<-0.01): {n_harm_0010}",
        fontsize=11
    )
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        x_text = width + (0.002 if width >= 0 else -0.006)
        ha = "left" if width >= 0 else "right"
        axes[1].text(x_text, bar.get_y() + bar.get_height() / 2, f"{width:+.4f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    fig_path = CORR_DIR / "geoid_r2_vs_r0_comparison_figure_nan_corrected.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure: {fig_path}")

    # 9. LaTeX table snippet
    tex_path = CORR_DIR / "geoid_table_snippet_nan_corrected.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% ChangeGuard NaN-Corrected External Evaluation on GEOID-Flood Held-Out Benchmark\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Zero-shot cross-benchmark evaluation on GEOID-Flood held-out partition (1,301 tiles across 9 independent Copernicus EMS activations). Models trained purely on Kuro Siwo without target adaptation. Evaluated with source-matched NaN padding ($v=0.15$).}\n")
        f.write("\\label{tab:geoid_external_eval_corrected}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Recipe & Macro F1 (9 Ev.) & Macro F1 (8 Grp.) & Pooled F1 & Macro IoU \\\\\n\\midrule\n")
        f.write(f"R0 (Standard Source) & {corr_r0_macro_f1:.4f} $\\pm$ {r0_f1_9.std(ddof=1):.4f} & {r0_f1_8.mean():.4f} $\\pm$ {r0_f1_8.std(ddof=1):.4f} & {r0_pool_f1.mean():.4f} $\\pm$ {r0_pool_f1.std(ddof=1):.4f} & {r0_iou_9.mean():.4f} \\\\\n")
        f.write(f"R2 (Robust Source)   & {corr_r2_macro_f1:.4f} $\\pm$ {r2_f1_9.std(ddof=1):.4f} & {r2_f1_8.mean():.4f} $\\pm$ {r2_f1_8.std(ddof=1):.4f} & {r2_pool_f1.mean():.4f} $\\pm$ {r2_pool_f1.std(ddof=1):.4f} & {r2_iou_9.mean():.4f} \\\\\n\\midrule\n")
        f.write(f"$\\Delta(\\text{{R2}} - \\text{{R0}})$ & {corr_delta_macro_f1:+.4f} & {r2_f1_8.mean() - r0_f1_8.mean():+.4f} & {r2_pool_f1.mean() - r0_pool_f1.mean():+.4f} & {r2_iou_9.mean() - r0_iou_9.mean():+.4f} \\\\\n")
        f.write(f"95\\% Event Bootstrap CI & [{ci_9_event['ci_low']:+.4f}, {ci_9_event['ci_high']:+.4f}] & [{ci_8_event['ci_low']:+.4f}, {ci_8_event['ci_high']:+.4f}] & -- & -- \\\\\n")
        f.write(f"95\\% Seed Bootstrap CI  & [{ci_9_seeds['ci_low']:+.4f}, {ci_9_seeds['ci_high']:+.4f}] & -- & [{ci_pool_seeds['ci_low']:+.4f}, {ci_pool_seeds['ci_high']:+.4f}] & -- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved LaTeX table snippet: {tex_path}")

    # 10. Print comprehensive summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: GEOID-FLOOD NAN-CORRECTED EVALUATION")
    print("=" * 70)
    print(f"Canonical 9-Event Macro F1:")
    print(f"  R0: {corr_r0_macro_f1:.6f} ± {r0_f1_9.std(ddof=1):.6f}  (seeds: {[f'{x:.6f}' for x in r0_f1_9]})")
    print(f"  R2: {corr_r2_macro_f1:.6f} ± {r2_f1_9.std(ddof=1):.6f}  (seeds: {[f'{x:.6f}' for x in r2_f1_9]})")
    print(f"  Δ(R2 - R0): {corr_delta_macro_f1:+.6f} ({corr_delta_macro_f1*100:+.2f} pp)")
    print(f"  95% Event Bootstrap CI (9 events): [{ci_9_event['ci_low']:+.6f}, {ci_9_event['ci_high']:+.6f}]")
    print(f"  95% Seed Bootstrap CI (3 seeds):   [{ci_9_seeds['ci_low']:+.6f}, {ci_9_seeds['ci_high']:+.6f}]")
    print(f"\nPrecision / Recall Trade-off (9 events):")
    print(f"  Macro Precision: R0 = {r0_prec_9.mean():.6f}, R2 = {r2_prec_9.mean():.6f} (Δ = {r2_prec_9.mean() - r0_prec_9.mean():+.6f})")
    print(f"  Macro Recall:    R0 = {r0_rec_9.mean():.6f}, R2 = {r2_rec_9.mean():.6f} (Δ = {r2_rec_9.mean() - r0_rec_9.mean():+.6f})")
    print(f"  Macro IoU:       R0 = {r0_iou_9.mean():.6f}, R2 = {r2_iou_9.mean():.6f} (Δ = {r2_iou_9.mean() - r0_iou_9.mean():+.6f})")
    print(f"\nGrouped 8-Event Sensitivity (Iberia combined):")
    print(f"  R0: {r0_f1_8.mean():.6f} ± {r0_f1_8.std(ddof=1):.6f}")
    print(f"  R2: {r2_f1_8.mean():.6f} ± {r2_f1_8.std(ddof=1):.6f}")
    print(f"  Δ(R2 - R0): {r2_f1_8.mean() - r0_f1_8.mean():+.6f}")
    print(f"  95% Event Bootstrap CI (8 groups): [{ci_8_event['ci_low']:+.6f}, {ci_8_event['ci_high']:+.6f}]")
    print(f"\nPooled Dataset F1 (873M pixels):")
    print(f"  R0: {r0_pool_f1.mean():.6f} | R2: {r2_pool_f1.mean():.6f} | Δ: {r2_pool_f1.mean() - r0_pool_f1.mean():+.6f}")
    print(f"\nEvent Benefit / Neutral / Harm Counts:")
    print(f"  Band ±0.0100: Benefit={n_benefit_0010}, Neutral={n_neutral_0010}, Harm={n_harm_0010}")
    print(f"  Band ±0.0050: Benefit={n_benefit_0005}, Neutral={n_neutral_0005}, Harm={n_harm_0005}")
    print(f"\nShift from NaN-Correction (nan=0.0 -> nan=0.15):")
    print(f"  R0 Macro F1 Shift: {macro_f1_r0_shift:+.6f} ({macro_f1_r0_shift*100:+.4f} pp)")
    print(f"  R2 Macro F1 Shift: {macro_f1_r2_shift:+.6f} ({macro_f1_r2_shift*100:+.4f} pp)")
    print(f"  Δ(R2 - R0) Shift:  {delta_macro_f1_shift:+.6f} ({delta_macro_f1_shift*100:+.4f} pp)")
    print("=" * 70)


if __name__ == "__main__":
    main()
