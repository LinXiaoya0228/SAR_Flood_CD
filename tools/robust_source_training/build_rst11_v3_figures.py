#!/usr/bin/env python3
"""Generates authoritative publication-grade figures for RST11 v3 FYP report and deliverables.

Addresses all audited visual QA and numerical issues:
- Figure 1 (formerly Fig 6): Uses verified 27-event inventory (7,556 max, 21 min, 21,462 total, 359.8x ratio).
  Labels M1 balanced line explicitly as 'Expected Draws Under M1 Event Balancing (~795 draws / epoch)'.
- Figure 2 (formerly Fig 1): Component discovery staircase on Fold C (RST4).
- Figure 3 (formerly Fig 2): 18 confirmatory event paired deltas on SNUNet-ECAM (RST7).
  Accurately reflects 9 positive and 9 negative mean differences.
- Figure 4 (formerly Fig 3): Tail reliability vs average generalization. Legend moved to upper right
  with expanded y-limits [0.20, 0.95] to prevent collision with data annotations.
- Figure 5 (formerly Fig 4): Precision-recall trade-off with expanded top margin preventing title crowding.
- Figure 6 (formerly Fig 5): Seed stability with expanded top margin and clean legend placement.
- Figure 7 (formerly Fig 7): Cross-backbone architecture transfer on SiamUnet_diff.
- Figure 8 (formerly Fig 8): Retrospective historical-target extension deltas.
- Exports both 300 DPI PNG and vector PDF, along with backing CSV files.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Professional, colorblind-friendly academic palette
PALETTE = {
    "r0": "#4A6FA5",        # Muted Steel Blue (Baseline R0)
    "r1": "#47B8B8",        # Cyan/Teal (R1)
    "r2": "#E06D53",        # Coral/Rust Red (R2 Selected)
    "r3": "#DDA7A5",        # Soft Rose/Muted Mauve (R3)
    "fold_a": "#2E7D32",    # Forest Green
    "fold_b": "#1565C0",    # Strong Blue
    "dev": "#5C6BC0",       # Indigo (Historical Dev)
    "test": "#D84315",      # Deep Orange (Historical Test)
    "pos": "#2E7D32",       # Green for positive gain
    "neg": "#C62828",       # Red for negative loss
    "neutral": "#616161",   # Neutral Grey
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save_fig(fig: plt.Figure, base_path: Path, data_rows: List[Dict[str, Any]], fieldnames: List[str]):
    """Saves figure in PNG and PDF formats, and exports backing CSV data."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=300)
    fig.savefig(f"{base_path}.pdf")
    plt.close(fig)

    data_dir = base_path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{base_path.name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
    print(f"Saved: {base_path}.png, {base_path}.pdf, {csv_path}")


def build_fig1_event_sampling_distribution(inventory_csv: Path, out_dir: Path):
    """Figure 1 (formerly Fig 6): Event sampling distribution before and after M1 balancing."""
    events = []
    with open(inventory_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            events.append({
                "event_id": str(r["event_id"]),
                "eligible_patches": int(r["eligible_training_patches"]),
            })

    # Sort by original patch count
    events.sort(key=lambda x: x["eligible_patches"])
    event_ids = [e["event_id"] for e in events]
    orig_counts = [e["eligible_patches"] for e in events]
    total_patches = sum(orig_counts)
    min_count = orig_counts[0]
    max_count = orig_counts[-1]
    disparity_ratio = max_count / min_count
    expected_draws = round(total_patches / len(events))  # 21462 / 27 = 794.888... -> 795

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(events))

    ax.bar(x, orig_counts, color=PALETTE["r0"], alpha=0.85, edgecolor="black", linewidth=0.7,
           label=f"Original Dataset Patches ({disparity_ratio:.1f}× Imbalance)")
    ax.axhline(expected_draws, color=PALETTE["r2"], linestyle="--", linewidth=2.0,
               label=f"Expected Draws Under M1 Event Balancing (~{expected_draws} draws / epoch)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(event_ids, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Samples / Draws per Epoch (Log Scale)")
    ax.set_xlabel("Authoritative Source Event Identifier (Sorted by Patch Count)")
    ax.set_title(
        f"Figure 1: Equalization of Event Sampling Frequencies Under M1\n"
        f"(Authoritative 27 Source Events: {min_count} Patches Min vs. {max_count:,} Patches Max; {disparity_ratio:.1f}× Disparity)",
        fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # Annotate min and max
    ax.annotate(f"Min: {orig_counts[0]} patches\n(Event {event_ids[0]})", xy=(0, orig_counts[0]), xytext=(1.5, 15),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"), fontsize=8)
    ax.annotate(f"Max: {orig_counts[-1]:,} patches\n(Event {event_ids[-1]})", xy=(len(events)-1, orig_counts[-1]), xytext=(len(events)-5, 3000),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"), fontsize=8)

    data_export = [
        {"event_id": eid, "original_patches": orig, "m1_expected_draws_per_epoch": expected_draws}
        for eid, orig in zip(event_ids, orig_counts)
    ]
    _save_fig(fig, out_dir / "fig1_event_sampling_distribution", data_export,
              ["event_id", "original_patches", "m1_expected_draws_per_epoch"])


def build_fig2_component_staircase(rst4_summary_csv: Path, out_dir: Path):
    """Figure 2 (formerly Fig 1): Component discovery staircase on Fold C."""
    rows = []
    with open(rst4_summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    recipe_ids = [r["recipe_id"] for r in rows]
    labels = [
        "R0\n(Baseline)",
        "R1\n(+M1 Event-Bal)",
        "R2 (Selected)\n(+M1 & +M2 Loss)",
        "R3\n(+M1, +M2, +M3 Aug)"
    ]

    macro_f1 = [float(r["event_macro_f1"]) for r in rows]
    bottom3_f1 = [float(r["bottom_3_event_mean_f1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(recipe_ids))
    width = 0.35

    rects1 = ax.bar(x - width/2, macro_f1, width, label="Grand Macro F1", color=PALETTE["r0"], alpha=0.88, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, bottom3_f1, width, label="Bottom-3 Tail Event Mean F1", color=PALETTE["r2"], alpha=0.88, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("F1 Score")
    ax.set_title("Figure 2: Component Discovery Staircase on Fold C (RST4, Seed 0)\n(Progression from Baseline R0 to Frozen Recipe R2)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.40, 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # Add value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    data_export = [
        {"recipe_id": r["recipe_id"], "macro_f1": round(m, 4), "bottom3_f1": round(b, 4)}
        for r, m, b in zip(rows, macro_f1, bottom3_f1)
    ]
    _save_fig(fig, out_dir / "fig2_component_staircase", data_export,
              ["recipe_id", "macro_f1", "bottom3_f1"])


def build_fig3_primary_per_event_paired_deltas(rst7_canonical_csv: Path, out_dir: Path):
    """Figure 3 (formerly Fig 2): Paired per-event differences across 18 confirmatory events."""
    rows = []
    with open(rst7_canonical_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by event_id
    events = {}
    for r in rows:
        eid = int(r["event_id"])
        fold = "Fold A" if "folda" in r["fold_id"].lower() or "fold_a" in r["fold_id"].lower() else "Fold B"
        method = r["method"].upper()
        seed = int(r["seed"])
        f1 = float(r["f1"])
        if eid not in events:
            events[eid] = {"fold": fold, "R0": {}, "R2": {}}
        events[eid][method][seed] = f1

    event_summary = []
    for eid, info in sorted(events.items()):
        seeds = sorted(info["R0"].keys())
        r0_mean = float(np.mean([info["R0"][s] for s in seeds]))
        r2_mean = float(np.mean([info["R2"][s] for s in seeds]))
        seed_deltas = [info["R2"][s] - info["R0"][s] for s in seeds]
        mean_d = r2_mean - r0_mean
        event_summary.append({
            "event_id": eid,
            "fold": info["fold"],
            "r0_mean": r0_mean,
            "r2_mean": r2_mean,
            "mean_delta_f1": mean_d,
            "seed_deltas": seed_deltas,
        })

    # Sort ascending by mean paired difference
    event_summary.sort(key=lambda x: x["mean_delta_f1"])

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    y_pos = np.arange(len(event_summary))

    data_export = []
    pos_count = 0
    neg_count = 0

    for idx, es in enumerate(event_summary):
        mean_val = es["mean_delta_f1"]
        if mean_val >= 0:
            pos_count += 1
        else:
            neg_count += 1

        fold_color = PALETTE["fold_a"] if es["fold"] == "Fold A" else PALETTE["fold_b"]
        marker_color = PALETTE["pos"] if mean_val >= 0 else PALETTE["neg"]

        # Plot individual seed deltas
        ax.scatter(es["seed_deltas"], [idx]*len(es["seed_deltas"]), color=fold_color, alpha=0.45, s=28, zorder=3)
        # Plot stem line from 0 to mean
        ax.plot([0, mean_val], [idx, idx], color=fold_color, alpha=0.6, linewidth=1.8, zorder=2)
        # Plot mean diamond marker
        ax.scatter(mean_val, idx, color=marker_color, edgecolors="black", s=65, zorder=4)

        data_export.append({
            "event_id": es["event_id"],
            "fold": es["fold"],
            "r0_mean_f1": round(es["r0_mean"], 4),
            "r2_mean_f1": round(es["r2_mean"], 4),
            "mean_delta_f1": round(mean_val, 4),
            "seed0_delta": round(es["seed_deltas"][0], 4),
            "seed1_delta": round(es["seed_deltas"][1], 4),
            "seed2_delta": round(es["seed_deltas"][2], 4),
        })

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Event {es['event_id']} ({'A' if es['fold']=='Fold A' else 'B'})" for es in event_summary])
    ax.set_xlabel("Paired Difference in F1 Score (Recipe R2 − Baseline R0)")
    ax.set_title(
        f"Figure 3: Primary Confirmatory Event-Level Paired Deltas (RST7)\n"
        f"(18 Held-Out Source Events across Seeds 0, 1, 2: {pos_count} Positive Mean Differences, {neg_count} Negative Mean Differences)",
        fontweight="bold"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['fold_a'], label='Fold A Seed Runs', markersize=6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['fold_b'], label='Fold B Seed Runs', markersize=6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['pos'], markeredgecolor='black', label=f'Positive Mean Δ ({pos_count} events)', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['neg'], markeredgecolor='black', label=f'Negative Mean Δ ({neg_count} events)', markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    _save_fig(fig, out_dir / "fig3_primary_per_event_paired_deltas", data_export,
              ["event_id", "fold", "r0_mean_f1", "r2_mean_f1", "mean_delta_f1", "seed0_delta", "seed1_delta", "seed2_delta"])


def build_fig4_worst_bottom_event_comparison(rst7_summary_json: Path, out_dir: Path):
    """Figure 4 (formerly Fig 3): Worst and bottom-event comparison in RST7.

    Layout Fix: Moves legend to upper right and sets y-limits [0.20, 0.95] to prevent collision with annotations.
    """
    with open(rst7_summary_json, "r", encoding="utf-8") as f:
        d = json.load(f)

    gm = d["grand_metrics"]
    categories = ["Grand Macro F1", "Bottom-3 Mean F1", "Worst Event F1"]
    r0_vals = [gm["r0"]["f1"], gm["r0"]["bottom3_mean_f1"], gm["r0"]["worst_event_f1"]]
    r2_vals = [gm["r2"]["f1"], gm["r2"]["bottom3_mean_f1"], gm["r2"]["worst_event_f1"]]
    deltas = [gm["delta"]["f1"], gm["delta"]["bottom3_mean_f1"], gm["delta"]["worst_event_f1"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, r0_vals, width, label="R0 Baseline", color=PALETTE["r0"], alpha=0.88, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, r2_vals, width, label="R2 Recipe (Selected)", color=PALETTE["r2"], alpha=0.88, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("F1 Score")
    ax.set_title("Figure 4: Tail Reliability vs. Average Generalization in RST7\n(Source Domain Confirmation on SNUNet-ECAM)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.20, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Legend placed in upper right to avoid data overlap completely
    ax.legend(loc="upper right")

    # Annotate deltas clearly above bars
    for i, d_val in enumerate(deltas):
        color = PALETTE["pos"] if d_val >= 0 else PALETTE["neg"]
        ax.annotate(f"Δ: {d_val:+.4f}", xy=(x[i], max(r0_vals[i], r2_vals[i]) + 0.035),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h / 2),
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h / 2),
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    data_export = [
        {"metric_group": cat, "r0_score": round(r0, 4), "r2_score": round(r2, 4), "delta": round(d_val, 4)}
        for cat, r0, r2, d_val in zip(categories, r0_vals, r2_vals, deltas)
    ]
    _save_fig(fig, out_dir / "fig4_worst_bottom_event_comparison", data_export,
              ["metric_group", "r0_score", "r2_score", "delta"])


def build_fig5_precision_recall_tradeoff(rst7_summary_json: Path, rst9_analysis_json: Path, out_dir: Path):
    """Figure 5 (formerly Fig 4): Precision/recall trade-off separated by scope.

    Layout Fix: Expanded top margin and subplots adjustment to eliminate crowding between suptitle and panel titles.
    """
    with open(rst7_summary_json, "r", encoding="utf-8") as f:
        d7 = json.load(f)
    with open(rst9_analysis_json, "r", encoding="utf-8") as f:
        d9 = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True, sharex=True)

    # Panel A: RST7 Source Domain Confirmation
    r0_p7 = d7["grand_metrics"]["r0"]["precision"]
    r0_r7 = d7["grand_metrics"]["r0"]["recall"]
    r2_p7 = d7["grand_metrics"]["r2"]["precision"]
    r2_r7 = d7["grand_metrics"]["r2"]["recall"]

    ax1.scatter(r0_r7, r0_p7, color=PALETTE["r0"], s=100, label="R0 Baseline", edgecolors="black", zorder=3)
    ax1.scatter(r2_r7, r2_p7, color=PALETTE["r2"], s=100, label="R2 Recipe", edgecolors="black", zorder=3)
    ax1.annotate("", xy=(r2_r7, r2_p7), xytext=(r0_r7, r0_p7),
                 arrowprops=dict(arrowstyle="->", lw=2.0, color="#1E293B", mutation_scale=15))
    ax1.set_title("Panel A: RST7 Primary Source Confirmation\n(18 Held-Out Events, 3 Seeds)", fontsize=10, fontweight="bold", pad=8)
    ax1.set_xlabel("Event-Macro Recall")
    ax1.set_ylabel("Event-Macro Precision")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left")
    ax1.text(0.73, 0.73, f"Δ Precision: {r2_p7 - r0_p7:+.4f}\nΔ Recall:    {r2_r7 - r0_r7:+.4f}",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1"))

    # Panel B: RST9 Retrospective Historical Extension
    r0_p9 = float(np.mean([e["r0_precision_mean"] for e in d9["per_event_table"]]))
    r0_r9 = float(np.mean([e["r0_recall_mean"] for e in d9["per_event_table"]]))
    r2_p9 = float(np.mean([e["r2_precision_mean"] for e in d9["per_event_table"]]))
    r2_r9 = float(np.mean([e["r2_recall_mean"] for e in d9["per_event_table"]]))

    ax2.scatter(r0_r9, r0_p9, color=PALETTE["r0"], s=100, label="R0 Baseline", edgecolors="black", zorder=3)
    ax2.scatter(r2_r9, r2_p9, color=PALETTE["r2"], s=100, label="R2 Recipe", edgecolors="black", zorder=3)
    ax2.annotate("", xy=(r2_r9, r2_p9), xytext=(r0_r9, r0_p9),
                 arrowprops=dict(arrowstyle="->", lw=2.0, color="#1E293B", mutation_scale=15))
    ax2.set_title("Panel B: RST9 Retrospective Extension\n(16 Previously Observed Events, 3 Seeds)", fontsize=10, fontweight="bold", pad=8)
    ax2.set_xlabel("Event-Macro Recall")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left")
    ax2.text(0.71, 0.74, f"Δ Precision: {r2_p9 - r0_p9:+.4f}\nΔ Recall:    {r2_r9 - r0_r9:+.4f}",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1"))

    fig.suptitle("Figure 5: Precision vs. Recall Dynamics Under Recipe R2", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.82, wspace=0.22)

    data_export = [
        {"scope": "RST7_Primary_Source", "r0_precision": round(r0_p7, 4), "r0_recall": round(r0_r7, 4),
         "r2_precision": round(r2_p7, 4), "r2_recall": round(r2_r7, 4)},
        {"scope": "RST9_Retrospective_Target", "r0_precision": round(r0_p9, 4), "r0_recall": round(r0_r9, 4),
         "r2_precision": round(r2_p9, 4), "r2_recall": round(r2_r9, 4)},
    ]
    _save_fig(fig, out_dir / "fig5_precision_recall_tradeoff", data_export,
              ["scope", "r0_precision", "r0_recall", "r2_precision", "r2_recall"])


def build_fig6_seed_stability(rst7_summary_json: Path, rst9_analysis_json: Path, out_dir: Path):
    """Figure 6 (formerly Fig 5): Seed stability across RST7 (Folds A & B) and RST9 (Historical Target).

    Layout Fix: Expanded top margin and subplots adjustment to eliminate crowding between suptitle and panel titles.
    """
    with open(rst7_summary_json, "r", encoding="utf-8") as f:
        d7 = json.load(f)
    with open(rst9_analysis_json, "r", encoding="utf-8") as f:
        d9 = json.load(f)

    fa = d7["fold_summaries"]["fold_a"]
    fb = d7["fold_summaries"]["fold_b"]

    fa_r0_seeds = fa["r0"]["seed_macro_f1"]
    fa_r2_seeds = fa["r2"]["seed_macro_f1"]
    fb_r0_seeds = fb["r0"]["seed_macro_f1"]
    fb_r2_seeds = fb["r2"]["seed_macro_f1"]

    rst9_s_means = {
        int(s): (d9["per_seed_summary"][s]["r0"]["grand_macro_f1"], d9["per_seed_summary"][s]["r2"]["grand_macro_f1"])
        for s in ("0", "1", "2")
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.6), sharey=True)

    seed_colors = [PALETTE["r0"], PALETTE["fold_b"], PALETTE["r3"]]

    # Panel A: RST7 Fold A
    for s in (0, 1, 2):
        r0 = fa_r0_seeds[s]
        r2 = fa_r2_seeds[s]
        ax1.plot([0, 1], [r0, r2], marker="o", lw=2.2, color=seed_colors[s], label=f"Seed {s} (Δ: {r2-r0:+.4f})", zorder=3)
    ax1.plot([0, 1], [fa["r0"]["mean_macro_f1"], fa["r2"]["mean_macro_f1"]],
             marker="D", lw=2.8, ls="--", color="black", label=f"Mean (SD: {fa['r0']['sd_macro_f1']:.4f}→{fa['r2']['sd_macro_f1']:.4f})", zorder=4)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["R0 Baseline", "R2 Recipe"], fontweight="bold")
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("Panel A: RST7 Fold A (9 Events)\n(SD: 0.0120 → 0.0106; Δ = +0.0083)", fontsize=10, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel B: RST7 Fold B
    for s in (0, 1, 2):
        r0 = fb_r0_seeds[s]
        r2 = fb_r2_seeds[s]
        ax2.plot([0, 1], [r0, r2], marker="o", lw=2.2, color=seed_colors[s], label=f"Seed {s} (Δ: {r2-r0:+.4f})", zorder=3)
    ax2.plot([0, 1], [fb["r0"]["mean_macro_f1"], fb["r2"]["mean_macro_f1"]],
             marker="D", lw=2.8, ls="--", color="black", label=f"Mean (SD: {fb['r0']['sd_macro_f1']:.4f}→{fb['r2']['sd_macro_f1']:.4f})", zorder=4)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["R0 Baseline", "R2 Recipe"], fontweight="bold")
    ax2.set_title("Panel B: RST7 Fold B (9 Events)\n(SD: 0.0035 → 0.0165; P3 FAILED)", fontsize=10, fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5)

    # Panel C: RST9 Retrospective Target
    t_r0_seeds = [rst9_s_means[s][0] for s in (0, 1, 2)]
    t_r2_seeds = [rst9_s_means[s][1] for s in (0, 1, 2)]
    t_r0_mean = float(np.mean(t_r0_seeds))
    t_r2_mean = float(np.mean(t_r2_seeds))
    t_r0_sd = float(np.std(t_r0_seeds, ddof=1))
    t_r2_sd = float(np.std(t_r2_seeds, ddof=1))

    for s in (0, 1, 2):
        r0, r2 = rst9_s_means[s]
        ax3.plot([0, 1], [r0, r2], marker="s", lw=2.2, color=seed_colors[s], label=f"Seed {s} (Δ: {r2-r0:+.4f})", zorder=3)
    ax3.plot([0, 1], [t_r0_mean, t_r2_mean],
             marker="D", lw=2.8, ls="--", color="black", label=f"Mean (SD: {t_r0_sd:.4f}→{t_r2_sd:.4f})", zorder=4)

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["R0 Baseline", "R2 Recipe"], fontweight="bold")
    ax3.set_title("Panel C: RST9 Retrospective Target\n(SD: 0.0223 → 0.0059; Δ = +0.0507)", fontsize=10, fontweight="bold", pad=8)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Figure 6: Evaluation of Seed Stability Across Random Seeds 0, 1, 2 (RST7 vs RST9)", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.82, wspace=0.22)

    data_export = [
        {"scope": "RST7_Primary_Source", "partition": "Fold A", "seed": f"Seed {s}",
         "r0_macro_f1": round(fa_r0_seeds[s], 4), "r2_macro_f1": round(fa_r2_seeds[s], 4), "delta": round(fa_r2_seeds[s] - fa_r0_seeds[s], 4)}
        for s in (0, 1, 2)
    ] + [
        {"scope": "RST7_Primary_Source", "partition": "Fold B", "seed": f"Seed {s}",
         "r0_macro_f1": round(fb_r0_seeds[s], 4), "r2_macro_f1": round(fb_r2_seeds[s], 4), "delta": round(fb_r2_seeds[s] - fb_r0_seeds[s], 4)}
        for s in (0, 1, 2)
    ] + [
        {"scope": "RST9_Retrospective_Target", "partition": "Historical_16", "seed": f"Seed {s}",
         "r0_macro_f1": round(rst9_s_means[s][0], 4), "r2_macro_f1": round(rst9_s_means[s][1], 4), "delta": round(rst9_s_means[s][1] - rst9_s_means[s][0], 4)}
        for s in (0, 1, 2)
    ]
    _save_fig(fig, out_dir / "fig6_seed_stability", data_export,
              ["scope", "partition", "seed", "r0_macro_f1", "r2_macro_f1", "delta"])


def build_fig7_architecture_transfer_deltas(rst7_summary_json: Path, rst8_summary_json: Path, out_dir: Path):
    """Figure 7 (formerly Fig 7): Architecture transfer deltas on SiamUnet_diff."""
    with open(rst7_summary_json, "r", encoding="utf-8") as f:
        d7 = json.load(f)
    with open(rst8_summary_json, "r", encoding="utf-8") as f:
        d8 = json.load(f)

    metrics = ["Grand Macro F1", "Bottom-3 Mean F1", "Worst Event F1"]

    sn_gm = d7["grand_metrics"]
    snunet_r0 = [sn_gm["r0"]["f1"], sn_gm["r0"]["bottom3_mean_f1"], sn_gm["r0"]["worst_event_f1"]]
    snunet_r2 = [sn_gm["r2"]["f1"], sn_gm["r2"]["bottom3_mean_f1"], sn_gm["r2"]["worst_event_f1"]]
    snunet_deltas = [sn_gm["delta"]["f1"], sn_gm["delta"]["bottom3_mean_f1"], sn_gm["delta"]["worst_event_f1"]]

    siam_r0 = [
        d8["grand_metrics"]["r0"]["f1"],
        d8["grand_metrics"]["r0"]["bottom3_mean_f1"],
        d8["grand_metrics"]["r0"]["worst_event_f1"],
    ]
    siam_r2 = [
        d8["grand_metrics"]["r2"]["f1"],
        d8["grand_metrics"]["r2"]["bottom3_mean_f1"],
        d8["grand_metrics"]["r2"]["worst_event_f1"],
    ]
    siam_deltas = [
        d8["grand_metrics"]["delta"]["f1"],
        d8["grand_metrics"]["delta"]["bottom3_mean_f1"],
        d8["grand_metrics"]["delta"]["worst_event_f1"],
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax.bar(x - width/2, snunet_deltas, width, label="SNUNet-ECAM (Primary, 3 Seeds)", color=PALETTE["r0"], alpha=0.88, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, siam_deltas, width, label="SiamUnet_diff (Transfer, Seed 0)", color=PALETTE["r2"], alpha=0.88, edgecolor="black", linewidth=0.8)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("F1 Score Delta (Recipe R2 − Baseline R0)")
    ax.set_title("Figure 7: Cross-Backbone Architecture Transfer Evaluation on SiamUnet_diff (RST8)\n(Positive Confirmation on SNUNet-ECAM vs. Negative Transfer on SiamUnet_diff)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(-0.06, 0.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    for rect in rects1:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -9),
                    textcoords="offset points", ha="center", va=va, fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -9),
                    textcoords="offset points", ha="center", va=va, fontsize=8.5, fontweight="bold")

    data_export = [
        {
            "metric": m,
            "snunet_r0": round(sn_r0, 4),
            "snunet_r2": round(sn_r2, 4),
            "snunet_ecam_delta": round(sn, 4),
            "siamunet_r0": round(si_r0, 4),
            "siamunet_r2": round(si_r2, 4),
            "siamunet_diff_delta": round(si, 4),
        }
        for m, sn_r0, sn_r2, sn, si_r0, si_r2, si in zip(
            metrics, snunet_r0, snunet_r2, snunet_deltas, siam_r0, siam_r2, siam_deltas
        )
    ]
    _save_fig(fig, out_dir / "fig7_architecture_transfer_deltas", data_export,
              ["metric", "snunet_r0", "snunet_r2", "snunet_ecam_delta", "siamunet_r0", "siamunet_r2", "siamunet_diff_delta"])


def build_fig8_historical_target_extension_deltas(rst9_analysis_json: Path, out_dir: Path):
    """Figure 8 (formerly Fig 8): Historical target extension paired deltas across 16 events."""
    with open(rst9_analysis_json, "r", encoding="utf-8") as f:
        d = json.load(f)

    events = d["per_event_table"]
    events.sort(key=lambda x: x["delta_f1"])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y_pos = np.arange(len(events))

    data_export = []
    for idx, e in enumerate(events):
        color = PALETTE["dev"] if e["event_type"] == "dev" else PALETTE["test"]
        d_val = e["delta_f1"]
        ax.barh(idx, d_val, color=color, alpha=0.88, edgecolor="black", linewidth=0.7)
        va = "left" if d_val >= 0 else "right"
        offset = 0.003 if d_val >= 0 else -0.003
        ax.text(d_val + offset, idx, f"{d_val:+.3f}", va="center", ha=va, fontsize=8)

        data_export.append({
            "event_id": e["event_id"],
            "event_type": e["event_type"],
            "r0_f1": e["r0_f1_mean"],
            "r2_f1": e["r2_f1_mean"],
            "delta_f1": e["delta_f1"],
        })

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Event {e['event_id']} ({e['event_type']})" for e in events])
    ax.set_xlabel("Paired Difference in F1 Score (Recipe R2 − Baseline R0)")
    ax.set_title("Figure 8: RST9 Retrospective Historical-Target Extension\n(16 Previously Observed Events, 3 Seeds — Retrospective Analysis)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["dev"], edgecolor="black", label="6 Historical Development Events"),
        Patch(facecolor=PALETTE["test"], edgecolor="black", label="10 Historical Test Events"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    _save_fig(fig, out_dir / "fig8_historical_target_extension_deltas", data_export,
              ["event_id", "event_type", "r0_f1", "r2_f1", "delta_f1"])


def main():
    out_dir = REPO_ROOT / "outputs/robust_source_training/rst11_v3/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating authoritative RST11 v3 publication figures in: {out_dir}")

    build_fig1_event_sampling_distribution(
        REPO_ROOT / "configs/robust_source_training/rst1_source_event_inventory_v1.csv",
        out_dir,
    )

    build_fig2_component_staircase(
        REPO_ROOT / "outputs/robust_source_training/rst4/analysis_v1/discovery_recipe_summary.csv",
        out_dir,
    )

    build_fig3_primary_per_event_paired_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst7/canonical_event_results_v1.csv",
        out_dir,
    )

    build_fig4_worst_bottom_event_comparison(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        out_dir,
    )

    build_fig5_precision_recall_tradeoff(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        out_dir,
    )

    build_fig6_seed_stability(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        out_dir,
    )

    build_fig7_architecture_transfer_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst8/stage_a_summary_v1.json",
        out_dir,
    )

    build_fig8_historical_target_extension_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        out_dir,
    )

    print("All 8 figures successfully generated!")


if __name__ == "__main__":
    main()
