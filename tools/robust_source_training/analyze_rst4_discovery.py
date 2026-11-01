#!/usr/bin/env python3
"""Deterministic offline analysis and promotion decision engine for RST4 Discovery Staircase.

Consumes completed manifests and event metric files directly from disk.
Generates tabular summaries, pairwise component differences, and automated promotion decision.
"""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="RST4 Discovery Offline Analysis Tool")
    parser.add_argument(
        "--matrix",
        type=str,
        default="configs/robust_source_training/rst4_discovery_matrix_v1.json",
        help="Path to discovery matrix config",
    )
    parser.add_argument(
        "--promotion_rule",
        type=str,
        default="configs/robust_source_training/rst4_promotion_rule_v1.json",
        help="Path to promotion rule config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/robust_source_training/rst4/analysis_v1",
        help="Directory to save analysis CSVs and JSONs",
    )
    return parser.parse_args()


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def analyze_discovery_runs(matrix_path: Path, rule_path: Path, output_dir: Path):
    matrix_cfg = load_json(matrix_path)
    rule_cfg = load_json(rule_path)
    repo_root = matrix_path.resolve().parents[2]

    output_dir.mkdir(parents=True, exist_ok=True)

    runs = matrix_cfg.get("runs", [])
    if len(runs) != 4:
        raise ValueError(f"Expected 4 runs in matrix, found {len(runs)}")

    recipe_data: Dict[str, Dict[str, Any]] = {}
    per_event_rows: List[Dict[str, Any]] = []

    for r_entry in runs:
        recipe_id = r_entry["recipe_id"]
        run_id = r_entry["run_id"]
        out_root = repo_root / r_entry["output_root"]

        term_status_file = out_root / "terminal_status.json"
        if not term_status_file.exists():
            raise FileNotFoundError(f"Missing terminal_status.json for {run_id} at {term_status_file}")
        term_status = load_json(term_status_file)
        if term_status.get("status") != "COMPLETED_SCIENTIFIC_RUN":
            raise RuntimeError(f"Run {run_id} did not complete successfully: {term_status}")

        pref_manifest_file = out_root / "preflight_manifest.json"
        final_manifest_file = out_root / "final_manifest.json"
        metrics_file = out_root / "event_metrics.json"

        pref = load_json(pref_manifest_file)
        final = load_json(final_manifest_file)
        metrics = load_json(metrics_file)

        m2_weights_file = out_root / "training_class_weights.json"
        m2_weights = None
        if m2_weights_file.exists():
            m2_weights = load_json(m2_weights_file)

        per_event_list = metrics.get("per_event", [])
        if len(per_event_list) != 9:
            raise ValueError(f"Run {run_id} evaluated {len(per_event_list)} events; expected exactly 9 Group C events.")

        # Compute bottom-3 event mean F1
        f1_sorted = sorted([ev["f1"] for ev in per_event_list])
        bottom_3_f1 = sum(f1_sorted[:3]) / 3.0

        worst_ev = metrics.get("worst_event", {})

        obs = final.get("observed_execution", {})
        macro = metrics.get("event_macro", {})
        pooled = metrics.get("pooled", {})

        recipe_data[recipe_id] = {
            "recipe_id": recipe_id,
            "run_id": run_id,
            "components": r_entry["components"],
            "terminal_status": term_status.get("status"),
            "event_macro_f1": float(macro.get("f1", 0.0)),
            "event_macro_iou": float(macro.get("iou", 0.0)),
            "event_macro_precision": float(macro.get("precision", 0.0)),
            "event_macro_recall": float(macro.get("recall", 0.0)),
            "pooled_f1": float(pooled.get("f1", 0.0)),
            "pooled_iou": float(pooled.get("iou", 0.0)),
            "worst_event_id": worst_ev.get("event_id"),
            "worst_event_f1": float(worst_ev.get("f1", 0.0)),
            "bottom_3_event_mean_f1": float(bottom_3_f1),
            "total_observed_samples": obs.get("observed_sample_count"),
            "total_optimizer_steps": obs.get("optimizer_steps"),
            "wall_clock_runtime_seconds": obs.get("wall_clock_runtime_seconds"),
            "peak_vram_mb": obs.get("peak_vram_mb"),
            "final_checkpoint_sha256": final.get("output_artifacts", {}).get("final_checkpoint_sha256"),
            "class_weights": m2_weights.get("final_weights_list") if m2_weights else [1.0, 1.0, 1.0],
            "per_event": {ev["event_id"]: ev for ev in per_event_list},
        }

        for ev in per_event_list:
            per_event_rows.append({
                "recipe_id": recipe_id,
                "run_id": run_id,
                "event_id": ev["event_id"],
                "sample_count": ev["sample_count"],
                "valid_pixel_count": ev["valid_pixel_count"],
                "tp": ev["tp"],
                "fp": ev["fp"],
                "fn": ev["fn"],
                "tn": ev["tn"],
                "pred_flood_pixels": ev["pred_flood_pixels"],
                "ref_flood_pixels": ev["ref_flood_pixels"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
                "iou": ev["iou"],
            })

    # Write 1. discovery_event_metrics.csv
    event_metrics_csv = output_dir / "discovery_event_metrics.csv"
    with open(event_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "recipe_id", "run_id", "event_id", "sample_count", "valid_pixel_count",
            "tp", "fp", "fn", "tn", "pred_flood_pixels", "ref_flood_pixels",
            "precision", "recall", "f1", "iou"
        ])
        writer.writeheader()
        writer.writerows(per_event_rows)

    # 2. Compute differences and promotion rule
    r0 = recipe_data["R0"]
    diff_rows: List[Dict[str, Any]] = []
    comparisons = [("R1", "R0", "M1 (Event Balance)"),
                   ("R2", "R1", "M2 (Flood-Aware Loss)"),
                   ("R3", "R2", "M3 (Paired Augmentation)"),
                   ("R2", "R0", "M1 + M2 vs Baseline"),
                   ("R3", "R0", "Full Composite vs Baseline")]

    for target, base, comp_name in comparisons:
        t_data = recipe_data[target]
        b_data = recipe_data[base]

        delta_f1 = t_data["event_macro_f1"] - b_data["event_macro_f1"]
        delta_iou = t_data["event_macro_iou"] - b_data["event_macro_iou"]
        delta_prec = t_data["event_macro_precision"] - b_data["event_macro_precision"]
        delta_rec = t_data["event_macro_recall"] - b_data["event_macro_recall"]
        delta_worst = t_data["worst_event_f1"] - b_data["worst_event_f1"]
        delta_b3 = t_data["bottom_3_event_mean_f1"] - b_data["bottom_3_event_mean_f1"]

        # Per event benefits / harms
        n_improved = 0
        n_harmed = 0
        n_neutral = 0
        for eid in t_data["per_event"]:
            t_f1 = t_data["per_event"][eid]["f1"]
            b_f1 = b_data["per_event"][eid]["f1"]
            diff = t_f1 - b_f1
            if diff >= 0.01:
                n_improved += 1
            elif diff <= -0.01:
                n_harmed += 1
            else:
                n_neutral += 1

        diff_rows.append({
            "comparison": f"{target} - {base}",
            "component_analyzed": comp_name,
            "target_recipe": target,
            "baseline_recipe": base,
            "delta_event_macro_f1": round(delta_f1, 6),
            "delta_event_macro_iou": round(delta_iou, 6),
            "delta_event_macro_precision": round(delta_prec, 6),
            "delta_event_macro_recall": round(delta_rec, 6),
            "delta_worst_event_f1": round(delta_worst, 6),
            "delta_bottom_3_event_mean_f1": round(delta_b3, 6),
            "improved_events_count": n_improved,
            "harmed_events_count": n_harmed,
            "neutral_events_count": n_neutral,
        })

    # Write 2. component_differences.csv
    comp_diff_csv = output_dir / "component_differences.csv"
    with open(comp_diff_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "comparison", "component_analyzed", "target_recipe", "baseline_recipe",
            "delta_event_macro_f1", "delta_event_macro_iou", "delta_event_macro_precision",
            "delta_event_macro_recall", "delta_worst_event_f1", "delta_bottom_3_event_mean_f1",
            "improved_events_count", "harmed_events_count", "neutral_events_count"
        ])
        writer.writeheader()
        writer.writerows(diff_rows)

    # 3. Write discovery_recipe_summary.csv
    summary_rows: List[Dict[str, Any]] = []
    for r in ["R0", "R1", "R2", "R3"]:
        d = recipe_data[r]
        # find vs R0 delta
        d_f1 = d["event_macro_f1"] - r0["event_macro_f1"]
        d_b3 = d["bottom_3_event_mean_f1"] - r0["bottom_3_event_mean_f1"]
        n_imp = 0
        n_harm = 0
        for eid in d["per_event"]:
            diff = d["per_event"][eid]["f1"] - r0["per_event"][eid]["f1"]
            if diff >= 0.01:
                n_imp += 1
            elif diff <= -0.01:
                n_harm += 1

        summary_rows.append({
            "recipe_id": r,
            "run_id": d["run_id"],
            "m1_balance": d["components"]["m1_event_balance"],
            "m2_loss": d["components"]["m2_flood_aware_loss"],
            "m3_aug": d["components"]["m3_paired_augmentation"],
            "event_macro_f1": round(d["event_macro_f1"], 6),
            "event_macro_iou": round(d["event_macro_iou"], 6),
            "event_macro_precision": round(d["event_macro_precision"], 6),
            "event_macro_recall": round(d["event_macro_recall"], 6),
            "pooled_f1": round(d["pooled_f1"], 6),
            "pooled_iou": round(d["pooled_iou"], 6),
            "worst_event_id": d["worst_event_id"],
            "worst_event_f1": round(d["worst_event_f1"], 6),
            "bottom_3_event_mean_f1": round(d["bottom_3_event_mean_f1"], 6),
            "delta_f1_vs_r0": round(d_f1, 6),
            "delta_bottom3_vs_r0": round(d_b3, 6),
            "improved_events_vs_r0": n_imp,
            "harmed_events_vs_r0": n_harm,
            "runtime_seconds": d["wall_clock_runtime_seconds"],
            "peak_vram_mb": d["peak_vram_mb"],
            "final_checkpoint_sha256": d["final_checkpoint_sha256"],
        })

    summary_csv = output_dir / "discovery_recipe_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "recipe_id", "run_id", "m1_balance", "m2_loss", "m3_aug",
            "event_macro_f1", "event_macro_iou", "event_macro_precision", "event_macro_recall",
            "pooled_f1", "pooled_iou", "worst_event_id", "worst_event_f1", "bottom_3_event_mean_f1",
            "delta_f1_vs_r0", "delta_bottom3_vs_r0", "improved_events_vs_r0", "harmed_events_vs_r0",
            "runtime_seconds", "peak_vram_mb", "final_checkpoint_sha256"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # 4. Evaluate promotion rule
    eligible_candidates: Dict[str, Dict[str, Any]] = {}
    for r in ["R1", "R2", "R3"]:
        d = recipe_data[r]
        delta_f1 = d["event_macro_f1"] - r0["event_macro_f1"]
        delta_b3 = d["bottom_3_event_mean_f1"] - r0["bottom_3_event_mean_f1"]
        n_imp = 0
        n_harm = 0
        for eid in d["per_event"]:
            diff = d["per_event"][eid]["f1"] - r0["per_event"][eid]["f1"]
            if diff >= 0.01:
                n_imp += 1
            elif diff <= -0.01:
                n_harm += 1

        route_d1 = (delta_f1 > 0.0) and (n_imp >= n_harm)
        route_d2 = (delta_b3 >= 0.02) and (delta_f1 >= -0.005)

        if route_d1 or route_d2:
            eligible_candidates[r] = {
                "recipe_id": r,
                "event_macro_f1": d["event_macro_f1"],
                "delta_f1_vs_r0": delta_f1,
                "delta_b3_vs_r0": delta_b3,
                "route_d1_satisfied": bool(route_d1),
                "route_d2_satisfied": bool(route_d2),
            }

    selected_recipe = None
    selection_reason = ""
    decision_status = ""

    if not eligible_candidates:
        selected_recipe = None
        decision_status = "NO_RECIPE_PROMOTED"
        selection_reason = "No candidate recipe satisfied Route D1 (mean improvement) or Route D2 (tail reliability improvement) relative to R0."
    else:
        # Step 3: Highest event macro F1 among eligible
        max_f1 = max(c["event_macro_f1"] for c in eligible_candidates.values())
        # Step 4: Within 0.005 bandwidth of best
        bandwidth_pool = [
            r for r, c in eligible_candidates.items()
            if (max_f1 - c["event_macro_f1"]) <= 0.005
        ]
        # Step 5: Simplest hierarchy R1 -> R2 -> R3
        hierarchy = ["R1", "R2", "R3"]
        for cand in hierarchy:
            if cand in bandwidth_pool:
                selected_recipe = cand
                break

        decision_status = f"RECIPE_{selected_recipe}_PROMOTED"
        selection_reason = (
            f"Recipe {selected_recipe} promoted: eligible under criteria "
            f"(D1={eligible_candidates[selected_recipe]['route_d1_satisfied']}, "
            f"D2={eligible_candidates[selected_recipe]['route_d2_satisfied']}), "
            f"achieved event-macro F1 {eligible_candidates[selected_recipe]['event_macro_f1']:.6f} "
            f"(within 0.005 bandwidth of max {max_f1:.6f}), and selected by parsimony preference."
        )

    promotion_decision = {
        "$schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision_status,
        "selected_recipe": selected_recipe,
        "selection_reason": selection_reason,
        "rule_reference": {
            "path": str(rule_path),
            "sha256": sha256_file(rule_path),
        },
        "baseline_summary": {
            "recipe_id": "R0",
            "run_id": r0["run_id"],
            "event_macro_f1": r0["event_macro_f1"],
            "event_macro_iou": r0["event_macro_iou"],
            "worst_event_f1": r0["worst_event_f1"],
            "bottom_3_event_mean_f1": r0["bottom_3_event_mean_f1"],
        },
        "candidate_evaluations": eligible_candidates,
    }

    decision_json_path = output_dir / "promotion_decision.json"
    with open(decision_json_path, "w", encoding="utf-8") as f:
        json.dump(promotion_decision, f, indent=2)
        f.write("\n")

    # 5. Write analysis_manifest.json
    analysis_manifest = {
        "$schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_config_sha256": sha256_file(matrix_path),
        "promotion_rule_sha256": sha256_file(rule_path),
        "analyzed_runs": {
            r: {
                "run_id": recipe_data[r]["run_id"],
                "final_checkpoint_sha256": recipe_data[r]["final_checkpoint_sha256"],
                "event_metrics_sha256": sha256_file(repo_root / matrix_cfg["runs"][i]["output_root"] / "event_metrics.json"),
            } for i, r in enumerate(["R0", "R1", "R2", "R3"])
        },
        "output_artifacts": {
            "discovery_recipe_summary_csv": sha256_file(summary_csv),
            "discovery_event_metrics_csv": sha256_file(event_metrics_csv),
            "component_differences_csv": sha256_file(comp_diff_csv),
            "promotion_decision_json": sha256_file(decision_json_path),
        },
    }

    manifest_json_path = output_dir / "analysis_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_manifest, f, indent=2)
        f.write("\n")

    print(f"Analysis complete. Outputs generated in {output_dir}")
    print(f"Decision: {decision_status} (Selected: {selected_recipe})")


if __name__ == "__main__":
    args = parse_args()
    analyze_discovery_runs(
        matrix_path=Path(args.matrix),
        rule_path=Path(args.promotion_rule),
        output_dir=Path(args.output_dir),
    )

