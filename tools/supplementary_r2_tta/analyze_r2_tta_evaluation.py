#!/usr/bin/env python3
"""ChangeGuard Supplementary Evaluation Analysis Engine: R2 + Tent / CoTTA.

Performs rigorous statistical analysis across the six experimental conditions:
1. Fixed R0 (reused historical baseline)
2. R0 + Tent (reused E11 final test)
3. R0 + CoTTA (reused E11 final test)
4. Fixed R2 (reused RST9 robust source model)
5. R2 + Tent (new supplementary evaluation)
6. R2 + CoTTA (new supplementary evaluation)

Statistical Invariants:
- 10 Kuro Siwo historical target events treated as 10 independent event units.
- Multi-seed evaluations averaged within each event prior to event-level paired analysis.
- Paired event-level bootstrap with 10,000 resamples (seed 42), reusing exact resample indices across contrasts.
- Pointwise 95% bootstrap confidence intervals conditional on frozen trained checkpoints.
- Practical threshold band: ±0.010 (±1.0 percentage points) for benefit / neutral / harm classification.
- Full reporting of per-event, macro, pooled, worst-event, and bottom-3 metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HISTORICAL_TEST_EVENTS = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]

INPUT_RESOURCES = {
    "rst9_target_metrics": REPO_ROOT / "outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv",
    "e11_summary": REPO_ROOT / "outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json",
}


def classify_delta(delta: float, threshold: float = 0.010) -> str:
    if delta >= threshold:
        return "benefit"
    elif delta <= -threshold:
        return "harm"
    else:
        return "neutral"


def run_paired_event_bootstrap(
    event_deltas: Dict[str, np.ndarray],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Runs paired event-level bootstrap across 10 events reusing identical bootstrap indices."""
    rng = np.random.RandomState(seed)
    n_events = 10

    # Draw common bootstrap indices
    boot_indices = [rng.randint(0, n_events, size=n_events) for _ in range(n_resamples)]

    results = {}
    for contrast_name, deltas in event_deltas.items():
        assert len(deltas) == n_events, f"{contrast_name} has {len(deltas)} events, expected {n_events}"
        boot_means = np.zeros(n_resamples, dtype=np.float64)
        for b in range(n_resamples):
            boot_means[b] = np.mean(deltas[boot_indices[b]])

        boot_means.sort()
        obs_mean = float(np.mean(deltas))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        results[contrast_name] = {
            "contrast_name": contrast_name,
            "observed_mean_delta": obs_mean,
            "ci_95": [ci_lower, ci_upper],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_resamples": n_resamples,
            "bootstrap_seed": seed,
        }
    return results


class R2TTAAnalyzer:
    """Analyzes and synthesizes the 6-condition comparison."""

    def __init__(
        self,
        supplementary_output_dir: Path,
        analysis_output_dir: Path,
    ):
        self.supp_dir = supplementary_output_dir.resolve()
        self.out_dir = analysis_output_dir.resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_historical_r0_r2(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads fixed R0 and fixed R2 metrics from all_historical_target_metrics.csv."""
        csv_path = INPUT_RESOURCES["rst9_target_metrics"]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {csv_path}")

        r0_by_seed_event = {0: {}, 1: {}, 2: {}}
        r2_by_seed_event = {0: {}, 1: {}, 2: {}}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recipe = row["recipe"]
                seed = int(row["seed"])
                event_id = int(row["event_id"])
                if event_id not in HISTORICAL_TEST_EVENTS:
                    continue
                metrics = {
                    "event_id": event_id,
                    "samples": int(row["sample_count"]),
                    "valid_pixels": int(row["valid_pixel_count"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tn": int(row["tn"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "iou": float(row["iou"]),
                }
                if recipe == "R0":
                    r0_by_seed_event[seed][event_id] = metrics
                elif recipe == "R2":
                    r2_by_seed_event[seed][event_id] = metrics

        return r0_by_seed_event, r2_by_seed_event

    def load_e11_tta(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads R0 + Tent and R0 + CoTTA metrics from E11 consolidated summary."""
        e11_path = INPUT_RESOURCES["e11_summary"]
        if not e11_path.is_file():
            raise FileNotFoundError(f"Missing {e11_path}")

        with open(e11_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        snunet_data = data["SNUNet_ECAM"]
        tent_seeds = snunet_data["tent"]["seeds"]
        cotta_seeds = snunet_data["cotta"]["seeds"]

        def parse_method(seeds_dict):
            out = {0: {}, 1: {}, 2: {}}
            for s_str, s_info in seeds_dict.items():
                s = int(s_str)
                for ev in s_info["event_table"]:
                    eid = int(ev["event_id"])
                    if eid in HISTORICAL_TEST_EVENTS:
                        out[s][eid] = {
                            "event_id": eid,
                            "samples": int(ev["sample_count"]),
                            "valid_pixels": int(ev["valid_pixels"]),
                            "tp": int(ev["tp"]),
                            "fp": int(ev["fp"]),
                            "fn": int(ev["fn"]),
                            "tn": int(ev["tn"]),
                            "precision": float(ev["precision"]),
                            "recall": float(ev["recall"]),
                            "f1": float(ev["f1"]),
                            "iou": float(ev["iou"]),
                        }
            return out

        return parse_method(tent_seeds), parse_method(cotta_seeds)

    def load_supplementary_r2_tta(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads newly evaluated R2 + Tent and R2 + CoTTA stream summaries."""
        r2_tent_by_seed_event = {0: {}, 1: {}, 2: {}}
        r2_cotta_by_seed_event = {0: {}, 1: {}, 2: {}}

        for s in [0, 1, 2]:
            # Tent
            t_path = self.supp_dir / f"r2_snunet_tent_seed{s}" / "stream_summary.json"
            if not t_path.is_file():
                raise FileNotFoundError(f"Missing completed stream summary: {t_path}")
            with open(t_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            for ev in t_data["event_summaries"]:
                eid = int(ev["event_id"])
                c = ev["confusion"]
                r2_tent_by_seed_event[s][eid] = {
                    "event_id": eid,
                    "samples": ev["samples"],
                    "valid_pixels": c["valid_pixels"],
                    "tp": c["tp"],
                    "fp": c["fp"],
                    "fn": c["fn"],
                    "tn": c["tn"],
                    "precision": ev["flood_precision"],
                    "recall": ev["flood_recall"],
                    "f1": ev["flood_f1"],
                    "iou": ev["flood_iou"],
                    "updates_applied": ev["updates_applied"],
                    "updates_skipped": ev["updates_skipped"],
                    "updates_failed": ev["updates_failed"],
                }

            # CoTTA
            c_path = self.supp_dir / f"r2_snunet_cotta_seed{s}" / "stream_summary.json"
            if not c_path.is_file():
                raise FileNotFoundError(f"Missing completed stream summary: {c_path}")
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            for ev in c_data["event_summaries"]:
                eid = int(ev["event_id"])
                c = ev["confusion"]
                r2_cotta_by_seed_event[s][eid] = {
                    "event_id": eid,
                    "samples": ev["samples"],
                    "valid_pixels": c["valid_pixels"],
                    "tp": c["tp"],
                    "fp": c["fp"],
                    "fn": c["fn"],
                    "tn": c["tn"],
                    "precision": ev["flood_precision"],
                    "recall": ev["flood_recall"],
                    "f1": ev["flood_f1"],
                    "iou": ev["flood_iou"],
                    "updates_applied": ev["updates_applied"],
                    "updates_skipped": ev["updates_skipped"],
                    "updates_failed": ev["updates_failed"],
                }

        return r2_tent_by_seed_event, r2_cotta_by_seed_event

    def aggregate_condition(
        self,
        seed_event_dict: Dict[int, Dict[int, Dict[str, Any]]],
        condition_name: str,
    ) -> Dict[str, Any]:
        """Aggregates metrics for a condition across seeds and events."""
        seeds = [0, 1, 2]
        events = HISTORICAL_TEST_EVENTS

        # 1. Per-event seed averages
        event_f1s = []
        event_ious = []
        event_precs = []
        event_recs = []
        per_event_table = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_valid = 0
        total_samples = 0

        for eid in events:
            f1_seeds = [seed_event_dict[s][eid]["f1"] for s in seeds]
            iou_seeds = [seed_event_dict[s][eid]["iou"] for s in seeds]
            prec_seeds = [seed_event_dict[s][eid]["precision"] for s in seeds]
            rec_seeds = [seed_event_dict[s][eid]["recall"] for s in seeds]

            mean_f1 = float(np.mean(f1_seeds))
            mean_iou = float(np.mean(iou_seeds))
            mean_prec = float(np.mean(prec_seeds))
            mean_rec = float(np.mean(rec_seeds))

            ev_samples = seed_event_dict[0][eid]["samples"]
            ev_valid = seed_event_dict[0][eid]["valid_pixels"]

            # Sum confusion across all 3 seeds for pooled
            for s in seeds:
                total_tp += seed_event_dict[s][eid]["tp"]
                total_fp += seed_event_dict[s][eid]["fp"]
                total_fn += seed_event_dict[s][eid]["fn"]
                total_tn += seed_event_dict[s][eid]["tn"]
                total_valid += seed_event_dict[s][eid]["valid_pixels"]
                total_samples += seed_event_dict[s][eid]["samples"]

            event_f1s.append(mean_f1)
            event_ious.append(mean_iou)
            event_precs.append(mean_prec)
            event_recs.append(mean_rec)

            per_event_table.append({
                "event_id": eid,
                "samples": ev_samples,
                "valid_pixels": ev_valid,
                "f1": mean_f1,
                "iou": mean_iou,
                "precision": mean_prec,
                "recall": mean_rec,
                "f1_by_seed": f1_seeds,
            })

        # Macro averages
        macro_f1 = float(np.mean(event_f1s))
        macro_iou = float(np.mean(event_ious))
        macro_prec = float(np.mean(event_precs))
        macro_rec = float(np.mean(event_recs))

        # Per-seed macro F1s
        seed_macro_f1s = []
        for s in seeds:
            s_f1s = [seed_event_dict[s][eid]["f1"] for eid in events]
            seed_macro_f1s.append(float(np.mean(s_f1s)))

        seed_mean_f1 = float(np.mean(seed_macro_f1s))
        seed_sd_f1 = float(np.std(seed_macro_f1s, ddof=1))

        # Worst event and Bottom-3
        sorted_f1s = sorted(event_f1s)
        worst_event_f1 = float(sorted_f1s[0])
        bottom_3_f1 = float(np.mean(sorted_f1s[:3]))

        # Pooled metrics
        pooled_prec = float(total_tp / max(total_tp + total_fp, 1))
        pooled_rec = float(total_tp / max(total_tp + total_fn, 1))
        pooled_f1 = float(
            2 * pooled_prec * pooled_rec / max(pooled_prec + pooled_rec, 1e-12)
        )
        pooled_iou = float(
            total_tp / max(total_tp + total_fp + total_fn, 1)
        )

        return {
            "condition": condition_name,
            "macro_f1": macro_f1,
            "macro_iou": macro_iou,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "seed_macro_f1s": seed_macro_f1s,
            "seed_mean_f1": seed_mean_f1,
            "seed_sd_f1": seed_sd_f1,
            "worst_event_f1": worst_event_f1,
            "bottom_3_f1": bottom_3_f1,
            "pooled_f1": pooled_f1,
            "pooled_iou": pooled_iou,
            "event_f1_array": np.array(event_f1s, dtype=np.float64),
            "per_event_table": per_event_table,
        }

    def run_analysis(self) -> Dict[str, Any]:
        """Executes complete six-condition comparative analysis."""
        print(">>> Loading historical R0, R2, and E11 TTA datasets...")
        r0_dict, r2_dict = self.load_historical_r0_r2()
        r0_tent_dict, r0_cotta_dict = self.load_e11_tta()

        print(">>> Loading newly evaluated R2 + Tent and R2 + CoTTA results...")
        r2_tent_dict, r2_cotta_dict = self.load_supplementary_r2_tta()

        conditions = {
            "fixed_r0": self.aggregate_condition(r0_dict, "Fixed R0"),
            "r0_tent": self.aggregate_condition(r0_tent_dict, "R0 + Tent"),
            "r0_cotta": self.aggregate_condition(r0_cotta_dict, "R0 + CoTTA"),
            "fixed_r2": self.aggregate_condition(r2_dict, "Fixed R2"),
            "r2_tent": self.aggregate_condition(r2_tent_dict, "R2 + Tent"),
            "r2_cotta": self.aggregate_condition(r2_cotta_dict, "R2 + CoTTA"),
        }

        # Event-level deltas
        deltas = {
            # Primary contrasts
            "r2_tent_vs_fixed_r2": conditions["r2_tent"]["event_f1_array"] - conditions["fixed_r2"]["event_f1_array"],
            "r2_cotta_vs_fixed_r2": conditions["r2_cotta"]["event_f1_array"] - conditions["fixed_r2"]["event_f1_array"],
            # Secondary matched contrasts
            "r2_tent_vs_r0_tent": conditions["r2_tent"]["event_f1_array"] - conditions["r0_tent"]["event_f1_array"],
            "r2_cotta_vs_r0_cotta": conditions["r2_cotta"]["event_f1_array"] - conditions["r0_cotta"]["event_f1_array"],
            # Contextual contrasts
            "fixed_r2_vs_fixed_r0": conditions["fixed_r2"]["event_f1_array"] - conditions["fixed_r0"]["event_f1_array"],
            "r0_tent_vs_fixed_r0": conditions["r0_tent"]["event_f1_array"] - conditions["fixed_r0"]["event_f1_array"],
            "r0_cotta_vs_fixed_r0": conditions["r0_cotta"]["event_f1_array"] - conditions["fixed_r0"]["event_f1_array"],
        }

        print(">>> Running paired event-level bootstrap (10,000 resamples, seed 42)...")
        bootstrap_results = run_paired_event_bootstrap(deltas, n_resamples=10000, seed=42)

        # Categorize deltas across 10 events
        def get_delta_counts(arr: np.ndarray) -> Dict[str, int]:
            benefit = int(np.sum(arr >= 0.010))
            harm = int(np.sum(arr <= -0.010))
            neutral = int(np.sum(np.abs(arr) < 0.010))
            pos = int(np.sum(arr > 0))
            neg = int(np.sum(arr < 0))
            zero = int(np.sum(arr == 0))
            return {
                "benefit": benefit,
                "harm": harm,
                "neutral": neutral,
                "positive": pos,
                "negative": neg,
                "zero": zero,
            }

        delta_counts = {k: get_delta_counts(v) for k, v in deltas.items()}

        # Collect updates telemetry for R2 streams
        telemetry = {}
        for s in [0, 1, 2]:
            t_path = self.supp_dir / f"r2_snunet_tent_seed{s}" / "stream_summary.json"
            c_path = self.supp_dir / f"r2_snunet_cotta_seed{s}" / "stream_summary.json"
            with open(t_path, "r", encoding="utf-8") as f:
                t_sum = json.load(f)
            with open(c_path, "r", encoding="utf-8") as f:
                c_sum = json.load(f)
            telemetry[f"r2_tent_s{s}"] = {
                "elapsed_seconds": t_sum["elapsed_seconds"],
                "peak_vram_mb": t_sum["peak_vram_mb"],
                "updates_applied": t_sum["total_updates_applied"],
                "updates_skipped": t_sum["total_updates_skipped"],
                "updates_failed": t_sum["total_updates_failed"],
            }
            telemetry[f"r2_cotta_s{s}"] = {
                "elapsed_seconds": c_sum["elapsed_seconds"],
                "peak_vram_mb": c_sum["peak_vram_mb"],
                "updates_applied": c_sum["total_updates_applied"],
                "updates_skipped": c_sum["total_updates_skipped"],
                "updates_failed": c_sum["total_updates_failed"],
            }

        # Serialized output dictionary
        analysis_summary = {
            "title": "ChangeGuard Supplementary Evaluation: R2 + Tent / CoTTA Analysis",
            "conditions": {
                k: {
                    m: v[m] for m in v if m != "event_f1_array"
                }
                for k, v in conditions.items()
            },
            "bootstrap_contrasts": bootstrap_results,
            "delta_counts": delta_counts,
            "r2_tta_telemetry": telemetry,
        }

        # Write JSON summary
        json_out_path = self.out_dir / "r2_tta_analysis_summary.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_summary, f, indent=2)
        print(f"✓ Summary JSON written to: {json_out_path}")

        # Generate comprehensive markdown report
        self._write_markdown_report(conditions, bootstrap_results, delta_counts, telemetry)
        return analysis_summary

    def _write_markdown_report(
        self,
        conditions: Dict[str, Any],
        bootstrap_results: Dict[str, Any],
        delta_counts: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> None:
        """Writes detailed markdown analysis report."""
        report_path = self.out_dir / "r2_tta_analysis_report.md"

        c_r0 = conditions["fixed_r0"]
        c_r0_tent = conditions["r0_tent"]
        c_r0_cotta = conditions["r0_cotta"]
        c_r2 = conditions["fixed_r2"]
        c_r2_tent = conditions["r2_tent"]
        c_r2_cotta = conditions["r2_cotta"]

        lines = [
            "# Supplementary Evaluation Report: Test-Time Adaptation on Robust Source Model (R2 + TTA)",
            "",
            "**Research Question:** Does Tent or CoTTA provide additional benefit when initialized from R2, compared with deploying the same R2 checkpoint without adaptation?",
            "",
            "---",
            "",
            "## 1. Six-Condition Consolidated Benchmark Table",
            "",
            "| Condition | Macro F1 | Macro IoU | Macro Prec | Macro Rec | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Seed Mean ± SD | Worst Event | Bottom-3 F1 | Pooled F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Fixed R0** | {c_r0['macro_f1']:.4f} | {c_r0['macro_iou']:.4f} | {c_r0['macro_precision']:.4f} | {c_r0['macro_recall']:.4f} | {c_r0['seed_macro_f1s'][0]:.4f} | {c_r0['seed_macro_f1s'][1]:.4f} | {c_r0['seed_macro_f1s'][2]:.4f} | {c_r0['seed_mean_f1']:.4f} ± {c_r0['seed_sd_f1']:.4f} | {c_r0['worst_event_f1']:.4f} | {c_r0['bottom_3_f1']:.4f} | {c_r0['pooled_f1']:.4f} |",
            f"| **R0 + Tent** | {c_r0_tent['macro_f1']:.4f} | {c_r0_tent['macro_iou']:.4f} | {c_r0_tent['macro_precision']:.4f} | {c_r0_tent['macro_recall']:.4f} | {c_r0_tent['seed_macro_f1s'][0]:.4f} | {c_r0_tent['seed_macro_f1s'][1]:.4f} | {c_r0_tent['seed_macro_f1s'][2]:.4f} | {c_r0_tent['seed_mean_f1']:.4f} ± {c_r0_tent['seed_sd_f1']:.4f} | {c_r0_tent['worst_event_f1']:.4f} | {c_r0_tent['bottom_3_f1']:.4f} | {c_r0_tent['pooled_f1']:.4f} |",
            f"| **R0 + CoTTA** | {c_r0_cotta['macro_f1']:.4f} | {c_r0_cotta['macro_iou']:.4f} | {c_r0_cotta['macro_precision']:.4f} | {c_r0_cotta['macro_recall']:.4f} | {c_r0_cotta['seed_macro_f1s'][0]:.4f} | {c_r0_cotta['seed_macro_f1s'][1]:.4f} | {c_r0_cotta['seed_macro_f1s'][2]:.4f} | {c_r0_cotta['seed_mean_f1']:.4f} ± {c_r0_cotta['seed_sd_f1']:.4f} | {c_r0_cotta['worst_event_f1']:.4f} | {c_r0_cotta['bottom_3_f1']:.4f} | {c_r0_cotta['pooled_f1']:.4f} |",
            f"| **Fixed R2** | {c_r2['macro_f1']:.4f} | {c_r2['macro_iou']:.4f} | {c_r2['macro_precision']:.4f} | {c_r2['macro_recall']:.4f} | {c_r2['seed_macro_f1s'][0]:.4f} | {c_r2['seed_macro_f1s'][1]:.4f} | {c_r2['seed_macro_f1s'][2]:.4f} | {c_r2['seed_mean_f1']:.4f} ± {c_r2['seed_sd_f1']:.4f} | {c_r2['worst_event_f1']:.4f} | {c_r2['bottom_3_f1']:.4f} | {c_r2['pooled_f1']:.4f} |",
            f"| **R2 + Tent** | {c_r2_tent['macro_f1']:.4f} | {c_r2_tent['macro_iou']:.4f} | {c_r2_tent['macro_precision']:.4f} | {c_r2_tent['macro_recall']:.4f} | {c_r2_tent['seed_macro_f1s'][0]:.4f} | {c_r2_tent['seed_macro_f1s'][1]:.4f} | {c_r2_tent['seed_macro_f1s'][2]:.4f} | {c_r2_tent['seed_mean_f1']:.4f} ± {c_r2_tent['seed_sd_f1']:.4f} | {c_r2_tent['worst_event_f1']:.4f} | {c_r2_tent['bottom_3_f1']:.4f} | {c_r2_tent['pooled_f1']:.4f} |",
            f"| **R2 + CoTTA** | {c_r2_cotta['macro_f1']:.4f} | {c_r2_cotta['macro_iou']:.4f} | {c_r2_cotta['macro_precision']:.4f} | {c_r2_cotta['macro_recall']:.4f} | {c_r2_cotta['seed_macro_f1s'][0]:.4f} | {c_r2_cotta['seed_macro_f1s'][1]:.4f} | {c_r2_cotta['seed_macro_f1s'][2]:.4f} | {c_r2_cotta['seed_mean_f1']:.4f} ± {c_r2_cotta['seed_sd_f1']:.4f} | {c_r2_cotta['worst_event_f1']:.4f} | {c_r2_cotta['bottom_3_f1']:.4f} | {c_r2_cotta['pooled_f1']:.4f} |",
            "",
            "---",
            "",
            "## 2. Statistical Contrasts and Bootstrap Intervals",
            "",
            "Pointwise paired event-bootstrap intervals (10,000 resamples, seed 42) across the 10 historical target events:",
            "",
            "| Contrast Type | Comparison | Observed $\\Delta$ F1 | 95% Bootstrap CI | Benefit ($\\ge +0.010$) | Neutral ($|\\Delta| < 0.010$) | Harm ($\\le -0.010$) | Direction (Pos / Neg / Zero) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        def format_row(ctype, cname, key):
            b = bootstrap_results[key]
            dc = delta_counts[key]
            obs = b["observed_mean_delta"]
            ci = f"[{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}]"
            sign = "+" if obs >= 0 else ""
            b_cnt = f"{dc['benefit']}B / {dc['neutral']}N / {dc['harm']}H"
            d_cnt = f"{dc['positive']} / {dc['negative']} / {dc['zero']}"
            return f"| **{ctype}** | {cname} | **{sign}{obs:.4f}** | **{ci}** | {dc['benefit']} | {dc['neutral']} | {dc['harm']} | {d_cnt} |"

        lines.append(format_row("Primary", "R2 + Tent vs. Fixed R2", "r2_tent_vs_fixed_r2"))
        lines.append(format_row("Primary", "R2 + CoTTA vs. Fixed R2", "r2_cotta_vs_fixed_r2"))
        lines.append(format_row("Secondary", "R2 + Tent vs. R0 + Tent", "r2_tent_vs_r0_tent"))
        lines.append(format_row("Secondary", "R2 + CoTTA vs. R0 + CoTTA", "r2_cotta_vs_r0_cotta"))
        lines.append(format_row("Contextual", "Fixed R2 vs. Fixed R0", "fixed_r2_vs_fixed_r0"))
        lines.append(format_row("Contextual", "R0 + Tent vs. Fixed R0", "r0_tent_vs_fixed_r0"))
        lines.append(format_row("Contextual", "R0 + CoTTA vs. Fixed R0", "r0_cotta_vs_fixed_r0"))

        lines.extend([
            "",
            "---",
            "",
            "## 3. Ten-Event Detail Breakdown (Seed-Averaged F1 Scores)",
            "",
            "| Event ID | Samples | Valid Pixels | Fixed R0 | R0 + Tent | R0 + CoTTA | Fixed R2 | R2 + Tent | R2 + CoTTA | $\\Delta$(R2+Tent - R2) | $\\Delta$(R2+CoTTA - R2) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for i, eid in enumerate(HISTORICAL_TEST_EVENTS):
            f1_r0 = c_r0["per_event_table"][i]["f1"]
            f1_r0_t = c_r0_tent["per_event_table"][i]["f1"]
            f1_r0_c = c_r0_cotta["per_event_table"][i]["f1"]
            f1_r2 = c_r2["per_event_table"][i]["f1"]
            f1_r2_t = c_r2_tent["per_event_table"][i]["f1"]
            f1_r2_c = c_r2_cotta["per_event_table"][i]["f1"]

            d_t = f1_r2_t - f1_r2
            d_c = f1_r2_c - f1_r2
            samples = c_r2["per_event_table"][i]["samples"]
            valid = c_r2["per_event_table"][i]["valid_pixels"]

            lines.append(
                f"| **{eid}** | {samples:,} | {valid:,} | {f1_r0:.4f} | {f1_r0_t:.4f} | {f1_r0_c:.4f} | "
                f"{f1_r2:.4f} | {f1_r2_t:.4f} | {f1_r2_c:.4f} | {d_t:+.4f} | {d_c:+.4f} |"
            )

        # Averages row
        lines.append(
            f"| **Mean** | **15,896** | **787,338,783** | **{c_r0['macro_f1']:.4f}** | **{c_r0_tent['macro_f1']:.4f}** | **{c_r0_cotta['macro_f1']:.4f}** | "
            f"**{c_r2['macro_f1']:.4f}** | **{c_r2_tent['macro_f1']:.4f}** | **{c_r2_cotta['macro_f1']:.4f}** | "
            f"**{bootstrap_results['r2_tent_vs_fixed_r2']['observed_mean_delta']:+.4f}** | **{bootstrap_results['r2_cotta_vs_fixed_r2']['observed_mean_delta']:+.4f}** |"
        )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Execution Telemetry & Compute Overhead",
            "",
            "| Stream | Updates Applied | Updates Skipped | Updates Failed | Runtime (s) | Peak VRAM (MB) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for k, v in telemetry.items():
            lines.append(
                f"| `{k}` | {v['updates_applied']} | {v['updates_skipped']} | {v['updates_failed']} | {v['elapsed_seconds']:.1f}s | {v['peak_vram_mb']:.1f}MB |"
            )

        lines.append("")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✓ Markdown report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard R2 + TTA Analysis Engine.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=REPO_ROOT / "outputs/supplementary_r2_tta",
        help="Directory containing completed stream summaries.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs/supplementary_r2_tta/analysis",
        help="Directory to save analysis report and JSON.",
    )
    args = parser.parse_args()

    analyzer = R2TTAAnalyzer(
        supplementary_output_dir=args.results_dir,
        analysis_output_dir=args.output_dir,
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

