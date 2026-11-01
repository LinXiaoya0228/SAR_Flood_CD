#!/usr/bin/env python3
"""ChangeGuard Work Package E10: Robustness and Backbone Extension Analysis Engine.

Performs offline analysis of:
1. Second Lightweight Backbone (SiamUnet_diff):
   - Cross-seed (Seeds 0, 1, 2) evaluation of source_only, bn_no_update, and tent on 6 dev events.
   - Compares against canonical SNUNet results from E8 confirmation.
   - Evaluates whether architectural scale (1.35M vs 12M params) and Siamese difference affect TTA vulnerability.
2. Synthetic Domain-Shift Family:
   - Evaluates multiplicative backscatter shifts s in {0.0, 0.05, 0.10, 0.20} on SNUNet and SiamUnet_diff.
   - Quantifies degradation trajectories and whether TTA mitigates or exacerbates domain shift.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DECLARED_DEV_EVENTS: Tuple[int, ...] = (279, 437, 520, 559, 1111003, 1111008)
FORBIDDEN_SEALED_TEST_EVENTS: Tuple[int, ...] = (
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
)


class E10RobustnessAnalyzer:
    """Consolidates and analyzes E10 second backbone and synthetic shift experiments."""

    def __init__(
        self,
        output_dir: Path,
    ):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_stream_summary(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Verify no sealed test
        for row in data.get("event_table", []):
            eid = int(row["event_id"])
            if eid in FORBIDDEN_SEALED_TEST_EVENTS:
                raise RuntimeError(f"CRITICAL INTEGRITY VIOLATION: Sealed test event {eid} in stream!")
        return data

    def consolidate_second_backbone_results(
        self,
        dev_eval_dir: Path,
        snunet_e8_summary_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Consolidates second backbone streams across seeds and methods."""
        results: Dict[str, Dict[str, Any]] = {}
        methods = ["source_only", "bn_no_update", "tent"]
        seeds = [0, 1, 2]

        for method in methods:
            results[method] = {
                "seeds": {},
                "macro_f1_list": [],
                "worst_f1_list": [],
                "macro_iou_list": [],
                "macro_recall_list": [],
            }
            for seed in seeds:
                stream_id = f"siam_diff_{method}_seed{seed}"
                stream_path = dev_eval_dir / stream_id / "stream_summary.json"
                if stream_path.exists():
                    data = self.load_stream_summary(stream_path)
                    results[method]["seeds"][str(seed)] = data
                    results[method]["macro_f1_list"].append(data["macro_f1"])
                    results[method]["worst_f1_list"].append(data["worst_event_f1"])
                    results[method]["macro_iou_list"].append(data["macro_iou"])
                    results[method]["macro_recall_list"].append(data["macro_recall"])

            # Compute seed averages
            if results[method]["macro_f1_list"]:
                results[method]["mean_macro_f1"] = float(np.mean(results[method]["macro_f1_list"]))
                results[method]["std_macro_f1"] = float(np.std(results[method]["macro_f1_list"], ddof=1)) if len(results[method]["macro_f1_list"]) > 1 else 0.0
                results[method]["mean_worst_f1"] = float(np.mean(results[method]["worst_f1_list"]))
                results[method]["mean_macro_iou"] = float(np.mean(results[method]["macro_iou_list"]))
                results[method]["mean_macro_recall"] = float(np.mean(results[method]["macro_recall_list"]))

        # Compute delta TTA vs source_only
        if "source_only" in results and results["source_only"]["macro_f1_list"]:
            so_mean = results["source_only"]["mean_macro_f1"]
            for method in ["bn_no_update", "tent"]:
                if results[method]["macro_f1_list"]:
                    results[method]["delta_f1_vs_source"] = results[method]["mean_macro_f1"] - so_mean

        # Save consolidated JSON
        out_json = self.output_dir / "e10_second_backbone_consolidation.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def consolidate_synthetic_shift_results(
        self,
        shift_eval_dir: Path,
    ) -> Dict[str, Any]:
        """Consolidates synthetic shift results across architectures, methods, and severities."""
        severities = ["clean", "severity_1", "severity_2", "severity_3"]
        sev_factors = [0.0, 0.05, 0.10, 0.20]
        archs = ["SNUNet_ECAM", "SiamUnet_diff"]
        methods = ["source_only", "bn_no_update", "tent"]
        seeds = [0, 1, 2]

        consolidation: Dict[str, Any] = {}

        for arch in archs:
            consolidation[arch] = {}
            for method in methods:
                consolidation[arch][method] = {}
                for sev, factor in zip(severities, sev_factors):
                    f1_vals = []
                    worst_vals = []
                    iou_vals = []
                    for seed in seeds:
                        prefix = "snunet" if arch == "SNUNet_ECAM" else "siam_diff"
                        stream_id = f"shift_{prefix}_{method}_{sev}_seed{seed}"
                        summary_file = shift_eval_dir / stream_id / "stream_summary.json"
                        if summary_file.exists():
                            d = self.load_stream_summary(summary_file)
                            f1_vals.append(d["macro_f1"])
                            worst_vals.append(d["worst_event_f1"])
                            iou_vals.append(d["macro_iou"])

                    if f1_vals:
                        consolidation[arch][method][sev] = {
                            "severity_factor": factor,
                            "n_seeds": len(f1_vals),
                            "mean_macro_f1": float(np.mean(f1_vals)),
                            "std_macro_f1": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                            "mean_worst_f1": float(np.mean(worst_vals)),
                            "mean_macro_iou": float(np.mean(iou_vals)),
                        }

        # Save consolidated JSON
        out_json = self.output_dir / "e10_synthetic_shift_consolidation.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(consolidation, f, indent=2)

        return consolidation

    def generate_summary_tables(
        self,
        second_backbone_data: Dict[str, Any],
        synthetic_shift_data: Dict[str, Any],
    ) -> str:
        """Generates markdown summary tables for second backbone and synthetic shifts."""
        lines = []
        lines.append("# ChangeGuard Work Package E10: Consolidated Robustness Evaluation Tables\n")
        lines.append("## 1. Second Lightweight Backbone (SiamUnet_diff) vs Baselines\n")
        lines.append("| Architecture | Method | Seed 0 Macro F1 | Seed 1 Macro F1 | Seed 2 Macro F1 | Mean Macro F1 | Std F1 | Mean Worst F1 | Mean Macro IoU | Δ_TTA vs Source |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m in ["source_only", "bn_no_update", "tent"]:
            if m in second_backbone_data:
                d = second_backbone_data[m]
                s0 = f"{d['seeds'].get('0', {}).get('macro_f1', 0.0):.4f}" if '0' in d['seeds'] else "N/A"
                s1 = f"{d['seeds'].get('1', {}).get('macro_f1', 0.0):.4f}" if '1' in d['seeds'] else "N/A"
                s2 = f"{d['seeds'].get('2', {}).get('macro_f1', 0.0):.4f}" if '2' in d['seeds'] else "N/A"
                mean_f1 = f"{d.get('mean_macro_f1', 0.0):.4f}"
                std_f1 = f"{d.get('std_macro_f1', 0.0):.4f}"
                worst = f"{d.get('mean_worst_f1', 0.0):.4f}"
                iou = f"{d.get('mean_macro_iou', 0.0):.4f}"
                delta = f"{d.get('delta_f1_vs_source', 0.0):+.4f}" if m != "source_only" else "Ref (0.0000)"
                lines.append(f"| SiamUnet_diff | {m} | {s0} | {s1} | {s2} | {mean_f1} | {std_f1} | {worst} | {iou} | {delta} |")
        lines.append("\n---\n")
        lines.append("## 2. Synthetic Domain-Shift Stress Test Matrix\n")
        lines.append("| Architecture | Method | Clean (s=0.0) | Sev 1 (s=0.05) | Sev 2 (s=0.10) | Sev 3 (s=0.20) | Δ(s=0.20 - Clean) | Slope dF1/ds |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for arch in ["SNUNet_ECAM", "SiamUnet_diff"]:
            if arch in synthetic_shift_data:
                for m in ["source_only", "bn_no_update", "tent"]:
                    if m in synthetic_shift_data[arch]:
                        m_data = synthetic_shift_data[arch][m]
                        c_f1 = m_data.get("clean", {}).get("mean_macro_f1", 0.0)
                        s1_f1 = m_data.get("severity_1", {}).get("mean_macro_f1", 0.0)
                        s2_f1 = m_data.get("severity_2", {}).get("mean_macro_f1", 0.0)
                        s3_f1 = m_data.get("severity_3", {}).get("mean_macro_f1", 0.0)
                        delta_shift = s3_f1 - c_f1
                        slope = (s3_f1 - c_f1) / 0.20 if c_f1 > 0 else 0.0
                        lines.append(f"| {arch} | {m} | {c_f1:.4f} | {s1_f1:.4f} | {s2_f1:.4f} | {s3_f1:.4f} | {delta_shift:+.4f} | {slope:+.4f} |")
        report = "\n".join(lines) + "\n"
        out_md = self.output_dir / "e10_summary_tables.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(report)
        return report


def main():
    parser = argparse.ArgumentParser(description="E10 robustness analysis engine.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs/e10_consolidated",
        help="Directory to save consolidated outputs.",
    )
    args = parser.parse_args()

    analyzer = E10RobustnessAnalyzer(output_dir=args.output_dir)

    bb_data = {}
    dev_eval_dir = REPO_ROOT / "outputs/e10_second_backbone/dev_eval"
    if dev_eval_dir.exists():
        print(f"Consolidating second backbone results from {dev_eval_dir}...")
        bb_data = analyzer.consolidate_second_backbone_results(dev_eval_dir)

    shift_data = {}
    shift_eval_dir = REPO_ROOT / "outputs/e10_synthetic_shift"
    if shift_eval_dir.exists():
        print(f"Consolidating synthetic shift results from {shift_eval_dir}...")
        shift_data = analyzer.consolidate_synthetic_shift_results(shift_eval_dir)

    if bb_data and shift_data:
        md_tables = analyzer.generate_summary_tables(bb_data, shift_data)
        print("\n" + md_tables)


if __name__ == "__main__":
    main()



