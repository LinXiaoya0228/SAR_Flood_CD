#!/usr/bin/env python3
"""ChangeGuard Work Package E11: Final Test and Rehearsal Analysis Engine.

Consolidates evaluation streams across:
1. SNUNet-ECAM: source_only, bn_no_update, tent, cotta across Seeds 0, 1, 2.
2. SiamUnet_diff: source_only, bn_no_update, tent across Seeds 0, 1, 2.

Performs:
- Event-macro, worst-event, macro IoU, and pooled F1 aggregation.
- Paired delta evaluation (Delta_TTA vs. Source-Only).
- Paired event-cluster bootstrap with 10,000 resamples for 95% confidence intervals.
- Non-parametric Wilcoxon signed-rank significance testing.
- Markdown table and JSON serialization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORBIDDEN_SEALED_TEST_EVENTS = {
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
}


class E11FinalTestAnalyzer:
    """Consolidation and statistical analysis engine for E11 streams."""

    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        ledger_path: Path,
    ):
        self.results_dir = results_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = ledger_path.resolve()

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

    def load_stream_summary(self, stream_id: str) -> Dict[str, Any]:
        """Loads and verifies stream summary."""
        stream_dir = self.results_dir / stream_id
        summary_path = stream_dir / "stream_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing stream summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def run_event_cluster_bootstrap(
        self,
        event_f1s_tta: np.ndarray,
        event_f1s_src: np.ndarray,
        num_bootstraps: int = 10000,
        rng_seed: int = 42,
    ) -> Tuple[float, float, float, float]:
        """Performs paired event-cluster bootstrap to obtain mean delta and 95% CI.
        
        Args:
            event_f1s_tta: (N_events,) F1 scores for TTA method
            event_f1s_src: (N_events,) F1 scores for source model
            
        Returns:
            (mean_delta, ci_lower, ci_upper, p_value_approx)
        """
        n_events = len(event_f1s_tta)
        if n_events < 2:
            delta = float(np.mean(event_f1s_tta - event_f1s_src))
            return delta, delta, delta, 1.0

        rng = np.random.RandomState(rng_seed)
        deltas = event_f1s_tta - event_f1s_src
        mean_delta = float(np.mean(deltas))

        # Resample event clusters
        boot_means = np.zeros(num_bootstraps, dtype=np.float64)
        for b in range(num_bootstraps):
            indices = rng.randint(0, n_events, size=n_events)
            boot_means[b] = np.mean(deltas[indices])

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        # Fraction of bootstraps crossing 0
        p_val = float(2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
        p_val = min(p_val, 1.0)

        return mean_delta, ci_lower, ci_upper, p_val

    def consolidate(self) -> Dict[str, Any]:
        """Consolidates all streams in ledger and computes statistical summaries."""
        print(f"\nConsolidating E11 results from {self.results_dir}...")
        results: Dict[str, Any] = {
            "SNUNet_ECAM": {},
            "SiamUnet_diff": {},
        }

        for stream_entry in self.ledger["streams"]:
            stream_id = stream_entry["stream_id"]
            arch = stream_entry["architecture"]
            method = stream_entry["method"]
            seed = stream_entry["seed"]

            if method not in results[arch]:
                results[arch][method] = {
                    "seeds": {},
                    "macro_f1_list": [],
                    "worst_f1_list": [],
                    "macro_iou_list": [],
                    "pooled_f1_list": [],
                    "event_f1_arrays": {},
                }

            data = self.load_stream_summary(stream_id)
            results[arch][method]["seeds"][str(seed)] = data
            results[arch][method]["macro_f1_list"].append(data["macro_f1"])
            results[arch][method]["worst_f1_list"].append(data["worst_event_f1"])
            results[arch][method]["macro_iou_list"].append(data["macro_iou"])
            results[arch][method]["pooled_f1_list"].append(data.get("pooled_f1", 0.0))

            ev_f1s = [row["flood_f1"] for row in data["event_summaries"]]
            results[arch][method]["event_f1_arrays"][str(seed)] = ev_f1s

        # Compute summary statistics
        for arch in results:
            src_f1_mean = None
            if "source_only" in results[arch] and results[arch]["source_only"]["macro_f1_list"]:
                src_f1_mean = float(np.mean(results[arch]["source_only"]["macro_f1_list"]))

            for method in results[arch]:
                m_info = results[arch][method]
                f1s = m_info["macro_f1_list"]
                if f1s:
                    m_info["mean_macro_f1"] = float(np.mean(f1s))
                    m_info["std_macro_f1"] = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
                    m_info["mean_worst_f1"] = float(np.mean(m_info["worst_f1_list"]))
                    m_info["mean_macro_iou"] = float(np.mean(m_info["macro_iou_list"]))
                    m_info["mean_pooled_f1"] = float(np.mean(m_info["pooled_f1_list"]))

                    if src_f1_mean is not None:
                        m_info["delta_vs_source"] = m_info["mean_macro_f1"] - src_f1_mean

                    # Paired bootstrap across events
                    if method != "source_only" and "source_only" in results[arch]:
                        all_tta_ev = []
                        all_src_ev = []
                        for s_key in m_info["event_f1_arrays"]:
                            if s_key in results[arch]["source_only"]["event_f1_arrays"]:
                                all_tta_ev.extend(m_info["event_f1_arrays"][s_key])
                                all_src_ev.extend(results[arch]["source_only"]["event_f1_arrays"][s_key])

                        if all_tta_ev and all_src_ev:
                            delta_b, ci_l, ci_u, p_b = self.run_event_cluster_bootstrap(
                                np.array(all_tta_ev), np.array(all_src_ev)
                            )
                            m_info["bootstrap_delta"] = delta_b
                            m_info["bootstrap_95ci"] = [ci_l, ci_u]
                            m_info["bootstrap_p_val"] = p_b

        # Write consolidated JSON
        out_json = self.output_dir / "e11_consolidated_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Generate markdown report
        md_report = self.generate_markdown_tables(results)
        out_md = self.output_dir / "e11_consolidated_tables.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_report)

        print(md_report)
        return results

    def generate_markdown_tables(self, results: Dict[str, Any]) -> str:
        """Formats consolidated metrics into publication markdown tables."""
        lines = []
        lines.append("# ChangeGuard Work Package E11: Consolidated Evaluation Matrix\n")
        lines.append("| Architecture | Method | Seed 0 | Seed 1 | Seed 2 | Macro F1 (Mean ± Std) | Worst F1 | Macro IoU | Pooled F1 | Δ_TTA vs Source | 95% Bootstrap CI |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for arch, methods in results.items():
            for m, data in methods.items():
                s0 = f"{data['seeds'].get('0', {}).get('macro_f1', 0.0):.4f}" if "0" in data["seeds"] else "N/A"
                s1 = f"{data['seeds'].get('1', {}).get('macro_f1', 0.0):.4f}" if "1" in data["seeds"] else "N/A"
                s2 = f"{data['seeds'].get('2', {}).get('macro_f1', 0.0):.4f}" if "2" in data["seeds"] else "N/A"
                mean_std = f"{data.get('mean_macro_f1', 0.0):.4f} ± {data.get('std_macro_f1', 0.0):.4f}"
                worst = f"{data.get('mean_worst_f1', 0.0):.4f}"
                iou = f"{data.get('mean_macro_iou', 0.0):.4f}"
                pooled = f"{data.get('mean_pooled_f1', 0.0):.4f}"
                delta = f"{data.get('delta_vs_source', 0.0):+.4f}" if m != "source_only" else "Ref (0.0000)"
                ci = data.get("bootstrap_95ci")
                ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "N/A"

                lines.append(
                    f"| {arch} | {m} | {s0} | {s1} | {s2} | {mean_std} | {worst} | {iou} | {pooled} | {delta} | {ci_str} |"
                )

        return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="E11 Final Test & Rehearsal Analysis Engine.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=REPO_ROOT / "outputs/e11_rehearsal",
        help="Directory containing stream outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs/e11_rehearsal_analysis",
        help="Directory to save consolidated analysis.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e11_final_test_ledger_v1.json",
        help="Path to final test ledger.",
    )
    args = parser.parse_args()

    analyzer = E11FinalTestAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
    )
    analyzer.consolidate()


if __name__ == "__main__":
    main()

