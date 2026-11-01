#!/usr/bin/env python3
"""ChangeGuard Work Package E7: Branch C Empirical Comparison and LOEO Analysis Engine.

Performs offline statistical comparison, cross-event generalization (LOEO),
and sensitivity diagnostics across all evaluated adaptation methods:
- Source-Only, AdaBN, Tent, EATA, SAR, Candidate Consistency, CoTTA Continual, CoTTA Reset.
- Multi-seed (Seeds 0, 1, 2) and multi-order (Order A, Order B) coverage.
- Strictly adheres to Branch C semantics: zero policy calibration, no pseudo-replication.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DECLARED_DEVELOPMENT_EVENTS: Tuple[int, ...] = (279, 437, 520, 559, 1111003, 1111008)
FORBIDDEN_SEALED_TEST_EVENTS: Tuple[int, ...] = (
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
)


def compute_f1_from_cells(tp: int, fp: int, fn: int) -> float:
    """Computes F1 score from confusion matrix cells."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def compute_iou_from_cells(tp: int, fp: int, fn: int) -> float:
    """Computes IoU from confusion matrix cells."""
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


class E7EmpiricalAnalyzer:
    """Offline comparative analysis and LOEO evaluation engine for Work Package E7."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        allow_overwrite: bool = False,
    ):
        self.source_dir = source_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.allow_overwrite = allow_overwrite

    def validate_inputs(self) -> Dict[str, Any]:
        """Validates source input directory and preflight requirements."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source results directory not found: {self.source_dir}")

        if not self.allow_overwrite and self.output_dir.exists():
            # Check if output directory already contains completed report
            summary_p = self.output_dir / "e7_empirical_synthesis_summary.json"
            if summary_p.is_file():
                raise FileExistsError(
                    f"Destination output directory already exists and contains completed analysis: {self.output_dir}. "
                    "Overwriting is prohibited under Plan V3 rules."
                )

        return {
            "status": "VALIDATION_PASSED",
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "development_events": list(DECLARED_DEVELOPMENT_EVENTS),
            "quarantined_events": list(FORBIDDEN_SEALED_TEST_EVENTS),
        }

    def load_stream_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Loads and indexes all completed stream summary files from source directory."""
        stream_data = {}
        for stream_path in self.source_dir.glob("**/stream_summary.json"):
            try:
                with open(stream_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                s_id = summary.get("stream_id", stream_path.parent.name)
                stream_data[s_id] = summary
            except Exception as e:
                print(f"Warning: Failed to load stream summary from {stream_path}: {e}")
        return stream_data

    def run_loeo_generalization_analysis(
        self,
        stream_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Performs Leave-One-Event-Out (LOEO) evaluation under Branch C.

        Under Branch C:
        - There is NO policy threshold calibration.
        - LOEO assesses cross-event generalization of baseline performance by holding out
          each event in turn, reporting held-out metrics without pseudo-replication.
        """
        loeo_folds = {}
        for held_out_event in DECLARED_DEVELOPMENT_EVENTS:
            fold_key = f"held_out_{held_out_event}"
            fold_results = {}

            for s_id, s_summary in stream_data.items():
                ev_metrics = {m["event_id"]: m for m in s_summary.get("event_metrics", [])}
                if held_out_event in ev_metrics:
                    m = ev_metrics[held_out_event]
                    fold_results[s_id] = {
                        "event_id": held_out_event,
                        "f1": m.get("f1", 0.0),
                        "precision": m.get("precision", 0.0),
                        "recall": m.get("recall", 0.0),
                        "iou": m.get("iou", 0.0),
                        "samples": m.get("samples", 0),
                    }

            loeo_folds[fold_key] = {
                "held_out_event": held_out_event,
                "stream_metrics": fold_results,
            }

        return {
            "study_branch": "Branch C — Empirical Study",
            "policy_calibration": None,
            "calibration_comment": "Branch C rejects policy threshold calibration; LOEO evaluates cross-event baseline generalization directly.",
            "total_folds": len(loeo_folds),
            "folds": loeo_folds,
        }

    def compute_synthesis_summary(
        self,
        stream_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Computes multi-seed, multi-order, and cross-method synthesis metrics."""
        master_rows = []
        source_baselines: Dict[int, Dict[int, float]] = {}

        # First pass: extract source-only baselines per seed and event
        for s_id, s_summary in stream_data.items():
            method = s_summary.get("method")
            seed = s_summary.get("seed")
            if method == "source_only" and seed is not None:
                source_baselines[seed] = {
                    m["event_id"]: m.get("f1", 0.0)
                    for m in s_summary.get("event_metrics", [])
                }

        # Second pass: compute deltas and classifications
        for s_id, s_summary in sorted(stream_data.items()):
            method = s_summary.get("method", "unknown")
            seed = s_summary.get("seed", 0)
            order_id = s_summary.get("order_id", "order_a")
            ev_metrics = {m["event_id"]: m for m in s_summary.get("event_metrics", [])}

            ev_f1s = {}
            ev_deltas = {}
            harm_count = 0
            benefit_count = 0
            neutral_count = 0

            for eid in DECLARED_DEVELOPMENT_EVENTS:
                f1_val = ev_metrics.get(eid, {}).get("f1", 0.0)
                ev_f1s[eid] = f1_val

                src_f1 = source_baselines.get(seed, {}).get(eid, f1_val)
                delta = f1_val - src_f1
                ev_deltas[eid] = delta

                if delta <= -0.0100:
                    harm_count += 1
                elif delta >= 0.0100:
                    benefit_count += 1
                else:
                    neutral_count += 1

            macro_f1 = s_summary.get("macro_f1", float(np.mean(list(ev_f1s.values()))))
            pooled_f1 = s_summary.get("pooled_f1", 0.0)
            worst_f1 = min(ev_f1s.values()) if ev_f1s else 0.0

            master_rows.append({
                "stream_id": s_id,
                "method": method,
                "seed": seed,
                "order_id": order_id,
                "macro_f1": macro_f1,
                "pooled_f1": pooled_f1,
                "worst_event_f1": worst_f1,
                "f1_ev279": ev_f1s.get(279, 0.0),
                "f1_ev437": ev_f1s.get(437, 0.0),
                "f1_ev520": ev_f1s.get(520, 0.0),
                "f1_ev559": ev_f1s.get(559, 0.0),
                "f1_ev1111003": ev_f1s.get(1111003, 0.0),
                "f1_ev1111008": ev_f1s.get(1111008, 0.0),
                "delta_ev279": ev_deltas.get(279, 0.0),
                "delta_ev437": ev_deltas.get(437, 0.0),
                "delta_ev520": ev_deltas.get(520, 0.0),
                "delta_ev559": ev_deltas.get(559, 0.0),
                "delta_ev1111003": ev_deltas.get(1111003, 0.0),
                "delta_ev1111008": ev_deltas.get(1111008, 0.0),
                "harm_count": harm_count,
                "benefit_count": benefit_count,
                "neutral_count": neutral_count,
            })

        return {
            "status": "ANALYSIS_COMPLETE",
            "total_streams_evaluated": len(master_rows),
            "master_comparison_rows": master_rows,
        }

    def execute_and_export(self) -> Dict[str, Any]:
        """Runs full offline comparison and writes master artifacts."""
        self.validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stream_data = self.load_stream_summaries()
        print(f"Loaded {len(stream_data)} stream summary records from {self.source_dir}")

        loeo_report = self.run_loeo_generalization_analysis(stream_data)
        synthesis_summary = self.compute_synthesis_summary(stream_data)

        # 1. Export master CSV
        csv_path = self.output_dir / "e7_master_comparison_table.csv"
        if synthesis_summary["master_comparison_rows"]:
            fieldnames = list(synthesis_summary["master_comparison_rows"][0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in synthesis_summary["master_comparison_rows"]:
                    writer.writerow(r)
            print(f"Exported master comparison table to {csv_path}")

        # 2. Export LOEO JSON
        loeo_path = self.output_dir / "e7_loeo_generalization_report.json"
        with open(loeo_path, "w", encoding="utf-8") as f:
            json.dump(loeo_report, f, indent=2)
        print(f"Exported LOEO report to {loeo_path}")

        # 3. Export synthesis summary JSON
        summary_path = self.output_dir / "e7_empirical_synthesis_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(synthesis_summary, f, indent=2)
        print(f"Exported synthesis summary to {summary_path}")

        return synthesis_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard E7 Empirical Comparison Analyzer")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("outputs/e7_empirical_evidence"),
        help="Directory containing completed stream evaluation outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/e7_empirical_evidence/analysis_v1"),
        help="Directory to write comparative tables and reports",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate source inputs and preflights without exporting artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permit overwrite if output directory already exists",
    )
    args = parser.parse_args()

    analyzer = E7EmpiricalAnalyzer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        allow_overwrite=args.force,
    )

    if args.validate_only:
        report = analyzer.validate_inputs()
        print(f"Input validation passed: {report}")
        sys.exit(0)

    analyzer.execute_and_export()


if __name__ == "__main__":
    main()

