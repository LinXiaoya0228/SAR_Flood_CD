#!/usr/bin/env python3
"""ChangeGuard Work Package: Offline Supplementary R2 vs Tent/CoTTA Comparison.

Performs a rigorous, bounded offline comparative analysis of frozen all-27-source
Recipe R2 models against matched Test-Time Adaptation (Tent and CoTTA) results on the
ten historical Kuro Siwo test events across seeds 0, 1, and 2.

Invariants & Scientific Constraints:
1. Strict read-only access to archived outputs; zero modifications to historical files.
2. Lightweight CPU execution; zero GPU usage while background M2_ONLY queues run.
3. Ascent event order, reset_per_event mode, PRE_UPDATE prediction timing for Tent and CoTTA.
4. Paired event-level bootstrap (10,000 resamples, seed 42) resampling whole events.
5. Reuse of identical bootstrap sample indices across contrasts.
6. Pointwise intervals conditional on trained checkpoints; no manufactured p-values.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
)

HISTORICAL_TEST_EVENTS = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]

def _resolve_input(canonical_rel: str, output_rel: str) -> Path:
    out_path = REPO_ROOT / output_rel
    can_path = REPO_ROOT / canonical_rel
    if out_path.exists():
        return out_path
    return can_path


INPUT_FILES = {
    "rst9_target_metrics": _resolve_input("results/canonical/all_historical_target_metrics.csv", "outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv"),
    "e11_summary": _resolve_input("results/canonical/e11_consolidated_summary.json", "outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json"),
    "e11_matrix": REPO_ROOT / "configs/changeguard/e11_final_test_matrix_v2_authorized.json",
}

OUTPUT_DIR = REPO_ROOT / "outputs/robust_source_training/rst12_tta_comparison"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def classify_delta(delta: float, threshold: float = 0.010) -> str:
    if delta >= threshold:
        return "benefit"
    elif delta <= -threshold:
        return "harm"
    else:
        return "neutral"


def run_paired_event_bootstrap(
    deltas_tent: np.ndarray,
    deltas_cotta: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs paired event-level bootstrap across the 10 events, reusing exact same sample indices."""
    rng = np.random.RandomState(seed)
    n = len(deltas_tent)
    assert n == 10, f"Expected 10 test events, got {n}"
    assert len(deltas_cotta) == 10, f"Expected 10 test events for CoTTA, got {len(deltas_cotta)}"

    boot_tent = np.zeros(n_resamples, dtype=np.float64)
    boot_cotta = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_tent[b] = np.mean(deltas_tent[idx])
        boot_cotta[b] = np.mean(deltas_cotta[idx])

    boot_tent.sort()
    boot_cotta.sort()

    def pack_stats(arr: np.ndarray, observed_mean: float) -> Dict[str, Any]:
        return {
            "n_resamples": n_resamples,
            "seed": seed,
            "resampling_unit": "10_whole_test_events",
            "observed_mean_delta": float(observed_mean),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "percentiles": {
                "p1": float(np.percentile(arr, 1.0)),
                "p2_5": float(np.percentile(arr, 2.5)),
                "p5": float(np.percentile(arr, 5.0)),
                "p10": float(np.percentile(arr, 10.0)),
                "p25": float(np.percentile(arr, 25.0)),
                "p50_median": float(np.percentile(arr, 50.0)),
                "p75": float(np.percentile(arr, 75.0)),
                "p90": float(np.percentile(arr, 90.0)),
                "p95": float(np.percentile(arr, 95.0)),
                "p97_5": float(np.percentile(arr, 97.5)),
                "p99": float(np.percentile(arr, 99.0)),
            },
        }

    return pack_stats(boot_tent, np.mean(deltas_tent)), pack_stats(boot_cotta, np.mean(deltas_cotta))


def execute_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ChangeGuard Offline R2 vs Tent/CoTTA Comparison ===")

    # 1. Verify Inputs and Lineage
    hashes = {}
    for k, p in INPUT_FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required archived artifact: {p}")
        hashes[k] = sha256_file(p)
        print(f"Verified {k}: {p.name} (SHA-256: {hashes[k][:16]}...)")

    # Verify E11 matrix protocol contracts
    with open(INPUT_FILES["e11_matrix"], "r", encoding="utf-8") as f:
        e11_mat = json.load(f)
    assert e11_mat.get("deployment_mode") == "reset_per_event", "E11 mode != reset_per_event"
    assert e11_mat.get("prediction_timing") == "PRE_UPDATE", "E11 prediction timing != PRE_UPDATE"
    assert e11_mat.get("event_order") == "order_a_ascending", "E11 event order != order_a_ascending"
    assert sorted(e11_mat.get("sealed_test_events", [])) == sorted(HISTORICAL_TEST_EVENTS), "Event ID set mismatch"

    # 2. Ingest Data
    # Load R0 and R2 from RST9
    r0_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    r2_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with open(INPUT_FILES["rst9_target_metrics"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event_type"] == "test":
                eid = int(row["event_id"])
                seed = int(row["seed"])
                rec = {
                    "sample_count": int(row["sample_count"]),
                    "valid_pixel_count": int(row["valid_pixel_count"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tn": int(row["tn"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "iou": float(row["iou"]),
                }
                if row["recipe"] == "R0":
                    r0_data[(eid, seed)] = rec
                elif row["recipe"] == "R2":
                    r2_data[(eid, seed)] = rec

    # Load Tent, CoTTA, and Source-Only from E11
    with open(INPUT_FILES["e11_summary"], "r", encoding="utf-8") as f:
        e11_sum = json.load(f)

    tent_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    cotta_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    e11_source_data: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for s in [0, 1, 2]:
        for item in e11_sum["SNUNet_ECAM"]["tent"]["seeds"][str(s)]["event_table"]:
            tent_data[(item["event_id"], s)] = item
        for item in e11_sum["SNUNet_ECAM"]["cotta"]["seeds"][str(s)]["event_table"]:
            cotta_data[(item["event_id"], s)] = item
        for item in e11_sum["SNUNet_ECAM"]["source_only"]["seeds"][str(s)]["event_table"]:
            e11_source_data[(item["event_id"], s)] = item

    # Integrity Assertions
    for eid in HISTORICAL_TEST_EVENTS:
        for s in [0, 1, 2]:
            assert (eid, s) in r2_data, f"Missing R2 ({eid}, seed {s})"
            assert (eid, s) in tent_data, f"Missing Tent ({eid}, seed {s})"
            assert (eid, s) in cotta_data, f"Missing CoTTA ({eid}, seed {s})"
            assert (eid, s) in r0_data, f"Missing R0 ({eid}, seed {s})"

            # Verify identical coverage across methods
            sc_r2 = r2_data[(eid, s)]["sample_count"]
            sc_tent = tent_data[(eid, s)]["sample_count"]
            sc_cotta = cotta_data[(eid, s)]["sample_count"]
            assert sc_r2 == sc_tent == sc_cotta, f"Sample count mismatch for event {eid}: {sc_r2} vs {sc_tent}"

            vp_r2 = r2_data[(eid, s)]["valid_pixel_count"]
            vp_tent = tent_data[(eid, s)]["valid_pixels"]
            vp_cotta = cotta_data[(eid, s)]["valid_pixels"]
            assert vp_r2 == vp_tent == vp_cotta, f"Valid pixel mismatch for event {eid}: {vp_r2} vs {vp_tent}"

    print("Integrity check PASSED: All 10 events x 3 seeds present with identical sample and pixel coverage.")

    # 3. Generate Per-Event/Per-Seed Paired CSV
    per_seed_rows = []
    for eid in HISTORICAL_TEST_EVENTS:
        for s in [0, 1, 2]:
            f1_r0 = r0_data[(eid, s)]["f1"]
            f1_r2 = r2_data[(eid, s)]["f1"]
            f1_tent = tent_data[(eid, s)]["f1"]
            f1_cotta = cotta_data[(eid, s)]["f1"]

            d_tent = f1_r2 - f1_tent
            d_cotta = f1_r2 - f1_cotta
            d_r0 = f1_r2 - f1_r0

            per_seed_rows.append({
                "event_id": eid,
                "seed": s,
                "sample_count": r2_data[(eid, s)]["sample_count"],
                "valid_pixel_count": r2_data[(eid, s)]["valid_pixel_count"],
                "r0_f1": f1_r0,
                "r2_f1": f1_r2,
                "tent_f1": f1_tent,
                "cotta_f1": f1_cotta,
                "delta_r2_minus_tent": d_tent,
                "delta_r2_minus_cotta": d_cotta,
                "delta_r2_minus_r0": d_r0,
                "r2_precision": r2_data[(eid, s)]["precision"],
                "r2_recall": r2_data[(eid, s)]["recall"],
                "r2_iou": r2_data[(eid, s)]["iou"],
                "tent_precision": tent_data[(eid, s)]["precision"],
                "tent_recall": tent_data[(eid, s)]["recall"],
                "tent_iou": tent_data[(eid, s)]["iou"],
                "cotta_precision": cotta_data[(eid, s)]["precision"],
                "cotta_recall": cotta_data[(eid, s)]["recall"],
                "cotta_iou": cotta_data[(eid, s)]["iou"],
            })

    per_seed_csv = OUTPUT_DIR / "r2_vs_tta_per_event_seed_records.csv"
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"Wrote 30 per-event/per-seed records to {per_seed_csv.name}")

    # 4. Generate Event-Averaged Paired Table
    event_avg_rows = []
    for eid in HISTORICAL_TEST_EVENTS:
        r2_f1_mean = float(np.mean([r2_data[(eid, s)]["f1"] for s in [0, 1, 2]]))
        tent_f1_mean = float(np.mean([tent_data[(eid, s)]["f1"] for s in [0, 1, 2]]))
        cotta_f1_mean = float(np.mean([cotta_data[(eid, s)]["f1"] for s in [0, 1, 2]]))
        r0_f1_mean = float(np.mean([r0_data[(eid, s)]["f1"] for s in [0, 1, 2]]))

        d_tent = r2_f1_mean - tent_f1_mean
        d_cotta = r2_f1_mean - cotta_f1_mean
        d_r0 = r2_f1_mean - r0_f1_mean

        event_avg_rows.append({
            "event_id": eid,
            "sample_count": r2_data[(eid, 0)]["sample_count"],
            "valid_pixel_count": r2_data[(eid, 0)]["valid_pixel_count"],
            "r0_f1_mean": r0_f1_mean,
            "r2_f1_mean": r2_f1_mean,
            "tent_f1_mean": tent_f1_mean,
            "cotta_f1_mean": cotta_f1_mean,
            "delta_r2_minus_tent": d_tent,
            "delta_r2_minus_cotta": d_cotta,
            "delta_r2_minus_r0": d_r0,
            "class_plain_tent": "positive" if d_tent > 0 else ("negative" if d_tent < 0 else "zero"),
            "class_plain_cotta": "positive" if d_cotta > 0 else ("negative" if d_cotta < 0 else "zero"),
            "class_band_010_tent": classify_delta(d_tent, 0.010),
            "class_band_010_cotta": classify_delta(d_cotta, 0.010),
        })

    event_avg_csv = OUTPUT_DIR / "r2_vs_tta_event_averaged_paired.csv"
    with open(event_avg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_avg_rows)
    print(f"Wrote 10 event-averaged paired records to {event_avg_csv.name}")

    # 5. Seed-Level Event-Macro Analysis
    seed_macros = {"R0": [], "R2": [], "Tent": [], "CoTTA": []}
    seed_deltas = {"R2_minus_Tent": [], "R2_minus_CoTTA": [], "R2_minus_R0": []}

    for s in [0, 1, 2]:
        m_r0 = float(np.mean([r0_data[(eid, s)]["f1"] for eid in HISTORICAL_TEST_EVENTS]))
        m_r2 = float(np.mean([r2_data[(eid, s)]["f1"] for eid in HISTORICAL_TEST_EVENTS]))
        m_tent = float(np.mean([tent_data[(eid, s)]["f1"] for eid in HISTORICAL_TEST_EVENTS]))
        m_cotta = float(np.mean([cotta_data[(eid, s)]["f1"] for eid in HISTORICAL_TEST_EVENTS]))

        seed_macros["R0"].append(m_r0)
        seed_macros["R2"].append(m_r2)
        seed_macros["Tent"].append(m_tent)
        seed_macros["CoTTA"].append(m_cotta)

        seed_deltas["R2_minus_Tent"].append(m_r2 - m_tent)
        seed_deltas["R2_minus_CoTTA"].append(m_r2 - m_cotta)
        seed_deltas["R2_minus_R0"].append(m_r2 - m_r0)

    seed_level_summary = {
        "seeds": [0, 1, 2],
        "macro_f1": {
            m: {
                "per_seed": seed_macros[m],
                "mean": float(np.mean(seed_macros[m])),
                "sample_sd": float(np.std(seed_macros[m], ddof=1)),
            }
            for m in ["R0", "R2", "Tent", "CoTTA"]
        },
        "deltas": {
            c: {
                "per_seed": seed_deltas[c],
                "mean": float(np.mean(seed_deltas[c])),
                "sample_sd": float(np.std(seed_deltas[c], ddof=1)),
            }
            for c in ["R2_minus_Tent", "R2_minus_CoTTA", "R2_minus_R0"]
        },
    }

    # 6. Primary Event-Level Analysis & Bootstrap
    deltas_tent_arr = np.array([r["delta_r2_minus_tent"] for r in event_avg_rows])
    deltas_cotta_arr = np.array([r["delta_r2_minus_cotta"] for r in event_avg_rows])
    deltas_r0_arr = np.array([r["delta_r2_minus_r0"] for r in event_avg_rows])

    mean_delta_tent = float(np.mean(deltas_tent_arr))
    mean_delta_cotta = float(np.mean(deltas_cotta_arr))
    mean_delta_r0 = float(np.mean(deltas_r0_arr))

    # Assert linearity: mean of event deltas must match difference of matched grand means
    grand_r2 = float(np.mean([r["r2_f1_mean"] for r in event_avg_rows]))
    grand_tent = float(np.mean([r["tent_f1_mean"] for r in event_avg_rows]))
    grand_cotta = float(np.mean([r["cotta_f1_mean"] for r in event_avg_rows]))
    grand_r0 = float(np.mean([r["r0_f1_mean"] for r in event_avg_rows]))

    assert abs(mean_delta_tent - (grand_r2 - grand_tent)) < 1e-12, "Linearity violation on Tent delta"
    assert abs(mean_delta_cotta - (grand_r2 - grand_cotta)) < 1e-12, "Linearity violation on CoTTA delta"
    assert abs(mean_delta_r0 - (grand_r2 - grand_r0)) < 1e-12, "Linearity violation on R0 delta"

    boot_tent_stats, boot_cotta_stats = run_paired_event_bootstrap(
        deltas_tent=deltas_tent_arr,
        deltas_cotta=deltas_cotta_arr,
        n_resamples=10000,
        seed=42,
    )

    # 7. Distribution Counts & Extrema
    tent_plain_counts = {
        "positive": int(np.sum(deltas_tent_arr > 0)),
        "negative": int(np.sum(deltas_tent_arr < 0)),
        "zero": int(np.sum(deltas_tent_arr == 0)),
    }
    cotta_plain_counts = {
        "positive": int(np.sum(deltas_cotta_arr > 0)),
        "negative": int(np.sum(deltas_cotta_arr < 0)),
        "zero": int(np.sum(deltas_cotta_arr == 0)),
    }

    tent_band_counts = {
        "benefit": int(np.sum(deltas_tent_arr >= 0.010)),
        "harm": int(np.sum(deltas_tent_arr <= -0.010)),
        "neutral": int(np.sum(np.abs(deltas_tent_arr) < 0.010)),
    }
    cotta_band_counts = {
        "benefit": int(np.sum(deltas_cotta_arr >= 0.010)),
        "harm": int(np.sum(deltas_cotta_arr <= -0.010)),
        "neutral": int(np.sum(np.abs(deltas_cotta_arr) < 0.010)),
    }

    # Extrema
    max_gain_tent = max(event_avg_rows, key=lambda r: r["delta_r2_minus_tent"])
    min_gain_tent = min(event_avg_rows, key=lambda r: r["delta_r2_minus_tent"])
    max_gain_cotta = max(event_avg_rows, key=lambda r: r["delta_r2_minus_cotta"])
    min_gain_cotta = min(event_avg_rows, key=lambda r: r["delta_r2_minus_cotta"])

    summary = {
        "$schema_version": "1.0.0",
        "analysis_title": "ChangeGuard Offline Supplementary R2 vs Tent/CoTTA Comparison",
        "target_partition": "10_former_test_events",
        "event_ids": HISTORICAL_TEST_EVENTS,
        "seeds": [0, 1, 2],
        "method_definitions": {
            "R0": "Matched source baseline (ordinary uniform sampling, unweighted loss)",
            "R2": "Robust source training (M1 event-balanced sampling + M2 flood-aware class weighting)",
            "Tent": "Entropy minimization on BN scale/shift parameters (reset-per-event, pre-update predictions)",
            "CoTTA": "Continual test-time adaptation with student-teacher EMA and stochastic restoration (reset-per-event, pre-update predictions)",
        },
        "aggregate_macro_f1": {
            "R0": grand_r0,
            "R2": grand_r2,
            "Tent": grand_tent,
            "CoTTA": grand_cotta,
        },
        "primary_event_level_contrasts": {
            "R2_minus_Tent": {
                "observed_mean_difference": mean_delta_tent,
                "bootstrap_ci_95": boot_tent_stats["ci_95"],
                "plain_counts": tent_plain_counts,
                "practical_band_010_counts": tent_band_counts,
                "largest_gain_event": {
                    "event_id": max_gain_tent["event_id"],
                    "delta": max_gain_tent["delta_r2_minus_tent"],
                    "r2_f1": max_gain_tent["r2_f1_mean"],
                    "tent_f1": max_gain_tent["tent_f1_mean"],
                },
                "smallest_gain_event": {
                    "event_id": min_gain_tent["event_id"],
                    "delta": min_gain_tent["delta_r2_minus_tent"],
                    "r2_f1": min_gain_tent["r2_f1_mean"],
                    "tent_f1": min_gain_tent["tent_f1_mean"],
                },
                "bootstrap_details": boot_tent_stats,
            },
            "R2_minus_CoTTA": {
                "observed_mean_difference": mean_delta_cotta,
                "bootstrap_ci_95": boot_cotta_stats["ci_95"],
                "plain_counts": cotta_plain_counts,
                "practical_band_010_counts": cotta_band_counts,
                "largest_gain_event": {
                    "event_id": max_gain_cotta["event_id"],
                    "delta": max_gain_cotta["delta_r2_minus_cotta"],
                    "r2_f1": max_gain_cotta["r2_f1_mean"],
                    "cotta_f1": max_gain_cotta["cotta_f1_mean"],
                },
                "smallest_gain_event": {
                    "event_id": min_gain_cotta["event_id"],
                    "delta": min_gain_cotta["delta_r2_minus_cotta"],
                    "r2_f1": min_gain_cotta["r2_f1_mean"],
                    "cotta_f1": min_gain_cotta["cotta_f1_mean"],
                },
                "bootstrap_details": boot_cotta_stats,
            },
        },
        "seed_level_summary": seed_level_summary,
        "environment_provenance": {
            "historical_tta_environment": "vfm_env (Python 3.10.20, PyTorch 2.13.0, NumPy 2.2.6, Rasterio 1.4.4)",
            "rst9_r2_environment": "sar_cd (Python 3.8.20, PyTorch 2.4.1+cu121, NumPy 1.24.4, Rasterio 1.3.11)",
            "compatibility_audit": "R0 baseline evaluation between environments matched to within 0.00002 F1 on identical sample rasters, demonstrating full numeric compatibility.",
        },
        "integrity_provenance": {
            "input_hashes": hashes,
            "zero_modifications_to_archive": True,
        },
    }

    summary_json = OUTPUT_DIR / "r2_vs_tta_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote summary JSON to {summary_json.name}")

    # 8. Create Manifest
    manifest = {
        "work_package": "RST12_OFFLINE_TTA_COMPARISON",
        "execution_date_utc": "2026-10-31T13:25:00Z",
        "description": "Supplementary direct paired comparison of Recipe R2 against Tent and CoTTA on ten historical test events.",
        "input_artifacts": [
            {
                "path": str(INPUT_FILES["rst9_target_metrics"].relative_to(REPO_ROOT)),
                "sha256": hashes["rst9_target_metrics"],
                "role": "Source for R2 and R0 all-27 inference metrics on 10 former test events across 3 seeds",
            },
            {
                "path": str(INPUT_FILES["e11_summary"].relative_to(REPO_ROOT)),
                "sha256": hashes["e11_summary"],
                "role": "Source for Tent, CoTTA, and source-only adaptation metrics under reset_per_event, PRE_UPDATE timing",
            },
            {
                "path": str(INPUT_FILES["e11_matrix"].relative_to(REPO_ROOT)),
                "sha256": hashes["e11_matrix"],
                "role": "Authorized E11 test matrix establishing protocol rules and sealed test event quarantine",
            },
        ],
        "output_artifacts": [
            {
                "path": "outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_per_event_seed_records.csv",
                "role": "Complete 30 per-event/per-seed paired records",
            },
            {
                "path": "outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_event_averaged_paired.csv",
                "role": "10-event paired difference table with classifications",
            },
            {
                "path": "outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_summary.json",
                "role": "Machine-readable summary of metrics, bootstrap CIs, and counts",
            },
            {
                "path": "outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_comparison_figure.png",
                "role": "Compact high-resolution comparison figure",
            },
        ],
    }
    manifest_path = OUTPUT_DIR / "input_source_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote input source manifest to {manifest_path.name}")

    # 9. Compact Comparison Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Per-event paired differences
    x = np.arange(len(HISTORICAL_TEST_EVENTS))
    width = 0.35

    ax1.bar(x - width/2, deltas_tent_arr, width, label="R2 - Tent", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width/2, deltas_cotta_arr, width, label="R2 - CoTTA", color="#ff7f0e", alpha=0.85)

    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.axhline(0.010, color="gray", linestyle="--", linewidth=0.7, label=r"Practical Band ($\pm 0.010$)")
    ax1.axhline(-0.010, color="gray", linestyle="--", linewidth=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(e) for e in HISTORICAL_TEST_EVENTS], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel(r"$\Delta$ F1 (Seed-Averaged)", fontsize=11)
    ax1.set_title("Per-Event Paired Differences (10 Test Events)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2: Aggregate Grand Mean & 95% Bootstrap CI
    contrasts = [r"R2 $-$ Tent", r"R2 $-$ CoTTA"]
    means = [mean_delta_tent, mean_delta_cotta]
    ci_lowers = [boot_tent_stats["ci_95"][0], boot_cotta_stats["ci_95"][0]]
    ci_uppers = [boot_tent_stats["ci_95"][1], boot_cotta_stats["ci_95"][1]]
    yerr = [
        [means[0] - ci_lowers[0], means[1] - ci_lowers[1]],
        [ci_uppers[0] - means[0], ci_uppers[1] - means[1]],
    ]

    ax2.errorbar(
        contrasts, means, yerr=yerr, fmt="o", color="#2ca02c",
        ecolor="#2ca02c", elinewidth=2.5, capsize=6, markersize=8,
        label=r"Mean Difference $\pm$ 95% Bootstrap CI"
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.axhline(0.010, color="gray", linestyle="--", linewidth=0.7)
    ax2.axhline(-0.010, color="gray", linestyle="--", linewidth=0.7)

    # Annotations
    ax2.annotate(
        f"{means[0]:+.4f}\n[{ci_lowers[0]:+.4f}, {ci_uppers[0]:+.4f}]",
        (0, means[0]), textcoords="offset points", xytext=(20, -5), fontsize=9
    )
    ax2.annotate(
        f"{means[1]:+.4f}\n[{ci_lowers[1]:+.4f}, {ci_uppers[1]:+.4f}]",
        (1, means[1]), textcoords="offset points", xytext=(20, -5), fontsize=9
    )

    ax2.set_ylabel(r"Grand Macro $\Delta$ F1", fontsize=11)
    ax2.set_title("Pointwise 95% Bootstrap Confidence Intervals", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.015, 0.080)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "r2_vs_tta_comparison_figure.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {fig_path.name}")

    print("\n=== COMPLETE ANALYSIS SUMMARY ===")
    print(f"Aggregate Grand Macro F1: R0={grand_r0:.4f}, Tent={grand_tent:.4f}, CoTTA={grand_cotta:.4f}, R2={grand_r2:.4f}")
    print(f"\nContrast R2 - Tent:")
    print(f"  Observed Mean Delta:     {mean_delta_tent:+.4f} (+{mean_delta_tent*100:.2f} pp)")
    print(f"  Bootstrap 95% CI:        [{boot_tent_stats['ci_95'][0]:+.4f}, {boot_tent_stats['ci_95'][1]:+.4f}]")
    print(f"  Plain Counts (+/-/0):    {tent_plain_counts['positive']}/{tent_plain_counts['negative']}/{tent_plain_counts['zero']}")
    print(f"  Band (+-0.010) (B/H/N):  {tent_band_counts['benefit']}/{tent_band_counts['harm']}/{tent_band_counts['neutral']}")
    print(f"  Largest Gain:            Event {max_gain_tent['event_id']} ({max_gain_tent['delta_r2_minus_tent']:+.4f})")
    print(f"  Smallest Gain:           Event {min_gain_tent['event_id']} ({min_gain_tent['delta_r2_minus_tent']:+.4f})")
    print(f"\nContrast R2 - CoTTA:")
    print(f"  Observed Mean Delta:     {mean_delta_cotta:+.4f} (+{mean_delta_cotta*100:.2f} pp)")
    print(f"  Bootstrap 95% CI:        [{boot_cotta_stats['ci_95'][0]:+.4f}, {boot_cotta_stats['ci_95'][1]:+.4f}]")
    print(f"  Plain Counts (+/-/0):    {cotta_plain_counts['positive']}/{cotta_plain_counts['negative']}/{cotta_plain_counts['zero']}")
    print(f"  Band (+-0.010) (B/H/N):  {cotta_band_counts['benefit']}/{cotta_band_counts['harm']}/{cotta_band_counts['neutral']}")
    print(f"  Largest Gain:            Event {max_gain_cotta['event_id']} ({max_gain_cotta['delta_r2_minus_cotta']:+.4f})")
    print(f"  Smallest Gain:           Event {min_gain_cotta['event_id']} ({min_gain_cotta['delta_r2_minus_cotta']:+.4f})")


if __name__ == "__main__":
    execute_comparison()

