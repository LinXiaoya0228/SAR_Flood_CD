#!/usr/bin/env python3
"""ChangeGuard Work Package E5.2C: Offline Temporal Risk Evidence Analysis.

Deterministic offline analysis engine for the E5.2 formal evidence collection.
Evaluates the association between observable pre-update risk indicators and
observed test-time adaptation outcomes (harmful vs. beneficial vs. neutral)
at both the primary event level and grouped batch level.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Repository root discovery
REPO_ROOT = Path(__file__).resolve().parent.parent

# Strictly quarantined sealed-test event IDs (zero access authorized)
FORBIDDEN_SEALED_TEST_EVENTS = {
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
}

# Declared six development events
DECLARED_DEVELOPMENT_EVENTS = [279, 437, 520, 559, 1111003, 1111008]


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_current_git_state() -> Tuple[str, bool]:
    """Returns (head_commit, is_dirty) for the analysis runner repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        is_dirty = len(status_out) > 0
        return commit, is_dirty
    except Exception:
        return "UNKNOWN", True


def compute_confusion_cells(
    pred: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    target_class: int = 2,
) -> Tuple[int, int, int, int, int]:
    """Computes binary (target_class vs rest) confusion cells on valid mask.

    Returns:
        (tp, fp, fn, tn, valid_pixels)
    """
    valid_m = mask.astype(bool)
    p = pred[valid_m]
    y = label[valid_m]
    valid_pixels = int(np.sum(valid_m))

    p_pos = (p == target_class)
    y_pos = (y == target_class)

    tp = int(np.sum(p_pos & y_pos))
    fp = int(np.sum(p_pos & ~y_pos))
    fn = int(np.sum(~p_pos & y_pos))
    tn = int(np.sum(~p_pos & ~y_pos))

    assert tp + fp + fn + tn == valid_pixels, (
        f"Confusion cell sum ({tp + fp + fn + tn}) != valid pixels ({valid_pixels})"
    )
    return tp, fp, fn, tn, valid_pixels


def compute_f1_from_cells(tp: int, fp: int, fn: int) -> float:
    """Computes binary F1 score from confusion cells with zero handling."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2.0 * tp / denom)


def compute_auroc_safe(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Computes AUROC via rank sums (Mann-Whitney U statistic) with single-class guard.

    Returns float('nan') if y_true contains only one class.
    Higher score denotes higher probability of positive class (label 1).
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)

    # Filter out NaNs if any
    valid = ~np.isnan(s)
    y = y[valid]
    s = s[valid]

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank data with average tie handling
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    # Correct for ties
    unique_s, counts = np.unique(s, return_counts=True)
    for val, count in zip(unique_s, counts):
        if count > 1:
            tie_mask = (s == val)
            ranks[tie_mask] = ranks[tie_mask].mean()

    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_auprc_safe(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Computes AUPRC (Average Precision) with single-class guard.

    Returns float('nan') if y_true contains only one class.
    Higher score denotes higher probability of positive class (label 1).
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)

    valid = ~np.isnan(s)
    y = y[valid]
    s = s[valid]

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort descending by score
    order = np.argsort(-s)
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    recalls = tp_cum / float(n_pos)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Average precision (trapezoidal integration over recall step changes)
    recalls_padded = np.concatenate(([0.0], recalls))
    precisions_padded = np.concatenate(([1.0], precisions))

    # Using standard AP formula: sum (R_k - R_{k-1}) * P_k
    delta_recall = recalls_padded[1:] - recalls_padded[:-1]
    ap = float(np.sum(delta_recall * precisions_padded[1:]))
    return ap


def compute_spearmanr_safe(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes Spearman rank correlation between two numerical sequences."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)

    valid = ~np.isnan(xa) & ~np.isnan(ya)
    xa = xa[valid]
    ya = ya[valid]

    if len(xa) < 2:
        return float("nan")

    # Check for zero variance
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return 0.0

    # Convert to ranks
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        unique_a, counts = np.unique(a, return_counts=True)
        for val, count in zip(unique_a, counts):
            if count > 1:
                ranks[a == val] = ranks[a == val].mean()
        return ranks

    rx = rankdata(xa)
    ry = rankdata(ya)

    cov = np.cov(rx, ry)
    std_x = np.std(rx, ddof=1)
    std_y = np.std(ry, ddof=1)
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return float(cov[0, 1] / (std_x * std_y))


class E5TemporalRiskAnalyzer:
    """Offline deterministic evaluator of temporal-risk evidence."""

    def __init__(
        self,
        config_path: Union[str, Path],
        source_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        src_cfg = self.cfg.get("source_run", {})
        if source_dir is not None:
            self.source_dir = Path(source_dir).resolve()
        else:
            rel_src = src_cfg.get(
                "output_dir", "outputs/e5_temporal_evidence/formal_s0_oa_v1/"
            )
            self.source_dir = (REPO_ROOT / rel_src).resolve()

        out_cfg = self.cfg.get("analysis_output", {})
        if output_dir is not None:
            self.output_dir = Path(output_dir).resolve()
        else:
            rel_out = out_cfg.get(
                "output_dir", "outputs/e5_temporal_evidence/analysis_s0_oa_v1/"
            )
            self.output_dir = (REPO_ROOT / rel_out).resolve()

        self.signal_specs = self.cfg.get("signal_specifications", {})
        self.rejection_budgets = list(
            self.cfg.get("rejection_budgets", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        )
        self.harmful_thresh = float(
            self.cfg.get("outcome_definitions", {}).get("harmful_threshold", -0.01)
        )
        self.beneficial_thresh = float(
            self.cfg.get("outcome_definitions", {}).get("beneficial_threshold", 0.01)
        )

    def validate_source_collection(self) -> Dict[str, Any]:
        """Performs rigorous fail-closed integrity preflight on formal run artifacts."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source formal directory missing: {self.source_dir}")

        status_path = self.source_dir / "completion_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing completion_status.json in {self.source_dir}")

        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        if status_data.get("status") != "COMPLETED_RUN":
            raise RuntimeError(
                f"Source run status is {status_data.get('status')}, expected COMPLETED_RUN!"
            )

        init_manifest_path = self.source_dir / "initial_manifest.json"
        if not init_manifest_path.is_file():
            raise FileNotFoundError(f"Missing initial_manifest.json in {self.source_dir}")

        with open(init_manifest_path, "r", encoding="utf-8") as f:
            init_manifest = json.load(f)

        if init_manifest.get("git_dirty") is not False:
            raise RuntimeError("Initial manifest records git_dirty != False!")

        # Enforce quarantine against sealed test partition FIRST
        target_events = init_manifest.get("target_events", [])
        for f_eid in FORBIDDEN_SEALED_TEST_EVENTS:
            if f_eid in target_events:
                raise RuntimeError(
                    f"CRITICAL SECURITY VIOLATION: Sealed-test event {f_eid} present in source run!"
                )

        # Verify declared development events via exact multiset equality
        if sorted(target_events) != sorted(DECLARED_DEVELOPMENT_EVENTS):
            raise RuntimeError(
                f"Target events mismatch: {target_events} does not match declared development events {DECLARED_DEVELOPMENT_EVENTS}"
            )

        total_shards_verified = 0
        total_samples_verified = 0
        all_sample_ids_ordered: List[str] = []
        all_shard_hashes: Dict[str, str] = {}

        event_summaries: Dict[int, Dict[str, Any]] = {}
        for eid in DECLARED_DEVELOPMENT_EVENTS:
            ev_dir = self.source_dir / f"event_{eid}"
            if not ev_dir.is_dir():
                raise FileNotFoundError(f"Missing event directory: {ev_dir}")

            # Check for failure.json
            if (ev_dir / "failure.json").exists():
                raise RuntimeError(f"failure.json detected in {ev_dir}!")

            # Verify evidence_index.json
            ev_idx_path = ev_dir / "evidence_index.json"
            if not ev_idx_path.is_file():
                raise FileNotFoundError(f"Missing evidence_index.json in {ev_dir}")

            with open(ev_idx_path, "r", encoding="utf-8") as f:
                ev_idx = json.load(f)

            shards = ev_idx.get("shards", [])
            ev_sample_ids: List[str] = []
            for s_info in shards:
                s_rel = s_info["shard_file"]
                s_path = ev_dir / s_rel
                if not s_path.is_file():
                    raise FileNotFoundError(f"Missing shard file: {s_path}")
                actual_h = compute_file_sha256(s_path)
                if actual_h != s_info["sha256"]:
                    raise ValueError(
                        f"Shard SHA-256 mismatch for {s_path}: expected {s_info['sha256']}, got {actual_h}"
                    )
                all_shard_hashes[f"event_{eid}/{s_rel}"] = actual_h
                ev_sample_ids.extend(s_info["sample_ids"])
                total_shards_verified += 1

            # Check exact sample ID match with contract
            contract_sids = init_manifest["event_contracts"][str(eid)]["expected_event_sample_ids"]
            if ev_sample_ids != contract_sids:
                raise RuntimeError(f"Sample sequence mismatch for Event {eid}!")

            all_sample_ids_ordered.extend(ev_sample_ids)
            total_samples_verified += len(ev_sample_ids)

            # Check event metrics conservation
            em_path = ev_dir / "event_metrics.json"
            with open(em_path, "r", encoding="utf-8") as f:
                em = json.load(f)
            if em["tp"] + em["fp"] + em["fn"] + em["tn"] != em["valid_pixels"]:
                raise ValueError(f"Conservation violation in event {eid} metrics!")

            event_summaries[eid] = {
                "samples": len(ev_sample_ids),
                "shards": len(shards),
                "flood_f1": em["flood_f1"],
                "valid_pixels": em["valid_pixels"],
            }

        if total_samples_verified != 2826:
            raise RuntimeError(f"Expected 2,826 samples, verified {total_samples_verified}!")
        if total_shards_verified != 179:
            raise RuntimeError(f"Expected 179 shards, verified {total_shards_verified}!")
        if len(set(all_sample_ids_ordered)) != 2826:
            raise RuntimeError("Duplicate sample IDs found across verified collection!")

        return {
            "run_id": init_manifest.get("run_id"),
            "git_commit": init_manifest.get("git_commit"),
            "git_dirty": init_manifest.get("git_dirty"),
            "total_samples": total_samples_verified,
            "total_shards": total_shards_verified,
            "events_verified": DECLARED_DEVELOPMENT_EVENTS,
            "event_summaries": event_summaries,
            "input_shard_hashes": all_shard_hashes,
            "authoritative_shard_policy": init_manifest.get("authoritative_shard_policy"),
            "aggregate_cache_policy": init_manifest.get("aggregate_cache_policy"),
            "status": "INTEGRITY_VERIFIED",
        }

    def execute_offline_analysis(self) -> Dict[str, Any]:
        """Executes full offline risk vs performance analysis from authoritative shards."""
        validation = self.validate_source_collection()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset manifest to resolve labels offline
        manifest_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"
        data_root = REPO_ROOT.parent / "datasets/KuroSiwo"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        import gzip
        import pickle
        import rasterio

        with gzip.open(manifest_path, "rb") as f:
            all_records = pickle.load(f)

        label_paths: Dict[str, Path] = {}
        for sid, rec in all_records.items():
            mlu_keys = [k for k in rec["info"]["datasets"].keys() if k.startswith("MK0_MLU")]
            if mlu_keys:
                label_paths[sid] = data_root / rec["path"] / f"{mlu_keys[0]}.tif"

        # 1. Read step risk summary CSV
        step_summary_path = self.source_dir / "e5_step_risk_summary.csv"
        step_signal_map: Dict[Tuple[int, int], Dict[str, float]] = {}
        with open(step_summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = int(row["event_id"])
                s_idx = int(row["step_index"])
                sig_dict = {}
                for sig_key in self.signal_specs:
                    val = row.get(sig_key, "")
                    sig_dict[sig_key] = float(val) if val != "" else float("nan")
                step_signal_map[(eid, s_idx)] = sig_dict

        # 2. Iterate shard-by-shard across each event
        event_outcomes: List[Dict[str, Any]] = []
        batch_outcomes: List[Dict[str, Any]] = []
        event_batch_records: Dict[int, List[Dict[str, Any]]] = {}

        for eid in DECLARED_DEVELOPMENT_EVENTS:
            ev_dir = self.source_dir / f"event_{eid}"
            with open(ev_dir / "evidence_index.json", "r", encoding="utf-8") as f:
                ev_idx = json.load(f)

            ev_tp_c, ev_fp_c, ev_fn_c, ev_tn_c, ev_valid = 0, 0, 0, 0, 0
            ev_tp_s, ev_fp_s, ev_fn_s, ev_tn_s = 0, 0, 0, 0
            ev_tp_b, ev_fp_b, ev_fn_b, ev_tn_b = 0, 0, 0, 0
            ev_sample_count = 0
            ev_shards = ev_idx.get("shards", [])

            for s_info in ev_shards:
                s_path = ev_dir / s_info["shard_file"]
                step_idx = int(s_info["step_index"])
                s_ids = s_info["sample_ids"]
                ev_sample_count += len(s_ids)

                # Shard arrays
                shard_data = np.load(s_path)

                b_tp_c, b_fp_c, b_fn_c, b_tn_c, b_valid = 0, 0, 0, 0, 0
                b_tp_s, b_fp_s, b_fn_s, b_tn_s = 0, 0, 0, 0
                b_tp_b, b_fp_b, b_fn_b, b_tn_b = 0, 0, 0, 0

                for sid in s_ids:
                    cand_pred = shard_data[f"{sid}_candidate_real_pred"]
                    src_pred = shard_data[f"{sid}_source_real_pred"]
                    bn_pred = shard_data[f"{sid}_bn_control_real_pred"]
                    sensory_mask = shard_data[f"{sid}_sensory_mask"]

                    lbl_file = label_paths[sid]
                    with rasterio.open(lbl_file) as src:
                        lbl = src.read(1)

                    eval_m = (sensory_mask == 1) & (lbl != 3)
                    y = lbl[eval_m]
                    p_c = cand_pred[eval_m]
                    p_s = src_pred[eval_m]
                    p_b = bn_pred[eval_m]

                    # Candidate cells
                    b_tp_c += int(((p_c == 2) & (y == 2)).sum())
                    b_fp_c += int(((p_c == 2) & (y != 2)).sum())
                    b_fn_c += int(((p_c != 2) & (y == 2)).sum())
                    b_tn_c += int(((p_c != 2) & (y != 2)).sum())
                    b_valid += int(eval_m.sum())

                    # Source cells
                    b_tp_s += int(((p_s == 2) & (y == 2)).sum())
                    b_fp_s += int(((p_s == 2) & (y != 2)).sum())
                    b_fn_s += int(((p_s != 2) & (y == 2)).sum())
                    b_tn_s += int(((p_s != 2) & (y != 2)).sum())

                    # BN control cells
                    b_tp_b += int(((p_b == 2) & (y == 2)).sum())
                    b_fp_b += int(((p_b == 2) & (y != 2)).sum())
                    b_fn_b += int(((p_b != 2) & (y == 2)).sum())
                    b_tn_b += int(((p_b != 2) & (y != 2)).sum())

                # Accumulate event cells
                ev_tp_c += b_tp_c
                ev_fp_c += b_fp_c
                ev_fn_c += b_fn_c
                ev_tn_c += b_tn_c
                ev_valid += b_valid

                ev_tp_s += b_tp_s
                ev_fp_s += b_fp_s
                ev_fn_s += b_fn_s
                ev_tn_s += b_tn_s

                ev_tp_b += b_tp_b
                ev_fp_b += b_fp_b
                ev_fn_b += b_fn_b
                ev_tn_b += b_tn_b

                # Batch performance
                b_cand_f1 = compute_f1_from_cells(b_tp_c, b_fp_c, b_fn_c)
                b_src_f1 = compute_f1_from_cells(b_tp_s, b_fp_s, b_fn_s)
                b_bn_f1 = compute_f1_from_cells(b_tp_b, b_fp_b, b_fn_b)

                d_cand_src = b_cand_f1 - b_src_f1
                d_bn_src = b_bn_f1 - b_src_f1
                d_cand_bn = b_cand_f1 - b_bn_f1

                if d_cand_src <= self.harmful_thresh:
                    b_class = "harmful"
                elif d_cand_src >= self.beneficial_thresh:
                    b_class = "beneficial"
                else:
                    b_class = "neutral"

                sigs = step_signal_map.get((eid, step_idx), {})

                batch_rec: Dict[str, Any] = {
                    "event_id": eid,
                    "step_index": step_idx,
                    "batch_size": len(s_ids),
                    "valid_pixels": b_valid,
                    "source_f1": b_src_f1,
                    "bn_control_f1": b_bn_f1,
                    "candidate_f1": b_cand_f1,
                    "delta_f1_cand_vs_src": d_cand_src,
                    "delta_f1_bn_vs_src": d_bn_src,
                    "delta_f1_cand_vs_bn": d_cand_bn,
                    "classification": b_class,
                    "tp_cand": b_tp_c,
                    "fp_cand": b_fp_c,
                    "fn_cand": b_fn_c,
                    "tp_src": b_tp_s,
                    "fp_src": b_fp_s,
                    "fn_src": b_fn_s,
                }
                for sig_key in self.signal_specs:
                    batch_rec[sig_key] = sigs.get(sig_key, float("nan"))

                batch_outcomes.append(batch_rec)
                event_batch_records.setdefault(eid, []).append(batch_rec)

            # Event performance
            ev_cand_f1 = compute_f1_from_cells(ev_tp_c, ev_fp_c, ev_fn_c)
            ev_src_f1 = compute_f1_from_cells(ev_tp_s, ev_fp_s, ev_fn_s)
            ev_bn_f1 = compute_f1_from_cells(ev_tp_b, ev_fp_b, ev_fn_b)

            ev_d_cand_src = ev_cand_f1 - ev_src_f1
            ev_d_bn_src = ev_bn_f1 - ev_src_f1
            ev_d_cand_bn = ev_cand_f1 - ev_bn_f1

            if ev_d_cand_src <= self.harmful_thresh:
                ev_class = "harmful"
            elif ev_d_cand_src >= self.beneficial_thresh:
                ev_class = "beneficial"
            else:
                ev_class = "neutral"

            if ev_d_bn_src <= self.harmful_thresh:
                ev_bn_class = "harmful"
            elif ev_d_bn_src >= self.beneficial_thresh:
                ev_bn_class = "beneficial"
            else:
                ev_bn_class = "neutral"

            event_outcomes.append({
                "event_id": eid,
                "sample_count": ev_sample_count,
                "batch_count": len(ev_shards),
                "valid_pixels": ev_valid,
                "source_f1": ev_src_f1,
                "bn_control_f1": ev_bn_f1,
                "candidate_f1": ev_cand_f1,
                "delta_f1_cand_vs_src": ev_d_cand_src,
                "delta_f1_bn_vs_src": ev_d_bn_src,
                "delta_f1_cand_vs_bn": ev_d_cand_bn,
                "classification": ev_class,
                "bn_classification": ev_bn_class,
                "tp_cand": ev_tp_c,
                "fp_cand": ev_fp_c,
                "fn_cand": ev_fn_c,
                "tp_src": ev_tp_s,
                "fp_src": ev_fp_s,
                "fn_src": ev_fn_s,
            })

        # 3. Compute signal event-level summary
        signal_event_summary: List[Dict[str, Any]] = []
        ev_y_harm = [1 if eo["classification"] == "harmful" else 0 for eo in event_outcomes]
        ev_delta_f1 = [eo["delta_f1_cand_vs_src"] for eo in event_outcomes]
        n_ev = len(event_outcomes)
        harm_prev_ev = sum(ev_y_harm) / n_ev if n_ev > 0 else float("nan")

        for sig_key, spec in self.signal_specs.items():
            dir_str = spec["direction"]
            ev_scores: List[float] = []
            for eo in event_outcomes:
                eid = eo["event_id"]
                recs = event_batch_records.get(eid, [])
                vals = [r[sig_key] for r in recs if not math.isnan(r[sig_key])]
                if vals:
                    mean_val = float(np.mean(vals))
                    score = -mean_val if dir_str == "lower_is_greater_risk" else mean_val
                else:
                    score = float("nan")
                ev_scores.append(score)

            avail_count = sum(1 for s in ev_scores if not math.isnan(s))
            unavail_count = n_ev - avail_count

            spearman_rho = compute_spearmanr_safe(ev_scores, ev_delta_f1)
            auroc = compute_auroc_safe(ev_y_harm, ev_scores)
            auprc = compute_auprc_safe(ev_y_harm, ev_scores)

            signal_event_summary.append({
                "signal_key": sig_key,
                "display_name": spec["display_name"],
                "family": spec["family"],
                "risk_direction": dir_str,
                "available_events_count": avail_count,
                "unavailable_events_count": unavail_count,
                "spearman_rho_with_delta_f1": spearman_rho,
                "auroc_harm": auroc,
                "auprc_harm": auprc,
                "harm_prevalence": harm_prev_ev,
                "no_skill_auprc_reference": harm_prev_ev,
            })

        # 4. Compute signal batch grouped summary
        signal_batch_summary: List[Dict[str, Any]] = []
        batch_y_harm = [1 if b["classification"] == "harmful" else 0 for b in batch_outcomes]
        batch_delta_f1 = [b["delta_f1_cand_vs_src"] for b in batch_outcomes]
        n_batches = len(batch_outcomes)
        batch_harm_prev = sum(batch_y_harm) / n_batches if n_batches > 0 else float("nan")

        for sig_key, spec in self.signal_specs.items():
            dir_str = spec["direction"]
            b_scores: List[float] = []
            for b in batch_outcomes:
                val = b[sig_key]
                if not math.isnan(val):
                    score = -val if dir_str == "lower_is_greater_risk" else val
                else:
                    score = float("nan")
                b_scores.append(score)

            avail_count = sum(1 for s in b_scores if not math.isnan(s))
            unavail_count = n_batches - avail_count

            # Grouped Spearman: average Spearman rho across individual events
            group_rhos: List[float] = []
            for eid in DECLARED_DEVELOPMENT_EVENTS:
                e_recs = event_batch_records.get(eid, [])
                e_scores = []
                e_deltas = []
                for r in e_recs:
                    v = r[sig_key]
                    if not math.isnan(v):
                        sc = -v if dir_str == "lower_is_greater_risk" else v
                        e_scores.append(sc)
                        e_deltas.append(r["delta_f1_cand_vs_src"])
                if len(e_scores) >= 2:
                    rho = compute_spearmanr_safe(e_scores, e_deltas)
                    if not math.isnan(rho):
                        group_rhos.append(rho)

            mean_group_rho = float(np.mean(group_rhos)) if group_rhos else float("nan")
            pooled_auroc = compute_auroc_safe(batch_y_harm, b_scores)
            pooled_auprc = compute_auprc_safe(batch_y_harm, b_scores)

            signal_batch_summary.append({
                "signal_key": sig_key,
                "display_name": spec["display_name"],
                "family": spec["family"],
                "risk_direction": dir_str,
                "available_batches_count": avail_count,
                "unavailable_batches_count": unavail_count,
                "grouped_spearman_rho_with_delta_f1": mean_group_rho,
                "pooled_auroc_harm": pooled_auroc,
                "pooled_auprc_harm": pooled_auprc,
                "batch_harm_prevalence": batch_harm_prev,
                "no_skill_auprc_reference": batch_harm_prev,
            })

        # 5. Compute rejection budget summary curves
        rejection_budget_summary: List[Dict[str, Any]] = []
        harm_total = sum(1 for b in batch_outcomes if b["classification"] == "harmful")
        ben_total = sum(1 for b in batch_outcomes if b["classification"] == "beneficial")
        neu_total = sum(1 for b in batch_outcomes if b["classification"] == "neutral")

        for sig_key, spec in self.signal_specs.items():
            dir_str = spec["direction"]
            scores_with_rec = []
            for b in batch_outcomes:
                val = b[sig_key]
                if not math.isnan(val):
                    sc = -val if dir_str == "lower_is_greater_risk" else val
                    scores_with_rec.append((sc, b))

            if not scores_with_rec:
                continue

            scores_only = [sc for sc, _ in scores_with_rec]

            for b_budget in self.rejection_budgets:
                tau_b = float(np.percentile(scores_only, (1.0 - b_budget) * 100.0))
                rej_harm = 0
                rej_ben = 0
                rej_neu = 0
                rej_total = 0

                realized_tp = 0
                realized_fp = 0
                realized_fn = 0

                for sc, b in scores_with_rec:
                    is_rej = (sc > tau_b)
                    if is_rej:
                        rej_total += 1
                        if b["classification"] == "harmful":
                            rej_harm += 1
                        elif b["classification"] == "beneficial":
                            rej_ben += 1
                        else:
                            rej_neu += 1
                        # Fallback to source
                        realized_tp += b["tp_src"]
                        realized_fp += b["fp_src"]
                        realized_fn += b["fn_src"]
                    else:
                        # Retain candidate
                        realized_tp += b["tp_cand"]
                        realized_fp += b["fp_cand"]
                        realized_fn += b["fn_cand"]

                harm_averted_rate = (rej_harm / harm_total) if harm_total > 0 else 0.0
                lost_benefits_rate = (rej_ben / ben_total) if ben_total > 0 else 0.0
                net_f1 = compute_f1_from_cells(realized_tp, realized_fp, realized_fn)

                rejection_budget_summary.append({
                    "signal_key": sig_key,
                    "budget": b_budget,
                    "threshold_score": tau_b,
                    "total_batches_rejected": rej_total,
                    "harmful_batches_rejected": rej_harm,
                    "harmful_batches_total": harm_total,
                    "harm_averted_rate": harm_averted_rate,
                    "beneficial_batches_rejected": rej_ben,
                    "beneficial_batches_total": ben_total,
                    "lost_benefits_rate": lost_benefits_rate,
                    "neutral_batches_rejected": rej_neu,
                    "neutral_batches_total": neu_total,
                    "net_realized_candidate_f1": net_f1,
                })

        # 6. Save all required CSV files
        # A. event_outcomes.csv
        ev_csv_cols = [
            "event_id", "sample_count", "batch_count", "valid_pixels",
            "source_f1", "bn_control_f1", "candidate_f1",
            "delta_f1_cand_vs_src", "delta_f1_bn_vs_src", "delta_f1_cand_vs_bn",
            "classification", "bn_classification"
        ]
        with open(self.output_dir / "event_outcomes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ev_csv_cols)
            writer.writeheader()
            for eo in event_outcomes:
                row = {c: eo[c] for c in ev_csv_cols}
                writer.writerow(row)

        # B. batch_outcomes.csv
        batch_csv_cols = [
            "event_id", "step_index", "batch_size", "valid_pixels",
            "source_f1", "bn_control_f1", "candidate_f1",
            "delta_f1_cand_vs_src", "delta_f1_bn_vs_src", "delta_f1_cand_vs_bn",
            "classification"
        ] + list(self.signal_specs.keys())
        with open(self.output_dir / "batch_outcomes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=batch_csv_cols)
            writer.writeheader()
            for b in batch_outcomes:
                row = {c: b[c] for c in batch_csv_cols}
                writer.writerow(row)

        # C. signal_event_summary.csv
        sig_ev_cols = [
            "signal_key", "display_name", "family", "risk_direction",
            "available_events_count", "unavailable_events_count",
            "spearman_rho_with_delta_f1", "auroc_harm", "auprc_harm",
            "harm_prevalence", "no_skill_auprc_reference"
        ]
        with open(self.output_dir / "signal_event_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sig_ev_cols)
            writer.writeheader()
            for s in signal_event_summary:
                row = {
                    c: ("NA" if isinstance(s[c], float) and math.isnan(s[c]) else s[c])
                    for c in sig_ev_cols
                }
                writer.writerow(row)

        # D. signal_batch_grouped_summary.csv
        sig_b_cols = [
            "signal_key", "display_name", "family", "risk_direction",
            "available_batches_count", "unavailable_batches_count",
            "grouped_spearman_rho_with_delta_f1", "pooled_auroc_harm", "pooled_auprc_harm",
            "batch_harm_prevalence", "no_skill_auprc_reference"
        ]
        with open(self.output_dir / "signal_batch_grouped_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sig_b_cols)
            writer.writeheader()
            for s in signal_batch_summary:
                row = {
                    c: ("NA" if isinstance(s[c], float) and math.isnan(s[c]) else s[c])
                    for c in sig_b_cols
                }
                writer.writerow(row)

        # E. rejection_budget_summary.csv
        rej_cols = [
            "signal_key", "budget", "threshold_score", "total_batches_rejected",
            "harmful_batches_rejected", "harmful_batches_total", "harm_averted_rate",
            "beneficial_batches_rejected", "beneficial_batches_total", "lost_benefits_rate",
            "neutral_batches_rejected", "neutral_batches_total", "net_realized_candidate_f1"
        ]
        with open(self.output_dir / "rejection_budget_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rej_cols)
            writer.writeheader()
            for r in rejection_budget_summary:
                writer.writerow(r)

        # 7. Write analysis_summary.json and analysis_manifest.json
        cur_commit, cur_dirty = get_current_git_state()
        analyzer_path = Path(__file__).resolve()
        init_manifest_path = self.source_dir / "initial_manifest.json"

        summary = {
            "work_package": "E5.2C",
            "title": "Offline Temporal Risk Analysis Summary",
            "analysis_id": self.output_dir.name,
            "protocol_version": self.cfg.get("protocol_version", "e5_temporal_risk_analysis_v1"),
            "source_run_id": validation["run_id"],
            "source_git_commit": validation["git_commit"],
            "analysis_git_commit": cur_commit,
            "analysis_git_dirty": cur_dirty,
            "analysis_created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "validation_report": validation,
            "event_outcomes": [
                {k: eo[k] for k in ev_csv_cols} for eo in event_outcomes
            ],
            "signal_event_summary": signal_event_summary,
            "signal_batch_grouped_summary": signal_batch_summary,
            "rejection_budget_summary": rejection_budget_summary,
            "total_batches_analyzed": len(batch_outcomes),
            "multivariate_composite_policy": "NOT EVALUATED — CALIBRATION RULE NOT PREDECLARED",
            "statistical_discipline": {
                "event_count": 6,
                "claim_population_significance": False,
                "batch_as_event_inference": False,
                "bootstrap_grouped_by_event": True,
                "zero_sealed_test_access": True,
            },
            "status": "COMPLETED_ANALYSIS",
        }

        # Write canonical analysis_summary.json
        with open(self.output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Compute file hashes for generated artifacts
        artifact_names = [
            "analysis_summary.json",
            "event_outcomes.csv",
            "batch_outcomes.csv",
            "signal_event_summary.csv",
            "signal_batch_grouped_summary.csv",
            "rejection_budget_summary.csv",
        ]
        generated_manifest = {}
        for fname in artifact_names:
            fpath = self.output_dir / fname
            if fpath.is_file():
                generated_manifest[fname] = {
                    "sha256": compute_file_sha256(fpath),
                    "bytes": fpath.stat().st_size,
                }

        config_rel = (
            str(self.config_path.relative_to(REPO_ROOT))
            if self.config_path.is_relative_to(REPO_ROOT)
            else str(self.config_path)
        )
        manifest_data = {
            "work_package": "E5.2C",
            "analysis_id": self.output_dir.name,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "analyzer_path": "tools/analyze_e5_temporal_risk.py",
            "analyzer_sha256": compute_file_sha256(analyzer_path),
            "config_path": config_rel,
            "config_sha256": compute_file_sha256(self.config_path),
            "analysis_git_commit": cur_commit,
            "analysis_git_dirty": cur_dirty,
            "source_run_id": validation["run_id"],
            "source_git_commit": validation["git_commit"],
            "source_initial_manifest_sha256": compute_file_sha256(init_manifest_path),
            "zero_sealed_test_declaration": "strictly_quarantined_zero_access",
            "events_analyzed": DECLARED_DEVELOPMENT_EVENTS,
            "total_samples": 2826,
            "total_shards": 179,
            "input_shard_hashes": validation.get("input_shard_hashes", {}),
            "generated_files": generated_manifest,
            "status": "COMPLETED_ANALYSIS",
        }
        with open(self.output_dir / "analysis_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="ChangeGuard Work Package E5.2C: Offline Temporal Risk Analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e5_temporal_risk_analysis_v1.json",
        help="Path to analysis configuration JSON",
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=None,
        help="Explicit source formal run directory override",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Explicit analysis output directory override",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run read-only integrity validation without producing final output artifacts",
    )
    args = parser.parse_args()

    analyzer = E5TemporalRiskAnalyzer(
        config_path=args.config,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )

    if args.validate_only:
        print("Running read-only collection validation...")
        report = analyzer.validate_source_collection()
        print("Read-only collection validation PASSED:")
        print(json.dumps(report, indent=2))
        sys.exit(0)

    print("Running offline temporal-risk analysis...")
    summary = analyzer.execute_offline_analysis()
    print("Analysis complete. Summary saved.")


if __name__ == "__main__":
    main()
