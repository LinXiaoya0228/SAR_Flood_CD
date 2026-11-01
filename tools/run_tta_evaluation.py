"""
Full Validation Evaluation Suite for Days 3–5:
Unsupervised Test-Time Adaptation (AdaBN and Tent) vs Corrected Canonical Baseline.

High-performance in-memory tensor execution:
1. Preloads all 2,826 validation samples into host RAM once (240 GB available).
2. Executes episodic online adaptation across all 13 experimental configurations with zero disk I/O bottleneck:
   - Source-only (Seed 0)
   - AdaBN (Seeds 0, 1, 2)
   - Tent at learning rates 1e-5, 1e-4, 1e-3 (Seeds 0, 1, 2 each)
3. Enforces:
   - CompletenessTracker (asserts 2,826 samples, 0 missing, 0 duplicates)
   - RunManifest (proves 0 test events accessed, 0 label leakage)
   - Zero label usage in adaptation: labels joined strictly post-run for offline metric computation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.source_only import SourceOnlyAdapter
from src.tta.adabn import AdaBNAdapter
from src.tta.tent import TentAdapter
from src.evaluation.metrics import EventMetricAccumulator
from src.evaluation.completeness import CompletenessTracker
from src.evaluation.run_manifest import RunManifest


VALIDATION_EVENTS = [279, 437, 520, 559, 1111003, 1111008]
EXPECTED_COUNTS = {
    279: 2381,
    437: 169,
    520: 92,
    559: 42,
    1111003: 19,
    1111008: 123,
}
TOTAL_EXPECTED_SAMPLES = 2826

# Diagnostic harm thresholds from protocol
HARMFUL_F1_THRESHOLD = -0.03
CRITICAL_RECALL_HARM_THRESHOLD = -0.10
CHANGE_COLLAPSE_RATIO = 0.50
BENEFICIAL_F1_THRESHOLD = 0.03


def preload_dataset_into_ram(
    dataset: KuroValidationDataset,
    batch_size: int = 32,
    num_workers: int = 16,
    cache_path: str = "/tmp/kuro_val_2826_samples.pt",
) -> Dict[str, Dict[str, Any]]:
    """Loads all dataset samples into host RAM for ultra-fast multi-seed adaptation."""
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"Loading preloaded samples from disk cache: {cache_file}...", flush=True)
        cached_samples = torch.load(cache_file, map_location="cpu")
        print(f"Loaded {len(cached_samples)} samples from cache successfully.", flush=True)
        return cached_samples

    print(f"Preloading {len(dataset)} samples into host memory using {num_workers} parallel workers...", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    cached_samples: Dict[str, Dict[str, Any]] = {}
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        sids = batch["sample_id"]
        eids = batch["event_id"].tolist()
        xA = batch["xA"]
        xB = batch["xB"]
        sar_valid = batch["valid_mask"]
        labels = batch["label"]

        for i, sid in enumerate(sids):
            cached_samples[sid] = {
                "sample_id": sid,
                "event_id": eids[i],
                "xA": xA[i].clone(),
                "xB": xB[i].clone(),
                "valid_mask": sar_valid[i].clone(),
                "label": labels[i].clone(),
            }
        if (batch_idx + 1) % 10 == 0 or len(cached_samples) == len(dataset):
            print(f"  Preloaded {len(cached_samples)}/{len(dataset)} samples ({time.time() - start_time:.1f}s elapsed)...", flush=True)

    elapsed = time.time() - start_time
    print(f"Preloaded {len(cached_samples)} samples in {elapsed:.1f}s. Saving to cache {cache_file}...", flush=True)
    try:
        torch.save(cached_samples, cache_file)
        print(f"Saved cache to {cache_file} successfully.", flush=True)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_file}: {e}", flush=True)
    print(f"Multi-seed adaptation will run in pure RAM.", flush=True)
    return cached_samples


def evaluate_event_stream_fast(
    adapter: Any,
    event_id: int,
    ordered_samples: List[Dict[str, Any]],
    source_predictions_cache: Optional[Dict[str, torch.Tensor]],
    batch_size: int = 16,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Runs episodic adaptation on preloaded in-memory samples for an event."""
    adapter.reset_to_source()

    trajectories: List[Dict[str, Any]] = []
    run_predictions: Dict[str, torch.Tensor] = {}
    offline_eval_records: List[Dict[str, Any]] = []

    num_samples = len(ordered_samples)
    batch_idx = 0

    for i in range(0, num_samples, batch_size):
        batch_idx += 1
        batch_slice = ordered_samples[i : i + batch_size]
        
        sids = [s["sample_id"] for s in batch_slice]
        eids = [s["event_id"] for s in batch_slice]
        
        # Sensory inputs to GPU
        xA = torch.stack([s["xA"] for s in batch_slice]).to(device)
        xB = torch.stack([s["xB"] for s in batch_slice]).to(device)
        sar_valid = torch.stack([s["valid_mask"] for s in batch_slice]).to(device)

        # Labels stay on CPU for offline evaluation
        labels = torch.stack([s["label"] for s in batch_slice])
        vlds = torch.stack([s["valid_mask"] for s in batch_slice])

        # Adapt and predict (strictly NO labels passed)
        out = adapter.adapt_and_predict(
            xA=xA,
            xB=xB,
            sar_valid_mask=sar_valid,
            sample_ids=sids,
            event_ids=eids,
        )

        pred_class = out["predicted_class"].cpu()
        step_diag = out["step_diagnostics"]
        drift_dict = step_diag.get("bn_drift", {"total_bn_drift_l2": 0.0})

        # Cache predictions
        for j, sid in enumerate(sids):
            run_predictions[sid] = pred_class[j]

        offline_eval_records.append({
            "event_id": event_id,
            "sample_ids": sids,
            "pred": pred_class,
            "label": labels,
            "valid_mask": vlds,
        })

        # Disagreement with source
        source_disagreement = 0
        if source_predictions_cache is not None:
            for j, sid in enumerate(sids):
                if sid in source_predictions_cache:
                    src_pred = source_predictions_cache[sid]
                    source_disagreement += int(torch.sum(pred_class[j] != src_pred).item())

        traj_row = {
            "event_id": event_id,
            "batch_idx": batch_idx,
            "batch_size": len(sids),
            "mean_entropy": step_diag["mean_entropy"],
            "mean_max_confidence": step_diag["mean_max_confidence"],
            "pred_flood_pixels": step_diag["pred_flood_pixels"],
            "pre_update_pred_flood": step_diag.get("pre_update_pred_flood_pixels", step_diag["pred_flood_pixels"]),
            "source_disagreement_pixels": source_disagreement,
            "total_bn_drift_l2": drift_dict["total_bn_drift_l2"],
            "cumulative_steps": step_diag["num_adaptation_steps"],
        }
        trajectories.append(traj_row)

    return trajectories, run_predictions, offline_eval_records


def run_full_experiment(
    cfg_path: str,
    output_root: str,
    device: str = "cuda",
) -> None:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 80)
    print("DAYS 3–5: FULL VALIDATION TEST-TIME ADAPTATION EVALUATION")
    print("=" * 80)
    print(f"Dataset root: {cfg['dataset_root']}")
    print(f"Checkpoint:   {cfg['checkpoint_path']}")
    print(f"Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Events:       {VALIDATION_EVENTS} (Total: {TOTAL_EXPECTED_SAMPLES} samples)")
    print("=" * 80)

    # 1. Initialize dataset with kuro_dem_sanitize_v1
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=VALIDATION_EVENTS,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
        preprocessing_version=cfg["dem_sanitization"]["preprocessing_version"],
        dem_nodata_threshold=cfg["dem_sanitization"]["nodata_threshold"],
        dem_replacement_value=cfg["dem_sanitization"]["replacement_value_normalized"],
    )

    # Preload samples into host RAM
    ram_cache = preload_dataset_into_ram(dataset, batch_size=32, num_workers=8)

    # Organize preloaded samples by event
    event_to_samples: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in VALIDATION_EVENTS}
    canonical_sample_ids: Dict[int, List[str]] = {eid: [] for eid in VALIDATION_EVENTS}
    for sample_meta in dataset.samples:
        eid = sample_meta["event_id"]
        sid = sample_meta["sample_id"]
        event_to_samples[eid].append(ram_cache[sid])
        canonical_sample_ids[eid].append(sid)

    # Verify counts
    for eid, expected in EXPECTED_COUNTS.items():
        assert len(event_to_samples[eid]) == expected, f"Event {eid} count mismatch!"

    # Load frozen source model
    raw_model, metadata = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)

    # Experimental matrix: 13 runs total
    experiments: List[Dict[str, Any]] = [
        {"method": "source_only", "lr": 0.0, "seed": 0, "run_id": "source_only_seed0"},
        {"method": "adabn", "lr": 0.0, "seed": 0, "run_id": "adabn_seed0"},
        {"method": "adabn", "lr": 0.0, "seed": 1, "run_id": "adabn_seed1"},
        {"method": "adabn", "lr": 0.0, "seed": 2, "run_id": "adabn_seed2"},
        {"method": "tent", "lr": 1e-5, "seed": 0, "run_id": "tent_lr1e-5_seed0"},
        {"method": "tent", "lr": 1e-5, "seed": 1, "run_id": "tent_lr1e-5_seed1"},
        {"method": "tent", "lr": 1e-5, "seed": 2, "run_id": "tent_lr1e-5_seed2"},
        {"method": "tent", "lr": 1e-4, "seed": 0, "run_id": "tent_lr1e-4_seed0"},
        {"method": "tent", "lr": 1e-4, "seed": 1, "run_id": "tent_lr1e-4_seed1"},
        {"method": "tent", "lr": 1e-4, "seed": 2, "run_id": "tent_lr1e-4_seed2"},
        {"method": "tent", "lr": 1e-3, "seed": 0, "run_id": "tent_lr1e-3_seed0"},
        {"method": "tent", "lr": 1e-3, "seed": 1, "run_id": "tent_lr1e-3_seed1"},
        {"method": "tent", "lr": 1e-3, "seed": 2, "run_id": "tent_lr1e-3_seed2"},
    ]

    source_predictions_cache: Optional[Dict[str, torch.Tensor]] = None
    source_event_metrics_lookup: Dict[Any, Any] = {}

    all_event_metrics_rows: List[Dict[str, Any]] = []

    total_runs = len(experiments)
    for run_idx, exp in enumerate(experiments, 1):
        method = exp["method"]
        lr = exp["lr"]
        seed = exp["seed"]
        run_id = exp["run_id"]

        print(f"\n[{run_idx}/{total_runs}] Starting Run: {run_id} (Method: {method.upper()}, LR: {lr}, Seed: {seed})", flush=True)

        manifest = RunManifest(
            run_id=run_id,
            config_path=cfg_path,
            checkpoint_path=cfg["checkpoint_path"],
            data_manifest=cfg["manifest_path"],
            accessed_event_ids=VALIDATION_EVENTS,
            sealed_test_events=cfg["sealed_test_event_ids"],
            random_seed=seed,
            labels_available_to_adaptation=False,
            labels_used_for_updates=False,
            is_partial_run=False,
            expected_sample_count=TOTAL_EXPECTED_SAMPLES,
        )

        completeness_tracker = CompletenessTracker(canonical_sample_ids, allow_partial=False)

        model = copy.deepcopy(raw_model)
        if method == "source_only":
            adapter = SourceOnlyAdapter(model, cfg["checkpoint_path"], device=device)
        elif method == "adabn":
            adapter = AdaBNAdapter(model, cfg["checkpoint_path"], device=device)
        elif method == "tent":
            adapter = TentAdapter(model, cfg["checkpoint_path"], learning_rate=lr, device=device)
        else:
            raise ValueError(f"Unknown method: {method}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        all_run_trajectories: List[Dict[str, Any]] = []
        all_eval_records: List[Dict[str, Any]] = []
        current_run_predictions: Dict[str, torch.Tensor] = {}

        for eid in VALIDATION_EVENTS:
            samples_for_event = event_to_samples[eid]
            rng = np.random.RandomState(seed + eid * 1000)
            perm = rng.permutation(len(samples_for_event))
            perm_samples = [samples_for_event[p] for p in perm]

            trajs, preds, eval_recs = evaluate_event_stream_fast(
                adapter=adapter,
                event_id=eid,
                ordered_samples=perm_samples,
                source_predictions_cache=source_predictions_cache,
                batch_size=16,
                device=device,
            )

            all_run_trajectories.extend(trajs)
            current_run_predictions.update(preds)
            all_eval_records.extend(eval_recs)

        elapsed_time = time.perf_counter() - start_time
        peak_gpu_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        if method == "source_only" and source_predictions_cache is None:
            source_predictions_cache = current_run_predictions

        # OFFLINE EVALUATION: Join labels and compute metrics
        accumulator = EventMetricAccumulator()
        for batch_rec in all_eval_records:
            beid = batch_rec["event_id"]
            bsids = batch_rec["sample_ids"]
            bpred = batch_rec["pred"]
            blabel = batch_rec["label"]
            bvld = batch_rec["valid_mask"]

            accumulator.update([beid] * len(bsids), bpred, blabel, bvld)

            for j, sid in enumerate(bsids):
                double_valid_pix = int(torch.sum((bvld[j] == 1) & (blabel[j] != 3)).item())
                completeness_tracker.update([beid], [sid], [double_valid_pix])

        completeness_report = completeness_tracker.validate_or_raise()
        manifest.processed_sample_count = completeness_report["total_processed_samples"]
        manifest.mark_completed("SUCCESS")

        # Save manifest
        manifest_path = out_dir / f"manifest_{run_id}.json"
        manifest.save(manifest_path)

        # Save batch trajectories CSV
        traj_csv_path = traj_dir / f"{run_id}_trajectory.csv"
        with open(traj_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_run_trajectories[0].keys()))
            writer.writeheader()
            for tr in all_run_trajectories:
                writer.writerow(tr)

        event_table = accumulator.compute_event_table()
        summary = accumulator.compute_summary()
        pooled_row = summary["pooled_metrics"]

        macro_f1 = float(np.mean([r["f1"] for r in event_table]))
        macro_iou = float(np.mean([r["iou"] for r in event_table]))
        macro_prec = float(np.mean([r["precision"] for r in event_table]))
        macro_rec = float(np.mean([r["recall"] for r in event_table]))
        total_pred_flood = sum(r["pred_flood_pixels"] for r in event_table)

        final_bn_drift = adapter.compute_bn_drift()["total_bn_drift_l2"]

        if method == "source_only":
            for r in event_table:
                source_event_metrics_lookup[r["event_id"]] = copy.deepcopy(r)
            source_event_metrics_lookup["macro"] = {
                "f1": macro_f1, "iou": macro_iou, "precision": macro_prec, "recall": macro_rec,
                "pred_flood_pixels": total_pred_flood,
            }
            source_event_metrics_lookup["pooled"] = copy.deepcopy(pooled_row)

            print("Verifying Source-Only baseline reproduction against corrected canonical metrics...")
            EXPECTED_F1S = {279: 0.6301, 437: 0.7481, 520: 0.7660, 559: 0.0816, 1111003: 0.3291, 1111008: 0.8325}
            for r in event_table:
                eid = r["event_id"]
                exp_f1 = EXPECTED_F1S[eid]
                diff = abs(r["f1"] - exp_f1)
                assert diff < 1e-3, f"Event {eid} baseline reproduction mismatch: expected {exp_f1}, got {r['f1']:.4f}"
            print("SOURCE-ONLY BASELINE EXACTLY REPRODUCED WITHIN DETERMINISTIC TOLERANCE!")

        for r in event_table:
            eid = r["event_id"]
            src_m = source_event_metrics_lookup[eid]
            delta_f1 = r["f1"] - src_m["f1"]
            delta_rec = r["recall"] - src_m["recall"]
            delta_prec = r["precision"] - src_m["precision"]
            pred_ratio = (r["pred_flood_pixels"] / src_m["pred_flood_pixels"]) if src_m["pred_flood_pixels"] > 0 else 1.0

            is_harmful_f1 = delta_f1 <= HARMFUL_F1_THRESHOLD
            is_critical_recall = delta_rec <= CRITICAL_RECALL_HARM_THRESHOLD
            is_change_collapse = (pred_ratio < CHANGE_COLLAPSE_RATIO) and (delta_rec < 0.0)
            is_beneficial = delta_f1 >= BENEFICIAL_F1_THRESHOLD

            event_row = {
                "run_id": run_id,
                "method": method,
                "lr": lr,
                "seed": seed,
                "event_id": eid,
                "samples": r["sample_count"],
                "valid_pixels": r["valid_pixels"],
                "gt_flood_pixels": r["gt_flood_pixels"],
                "pred_flood_pixels": r["pred_flood_pixels"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "tn": r["tn"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "iou": r["iou"],
                "delta_precision": delta_prec,
                "delta_recall": delta_rec,
                "delta_f1": delta_f1,
                "pred_flood_ratio_to_source": pred_ratio,
                "harmful_f1": is_harmful_f1,
                "critical_recall_harm": is_critical_recall,
                "change_collapse": is_change_collapse,
                "beneficial": is_beneficial,
                "final_bn_drift_l2": final_bn_drift,
                "runtime_seconds": elapsed_time,
                "peak_gpu_mb": peak_gpu_mb,
            }
            all_event_metrics_rows.append(event_row)

        print(f"[{run_id}] Finished in {elapsed_time:.1f}s | Peak GPU: {peak_gpu_mb:.1f} MB | Final BN Drift: {final_bn_drift:.4f}", flush=True)
        print(f"[{run_id}] Macro F1: {macro_f1:.4f} (Delta: {macro_f1 - source_event_metrics_lookup['macro']['f1']:+.4f}) | Pooled F1: {pooled_row['f1']:.4f}", flush=True)

    # Write all event metrics to CSV
    all_events_csv = out_dir / "all_runs_event_metrics.csv"
    with open(all_events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_event_metrics_rows[0].keys()))
        writer.writeheader()
        for r in all_event_metrics_rows:
            writer.writerow(r)
    print(f"\nSaved all runs event metrics to: {all_events_csv}")

    # Write summary comparison table with macro and pooled aggregations
    generate_summary_and_gate_report(all_event_metrics_rows, experiments, source_event_metrics_lookup, out_dir)


def generate_summary_and_gate_report(
    all_event_rows: List[Dict[str, Any]],
    experiments: List[Dict[str, Any]],
    source_lookup: Dict[Any, Any],
    out_dir: Path,
) -> None:
    """Generates comparison tables, failure counts, and Gate A decision analysis."""
    summary_rows = []
    run_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_event_rows:
        run_to_rows.setdefault(r["run_id"], []).append(r)

    total_harmful_f1_events = 0
    total_critical_recall_events = 0
    total_change_collapse_events = 0
    total_beneficial_events = 0

    run_diagnostics = {}

    for exp in experiments:
        rid = exp["run_id"]
        rows = run_to_rows[rid]

        macro_f1 = float(np.mean([r["f1"] for r in rows]))
        macro_rec = float(np.mean([r["recall"] for r in rows]))
        macro_prec = float(np.mean([r["precision"] for r in rows]))
        macro_iou = float(np.mean([r["iou"] for r in rows]))

        src_macro_f1 = source_lookup["macro"]["f1"]
        delta_macro_f1 = macro_f1 - src_macro_f1

        cnt_harmful = sum(1 for r in rows if r["harmful_f1"])
        cnt_recall_harm = sum(1 for r in rows if r["critical_recall_harm"])
        cnt_collapse = sum(1 for r in rows if r["change_collapse"])
        cnt_beneficial = sum(1 for r in rows if r["beneficial"])

        total_harmful_f1_events += cnt_harmful
        total_critical_recall_events += cnt_recall_harm
        total_change_collapse_events += cnt_collapse
        total_beneficial_events += cnt_beneficial

        summary_rows.append({
            "run_id": rid,
            "method": exp["method"],
            "lr": exp["lr"],
            "seed": exp["seed"],
            "macro_f1": macro_f1,
            "delta_macro_f1": delta_macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_iou": macro_iou,
            "harmful_f1_events": cnt_harmful,
            "critical_recall_events": cnt_recall_harm,
            "change_collapse_events": cnt_collapse,
            "beneficial_events": cnt_beneficial,
            "final_bn_drift_l2": rows[0]["final_bn_drift_l2"],
            "runtime_seconds": rows[0]["runtime_seconds"],
        })

        run_diagnostics[rid] = {
            "macro_f1": macro_f1,
            "delta_macro_f1": delta_macro_f1,
            "harmful_events": [r["event_id"] for r in rows if r["harmful_f1"]],
            "collapse_events": [r["event_id"] for r in rows if r["change_collapse"]],
            "beneficial_events": [r["event_id"] for r in rows if r["beneficial"]],
        }

    summary_csv = out_dir / "summary_comparison.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"Saved summary comparison CSV to: {summary_csv}")

    # Evaluate Gate A Decision Criteria
    reasonable_runs = [r for r in summary_rows if r["method"] == "adabn" or (r["method"] == "tent" and r["lr"] in [1e-5, 1e-4])]
    reasonable_harmful_runs = [r for r in reasonable_runs if r["harmful_f1_events"] > 0 or r["change_collapse_events"] > 0]

    seeds_with_harm = set(r["seed"] for r in reasonable_harmful_runs)
    events_with_harm = set()
    for exp in reasonable_harmful_runs:
        events_with_harm.update(run_diagnostics[exp["run_id"]]["harmful_events"])
        events_with_harm.update(run_diagnostics[exp["run_id"]]["collapse_events"])

    if len(reasonable_harmful_runs) > 0 and len(seeds_with_harm) >= 2 and len(events_with_harm) >= 1:
        gate_a_decision = "STRONG_GO"
        gate_a_rationale = (
            f"Harmful adaptation reproduces across {len(seeds_with_harm)} stream seeds in reasonable configurations "
            f"(AdaBN / Tent 1e-4). Affected events: {sorted(list(events_with_harm))}. "
            "Confirms that standard unsupervised TTA causes repeatable change-detection degradation without safety controllers."
        )
    elif len(reasonable_harmful_runs) > 0:
        gate_a_decision = "WEAK_CONDITIONAL_GO"
        gate_a_rationale = "Harmful adaptation observed but limited across seeds or events."
    else:
        gate_a_decision = "NO_GO"
        gate_a_rationale = "No meaningful harm observed under reasonable TTA configurations."

    gate_report = {
        "gate_a_decision": gate_a_decision,
        "gate_a_rationale": gate_a_rationale,
        "total_runs_executed": len(experiments),
        "total_event_evaluations": len(all_event_rows),
        "aggregated_counts": {
            "total_harmful_f1_event_instances": total_harmful_f1_events,
            "total_critical_recall_harm_instances": total_critical_recall_events,
            "total_change_collapse_instances": total_change_collapse_events,
            "total_beneficial_event_instances": total_beneficial_events,
        },
        "run_diagnostics": run_diagnostics,
    }

    gate_json = out_dir / "gate_a_evaluation.json"
    with open(gate_json, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)
    print(f"Saved Gate A evaluation to: {gate_json}")
    print("\n" + "=" * 80)
    print(f"GATE A DECISION: {gate_a_decision}")
    print(f"Rationale: {gate_a_rationale}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_config.json")
    parser.add_argument("--output_root", default="outputs/day3_5_tta")
    args = parser.parse_args()
    run_full_experiment(args.config, args.output_root)
