"""Day 7: Full label-free risk generation and prediction generation for Gate B.

Strict label-free execution:
- Does NOT load or access ground-truth labels during risk feature generation.
- Generates risk features and predictions for all 6 validation events across Seeds 0, 1, and 2.
- Saves artifacts into outputs/day6_7_gate_b/ without modifying or overwriting outputs/day3_5_tta/.
"""

from __future__ import annotations

import sys
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroUnlabeledValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.tent import TentAdapter
from src.risk.collector import RiskFeatureCollector
from src.risk.schema import validate_risk_feature_dict
from src.evaluation.run_manifest import RunManifest


def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_full_risk_generation(
    cfg_path: Path,
    output_root: Path,
    device: str = "cuda",
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    risk_dir = output_root / "risk_generation"
    pred_dir = output_root / "predictions"
    manifest_dir = output_root / "manifests"

    risk_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol_hash = compute_file_sha256(cfg_path)
    checkpoint_hash = cfg["checkpoint_sha256"]

    val_events = cfg["validation_event_ids"]
    seeds = cfg["seeds"]
    lr = cfg["learning_rate"]
    batch_size = cfg["batch_size"]

    print("=" * 80)
    print("DAYS 6–7: GATE B FULL LABEL-FREE RISK GENERATION")
    print(f"Protocol:   {cfg_path.name} (SHA-256: {protocol_hash})")
    print(f"Checkpoint: {cfg['checkpoint_path']} (SHA-256: {checkpoint_hash})")
    print(f"Events:     {val_events}")
    print(f"Seeds:      {seeds}")
    print(f"LR:         {lr}")
    print(f"Device:     {device}")
    print("=" * 80)

    # 1. Initialize strictly unlabeled dataset (does NOT read MK0_MLU label rasters)
    dataset = KuroUnlabeledValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=val_events,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        preprocessing_version=cfg["preprocessing_version"],
    )

    # Preload samples into host memory organized by event
    print(f"Loading {len(dataset)} validation samples into host memory (unlabeled)...")
    samples_by_event: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in val_events}
    for item in dataset:
        eid = item["event_id"]
        assert "label" not in item, f"Security Violation: Label found in unlabeled dataset item {item['sample_id']}!"
        # Retain sensory tensors and sample ID (zero ground truth)
        samples_by_event[eid].append({
            "sample_id": item["sample_id"],
            "event_id": eid,
            "xA": item["xA"],
            "xB": item["xB"],
            "pre1": item["pre1"],
            "pre2": item["pre2"],
            "post": item["post"],
            "valid_mask": item["valid_mask"],
        })

    # 2. Load models
    source_model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    source_model.eval()

    candidate_model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    adapter = TentAdapter(
        candidate_model,
        checkpoint_path=cfg["checkpoint_path"],
        learning_rate=lr,
        device=device,
    )

    collector = RiskFeatureCollector(
        source_model=source_model,
        protocol_hash=protocol_hash,
        checkpoint_hash=checkpoint_hash,
        device=device,
    )

    total_runs = len(seeds)
    for run_idx, seed in enumerate(seeds, 1):
        run_id = f"gate_b_tent_lr1e-4_seed{seed}"
        print(f"\n[{run_idx}/{total_runs}] Starting Run: {run_id} (Seed {seed})...", flush=True)

        manifest = RunManifest(
            run_id=run_id,
            config_path=str(cfg_path),
            checkpoint_path=cfg["checkpoint_path"],
            data_manifest=cfg["manifest_path"],
            accessed_event_ids=val_events,
            sealed_test_events=cfg["sealed_test_event_ids"],
            random_seed=seed,
            labels_available_to_adaptation=False,
            labels_used_for_updates=False,
            expected_sample_count=len(dataset),
        )

        run_risk_records: List[Dict[str, Any]] = []
        run_predictions: Dict[str, torch.Tensor] = {}

        run_start_time = time.time()

        for eid in val_events:
            adapter.reset_to_source()
            event_samples = samples_by_event[eid]

            # Reproducible stream order permuted by seed
            rng = np.random.RandomState(seed + eid * 1000)
            perm = rng.permutation(len(event_samples))
            ordered_samples = [event_samples[p] for p in perm]

            num_event_samples = len(ordered_samples)
            num_batches = (num_event_samples + batch_size - 1) // batch_size
            print(f"  Event {eid}: Streaming {num_event_samples} samples across {num_batches} batches...", flush=True)

            for b_idx in range(0, num_event_samples, batch_size):
                actual_batch_idx = (b_idx // batch_size) + 1
                batch_slice = ordered_samples[b_idx : b_idx + batch_size]

                sids = [s["sample_id"] for s in batch_slice]
                xA = torch.stack([s["xA"] for s in batch_slice]).to(device)
                xB = torch.stack([s["xB"] for s in batch_slice]).to(device)
                pre1 = torch.stack([s["pre1"] for s in batch_slice]).to(device)
                pre2 = torch.stack([s["pre2"] for s in batch_slice]).to(device)
                post = torch.stack([s["post"] for s in batch_slice]).to(device)
                sar_valid = torch.stack([s["valid_mask"] for s in batch_slice]).to(device)

                # 1. Candidate adaptation & prediction
                out = adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_valid)
                pred_class = out["predicted_class"].cpu()

                for j, sid in enumerate(sids):
                    run_predictions[sid] = pred_class[j]

                # 2. Risk probing (zero-mutation verified)
                batch_sensory = {
                    "sample_ids": sids,
                    "xA": xA,
                    "xB": xB,
                    "pre1": pre1,
                    "pre2": pre2,
                    "post": post,
                    "valid_mask": sar_valid,
                }

                record = collector.collect_risk_record(
                    candidate_adapter=adapter,
                    batch_sensory=batch_sensory,
                    run_id=run_id,
                    event_id=eid,
                    seed=seed,
                    batch_idx=actual_batch_idx,
                )
                run_risk_records.append(record)

        run_elapsed = time.time() - run_start_time

        # Save risk features JSON
        risk_file = risk_dir / f"{run_id}_risk_features.json"
        with open(risk_file, "w", encoding="utf-8") as f:
            json.dump(run_risk_records, f, indent=2)

        # Save predictions tensor dict
        pred_file = pred_dir / f"{run_id}_predictions.pt"
        torch.save(run_predictions, pred_file)

        manifest.processed_sample_count = len(run_predictions)
        manifest.mark_completed("SUCCESS")
        manifest.output_paths = {
            "risk_features": str(risk_file),
            "predictions": str(pred_file),
        }
        manifest_file = manifest_dir / f"manifest_{run_id}.json"
        manifest.save(manifest_file)

        print(f"[{run_idx}/{total_runs}] Run {run_id} finished in {run_elapsed:.1f}s. Saved artifacts:")
        print(f"  Risk features: {risk_file}")
        print(f"  Predictions:   {pred_file}")
        print(f"  Manifest:      {manifest_file}")

    print("\nALL GATE B RISK GENERATION RUNS FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    cfg_p = PROJECT_ROOT / "configs/gate_b_risk_protocol.json"
    out_p = PROJECT_ROOT / "outputs/day6_7_gate_b"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_full_risk_generation(cfg_p, out_p, device=dev)
