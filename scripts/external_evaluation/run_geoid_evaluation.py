#!/usr/bin/env python3
"""
Executes evaluation of a frozen ChangeGuard model on the GEOID-Flood held-out partition.
Computes canonical 9-event metrics, 8-event grouped sensitivity metrics, and pooled metrics.
Outputs JSON results with full audit trail.
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party/KuroSiwo") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party/KuroSiwo"))

from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import (
    compute_flood_metrics,
    compute_macro_metrics,
    compute_pooled_metrics,
)
from src.external_evaluation.geoid_adapter import (
    discover_geoid_heldout_tiles,
    GEOIDTileDataset,
    evaluate_model_on_geoid,
)


def compute_sha256(filepath: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_grouped_iberia_metrics(event_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes sensitivity metrics with EMSR861 and EMSR864 merged into one Iberian event."""
    grouped_accumulators: Dict[str, Dict[str, int]] = {}

    for em in event_metrics:
        ev_id = em["event_id"]
        group_key = "EMSR861_864_Iberia" if ev_id in ["EMSR861", "EMSR864"] else ev_id

        if group_key not in grouped_accumulators:
            grouped_accumulators[group_key] = {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "valid_pixels": 0, "gt_flood_pixels": 0, "pred_flood_pixels": 0,
            }
        for k in ["tp", "fp", "fn", "tn", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels"]:
            grouped_accumulators[group_key][k] += em[k]

    grouped_rows = []
    for g_id, counts in sorted(grouped_accumulators.items()):
        rates = compute_flood_metrics(counts)
        row = {
            "event_id": g_id,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
            "valid_pixels": counts["valid_pixels"],
            "gt_flood_pixels": counts["gt_flood_pixels"],
            "pred_flood_pixels": counts["pred_flood_pixels"],
            "precision": rates["precision"],
            "recall": rates["recall"],
            "f1": rates["f1"],
            "iou": rates["iou"],
            "is_evaluable": rates["f1"] is not None,
        }
        grouped_rows.append(row)

    grouped_macro = compute_macro_metrics(grouped_rows)
    return {
        "grouped_event_metrics": grouped_rows,
        "grouped_macro_metrics": grouped_macro,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChangeGuard model on GEOID-Flood.")
    parser.add_argument("--config", type=str, default="configs/external_evaluation/geoid_eval_config_v1.json")
    parser.add_argument("--model_id", type=str, required=True, help="e.g. R0_s0, R2_s1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/external_evaluation")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Find model entry
    model_cfg = next((m for m in cfg["models"] if m["model_id"] == args.model_id), None)
    if not model_cfg:
        raise ValueError(f"Model ID '{args.model_id}' not found in {args.config}")

    ckpt_path = REPO_ROOT / model_cfg["checkpoint_path"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Verify SHA-256 hash
    print(f"Verifying checkpoint SHA-256 for {args.model_id}...")
    actual_hash = compute_sha256(ckpt_path)
    expected_hash = model_cfg["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {ckpt_path}! Expected {expected_hash}, got {actual_hash}")
    print(f"  Verified SHA-256: {actual_hash[:16]}...")

    # Load Model
    print(f"Loading SNUNet-ECAM checkpoint on {args.device}...")
    model, _ = SNUNetWrapper.load_from_checkpoint(
        str(ckpt_path),
        in_channels=2,
        out_ch=3,
        base_channel=32,
        device=args.device,
    )
    model.eval()

    # Discover Dataset
    base_dir = REPO_ROOT / cfg["data_paths"]["base_dir"]
    catalog_path = REPO_ROOT / cfg["data_paths"]["catalog_path"]
    print(f"Discovering tiles in {base_dir}...")
    records = discover_geoid_heldout_tiles(base_dir, catalog_path)
    print(f"  Discovered {len(records)} 4-component tiles.")

    prep_cfg = cfg["preprocessing"]
    dataset = GEOIDTileDataset(
        records=records,
        sar_clamp_max=prep_cfg["sar_clamp_max"],
        sar_mean=tuple(prep_cfg["sar_mean"]),
        sar_std=tuple(prep_cfg["sar_std"]),
    )

    # Run Evaluation
    print(f"Starting evaluation of {args.model_id} on {len(dataset)} tiles ({len(dataset) * 16} chips)...")
    t0 = time.time()
    res = evaluate_model_on_geoid(
        model=model,
        dataset=dataset,
        device=args.device,
        batch_size_tiles=cfg["runtime"]["batch_size_tiles"],
        num_workers=cfg["runtime"]["num_workers"],
        flood_class=prep_cfg["flood_class"],
        ignore_class=prep_cfg["ignore_class"],
    )
    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed:.2f}s ({elapsed/len(records):.3f}s/tile)!")

    # Sensitivity grouping
    iberia_grouping = compute_grouped_iberia_metrics(res["event_metrics"])

    # Compile Final Output
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model_id}_geoid_eval_results.json"

    final_payload = {
        "model_id": args.model_id,
        "recipe": model_cfg["recipe"],
        "seed": model_cfg["seed"],
        "checkpoint_path": str(model_cfg["checkpoint_path"]),
        "sha256": actual_hash,
        "evaluation_device": args.device,
        "evaluation_time_seconds": float(elapsed),
        "total_tiles_evaluated": len(records),
        "total_chips_evaluated": len(records) * 16,
        "macro_metrics_9_events": res["macro_metrics"],
        "macro_metrics_8_events_grouped": iberia_grouping["grouped_macro_metrics"],
        "pooled_metrics": res["pooled_metrics"],
        "event_metrics_9_events": res["event_metrics"],
        "event_metrics_8_events_grouped": iberia_grouping["grouped_event_metrics"],
        "tile_confusion_counts": res["tile_records"],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"Saved evaluation results to {out_file}")
    print(f"\n--- Summary for {args.model_id} ---")
    print(f"Macro F1 (9 events):  {res['macro_metrics']['macro_f1']:.4f}")
    print(f"Macro F1 (8 grouped): {iberia_grouping['grouped_macro_metrics']['macro_f1']:.4f}")
    print(f"Pooled F1:            {res['pooled_metrics']['pooled_f1']:.4f}")
    print(f"Macro Precision:      {res['macro_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall:         {res['macro_metrics']['macro_recall']:.4f}")
    print(f"Macro IoU:            {res['macro_metrics']['macro_iou']:.4f}")


if __name__ == "__main__":
    main()
