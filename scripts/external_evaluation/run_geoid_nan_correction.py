#!/usr/bin/env python3
"""
Executes the bounded NaN preprocessing correction for a ChangeGuard model on GEOID-Flood.
1. Replays old-fill (nan=0.0) on affected batches and verifies exact bitwise match with archived counts.
2. Runs source-matched transform (nan=0.15) on affected batches.
3. Merges corrected counts for the 143 affected tiles with archived counts for the 1,158 unaffected tiles.
4. Checks all pixel and confusion invariants across all 1,301 tiles.
5. Recomputes canonical 9-event, 8-group, and pooled metrics.
6. Outputs full JSON results and replay audit records.
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party/KuroSiwo") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party/KuroSiwo"))

from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import (
    compute_confusion_counts,
    compute_flood_metrics,
    compute_macro_metrics,
    compute_pooled_metrics,
)
from src.external_evaluation.geoid_adapter import (
    discover_geoid_heldout_tiles,
    GEOIDTileDataset,
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


def evaluate_records_with_nan_val(
    model: torch.nn.Module,
    records: List[Any],
    device: str,
    sar_nan_val: float,
    sar_clamp_max: float = 0.15,
    sar_mean: tuple = (0.0953, 0.0264),
    sar_std: tuple = (0.0427, 0.0215),
    flood_class: int = 2,
    ignore_class: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates batches of records, returning tile_confusion_counts mapped by tile_key."""
    dataset = GEOIDTileDataset(
        records=records,
        sar_clamp_max=sar_clamp_max,
        sar_nan_val=sar_nan_val,
        sar_mean=sar_mean,
        sar_std=sar_std,
    )
    # Maintain exact batch size of len(records) (1 or 2)
    loader = DataLoader(dataset, batch_size=len(records), shuffle=False, num_workers=0)

    tile_results = {}
    with torch.no_grad():
        for batch in loader:
            tile_keys = batch["tile_key"]
            event_ids = batch["event_id"]
            xA = batch["xA"].view(-1, 2, 256, 256).to(device, non_blocking=True)
            xB = batch["xB"].view(-1, 2, 256, 256).to(device, non_blocking=True)
            labels = batch["labels"].view(-1, 256, 256).cpu().numpy()
            valid_masks = batch["valid_mask"].view(-1, 256, 256).cpu().numpy()

            logits = model(xA, xB)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            chips_per_tile = 16
            for t_idx, (t_key, ev_id) in enumerate(zip(tile_keys, event_ids)):
                c_start = t_idx * chips_per_tile
                c_end = c_start + chips_per_tile
                t_preds = preds[c_start:c_end]
                t_labels = labels[c_start:c_end]
                t_val = valid_masks[c_start:c_end]

                t_counts = compute_confusion_counts(
                    pred_classes=t_preds,
                    labels=t_labels,
                    valid_mask=t_val,
                    flood_class=flood_class,
                    ignore_class=ignore_class,
                )
                t_counts["tile_key"] = t_key
                t_counts["event_id"] = ev_id
                tile_results[t_key] = t_counts

    return tile_results


def main():
    parser = argparse.ArgumentParser(description="Bounded NaN Correction for GEOID-Flood evaluation.")
    parser.add_argument("--config", type=str, default="configs/external_evaluation/geoid_eval_config_v1.json")
    parser.add_argument("--model_id", type=str, required=True, help="e.g. R0_s0, R2_s1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/external_evaluation/geoid_nan_corrected_v1")
    parser.add_argument("--log_dir", type=str, default="logs/external_evaluation/geoid_nan_corrected_v1")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = REPO_ROOT / args.output_dir
    log_dir = REPO_ROOT / args.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = REPO_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = next((m for m in cfg["models"] if m["model_id"] == args.model_id), None)
    if not model_cfg:
        raise ValueError(f"Model ID '{args.model_id}' not found in {args.config}")

    ckpt_path = REPO_ROOT / model_cfg["checkpoint_path"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Verify SHA-256 hash
    print(f"[{args.model_id}] Verifying checkpoint SHA-256...")
    actual_hash = compute_sha256(ckpt_path)
    expected_hash = model_cfg["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {ckpt_path}! Expected {expected_hash}, got {actual_hash}")
    print(f"[{args.model_id}] Checkpoint SHA-256 verified: {actual_hash[:16]}...")

    # Load Model
    print(f"[{args.model_id}] Loading model strictly on {args.device}...")
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
    print(f"[{args.model_id}] Discovering tiles in {base_dir}...")
    records = discover_geoid_heldout_tiles(base_dir, catalog_path)
    if len(records) != 1301:
        raise ValueError(f"Expected 1,301 tiles, found {len(records)}")

    # Load affected IDs
    nan_audit_path = REPO_ROOT / "outputs/external_evaluation/input_checks/geoid_check1_nan_audit.json"
    with open(nan_audit_path, "r", encoding="utf-8") as f:
        nan_audit = json.load(f)
    affected_ids = set(nan_audit["affected_tile_ids"])
    print(f"[{args.model_id}] Loaded {len(affected_ids)} verified affected tile IDs.")

    # Load original archived results
    archived_file = REPO_ROOT / f"outputs/external_evaluation/{args.model_id}_geoid_eval_results.json"
    with open(archived_file, "r", encoding="utf-8") as f:
        archived_data = json.load(f)
    archived_counts_by_tile = {t["tile_key"]: t for t in archived_data["tile_confusion_counts"]}

    # Identify batches containing affected tiles (preserving original batch size = 2)
    batch_size_tiles = cfg["runtime"]["batch_size_tiles"]  # 2
    affected_batches = []
    unaffected_batches_count = 0

    for i in range(0, len(records), batch_size_tiles):
        b_records = records[i:i + batch_size_tiles]
        if any(r.tile_key in affected_ids for r in b_records):
            affected_batches.append(b_records)
        else:
            unaffected_batches_count += 1

    print(f"[{args.model_id}] Total batches: {len(records)//2 + 1}. Affected batches: {len(affected_batches)}. Unaffected batches: {unaffected_batches_count}.")

    # PHASE 1: Replay old-fill (nan=0.0) on affected batches
    print(f"[{args.model_id}] Phase 1: Replaying old-fill (nan=0.0) on affected batches...")
    replay_mismatches = 0
    replay_records = {}

    for b_recs in affected_batches:
        b_res = evaluate_records_with_nan_val(
            model=model,
            records=b_recs,
            device=args.device,
            sar_nan_val=0.0,
            sar_clamp_max=cfg["preprocessing"]["sar_clamp_max"],
            sar_mean=tuple(cfg["preprocessing"]["sar_mean"]),
            sar_std=tuple(cfg["preprocessing"]["sar_std"]),
        )
        for t_key, t_counts in b_res.items():
            replay_records[t_key] = t_counts
            arch = archived_counts_by_tile[t_key]
            for k in ["tp", "fp", "fn", "tn", "valid_pixels"]:
                if t_counts[k] != arch[k]:
                    print(f"[{args.model_id}] REPLAY MISMATCH on {t_key} ({k}): replayed={t_counts[k]}, archived={arch[k]}")
                    replay_mismatches += 1

    if replay_mismatches > 0:
        raise RuntimeError(f"[{args.model_id}] Old-fill replay failed with {replay_mismatches} mismatches!")
    print(f"[{args.model_id}] Old-fill replay successful: exactly 0 mismatches across {len(replay_records)} forwarded tiles.")

    # Save replay verification log
    replay_log_file = log_dir / f"replay_verification_{args.model_id}.json"
    with open(replay_log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "REPLAY_VERIFIED_BITWISE_IDENTICAL",
            "forwarded_tiles_count": len(replay_records),
            "affected_tiles_checked": len([k for k in replay_records if k in affected_ids]),
            "mismatches": 0,
        }, f, indent=2)

    # PHASE 2: Run source-matched transform (nan=0.15) on affected batches
    print(f"[{args.model_id}] Phase 2: Running source-matched transform (nan=0.15) on affected batches...")
    corrected_records = {}

    for b_recs in affected_batches:
        b_res = evaluate_records_with_nan_val(
            model=model,
            records=b_recs,
            device=args.device,
            sar_nan_val=0.15,
            sar_clamp_max=cfg["preprocessing"]["sar_clamp_max"],
            sar_mean=tuple(cfg["preprocessing"]["sar_mean"]),
            sar_std=tuple(cfg["preprocessing"]["sar_std"]),
        )
        for t_key, t_counts in b_res.items():
            corrected_records[t_key] = t_counts

    # Compute difference on affected tiles
    affected_diffs = []
    affected_tp_shift = 0
    affected_fp_shift = 0
    affected_fn_shift = 0
    affected_tn_shift = 0

    for t_key in sorted(affected_ids):
        old_c = replay_records[t_key]
        new_c = corrected_records[t_key]
        d_tp = new_c["tp"] - old_c["tp"]
        d_fp = new_c["fp"] - old_c["fp"]
        d_fn = new_c["fn"] - old_c["fn"]
        d_tn = new_c["tn"] - old_c["tn"]

        affected_tp_shift += d_tp
        affected_fp_shift += d_fp
        affected_fn_shift += d_fn
        affected_tn_shift += d_tn

        if d_tp != 0 or d_fp != 0 or d_fn != 0 or d_tn != 0:
            affected_diffs.append({
                "tile_key": t_key,
                "event_id": old_c["event_id"],
                "delta_tp": d_tp,
                "delta_fp": d_fp,
                "delta_fn": d_fn,
                "delta_tn": d_tn,
            })

    print(f"[{args.model_id}] Affected tiles with non-zero shift: {len(affected_diffs)}/{len(affected_ids)}.")
    print(f"[{args.model_id}] Net shift on affected population: dTP={affected_tp_shift:+d}, dFP={affected_fp_shift:+d}, dFN={affected_fn_shift:+d}, dTN={affected_tn_shift:+d}.")

    diff_log_file = out_dir / f"{args.model_id}_affected_tiles_diff.json"
    with open(diff_log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "total_affected_tiles": len(affected_ids),
            "tiles_with_shifts": len(affected_diffs),
            "net_shifts": {
                "delta_tp": affected_tp_shift,
                "delta_fp": affected_fp_shift,
                "delta_fn": affected_fn_shift,
                "delta_tn": affected_tn_shift,
            },
            "shifts_per_tile": affected_diffs,
        }, f, indent=2)

    # PHASE 3: Merge into full 1,301-tile dataset
    print(f"[{args.model_id}] Phase 3: Merging into complete 1,301-tile dataset...")
    merged_tile_records = []
    event_accumulators = {}

    for r in records:
        t_key = r.tile_key
        ev_id = r.event_id

        if t_key in affected_ids:
            counts = corrected_records[t_key]
        else:
            counts = archived_counts_by_tile[t_key]

        # Invariant checks per tile
        assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == counts["valid_pixels"], f"Sum != valid on {t_key}"
        assert counts["tp"] + counts["fn"] == counts["gt_flood_pixels"], f"TP+FN != GT on {t_key}"
        assert counts["tp"] + counts["fp"] == counts["pred_flood_pixels"], f"TP+FP != Pred on {t_key}"

        merged_tile_records.append(counts)

        if ev_id not in event_accumulators:
            event_accumulators[ev_id] = {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "valid_pixels": 0, "gt_flood_pixels": 0, "pred_flood_pixels": 0,
            }
        for k in ["tp", "fp", "fn", "tn", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels"]:
            event_accumulators[ev_id][k] += counts[k]

    assert len(merged_tile_records) == 1301, f"Expected 1,301 merged records, got {len(merged_tile_records)}"
    total_valid = sum(t["valid_pixels"] for t in merged_tile_records)
    total_gt = sum(t["gt_flood_pixels"] for t in merged_tile_records)
    assert total_valid == 873090528, f"Total valid mismatch: {total_valid}"
    assert total_gt == 45048201, f"Total GT flood mismatch: {total_gt}"
    print(f"[{args.model_id}] Invariant verified: exactly {total_valid} valid pixels and {total_gt} GT flood pixels.")

    # Compute event-level metrics
    event_rows = []
    for ev_id, counts in sorted(event_accumulators.items()):
        rates = compute_flood_metrics(counts)
        row = {
            "event_id": ev_id,
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
        event_rows.append(row)

    macro_metrics = compute_macro_metrics(event_rows)
    pooled_metrics = compute_pooled_metrics(event_rows)
    iberia_grouping = compute_grouped_iberia_metrics(event_rows)

    elapsed = time.time() - t_start

    final_payload = {
        "model_id": args.model_id,
        "recipe": model_cfg["recipe"],
        "seed": model_cfg["seed"],
        "checkpoint_path": str(model_cfg["checkpoint_path"]),
        "sha256": actual_hash,
        "evaluation_device": args.device,
        "correction_runtime_seconds": float(elapsed),
        "total_tiles_evaluated": len(merged_tile_records),
        "total_chips_evaluated": len(merged_tile_records) * 16,
        "nan_treatment_correction": {
            "old_nan_fill": 0.0,
            "corrected_nan_fill": 0.15,
            "affected_tiles_recomputed": len(affected_ids),
            "unaffected_tiles_reused": 1301 - len(affected_ids),
        },
        "macro_metrics_9_events": macro_metrics,
        "macro_metrics_8_events_grouped": iberia_grouping["grouped_macro_metrics"],
        "pooled_metrics": pooled_metrics,
        "event_metrics_9_events": event_rows,
        "event_metrics_8_events_grouped": iberia_grouping["grouped_event_metrics"],
        "tile_confusion_counts": merged_tile_records,
    }

    final_out_file = out_dir / f"{args.model_id}_geoid_eval_results_nan_corrected.json"
    with open(final_out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"[{args.model_id}] Saved corrected results to {final_out_file}")
    print(f"\n--- Corrected Summary for {args.model_id} ---")
    print(f"Macro F1 (9 events):  {macro_metrics['macro_f1']:.6f} (was {archived_data['macro_metrics_9_events']['macro_f1']:.6f}, diff: {macro_metrics['macro_f1'] - archived_data['macro_metrics_9_events']['macro_f1']:+.6f})")
    print(f"Macro F1 (8 grouped): {iberia_grouping['grouped_macro_metrics']['macro_f1']:.6f}")
    print(f"Pooled F1:            {pooled_metrics['pooled_f1']:.6f}")
    print(f"Macro Precision:      {macro_metrics['macro_precision']:.6f}")
    print(f"Macro Recall:         {macro_metrics['macro_recall']:.6f}")
    print(f"Macro IoU:            {macro_metrics['macro_iou']:.6f}")
    print(f"Total correction runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

