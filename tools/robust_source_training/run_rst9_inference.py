#!/usr/bin/env python3
"""Inference runner for ChangeGuard RST9 Historical-Target Extension.

Evaluates trained SNUNet-ECAM checkpoints (R2 or matched R0) on the 16 historical
target events (6 dev events + 10 former test events).

Research Integrity:
- All 16 events were observed historically and do NOT constitute an unseen confirmatory test set.
- Evaluation runs strictly in model.eval() with no gradient updates or label access during inference.
- Binary flood metrics use Class 2 (newly inundated flood water) as positive class.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe numpy alias for older checkpoints if necessary
try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    sys.modules.setdefault("numpy._core.multiarray", _npc.multiarray)
except Exception:
    pass

from src.data.evaluation_dataset import KuroEvaluationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.robust_source_training.checkpoint import load_rst_checkpoint
from src.robust_source_training.evaluation import (
    compute_binary_confusion_counts,
    compute_rates_from_counts,
)

HISTORICAL_DEV_EVENTS: List[int] = [279, 437, 520, 559, 1111003, 1111008]
HISTORICAL_TEST_EVENTS: List[int] = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
ALL_HISTORICAL_TARGET_EVENTS: List[int] = HISTORICAL_DEV_EVENTS + HISTORICAL_TEST_EVENTS


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, str]:
    """Loads SNUNetWrapper and weights from modern RST or legacy week5 checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    ckpt_bytes = checkpoint_path.read_bytes()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()

    # Attempt modern RST checkpoint load first
    try:
        load_rst_checkpoint(path=checkpoint_path, model=model, device=str(device))
    except Exception:
        # Fall back to raw PyTorch load for legacy Week 5 format
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw))
        # Strip potential module. prefix
        cleaned_sd = {}
        for k, v in sd.items():
            cleaned_k = k[7:] if k.startswith("module.") else k
            cleaned_sd[cleaned_k] = v
        model.load_state_dict(cleaned_sd)

    model.to(device)
    model.eval()
    return model, ckpt_sha256


def evaluate_single_event(
    model: nn.Module,
    event_id: int,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates the model on a single target event."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_valid = 0
    total_samples = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            xA = batch["xA"].to(device)
            xB = batch["xB"].to(device)
            labels = batch["label"].cpu().numpy()
            valid_masks = batch["valid_mask"].cpu().numpy()

            logits = model(xA, xB)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            counts = compute_binary_confusion_counts(preds, labels, valid_masks)
            total_tp += counts["tp"]
            total_fp += counts["fp"]
            total_fn += counts["fn"]
            total_tn += counts["tn"]
            total_valid += counts["valid_pixels"]
            total_samples += len(batch["sample_id"])

    rates = compute_rates_from_counts(total_tp, total_fp, total_fn, total_tn)
    event_type = "dev" if event_id in HISTORICAL_DEV_EVENTS else "test"

    return {
        "event_id": event_id,
        "event_type": event_type,
        "sample_count": total_samples,
        "valid_pixel_count": total_valid,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "pred_flood_pixels": total_tp + total_fp,
        "ref_flood_pixels": total_tp + total_fn,
        **rates,
    }


def aggregate_historical_metrics(event_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes dev-macro, test-macro, grand-macro, and tail metrics."""
    dev_results = [r for r in event_results if r["event_type"] == "dev"]
    test_results = [r for r in event_results if r["event_type"] == "test"]

    dev_macro_f1 = float(np.mean([r["f1"] for r in dev_results])) if dev_results else 0.0
    dev_macro_iou = float(np.mean([r["iou"] for r in dev_results])) if dev_results else 0.0

    test_macro_f1 = float(np.mean([r["f1"] for r in test_results])) if test_results else 0.0
    test_macro_iou = float(np.mean([r["iou"] for r in test_results])) if test_results else 0.0

    grand_macro_f1 = float(np.mean([r["f1"] for r in event_results])) if event_results else 0.0
    grand_macro_iou = float(np.mean([r["iou"] for r in event_results])) if event_results else 0.0
    grand_macro_prec = float(np.mean([r["precision"] for r in event_results])) if event_results else 0.0
    grand_macro_rec = float(np.mean([r["recall"] for r in event_results])) if event_results else 0.0

    # Tail reliability: Bottom-3 mean F1
    sorted_by_f1 = sorted(event_results, key=lambda r: (r["f1"], r["event_id"]))
    bottom3 = sorted_by_f1[:3]
    bottom3_mean_f1 = float(np.mean([r["f1"] for r in bottom3])) if bottom3 else 0.0
    worst_event = sorted_by_f1[0] if sorted_by_f1 else {}

    # Pooled sums
    pooled_tp = sum(r["tp"] for r in event_results)
    pooled_fp = sum(r["fp"] for r in event_results)
    pooled_fn = sum(r["fn"] for r in event_results)
    pooled_tn = sum(r["tn"] for r in event_results)
    pooled_rates = compute_rates_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

    return {
        "n_events_total": len(event_results),
        "n_dev_events": len(dev_results),
        "n_test_events": len(test_results),
        "dev_macro": {
            "f1": round(dev_macro_f1, 6),
            "iou": round(dev_macro_iou, 6),
        },
        "test_macro": {
            "f1": round(test_macro_f1, 6),
            "iou": round(test_macro_iou, 6),
        },
        "grand_macro": {
            "f1": round(grand_macro_f1, 6),
            "iou": round(grand_macro_iou, 6),
            "precision": round(grand_macro_prec, 6),
            "recall": round(grand_macro_rec, 6),
        },
        "bottom3_tail": {
            "mean_f1": round(bottom3_mean_f1, 6),
            "events": [
                {"event_id": r["event_id"], "event_type": r["event_type"], "f1": r["f1"]}
                for r in bottom3
            ],
        },
        "worst_event": {
            "event_id": worst_event.get("event_id"),
            "event_type": worst_event.get("event_type"),
            "f1": worst_event.get("f1"),
        },
        "pooled": {
            "tp": pooled_tp,
            "fp": pooled_fp,
            "fn": pooled_fn,
            "tn": pooled_tn,
            **pooled_rates,
        },
        "per_event": event_results,
    }


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard RST9 Historical-Target Extension Runner")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--recipe", type=str, required=True, choices=["R0", "R2"], help="Recipe (R0 or R2)")
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1, 2], help="Random seed (0, 1, 2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="outputs/robust_source_training/rst9_inference", help="Output directory")
    parser.add_argument("--max_batches_per_event", type=int, default=None, help="Batch cap for testing")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = (REPO_ROOT / args.output_dir / args.run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    master_csv_path = (REPO_ROOT / args.output_dir / "all_historical_target_metrics.csv").resolve()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Loading model from {ckpt_path} on {device}...")
    model, ckpt_sha = load_model_checkpoint(ckpt_path, device)
    print(f"Model loaded successfully (SHA-256: {ckpt_sha[:16]}...)")

    dataset_root = REPO_ROOT.parent / "datasets/KuroSiwo"
    manifest_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"

    print("Loading historical evaluation manifest (single pass)...", flush=True)
    full_ds = KuroEvaluationDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        allowed_events=ALL_HISTORICAL_TARGET_EVENTS,
        load_labels=True,
        allow_sealed_test=True,
    )
    event_indices = {eid: [] for eid in ALL_HISTORICAL_TARGET_EVENTS}
    for idx, s in enumerate(full_ds.samples):
        eid = int(s["event_id"])
        if eid in event_indices:
            event_indices[eid].append(idx)

    event_results = []
    t0 = time.time()

    print(f"Evaluating across {len(ALL_HISTORICAL_TARGET_EVENTS)} historical target events ({len(full_ds)} total samples)...", flush=True)
    for idx, eid in enumerate(ALL_HISTORICAL_TARGET_EVENTS):
        t_event_0 = time.time()
        subset_ds = torch.utils.data.Subset(full_ds, event_indices[eid])
        res = evaluate_single_event(
            model=model,
            event_id=eid,
            dataset=subset_ds,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches_per_event,
        )
        elapsed_ev = time.time() - t_event_0
        event_results.append(res)
        print(
            f"  [{idx+1:02d}/{len(ALL_HISTORICAL_TARGET_EVENTS)}] Event {eid:7d} ({res['event_type']:4s}): "
            f"Samples={res['sample_count']:4d} | F1={res['f1']:.4f} | IoU={res['iou']:.4f} | "
            f"Prec={res['precision']:.4f} | Rec={res['recall']:.4f} ({elapsed_ev:.1f}s)",
            flush=True,
        )

    total_time = time.time() - t0
    summary = aggregate_historical_metrics(event_results)
    summary["run_id"] = args.run_id
    summary["recipe"] = args.recipe
    summary["seed"] = args.seed
    summary["checkpoint_path"] = str(ckpt_path)
    summary["checkpoint_sha256"] = ckpt_sha
    summary["total_runtime_seconds"] = round(total_time, 2)
    summary["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Save run-level JSON
    json_path = out_dir / "event_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved event metrics summary to {json_path}")

    # Save run-level CSV
    run_csv_path = out_dir / "event_metrics.csv"
    fieldnames = [
        "run_id", "recipe", "seed", "event_id", "event_type", "sample_count",
        "valid_pixel_count", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou"
    ]
    with open(run_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in event_results:
            writer.writerow({
                "run_id": args.run_id,
                "recipe": args.recipe,
                "seed": args.seed,
                "event_id": r["event_id"],
                "event_type": r["event_type"],
                "sample_count": r["sample_count"],
                "valid_pixel_count": r["valid_pixel_count"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "tn": r["tn"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "iou": r["iou"],
            })

    # Append to master CSV
    write_header = not master_csv_path.exists()
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in event_results:
            writer.writerow({
                "run_id": args.run_id,
                "recipe": args.recipe,
                "seed": args.seed,
                "event_id": r["event_id"],
                "event_type": r["event_type"],
                "sample_count": r["sample_count"],
                "valid_pixel_count": r["valid_pixel_count"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "tn": r["tn"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "iou": r["iou"],
            })

    print(
        f"\n--- {args.run_id} ({args.recipe}, Seed {args.seed}) Summary ---\n"
        f"Grand Macro F1: {summary['grand_macro']['f1']:.4f} (IoU: {summary['grand_macro']['iou']:.4f})\n"
        f"Dev Macro F1:   {summary['dev_macro']['f1']:.4f} (IoU: {summary['dev_macro']['iou']:.4f})\n"
        f"Test Macro F1:  {summary['test_macro']['f1']:.4f} (IoU: {summary['test_macro']['iou']:.4f})\n"
        f"Bottom-3 Mean:  {summary['bottom3_tail']['mean_f1']:.4f}\n"
        f"Worst Event:    {summary['worst_event']['event_id']} (F1: {summary['worst_event']['f1']:.4f})\n"
        f"Total Runtime:  {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()

