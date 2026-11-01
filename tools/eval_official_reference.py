"""Bounded evaluation of official KuroSiwo SNUNet checkpoint as an external reference.

Evaluates checkpoints/kuro_snunet_official.pt on the 5 source-inner-validation events:
[267, 275, 497, 518, 1111005]
using the Policy S2 slope candidate reconstruction [VV, VH, slope].

Guarantees:
1. Strict checkpoint loading (strict=True, frozen parameters).
2. Zero adaptation, zero optimizer, zero backward calls, zero parameter updates.
3. Weights hashed before and after inference to verify zero modification.
4. Runtime access audit guaranteeing zero access to quarantined TTA-val or sealed test events.
5. Produces event_validation_metrics.csv, reference_summary.json, run_manifest.json, access_trace.json, eval.log.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.source_dataset import KuroSourceDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import EventMetricAccumulator, compute_flood_metrics

FORBIDDEN_TTA_VAL_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def hash_model_parameters(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode("utf-8"))
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_git_info(cwd: Path) -> Dict[str, Any]:
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd).decode().strip()
        return {"commit": commit, "dirty": len(status) > 0}
    except Exception:
        return {"commit": "UNKNOWN", "dirty": True}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate official checkpoint on source-inner validation set.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reference/official_snunet_reference_slope_candidate.json",
        help="Path to reference configuration JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/week3_official_reference",
        help="Output directory for reference evaluation artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., cuda:0 or cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = (REPO_ROOT / (args.output_dir or config.get("output_specification", {}).get("output_dir", "outputs/week3_official_reference"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("eval_official_ref")

    logger.info("=" * 70)
    logger.info("ChangeGuard Official Checkpoint Bounded Reference Evaluation")
    logger.info(f"Config:      {config_path}")
    logger.info(f"Output Dir:  {output_dir}")
    logger.info("=" * 70)

    # 1. Hashes and Provenance Verification
    config_hash = sha256_file(config_path)
    logger.info(f"Config SHA-256: {config_hash}")

    ckpt_rel = config["official_checkpoint"]["relative_path"]
    ckpt_path = (REPO_ROOT / ckpt_rel).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Official checkpoint not found: {ckpt_path}")
    ckpt_hash = sha256_file(ckpt_path)
    expected_ckpt_hash = config["official_checkpoint"]["sha256"]
    assert ckpt_hash == expected_ckpt_hash, f"Checkpoint hash mismatch: {ckpt_hash} vs {expected_ckpt_hash}"
    logger.info(f"Checkpoint verified: {ckpt_path} (SHA-256: {ckpt_hash})")

    split_rel = config["dataset_specification"]["split_config_path"]
    split_path = (REPO_ROOT / split_rel).resolve()
    split_hash = sha256_file(split_path)
    expected_split_hash = config["dataset_specification"]["split_config_sha256"]
    assert split_hash == expected_split_hash, f"Split hash mismatch: {split_hash} vs {expected_split_hash}"
    logger.info(f"Split verified: {split_path} (SHA-256: {split_hash})")

    slope_stats_rel = config["input_specification"]["slope_specification"]["statistics_source"]
    slope_stats_path = (REPO_ROOT / slope_stats_rel).resolve()
    slope_stats_hash = sha256_file(slope_stats_path)
    expected_stats_hash = config["input_specification"]["slope_specification"]["statistics_sha256"]
    assert slope_stats_hash == expected_stats_hash, f"Slope stats hash mismatch: {slope_stats_hash} vs {expected_stats_hash}"
    logger.info(f"Slope stats verified: {slope_stats_path} (SHA-256: {slope_stats_hash})")

    # 2. Device Selection
    device_str = args.device or config.get("inference_protocol", {}).get("device", "cuda:0")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to cpu.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Computation Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    # 3. Model Loading & Freezing
    logger.info("Loading official checkpoint strictly into SNUNetWrapper(in_channels=3, out_ch=3, base_channel=32)...")
    wrapper, ckpt_meta = SNUNetWrapper.load_from_checkpoint(
        str(ckpt_path),
        in_channels=3,
        out_ch=3,
        base_channel=32,
        device=str(device),
    )
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad = False

    initial_weights_hash = hash_model_parameters(wrapper.model)
    logger.info(f"Initial model state hash: {initial_weights_hash}")
    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info(f"Strict load complete. Total parameters: {total_params:,}")

    # 4. Dataset Loading
    logger.info("Initializing KuroSourceDataset for split='val' with include_slope=True...")
    dataset = KuroSourceDataset(
        split_config_path=str(split_path),
        split="val",
        include_slope=True,
        slope_stats_path=str(slope_stats_path),
        canonical_preevent=config["dataset_specification"].get("canonical_preevent", "pre_event_2"),
        sar_clamp_max=config["input_specification"]["sar_clamping"]["max"],
        sar_mean=tuple(config["input_specification"]["sar_normalization"]["mean"]),
        sar_std=tuple(config["input_specification"]["sar_normalization"]["std"]),
    )
    expected_samples = config["dataset_specification"]["expected_validation_sample_count"]
    assert len(dataset) == expected_samples, f"Sample count mismatch: {len(dataset)} vs {expected_samples}"
    logger.info(f"Dataset successfully loaded: {len(dataset)} samples across allowed events: {sorted(list(dataset.allowed_events))}")

    batch_size = int(config["inference_protocol"].get("batch_size", 16))
    num_workers = int(config["inference_protocol"].get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 5. Access Auditing Setup
    accessed_samples = set()
    accessed_events = set()
    quarantined_accessed = set()

    # 6. Evaluation Loop
    accumulator = EventMetricAccumulator()
    total_class_pixels = {0: 0, 1: 0, 2: 0, "invalid": 0}
    total_valid_pixels = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    logger.info("Starting deterministic inference pass (torch.no_grad(), zero optimizer, zero gradient updates)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            sample_ids = batch["sample_id"]
            event_ids = batch["event_id"]
            if torch.is_tensor(event_ids):
                event_ids = event_ids.cpu().tolist()

            for sid, eid in zip(sample_ids, event_ids):
                sid_str = str(sid)
                eid_int = int(eid)
                if eid_int in FORBIDDEN_TTA_VAL_EVENTS or eid_int in FORBIDDEN_SEALED_TEST_EVENTS:
                    quarantined_accessed.add(eid_int)
                    raise RuntimeError(f"CRITICAL RESEARCH INTEGRITY VIOLATION: Access attempted on forbidden event {eid_int}!")
                accessed_samples.add(sid_str)
                accessed_events.add(eid_int)

            xA = batch["xA"].to(device, non_blocking=True)
            xB = batch["xB"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)

            logits = wrapper(xA, xB)
            preds = torch.argmax(logits, dim=1)

            # Confusion matrix accumulation
            accumulator.update(
                event_ids=event_ids,
                pred_classes=preds,
                labels=labels,
                valid_masks=valid_masks,
            )

            # Class proportion accumulation
            valid_bool = (valid_masks == 1) & (labels != 3)
            for c in [0, 1, 2]:
                total_class_pixels[c] += int((preds[valid_bool] == c).sum().item())
            total_class_pixels["invalid"] += int((labels == 3).sum().item())
            total_valid_pixels += int(valid_bool.sum().item())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
                logger.info(f"Processed batch {batch_idx + 1:03d} / {len(loader):03d} (samples {len(accessed_samples):04d} / {len(dataset):04d})")

    elapsed_sec = time.time() - start_time
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    # 7. Post-Inference Invariant Verification
    final_weights_hash = hash_model_parameters(wrapper.model)
    assert initial_weights_hash == final_weights_hash, "CRITICAL ERROR: Model weights were modified during inference!"
    logger.info(f"Post-inference model state hash verified identical: {final_weights_hash}")
    assert len(quarantined_accessed) == 0, "Quarantined events were accessed!"

    # 8. Compute Metrics
    summary = accumulator.compute_summary()
    event_rows = accumulator.compute_event_table()

    # Find worst event
    worst_event = min(event_rows, key=lambda r: r["f1"])

    # Compute class proportions
    class_proportions = {
        "class_0_no_water": total_class_pixels[0] / total_valid_pixels if total_valid_pixels > 0 else 0.0,
        "class_1_permanent_water": total_class_pixels[1] / total_valid_pixels if total_valid_pixels > 0 else 0.0,
        "class_2_flood_water": total_class_pixels[2] / total_valid_pixels if total_valid_pixels > 0 else 0.0,
        "total_valid_pixels": total_valid_pixels,
        "total_invalid_pixels": total_class_pixels["invalid"],
    }

    pooled = summary["pooled_metrics"]
    logger.info("=" * 70)
    logger.info("OFFICIAL CHECKPOINT REFERENCE EVALUATION RESULTS")
    logger.info(f"Event-Macro Flood F1:   {summary['macro_f1']:.6f}")
    logger.info(f"Pixel-Pooled Flood F1:  {pooled['f1']:.6f}")
    logger.info(f"Worst Event F1:         {worst_event['f1']:.6f} (Event {worst_event['event_id']})")
    logger.info(f"Macro Precision / Recall: {summary['macro_precision']:.6f} / {summary['macro_recall']:.6f}")
    logger.info(f"Macro IoU:              {summary['macro_iou']:.6f}")
    logger.info(f"Pooled Precision / Rec: {pooled['precision']:.6f} / {pooled['recall']:.6f}")
    logger.info(f"Pooled IoU:             {pooled['iou']:.6f}")
    logger.info(f"Predicted Flood Pct:    {class_proportions['class_2_flood_water'] * 100:.2f}% of valid pixels")
    logger.info(f"Runtime:                {elapsed_sec:.2f} seconds | Peak GPU Memory: {peak_gpu_mem_mb:.1f} MB")
    logger.info("=" * 70)

    # 9. Save Artifacts
    # A. event_validation_metrics.csv
    event_csv_path = output_dir / "event_validation_metrics.csv"
    with open(event_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "event_id", "sample_count", "valid_pixels", "gt_flood_pixels",
            "pred_flood_pixels", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "iou"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in event_rows:
            writer.writerow(r)
    logger.info(f"Saved: {event_csv_path}")

    # B. reference_summary.json
    summary_data = {
        "run_id": config.get("output_specification", {}).get("run_id", "official_snunet_reference_slope_candidate"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": ckpt_hash,
            "architecture": "SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)",
            "weights_hash": final_weights_hash,
        },
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "evaluation_scope": {
            "split": "val",
            "allowed_events": sorted(list(dataset.allowed_events)),
            "processed_samples": len(accessed_samples),
            "expected_samples": expected_samples,
        },
        "macro_metrics": {
            "f1": summary["macro_f1"],
            "iou": summary["macro_iou"],
            "precision": summary["macro_precision"],
            "recall": summary["macro_recall"],
        },
        "pooled_metrics": {
            "f1": pooled["f1"],
            "iou": pooled["iou"],
            "precision": pooled["precision"],
            "recall": pooled["recall"],
        },
        "worst_event": {
            "event_id": worst_event["event_id"],
            "f1": worst_event["f1"],
            "iou": worst_event["iou"],
            "precision": worst_event["precision"],
            "recall": worst_event["recall"],
        },
        "class_proportions": class_proportions,
        "performance": {
            "runtime_seconds": elapsed_sec,
            "peak_gpu_memory_mb": peak_gpu_mem_mb,
            "device": str(device),
        },
        "per_event_metrics": {r["event_id"]: r for r in event_rows},
        "invariants": {
            "weights_identical_before_and_after": initial_weights_hash == final_weights_hash,
            "zero_optimizer_created": True,
            "zero_backward_calls": True,
            "zero_parameter_updates": True,
            "zero_tta_validation_accessed": len(quarantined_accessed) == 0,
            "zero_sealed_test_accessed": len(accessed_events & FORBIDDEN_SEALED_TEST_EVENTS) == 0,
        },
    }
    summary_json_path = output_dir / "reference_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved: {summary_json_path}")

    # C. access_trace.json
    trace_data = {
        "model_type": "official_snunet_reference_slope_candidate",
        "val_unique_samples_count": len(accessed_samples),
        "val_events_accessed": sorted(list(accessed_events)),
        "val_events_count": len(accessed_events),
        "quarantined_events_accessed": sorted(list(quarantined_accessed)),
        "assertions": {
            "no_tta_val_accessed": len(quarantined_accessed) == 0,
            "no_sealed_test_accessed": len(accessed_events & FORBIDDEN_SEALED_TEST_EVENTS) == 0,
            "accessed_events_match_inner_val": sorted(list(accessed_events)) == sorted(list(dataset.allowed_events)),
            "processed_samples_match_expected": len(accessed_samples) == expected_samples,
        },
    }
    trace_path = output_dir / "access_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    logger.info(f"Saved: {trace_path}")

    # D. run_manifest.json
    git_info = get_git_info(REPO_ROOT)
    manifest_data = {
        "reproducibility": {
            "run_id": config.get("output_specification", {}).get("run_id", "official_snunet_reference_slope_candidate"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_type": "official_checkpoint_reference",
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "python_executable": sys.executable,
            "environment": f"Linux {os.uname().release} ({os.uname().machine})",
            "gpu_device": str(device),
            "command": " ".join(sys.argv),
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "data_manifest": str(dataset.manifest_path),
            "completion_status": "COMPLETED",
            "split_sha256": split_hash,
            "slope_stats_sha256": slope_stats_hash,
            "output_paths": {
                "event_validation_metrics": str(event_csv_path),
                "reference_summary": str(summary_json_path),
                "access_trace": str(trace_path),
                "log": str(log_path),
            },
        },
        "compliance": {
            "sealed_test_events": sorted(list(FORBIDDEN_SEALED_TEST_EVENTS)),
            "quarantined_tta_events": sorted(list(FORBIDDEN_TTA_VAL_EVENTS)),
            "accessed_event_ids": sorted(list(accessed_events)),
            "sealed_event_accessed": False,
            "quarantined_tta_event_accessed": False,
            "labels_available_to_adaptation": False,
            "labels_used_for_updates": False,
            "is_partial_run": False,
            "expected_sample_count": expected_samples,
            "processed_sample_count": len(accessed_samples),
            "sample_count_matched": len(accessed_samples) == expected_samples,
            "zero_optimizer_created": True,
            "zero_backward_calls": True,
            "zero_parameter_updates": True,
            "weights_identical_before_and_after": initial_weights_hash == final_weights_hash,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved: {manifest_path}")

    logger.info("Official checkpoint reference evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
