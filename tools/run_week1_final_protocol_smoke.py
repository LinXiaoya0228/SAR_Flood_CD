"""Week 1 final approved protocol smoke confirmation script for Model A and Model B.

Executes a minimal protocol confirmation (10 optimizer steps) using:
- Restored frozen primary optimizer: Adam (lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999))
- No learning rate scheduler (scheduler = None)
- Effective batch size 32 (physical microbatch 8, accumulation 4)
- AMP FP16 on single GPU 0
- ControlledSourceLoss (CE + Soft Dice)
- Inner-train (22 events) and inner-val (5 events)
- Runtime access tracking and strict integrity enforcement
- Standardized RunManifest with run_type="source_training"
- Saves outputs under outputs/week1_final_protocol_smoke/
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KURO_ROOT = REPO_ROOT.parent / "third_party/KuroSiwo"
if str(KURO_ROOT) not in sys.path:
    sys.path.insert(0, str(KURO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.source_dataset import KuroSourceDataset
from src.training.source_trainer import SourceTrainer
from src.evaluation.run_manifest import RunManifest, compute_file_sha256


class TrackingDataLoader:
    """Wrapper around DataLoader to trace exact accessed samples and events."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.accessed_samples: List[str] = []
        self.accessed_events: Set[int] = set()

    def __iter__(self):
        for batch in self.dataloader:
            sample_ids = batch.get("sample_id", [])
            event_ids = batch.get("event_id", [])
            if torch.is_tensor(event_ids):
                event_ids = event_ids.cpu().tolist()
            for s, e in zip(sample_ids, event_ids):
                self.accessed_samples.append(str(s))
                self.accessed_events.add(int(e))
            yield batch

    def __len__(self):
        return len(self.dataloader)


def run_final_protocol_smoke(
    model_type: str,
    output_dir: Path,
    steps: int = 10,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict[str, Any]:
    assert model_type in ("model_a", "model_b")
    include_slope = (model_type == "model_b")
    in_channels = 3 if include_slope else 2
    output_dir.mkdir(parents=True, exist_ok=True)

    config_rel_path = (
        "configs/source/snunet_vvvh_v1.json"
        if model_type == "model_a"
        else "configs/source/snunet_vvvh_slope_v1.json"
    )
    config_abs_path = REPO_ROOT / config_rel_path
    config_hash = compute_file_sha256(config_abs_path)
    with open(config_abs_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n=======================================================")
    print(f"Final Protocol Smoke Confirmation: {model_type.upper()}")
    print(f"Config: {config_rel_path} (SHA-256: {config_hash})")
    print(f"In Channels: {in_channels} | Include Slope: {include_slope}")
    print(f"Target Steps: {steps} | Device: {device} | Output: {output_dir}")
    print(f"=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Datasets
    print("Initializing datasets...")
    slope_stats = "artifacts/week1/source_train_slope_statistics.json" if include_slope else None
    train_dataset = KuroSourceDataset(
        split="train",
        include_slope=include_slope,
        slope_stats_path=slope_stats,
    )
    val_dataset = KuroSourceDataset(
        split="val",
        include_slope=include_slope,
        slope_stats_path=slope_stats,
    )

    raw_train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    raw_val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    train_tracker = TrackingDataLoader(raw_train_loader)
    val_tracker = TrackingDataLoader(raw_val_loader)

    # 2. Model & Trainer from Config
    model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
    trainer = SourceTrainer.from_config(
        model=model,
        config=config,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )

    # Verify optimizer and scheduler runtime configuration
    assert isinstance(trainer.optimizer, torch.optim.Adam), "Optimizer must be torch.optim.Adam"
    assert not isinstance(trainer.optimizer, torch.optim.AdamW), "Optimizer must NOT be AdamW"
    assert trainer.lr_scheduler is None, "Scheduler must be None"
    opt_pg = trainer.optimizer.param_groups[0]
    assert opt_pg["lr"] == 1e-3, f"Learning rate must be 1e-3, got {opt_pg['lr']}"
    assert opt_pg["weight_decay"] == 0.0, f"Weight decay must be 0.0, got {opt_pg['weight_decay']}"
    assert opt_pg["betas"] == (0.9, 0.999), f"Betas must be (0.9, 0.999), got {opt_pg['betas']}"
    print(f"Verified Trainer instantiated from config: Adam(lr=1e-3, betas=(0.9, 0.999), wd=0.0), scheduler=None")

    # 3. Train steps
    start_time = time.time()
    print(f"Executing {steps} optimizer steps (microbatch 8, accum 4, effective batch 32)...")
    step_metrics = trainer.train_steps(train_tracker, num_optimizer_steps=steps)
    train_duration = time.time() - start_time

    first_loss = step_metrics[0]["loss_total"]
    final_loss = step_metrics[-1]["loss_total"]
    print(f"Step 01 Loss: {first_loss:.4f} -> Step {steps:02d} Loss: {final_loss:.4f} (Duration: {train_duration:.1f}s)")
    assert np.isfinite(first_loss) and np.isfinite(final_loss), "Losses must be finite!"

    # 4. Evaluate on validation subset (10 batches = 80 samples)
    print("Running validation subset pass (10 batches)...")
    val_summary, val_events = trainer.evaluate(val_tracker, max_batches=10)
    print(f"Validation Macro F1: {val_summary.get('macro_f1', 0.0):.4f} across {val_summary.get('num_events', 0)} events")

    # 5. Checkpoint saving with full provenance
    extra_meta = {
        "model_type": model_type,
        "in_channels": in_channels,
        "out_channels": 3,
        "base_channel": 32,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "lr_scheduler": None,
    }
    latest_ckpt = trainer.save_checkpoint(
        "checkpoint_latest.pt",
        metrics=val_summary,
        config_path=config_rel_path,
        config_hash=config_hash,
        extra_metadata=extra_meta,
    )
    best_ckpt = trainer.save_checkpoint(
        "checkpoint_best.pt",
        metrics=val_summary,
        config_path=config_rel_path,
        config_hash=config_hash,
        extra_metadata=extra_meta,
    )
    print(f"Saved checkpoints to {output_dir}")

    # 6. Test checkpoint reloading & RNG state
    test_reload_model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
    test_reload_trainer = SourceTrainer.from_config(
        model=test_reload_model,
        config=config,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
    resumed_state = test_reload_trainer.resume_from_checkpoint(latest_ckpt, restore_rng=True)
    assert test_reload_trainer.global_step == steps
    assert "rng_state" in resumed_state and resumed_state["rng_state"] is not None
    assert resumed_state["config_hash"] == config_hash
    print("Checkpoint reload and RNG state restoration verified.")

    # 7. Compile access trace
    all_accessed_events = sorted(list(train_tracker.accessed_events | val_tracker.accessed_events))
    trace_data = {
        "model_type": model_type,
        "train_samples_count": len(train_tracker.accessed_samples),
        "train_events_accessed": sorted(list(train_tracker.accessed_events)),
        "val_samples_count": len(val_tracker.accessed_samples),
        "val_events_accessed": sorted(list(val_tracker.accessed_events)),
        "all_accessed_events": all_accessed_events,
        "sample_intersection": list(set(train_tracker.accessed_samples) & set(val_tracker.accessed_samples)),
        "event_intersection": list(train_tracker.accessed_events & val_tracker.accessed_events),
    }
    trace_path = output_dir / "access_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    # 8. Compile standardized RunManifest
    manifest_path = output_dir / "run_manifest.json"
    manifest = RunManifest(
        run_id=f"week1_final_protocol_smoke_{model_type}",
        config_path=config_rel_path,
        config_hash=config_hash,
        checkpoint_path=latest_ckpt,
        data_manifest=str(train_dataset.manifest_path),
        accessed_event_ids=all_accessed_events,
        random_seed=seed,
        labels_available_to_adaptation=False,
        labels_used_for_updates=True,
        is_partial_run=True,
        expected_sample_count=len(train_tracker.accessed_samples),
        processed_sample_count=len(train_tracker.accessed_samples),
        output_paths={
            "checkpoint_latest": str(latest_ckpt),
            "checkpoint_best": str(best_ckpt),
            "metrics": str(output_dir / "smoke_run_metrics.json"),
            "access_trace": str(trace_path),
        },
        run_type="source_training",
    )
    manifest.mark_completed("SUCCESS")
    manifest.save(manifest_path)

    # 9. Save metrics summary
    metrics_path = output_dir / "smoke_run_metrics.json"
    metrics_data = {
        "model_type": model_type,
        "steps": steps,
        "optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "scheduler": None,
        },
        "config_sha256": config_hash,
        "train_duration_sec": train_duration,
        "step_metrics": step_metrics,
        "val_summary": val_summary,
        "val_events": val_events,
        "completion_status": "SUCCESS",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Final protocol smoke for {model_type} completed successfully.")
    return {
        "model_type": model_type,
        "config_hash": config_hash,
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "trace_path": str(trace_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Week 1 final approved protocol smoke confirmation.")
    parser.add_argument("--model_type", choices=["model_a", "model_b", "both"], default="both")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_base", type=str, default="outputs/week1_final_protocol_smoke")
    args = parser.parse_args()

    base_out = REPO_ROOT / args.output_base
    base_out.mkdir(parents=True, exist_ok=True)

    models_to_run = ["model_a", "model_b"] if args.model_type == "both" else [args.model_type]
    results = {}
    for m in models_to_run:
        out_dir = base_out / m
        results[m] = run_final_protocol_smoke(
            model_type=m,
            output_dir=out_dir,
            steps=args.steps,
            device=args.device,
            seed=args.seed,
        )

    print("\n=======================================================")
    print("All final protocol smoke confirmations complete!")
    print(json.dumps(results, indent=2))
    print("=======================================================")


if __name__ == "__main__":
    main()
