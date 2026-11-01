"""Week 1 smoke training script for Model A and Model B.

Executes short controlled smoke runs (25 optimizer steps) using:
- Effective batch size 32 (physical microbatch 8, accumulation 4)
- AMP FP16 on GPU 0
- ControlledSourceLoss (CE + Soft Dice)
- Inner-train (22 events) and inner-val (5 events)
- Saves checkpoints and run manifests in outputs/week1_smoke/
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
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
from src.training.source_trainer import SourceTrainer, is_better_checkpoint
from src.evaluation.run_manifest import RunManifest


def run_smoke(
    model_type: str,
    output_dir: Path,
    steps: int = 25,
    device: str = "cuda:0",
    seed: int = 42,
) -> dict:
    assert model_type in ("model_a", "model_b")
    include_slope = (model_type == "model_b")
    in_channels = 3 if include_slope else 2
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting Week 1 Smoke Run: {model_type.upper()}")
    print(f"In Channels: {in_channels} | Include Slope: {include_slope}")
    print(f"Target Steps: {steps} | Device: {device} | Output: {output_dir}")
    print(f"=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Datasets
    print("Initializing datasets...")
    train_dataset = KuroSourceDataset(
        split="train",
        include_slope=include_slope,
        slope_stats_path="outputs/week1/source_train_slope_statistics.json" if include_slope else None,
    )
    val_dataset = KuroSourceDataset(
        split="val",
        include_slope=include_slope,
        slope_stats_path="outputs/week1/source_train_slope_statistics.json" if include_slope else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model & Trainer
    model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
    trainer = SourceTrainer(
        model=model,
        output_dir=output_dir,
        device=device,
        effective_batch_size=32,
        physical_microbatch_size=8,
        mixed_precision=(device != "cpu"),
        seed=seed,
    )

    # 3. Train steps
    start_time = time.time()
    print(f"Executing {steps} optimizer steps (effective batch size 32)...")
    step_metrics = trainer.train_steps(train_loader, num_optimizer_steps=steps)
    train_duration = time.time() - start_time

    first_loss = step_metrics[0]["loss_total"]
    final_loss = step_metrics[-1]["loss_total"]
    print(f"Step 01 Loss: {first_loss:.4f} -> Step {steps:02d} Loss: {final_loss:.4f} (Duration: {train_duration:.1f}s)")

    # 4. Evaluate on validation subset (first 25 batches to test metric calculation)
    print("Running validation pass (25 batches)...")
    val_summary, val_events = trainer.evaluate(val_loader, max_batches=25)
    print(f"Validation Macro F1: {val_summary.get('macro_f1', 0.0):.4f} across {val_summary.get('num_events', 0)} events")

    # 5. Checkpoint saving
    latest_ckpt = trainer.save_checkpoint("checkpoint_latest.pt", metrics=val_summary)
    best_ckpt = trainer.save_checkpoint("checkpoint_best.pt", metrics=val_summary)
    print(f"Saved checkpoints to {output_dir}")

    # 6. Test checkpoint reloading
    test_reload_model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
    test_reload_trainer = SourceTrainer(
        model=test_reload_model,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
    test_reload_trainer.resume_from_checkpoint(latest_ckpt)
    assert test_reload_trainer.global_step == steps

    # 7. Record run summary
    summary = {
        "model_type": model_type,
        "in_channels": in_channels,
        "include_slope": include_slope,
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "effective_batch_size": 32,
        "physical_microbatch_size": 8,
        "accumulation_steps": 4,
        "train_duration_seconds": round(train_duration, 2),
        "initial_step_loss": round(first_loss, 6),
        "final_step_loss": round(final_loss, 6),
        "step_metrics": step_metrics,
        "validation_summary": val_summary,
        "status": "SMOKE_RUN_PASS",
    }

    summary_file = output_dir / "smoke_run_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_smoke_dir = REPO_ROOT / "outputs/week1_smoke"

    # Model A Smoke Run
    dir_a = base_smoke_dir / "model_a"
    res_a = run_smoke("model_a", dir_a, steps=25, device=device)

    # Model B Smoke Run
    dir_b = base_smoke_dir / "model_b"
    res_b = run_smoke("model_b", dir_b, steps=25, device=device)

    print("\n=======================================================")
    print("ALL WEEK 1 SMOKE RUNS COMPLETED SUCCESSFULLY!")
    print(f"Model A Final Loss: {res_a['final_step_loss']:.4f}")
    print(f"Model B Final Loss: {res_b['final_step_loss']:.4f}")
    print("=======================================================")


if __name__ == "__main__":
    main()
