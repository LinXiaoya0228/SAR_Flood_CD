#!/usr/bin/env python3
"""Second backbone (SiamUnet_diff) training engine for ChangeGuard E10.

Supports:
1. Source-inner recipe screening:
   - Trains on 22 inner-train events (19,178 samples).
   - Evaluates after each epoch on 5 inner-validation events (2,284 samples).
   - Computes Event-Macro Flood F1, Worst-Event Flood F1, Event-Macro IoU, Loss.
   - Saves best checkpoint by Event-Macro Flood F1 and final checkpoint.
2. Full all-source training:
   - Trains on all 27 source events (21,462 samples) for a fixed epoch budget.
   - Evaluates zero validation (fixed budget protocol).
   - Saves final checkpoint.

Strict research integrity:
- Zero access to the 6 TTA development events or 10 sealed-test events.
- Deterministic seeding and reproducibility.
- Bit-exact artifact and config hash tracking.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.source_dataset import KuroSourceDataset
from src.evaluation.losses import ControlledSourceLoss
from src.evaluation.metrics import EventMetricAccumulator
from src.models.siam_diff_wrapper import SiamDiffWrapper

FORBIDDEN_TTA_VAL_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SecondBackboneTrainer:
    """Trainer for SiamUnet_diff on KuroSiwo source partitions."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        device: str = "cuda:0",
    ):
        self.config = config
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_str = device
        self.device = torch.device(device)

        self.seed = int(config.get("seed", 0))
        seed_everything(self.seed)

        self.lr = float(config.get("learning_rate", 0.001))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.max_epochs = int(config.get("max_epochs", 20))
        self.effective_batch_size = int(config.get("effective_batch_size", 32))
        self.microbatch_size = int(config.get("microbatch_size", 16))
        self.accumulation_steps = self.effective_batch_size // self.microbatch_size

        self.loss_fn = ControlledSourceLoss(
            weight_ce=0.5,
            weight_dice=0.5,
            classes_in_dice=(0, 1, 2),
            ignore_index=3,
            smooth_epsilon=1.0,
            reduction="batch",
        ).to(self.device)

        # Initialize model
        self.wrapper = SiamDiffWrapper(in_channels=2, out_channels=3, dropout_p=0.2).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.wrapper.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.wrapper.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        for b_idx, batch in enumerate(dataloader):
            xA = batch["xA"].to(self.device)
            xB = batch["xB"].to(self.device)
            target = batch["label"].to(self.device)

            logits = self.wrapper(xA, xB)
            loss_tensor = self.loss_fn(logits, target)
            loss = loss_tensor / self.accumulation_steps
            loss.backward()

            total_loss += loss_tensor.item()
            n_batches += 1

            if (b_idx + 1) % self.accumulation_steps == 0 or (b_idx + 1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

        return float(total_loss / n_batches) if n_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.wrapper.eval()
        accumulator = EventMetricAccumulator()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            xA = batch["xA"].to(self.device)
            xB = batch["xB"].to(self.device)
            target = batch["label"].to(self.device)
            event_ids = batch["event_id"]
            valid_masks = batch["valid_mask"]

            logits = self.wrapper(xA, xB)
            loss_tensor = self.loss_fn(logits, target)
            total_loss += loss_tensor.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = target.cpu().numpy()
            vm_np = valid_masks.cpu().numpy()
            eids = [int(e) for e in event_ids]

            accumulator.update(
                event_ids=eids,
                pred_classes=preds,
                labels=targets_np,
                valid_masks=vm_np,
            )

        summary = accumulator.compute_summary()
        worst_f1 = (summary.get("worst_event_f1") or {}).get("f1")
        pooled_f1 = (summary.get("pooled_metrics") or {}).get("f1")

        metrics = {
            "macro_flood_f1": float(summary["macro_f1"]) if summary.get("macro_f1") is not None else 0.0,
            "worst_event_flood_f1": float(worst_f1) if worst_f1 is not None else 0.0,
            "macro_flood_iou": float(summary["macro_iou"]) if summary.get("macro_iou") is not None else 0.0,
            "macro_flood_recall": float(summary["macro_recall"]) if summary.get("macro_recall") is not None else 0.0,
            "pooled_flood_f1": float(pooled_f1) if pooled_f1 is not None else 0.0,
            "val_loss": float(total_loss / n_batches) if n_batches > 0 else 0.0,
            "event_table": accumulator.compute_event_table(),
        }
        return metrics

    def run_source_inner_screen(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Runs training and validation across max_epochs on source-inner split."""
        csv_path = self.output_dir / "training_history.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "epoch", "train_loss", "val_loss", "macro_flood_f1",
            "worst_event_flood_f1", "macro_flood_iou", "macro_flood_recall",
            "pooled_flood_f1", "elapsed_s"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        best_epoch = -1
        best_macro_f1 = -1.0
        best_worst_f1 = -1.0
        best_metrics: Dict[str, Any] = {}
        history = []

        print(f"\n================================================================================")
        print(f"Starting Source-Inner Recipe Screen: {self.output_dir.name}")
        print(f"Epochs: {self.max_epochs} | LR: {self.lr} | Effective Batch: {self.effective_batch_size}")
        print(f"================================================================================")

        for ep in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, ep)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            macro_f1 = val_metrics["macro_flood_f1"]
            worst_f1 = val_metrics["worst_event_flood_f1"]
            macro_iou = val_metrics["macro_flood_iou"]
            macro_rec = val_metrics["macro_flood_recall"]
            pooled_f1 = val_metrics["pooled_flood_f1"]
            val_loss = val_metrics["val_loss"]

            row = {
                "epoch": ep,
                "train_loss": f"{train_loss:.5f}",
                "val_loss": f"{val_loss:.5f}",
                "macro_flood_f1": f"{macro_f1:.5f}",
                "worst_event_flood_f1": f"{worst_f1:.5f}",
                "macro_flood_iou": f"{macro_iou:.5f}",
                "macro_flood_recall": f"{macro_rec:.5f}",
                "pooled_flood_f1": f"{pooled_f1:.5f}",
                "elapsed_s": f"{elapsed:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()
            history.append(row)

            print(
                f"Epoch {ep:2d}/{self.max_epochs:2d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Macro F1: {macro_f1:.4f} | Worst F1: {worst_f1:.4f} | "
                f"Macro IoU: {macro_iou:.4f}"
            )

            # Checkpoint selection: higher macro_flood_f1, then higher worst_event_flood_f1
            is_better = (macro_f1 > best_macro_f1 + 1e-5) or (
                abs(macro_f1 - best_macro_f1) <= 1e-5 and worst_f1 > best_worst_f1
            )
            if is_better:
                best_macro_f1 = macro_f1
                best_worst_f1 = worst_f1
                best_epoch = ep
                best_metrics = val_metrics

                torch.save(
                    {
                        "model_state_dict": self.wrapper.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                        "best_macro_f1": best_macro_f1,
                        "best_worst_f1": best_worst_f1,
                        "val_metrics": val_metrics,
                        "config": self.config,
                    },
                    self.output_dir / "best_checkpoint.pt",
                )

        csv_file.close()

        # Save final checkpoint
        torch.save(
            {
                "model_state_dict": self.wrapper.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.max_epochs,
                "config": self.config,
            },
            self.output_dir / "final_checkpoint.pt",
        )

        summary = {
            "output_directory": str(self.output_dir),
            "recipe_config": self.config,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "best_worst_f1": best_worst_f1,
            "best_val_metrics": best_metrics,
            "history": history,
        }
        with open(self.output_dir / "screen_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_all_source_train(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Runs full training across all 27 source events for fixed max_epochs."""
        csv_path = self.output_dir / "training_history.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = ["epoch", "train_loss", "elapsed_s"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print(f"\n================================================================================")
        print(f"Starting All-Source Training: {self.output_dir.name}")
        print(f"Seed: {self.seed} | Fixed Epoch Budget: {self.max_epochs} | LR: {self.lr}")
        print(f"================================================================================")

        for ep in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, ep)
            elapsed = time.time() - t0

            row = {
                "epoch": ep,
                "train_loss": f"{train_loss:.5f}",
                "elapsed_s": f"{elapsed:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()

            print(f"Epoch {ep:2d}/{self.max_epochs:2d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f}")

            # Save epoch checkpoint
            torch.save(
                {
                    "model_state_dict": self.wrapper.state_dict(),
                    "epoch": ep,
                    "train_loss": train_loss,
                    "seed": self.seed,
                    "config": self.config,
                },
                self.output_dir / f"checkpoint_epoch_{ep:02d}.pt",
            )

        csv_file.close()

        # Save final checkpoint
        final_ckpt_path = self.output_dir / "final_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": self.wrapper.state_dict(),
                "epoch": self.max_epochs,
                "seed": self.seed,
                "architecture": "SiamUnet_diff",
                "config": self.config,
            },
            final_ckpt_path,
        )

        final_sha256 = compute_file_sha256(final_ckpt_path)
        summary = {
            "output_directory": str(self.output_dir),
            "seed": self.seed,
            "architecture": "SiamUnet_diff",
            "fixed_epochs": self.max_epochs,
            "final_checkpoint": str(final_ckpt_path),
            "final_checkpoint_sha256": final_sha256,
            "config": self.config,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Second backbone trainer for E10.")
    parser.add_argument("--mode", choices=["source_inner_screen", "all_source_train"], required=True)
    parser.add_argument("--config", type=Path, required=True, help="Path to training config JSON.")
    parser.add_argument("--device", type=str, default="cuda:1", help="CUDA device.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = Path(cfg["output_directory"])
    trainer = SecondBackboneTrainer(config=cfg, output_dir=out_dir, device=args.device)

    if args.mode == "source_inner_screen":
        train_ds = KuroSourceDataset(
            split_config_path=cfg.get("split_path", "configs/splits/kuro_source_inner_split_v1.json"),
            split="train",
        )
        val_ds = KuroSourceDataset(
            split_config_path=cfg.get("split_path", "configs/splits/kuro_source_inner_split_v1.json"),
            split="val",
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=trainer.microbatch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=trainer.microbatch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        res = trainer.run_source_inner_screen(train_loader, val_loader)
        print(f"Screen complete. Best Epoch: {res['best_epoch']} | Best Macro F1: {res['best_macro_f1']:.4f}")

    elif args.mode == "all_source_train":
        all_ds = KuroSourceDataset(
            split_config_path=cfg.get("split_path", "configs/splits/kuro_source_all27_split_v1.json"),
            split="all_source",
        )
        train_loader = DataLoader(
            all_ds,
            batch_size=trainer.microbatch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        res = trainer.run_all_source_train(train_loader)
        print(f"All-Source Training complete for seed {res['seed']}. Final Checkpoint SHA-256: {res['final_checkpoint_sha256']}")


if __name__ == "__main__":
    main()
