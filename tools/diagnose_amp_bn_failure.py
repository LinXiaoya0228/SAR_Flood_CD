"""Bounded causal diagnostic replay for AMP and BatchNorm numerical stability.

Instruments SNUNet training from an untouched checkpoint to record:
1. Exact per-batch sample IDs and event IDs.
2. Max absolute activation per major block.
3. First layer whose output contains NaN or Inf.
4. Pre- and post-forward BatchNorm running buffer finiteness.
5. Loss, logits, and gradient finiteness.
6. Fail-fast immediately at the first nonfinite event.
Supports both FP16 (AMP) and matched FP32 modes.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KURO_ROOT = REPO_ROOT.parent / "third_party/KuroSiwo"
if str(KURO_ROOT) not in sys.path:
    sys.path.insert(0, str(KURO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.source_dataset import KuroSourceDataset
from src.evaluation.losses import ControlledSourceLoss
from src.evaluation.run_manifest import compute_file_sha256

FORBIDDEN_TTA_VAL_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}


class InstrumentedDiagnosticRunner:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        output_dir: Path,
        precision_mode: str = "fp16",
        device: str = "cuda:0",
        max_epoch: int = 25,
    ):
        assert precision_mode in ("fp16", "fp32"), f"Invalid precision mode: {precision_mode}"
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.precision_mode = precision_mode
        self.device = torch.device(device)
        self.max_epoch = max_epoch
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = json.load(f)

        in_channels = int(self.config["architecture"]["in_channels_per_branch"])
        self.include_slope = (in_channels == 3)
        self.model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
        self.model.to(self.device)

        # Loss and optimizer setup matching protocol
        self.loss_fn = ControlledSourceLoss(
            weight_ce=0.5,
            weight_dice=0.5,
            classes_in_dice=[0, 1, 2],
            ignore_index=3,
            smooth_epsilon=1.0,
            squared_denominator=True,
            reduction="batch",
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        self.scaler = GradScaler() if self.precision_mode == "fp16" else None
        self.accumulation_steps = 4

        # Instrumentation buffers
        self.current_activations: Dict[str, float] = {}
        self.first_nonfinite_layer: Optional[str] = None
        self.layer_nonfinite_type: Optional[str] = None
        self.hooks = []
        self._register_hooks()

        # CSV loggers
        self.batch_log_path = self.output_dir / f"diagnostic_batches_{self.precision_mode}.csv"
        self.batch_csv_file = open(self.batch_log_path, "w", newline="", encoding="utf-8")
        self.batch_writer = csv.writer(self.batch_csv_file)
        self.batch_writer.writerow([
            "epoch", "batch_idx", "step", "sample_ids", "event_ids",
            "inputs_finite", "max_activation_layer", "max_activation_value",
            "first_nonfinite_layer", "logits_finite", "loss_finite", "loss_total",
            "loss_ce", "loss_dice", "gradients_finite", "scaler_scale",
            "bn_buffers_finite_pre", "bn_buffers_finite_post", "corrupted_bn_layers"
        ])

        # Summary metadata
        self.diagnostic_summary = {
            "precision_mode": self.precision_mode,
            "device": str(self.device),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": compute_file_sha256(self.checkpoint_path),
            "start_epoch": 21,
            "target_max_epoch": self.max_epoch,
            "failure_detected": False,
            "failure_details": None,
            "total_batches_processed": 0,
        }

    def _register_hooks(self):
        """Registers forward hooks on major SNUNet submodules to monitor activations."""
        monitored_submodules = [
            "conv0_0", "conv1_0", "conv2_0", "conv3_0", "conv4_0",
            "conv0_1", "conv1_1", "conv2_1", "conv3_1",
            "conv0_2", "conv1_2", "conv2_2",
            "conv0_3", "conv1_3",
            "conv0_4",
            "ca", "ca1",
            "conv_final"
        ]

        def get_hook(name):
            def hook(module, inp, out):
                if torch.is_tensor(out):
                    tensor = out
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    tensor = out[0]
                else:
                    return

                t_float = tensor.detach().float()
                has_nan = torch.isnan(t_float).any().item()
                has_inf = torch.isinf(t_float).any().item()

                if has_nan or has_inf:
                    if self.first_nonfinite_layer is None:
                        self.first_nonfinite_layer = name
                        self.layer_nonfinite_type = "NaN" if has_nan else "Inf"
                    self.current_activations[name] = float("nan")
                else:
                    self.current_activations[name] = float(t_float.abs().max().item())

            return hook

        for name, mod in self.model.model.named_children():
            if name in monitored_submodules:
                h = mod.register_forward_hook(get_hook(name))
                self.hooks.append(h)

    def audit_bn_buffers(self) -> Tuple[bool, List[str]]:
        """Audits every BatchNorm running_mean and running_var for finiteness."""
        all_finite = True
        corrupted = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                rm = mod.running_mean
                rv = mod.running_var
                rm_fin = torch.isfinite(rm.float()).all().item() if rm is not None else True
                rv_fin = torch.isfinite(rv.float()).all().item() if rv is not None else True
                if not (rm_fin and rv_fin):
                    all_finite = False
                    corrupted.append(name)
        return all_finite, corrupted

    def restore_exact_state(self):
        """Loads model, optimizer, scaler, and exact RNG states from checkpoint."""
        print(f"\n[DIAGNOSTIC] Loading state from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Restore RNG states
        if ckpt.get("rng_state") is not None:
            rng_s = ckpt["rng_state"].cpu() if hasattr(ckpt["rng_state"], "cpu") else ckpt["rng_state"]
            torch.set_rng_state(rng_s)
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            cuda_s = [s.cpu() if hasattr(s, "cpu") else s for s in ckpt["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_s)
        if ckpt.get("np_rng_state") is not None:
            np.random.set_state(ckpt["np_rng_state"])

        self.start_epoch = int(ckpt.get("epoch", 20)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        print(f"[DIAGNOSTIC] Resumed successfully at Epoch {self.start_epoch}, Step {self.global_step}")

    def run(self) -> Dict[str, Any]:
        self.restore_exact_state()

        split_path = REPO_ROOT / "configs/splits/kuro_source_inner_split_v1.json"
        slope_stats_path = REPO_ROOT / "artifacts/week1/source_train_slope_statistics.json"

        train_dataset = KuroSourceDataset(
            split_config_path=str(split_path),
            split="train",
            include_slope=self.include_slope,
            slope_stats_path=str(slope_stats_path) if self.include_slope else None,
        )

        def seed_worker(worker_id):
            worker_seed = (0 + worker_id + 1) % (2**32)
            np.random.seed(worker_seed)
            import random
            random.seed(worker_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        print(f"\n[DIAGNOSTIC START] Running Mode: {self.precision_mode.upper()} from Epoch {self.start_epoch} to {self.max_epoch}")
        total_batches = len(train_loader)
        stop_diagnostic = False

        try:
            for epoch in range(self.start_epoch, self.max_epoch + 1):
                if stop_diagnostic:
                    break

                self.model.train()
                print(f"\n--- Epoch {epoch:03d}/{self.max_epoch:03d} Diagnostic Pass ---")

                for batch_idx, batch in enumerate(train_loader):
                    self.diagnostic_summary["total_batches_processed"] += 1
                    self.current_activations.clear()
                    self.first_nonfinite_layer = None
                    self.layer_nonfinite_type = None

                    xA = batch["xA"].to(self.device, non_blocking=True)
                    xB = batch["xB"].to(self.device, non_blocking=True)
                    label = batch["label"].to(self.device, non_blocking=True)
                    sample_ids = batch.get("sample_id", [])
                    event_ids = batch.get("event_id", [])
                    if torch.is_tensor(event_ids):
                        event_ids = event_ids.cpu().tolist()

                    # Integrity check on quarantined events
                    for eid in event_ids:
                        if int(eid) in FORBIDDEN_TTA_VAL_EVENTS or int(eid) in FORBIDDEN_SEALED_TEST_EVENTS:
                            raise RuntimeError(f"CRITICAL VIOLATION: Quarantined event {eid} accessed!")

                    # 1. Inputs finiteness
                    xA_fin = torch.isfinite(xA).all().item()
                    xB_fin = torch.isfinite(xB).all().item()
                    inputs_finite = (xA_fin and xB_fin)

                    # 2. Check BN buffers before forward
                    bn_pre_fin, bn_corrupted_pre = self.audit_bn_buffers()

                    # 3. Forward pass
                    if self.precision_mode == "fp16":
                        with autocast():
                            logits = self.model(xA, xB)
                            loss, comps = self.loss_fn(logits, label, return_components=True)
                            scaled_loss = loss / self.accumulation_steps
                    else:
                        logits = self.model(xA, xB)
                        loss, comps = self.loss_fn(logits, label, return_components=True)
                        scaled_loss = loss / self.accumulation_steps

                    # 4. Check BN buffers after forward
                    bn_post_fin, bn_corrupted_post = self.audit_bn_buffers()

                    # 5. Check Logits and Loss finiteness
                    logits_fin = torch.isfinite(logits).all().item()
                    loss_fin = torch.isfinite(loss).all().item()

                    # Find max activation across blocks
                    max_act_layer = "none"
                    max_act_val = 0.0
                    for k, v in self.current_activations.items():
                        if not math.isnan(v) and v > max_act_val:
                            max_act_val = v
                            max_act_layer = k

                    # 6. Backward pass
                    grads_finite = True
                    if self.precision_mode == "fp16":
                        self.scaler.scale(scaled_loss).backward()
                        # Check scaled gradients
                        for p in self.model.parameters():
                            if p.grad is not None:
                                if not torch.isfinite(p.grad).all().item():
                                    grads_finite = False
                                    break
                    else:
                        scaled_loss.backward()
                        for p in self.model.parameters():
                            if p.grad is not None:
                                if not torch.isfinite(p.grad).all().item():
                                    grads_finite = False
                                    break

                    scaler_scale = self.scaler.get_scale() if self.scaler else 1.0

                    # 7. Write per-batch log
                    s_ids_str = "|".join(str(s) for s in sample_ids)
                    e_ids_str = "|".join(str(e) for e in event_ids)
                    new_corrupted_bn = [b for b in bn_corrupted_post if b not in bn_corrupted_pre]
                    corrupted_str = "|".join(new_corrupted_bn) if new_corrupted_bn else ("none" if bn_post_fin else "|".join(bn_corrupted_post))

                    self.batch_writer.writerow([
                        epoch, batch_idx + 1, self.global_step, s_ids_str, e_ids_str,
                        inputs_finite, max_act_layer, f"{max_act_val:.2f}",
                        self.first_nonfinite_layer or "none", logits_fin, loss_fin,
                        f"{comps['loss_total']:.6f}" if loss_fin else "nan",
                        f"{comps['loss_ce']:.6f}" if loss_fin else "nan",
                        f"{comps['loss_dice']:.6f}" if loss_fin else "nan",
                        grads_finite, scaler_scale,
                        bn_pre_fin, bn_post_fin, corrupted_str
                    ])
                    self.batch_csv_file.flush()

                    # Step optimizer at accumulation boundary
                    is_accum = ((batch_idx + 1) % self.accumulation_steps == 0)
                    is_last = (batch_idx + 1 == total_batches)
                    if is_accum or is_last:
                        if self.precision_mode == "fp16":
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                    # 8. Check for Failure
                    has_failure = (
                        not inputs_finite or
                        not logits_fin or
                        not loss_fin or
                        not bn_post_fin or
                        not grads_finite or
                        self.first_nonfinite_layer is not None
                    )

                    if has_failure:
                        stop_diagnostic = True
                        fail_info = {
                            "epoch": epoch,
                            "batch_idx": batch_idx + 1,
                            "global_step": self.global_step,
                            "sample_ids": [str(s) for s in sample_ids],
                            "event_ids": [int(e) for e in event_ids],
                            "first_nonfinite_layer": self.first_nonfinite_layer,
                            "nonfinite_type": self.layer_nonfinite_type,
                            "inputs_finite": inputs_finite,
                            "logits_finite": logits_fin,
                            "loss_finite": loss_fin,
                            "grads_finite": grads_finite,
                            "bn_pre_finite": bn_pre_fin,
                            "bn_post_finite": bn_post_fin,
                            "corrupted_bn_buffers": new_corrupted_bn or bn_corrupted_post,
                            "max_activation_layer": max_act_layer,
                            "max_activation_value": max_act_val,
                            "scaler_scale": scaler_scale,
                        }
                        self.diagnostic_summary["failure_detected"] = True
                        self.diagnostic_summary["failure_details"] = fail_info
                        print("\n" + "!" * 70)
                        print(f"[FAIL-FAST TRIGGERED] First nonfinite anomaly detected in {self.precision_mode.upper()} mode!")
                        print(f"  Epoch: {epoch}, Batch: {batch_idx + 1}, Step: {self.global_step}")
                        print(f"  First Nonfinite Layer: {self.first_nonfinite_layer} ({self.layer_nonfinite_type})")
                        print(f"  BN Buffers Post-Forward Finite: {bn_post_fin}")
                        print(f"  Corrupted BN Buffers: {new_corrupted_bn or bn_corrupted_post}")
                        print(f"  Logits Finite: {logits_fin} | Loss Finite: {loss_fin} | Grads Finite: {grads_finite}")
                        print("!" * 70 + "\n")
                        break

                    if (batch_idx + 1) % 50 == 0:
                        print(
                            f"Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{total_batches:04d} | "
                            f"Max Act: {max_act_val:.1f} ({max_act_layer}) | Scale: {scaler_scale:.0f} | "
                            f"Loss: {comps['loss_total']:.4f} | BN Fin: {bn_post_fin}"
                        )
        finally:
            # Cleanup and guaranteed summary output
            for h in self.hooks:
                h.remove()
            self.batch_csv_file.close()

            summary_path = self.output_dir / f"diagnostic_summary_{self.precision_mode}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.diagnostic_summary, f, indent=2)
            print(f"[DIAGNOSTIC] Saved summary to {summary_path}")
        return self.diagnostic_summary


def main():
    parser = argparse.ArgumentParser(description="Instrumented AMP / BatchNorm Diagnostic Replay")
    parser.add_argument("--config", type=str, required=True, help="Path to source model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Epoch 20 best checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save diagnostic outputs")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Precision mode")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to execute on")
    parser.add_argument("--max_epoch", type=int, default=25, help="Maximum epoch to run until")
    args = parser.parse_args()

    runner = InstrumentedDiagnosticRunner(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        precision_mode=args.precision,
        device=args.device,
        max_epoch=args.max_epoch,
    )
    summary = runner.run()
    sys.exit(0 if not summary["failure_detected"] else 2)


if __name__ == "__main__":
    main()

