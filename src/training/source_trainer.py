"""Source model training engine for controlled SNUNet training.

Implements:
1. PyTorch AMP mixed precision (autocast + GradScaler).
2. Gradient accumulation for effective batch size 32.
3. ControlledSourceLoss (unweighted CE + Soft Dice).
4. Validation pass computing Event-Macro Flood F1, Worst-Event Flood F1, Recall, and IoU.
5. Deterministic tie-breaking for checkpoint selection (Section 7.2).
6. Full checkpoint save and resume capability.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.evaluation.losses import ControlledSourceLoss
from src.evaluation.metrics import EventMetricAccumulator



class NumericalInstabilityError(RuntimeError):
    """Raised immediately when any nonfinite value is detected in training/validation."""

    def __init__(
        self,
        message: str,
        tensor_name: str = "",
        stage: str = "",
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        sample_ids: Optional[List[Any]] = None,
        event_ids: Optional[List[Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.tensor_name = tensor_name
        self.stage = stage
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.global_step = global_step
        self.sample_ids = sample_ids or []
        self.event_ids = event_ids or []
        self.details = details or {}


class PredictiveCollapseError(RuntimeError):
    """Raised when predictive collapse is detected on known-positive validation set."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        consecutive_zero_epochs: int = 0,
        warmup_epochs: int = 5,
        patience: int = 3,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.epoch = epoch
        self.consecutive_zero_epochs = consecutive_zero_epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.details = details or {}


class PredictiveCollapseDetector:
    """Predictive collapse detector for known-positive validation set (Protocol v2 Section 5.2)."""

    def __init__(self, warmup_epochs: int = 5, patience: int = 3):
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.consecutive_zero_epochs = 0
        self.triggered = False

    def update(self, epoch: int, pred_flood_pixels: int) -> bool:
        """Updates collapse tracking.

        Returns True if predictive collapse triggered, False otherwise.
        """
        if pred_flood_pixels == 0:
            if epoch > self.warmup_epochs:
                self.consecutive_zero_epochs += 1
            else:
                self.consecutive_zero_epochs = 0
        else:
            self.consecutive_zero_epochs = 0

        if epoch > self.warmup_epochs and self.consecutive_zero_epochs >= self.patience:
            self.triggered = True
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_epochs": self.warmup_epochs,
            "patience": self.patience,
            "consecutive_zero_epochs": self.consecutive_zero_epochs,
            "triggered": self.triggered,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
        self.patience = int(state.get("patience", self.patience))
        self.consecutive_zero_epochs = int(state.get("consecutive_zero_epochs", 0))
        self.triggered = bool(state.get("triggered", False))


def is_better_checkpoint(
    cand_metrics: Dict[str, Any],
    cand_epoch: int,
    best_metrics: Optional[Dict[str, Any]],
    best_epoch: Optional[int],
    tol: float = 1e-5,
) -> bool:
    """Deterministic epoch tie-breaking rule (Section 7.2).

    1. Higher event-macro Flood F1;
    2. If tied to within 1e-5, higher worst-event Flood F1;
    3. If still tied, higher event-macro Flood Recall;
    4. If still tied, earlier epoch (smaller integer).
    """
    if best_metrics is None or best_epoch is None:
        return True

    cand_f1 = float(cand_metrics["macro_f1"])
    best_f1 = float(best_metrics["macro_f1"])
    if cand_f1 - best_f1 > tol:
        return True
    if best_f1 - cand_f1 > tol:
        return False

    # Tied on Macro F1: check Worst Event F1
    cand_worst = float(cand_metrics.get("worst_event_f1", {}).get("f1", 0.0))
    best_worst = float(best_metrics.get("worst_event_f1", {}).get("f1", 0.0))
    if cand_worst - best_worst > tol:
        return True
    if best_worst - cand_worst > tol:
        return False

    # Tied on Worst F1: check Macro Recall
    cand_rec = float(cand_metrics["macro_recall"])
    best_rec = float(best_metrics["macro_recall"])
    if cand_rec - best_rec > tol:
        return True
    if best_rec - cand_rec > tol:
        return False

    # Tied on Recall: smaller epoch wins
    return cand_epoch < best_epoch



class SourceTrainer:
    """Trainer for controlled source change-detection models."""

    def __init__(
        self,
        model: nn.Module,
        output_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        mixed_precision: bool = True,
        effective_batch_size: int = 32,
        physical_microbatch_size: int = 8,
        loss_fn: Optional[nn.Module] = None,
        seed: int = 42,
        optimizer_name: str = "Adam",
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.effective_batch_size = int(effective_batch_size)
        self.physical_microbatch_size = int(physical_microbatch_size)
        assert self.effective_batch_size % self.physical_microbatch_size == 0, (
            f"effective_batch_size ({self.effective_batch_size}) must be divisible by "
            f"physical_microbatch_size ({self.physical_microbatch_size})"
        )
        self.accumulation_steps = self.effective_batch_size // self.physical_microbatch_size

        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.scaler = GradScaler() if self.mixed_precision else None

        self.optimizer_name = str(optimizer_name)
        if self.optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(learning_rate),
                betas=betas,
                weight_decay=float(weight_decay),
            )
        elif self.optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(learning_rate),
                betas=betas,
                weight_decay=float(weight_decay),
            )
        else:
            raise ValueError(f"Unsupported optimizer: '{self.optimizer_name}'")

        self.lr_scheduler = None
        self.loss_fn = loss_fn if loss_fn is not None else ControlledSourceLoss()
        self.seed = int(seed)

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.best_epoch: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self.stop_requested = False

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> "SourceTrainer":
        """Instantiates SourceTrainer strictly from the governing configuration dictionary."""
        opt_cfg = config.get("optimization_protocol", {})
        batch_cfg = config.get("batch_size_policy", {})

        opt_name = str(opt_cfg.get("optimizer", "Adam"))
        lr = float(opt_cfg.get("learning_rate", 1e-3))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        mp = bool(opt_cfg.get("mixed_precision", True))
        if (
            str(opt_cfg.get("precision", "")).lower() == "fp32"
            or opt_cfg.get("autocast") is False
            or opt_cfg.get("grad_scaler") is False
        ):
            mp = False

        eff_bs = int(batch_cfg.get("effective_batch_size", 32))
        micro_bs = int(batch_cfg.get("physical_microbatch_size", 8))

        if seed is None:
            seeds = opt_cfg.get("seeds", [42])
            seed = int(seeds[0]) if seeds else 42

        return cls(
            model=model,
            output_dir=output_dir,
            device=device,
            learning_rate=lr,
            weight_decay=wd,
            betas=betas,
            mixed_precision=mp,
            effective_batch_size=eff_bs,
            physical_microbatch_size=micro_bs,
            loss_fn=loss_fn,
            seed=seed,
            optimizer_name=opt_name,
        )

    def train_steps(
        self,
        train_loader: DataLoader,
        num_optimizer_steps: int,
    ) -> List[Dict[str, float]]:
        """Executes a specific number of optimizer update steps."""
        self.model.train()
        step_metrics = []
        loader_iter = iter(train_loader)

        # Advance iterator to match global_step if resuming mid-cycle
        num_batches_in_loader = len(train_loader)
        if num_batches_in_loader > 0:
            batches_to_skip = (self.global_step * self.accumulation_steps) % num_batches_in_loader
            for _ in range(batches_to_skip):
                try:
                    next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    next(loader_iter)

        for _ in range(num_optimizer_steps):
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss_total = 0.0
            accum_loss_ce = 0.0
            accum_loss_dice = 0.0

            for acc_idx in range(self.accumulation_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)

                xA = batch["xA"].to(self.device, non_blocking=True)
                xB = batch["xB"].to(self.device, non_blocking=True)
                label = batch["label"].to(self.device, non_blocking=True)

                if not torch.isfinite(xA).all():
                    raise NumericalInstabilityError(
                        "Nonfinite values detected in input tensor xA",
                        tensor_name="xA",
                        stage="train_input",
                        global_step=self.global_step,
                    )
                if not torch.isfinite(xB).all():
                    raise NumericalInstabilityError(
                        "Nonfinite values detected in input tensor xB",
                        tensor_name="xB",
                        stage="train_input",
                        global_step=self.global_step,
                    )

                if self.mixed_precision:
                    with autocast():
                        logits = self.model(xA, xB)
                        if not torch.isfinite(logits).all():
                            raise NumericalInstabilityError(
                                "Nonfinite values detected in forward logits",
                                tensor_name="logits",
                                stage="train_forward_logits",
                                global_step=self.global_step,
                            )
                        loss, components = self.loss_fn(logits, label, return_components=True)
                        if not torch.isfinite(loss):
                            raise NumericalInstabilityError(
                                "Nonfinite loss detected pre-backward",
                                tensor_name="loss",
                                stage="train_loss",
                                global_step=self.global_step,
                            )
                        scaled_loss = loss / self.accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                else:
                    logits = self.model(xA, xB)
                    if not torch.isfinite(logits).all():
                        raise NumericalInstabilityError(
                            "Nonfinite values detected in forward logits",
                            tensor_name="logits",
                            stage="train_forward_logits",
                            global_step=self.global_step,
                        )
                    loss, components = self.loss_fn(logits, label, return_components=True)
                    if not torch.isfinite(loss):
                        raise NumericalInstabilityError(
                            "Nonfinite loss detected pre-backward",
                            tensor_name="loss",
                            stage="train_loss",
                            global_step=self.global_step,
                        )
                    scaled_loss = loss / self.accumulation_steps
                    scaled_loss.backward()

                accum_loss_total += components["loss_total"] / self.accumulation_steps
                accum_loss_ce += components["loss_ce"] / self.accumulation_steps
                accum_loss_dice += components["loss_dice"] / self.accumulation_steps

            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
            for p_name, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise NumericalInstabilityError(
                        f"Nonfinite gradients detected in parameter {p_name} before optimizer step.",
                        tensor_name=f"grad_{p_name}",
                        stage="train_gradients",
                        global_step=self.global_step,
                    )

            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            for p_name, p in self.model.named_parameters():
                if not torch.isfinite(p).all():
                    raise NumericalInstabilityError(
                        f"Nonfinite parameter values detected after optimizer step in {p_name}.",
                        tensor_name=f"param_{p_name}",
                        stage="train_step_params",
                        global_step=self.global_step,
                    )

            self.global_step += 1
            step_record = {
                "step": self.global_step,
                "loss_total": float(accum_loss_total),
                "loss_ce": float(accum_loss_ce),
                "loss_dice": float(accum_loss_dice),
            }
            step_metrics.append(step_record)

        return step_metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int = 50,
        epoch_idx: Optional[int] = None,
        tracker: Optional[Any] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes one complete training epoch with gradient accumulation.

        Accumulates gradients across physical microbatches to match effective batch size.
        Returns dictionary of training metrics for the epoch.
        """
        self.model.train()
        start_time = time.time()
        if epoch_idx is not None:
            self.current_epoch = int(epoch_idx)
        else:
            self.current_epoch += 1

        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_accum_steps = 0

        accum_loss_total = 0.0
        accum_loss_ce = 0.0
        accum_loss_dice = 0.0
        samples_observed = 0
        batches_observed = 0

        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        if max_batches is not None and max_batches > 0:
            total_batches = min(total_batches, int(max_batches))

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= total_batches:
                break
            if getattr(self, "stop_requested", False):
                print(f"\n[INTERRUPT] Stop requested; breaking epoch {self.current_epoch} at batch {batch_idx}/{total_batches}.")
                break

            xA = batch["xA"].to(self.device, non_blocking=True)
            xB = batch["xB"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            sample_ids = batch.get("sample_id", [])
            event_ids = batch.get("event_id", [])
            if torch.is_tensor(event_ids):
                event_ids = event_ids.cpu().tolist()

            batch_samples = xA.size(0)
            samples_observed += batch_samples
            batches_observed += 1

            if tracker is not None and hasattr(tracker, "record_batch"):
                tracker.record_batch(batch)

            if not torch.isfinite(xA).all():
                raise NumericalInstabilityError(
                    "Nonfinite values detected in input tensor xA",
                    tensor_name="xA",
                    stage="train_input",
                    epoch=self.current_epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    sample_ids=sample_ids,
                    event_ids=event_ids,
                )
            if not torch.isfinite(xB).all():
                raise NumericalInstabilityError(
                    "Nonfinite values detected in input tensor xB",
                    tensor_name="xB",
                    stage="train_input",
                    epoch=self.current_epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    sample_ids=sample_ids,
                    event_ids=event_ids,
                )

            if self.mixed_precision:
                with autocast():
                    logits = self.model(xA, xB)
                    if not torch.isfinite(logits).all():
                        raise NumericalInstabilityError(
                            "Nonfinite values detected in forward logits",
                            tensor_name="logits",
                            stage="train_forward_logits",
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )
                    loss, components = self.loss_fn(logits, label, return_components=True)
                    if not torch.isfinite(loss):
                        raise NumericalInstabilityError(
                            "Nonfinite loss detected pre-backward",
                            tensor_name="loss",
                            stage="train_loss",
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )
                    scaled_loss = loss / self.accumulation_steps
                self.scaler.scale(scaled_loss).backward()
            else:
                logits = self.model(xA, xB)
                if not torch.isfinite(logits).all():
                    raise NumericalInstabilityError(
                        "Nonfinite values detected in forward logits",
                        tensor_name="logits",
                        stage="train_forward_logits",
                        epoch=self.current_epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        sample_ids=sample_ids,
                        event_ids=event_ids,
                    )
                loss, components = self.loss_fn(logits, label, return_components=True)
                if not torch.isfinite(loss):
                    raise NumericalInstabilityError(
                        "Nonfinite loss detected pre-backward",
                        tensor_name="loss",
                        stage="train_loss",
                        epoch=self.current_epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        sample_ids=sample_ids,
                        event_ids=event_ids,
                    )
                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

            accum_loss_total += components["loss_total"]
            accum_loss_ce += components["loss_ce"]
            accum_loss_dice += components["loss_dice"]

            is_accum_boundary = ((batch_idx + 1) % self.accumulation_steps == 0)
            is_last_batch = (batch_idx + 1 == total_batches)

            if is_accum_boundary or is_last_batch:
                window_microbatches = (
                    (batch_idx % self.accumulation_steps) + 1
                    if is_last_batch and not is_accum_boundary
                    else self.accumulation_steps
                )

                if is_last_batch and not is_accum_boundary and window_microbatches < self.accumulation_steps:
                    scale_factor = self.accumulation_steps / window_microbatches
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                for p_name, p in self.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise NumericalInstabilityError(
                            f"Nonfinite gradients detected in parameter {p_name} before optimizer step.",
                            tensor_name=f"grad_{p_name}",
                            stage="train_gradients",
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                for p_name, p in self.model.named_parameters():
                    if not torch.isfinite(p).all():
                        raise NumericalInstabilityError(
                            f"Nonfinite parameter values detected after optimizer step in {p_name}.",
                            tensor_name=f"param_{p_name}",
                            stage="train_step_params",
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                step_loss = accum_loss_total / window_microbatches
                step_ce = accum_loss_ce / window_microbatches
                step_dice = accum_loss_dice / window_microbatches

                total_loss += step_loss
                total_ce += step_ce
                total_dice += step_dice
                num_accum_steps += 1

                accum_loss_total = 0.0
                accum_loss_ce = 0.0
                accum_loss_dice = 0.0

                if log_interval > 0 and (num_accum_steps % log_interval == 0 or is_last_batch):
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {self.current_epoch:03d} | Batch {batch_idx+1:04d}/{total_batches:04d} | "
                        f"Step {self.global_step:05d} | Loss: {step_loss:.4f} (CE: {step_ce:.4f}, Dice: {step_dice:.4f}) | "
                        f"LR: {lr:.6f}"
                    )

        # End of epoch: audit BatchNorm running buffers and all floating point buffers
        for b_name, buf in self.model.named_buffers():
            if buf.is_floating_point() and not torch.isfinite(buf).all():
                raise NumericalInstabilityError(
                    f"Nonfinite values detected in buffer {b_name} at epoch end.",
                    tensor_name=f"buffer_{b_name}",
                    stage="train_epoch_bn_audit",
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                )

        epoch_duration = time.time() - start_time
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )

        return {
            "epoch": self.current_epoch,
            "loss_total": float(total_loss / max(num_accum_steps, 1)),
            "loss_ce": float(total_ce / max(num_accum_steps, 1)),
            "loss_dice": float(total_dice / max(num_accum_steps, 1)),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "optimizer_steps": self.global_step,
            "samples_observed": samples_observed,
            "batches_observed": batches_observed,
            "epoch_duration_sec": float(epoch_duration),
            "peak_gpu_memory_mb": float(peak_mem_mb),
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None,
        compute_loss: bool = True,
        tracker: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluates model on validation dataset across events."""
        self.model.eval()
        accumulator = EventMetricAccumulator()
        batch_count = 0
        total_val_loss = 0.0
        total_val_ce = 0.0
        total_val_dice = 0.0
        val_loss_steps = 0

        for batch in val_loader:
            xA = batch["xA"].to(self.device, non_blocking=True)
            xB = batch["xB"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
            sample_ids = batch.get("sample_id", [])
            event_ids = batch.get("event_id", [])
            if torch.is_tensor(event_ids):
                event_ids = event_ids.cpu().tolist()

            if tracker is not None and hasattr(tracker, "record_batch"):
                tracker.record_batch(batch)

            if not torch.isfinite(xA).all():
                raise NumericalInstabilityError(
                    "Nonfinite values detected in validation input tensor xA",
                    tensor_name="xA",
                    stage="val_input",
                    epoch=self.current_epoch,
                    batch_idx=batch_count,
                    sample_ids=sample_ids,
                    event_ids=event_ids,
                )
            if not torch.isfinite(xB).all():
                raise NumericalInstabilityError(
                    "Nonfinite values detected in validation input tensor xB",
                    tensor_name="xB",
                    stage="val_input",
                    epoch=self.current_epoch,
                    batch_idx=batch_count,
                    sample_ids=sample_ids,
                    event_ids=event_ids,
                )

            if self.mixed_precision:
                with autocast():
                    logits = self.model(xA, xB)
                    if not torch.isfinite(logits).all():
                        raise NumericalInstabilityError(
                            "Nonfinite values detected in validation logits",
                            tensor_name="logits",
                            stage="val_logits",
                            epoch=self.current_epoch,
                            batch_idx=batch_count,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )
                    if compute_loss:
                        loss, components = self.loss_fn(logits, label, return_components=True)
                        if not torch.isfinite(loss):
                            raise NumericalInstabilityError(
                                "Nonfinite validation loss detected",
                                tensor_name="loss",
                                stage="val_loss",
                                epoch=self.current_epoch,
                                batch_idx=batch_count,
                                sample_ids=sample_ids,
                                event_ids=event_ids,
                            )
            else:
                logits = self.model(xA, xB)
                if not torch.isfinite(logits).all():
                    raise NumericalInstabilityError(
                        "Nonfinite values detected in validation logits",
                        tensor_name="logits",
                        stage="val_logits",
                        epoch=self.current_epoch,
                        batch_idx=batch_count,
                        sample_ids=sample_ids,
                        event_ids=event_ids,
                    )
                if compute_loss:
                    loss, components = self.loss_fn(logits, label, return_components=True)
                    if not torch.isfinite(loss):
                        raise NumericalInstabilityError(
                            "Nonfinite validation loss detected",
                            tensor_name="loss",
                            stage="val_loss",
                            epoch=self.current_epoch,
                            batch_idx=batch_count,
                            sample_ids=sample_ids,
                            event_ids=event_ids,
                        )

            if compute_loss:
                total_val_loss += components["loss_total"]
                total_val_ce += components["loss_ce"]
                total_val_dice += components["loss_dice"]
                val_loss_steps += 1

            pred_class = torch.argmax(logits, dim=1)
            accumulator.update(
                event_ids=event_ids,
                pred_classes=pred_class,
                labels=label,
                valid_masks=valid_mask,
            )

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        summary = accumulator.compute_summary()
        if compute_loss and val_loss_steps > 0:
            summary["loss_total"] = float(total_val_loss / val_loss_steps)
            summary["loss_ce"] = float(total_val_ce / val_loss_steps)
            summary["loss_dice"] = float(total_val_dice / val_loss_steps)
        event_table = accumulator.compute_event_table()
        return summary, event_table

    def save_checkpoint(
        self,
        filename: str,
        metrics: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        config_hash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves complete trainer checkpoint with atomic write, finiteness verification, and provenance."""
        # 1. Audit model parameters and buffers for finiteness before saving
        for p_name, p in self.model.named_parameters():
            if not torch.isfinite(p).all():
                raise NumericalInstabilityError(
                    f"Cannot save checkpoint: nonfinite values in parameter {p_name}.",
                    tensor_name=f"param_{p_name}",
                    stage="save_checkpoint_param_audit",
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                )
        for b_name, buf in self.model.named_buffers():
            if buf.is_floating_point() and not torch.isfinite(buf).all():
                raise NumericalInstabilityError(
                    f"Cannot save checkpoint: nonfinite values in buffer {b_name}.",
                    tensor_name=f"buffer_{b_name}",
                    stage="save_checkpoint_buffer_audit",
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                )

        ckpt_path = self.output_dir / filename
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if (torch.cuda.is_available() and self.device.type == "cuda")
            else None
        )
        np_rng_state = np.random.get_state()

        from src.evaluation.run_manifest import get_git_commit, is_git_dirty
        git_commit = get_git_commit(self.output_dir)
        git_dirty = is_git_dirty(self.output_dir)

        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_metrics": self.best_metrics,
            "best_epoch": self.best_epoch,
            "current_metrics": metrics,
            "seed": self.seed,
            "rng_state": rng_state,
            "cuda_rng_state": cuda_rng_state,
            "np_rng_state": np_rng_state,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "config_path": str(config_path) if config_path else None,
            "config_hash": str(config_hash) if config_hash else None,
            "extra_metadata": extra_metadata or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Atomic write via temporary file
        tmp_path = self.output_dir / f"{filename}.tmp.{os.getpid()}_{time.time_ns()}"
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, ckpt_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return ckpt_path

    def resume_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Resumes trainer state from checkpoint, optionally restoring RNG states."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

        try:
            state = torch.load(ckpt_p, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_p, map_location=self.device)

        # Refuse corrupted checkpoints
        if state.get("is_corrupted") or state.get("status") in ("FAILED_NUMERICAL", "FAILED_COLLAPSE", "FAILED"):
            raise ValueError(f"Refusing to resume from checkpoint flagged as failed/corrupted: {ckpt_p}")

        self.model.load_state_dict(state["model_state_dict"])

        # Audit loaded parameters and buffers for finiteness
        for p_name, p in self.model.named_parameters():
            if not torch.isfinite(p).all():
                raise ValueError(f"Corrupted checkpoint: parameter {p_name} contains nonfinite values")
        for b_name, buf in self.model.named_buffers():
            if buf.is_floating_point() and not torch.isfinite(buf).all():
                raise ValueError(f"Corrupted checkpoint: buffer {b_name} contains nonfinite values")

        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scaler and state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.current_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_metrics = state.get("best_metrics")
        self.best_epoch = state.get("best_epoch")

        if restore_rng:
            if "rng_state" in state and state["rng_state"] is not None:
                try:
                    torch.set_rng_state(state["rng_state"].cpu() if hasattr(state["rng_state"], "cpu") else state["rng_state"])
                except Exception as e:
                    print(f"Warning: could not restore torch RNG state: {e}")
            if "cuda_rng_state" in state and state["cuda_rng_state"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in state["cuda_rng_state"]])
                except Exception as e:
                    print(f"Warning: could not restore CUDA RNG state: {e}")
            if "np_rng_state" in state and state["np_rng_state"] is not None:
                try:
                    np.random.set_state(state["np_rng_state"])
                except Exception as e:
                    print(f"Warning: could not restore numpy RNG state: {e}")

        print(f"Resumed trainer state from {ckpt_p} at epoch={self.current_epoch}, step={self.global_step}")
        return state
