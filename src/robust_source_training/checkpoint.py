"""Atomic checkpoint saving, loading, and strict state-equality resume verification."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.robust_source_training.contracts import ContractViolationError


def save_rst_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    run_id: str,
    config_hash: str,
    fold_hash: str,
    train_event_ids: List[int],
    evaluation_event_ids: List[int],
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Path:
    """Atomically serializes a complete RST checkpoint with full provenance and RNG state."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")

    rng_states = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else "NA",
    }

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else "NA",
        "scaler_state_dict": scaler.state_dict() if scaler is not None else "NA",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "seed": int(seed),
        "rng_states": rng_states,
        "metadata": {
            "run_id": str(run_id),
            "config_hash": str(config_hash),
            "fold_hash": str(fold_hash),
            "train_event_ids": sorted(list(train_event_ids)),
            "evaluation_event_ids": sorted(list(evaluation_event_ids)),
        },
    }

    torch.save(checkpoint_dict, temp_path)
    temp_path.replace(p)
    return p


def load_rst_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    expected_config_hash: Optional[str] = None,
    expected_fold_hash: Optional[str] = None,
    expected_train_events: Optional[List[int]] = None,
    expected_eval_events: Optional[List[int]] = None,
    restore_rng: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads and strictly verifies an RST checkpoint against expected metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")

    ckpt = torch.load(p, map_location=device)
    meta = ckpt.get("metadata", {})

    # Strict metadata assertions
    if expected_config_hash and meta.get("config_hash") != expected_config_hash:
        raise ContractViolationError(
            f"Checkpoint config hash mismatch! Expected {expected_config_hash}, got {meta.get('config_hash')}"
        )

    if expected_fold_hash and meta.get("fold_hash") != expected_fold_hash:
        raise ContractViolationError(
            f"Checkpoint fold hash mismatch! Expected {expected_fold_hash}, got {meta.get('fold_hash')}"
        )

    if expected_train_events is not None:
        ckpt_train = set(meta.get("train_event_ids", []))
        if ckpt_train != set(expected_train_events):
            raise ContractViolationError(
                f"Checkpoint train event list mismatch! Discrepancy: {ckpt_train ^ set(expected_train_events)}"
            )

    if expected_eval_events is not None:
        ckpt_eval = set(meta.get("evaluation_event_ids", []))
        if ckpt_eval != set(expected_eval_events):
            raise ContractViolationError(
                f"Checkpoint eval event list mismatch! Discrepancy: {ckpt_eval ^ set(expected_eval_events)}"
            )

    # Load model and optimizer weights
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Restore RNG if requested
    if restore_rng and "rng_states" in ckpt:
        rngs = ckpt["rng_states"]
        if "python_rng" in rngs:
            random.setstate(rngs["python_rng"])
        if "numpy_rng" in rngs:
            np.random.set_state(rngs["numpy_rng"])
        if "torch_cpu_rng" in rngs:
            cpu_rng = rngs["torch_cpu_rng"]
            if isinstance(cpu_rng, torch.Tensor):
                cpu_rng = cpu_rng.cpu()
            torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and rngs.get("torch_cuda_rng") != "NA":
            cuda_rng = rngs["torch_cuda_rng"]
            if isinstance(cuda_rng, list):
                cuda_rng = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_rng]
            torch.cuda.set_rng_state_all(cuda_rng)

    return ckpt


def assert_states_equal(model1: nn.Module, model2: nn.Module) -> None:
    """Verifies that two models have exact bit-level parameter and buffer equality."""
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    if set(sd1.keys()) != set(sd2.keys()):
        raise AssertionError("Model state dict keys differ!")

    for k in sd1:
        t1 = sd1[k].cpu()
        t2 = sd2[k].cpu()
        if not torch.equal(t1, t2):
            diff = (t1 - t2).abs().max().item()
            raise AssertionError(f"Parameter/buffer mismatch on key '{k}', max diff: {diff}")

