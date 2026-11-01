"""Production controlled source-model training launcher for SNUNet (Model A and Model B).

Executes full-epoch training under the frozen Week 0/1 research protocol:
- Optimizer: Adam (lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
- Scheduler: None
- Effective batch size: 32 (physical microbatch 8, accumulation 4)
- Mixed precision: PyTorch AMP (autocast + GradScaler)
- Loss: ControlledSourceLoss (unweighted CE + Soft Dice with ignore_index=3)
- Dataset: KuroSourceDataset with 22 inner-train and 5 inner-val events
- Early stopping: patience 20 on inner-validation event-macro Flood F1
- Maximum epochs: 150
- Full provenance, access tracking, CSV history logging, and deterministic resumption.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
from src.training.source_trainer import (
    SourceTrainer,
    is_better_checkpoint,
    NumericalInstabilityError,
    PredictiveCollapseError,
    PredictiveCollapseDetector,
)
from src.evaluation.run_manifest import RunManifest, compute_file_sha256, get_git_commit, is_git_dirty

# Frozen Protocol References
LOCK_MANIFEST_DEFAULT_PATH = REPO_ROOT / "configs/source/source_training_v2_lock.json"
FROZEN_SPLIT_HASH = "7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3"
FROZEN_SLOPE_STATS_HASH = "bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04"
FROZEN_MODEL_A_V1_HASH = "9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc"
FROZEN_MODEL_B_V1_HASH = "f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec"

# Backward compatibility aliases
FROZEN_MODEL_A_HASH = FROZEN_MODEL_A_V1_HASH
FROZEN_MODEL_B_HASH = FROZEN_MODEL_B_V1_HASH

FORBIDDEN_TTA_VAL_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}


class TeeLogger:
    """Duplicates stdout/stderr to a persistent log file and terminal."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


class AccessTracker:
    """Audits data loading at runtime to guarantee zero quarantined data access."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.train_samples: Set[str] = set()
        self.train_events: Set[int] = set()
        self.val_samples: Set[str] = set()
        self.val_events: Set[int] = set()
        self.forbidden_accessed: Set[int] = set()
        self.current_mode = "train"

    def set_mode(self, mode: str):
        assert mode in ("train", "val")
        self.current_mode = mode

    def record_batch(self, batch: Dict[str, Any]):
        sample_ids = batch.get("sample_id", [])
        event_ids = batch.get("event_id", [])
        if torch.is_tensor(event_ids):
            event_ids = event_ids.cpu().tolist()

        for s, e in zip(sample_ids, event_ids):
            eid = int(e)
            sid = str(s)
            if eid in FORBIDDEN_TTA_VAL_EVENTS or eid in FORBIDDEN_SEALED_TEST_EVENTS:
                self.forbidden_accessed.add(eid)
                raise RuntimeError(
                    f"CRITICAL RESEARCH INTEGRITY VIOLATION: Access attempted on forbidden event {eid} (sample {sid})!"
                )

            if self.current_mode == "train":
                self.train_samples.add(sid)
                self.train_events.add(eid)
            else:
                self.val_samples.add(sid)
                self.val_events.add(eid)

    def to_dict(self) -> Dict[str, Any]:
        all_accessed = sorted(list(self.train_events | self.val_events))
        return {
            "model_type": self.model_type,
            "train_unique_samples_count": len(self.train_samples),
            "train_events_accessed": sorted(list(self.train_events)),
            "train_events_count": len(self.train_events),
            "val_unique_samples_count": len(self.val_samples),
            "val_events_accessed": sorted(list(self.val_events)),
            "val_events_count": len(self.val_events),
            "all_accessed_events": all_accessed,
            "quarantined_events_accessed": sorted(list(self.forbidden_accessed)),
            "sample_intersection": sorted(list(self.train_samples & self.val_samples)),
            "event_intersection": sorted(list(self.train_events & self.val_events)),
            "assertions": {
                "no_tta_val_accessed": len(self.forbidden_accessed & FORBIDDEN_TTA_VAL_EVENTS) == 0,
                "no_sealed_test_accessed": len(self.forbidden_accessed & FORBIDDEN_SEALED_TEST_EVENTS) == 0,
                "train_val_events_disjoint": len(self.train_events & self.val_events) == 0,
                "train_val_samples_disjoint": len(self.train_samples & self.val_samples) == 0,
            },
        }

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def generate_curves(output_dir: Path):
    """Generates loss and F1 training curves from saved CSV history files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: matplotlib not available, skipping curve plot generation.")
        return

    train_csv = output_dir / "train_history.csv"
    val_csv = output_dir / "validation_history.csv"
    if not train_csv.exists() or not val_csv.exists():
        return

    train_epochs, train_losses = [], []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            train_epochs.append(int(r["epoch"]))
            train_losses.append(float(r["loss_total"]))

    val_epochs, val_losses, macro_f1s, worst_f1s = [], [], [], []
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val_epochs.append(int(r["epoch"]))
            val_losses.append(float(r["val_loss_total"]))
            macro_f1s.append(float(r["macro_f1"]))
            worst_f1s.append(float(r["worst_event_f1"]))

    if not train_epochs:
        return

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_epochs, train_losses, label="Train Loss (CE + Dice)", color="#1f77b4", lw=2)
    if val_epochs and val_losses:
        ax.plot(val_epochs, val_losses, label="Val Loss (CE + Dice)", color="#ff7f0e", lw=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title_suffix = f" ({output_dir.name})"
    ax.set_title(f"Training Loss{title_suffix}" if not val_epochs else f"Training and Validation Loss{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Plot 2: Macro and Worst F1 (only if validation data exists)
    if val_epochs and macro_f1s and worst_f1s:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(val_epochs, macro_f1s, label="Event-Macro Flood F1", color="#2ca02c", lw=2)
        ax.plot(val_epochs, worst_f1s, label="Worst-Event Flood F1", color="#d62728", lw=2, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Validation Flood F1 Metrics ({output_dir.name})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "f1_curve.png", dpi=150)
        plt.close(fig)


def update_checkpoint_hashes(output_dir: Path, best_epoch: Optional[int], last_epoch: int) -> Path:
    """Updates the checkpoint SHA-256 ledger file."""
    hash_file = output_dir / "checkpoint_hashes.json"
    hashes: Dict[str, Any] = {}

    best_pt = output_dir / "best_checkpoint.pt"
    if best_pt.exists():
        hashes["best_checkpoint"] = {
            "filename": "best_checkpoint.pt",
            "epoch": best_epoch,
            "sha256": compute_file_sha256(best_pt),
        }

    final_pt = output_dir / "final_checkpoint.pt"
    if final_pt.exists():
        hashes["final_checkpoint"] = {
            "filename": "final_checkpoint.pt",
            "epoch": last_epoch,
            "sha256": compute_file_sha256(final_pt),
        }

    last_pt = output_dir / "last_checkpoint.pt"
    if last_pt.exists():
        hashes["last_checkpoint"] = {
            "filename": "last_checkpoint.pt",
            "epoch": last_epoch,
            "sha256": compute_file_sha256(last_pt),
        }

    for pt in sorted(output_dir.glob("epoch_*_checkpoint.pt")):
        try:
            ep_num = int(pt.stem.split("_")[1])
        except (IndexError, ValueError):
            ep_num = None
        hashes[pt.stem] = {
            "filename": pt.name,
            "epoch": ep_num,
            "sha256": compute_file_sha256(pt),
        }

    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    return hash_file


def validate_provenance_and_config(
    config_path: Union[str, Path],
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    lock_path: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validates configuration against immutable lock manifest and protocol.

    Verifies:
    1. Lock manifest exists and matches expected schema.
    2. Actual Protocol-v2 hash on disk equals the lock's recorded protocol hash.
    3. The config's governing_protocol_sha256 equals the actual Protocol-v2 hash.
    4. Actual config hash on disk equals the lock's recorded config hash.
    5. Actual split hash equals both the lock's and config's recorded split hash.
    6. For Model B, actual slope-statistics hash equals both lock's and config's recorded hash.
    7. Precision is 'fp32', and mixed_precision, autocast, grad_scaler are all False.
    8. Requested run_id (if provided) is in approved_run_ids in lock and matches model spec.
    9. Requested seed (if provided) is in approved_seeds in lock.

    Returns a dictionary of verified provenance metadata.
    Raises RuntimeError, ValueError, or FileNotFoundError if any check fails.
    """
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    actual_config_hash = compute_file_sha256(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Determine lock manifest path
    if lock_path is None:
        lock_rel = config.get("lock_manifest_path", "configs/source/source_training_v2_lock.json")
        lock_path = repo_root / lock_rel
    else:
        lock_path = Path(lock_path)
        if not lock_path.is_absolute():
            lock_path = repo_root / lock_path

    if not lock_path.exists():
        raise RuntimeError(f"Lock manifest file not found: {lock_path}")

    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)

    # Model classification
    in_channels = int(config.get("architecture", {}).get("in_channels_per_branch", 2))
    include_slope = (in_channels == 3)
    model_key = "model_b" if include_slope else "model_a"

    # 1. Protocol Verification
    protocol_spec = lock.get("protocol") or lock.get("governing_protocol", {})
    proto_rel = protocol_spec.get("path")
    proto_expected = protocol_spec.get("sha256")
    if not proto_rel or not proto_expected:
        raise RuntimeError("Lock manifest missing protocol specification")
    proto_path = repo_root / proto_rel
    if not proto_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {proto_path}")
    actual_proto_hash = compute_file_sha256(proto_path)
    if actual_proto_hash != proto_expected:
        raise RuntimeError(
            f"Protocol hash mismatch: actual '{actual_proto_hash}' != lock '{proto_expected}'"
        )

    # 2. Config's Recorded Protocol Hash Verification
    cfg_proto_hash = config.get("governing_protocol_sha256")
    if cfg_proto_hash != actual_proto_hash:
        raise RuntimeError(
            f"Config recorded protocol hash mismatch: config has '{cfg_proto_hash}' "
            f"but actual protocol is '{actual_proto_hash}'"
        )

    # 3. Config Hash Verification against Lock
    model_spec = None
    try:
        rel_config_path = str(config_path.relative_to(repo_root))
    except ValueError:
        rel_config_path = str(config_path)
    cfg_id = config.get("configuration_id")

    # Check lock models dictionary if present
    if "models" in lock and isinstance(lock["models"], dict):
        for k, spec in lock["models"].items():
            if isinstance(spec, dict) and (spec.get("config_path") == rel_config_path or k == cfg_id):
                model_spec = spec
                break

    # Check top-level dictionary entries matching config_path or configuration_id
    if model_spec is None:
        for k, spec in lock.items():
            if isinstance(spec, dict) and (spec.get("config_path") == rel_config_path or k == cfg_id):
                model_spec = spec
                break

    # Fallback to legacy model_key ('model_a' / 'model_b')
    if model_spec is None:
        model_spec = lock.get(model_key, {})

    expected_config_hash = model_spec.get("config_sha256")
    if not expected_config_hash:
        raise RuntimeError(f"Lock manifest missing {model_key} config hash for {rel_config_path}")
    if actual_config_hash != expected_config_hash:
        raise RuntimeError(
            f"{model_key} config hash mismatch: actual '{actual_config_hash}' != lock '{expected_config_hash}'"
        )

    # 4. Split Hash Verification
    split_spec = lock.get("frozen_split", lock.get("split", {}))
    split_rel = split_spec.get("path")
    split_expected = split_spec.get("sha256")
    if not split_rel or not split_expected:
        raise RuntimeError("Lock manifest missing frozen_split specification")
    split_path = repo_root / split_rel
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    actual_split_hash = compute_file_sha256(split_path)
    if actual_split_hash != split_expected:
        raise RuntimeError(
            f"Split file hash mismatch: actual '{actual_split_hash}' != lock '{split_expected}'"
        )
    cfg_split_hash = config.get("data_split", {}).get("split_config_sha256")
    if cfg_split_hash != actual_split_hash:
        raise RuntimeError(
            f"Config recorded split hash mismatch: config has '{cfg_split_hash}' "
            f"but actual split is '{actual_split_hash}'"
        )

    # 5. Slope Statistics Verification (if Model B)
    actual_slope_hash = None
    if include_slope:
        slope_spec = lock.get("slope_statistics", {})
        slope_rel = slope_spec.get("path")
        slope_expected = slope_spec.get("sha256")
        if not slope_rel or not slope_expected:
            raise RuntimeError("Lock manifest missing slope_statistics specification for Model B")
        slope_path = repo_root / slope_rel
        if not slope_path.exists():
            raise FileNotFoundError(f"Slope statistics artifact not found: {slope_path}")
        actual_slope_hash = compute_file_sha256(slope_path)
        if actual_slope_hash != slope_expected:
            raise RuntimeError(
                f"Slope statistics hash mismatch: actual '{actual_slope_hash}' != lock '{slope_expected}'"
            )
        cfg_slope_hash = (
            config.get("input_specification", {})
            .get("slope_policy", {})
            .get("statistics_artifact_sha256")
        )
        if cfg_slope_hash != actual_slope_hash:
            raise RuntimeError(
                f"Config recorded slope stats hash mismatch: config has '{cfg_slope_hash}' "
                f"but actual slope stats is '{actual_slope_hash}'"
            )

    # 6. Precision Verification
    opt_cfg = config.get("optimization_protocol", {})
    if opt_cfg.get("precision") != "fp32":
        raise RuntimeError(f"Invalid precision in config: expected 'fp32', got '{opt_cfg.get('precision')}'")
    if opt_cfg.get("mixed_precision") is not False:
        raise RuntimeError(f"mixed_precision must be False, got {opt_cfg.get('mixed_precision')}")
    if opt_cfg.get("autocast") is not False:
        raise RuntimeError(f"autocast must be False, got {opt_cfg.get('autocast')}")
    if opt_cfg.get("grad_scaler") is not False:
        raise RuntimeError(f"grad_scaler must be False, got {opt_cfg.get('grad_scaler')}")

    # 7. Mode & Constraint Verification
    is_all_source = (
        config.get("training_mode") == "all_source_fixed_epoch"
        or config.get("run_type") == "all_source_training"
    )
    if is_all_source:
        max_epochs = opt_cfg.get("max_epochs")
        if max_epochs != 16:
            raise RuntimeError(
                f"All-source fixed-epoch mode requires exactly 16 epochs, got {max_epochs}"
            )
        patience = opt_cfg.get("early_stopping_patience")
        if patience is not None:
            raise RuntimeError(
                f"All-source fixed-epoch mode prohibits early stopping, but early_stopping_patience is {patience}"
            )

    # 8. Run ID & Seed Verification
    if run_id is not None:
        approved_run_ids = lock.get("approved_run_ids", [])
        if run_id not in approved_run_ids:
            raise RuntimeError(
                f"Requested run ID '{run_id}' not in approved lock run IDs: {approved_run_ids}"
            )
        model_approved_run = model_spec.get("approved_run_id")
        if model_approved_run and run_id != model_approved_run:
            raise RuntimeError(
                f"Requested run ID '{run_id}' does not match {model_key} approved run ID '{model_approved_run}'"
            )

    cfg_seeds = config.get("optimization_protocol", {}).get("seeds", [])
    config_declared_seed = cfg_seeds[0] if cfg_seeds else None

    approved_seeds = lock.get("approved_seeds", [0])
    if seed is not None:
        if int(seed) not in approved_seeds:
            raise RuntimeError(
                f"Requested seed {seed} not in approved lock seeds: {approved_seeds}"
            )
        if config_declared_seed is not None and int(seed) != int(config_declared_seed):
            raise RuntimeError(
                f"Seed mismatch: requested seed {seed} does not match config declared seed {config_declared_seed}"
            )
    elif config_declared_seed is not None:
        if int(config_declared_seed) not in approved_seeds:
            raise RuntimeError(
                f"Config declared seed {config_declared_seed} not in approved lock seeds: {approved_seeds}"
            )

    return {
        "status": "VALIDATED",
        "model_key": model_key,
        "config_hash": actual_config_hash,
        "protocol_hash": actual_proto_hash,
        "split_hash": actual_split_hash,
        "slope_stats_hash": actual_slope_hash,
        "precision": "fp32",
        "lock_path": str(lock_path),
        "config": config,
        "lock": lock,
    }


def train_source_model(args: argparse.Namespace) -> int:
    """Main production training execution function."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 0. Early Provenance & Lock Validation
    # Fails immediately before output directory creation, GPU allocation, or dataset loading.
    provenance = validate_provenance_and_config(
        config_path=config_path,
        run_id=args.run_id,
        seed=args.seed,
    )
    config = provenance.get("config")
    config_hash = provenance.get("config_hash")
    model_type = provenance.get("model_key")
    protocol_hash = provenance.get("protocol_hash")
    split_hash = provenance.get("split_hash")
    slope_stats_hash = provenance.get("slope_stats_hash")
    lock_path = provenance.get("lock_path")

    # Strict provenance assertion: fail before creating output directories or loading datasets
    if not config or not isinstance(config, dict):
        raise RuntimeError("Verified config dictionary missing from provenance validation result")
    if not config_hash or not isinstance(config_hash, str) or len(config_hash) != 64:
        raise RuntimeError(f"Verified config_hash missing or invalid: {config_hash}")
    if not protocol_hash or not isinstance(protocol_hash, str) or len(protocol_hash) != 64:
        raise RuntimeError(f"Verified protocol_hash missing or invalid: {protocol_hash}")
    if not split_hash or not isinstance(split_hash, str) or len(split_hash) != 64:
        raise RuntimeError(f"Verified split_hash missing or invalid: {split_hash}")
    if not lock_path or not Path(lock_path).exists():
        raise RuntimeError(f"Verified lock_path missing or invalid: {lock_path}")
    lock_hash = compute_file_sha256(lock_path)
    if not lock_hash or len(lock_hash) != 64:
        raise RuntimeError(f"Failed to compute SHA-256 for lock manifest: {lock_path}")

    in_channels = int(config["architecture"]["in_channels_per_branch"])
    include_slope = (in_channels == 3)
    if include_slope:
        if not slope_stats_hash or not isinstance(slope_stats_hash, str) or len(slope_stats_hash) != 64:
            raise RuntimeError(f"Verified slope_stats_hash missing or invalid for Model B: {slope_stats_hash}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persistent logging
    log_file_path = output_dir / "train.log"
    tee = TeeLogger(log_file_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee

    interrupted = False
    trainer = None

    def sig_handler(signum, frame):
        nonlocal interrupted, trainer
        sig_name = signal.Signals(signum).name
        print(f"\n[SIGNAL] Caught signal {sig_name} ({signum}). Initiating graceful termination...")
        interrupted = True
        if trainer is not None:
            trainer.stop_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        print("=================================================================")
        print(f"ChangeGuard Controlled Source Model Training: {args.run_id}")
        print(f"Start Time (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        print(f"Host: {os.uname().nodename} | PID: {os.getpid()}")
        print(f"Device: {args.device} | Seed: {args.seed}")
        print(f"Config: {config_path} (SHA-256: {config_hash})")
        print(f"In Channels: {in_channels} | Include Slope: {include_slope}")
        print(f"Output Directory: {output_dir}")
        print("Verified against immutable provenance lock:")
        print(f"  Lock Manifest:       {provenance['lock_path']}")
        print(f"  Protocol SHA-256:    {provenance['protocol_hash']}")
        print(f"  Config SHA-256:      {provenance['config_hash']}")
        print(f"  Split SHA-256:       {provenance['split_hash']}")
        if include_slope:
            print(f"  Slope Stats SHA-256: {provenance['slope_stats_hash']}")
        print(f"  Precision:           {provenance['precision']} (mixed_precision=False, autocast=False, grad_scaler=False)")
        print("=================================================================")

        # Validated split and slope stats paths
        split_path = REPO_ROOT / provenance["lock"]["frozen_split"]["path"]
        slope_stats_rel = provenance["lock"]["slope_statistics"]["path"]
        slope_stats_path = REPO_ROOT / slope_stats_rel if include_slope else None

        # Mode determination
        is_all_source_mode = (
            bool(getattr(args, "all_source", False))
            or bool(getattr(args, "no_val", False))
            or config.get("training_mode") == "all_source_fixed_epoch"
            or config.get("run_type") == "all_source_training"
        )

        # Hyperparameters
        opt_cfg = config.get("optimization_protocol", {})
        batch_cfg = config.get("batch_size_policy", {})
        if is_all_source_mode:
            if args.patience is not None:
                raise RuntimeError(
                    f"All-source fixed-epoch mode prohibits early stopping, but --patience was set to {args.patience}"
                )
            max_epochs = int(args.max_epochs or opt_cfg.get("max_epochs", 16))
            patience = None
        else:
            max_epochs = int(args.max_epochs or opt_cfg.get("max_epochs", 150))
            patience = int(args.patience or opt_cfg.get("early_stopping_patience", 20))
        micro_bs = int(batch_cfg.get("physical_microbatch_size", 8))
        effective_bs = int(batch_cfg.get("effective_batch_size", 32))
        accum_steps = effective_bs // micro_bs

        # Seed setup
        cfg_seeds = config.get("optimization_protocol", {}).get("seeds", [])
        config_declared_seed = cfg_seeds[0] if cfg_seeds else None

        if args.seed is None:
            seed = int(config_declared_seed) if config_declared_seed is not None else 0
        else:
            seed = int(args.seed)
            if config_declared_seed is not None and seed != int(config_declared_seed):
                raise RuntimeError(
                    f"CLI seed override {seed} contradicts config declared seed {config_declared_seed}"
                )
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        def seed_worker(worker_id):
            worker_seed = (seed + worker_id + 1) % (2**32)
            np.random.seed(worker_seed)
            import random
            random.seed(worker_seed)

        # 2. Datasets & Loaders
        if is_all_source_mode:
            print("\nInitializing all-source dataset (all 27 events, no validation split)...")
            train_dataset = KuroSourceDataset(
                split_config_path=str(split_path),
                split="all_source",
                include_slope=include_slope,
                slope_stats_path=str(slope_stats_path) if slope_stats_path else None,
            )
            val_dataset = None
            val_loader = None
        else:
            print("\nInitializing datasets...")
            train_dataset = KuroSourceDataset(
                split_config_path=str(split_path),
                split="train",
                include_slope=include_slope,
                slope_stats_path=str(slope_stats_path) if slope_stats_path else None,
            )
            val_dataset = KuroSourceDataset(
                split_config_path=str(split_path),
                split="val",
                include_slope=include_slope,
                slope_stats_path=str(slope_stats_path) if slope_stats_path else None,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=micro_bs,
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=True,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=micro_bs,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # 3. Model & Trainer
        model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32)
        trainer = SourceTrainer.from_config(
            model=model,
            config=config,
            output_dir=output_dir,
            device=args.device,
            seed=seed,
        )

        # Verify optimizer
        assert isinstance(trainer.optimizer, torch.optim.Adam), "Optimizer must be torch.optim.Adam"
        assert not isinstance(trainer.optimizer, torch.optim.AdamW), "Optimizer must not be AdamW"
        assert trainer.lr_scheduler is None, "Scheduler must be None"
        opt_pg = trainer.optimizer.param_groups[0]
        assert opt_pg["lr"] == 1e-3, f"Learning rate must be 1e-3, got {opt_pg['lr']}"
        assert opt_pg["weight_decay"] == 0.0, f"Weight decay must be 0.0, got {opt_pg['weight_decay']}"
        assert opt_pg["betas"] == (0.9, 0.999), f"Betas must be (0.9, 0.999), got {opt_pg['betas']}"

        # 4. Resume or Fresh Setup
        last_ckpt_path = output_dir / "last_checkpoint.pt"
        best_ckpt_path = output_dir / "best_checkpoint.pt"
        final_ckpt_path = output_dir / "final_checkpoint.pt"
        manifest_path = output_dir / "run_manifest.json"
        trace_path = output_dir / "access_trace.json"

        train_csv_path = output_dir / "train_history.csv"
        val_csv_path = output_dir / "validation_history.csv"
        event_csv_path = output_dir / "event_validation_metrics.csv"

        start_epoch = 0
        early_stopping_counter = 0
        access_tracker = AccessTracker(model_type=model_type)
        if is_all_source_mode:
            collapse_detector = None
        else:
            collapse_detector = PredictiveCollapseDetector(
                warmup_epochs=int(opt_cfg.get("collapse_warmup_epochs", 5)),
                patience=int(opt_cfg.get("collapse_patience", 3)),
            )

        resume_target: Optional[Path] = None
        if args.resume:
            resume_target = Path(args.resume)
        elif args.resume_auto and last_ckpt_path.exists():
            resume_target = last_ckpt_path

        if resume_target is not None:
            if not resume_target.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")
            print(f"\nResuming training from checkpoint: {resume_target}")
            resumed_state = trainer.resume_from_checkpoint(resume_target, restore_rng=True)
            start_epoch = int(resumed_state["epoch"])
            early_stopping_counter = int(resumed_state.get("early_stopping_counter", 0))
            if collapse_detector is not None and "collapse_detector_state" in resumed_state.get("extra_metadata", {}):
                collapse_detector.load_state_dict(resumed_state["extra_metadata"]["collapse_detector_state"])

            # Resume integrity assertions
            assert resumed_state.get("config_hash") == config_hash, "Resume config_hash mismatch!"
            assert resumed_state.get("seed") == seed, f"Resume seed mismatch: {resumed_state.get('seed')} vs {seed}"

            # Prune CSVs to match resumed epoch (prevent duplicate entries)
            def _filter_csv(csv_path: Path, max_ep: int):
                if not csv_path.exists():
                    return
                with open(csv_path, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) <= 1:
                    return
                header, data = rows[0], rows[1:]
                kept = [r for r in data if int(r[0]) <= max_ep]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(kept)

            _filter_csv(train_csv_path, start_epoch)
            _filter_csv(val_csv_path, start_epoch)
            _filter_csv(event_csv_path, start_epoch)
            print(f"Resumed at completed epoch {start_epoch}, next epoch is {start_epoch + 1}.")
        else:
            if last_ckpt_path.exists() and not args.resume_auto:
                raise RuntimeError(
                    f"Existing checkpoint {last_ckpt_path} detected but neither --resume nor --resume_auto was passed! "
                    "Use --resume_auto to safely resume, or choose a new output directory."
                )
            # Fresh run: Initialize CSV headers
            with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "loss_total", "loss_ce", "loss_dice", "learning_rate",
                    "optimizer_steps", "samples_observed", "batches_observed",
                    "epoch_duration_sec", "peak_gpu_memory_mb"
                ])
            if not is_all_source_mode:
                with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch", "val_loss_total", "val_loss_ce", "val_loss_dice",
                        "macro_precision", "macro_recall", "macro_f1", "macro_iou",
                        "pooled_precision", "pooled_recall", "pooled_f1", "pooled_iou",
                        "worst_event_id", "worst_event_f1", "pred_flood_pixels",
                        "gt_flood_pixels", "valid_pixels", "is_best_epoch", "collapse_counter"
                    ])
                with open(event_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch", "event_id", "sample_count", "valid_pixels",
                        "gt_flood_pixels", "pred_flood_pixels", "tp", "fp", "fn", "tn",
                        "precision", "recall", "f1", "iou"
                    ])

        # 5. Initialize RunManifest
        output_paths = {
            "last_checkpoint": str(last_ckpt_path),
            "train_history": str(train_csv_path),
            "access_trace": str(trace_path),
            "log": str(log_file_path),
        }
        if is_all_source_mode:
            output_paths["final_checkpoint"] = str(final_ckpt_path)
        else:
            output_paths["best_checkpoint"] = str(best_ckpt_path)
            output_paths["validation_history"] = str(val_csv_path)
            output_paths["event_validation_metrics"] = str(event_csv_path)

        manifest = RunManifest(
            run_id=args.run_id,
            config_path=str(config_path),
            config_hash=config_hash,
            checkpoint_path=final_ckpt_path if final_ckpt_path.exists() else (best_ckpt_path if best_ckpt_path.exists() else (last_ckpt_path if last_ckpt_path.exists() else None)),
            data_manifest=str(train_dataset.manifest_path),
            accessed_event_ids=[],
            random_seed=seed,
            labels_available_to_adaptation=False,
            labels_used_for_updates=True,
            is_partial_run=True,
            expected_sample_count=len(train_dataset),
            processed_sample_count=0,
            output_paths=output_paths,
            run_type="all_source_training" if is_all_source_mode else "source_training",
            protocol_hash=protocol_hash,
            lock_hash=lock_hash,
            split_hash=split_hash,
            slope_stats_hash=slope_stats_hash if include_slope else None,
        )
        manifest.status = "RUNNING"
        manifest.save(manifest_path)

        # 6. Training Loop
        if is_all_source_mode:
            print(f"\nStarting all-source fixed-epoch training loop: Epoch {start_epoch + 1} to {max_epochs} (16 fixed epochs, no validation split)\n")
            completed_status = "COMPLETED_FIXED_EPOCHS"
        else:
            print(f"\nStarting controlled source training loop: Epoch {start_epoch + 1} to {max_epochs}")
            print(f"Early stopping patience: {patience} epochs on inner-validation Macro F1\n")
            completed_status = "COMPLETED_MAX_EPOCH"

        for epoch in range(start_epoch + 1, max_epochs + 1):
            if interrupted:
                break

            epoch_start_time = time.time()
            access_tracker.set_mode("train")

            # Training epoch
            train_metrics = trainer.train_epoch(
                train_loader=train_loader,
                log_interval=int(args.log_interval),
                epoch_idx=epoch,
                tracker=access_tracker,
                max_batches=getattr(args, "max_train_batches", None),
            )

            if interrupted:
                break

            if is_all_source_mode:
                # All-source fixed-epoch mode: skip validation and early stopping
                extra_meta = {
                    "run_id": args.run_id,
                    "model_type": model_type,
                    "in_channels": in_channels,
                    "include_slope": include_slope,
                    "protocol_hash": protocol_hash,
                    "config_hash": config_hash,
                    "lock_hash": lock_hash,
                    "split_hash": split_hash,
                    "slope_stats_hash": slope_stats_hash if include_slope else None,
                    "training_mode": "all_source_fixed_epoch",
                    "epoch": epoch,
                    "max_epochs": max_epochs,
                }

                save_every = config.get("checkpoint_policy", {}).get("save_every_epoch", True)
                if save_every:
                    trainer.save_checkpoint(
                        f"epoch_{epoch:03d}_checkpoint.pt",
                        metrics={"loss_total": train_metrics["loss_total"]},
                        config_path=str(config_path),
                        config_hash=config_hash,
                        extra_metadata=extra_meta,
                    )

                trainer.save_checkpoint(
                    "last_checkpoint.pt",
                    metrics={"loss_total": train_metrics["loss_total"]},
                    config_path=str(config_path),
                    config_hash=config_hash,
                    extra_metadata=extra_meta,
                )

                if epoch == max_epochs:
                    trainer.save_checkpoint(
                        "final_checkpoint.pt",
                        metrics={"loss_total": train_metrics["loss_total"]},
                        config_path=str(config_path),
                        config_hash=config_hash,
                        extra_metadata=extra_meta,
                    )

                update_checkpoint_hashes(output_dir, None, epoch)

                with open(train_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        f"{train_metrics['loss_total']:.6f}",
                        f"{train_metrics['loss_ce']:.6f}",
                        f"{train_metrics['loss_dice']:.6f}",
                        f"{train_metrics['learning_rate']:.6f}",
                        train_metrics["optimizer_steps"],
                        train_metrics["samples_observed"],
                        train_metrics["batches_observed"],
                        f"{train_metrics['epoch_duration_sec']:.2f}",
                        f"{train_metrics['peak_gpu_memory_mb']:.2f}",
                    ])

                access_tracker.save(trace_path)
                manifest.accessed_event_ids = sorted(list(access_tracker.train_events))
                manifest.processed_sample_count = len(access_tracker.train_samples)
                if final_ckpt_path.exists():
                    manifest.update_checkpoint(final_ckpt_path)
                elif last_ckpt_path.exists():
                    manifest.update_checkpoint(last_ckpt_path)
                manifest.save(manifest_path)

                epoch_total_time = time.time() - epoch_start_time
                print(
                    f"\n>>> Epoch {epoch:03d}/{max_epochs:03d} Summary ({epoch_total_time:.1f}s) <<<\n"
                    f"  Train: Loss={train_metrics['loss_total']:.4f} (CE={train_metrics['loss_ce']:.4f}, Dice={train_metrics['loss_dice']:.4f}) | Peak VRAM: {train_metrics['peak_gpu_memory_mb']:.1f} MiB\n"
                    f"  Status: ALL-SOURCE FIXED-EPOCH PROGRESS ({epoch}/{max_epochs})\n"
                )
            else:
                # Validation epoch
                access_tracker.set_mode("val")
                val_summary, val_events = trainer.evaluate(
                    val_loader=val_loader,
                    max_batches=getattr(args, "max_val_batches", None),
                    compute_loss=True,
                    tracker=access_tracker,
                )

                # Predictive collapse detection (Protocol v2 Section 5.2)
                total_pred_flood = sum(e["pred_flood_pixels"] for e in val_events)
                is_collapsed = collapse_detector.update(epoch, total_pred_flood)
                if is_collapsed:
                    raise PredictiveCollapseError(
                        f"Predictive collapse triggered at Epoch {epoch}: {collapse_detector.patience} consecutive "
                        f"validation epochs with 0 predicted flood pixels across known-positive validation set.",
                        epoch=epoch,
                        consecutive_zero_epochs=collapse_detector.consecutive_zero_epochs,
                        warmup_epochs=collapse_detector.warmup_epochs,
                        patience=collapse_detector.patience,
                    )

                # Tie-breaking checkpoint selection
                is_best = is_better_checkpoint(
                    cand_metrics=val_summary,
                    cand_epoch=epoch,
                    best_metrics=trainer.best_metrics,
                    best_epoch=trainer.best_epoch,
                )

                if is_best:
                    trainer.best_metrics = copy.deepcopy(val_summary)
                    trainer.best_epoch = epoch
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                # Extra metadata for checkpoint provenance
                extra_meta = {
                    "run_id": args.run_id,
                    "model_type": model_type,
                    "in_channels": in_channels,
                    "include_slope": include_slope,
                    "protocol_hash": protocol_hash,
                    "config_hash": config_hash,
                    "lock_hash": lock_hash,
                    "split_hash": split_hash,
                    "slope_stats_hash": slope_stats_hash if include_slope else None,
                    "early_stopping_counter": early_stopping_counter,
                    "patience": patience,
                    "is_best_epoch": is_best,
                    "collapse_detector_state": collapse_detector.state_dict(),
                }

                # Save last checkpoint
                trainer.save_checkpoint(
                    "last_checkpoint.pt",
                    metrics=val_summary,
                    config_path=str(config_path),
                    config_hash=config_hash,
                    extra_metadata=extra_meta,
                )

                # Save best checkpoint if new best
                if is_best:
                    trainer.save_checkpoint(
                        "best_checkpoint.pt",
                        metrics=val_summary,
                        config_path=str(config_path),
                        config_hash=config_hash,
                        extra_metadata=extra_meta,
                    )

                # Update checkpoint hashes
                update_checkpoint_hashes(output_dir, trainer.best_epoch, epoch)

                # Append to train_history.csv
                with open(train_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        f"{train_metrics['loss_total']:.6f}",
                        f"{train_metrics['loss_ce']:.6f}",
                        f"{train_metrics['loss_dice']:.6f}",
                        f"{train_metrics['learning_rate']:.6f}",
                        train_metrics["optimizer_steps"],
                        train_metrics["samples_observed"],
                        train_metrics["batches_observed"],
                        f"{train_metrics['epoch_duration_sec']:.2f}",
                        f"{train_metrics['peak_gpu_memory_mb']:.2f}",
                    ])

                # Append to validation_history.csv
                worst_ev = val_summary.get("worst_event_f1", {})
                pooled_m = val_summary.get("pooled_metrics", {})
                with open(val_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        f"{val_summary.get('loss_total', 0.0):.6f}",
                        f"{val_summary.get('loss_ce', 0.0):.6f}",
                        f"{val_summary.get('loss_dice', 0.0):.6f}",
                        f"{val_summary.get('macro_precision', 0.0):.6f}",
                        f"{val_summary.get('macro_recall', 0.0):.6f}",
                        f"{val_summary.get('macro_f1', 0.0):.6f}",
                        f"{val_summary.get('macro_iou', 0.0):.6f}",
                        f"{pooled_m.get('precision', 0.0):.6f}",
                        f"{pooled_m.get('recall', 0.0):.6f}",
                        f"{pooled_m.get('f1', 0.0):.6f}",
                        f"{pooled_m.get('iou', 0.0):.6f}",
                        worst_ev.get("event_id", -1),
                        f"{worst_ev.get('f1', 0.0):.6f}",
                        total_pred_flood,
                        sum(e["gt_flood_pixels"] for e in val_events),
                        sum(e["valid_pixels"] for e in val_events),
                        "TRUE" if is_best else "FALSE",
                        collapse_detector.consecutive_zero_epochs,
                    ])

                # Append to event_validation_metrics.csv
                with open(event_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for ev in val_events:
                        writer.writerow([
                            epoch,
                            ev["event_id"],
                            ev["sample_count"],
                            ev["valid_pixels"],
                            ev["gt_flood_pixels"],
                            ev["pred_flood_pixels"],
                            ev["tp"],
                            ev["fp"],
                            ev["fn"],
                            ev["tn"],
                            f"{ev['precision']:.6f}",
                            f"{ev['recall']:.6f}",
                            f"{ev['f1']:.6f}",
                            f"{ev['iou']:.6f}",
                        ])

                # Update trace and manifest
                access_tracker.save(trace_path)
                manifest.accessed_event_ids = sorted(list(access_tracker.train_events | access_tracker.val_events))
                manifest.processed_sample_count = len(access_tracker.train_samples)
                if best_ckpt_path.exists():
                    manifest.update_checkpoint(best_ckpt_path)
                elif last_ckpt_path.exists():
                    manifest.update_checkpoint(last_ckpt_path)
                manifest.save(manifest_path)

                # Epoch summary banner
                epoch_total_time = time.time() - epoch_start_time
                best_marker = " [★ NEW BEST]" if is_best else f" [Patience: {early_stopping_counter}/{patience}]"
                collapse_info = f" [Collapse: {collapse_detector.consecutive_zero_epochs}/{collapse_detector.patience}]"
                print(
                    f"\n>>> Epoch {epoch:03d}/{max_epochs:03d} Summary ({epoch_total_time:.1f}s) <<<\n"
                    f"  Train: Loss={train_metrics['loss_total']:.4f} (CE={train_metrics['loss_ce']:.4f}, Dice={train_metrics['loss_dice']:.4f}) | Peak VRAM: {train_metrics['peak_gpu_memory_mb']:.1f} MiB\n"
                    f"  Val:   Loss={val_summary.get('loss_total', 0.0):.4f} | Macro F1={val_summary.get('macro_f1', 0.0):.4f} | Macro Rec={val_summary.get('macro_recall', 0.0):.4f} | Worst F1={worst_ev.get('f1', 0.0):.4f} (Ev {worst_ev.get('event_id', -1)})\n"
                    f"  Status:{best_marker}{collapse_info} (Best Epoch: {trainer.best_epoch})\n"
                )

                # Early stopping check
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered at Epoch {epoch}! No improvement for {patience} consecutive epochs.")
                    print(f"Best model was checkpointed at Epoch {trainer.best_epoch} with Macro F1={trainer.best_metrics.get('macro_f1', 0.0):.4f}.")
                    completed_status = "COMPLETED_EARLY_STOP"
                    break
        else:
            if is_all_source_mode:
                completed_status = "COMPLETED_FIXED_EPOCHS"
            else:
                completed_status = "COMPLETED_MAX_EPOCH"

        # 7. Post-Loop Completion Handling
        if interrupted:
            print("\n[INTERRUPTED] Training loop terminated early by signal.")
            manifest.mark_completed("INTERRUPTED")
            manifest.save(manifest_path)
            generate_curves(output_dir)
            return 130

        print("\n=================================================================")
        print(f"Training completed successfully for {args.run_id}!")
        print(f"Completion Status: {completed_status}")
        if is_all_source_mode:
            print(f"Total Epochs Completed: {trainer.current_epoch} (Fixed budget: {max_epochs})")
            print(f"Final Train Loss: {train_metrics['loss_total']:.4f} (CE={train_metrics['loss_ce']:.4f}, Dice={train_metrics['loss_dice']:.4f})")
            print(f"Final Checkpoint: {final_ckpt_path}")
            print(f"Last Checkpoint: {last_ckpt_path}")
        else:
            print(f"Final Epoch: {trainer.current_epoch} | Best Epoch: {trainer.best_epoch}")
            print(f"Best Inner-Val Macro F1: {trainer.best_metrics.get('macro_f1', 0.0):.4f}")
            print(f"Best Checkpoint: {best_ckpt_path}")
            print(f"Last Checkpoint: {last_ckpt_path}")
        print("=================================================================")

        manifest.mark_completed(completed_status)
        manifest.is_partial_run = False
        if is_all_source_mode:
            if final_ckpt_path.exists():
                manifest.update_checkpoint(final_ckpt_path)
            elif last_ckpt_path.exists():
                manifest.update_checkpoint(last_ckpt_path)
        else:
            if best_ckpt_path.exists():
                manifest.update_checkpoint(best_ckpt_path)
            elif last_ckpt_path.exists():
                manifest.update_checkpoint(last_ckpt_path)
        manifest.save(manifest_path)

        # Generate plots
        generate_curves(output_dir)
        return 0

    except NumericalInstabilityError as e:
        print(f"\n[FAILED_NUMERICAL] Numerical instability detected: {e}")
        failure_manifest_path = output_dir / "failure_manifest.json"
        failure_data = {
            "run_id": args.run_id,
            "status": "FAILED_NUMERICAL",
            "failure_type": "numerical_instability",
            "epoch": getattr(e, "epoch", None),
            "batch_idx": getattr(e, "batch_idx", None),
            "global_step": getattr(e, "global_step", trainer.global_step if trainer else None),
            "tensor_name": getattr(e, "tensor_name", "unknown"),
            "stage": getattr(e, "stage", "unknown"),
            "sample_ids": getattr(e, "sample_ids", []),
            "event_ids": getattr(e, "event_ids", []),
            "error_message": str(e),
            "details": getattr(e, "details", {}),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": config_hash if "config_hash" in locals() else None,
            "git_commit": get_git_commit(output_dir),
            "git_dirty": is_git_dirty(output_dir),
        }
        with open(failure_manifest_path, "w", encoding="utf-8") as f:
            json.dump(failure_data, f, indent=2)
        print(f"Wrote failure manifest to {failure_manifest_path}")

        try:
            if "manifest" in locals() and manifest is not None:
                manifest.mark_completed("FAILED_NUMERICAL")
                manifest.is_partial_run = True
                manifest.save(manifest_path)
        except Exception:
            pass
        generate_curves(output_dir)
        return 1

    except PredictiveCollapseError as e:
        print(f"\n[FAILED_COLLAPSE] Predictive collapse detected: {e}")
        failure_manifest_path = output_dir / "failure_manifest.json"
        failure_data = {
            "run_id": args.run_id,
            "status": "FAILED_COLLAPSE",
            "failure_type": "predictive_collapse",
            "epoch": getattr(e, "epoch", None),
            "consecutive_zero_epochs": getattr(e, "consecutive_zero_epochs", 0),
            "warmup_epochs": getattr(e, "warmup_epochs", 5),
            "patience": getattr(e, "patience", 3),
            "error_message": str(e),
            "details": getattr(e, "details", {}),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": config_hash if "config_hash" in locals() else None,
            "git_commit": get_git_commit(output_dir),
            "git_dirty": is_git_dirty(output_dir),
        }
        with open(failure_manifest_path, "w", encoding="utf-8") as f:
            json.dump(failure_data, f, indent=2)
        print(f"Wrote failure manifest to {failure_manifest_path}")

        try:
            if "manifest" in locals() and manifest is not None:
                manifest.mark_completed("FAILED_COLLAPSE")
                manifest.is_partial_run = True
                manifest.save(manifest_path)
        except Exception:
            pass
        generate_curves(output_dir)
        return 1

    except Exception as e:
        print(f"\n[FATAL ERROR] Training run failed with exception: {e}")
        import traceback
        traceback.print_exc()
        try:
            if "manifest" in locals() and manifest is not None:
                manifest.mark_completed("FAILED")
                manifest.is_partial_run = True
                if hasattr(manifest, "extra_metadata"):
                    manifest.extra_metadata["error"] = str(e)
                manifest.save(manifest_path)
        except Exception as mex:
            print(f"[WARNING] Could not update run_manifest on failure: {mex}")
        try:
            failure_manifest_path = output_dir / "failure_manifest.json"
            failure_data = {
                "run_id": getattr(args, "run_id", "unknown"),
                "status": "FAILED",
                "failure_type": "unhandled_exception",
                "error_message": str(e),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config_hash": config_hash if "config_hash" in locals() else None,
                "git_commit": get_git_commit(output_dir),
                "git_dirty": is_git_dirty(output_dir),
            }
            with open(failure_manifest_path, "w", encoding="utf-8") as f:
                json.dump(failure_data, f, indent=2)
            print(f"Wrote failure manifest to {failure_manifest_path}")
        except Exception:
            pass
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()


def main():
    parser = argparse.ArgumentParser(description="Production Controlled Source-Model Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to source model configuration JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--run_id", type=str, default=None, help="Canonical Run ID")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device (e.g. cuda:0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config declared seed)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum epochs override")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience override")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--log_interval", type=int, default=50, help="Print progress every N optimizer steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_auto", action="store_true", help="Auto-resume from output_dir/last_checkpoint.pt")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional batch limit per training epoch (e.g. for smoke tests)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional batch limit per validation epoch (e.g. for smoke tests)")
    parser.add_argument("--validate_only", action="store_true", help="Perform provenance and config validation against lock manifest and exit without GPU allocation, dataset loading, or training")
    parser.add_argument("--all_source", action="store_true", help="Enable all-source fixed-epoch training mode (no validation split)")
    parser.add_argument("--no_val", action="store_true", help="Synonym for --all_source (skip validation split)")

    args = parser.parse_args()

    if args.validate_only:
        try:
            res = validate_provenance_and_config(
                config_path=Path(args.config),
                run_id=args.run_id,
                seed=args.seed,
            )
            print("=================================================================")
            print("ChangeGuard Provenance & Configuration Validation: PASSED")
            print(f"Model Key:            {res['model_key']} ({args.config})")
            print(f"Config SHA-256:       {res['config_hash']}")
            print(f"Protocol SHA-256:     {res['protocol_hash']}")
            print(f"Split SHA-256:        {res['split_hash']}")
            if res['slope_stats_hash']:
                print(f"Slope Stats SHA-256:  {res['slope_stats_hash']}")
            print(f"Precision:            {res['precision']} (mixed_precision=False, autocast=False, grad_scaler=False)")
            print(f"Lock Manifest:        {res['lock_path']}")
            print("Status: 100% VALIDATED AGAINST IMMUTABLE PROVENANCE LOCK")
            print("=================================================================")
            sys.exit(0)
        except Exception as e:
            print(f"PROVENANCE VALIDATION FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.output_dir or not args.run_id:
        parser.error("--output_dir and --run_id are required when --validate_only is not specified.")

    exit_code = train_source_model(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
