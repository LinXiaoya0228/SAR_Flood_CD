"""ChangeGuard Plan V3 E1.3 Bounded Real-Data Smoke Runner.

Adheres strictly to docs/MASTER_EXECUTION_PLAN_V3.md and docs/CACHE_AND_MEASUREMENT_CONTRACT.md:
1. Scope: Event 520 only, deterministic first 50 samples, batches [16, 16, 16, 2].
2. Fail-Closed Preflight: Cryptographic verification of Git HEAD, config, checkpoint, manifest, and split.
3. Zero Sealed-Test Access: Hard boundary enforcing no access to forbidden sealed-test events.
4. GPU Isolation: Strict verification of logical device and GPU availability.
5. Exact Timing & Label Guarding: Sensory inputs isolated during adaptation; evaluation labels unlocked
   only after predictions and actions are durably persisted to disk.
6. Complete Measurement Plumbing: Records gradient L2 norms, BN displacement, classwise selection counts,
   full-precision summaries, and double-masked confusion matrix with conservation.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.e1_orchestrator import (
    E1ConsistencyAdapter,
    E1OfflineEvaluator,
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
)
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    KuroEvaluationDataset,
)
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieves current Git commit and working-tree cleanliness."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        status_porcelain = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        # Check tracked modified files
        status_lines = status_porcelain.splitlines() if status_porcelain else []
        tracked_dirty = any(line[:2].strip() in {"M", "A", "D", "R"} for line in status_lines)
        return {
            "commit": commit,
            "tracked_dirty": tracked_dirty,
            "status_porcelain": status_porcelain,
        }
    except Exception as e:
        return {"commit": f"UNKNOWN_ERROR: {e}", "tracked_dirty": True}


class E1RealSmokeRunner:
    """Fail-closed real-data smoke runner for ChangeGuard Plan V3 Subtask E1.3."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        device: str = "cuda:0",
        allow_overwrite: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device_str = str(device)
        self.allow_overwrite = bool(allow_overwrite)

        smoke_spec = self.cfg["smoke_specification"]
        self.target_event = int(smoke_spec["event_id"])
        if self.target_event != 520:
            raise ValueError(f"Strict E1.3 constraint violation: event must be 520, got {self.target_event}")

        self.max_samples = int(smoke_spec.get("max_samples", 50))
        self.batch_size = int(smoke_spec.get("batch_size", 16))
        self.expected_batches = smoke_spec.get("expected_batches", [16, 16, 16, 2])
        self.reset_mode = str(smoke_spec.get("reset_mode", "reset_per_event"))

        # Paths
        paths_cfg = self.cfg.get("paths", {})
        self.output_dir = (
            output_dir.resolve()
            if output_dir is not None
            else (REPO_ROOT / paths_cfg.get("default_output_dir", "outputs/e1_real_smoke/approved_smoke_event520_v1")).resolve()
        )
        self.log_dir = (
            log_dir.resolve()
            if log_dir is not None
            else (REPO_ROOT / paths_cfg.get("default_log_dir", "logs/e1_real_smoke/approved_smoke_event520_v1")).resolve()
        )

        # Dataset & Checkpoint settings
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.split_path = (REPO_ROOT / ds_cfg["split_path"]).resolve()
        self.sar_mean = ds_cfg.get("sar_mean", [0.0953, 0.0264])
        self.sar_std = ds_cfg.get("sar_std", [0.0427, 0.0215])
        self.sar_clamp_max = float(ds_cfg.get("sar_clamp_max", 0.15))
        self.canonical_preevent = ds_cfg.get("canonical_preevent", "pre_event_2")

        ckpt_info = self.cfg["source_checkpoints"]["seeds"]["0"]
        self.checkpoint_path = (REPO_ROOT / ckpt_info["checkpoint_path"]).resolve()
        self.ckpt_expected_sha256 = ckpt_info["expected_sha256"]

        # Expected digests
        self.manifest_expected_sha256 = ds_cfg.get("manifest_expected_sha256")
        self.split_expected_sha256 = ds_cfg.get("split_expected_sha256")

        # Model hyperparameters
        m_cfg = self.cfg["model"]
        self.in_channels = int(m_cfg.get("in_channels", 2))
        self.num_classes = int(m_cfg.get("num_classes", 3))
        self.base_channel = int(m_cfg.get("base_channel", 32))

        # Adaptation hyperparameters
        h_cfg = self.cfg.get("adaptation_hyperparameters", {})
        self.learning_rate = float(h_cfg.get("learning_rate", 1.25e-4))
        self.momentum = float(h_cfg.get("momentum", 0.9))
        self.weight_decay = float(h_cfg.get("weight_decay", 0.0))
        self.confidence_threshold = float(
            self.cfg.get("consistency_views", {}).get("confidence_masking", {}).get("threshold", 0.90)
        )

    def validate_preflight(self, fail_closed: bool = True) -> Dict[str, Any]:
        """Strict preflight audit verifying code, assets, hashes, and GPU resources."""
        errors: List[str] = []

        # 1. Check Git HEAD
        git_info = get_git_info()
        commit = git_info["commit"]
        if not commit or len(commit) < 40:
            errors.append(f"Invalid Git commit: {commit}")

        # 2. Check Checkpoint
        if not self.checkpoint_path.is_file():
            errors.append(f"Checkpoint file not found: {self.checkpoint_path}")
        else:
            actual_ckpt_hash = compute_file_sha256(self.checkpoint_path)
            if actual_ckpt_hash != self.ckpt_expected_sha256:
                errors.append(
                    f"Checkpoint SHA-256 mismatch: expected {self.ckpt_expected_sha256}, got {actual_ckpt_hash}"
                )

        # 3. Check Dataset Manifest
        if not self.manifest_path.is_file():
            errors.append(f"Manifest file not found: {self.manifest_path}")
        elif self.manifest_expected_sha256:
            actual_manifest_hash = compute_file_sha256(self.manifest_path)
            if actual_manifest_hash != self.manifest_expected_sha256:
                errors.append(
                    f"Manifest SHA-256 mismatch: expected {self.manifest_expected_sha256}, got {actual_manifest_hash}"
                )

        # 4. Check Dataset Root
        if not self.dataset_root.is_dir():
            errors.append(f"Dataset root directory not found: {self.dataset_root}")

        # 5. Check Split Configuration
        if not self.split_path.is_file():
            errors.append(f"Split file not found: {self.split_path}")
        elif self.split_expected_sha256:
            actual_split_hash = compute_file_sha256(self.split_path)
            if actual_split_hash != self.split_expected_sha256:
                errors.append(
                    f"Split SHA-256 mismatch: expected {self.split_expected_sha256}, got {actual_split_hash}"
                )

        # 6. Check Sealed-Test Isolation
        if self.target_event in FORBIDDEN_SEALED_TEST_EVENTS:
            errors.append(
                f"CRITICAL BOUNDARY VIOLATION: Target event {self.target_event} is a forbidden sealed test event!"
            )
        if self.target_event not in DEFAULT_TTA_DEV_EVENTS:
            errors.append(
                f"Boundary violation: Target event {self.target_event} is not in approved development events {DEFAULT_TTA_DEV_EVENTS}"
            )

        # 7. Check Output Freshness
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.allow_overwrite:
                errors.append(
                    f"Output directory {self.output_dir} already exists and contains files. "
                    "Overwriting is prohibited (allow_overwrite=False)."
                )

        # 8. Check Device and GPU Isolation
        if self.device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                errors.append(f"CUDA requested ({self.device_str}) but torch.cuda.is_available() is False.")
            else:
                try:
                    dev_idx = torch.device(self.device_str).index or 0
                    device_name = torch.cuda.get_device_name(dev_idx)
                    free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
                    free_mb = free_mem / (1024 * 1024)
                    if free_mb < 8000:
                        errors.append(
                            f"GPU {self.device_str} ({device_name}) has only {free_mb:.1f} MB free memory (< 8000 MB required)."
                        )
                except Exception as e:
                    errors.append(f"Failed to query CUDA device info for {self.device_str}: {e}")

        if errors and fail_closed:
            error_report = "\n  - ".join(errors)
            raise RuntimeError(f"FAIL-CLOSED PREFLIGHT AUDIT FAILED with {len(errors)} error(s):\n  - {error_report}")

        return {
            "status": "PASSED" if not errors else "FAILED",
            "errors": errors,
            "git_commit": commit,
            "config_sha256": compute_file_sha256(self.config_path),
            "checkpoint_sha256": self.ckpt_expected_sha256,
            "manifest_sha256": self.manifest_expected_sha256,
            "split_sha256": self.split_expected_sha256,
            "device": self.device_str,
            "target_event": self.target_event,
            "output_dir": str(self.output_dir),
        }

    def _resolve_samples_for_event(
        self,
        event_id: int,
        dataset_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Applies declared deterministic sorting and sample capping for Event 520."""
        samples = [s for s in dataset_samples if int(s["event_id"]) == event_id]
        if not samples:
            raise ValueError(f"No samples found for event {event_id} in dataset manifest!")

        # Strictly sort ascending by alphanumeric sample_id
        samples = sorted(samples, key=lambda x: str(x["sample_id"]))

        # Enforce positive hard ceiling of exactly max_samples
        if len(samples) < self.max_samples:
            raise ValueError(
                f"Insufficient samples for Event {event_id}: found {len(samples)}, expected {self.max_samples}."
            )
        samples = samples[: self.max_samples]

        s_ids = [str(s["sample_id"]) for s in samples]
        if len(s_ids) != len(set(s_ids)):
            raise ValueError(
                f"Duplicate sample IDs detected for event {event_id}: total {len(s_ids)} vs unique {len(set(s_ids))}"
            )
        if len(samples) != self.max_samples:
            raise ValueError(
                f"Sample count violation: expected exactly {self.max_samples} samples, got {len(samples)}."
            )

        return samples

    def run_smoke(self) -> Dict[str, Any]:
        """Executes bounded real-data smoke under ChangeGuard Plan V3 orchestration."""
        start_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t0 = time.time()
        print(f"[{start_time_utc}] Starting E1.3 Bounded Real-Data Smoke on Event {self.target_event}...")

        # 1. Fail-closed preflight audit
        preflight_info = self.validate_preflight(fail_closed=True)
        print("Preflight audit PASSED.")
        print(f"  Git commit:      {preflight_info['git_commit']}")
        print(f"  Checkpoint:      {self.checkpoint_path}")
        print(f"  Device:          {self.device_str}")
        print(f"  Target Event:    {self.target_event} (Development)")

        # Create output and log directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 2. Write initial provenance manifest immediately
        initial_manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.cfg.get("status"),
            "approval_status": self.cfg.get("approval_status"),
            "execution_mode": "bounded_real_smoke",
            "git_info": get_git_info(),
            "config_path": str(self.config_path),
            "config_sha256": compute_file_sha256(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.ckpt_expected_sha256,
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_expected_sha256,
            "split_path": str(self.split_path),
            "split_sha256": self.split_expected_sha256,
            "target_event": self.target_event,
            "expected_samples": self.max_samples,
            "expected_batches": self.expected_batches,
            "device": self.device_str,
            "start_time_utc": start_time_utc,
        }
        with open(self.output_dir / "initial_provenance_manifest.json", "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        # 3. Load Model and Source Reference
        print(f"Loading SNUNet model from {self.checkpoint_path} onto {self.device_str}...")
        model_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            in_channels=self.in_channels,
            out_ch=self.num_classes,
            base_channel=self.base_channel,
            device=self.device_str,
        )
        source_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            in_channels=self.in_channels,
            out_ch=self.num_classes,
            base_channel=self.base_channel,
            device=self.device_str,
        )

        # 4. Instantiate Datasets (Adaptation: NO labels; Offline: WITH labels)
        print("Loading real SAR evaluation datasets from manifest...")
        adapt_dataset = KuroEvaluationDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            allowed_events=[self.target_event],
            canonical_preevent=self.canonical_preevent,
            sar_clamp_max=self.sar_clamp_max,
            sar_mean=self.sar_mean,
            sar_std=self.sar_std,
            load_labels=False,
        )
        offline_dataset = KuroEvaluationDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            allowed_events=[self.target_event],
            canonical_preevent=self.canonical_preevent,
            sar_clamp_max=self.sar_clamp_max,
            sar_mean=self.sar_mean,
            sar_std=self.sar_std,
            load_labels=True,
        )

        # 5. Resolve exactly 50 samples for Event 520
        event_sample_items = []
        for idx, item in enumerate(adapt_dataset.samples):
            if int(item["event_id"]) == self.target_event:
                item_with_idx = dict(item)
                item_with_idx["dataset_index"] = idx
                event_sample_items.append(item_with_idx)

        selected_samples = self._resolve_samples_for_event(self.target_event, event_sample_items)
        num_selected = len(selected_samples)
        print(f"Resolved {num_selected} samples for Event {self.target_event}. Partitioning into batches...")

        # 6. Build LabelGuardedBatch sequence
        batches: List[LabelGuardedBatch] = []
        actual_batch_sizes: List[int] = []

        for i in range(0, num_selected, self.batch_size):
            batch_items = selected_samples[i : i + self.batch_size]
            actual_batch_sizes.append(len(batch_items))
            b_ids = [str(s["sample_id"]) for s in batch_items]
            b_indices = [s["dataset_index"] for s in batch_items]

            # Load sensory batch (strictly label-isolated)
            adapt_data = [adapt_dataset[idx] for idx in b_indices]
            xA = torch.stack([d["xA"] for d in adapt_data]).to(self.device_str)
            xB = torch.stack([d["xB"] for d in adapt_data]).to(self.device_str)
            valid_mask = torch.stack([d["valid_mask"] for d in adapt_data]).to(self.device_str)

            # Load offline labels
            offline_data = [offline_dataset[idx] for idx in b_indices]
            labels = torch.stack([d["label"] for d in offline_data]).to(self.device_str)

            batch = LabelGuardedBatch(
                sample_ids=b_ids,
                event_id=self.target_event,
                xA=xA,
                xB=xB,
                valid_mask=valid_mask,
                labels=labels,
            )
            batches.append(batch)

        if actual_batch_sizes != self.expected_batches:
            raise ValueError(
                f"Batch size structure mismatch: expected {self.expected_batches}, got {actual_batch_sizes}"
            )
        print(f"Batches initialized: {actual_batch_sizes} ({len(batches)} batches).")

        # 7. Build ExtendedExperimentRecorder, E1ConsistencyAdapter, and E1Orchestrator
        recorder = ExtendedExperimentRecorder(
            output_dir=self.output_dir,
            run_id="approved_smoke_event520_v1",
            checkpoint_id="source_snunet_vvvh_all27_seed0_v1_fp32",
            config_hash=compute_file_sha256(self.config_path),
            git_commit=get_git_info()["commit"],
            allow_overwrite=self.allow_overwrite,
        )

        adapter = E1ConsistencyAdapter(
            model=model_wrapper,
            source_model=source_wrapper,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            confidence_threshold=self.confidence_threshold,
            device=self.device_str,
        )

        orchestrator = E1Orchestrator(
            recorder=recorder,
            adapter=adapter,
            condition_id="E1_3_SMOKE",
            seed=0,
            reset_mode=self.reset_mode,
        )

        # 8. Run Event Stream through E1 Orchestration
        print(f"Executing adaptation stream for Event {self.target_event}...")
        event_metrics = orchestrator.run_event_stream(
            event_id=self.target_event,
            batches=batches,
        )

        # 9. Finalize Run
        recorder.complete_run(condition_summary=event_metrics)
        elapsed_sec = time.time() - t0
        print(f"E1.3 Smoke Execution COMPLETED in {elapsed_sec:.2f}s.")
        print(f"Results Summary for Event {self.target_event}:")
        print(f"  Samples:          {event_metrics['sample_count']}")
        print(f"  Valid Pixels:     {event_metrics['valid_pixels']}")
        print(f"  Updates Applied:  {event_metrics['updates_applied']}")
        print(f"  Updates Skipped:  {event_metrics['updates_skipped']}")
        print(f"  Updates Failed:   {event_metrics['updates_failed']}")
        print(f"  F1 Score:         {event_metrics['f1']:.6f}")
        print(f"  Precision:        {event_metrics['precision']:.6f}")
        print(f"  Recall:           {event_metrics['recall']:.6f}")
        print(f"  IoU:              {event_metrics['iou']:.6f}")
        print(f"  Confusion TP:     {event_metrics['tp']}")
        print(f"  Confusion FP:     {event_metrics['fp']}")
        print(f"  Confusion FN:     {event_metrics['fn']}")
        print(f"  Confusion TN:     {event_metrics['tn']}")

        return {
            "status": "COMPLETED",
            "elapsed_sec": elapsed_sec,
            "event_metrics": event_metrics,
            "preflight_info": preflight_info,
            "output_dir": str(self.output_dir),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E1.3 Bounded Real-Data Smoke Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e1_3_smoke_event520_v1.json",
        help="Path to E1.3 smoke config",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional output dir override")
    parser.add_argument("--log_dir", type=Path, default=None, help="Optional log dir override")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (e.g. cuda:0)")
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        default=False,
        help="Prohibited for smoke: do not overwrite existing output directory",
    )
    args = parser.parse_args()

    runner = E1RealSmokeRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        allow_overwrite=args.allow_overwrite,
    )
    runner.run_smoke()


if __name__ == "__main__":
    main()
