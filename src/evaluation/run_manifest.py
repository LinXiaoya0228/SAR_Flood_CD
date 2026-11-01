"""
Run Manifest and Research Integrity Helper for ChangeGuard.

Guarantees full experimental provenance, auditability, and enforcement
of research integrity rules (data boundaries, label isolation, and run completeness).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

DEFAULT_SEALED_TEST_EVENTS = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
DEFAULT_TTA_VAL_EVENTS = [279, 437, 520, 559, 1111003, 1111008]


def get_git_commit(repo_dir: Optional[Union[str, Path]] = None) -> str:
    """Retrieve the current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def is_git_dirty(repo_dir: Optional[Union[str, Path]] = None) -> bool:
    """Check if the git working tree has uncommitted modifications."""
    try:
        cmd = ["git", "status", "--porcelain"]
        output = subprocess.check_output(cmd, cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip()
        return len(output) > 0
    except Exception:
        return False


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file if it exists, otherwise return 'FILE_NOT_FOUND'."""
    p = Path(filepath)
    if not p.is_file():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"HASH_ERROR: {str(e)}"


def get_gpu_info() -> str:
    """Retrieve GPU device information via PyTorch if available."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return f"CUDA ({count} device(s): {', '.join(names)})"
        return "CPU"
    except Exception:
        return "UNKNOWN"


class RunManifest:
    """
    Standardized reproducibility and compliance manifest.
    
    Validates protocol rules upon instantiation:
    - Rejects any overlap between accessed events and sealed test events.
    - Rejects any run where target labels are indicated as used for updates (TTA mode).
    - Rejects any overlap with quarantined TTA validation events (source training mode).
    """

    def __init__(
        self,
        run_id: str,
        config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        data_manifest: Union[str, Path],
        accessed_event_ids: Sequence[int],
        sealed_test_events: Optional[Sequence[int]] = None,
        random_seed: int = 42,
        labels_available_to_adaptation: bool = False,
        labels_used_for_updates: bool = False,
        is_partial_run: bool = False,
        expected_sample_count: int = 0,
        processed_sample_count: int = 0,
        output_paths: Optional[Dict[str, Union[str, Path]]] = None,
        modified_protocol_fields: Optional[List[str]] = None,
        repo_dir: Optional[Union[str, Path]] = None,
        completion_status: str = "INITIALIZED",
        run_type: str = "tta",
        config_hash: Optional[str] = None,
        quarantined_events: Optional[Sequence[int]] = None,
        protocol_hash: Optional[str] = None,
        lock_hash: Optional[str] = None,
        split_hash: Optional[str] = None,
        slope_stats_hash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.run_id = str(run_id)
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.config_path = str(config_path)
        self.config_hash = str(config_hash) if config_hash else compute_file_sha256(self.config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.data_manifest = str(data_manifest)
        self.accessed_event_ids = sorted(list(accessed_event_ids))
        self.sealed_test_events = sorted(
            list(sealed_test_events if sealed_test_events is not None else DEFAULT_SEALED_TEST_EVENTS)
        )
        self.run_type = str(run_type).lower()
        self.quarantined_events = sorted(
            list(quarantined_events if quarantined_events is not None else DEFAULT_TTA_VAL_EVENTS)
        )
        self.random_seed = int(random_seed)
        self.labels_available_to_adaptation = bool(labels_available_to_adaptation)
        self.labels_used_for_updates = bool(labels_used_for_updates)
        self.is_partial_run = bool(is_partial_run)
        self.expected_sample_count = int(expected_sample_count)
        self.processed_sample_count = int(processed_sample_count)
        self.output_paths = {k: str(v) for k, v in (output_paths or {}).items()}
        self.modified_protocol_fields = list(modified_protocol_fields or [])
        self.completion_status = completion_status
        self.protocol_hash = str(protocol_hash) if protocol_hash else None
        self.lock_hash = str(lock_hash) if lock_hash else None
        self.split_hash = str(split_hash) if split_hash else None
        self.slope_stats_hash = str(slope_stats_hash) if slope_stats_hash else None
        self.extra_metadata = dict(extra_metadata or {})

        # System provenance
        self.repo_dir = repo_dir
        self.git_commit = get_git_commit(repo_dir)
        self.git_dirty = is_git_dirty(repo_dir)
        self.python_executable = sys.executable
        self.environment = f"{platform.system()} {platform.release()} ({platform.machine()})"
        self.gpu_device = get_gpu_info()
        self.command = " ".join(sys.argv)
        self.checkpoint_hash = compute_file_sha256(self.checkpoint_path)

        # Enforce compliance
        self.validate()

    def validate(self) -> None:
        """Enforce strict research-integrity constraints."""
        # 1. Sealed test event boundary
        overlap = set(self.accessed_event_ids).intersection(set(self.sealed_test_events))
        if overlap:
            raise ValueError(
                f"RESEARCH INTEGRITY VIOLATION: Accessed event IDs {sorted(list(overlap))} "
                f"belong to sealed test events {self.sealed_test_events}. "
                f"Accessing sealed test events during development is strictly prohibited!"
            )

        # 2. Quarantined TTA validation boundary (for source training)
        if self.run_type in ("source_training", "all_source_training"):
            tta_overlap = set(self.accessed_event_ids).intersection(set(self.quarantined_events))
            if tta_overlap:
                raise ValueError(
                    f"RESEARCH INTEGRITY VIOLATION: Accessed event IDs {sorted(list(tta_overlap))} "
                    f"belong to quarantined TTA-validation events {self.quarantined_events}. "
                    f"Accessing TTA-validation events during source training is strictly prohibited!"
                )
            if "test" in Path(self.data_manifest).name.lower():
                raise ValueError(
                    f"RESEARCH INTEGRITY VIOLATION: Source training run declared test data manifest: {self.data_manifest}"
                )

        # 3. Label isolation (for TTA)
        if self.run_type == "tta" and self.labels_used_for_updates:
            raise ValueError(
                "RESEARCH INTEGRITY VIOLATION: Experiment declared that target labels were used "
                "for model updates. Unsupervised test-time adaptation methods must NEVER "
                "use target ground-truth labels for optimization or updates!"
            )

    def mark_completed(self, status: str = "SUCCESS") -> None:
        """Update run completion status."""
        self.completion_status = status

    def update_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Update checkpoint path and its verified hash."""
        p = Path(checkpoint_path)
        if p.exists():
            self.checkpoint_path = str(p)
            self.checkpoint_hash = compute_file_sha256(p)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a serializable dictionary."""
        rep = {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp,
            "run_type": self.run_type,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "python_executable": self.python_executable,
            "environment": self.environment,
            "gpu_device": self.gpu_device,
            "random_seed": self.random_seed,
            "command": self.command,
            "config_path": self.config_path,
            "config_sha256": self.config_hash,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_hash,
            "data_manifest": self.data_manifest,
            "event_ids": self.accessed_event_ids,
            "output_paths": self.output_paths,
            "completion_status": self.completion_status,
        }
        if self.protocol_hash is not None:
            rep["protocol_sha256"] = self.protocol_hash
        if self.lock_hash is not None:
            rep["lock_sha256"] = self.lock_hash
        if self.split_hash is not None:
            rep["split_sha256"] = self.split_hash
        if self.slope_stats_hash is not None:
            rep["slope_stats_sha256"] = self.slope_stats_hash
        if self.extra_metadata:
            rep["extra_metadata"] = self.extra_metadata

        return {
            "reproducibility": rep,
            "compliance": {
                "sealed_test_events": self.sealed_test_events,
                "quarantined_tta_events": self.quarantined_events if self.run_type == "source_training" else None,
                "accessed_event_ids": self.accessed_event_ids,
                "sealed_event_accessed": False,
                "quarantined_tta_event_accessed": False if self.run_type == "source_training" else None,
                "labels_available_to_adaptation": self.labels_available_to_adaptation,
                "labels_used_for_updates": self.labels_used_for_updates,
                "is_partial_run": self.is_partial_run,
                "expected_sample_count": self.expected_sample_count,
                "processed_sample_count": self.processed_sample_count,
                "sample_count_matched": (
                    self.expected_sample_count == self.processed_sample_count
                    if not self.is_partial_run
                    else None
                ),
                "modified_protocol_fields": self.modified_protocol_fields,
            },
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save the manifest as a formatted JSON document."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path
