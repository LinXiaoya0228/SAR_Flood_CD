"""ChangeGuard Week 11 Real-Data Smoke Infrastructure Runner.

Governing Protocol: docs/protocols/week11_evidence_consistency_protocol_v1.md
Governing Draft Config: configs/changeguard/week11_smoke_event520_draft.json

Authorizations & Fail-Closed Safeguards (W11C-R01 through W11C-R07, v2):
1. Authorization & Stage Enforcement (R04):
   - Config field 'execution_authorizations' must define JSON boolean flags:
     'real_data_execution_authorized' and 'gpu_execution_authorized'.
   - Permitted synthetic statuses: DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED,
     APPROVED_FOR_CPU_SYNTHETIC_SMOKE_VERIFICATION.
   - Real-data execution requires status 'APPROVED_FOR_REAL_DATA_SMOKE' and
     approval_status 'APPROVED_FOR_EXECUTION'. Draft/missing/null/unknown status
     fails closed before asset or device access. Boolean flags cannot override draft status.
   - Device syntax enforcement:
     * Synthetic/mock execution strictly requires device 'cpu'.
     * Real-data execution strictly requires format 'cuda:<non-negative integer>' (e.g. 'cuda:0').
     * Bare 'cuda', 'cpu', 'mps', negative indices, malformed strings, and arbitrary names
       are rejected before asset or device access, without initializing CUDA.
2. Bounded Smoke Identity Enforcement (R05):
   - Strictly enforces seed 0, Event 520, stream_order 'order_a', reset_mode 'reset_per_event',
     batch_size 16, max_samples 50 (hard ceiling), sample_selection_rule
     'deterministic_sorted_sample_id_ascending_head_50', and expected batches [16, 16, 16, 2].
   - Rejects unbounded (<= 0), oversized, reversed (order_b), or contradictory settings in preflight.
3. Path Resolution & Path Metadata Validation (R01):
   - Resolves dataset_root and manifest_path against canonical frozen locations.
   - In real mode, preflight validates that dataset_root is a directory and manifest_path is a file
     before model or device work, without reading raster or manifest byte contents.
4. Pure Synthetic Asset Independence & Complete Provenance (R06 & R02):
   - In synthetic_mock mode, zero physical asset reads occur (no checkpoint hashing or raster reads).
   - Provenance records 'MOCKED_SYNTHETIC' and 'MOCKED_NOT_APPLICABLE'.
   - In real mode, validates and records actual checkpoint SHA-256 and input-manifest SHA-256.
5. Phase-Aware Lifecycle & Durable Progress (R02):
   - Writes initial_manifest.json on start with code/commit/config hashes and provenance.
   - Saves adaptation_progress.json after EVERY completed adaptation batch (batch index, completed IDs,
     actions, and cumulative counters).
   - Saves frozen_predictions.npz, adaptation_step_records.json/csv, and adaptation_progress.json
     BEFORE any ground truth access.
   - Covers initialization, adaptation, persistence, and offline scoring in phase-aware try/except.
   - On ANY failure, writes execution_failure.json with terminal failure details, failing phase,
     and last durable progress. A partial run is never marked complete.
   - Writes final completion_status.json and smoke_manifest.json only upon successful completion.
6. Confusion Matrix Validity & True Negatives (R03):
   - Restricts TN to valid pixels: tn = sum(~p_flood & ~g_flood & valid).
   - Asserts confusion-cell sum (tp + fp + fn + tn) equals evaluated valid pixels.
7. Output Protection:
   - Output directory collision protection: refuses to overwrite non-empty output directory
     unless --allow_overwrite is explicitly provided.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter
from src.data.evaluation_dataset import (
    KuroEvaluationDataset,
    FORBIDDEN_SEALED_TEST_EVENTS,
    DEFAULT_TTA_DEV_EVENTS,
)
from src.models.snunet_wrapper import SNUNetWrapper


SUPPORTED_SYNTHETIC_STATUSES: Set[str] = {
    "DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED",
    "APPROVED_FOR_CPU_SYNTHETIC_SMOKE_VERIFICATION",
}

SUPPORTED_REAL_STATUSES: Set[str] = {
    "APPROVED_FOR_REAL_DATA_SMOKE",
}

CUDA_DEVICE_REGEX = re.compile(r"^cuda:(0|[1-9]\d*)$")

REVIEWED_CODE_FILES: List[str] = [
    "src/changeguard/adapter.py",
    "tools/run_week11_real_data_smoke.py",
    "tools/run_week11_cpu_evidence.py",
    "configs/changeguard/week11_smoke_event520_draft.json",
    "configs/changeguard/week11_evidence_consistency_v1.json",
    "docs/protocols/week11_evidence_consistency_protocol_v1.md",
]


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
            .decode("ascii")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        # Record reviewed code file hashes for exact implementation identity
        code_hashes = {}
        for rel_f in REVIEWED_CODE_FILES:
            f_p = REPO_ROOT / rel_f
            if f_p.is_file():
                code_hashes[rel_f] = compute_file_sha256(f_p)
            else:
                code_hashes[rel_f] = "MISSING"

        return {
            "commit": commit,
            "is_clean": len(status) == 0,
            "reviewed_code_hashes": code_hashes,
        }
    except Exception as e:
        return {"commit": "unknown", "is_clean": False, "error": str(e)}


class MockSNUNet(nn.Module):
    """Synthetic proxy model matching SNUNet multi-temporal interface for CPU testing."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)


class Week11RealDataSmokeRunner:
    """Fail-closed real-data smoke runner for ChangeGuard Week 11."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cpu",
        synthetic_mock: bool = False,
        allow_overwrite: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.synthetic_mock = bool(synthetic_mock)
        self.allow_overwrite = bool(allow_overwrite)
        self.device_str = str(device).lower()

        # Output directory resolution
        if output_dir is not None:
            self.output_dir = output_dir.resolve()
        else:
            rel_out = self.cfg.get("outputs", {}).get("smoke_dir", "outputs/week11_real_smoke/draft_smoke_v1")
            self.output_dir = (REPO_ROOT / rel_out).resolve()

        self.log_dir = (REPO_ROOT / self.cfg.get("outputs", {}).get("log_dir", "logs/week11_real_smoke")).resolve()

        # Status and stage metadata
        self.status = self.cfg.get("status")
        self.approval_status = self.cfg.get("approval_status")

        # Dataset configuration
        d_cfg = self.cfg.get("dataset", {})
        self.target_events: List[int] = [int(e) for e in d_cfg.get("target_events", [520])]
        self.development_events: List[int] = [int(e) for e in d_cfg.get("development_events", DEFAULT_TTA_DEV_EVENTS)]
        self.forbidden_events: Set[int] = set(
            int(e) for e in d_cfg.get("forbidden_sealed_test_events", list(FORBIDDEN_SEALED_TEST_EVENTS))
        )
        self.canonical_preevent: str = d_cfg.get("canonical_preevent", "pre_event_2")
        self.sar_mean = tuple(d_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = tuple(d_cfg.get("sar_std", [0.0427, 0.0215]))
        self.sar_clamp_max = float(d_cfg.get("sar_clamp_max", 0.15))

        # Resolve dataset paths (R01)
        raw_root = d_cfg.get("dataset_root", "../datasets/KuroSiwo")
        raw_manifest = d_cfg.get("manifest_path", "../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz")
        self.dataset_root = self._resolve_path(raw_root)
        self.manifest_path = self._resolve_path(raw_manifest)

        # Smoke specification (R05)
        s_cfg = self.cfg.get("smoke_specification", {})
        self.checkpoint_seed = int(s_cfg.get("checkpoint_seed", 0))
        self.stream_order = s_cfg.get("stream_order", "order_a")
        self.reset_mode = s_cfg.get("reset_mode", "reset_per_event")
        self.max_samples = int(s_cfg.get("max_samples", 50))
        self.sample_selection_rule = s_cfg.get(
            "sample_selection_rule", "deterministic_sorted_sample_id_ascending_head_50"
        )
        self.batch_size = int(self.cfg.get("adaptation_hyperparameters", {}).get("batch_size", 16))

        # Model configuration
        m_cfg = self.cfg.get("model", {})
        self.model_architecture = m_cfg.get("architecture", "SNUNet_ECAM")
        self.in_channels = int(m_cfg.get("in_channels", 2))
        self.num_classes = int(m_cfg.get("num_classes", 3))
        self.base_channel = int(m_cfg.get("base_channel", 32))
        self.input_channels = m_cfg.get("input_channels", ["VV", "VH"])
        self.precision = m_cfg.get("precision", "fp32")

        # Provenance digest holders
        self.manifest_sha256: Optional[str] = None
        self.ckpt_actual_sha256: Optional[str] = None
        self.ckpt_expected_sha256: Optional[str] = None

        # Enforce authorizations early
        self._enforce_authorizations()

    def _resolve_path(self, path_str: str) -> Path:
        """Resolves path supporting absolute, REPO_ROOT-relative, and parent-relative spellings."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        if (REPO_ROOT / p).exists():
            return (REPO_ROOT / p).resolve()
        return (REPO_ROOT.parent / p).resolve()

    def _enforce_authorizations(self) -> None:
        """Validates authorization flags and enforces strict execution boundaries (R04)."""
        auth = self.cfg.get("execution_authorizations")
        if not isinstance(auth, dict):
            raise TypeError("Configuration missing or invalid 'execution_authorizations' dictionary.")

        real_auth = auth.get("real_data_execution_authorized")
        gpu_auth = auth.get("gpu_execution_authorized")

        if not isinstance(real_auth, bool):
            raise TypeError(f"'real_data_execution_authorized' must be a JSON boolean, got {type(real_auth)}.")
        if not isinstance(gpu_auth, bool):
            raise TypeError(f"'gpu_execution_authorized' must be a JSON boolean, got {type(gpu_auth)}.")

        # Validate status presence
        if self.status is None:
            raise ValueError("Configuration 'status' is missing or null.")
        if not isinstance(self.status, str):
            raise TypeError(f"Configuration 'status' must be a string, got {type(self.status)}.")

        # Synthetic mock mode
        if self.synthetic_mock:
            if self.device_str != "cpu":
                raise PermissionError(f"Synthetic mock execution supports 'cpu' only, got '{self.device_str}'.")
            if self.status not in SUPPORTED_SYNTHETIC_STATUSES and self.status not in SUPPORTED_REAL_STATUSES:
                raise ValueError(f"Unsupported configuration status for synthetic mock: '{self.status}'.")
            return

        # Real-data execution path (strictly fail-closed if draft or unauthorized)
        if self.status == "DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED" or self.status.startswith("DRAFT_"):
            raise PermissionError(
                f"Real-data execution is rejected: configuration status is '{self.status}'. "
                "Boolean flags cannot override draft status."
            )

        if self.status not in SUPPORTED_REAL_STATUSES:
            raise ValueError(f"Unsupported configuration status for real execution: '{self.status}'.")

        if self.approval_status != "APPROVED_FOR_EXECUTION":
            raise PermissionError(
                f"Real-data execution is rejected: approval_status is '{self.approval_status}' "
                "(requires 'APPROVED_FOR_EXECUTION')."
            )

        if not real_auth:
            raise PermissionError(
                "Real-data execution is NOT authorized. "
                "'real_data_execution_authorized' is false in configuration."
            )

        # R04: Device syntax validation for real execution
        if not CUDA_DEVICE_REGEX.match(self.device_str):
            raise ValueError(
                f"Real smoke execution strictly requires device format 'cuda:<non-negative integer>' (e.g. 'cuda:0'), "
                f"got '{self.device_str}'. Bare 'cuda', 'cpu', 'mps', negative indices and arbitrary names are rejected."
            )

        if not gpu_auth:
            raise PermissionError(
                f"GPU device '{self.device_str}' requested, but "
                "'gpu_execution_authorized' is false in configuration."
            )

    def validate_preflight(self, fail_closed: bool = True) -> Dict[str, Any]:
        """Audits configuration, bounded identity, event boundaries, path metadata, and outputs."""
        # 1. Authorizations & Stage audit (R04)
        self._enforce_authorizations()

        # 2. Bounded Smoke Identity Audit (R05)
        if self.checkpoint_seed != 0:
            raise ValueError(
                f"Week 11 bounded smoke strictly requires checkpoint_seed 0, got {self.checkpoint_seed}."
            )
        if self.target_events != [520]:
            raise ValueError(
                f"Week 11 bounded smoke strictly requires target_events [520], got {self.target_events}."
            )
        if self.stream_order != "order_a":
            raise ValueError(
                f"Week 11 bounded smoke strictly requires stream_order 'order_a', got '{self.stream_order}'."
            )
        if self.reset_mode != "reset_per_event":
            raise ValueError(
                f"Week 11 bounded smoke strictly requires reset_mode 'reset_per_event', got '{self.reset_mode}'."
            )
        if self.batch_size != 16:
            raise ValueError(
                f"Week 11 bounded smoke strictly requires batch_size 16, got {self.batch_size}."
            )
        if self.max_samples != 50:
            raise ValueError(
                f"Week 11 bounded smoke strictly requires max_samples 50 (hard ceiling), got {self.max_samples}."
            )
        if self.sample_selection_rule != "deterministic_sorted_sample_id_ascending_head_50":
            raise ValueError(
                f"Week 11 bounded smoke strictly requires sample_selection_rule "
                f"'deterministic_sorted_sample_id_ascending_head_50', got '{self.sample_selection_rule}'."
            )

        # Frozen model & preprocessing invariants
        if self.model_architecture != "SNUNet_ECAM":
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")
        if self.in_channels != 2 or self.num_classes != 3 or self.base_channel != 32:
            raise ValueError(
                f"Model channel/class invariants violated: in_channels={self.in_channels}, "
                f"num_classes={self.num_classes}, base_channel={self.base_channel}"
            )
        if self.input_channels != ["VV", "VH"]:
            raise ValueError(f"Input channels must be ['VV', 'VH'], got {self.input_channels}")
        if self.canonical_preevent != "pre_event_2":
            raise ValueError(f"Canonical preevent must be 'pre_event_2', got '{self.canonical_preevent}'")
        if self.precision != "fp32":
            raise ValueError(f"Precision must be 'fp32', got '{self.precision}'")
        if self.sar_mean != (0.0953, 0.0264) or self.sar_std != (0.0427, 0.0215) or self.sar_clamp_max != 0.15:
            raise ValueError("Frozen SAR normalization constants altered in configuration.")

        # 3. Event quarantine audit
        target_set = set(self.target_events)
        forbidden_intersection = target_set.intersection(self.forbidden_events)
        if forbidden_intersection:
            raise RuntimeError(
                f"CRITICAL RESEARCH INTEGRITY VIOLATION: Target events {self.target_events} "
                f"overlap with sealed test events: {sorted(list(forbidden_intersection))}!"
            )

        dev_set = set(self.development_events)
        out_of_dev = target_set.difference(dev_set)
        if out_of_dev:
            raise RuntimeError(
                f"Scope violation: Target events {self.target_events} contain events outside "
                f"approved development events: {sorted(list(out_of_dev))}!"
            )

        # 4. Path metadata validation & digest propagation (R01, R02 & R06)
        dataset_root_valid = False
        manifest_path_valid = False
        ckpt_path_str: Optional[str] = None
        ckpt_exists = False
        ckpt_hash_valid = False
        ckpt_error: Optional[str] = None

        if self.synthetic_mock:
            # R06: Synthetic mode performs ZERO physical asset reads
            dataset_root_valid = True
            manifest_path_valid = True
            ckpt_path_str = "MOCKED_SYNTHETIC"
            ckpt_exists = True
            ckpt_hash_valid = True
            self.ckpt_expected_sha256 = "MOCKED_NOT_APPLICABLE"
            self.ckpt_actual_sha256 = "MOCKED_NOT_APPLICABLE"
            self.manifest_sha256 = "MOCKED_NOT_APPLICABLE"
        else:
            # R01 & R02: Real mode validates path metadata and calculates digests
            if not self.dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
            dataset_root_valid = True

            if not self.manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
            manifest_path_valid = True
            self.manifest_sha256 = compute_file_sha256(self.manifest_path)

            # Real checkpoint verification
            ckpt_info = self.cfg.get("source_checkpoints", {}).get("seeds", {}).get(str(self.checkpoint_seed), {})
            rel_ckpt = ckpt_info.get("checkpoint_path")
            exp_hash = ckpt_info.get("expected_sha256")
            self.ckpt_expected_sha256 = exp_hash
            ckpt_path = (REPO_ROOT / rel_ckpt).resolve() if rel_ckpt else None
            ckpt_path_str = str(ckpt_path) if ckpt_path else None
            ckpt_exists = ckpt_path.is_file() if ckpt_path else False

            if ckpt_exists and ckpt_path is not None:
                try:
                    actual_hash = compute_file_sha256(ckpt_path)
                    self.ckpt_actual_sha256 = actual_hash
                    ckpt_hash_valid = (actual_hash == exp_hash)
                    if not ckpt_hash_valid:
                        ckpt_error = f"Hash mismatch: actual {actual_hash} vs expected {exp_hash}"
                except Exception as e:
                    self.ckpt_actual_sha256 = "ERROR"
                    ckpt_error = str(e)
            else:
                self.ckpt_actual_sha256 = "MISSING"
                ckpt_error = f"Checkpoint file missing: {ckpt_path}"

            if fail_closed:
                if not ckpt_exists:
                    raise FileNotFoundError(f"Source checkpoint not found at {ckpt_path}")
                if not ckpt_hash_valid:
                    raise ValueError(f"Source checkpoint validation failed: {ckpt_error}")

        # 5. Output collision audit
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.allow_overwrite:
                raise FileExistsError(
                    f"Output directory {self.output_dir} already exists and contains files. "
                    "Re-running requires explicit allow_overwrite authorization."
                )

        return {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "target_events": self.target_events,
            "forbidden_events": sorted(list(self.forbidden_events)),
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_path": ckpt_path_str,
            "checkpoint_exists": ckpt_exists,
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "checkpoint_hash_valid": ckpt_hash_valid,
            "checkpoint_error": ckpt_error,
            "dataset_root": str(self.dataset_root) if not self.synthetic_mock else "MOCKED_SYNTHETIC",
            "manifest_path": str(self.manifest_path) if not self.synthetic_mock else "MOCKED_SYNTHETIC",
            "manifest_sha256": self.manifest_sha256,
            "dataset_root_valid": dataset_root_valid,
            "manifest_path_valid": manifest_path_valid,
            "output_dir": str(self.output_dir),
            "output_dir_clean": not (self.output_dir.exists() and any(self.output_dir.iterdir())),
            "allow_overwrite": self.allow_overwrite,
            "synthetic_mock": self.synthetic_mock,
            "device": self.device_str,
        }

    def _resolve_samples_for_event(
        self,
        event_id: int,
        dataset_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Applies declared deterministic sorting and sample capping for Event 520 (R05)."""
        samples = [s for s in dataset_samples if int(s["event_id"]) == event_id]
        if not samples:
            raise ValueError(f"No samples found for event {event_id}!")

        # Strictly sort ascending by alphanumeric sample_id
        samples = sorted(samples, key=lambda x: str(x["sample_id"]))

        # Enforce positive hard ceiling of exactly 50 samples
        if len(samples) < self.max_samples:
            raise ValueError(
                f"Insufficient samples for Event {event_id}: found {len(samples)}, expected {self.max_samples}."
            )
        samples = samples[: self.max_samples]

        # Verify uniqueness
        s_ids = [str(s["sample_id"]) for s in samples]
        if len(s_ids) != len(set(s_ids)):
            raise ValueError(
                f"Integrity violation: duplicate sample IDs found for event {event_id}: "
                f"total {len(s_ids)} vs unique {len(set(s_ids))}"
            )
        if len(samples) != 50:
            raise ValueError(f"Sample count violation: expected exactly 50 samples, resolved {len(samples)}.")

        return samples

    def run_smoke(self, mock_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Executes the bounded real-data smoke with phase-aware exception handling & durable progress (R02)."""
        start_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t0 = time.time()
        current_phase = "PREFLIGHT"
        latest_progress_data: Optional[Dict[str, Any]] = None

        # Run preflight audit
        preflight_info = self.validate_preflight(fail_closed=True)

        # Build output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # R02: Write initial provenance manifest immediately upon start with complete digests
        initial_manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "execution_mode": "synthetic_mock" if self.synthetic_mock else "real",
            "git_info": get_git_info(),
            "config_path": str(self.config_path),
            "config_sha256": compute_file_sha256(self.config_path),
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_path": preflight_info.get("checkpoint_path"),
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "dataset_root": preflight_info.get("dataset_root"),
            "manifest_path": preflight_info.get("manifest_path"),
            "manifest_sha256": self.manifest_sha256,
            "target_event": self.target_events[0],
            "sample_selection_rule": self.sample_selection_rule,
            "expected_samples": 50,
            "expected_batches": [16, 16, 16, 2],
            "execution_authorizations": self.cfg.get("execution_authorizations"),
            "device": self.device_str,
            "lifecycle_state": "INITIALIZED",
            "start_time_utc": start_time_utc,
        }
        with open(self.output_dir / "initial_manifest.json", "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        try:
            current_phase = "INITIALIZATION"
            # Resolve model
            if self.synthetic_mock:
                wrapper = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
                wrapper.eval()
                wrapper.to(self.device_str)
            else:
                ckpt_info = self.cfg["source_checkpoints"]["seeds"][str(self.checkpoint_seed)]
                ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
                wrapper, _ = SNUNetWrapper.load_from_checkpoint(
                    checkpoint_path=str(ckpt_path),
                    in_channels=self.in_channels,
                    out_ch=self.num_classes,
                    base_channel=self.base_channel,
                    device=self.device_str,
                )

            # Build EvidenceConsistencyAdapter
            h_cfg = self.cfg.get("adaptation_hyperparameters", {})
            adapter = EvidenceConsistencyAdapter(
                model=wrapper,
                learning_rate=h_cfg.get("learning_rate", 1.25e-4),
                momentum=h_cfg.get("momentum", 0.9),
                weight_decay=h_cfg.get("weight_decay", 0.0),
                confidence_threshold=self.cfg.get("consistency_views", {}).get("confidence_masking", {}).get("threshold", 0.90),
                device=self.device_str,
            )

            # Resolve dataset
            if self.synthetic_mock:
                if mock_records is None:
                    mock_records = []
                    for i in range(self.max_samples):
                        mock_records.append({
                            "sample_id": f"synthetic_520_{i:04d}",
                            "event_id": 520,
                            "xA": torch.randn((2, 224, 224), dtype=torch.float32),
                            "xB": torch.randn((2, 224, 224), dtype=torch.float32),
                            "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
                            "label": torch.zeros((224, 224), dtype=torch.int64),
                        })
                adapt_records = []
                offline_records = []
                for r in mock_records:
                    adapt_r = {
                        "sample_id": r["sample_id"],
                        "event_id": r["event_id"],
                        "xA": r["xA"],
                        "xB": r["xB"],
                        "valid_mask": r["valid_mask"],
                    }
                    adapt_records.append(adapt_r)
                    offline_records.append(r)

                adapt_dataset = KuroEvaluationDataset(records=adapt_records, allowed_events=self.target_events, load_labels=False)
                offline_dataset = KuroEvaluationDataset(records=offline_records, allowed_events=self.target_events, load_labels=True)
            else:
                adapt_dataset = KuroEvaluationDataset(
                    dataset_root=self.dataset_root,
                    manifest_path=self.manifest_path,
                    allowed_events=self.target_events,
                    canonical_preevent=self.canonical_preevent,
                    sar_clamp_max=self.sar_clamp_max,
                    sar_mean=self.sar_mean,
                    sar_std=self.sar_std,
                    load_labels=False,
                )
                offline_dataset = KuroEvaluationDataset(
                    dataset_root=self.dataset_root,
                    manifest_path=self.manifest_path,
                    allowed_events=self.target_events,
                    canonical_preevent=self.canonical_preevent,
                    sar_clamp_max=self.sar_clamp_max,
                    sar_mean=self.sar_mean,
                    sar_std=self.sar_std,
                    load_labels=True,
                )

            # Select samples for target event (Event 520)
            target_event = self.target_events[0]
            event_sample_items = []
            for idx, item in enumerate(adapt_dataset.samples):
                if int(item["event_id"]) == target_event:
                    item_with_idx = dict(item)
                    item_with_idx["dataset_index"] = idx
                    event_sample_items.append(item_with_idx)

            selected_samples = self._resolve_samples_for_event(target_event, event_sample_items)
            num_selected = len(selected_samples)

            # Reset model to source if reset_per_event
            if self.reset_mode == "reset_per_event":
                adapter.reset_to_source()

            # Execute adaptation batches
            current_phase = "ADAPTATION"
            frozen_predictions: Dict[str, np.ndarray] = {}
            step_records: List[Dict[str, Any]] = []
            batch_sizes_executed: List[int] = []
            completed_sample_ids: List[str] = []

            for i in range(0, num_selected, self.batch_size):
                batch_items = selected_samples[i : i + self.batch_size]
                batch_sizes_executed.append(len(batch_items))
                b_ids = [str(s["sample_id"]) for s in batch_items]
                b_indices = [s["dataset_index"] for s in batch_items]

                # Load sensory batch (strictly label-isolated)
                batch_data = [adapt_dataset[idx] for idx in b_indices]
                xA = torch.stack([d["xA"] for d in batch_data]).to(self.device_str)
                xB = torch.stack([d["xB"] for d in batch_data]).to(self.device_str)
                m = torch.stack([d["valid_mask"] for d in batch_data]).to(self.device_str)

                # Adapt and predict (label-free)
                step_res = adapter.adapt_and_predict(
                    xA,
                    xB,
                    sar_valid_mask=m,
                    sample_ids=b_ids,
                    event_ids=[target_event] * len(b_ids),
                )

                # Freeze primary PRE_UPDATE predictions
                preds = step_res["primary_prediction"]["predicted_class"].cpu().numpy()
                for idx, s_id in enumerate(b_ids):
                    if s_id in frozen_predictions:
                        raise ValueError(f"Integrity violation: duplicate prediction for {s_id} in event {target_event}")
                    frozen_predictions[s_id] = preds[idx]

                completed_sample_ids.extend(b_ids)

                # Record step metadata
                adapt_meta = step_res.get("adaptation_metadata", {})
                risk_meta = step_res.get("risk_evidence", {})
                step_record = {
                    "step_index": len(step_records) + 1,
                    "event_id": target_event,
                    "batch_size": len(batch_items),
                    "sample_ids": b_ids,
                    "action": adapt_meta.get("action"),
                    "update_applied": adapt_meta.get("update_applied"),
                    "failure_reason": adapt_meta.get("failure_reason"),
                    "loss": adapt_meta.get("loss"),
                    "selected_pixels": adapt_meta.get("selected_pixels"),
                    "valid_pixels": adapt_meta.get("valid_pixels"),
                    "predicted_flood_pixels": step_res["primary_prediction"].get("predicted_flood_pixels"),
                    "update_time_ms": adapt_meta.get("update_time_ms"),
                    "entropy_norm": risk_meta.get("entropy_norm"),
                    "max_confidence": risk_meta.get("max_confidence"),
                    "source_disagreement_rate_pre_update": risk_meta.get("source_disagreement_rate_pre_update"),
                    "null_score_primary_post_update": risk_meta.get("null_score_primary_post_update"),
                    "null_pre2_delta": risk_meta.get("null_pre2_delta"),
                    "null_post_delta": risk_meta.get("null_post_delta"),
                    "cand_bn_context": risk_meta.get("bn_context", {}).get("candidate_bn"),
                    "src_bn_context": risk_meta.get("bn_context", {}).get("source_bn"),
                }
                step_records.append(step_record)

                # R02: Persist progress after EVERY completed adaptation batch
                latest_progress_data = {
                    "status": "RUNNING_ADAPTATION",
                    "batches_completed": len(batch_sizes_executed),
                    "total_expected_batches": 4,
                    "batch_sizes_completed": list(batch_sizes_executed),
                    "samples_completed": len(completed_sample_ids),
                    "completed_sample_ids": list(completed_sample_ids),
                    "latest_action": adapt_meta.get("action"),
                    "updates_applied": adapter.num_updates_applied,
                    "updates_skipped": adapter.num_updates_skipped,
                    "updates_failed": adapter.num_updates_failed,
                    "total_steps": adapter.step_count,
                    "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                with open(self.output_dir / "adaptation_progress.json", "w", encoding="utf-8") as f:
                    json.dump(latest_progress_data, f, indent=2)

            # Validate expected batch sizes
            if batch_sizes_executed != [16, 16, 16, 2]:
                raise RuntimeError(
                    f"Batch execution mismatch: expected [16, 16, 16, 2], got {batch_sizes_executed}"
                )

            # ---------------------------------------------------------------------
            # R02: PERSISTENCE PHASE BEFORE ANY GT ACCESS
            # ---------------------------------------------------------------------
            current_phase = "PERSISTENCE"
            # 1. Save compressed predictions
            np.savez_compressed(self.output_dir / "frozen_predictions.npz", **frozen_predictions)

            # 2. Save pure adaptation step records
            with open(self.output_dir / "adaptation_step_records.json", "w", encoding="utf-8") as f:
                json.dump(step_records, f, indent=2)

            fieldnames_adapt = [k for k in step_records[0].keys() if k != "sample_ids"] + ["sample_ids"]
            with open(self.output_dir / "adaptation_step_records.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames_adapt)
                for rec in step_records:
                    row = []
                    for field in fieldnames_adapt:
                        val = rec.get(field)
                        if val is None:
                            row.append("null")
                        elif field == "sample_ids" and isinstance(val, list):
                            row.append(";".join(str(s) for s in val))
                        else:
                            row.append(str(val))
                    writer.writerow(row)

            # 3. Update adaptation progress record to complete
            latest_progress_data["status"] = "ADAPTATION_COMPLETED_PENDING_OFFLINE_SCORING"
            latest_progress_data["adaptation_completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(self.output_dir / "adaptation_progress.json", "w", encoding="utf-8") as f:
                json.dump(latest_progress_data, f, indent=2)

            # ---------------------------------------------------------------------
            # Separate Offline Evaluation (Strictly on Frozen Predictions)
            # ---------------------------------------------------------------------
            current_phase = "OFFLINE_SCORING"
            tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
            valid_pixels_total = 0
            predicted_flood_pixels_total = 0
            gt_flood_pixels_total = 0

            for step_rec in step_records:
                step_tp, step_fp, step_fn, step_tn = 0, 0, 0, 0
                step_valid_pixels = 0
                step_gt_flood_pixels = 0
                step_pred_flood_pixels = 0

                for s_id in step_rec["sample_ids"]:
                    pred = frozen_predictions[s_id]
                    matching_sample = next(s for s in selected_samples if str(s["sample_id"]) == s_id)
                    offline_data = offline_dataset[matching_sample["dataset_index"]]

                    gt = offline_data["label"].numpy() if torch.is_tensor(offline_data["label"]) else offline_data["label"]
                    valid_mask = (
                        offline_data["valid_mask"].squeeze().numpy()
                        if torch.is_tensor(offline_data["valid_mask"])
                        else offline_data["valid_mask"]
                    )

                    # R03: Double-masking validity: valid = (valid_mask == 1) & (gt != 3)
                    valid = (valid_mask == 1) & (gt != 3)
                    p_flood = (pred == 2) & valid
                    g_flood = (gt == 2) & valid

                    tp = int(np.sum(p_flood & g_flood))
                    fp = int(np.sum(p_flood & ~g_flood))
                    fn = int(np.sum(~p_flood & g_flood))
                    # R03: Restrict True Negatives strictly to valid pixels
                    tn = int(np.sum(~p_flood & ~g_flood & valid))

                    sample_valid_pixels = int(np.sum(valid))
                    # Assert confusion-cell sum equals valid pixels
                    assert (tp + fp + fn + tn) == sample_valid_pixels, (
                        f"Sample confusion mismatch: {tp + fp + fn + tn} != {sample_valid_pixels}"
                    )

                    step_tp += tp
                    step_fp += fp
                    step_fn += fn
                    step_tn += tn
                    step_valid_pixels += sample_valid_pixels
                    step_gt_flood_pixels += int(np.sum(g_flood))
                    step_pred_flood_pixels += int(np.sum(p_flood))

                # Step confusion sum check
                assert (step_tp + step_fp + step_fn + step_tn) == step_valid_pixels, (
                    f"Step confusion mismatch: {step_tp + step_fp + step_fn + step_tn} != {step_valid_pixels}"
                )

                step_rec["tp"] = step_tp
                step_rec["fp"] = step_fp
                step_rec["fn"] = step_fn
                step_rec["tn"] = step_tn
                step_rec["offline_valid_pixels"] = step_valid_pixels
                step_rec["gt_flood_pixels"] = step_gt_flood_pixels
                denom_prec = step_tp + step_fp
                denom_rec = step_tp + step_fn
                step_rec["flood_precision"] = float(step_tp / denom_prec) if denom_prec > 0 else 0.0
                step_rec["flood_recall"] = float(step_tp / denom_rec) if denom_rec > 0 else 0.0
                denom_f1 = 2 * step_tp + step_fp + step_fn
                step_rec["flood_f1"] = float(2 * step_tp / denom_f1) if denom_f1 > 0 else 0.0
                denom_iou = step_tp + step_fp + step_fn
                step_rec["flood_iou"] = float(step_tp / denom_iou) if denom_iou > 0 else 0.0

                tp_total += step_tp
                fp_total += step_fp
                fn_total += step_fn
                tn_total += step_tn
                valid_pixels_total += step_valid_pixels
                predicted_flood_pixels_total += step_pred_flood_pixels
                gt_flood_pixels_total += step_gt_flood_pixels

            # Event-level confusion cell sum assertion
            assert (tp_total + fp_total + fn_total + tn_total) == valid_pixels_total, (
                f"Event confusion mismatch: {tp_total + fp_total + fn_total + tn_total} != {valid_pixels_total}"
            )

            # Calculate overall event metrics
            denom_prec = tp_total + fp_total
            denom_rec = tp_total + fn_total
            event_precision = float(tp_total / denom_prec) if denom_prec > 0 else 0.0
            event_recall = float(tp_total / denom_rec) if denom_rec > 0 else 0.0
            denom_f1 = 2 * tp_total + fp_total + fn_total
            event_f1 = float(2 * tp_total / denom_f1) if denom_f1 > 0 else 0.0
            denom_iou = tp_total + fp_total + fn_total
            event_iou = float(tp_total / denom_iou) if denom_iou > 0 else 0.0

            event_row = {
                "event_id": target_event,
                "sample_count": num_selected,
                "batch_count": len(step_records),
                "batch_sizes": batch_sizes_executed,
                "tail_batch_size": batch_sizes_executed[-1] if batch_sizes_executed else 0,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
                "tn": tn_total,
                "flood_f1": event_f1,
                "flood_iou": event_iou,
                "flood_precision": event_precision,
                "flood_recall": event_recall,
                "valid_pixels": valid_pixels_total,
                "predicted_flood_pixels": predicted_flood_pixels_total,
                "gt_flood_pixels": gt_flood_pixels_total,
                "updates_applied": adapter.num_updates_applied,
                "updates_skipped": adapter.num_updates_skipped,
                "updates_failed": adapter.num_updates_failed,
                "total_steps": adapter.step_count,
            }

            # Completion Phase
            current_phase = "COMPLETION"
            self._save_outputs(
                event_row=event_row,
                step_records=step_records,
                preflight_info=preflight_info,
                selected_sample_ids=[str(s["sample_id"]) for s in selected_samples],
                start_time_utc=start_time_utc,
                elapsed_seconds=round(time.time() - t0, 3),
            )

            # Write completion status
            completion_status = {
                "status": "COMPLETED",
                "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - t0, 3),
            }
            with open(self.output_dir / "completion_status.json", "w", encoding="utf-8") as f:
                json.dump(completion_status, f, indent=2)

            return {
                "event_row": event_row,
                "step_records": step_records,
                "preflight_info": preflight_info,
                "output_dir": str(self.output_dir),
            }

        except Exception as exc:
            # R02: Phase-aware terminal failure handling. Preserves all durable progress.
            failure_record = {
                "status": "FAILED",
                "phase": current_phase,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "last_durable_progress": latest_progress_data,
            }
            with open(self.output_dir / "execution_failure.json", "w", encoding="utf-8") as f:
                json.dump(failure_record, f, indent=2)
            raise

    def _save_outputs(
        self,
        event_row: Dict[str, Any],
        step_records: List[Dict[str, Any]],
        preflight_info: Dict[str, Any],
        selected_sample_ids: List[str],
        start_time_utc: str,
        elapsed_seconds: float,
    ) -> None:
        """Serializes manifest, JSON, and CSV metrics with strict null fidelity."""
        # 1. Final Manifest
        manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "execution_mode": "synthetic_mock" if self.synthetic_mock else "real",
            "git_info": get_git_info(),
            "config_path": str(self.config_path),
            "config_sha256": compute_file_sha256(self.config_path),
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_path": preflight_info.get("checkpoint_path"),
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "dataset_root": preflight_info.get("dataset_root"),
            "manifest_path": preflight_info.get("manifest_path"),
            "manifest_sha256": self.manifest_sha256,
            "target_event": event_row["event_id"],
            "sample_count": event_row["sample_count"],
            "batch_count": event_row["batch_count"],
            "batch_sizes": event_row["batch_sizes"],
            "tail_batch_size": event_row["tail_batch_size"],
            "sample_selection_rule": self.sample_selection_rule,
            "sample_ids": selected_sample_ids,
            "execution_authorizations": self.cfg.get("execution_authorizations"),
            "device": self.device_str,
            "counters": {
                "updates_applied": event_row["updates_applied"],
                "updates_skipped": event_row["updates_skipped"],
                "updates_failed": event_row["updates_failed"],
                "total_steps": event_row["total_steps"],
            },
            "timestamps": {
                "start_utc": start_time_utc,
                "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "elapsed_seconds": elapsed_seconds,
            },
            "lifecycle_state": "COMPLETED",
        }
        with open(self.output_dir / "smoke_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 2. Event metrics (JSON and CSV)
        with open(self.output_dir / "smoke_event_metrics.json", "w", encoding="utf-8") as f:
            json.dump([event_row], f, indent=2)

        with open(self.output_dir / "smoke_event_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_row.keys()))
            writer.writeheader()
            writer.writerow(event_row)

        # 3. Step records (JSON with native nulls, CSV with text "null")
        with open(self.output_dir / "smoke_step_records.json", "w", encoding="utf-8") as f:
            json.dump(step_records, f, indent=2)

        if step_records:
            fieldnames = [k for k in step_records[0].keys() if k != "sample_ids"] + ["sample_ids"]
            with open(self.output_dir / "smoke_step_records.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                for rec in step_records:
                    row = []
                    for field in fieldnames:
                        val = rec.get(field)
                        if val is None:
                            row.append("null")
                        elif field == "sample_ids" and isinstance(val, list):
                            row.append(";".join(str(s) for s in val))
                        else:
                            row.append(str(val))
                    writer.writerow(row)

        # 4. Summary JSON
        summary = {
            "event_summary": event_row,
            "preflight": preflight_info,
            "timestamps": manifest["timestamps"],
        }
        with open(self.output_dir / "smoke_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChangeGuard Week 11 Real-Data Smoke Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/week11_smoke_event520_draft.json",
        help="Path to smoke configuration JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to output directory (overrides config smoke_dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target execution device (default: cpu)",
    )
    parser.add_argument(
        "--synthetic_mock",
        action="store_true",
        help="Execute CPU synthetic mock orchestration verification without real data/checkpoints",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Perform preflight validation checks and exit without execution",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting non-empty output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("================================================================================")
    print(" ChangeGuard Week 11 Real-Data Smoke Runner")
    print("================================================================================")
    print(f"Config:          {args.config}")
    print(f"Device:          {args.device}")
    print(f"Synthetic Mock:  {args.synthetic_mock}")
    print(f"Validate Only:   {args.validate_only}")
    print(f"Allow Overwrite: {args.allow_overwrite}")
    print("--------------------------------------------------------------------------------")

    try:
        runner = Week11RealDataSmokeRunner(
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
            synthetic_mock=args.synthetic_mock,
            allow_overwrite=args.allow_overwrite,
        )

        preflight = runner.validate_preflight(fail_closed=True)
        print("PREFLIGHT AUDIT PASSED:")
        print(f"  Protocol:          {preflight['protocol_version']}")
        print(f"  Status:            {preflight['status']}")
        print(f"  Target Events:     {preflight['target_events']}")
        print(f"  Checkpoint Seed:   {preflight['checkpoint_seed']}")
        print(f"  Output Dir:        {preflight['output_dir']}")
        print("--------------------------------------------------------------------------------")

        if args.validate_only:
            print("Validate-only requested. Exiting 0.")
            return 0

        res = runner.run_smoke()
        event_res = res["event_row"]
        print("SMOKE EXECUTION COMPLETED:")
        print(f"  Event:             {event_res['event_id']}")
        print(f"  Samples:           {event_res['sample_count']}")
        print(f"  Batches:           {event_res['batch_count']} (sizes: {event_res['batch_sizes']})")
        print(f"  Tail Batch:        {event_res['tail_batch_size']}")
        print(f"  Updates Applied:   {event_res['updates_applied']}")
        print(f"  Updates Skipped:   {event_res['updates_skipped']}")
        print(f"  Updates Failed:    {event_res['updates_failed']}")
        print(f"  Flood F1:          {event_res['flood_f1']:.4f}")
        print(f"  Flood IoU:         {event_res['flood_iou']:.4f}")
        print(f"  Outputs saved to:  {res['output_dir']}")
        print("================================================================================")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
