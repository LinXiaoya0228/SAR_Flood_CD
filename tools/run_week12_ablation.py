"""ChangeGuard Week 12 Exploratory Ablation Runner.

Governing Protocol: docs/protocols/week12_exploratory_ablation_protocol_v1.md
Governing Config:   configs/changeguard/week12_ablation_dev6_v1.json

Executes the four-condition minimal ablation across all six development events:
1. source_only
2. matched_tent_pre_update_v1
3. C1_consistency_no_conf (weak-strong consistency without confidence filter)
4. C2_consistency_conf90 (weak-strong consistency with confidence >= 0.90)

Key Research Invariants:
- Single source checkpoint seed: 0
- 6 Development Events only: [279, 437, 520, 559, 1111003, 1111008]
- Sealed test events strictly quarantined
- Stream order: order_a (deterministic sorted ascending sample_id within each event)
- Reset mode: reset_per_event (parameters and optimizer re-initialized from source before each event)
- Primary predictions: Strictly PRE_UPDATE timing across all four conditions
- Double-masking offline evaluation: valid = (MK0_MNA == 1) & (MK0_MLU != 3)
- Confusion conservation: assert (tp + fp + fn + tn) == valid_pixels
- Evidence Level: EXPLORATORY
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.evaluation_dataset import KuroEvaluationDataset
from src.changeguard.adapter import EvidenceConsistencyAdapter

CUDA_DEVICE_REGEX = re.compile(r"^cuda:(0|[1-9]\d*)$")
DEFAULT_TTA_DEV_EVENTS = [279, 437, 520, 559, 1111003, 1111008]
FORBIDDEN_SEALED_TEST_EVENTS = {
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
}
SUPPORTED_SYNTHETIC_STATUSES = {
    "DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED",
    "APPROVED_FOR_CPU_SYNTHETIC_SMOKE_VERIFICATION",
    "APPROVED_FOR_WEEK12_EXPLORATORY_EXECUTION",
}
SUPPORTED_REAL_STATUSES = {
    "APPROVED_FOR_WEEK12_EXPLORATORY_EXECUTION",
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class MockSNUNet(nn.Module):
    """Synthetic proxy model for unit testing and CPU synthetic verification."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([a, b], dim=1)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        return self.head(z)


class SourceOnlyAdapter:
    """Non-adapted source model adapter evaluating frozen PRE_UPDATE predictions."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def adapt_and_predict(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: torch.Tensor,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if sar_valid_mask.dim() == 4 and sar_valid_mask.size(1) == 1:
            sar_valid_mask = sar_valid_mask.squeeze(1)

        valid_count = int((sar_valid_mask == 1).sum().item())

        with torch.no_grad():
            self.model.eval()
            logits = self.model(xA, xB)
            pred_class = torch.argmax(logits, dim=1)

        self.step_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primary_prediction": {
                "predicted_class": pred_class.detach(),
                "logits": logits.detach(),
            },
            "adaptation_metadata": {
                "action": "source_eval",
                "update_applied": False,
                "loss": None,
                "valid_pixels": valid_count,
                "selected_pixels": 0,
                "selected_pixel_fraction": 0.0,
                "failure_reason": None,
                "update_time_ms": elapsed_ms,
            },
        }


class MatchedTentAdapter:
    """Matched Tent adapter operating strictly under PRE_UPDATE prediction timing."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

        self._configure_parameters()
        self._init_optimizer()

    def _configure_parameters(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

    def _init_optimizer(self) -> None:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            trainable,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def adapt_and_predict(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: torch.Tensor,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if sar_valid_mask.dim() == 4 and sar_valid_mask.size(1) == 1:
            sar_valid_mask = sar_valid_mask.squeeze(1)

        valid_mask_bool = (sar_valid_mask == 1)
        valid_count = int(valid_mask_bool.sum().item())

        # PRE_UPDATE forward pass on candidate
        logits = self.model(xA, xB)
        pred_class = torch.argmax(logits, dim=1).detach()

        action = "skip"
        update_applied = False
        loss_val: Optional[float] = None
        failure_reason: Optional[str] = None
        selected_pixels = valid_count

        if valid_count == 0:
            self.num_updates_skipped += 1
            action = "skip"
        else:
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            loss = (entropy * valid_mask_bool.float()).sum() / valid_mask_bool.sum().clamp(min=1.0)

            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = f"Non-finite entropy loss: {loss.item()}"
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                trainable = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [p for p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    failure_reason = "Non-finite gradient in backward pass"
                    self.optimizer.zero_grad()
                else:
                    pre_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                    pre_opt = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()

                    # Safeguard post-step finite check
                    corrupt = any(not torch.isfinite(p).all() for p in trainable)
                    if corrupt:
                        for n, p in self.model.named_parameters():
                            if p.requires_grad:
                                p.data.copy_(pre_params[n])
                        self.optimizer.load_state_dict(pre_opt)
                        self.num_updates_failed += 1
                        action = "failure"
                        failure_reason = "Optimizer step corrupted parameters"
                    else:
                        self.num_updates_applied += 1
                        update_applied = True
                        action = "update"
                        loss_val = float(loss.item())

        self.step_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primary_prediction": {
                "predicted_class": pred_class,
                "logits": logits.detach(),
            },
            "adaptation_metadata": {
                "action": action,
                "update_applied": update_applied,
                "loss": loss_val,
                "valid_pixels": valid_count,
                "selected_pixels": selected_pixels,
                "selected_pixel_fraction": 1.0 if valid_count > 0 else 0.0,
                "failure_reason": failure_reason,
                "update_time_ms": elapsed_ms,
            },
        }


class Week12AblationRunner:
    """Orchestrates the 4-condition Week 12 exploratory ablation over all 6 dev events."""

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

        if output_dir is not None:
            self.output_dir = output_dir.resolve()
        else:
            rel_out = self.cfg.get("outputs", {}).get("experiment_dir", "outputs/week12_ablation/approved_dev6_s0_v1")
            self.output_dir = (REPO_ROOT / rel_out).resolve()

        self.log_dir = (REPO_ROOT / self.cfg.get("outputs", {}).get("log_dir", "logs/week12_ablation/approved_dev6_s0_v1")).resolve()

        self.status = self.cfg.get("status")
        self.approval_status = self.cfg.get("approval_status")
        self.evidence_level = self.cfg.get("evidence_level", "EXPLORATORY")

        # Dataset configuration
        d_cfg = self.cfg.get("dataset", {})
        self.target_events: List[int] = [int(e) for e in d_cfg.get("target_events", DEFAULT_TTA_DEV_EVENTS)]
        self.development_events: List[int] = [int(e) for e in d_cfg.get("development_events", DEFAULT_TTA_DEV_EVENTS)]
        self.forbidden_events: Set[int] = set(
            int(e) for e in d_cfg.get("forbidden_sealed_test_events", list(FORBIDDEN_SEALED_TEST_EVENTS))
        )
        self.canonical_preevent: str = d_cfg.get("canonical_preevent", "pre_event_2")
        self.sar_mean = tuple(d_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = tuple(d_cfg.get("sar_std", [0.0427, 0.0215]))
        self.sar_clamp_max = float(d_cfg.get("sar_clamp_max", 0.15))

        raw_root = d_cfg.get("dataset_root", "../datasets/KuroSiwo")
        raw_manifest = d_cfg.get("manifest_path", "../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz")
        self.dataset_root = self._resolve_path(raw_root)
        self.manifest_path = self._resolve_path(raw_manifest)

        # Ablation specification
        a_cfg = self.cfg.get("ablation_specification", {})
        self.checkpoint_seed = int(a_cfg.get("checkpoint_seed", 0))
        self.stream_order = a_cfg.get("stream_order", "order_a")
        self.event_sequence = [int(e) for e in a_cfg.get("event_sequence", DEFAULT_TTA_DEV_EVENTS)]
        self.reset_mode = a_cfg.get("reset_mode", "reset_per_event")
        self.batch_size = int(a_cfg.get("batch_size", 16))
        self.condition_specs = a_cfg.get("conditions", [])

        # Model configuration
        m_cfg = self.cfg.get("model", {})
        self.model_architecture = m_cfg.get("architecture", "SNUNet_ECAM")
        self.in_channels = int(m_cfg.get("in_channels", 2))
        self.num_classes = int(m_cfg.get("num_classes", 3))
        self.base_channel = int(m_cfg.get("base_channel", 32))

        # Provenance digest holders
        self.manifest_sha256: Optional[str] = None
        self.ckpt_actual_sha256: Optional[str] = None
        self.ckpt_expected_sha256: Optional[str] = None

        self._enforce_authorizations()

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        if (REPO_ROOT / p).exists():
            return (REPO_ROOT / p).resolve()
        return (REPO_ROOT.parent / p).resolve()

    def _enforce_authorizations(self) -> None:
        auth = self.cfg.get("execution_authorizations")
        if not isinstance(auth, dict):
            raise TypeError("Configuration missing or invalid 'execution_authorizations' dictionary.")

        real_auth = auth.get("real_data_execution_authorized")
        gpu_auth = auth.get("gpu_execution_authorized")

        if not isinstance(real_auth, bool):
            raise TypeError(f"'real_data_execution_authorized' must be a JSON boolean, got {type(real_auth)}.")
        if not isinstance(gpu_auth, bool):
            raise TypeError(f"'gpu_execution_authorized' must be a JSON boolean, got {type(gpu_auth)}.")

        if self.status is None:
            raise ValueError("Configuration 'status' is missing or null.")
        if not isinstance(self.status, str):
            raise TypeError(f"Configuration 'status' must be a string, got {type(self.status)}.")

        if self.synthetic_mock:
            if self.device_str != "cpu":
                raise PermissionError(f"Synthetic mock execution supports 'cpu' only, got '{self.device_str}'.")
            if self.status not in SUPPORTED_SYNTHETIC_STATUSES and self.status not in SUPPORTED_REAL_STATUSES:
                raise ValueError(f"Unsupported configuration status for synthetic mock: '{self.status}'.")
            return

        # Real execution enforcement
        if self.status.startswith("DRAFT_"):
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
            raise PermissionError("Real-data execution is NOT authorized ('real_data_execution_authorized' is false).")

        if not CUDA_DEVICE_REGEX.match(self.device_str):
            raise ValueError(
                f"Real smoke execution strictly requires device format 'cuda:<non-negative integer>' (e.g. 'cuda:0'), "
                f"got '{self.device_str}'."
            )

        if not gpu_auth:
            raise PermissionError(f"GPU device '{self.device_str}' requested, but 'gpu_execution_authorized' is false.")

    def validate_preflight(self, fail_closed: bool = True) -> Dict[str, Any]:
        self._enforce_authorizations()

        if self.checkpoint_seed != 0:
            raise ValueError(f"Week 12 exploratory ablation strictly requires checkpoint_seed 0, got {self.checkpoint_seed}.")

        if self.stream_order != "order_a":
            raise ValueError(f"Week 12 exploratory ablation strictly requires stream_order 'order_a', got '{self.stream_order}'.")

        if self.reset_mode != "reset_per_event":
            raise ValueError(f"Week 12 exploratory ablation strictly requires reset_mode 'reset_per_event', got '{self.reset_mode}'.")

        if self.event_sequence != DEFAULT_TTA_DEV_EVENTS:
            raise ValueError(f"Week 12 event sequence mismatch: expected {DEFAULT_TTA_DEV_EVENTS}, got {self.event_sequence}.")

        target_set = set(self.target_events)
        forbidden_intersection = target_set.intersection(self.forbidden_events)
        if forbidden_intersection:
            raise RuntimeError(
                f"RESEARCH INTEGRITY VIOLATION: Target events {self.target_events} overlap with sealed test events: "
                f"{sorted(list(forbidden_intersection))}!"
            )

        dev_set = set(self.development_events)
        out_of_dev = target_set.difference(dev_set)
        if out_of_dev:
            raise RuntimeError(f"Scope violation: Target events contain out-of-dev events: {sorted(list(out_of_dev))}!")

        dataset_root_valid = False
        manifest_path_valid = False
        ckpt_path_str: Optional[str] = None
        ckpt_exists = False
        ckpt_hash_valid = False
        ckpt_error: Optional[str] = None

        if self.synthetic_mock:
            dataset_root_valid = True
            manifest_path_valid = True
            ckpt_path_str = "MOCKED_SYNTHETIC"
            ckpt_exists = True
            ckpt_hash_valid = True
            self.ckpt_expected_sha256 = "MOCKED_NOT_APPLICABLE"
            self.ckpt_actual_sha256 = "MOCKED_NOT_APPLICABLE"
            self.manifest_sha256 = "MOCKED_NOT_APPLICABLE"
        else:
            if not self.dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
            dataset_root_valid = True

            if not self.manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
            manifest_path_valid = True
            self.manifest_sha256 = compute_file_sha256(self.manifest_path)

            ckpt_info = self.cfg.get("source_checkpoints", {}).get("seeds", {}).get(str(self.checkpoint_seed), {})
            rel_ckpt = ckpt_info.get("checkpoint_path")
            exp_hash = ckpt_info.get("expected_sha256")
            self.ckpt_expected_sha256 = exp_hash

            ckpt_path = (REPO_ROOT / rel_ckpt).resolve() if rel_ckpt else None
            ckpt_path_str = str(ckpt_path) if ckpt_path else None
            ckpt_exists = bool(ckpt_path and ckpt_path.is_file())

            if ckpt_exists and ckpt_path:
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
            "evidence_level": self.evidence_level,
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

    def _resolve_samples_for_event(self, event_id: int, event_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not event_items:
            raise ValueError(f"No samples found for event {event_id}.")

        sorted_items = sorted(event_items, key=lambda x: str(x["sample_id"]))
        seen_ids = set()
        unique_items = []
        for it in sorted_items:
            sid = it["sample_id"]
            if sid in seen_ids:
                raise ValueError(f"Duplicate sample ID {sid} encountered in event {event_id}.")
            seen_ids.add(sid)
            unique_items.append(it)

        return unique_items

    def _instantiate_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        if self.synthetic_mock:
            model = MockSNUNet(in_channels=self.in_channels, num_classes=self.num_classes, base_channels=8)
            model.eval()
            model.to(self.device_str)
            return model, {"mock": True}
        else:
            ckpt_info = self.cfg["source_checkpoints"]["seeds"][str(self.checkpoint_seed)]
            ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
            wrapper, meta = SNUNetWrapper.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                in_channels=self.in_channels,
                out_ch=self.num_classes,
                base_channel=self.base_channel,
                device=self.device_str,
            )
            return wrapper, meta

    def _build_adapter(self, condition_id: str, model: nn.Module) -> Any:
        h_cfg = self.cfg.get("adaptation_hyperparameters", {})
        lr = h_cfg.get("learning_rate", 1.25e-4)
        mom = h_cfg.get("momentum", 0.9)
        wd = h_cfg.get("weight_decay", 0.0)

        if condition_id == "source_only":
            return SourceOnlyAdapter(model=model, device=self.device_str)
        elif condition_id == "matched_tent_pre_update_v1":
            return MatchedTentAdapter(
                model=model,
                learning_rate=lr,
                momentum=mom,
                weight_decay=wd,
                device=self.device_str,
            )
        elif condition_id == "C1_consistency_no_conf":
            # C1: Weak-strong consistency with NO confidence threshold (0.0 selects all valid pixels)
            return EvidenceConsistencyAdapter(
                model=model,
                learning_rate=lr,
                momentum=mom,
                weight_decay=wd,
                confidence_threshold=0.0,
                device=self.device_str,
            )
        elif condition_id == "C2_consistency_conf90":
            # C2: Weak-strong consistency with confidence >= 0.90
            return EvidenceConsistencyAdapter(
                model=model,
                learning_rate=lr,
                momentum=mom,
                weight_decay=wd,
                confidence_threshold=0.90,
                device=self.device_str,
            )
        else:
            raise ValueError(f"Unknown condition ID: '{condition_id}'")

    def run_ablation(self, mock_records_by_event: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        preflight = self.validate_preflight(fail_closed=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        initial_manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "evidence_level": self.evidence_level,
            "git_commit": self._get_git_commit(),
            "config_sha256": compute_file_sha256(self.config_path),
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "manifest_sha256": self.manifest_sha256,
            "event_sequence": self.event_sequence,
            "conditions": [c["condition_id"] for c in self.condition_specs],
            "device": self.device_str,
            "start_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(self.output_dir / "ablation_initial_manifest.json", "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        # Datasets resolution
        if self.synthetic_mock and mock_records_by_event is not None:
            all_records_offline = []
            all_records_adapt = []
            for ev, recs in mock_records_by_event.items():
                for r in recs:
                    all_records_offline.append(r)
                    all_records_adapt.append({
                        "sample_id": r["sample_id"],
                        "event_id": r["event_id"],
                        "xA": r["xA"],
                        "xB": r["xB"],
                        "valid_mask": r["valid_mask"],
                    })
            adapt_dataset = KuroEvaluationDataset(records=all_records_adapt, allowed_events=self.event_sequence, load_labels=False)
            offline_dataset = KuroEvaluationDataset(records=all_records_offline, allowed_events=self.event_sequence, load_labels=True)
        else:
            adapt_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.event_sequence,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=False,
            )
            offline_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.event_sequence,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=True,
            )

        # Group sample items by event
        samples_by_event: Dict[int, List[Dict[str, Any]]] = {ev: [] for ev in self.event_sequence}
        for idx, item in enumerate(adapt_dataset.samples):
            ev = int(item["event_id"])
            if ev in samples_by_event:
                item_with_idx = dict(item)
                item_with_idx["dataset_index"] = idx
                samples_by_event[ev].append(item_with_idx)

        # Sort within each event ascending by sample_id
        sorted_samples_by_event = {
            ev: self._resolve_samples_for_event(ev, items)
            for ev, items in samples_by_event.items()
        }

        # Cache source state dict for clean resets
        base_model, _ = self._instantiate_model()
        source_state_dict = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

        all_condition_results: Dict[str, Any] = {}
        t_global_start = time.perf_counter()

        for c_idx, c_spec in enumerate(self.condition_specs):
            cond_id = c_spec["condition_id"]
            cond_dir = self.output_dir / cond_id
            cond_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> Running Condition [{c_idx + 1}/4]: {cond_id} ...")
            cond_event_summaries = []
            cond_total_updates = 0
            cond_total_skips = 0
            cond_total_failures = 0
            cond_total_valid_pixels = 0
            cond_total_selected_pixels = 0
            cond_t0 = time.perf_counter()

            for ev_idx, event_id in enumerate(self.event_sequence):
                ev_items = sorted_samples_by_event[event_id]
                event_dir = cond_dir / f"event_{event_id}"
                event_dir.mkdir(parents=True, exist_ok=True)

                # RESET POLICY: Instantiate a fresh model and adapter for each event
                # This guarantees zero buffer mutation leakage or running_mean/var collision across events
                model, _ = self._instantiate_model()
                adapter = self._build_adapter(cond_id, model)

                # Batching (B=16 with natural tail)
                batches = [ev_items[i : i + self.batch_size] for i in range(0, len(ev_items), self.batch_size)]
                frozen_predictions: Dict[str, np.ndarray] = {}
                step_records: List[Dict[str, Any]] = []
                completed_sample_ids: List[str] = []

                # Progress snapshot holder
                progress_file = event_dir / "adaptation_progress.json"

                try:
                    for b_idx, b_items in enumerate(batches):
                        b_indices = [it["dataset_index"] for it in b_items]
                        b_ids = [it["sample_id"] for it in b_items]

                        batch_data = [adapt_dataset[idx] for idx in b_indices]
                        xA = torch.stack([d["xA"] for d in batch_data]).to(self.device_str)
                        xB = torch.stack([d["xB"] for d in batch_data]).to(self.device_str)
                        m = torch.stack([d["valid_mask"] for d in batch_data]).to(self.device_str)

                        # Adapt and predict (label-free, PRE_UPDATE)
                        step_res = adapter.adapt_and_predict(
                            xA,
                            xB,
                            sar_valid_mask=m,
                            sample_ids=b_ids,
                            event_ids=[event_id] * len(b_ids),
                        )

                        preds = step_res["primary_prediction"]["predicted_class"].cpu().numpy()
                        for idx_s, sid in enumerate(b_ids):
                            frozen_predictions[sid] = preds[idx_s]

                        completed_sample_ids.extend(b_ids)
                        adapt_meta = step_res.get("adaptation_metadata", {})

                        v_px = adapt_meta.get("valid_pixels", 0)
                        s_px = adapt_meta.get("selected_pixels", 0)
                        frac = (s_px / v_px) if v_px > 0 else 0.0

                        rec = {
                            "step_index": b_idx + 1,
                            "event_id": event_id,
                            "condition_id": cond_id,
                            "batch_size": len(b_items),
                            "sample_ids": b_ids,
                            "action": adapt_meta.get("action"),
                            "update_applied": adapt_meta.get("update_applied"),
                            "loss": adapt_meta.get("loss"),
                            "valid_pixels": v_px,
                            "selected_pixels": s_px,
                            "selected_pixel_fraction": frac,
                            "failure_reason": adapt_meta.get("failure_reason"),
                            "batch_time_ms": adapt_meta.get("update_time_ms"),
                        }
                        step_records.append(rec)

                        # Durable progress after each batch
                        prog_data = {
                            "status": "RUNNING_ADAPTATION",
                            "condition_id": cond_id,
                            "event_id": event_id,
                            "batches_completed": b_idx + 1,
                            "total_expected_batches": len(batches),
                            "samples_completed": len(completed_sample_ids),
                            "updates_applied": adapter.num_updates_applied,
                            "updates_skipped": adapter.num_updates_skipped,
                            "updates_failed": adapter.num_updates_failed,
                            "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                        with open(progress_file, "w", encoding="utf-8") as f:
                            json.dump(prog_data, f, indent=2)

                    # PERSISTENCE PHASE BEFORE ANY GT ACCESS
                    np.savez_compressed(event_dir / "frozen_predictions.npz", **frozen_predictions)
                    with open(event_dir / "step_records.json", "w", encoding="utf-8") as f:
                        json.dump(step_records, f, indent=2)

                    # OFFLINE SCORING PHASE
                    tp, fp, fn, tn = 0, 0, 0, 0
                    pred_flood_total = 0
                    gt_flood_total = 0
                    evaluated_valid_pixels = 0

                    for item in ev_items:
                        sid = item["sample_id"]
                        d_idx = item["dataset_index"]
                        gt_rec = offline_dataset[d_idx]

                        pred = frozen_predictions[sid]
                        label = gt_rec["label"].numpy()
                        vmask = gt_rec["valid_mask"].numpy()

                        # Double masking: valid = (MK0_MNA == 1) & (MK0_MLU != 3)
                        valid = (vmask == 1) & (label != 3)
                        v_count = int(np.sum(valid))
                        evaluated_valid_pixels += v_count

                        p_flood = (pred == 2)
                        g_flood = (label == 2)

                        s_tp = int(np.sum(p_flood & g_flood & valid))
                        s_fp = int(np.sum(p_flood & ~g_flood & valid))
                        s_fn = int(np.sum(~p_flood & g_flood & valid))
                        s_tn = int(np.sum(~p_flood & ~g_flood & valid))

                        assert (s_tp + s_fp + s_fn + s_tn) == v_count

                        tp += s_tp
                        fp += s_fp
                        fn += s_fn
                        tn += s_tn
                        pred_flood_total += int(np.sum(p_flood & valid))
                        gt_flood_total += int(np.sum(g_flood & valid))

                    # Event conservation check
                    assert (tp + fp + fn + tn) == evaluated_valid_pixels

                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                    ev_summary = {
                        "event_id": event_id,
                        "sample_count": len(ev_items),
                        "batch_count": len(batches),
                        "batch_sizes": [len(b) for b in batches],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "valid_pixels": evaluated_valid_pixels,
                        "pred_flood_pixels": pred_flood_total,
                        "gt_flood_pixels": gt_flood_total,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "iou": iou,
                        "updates_applied": adapter.num_updates_applied,
                        "updates_skipped": adapter.num_updates_skipped,
                        "updates_failed": adapter.num_updates_failed,
                    }
                    with open(event_dir / "event_metrics.json", "w", encoding="utf-8") as f:
                        json.dump(ev_summary, f, indent=2)

                    cond_event_summaries.append(ev_summary)
                    cond_total_updates += adapter.num_updates_applied
                    cond_total_skips += adapter.num_updates_skipped
                    cond_total_failures += adapter.num_updates_failed

                    cond_total_valid_pixels += sum(r["valid_pixels"] for r in step_records)
                    cond_total_selected_pixels += sum(r["selected_pixels"] for r in step_records)

                    print(f"  Event {event_id} ({len(ev_items)} samples): F1={f1:.4f}, IoU={iou:.4f}, Updates={adapter.num_updates_applied}/{len(batches)}")

                except Exception as exc:
                    fail_record = {
                        "status": "FAILED",
                        "condition_id": cond_id,
                        "event_id": event_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(event_dir / "execution_failure.json", "w", encoding="utf-8") as f:
                        json.dump(fail_record, f, indent=2)
                    raise

            # Condition aggregate metrics
            pooled_tp = sum(e["tp"] for e in cond_event_summaries)
            pooled_fp = sum(e["fp"] for e in cond_event_summaries)
            pooled_fn = sum(e["fn"] for e in cond_event_summaries)
            pooled_tn = sum(e["tn"] for e in cond_event_summaries)
            pooled_valid = sum(e["valid_pixels"] for e in cond_event_summaries)

            assert (pooled_tp + pooled_fp + pooled_fn + pooled_tn) == pooled_valid

            pooled_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
            pooled_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
            pooled_f1 = 2 * (pooled_prec * pooled_rec) / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
            pooled_iou = pooled_tp / (pooled_tp + pooled_fp + pooled_fn) if (pooled_tp + pooled_fp + pooled_fn) > 0 else 0.0

            macro_f1 = float(np.mean([e["f1"] for e in cond_event_summaries]))
            macro_iou = float(np.mean([e["iou"] for e in cond_event_summaries]))
            macro_prec = float(np.mean([e["precision"] for e in cond_event_summaries]))
            macro_rec = float(np.mean([e["recall"] for e in cond_event_summaries]))

            cond_time = time.perf_counter() - cond_t0
            sel_frac = (cond_total_selected_pixels / cond_total_valid_pixels) if cond_total_valid_pixels > 0 else 0.0

            cond_aggregate = {
                "condition_id": cond_id,
                "evidence_level": "EXPLORATORY",
                "events_evaluated": self.event_sequence,
                "total_samples": sum(e["sample_count"] for e in cond_event_summaries),
                "total_batches": sum(e["batch_count"] for e in cond_event_summaries),
                "pooled_metrics": {
                    "tp": pooled_tp,
                    "fp": pooled_fp,
                    "fn": pooled_fn,
                    "tn": pooled_tn,
                    "valid_pixels": pooled_valid,
                    "precision": pooled_prec,
                    "recall": pooled_rec,
                    "f1": pooled_f1,
                    "iou": pooled_iou,
                },
                "macro_metrics": {
                    "macro_f1": macro_f1,
                    "macro_iou": macro_iou,
                    "macro_precision": macro_prec,
                    "macro_recall": macro_rec,
                },
                "adaptation_behavior": {
                    "updates_applied": cond_total_updates,
                    "updates_skipped": cond_total_skips,
                    "updates_failed": cond_total_failures,
                    "total_valid_pixels": cond_total_valid_pixels,
                    "total_selected_pixels": cond_total_selected_pixels,
                    "selected_pixel_fraction": sel_frac,
                    "elapsed_seconds": cond_time,
                },
                "event_rows": cond_event_summaries,
            }
            with open(cond_dir / "condition_summary.json", "w", encoding="utf-8") as f:
                json.dump(cond_aggregate, f, indent=2)

            all_condition_results[cond_id] = cond_aggregate

        # Final cross-condition summary
        total_time = time.perf_counter() - t_global_start
        final_summary = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": "COMPLETED",
            "evidence_level": "EXPLORATORY",
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": total_time,
            "conditions": all_condition_results,
        }
        with open(self.output_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)

        completion_marker = {
            "status": "COMPLETED",
            "evidence_level": "EXPLORATORY",
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": total_time,
        }
        with open(self.output_dir / "completion_status.json", "w", encoding="utf-8") as f:
            json.dump(completion_marker, f, indent=2)

        print("\n================================================================================")
        print(" ChangeGuard Week 12 Exploratory Ablation Completed Successfully!")
        print("================================================================================")
        print(f"Elapsed: {total_time:.2f}s | Output: {self.output_dir}")
        return final_summary

    def _get_git_commit(self) -> str:
        try:
            import subprocess
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
            return out.strip()
        except Exception:
            return "UNKNOWN"


def parse_args():
    parser = argparse.ArgumentParser(description="Week 12 Exploratory Ablation Runner")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/changeguard/week12_ablation_dev6_v1.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--synthetic_mock", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("================================================================================")
    print(" ChangeGuard Week 12 Exploratory Ablation Runner")
    print("================================================================================")
    print(f"Config:          {args.config}")
    print(f"Device:          {args.device}")
    print(f"Synthetic Mock:  {args.synthetic_mock}")
    print(f"Validate Only:   {args.validate_only}")
    print(f"Allow Overwrite: {args.allow_overwrite}")
    print("--------------------------------------------------------------------------------")

    try:
        runner = Week12AblationRunner(
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
        print(f"  Output Dir:        {preflight['output_dir']}")
        print("--------------------------------------------------------------------------------")

        if args.validate_only:
            print("Validate-only requested. Exiting 0.")
            return 0

        runner.run_ablation()
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
