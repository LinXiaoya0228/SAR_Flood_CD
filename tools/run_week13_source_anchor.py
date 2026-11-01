#!/usr/bin/env python3
"""ChangeGuard Week 13 Frozen-Source Probability-Anchor Runner.

Governing Protocol: docs/protocols/week13_source_anchor_protocol_v1.md
Governing Config:   configs/changeguard/week13_source_anchor_dev6_v1.json

Executes the single planned condition:
C3_anchored_consistency_v1:
  total_loss = consistency_loss + 0.50 * mean_valid(KL(p_source_detached || p_candidate_aligned_strong))

Key Research Invariants:
- Single source checkpoint seed: 0
- 6 Development Events only: [279, 437, 520, 559, 1111003, 1111008]
- 10 Sealed Test Events strictly forbidden: [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
- Stream Order: order_a (lexicographic ascending path order)
- Reset Mode: reset_per_event = true
- Batch Size: 16 with natural tail
- SGD: LR 1.25e-4, Momentum 0.9, Weight Decay 0.0
- Trainable Scope: selected candidate BatchNorm affine weights/biases only
- Source Model: eval mode, frozen training running stats, strictly detached under torch.no_grad()
- Predictions: PRE_UPDATE primary predictions persisted before ground-truth access
- Double-Masked Offline Scoring: valid = (MK0_MNA == 1) & (MK0_MLU != 3)
- Strict Conservation: TP + FP + FN + TN == valid_pixels
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.evaluation_dataset import KuroEvaluationDataset

DEFAULT_TTA_DEV_EVENTS = [279, 437, 520, 559, 1111003, 1111008]
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
CUDA_DEVICE_REGEX = re.compile(r"^cuda:(0|[1-9]\d*)$")

SUPPORTED_REAL_STATUSES = {
    "APPROVED_FOR_WEEK13_SOURCE_ANCHOR_EXECUTION",
}

SUPPORTED_SYNTHETIC_STATUSES = {
    "APPROVED_FOR_WEEK13_SOURCE_ANCHOR_EXECUTION",
    "APPROVED_FOR_CPU_IMPLEMENTATION",
    "DRAFT_PROTOCOL_EXECUTION_UNAUTHORIZED",
}


def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class MockSNUNet(nn.Module):
    """Lightweight 2-stream convolutional network for CPU synthetic testing."""

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


class SourceAnchorConsistencyAdapter:
    """Weak-Strong consistency adapter with frozen-source probability anchor.

    Total loss:
      L_total = L_cons + lambda_anc * L_anchor
    where:
      L_cons   = mean_{Omega_sel} KL(detach(p_weak) || p_candidate_aligned_strong)
      L_anchor = mean_{Omega_valid} KL(detach(p_source) || p_candidate_aligned_strong)
    """

    def __init__(
        self,
        model: nn.Module,
        source_model: nn.Module,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        confidence_threshold: float = 0.90,
        anchor_lambda: float = 0.50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.confidence_threshold = float(confidence_threshold)
        self.anchor_lambda = float(anchor_lambda)

        # Cache source state dict for candidate model restorations
        self._source_state_dict = {
            k: v.detach().clone().to(self.device)
            for k, v in self.model.state_dict().items()
        }

        # Cache BN buffer shapes
        self._bn_buffer_shapes: Dict[str, Tuple[torch.Size, torch.Size]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mean_shape = (
                    module.running_mean.shape
                    if module.running_mean is not None
                    else torch.Size([module.num_features])
                )
                var_shape = (
                    module.running_var.shape
                    if module.running_var is not None
                    else torch.Size([module.num_features])
                )
                self._bn_buffer_shapes[name] = (mean_shape, var_shape)

        # Frozen source model (eval mode, immutable, independent storage)
        if source_model is self.model:
            raise RuntimeError("Source model cannot be the identical object to candidate model.")
        self.source_model = source_model.to(self.device)
        self.source_model.eval()
        self.source_model.requires_grad_(False)

        # Ensure no storage aliasing between candidate and source
        cand_ptrs = {p.data_ptr() for p in self.model.parameters()} | {
            b.data_ptr() for b in self.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in self.source_model.parameters()} | {
            b.data_ptr() for b in self.source_model.buffers() if b is not None
        }
        shared = cand_ptrs.intersection(src_ptrs)
        if shared:
            raise RuntimeError(f"Storage aliasing detected between candidate and source model: {shared}")

        # Configure candidate model BN
        self._configure_model_bn()
        self._init_optimizer()

        # Step and update counters
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def _configure_model_bn(self) -> None:
        """Configures candidate model: eval globally, selected BN train() with no tracking."""
        self.model.eval()
        self.model.requires_grad_(False)

        self.bn_param_names: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    self.bn_param_names.append(f"{name}.weight")
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    self.bn_param_names.append(f"{name}.bias")

    def _init_optimizer(self) -> None:
        """Initializes SGD optimizer over active BatchNorm affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BatchNorm parameters found in candidate model!")
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def _is_state_dict_finite(obj: Any) -> bool:
        """Recursively checks if all tensors and floats in a state dict are finite."""
        if isinstance(obj, torch.Tensor):
            return bool(torch.isfinite(obj).all())
        if isinstance(obj, dict):
            return all(SourceAnchorConsistencyAdapter._is_state_dict_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(SourceAnchorConsistencyAdapter._is_state_dict_finite(v) for v in obj)
        if isinstance(obj, float):
            return math.isfinite(obj)
        return True

    def reset_to_source(self) -> None:
        """Restores candidate model state bit-exactly to source checkpoint and resets optimizer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape, device=self.device))
                    module.register_buffer("running_var", torch.ones(var_shape, device=self.device))
                    module.register_buffer(
                        "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=self.device)
                    )

        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)
        self._configure_model_bn()
        self._init_optimizer()
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs anchored consistency step and records PRE_UPDATE predictions."""
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        sar_valid_mask = sar_valid_mask.to(self.device)

        if xA.shape != xB.shape:
            raise ValueError(f"Temporal inputs xA {xA.shape} and xB {xB.shape} must have identical shape.")
        if sar_valid_mask.dim() == 4 and sar_valid_mask.shape[1] == 1:
            sar_valid_mask = sar_valid_mask.squeeze(1)
        elif sar_valid_mask.dim() != 3:
            raise ValueError(
                f"Unsupported sensory mask shape {sar_valid_mask.shape}. "
                f"Expected 3D tensor (B, H, W) or 4D with singleton channel (B, 1, H, W)."
            )
        if sar_valid_mask.shape[0] != xA.shape[0] or sar_valid_mask.shape[1:] != xA.shape[2:]:
            raise ValueError(
                f"Mask shape {sar_valid_mask.shape} does not match spatial dimensions of inputs {xA.shape}."
            )

        valid_mask_bool = (sar_valid_mask == 1)
        valid_pixel_count = int(valid_mask_bool.sum().item())

        # ---------------------------------------------------------------------
        # 1. Primary PRE_UPDATE Candidate Forward Pass on Weak View
        # ---------------------------------------------------------------------
        logits_weak = self.model(xA, xB)
        probs_weak = F.softmax(logits_weak, dim=1)
        pred_weak = probs_weak.argmax(dim=1)
        probs_weak_det = probs_weak.detach()
        probs_weak_max = probs_weak_det.max(dim=1)[0]  # (B, H, W)

        primary_prediction = {
            "logits": logits_weak.detach(),
            "probabilities": probs_weak_det,
            "predicted_class": pred_weak.detach(),
            "predicted_flood_pixels": int((pred_weak == 2).sum().item()),
            "valid_pixels": valid_pixel_count,
        }

        # ---------------------------------------------------------------------
        # 2. Frozen Source Model Forward Pass (Strictly no_grad)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            self.source_model.eval()
            logits_source = self.source_model(xA, xB)
            probs_source = F.softmax(logits_source, dim=1).detach()

        # ---------------------------------------------------------------------
        # 3. Candidate Strong View Forward Pass & Realignment
        # ---------------------------------------------------------------------
        xA_strong = torch.flip(xA, dims=[-1])
        xB_strong = torch.flip(xB, dims=[-1])

        logits_strong = self.model(xA_strong, xB_strong)
        logits_strong_aligned = torch.flip(logits_strong, dims=[-1])
        log_probs_strong_aligned = F.log_softmax(logits_strong_aligned, dim=1)

        # ---------------------------------------------------------------------
        # 4. Consistency Masking & Loss Formulation
        # ---------------------------------------------------------------------
        conf_mask = (probs_weak_max >= self.confidence_threshold) & valid_mask_bool
        selected_pixels = int(conf_mask.sum().item())

        update_applied = False
        action = "skip"
        loss_val: Optional[float] = None
        consistency_loss_val: Optional[float] = None
        anchor_loss_val: Optional[float] = None
        failure_reason: Optional[str] = None

        if valid_pixel_count == 0 or (selected_pixels == 0 and self.anchor_lambda == 0.0):
            self.num_updates_skipped += 1
            action = "skip"
        else:
            # Consistency term
            if selected_pixels > 0:
                kl_cons_pixel = (
                    probs_weak_det * (torch.log(probs_weak_det + 1e-12) - log_probs_strong_aligned)
                ).sum(dim=1)
                loss_cons = (kl_cons_pixel * conf_mask.float()).sum() / conf_mask.sum().clamp(min=1.0)
            else:
                loss_cons = torch.tensor(0.0, device=self.device)

            # Source anchor term over all valid pixels
            if self.anchor_lambda > 0.0 and valid_pixel_count > 0:
                kl_anc_pixel = (
                    probs_source * (torch.log(probs_source + 1e-12) - log_probs_strong_aligned)
                ).sum(dim=1)
                loss_anc = (kl_anc_pixel * valid_mask_bool.float()).sum() / valid_mask_bool.sum().clamp(min=1.0)
            else:
                loss_anc = torch.tensor(0.0, device=self.device)

            loss = loss_cons + self.anchor_lambda * loss_anc

            # Numerical boundary check 1: Finite loss
            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = f"Non-finite loss encountered: {loss.item()}"
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                # Numerical boundary check 2: Finite gradients
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [
                    p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()
                ]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    failure_reason = "Non-finite gradient encountered during backward pass"
                    self.optimizer.zero_grad()
                else:
                    # Save pre-step state for safe rollback
                    pre_step_params = {
                        n: p.detach().clone()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                    }
                    pre_step_opt_state = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()

                    # Numerical boundary check 3: Finite parameters and optimizer state post-step
                    corrupt_params = any(not torch.isfinite(p).all() for p in trainable_params)
                    corrupt_opt = not self._is_state_dict_finite(self.optimizer.state_dict())

                    if corrupt_params or corrupt_opt:
                        for n, p in self.model.named_parameters():
                            if p.requires_grad:
                                p.data.copy_(pre_step_params[n])
                        self.optimizer.load_state_dict(pre_step_opt_state)

                        self.num_updates_failed += 1
                        action = "failure"
                        failure_reason = (
                            "Post-step numerical corruption in parameters"
                            if corrupt_params
                            else "Post-step numerical corruption in optimizer state"
                        )
                    else:
                        self.num_updates_applied += 1
                        update_applied = True
                        action = "update"
                        loss_val = float(loss.item())
                        consistency_loss_val = float(loss_cons.item())
                        anchor_loss_val = float(loss_anc.item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": primary_prediction,
            "adaptation_metadata": {
                "action": action,
                "update_applied": update_applied,
                "loss": loss_val,
                "consistency_loss": consistency_loss_val,
                "anchor_loss": anchor_loss_val,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixel_fraction": float(selected_pixels / valid_pixel_count) if valid_pixel_count > 0 else 0.0,
                "failure_reason": failure_reason,
                "update_time_ms": elapsed_ms,
            },
        }


class Week13SourceAnchorRunner:
    """Orchestrates Week 13 source-anchored consistency adaptation over 6 dev events."""

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
            rel_out = self.cfg.get("outputs", {}).get("experiment_dir", "outputs/week13_source_anchor/approved_dev6_s0_v1")
            self.output_dir = (REPO_ROOT / rel_out).resolve()

        self.log_dir = (REPO_ROOT / self.cfg.get("outputs", {}).get("log_dir", "logs/week13_source_anchor/approved_dev6_s0_v1")).resolve()

        self.status = self.cfg.get("status")
        self.approval_status = self.cfg.get("approval_status")
        self.evidence_level = self.cfg.get("evidence_level", "CONTROLLED_DEVELOPMENT_ABLATION")

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

        # Experiment specification
        e_cfg = self.cfg.get("experiment_specification", {})
        self.checkpoint_seed = int(e_cfg.get("checkpoint_seed", 0))
        self.stream_order = e_cfg.get("stream_order", "order_a")
        self.event_sequence = [int(e) for e in e_cfg.get("event_sequence", DEFAULT_TTA_DEV_EVENTS)]
        self.reset_mode = e_cfg.get("reset_mode", "reset_per_event")
        self.batch_size = int(e_cfg.get("batch_size", 16))
        self.condition_spec = e_cfg.get("condition", {})
        self.condition_id = self.condition_spec.get("condition_id", "C3_anchored_consistency_v1")

        # Adaptation hyperparameters
        h_cfg = self.cfg.get("adaptation_hyperparameters", {})
        self.learning_rate = float(h_cfg.get("learning_rate", 1.25e-4))
        self.momentum = float(h_cfg.get("momentum", 0.9))
        self.weight_decay = float(h_cfg.get("weight_decay", 0.0))
        self.confidence_threshold = float(h_cfg.get("confidence_threshold", 0.90))
        self.anchor_lambda = float(h_cfg.get("anchor_lambda", 0.50))

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

        if self.status.startswith("DRAFT_"):
            raise PermissionError(
                f"Real-data execution is rejected: configuration status is '{self.status}'."
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
                f"Real execution strictly requires device format 'cuda:<non-negative integer>' (e.g. 'cuda:0'), "
                f"got '{self.device_str}'."
            )

        if not gpu_auth:
            raise PermissionError(f"GPU device '{self.device_str}' requested, but 'gpu_execution_authorized' is false.")

    def validate_preflight(self, fail_closed: bool = True) -> Dict[str, Any]:
        self._enforce_authorizations()

        if self.checkpoint_seed != 0:
            raise ValueError(f"Week 13 strictly requires checkpoint_seed 0, got {self.checkpoint_seed}.")

        if self.stream_order != "order_a":
            raise ValueError(f"Week 13 strictly requires stream_order 'order_a', got '{self.stream_order}'.")

        if self.reset_mode != "reset_per_event":
            raise ValueError(f"Week 13 strictly requires reset_mode 'reset_per_event', got '{self.reset_mode}'.")

        if self.event_sequence != DEFAULT_TTA_DEV_EVENTS:
            raise ValueError(f"Week 13 event sequence mismatch: expected {DEFAULT_TTA_DEV_EVENTS}, got {self.event_sequence}.")

        if self.condition_id != "C3_anchored_consistency_v1":
            raise ValueError(f"Week 13 strictly targets 'C3_anchored_consistency_v1', got '{self.condition_id}'.")

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

    def _instantiate_models(self) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
        """Instantiates separate candidate and frozen source models."""
        if self.synthetic_mock:
            cand = MockSNUNet(in_channels=self.in_channels, num_classes=self.num_classes, base_channels=8)
            cand.to(self.device_str)
            src = MockSNUNet(in_channels=self.in_channels, num_classes=self.num_classes, base_channels=8)
            src.load_state_dict(cand.state_dict())
            src.to(self.device_str)
            src.eval()
            src.requires_grad_(False)
            return cand, src, {"mock": True}
        else:
            ckpt_info = self.cfg["source_checkpoints"]["seeds"][str(self.checkpoint_seed)]
            ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
            cand_wrapper, meta = SNUNetWrapper.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                in_channels=self.in_channels,
                out_ch=self.num_classes,
                base_channel=self.base_channel,
                device=self.device_str,
            )
            src_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                in_channels=self.in_channels,
                out_ch=self.num_classes,
                base_channel=self.base_channel,
                device=self.device_str,
            )
            src_wrapper.eval()
            src_wrapper.requires_grad_(False)
            return cand_wrapper, src_wrapper, meta

    def _build_adapter(self, candidate_model: nn.Module, source_model: nn.Module) -> SourceAnchorConsistencyAdapter:
        return SourceAnchorConsistencyAdapter(
            model=candidate_model,
            source_model=source_model,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            confidence_threshold=self.confidence_threshold,
            anchor_lambda=self.anchor_lambda,
            device=self.device_str,
        )

    def run_experiment(self, mock_records_by_event: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        preflight = self.validate_preflight(fail_closed=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        initial_manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "evidence_level": self.evidence_level,
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "manifest_sha256": self.manifest_sha256,
            "event_sequence": self.event_sequence,
            "condition": self.condition_id,
            "anchor_lambda": self.anchor_lambda,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device_str,
            "start_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        manifest_path = self.output_dir / self.cfg.get("outputs", {}).get("manifest_json", "anchor_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        start_time_total = time.perf_counter()
        condition_dir = self.output_dir / self.condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)

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
                        "valid_mask": r["valid_mask"] if "valid_mask" in r else r["sar_valid_mask"],
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

        event_summaries: List[Dict[str, Any]] = []

        for ev_idx, event_id in enumerate(self.event_sequence):
            ev_items = sorted_samples_by_event[event_id]
            event_dir = condition_dir / f"event_{event_id}"
            event_dir.mkdir(parents=True, exist_ok=True)

            # RESET POLICY: Instantiate fresh models and adapter for each event
            cand_model, src_model, _ = self._instantiate_models()
            adapter = self._build_adapter(candidate_model=cand_model, source_model=src_model)

            batches = [ev_items[i : i + self.batch_size] for i in range(0, len(ev_items), self.batch_size)]
            frozen_predictions: Dict[str, np.ndarray] = {}
            step_records: List[Dict[str, Any]] = []
            completed_sample_ids: List[str] = []
            progress_file = event_dir / "adaptation_progress.json"

            for b_idx, b_items in enumerate(batches):
                b_indices = [it["dataset_index"] for it in b_items]
                b_ids = [it["sample_id"] for it in b_items]

                batch_data = [adapt_dataset[idx] for idx in b_indices]
                xA = torch.stack([d["xA"] for d in batch_data]).to(self.device_str)
                xB = torch.stack([d["xB"] for d in batch_data]).to(self.device_str)
                m = torch.stack([d["valid_mask"] for d in batch_data]).to(self.device_str)

                # Adapt and predict (label-free, PRE_UPDATE)
                step_res = adapter.adapt_and_predict(
                    xA=xA,
                    xB=xB,
                    sar_valid_mask=m,
                    sample_ids=b_ids,
                    event_ids=[event_id] * len(b_ids),
                )

                preds = step_res["primary_prediction"]["predicted_class"].cpu().numpy()
                for idx_s, sid in enumerate(b_ids):
                    frozen_predictions[sid] = preds[idx_s].astype(np.uint8)

                completed_sample_ids.extend(b_ids)
                adapt_meta = step_res.get("adaptation_metadata", {})

                v_px = adapt_meta.get("valid_pixels", 0)
                s_px = adapt_meta.get("selected_pixels", 0)
                frac = (s_px / v_px) if v_px > 0 else 0.0

                rec = {
                    "step_index": b_idx + 1,
                    "event_id": event_id,
                    "condition_id": self.condition_id,
                    "batch_size": len(b_items),
                    "sample_ids": b_ids,
                    "action": adapt_meta.get("action"),
                    "update_applied": adapt_meta.get("update_applied"),
                    "loss": adapt_meta.get("loss"),
                    "consistency_loss": adapt_meta.get("consistency_loss"),
                    "anchor_loss": adapt_meta.get("anchor_loss"),
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
                    "condition_id": self.condition_id,
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
            assert (tp + fp + fn + tn) == evaluated_valid_pixels, (
                f"Confusion sum {tp + fp + fn + tn} != valid pixels {evaluated_valid_pixels} on event {event_id}!"
            )

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
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "iou": float(iou),
                "updates_applied": adapter.num_updates_applied,
                "updates_skipped": adapter.num_updates_skipped,
                "updates_failed": adapter.num_updates_failed,
            }
            with open(event_dir / "event_metrics.json", "w", encoding="utf-8") as f:
                json.dump(ev_summary, f, indent=2)

            event_summaries.append(ev_summary)
            print(
                f"  Event {event_id} ({len(ev_items)} samples): "
                f"F1={f1:.4f}, IoU={iou:.4f}, Updates={adapter.num_updates_applied}/{len(batches)}"
            )

        # Aggregate metrics
        macro_f1 = float(np.mean([m["f1"] for m in event_summaries])) if event_summaries else 0.0
        macro_iou = float(np.mean([m["iou"] for m in event_summaries])) if event_summaries else 0.0
        macro_prec = float(np.mean([m["precision"] for m in event_summaries])) if event_summaries else 0.0
        macro_rec = float(np.mean([m["recall"] for m in event_summaries])) if event_summaries else 0.0

        tot_tp = sum(m["tp"] for m in event_summaries)
        tot_fp = sum(m["fp"] for m in event_summaries)
        tot_fn = sum(m["fn"] for m in event_summaries)
        tot_tn = sum(m["tn"] for m in event_summaries)
        tot_valid = sum(m["valid_pixels"] for m in event_summaries)

        pooled_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        pooled_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        pooled_f1 = 2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = tot_tp / (tot_tp + tot_fp + tot_fn) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

        worst_event = min(event_summaries, key=lambda x: x["f1"]) if event_summaries else {}

        tot_updates_applied = sum(m["updates_applied"] for m in event_summaries)
        tot_updates_skipped = sum(m["updates_skipped"] for m in event_summaries)
        tot_updates_failed = sum(m["updates_failed"] for m in event_summaries)

        summary_data = {
            "condition_id": self.condition_id,
            "anchor_lambda": self.anchor_lambda,
            "confidence_threshold": self.confidence_threshold,
            "macro_f1": macro_f1,
            "macro_iou": macro_iou,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "pooled_f1": pooled_f1,
            "pooled_iou": pooled_iou,
            "pooled_precision": float(pooled_prec),
            "pooled_recall": float(pooled_rec),
            "worst_event_id": worst_event.get("event_id"),
            "worst_event_f1": worst_event.get("f1"),
            "total_tp": tot_tp,
            "total_fp": tot_fp,
            "total_fn": tot_fn,
            "total_tn": tot_tn,
            "total_valid_pixels": tot_valid,
            "total_updates_applied": tot_updates_applied,
            "total_updates_skipped": tot_updates_skipped,
            "total_updates_failed": tot_updates_failed,
            "event_metrics": event_summaries,
        }

        with open(condition_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        elapsed_total = time.perf_counter() - start_time_total
        completion_status = {
            "status": "COMPLETED",
            "evidence_level": self.evidence_level,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_total,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
        }
        with open(self.output_dir / "completion_status.json", "w", encoding="utf-8") as f:
            json.dump(completion_status, f, indent=2)

        return summary_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChangeGuard Week 13 Source-Anchored Consistency Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/week13_source_anchor_dev6_v1.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Explicit output directory override")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--synthetic_mock", action="store_true", help="Run with CPU synthetic mock data")
    parser.add_argument("--allow_overwrite", action="store_true", help="Permit writing into non-empty output directory")
    parser.add_argument("--validate_only", action="store_true", help="Run preflight validation and exit without running")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Week13SourceAnchorRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        synthetic_mock=args.synthetic_mock,
        allow_overwrite=args.allow_overwrite,
    )

    if args.validate_only:
        preflight = runner.validate_preflight(fail_closed=True)
        print("Preflight validation PASSED:")
        for k, v in preflight.items():
            print(f"  {k}: {v}")
        return

    print(f"\n>>> Running Week 13 Source-Anchored Adaptation: {runner.condition_id} ...")
    summary = runner.run_experiment()
    print("\n" + "=" * 80)
    print(" ChangeGuard Week 13 Source-Anchored Adaptation Completed Successfully!")
    print(f" Macro F1: {summary['macro_f1']:.4f} | Pooled F1: {summary['pooled_f1']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

