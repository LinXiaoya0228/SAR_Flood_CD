"""ChangeGuard Plan V3 Work Package E2.2: Augmentation Adapters.

Implements geometric test-time consistency adaptation variants:
- A0_HFLIP: Paired Horizontal Flip (reference condition).
- A1_VFLIP: Paired Vertical Flip (exact coordinate inversion).
- A2_TRANSLATION: Small Paired Integer Translation with declared overlap mask and padding exclusion.
- A3_INDEPENDENT_WEAK: Independent Weak Views (Weak 1 H-flip vs Weak 2 V-flip).

Strict Invariants:
1. Paired transformations are synchronized across both temporal dates (xA and xB).
2. Strong predictions, probabilities and validity masks return to native/original coordinates before loss.
3. Translation consistency is evaluated strictly on the declared geometric overlap (padding excluded).
4. All evaluation, primary caching, and metrics are computed strictly in original coordinates.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.changeguard.recorder import (
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
)


class E2AugmentationAdapter:
    """Configurable Weak-Strong Consistency Adapter supporting Plan V3 Augmentation Grid."""

    SUPPORTED_AUGS = {"horizontal_flip", "vertical_flip", "translation", "independent_weak"}

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        aug_type: str = "horizontal_flip",
        shift: Tuple[int, int] = (4, 4),
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        confidence_threshold: float = 0.90,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.aug_type = str(aug_type).lower()
        if self.aug_type not in self.SUPPORTED_AUGS:
            raise ValueError(f"Unsupported aug_type '{self.aug_type}'. Must be one of {self.SUPPORTED_AUGS}.")

        self.shift = (int(shift[0]), int(shift[1]))
        if self.shift[0] < 0 or self.shift[1] < 0:
            raise ValueError(f"Negative shifts not supported: {self.shift}")

        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.confidence_threshold = float(confidence_threshold)

        # Cache source state dict for exact bit-level restoration
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

        # Source reference model (immutable reference, independent storage)
        if source_model is not None:
            self.source_model = copy.deepcopy(source_model).to(self.device)
        else:
            self.source_model = copy.deepcopy(self.model).to(self.device)
        self.source_model.eval()
        self.source_model.requires_grad_(False)

        # Verify no aliasing
        if self.source_model is self.model:
            raise RuntimeError("Source model cannot be identical object to candidate model.")
        cand_ptrs = {p.data_ptr() for p in self.model.parameters()} | {
            b.data_ptr() for b in self.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in self.source_model.parameters()} | {
            b.data_ptr() for b in self.source_model.buffers() if b is not None
        }
        shared = cand_ptrs.intersection(src_ptrs)
        if shared:
            raise RuntimeError(f"Storage aliasing detected between candidate and source model: {shared}")

        self._configure_model_bn()
        self._init_optimizer()

        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def _configure_model_bn(self) -> None:
        """Configures candidate model: eval mode globally, selected BN modules train() with no tracking."""
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
            raise RuntimeError("No trainable BatchNorm parameters found in candidate model.")
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
            return all(E2AugmentationAdapter._is_state_dict_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(E2AugmentationAdapter._is_state_dict_finite(v) for v in obj)
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

    def adapt_and_predict_step(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_ids: Sequence[str],
        event_id: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs paired consistency adaptation under the configured augmentation strategy."""
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        valid_mask = valid_mask.to(self.device)

        if xA.shape != xB.shape:
            raise ValueError(f"Temporal inputs xA {xA.shape} and xB {xB.shape} must have identical shape.")
        if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)
        elif valid_mask.dim() != 3:
            raise ValueError(f"Unsupported sensory mask shape {valid_mask.shape}.")

        B, C, H, W = xA.shape
        sensory_valid = (valid_mask == 1)
        sensory_valid_pixels = int(sensory_valid.sum().item())

        # ---------------------------------------------------------------------
        # 1. Primary Weak Forward Pass (in original/native coordinate frame)
        # ---------------------------------------------------------------------
        if self.aug_type in {"horizontal_flip", "vertical_flip", "translation"}:
            # Standard weak view = Identity
            logits_cand_pre = self.model(xA, xB)
            probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
            pred_cand_pre = probs_cand_pre.argmax(dim=1)
            probs_cand_pre_det = probs_cand_pre.detach()
            probs_cand_max = probs_cand_pre_det.max(dim=1)[0]

            if self.aug_type in {"horizontal_flip", "vertical_flip"}:
                overlap_mask = torch.ones_like(sensory_valid, dtype=torch.bool)
            else:  # translation
                dy, dx = self.shift
                overlap_mask = torch.zeros_like(sensory_valid, dtype=torch.bool)
                overlap_mask[:, :H - dy, :W - dx] = True

        elif self.aug_type == "independent_weak":
            # Weak View 1 = Horizontal Flip; primary prediction inverted to original coordinates
            xA_w1 = torch.flip(xA, dims=[-1])
            xB_w1 = torch.flip(xB, dims=[-1])
            logits_w1 = self.model(xA_w1, xB_w1)
            logits_cand_pre = torch.flip(logits_w1, dims=[-1])  # Realigned to native coordinates!
            probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
            pred_cand_pre = probs_cand_pre.argmax(dim=1)
            probs_cand_pre_det = probs_cand_pre.detach()
            probs_cand_max = probs_cand_pre_det.max(dim=1)[0]
            overlap_mask = torch.ones_like(sensory_valid, dtype=torch.bool)
        else:
            raise ValueError(f"Unhandled aug_type: {self.aug_type}")

        # Active validity mask in original coordinates (excludes non-overlap & padding)
        V_active = sensory_valid & overlap_mask
        valid_pixel_count = int(V_active.sum().item())
        overlap_pixel_count = int(overlap_mask.sum().item())

        forward_finite = (
            torch.isfinite(logits_cand_pre).all() and torch.isfinite(probs_cand_pre).all()
        )

        # Full-precision summary statistics over valid pixels
        if valid_pixel_count > 0 and forward_finite:
            H_pixel = -(probs_cand_pre_det * torch.log(probs_cand_pre_det + 1e-12)).sum(dim=1)
            mean_entropy = float(((H_pixel * V_active.float()).sum() / (V_active.sum() * math.log(3))).item())
            mean_max_conf = float(((probs_cand_max * V_active.float()).sum() / V_active.sum()).item())
            p_c2_pre = probs_cand_pre_det[:, 2]
            mean_prob_c2 = float(((p_c2_pre * V_active.float()).sum() / V_active.sum()).item())
        else:
            mean_entropy = None
            mean_max_conf = None
            mean_prob_c2 = None

        # Selection Masking
        if forward_finite:
            selection_mask = (probs_cand_max >= self.confidence_threshold) & V_active
            selected_pixels = int(selection_mask.sum().item())
            selected_pixel_fraction = (
                float(selected_pixels / valid_pixel_count) if valid_pixel_count > 0 else 0.0
            )

            # Classwise selection breakdown
            c_counts = compute_per_class_pixel_counts(
                predictions=pred_cand_pre,
                selection_mask=selection_mask,
                valid_mask=V_active,
            )
            selected_c0 = c_counts[0]
            selected_c1 = c_counts[1]
            selected_c2 = c_counts[2]
        else:
            selection_mask = torch.zeros_like(V_active, dtype=torch.bool)
            selected_pixels = 0
            selected_pixel_fraction = 0.0
            selected_c0 = 0
            selected_c1 = 0
            selected_c2 = 0

        # ---------------------------------------------------------------------
        # 2. Immutable Source Reference PRE_UPDATE Forward Pass (Native Coordinates)
        # ---------------------------------------------------------------------
        src_rng_state = torch.get_rng_state()
        src_cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if (self.device.type == "cuda" and torch.cuda.is_available())
            else None
        )
        src_buffers_before = {k: v.clone() for k, v in self.source_model.named_buffers()}
        try:
            with torch.no_grad():
                logits_src = self.source_model(xA, xB)
                probs_src = F.softmax(logits_src, dim=1)
                pred_src = probs_src.argmax(dim=1)
                if valid_pixel_count > 0:
                    src_disagreement = float(
                        ((pred_src != pred_cand_pre) & V_active).float().sum().item() / valid_pixel_count
                    )
                else:
                    src_disagreement = None
        finally:
            torch.set_rng_state(src_rng_state)
            if src_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(src_cuda_rng_state)
            for k, v in self.source_model.named_buffers():
                if k in src_buffers_before:
                    v.copy_(src_buffers_before[k])

        # ---------------------------------------------------------------------
        # 3. Strong View Forward Pass & Realignment to Original Frame
        # ---------------------------------------------------------------------
        if self.aug_type == "horizontal_flip":
            xA_strong = torch.flip(xA, dims=[-1])
            xB_strong = torch.flip(xB, dims=[-1])
            logits_strong = self.model(xA_strong, xB_strong)
            logits_strong_aligned = torch.flip(logits_strong, dims=[-1])

        elif self.aug_type == "vertical_flip":
            xA_strong = torch.flip(xA, dims=[-2])
            xB_strong = torch.flip(xB, dims=[-2])
            logits_strong = self.model(xA_strong, xB_strong)
            logits_strong_aligned = torch.flip(logits_strong, dims=[-2])

        elif self.aug_type == "translation":
            dy, dx = self.shift
            xA_strong = torch.zeros_like(xA)
            xB_strong = torch.zeros_like(xB)
            xA_strong[:, :, dy:, dx:] = xA[:, :, :H - dy, :W - dx]
            xB_strong[:, :, dy:, dx:] = xB[:, :, :H - dy, :W - dx]
            logits_strong = self.model(xA_strong, xB_strong)

            logits_strong_aligned = torch.zeros_like(logits_strong)
            logits_strong_aligned[:, :, :H - dy, :W - dx] = logits_strong[:, :, dy:, dx:]

        elif self.aug_type == "independent_weak":
            # Weak View 2 = Vertical Flip; realigned to native coordinates
            xA_w2 = torch.flip(xA, dims=[-2])
            xB_w2 = torch.flip(xB, dims=[-2])
            logits_w2 = self.model(xA_w2, xB_w2)
            logits_strong_aligned = torch.flip(logits_w2, dims=[-2])

        log_probs_strong_aligned = F.log_softmax(logits_strong_aligned, dim=1)

        # ---------------------------------------------------------------------
        # 4. Consistency Loss & Optimizer Step
        # ---------------------------------------------------------------------
        action = "skip"
        update_applied = False
        loss_val: Optional[float] = None
        grad_norm_l2: Optional[float] = None
        failure_reason: Optional[str] = None

        if not forward_finite:
            self.num_updates_failed += 1
            action = "failure"
            update_applied = False
            failure_reason = "Non-finite logits or probabilities encountered during candidate forward pass"
            loss_val = None
            grad_norm_l2 = None
        elif selected_pixels == 0 or valid_pixel_count == 0:
            self.num_updates_skipped += 1
            action = "skip"
            update_applied = False
            loss_val = None
            grad_norm_l2 = None
        else:
            kl_pixel = (
                probs_cand_pre_det * (torch.log(probs_cand_pre_det + 1e-12) - log_probs_strong_aligned)
            ).sum(dim=1)
            loss = (kl_pixel * selection_mask.float()).sum() / selection_mask.sum().clamp(min=1.0)

            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                action = "failure"
                update_applied = False
                failure_reason = f"Non-finite loss encountered: {loss.item()}"
                loss_val = None
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())

                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [
                    p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()
                ]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    update_applied = False
                    failure_reason = "Non-finite gradient encountered during backward pass"
                    loss_val = None
                    self.optimizer.zero_grad()
                else:
                    pre_step_params = {
                        n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
                    }
                    pre_step_opt_state = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nonfinite_params = [
                        n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()
                    ]
                    optimizer_state_finite = self._is_state_dict_finite(self.optimizer.state_dict())

                    if nonfinite_params or not optimizer_state_finite:
                        for n, p in self.model.named_parameters():
                            if n in pre_step_params:
                                p.data.copy_(pre_step_params[n])
                        self.optimizer.load_state_dict(pre_step_opt_state)
                        self.optimizer.zero_grad()

                        self.num_updates_failed += 1
                        action = "failure"
                        update_applied = False
                        failure_reason = "Post-step corruption detected; restored pre-step state"
                        loss_val = None
                    else:
                        self.num_updates_applied += 1
                        action = "update"
                        update_applied = True
                        loss_val = float(loss.item())

        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # ---------------------------------------------------------------------
        # 5. POST_UPDATE Diagnostic Forward Pass (Native Coordinates)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            logits_cand_post = self.model(xA, xB)
            probs_cand_post = F.softmax(logits_cand_post, dim=1)
            pred_cand_post = probs_cand_post.argmax(dim=1)

        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach(),
                "probs_cand_pre": probs_cand_pre_det,
                "logits_cand_pre": logits_cand_pre.detach(),
                "pred_src": pred_src.detach(),
                "probs_src": probs_src.detach(),
                "valid_pixels": valid_pixel_count,
                "sensory_valid_pixels": sensory_valid_pixels,
                "overlap_pixels": overlap_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred_cand_post.detach(),
                "probs_cand_post": probs_cand_post.detach(),
                "logits_cand_post": logits_cand_post.detach(),
            },
            "adaptation_metadata": {
                "action": action,
                "update_applied": update_applied,
                "total_loss": loss_val,
                "consistency_loss": loss_val,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "sensory_valid_pixels": sensory_valid_pixels,
                "overlap_pixels": overlap_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixels_c0": selected_c0,
                "selected_pixels_c1": selected_c1,
                "selected_pixels_c2": selected_c2,
                "selected_pixel_fraction": selected_pixel_fraction,
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
                "confidence_threshold": self.confidence_threshold,
                "aug_type": self.aug_type,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }

