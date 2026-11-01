"""ChangeGuard Work Package E7: CoTTA Adaptation Adapter.

Implements the official CoTTA (Continual Test-Time Domain Adaptation, Wang et al., CVPR 2022)
baseline adapted to paired multi-temporal SAR flood change detection.

Adheres strictly to docs/protocols/e6_cotta_compatibility_protocol_v1.md and
docs/protocols/e7_branch_c_empirical_protocol_v1.md:
1. Mean Teacher Network with EMA:
   Teacher theta_t tracks student theta_s via theta_t <- alpha * theta_t + (1 - alpha) * theta_s
   with alpha = 0.999. Teacher parameters never receive direct gradients.
2. Augmentation-Averaged Teacher Predictions:
   Evaluates 4 exact SAR spatial transformations (Identity, Horizontal Flip, Vertical Flip, Rot180)
   applied identically to (xA, xB). Inverted via A_k^{-1} and averaged.
3. Stochastic Source Restoration (SSR):
   At each step, each trainable parameter element has an independent probability p = 0.01 of
   resetting bit-exactly to initial source checkpoint theta_0.
4. Trainable Parameters:
   BatchNorm affine parameters (weight, bias) only; conv weights frozen; tracking stats disabled.
5. Sensory-Valid Mask Enforcement:
   Loss is evaluated strictly on valid pixels (M_NA == 1) where teacher confidence >= tau = 0.90.
6. Non-Aliasing:
   Student, teacher, and source models share zero memory storage or pointers.
7. Timing:
   Primary predictions are strictly PRE_UPDATE teacher predictions.
8. Operational Modes:
   cotta_continual (unbroken stream) and cotta_reset_per_event (episodic reset).
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.changeguard.baseline_adapters import BaseE1Adapter
from src.changeguard.recorder import (
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
)


def apply_sar_spatial_augmentation(
    xA: torch.Tensor, xB: torch.Tensor, view_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies exact spatial transformation jointly to bi-temporal SAR rasters.

    Views:
        0: Identity (rot0)
        1: Horizontal Flip (hflip)
        2: Vertical Flip (vflip)
        3: Horizontal + Vertical Flip (rot180)
    """
    if view_idx == 0:
        return xA, xB
    elif view_idx == 1:
        return torch.flip(xA, dims=[-1]), torch.flip(xB, dims=[-1])
    elif view_idx == 2:
        return torch.flip(xA, dims=[-2]), torch.flip(xB, dims=[-2])
    elif view_idx == 3:
        return torch.flip(xA, dims=[-2, -1]), torch.flip(xB, dims=[-2, -1])
    else:
        raise ValueError(f"Invalid view index {view_idx}; expected 0, 1, 2, or 3.")


def invert_sar_spatial_tensor(tensor: torch.Tensor, view_idx: int) -> torch.Tensor:
    """Inverts the spatial transformation for a tensor (self-inverse spatial transforms)."""
    if view_idx == 0:
        return tensor
    elif view_idx == 1:
        return torch.flip(tensor, dims=[-1])
    elif view_idx == 2:
        return torch.flip(tensor, dims=[-2])
    elif view_idx == 3:
        return torch.flip(tensor, dims=[-2, -1])
    else:
        raise ValueError(f"Invalid view index {view_idx}; expected 0, 1, 2, or 3.")


def assert_module_storage_disjoint(
    m1: nn.Module, m2: nn.Module, name1: str, name2: str
) -> None:
    """Verifies that two models share zero parameter or buffer storage addresses."""
    if m1 is m2:
        raise RuntimeError(f"FATAL ALIASING: {name1} and {name2} are identical objects.")
    ptrs1 = {p.data_ptr() for p in m1.parameters()} | {
        b.data_ptr() for b in m1.buffers() if b is not None
    }
    ptrs2 = {p.data_ptr() for p in m2.parameters()} | {
        b.data_ptr() for b in m2.buffers() if b is not None
    }
    shared = ptrs1.intersection(ptrs2)
    if shared:
        raise RuntimeError(
            f"FATAL STORAGE ALIASING: {len(shared)} shared pointer(s) between {name1} and {name2}."
        )


class E7CoTTAAdapter(BaseE1Adapter):
    """ChangeGuard E7 Continual Test-Time Domain Adaptation (CoTTA) Adapter."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
        ema_alpha: float = 0.999,
        restoration_prob: float = 0.01,
        confidence_threshold: float = 0.90,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        mode: str = "reset_per_event",
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.learning_rate = float(learning_rate)
        self.ema_alpha = float(ema_alpha)
        self.restoration_prob = float(restoration_prob)
        self.confidence_threshold = float(confidence_threshold)
        self.betas = tuple(betas)
        self.weight_decay = float(weight_decay)

        if mode not in ("reset_per_event", "continual"):
            raise ValueError(f"Mode must be 'reset_per_event' or 'continual', got '{mode}'")
        self.mode = mode
        self.seed = int(seed)

        # Build independent teacher model from source checkpoint
        self.teacher_model = copy.deepcopy(self.source_model).to(self.device)
        self._configure_teacher_bn()

        # Configure student candidate model BN
        self._configure_student_bn()

        # Verify strict non-aliasing across all three models
        self._assert_non_aliasing()

        # Initialize optimizer for student trainable parameters
        self._init_optimizer()

        # Initialize deterministic RNG for stochastic source restoration
        self.rng = torch.Generator(device="cpu").manual_seed(self.seed)

    def _configure_teacher_bn(self) -> None:
        """Configures teacher model: BN in train mode without running stats tracking."""
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        for module in self.teacher_model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)

    def _configure_student_bn(self) -> None:
        """Configures student model: BN affine parameters trainable, stats untracked."""
        self.model.eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    def _assert_non_aliasing(self) -> None:
        """Asserts zero memory overlap between student, teacher, and source models."""
        assert_module_storage_disjoint(self.model, self.source_model, "student", "source")
        assert_module_storage_disjoint(self.model, self.teacher_model, "student", "teacher")
        assert_module_storage_disjoint(self.teacher_model, self.source_model, "teacher", "source")

    def _init_optimizer(self) -> None:
        """Initializes Adam optimizer for student trainable BN affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BN affine parameters found for CoTTA student.")
        self.optimizer = optim.Adam(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def reset_to_source(self) -> None:
        """Restores student and teacher state bit-exactly to source checkpoint."""
        # 1. Reset student BN buffers and load source state
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
        self._configure_student_bn()

        # 2. Reset teacher BN buffers and load source state
        for name, module in self.teacher_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape, device=self.device))
                    module.register_buffer("running_var", torch.ones(var_shape, device=self.device))
                    module.register_buffer(
                        "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=self.device)
                    )
        self.teacher_model.load_state_dict(self._source_state_dict, strict=True)
        self.teacher_model.to(self.device)
        self._configure_teacher_bn()

        # 3. Assert non-aliasing holds after reset
        self._assert_non_aliasing()

        # 4. Reinitialize optimizer and deterministic RNG
        self._init_optimizer()
        self.rng = torch.Generator(device="cpu").manual_seed(self.seed)

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
        """Executes CoTTA adaptation step and returns predictions and diagnostics.

        Strict Plan V3 Timing:
        - Primary prediction recorded is PRE_UPDATE augmentation-averaged teacher prediction.
        - Zero ground-truth label access permitted during execution.
        """
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        # 1. Source reference forward (frozen eval)
        with torch.no_grad():
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # 2. PRE_UPDATE Augmentation-Averaged Teacher Forward (K=4 views)
        with torch.no_grad():
            probs_views = []
            logits_views = []
            for k in range(4):
                xA_k, xB_k = apply_sar_spatial_augmentation(xA, xB, k)
                logits_k = self.teacher_model(xA_k, xB_k)
                probs_k = F.softmax(logits_k, dim=1)
                inv_probs_k = invert_sar_spatial_tensor(probs_k, k)
                inv_logits_k = invert_sar_spatial_tensor(logits_k, k)
                probs_views.append(inv_probs_k)
                logits_views.append(inv_logits_k)

            probs_t_avg = torch.stack(probs_views, dim=0).mean(dim=0)
            logits_t_avg = torch.stack(logits_views, dim=0).mean(dim=0)

        # Primary PRE_UPDATE predictions
        pred_t_pre = probs_t_avg.argmax(dim=1)
        max_conf_t, _ = torch.max(probs_t_avg, dim=1)
        pseudo_label = pred_t_pre

        # 3. Sensory-valid selection mask: M_NA == 1 AND max_conf >= tau (0.90)
        conf_mask = (max_conf_t >= self.confidence_threshold).float()
        selection_mask = V * conf_mask
        selected_pixels = int(selection_mask.sum().item())

        # Compute pre-update risk indicators over valid pixels
        eps = 1e-12
        log_probs_t = torch.log(probs_t_avg + eps)
        entropy = -torch.sum(probs_t_avg * log_probs_t, dim=1)
        mean_entropy = float((entropy * V).sum().item() / max(valid_pixel_count, 1))
        mean_max_conf = float((max_conf_t * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_t_avg[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))
        src_disagreement = float(((pred_t_pre != pred_src).float() * V).sum().item() / max(valid_pixel_count, 1))

        # 4. Student Adaptation Step
        action = "skip"
        update_applied = False
        loss_val = None
        grad_norm_l2 = None
        failure_reason = None

        if selected_pixels == 0:
            self.num_updates_skipped += 1
            action = "skip"
        else:
            # Student forward pass on original unaugmented views
            logits_student = self.model(xA, xB)
            ce_loss = F.cross_entropy(logits_student, pseudo_label, reduction="none")
            loss = (ce_loss * selection_mask).sum() / max(selection_mask.sum(), 1.0)

            # Check loss finiteness
            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                action = "rollback"
                failure_reason = "Non-finite student loss encountered"
            else:
                # Cache pre-step states for fail-closed rollback
                pre_step_student = {
                    n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
                }
                pre_step_teacher = {
                    n: p.detach().clone() for n, p in self.teacher_model.named_parameters()
                }
                pre_step_opt = copy.deepcopy(self.optimizer.state_dict())

                self.optimizer.zero_grad()
                loss.backward()

                # Verify gradient finiteness
                grads_finite = True
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break

                if not grads_finite:
                    # Roll back immediately
                    self.optimizer.zero_grad()
                    self.num_updates_failed += 1
                    action = "rollback"
                    failure_reason = "Non-finite gradients encountered; rolled back"
                else:
                    grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 5. Mean Teacher EMA Update: theta_t <- alpha * theta_t + (1 - alpha) * theta_s
                    with torch.no_grad():
                        for (n_t, p_t), (n_s, p_s) in zip(
                            self.teacher_model.named_parameters(), self.model.named_parameters()
                        ):
                            p_t.data.mul_(self.ema_alpha).add_(p_s.data, alpha=1.0 - self.ema_alpha)

                    # 6. Stochastic Source Restoration (SSR): p = 0.01 per trainable element
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and name in self._source_state_dict:
                                src_val = self._source_state_dict[name]
                                rand_t = torch.rand(
                                    param.shape, generator=self.rng, device="cpu"
                                ).to(param.device)
                                restore_mask = rand_t < self.restoration_prob
                                param.data.copy_(torch.where(restore_mask, src_val, param.data))

                    # 7. Post-step parameter sanity check
                    student_finite = self._is_state_dict_finite(self.model.state_dict())
                    teacher_finite = self._is_state_dict_finite(self.teacher_model.state_dict())

                    if not (student_finite and teacher_finite):
                        # Roll back student, teacher, and optimizer
                        for n, p in self.model.named_parameters():
                            if n in pre_step_student:
                                p.data.copy_(pre_step_student[n])
                        for n, p in self.teacher_model.named_parameters():
                            if n in pre_step_teacher:
                                p.data.copy_(pre_step_teacher[n])
                        self.optimizer.load_state_dict(pre_step_opt)
                        self.optimizer.zero_grad()
                        self.num_updates_failed += 1
                        action = "rollback"
                        failure_reason = "Non-finite parameters after step/EMA/SSR; rolled back"
                    else:
                        self.num_updates_applied += 1
                        update_applied = True
                        action = "update"
                        loss_val = float(loss.item())

        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # Diagnostic POST_UPDATE forward pass (teacher model)
        with torch.no_grad():
            logits_post = self.teacher_model(xA, xB)
            probs_post = F.softmax(logits_post, dim=1)
            pred_post = probs_post.argmax(dim=1)

        cls_counts = compute_per_class_pixel_counts(pred_t_pre, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_t_pre.detach(),
                "probs_cand_pre": probs_t_avg.detach(),
                "logits_cand_pre": logits_t_avg.detach(),
                "pred_src": pred_src.detach(),
                "probs_src": probs_src.detach(),
                "valid_pixels": valid_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred_post.detach(),
                "probs_cand_post": probs_post.detach(),
                "logits_cand_post": logits_post.detach(),
            },
            "adaptation_metadata": {
                "action": action,
                "update_applied": update_applied,
                "total_loss": loss_val,
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": float(selected_pixels / max(valid_pixel_count, 1)),
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
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
