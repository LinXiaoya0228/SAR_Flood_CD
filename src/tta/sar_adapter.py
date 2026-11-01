"""
SAR (Sharpness-Aware and Reliable Test-Time Adaptation) adapter for ChangeGuard.

Upstream Provenance:
- Paper: "Towards Stable Test-Time Adaptation in Dynamic Wild World" (ICLR 2023 Oral)
- Upstream Authors: Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Minhao Chen, Peilin Zhao, Pengcheng Yuan, Mingkui Tan
- Official Repository: https://github.com/mr-eggplant/SAR
- Commit: 20f6e24b17525f34503510afccedc0629b67b7c4 (2023-09-08)
- License: BSD 3-Clause License

Semantics & Adaptations to Bi-Temporal SAR Flood Change Detection:
1. Model Configuration:
   - Sets model to train mode with track_running_stats = False (batch stats only).
   - Only BatchNorm affine parameters (weight, bias) receive gradients and updates.
   - All convolution, attention, classifier, and non-BN parameters remain frozen.
2. Sensory-Only Validity Masking:
   - Spatial entropy reduction averaged strictly over valid sensory pixels (MK0_MNA == 1).
   - Ground-truth labels (MK0_MLU) are strictly prohibited and never accessed.
3. Reliability Filtering:
   - Patch-level mean entropy H_b < E_0 = 0.40 * ln(3) ~ 0.4394449.
4. Two-Stage SAM Optimization:
   - Stage 1: Computes gradients on reliable samples and perturbs weights to theta + epsilon(theta).
   - Stage 2: Evaluates perturbed model, re-filters reliable samples, computes gradients, restores
     weights to theta, and steps base optimizer in the flat-minima direction.
5. Model Recovery Scheme:
   - Tracks running exponential moving average (EMA) of adapted entropy loss.
   - Configurable recovery threshold: e_m = 0.2 * ln(3) / ln(1000) ~ 0.03180808 (replaces upstream hardcoded 0.2).
   - If EMA drops below e_m, the model automatically resets to the initial source checkpoint.
6. Safety Fixes:
   - If stage 1 has zero reliable samples, update is skipped cleanly.
   - If stage 2 cannot proceed (zero reliable samples or numerical anomaly), `restore_perturbation()`
     safely reverts parameters from the perturbed state back to original weights before returning.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_diagnostics
from src.tta.sam import SAM


class SARAdapter(BaseOnlineAdapter):
    """SAR adapter adhering strictly to upstream mr-eggplant/SAR ICLR 2023 semantics."""

    ADAPTER_IDENTIFIER = "sar_v1"

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        margin_e0: Optional[float] = None,
        reset_constant_em: Optional[float] = None,
        sam_rho: float = 0.05,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__(model, checkpoint_path, device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.sam_rho = float(sam_rho)

        # Default E0 = 0.40 * ln(3) ~ 0.4394449 for 3-class segmentation
        if margin_e0 is None:
            self.margin_e0 = float(0.40 * math.log(3))
        else:
            self.margin_e0 = float(margin_e0)

        # Default recovery threshold: 0.2 * ln(3) / ln(1000) ~ 0.03180808
        if reset_constant_em is None:
            self.reset_constant_em = float(0.20 * math.log(3) / math.log(1000))
        else:
            self.reset_constant_em = float(reset_constant_em)

        self.ema: Optional[float] = None

        # Counters for execution diagnostics
        self.step_count = 0
        self.num_reliable_samples = 0
        self.num_adaptation_steps = 0
        self.num_skipped_updates = 0
        self.num_aborted_stage2 = 0
        self.num_recoveries = 0
        self.num_zero_valid_patches = 0

        self.optimizer: Optional[SAM] = None

        # Cache source BN running buffer shapes for clean restoration
        self._bn_buffer_shapes: Dict[str, Tuple[torch.Size, torch.Size]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.running_mean is not None and module.running_var is not None:
                    self._bn_buffer_shapes[name] = (
                        module.running_mean.shape,
                        module.running_var.shape,
                    )

        self._configure_model()
        self._init_optimizer()

    def _configure_model(self) -> None:
        """Configures model for SAR: train mode, non-BN frozen, BN affine trainable, track_running_stats=False."""
        self.model.train()
        self.model.requires_grad_(False)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def _init_optimizer(self) -> None:
        """Initializes SAM optimizer wrapping base SGD (upstream SAR default)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        base_optimizer = optim.SGD
        self.optimizer = SAM(
            params,
            base_optimizer,
            rho=self.sam_rho,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _recover_model_to_source(self) -> None:
        """Restores model weights to source and resets optimizer/EMA without clearing cumulative audit counters."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape))
                    module.register_buffer("running_var", torch.ones(var_shape))
                    module.register_buffer(
                        "num_batches_tracked", torch.tensor(0, dtype=torch.long)
                    )

        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)
        self._configure_model()
        self.ema = None
        self._init_optimizer()

    def _reset_optimizer(self) -> None:
        """Resets optimizer and tracking state for episodic adaptation."""
        self.step_count = 0
        self.num_reliable_samples = 0
        self.num_adaptation_steps = 0
        self.num_skipped_updates = 0
        self.num_aborted_stage2 = 0
        self.num_recoveries = 0
        self.num_zero_valid_patches = 0
        self.ema = None
        self._init_optimizer()

    def reset_to_source(self) -> None:
        """Restores model state dict bit-exactly to source checkpoint and resets optimizer/EMA."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape))
                    module.register_buffer("running_var", torch.ones(var_shape))
                    module.register_buffer(
                        "num_batches_tracked", torch.tensor(0, dtype=torch.long)
                    )

        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)

        self._configure_model()
        self._reset_optimizer()

    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Runs SAR adaptation and returns predictions from the initial forward pass."""
        self.model.train()
        batch_size = xA.size(0)

        # 1. Forward pass computes initial predictions (returned for evaluation)
        logits = self.model(xA, xB)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)

        # 2. Compute per-patch mean entropy over valid sensory pixels
        eps = 1e-12
        log_probs = torch.log(probs + eps)
        pixel_entropy = -torch.sum(probs * log_probs, dim=1)

        if sar_valid_mask is not None:
            mask = (
                sar_valid_mask.squeeze(1).float()
                if sar_valid_mask.dim() == 4
                else sar_valid_mask.float()
            )
        else:
            mask = torch.ones_like(pixel_entropy)

        patch_entropies = []
        valid_pixel_counts = []
        for b in range(batch_size):
            m_b = mask[b]
            v_b = torch.sum(m_b)
            valid_pixel_counts.append(v_b)
            if v_b > 0:
                h_b = torch.sum(pixel_entropy[b] * m_b) / v_b
            else:
                h_b = None
            patch_entropies.append(h_b)

        # Track and explicitly exclude patches with zero valid sensory pixels
        zero_valid_ids = [b for b, v in enumerate(valid_pixel_counts) if v == 0]
        self.num_zero_valid_patches += len(zero_valid_ids)

        # 3. Filter 1: Reliability Filtering (H_b < E_0 and v_b > 0)
        reliable_ids_1 = [
            b
            for b, h in enumerate(patch_entropies)
            if h is not None and float(h.item()) < self.margin_e0
        ]
        self.num_reliable_samples += len(reliable_ids_1)

        loss_val = 0.0
        stage1_loss_val = 0.0
        stage2_loss_val = 0.0
        recovered_this_step = False
        updated_this_step = False
        reliable_ids_2 = []

        if reliable_ids_1 and self.optimizer is not None:
            # SAM Stage 1: Ascent step on reliable samples
            loss1 = torch.stack([patch_entropies[b] for b in reliable_ids_1]).mean()
            stage1_loss_val = float(loss1.item())

            self.optimizer.zero_grad()
            loss1.backward()
            self.optimizer.first_step(zero_grad=True)

            try:
                # SAM Stage 2: Second forward pass with perturbed weights theta + epsilon(theta)
                logits2 = self.model(xA, xB)
                probs2 = F.softmax(logits2, dim=1)
                log_probs2 = torch.log(probs2 + eps)
                pixel_entropy2 = -torch.sum(probs2 * log_probs2, dim=1)

                patch_entropies2 = []
                for b in range(batch_size):
                    m_b = mask[b]
                    v_b = torch.sum(m_b)
                    if v_b > 0:
                        h_b2 = torch.sum(pixel_entropy2[b] * m_b) / v_b
                    else:
                        h_b2 = None
                    patch_entropies2.append(h_b2)

                # Re-filter reliable samples at perturbed state
                reliable_ids_2 = [
                    b
                    for b in reliable_ids_1
                    if patch_entropies2[b] is not None
                    and float(patch_entropies2[b].item()) < self.margin_e0
                ]

                if reliable_ids_2:
                    loss2 = torch.stack(
                        [patch_entropies2[b] for b in reliable_ids_2]
                    ).mean()
                    stage2_loss_val = float(loss2.item())
                    loss_val = stage2_loss_val

                    # Update running EMA of loss
                    if not math.isnan(stage2_loss_val) and not math.isinf(stage2_loss_val):
                        if self.ema is None:
                            self.ema = stage2_loss_val
                        else:
                            self.ema = 0.9 * self.ema + 0.1 * stage2_loss_val

                    # Model Recovery Scheme check: if ema < reset_constant_em, recover model
                    if self.ema is not None and self.ema < self.reset_constant_em:
                        self.optimizer.restore_perturbation(zero_grad=True)
                        self._recover_model_to_source()
                        self.num_recoveries += 1
                        recovered_this_step = True
                    else:
                        if torch.isfinite(loss2):
                            self.optimizer.zero_grad()
                            loss2.backward()
                            applied_step = self.optimizer.second_step(zero_grad=True)
                            if applied_step:
                                self.num_adaptation_steps += 1
                                updated_this_step = True
                            else:
                                self.num_skipped_updates += 1
                                updated_this_step = False
                        else:
                            self.optimizer.restore_perturbation(zero_grad=True)
                            self.num_skipped_updates += 1
                else:
                    # Safety restoration: stage 2 had no reliable samples, safely restore weights
                    self.optimizer.restore_perturbation(zero_grad=True)
                    self.num_aborted_stage2 += 1
                    loss_val = stage1_loss_val
            except Exception:
                # Crucial safety guarantee: ensure weights are restored if stage 2 raises an exception
                if getattr(self.optimizer, "perturbed", False):
                    self.optimizer.restore_perturbation(zero_grad=True)
                raise
        else:
            self.num_skipped_updates += 1

        self.step_count += 1

        # Diagnostics
        with torch.no_grad():
            diagnostics = compute_entropy_diagnostics(logits, sar_valid_mask)
            pred_flood = int(torch.sum(pred_class == 2).item())

            step_diagnostics = {
                "step_index": self.step_count,
                "batch_size": batch_size,
                "num_zero_valid_in_batch": len(zero_valid_ids),
                "num_reliable_in_batch": len(reliable_ids_1),
                "num_stage2_reliable_in_batch": len(reliable_ids_2),
                "updated_this_step": updated_this_step,
                "adaptation_loss": loss_val,
                "stage1_loss": stage1_loss_val,
                "stage2_loss": stage2_loss_val,
                "loss_ema": self.ema if self.ema is not None else float("nan"),
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": pred_flood,
                "bn_drift": self.compute_bn_drift(),
                "num_zero_valid_patches": self.num_zero_valid_patches,
                "num_reliable_samples": self.num_reliable_samples,
                "num_adaptation_steps": self.num_adaptation_steps,
                "num_skipped_updates": self.num_skipped_updates,
                "num_aborted_stage2": self.num_aborted_stage2,
                "num_recoveries": self.num_recoveries,
                "recovered_this_step": recovered_this_step,
            }

        return {
            "logits": logits.detach(),
            "probabilities": probs.detach(),
            "predicted_class": pred_class.detach(),
            "step_diagnostics": step_diagnostics,
        }

