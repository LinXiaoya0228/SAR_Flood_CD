"""
EATA (Efficient Anti-Forget Test-Time Adaptation) adapter for ChangeGuard.

Upstream Provenance:
- Paper: "Efficient Test-Time Adaptation against Distribution Shifts" (ICML 2022)
- Upstream Authors: Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan
- Official Repository: https://github.com/mr-eggplant/EATA
- Commit: f739b3668cc7617e9b9f1979c1a358497a3472c3 (2023-05-19)
- License: MIT License

Semantics & Adaptations to Bi-Temporal SAR Flood Change Detection:
1. Model Configuration:
   - Sets model to train mode with track_running_stats = False (batch stats only).
   - Only BatchNorm affine parameters (weight, bias) receive gradients and updates.
   - All convolution, attention, classifier, and non-BN parameters remain frozen.
2. Sensory-Only Validity Masking:
   - Entropy and probability vectors are averaged strictly over valid sensory pixels (MK0_MNA == 1).
   - Ground-truth labels (MK0_MLU) and label-derived masks are strictly prohibited and never accessed.
3. Reliability Filtering:
   - Patch-level mean entropy H_b < E_0 = 0.40 * ln(3) ~ 0.4394449.
4. Redundancy Filtering:
   - Compares patch-mean class probability vector p_b against running EMA p_bar.
   - Reliable sample b is retained only if |cos(p_b, p_bar)| < d_margin (0.05).
5. Loss Reweighting:
   - Weight w_b = 1 / exp(H_b - E_0) applied to each active sample's entropy.
6. Fisher Anti-Forgetting Regularization:
   - Quadratic penalty against initial source weights: sum_l (alpha / 2) * F_l * (theta_l - theta_l^0)^2.
   - Precomputed strictly on approved source training events (events in configs/splits/kuro_source_all27_split_v1.json).
7. Safety Fixes:
   - If no samples pass filtering in a batch, update is cleanly skipped (no backward, no step).
   - Zero GT access verified.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_diagnostics


class EATAAdapter(BaseOnlineAdapter):
    """EATA adapter adhering strictly to upstream mr-eggplant/EATA ICML 2022 semantics."""

    ADAPTER_IDENTIFIER = "eata_v1"

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2000.0,
        e_margin: Optional[float] = None,
        d_margin: float = 0.05,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__(model, checkpoint_path, device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.fisher_alpha = float(fisher_alpha)
        self.d_margin = float(d_margin)

        # Default E0 = 0.40 * ln(3) ~ 0.4394449 for 3-class segmentation
        if e_margin is None:
            self.e_margin = float(0.40 * math.log(3))
        else:
            self.e_margin = float(e_margin)

        # Fisher regularizer: dict of {param_name: (fisher_diagonal, initial_param_copy)}
        self.fishers = fishers

        # Running probability vector EMA (C=3)
        self.current_model_probs: Optional[torch.Tensor] = None

        # Counters for execution diagnostics
        self.step_count = 0
        self.num_reliable_samples = 0
        self.num_redundant_samples = 0
        self.num_active_samples = 0
        self.num_adaptation_steps = 0
        self.num_skipped_updates = 0
        self.num_zero_valid_patches = 0

        self.optimizer: Optional[optim.SGD] = None

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
        """Configures model for EATA: train mode, non-BN frozen, BN affine trainable, track_running_stats=False."""
        self.model.train()
        self.model.requires_grad_(False)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def _init_optimizer(self) -> None:
        """Initializes SGD optimizer over active BN affine parameters (upstream EATA default)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _reset_optimizer(self) -> None:
        """Resets optimizer and counters for episodic adaptation."""
        self.step_count = 0
        self.num_reliable_samples = 0
        self.num_redundant_samples = 0
        self.num_active_samples = 0
        self.num_adaptation_steps = 0
        self.num_skipped_updates = 0
        self.current_model_probs = None
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
        """Runs EATA adaptation and returns predictions from the initial forward pass."""
        self.model.train()
        batch_size = xA.size(0)

        # 1. Forward pass computes logits and predictions
        logits = self.model(xA, xB)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        pred_class = torch.argmax(logits, dim=1)  # (B, H, W)

        # 2. Compute per-patch mean entropy and probability vectors over valid sensory pixels
        eps = 1e-12
        log_probs = torch.log(probs + eps)
        pixel_entropy = -torch.sum(probs * log_probs, dim=1)  # (B, H, W)

        if sar_valid_mask is not None:
            mask = (
                sar_valid_mask.squeeze(1).float()
                if sar_valid_mask.dim() == 4
                else sar_valid_mask.float()
            )
        else:
            mask = torch.ones_like(pixel_entropy)

        patch_entropies = []
        patch_probs = []
        valid_pixel_counts = []
        for b in range(batch_size):
            m_b = mask[b]
            v_b = torch.sum(m_b)
            valid_pixel_counts.append(v_b)
            if v_b > 0:
                h_b = torch.sum(pixel_entropy[b] * m_b) / v_b
                p_b = torch.sum(probs[b] * m_b.unsqueeze(0), dim=(1, 2)) / v_b
            else:
                h_b = None
                p_b = None
            patch_entropies.append(h_b)
            patch_probs.append(p_b)

        # Track and explicitly exclude patches with zero valid sensory pixels
        zero_valid_ids = [b for b, v in enumerate(valid_pixel_counts) if v == 0]
        self.num_zero_valid_patches += len(zero_valid_ids)

        # 3. Filter 1: Reliability Filtering (H_b < E_0 and v_b > 0)
        reliable_ids = [
            b
            for b, h in enumerate(patch_entropies)
            if h is not None and float(h.item()) < self.e_margin
        ]
        self.num_reliable_samples += len(reliable_ids)

        # 4. Filter 2: Redundancy Filtering (Cosine similarity against running EMA probability vector)
        active_ids = []
        if self.current_model_probs is None:
            # First time: all reliable samples pass redundancy filter
            active_ids = list(reliable_ids)
        else:
            target_ema = self.current_model_probs.to(self.device).unsqueeze(0)
            for b in reliable_ids:
                p_b_vec = patch_probs[b].unsqueeze(0)
                cos_sim = F.cosine_similarity(target_ema, p_b_vec, dim=1).item()
                if abs(cos_sim) < self.d_margin:
                    active_ids.append(b)
                else:
                    self.num_redundant_samples += 1

        self.num_active_samples += len(active_ids)

        # Update EMA probability vector with newly selected active samples (never with zero-valid patches)
        if active_ids:
            with torch.no_grad():
                new_probs_mean = torch.stack([patch_probs[b] for b in active_ids]).mean(
                    dim=0
                )
                if self.current_model_probs is None:
                    self.current_model_probs = new_probs_mean.detach().clone()
                else:
                    self.current_model_probs = (
                        0.9 * self.current_model_probs + 0.1 * new_probs_mean
                    ).detach().clone()

        # 5. Compute loss and optimize if active samples exist
        loss_val = 0.0
        entropy_loss_val = 0.0
        fisher_loss_val = 0.0
        updated_this_step = False

        if active_ids and self.optimizer is not None:
            # Reweighted entropy loss
            weighted_losses = []
            for b in active_ids:
                h_b = patch_entropies[b]
                coeff = 1.0 / torch.exp(h_b.detach() - self.e_margin)
                weighted_losses.append(h_b * coeff)
            entropy_loss = torch.stack(weighted_losses).mean()
            entropy_loss_val = float(entropy_loss.item())

            # Fisher anti-forgetting regularization
            fisher_loss = torch.tensor(0.0, device=self.device)
            if self.fishers is not None:
                for name, param in self.model.named_parameters():
                    if name in self.fishers:
                        f_diag, p_init = self.fishers[name]
                        f_diag = f_diag.to(param.device)
                        p_init = p_init.to(param.device)
                        fisher_loss = fisher_loss + self.fisher_alpha * (
                            f_diag * ((param - p_init) ** 2)
                        ).sum()
                fisher_loss_val = float(fisher_loss.item())

            total_loss = entropy_loss + fisher_loss
            loss_val = float(total_loss.item())

            if torch.isfinite(total_loss):
                self.optimizer.zero_grad()
                total_loss.backward()

                # Ensure gradients are finite
                all_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        all_finite = False
                        break

                if all_finite:
                    self.optimizer.step()
                    self.num_adaptation_steps += 1
                    updated_this_step = True
                else:
                    self.num_skipped_updates += 1
            else:
                self.num_skipped_updates += 1
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
                "num_reliable_in_batch": len(reliable_ids),
                "num_active_in_batch": len(active_ids),
                "num_redundant_in_batch": len(reliable_ids) - len(active_ids),
                "updated_this_step": updated_this_step,
                "adaptation_loss": loss_val,
                "entropy_loss": entropy_loss_val,
                "fisher_loss": fisher_loss_val,
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": pred_flood,
                "bn_drift": self.compute_bn_drift(),
                "num_zero_valid_patches": self.num_zero_valid_patches,
                "num_reliable_samples": self.num_reliable_samples,
                "num_redundant_samples": self.num_redundant_samples,
                "num_active_samples": self.num_active_samples,
                "num_adaptation_steps": self.num_adaptation_steps,
                "num_skipped_updates": self.num_skipped_updates,
            }

        return {
            "logits": logits.detach(),
            "probabilities": probs.detach(),
            "predicted_class": pred_class.detach(),
            "step_diagnostics": step_diagnostics,
        }

