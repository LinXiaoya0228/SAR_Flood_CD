"""
Reference-aligned Tent (Test-Time Entropy Minimization) adapter for ChangeGuard.

Reference:
DequanWang/tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
Upstream implementation: https://github.com/DequanWang/tent/blob/master/tent.py
Revision: master branch (commit reference DequanWang/tent@master)

Semantics aligned with upstream reference:
1. Only BatchNorm affine parameters (weight, bias) receive gradients and optimizer updates.
   All convolution, attention, classifier, and non-BN parameters remain strictly frozen.
2. Disable BatchNorm running-statistics tracking (track_running_stats = False) and use
   batch statistics for normalization in both train and eval modes (running_mean = None,
   running_var = None), exactly as in upstream tent.py configure_model().
3. Single forward pass per batch: computes both the entropy loss and final predictions.
4. One optimizer step per batch (Adam with lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0).
5. Predictions returned are those computed by the single forward pass; NO second post-update
   forward is executed.

SAR-Specific Adaptations (Explicitly Documented):
1. Spatial entropy reduction over independently obtained SAR-valid pixels only (MK0_MNA == 1).
   Pixels outside valid radar coverage are masked out from the entropy calculation.
2. Ground-truth labels (MK0_MLU) and label-derived masks (label != 3) are STRICTLY FORBIDDEN
   and never accessed during adaptation.
3. Dual-input bi-temporal SAR observations (xA: pre-event, xB: post-event).

NOTE: This is NOT an exact reproduction of the original paper's CIFAR-10-C / ImageNet-C
experiments, but a faithful adaptation of the reference Tent update and prediction semantics
to bi-temporal SAR flood change detection. The legacy two-forward Tent implementation
is preserved in src/tta/tent.py.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_loss, compute_entropy_diagnostics


class ReferenceTentAdapter(BaseOnlineAdapter):
    """Reference-aligned Tent adapter adhering strictly to upstream DequanWang/tent semantics."""

    ADAPTER_IDENTIFIER = "reference_tent_v1"

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__(model, checkpoint_path, device)
        self.learning_rate = float(learning_rate)
        self.betas = betas
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.optimizer: Optional[optim.Adam] = None

        # Cache source BN running buffer shapes for clean restoration
        self._bn_buffer_shapes: Dict[str, Tuple[torch.Size, torch.Size]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.running_mean is not None and module.running_var is not None:
                    self._bn_buffer_shapes[name] = (
                        module.running_mean.shape,
                        module.running_var.shape,
                    )

        self._configure_reference_tent()
        self._init_optimizer()

    def _configure_reference_tent(self) -> None:
        """Configures model according to upstream tent.py configure_model():

        1. Sets model to train mode.
        2. Freezes all parameters.
        3. For all BN modules:
           - enables grad on weight and bias
           - sets track_running_stats = False
           - sets running_mean = None and running_var = None
        """
        self.model.train()
        self.model.requires_grad_(False)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def _init_optimizer(self) -> None:
        """Initializes fresh Adam optimizer over active BN affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def _reset_optimizer(self) -> None:
        """Resets optimizer state and step counter for episodic adaptation."""
        self.step_count = 0
        self._init_optimizer()

    def reset_to_source(self) -> None:
        """Restores model state dict bit-exactly to source checkpoint and resets optimizer.

        Before loading source state dict, restores track_running_stats and re-registers
        running_mean and running_var buffers so load_state_dict(strict=True) succeeds.
        Then re-applies reference Tent configuration and re-initializes a fresh optimizer.
        """
        # Re-register running buffers to allow strict state dict loading
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape))
                    module.register_buffer("running_var", torch.ones(var_shape))
                    module.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)

        # Re-apply reference Tent configuration (disable running stats tracking)
        self._configure_reference_tent()
        self._reset_optimizer()

    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Runs reference-aligned Tent: single forward pass, one backward step, no second forward.

        Adheres strictly to upstream tent.py:
        1. Forward pass computes outputs using batch normalization with batch statistics.
        2. Softmax entropy loss computed over valid pixels.
        3. Backward pass and optimizer step update BN affine parameters.
        4. Outputs from the single forward pass are returned as final predictions.
        """
        # Ensure model is in train mode with non-BN frozen and BN track_running_stats=False
        self.model.train()

        # 1. Single forward pass computes logits and predictions
        logits = self.model(xA, xB)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)

        # 2. Entropy loss computed over SAR-valid pixels only (MK0_MNA)
        loss = compute_entropy_loss(logits, sar_valid_mask)
        loss_val = float(loss.item())

        # 3. Adaptation step: update BN affine parameters
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.step_count += 1

        # 4. Diagnostics without second forward pass
        with torch.no_grad():
            diagnostics = compute_entropy_diagnostics(logits, sar_valid_mask)
            pred_flood = int(torch.sum(pred_class == 2).item())

            step_diagnostics = {
                "adaptation_loss": loss_val,
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": pred_flood,
                "bn_drift": self.compute_bn_drift(),
                "num_adaptation_steps": self.step_count,
            }

        return {
            "logits": logits.detach(),
            "probabilities": probs.detach(),
            "predicted_class": pred_class.detach(),
            "step_diagnostics": step_diagnostics,
        }

