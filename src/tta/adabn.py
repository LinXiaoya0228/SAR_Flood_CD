"""
AdaBN (Adaptive Batch Normalization) adapter for ChangeGuard.

Reference: Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation",
Pattern Recognition 2018; Schneider et al., "Improving robustness against common corruptions
by covariate shift adaptation", NeurIPS 2020.

Adapts BatchNorm running statistics using unlabeled target stream inputs without gradient updates.
Affine parameters (gamma, beta) and non-BN weights remain strictly frozen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_diagnostics


class AdaBNAdapter(BaseOnlineAdapter):
    """Adaptive Batch Normalization (AdaBN) baseline.

    In online episodic adaptation:
    1. At the start of each event, running statistics (mean and variance) and affine
       parameters are reset to the exact source checkpoint state dict.
    2. During event streaming, BN layers operate in training mode (m.train()) under
       torch.no_grad(), normalizing each batch using its batch statistics and updating
       the running statistics via moving average.
    3. All convolution, classifier, and BN-affine parameters remain strictly frozen.
    """

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        super().__init__(model, checkpoint_path, device)
        self.step_count = 0
        self._setup_adabn_mode()

    def _setup_adabn_mode(self) -> None:
        """Configures model for AdaBN: freezes all parameters, sets BN to train mode."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Put only BatchNorm modules into train mode to update statistics
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()

    def _reset_optimizer(self) -> None:
        """Resets step count and restores AdaBN mode after checkpoint reload."""
        self.step_count = 0
        self._setup_adabn_mode()

    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Runs AdaBN inference: normalizes with target batch stats, updates running stats."""
        # Ensure non-BN layers remain in eval mode while BN layers remain in train mode
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()

        with torch.no_grad():
            logits = self.model(xA, xB)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1)

            diagnostics = compute_entropy_diagnostics(logits, sar_valid_mask)
            pred_flood = int(torch.sum(pred_class == 2).item())
            self.step_count += 1

            step_diagnostics = {
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": pred_flood,
                "bn_drift": self.compute_bn_drift(),  # strictly 0.0 for affine params
                "num_adaptation_steps": self.step_count,
            }

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_class,
            "step_diagnostics": step_diagnostics,
        }
