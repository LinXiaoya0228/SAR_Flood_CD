"""
Source-only baseline adapter for ChangeGuard.

Operates in frozen evaluation mode without parameter updates or running-stat updates.
Serves as the frozen reference point to measure adaptation benefit or harm.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import torch
import torch.nn.functional as F

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_diagnostics


class SourceOnlyAdapter(BaseOnlineAdapter):
    """Frozen source model adapter with zero test-time updates."""

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        super().__init__(model, checkpoint_path, device)
        self.model.freeze()

    def _reset_optimizer(self) -> None:
        """No optimizer to reset for source-only."""
        self.model.freeze()

    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Runs frozen inference without any adaptation."""
        self.model.eval()
        with torch.no_grad():
            out = self.model.predict(xA, xB)
            logits = out["logits"]
            probs = out["probabilities"]
            pred_class = out["predicted_class"]

            diagnostics = compute_entropy_diagnostics(logits, sar_valid_mask)
            pred_flood = int(torch.sum(pred_class == 2).item())

            step_diagnostics = {
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": pred_flood,
                "bn_drift": self.compute_bn_drift(),
                "num_adaptation_steps": 0,
            }

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_class,
            "step_diagnostics": step_diagnostics,
        }
