"""
Base class for test-time adaptation (TTA) algorithms in ChangeGuard.

Enforces strict label isolation by construction:
The adapt_and_predict method accepts only sensory inputs (xA, xB) and physical
observation validity masks. Ground-truth labels are strictly forbidden.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from src.models.snunet_wrapper import SNUNetWrapper


class BaseOnlineAdapter(ABC):
    """Abstract base class for episodic online test-time adaptation methods."""

    def __init__(
        self,
        model: SNUNetWrapper,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model.to(self.device)

        # Cache source state dict on CPU to enable exact zero-overhead resets
        self._source_state_dict = copy.deepcopy(self.model.state_dict())

        # Collect and cache initial BatchNorm affine parameter tensors
        self._source_bn_weights: Dict[str, torch.Tensor] = {}
        self._source_bn_biases: Dict[str, torch.Tensor] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    self._source_bn_weights[name] = module.weight.detach().clone().cpu()
                if module.bias is not None:
                    self._source_bn_biases[name] = module.bias.detach().clone().cpu()

    def reset_to_source(self) -> None:
        """Resets model parameters and running statistics exactly to the source checkpoint."""
        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)
        self._reset_optimizer()

    @abstractmethod
    def _reset_optimizer(self) -> None:
        """Resets any adaptation optimizer and step counters."""
        pass

    def compute_bn_drift(self) -> Dict[str, float]:
        """Computes L2 norm drift of current BN affine parameters relative to source.

        Returns:
            Dictionary with 'weight_drift_l2', 'bias_drift_l2', 'total_bn_drift_l2'.
        """
        w_drift_sq = 0.0
        b_drift_sq = 0.0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None and name in self._source_bn_weights:
                    diff_w = module.weight.detach().cpu() - self._source_bn_weights[name]
                    w_drift_sq += float(torch.sum(diff_w ** 2).item())
                if module.bias is not None and name in self._source_bn_biases:
                    diff_b = module.bias.detach().cpu() - self._source_bn_biases[name]
                    b_drift_sq += float(torch.sum(diff_b ** 2).item())

        total_drift_sq = w_drift_sq + b_drift_sq
        return {
            "bn_weight_drift_l2": float(w_drift_sq ** 0.5),
            "bn_bias_drift_l2": float(b_drift_sq ** 0.5),
            "total_bn_drift_l2": float(total_drift_sq ** 0.5),
        }

    def adapt_and_predict(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Public interface for transductive online adaptation and prediction.

        Accepts ONLY unlabeled inputs. If any label or ground-truth key is passed,
        raises a fatal Research Integrity violation.
        """
        # Guard against label leakage
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "batch"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(
                f"RESEARCH INTEGRITY VIOLATION: adapt_and_predict received forbidden keys {leaked}. "
                f"Test-time adaptation must NEVER have access to ground-truth labels!"
            )

        return self._adapt_and_predict_impl(
            xA=xA,
            xB=xB,
            sar_valid_mask=sar_valid_mask,
            sample_ids=sample_ids,
            event_ids=event_ids,
        )

    @abstractmethod
    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Subclass implementation of inference and adaptation."""
        pass
