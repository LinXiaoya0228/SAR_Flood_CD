"""
Tent (Test-Time Entropy Minimization) adapter for ChangeGuard.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization",
ICLR 2021 (https://github.com/DequanWang/tent).

Faithful source-model adaptation:
- Updates ONLY BatchNorm affine parameters (weight, bias).
- Convolution, classifier, attention, and all non-BN parameters remain strictly frozen.
- Unsupervised Shannon entropy loss computed over SAR-valid pixels (MK0_MNA).
- Ground-truth labels are strictly forbidden from entering the adaptation loss.
- Resets to clean source checkpoint and fresh optimizer before every event.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.tta.entropy import compute_entropy_loss, compute_entropy_diagnostics


class TentAdapter(BaseOnlineAdapter):
    """Test-Time Entropy Minimization (Tent) baseline.

    Optimizes BatchNorm affine parameters online via prediction entropy minimization.
    Following Wang et al. (2021):
    - Uses Adam optimizer on BN weights and biases.
    - Sets non-BN layers to eval mode; sets BN layers to train mode during adaptation.
    - Running statistics update via moving average during streaming; both affine
      parameters and running statistics are reset at the beginning of each event.
    """

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

        self._configure_tent_parameters()
        self._init_optimizer()

    def _configure_tent_parameters(self) -> List[nn.Parameter]:
        """Freezes all non-BN parameters; enables gradients only for BN affine parameters."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        params_to_opt: List[nn.Parameter] = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                if m.weight is not None:
                    m.weight.requires_grad = True
                    params_to_opt.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params_to_opt.append(m.bias)

        return params_to_opt

    def _init_optimizer(self) -> None:
        """Initializes fresh Adam optimizer over the active BN affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def _reset_optimizer(self) -> None:
        """Resets parameter gradients, optimizer state, and step counter."""
        self.step_count = 0
        self._configure_tent_parameters()
        self._init_optimizer()

    def _adapt_and_predict_impl(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Performs transductive online adaptation: adapts on current batch, then predicts."""
        # 1. Ensure correct mode: non-BN eval, BN train
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()

        # 2. Forward pass for adaptation loss
        logits_pre = self.model(xA, xB)

        with torch.no_grad():
            pre_pred_class = torch.argmax(logits_pre, dim=1)
            pre_pred_flood = int(torch.sum(pre_pred_class == 2).item())

        loss = compute_entropy_loss(logits_pre, sar_valid_mask)
        loss_val = float(loss.item())

        # 3. Adaptation step: update BN affine parameters
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.step_count += 1

        # 4. Transductive post-update prediction on the same batch
        with torch.no_grad():
            logits_post = self.model(xA, xB)
            probs_post = F.softmax(logits_post, dim=1)
            pred_class_post = torch.argmax(logits_post, dim=1)

            post_flood = int(torch.sum(pred_class_post == 2).item())
            diagnostics = compute_entropy_diagnostics(logits_post, sar_valid_mask)
            disagreement = int(torch.sum(pre_pred_class != pred_class_post).item())

            step_diagnostics = {
                "adaptation_loss": loss_val,
                "mean_entropy": diagnostics["mean_entropy"],
                "mean_max_confidence": diagnostics["mean_max_confidence"],
                "pred_flood_pixels": post_flood,
                "pre_update_pred_flood_pixels": pre_pred_flood,
                "disagreement_with_source": disagreement,
                "bn_drift": self.compute_bn_drift(),
                "num_adaptation_steps": self.step_count,
            }

        return {
            "logits": logits_post,
            "probabilities": probs_post,
            "predicted_class": pred_class_post,
            "step_diagnostics": step_diagnostics,
        }
