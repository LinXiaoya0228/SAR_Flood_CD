"""Loss functions for controlled source model training.

Implements the frozen unweighted Cross-Entropy + Soft Dice loss
specified in docs/source_training_protocol_v1.md (Section 6.2).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Absent-class stable Soft Dice Loss for multiclass segmentation.
    
    Formula per class c:
        L_Dice^(c) = 1 - (2 * sum_{i in M}(p_{c, i} * t_{c, i}) + eps) /
                         (sum_{i in M}(p_{c, i}^2) + sum_{i in M}(t_{c, i}^2) + eps)
    
    where M is the mask of valid pixels (target != ignore_index).
    eps = 1.0 guarantees:
    - No division by zero.
    - If class c is absent (sum t = 0) and the model predicts p ~ 0, loss = 0.0.
    - If class c is absent and model has false alarms (p > 0), loss increases smoothly.
    - Stable, bounded gradients across all batch distributions.
    """

    def __init__(
        self,
        classes: List[int] = (0, 1, 2),
        ignore_index: int = 3,
        eps: float = 1.0,
        squared_denominator: bool = True,
        reduction: str = "batch",
    ):
        super().__init__()
        self.classes = list(classes)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.squared_denominator = bool(squared_denominator)
        if reduction not in ("batch", "mean"):
            raise ValueError(f"reduction must be 'batch' or 'mean', got '{reduction}'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes multiclass soft dice loss.
        
        Args:
            logits: (B, C, H, W) raw class logits
            target: (B, H, W) integer ground truth labels
            
        Returns:
            Scalar tensor representing average Dice loss across specified classes.
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        valid_mask = (target != self.ignore_index)  # (B, H, W) bool
        if not valid_mask.any():
            # Zero valid pixels in batch, return zero with gradient tracking
            return logits.sum() * 0.0

        valid_mask_f = valid_mask.float()  # (B, H, W)

        dice_per_class = []
        for c in self.classes:
            p_c = probs[:, c, :, :] * valid_mask_f  # (B, H, W)
            t_c = (target == c).float() * valid_mask_f  # (B, H, W)

            if self.reduction == "batch":
                intersection = torch.sum(p_c * t_c)
                if self.squared_denominator:
                    cardinality = torch.sum(p_c ** 2) + torch.sum(t_c ** 2)
                else:
                    cardinality = torch.sum(p_c) + torch.sum(t_c)
                dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
                loss_c = 1.0 - dice_c
            else:  # reduction == "mean"
                intersection = torch.sum(p_c * t_c, dim=(1, 2))  # (B,)
                if self.squared_denominator:
                    cardinality = torch.sum(p_c ** 2, dim=(1, 2)) + torch.sum(t_c ** 2, dim=(1, 2))
                else:
                    cardinality = torch.sum(p_c, dim=(1, 2)) + torch.sum(t_c, dim=(1, 2))
                dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
                loss_c = torch.mean(1.0 - dice_c)

            dice_per_class.append(loss_c)

        return torch.stack(dice_per_class).mean()


class ControlledSourceLoss(nn.Module):
    """Frozen unweighted Cross-Entropy + Soft Dice Loss for controlled source training.
    
    L_total = 0.5 * L_CE + 0.5 * L_Dice
    """

    def __init__(
        self,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        classes_in_dice: List[int] = (0, 1, 2),
        ignore_index: int = 3,
        smooth_epsilon: float = 1.0,
        squared_denominator: bool = True,
        reduction: str = "batch",
    ):
        super().__init__()
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.ignore_index = int(ignore_index)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.dice = SoftDiceLoss(
            classes=classes_in_dice,
            ignore_index=self.ignore_index,
            eps=smooth_epsilon,
            squared_denominator=squared_denominator,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute combined loss.
        
        Args:
            logits: (B, C, H, W) model predictions
            target: (B, H, W) class indices
            return_components: if True, also return dictionary of individual loss components.
        """
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            loss_total = logits.sum() * 0.0
            if return_components:
                return loss_total, {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0}
            return loss_total

        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        loss_total = self.weight_ce * loss_ce + self.weight_dice * loss_dice

        if return_components:
            components = {
                "loss_total": float(loss_total.detach().cpu().item()),
                "loss_ce": float(loss_ce.detach().cpu().item()),
                "loss_dice": float(loss_dice.detach().cpu().item()),
            }
            return loss_total, components

        return loss_total
