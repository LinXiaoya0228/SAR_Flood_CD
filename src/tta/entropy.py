"""
Entropy computation and diagnostic utilities for unsupervised test-time adaptation.

Strictly operates on model logits and optional SAR-valid masks (MK0_MNA).
Under no circumstance does this module accept or process ground-truth labels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_entropy_loss(
    logits: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes mean Shannon entropy loss over SAR-valid pixels.

    Args:
        logits: (B, C, H, W) raw class logits from the model (C=3).
        sar_valid_mask: Optional (B, H, W) or (B, 1, H, W) binary mask from MK0_MNA
            indicating physical SAR observation validity (1=valid, 0=invalid/margin).
            MUST NOT be derived from ground-truth labels.
        eps: Small epsilon to prevent log(0).

    Returns:
        Scalar entropy loss tensor with gradients attached.
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    log_probs = torch.log(probs + eps)
    pixel_entropy = -torch.sum(probs * log_probs, dim=1)  # (B, H, W)

    if sar_valid_mask is not None:
        mask = sar_valid_mask.squeeze(1).float() if sar_valid_mask.dim() == 4 else sar_valid_mask.float()
        valid_pixel_count = torch.sum(mask)
        if valid_pixel_count > 0:
            loss = torch.sum(pixel_entropy * mask) / (valid_pixel_count + eps)
        else:
            loss = torch.mean(pixel_entropy)
    else:
        loss = torch.mean(pixel_entropy)

    return loss


@torch.no_grad()
def compute_entropy_diagnostics(
    logits: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes diagnostic statistics (entropy, max confidence) without tracking gradients."""
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + eps)
    pixel_entropy = -torch.sum(probs * log_probs, dim=1)
    max_conf, _ = torch.max(probs, dim=1)

    if sar_valid_mask is not None:
        mask = sar_valid_mask.squeeze(1).float() if sar_valid_mask.dim() == 4 else sar_valid_mask.float()
        valid_count = torch.sum(mask).item()
        if valid_count > 0:
            mean_entropy = (torch.sum(pixel_entropy * mask) / (valid_count + eps)).item()
            mean_max_conf = (torch.sum(max_conf * mask) / (valid_count + eps)).item()
        else:
            mean_entropy = torch.mean(pixel_entropy).item()
            mean_max_conf = torch.mean(max_conf).item()
    else:
        mean_entropy = torch.mean(pixel_entropy).item()
        mean_max_conf = torch.mean(max_conf).item()

    return {
        "mean_entropy": mean_entropy,
        "mean_max_confidence": mean_max_conf,
    }
