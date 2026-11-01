"""Synthetic domain-shift transforms for ChangeGuard E10 robustness evaluations.

Implements physically interpretable radiometric backscatter calibration shifts:
    x' = clip((1 + s) * x, 0.0, clamp_max)
applied on clean linear SAR amplitude prior to standard z-score normalization.

Severities:
- s = 0.00: Clean baseline (exact match to unshifted data)
- s = 0.05: Mild (+5% backscatter gain / calibration drift)
- s = 0.10: Moderate (+10% backscatter gain / calibration drift)
- s = 0.20: Severe (+20% backscatter gain / calibration drift)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


SUPPORTED_SEVERITIES: Dict[str, float] = {
    "clean": 0.0,
    "severity_1": 0.05,
    "severity_2": 0.10,
    "severity_3": 0.20,
}


class MultiplicativeBackscatterShift:
    """Applies a multiplicative backscatter calibration shift to raw linear SAR amplitude."""

    def __init__(
        self,
        severity_factor: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 0.15,
        sar_mean: Tuple[float, float] = (0.0953, 0.0264),
        sar_std: Tuple[float, float] = (0.0427, 0.0215),
    ):
        self.s = float(severity_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]

    def transform_raw_sar(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
    ) -> np.ndarray:
        """Transforms raw unnormalized SAR channels [VV, VH] with calibration shift.
        
        Args:
            vv: (H, W) raw linear amplitude VV
            vh: (H, W) raw linear amplitude VH
            
        Returns:
            (2, H, W) normalized SAR array
        """
        sar = np.stack([vv, vh], axis=0)  # (2, H, W)
        # Apply multiplicative calibration factor before normalization
        if abs(self.s) > 1e-7:
            sar = sar * (1.0 + self.s)
        sar = np.clip(sar, self.clamp_min, self.clamp_max)
        sar = np.nan_to_num(sar, nan=self.clamp_max, posinf=self.clamp_max, neginf=self.clamp_min)
        sar_norm = (sar - self.sar_mean) / self.sar_std
        return sar_norm.astype(np.float32)

    def transform_normalized_tensor(
        self,
        x_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Applies shift to already normalized tensor by inverting, shifting, and re-normalizing.
        
        Args:
            x_norm: (B, 2, H, W) or (2, H, W) standardized tensor
            
        Returns:
            Shifted standardized tensor of same shape
        """
        if abs(self.s) < 1e-7:
            return x_norm

        device = x_norm.device
        dtype = x_norm.dtype
        mean = torch.tensor(self.sar_mean, device=device, dtype=dtype)
        std = torch.tensor(self.sar_std, device=device, dtype=dtype)

        # Invert standardization to recover clamped linear amplitude
        x_linear = x_norm * std + mean
        # Apply multiplicative shift
        x_shifted = x_linear * (1.0 + self.s)
        x_clamped = torch.clamp(x_shifted, self.clamp_min, self.clamp_max)
        # Re-standardize
        x_out = (x_clamped - mean) / std
        return x_out

