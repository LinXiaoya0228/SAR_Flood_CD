"""SiamUnet_diff (FC-Siam-diff) model and wrapper for ChangeGuard E10.

Architectural reference:
Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection".
In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

Provides dense 3-class change detection logits for [No Water, Permanent Water, Flood]
given bi-temporal dual-polarization [VV, VH] SAR inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


class SiamUnet_diff(nn.Module):
    """Fully Convolutional Siamese Difference Network for Change Detection."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3, dropout_p: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder Stage 1
        self.conv11 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=dropout_p)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=dropout_p)

        # Encoder Stage 2
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=dropout_p)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=dropout_p)

        # Encoder Stage 3
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=dropout_p)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=dropout_p)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=dropout_p)

        # Encoder Stage 4
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=dropout_p)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=dropout_p)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=dropout_p)

        # Decoder Stage 4d
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=dropout_p)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=dropout_p)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=dropout_p)

        # Decoder Stage 3d
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=dropout_p)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=dropout_p)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=dropout_p)

        # Decoder Stage 2d
        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=dropout_p)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=dropout_p)

        # Decoder Stage 1d
        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=dropout_p)
        self.conv11d = nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass taking paired observations x1 (pre) and x2 (post).

        Args:
            x1: (B, in_channels, H, W) pre-event tensor
            x2: (B, in_channels, H, W) post-event tensor
        Returns:
            logits: (B, out_channels, H, W) raw class logits
        """
        # Encoder branch 1 (pre)
        x11_1 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11_1))))
        x1p_1 = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        x21_1 = self.do21(F.relu(self.bn21(self.conv21(x1p_1))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21_1))))
        x2p_1 = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        x31_1 = self.do31(F.relu(self.bn31(self.conv31(x2p_1))))
        x32_1 = self.do32(F.relu(self.bn32(self.conv32(x31_1))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32_1))))
        x3p_1 = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        x41_1 = self.do41(F.relu(self.bn41(self.conv41(x3p_1))))
        x42_1 = self.do42(F.relu(self.bn42(self.conv42(x41_1))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42_1))))
        x4p_1 = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        # Encoder branch 2 (post) - shared weights
        x11_2 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11_2))))
        x1p_2 = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        x21_2 = self.do21(F.relu(self.bn21(self.conv21(x1p_2))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21_2))))
        x2p_2 = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        x31_2 = self.do31(F.relu(self.bn31(self.conv31(x2p_2))))
        x32_2 = self.do32(F.relu(self.bn32(self.conv32(x31_2))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32_2))))
        x3p_2 = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        x41_2 = self.do41(F.relu(self.bn41(self.conv41(x3p_2))))
        x42_2 = self.do42(F.relu(self.bn42(self.conv42(x41_2))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42_2))))
        x4p_2 = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        # Difference Decoder Stage 4d
        x4d = self.upconv4(x4p_1)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), dim=1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Difference Decoder Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), dim=1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Difference Decoder Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), dim=1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Difference Decoder Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), dim=1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        logits = self.conv11d(x12d)

        return logits


class SiamDiffWrapper(nn.Module):
    """Wrapper around SiamUnet_diff adhering to the ChangeGuard evaluation interface."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3, dropout_p: float = 0.2):
        super().__init__()
        self.model = SiamUnet_diff(in_channels=in_channels, out_channels=out_channels, dropout_p=dropout_p)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """Forward pass taking pre-event (xA) and post-event (xB) dual-polarization observations.

        Args:
            xA: (B, 2, H, W) pre-event observation [VV, VH]
            xB: (B, 2, H, W) post-event observation [VV, VH]
        Returns:
            logits: (B, 3, H, W) unnormalized class logits for [No Water, Permanent Water, Flood]
        """
        # Ensure input channels match expected (if 3 channels are passed, extract first 2)
        if xA.size(1) > self.in_channels:
            xA = xA[:, :self.in_channels, :, :]
        if xB.size(1) > self.in_channels:
            xB = xB[:, :self.in_channels, :, :]
        return self.model(xA, xB)

    @torch.no_grad()
    def predict(self, xA: torch.Tensor, xB: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs inference and returns decoded predictions and probabilities."""
        logits = self.forward(xA, xB)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)
        flood_prob = probs[:, 2, :, :]
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_class,
            "flood_probability": flood_prob,
        }

    def freeze(self) -> None:
        """Freezes all model parameters and sets evaluation mode."""
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        in_channels: int = 2,
        out_channels: int = 3,
        device: str = "cpu",
    ) -> Tuple[SiamDiffWrapper, Dict[str, Any]]:
        """Strictly loads checkpoint weights into SiamDiffWrapper."""
        wrapper = cls(in_channels=in_channels, out_channels=out_channels)
        ckpt_path = Path(checkpoint_path).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
        else:
            state_dict = ckpt
            metadata = {}

        # Handle prefix stripping/adding
        cleaned_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if not k.startswith("model."):
                k = f"model.{k}"
            cleaned_sd[k] = v

        missing, unexpected = wrapper.load_state_dict(cleaned_sd, strict=True)
        wrapper.to(device)
        return wrapper, metadata

