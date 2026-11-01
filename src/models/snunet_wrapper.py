"""Thin, safe wrapper around official SNUNet_ECAM change detection model."""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import official architecture definition read-only from third_party/KuroSiwo
import os

candidate_kuro_paths = [
    os.environ.get("KURO_ROOT"),
    str(Path(__file__).resolve().parents[2] / "third_party/KuroSiwo"),
    str(Path(__file__).resolve().parents[3] / "third_party/KuroSiwo"),
    str(Path(__file__).resolve().parents[4] / "third_party/KuroSiwo") if len(Path(__file__).resolve().parents) > 4 else None,
]

kuro_found = False
for cp in candidate_kuro_paths:
    if cp and os.path.isdir(cp):
        if cp not in sys.path:
            sys.path.insert(0, cp)
        try:
            from models.snunet import SNUNet_ECAM
            kuro_found = True
            break
        except ImportError:
            pass

if not kuro_found:
    try:
        from models.snunet import SNUNet_ECAM
    except ImportError as e:
        raise ImportError(
            f"Could not import SNUNet_ECAM from third_party/KuroSiwo. Checked candidate paths: {candidate_kuro_paths}. "
            "Please clone KuroSiwo (git clone https://github.com/Orion-AI-Lab/KuroSiwo.git third_party/KuroSiwo) "
            "or set KURO_ROOT environment variable."
        ) from e


class SNUNetWrapper(nn.Module):
    """Wrapper around SNUNet_ECAM for bi-temporal SAR flood change detection."""

    def __init__(self, in_channels: int = 3, out_ch: int = 3, base_channel: int = 32):
        super().__init__()
        self.model = SNUNet_ECAM(in_channels=in_channels, out_ch=out_ch, base_channel=base_channel)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """Forward pass taking two temporal branches.

        Args:
            xA: (B, 3, H, W) pre-event observation [VV, VH, DEM]
            xB: (B, 3, H, W) post-event observation [VV, VH, DEM]
        Returns:
            logits: (B, 3, H, W) raw class logits for [No Water, Permanent Water, Flood]
        """
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

    def freeze(self):
        """Freezes all model parameters and sets evaluation mode."""
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        in_channels: int = 3,
        out_ch: int = 3,
        base_channel: int = 32,
        device: str = "cpu",
    ) -> Tuple["SNUNetWrapper", Dict[str, Any]]:
        """Strictly loads the official checkpoint into SNUNetWrapper."""
        wrapper = cls(in_channels=in_channels, out_ch=out_ch, base_channel=base_channel)

        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        else:
            state_dict = ckpt
            metadata = {}

        # Handle whether state_dict was saved from SNUNetWrapper (prefixed with 'model.')
        # or directly from SNUNet_ECAM (unprefixed)
        if any(k.startswith("model.") for k in state_dict.keys()):
            wrapper.load_state_dict(state_dict, strict=True)
        else:
            wrapper.model.load_state_dict(state_dict, strict=True)
        wrapper.to(device)
        wrapper.freeze()
        return wrapper, metadata
