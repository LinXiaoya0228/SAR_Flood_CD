"""Synchronized paired SAR data augmentation (Component M3)."""

import hashlib
import math
import random
from typing import Any, Dict, Tuple
import numpy as np
import torch


def compute_m3_sample_seed(run_seed: int, epoch: int, sample_idx_or_id: Any) -> int:
    """Computes deterministic M3 per-sample seed."""
    if isinstance(sample_idx_or_id, str):
        sid_int = int(hashlib.sha256(sample_idx_or_id.encode("utf-8")).hexdigest()[:8], 16)
    else:
        sid_int = int(sample_idx_or_id)
    return (int(run_seed) * 1000003 + int(epoch) * 997 + sid_int * 31 + 509) & 0xFFFFFFFF


class SynchronizedPairedSARAugmentation:
    """Synchronized geometric and radiometric augmentation for bi-temporal SAR change detection.

    Applied in the raw linear amplitude backscatter domain BEFORE clipping [0.0, 0.15]
    and standardization.

    Guarantees:
    1. Pre-event and post-event images receive identical geometric and radiometric transforms.
    2. Labels and valid masks receive identical geometric transform.
    3. VV and VH channels receive identical radiometric gain (common-mode).
    4. Label values (0, 1, 2) and ignore indices (3) remain uncorrupted.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotations_deg: Tuple[int, ...] = (0, 90, 180, 270),
        min_gain: float = 0.95,
        max_gain: float = 1.05,
    ):
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotations = tuple(rotations_deg)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

    def sample_parameters(self, sample_seed: int) -> Dict[str, Any]:
        """Deterministically samples transform parameters from seed."""
        rng = random.Random(sample_seed)
        hflip = rng.random() < self.hflip_prob
        vflip = rng.random() < self.vflip_prob
        rot_deg = rng.choice(self.rotations)
        rot_k = (rot_deg // 90) % 4

        # Common-mode amplitude gain: gain ~ Uniform(0.95, 1.05)
        gain = rng.uniform(self.min_gain, self.max_gain)

        return {
            "hflip": bool(hflip),
            "vflip": bool(vflip),
            "rot_deg": int(rot_deg),
            "rot_k": int(rot_k),
            "gain": float(gain),
            "sample_seed": int(sample_seed),
        }

    def apply_geometry(self, arr: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Applies spatial flips and rotations to 2D (H, W) or 3D (C, H, W) numpy array."""
        # Horizontal flip along width (last dimension)
        if params["hflip"]:
            arr = np.flip(arr, axis=-1)
        # Vertical flip along height (second to last dimension)
        if params["vflip"]:
            arr = np.flip(arr, axis=-2)
        # Rotation by k * 90 degrees in (H, W) plane (axes -2, -1)
        k = params["rot_k"]
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(-2, -1))
        return np.ascontiguousarray(arr)

    def __call__(
        self,
        raw_pre: np.ndarray,
        raw_post: np.ndarray,
        raw_label: np.ndarray,
        raw_valid_mask: np.ndarray,
        sample_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Augments a paired sample.

        Args:
            raw_pre: Linear amplitude pre-event (2, H, W)
            raw_post: Linear amplitude post-event (2, H, W)
            raw_label: Ground truth label (H, W)
            raw_valid_mask: Valid pixel mask (H, W)
            sample_seed: Seed for deterministic sampling

        Returns:
            (aug_pre, aug_post, aug_label, aug_valid_mask, transform_params)
        """
        params = self.sample_parameters(sample_seed)

        # 1. Synchronized Geometric Augmentation
        aug_pre = self.apply_geometry(raw_pre.copy(), params)
        aug_post = self.apply_geometry(raw_post.copy(), params)
        aug_label = self.apply_geometry(raw_label.copy(), params)
        aug_mask = self.apply_geometry(raw_valid_mask.copy(), params)

        # 2. Synchronized Radiometric Gain (applied in linear amplitude)
        gain = params["gain"]
        aug_pre = aug_pre * gain
        aug_post = aug_post * gain

        return aug_pre, aug_post, aug_label, aug_mask, params

    def derive_sample_seed(self, base_seed: int, epoch: int, sample_id: Any) -> int:
        """Derives sample seed from base seed, epoch, and sample identifier."""
        return compute_m3_sample_seed(base_seed, epoch, sample_id)

    def augment(
        self,
        raw_pre: np.ndarray,
        raw_post: np.ndarray,
        raw_label: np.ndarray,
        raw_valid_mask: np.ndarray,
        sample_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Alias for __call__."""
        return self(raw_pre, raw_post, raw_label, raw_valid_mask, sample_seed)
