"""Change-specific null evidence and change-disappearance probing.

All probe forward passes run strictly in eval mode under torch.no_grad()
without mutating parameters, BN running statistics, or optimizer states.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.risk.indicators import compute_augmentation_inconsistency


def compute_spatial_concentration(
    pred_class2_mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> float:
    """Measures spatial clustering of class-2 predictions using 3x3 local density.

    Args:
        pred_class2_mask: (B, H, W) binary tensor of class-2 predictions.
    Returns:
        Average density in [0.0, 1.0] where contiguous clusters -> 1.0, isolated noise -> ~0.11.
    """
    total_positives = float(torch.sum(pred_class2_mask).item())
    if total_positives == 0:
        return 0.0

    mask_float = pred_class2_mask.unsqueeze(1).float()  # (B, 1, H, W)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=pred_class2_mask.device) / 9.0
    density = F.conv2d(mask_float, kernel, padding=1).squeeze(1)  # (B, H, W)
    mean_density = torch.sum(density * mask_float.squeeze(1)) / (total_positives + epsilon)
    return float(mean_density.item())


@torch.no_grad()
def evaluate_probe_pair(
    candidate_model: nn.Module,
    source_model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
    prefix: str = "null_pre2_pre2",
) -> Dict[str, float]:
    """Evaluates a pair (x1, x2) under both candidate and source models in eval mode.

    Returns label-free metrics prefixed with `prefix`.
    """
    candidate_model.eval()
    source_model.eval()

    if sar_valid_mask is not None:
        valid_bool = (sar_valid_mask > 0).to(x1.device)
    else:
        valid_bool = torch.ones((x1.size(0), x1.size(2), x1.size(3)), dtype=torch.bool, device=x1.device)

    total_valid_pixels = float(torch.sum(valid_bool).item())
    if total_valid_pixels == 0:
        total_valid_pixels = 1.0

    # 1. Forward passes
    cand_logits = candidate_model(x1, x2)
    src_logits = source_model(x1, x2)

    cand_probs = F.softmax(cand_logits, dim=1)
    src_probs = F.softmax(src_logits, dim=1)

    cand_preds = torch.argmax(cand_logits, dim=1)
    src_preds = torch.argmax(src_logits, dim=1)

    cand_class2_prob = cand_probs[:, 2, :, :]  # (B, H, W)
    src_class2_prob = src_probs[:, 2, :, :]

    cand_class2_mask = (cand_preds == 2) & valid_bool
    src_class2_mask = (src_preds == 2) & valid_bool

    cand_class2_area = float(torch.sum(cand_class2_mask).item())
    src_class2_area = float(torch.sum(src_class2_mask).item())

    mean_cand_prob = float((torch.sum(cand_class2_prob * valid_bool) / total_valid_pixels).item())
    mean_src_prob = float((torch.sum(src_class2_prob * valid_bool) / total_valid_pixels).item())

    prob_increase = mean_cand_prob - mean_src_prob
    area_increase = cand_class2_area - src_class2_area
    activation_rate = cand_class2_area / total_valid_pixels
    area_ratio = (cand_class2_area + 1.0) / (src_class2_area + 1.0)

    # Augmentation inconsistency on the probe
    aug_inconsistency = compute_augmentation_inconsistency(
        candidate_model, x1, x2, cand_preds, sar_valid_mask=sar_valid_mask
    )

    # Spatial concentration of candidate class-2 activation
    spatial_conc = compute_spatial_concentration(cand_class2_mask)

    return {
        f"{prefix}_mean_class2_prob": mean_cand_prob,
        f"{prefix}_class2_area": cand_class2_area,
        f"{prefix}_source_class2_area": src_class2_area,
        f"{prefix}_prob_increase": prob_increase,
        f"{prefix}_area_increase": area_increase,
        f"{prefix}_activation_rate": activation_rate,
        f"{prefix}_area_ratio": area_ratio,
        f"{prefix}_aug_inconsistency": aug_inconsistency,
        f"{prefix}_spatial_concentration": spatial_conc,
    }


def run_null_probes(
    candidate_model: nn.Module,
    source_model: nn.Module,
    pre1: torch.Tensor,
    pre2: torch.Tensor,
    post: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Runs primary counterfactual, secondary identical, and natural stability probes.

    1. Primary counterfactual null probe: (pre2, pre2)
    2. Secondary identical-input probes: (pre1, pre1), (post, post)
    3. Auxiliary natural stability pair: (pre1, pre2)
    """
    results: Dict[str, float] = {}

    # Primary counterfactual: (pre2, pre2)
    res_pre2 = evaluate_probe_pair(
        candidate_model, source_model, pre2, pre2,
        sar_valid_mask=sar_valid_mask, prefix="null_pre2_pre2"
    )
    results.update(res_pre2)

    # Secondary identical-input: (pre1, pre1)
    res_pre1 = evaluate_probe_pair(
        candidate_model, source_model, pre1, pre1,
        sar_valid_mask=sar_valid_mask, prefix="null_pre1_pre1"
    )
    results.update(res_pre1)

    # Secondary identical-input: (post, post)
    res_post = evaluate_probe_pair(
        candidate_model, source_model, post, post,
        sar_valid_mask=sar_valid_mask, prefix="null_post_post"
    )
    results.update(res_post)

    # Auxiliary natural stability: (pre1, pre2)
    res_nat = evaluate_probe_pair(
        candidate_model, source_model, pre1, pre2,
        sar_valid_mask=sar_valid_mask, prefix="stability_pre1_pre2"
    )
    results.update(res_nat)

    return results


def run_disappearance_probes(
    candidate_logits: torch.Tensor,
    source_logits: torch.Tensor,
    candidate_features: Optional[torch.Tensor],
    source_features: Optional[torch.Tensor],
    candidate_model: nn.Module,
    xA: torch.Tensor,
    xB: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Computes label-free change-disappearance evidence on the real (pre2, post) pair."""
    B, C, H, W = candidate_logits.shape
    if sar_valid_mask is not None:
        valid_bool = (sar_valid_mask > 0).to(candidate_logits.device)
    else:
        valid_bool = torch.ones((B, H, W), dtype=torch.bool, device=candidate_logits.device)

    total_valid = float(torch.sum(valid_bool).item())
    if total_valid == 0:
        total_valid = 1.0

    cand_probs = F.softmax(candidate_logits, dim=1)
    src_probs = F.softmax(source_logits, dim=1)
    cand_preds = torch.argmax(candidate_logits, dim=1)
    src_preds = torch.argmax(source_logits, dim=1)

    cand_flood_mask = (cand_preds == 2) & valid_bool
    src_flood_mask = (src_preds == 2) & valid_bool

    cand_flood_area = float(torch.sum(cand_flood_mask).item())
    src_flood_area = float(torch.sum(src_flood_mask).item())

    cand_mean_p2 = float((torch.sum(cand_probs[:, 2, :, :] * valid_bool) / total_valid).item())
    src_mean_p2 = float((torch.sum(src_probs[:, 2, :, :] * valid_bool) / total_valid).item())

    # Adapted / Source predicted flood area and probability ratios
    area_ratio = (cand_flood_area + 1.0) / (src_flood_area + 1.0)
    prob_ratio = (cand_mean_p2 + 1e-4) / (src_mean_p2 + 1e-4)

    # Feature norm ratio
    if candidate_features is not None and source_features is not None:
        cand_fnorm = float(torch.norm(candidate_features).item())
        src_fnorm = float(torch.norm(source_features).item())
        fnorm_ratio = (cand_fnorm + 1e-4) / (src_fnorm + 1e-4)
    else:
        fnorm_ratio = 1.0

    # Fraction of source-predicted flood pixels erased by adaptation
    # i.e. pixels where source predicted 2, but adapted model predicts != 2
    erased_mask = src_flood_mask & (~cand_flood_mask)
    erased_pixels = float(torch.sum(erased_mask).item())
    fraction_source_erased = erased_pixels / (src_flood_area + 1.0)

    # Disagreement
    disagree_mask = (cand_preds != src_preds) & valid_bool
    disagree_fraction = float(torch.sum(disagree_mask).item()) / total_valid

    # Augmentation inconsistency on real pair
    aug_inconsistency = compute_augmentation_inconsistency(
        candidate_model, xA, xB, cand_preds, sar_valid_mask=sar_valid_mask
    )

    return {
        "disappear_area_ratio": area_ratio,
        "disappear_prob_ratio": prob_ratio,
        "disappear_feature_norm_ratio": fnorm_ratio,
        "disappear_erased_pixels": erased_pixels,
        "disappear_fraction_source_erased": fraction_source_erased,
        "disappear_disagreement_fraction": disagree_fraction,
        "disappear_aug_inconsistency": aug_inconsistency,
    }

