"""Generic label-free indicators for test-time adaptation risk monitoring.

Every indicator is computed strictly without access to ground truth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.snunet_wrapper import SNUNetWrapper


class IntermediateFeatureHook:
    """Captures intermediate change representations at a single predeclared layer.
    
    Predeclared Layer: model.conv_final input (128 channels), representing the
    deepest and richest bi-temporal change representation before linear projection to logits.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.features: Optional[torch.Tensor] = None
        self._hook_handle = None
        
        # Locate the conv_final module inside SNUNet_ECAM
        target_module = getattr(model, "conv_final", None)
        if target_module is None and hasattr(model, "model"):
            target_module = getattr(model.model, "conv_final", None)
            
        if target_module is None:
            raise AttributeError("Could not find target layer 'conv_final' in model.")
            
        self._hook_handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: torch.Tensor) -> None:
        # inp[0] is the 128-channel dense change embedding
        self.features = inp[0].detach()

    def remove(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


def compute_generic_indicators(
    candidate_logits: torch.Tensor,
    source_logits: torch.Tensor,
    candidate_features: Optional[torch.Tensor],
    source_features: Optional[torch.Tensor],
    candidate_model: nn.Module,
    source_bn_state: Dict[str, torch.Tensor],
    sar_valid_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """Computes all predeclared generic label-free indicators for a batch.

    Args:
        candidate_logits: (B, 3, H, W) logits from adapted model.
        source_logits: (B, 3, H, W) logits from frozen source model.
        candidate_features: (B, C, H, W) intermediate feature from adapted model.
        source_features: (B, C, H, W) intermediate feature from frozen source model.
        candidate_model: Active adapted model instance.
        source_bn_state: Cached source BN parameters and running buffers.
        sar_valid_mask: (B, H, W) boolean/uint8 physical SAR validity mask.
        epsilon: Numerical stability constant.

    Returns:
        Dictionary of label-free generic risk metrics.
    """
    B, C, H, W = candidate_logits.shape
    if sar_valid_mask is not None:
        valid_bool = (sar_valid_mask > 0).to(candidate_logits.device)
    else:
        valid_bool = torch.ones((B, H, W), dtype=torch.bool, device=candidate_logits.device)

    total_valid_pixels = float(torch.sum(valid_bool).item())
    if total_valid_pixels == 0:
        total_valid_pixels = 1.0

    # 1. Softmax probabilities and predictions
    cand_probs = F.softmax(candidate_logits, dim=1)
    src_probs = F.softmax(source_logits, dim=1)
    cand_preds = torch.argmax(candidate_logits, dim=1)
    src_preds = torch.argmax(source_logits, dim=1)

    # 2. Mean Prediction Entropy: H(p) = - sum(p * log(p)) over valid mask
    cand_log_probs = torch.log(torch.clamp(cand_probs, min=epsilon, max=1.0))
    pixel_entropy = -torch.sum(cand_probs * cand_log_probs, dim=1)  # (B, H, W)
    mean_entropy = float((torch.sum(pixel_entropy * valid_bool) / total_valid_pixels).item())

    # 3. Mean Maximum Confidence: max_c p_c over valid mask
    max_conf, _ = torch.max(cand_probs, dim=1)  # (B, H, W)
    mean_max_conf = float((torch.sum(max_conf * valid_bool) / total_valid_pixels).item())

    # 4. Source-vs-Adapted KL Divergence: D_KL(P_adapt || P_source)
    src_log_probs = torch.log(torch.clamp(src_probs, min=epsilon, max=1.0))
    kl_pixel = torch.sum(cand_probs * (cand_log_probs - src_log_probs), dim=1)  # (B, H, W)
    mean_kl = float((torch.sum(kl_pixel * valid_bool) / total_valid_pixels).item())

    # 5. Source-vs-Adapted Disagreement Fraction
    disagree_mask = (cand_preds != src_preds) & valid_bool
    disagree_pixels = float(torch.sum(disagree_mask).item())
    disagreement_fraction = disagree_pixels / total_valid_pixels

    # 6. BN Affine and Running Statistic Drift
    w_drift_sq = 0.0
    b_drift_sq = 0.0
    mean_drift_sq = 0.0
    var_drift_sq = 0.0

    for name, module in candidate_model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            rm_key = f"{name}.running_mean"
            rv_key = f"{name}.running_var"

            if module.weight is not None and w_key in source_bn_state:
                diff_w = module.weight.detach() - source_bn_state[w_key].to(module.weight.device)
                w_drift_sq += float(torch.sum(diff_w ** 2).item())
            if module.bias is not None and b_key in source_bn_state:
                diff_b = module.bias.detach() - source_bn_state[b_key].to(module.bias.device)
                b_drift_sq += float(torch.sum(diff_b ** 2).item())
            if module.running_mean is not None and rm_key in source_bn_state:
                diff_rm = module.running_mean.detach() - source_bn_state[rm_key].to(module.running_mean.device)
                mean_drift_sq += float(torch.sum(diff_rm ** 2).item())
            if module.running_var is not None and rv_key in source_bn_state:
                diff_rv = module.running_var.detach() - source_bn_state[rv_key].to(module.running_var.device)
                var_drift_sq += float(torch.sum(diff_rv ** 2).item())

    bn_affine_drift_l2 = float((w_drift_sq + b_drift_sq) ** 0.5)
    bn_running_stat_drift_l2 = float((mean_drift_sq + var_drift_sq) ** 0.5)

    # 7. Intermediate Feature Drift at model.conv_final input
    if candidate_features is not None and source_features is not None:
        diff_feat = candidate_features - source_features
        l2_feat_pixel = torch.sqrt(torch.sum(diff_feat ** 2, dim=1) + epsilon)  # (B, H, W)
        mean_feature_drift = float((torch.sum(l2_feat_pixel * valid_bool) / total_valid_pixels).item())
    else:
        mean_feature_drift = 0.0

    # 8. Predicted Flood Area (Class 2) and Ratio
    cand_flood_mask = (cand_preds == 2) & valid_bool
    src_flood_mask = (src_preds == 2) & valid_bool
    cand_flood_area = float(torch.sum(cand_flood_mask).item())
    src_flood_area = float(torch.sum(src_flood_mask).item())
    area_ratio = (cand_flood_area + 1.0) / (src_flood_area + 1.0)

    return {
        "mean_prediction_entropy": mean_entropy,
        "mean_max_confidence": mean_max_conf,
        "kl_divergence_source_adapted": mean_kl,
        "source_disagreement_fraction": disagreement_fraction,
        "source_disagreement_pixels": disagree_pixels,
        "bn_affine_drift_l2": bn_affine_drift_l2,
        "bn_running_stat_drift_l2": bn_running_stat_drift_l2,
        "intermediate_feature_drift_l2": mean_feature_drift,
        "predicted_flood_area": cand_flood_area,
        "source_predicted_flood_area": src_flood_area,
        "adapted_to_source_area_ratio": area_ratio,
    }


@torch.no_grad()
def compute_augmentation_inconsistency(
    model: nn.Module,
    xA: torch.Tensor,
    xB: torch.Tensor,
    pred_orig: torch.Tensor,
    sar_valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Computes horizontal flip augmentation inconsistency.
    
    Flips inputs spatially, evaluates with model in eval mode, inverts the flip
    on predictions, and measures the disagreement fraction on valid pixels.
    """
    model.eval()
    xA_flip = torch.flip(xA, dims=[-1])
    xB_flip = torch.flip(xB, dims=[-1])
    logits_flip = model(xA_flip, xB_flip)
    preds_flip = torch.argmax(logits_flip, dim=1)
    preds_flip_inv = torch.flip(preds_flip, dims=[-1])

    if sar_valid_mask is not None:
        valid_bool = (sar_valid_mask > 0).to(xA.device)
    else:
        valid_bool = torch.ones_like(preds_flip_inv, dtype=torch.bool)

    tot_valid = float(torch.sum(valid_bool).item())
    if tot_valid == 0:
        return 0.0

    inconsistent_pixels = float(torch.sum((pred_orig != preds_flip_inv) & valid_bool).item())
    return inconsistent_pixels / tot_valid
