"""ChangeGuard Plan V3 Work Package E5: Temporal Risk Probe Infrastructure.

Implements state-neutral bi-temporal null probes and generic risk discriminators:
1. Primary Temporal Null Risk Family:
   s_null = 0.5 * sum_{z in {pre2/pre2, post/post}} max(0, delta_null(z))
   delta_null(z) = mean_{sensory_valid} p_candidate(flood | z) - mean_{sensory_valid} p_source(flood | z)
   Predeclared direction: Higher indicates greater risk of hallucinated flood change.
   Zero-valid boundary: Unavailable (None), never numeric zero.

2. Secondary Source-Referenced Flood Change Family (on real pre2/post pair):
   s_flood_expansion = mean_{sensory_valid} max(0, p_candidate(flood) - p_source(flood))
   s_flood_disappearance = mean_{sensory_valid} max(0, p_source(flood) - p_candidate(flood))
   Preserves signed directions; does NOT assert source model correctness.

3. Generic Pre-Update Risk Evidence:
   - Candidate predictive entropy H(p)
   - Candidate maximum softmax confidence max_c p_c
   - Source-to-candidate KL divergence D_KL(p_cand || p_src) over all 3 classes
   - Hard prediction disagreement rate
   - Signed and absolute predicted flood-pixel count change
   - Weak/strong horizontal-flip augmentation inconsistency in native coordinates
   - BN affine weight and bias displacements (L2 norms)
   - Sensory-valid pixel count and probe availability flags

Strict Contracts:
- Probes execute strictly PRE_UPDATE without altering candidate parameters, buffers,
  optimizer states, RNG streams, or adaptation counters.
- Source model is evaluated in eval() mode with frozen running statistics and zero updates.
- Source and candidate models must never alias parameter or buffer memory pointers.
- Online probe inputs use sensory validity (MK0_MNA) ONLY. Ground-truth labels (MK0_MLU)
  and label != 3 masking are strictly forbidden online.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.changeguard.recorder import (
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
)


def assert_model_non_aliasing(model_a: nn.Module, model_b: nn.Module) -> None:
    """Verifies that model_a and model_b share zero parameter or buffer memory pointers."""
    if model_a is model_b:
        raise RuntimeError("Model aliasing violation: model_a and model_b are the identical Python object.")

    a_param_ptrs = {p.data_ptr() for p in model_a.parameters() if p.numel() > 0}
    b_param_ptrs = {p.data_ptr() for p in model_b.parameters() if p.numel() > 0}
    shared_params = a_param_ptrs.intersection(b_param_ptrs)
    if shared_params:
        raise RuntimeError(
            f"Model parameter memory aliasing detected! {len(shared_params)} shared parameter pointers found."
        )

    a_buf_ptrs = {b.data_ptr() for b in model_a.buffers() if b.numel() > 0}
    b_buf_ptrs = {b.data_ptr() for b in model_b.buffers() if b.numel() > 0}
    shared_bufs = a_buf_ptrs.intersection(b_buf_ptrs)
    if shared_bufs:
        raise RuntimeError(
            f"Model buffer memory aliasing detected! {len(shared_bufs)} shared buffer pointers found."
        )


def _normalize_prob_tensor(t: torch.Tensor) -> torch.Tensor:
    """Ensures probability map tensor has spatial shape (B, H, W)."""
    if t.dim() == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def compute_null_pair_risk_score(
    cand_prob_c2_pre2: torch.Tensor,
    cand_prob_c2_post: torch.Tensor,
    src_prob_c2_pre2: torch.Tensor,
    src_prob_c2_post: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Computes the primary temporal null-pair risk score s_null and component deltas.
    
    Formula:
        delta_null(z) = mean_{valid} p_candidate(flood | z) - mean_{valid} p_source(flood | z)
        s_null = 0.5 * (max(0, delta_null(pre2/pre2)) + max(0, delta_null(post/post)))

    Boundary Contract:
        If valid_mask has zero valid pixels, s_null and deltas are unavailable (None).
        Zero-valid is NEVER reported as safe or numeric 0.0.

    Returns:
        (s_null, delta_pre2, delta_post, cand_mean_pre2, cand_mean_post)
    """
    cand_prob_c2_pre2 = _normalize_prob_tensor(cand_prob_c2_pre2)
    cand_prob_c2_post = _normalize_prob_tensor(cand_prob_c2_post)
    src_prob_c2_pre2 = _normalize_prob_tensor(src_prob_c2_pre2)
    src_prob_c2_post = _normalize_prob_tensor(src_prob_c2_post)

    if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
        v_mask = (valid_mask.squeeze(1) == 1)
    else:
        v_mask = (valid_mask == 1)

    valid_count = int(v_mask.sum().item())
    if valid_count == 0:
        return None, None, None, None, None

    cand_pre2_valid = cand_prob_c2_pre2[v_mask]
    cand_post_valid = cand_prob_c2_post[v_mask]
    src_pre2_valid = src_prob_c2_pre2[v_mask]
    src_post_valid = src_prob_c2_post[v_mask]

    # Non-finite guard
    if not (
        torch.isfinite(cand_pre2_valid).all()
        and torch.isfinite(cand_post_valid).all()
        and torch.isfinite(src_pre2_valid).all()
        and torch.isfinite(src_post_valid).all()
    ):
        raise FloatingPointError("Non-finite probability values encountered during null probe evaluation.")

    cand_mean_pre2 = float(cand_pre2_valid.mean().item())
    cand_mean_post = float(cand_post_valid.mean().item())
    src_mean_pre2 = float(src_pre2_valid.mean().item())
    src_mean_post = float(src_post_valid.mean().item())

    delta_pre2 = cand_mean_pre2 - src_mean_pre2
    delta_post = cand_mean_post - src_mean_post

    s_null = 0.5 * (max(0.0, delta_pre2) + max(0.0, delta_post))

    return s_null, delta_pre2, delta_post, cand_mean_pre2, cand_mean_post


def compute_flood_change_probes(
    cand_prob_c2_real: torch.Tensor,
    src_prob_c2_real: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[Optional[float], Optional[float]]:
    """Computes source-referenced flood probability expansion and disappearance on real pre2/post pair.
    
    Formulas:
        s_flood_expansion = mean_{valid} max(0, p_candidate(flood) - p_source(flood))
        s_flood_disappearance = mean_{valid} max(0, p_source(flood) - p_candidate(flood))

    Returns:
        (s_flood_expansion, s_flood_disappearance)
    """
    cand_prob_c2_real = _normalize_prob_tensor(cand_prob_c2_real)
    src_prob_c2_real = _normalize_prob_tensor(src_prob_c2_real)

    if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
        v_mask = (valid_mask.squeeze(1) == 1)
    else:
        v_mask = (valid_mask == 1)

    valid_count = int(v_mask.sum().item())
    if valid_count == 0:
        return None, None

    p_cand = cand_prob_c2_real[v_mask]
    p_src = src_prob_c2_real[v_mask]

    if not (torch.isfinite(p_cand).all() and torch.isfinite(p_src).all()):
        raise FloatingPointError("Non-finite probability values in flood change probe.")

    diff = p_cand - p_src
    expansion = float(torch.clamp(diff, min=0.0).mean().item())
    disappearance = float(torch.clamp(-diff, min=0.0).mean().item())

    return expansion, disappearance


def compute_generic_risk_evidence(
    cand_probs: torch.Tensor,
    src_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    cand_probs_strong_aligned: Optional[torch.Tensor] = None,
    bn_weight_displacement: Optional[float] = None,
    bn_bias_displacement: Optional[float] = None,
    eps: float = 1e-7,
) -> Dict[str, Any]:
    """Computes generic pre-update risk indicators over sensory-valid pixels.
    
    Includes:
    - mean_entropy: - 1/N sum_i sum_c p_cand,c log(p_cand,c + eps)
    - mean_max_confidence: 1/N sum_i max_c p_cand,c
    - source_to_candidate_kl: 1/N sum_i sum_c p_cand,c log((p_cand,c + eps) / (p_src,c + eps))
    - hard_disagreement_rate: fraction where argmax(p_cand) != argmax(p_src)
    - delta_flood_pixels: count(pred_cand == 2) - count(pred_src == 2)
    - abs_delta_flood_pixels: |delta_flood_pixels|
    - aug_inconsistency: Total variation between weak and aligned strong views
    """
    if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
        v_mask = (valid_mask.squeeze(1) == 1)
    else:
        v_mask = (valid_mask == 1)

    valid_count = int(v_mask.sum().item())
    if valid_count == 0:
        return {
            "valid_pixels": 0,
            "mean_entropy": None,
            "mean_max_confidence": None,
            "source_to_candidate_kl": None,
            "hard_disagreement_rate": None,
            "delta_flood_pixels": 0,
            "abs_delta_flood_pixels": 0,
            "aug_inconsistency": None,
            "bn_weight_displacement": bn_weight_displacement,
            "bn_bias_displacement": bn_bias_displacement,
            "probe_available": False,
        }

    # Extract valid pixel distributions: (N_valid, num_classes)
    p_cand_v = cand_probs.permute(0, 2, 3, 1)[v_mask]
    p_src_v = src_probs.permute(0, 2, 3, 1)[v_mask]

    if not (torch.isfinite(p_cand_v).all() and torch.isfinite(p_src_v).all()):
        raise FloatingPointError("Non-finite probability values in generic risk evidence.")

    # 1. Predictive Entropy
    log_p_cand = torch.log(torch.clamp(p_cand_v, min=eps, max=1.0))
    entropy_pixel = -(p_cand_v * log_p_cand).sum(dim=-1)
    mean_entropy = float(entropy_pixel.mean().item())

    # 2. Maximum Confidence
    max_conf_pixel = p_cand_v.max(dim=-1).values
    mean_max_confidence = float(max_conf_pixel.mean().item())

    # 3. Source-to-Candidate KL Divergence: KL(p_cand || p_src)
    # Fixed direction: candidate relative to immutable source reference
    log_ratio = torch.log(torch.clamp(p_cand_v, min=eps, max=1.0)) - torch.log(
        torch.clamp(p_src_v, min=eps, max=1.0)
    )
    kl_pixel = (p_cand_v * log_ratio).sum(dim=-1)
    # Clamp small negative numerical noise from float precision
    source_to_candidate_kl = float(torch.clamp(kl_pixel, min=0.0).mean().item())

    # 4. Hard Disagreement Rate
    pred_cand = p_cand_v.argmax(dim=-1)
    pred_src = p_src_v.argmax(dim=-1)
    hard_disagreement_rate = float((pred_cand != pred_src).float().mean().item())

    # 5. Flood Pixel Count Shifts
    cand_flood_count = int((pred_cand == 2).sum().item())
    src_flood_count = int((pred_src == 2).sum().item())
    delta_flood_pixels = cand_flood_count - src_flood_count
    abs_delta_flood_pixels = abs(delta_flood_pixels)

    # 6. Augmentation Inconsistency (Weak vs Strong Aligned)
    aug_inconsistency = None
    if cand_probs_strong_aligned is not None:
        p_strong_v = cand_probs_strong_aligned.permute(0, 2, 3, 1)[v_mask]
        if torch.isfinite(p_strong_v).all():
            # Total Variation distance: 0.5 * sum_c |p_weak_c - p_strong_c| bounded in [0, 1]
            tv_pixel = 0.5 * torch.abs(p_cand_v - p_strong_v).sum(dim=-1)
            aug_inconsistency = float(tv_pixel.mean().item())

    return {
        "valid_pixels": valid_count,
        "mean_entropy": mean_entropy,
        "mean_max_confidence": mean_max_confidence,
        "source_to_candidate_kl": source_to_candidate_kl,
        "hard_disagreement_rate": hard_disagreement_rate,
        "delta_flood_pixels": delta_flood_pixels,
        "abs_delta_flood_pixels": abs_delta_flood_pixels,
        "aug_inconsistency": aug_inconsistency,
        "bn_weight_displacement": bn_weight_displacement,
        "bn_bias_displacement": bn_bias_displacement,
        "probe_available": True,
    }


class E5RiskProbeEvaluator:
    """State-neutral evaluator for E5 temporal null probes and generic risk discriminators.
    
    Guarantees:
    - Exception safety: Restores PyTorch CPU/CUDA RNG states on any failure.
    - Buffer neutrality: Restores candidate model buffers after null-pair and strong forward passes.
    - Zero gradients: All probe forward passes wrapped in torch.no_grad().
    - Source immutability: Evaluated strictly in eval() mode with frozen running statistics.
    - Non-aliasing enforcement: Asserts zero shared memory pointers before execution.
    """

    def __init__(
        self,
        candidate_model: nn.Module,
        source_model: nn.Module,
        bn_control_model: Optional[nn.Module] = None,
        initial_bn_weights: Optional[Dict[str, torch.Tensor]] = None,
        initial_bn_biases: Optional[Dict[str, torch.Tensor]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.candidate_model = candidate_model.to(self.device)
        self.source_model = source_model.to(self.device)

        if bn_control_model is not None:
            self.bn_control_model = bn_control_model.to(self.device)
        else:
            self.bn_control_model = copy.deepcopy(source_model).to(self.device)

        # Configure matched_bn_no_update_bs16 for bn_control_model:
        # Zero backward calls, zero parameter changes, eval mode on parameters,
        # but current-batch statistics on BatchNorm modules with track_running_stats=False.
        self.bn_control_model.eval()
        self.bn_control_model.requires_grad_(False)
        for m in self.bn_control_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.track_running_stats = False

        # Enforce non-aliasing across all three models at initialization
        assert_model_non_aliasing(self.candidate_model, self.source_model)
        assert_model_non_aliasing(self.candidate_model, self.bn_control_model)
        assert_model_non_aliasing(self.source_model, self.bn_control_model)

        # Cache initial BN affine parameters for displacement computation
        if initial_bn_weights is None or initial_bn_biases is None:
            self.initial_bn_weights: Dict[str, torch.Tensor] = {}
            self.initial_bn_biases: Dict[str, torch.Tensor] = {}
            for name, m in self.candidate_model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if m.weight is not None:
                        self.initial_bn_weights[name] = m.weight.detach().clone()
                    if m.bias is not None:
                        self.initial_bn_biases[name] = m.bias.detach().clone()
        else:
            self.initial_bn_weights = {k: v.detach().clone() for k, v in initial_bn_weights.items()}
            self.initial_bn_biases = {k: v.detach().clone() for k, v in initial_bn_biases.items()}

    def _get_bn_displacements(self) -> Tuple[float, float]:
        """Calculates current candidate model BN affine displacements from source."""
        w_diffs = []
        b_diffs = []
        for name, m in self.candidate_model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None and name in self.initial_bn_weights:
                    w_diffs.append((m.weight - self.initial_bn_weights[name].to(m.weight.device)).flatten())
                if m.bias is not None and name in self.initial_bn_biases:
                    b_diffs.append((m.bias - self.initial_bn_biases[name].to(m.bias.device)).flatten())

        w_disp = float(torch.norm(torch.cat(w_diffs), p=2).item()) if w_diffs else 0.0
        b_disp = float(torch.norm(torch.cat(b_diffs), p=2).item()) if b_diffs else 0.0
        return w_disp, b_disp

    def evaluate_batch_probes_state_neutral(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluates all temporal null probes and generic risk signals state-neutrally.
        
        Args:
            xA: (B, C, H, W) pre2 SAR raster tensor.
            xB: (B, C, H, W) post SAR raster tensor.
            valid_mask: (B, H, W) binary sensory validity mask (1=valid, 0=nodata).
        
        Returns:
            Dictionary containing all telemetry, risk scores, probability maps, and normalization contexts.
        """
        # 1. Capture RNG states for exception-safe restoration
        is_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if is_cuda else None

        # 2. Snapshot buffers and training modes to guarantee state neutrality
        cand_buffers_snapshot = {k: v.clone() for k, v in self.candidate_model.named_buffers()}
        src_buffers_snapshot = {k: v.clone() for k, v in self.source_model.named_buffers()}
        bn_buffers_snapshot = {k: v.clone() for k, v in self.bn_control_model.named_buffers()}
        orig_cand_training_mode = self.candidate_model.training
        orig_src_training_mode = self.source_model.training
        orig_bn_training_mode = self.bn_control_model.training

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        valid_mask = valid_mask.to(self.device)

        if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)

        try:
            with torch.no_grad():
                # ==============================================================
                # Context 1: Immutable Source Reference (eval, frozen running stats)
                # ==============================================================
                self.source_model.eval()
                # Real pair: (pre2, post)
                src_logits_real = self.source_model(xA, xB)
                src_probs_real = F.softmax(src_logits_real, dim=1)
                src_prob_c2_real = src_probs_real[:, 2, :, :]
                src_pred_real = src_probs_real.argmax(dim=1).to(torch.uint8)

                # Null pre2 pair: (pre2, pre2)
                src_logits_pre2 = self.source_model(xA, xA)
                src_probs_pre2 = F.softmax(src_logits_pre2, dim=1)
                src_prob_c2_pre2 = src_probs_pre2[:, 2, :, :]
                src_pred_pre2 = src_probs_pre2.argmax(dim=1).to(torch.uint8)

                # Null post pair: (post, post)
                src_logits_post = self.source_model(xB, xB)
                src_probs_post = F.softmax(src_logits_post, dim=1)
                src_prob_c2_post = src_probs_post[:, 2, :, :]
                src_pred_post = src_probs_post.argmax(dim=1).to(torch.uint8)

                # ==============================================================
                # Context 2: Matched BN No-Update Control (current batch stats)
                # ==============================================================
                # Real pair: (pre2, post)
                bn_logits_real = self.bn_control_model(xA, xB)
                bn_probs_real = F.softmax(bn_logits_real, dim=1)
                bn_prob_c2_real = bn_probs_real[:, 2, :, :]
                bn_pred_real = bn_probs_real.argmax(dim=1).to(torch.uint8)

                # Null pre2 pair: (pre2, pre2)
                bn_logits_pre2 = self.bn_control_model(xA, xA)
                bn_probs_pre2 = F.softmax(bn_logits_pre2, dim=1)
                bn_prob_c2_pre2 = bn_probs_pre2[:, 2, :, :]
                bn_pred_pre2 = bn_probs_pre2.argmax(dim=1).to(torch.uint8)

                # Null post pair: (post, post)
                bn_logits_post = self.bn_control_model(xB, xB)
                bn_probs_post = F.softmax(bn_logits_post, dim=1)
                bn_prob_c2_post = bn_probs_post[:, 2, :, :]
                bn_pred_post = bn_probs_post.argmax(dim=1).to(torch.uint8)

                # ==============================================================
                # Context 3: Candidate Adaptation PRE_UPDATE
                # ==============================================================
                # Real pair: (pre2, post)
                cand_logits_real = self.candidate_model(xA, xB)
                cand_probs_real = F.softmax(cand_logits_real, dim=1)
                cand_prob_c2_real = cand_probs_real[:, 2, :, :]
                cand_pred_real = cand_probs_real.argmax(dim=1).to(torch.uint8)

                # Null pre2 pair: (pre2, pre2)
                cand_logits_pre2 = self.candidate_model(xA, xA)
                cand_probs_pre2 = F.softmax(cand_logits_pre2, dim=1)
                cand_prob_c2_pre2 = cand_probs_pre2[:, 2, :, :]
                cand_pred_pre2 = cand_probs_pre2.argmax(dim=1).to(torch.uint8)

                # Null post pair: (post, post)
                cand_logits_post = self.candidate_model(xB, xB)
                cand_probs_post = F.softmax(cand_logits_post, dim=1)
                cand_prob_c2_post = cand_probs_post[:, 2, :, :]
                cand_pred_post = cand_probs_post.argmax(dim=1).to(torch.uint8)

                # Strong Augmentation View (Horizontal Flip on real pair)
                xA_strong = torch.flip(xA, dims=[-1])
                xB_strong = torch.flip(xB, dims=[-1])
                cand_logits_strong = self.candidate_model(xA_strong, xB_strong)
                cand_probs_strong = F.softmax(cand_logits_strong, dim=1)
                cand_probs_strong_aligned = torch.flip(cand_probs_strong, dims=[-1])

                # Displacement metrics
                bn_w_disp, bn_b_disp = self._get_bn_displacements()

                # Primary Null Risk Score
                s_null, delta_pre2, delta_post, cand_m_pre2, cand_m_post = compute_null_pair_risk_score(
                    cand_prob_c2_pre2=cand_prob_c2_pre2,
                    cand_prob_c2_post=cand_prob_c2_post,
                    src_prob_c2_pre2=src_prob_c2_pre2,
                    src_prob_c2_post=src_prob_c2_post,
                    valid_mask=valid_mask,
                )

                # Secondary Flood Change Probes
                s_flood_exp, s_flood_dis = compute_flood_change_probes(
                    cand_prob_c2_real=cand_prob_c2_real,
                    src_prob_c2_real=src_prob_c2_real,
                    valid_mask=valid_mask,
                )

                # Generic Risk Evidence
                generic_evidence = compute_generic_risk_evidence(
                    cand_probs=cand_probs_real,
                    src_probs=src_probs_real,
                    valid_mask=valid_mask,
                    cand_probs_strong_aligned=cand_probs_strong_aligned,
                    bn_weight_displacement=bn_w_disp,
                    bn_bias_displacement=bn_b_disp,
                )

                # ==============================================================
                # Comparative Telemetry: Source vs BN vs Candidate across 3 pairs
                # ==============================================================
                V = (valid_mask == 1).float()
                N_valid = int(V.sum().item())

                if N_valid > 0:
                    # 1. Real Pair
                    src_mean_c2_real = float((src_prob_c2_real * V).sum().item() / N_valid)
                    bn_mean_c2_real = float((bn_prob_c2_real * V).sum().item() / N_valid)
                    cand_mean_c2_real = float((cand_prob_c2_real * V).sum().item() / N_valid)

                    disagree_src_bn_real = float(((src_pred_real != bn_pred_real).float() * V).sum().item() / N_valid)
                    disagree_bn_cand_real = float(((bn_pred_real != cand_pred_real).float() * V).sum().item() / N_valid)
                    disagree_src_cand_real = float(((src_pred_real != cand_pred_real).float() * V).sum().item() / N_valid)

                    diff_c2_src_bn_real = float(((bn_prob_c2_real - src_prob_c2_real) * V).sum().item() / N_valid)
                    diff_c2_bn_cand_real = float(((cand_prob_c2_real - bn_prob_c2_real) * V).sum().item() / N_valid)
                    diff_c2_src_cand_real = float(((cand_prob_c2_real - src_prob_c2_real) * V).sum().item() / N_valid)

                    # 2. Null Pre2 Pair
                    src_mean_c2_pre2 = float((src_prob_c2_pre2 * V).sum().item() / N_valid)
                    bn_mean_c2_pre2 = float((bn_prob_c2_pre2 * V).sum().item() / N_valid)
                    cand_mean_c2_pre2 = float((cand_prob_c2_pre2 * V).sum().item() / N_valid)

                    disagree_src_bn_pre2 = float(((src_pred_pre2 != bn_pred_pre2).float() * V).sum().item() / N_valid)
                    disagree_bn_cand_pre2 = float(((bn_pred_pre2 != cand_pred_pre2).float() * V).sum().item() / N_valid)
                    disagree_src_cand_pre2 = float(((src_pred_pre2 != cand_pred_pre2).float() * V).sum().item() / N_valid)

                    diff_c2_src_bn_pre2 = float(((bn_prob_c2_pre2 - src_prob_c2_pre2) * V).sum().item() / N_valid)
                    diff_c2_bn_cand_pre2 = float(((cand_prob_c2_pre2 - bn_prob_c2_pre2) * V).sum().item() / N_valid)
                    diff_c2_src_cand_pre2 = float(((cand_prob_c2_pre2 - src_prob_c2_pre2) * V).sum().item() / N_valid)

                    # 3. Null Post Pair
                    src_mean_c2_post = float((src_prob_c2_post * V).sum().item() / N_valid)
                    bn_mean_c2_post = float((bn_prob_c2_post * V).sum().item() / N_valid)
                    cand_mean_c2_post = float((cand_prob_c2_post * V).sum().item() / N_valid)

                    disagree_src_bn_post = float(((src_pred_post != bn_pred_post).float() * V).sum().item() / N_valid)
                    disagree_bn_cand_post = float(((bn_pred_post != cand_pred_post).float() * V).sum().item() / N_valid)
                    disagree_src_cand_post = float(((src_pred_post != cand_pred_post).float() * V).sum().item() / N_valid)

                    diff_c2_src_bn_post = float(((bn_prob_c2_post - src_prob_c2_post) * V).sum().item() / N_valid)
                    diff_c2_bn_cand_post = float(((cand_prob_c2_post - bn_prob_c2_post) * V).sum().item() / N_valid)
                    diff_c2_src_cand_post = float(((cand_prob_c2_post - src_prob_c2_post) * V).sum().item() / N_valid)
                else:
                    src_mean_c2_real = bn_mean_c2_real = cand_mean_c2_real = None
                    disagree_src_bn_real = disagree_bn_cand_real = disagree_src_cand_real = None
                    diff_c2_src_bn_real = diff_c2_bn_cand_real = diff_c2_src_cand_real = None

                    src_mean_c2_pre2 = bn_mean_c2_pre2 = cand_mean_c2_pre2 = None
                    disagree_src_bn_pre2 = disagree_bn_cand_pre2 = disagree_src_cand_pre2 = None
                    diff_c2_src_bn_pre2 = diff_c2_bn_cand_pre2 = diff_c2_src_cand_pre2 = None

                    src_mean_c2_post = bn_mean_c2_post = cand_mean_c2_post = None
                    disagree_src_bn_post = disagree_bn_cand_post = disagree_src_cand_post = None
                    diff_c2_src_bn_post = diff_c2_bn_cand_post = diff_c2_src_cand_post = None

                return {
                    # Spatial Predictions & Probability Tensors
                    # Real Pair
                    "cand_pred": cand_pred_real,
                    "cand_pred_real": cand_pred_real,
                    "src_pred": src_pred_real,
                    "src_pred_real": src_pred_real,
                    "bn_pred_real": bn_pred_real,
                    "cand_probs_real": cand_probs_real,
                    "src_probs_real": src_probs_real,
                    "bn_probs_real": bn_probs_real,
                    "cand_probs_strong_aligned": cand_probs_strong_aligned,
                    "cand_prob_c2": cand_prob_c2_real,
                    "src_prob_c2": src_prob_c2_real,
                    "bn_prob_c2_real": bn_prob_c2_real,

                    # Null Pre2 Pair
                    "cand_pred_null_pre2": cand_pred_pre2,
                    "src_pred_null_pre2": src_pred_pre2,
                    "bn_pred_null_pre2": bn_pred_pre2,
                    "cand_probs_null_pre2": cand_probs_pre2,
                    "src_probs_null_pre2": src_probs_pre2,
                    "bn_probs_null_pre2": bn_probs_pre2,
                    "cand_prob_c2_null_pre2": cand_prob_c2_pre2,
                    "src_prob_c2_null_pre2": src_prob_c2_pre2,
                    "bn_prob_c2_null_pre2": bn_prob_c2_pre2,

                    # Null Post Pair
                    "cand_pred_null_post": cand_pred_post,
                    "src_pred_null_post": src_pred_post,
                    "bn_pred_null_post": bn_pred_post,
                    "cand_probs_null_post": cand_probs_post,
                    "src_probs_null_post": src_probs_post,
                    "bn_probs_null_post": bn_probs_post,
                    "cand_prob_c2_null_post": cand_prob_c2_post,
                    "src_prob_c2_null_post": src_prob_c2_post,
                    "bn_prob_c2_null_post": bn_prob_c2_post,

                    # Sensory valid mask
                    "valid_mask": valid_mask,

                    # Primary Temporal Null Risk
                    "s_null": s_null,
                    "delta_null_pre2": delta_pre2,
                    "delta_null_post": delta_post,
                    "cand_null_pre2_mean": cand_m_pre2,
                    "cand_null_post_mean": cand_m_post,

                    # Secondary Flood Change Probes
                    "s_flood_expansion": s_flood_exp,
                    "s_flood_disappearance": s_flood_dis,

                    # Generic Risk Evidence
                    "mean_entropy": generic_evidence["mean_entropy"],
                    "mean_max_confidence": generic_evidence["mean_max_confidence"],
                    "source_to_candidate_kl": generic_evidence["source_to_candidate_kl"],
                    "hard_disagreement_rate": generic_evidence["hard_disagreement_rate"],
                    "delta_flood_pixels": generic_evidence["delta_flood_pixels"],
                    "abs_delta_flood_pixels": generic_evidence["abs_delta_flood_pixels"],
                    "aug_inconsistency": generic_evidence["aug_inconsistency"],
                    "bn_weight_displacement": bn_w_disp,
                    "bn_bias_displacement": bn_b_disp,
                    "valid_pixels": generic_evidence["valid_pixels"],
                    "probe_available": generic_evidence["probe_available"] and (s_null is not None),

                    # Provenance & Normalization Context
                    "source_norm_context": "source_frozen_running_stats",
                    "bn_control_norm_context": "matched_bn_no_update_bs16",
                    "candidate_norm_context": "candidate_e2_conf90_pre_update",

                    # BN Diagnostic Comparative Telemetry: Real Pair
                    "src_mean_c2_real": src_mean_c2_real,
                    "bn_mean_c2_real": bn_mean_c2_real,
                    "cand_mean_c2_real": cand_mean_c2_real,
                    "disagree_src_bn_real": disagree_src_bn_real,
                    "disagree_bn_cand_real": disagree_bn_cand_real,
                    "disagree_src_cand_real": disagree_src_cand_real,
                    "diff_c2_src_bn_real": diff_c2_src_bn_real,
                    "diff_c2_bn_cand_real": diff_c2_bn_cand_real,
                    "diff_c2_src_cand_real": diff_c2_src_cand_real,

                    # BN Diagnostic Comparative Telemetry: Null Pre2 Pair
                    "src_mean_c2_pre2": src_mean_c2_pre2,
                    "bn_mean_c2_pre2": bn_mean_c2_pre2,
                    "cand_mean_c2_pre2": cand_mean_c2_pre2,
                    "disagree_src_bn_pre2": disagree_src_bn_pre2,
                    "disagree_bn_cand_pre2": disagree_bn_cand_pre2,
                    "disagree_src_cand_pre2": disagree_src_cand_pre2,
                    "diff_c2_src_bn_pre2": diff_c2_src_bn_pre2,
                    "diff_c2_bn_cand_pre2": diff_c2_bn_cand_pre2,
                    "diff_c2_src_cand_pre2": diff_c2_src_cand_pre2,

                    # BN Diagnostic Comparative Telemetry: Null Post Pair
                    "src_mean_c2_post": src_mean_c2_post,
                    "bn_mean_c2_post": bn_mean_c2_post,
                    "cand_mean_c2_post": cand_mean_c2_post,
                    "disagree_src_bn_post": disagree_src_bn_post,
                    "disagree_bn_cand_post": disagree_bn_cand_post,
                    "disagree_src_cand_post": disagree_src_cand_post,
                    "diff_c2_src_bn_post": diff_c2_src_bn_post,
                    "diff_c2_bn_cand_post": diff_c2_bn_cand_post,
                    "diff_c2_src_cand_post": diff_c2_src_cand_post,
                }
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            for k, v in self.candidate_model.named_buffers():
                if k in cand_buffers_snapshot:
                    v.copy_(cand_buffers_snapshot[k])

            for k, v in self.source_model.named_buffers():
                if k in src_buffers_snapshot:
                    v.copy_(src_buffers_snapshot[k])

            for k, v in self.bn_control_model.named_buffers():
                if k in bn_buffers_snapshot:
                    v.copy_(bn_buffers_snapshot[k])

            self.candidate_model.train(orig_cand_training_mode)
            self.source_model.train(orig_src_training_mode)
            self.bn_control_model.train(orig_bn_training_mode)
