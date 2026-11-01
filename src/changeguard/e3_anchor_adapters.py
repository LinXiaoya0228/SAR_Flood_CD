"""ChangeGuard Plan V3 Work Package E3: Source-Anchor Adapters.

Implements the three frozen-source anchor families:
1. Probability Anchor: Forward KL(p_source || p_candidate_aligned)
2. Logit Anchor: Per-pixel class-mean centered MSE with fixed scale (T=1.0)
3. Feature Anchor: Cosine feature distance on penultimate ECAM dense representation

Anchor-View Convention:
- Immutable source-reference model evaluates original/weak view (xA, xB).
- Candidate model evaluates strong view (horizontal flip) realigned into original coordinates.
- Raw anchor discrepancy contains both normalization-context and view/augmentation differences,
  and is not attributed solely to model parameter drift.

Strict Immutability & Telemetry Invariants:
1. Source model is an entirely separate object; zero storage pointer aliasing.
2. Source model is in eval mode, requires_grad=False, zero parameter gradients.
3. Cryptographic SHA-256 hash of source model state dict verified before and after each stream.
4. Expanded gradient telemetry records raw and weighted norms, ratio, and cosine conflict.
5. PRE_UPDATE primary predictions cached before ground-truth access.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.changeguard.recorder import (
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
)


class E3SourceAnchorAdapter:
    """Configurable Weak-Strong Consistency Adapter with Frozen Source Anchors."""

    SUPPORTED_FAMILIES = {"prob", "logit", "feat"}

    def __init__(
        self,
        model: nn.Module,
        source_model: nn.Module,
        family: str = "prob",
        lambda_val: float = 0.50,
        feature_layer_name: str = "conv_final_input",
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        confidence_threshold: float = 0.90,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        family_clean = str(family).lower().strip()
        if family_clean not in self.SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported anchor family '{family}'. Must be one of {self.SUPPORTED_FAMILIES}.")
        self.family = family_clean
        self.lambda_val = float(lambda_val)
        if self.lambda_val < 0.0:
            raise ValueError(f"Negative lambda_val {self.lambda_val} not allowed.")

        self.feature_layer_name = str(feature_layer_name)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.confidence_threshold = float(confidence_threshold)

        # ---------------------------------------------------------------------
        # 1. Physical Storage Independence & Source Verification
        # ---------------------------------------------------------------------
        if source_model is self.model:
            raise RuntimeError("Source model cannot be the identical Python object to candidate model.")

        self.source_model = source_model.to(self.device)
        self.source_model.eval()
        self.source_model.requires_grad_(False)

        # Ensure no memory pointer overlap
        cand_ptrs = {p.data_ptr() for p in self.model.parameters()} | {
            b.data_ptr() for b in self.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in self.source_model.parameters()} | {
            b.data_ptr() for b in self.source_model.buffers() if b is not None
        }
        shared_ptrs = cand_ptrs.intersection(src_ptrs)
        if shared_ptrs:
            raise RuntimeError(f"CRITICAL MEMORY ALIASING: {len(shared_ptrs)} shared pointers between source and candidate.")

        # Cache source state dict for candidate model resets
        self._source_state_dict = {
            k: v.detach().clone().to(self.device)
            for k, v in self.source_model.state_dict().items()
        }

        # Cache source cryptographic SHA-256 checksum
        self.initial_source_sha256 = self.compute_source_sha256()

        # Cache BN buffer shapes
        self._bn_buffer_shapes: Dict[str, Tuple[torch.Size, torch.Size]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mean_shape = (
                    module.running_mean.shape
                    if module.running_mean is not None
                    else torch.Size([module.num_features])
                )
                var_shape = (
                    module.running_var.shape
                    if module.running_var is not None
                    else torch.Size([module.num_features])
                )
                self._bn_buffer_shapes[name] = (mean_shape, var_shape)

        # ---------------------------------------------------------------------
        # 2. Feature Hook Setup (if family == "feat")
        # ---------------------------------------------------------------------
        self._cand_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._src_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._last_cand_feature: Optional[torch.Tensor] = None
        self._last_src_feature: Optional[torch.Tensor] = None

        if self.family == "feat":
            self._setup_feature_hooks()

        # ---------------------------------------------------------------------
        # 3. Configure Candidate BatchNorm Affine Updates & Optimizer
        # ---------------------------------------------------------------------
        self._configure_candidate_bn()
        self._init_optimizer()

        # Step and update counters
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def compute_source_sha256(self) -> str:
        """Computes deterministic SHA-256 hash over all source parameters and buffers."""
        h = hashlib.sha256()
        sd = self.source_model.state_dict()
        for k in sorted(sd.keys()):
            tensor = sd[k]
            h.update(k.encode("utf-8"))
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def verify_source_immutability(self) -> None:
        """Verifies bit-exact SHA-256 state and zero gradients on immutable source-reference model."""
        current_hash = self.compute_source_sha256()
        if current_hash != self.initial_source_sha256:
            raise RuntimeError(
                f"CRITICAL SOURCE MUTATION: Source model SHA-256 changed from {self.initial_source_sha256} to {current_hash}"
            )
        for name, p in self.source_model.named_parameters():
            if p.grad is not None:
                raise RuntimeError(f"CRITICAL LEAKAGE: Source parameter '{name}' has non-None gradient!")

    def _locate_target_module(self, root_module: nn.Module) -> nn.Module:
        """Locates the layer preceding conv_final for feature hooking."""
        if hasattr(root_module, "model") and hasattr(root_module.model, "conv_final"):
            return root_module.model.conv_final
        if hasattr(root_module, "conv_final"):
            return root_module.conv_final
        if hasattr(root_module, "head"):
            # For MockSNUNet
            return root_module.head
        raise AttributeError(f"Could not locate 'conv_final' or 'head' on module {root_module.__class__.__name__}")

    def _setup_feature_hooks(self) -> None:
        """Installs forward hooks on conv_final input to capture 128-channel dense ECAM features."""
        cand_target = self._locate_target_module(self.model)
        src_target = self._locate_target_module(self.source_model)

        def cand_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self._last_cand_feature = inp[0]

        def src_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self._last_src_feature = inp[0]

        self._cand_hook_handle = cand_target.register_forward_hook(cand_hook)
        self._src_hook_handle = src_target.register_forward_hook(src_hook)

    def _configure_candidate_bn(self) -> None:
        """Configures candidate model: eval globally, active BN train() with track_running_stats=False."""
        self.model.eval()
        self.model.requires_grad_(False)

        self.bn_param_names: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    self.bn_param_names.append(f"{name}.weight")
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    self.bn_param_names.append(f"{name}.bias")

    def _init_optimizer(self) -> None:
        """Initializes SGD optimizer strictly over active candidate BatchNorm affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BatchNorm affine parameters found in candidate model!")
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def reset_to_source(self) -> None:
        """Restores candidate model state bit-exactly to source checkpoint and resets optimizer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.track_running_stats = True
                if name in self._bn_buffer_shapes:
                    mean_shape, var_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(mean_shape, device=self.device))
                    module.register_buffer("running_var", torch.ones(var_shape, device=self.device))
                    module.register_buffer(
                        "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=self.device)
                    )

        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)
        self._configure_candidate_bn()
        self._init_optimizer()
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def adapt_and_predict(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: torch.Tensor,
        sample_ids: Optional[Sequence[str]] = None,
        event_ids: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs one source-anchored adaptation step and returns PRE_UPDATE predictions and telemetry."""
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        sar_valid_mask = sar_valid_mask.to(self.device)

        if xA.shape != xB.shape:
            raise ValueError(f"Temporal inputs xA {xA.shape} and xB {xB.shape} must have identical shape.")
        if sar_valid_mask.dim() == 4 and sar_valid_mask.shape[1] == 1:
            sar_valid_mask = sar_valid_mask.squeeze(1)
        elif sar_valid_mask.dim() != 3:
            raise ValueError(f"Unsupported mask shape {sar_valid_mask.shape}. Expected (B, H, W).")

        valid_mask_bool = (sar_valid_mask == 1)
        valid_pixel_count = int(valid_mask_bool.sum().item())

        # ---------------------------------------------------------------------
        # 1. Primary PRE_UPDATE Candidate Forward Pass on Weak View
        # ---------------------------------------------------------------------
        self._last_cand_feature = None
        logits_weak = self.model(xA, xB)
        probs_weak = F.softmax(logits_weak, dim=1)
        pred_weak = probs_weak.argmax(dim=1)
        probs_weak_det = probs_weak.detach()
        probs_weak_max = probs_weak_det.max(dim=1)[0]

        primary_prediction = {
            "logits": logits_weak.detach(),
            "probabilities": probs_weak_det,
            "predicted_class": pred_weak.detach(),
            "predicted_flood_pixels": int((pred_weak == 2).sum().item()),
            "valid_pixels": valid_pixel_count,
        }

        # ---------------------------------------------------------------------
        # 2. Immutable Source-Reference Forward Pass on Weak View (no_grad)
        # ---------------------------------------------------------------------
        self._last_src_feature = None
        with torch.no_grad():
            self.source_model.eval()
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1).detach()
            feat_src = self._last_src_feature.detach() if self.family == "feat" else None

        # Pre-adaptation probability disagreement over valid pixels
        if valid_pixel_count > 0:
            l1_disagreement = 0.5 * torch.abs(probs_weak_det - probs_src).sum(dim=1)
            prob_disagreement_val = float((l1_disagreement * valid_mask_bool.float()).sum().item() / valid_pixel_count)
        else:
            prob_disagreement_val = 0.0

        # ---------------------------------------------------------------------
        # 3. Candidate Strong View Forward Pass & Realignment
        # ---------------------------------------------------------------------
        xA_strong = torch.flip(xA, dims=[-1])
        xB_strong = torch.flip(xB, dims=[-1])

        self._last_cand_feature = None
        logits_strong = self.model(xA_strong, xB_strong)
        logits_strong_aligned = torch.flip(logits_strong, dims=[-1])
        log_probs_strong_aligned = F.log_softmax(logits_strong_aligned, dim=1)
        feat_cand_aligned = (
            torch.flip(self._last_cand_feature, dims=[-1]) if self.family == "feat" else None
        )

        # ---------------------------------------------------------------------
        # 4. Consistency Masking & Loss Formulations
        # ---------------------------------------------------------------------
        conf_mask = (probs_weak_max >= self.confidence_threshold) & valid_mask_bool
        selected_pixels = int(conf_mask.sum().item())

        update_applied = False
        action = "skip"
        loss_val: Optional[float] = None
        loss_cons_raw_val: Optional[float] = None
        loss_anc_raw_val: Optional[float] = None
        loss_anc_weighted_val: Optional[float] = None

        grad_norm_cons: float = 0.0
        grad_norm_anc_raw: float = 0.0
        grad_norm_anc_weighted: float = 0.0
        grad_norm_total: float = 0.0
        grad_norm_ratio_weighted: float = 0.0
        grad_cosine_conflict: float = 0.0
        update_norm_l2: float = 0.0
        failure_reason: Optional[str] = None

        if valid_pixel_count == 0 or (selected_pixels == 0 and self.lambda_val == 0.0):
            self.num_updates_skipped += 1
            action = "skip"
        else:
            # A. Base Consistency Loss
            if selected_pixels > 0:
                kl_cons_pixel = (
                    probs_weak_det * (torch.log(probs_weak_det + 1e-12) - log_probs_strong_aligned)
                ).sum(dim=1)
                loss_cons = (kl_cons_pixel * conf_mask.float()).sum() / conf_mask.sum().clamp(min=1.0)
            else:
                loss_cons = torch.tensor(0.0, device=self.device)

            # B. Source Anchor Loss
            if self.lambda_val > 0.0 and valid_pixel_count > 0:
                if self.family == "prob":
                    # Forward KL: D_KL(p_src || p_cand_aligned)
                    kl_anc_pixel = (
                        probs_src * (torch.log(probs_src + 1e-12) - log_probs_strong_aligned)
                    ).sum(dim=1)
                    loss_anc = (kl_anc_pixel * valid_mask_bool.float()).sum() / valid_mask_bool.sum().clamp(min=1.0)
                elif self.family == "logit":
                    # Per-pixel class-mean centering (shift-invariant MSE)
                    z_src_c = logits_src - logits_src.mean(dim=1, keepdim=True)
                    z_cand_c = logits_strong_aligned - logits_strong_aligned.mean(dim=1, keepdim=True)
                    mse_pixel = ((z_cand_c - z_src_c) ** 2).mean(dim=1)
                    loss_anc = (mse_pixel * valid_mask_bool.float()).sum() / valid_mask_bool.sum().clamp(min=1.0)
                elif self.family == "feat":
                    # Bounded cosine distance on penultimate features
                    f_src_norm = F.normalize(feat_src, p=2, dim=1, eps=1e-8)
                    f_cand_norm = F.normalize(feat_cand_aligned, p=2, dim=1, eps=1e-8)
                    cos_dist_pixel = 1.0 - (f_cand_norm * f_src_norm).sum(dim=1)
                    loss_anc = (cos_dist_pixel * valid_mask_bool.float()).sum() / valid_mask_bool.sum().clamp(min=1.0)
                else:
                    raise RuntimeError(f"Unexpected family {self.family}")
            else:
                loss_anc = torch.tensor(0.0, device=self.device)

            loss_total = loss_cons + self.lambda_val * loss_anc

            # Numerical boundary check: finite loss
            if not torch.isfinite(loss_total):
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = f"Non-finite loss encountered: {loss_total.item()}"
                self.optimizer.zero_grad()
            else:
                loss_val = float(loss_total.item())
                loss_cons_raw_val = float(loss_cons.item())
                loss_anc_raw_val = float(loss_anc.item())
                loss_anc_weighted_val = float(self.lambda_val * loss_anc_raw_val)

                # -----------------------------------------------------------------
                # 5. Expanded Decomposed Gradient Telemetry
                # -----------------------------------------------------------------
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]

                # Component 1: Consistency gradient vector
                if selected_pixels > 0 and loss_cons.requires_grad:
                    g_cons_list = torch.autograd.grad(
                        loss_cons, trainable_params, retain_graph=True, allow_unused=True
                    )
                    g_cons_vec = torch.cat([
                        g.contiguous().view(-1) if g is not None else torch.zeros(p.numel(), device=self.device)
                        for g, p in zip(g_cons_list, trainable_params)
                    ])
                else:
                    g_cons_vec = torch.zeros(sum(p.numel() for p in trainable_params), device=self.device)

                # Component 2: Anchor gradient vector
                if self.lambda_val > 0.0 and valid_pixel_count > 0 and loss_anc.requires_grad:
                    g_anc_list = torch.autograd.grad(
                        loss_anc, trainable_params, retain_graph=True, allow_unused=True
                    )
                    g_anc_vec = torch.cat([
                        g.contiguous().view(-1) if g is not None else torch.zeros(p.numel(), device=self.device)
                        for g, p in zip(g_anc_list, trainable_params)
                    ])
                else:
                    g_anc_vec = torch.zeros(sum(p.numel() for p in trainable_params), device=self.device)

                # Norms and conflict metrics
                grad_norm_cons = float(torch.norm(g_cons_vec, p=2).item())
                grad_norm_anc_raw = float(torch.norm(g_anc_vec, p=2).item())
                grad_norm_anc_weighted = float(self.lambda_val * grad_norm_anc_raw)

                g_total_vec = g_cons_vec + (self.lambda_val * g_anc_vec)
                grad_norm_total = float(torch.norm(g_total_vec, p=2).item())

                grad_norm_ratio_weighted = float(grad_norm_anc_weighted / (grad_norm_cons + 1e-12))
                denom_cos = (grad_norm_cons * grad_norm_anc_raw) + 1e-12
                grad_cosine_conflict = float(torch.dot(g_cons_vec, g_anc_vec).item() / denom_cos)

                # -----------------------------------------------------------------
                # 6. Parameter Update & Displacement Measurement
                # -----------------------------------------------------------------
                self.optimizer.zero_grad()
                loss_total.backward()

                # Verify all gradients finite
                all_finite = all(
                    torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None
                )
                if not all_finite:
                    self.num_updates_failed += 1
                    action = "failure"
                    failure_reason = "Non-finite parameter gradients detected."
                    self.optimizer.zero_grad()
                else:
                    # Cache pre-step parameter values
                    pre_params = torch.cat([p.detach().clone().view(-1) for p in trainable_params])
                    self.optimizer.step()
                    post_params = torch.cat([p.detach().clone().view(-1) for p in trainable_params])

                    update_norm_l2 = float(torch.norm(post_params - pre_params, p=2).item())
                    update_applied = True
                    action = "update"
                    self.num_updates_applied += 1

        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        # BN affine displacements relative to source
        bn_weight_disp, bn_bias_disp = compute_bn_affine_displacement(
            self.model, self._source_state_dict
        )

        step_telemetry = {
            "step_index": self.step_count,
            "action": action,
            "update_applied": update_applied,
            "loss": loss_val,
            "consistency_loss": loss_cons_raw_val,
            "anchor_loss": loss_anc_raw_val,
            "anchor_loss_weighted": loss_anc_weighted_val,
            "gradient_norm_consistency": grad_norm_cons,
            "gradient_norm_anchor_raw": grad_norm_anc_raw,
            "gradient_norm_anchor_weighted": grad_norm_anc_weighted,
            "gradient_norm_total": grad_norm_total,
            "gradient_norm_ratio_weighted": grad_norm_ratio_weighted,
            "gradient_cosine_conflict": grad_cosine_conflict,
            "update_norm_l2": update_norm_l2,
            "prob_disagreement_pre": prob_disagreement_val,
            "bn_weight_displacement": bn_weight_disp,
            "bn_bias_displacement": bn_bias_disp,
            "valid_pixels": valid_pixel_count,
            "selected_pixels": selected_pixels,
            "selected_pixel_fraction": float(selected_pixels / max(1, valid_pixel_count)),
            "failure_reason": failure_reason,
            "batch_time_ms": batch_time_ms,
        }

        return {
            "primary_prediction": primary_prediction,
            "step_telemetry": step_telemetry,
        }

    def close(self) -> None:
        """Removes forward hooks and cleans up resources."""
        if self._cand_hook_handle is not None:
            self._cand_hook_handle.remove()
            self._cand_hook_handle = None
        if self._src_hook_handle is not None:
            self._src_hook_handle.remove()
            self._src_hook_handle = None


def compute_offline_interpretation_metrics(
    candidate_preds_by_sample: Dict[str, np.ndarray],
    base_preds_by_sample: Dict[str, np.ndarray],
    source_preds_by_sample: Dict[str, np.ndarray],
    labels_by_sample: Dict[str, np.ndarray],
    valid_masks_by_sample: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Computes offline source-anchor interpretation metrics strictly post-persistence.
    
    Guards & Invariants:
    1. Double-masking is strictly applied: eval_mask = (valid_mask == 1) & (labels != 3).
    2. Computed only during offline evaluation after labels are unlocked.
    3. Never influences adaptation, masking, loss construction, or optimizer updates.
    4. Evaluates binary flood detection (Class 2 vs Non-Flood [0, 1]).
    5. Explicitly calculates harmful-error recovery alongside beneficial-adaptation loss.
    """
    total_valid_pixels = 0
    disagree_src_cnt = 0
    disagree_3class_src_cnt = 0
    agree_src_cnt = 0
    agree_base_cnt = 0

    harmful_recovery_cnt = 0
    harmful_base_error_total = 0

    beneficial_retention_cnt = 0
    beneficial_loss_cnt = 0
    beneficial_adaptation_total = 0

    regression_from_base_cnt = 0
    correction_over_base_cnt = 0
    base_correct_total = 0
    base_wrong_total = 0
    source_correct_total = 0

    for sid in candidate_preds_by_sample:
        cand_p = candidate_preds_by_sample[sid]
        base_p = base_preds_by_sample[sid]
        src_p = source_preds_by_sample[sid]
        lbl = labels_by_sample[sid]
        v_mask = valid_masks_by_sample[sid]

        eval_mask = (v_mask == 1) & (lbl != 3)
        valid_cnt = int(np.sum(eval_mask))
        if valid_cnt == 0:
            continue

        total_valid_pixels += valid_cnt

        y = (lbl[eval_mask] == 2)
        c = (cand_p[eval_mask] == 2)
        b = (base_p[eval_mask] == 2)
        s = (src_p[eval_mask] == 2)

        # Disagreement & Agreement
        disagree_src_cnt += int(np.sum(c != s))
        disagree_3class_src_cnt += int(np.sum(cand_p[eval_mask] != src_p[eval_mask]))
        agree_src_cnt += int(np.sum(c == s))
        agree_base_cnt += int(np.sum(c == b))

        s_corr = (s == y)
        s_wrong = ~s_corr
        b_corr = (b == y)
        b_wrong = ~b_corr
        c_corr = (c == y)
        c_wrong = ~c_corr

        source_correct_total += int(np.sum(s_corr))
        base_correct_total += int(np.sum(b_corr))
        base_wrong_total += int(np.sum(b_wrong))

        # Harmful base error recovery: source correct, base wrong, candidate correct
        harmful_recovery_cnt += int(np.sum(s_corr & b_wrong & c_corr))
        harmful_base_error_total += int(np.sum(s_corr & b_wrong))

        # Beneficial adaptation retention & loss: source wrong, base correct
        beneficial_retention_cnt += int(np.sum(s_wrong & b_corr & c_corr))
        beneficial_loss_cnt += int(np.sum(s_wrong & b_corr & c_wrong))
        beneficial_adaptation_total += int(np.sum(s_wrong & b_corr))

        # Net shifts relative to base consistency
        regression_from_base_cnt += int(np.sum(b_corr & c_wrong))
        correction_over_base_cnt += int(np.sum(b_wrong & c_corr))

    # Conservation identity check: retention + loss == total
    if (beneficial_retention_cnt + beneficial_loss_cnt) != beneficial_adaptation_total:
        raise ValueError("Beneficial adaptation conservation identity violated: retention + loss != total")

    return {
        "total_valid_pixels": total_valid_pixels,
        "source_candidate_disagreement": {
            "count": disagree_src_cnt,
            "rate_rel_valid": float(disagree_src_cnt / max(1, total_valid_pixels)),
            "three_class_count": disagree_3class_src_cnt,
            "three_class_rate_rel_valid": float(disagree_3class_src_cnt / max(1, total_valid_pixels)),
        },
        "source_agreement": {
            "count": agree_src_cnt,
            "rate_rel_valid": float(agree_src_cnt / max(1, total_valid_pixels)),
        },
        "base_consistency_agreement": {
            "count": agree_base_cnt,
            "rate_rel_valid": float(agree_base_cnt / max(1, total_valid_pixels)),
        },
        "harmful_base_error_recovery": {
            "recovered_count": harmful_recovery_cnt,
            "denominator_harmful_base_errors": harmful_base_error_total,
            "recovery_rate_rel_harmful": float(harmful_recovery_cnt / max(1, harmful_base_error_total)),
            "rate_rel_valid": float(harmful_recovery_cnt / max(1, total_valid_pixels)),
        },
        "beneficial_adaptation_retention": {
            "retained_count": beneficial_retention_cnt,
            "denominator_beneficial_adaptations": beneficial_adaptation_total,
            "retention_rate_rel_beneficial": float(beneficial_retention_cnt / max(1, beneficial_adaptation_total)),
            "rate_rel_valid": float(beneficial_retention_cnt / max(1, total_valid_pixels)),
        },
        "beneficial_adaptation_loss": {
            "lost_count": beneficial_loss_cnt,
            "denominator_beneficial_adaptations": beneficial_adaptation_total,
            "loss_rate_rel_beneficial": float(beneficial_loss_cnt / max(1, beneficial_adaptation_total)),
            "rate_rel_valid": float(beneficial_loss_cnt / max(1, total_valid_pixels)),
        },
        "regression_from_base": {
            "regressed_count": regression_from_base_cnt,
            "denominator_base_correct": base_correct_total,
            "regression_rate_rel_base_correct": float(regression_from_base_cnt / max(1, base_correct_total)),
            "rate_rel_valid": float(regression_from_base_cnt / max(1, total_valid_pixels)),
        },
        "correction_over_base": {
            "corrected_count": correction_over_base_cnt,
            "denominator_base_wrong": base_wrong_total,
            "correction_rate_rel_base_wrong": float(correction_over_base_cnt / max(1, base_wrong_total)),
            "rate_rel_valid": float(correction_over_base_cnt / max(1, total_valid_pixels)),
        },
    }
