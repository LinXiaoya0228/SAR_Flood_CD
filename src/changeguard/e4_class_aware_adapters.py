"""ChangeGuard Plan V3 Work Package E4: Class-Aware Intervention Adapters.

Implements the four isolated class-aware adaptation interventions (K1-K4)
evaluated on top of the fixed, unmodified E2 consistency base (E2_CONF_90):

1. K1 (Class-Balanced Selection): Imposes deterministic per-predicted-class quotas
   using source-calibrated priors (pi_0=0.7470, pi_1=0.0704, pi_2=0.1826) to prevent
   majority background classes from overwhelming pseudo-label selection.
2. K2 (Flood-Aware Loss Weighting): Applies a source-calibrated weight boost (w_flood=4.0)
   to predicted flood pixels inside the adaptation loss denominator, preserving full
   prediction coverage and streaming order.
3. K3 (Permanent-Water Protection): Uses frozen-source class-1 (permanent water) predictions
   to mask out permanent water pixels from consistency adaptation, preventing water-to-flood
   false alarms.
4. K4 (Flood/Change Preservation): Adds an auxiliary log-loss penalty on pixels where the
   frozen source model detected flood with high confidence (p_src(2) >= 0.50), preventing
   catastrophic flood erasure while monitoring expansion.

Strict Invariants & Epistemic Boundaries:
- Interventions K1-K4 are strictly isolated; no combinations in E4.1/E4.2.
- No ground-truth labels or GT class masks are accessible during adaptation.
- Frozen source model is immutable, detached, and evaluated in eval() mode.
- Source agreement is explicitly distinct from correctness.
- Complete per-class selection, confusion, and transition diagnostics are tracked.
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

# Declared Source Calibration Class Distribution across 27 source events (1.06B pixels)
SOURCE_CALIBRATION_PI = {
    0: 0.7469878,  # Land / Background (~74.70%)
    1: 0.0704130,  # Permanent Water (~7.04%)
    2: 0.1825992,  # Flood Water (~18.26%)
}


class E4ClassAwareAdapter:
    """Weak-Strong Consistency Adapter with Isolated Class-Aware Interventions (K1-K4)."""

    SUPPORTED_INTERVENTIONS = {
        "base",
        "k1_balanced_selection",
        "k2_flood_weighting",
        "k3_perm_protection",
        "k4_flood_preservation",
    }

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        intervention: str = "base",
        confidence_threshold: float = 0.90,
        learning_rate: float = 1.0e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        # K1 parameters
        k1_quota_source: Optional[Dict[int, float]] = None,
        # K2 parameters
        k2_class_weights: Optional[Dict[int, float]] = None,
        # K3 parameters
        k3_perm_threshold: float = 0.50,
        # K4 parameters
        k4_flood_threshold: float = 0.50,
        k4_lambda_pres: float = 0.50,
        device: str = "cpu",
    ):
        if intervention not in self.SUPPORTED_INTERVENTIONS:
            raise ValueError(
                f"Unsupported intervention '{intervention}'. "
                f"Must be one of {sorted(self.SUPPORTED_INTERVENTIONS)}"
            )

        self.intervention = intervention
        self.device = torch.device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.learning_rate = float(learning_rate)

        # K1 configuration
        self.k1_priors = (
            dict(k1_quota_source)
            if k1_quota_source is not None
            else dict(SOURCE_CALIBRATION_PI)
        )

        # K2 configuration: default w = (1.0, 1.0, 4.0) from pi_0 / pi_2 ~= 4.09
        self.k2_weights = (
            dict(k2_class_weights)
            if k2_class_weights is not None
            else {0: 1.0, 1: 1.0, 2: 4.0}
        )

        # K3 configuration
        self.k3_perm_threshold = float(k3_perm_threshold)

        # K4 configuration
        self.k4_flood_threshold = float(k4_flood_threshold)
        self.k4_lambda_pres = float(k4_lambda_pres)

        # Candidate Model Setup
        self.model = model.to(self.device)
        self.model.train()

        # Identify affine BN parameters for adaptation
        self.adapted_param_names: List[str] = []
        self.adapted_params: List[nn.Parameter] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if module.weight is not None:
                    module.weight.requires_grad = True
                    self.adapted_params.append(module.weight)
                    self.adapted_param_names.append(f"{name}.weight")
                if module.bias is not None:
                    module.bias.requires_grad = True
                    self.adapted_params.append(module.bias)
                    self.adapted_param_names.append(f"{name}.bias")

        # Freeze all non-BN parameters
        for name, param in self.model.named_parameters():
            if name not in self.adapted_param_names:
                param.requires_grad = False

        if not self.adapted_params:
            raise RuntimeError("No BatchNorm affine parameters found to adapt in candidate model.")

        self.optimizer = optim.Adam(
            self.adapted_params,
            lr=self.learning_rate,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        )

        # Cache initial state dict for BN affine displacement tracking
        self._initial_state_dict = {
            k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
        }

        # Source Model Setup (Required for K3 and K4; optional for base, K1, K2)
        self.source_model: Optional[nn.Module] = None
        self.source_initial_state_hash: Optional[str] = None
        if source_model is not None:
            self._init_source_model(source_model)
        elif self.intervention in {"k3_perm_protection", "k4_flood_preservation"}:
            raise ValueError(f"Intervention '{self.intervention}' requires a non-null source_model.")

        # Step accounting
        self.num_updates_applied: int = 0
        self.num_updates_skipped: int = 0
        self.num_updates_failed: int = 0

    def _init_source_model(self, source_model: nn.Module) -> None:
        """Initializes and verifies the immutable source reference model."""
        if source_model is self.model:
            raise ValueError("source_model must not be identical object to candidate model.")

        self.source_model = source_model.to(self.device)
        self.source_model.eval()
        for p in self.source_model.parameters():
            p.requires_grad = False

        # Verify zero parameter pointer aliasing
        cand_ptrs = {p.data_ptr() for p in self.model.parameters()}
        for p in self.source_model.parameters():
            if p.data_ptr() in cand_ptrs:
                raise RuntimeError("Security violation: source_model shares memory pointer with candidate model.")

        self.source_initial_state_hash = self._compute_source_state_hash()

    def _compute_source_state_hash(self) -> str:
        """Computes SHA-256 digest of source model state dict."""
        if self.source_model is None:
            return ""
        hasher = hashlib.sha256()
        for k, v in sorted(self.source_model.state_dict().items()):
            hasher.update(k.encode("utf-8"))
            hasher.update(v.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def verify_source_immutability(self) -> bool:
        """Verifies that source model state dict has not changed."""
        if self.source_model is None:
            return True
        current_hash = self._compute_source_state_hash()
        return current_hash == self.source_initial_state_hash

    # -------------------------------------------------------------------------
    # Spatial Realignment Transform (Weak vs Strong View)
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_strong_view(xA: torch.Tensor, xB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies locked horizontal flip strong augmentation."""
        return torch.flip(xA, dims=[3]), torch.flip(xB, dims=[3])

    @staticmethod
    def realign_strong_logits(logits_strong: torch.Tensor) -> torch.Tensor:
        """Inverses horizontal flip on strong output tensor."""
        return torch.flip(logits_strong, dims=[3])

    # -------------------------------------------------------------------------
    # Forward Step & Adaptation Execution
    # -------------------------------------------------------------------------
    def adapt_step(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_ids: Sequence[str],
        event_id: int,
    ) -> Dict[str, Any]:
        """Executes a single test-time adaptation step on a batch.
        
        Enforces:
        1. Pre-update primary prediction extracted before any parameter modifications.
        2. Clean separation of sensory data; zero ground-truth label access.
        3. Detailed telemetry of selection, weighting, and source-transition metrics.
        """
        start_time = time.perf_counter()
        xA = xA.to(self.device, non_blocking=True)
        xB = xB.to(self.device, non_blocking=True)
        V = valid_mask.to(self.device, non_blocking=True).bool()

        B, _, H, W = xA.shape
        valid_pixel_count = int(V.sum().item())

        # 1. Forward Candidate Model on Weak View (PRE_UPDATE)
        self.model.eval()
        with torch.no_grad():
            logits_weak_pre = self.model(xA, xB)
            probs_weak_pre = F.softmax(logits_weak_pre, dim=1)
            pred_cand_pre = probs_weak_pre.argmax(dim=1)
            conf_weak_pre, _ = probs_weak_pre.max(dim=1)
        self.model.train()

        # 2. Forward Source Model on Weak View (if available)
        pred_src: Optional[torch.Tensor] = None
        probs_src: Optional[torch.Tensor] = None
        if self.source_model is not None:
            src_rng_state = torch.get_rng_state()
            src_cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if (self.device.type == "cuda" and torch.cuda.is_available())
                else None
            )
            src_buffers_before = {k: v.clone() for k, v in self.source_model.named_buffers()}
            try:
                with torch.no_grad():
                    logits_src = self.source_model(xA, xB)
                    probs_src = F.softmax(logits_src, dim=1)
                    pred_src = probs_src.argmax(dim=1)
            finally:
                torch.set_rng_state(src_rng_state)
                if src_cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(src_cuda_rng_state)
                for k, v in self.source_model.named_buffers():
                    if k in src_buffers_before:
                        v.copy_(src_buffers_before[k])

        # 3. Base Selection Mask: V and conf_weak_pre >= threshold
        base_conf_mask = V & (conf_weak_pre >= self.confidence_threshold)

        # 4. Intervention-Specific Selection & Weighting Diagnostics
        intervention_meta: Dict[str, Any] = {}
        loss_pixel_mask: torch.Tensor = base_conf_mask
        pixel_loss_weights: Optional[torch.Tensor] = None
        aux_loss: Optional[torch.Tensor] = None

        if self.intervention == "base":
            intervention_meta = self._compute_base_meta(base_conf_mask, pred_cand_pre, V)

        elif self.intervention == "k1_balanced_selection":
            loss_pixel_mask, k1_meta = self._apply_k1_selection(
                base_conf_mask=base_conf_mask,
                pred_cand=pred_cand_pre,
                conf_weak=conf_weak_pre,
                valid_mask=V,
                valid_pixel_count=valid_pixel_count,
            )
            intervention_meta = k1_meta

        elif self.intervention == "k2_flood_weighting":
            pixel_loss_weights, k2_meta = self._apply_k2_weighting(
                base_conf_mask=base_conf_mask,
                pred_cand=pred_cand_pre,
                valid_mask=V,
            )
            intervention_meta = k2_meta

        elif self.intervention == "k3_perm_protection":
            assert pred_src is not None
            loss_pixel_mask, k3_meta = self._apply_k3_protection(
                base_conf_mask=base_conf_mask,
                pred_src=pred_src,
                probs_src=probs_src,
                pred_cand=pred_cand_pre,
                valid_mask=V,
            )
            intervention_meta = k3_meta

        elif self.intervention == "k4_flood_preservation":
            assert probs_src is not None and pred_src is not None
            k4_meta = self._apply_k4_preservation(
                pred_src=pred_src,
                probs_src=probs_src,
                pred_cand=pred_cand_pre,
                valid_mask=V,
            )
            intervention_meta = k4_meta

        # 5. Compute Adaptation Loss and Gradient Step
        num_loss_pixels = int(loss_pixel_mask.sum().item())
        has_k4_aux = (
            self.intervention == "k4_flood_preservation"
            and bool(intervention_meta.get("k4_active_pixels", 0) > 0)
        )

        if num_loss_pixels == 0 and not has_k4_aux:
            # Skip update if no pixels to adapt
            self.num_updates_skipped += 1
            action = "SKIPPED_EMPTY_MASK"
            step_loss_val = 0.0
            grad_norm = 0.0
            update_norm = 0.0
        else:
            self.optimizer.zero_grad()

            # Forward candidate on Strong View (Horizontally Flipped)
            xA_strong, xB_strong = self.apply_strong_view(xA, xB)
            logits_strong = self.model(xA_strong, xB_strong)
            logits_strong_aligned = self.realign_strong_logits(logits_strong)
            probs_strong_aligned = F.softmax(logits_strong_aligned, dim=1)

            # Re-evaluate weak view in train mode with gradients attached
            logits_weak_train = self.model(xA, xB)
            p_weak = F.softmax(logits_weak_train.detach(), dim=1)
            p_strong = probs_strong_aligned

            # Consistency cross-entropy per pixel: - sum_c p_weak(c) * log p_strong(c)
            # Clip for numerical safety
            log_p_strong = torch.log(p_strong.clamp(min=1e-12))
            per_pixel_loss = -(p_weak * log_p_strong).sum(dim=1)  # [B, H, W]

            if num_loss_pixels > 0:
                if pixel_loss_weights is not None:
                    # K2 weighted consistency loss
                    w_sum = (pixel_loss_weights * loss_pixel_mask.float()).sum()
                    if w_sum > 0:
                        loss_cons = (per_pixel_loss * pixel_loss_weights * loss_pixel_mask.float()).sum() / w_sum
                    else:
                        loss_cons = torch.tensor(0.0, device=self.device)
                else:
                    loss_cons = per_pixel_loss[loss_pixel_mask].mean()
            else:
                loss_cons = torch.tensor(0.0, device=self.device)

            # Combine with auxiliary loss (K4)
            if has_k4_aux and probs_src is not None:
                src_flood_mask = V & (probs_src[:, 2] >= self.k4_flood_threshold)
                if src_flood_mask.sum() > 0:
                    p2_strong = probs_strong_aligned[:, 2].clamp(min=1e-12)
                    loss_pres = -torch.log(p2_strong[src_flood_mask]).mean()
                    total_loss = loss_cons + self.k4_lambda_pres * loss_pres
                    intervention_meta["raw_preservation_loss"] = float(loss_pres.item())
                    intervention_meta["weighted_preservation_loss"] = float((self.k4_lambda_pres * loss_pres).item())
                else:
                    total_loss = loss_cons
            else:
                total_loss = loss_cons

            intervention_meta["consistency_loss"] = float(loss_cons.item())
            intervention_meta["total_loss"] = float(total_loss.item())

            step_loss_val = float(total_loss.item())

            # Backward and Optimize
            if not math.isnan(step_loss_val) and not math.isinf(step_loss_val) and total_loss.requires_grad:
                pre_params = torch.cat([p.detach().clone().view(-1) for p in self.adapted_params])
                total_loss.backward()
                grad_norm = compute_gradient_l2_norm(self.adapted_params)
                self.optimizer.step()
                post_params = torch.cat([p.detach().clone().view(-1) for p in self.adapted_params])
                update_norm = float(torch.norm(post_params - pre_params, p=2).item())
                self.num_updates_applied += 1
                action = "APPLIED"
            else:
                self.num_updates_failed += 1
                action = "FAILED_NUMERICAL"
                grad_norm = 0.0
                update_norm = 0.0

        # Compute BN affine displacement relative to initial source checkpoint
        bn_weight_disp, bn_bias_disp = compute_bn_affine_displacement(
            self.model, self._initial_state_dict
        )

        # 6. Post-Update Forward (POST_UPDATE diagnostic)
        self.model.eval()
        with torch.no_grad():
            logits_weak_post = self.model(xA, xB)
            probs_weak_post = F.softmax(logits_weak_post, dim=1)
            pred_cand_post = probs_weak_post.argmax(dim=1)
        self.model.train()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "action": action,
            "step_loss": step_loss_val,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "bn_weight_displacement": bn_weight_disp,
            "bn_bias_displacement": bn_bias_disp,
            "elapsed_ms": elapsed_ms,
            "num_loss_pixels": num_loss_pixels,
            "valid_pixel_count": valid_pixel_count,
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach().cpu().numpy(),
                "probs_cand_pre": probs_weak_pre.detach().cpu().numpy(),
                "pred_src": pred_src.detach().cpu().numpy() if pred_src is not None else None,
                "probs_src": probs_src.detach().cpu().numpy() if probs_src is not None else None,
            },
            "post_prediction": {
                "pred_cand_post": pred_cand_post.detach().cpu().numpy(),
            },
            "intervention_meta": intervention_meta,
        }

    # -------------------------------------------------------------------------
    # Internal Intervention Handlers
    # -------------------------------------------------------------------------
    def _compute_base_meta(
        self, base_conf_mask: torch.Tensor, pred_cand: torch.Tensor, V: torch.Tensor
    ) -> Dict[str, Any]:
        """Extracts per-class selection counts for unmodified base."""
        meta = {"intervention": "base"}
        for c in range(3):
            c_mask = base_conf_mask & (pred_cand == c)
            meta[f"selected_pixels_c{c}"] = int(c_mask.sum().item())
        meta["total_selected_pixels"] = int(base_conf_mask.sum().item())
        meta["total_valid_pixels"] = int(V.sum().item())
        return meta

    def _apply_k1_selection(
        self,
        base_conf_mask: torch.Tensor,
        pred_cand: torch.Tensor,
        conf_weak: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_pixel_count: int,
        predeclared_cap: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """K1: Genuinely class-balanced confidence selection.

        1. Starts from eligible pixels satisfying sensory-valid and confidence >= threshold.
        2. Partitions eligible pixels by candidate predicted class c in {0, 1, 2}.
        3. Computes eligible count A_c for each class.
        4. Enforces uniform deterministic batch cap C = floor(N_eligible / 3) for all non-empty classes.
        5. Empty classes (A_c == 0) receive zero selections; unused capacity is strictly discarded.
        6. Breaks confidence ties via exact deterministic lexicographical sort:
           Primary key: descending confidence
           Secondary key: ascending flattened spatial index
           Zero floating-point perturbations applied.
        """
        B, H, W = valid_mask.shape
        device = valid_mask.device
        final_selected_mask = torch.zeros_like(base_conf_mask, dtype=torch.bool)

        n_eligible = int(base_conf_mask.sum().item())

        # Determine uniform batch-level cap per class across the 3 semantic classes
        if predeclared_cap is not None:
            batch_cap = int(predeclared_cap)
        elif n_eligible > 0:
            batch_cap = max(1, math.floor(n_eligible / 3))
        else:
            batch_cap = 0

        k1_meta: Dict[str, Any] = {
            "intervention": "k1_balanced_selection",
            "batch_cap": batch_cap,
            "total_eligible_pixels": n_eligible,
            "available_pixels": {},
            "selected_pixels": {},
            "requested_quota": {},
            "realised_quota": {},
            "empty_classes": {},
        }

        for c in range(3):
            cand_c_mask = base_conf_mask & (pred_cand == c)
            a_c = int(cand_c_mask.sum().item())
            k1_meta["available_pixels"][c] = a_c
            k1_meta["requested_quota"][c] = batch_cap

            if a_c == 0:
                k1_meta["realised_quota"][c] = 0
                k1_meta["selected_pixels"][c] = 0
                k1_meta["empty_classes"][c] = True
                continue

            k1_meta["empty_classes"][c] = False
            r_c = min(a_c, batch_cap)
            k1_meta["realised_quota"][c] = r_c

            if r_c == a_c:
                final_selected_mask |= cand_c_mask
                k1_meta["selected_pixels"][c] = r_c
            else:
                # Top-r_c pixels selected via exact deterministic lexicographical ordering:
                # 1. Descending confidence
                # 2. Ascending flattened spatial index
                indices = torch.nonzero(cand_c_mask, as_tuple=False)  # [N, 3] (b, h, w)
                confs = conf_weak[indices[:, 0], indices[:, 1], indices[:, 2]]  # [N]
                flat_spatial_idx = indices[:, 0] * (H * W) + indices[:, 1] * W + indices[:, 2]

                # Step 1: Sort by spatial index ascending (stable)
                spatial_order = torch.argsort(flat_spatial_idx, descending=False, stable=True)
                indices_sorted = indices[spatial_order]
                confs_sorted = confs[spatial_order]

                # Step 2: Sort by confidence descending (stable)
                # Preserves ascending spatial index for ties bit-exactly
                conf_order = torch.argsort(confs_sorted, descending=True, stable=True)
                chosen_indices = indices_sorted[conf_order[:r_c]]

                final_selected_mask[chosen_indices[:, 0], chosen_indices[:, 1], chosen_indices[:, 2]] = True
                k1_meta["selected_pixels"][c] = r_c

        k1_meta["total_selected_pixels"] = int(final_selected_mask.sum().item())
        k1_meta["total_valid_pixels"] = valid_pixel_count
        k1_meta["selected_fraction"] = (
            float(k1_meta["total_selected_pixels"] / valid_pixel_count)
            if valid_pixel_count > 0
            else 0.0
        )
        return final_selected_mask, k1_meta

    def _apply_k2_weighting(
        self,
        base_conf_mask: torch.Tensor,
        pred_cand: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """K2: Flood-aware loss weighting inside adaptation loss denominator."""
        weights = torch.ones_like(base_conf_mask, dtype=torch.float32)

        k2_meta: Dict[str, Any] = {
            "intervention": "k2_flood_weighting",
            "candidate_counts": {},
            "selected_counts": {},
            "effective_weights": self.k2_weights,
        }

        for c in range(3):
            c_mask = valid_mask & (pred_cand == c)
            sel_c_mask = base_conf_mask & (pred_cand == c)
            k2_meta["candidate_counts"][c] = int(c_mask.sum().item())
            k2_meta["selected_counts"][c] = int(sel_c_mask.sum().item())
            w_c = self.k2_weights.get(c, 1.0)
            weights[pred_cand == c] = w_c

        k2_meta["total_selected_pixels"] = int(base_conf_mask.sum().item())
        k2_meta["total_valid_pixels"] = int(valid_mask.sum().item())
        return weights, k2_meta

    def _apply_k3_exclusion(
        self,
        base_conf_mask: torch.Tensor,
        pred_src: torch.Tensor,
        probs_src: torch.Tensor,
        pred_cand: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """K3: Source-predicted permanent-water exclusion.

        Excludes pixels predicted as permanent water (class 1) by the frozen source model
        (pred_src == 1, frozen-source argmax) from the consistency adaptation loss mask.

        Definitional Boundary & Explicit Claim:
        This operation excludes source-identified permanent water pixels from directly
        contributing to candidate adaptation updates. It does NOT guarantee invariant
        candidate predictions at those pixels because shared normalization statistics
        and affine parameters still adapt on other pixels.
        """
        # Permanent water excluded mask: where source model predicted class 1 (argmax)
        src_perm_mask = valid_mask & (pred_src == 1)
        num_excluded = int(src_perm_mask.sum().item())
        valid_pixel_count = int(valid_mask.sum().item())

        # Candidate selection mask with permanent water excluded
        adapted_mask = base_conf_mask & (~src_perm_mask)

        # Confidence distribution on excluded pixels
        p1_src = probs_src[:, 1]
        if num_excluded > 0:
            p1_excluded = p1_src[src_perm_mask]
            conf_mean = float(p1_excluded.mean().item())
            conf_min = float(p1_excluded.min().item())
            conf_max = float(p1_excluded.max().item())
        else:
            conf_mean = conf_min = conf_max = 0.0

        # Candidate changes away from source-predicted class 1
        cand_on_perm = pred_cand[src_perm_mask]
        cand_drift_away_cnt = int((cand_on_perm != 1).sum().item()) if num_excluded > 0 else 0

        k3_meta = {
            "intervention": "k3_perm_protection",
            "operation": "source_predicted_permanent_water_exclusion",
            "source_perm_criterion": "pred_src == 1 (frozen source argmax)",
            "excluded_pixels": num_excluded,
            "protected_pixels": num_excluded,  # retained for backward compatibility
            "excluded_fraction": float(num_excluded / max(1, valid_pixel_count)),
            "source_perm_conf_mean": conf_mean,
            "source_perm_conf_min": conf_min,
            "source_perm_conf_max": conf_max,
            "candidate_drift_away_from_src_perm_count": cand_drift_away_cnt,
            "total_selected_pixels": int(adapted_mask.sum().item()),
            "total_valid_pixels": valid_pixel_count,
        }
        return adapted_mask, k3_meta

    _apply_k3_protection = _apply_k3_exclusion

    def _apply_k4_preservation(
        self,
        pred_src: torch.Tensor,
        probs_src: torch.Tensor,
        pred_cand: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """K4: Frozen-source predicted flood/change preservation penalty."""
        # High confidence source flood evidence
        p2_src = probs_src[:, 2]
        src_flood_mask = valid_mask & (p2_src >= self.k4_flood_threshold)
        num_src_flood = int(src_flood_mask.sum().item())
        valid_pixel_count = int(valid_mask.sum().item())

        # Erasure and Expansion counts (relative to source predictions)
        # Erasure: Source predicted flood (class 2) but candidate predicts non-flood (!= 2)
        is_src_flood = valid_mask & (pred_src == 2)
        total_src_flood_cnt = int(is_src_flood.sum().item())
        erasure_cnt = int((is_src_flood & (pred_cand != 2)).sum().item())

        # Expansion: Source predicted non-flood (!= 2) but candidate predicts flood (== 2)
        is_src_nonflood = valid_mask & (pred_src != 2)
        total_src_nonflood_cnt = int(is_src_nonflood.sum().item())
        expansion_cnt = int((is_src_nonflood & (pred_cand == 2)).sum().item())

        k4_meta = {
            "intervention": "k4_flood_preservation",
            "k4_active_pixels": num_src_flood,
            "k4_active_fraction": float(num_src_flood / max(1, valid_pixel_count)),
            "source_flood_threshold": self.k4_flood_threshold,
            "erasure_count": erasure_cnt,
            "erasure_denominator": total_src_flood_cnt,
            "erasure_rate": float(erasure_cnt / max(1, total_src_flood_cnt)),
            "expansion_count": expansion_cnt,
            "expansion_denominator": total_src_nonflood_cnt,
            "expansion_rate": float(expansion_cnt / max(1, total_src_nonflood_cnt)),
            "total_valid_pixels": valid_pixel_count,
        }
        return k4_meta


# -----------------------------------------------------------------------------
# Offline Diagnostic Interpretation Metrics (Post-Persistence Analysis Schema)
# -----------------------------------------------------------------------------
def compute_e4_offline_interpretation_metrics(
    candidate_preds_by_sample: Dict[str, np.ndarray],
    base_preds_by_sample: Dict[str, np.ndarray],
    source_preds_by_sample: Dict[str, np.ndarray],
    labels_by_sample: Dict[str, np.ndarray],
    valid_masks_by_sample: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Computes comprehensive E4 class-aware offline diagnosis strictly post-persistence.
    
    Adheres strictly to the required analysis schema:
    1. 3x3 confusion matrix and class-wise precision, recall, F1.
    2. False positives and false negatives separated by true class, with explicit denominators.
    3. Transitions relative to frozen source prediction.
    4. Harmful source-correct-to-candidate-wrong transitions.
    5. Beneficial source-wrong-to-candidate-correct transitions.
    6. Lost beneficial corrections relative to base consistency.
    7. Preserved source errors.
    """
    total_valid_pixels = 0

    # 3x3 confusion matrix: C[true_class, pred_class]
    conf_3x3 = np.zeros((3, 3), dtype=np.int64)

    # Transitions from source to candidate: T[src_class, cand_class]
    trans_src_to_cand = np.zeros((3, 3), dtype=np.int64)

    # Source-relative error dynamics
    harmful_recovery_cnt = 0
    harmful_base_error_total = 0

    beneficial_retention_cnt = 0
    beneficial_loss_cnt = 0
    beneficial_adaptation_total = 0

    harmful_trans_src_corr_to_cand_wrong = 0
    beneficial_trans_src_wrong_to_cand_corr = 0
    preserved_source_error_cnt = 0
    total_source_correct_cnt = 0
    total_source_wrong_cnt = 0

    cand_pred_counts = {0: 0, 1: 0, 2: 0}
    source_pred_counts = {0: 0, 1: 0, 2: 0}
    base_pred_counts = {0: 0, 1: 0, 2: 0}

    for sid in candidate_preds_by_sample:
        cand_p = candidate_preds_by_sample[sid]
        base_p = base_preds_by_sample[sid]
        src_p = source_preds_by_sample[sid]
        lbl = labels_by_sample[sid]
        v_mask = valid_masks_by_sample[sid]

        # Sensory valid and non-ignore mask
        eval_mask = (v_mask == 1) & (lbl != 3)
        valid_cnt = int(np.sum(eval_mask))
        if valid_cnt == 0:
            continue

        total_valid_pixels += valid_cnt

        y = lbl[eval_mask]
        c = cand_p[eval_mask]
        b = base_p[eval_mask]
        s = src_p[eval_mask]

        # Update 3x3 confusion
        for true_c in range(3):
            for pred_c in range(3):
                conf_3x3[true_c, pred_c] += int(np.sum((y == true_c) & (c == pred_c)))

        # Update Transitions from Source to Candidate
        for src_c in range(3):
            for cand_c in range(3):
                trans_src_to_cand[src_c, cand_c] += int(np.sum((s == src_c) & (c == cand_c)))

        # Prediction counts
        for cls in range(3):
            cand_pred_counts[cls] += int(np.sum(c == cls))
            source_pred_counts[cls] += int(np.sum(s == cls))
            base_pred_counts[cls] += int(np.sum(b == cls))

        # Binary change evaluation (class 2 is change)
        y_change = (y == 2)
        c_change = (c == 2)
        b_change = (b == 2)
        s_change = (s == 2)

        s_corr = (s_change == y_change)
        s_wrong = ~s_corr
        b_corr = (b_change == y_change)
        b_wrong = ~b_corr
        c_corr = (c_change == y_change)
        c_wrong = ~c_corr

        total_source_correct_cnt += int(np.sum(s_corr))
        total_source_wrong_cnt += int(np.sum(s_wrong))

        # Harmful base error recovery: source correct, base wrong, candidate correct
        harmful_recovery_cnt += int(np.sum(s_corr & b_wrong & c_corr))
        harmful_base_error_total += int(np.sum(s_corr & b_wrong))

        # Beneficial adaptation retention & loss: source wrong, base correct
        beneficial_retention_cnt += int(np.sum(s_wrong & b_corr & c_corr))
        beneficial_loss_cnt += int(np.sum(s_wrong & b_corr & c_wrong))
        beneficial_adaptation_total += int(np.sum(s_wrong & b_corr))

        # 3-class source transitions
        # Harmful transition: source was 3-class correct (s == y), but candidate is wrong (c != y)
        s_3corr = (s == y)
        c_3wrong = (c != y)
        harmful_trans_src_corr_to_cand_wrong += int(np.sum(s_3corr & c_3wrong))

        # Beneficial transition: source was wrong (s != y), but candidate is correct (c == y)
        s_3wrong = (s != y)
        c_3corr = (c == y)
        beneficial_trans_src_wrong_to_cand_corr += int(np.sum(s_3wrong & c_3corr))

        # Preserved source error: source was wrong (s != y), and candidate reproduced source error (c == s)
        preserved_source_error_cnt += int(np.sum(s_3wrong & (c == s)))

    # Per-class precision, recall, F1 from 3x3 confusion
    class_metrics = {}
    for cls in range(3):
        tp_c = conf_3x3[cls, cls]
        fp_c = int(np.sum(conf_3x3[:, cls])) - tp_c
        fn_c = int(np.sum(conf_3x3[cls, :])) - tp_c
        prec_c = float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) > 0 else 0.0
        rec_c = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0
        f1_c = float(2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        class_metrics[cls] = {
            "tp": int(tp_c),
            "fp": int(fp_c),
            "fn": int(fn_c),
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
        }

    # Separated False Positives and False Negatives by True Class
    # True class 0 (Land), true class 1 (Perm Water), true class 2 (Flood)
    fp_land = int(conf_3x3[0, 2])   # c02
    fp_perm = int(conf_3x3[1, 2])   # c12
    fn_land = int(conf_3x3[2, 0])   # c20
    fn_perm = int(conf_3x3[2, 1])   # c21
    total_fp = fp_land + fp_perm
    total_fn = fn_land + fn_perm

    true_land_total = int(np.sum(conf_3x3[0, :]))
    true_perm_total = int(np.sum(conf_3x3[1, :]))
    true_flood_total = int(np.sum(conf_3x3[2, :]))

    # K4 Erasure & Expansion metrics relative to source predictions
    # Erasure: Source predicted flood (s == 2) but candidate predicted non-flood (c != 2)
    src_flood_total = int(trans_src_to_cand[2, :].sum())
    erasure_cnt = int(trans_src_to_cand[2, 0] + trans_src_to_cand[2, 1])

    # Expansion: Source predicted non-flood (s != 2) but candidate predicted flood (c == 2)
    src_nonflood_total = int(trans_src_to_cand[0, :].sum() + trans_src_to_cand[1, :].sum())
    expansion_cnt = int(trans_src_to_cand[0, 2] + trans_src_to_cand[1, 2])

    return {
        "total_valid_pixels": total_valid_pixels,
        "confusion_3x3": conf_3x3.tolist(),
        "transition_src_to_cand": trans_src_to_cand.tolist(),
        "candidate_prediction_counts": cand_pred_counts,
        "source_prediction_counts": source_pred_counts,
        "base_prediction_counts": base_pred_counts,
        "class_metrics": class_metrics,
        "separated_errors": {
            "fp_land_count": fp_land,
            "fp_land_denominator_true_land": true_land_total,
            "fp_land_rate_rel_true_land": float(fp_land / max(1, true_land_total)),
            "fp_land_share_of_total_fp": float(fp_land / max(1, total_fp)),
            "fp_perm_count": fp_perm,
            "fp_perm_denominator_true_perm": true_perm_total,
            "fp_perm_rate_rel_true_perm": float(fp_perm / max(1, true_perm_total)),
            "fp_perm_share_of_total_fp": float(fp_perm / max(1, total_fp)),
            "fn_land_count": fn_land,
            "fn_land_denominator_true_flood": true_flood_total,
            "fn_land_rate_rel_true_flood": float(fn_land / max(1, true_flood_total)),
            "fn_land_share_of_total_fn": float(fn_land / max(1, total_fn)),
            "fn_perm_count": fn_perm,
            "fn_perm_denominator_true_flood": true_flood_total,
            "fn_perm_rate_rel_true_flood": float(fn_perm / max(1, true_flood_total)),
            "fn_perm_share_of_total_fn": float(fn_perm / max(1, total_fn)),
        },
        "source_transition_dynamics": {
            "harmful_src_corr_to_cand_wrong": harmful_trans_src_corr_to_cand_wrong,
            "beneficial_src_wrong_to_cand_corr": beneficial_trans_src_wrong_to_cand_corr,
            "preserved_source_errors": preserved_source_error_cnt,
            "total_source_correct": total_source_correct_cnt,
            "total_source_wrong": total_source_wrong_cnt,
        },
        "base_consistency_comparison": {
            "harmful_base_error_recovery_count": harmful_recovery_cnt,
            "harmful_base_error_denominator": harmful_base_error_total,
            "harmful_recovery_rate": float(harmful_recovery_cnt / max(1, harmful_base_error_total)),
            "beneficial_adaptation_retention_count": beneficial_retention_cnt,
            "beneficial_adaptation_loss_count": beneficial_loss_cnt,
            "beneficial_adaptation_denominator": beneficial_adaptation_total,
            "beneficial_retention_rate": float(beneficial_retention_cnt / max(1, beneficial_adaptation_total)),
            "beneficial_loss_rate": float(beneficial_loss_cnt / max(1, beneficial_adaptation_total)),
        },
        "erasure_and_expansion": {
            "erasure_count": erasure_cnt,
            "erasure_denominator_src_flood": src_flood_total,
            "erasure_rate": float(erasure_cnt / max(1, src_flood_total)),
            "expansion_count": expansion_cnt,
            "expansion_denominator_src_nonflood": src_nonflood_total,
            "expansion_rate": float(expansion_cnt / max(1, src_nonflood_total)),
        },
    }
