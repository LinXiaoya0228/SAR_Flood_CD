"""ChangeGuard Evidence Consistency Adapter for Week 11A CPU Implementation.

Adheres strictly to docs/protocols/week11_evidence_consistency_protocol_v1.md:
1. Trainable scope: Selected BatchNorm affine parameters (weight, bias) only.
2. Candidate BN policy: Selected BN modules train() with track_running_stats=False,
   running_mean=None, running_var=None. Non-BN modules remain in eval() mode.
3. Frozen source model: Evaluated in eval() mode with original running stats, immutable.
   Eliminates and verifies no whole-object or storage aliasing (R06).
4. Consistency: Identity weak view; horizontal flip on both temporal inputs and mask for strong view;
   spatially inverted strong logits before one-way KL divergence with detached weak targets.
5. Numerical Safeguards: Guard against non-finite losses/gradients/parameters before and after
   stepping, restoring pre-step state on corrupting updates with explicit failure status (R02).
6. Evidence recording:
   - PRE_UPDATE: Normalized Shannon entropy, max confidence, source disagreement rate (R03).
   - POST_UPDATE: Null pairs (pre2, pre2) and (post, post) with primary null score (R03).
7. State preservation: Reference and null probe forwards are strictly state-neutral with
   exception-safe restoration of RNG and registered buffers (R01).
8. Input validation: Squeeze singleton sensory-mask channel without silent broadcasting (N01).
9. Loss representation: Skipped loss is consistently represented as None (N02).
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EvidenceConsistencyAdapter:
    """Minimal Weak-Strong Consistency Adapter with Evidence Recording."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        confidence_threshold: float = 0.90,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.confidence_threshold = float(confidence_threshold)

        # Cache source state dict for exact restoration
        self._source_state_dict = {
            k: v.detach().clone().to(self.device)
            for k, v in self.model.state_dict().items()
        }

        # Cache BN buffer shapes before setting them to None
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

        # R06: Source reference model (immutable reference, independent storage)
        if source_model is not None:
            # Safely eliminate whole-object and shared parameter/buffer storage aliasing
            self.source_model = copy.deepcopy(source_model).to(self.device)
        else:
            self.source_model = copy.deepcopy(self.model).to(self.device)
        self.source_model.eval()
        self.source_model.requires_grad_(False)

        # Eliminate and verify that candidate and source share no storage or modules
        if self.source_model is self.model:
            raise RuntimeError("Source model cannot be identical object to candidate model.")
        cand_ptrs = {p.data_ptr() for p in self.model.parameters()} | {
            b.data_ptr() for b in self.model.buffers() if b is not None
        }
        src_ptrs = {p.data_ptr() for p in self.source_model.parameters()} | {
            b.data_ptr() for b in self.source_model.buffers() if b is not None
        }
        shared = cand_ptrs.intersection(src_ptrs)
        if shared:
            raise RuntimeError(f"Storage aliasing detected between candidate and source model: {shared}")

        # Configure candidate model BN affine parameters
        self._configure_model_bn()
        self._init_optimizer()

        # Step and update counters
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def _configure_model_bn(self) -> None:
        """Configures candidate model: eval mode globally, selected BN modules train() with no tracking."""
        self.model.eval()
        self.model.requires_grad_(False)

        self.bn_param_names: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BN module computes batch statistics on forward pass
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
        """Initializes SGD optimizer over active BatchNorm affine parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BatchNorm parameters found in model!")
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def _is_state_dict_finite(obj: Any) -> bool:
        """Recursively checks if all tensors and floats in a state dict are finite."""
        if isinstance(obj, torch.Tensor):
            return bool(torch.isfinite(obj).all())
        if isinstance(obj, dict):
            return all(EvidenceConsistencyAdapter._is_state_dict_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(EvidenceConsistencyAdapter._is_state_dict_finite(v) for v in obj)
        if isinstance(obj, float):
            return math.isfinite(obj)
        return True

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
        self._configure_model_bn()
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
        """Runs weak-strong consistency step and evidence recording.
        
        Strict Label Separation:
        Raises ValueError immediately if ground-truth labels are present in inputs or kwargs.
        """
        # Guard against label leakage
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        sar_valid_mask = sar_valid_mask.to(self.device)

        # N01: Validate and normalize mask shape without silent broadcasting
        if xA.shape != xB.shape:
            raise ValueError(f"Temporal inputs xA {xA.shape} and xB {xB.shape} must have identical shape.")
        if sar_valid_mask.dim() == 4 and sar_valid_mask.shape[1] == 1:
            sar_valid_mask = sar_valid_mask.squeeze(1)
        elif sar_valid_mask.dim() != 3:
            raise ValueError(
                f"Unsupported sensory mask shape {sar_valid_mask.shape}. "
                f"Expected 3D tensor (B, H, W) or 4D with singleton channel (B, 1, H, W)."
            )
        if sar_valid_mask.shape[0] != xA.shape[0] or sar_valid_mask.shape[1:] != xA.shape[2:]:
            raise ValueError(
                f"Mask shape {sar_valid_mask.shape} does not match spatial dimensions of inputs {xA.shape}."
            )

        V = (sar_valid_mask == 1)
        valid_pixel_count = int(V.sum().item())

        # ---------------------------------------------------------------------
        # 1. Primary Batch Prediction on Weak View (Identity) BEFORE update
        # ---------------------------------------------------------------------
        logits_weak = self.model(xA, xB)
        probs_weak = F.softmax(logits_weak, dim=1)
        pred_weak = probs_weak.argmax(dim=1)
        probs_weak_det = probs_weak.detach()
        probs_weak_max = probs_weak_det.max(dim=1)[0]  # (B, H, W)

        # Primary batch prediction is cached from this pre-update forward pass
        primary_prediction = {
            "logits": logits_weak.detach(),
            "probabilities": probs_weak_det,
            "predicted_class": pred_weak.detach(),
            "predicted_flood_pixels": int((pred_weak == 2).sum().item()),
            "valid_pixels": valid_pixel_count,
        }

        # ---------------------------------------------------------------------
        # 2. Strong View Forward Pass & Realignment
        # ---------------------------------------------------------------------
        # Spatial transform: horizontal flip applied identically to both temporal inputs and mask
        xA_strong = torch.flip(xA, dims=[-1])
        xB_strong = torch.flip(xB, dims=[-1])
        valid_mask_strong = torch.flip(sar_valid_mask, dims=[-1])

        logits_strong = self.model(xA_strong, xB_strong)
        # Realignment: invert the horizontal flip
        logits_strong_aligned = torch.flip(logits_strong, dims=[-1])
        log_probs_strong_aligned = F.log_softmax(logits_strong_aligned, dim=1)

        # ---------------------------------------------------------------------
        # 3. Confidence Masking & One-Way KL Divergence Step
        # ---------------------------------------------------------------------
        conf_mask = (probs_weak_max >= self.confidence_threshold) & (sar_valid_mask == 1)
        selected_pixels = int(conf_mask.sum().item())

        update_applied = False
        action = "skip"  # "update" | "skip" | "failure"
        loss_val: Optional[float] = None
        failure_reason: Optional[str] = None

        if selected_pixels == 0 or valid_pixel_count == 0:
            # N02: Empty selection or zero-valid strictly skips optimizer update with loss=None
            self.num_updates_skipped += 1
            update_applied = False
            action = "skip"
            loss_val = None
        else:
            # One-way KL divergence from detached weak probabilities to strong branch
            kl_pixel = (probs_weak_det * (torch.log(probs_weak_det + 1e-12) - log_probs_strong_aligned)).sum(dim=1)
            loss = (kl_pixel * conf_mask.float()).sum() / conf_mask.sum().clamp(min=1.0)

            # R02: Numerical boundary check 1 - Check finite loss before backward
            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                update_applied = False
                action = "failure"
                failure_reason = f"Non-finite loss encountered: {loss.item()}"
                loss_val = None
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                # R02: Numerical boundary check 2 - Check finite gradients before stepping
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [
                    p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()
                ]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    update_applied = False
                    action = "failure"
                    failure_reason = "Non-finite gradient encountered during backward pass"
                    loss_val = None
                    self.optimizer.zero_grad()
                else:
                    # Save pre-step state in case optimizer step corrupts parameters/optimizer
                    pre_step_params = {
                        n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
                    }
                    pre_step_opt_state = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # R02: Numerical boundary check 3 - Check candidate and optimizer state after stepping
                    nonfinite_params = [
                        n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()
                    ]
                    optimizer_state_finite = self._is_state_dict_finite(self.optimizer.state_dict())

                    if nonfinite_params or not optimizer_state_finite:
                        # Corrupted! Restore pre-step parameters and optimizer state
                        for n, p in self.model.named_parameters():
                            if n in pre_step_params:
                                p.data.copy_(pre_step_params[n])
                        self.optimizer.load_state_dict(pre_step_opt_state)
                        self.optimizer.zero_grad()

                        self.num_updates_failed += 1
                        update_applied = False
                        action = "failure"
                        reasons = []
                        if nonfinite_params:
                            reasons.append(f"non-finite candidate parameters: {nonfinite_params}")
                        if not optimizer_state_finite:
                            reasons.append("non-finite optimizer state (e.g. momentum buffer)")
                        failure_reason = (
                            f"Post-step corruption detected ({'; '.join(reasons)}); restored pre-step state"
                        )
                        loss_val = None
                    else:
                        self.num_updates_applied += 1
                        update_applied = True
                        action = "update"
                        loss_val = float(loss.item())

        update_time_ms = (time.perf_counter() - start_time) * 1000.0

        # ---------------------------------------------------------------------
        # 4. Evidence Recording (Observation, NOT Gating)
        # ---------------------------------------------------------------------
        # Indicator 1: Sensory-valid normalized three-class Shannon entropy (PRE_UPDATE)
        H_pixel = - (probs_weak_det * torch.log(probs_weak_det + 1e-12)).sum(dim=1)
        if valid_pixel_count > 0:
            entropy_norm = float(((H_pixel * V.float()).sum() / (V.sum() * math.log(3))).item())
            max_confidence = float(((probs_weak_max * V.float()).sum() / V.sum()).item())
        else:
            entropy_norm = 0.0
            max_confidence = 0.0

        # Indicator 2: Source/Candidate Disagreement Rate (PRE_UPDATE)
        # R01 & W11C-R07: State-neutral execution: protect RNG and any forward-side buffers.
        # Only touch CUDA RNG if running in an intended CUDA execution context to avoid lazy init.
        is_cuda_context = (self.device.type == "cuda" and torch.cuda.is_available())
        src_rng_state = torch.get_rng_state()
        src_cuda_rng_state = torch.cuda.get_rng_state_all() if is_cuda_context else None
        src_buffers_before = {k: v.clone() for k, v in self.source_model.named_buffers()}
        try:
            with torch.no_grad():
                logits_src = self.source_model(xA, xB)
                probs_src = F.softmax(logits_src, dim=1)
                pred_src = probs_src.argmax(dim=1)
                if valid_pixel_count > 0:
                    disagreement_rate = float(
                        ((pred_src != pred_weak) & V).float().sum().item() / valid_pixel_count
                    )
                else:
                    disagreement_rate = 0.0
        finally:
            torch.set_rng_state(src_rng_state)
            if src_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(src_cuda_rng_state)
            for k, v in self.source_model.named_buffers():
                if k in src_buffers_before:
                    v.copy_(src_buffers_before[k])

        # Indicator 3: Null Pairs Probing (POST_UPDATE, State-Neutral Execution)
        null_metrics = self._evaluate_null_probes_state_neutral(xA, xB, V)

        return {
            "primary_prediction": primary_prediction,
            "sample_ids": list(sample_ids) if sample_ids is not None else [],
            "event_ids": list(event_ids) if event_ids is not None else [],
            "adaptation_metadata": {
                "update_applied": update_applied,
                "action": action,
                "failure_reason": failure_reason,
                "loss": loss_val,
                "selected_pixels": selected_pixels,
                "selected_pixel_fraction": (
                    float(selected_pixels / valid_pixel_count) if valid_pixel_count > 0 else 0.0
                ),
                "valid_pixels": valid_pixel_count,
                "confidence_threshold": self.confidence_threshold,
                "update_time_ms": update_time_ms,
                "step_count": self.step_count,
                "transform_applied": "horizontal_flip_aligned",
            },
            "risk_evidence": {
                "timing_contexts": {
                    "primary_prediction": "PRE_UPDATE",
                    "entropy_norm": "PRE_UPDATE",
                    "max_confidence": "PRE_UPDATE",
                    "source_disagreement_rate": "PRE_UPDATE",
                    "null_probes": "POST_UPDATE",
                },
                "entropy_norm": entropy_norm,
                "max_confidence": max_confidence,
                "source_disagreement_rate": disagreement_rate,
                "source_disagreement_rate_pre_update": disagreement_rate,
                "null_score_primary": null_metrics["null_score_primary"],
                "null_score_primary_post_update": null_metrics["null_score_primary"],
                "null_pre2_delta": null_metrics["null_pre2_delta"],
                "null_post_delta": null_metrics["null_post_delta"],
                "cand_null_pre2_prob": null_metrics["cand_null_pre2_prob"],
                "src_null_pre2_prob": null_metrics["src_null_pre2_prob"],
                "cand_null_post_prob": null_metrics["cand_null_post_prob"],
                "src_null_post_prob": null_metrics["src_null_post_prob"],
                "bn_context": {
                    "candidate_bn": "batch_stats_no_tracking",
                    "source_bn": "frozen_running_stats",
                },
                "erasure_expansion_interface": "DEFERRED",
            },
        }

    def _evaluate_null_probes_state_neutral(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluates (pre2, pre2) and (post, post) null pairs preserving all candidate/source/RNG state."""
        # Preserve RNG state (only touch CUDA RNG if running in an intended CUDA execution context)
        is_cuda_context = (self.device.type == "cuda" and torch.cuda.is_available())
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if is_cuda_context else None

        # Preserve candidate and source buffers
        cand_buffers = {k: v.clone() for k, v in self.model.named_buffers()}
        src_buffers = {k: v.clone() for k, v in self.source_model.named_buffers()}

        valid_count = float(valid_mask.sum().item())
        if valid_count == 0:
            return {
                "null_score_primary": 0.0,
                "null_pre2_delta": 0.0,
                "null_post_delta": 0.0,
                "cand_null_pre2_prob": 0.0,
                "src_null_pre2_prob": 0.0,
                "cand_null_post_prob": 0.0,
                "src_null_post_prob": 0.0,
            }

        try:
            with torch.no_grad():
                # Candidate forward passes on null pairs
                cand_p_pre2 = F.softmax(self.model(xA, xA), dim=1)[:, 2, :, :]
                cand_p_post = F.softmax(self.model(xB, xB), dim=1)[:, 2, :, :]
                cand_pre2_mean = float(((cand_p_pre2 * valid_mask.float()).sum() / valid_count).item())
                cand_post_mean = float(((cand_p_post * valid_mask.float()).sum() / valid_count).item())

                # Source forward passes on null pairs
                src_p_pre2 = F.softmax(self.source_model(xA, xA), dim=1)[:, 2, :, :]
                src_p_post = F.softmax(self.source_model(xB, xB), dim=1)[:, 2, :, :]
                src_pre2_mean = float(((src_p_pre2 * valid_mask.float()).sum() / valid_count).item())
                src_post_mean = float(((src_p_post * valid_mask.float()).sum() / valid_count).item())

                # Compute positive differences
                delta_pre2 = max(0.0, cand_pre2_mean - src_pre2_mean)
                delta_post = max(0.0, cand_post_mean - src_post_mean)
                null_score_primary = 0.5 * (delta_pre2 + delta_post)

            return {
                "null_score_primary": null_score_primary,
                "null_pre2_delta": delta_pre2,
                "null_post_delta": delta_post,
                "cand_null_pre2_prob": cand_pre2_mean,
                "src_null_pre2_prob": src_pre2_mean,
                "cand_null_post_prob": cand_post_mean,
                "src_null_post_prob": src_post_mean,
            }
        finally:
            # R01: Exception-safe restoration of RNG and buffers
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            for k, v in self.model.named_buffers():
                if k in cand_buffers:
                    v.copy_(cand_buffers[k])
            for k, v in self.source_model.named_buffers():
                if k in src_buffers:
                    v.copy_(src_buffers[k])
