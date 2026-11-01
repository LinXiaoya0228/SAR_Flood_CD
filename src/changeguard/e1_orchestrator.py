"""ChangeGuard E1 Orchestration Harness and Consistency Adapter for Plan V3.

Adheres strictly to docs/MASTER_EXECUTION_PLAN_V3.md and docs/CACHE_AND_MEASUREMENT_CONTRACT.md:
1. Versioned E1 Orchestration: Implements isolated PRE_UPDATE primary decisions and POST_UPDATE diagnostics.
2. Label Timing Enforcement: LabelGuardedBatch prevents access to evaluation labels until predictions and
   actions are durably persisted to disk.
3. Unambiguous Cache Contracts: Every spatial prediction and probability array identifies sample, branch,
   and phase with explicit schema persistence (cache_schema.json).
4. Full Measurement Plumbing: Computes gradient L2 norms before optimizer.step(), BN affine displacement
   against immutable source reference, per-class selection counts, and full-precision summaries without fabrication.
5. Double-Masking Invariant: (MK0_MNA == 1) & (label != 3). Invalid pixels are excluded from all confusion cells including TN.
6. Non-Reconstructibility: Full-precision score summaries (entropy, max confidence, mean prob) computed before float16 storage.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
    compute_per_class_pixel_counts,
)


class LabelGuardedBatch:
    """Batch container strictly enforcing separation of sensory adaptation data from evaluation labels.
    
    Evaluation labels cannot be accessed until predictions and adaptation actions have been
    durably persisted to disk (mark_persisted).
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        event_id: int,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        self.sample_ids = [str(sid) for sid in sample_ids]
        self.event_id = int(event_id)
        self.xA = xA
        self.xB = xB
        self.valid_mask = valid_mask
        self._labels = labels
        self._is_persisted = False

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def get_sensory_inputs(self) -> Dict[str, Any]:
        """Provides sensory inputs strictly free of any ground-truth labels."""
        return {
            "sample_ids": list(self.sample_ids),
            "event_id": self.event_id,
            "xA": self.xA,
            "xB": self.xB,
            "valid_mask": self.valid_mask,
        }

    def mark_persisted(self) -> None:
        """Unlocks label access after predictions and step records are durably written to disk."""
        self._is_persisted = True

    def get_labels(self) -> torch.Tensor:
        """Retrieves evaluation labels only after durable persistence is confirmed."""
        if not self._is_persisted:
            raise PermissionError(
                "CRITICAL LABEL TIMING VIOLATION: Evaluation labels cannot be accessed "
                "before predictions and adaptation actions are durably persisted to disk."
            )
        if self._labels is None:
            raise ValueError(f"No labels were provided for batch in event {self.event_id}.")
        return self._labels


class E1ConsistencyAdapter:
    """Versioned E1 Weak-Strong Consistency Adapter with Complete Measurement Plumbing."""

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

        # Cache source state dict for exact bit-level restoration
        self._source_state_dict = {
            k: v.detach().clone().to(self.device)
            for k, v in self.model.state_dict().items()
        }

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

        # Source reference model (immutable reference, independent storage)
        if source_model is not None:
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
            raise RuntimeError("No trainable BatchNorm parameters found in candidate model.")
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
            return all(E1ConsistencyAdapter._is_state_dict_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(E1ConsistencyAdapter._is_state_dict_finite(v) for v in obj)
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

    def adapt_and_predict_step(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_ids: Sequence[str],
        event_id: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs weak-strong consistency step and comprehensive evidence recording.
        
        Strict Label Separation:
        Raises ValueError immediately if ground-truth labels are present in inputs or kwargs.
        """
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        xA = xA.to(self.device)
        xB = xB.to(self.device)
        valid_mask = valid_mask.to(self.device)

        if xA.shape != xB.shape:
            raise ValueError(f"Temporal inputs xA {xA.shape} and xB {xB.shape} must have identical shape.")
        if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)
        elif valid_mask.dim() != 3:
            raise ValueError(
                f"Unsupported sensory mask shape {valid_mask.shape}. "
                f"Expected 3D tensor (B, H, W) or 4D with singleton channel (B, 1, H, W)."
            )
        if valid_mask.shape[0] != xA.shape[0] or valid_mask.shape[1:] != xA.shape[2:]:
            raise ValueError(
                f"Mask shape {valid_mask.shape} does not match spatial dimensions of inputs {xA.shape}."
            )

        V = (valid_mask == 1)
        valid_pixel_count = int(V.sum().item())

        # ---------------------------------------------------------------------
        # 1. Primary PRE_UPDATE Candidate Forward Pass (Weak / Identity View)
        # ---------------------------------------------------------------------
        logits_cand_pre = self.model(xA, xB)
        probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
        pred_cand_pre = probs_cand_pre.argmax(dim=1)
        probs_cand_pre_det = probs_cand_pre.detach()
        probs_cand_max = probs_cand_pre_det.max(dim=1)[0]  # (B, H, W)

        forward_finite = (
            torch.isfinite(logits_cand_pre).all() and torch.isfinite(probs_cand_pre).all()
        )

        # Full-precision summary statistics over valid pixels
        if valid_pixel_count > 0 and forward_finite:
            H_pixel = -(probs_cand_pre_det * torch.log(probs_cand_pre_det + 1e-12)).sum(dim=1)
            mean_entropy = float(((H_pixel * V.float()).sum() / (V.sum() * math.log(3))).item())
            mean_max_conf = float(((probs_cand_max * V.float()).sum() / V.sum()).item())
            p_c2_pre = probs_cand_pre_det[:, 2]
            mean_prob_c2 = float(((p_c2_pre * V.float()).sum() / V.sum()).item())
        else:
            mean_entropy = None
            mean_max_conf = None
            mean_prob_c2 = None

        # Selection Masking
        if forward_finite:
            selection_mask = (probs_cand_max >= self.confidence_threshold) & V
            selected_pixels = int(selection_mask.sum().item())
            selected_pixel_fraction = (
                float(selected_pixels / valid_pixel_count) if valid_pixel_count > 0 else 0.0
            )

            # Classwise selection breakdown
            c_counts = compute_per_class_pixel_counts(
                predictions=pred_cand_pre,
                selection_mask=selection_mask,
                valid_mask=V,
            )
            selected_c0 = c_counts[0]
            selected_c1 = c_counts[1]
            selected_c2 = c_counts[2]
        else:
            selection_mask = torch.zeros_like(V, dtype=torch.bool)
            selected_pixels = 0
            selected_pixel_fraction = 0.0
            selected_c0 = 0
            selected_c1 = 0
            selected_c2 = 0

        # ---------------------------------------------------------------------
        # 2. Immutable Source Reference PRE_UPDATE Forward Pass (State-Neutral)
        # ---------------------------------------------------------------------
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
                if valid_pixel_count > 0:
                    src_disagreement = float(
                        ((pred_src != pred_cand_pre) & V).float().sum().item() / valid_pixel_count
                    )
                else:
                    src_disagreement = None
        finally:
            torch.set_rng_state(src_rng_state)
            if src_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(src_cuda_rng_state)
            for k, v in self.source_model.named_buffers():
                if k in src_buffers_before:
                    v.copy_(src_buffers_before[k])

        # ---------------------------------------------------------------------
        # 3. Strong View Forward Pass & Realignment
        # ---------------------------------------------------------------------
        xA_strong = torch.flip(xA, dims=[-1])
        xB_strong = torch.flip(xB, dims=[-1])
        logits_strong = self.model(xA_strong, xB_strong)
        logits_strong_aligned = torch.flip(logits_strong, dims=[-1])
        log_probs_strong_aligned = F.log_softmax(logits_strong_aligned, dim=1)

        action = "skip"
        update_applied = False
        loss_val: Optional[float] = None
        grad_norm_l2: Optional[float] = None
        failure_reason: Optional[str] = None

        if not forward_finite:
            self.num_updates_failed += 1
            action = "failure"
            update_applied = False
            failure_reason = "Non-finite logits or probabilities encountered during candidate forward pass"
            loss_val = None
            grad_norm_l2 = None
        elif selected_pixels == 0 or valid_pixel_count == 0:
            self.num_updates_skipped += 1
            action = "skip"
            update_applied = False
            loss_val = None
            grad_norm_l2 = None
        else:
            kl_pixel = (
                probs_cand_pre_det * (torch.log(probs_cand_pre_det + 1e-12) - log_probs_strong_aligned)
            ).sum(dim=1)
            loss = (kl_pixel * selection_mask.float()).sum() / selection_mask.sum().clamp(min=1.0)

            if not torch.isfinite(loss):
                self.num_updates_failed += 1
                action = "failure"
                update_applied = False
                failure_reason = f"Non-finite loss encountered: {loss.item()}"
                loss_val = None
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                # MEASUREMENT POINT: Gradient L2 Norm before optimizer.step()
                grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())

                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [
                    p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()
                ]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    update_applied = False
                    failure_reason = "Non-finite gradient encountered during backward pass"
                    loss_val = None
                    self.optimizer.zero_grad()
                else:
                    pre_step_params = {
                        n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
                    }
                    pre_step_opt_state = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nonfinite_params = [
                        n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()
                    ]
                    optimizer_state_finite = self._is_state_dict_finite(self.optimizer.state_dict())

                    if nonfinite_params or not optimizer_state_finite:
                        # Post-step corruption detected: restore pre-step state
                        for n, p in self.model.named_parameters():
                            if n in pre_step_params:
                                p.data.copy_(pre_step_params[n])
                        self.optimizer.load_state_dict(pre_step_opt_state)
                        self.optimizer.zero_grad()

                        self.num_updates_failed += 1
                        action = "failure"
                        update_applied = False
                        failure_reason = "Post-step corruption detected; restored pre-step state"
                        loss_val = None
                    else:
                        self.num_updates_applied += 1
                        action = "update"
                        update_applied = True
                        loss_val = float(loss.item())

        # MEASUREMENT POINT: BN Affine Displacement from Source Reference
        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # ---------------------------------------------------------------------
        # 5. POST_UPDATE Diagnostic Forward Pass
        # ---------------------------------------------------------------------
        with torch.no_grad():
            logits_cand_post = self.model(xA, xB)
            probs_cand_post = F.softmax(logits_cand_post, dim=1)
            pred_cand_post = probs_cand_post.argmax(dim=1)

        # ---------------------------------------------------------------------
        # 6. State-Neutral Null Probes (POST_UPDATE Diagnostic)
        # ---------------------------------------------------------------------
        null_metrics = self._evaluate_null_probes_state_neutral(xA, xB, V)

        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach(),
                "probs_cand_pre": probs_cand_pre_det,
                "logits_cand_pre": logits_cand_pre.detach(),
                "pred_src": pred_src.detach(),
                "probs_src": probs_src.detach(),
                "valid_pixels": valid_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred_cand_post.detach(),
                "probs_cand_post": probs_cand_post.detach(),
                "logits_cand_post": logits_cand_post.detach(),
            },
            "adaptation_metadata": {
                "action": action,
                "update_applied": update_applied,
                "total_loss": loss_val,
                "consistency_loss": loss_val,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixels_c0": selected_c0,
                "selected_pixels_c1": selected_c1,
                "selected_pixels_c2": selected_c2,
                "selected_pixel_fraction": selected_pixel_fraction,
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
                "confidence_threshold": self.confidence_threshold,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": null_metrics.get("null_score_primary"),
                "null_pre2_delta": null_metrics.get("null_pre2_delta"),
                "null_post_delta": null_metrics.get("null_post_delta"),
                "cand_null_pre2_prob": null_metrics.get("cand_null_pre2_prob"),
                "src_null_pre2_prob": null_metrics.get("src_null_pre2_prob"),
                "cand_null_post_prob": null_metrics.get("cand_null_post_prob"),
                "src_null_post_prob": null_metrics.get("src_null_post_prob"),
            },
        }

    def _evaluate_null_probes_state_neutral(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluates (pre2, pre2) and (post, post) null pairs preserving all candidate/source/RNG state."""
        is_cuda_context = (self.device.type == "cuda" and torch.cuda.is_available())
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if is_cuda_context else None

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
                cand_p_pre2 = F.softmax(self.model(xA, xA), dim=1)[:, 2, :, :]
                cand_p_post = F.softmax(self.model(xB, xB), dim=1)[:, 2, :, :]
                cand_pre2_mean = float(((cand_p_pre2 * valid_mask.float()).sum() / valid_count).item())
                cand_post_mean = float(((cand_p_post * valid_mask.float()).sum() / valid_count).item())

                src_p_pre2 = F.softmax(self.source_model(xA, xA), dim=1)[:, 2, :, :]
                src_p_post = F.softmax(self.source_model(xB, xB), dim=1)[:, 2, :, :]
                src_pre2_mean = float(((src_p_pre2 * valid_mask.float()).sum() / valid_count).item())
                src_post_mean = float(((src_p_post * valid_mask.float()).sum() / valid_count).item())

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
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            for k, v in self.model.named_buffers():
                if k in cand_buffers:
                    v.copy_(cand_buffers[k])
            for k, v in self.source_model.named_buffers():
                if k in src_buffers:
                    v.copy_(src_buffers[k])


class E1OfflineEvaluator:
    """Offline ground-truth evaluator enforcing double-masking and confusion conservation."""

    @staticmethod
    def compute_event_metrics(
        event_id: int,
        predictions_by_sample: Dict[str, np.ndarray],
        labels_by_sample: Dict[str, np.ndarray],
        valid_masks_by_sample: Dict[str, np.ndarray],
        batch_sizes: Sequence[int],
        updates_applied: int,
        updates_skipped: int,
        updates_failed: int,
    ) -> Dict[str, Any]:
        """Calculates 3x3 confusion matrix and binary flood metrics strictly on double-masked pixels.
        
        Double Masking Invariant:
        eval_mask = (valid_mask == 1) & (labels != 3)
        All pixels with valid_mask == 0 or labels == 3 are excluded from all confusion cells including TN.
        """
        all_samples = list(predictions_by_sample.keys())
        total_valid_pixels = 0
        c_matrix = np.zeros((3, 3), dtype=np.int64)

        for sid in all_samples:
            pred = predictions_by_sample[sid]
            label = labels_by_sample[sid]
            v_mask = valid_masks_by_sample[sid]

            # Double-masking
            eval_mask = (v_mask == 1) & (label != 3)
            total_valid_pixels += int(np.sum(eval_mask))

            y_t = label[eval_mask]
            y_p = pred[eval_mask]

            for i in range(3):
                for j in range(3):
                    c_matrix[i, j] += int(np.sum((y_t == i) & (y_p == j)))

        # Invariant check
        matrix_sum = int(np.sum(c_matrix))
        if matrix_sum != total_valid_pixels:
            raise ValueError(
                f"Confusion matrix sum ({matrix_sum}) does not match valid pixels ({total_valid_pixels})."
            )

        c00 = int(c_matrix[0, 0])
        c01 = int(c_matrix[0, 1])
        c02 = int(c_matrix[0, 2])
        c10 = int(c_matrix[1, 0])
        c11 = int(c_matrix[1, 1])
        c12 = int(c_matrix[1, 2])
        c20 = int(c_matrix[2, 0])
        c21 = int(c_matrix[2, 1])
        c22 = int(c_matrix[2, 2])

        tp = c22
        fp = c02 + c12
        fn = c20 + c21
        tn = c00 + c01 + c10 + c11

        cell_sum = tp + fp + fn + tn
        if cell_sum != total_valid_pixels:
            raise ValueError(
                f"Binary confusion cell sum ({cell_sum}) does not match valid pixels ({total_valid_pixels})."
            )

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

        return {
            "event_id": int(event_id),
            "sample_count": len(all_samples),
            "batch_count": len(batch_sizes),
            "batch_sizes": list(batch_sizes),
            "valid_pixels": total_valid_pixels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "c00_land_land": c00,
            "c01_land_perm": c01,
            "c02_land_flood_FP": c02,
            "c10_perm_land": c10,
            "c11_perm_perm": c11,
            "c12_perm_flood_FP": c12,
            "c20_flood_land_FN": c20,
            "c21_flood_perm_FN": c21,
            "c22_flood_flood_TP": c22,
            "updates_applied": int(updates_applied),
            "updates_skipped": int(updates_skipped),
            "updates_failed": int(updates_failed),
        }


class E1Orchestrator:
    """End-to-end ChangeGuard Plan V3 E1 Orchestrator."""

    def __init__(
        self,
        recorder: ExtendedExperimentRecorder,
        adapter: E1ConsistencyAdapter,
        condition_id: str,
        seed: int,
        reset_mode: str = "reset_per_event",
    ):
        self.recorder = recorder
        self.adapter = adapter
        self.condition_id = str(condition_id)
        self.seed = int(seed)
        self.reset_mode = str(reset_mode)

        if self.reset_mode not in {"reset_per_event", "continual", "NOT_APPLICABLE_STATELESS"}:
            raise ValueError(
                f"Unsupported reset_mode '{self.reset_mode}'. "
                f"Must be 'reset_per_event', 'continual', or 'NOT_APPLICABLE_STATELESS'."
            )

    def run_event_stream(
        self,
        event_id: int,
        batches: Sequence[LabelGuardedBatch],
    ) -> Dict[str, Any]:
        """Executes sequential adaptation stream for an event, enforcing exact prediction-label timing."""
        self.recorder.start_event(event_id)

        # Event state reset handling
        if self.reset_mode == "reset_per_event":
            self.adapter.reset_to_source()

        batch_sizes: List[int] = []
        pre_preds_by_sample: Dict[str, np.ndarray] = {}
        labels_by_sample: Dict[str, np.ndarray] = {}
        masks_by_sample: Dict[str, np.ndarray] = {}

        try:
            for step_idx, batch in enumerate(batches, start=1):
                batch_sizes.append(batch.batch_size)

                # 1. Extract purely sensory inputs (strictly label-free)
                inputs = batch.get_sensory_inputs()
                xA = inputs["xA"]
                xB = inputs["xB"]
                valid_mask = inputs["valid_mask"]
                s_ids = inputs["sample_ids"]

                # 2. Run adaptation and dual pre/post prediction step
                step_res = self.adapter.adapt_and_predict_step(
                    xA=xA,
                    xB=xB,
                    valid_mask=valid_mask,
                    sample_ids=s_ids,
                    event_id=event_id,
                )

                primary = step_res["primary_prediction"]
                post = step_res["post_prediction"]
                adapt_meta = step_res["adaptation_metadata"]
                risk = step_res["risk_evidence"]

                pre_preds = primary["pred_cand_pre"]
                post_preds = post["pred_cand_post"]
                src_preds = primary["pred_src"]

                # 3. Cache spatial arrays in recorder (unambiguous keys)
                self.recorder.record_predictions(
                    sample_ids=s_ids,
                    pre_predictions=pre_preds,
                    post_predictions=post_preds,
                    src_predictions=src_preds,
                    prob_c2=primary["probs_cand_pre"][:, 2],
                    prob_phase="PRE_UPDATE",
                    prob_branch="candidate",
                )
                # Cache source Class-2 probabilities
                self.recorder.record_probability_map(
                    sample_ids=s_ids,
                    prob_c2=primary["probs_src"][:, 2],
                    prob_phase="FROZEN_SOURCE",
                    prob_branch="source",
                )
                # Cache post-update candidate Class-2 probabilities
                self.recorder.record_probability_map(
                    sample_ids=s_ids,
                    prob_c2=post["probs_cand_post"][:, 2],
                    prob_phase="POST_UPDATE",
                    prob_branch="candidate",
                )

                # 4. Atomically persist progressive step telemetry to disk
                self.recorder.record_step(
                    step_index=step_idx,
                    event_id=event_id,
                    condition_id=self.condition_id,
                    batch_size=len(s_ids),
                    sample_ids=s_ids,
                    action=adapt_meta["action"],
                    update_applied=adapt_meta["update_applied"],
                    total_loss=adapt_meta["total_loss"],
                    consistency_loss=adapt_meta["consistency_loss"],
                    anchor_loss=adapt_meta["anchor_loss"],
                    gradient_norm_l2=adapt_meta["gradient_norm_l2"],
                    bn_weight_displacement=adapt_meta["bn_weight_displacement"],
                    bn_bias_displacement=adapt_meta["bn_bias_displacement"],
                    valid_pixels=adapt_meta["valid_pixels"],
                    selected_pixels=adapt_meta["selected_pixels"],
                    selected_pixels_c0=adapt_meta["selected_pixels_c0"],
                    selected_pixels_c1=adapt_meta["selected_pixels_c1"],
                    selected_pixels_c2=adapt_meta["selected_pixels_c2"],
                    selected_pixel_fraction=adapt_meta["selected_pixel_fraction"],
                    mean_entropy=risk["mean_entropy"],
                    mean_max_confidence=risk["mean_max_confidence"],
                    mean_prob_c2=risk["mean_prob_c2"],
                    null_score_s_null=risk["null_score_s_null"],
                    batch_time_ms=adapt_meta["batch_time_ms"],
                    peak_vram_mb=None,
                    failure_reason=adapt_meta["failure_reason"],
                )

                # 5. UNLOCK LABELS ONLY AFTER PERSISTENCE CONFIRMED
                batch.mark_persisted()

                # 6. Retrieve labels and store predictions for offline scoring
                batch_labels = batch.get_labels().detach().cpu().numpy()
                pre_preds_np = pre_preds.detach().cpu().numpy()
                v_mask_np = valid_mask.detach().cpu().numpy()

                for i, sid in enumerate(s_ids):
                    pre_preds_by_sample[sid] = pre_preds_np[i]
                    labels_by_sample[sid] = batch_labels[i]
                    masks_by_sample[sid] = v_mask_np[i]

            # 7. Complete event with offline ground-truth evaluation
            event_metrics = E1OfflineEvaluator.compute_event_metrics(
                event_id=event_id,
                predictions_by_sample=pre_preds_by_sample,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=masks_by_sample,
                batch_sizes=batch_sizes,
                updates_applied=self.adapter.num_updates_applied,
                updates_skipped=self.adapter.num_updates_skipped,
                updates_failed=self.adapter.num_updates_failed,
            )

            self.recorder.complete_event(event_id, event_metrics)
            return event_metrics

        except Exception as e:
            # Durable failure recording on unhandled exception
            self.recorder.record_failure(
                error_message=str(e),
                phase="event_stream_adaptation",
                details={"event_id": event_id, "step_count": len(batch_sizes)},
            )
            raise
