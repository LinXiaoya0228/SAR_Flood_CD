"""ChangeGuard Plan V3 E1 Baseline Adapters.

Implements unified, isolated adaptation adapters for the five governing Plan V3 baselines:
1. E1SourceOnlyAdapter: Frozen training BN statistics, zero updates (evaluated once per seed).
2. E1AdaBNAdapter: Candidate BN moving average, zero gradient updates.
3. E1TentAdapter: Reference Tent test-time entropy minimization over BN affine parameters.
4. E1EATAAdapter: Efficient Anti-Forget TTA with V2 Fisher regularization, d_margin=0.05.
5. E1SARAdapter: Sharpness-Aware and Reliable TTA with two-stage SAM and model recovery.

Adheres strictly to docs/MASTER_EXECUTION_PLAN_V3.md and docs/CACHE_AND_MEASUREMENT_CONTRACT.md:
- Pre-update primary predictions separated from post-update diagnostics.
- Strict sensory isolation: zero ground-truth labels accessed during adaptation.
- Full telemetry: gradient L2 norms, BN affine displacements, selected pixel counts, entropy, confidence.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from src.tta.sam import SAM


class BaseE1Adapter:
    """Base class for Plan V3 E1 baseline adapters."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if source_model is not None:
            self.source_model = copy.deepcopy(source_model).to(self.device)
        else:
            self.source_model = copy.deepcopy(self.model).to(self.device)
        self.source_model.eval()
        self.source_model.requires_grad_(False)

        # Cache source state dict for bit-exact resets
        self._source_state_dict = {
            k: v.detach().clone().to(self.device)
            for k, v in self.model.state_dict().items()
        }

        # Cache BN running buffer shapes
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

        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def reset_to_source(self) -> None:
        """Restores model state bit-exactly to source checkpoint."""
        raise NotImplementedError

    def adapt_and_predict_step(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_ids: Sequence[str],
        event_id: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes adaptation step and returns predictions and diagnostics."""
        raise NotImplementedError

    @staticmethod
    def _is_state_dict_finite(obj: Any) -> bool:
        """Recursively checks if all tensors and floats in a state dict are finite."""
        if isinstance(obj, torch.Tensor):
            return bool(torch.isfinite(obj).all())
        if isinstance(obj, dict):
            return all(BaseE1Adapter._is_state_dict_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(BaseE1Adapter._is_state_dict_finite(v) for v in obj)
        if isinstance(obj, float):
            return math.isfinite(obj)
        return True


class E1SourceOnlyAdapter(BaseE1Adapter):
    """Source-Only baseline: frozen training BN statistics, zero updates."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.model.eval()
        self.model.requires_grad_(False)

    def reset_to_source(self) -> None:
        """Restores state dict to source checkpoint."""
        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
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
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1
        self.num_updates_skipped += 1

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        with torch.no_grad():
            logits = self.model(xA, xB)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

        # Risk metrics over valid pixels
        eps = 1e-12
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=1)
        mean_entropy = float((entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        # Class counts on valid pixels
        cls_counts = compute_per_class_pixel_counts(pred, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred.detach(),
                "probs_cand_pre": probs.detach(),
                "logits_cand_pre": logits.detach(),
                "pred_src": pred.detach(),
                "probs_src": probs.detach(),
                "valid_pixels": valid_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred.detach(),
                "probs_cand_post": probs.detach(),
                "logits_cand_post": logits.detach(),
            },
            "adaptation_metadata": {
                "action": "skip",
                "update_applied": False,
                "total_loss": None,
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": None,
                "bn_weight_displacement": 0.0,
                "bn_bias_displacement": 0.0,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": valid_pixel_count,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": 1.0,
                "batch_time_ms": batch_time_ms,
                "failure_reason": None,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": 0.0,
                "null_score_s_null": None,
            },
        }


class E1AdaBNAdapter(BaseE1Adapter):
    """AdaBN baseline: updates BN running statistics via moving average with zero gradient updates."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        momentum: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.momentum = float(momentum) if momentum is not None else None
        self._configure_model_adabn()

    def _configure_model_adabn(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = True
                if self.momentum is not None:
                    module.momentum = self.momentum

    def reset_to_source(self) -> None:
        """Restores model state dict and resets BN running buffers to source checkpoint."""
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
        self._configure_model_adabn()
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
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1
        self.num_updates_skipped += 1

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        with torch.no_grad():
            # Candidate forward (in train mode for BN -> updates running stats)
            logits_cand = self.model(xA, xB)
            probs_cand = F.softmax(logits_cand, dim=1)
            pred_cand = probs_cand.argmax(dim=1)

            # Source reference forward (frozen eval)
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # Risk metrics over valid pixels
        eps = 1e-12
        log_probs = torch.log(probs_cand + eps)
        entropy = -torch.sum(probs_cand * log_probs, dim=1)
        mean_entropy = float((entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs_cand, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_cand[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        diff_pixels = int(((pred_cand != pred_src).float() * V).sum().item())
        src_disagreement = float(diff_pixels / max(valid_pixel_count, 1))

        cls_counts = compute_per_class_pixel_counts(pred_cand, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand.detach(),
                "probs_cand_pre": probs_cand.detach(),
                "logits_cand_pre": logits_cand.detach(),
                "pred_src": pred_src.detach(),
                "probs_src": probs_src.detach(),
                "valid_pixels": valid_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred_cand.detach(),
                "probs_cand_post": probs_cand.detach(),
                "logits_cand_post": logits_cand.detach(),
            },
            "adaptation_metadata": {
                "action": "update",
                "update_applied": True,
                "total_loss": None,
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": None,
                "bn_weight_displacement": 0.0,
                "bn_bias_displacement": 0.0,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": valid_pixel_count,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": 1.0,
                "batch_time_ms": batch_time_ms,
                "failure_reason": None,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }


class E1TentAdapter(BaseE1Adapter):
    """Tent baseline: reference test-time entropy minimization over BN affine parameters."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.learning_rate = float(learning_rate)
        self.betas = tuple(betas)
        self.weight_decay = float(weight_decay)

        self._configure_model_tent()
        self._init_optimizer()

    def _configure_model_tent(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    def _init_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BN affine parameters found for Tent.")
        self.optimizer = optim.Adam(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def reset_to_source(self) -> None:
        """Restores model state bit-exactly to source checkpoint and resets optimizer."""
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
        self._configure_model_tent()
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
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        # 1. Source reference forward (frozen eval)
        with torch.no_grad():
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # 2. PRE_UPDATE Candidate Forward
        logits_cand_pre = self.model(xA, xB)
        probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
        pred_cand_pre = probs_cand_pre.argmax(dim=1)

        # 3. Compute spatial entropy loss over valid sensory pixels
        eps = 1e-12
        log_probs = torch.log(probs_cand_pre + eps)
        entropy = -torch.sum(probs_cand_pre * log_probs, dim=1)
        mean_entropy = float((entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs_cand_pre, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_cand_pre[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        diff_pixels = int(((pred_cand_pre != pred_src).float() * V).sum().item())
        src_disagreement = float(diff_pixels / max(valid_pixel_count, 1))

        loss = (entropy * V).sum() / (V.sum() + eps)

        # 4. Backward Pass and Optimizer Step
        grad_norm_l2 = None
        action = "update"
        update_applied = False
        failure_reason = None
        loss_val = None

        if not torch.isfinite(loss):
            self.num_updates_failed += 1
            action = "failure"
            failure_reason = f"Non-finite loss: {loss.item()}"
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            nonfinite_grads = [p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()]

            if nonfinite_grads:
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = "Non-finite gradient encountered"
                self.optimizer.zero_grad()
            else:
                pre_step_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                pre_step_opt = copy.deepcopy(self.optimizer.state_dict())

                self.optimizer.step()
                self.optimizer.zero_grad()

                nonfinite_params = [n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
                opt_finite = self._is_state_dict_finite(self.optimizer.state_dict())

                if nonfinite_params or not opt_finite:
                    # Rollback
                    for n, p in self.model.named_parameters():
                        if n in pre_step_params:
                            p.data.copy_(pre_step_params[n])
                    self.optimizer.load_state_dict(pre_step_opt)
                    self.optimizer.zero_grad()
                    self.num_updates_failed += 1
                    action = "rollback"
                    failure_reason = "Post-step parameter corruption; rolled back"
                else:
                    self.num_updates_applied += 1
                    update_applied = True
                    loss_val = float(loss.item())

        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # 5. Diagnostic POST_UPDATE forward pass
        with torch.no_grad():
            logits_cand_post = self.model(xA, xB)
            probs_cand_post = F.softmax(logits_cand_post, dim=1)
            pred_cand_post = probs_cand_post.argmax(dim=1)

        cls_counts = compute_per_class_pixel_counts(pred_cand_pre, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach(),
                "probs_cand_pre": probs_cand_pre.detach(),
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
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": valid_pixel_count,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": 1.0,
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }


class E1EATAAdapter(BaseE1Adapter):
    """EATA baseline: reliability filtering, redundancy filtering (d_margin=0.05), and Fisher regularization."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2000.0,
        e_margin: Optional[float] = None,
        d_margin: float = 0.05,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.fishers = fishers
        self.fisher_alpha = float(fisher_alpha)
        # Default upstream reliability threshold: E_0 = 0.40 * ln(3) ~ 0.4394449
        self.e_margin = float(e_margin) if e_margin is not None else float(0.40 * math.log(3.0))
        self.d_margin = float(d_margin)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        self.current_model_probs: Optional[torch.Tensor] = None
        self._configure_model_eata()
        self._init_optimizer()

    def _configure_model_eata(self) -> None:
        self.model.train()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def _init_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BN affine parameters found for EATA.")
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def reset_to_source(self) -> None:
        """Restores model state bit-exactly to source checkpoint and resets optimizer and EMA."""
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
        self._configure_model_eata()
        self._init_optimizer()
        self.current_model_probs = None
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
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1
        batch_size = xA.size(0)

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        # 1. Source reference forward
        with torch.no_grad():
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # 2. PRE_UPDATE Candidate Forward
        self.model.train()
        logits_cand_pre = self.model(xA, xB)
        probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
        pred_cand_pre = probs_cand_pre.argmax(dim=1)

        eps = 1e-12
        log_probs = torch.log(probs_cand_pre + eps)
        pixel_entropy = -torch.sum(probs_cand_pre * log_probs, dim=1)  # (B, H, W)

        # 3. Patch-level entropy and class probabilities over valid sensory pixels
        patch_entropies = []
        patch_probs = []
        for b in range(batch_size):
            m_b = V[b]
            v_b = torch.sum(m_b)
            if v_b > 0:
                h_b = torch.sum(pixel_entropy[b] * m_b) / v_b
                p_b = torch.sum(probs_cand_pre[b] * m_b.unsqueeze(0), dim=(1, 2)) / v_b
            else:
                h_b = None
                p_b = None
            patch_entropies.append(h_b)
            patch_probs.append(p_b)

        # 4. Filter 1: Reliability (H_b < E_0)
        reliable_ids = [
            b for b, h in enumerate(patch_entropies)
            if h is not None and float(h.item()) < self.e_margin
        ]

        # 5. Filter 2: Redundancy (cosine similarity against running EMA probability vector)
        active_ids = []
        if self.current_model_probs is None:
            active_ids = list(reliable_ids)
        else:
            target_ema = self.current_model_probs.to(self.device).unsqueeze(0)
            for b in reliable_ids:
                p_b_vec = patch_probs[b].unsqueeze(0)
                cos_sim = F.cosine_similarity(target_ema, p_b_vec, dim=1).item()
                if abs(cos_sim) < self.d_margin:
                    active_ids.append(b)

        # Update running EMA with active samples
        if active_ids:
            with torch.no_grad():
                new_probs_mean = torch.stack([patch_probs[b] for b in active_ids]).mean(dim=0)
                if self.current_model_probs is None:
                    self.current_model_probs = new_probs_mean.detach().clone()
                else:
                    self.current_model_probs = (
                        0.9 * self.current_model_probs + 0.1 * new_probs_mean
                    ).detach().clone()

        # 6. Adaptation Update
        action = "update"
        update_applied = False
        failure_reason = None
        loss_val = None
        grad_norm_l2 = None

        if not active_ids:
            self.num_updates_skipped += 1
            action = "skip"
        else:
            weighted_losses = []
            for b in active_ids:
                h_b = patch_entropies[b]
                coeff = 1.0 / torch.exp(h_b.detach() - self.e_margin)
                weighted_losses.append(h_b * coeff)
            entropy_loss = torch.stack(weighted_losses).mean()

            fisher_loss = torch.tensor(0.0, device=self.device)
            if self.fishers is not None:
                for name, param in self.model.named_parameters():
                    if name in self.fishers:
                        f_diag, p_init = self.fishers[name]
                        f_diag = f_diag.to(self.device)
                        p_init = p_init.to(self.device)
                        fisher_loss = fisher_loss + (f_diag * (param - p_init) ** 2).sum()

            total_loss = entropy_loss + (self.fisher_alpha / 2.0) * fisher_loss

            if not torch.isfinite(total_loss):
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = f"Non-finite total loss: {total_loss.item()}"
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())

                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    failure_reason = "Non-finite gradient encountered"
                    self.optimizer.zero_grad()
                else:
                    pre_step_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
                    pre_step_opt = copy.deepcopy(self.optimizer.state_dict())

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nonfinite_params = [n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
                    opt_finite = self._is_state_dict_finite(self.optimizer.state_dict())

                    if nonfinite_params or not opt_finite:
                        for n, p in self.model.named_parameters():
                            if n in pre_step_params:
                                p.data.copy_(pre_step_params[n])
                        self.optimizer.load_state_dict(pre_step_opt)
                        self.optimizer.zero_grad()
                        self.num_updates_failed += 1
                        action = "rollback"
                        failure_reason = "Post-step corruption; rolled back"
                    else:
                        self.num_updates_applied += 1
                        update_applied = True
                        loss_val = float(total_loss.item())

        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # 7. Diagnostic POST_UPDATE forward pass
        with torch.no_grad():
            logits_cand_post = self.model(xA, xB)
            probs_cand_post = F.softmax(logits_cand_post, dim=1)
            pred_cand_post = probs_cand_post.argmax(dim=1)

        mean_entropy = float((pixel_entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs_cand_pre, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_cand_pre[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        diff_pixels = int(((pred_cand_pre != pred_src).float() * V).sum().item())
        src_disagreement = float(diff_pixels / max(valid_pixel_count, 1))

        # Selected pixels: active patch valid pixels
        selected_pixels = 0
        for b in active_ids:
            selected_pixels += int(V[b].sum().item())
        selected_fraction = float(selected_pixels / max(valid_pixel_count, 1))

        cls_counts = compute_per_class_pixel_counts(pred_cand_pre, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach(),
                "probs_cand_pre": probs_cand_pre.detach(),
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
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": selected_fraction,
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }


class E1SARAdapter(BaseE1Adapter):
    """SAR baseline: two-stage SAM optimization with reliability filter and model recovery."""

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        e_margin: Optional[float] = None,
        sam_rho: float = 0.05,
        recovery_threshold: Optional[float] = None,
        learning_rate: float = 1.25e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self.e_margin = float(e_margin) if e_margin is not None else float(0.40 * math.log(3.0))
        self.sam_rho = float(sam_rho)
        # Upstream formula: e_m = 0.2 * ln(3) / ln(1000) ~ 0.03180808
        self.recovery_threshold = (
            float(recovery_threshold)
            if recovery_threshold is not None
            else float(0.2 * math.log(3.0) / math.log(1000.0))
        )
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        self.loss_ema: Optional[float] = None
        self._configure_model_sar()
        self._init_optimizer()

    def _configure_model_sar(self) -> None:
        self.model.train()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    def _init_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable BN affine parameters found for SAR.")
        self.optimizer = SAM(
            params,
            base_optimizer=optim.SGD,
            rho=self.sam_rho,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def reset_to_source(self) -> None:
        """Restores model state bit-exactly to source checkpoint and resets SAM optimizer."""
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
        self._configure_model_sar()
        self._init_optimizer()
        self.loss_ema = None
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
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1
        batch_size = xA.size(0)

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        # 1. Source reference forward
        with torch.no_grad():
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # 2. PRE_UPDATE Candidate Forward
        self.model.train()
        logits_cand_pre = self.model(xA, xB)
        probs_cand_pre = F.softmax(logits_cand_pre, dim=1)
        pred_cand_pre = probs_cand_pre.argmax(dim=1)

        eps = 1e-12
        log_probs = torch.log(probs_cand_pre + eps)
        pixel_entropy = -torch.sum(probs_cand_pre * log_probs, dim=1)

        # 3. Patch-level entropy over valid sensory pixels
        patch_entropies = []
        for b in range(batch_size):
            m_b = V[b]
            v_b = torch.sum(m_b)
            if v_b > 0:
                h_b = torch.sum(pixel_entropy[b] * m_b) / v_b
            else:
                h_b = None
            patch_entropies.append(h_b)

        reliable_ids = [
            b for b, h in enumerate(patch_entropies)
            if h is not None and float(h.item()) < self.e_margin
        ]

        # 4. Two-stage SAM Optimization
        action = "update"
        update_applied = False
        failure_reason = None
        loss_val = None
        grad_norm_l2 = None

        if not reliable_ids:
            self.num_updates_skipped += 1
            action = "skip"
        else:
            # Stage 1: evaluate reliable loss and perturb weights
            h_rel = torch.stack([patch_entropies[b] for b in reliable_ids]).mean()
            if not torch.isfinite(h_rel):
                self.num_updates_failed += 1
                action = "failure"
                failure_reason = f"Non-finite stage 1 entropy: {h_rel.item()}"
            else:
                h_rel.backward()
                grad_norm_l2 = compute_gradient_l2_norm(self.model.parameters())

                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                nonfinite_grads = [p for p in trainable_params if p.grad is not None and not torch.isfinite(p.grad).all()]

                if nonfinite_grads:
                    self.num_updates_failed += 1
                    action = "failure"
                    failure_reason = "Non-finite gradient in SAM stage 1"
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.first_step(zero_grad=True)

                    # Stage 2: evaluate perturbed model on re-filtered reliable samples
                    logits_pert = self.model(xA, xB)
                    probs_pert = F.softmax(logits_pert, dim=1)
                    log_probs_pert = torch.log(probs_pert + eps)
                    entropy_pert = -torch.sum(probs_pert * log_probs_pert, dim=1)

                    patch_entropies_stage2 = []
                    for b in range(batch_size):
                        m_b = V[b]
                        v_b = torch.sum(m_b)
                        if v_b > 0:
                            h_b2 = torch.sum(entropy_pert[b] * m_b) / v_b
                        else:
                            h_b2 = None
                        patch_entropies_stage2.append(h_b2)

                    reliable_ids_stage2 = [
                        b for b, h in enumerate(patch_entropies_stage2)
                        if h is not None and float(h.item()) < self.e_margin
                    ]

                    if not reliable_ids_stage2:
                        # Abort stage 2 safely: restore perturbation without optimizer step
                        self.optimizer.restore_perturbation()
                        self.optimizer.zero_grad()
                        self.num_updates_skipped += 1
                        action = "skip"
                    else:
                        h_stage2 = torch.stack([patch_entropies_stage2[b] for b in reliable_ids_stage2]).mean()
                        if not torch.isfinite(h_stage2):
                            self.optimizer.restore_perturbation()
                            self.optimizer.zero_grad()
                            self.num_updates_failed += 1
                            action = "failure"
                            failure_reason = f"Non-finite stage 2 entropy: {h_stage2.item()}"
                        else:
                            h_stage2.backward()
                            self.optimizer.second_step(zero_grad=True)

                            loss_val = float(h_rel.item())
                            self.num_updates_applied += 1
                            update_applied = True

                            # Update loss EMA and check model recovery
                            if self.loss_ema is None:
                                self.loss_ema = loss_val
                            else:
                                self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss_val

                            if self.loss_ema < self.recovery_threshold:
                                # Model recovery triggered: reset model to source checkpoint
                                for n, p in self.model.named_parameters():
                                    if n in self._source_state_dict:
                                        p.data.copy_(self._source_state_dict[n])
                                self._configure_model_sar()
                                self.loss_ema = None
                                action = "rollback"
                                update_applied = False
                                failure_reason = "Model recovery triggered; reset to source checkpoint"

        bn_w_disp, bn_b_disp = compute_bn_affine_displacement(self.model, self._source_state_dict)

        # 5. Diagnostic POST_UPDATE forward pass
        with torch.no_grad():
            logits_cand_post = self.model(xA, xB)
            probs_cand_post = F.softmax(logits_cand_post, dim=1)
            pred_cand_post = probs_cand_post.argmax(dim=1)

        mean_entropy = float((pixel_entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs_cand_pre, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_cand_pre[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        diff_pixels = int(((pred_cand_pre != pred_src).float() * V).sum().item())
        src_disagreement = float(diff_pixels / max(valid_pixel_count, 1))

        selected_pixels = sum(int(V[b].sum().item()) for b in reliable_ids)
        selected_fraction = float(selected_pixels / max(valid_pixel_count, 1))

        cls_counts = compute_per_class_pixel_counts(pred_cand_pre, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand_pre.detach(),
                "probs_cand_pre": probs_cand_pre.detach(),
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
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": grad_norm_l2,
                "bn_weight_displacement": bn_w_disp,
                "bn_bias_displacement": bn_b_disp,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": selected_pixels,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": selected_fraction,
                "batch_time_ms": batch_time_ms,
                "failure_reason": failure_reason,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }



class E1MatchedBNNoUpdateAdapter(BaseE1Adapter):
    """Matched BN no-update condition (matched_bn_no_update_pre_update_v1).
    
    Candidate forward uses current-batch statistics for all BatchNorm layers,
    with ZERO optimizer updates, ZERO backward passes, requires_grad=False on all parameters,
    and ZERO running-statistics accumulation (track_running_stats=False, running buffers None).
    """

    def __init__(
        self,
        model: nn.Module,
        source_model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__(model=model, source_model=source_model, device=device)
        self._configure_model_bn_no_update()

        # Cache initial parameter clones for strict immutability verification
        self._initial_param_clones = {
            name: param.detach().clone().cpu() for name, param in self.model.named_parameters()
        }

    def _configure_model_bn_no_update(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        self.bn_module_names: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if hasattr(module, "num_batches_tracked"):
                    module.num_batches_tracked = None
                self.bn_module_names.append(name)

    def reset_to_source(self) -> None:
        """Restores model state dict and reconfigures BN no-update context."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in self._bn_buffer_shapes:
                    m_shape, v_shape = self._bn_buffer_shapes[name]
                    module.register_buffer("running_mean", torch.zeros(m_shape, device=self.device))
                    module.register_buffer("running_var", torch.ones(v_shape, device=self.device))
                    module.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=self.device))
        self.model.load_state_dict(self._source_state_dict, strict=True)
        self.model.to(self.device)
        self._configure_model_bn_no_update()
        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def verify_immutability(self) -> Tuple[bool, float, float]:
        """Verifies bit-exact parameter preservation and zero BN affine displacement.
        
        Returns:
            (is_exact_match, max_weight_displacement, max_bias_displacement)
        """
        max_w_disp = 0.0
        max_b_disp = 0.0
        for name, param in self.model.named_parameters():
            init = self._initial_param_clones[name].to(param.device)
            if not torch.equal(param, init):
                diff = float((param - init).abs().max().item())
                if "weight" in name:
                    max_w_disp = max(max_w_disp, diff)
                else:
                    max_b_disp = max(max_b_disp, diff)
                return False, max_w_disp, max_b_disp
        return True, 0.0, 0.0

    def adapt_and_predict_step(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_ids: Sequence[str],
        event_id: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        forbidden_keys = {"label", "labels", "target", "targets", "gt", "ground_truth", "valid"}
        leaked = forbidden_keys.intersection(kwargs.keys())
        if leaked:
            raise ValueError(f"CRITICAL LABEL LEAKAGE: Forbidden ground-truth key(s) detected: {leaked}")

        start_time = time.perf_counter()
        self.step_count += 1

        V = (valid_mask == 1).float()
        valid_pixel_count = int(V.sum().item())

        with torch.no_grad():
            # Candidate forward: uses current-batch statistics (train mode, track_running_stats=False)
            logits_cand = self.model(xA, xB)
            probs_cand = F.softmax(logits_cand, dim=1)
            pred_cand = probs_cand.argmax(dim=1)

            # Source forward: uses frozen source running stats (eval mode, frozen)
            logits_src = self.source_model(xA, xB)
            probs_src = F.softmax(logits_src, dim=1)
            pred_src = probs_src.argmax(dim=1)

        # Risk metrics over valid pixels
        eps = 1e-12
        log_probs = torch.log(probs_cand + eps)
        entropy = -torch.sum(probs_cand * log_probs, dim=1)
        mean_entropy = float((entropy * V).sum().item() / max(valid_pixel_count, 1))
        max_conf, _ = torch.max(probs_cand, dim=1)
        mean_max_conf = float((max_conf * V).sum().item() / max(valid_pixel_count, 1))
        prob_c2 = probs_cand[:, 2]
        mean_prob_c2 = float((prob_c2 * V).sum().item() / max(valid_pixel_count, 1))

        diff_pixels = int(((pred_cand != pred_src).float() * V).sum().item())
        src_disagreement = float(diff_pixels / max(valid_pixel_count, 1))

        # Class counts on valid pixels
        cls_counts = compute_per_class_pixel_counts(pred_cand, valid_mask=valid_mask)
        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "primary_prediction": {
                "pred_cand_pre": pred_cand.detach(),
                "probs_cand_pre": probs_cand.detach(),
                "logits_cand_pre": logits_cand.detach(),
                "pred_src": pred_src.detach(),
                "probs_src": probs_src.detach(),
                "valid_pixels": valid_pixel_count,
            },
            "post_prediction": {
                "pred_cand_post": pred_cand.detach(),
                "probs_cand_post": probs_cand.detach(),
                "logits_cand_post": logits_cand.detach(),
            },
            "adaptation_metadata": {
                "action": "skip",
                "update_applied": False,
                "total_loss": None,
                "consistency_loss": None,
                "anchor_loss": None,
                "gradient_norm_l2": None,
                "bn_weight_displacement": 0.0,
                "bn_bias_displacement": 0.0,
                "valid_pixels": valid_pixel_count,
                "selected_pixels": valid_pixel_count,
                "selected_pixels_c0": cls_counts[0],
                "selected_pixels_c1": cls_counts[1],
                "selected_pixels_c2": cls_counts[2],
                "selected_pixel_fraction": 1.0,
                "batch_time_ms": batch_time_ms,
                "failure_reason": None,
                "step_count": self.step_count,
            },
            "risk_evidence": {
                "mean_entropy": mean_entropy,
                "mean_max_confidence": mean_max_conf,
                "mean_prob_c2": mean_prob_c2,
                "source_disagreement_rate": src_disagreement,
                "null_score_s_null": None,
            },
        }
