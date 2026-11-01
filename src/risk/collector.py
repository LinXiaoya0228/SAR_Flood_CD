"""RiskFeatureCollector: orchestrates non-mutating label-free risk generation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn

from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.base import BaseOnlineAdapter
from src.risk.indicators import (
    IntermediateFeatureHook,
    compute_generic_indicators,
    compute_augmentation_inconsistency,
)
from src.risk.probes import run_null_probes, run_disappearance_probes
from src.risk.scores import compute_predeclared_change_score
from src.risk.schema import validate_risk_feature_dict


class RiskFeatureCollector:
    """Orchestrates non-mutating label-free risk generation for Gate B.
    
    Guarantees:
    1. Zero ground-truth label leakage (strictly verified by schema validator).
    2. Zero mutation of candidate model weights, BN running statistics, optimizer state,
       or adaptation counters during risk probing.
    """

    def __init__(
        self,
        source_model: SNUNetWrapper,
        protocol_hash: str,
        checkpoint_hash: str,
        device: str = "cuda",
    ):
        self.source_model = source_model
        self.protocol_hash = protocol_hash
        self.checkpoint_hash = checkpoint_hash
        self.device = device

        # Cache source BN state dict for drift measurements
        self.source_bn_state: Dict[str, torch.Tensor] = {}
        for name, module in self.source_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    self.source_bn_state[f"{name}.weight"] = module.weight.detach().clone().cpu()
                if module.bias is not None:
                    self.source_bn_state[f"{name}.bias"] = module.bias.detach().clone().cpu()
                if module.running_mean is not None:
                    self.source_bn_state[f"{name}.running_mean"] = module.running_mean.detach().clone().cpu()
                if module.running_var is not None:
                    self.source_bn_state[f"{name}.running_var"] = module.running_var.detach().clone().cpu()

    def _snapshot_candidate_state(self, adapter: BaseOnlineAdapter) -> Dict[str, Any]:
        """Captures a bitwise snapshot of candidate adapter parameters and buffers."""
        return {
            "params": {name: p.detach().clone().cpu() for name, p in adapter.model.named_parameters()},
            "buffers": {name: b.detach().clone().cpu() for name, b in adapter.model.named_buffers()},
            "step_count": adapter.step_count,
        }

    def _verify_candidate_unmodified(self, adapter: BaseOnlineAdapter, snapshot: Dict[str, Any]) -> None:
        """Verifies bitwise identity against pre-probing snapshot."""
        # 1. Verify step count
        if adapter.step_count != snapshot["step_count"]:
            raise RuntimeError(
                f"STATE CORRUPTION: Risk probing mutated adapter step_count: "
                f"{snapshot['step_count']} -> {adapter.step_count}"
            )

        # 2. Verify parameters
        for name, p in adapter.model.named_parameters():
            snap_p = snapshot["params"][name]
            if not torch.equal(p.detach().cpu(), snap_p):
                raise RuntimeError(f"STATE CORRUPTION: Risk probing mutated parameter '{name}'!")

        # 3. Verify running buffers (mean, var, num_batches_tracked)
        for name, b in adapter.model.named_buffers():
            snap_b = snapshot["buffers"][name]
            if not torch.equal(b.detach().cpu(), snap_b):
                raise RuntimeError(f"STATE CORRUPTION: Risk probing mutated buffer '{name}'!")

    def collect_risk_record(
        self,
        candidate_adapter: BaseOnlineAdapter,
        batch_sensory: Dict[str, Any],
        run_id: str,
        event_id: int,
        seed: int,
        batch_idx: int,
    ) -> Dict[str, Any]:
        """Extracts all generic and change-specific risk indicators for the current batch.

        Args:
            candidate_adapter: Adapted model after the current Tent update step.
            batch_sensory: Unlabeled sensory batch dict (xA, xB, pre1, pre2, post, valid_mask, sample_ids).
            run_id: Unique run identifier.
            event_id: Event ID.
            seed: Random stream seed.
            batch_idx: 1-indexed batch counter.

        Returns:
            Validated label-free risk feature record dictionary.
        """
        sample_ids = batch_sensory["sample_ids"]
        xA = batch_sensory["xA"].to(self.device)  # canonical pre2
        xB = batch_sensory["xB"].to(self.device)  # canonical post
        pre1 = batch_sensory["pre1"].to(self.device)
        pre2 = batch_sensory["pre2"].to(self.device)
        post = batch_sensory["post"].to(self.device)
        sar_valid = batch_sensory["valid_mask"].to(self.device)

        # 1. Snapshot candidate state before probing
        snapshot = self._snapshot_candidate_state(candidate_adapter)

        # 2. Attach hooks to capture intermediate features at model.conv_final
        cand_hook = IntermediateFeatureHook(candidate_adapter.model)
        src_hook = IntermediateFeatureHook(self.source_model)

        risk_metrics: Dict[str, float] = {}

        try:
            with torch.no_grad():
                candidate_adapter.model.eval()
                self.source_model.eval()

                # Real pair forward pass (xA, xB)
                cand_logits = candidate_adapter.model(xA, xB)
                src_logits = self.source_model(xA, xB)

                cand_feat = cand_hook.features
                src_feat = src_hook.features

                # Generic indicators
                generic_dict = compute_generic_indicators(
                    candidate_logits=cand_logits,
                    source_logits=src_logits,
                    candidate_features=cand_feat,
                    source_features=src_feat,
                    candidate_model=candidate_adapter.model,
                    source_bn_state=self.source_bn_state,
                    sar_valid_mask=sar_valid,
                )
                risk_metrics.update(generic_dict)

                # Augmentation inconsistency on real pair
                aug_inconsistency = compute_augmentation_inconsistency(
                    candidate_adapter.model, xA, xB, torch.argmax(cand_logits, dim=1), sar_valid_mask=sar_valid
                )
                risk_metrics["augmentation_inconsistency"] = aug_inconsistency

                # Change-specific null probes
                null_dict = run_null_probes(
                    candidate_model=candidate_adapter.model,
                    source_model=self.source_model,
                    pre1=pre1,
                    pre2=pre2,
                    post=post,
                    sar_valid_mask=sar_valid,
                )
                risk_metrics.update(null_dict)

                # Change-disappearance probes
                disappear_dict = run_disappearance_probes(
                    candidate_logits=cand_logits,
                    source_logits=src_logits,
                    candidate_features=cand_feat,
                    source_features=src_feat,
                    candidate_model=candidate_adapter.model,
                    xA=xA,
                    xB=xB,
                    sar_valid_mask=sar_valid,
                )
                risk_metrics.update(disappear_dict)

                # Predeclared composite score
                total_valid_pixels = float(torch.sum(sar_valid > 0).item())
                score_dict = compute_predeclared_change_score(
                    risk_metrics, total_valid_pixels=total_valid_pixels
                )
                risk_metrics.update(score_dict)

        finally:
            cand_hook.remove()
            src_hook.remove()

        # 3. Verify that candidate model state was NOT mutated by probing
        self._verify_candidate_unmodified(candidate_adapter, snapshot)

        # 4. Construct complete record
        record: Dict[str, Any] = {
            "run_id": run_id,
            "event_id": event_id,
            "seed": seed,
            "batch_idx": batch_idx,
            "sample_ids": sample_ids,
            "batch_size": len(sample_ids),
            "protocol_hash": self.protocol_hash,
            "checkpoint_hash": self.checkpoint_hash,
        }
        record.update(risk_metrics)

        # 5. Schema verification (enforces zero GT leakage)
        validate_risk_feature_dict(record)

        return record

