"""Training-only flood-aware class weighting and loss objective (Component M2)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
import torch
import torch.nn as nn
from src.evaluation.losses import SoftDiceLoss
from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
)


def compute_training_class_weights(
    train_samples_metadata: List[Dict[str, Any]],
    train_event_ids: List[int],
    epsilon: float = 1e-12,
    clip_min: float = 0.5,
    clip_max: float = 2.0,
    source_inventory_hash: str = "",
    dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """Computes deterministic flood-aware class weights strictly from training events.

    Invariants:
    1. Zero access to evaluation events or forbidden target events.
    2. Valid pixels: (valid_mask == 1) & (label != 3).
    3. Formula:
       p_c = valid_c / total_valid
       raw_w_c = 1 / sqrt(max(p_c, epsilon))
       w_c = raw_w_c / mean(raw_w)
       w_c = clip(w_c, clip_min, clip_max)
       final_w_c = w_c / mean(w_c)
    """
    train_events_set = set(train_event_ids)
    forbidden_intersect = train_events_set.intersection(FORBIDDEN_TARGET_EVENTS)
    if forbidden_intersect:
        raise ContractViolationError(
            f"FATAL: Attempted to compute class weights using forbidden events: {forbidden_intersect}"
        )
    if not train_events_set.issubset(AUTHORITATIVE_SOURCE_EVENTS):
        raise ContractViolationError(
            f"Training events contain unauthorized IDs: {train_events_set - AUTHORITATIVE_SOURCE_EVENTS}"
        )

    per_event_counts: Dict[int, Dict[str, int]] = {
        eid: {"class_0": 0, "class_1": 0, "class_2": 0, "total_valid": 0}
        for eid in sorted(train_event_ids)
    }

    counted_sample_ids: List[str] = []

    for item in train_samples_metadata:
        eid = int(item["event_id"])
        if eid not in train_events_set:
            raise ContractViolationError(
                f"Sample event_id {eid} not in configured training allowlist {train_events_set}!"
            )

        counted_sample_ids.append(str(item["sample_id"]))
        if "label_path" in item and "valid_mask_path" in item:
            lbl_path = Path(item["label_path"])
            mna_path = Path(item["valid_mask_path"])
        elif dataset is not None:
            paths = dataset.get_sample_paths(item)
            lbl_path = paths["label_path"]
            mna_path = paths["valid_mask_path"]
        else:
            raise KeyError("Metadata item lacks 'label_path'/'valid_mask_path' and dataset was not provided.")

        with rasterio.open(lbl_path) as src_l, rasterio.open(mna_path) as src_m:
            lbl = src_l.read(1).astype(np.int64)
            mna = src_m.read(1).astype(np.uint8)

        # Invariant: valid pixels where mask == 1 and label != 3
        valid = (mna == 1) & (lbl != 3)
        c0 = int(((lbl == 0) & valid).sum())
        c1 = int(((lbl == 1) & valid).sum())
        c2 = int(((lbl == 2) & valid).sum())
        v_total = c0 + c1 + c2

        per_event_counts[eid]["class_0"] += c0
        per_event_counts[eid]["class_1"] += c1
        per_event_counts[eid]["class_2"] += c2
        per_event_counts[eid]["total_valid"] += v_total

    # Aggregate across training events
    agg_c0 = sum(per_event_counts[eid]["class_0"] for eid in train_events_set)
    agg_c1 = sum(per_event_counts[eid]["class_1"] for eid in train_events_set)
    agg_c2 = sum(per_event_counts[eid]["class_2"] for eid in train_events_set)
    total_valid = agg_c0 + agg_c1 + agg_c2

    if total_valid == 0:
        raise ContractViolationError("FATAL: Zero valid pixels found across training events!")

    counts_arr = np.array([agg_c0, agg_c1, agg_c2], dtype=np.float64)
    if np.any(counts_arr == 0):
        raise ContractViolationError(f"FATAL: One or more classes has zero valid pixels: {counts_arr}")

    # Proportions
    p = counts_arr / float(total_valid)

    # Raw weights
    raw_w = 1.0 / np.sqrt(np.maximum(p, float(epsilon)))
    norm_w = raw_w / float(np.mean(raw_w))

    # Clipping to [clip_min, clip_max]
    clipped_w = np.clip(norm_w, float(clip_min), float(clip_max))

    # Final normalization to mean 1.0
    final_w = clipped_w / float(np.mean(clipped_w))

    # Compute code hash of this implementation
    with open(__file__, "rb") as fh:
        code_hash = hashlib.sha256(fh.read()).hexdigest()

    samples_digest = hashlib.sha256(",".join(counted_sample_ids).encode("utf-8")).hexdigest()

    result = {
        "training_event_ids": sorted(list(train_event_ids)),
        "total_samples_counted": len(counted_sample_ids),
        "samples_digest": samples_digest,
        "per_event_class_counts": per_event_counts,
        "aggregate_counts": {
            "class_0": int(agg_c0),
            "class_1": int(agg_c1),
            "class_2": int(agg_c2),
            "total_valid": int(total_valid),
        },
        "class_proportions": {
            "class_0": float(round(p[0], 6)),
            "class_1": float(round(p[1], 6)),
            "class_2": float(round(p[2], 6)),
        },
        "raw_weights": {
            "class_0": float(round(raw_w[0], 6)),
            "class_1": float(round(raw_w[1], 6)),
            "class_2": float(round(raw_w[2], 6)),
        },
        "clipped_normalized_weights": {
            "class_0": float(round(final_w[0], 6)),
            "class_1": float(round(final_w[1], 6)),
            "class_2": float(round(final_w[2], 6)),
        },
        "final_weights_list": [float(round(final_w[c], 6)) for c in range(3)],
        "counting_code_hash": code_hash,
        "source_inventory_hash": source_inventory_hash,
    }

    result_json = json.dumps(result, indent=2) + "\n"
    result["artifact_sha256"] = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
    return result


class RobustSourceLoss(nn.Module):
    """Composite loss function supporting R0 unit weights and M2 flood-aware class weights.

    L_total = 0.5 * L_CE(w_c) + 0.5 * L_Dice(macro)
    """

    def __init__(
        self,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        classes_in_dice: Tuple[int, ...] = (0, 1, 2),
        ignore_index: int = 3,
        smooth_epsilon: float = 1.0,
        squared_denominator: bool = True,
        reduction: str = "batch",
    ):
        super().__init__()
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.ignore_index = int(ignore_index)

        if class_weights is not None:
            if isinstance(class_weights, list):
                w_tensor = torch.tensor(class_weights, dtype=torch.float32)
            else:
                w_tensor = class_weights.float()
            self.register_buffer("class_weights", w_tensor)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.dice = SoftDiceLoss(
            classes=list(classes_in_dice),
            ignore_index=self.ignore_index,
            eps=smooth_epsilon,
            squared_denominator=squared_denominator,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            loss_total = logits.sum() * 0.0
            if return_components:
                return loss_total, {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0}
            return loss_total

        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        loss_total = self.weight_ce * loss_ce + self.weight_dice * loss_dice

        if return_components:
            components = {
                "loss_total": float(loss_total.detach().cpu().item()),
                "loss_ce": float(loss_ce.detach().cpu().item()),
                "loss_dice": float(loss_dice.detach().cpu().item()),
            }
            return loss_total, components

        return loss_total
