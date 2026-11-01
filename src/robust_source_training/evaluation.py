"""Event-independent evaluation infrastructure and metrics calculation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def compute_binary_confusion_counts(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    valid_mask: np.ndarray,
    positive_class: int = 2,
    ignore_index: int = 3,
) -> Dict[str, int]:
    """Computes binary flood confusion counts (TP, FP, FN, TN) for valid pixels.

    Valid pixel invariant: (valid_mask == 1) & (gt_labels != ignore_index).
    Positive class is flood (2), negative classes are non-flood (0, 1).
    """
    valid = (valid_mask == 1) & (gt_labels != ignore_index)
    if not np.any(valid):
        return {
            "valid_pixels": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "pred_flood_pixels": 0,
            "ref_flood_pixels": 0,
        }

    y_true = gt_labels[valid] == positive_class
    y_pred = pred_labels[valid] == positive_class

    tp = int(np.logical_and(y_pred, y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())

    return {
        "valid_pixels": int(valid.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_flood_pixels": tp + fp,
        "ref_flood_pixels": tp + fn,
    }


def compute_rates_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Computes precision, recall, F1, and IoU with explicit zero-denominator handling."""
    denom_p = tp + fp
    denom_r = tp + fn
    denom_f1 = 2 * tp + fp + fn
    denom_iou = tp + fp + fn

    if denom_p == 0:
        precision = 1.0 if (tp + fn == 0) else 0.0
    else:
        precision = float(tp) / float(denom_p)

    if denom_r == 0:
        recall = 1.0 if (tp + fp == 0) else 0.0
    else:
        recall = float(tp) / float(denom_r)

    if denom_f1 == 0:
        f1 = 1.0 if (fp + fn == 0) else 0.0
    else:
        f1 = (2.0 * float(tp)) / float(denom_f1)

    if denom_iou == 0:
        iou = 1.0 if (fp + fn == 0) else 0.0
    else:
        iou = float(tp) / float(denom_iou)

    return {
        "precision": float(round(precision, 6)),
        "recall": float(round(recall, 6)),
        "f1": float(round(f1, 6)),
        "iou": float(round(iou, 6)),
    }


class EventIndependentEvaluator:
    """Evaluates models across held-out events strictly independently."""

    def __init__(
        self,
        evaluation_event_ids: List[int],
        device: torch.device,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self.evaluation_event_ids = sorted(evaluation_event_ids)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def evaluate_event(
        self,
        model: nn.Module,
        dataset: Dataset,
        event_id: int,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates one single event completely independently."""
        was_training = model.training
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_valid = 0
        total_samples = 0

        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            xA = batch["xA"].to(self.device)
            xB = batch["xB"].to(self.device)
            labels = batch["label"].cpu().numpy()
            valid_masks = batch["valid_mask"].cpu().numpy()

            logits = model(xA, xB)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            counts = compute_binary_confusion_counts(preds, labels, valid_masks)
            total_tp += counts["tp"]
            total_fp += counts["fp"]
            total_fn += counts["fn"]
            total_tn += counts["tn"]
            total_valid += counts["valid_pixels"]
            total_samples += len(batch["sample_id"])

        rates = compute_rates_from_counts(total_tp, total_fp, total_fn, total_tn)

        if was_training:
            model.train()

        return {
            "event_id": event_id,
            "sample_count": total_samples,
            "valid_pixel_count": total_valid,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "pred_flood_pixels": total_tp + total_fp,
            "ref_flood_pixels": total_tp + total_fn,
            "skipped_samples": 0,
            **rates,
        }

    def aggregate_metrics(self, event_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates event-macro arithmetic means and pooled summed metrics."""
        if not event_results:
            return {"error": "No event results to aggregate"}

        n_events = len(event_results)

        # 1. Event-macro averages
        macro_p = float(np.mean([r["precision"] for r in event_results]))
        macro_r = float(np.mean([r["recall"] for r in event_results]))
        macro_f1 = float(np.mean([r["f1"] for r in event_results]))
        macro_iou = float(np.mean([r["iou"] for r in event_results]))

        # 2. Pooled summed counts
        pooled_tp = sum(r["tp"] for r in event_results)
        pooled_fp = sum(r["fp"] for r in event_results)
        pooled_fn = sum(r["fn"] for r in event_results)
        pooled_tn = sum(r["tn"] for r in event_results)
        pooled_rates = compute_rates_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

        # 3. Worst-event finding
        worst_event = min(event_results, key=lambda r: (r["f1"], r["event_id"]))

        return {
            "n_events_evaluated": n_events,
            "event_macro": {
                "precision": round(macro_p, 6),
                "recall": round(macro_r, 6),
                "f1": round(macro_f1, 6),
                "iou": round(macro_iou, 6),
            },
            "pooled": {
                "tp": pooled_tp,
                "fp": pooled_fp,
                "fn": pooled_fn,
                "tn": pooled_tn,
                "precision": pooled_rates["precision"],
                "recall": pooled_rates["recall"],
                "f1": pooled_rates["f1"],
                "iou": pooled_rates["iou"],
            },
            "worst_event": {
                "event_id": worst_event["event_id"],
                "f1": worst_event["f1"],
                "iou": worst_event["iou"],
                "recall": worst_event["recall"],
                "precision": worst_event["precision"],
            },
            "per_event": event_results,
        }

