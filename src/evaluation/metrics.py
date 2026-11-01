"""Event-level and dataset-level evaluation metrics for binary SAR flood change detection.

Convention for zero-denominator handling:
- If tp + fp == 0 (no flood predicted):
    Precision is 1.0 if tp + fn == 0 (no ground-truth flood either), else 0.0.
- If tp + fn == 0 (no ground-truth flood exists):
    Recall is 1.0 if tp + fp == 0 (correctly predicted zero flood), else 0.0.
- If Precision + Recall == 0:
    F1 is 0.0 (unless Precision == 1.0 and Recall == 1.0, where F1 = 1.0).
- If tp + fp + fn == 0 (union of prediction and ground truth is empty):
    IoU is 1.0 (both predicted and ground truth contain zero flood pixels).
"""

from typing import Dict, Any, List
import numpy as np
import torch


def compute_confusion_counts(
    pred_classes: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    flood_class: int = 2,
    ignore_class: int = 3,
) -> Dict[str, int]:
    """Computes TP, FP, FN, TN for binary flood detection on valid pixels only.

    Rule: valid = (valid_mask == 1) & (labels != ignore_class)
    Positive target: labels == flood_class
    Negative target: labels != flood_class (and valid)
    Positive pred: pred_classes == flood_class
    Negative pred: pred_classes != flood_class
    """
    valid = (valid_mask == 1) & (labels != ignore_class)

    target_pos = (labels == flood_class) & valid
    target_neg = (labels != flood_class) & valid
    pred_pos = (pred_classes == flood_class) & valid
    pred_neg = (pred_classes != flood_class) & valid

    tp = int(np.count_nonzero(pred_pos & target_pos))
    fp = int(np.count_nonzero(pred_pos & target_neg))
    fn = int(np.count_nonzero(pred_neg & target_pos))
    tn = int(np.count_nonzero(pred_neg & target_neg))

    valid_pixels = int(np.count_nonzero(valid))
    gt_flood = int(np.count_nonzero(target_pos))
    pred_flood = int(np.count_nonzero(pred_pos))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": valid_pixels,
        "gt_flood_pixels": gt_flood,
        "pred_flood_pixels": pred_flood,
    }


def compute_flood_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    """Computes Precision, Recall, F1, and IoU from confusion counts.

    Zero-valid-pixel rule:
    If total valid pixels (tp + fp + fn + tn == 0) or counts.get("valid_pixels") == 0:
        The input is technically unevaluable (zero valid pixels). Metrics return None
        rather than awarding an artificial perfect score (1.0).
    Genuine flood-free rule:
        If valid pixels exist (tn > 0) and tp + fp + fn == 0:
        Precision, Recall, F1, and IoU return 1.0 (perfect negative agreement).
    """
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = tp + fp + fn + tn

    if total == 0 or counts.get("valid_pixels") == 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
        }

    # Precision
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        # No predicted flood
        precision = 1.0 if (tp + fn) == 0 else 0.0

    # Recall
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        # No ground-truth flood
        recall = 1.0 if (tp + fp) == 0 else 0.0

    # F1 Score
    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 1.0 if (precision == 1.0 and recall == 1.0) else 0.0

    # IoU
    union = tp + fp + fn
    if union > 0:
        iou = tp / union
    else:
        iou = 1.0  # Perfect agreement on empty flood target with valid negative pixels

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def compute_macro_metrics(event_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes unweighted equal-event macro metrics across evaluable event rows."""
    evaluable = [
        r for r in event_rows
        if r.get("is_evaluable", True) and r.get("f1") is not None
    ]
    if not evaluable:
        return {
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "macro_iou": None,
            "median_precision": None,
            "median_recall": None,
            "median_f1": None,
            "median_iou": None,
            "evaluable_events_count": 0,
            "total_events_count": len(event_rows),
        }
    precisions = [float(r["precision"]) for r in evaluable]
    recalls = [float(r["recall"]) for r in evaluable]
    f1s = [float(r["f1"]) for r in evaluable]
    ious = [float(r["iou"]) for r in evaluable]
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "macro_iou": float(np.mean(ious)),
        "median_precision": float(np.median(precisions)),
        "median_recall": float(np.median(recalls)),
        "median_f1": float(np.median(f1s)),
        "median_iou": float(np.median(ious)),
        "evaluable_events_count": len(evaluable),
        "total_events_count": len(event_rows),
    }


def compute_pooled_metrics(event_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes pooled confusion metrics across all evaluable event rows."""
    evaluable = [
        r for r in event_rows
        if r.get("is_evaluable", True) and r.get("valid_pixels", 1) > 0
    ]
    if not evaluable:
        return {
            "pooled_tp": 0,
            "pooled_fp": 0,
            "pooled_fn": 0,
            "pooled_tn": 0,
            "pooled_precision": None,
            "pooled_recall": None,
            "pooled_f1": None,
            "pooled_iou": None,
            "evaluable_events_count": 0,
            "total_events_count": len(event_rows),
        }
    pooled_tp = sum(r["tp"] for r in evaluable)
    pooled_fp = sum(r["fp"] for r in evaluable)
    pooled_fn = sum(r["fn"] for r in evaluable)
    pooled_tn = sum(r["tn"] for r in evaluable)
    pooled_counts = {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn, "tn": pooled_tn}
    m = compute_flood_metrics(pooled_counts)
    return {
        "pooled_tp": pooled_tp,
        "pooled_fp": pooled_fp,
        "pooled_fn": pooled_fn,
        "pooled_tn": pooled_tn,
        "pooled_precision": m["precision"],
        "pooled_recall": m["recall"],
        "pooled_f1": m["f1"],
        "pooled_iou": m["iou"],
        "evaluable_events_count": len(evaluable),
        "total_events_count": len(event_rows),
    }


class EventMetricAccumulator:
    """Accumulates confusion counts across batches for each event."""

    def __init__(self):
        self.events: Dict[int, Dict[str, int]] = {}

    def update(
        self,
        event_ids: List[int],
        pred_classes: np.ndarray,
        labels: np.ndarray,
        valid_masks: np.ndarray,
    ):
        """Update confusion counts per sample/event."""
        if isinstance(pred_classes, torch.Tensor):
            pred_classes = pred_classes.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(valid_masks, torch.Tensor):
            valid_masks = valid_masks.detach().cpu().numpy()

        batch_size = len(event_ids)
        for i in range(batch_size):
            eid = int(event_ids[i])
            counts = compute_confusion_counts(
                pred_classes[i], labels[i], valid_masks[i]
            )
            if eid not in self.events:
                self.events[eid] = {
                    "sample_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "valid_pixels": 0,
                    "gt_flood_pixels": 0,
                    "pred_flood_pixels": 0,
                }
            self.events[eid]["sample_count"] += 1
            for k in ["tp", "fp", "fn", "tn", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels"]:
                self.events[eid][k] += counts[k]

    def compute_event_table(self) -> List[Dict[str, Any]]:
        """Returns row-level metrics for each evaluated event."""
        rows = []
        for eid in sorted(self.events.keys()):
            counts = self.events[eid]
            valid_px = counts["valid_pixels"]
            if valid_px == 0:
                row = {
                    "event_id": eid,
                    "sample_count": counts["sample_count"],
                    "valid_pixels": 0,
                    "gt_flood_pixels": 0,
                    "pred_flood_pixels": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "iou": None,
                    "is_evaluable": False,
                    "evaluation_status": "technically_unevaluable_zero_valid_pixels",
                }
            else:
                metrics = compute_flood_metrics(counts)
                if counts["gt_flood_pixels"] == 0 and counts["pred_flood_pixels"] == 0:
                    status = "evaluable_flood_free_perfect_negative"
                elif counts["gt_flood_pixels"] > 0 and counts["pred_flood_pixels"] == 0:
                    status = "evaluable_zero_flood_predicted"
                else:
                    status = "evaluable"
                row = {
                    "event_id": eid,
                    "sample_count": counts["sample_count"],
                    "valid_pixels": counts["valid_pixels"],
                    "gt_flood_pixels": counts["gt_flood_pixels"],
                    "pred_flood_pixels": counts["pred_flood_pixels"],
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "tn": counts["tn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "iou": metrics["iou"],
                    "is_evaluable": True,
                    "evaluation_status": status,
                }
            rows.append(row)
        return rows

    def compute_summary(self) -> Dict[str, Any]:
        """Computes macro and pooled metrics across all evaluable events."""
        event_rows = self.compute_event_table()
        if not event_rows:
            return {}

        evaluable = [r for r in event_rows if r.get("is_evaluable", True) and r.get("f1") is not None]
        macro = compute_macro_metrics(event_rows)
        pooled = compute_pooled_metrics(event_rows)

        if evaluable:
            worst_f1_event = min(evaluable, key=lambda r: r["f1"])
            worst_recall_event = min(evaluable, key=lambda r: r["recall"])
            worst_f1_res = {"event_id": worst_f1_event["event_id"], "f1": worst_f1_event["f1"]}
            worst_recall_res = {"event_id": worst_recall_event["event_id"], "recall": worst_recall_event["recall"]}
        else:
            worst_f1_res = None
            worst_recall_res = None

        zero_pred_events = sum(1 for r in evaluable if r["pred_flood_pixels"] == 0)

        return {
            "num_events": len(event_rows),
            "evaluable_events_count": len(evaluable),
            "macro_precision": macro["macro_precision"],
            "macro_recall": macro["macro_recall"],
            "macro_f1": macro["macro_f1"],
            "macro_iou": macro["macro_iou"],
            "median_precision": macro["median_precision"],
            "median_recall": macro["median_recall"],
            "median_f1": macro["median_f1"],
            "median_iou": macro["median_iou"],
            "pooled_counts": {
                "tp": pooled["pooled_tp"],
                "fp": pooled["pooled_fp"],
                "fn": pooled["pooled_fn"],
                "tn": pooled["pooled_tn"],
            },
            "pooled_metrics": {
                "precision": pooled["pooled_precision"],
                "recall": pooled["pooled_recall"],
                "f1": pooled["pooled_f1"],
                "iou": pooled["pooled_iou"],
            },
            "worst_event_f1": worst_f1_res,
            "worst_event_recall": worst_recall_res,
            "events_with_zero_predicted_flood": zero_pred_events,
        }
