"""Source Model Evaluation Engine for ChangeGuard.

Guarantees:
1. Pure source-only evaluation: model.eval(), torch.no_grad().
2. Zero optimizer steps, zero backward passes, zero batch-norm running statistics adaptation.
3. Cryptographic model parameter and buffer hashing before and after inference to ensure state immutability.
4. Sealed test events [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013] strictly forbidden.
5. Exact confusion counts, flood metrics (precision, recall, F1, IoU), equal-event macro, pooled metrics, and worst-event identification.
6. Multi-seed metric aggregation using arithmetic mean +/- sample standard deviation (ddof=1) without inter-seed pooling.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.evaluation_dataset import (
    FORBIDDEN_SEALED_TEST_EVENTS,
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    TOTAL_EXPECTED_TTA_DEV_SAMPLES,
)
from src.evaluation.metrics import compute_confusion_counts, compute_flood_metrics


def compute_model_state_hash(model: nn.Module) -> str:
    """Computes SHA-256 hash across all model parameters and buffers to verify immutability."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode("utf-8"))
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class SourceModelEvaluator:
    """Evaluates a frozen source model on target/development events."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        flood_class: int = 2,
        ignore_class: int = 3,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.flood_class = int(flood_class)
        self.ignore_class = int(ignore_class)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        expected_counts: Optional[Dict[int, int]] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs deterministic source-only inference and returns comprehensive evaluation metrics."""
        self.model.eval()

        # Capture pre-inference model state hash
        pre_hash = compute_model_state_hash(self.model)

        events_data: Dict[int, Dict[str, Any]] = {}
        observed_sample_ids: Set[str] = set()
        duplicate_sample_ids: List[str] = []
        accessed_events: Set[int] = set()

        total_batches = len(data_loader)
        start_time = time.time()

        sample_valid_pixel_counts: Dict[str, int] = {}

        for batch_idx, batch in enumerate(data_loader):
            sample_ids = batch["sample_id"]
            event_ids = batch["event_id"]
            if torch.is_tensor(event_ids):
                event_ids = event_ids.cpu().tolist()

            # Quarantine assertions
            for eid, sid in zip(event_ids, sample_ids):
                eid_int = int(eid)
                sid_str = str(sid)
                if eid_int in FORBIDDEN_SEALED_TEST_EVENTS:
                    raise RuntimeError(
                        f"CRITICAL RESEARCH INTEGRITY VIOLATION: Sealed test event {eid_int} "
                        f"encountered during evaluation! Immediate halt."
                    )
                if sid_str in observed_sample_ids:
                    duplicate_sample_ids.append(sid_str)
                observed_sample_ids.add(sid_str)
                accessed_events.add(eid_int)

            xA = batch["xA"].to(self.device, non_blocking=True)
            xB = batch["xB"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)

            logits = self.model(xA, xB)
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Nonfinite logits detected in batch {batch_idx}! "
                    f"Sample IDs: {sample_ids}, Event IDs: {event_ids}"
                )

            preds = torch.argmax(logits, dim=1)

            # Accumulate metrics per sample
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            valid_masks_np = valid_masks.cpu().numpy()

            for i in range(len(sample_ids)):
                eid = int(event_ids[i])
                sid_str = str(sample_ids[i])
                if eid not in events_data:
                    events_data[eid] = {
                        "event_id": eid,
                        "sample_count": 0,
                        "sample_ids": [],
                        "valid_samples": [],
                        "invalid_only_samples": [],
                        "valid_pixels": 0,
                        "gt_flood_pixels": 0,
                        "pred_flood_pixels": 0,
                        "pred_background_pixels": 0,
                        "pred_permanent_water_pixels": 0,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "tn": 0,
                    }

                ed = events_data[eid]
                ed["sample_count"] += 1
                ed["sample_ids"].append(sid_str)

                # Double-masking rule: valid = (valid_mask == 1) & (label != 3)
                v_mask = (valid_masks_np[i] == 1) & (labels_np[i] != self.ignore_class)
                valid_count = int(np.count_nonzero(v_mask))
                ed["valid_pixels"] += valid_count
                sample_valid_pixel_counts[sid_str] = valid_count

                if valid_count == 0:
                    ed["invalid_only_samples"].append(sid_str)
                else:
                    ed["valid_samples"].append(sid_str)

                p_patch = preds_np[i]
                l_patch = labels_np[i]

                # Confusion counts for target flood class 2
                target_pos = (l_patch == self.flood_class) & v_mask
                target_neg = (l_patch != self.flood_class) & v_mask
                pred_pos = (p_patch == self.flood_class) & v_mask
                pred_neg = (p_patch != self.flood_class) & v_mask

                ed["tp"] += int(np.count_nonzero(pred_pos & target_pos))
                ed["fp"] += int(np.count_nonzero(pred_pos & target_neg))
                ed["fn"] += int(np.count_nonzero(pred_neg & target_pos))
                ed["tn"] += int(np.count_nonzero(pred_neg & target_neg))

                ed["gt_flood_pixels"] += int(np.count_nonzero(target_pos))
                ed["pred_flood_pixels"] += int(np.count_nonzero(pred_pos))
                ed["pred_background_pixels"] += int(np.count_nonzero((p_patch == 0) & v_mask))
                ed["pred_permanent_water_pixels"] += int(np.count_nonzero((p_patch == 1) & v_mask))

        elapsed_sec = time.time() - start_time

        # Verify model parameter & buffer immutability
        post_hash = compute_model_state_hash(self.model)
        if pre_hash != post_hash:
            raise RuntimeError(
                "CRITICAL ERROR: Model parameters or buffers changed during evaluation! "
                f"Pre-hash: {pre_hash}, Post-hash: {post_hash}"
            )

        # Assemble event-level table
        event_table: List[Dict[str, Any]] = []
        for eid in sorted(events_data.keys()):
            ed = events_data[eid]
            valid_px = ed["valid_pixels"]

            if valid_px == 0:
                # Zero valid pixels across the event: technically unevaluable
                row = {
                    "event_id": eid,
                    "sample_count": ed["sample_count"],
                    "invalid_only_samples_count": len(ed["invalid_only_samples"]),
                    "valid_pixels": 0,
                    "gt_flood_pixels": 0,
                    "pred_flood_pixels": 0,
                    "pred_flood_fraction": 0.0,
                    "pred_background_pixels": 0,
                    "pred_permanent_water_pixels": 0,
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
                conf = {"tp": ed["tp"], "fp": ed["fp"], "fn": ed["fn"], "tn": ed["tn"]}
                m = compute_flood_metrics(conf)
                pred_flood_px = ed["pred_flood_pixels"]
                pred_flood_fraction = float(pred_flood_px / valid_px)

                # Determine evaluation status distinguishing genuine flood-free from missed flood
                if ed["gt_flood_pixels"] == 0 and pred_flood_px == 0:
                    ev_status = "evaluable_flood_free_perfect_negative"
                elif ed["gt_flood_pixels"] > 0 and pred_flood_px == 0:
                    ev_status = "evaluable_zero_flood_predicted"
                else:
                    ev_status = "evaluable"

                row = {
                    "event_id": eid,
                    "sample_count": ed["sample_count"],
                    "invalid_only_samples_count": len(ed["invalid_only_samples"]),
                    "valid_pixels": valid_px,
                    "gt_flood_pixels": ed["gt_flood_pixels"],
                    "pred_flood_pixels": pred_flood_px,
                    "pred_flood_fraction": pred_flood_fraction,
                    "pred_background_pixels": ed["pred_background_pixels"],
                    "pred_permanent_water_pixels": ed["pred_permanent_water_pixels"],
                    "tp": ed["tp"],
                    "fp": ed["fp"],
                    "fn": ed["fn"],
                    "tn": ed["tn"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "iou": m["iou"],
                    "is_evaluable": True,
                    "evaluation_status": ev_status,
                }
            event_table.append(row)

        # Equal-event macro metrics across evaluable events only
        evaluable_rows = [r for r in event_table if r.get("is_evaluable", True) and r.get("f1") is not None]
        unevaluable_rows = [r for r in event_table if not r.get("is_evaluable", True)]

        if evaluable_rows:
            macro_precision = float(np.mean([r["precision"] for r in evaluable_rows]))
            macro_recall = float(np.mean([r["recall"] for r in evaluable_rows]))
            macro_f1 = float(np.mean([r["f1"] for r in evaluable_rows]))
            macro_iou = float(np.mean([r["iou"] for r in evaluable_rows]))

            # Pooled confusion metrics on evaluable events
            pooled_tp = sum(r["tp"] for r in evaluable_rows)
            pooled_fp = sum(r["fp"] for r in evaluable_rows)
            pooled_fn = sum(r["fn"] for r in evaluable_rows)
            pooled_tn = sum(r["tn"] for r in evaluable_rows)
            pooled_conf = {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn, "tn": pooled_tn}
            pooled_m = compute_flood_metrics(pooled_conf)

            worst_f1_row = min(evaluable_rows, key=lambda r: r["f1"])
            worst_event_f1 = {
                "event_id": worst_f1_row["event_id"],
                "f1": worst_f1_row["f1"],
                "recall": worst_f1_row["recall"],
                "precision": worst_f1_row["precision"],
            }

            all_background_events = [
                r["event_id"] for r in evaluable_rows if r["pred_flood_pixels"] == 0 and r["pred_permanent_water_pixels"] == 0
            ]
            all_permanent_water_events = [
                r["event_id"] for r in evaluable_rows if r["pred_background_pixels"] == 0 and r["pred_flood_pixels"] == 0
            ]
            all_flood_events = [
                r["event_id"] for r in evaluable_rows if r["pred_background_pixels"] == 0 and r["pred_permanent_water_pixels"] == 0
            ]
            zero_flood_events = [
                r["event_id"] for r in evaluable_rows if r["pred_flood_pixels"] == 0
            ]
            zero_flood_on_flood_positive_events = [
                r["event_id"] for r in evaluable_rows if r["gt_flood_pixels"] > 0 and r["pred_flood_pixels"] == 0
            ]
        else:
            macro_precision = macro_recall = macro_f1 = macro_iou = None
            pooled_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            pooled_m = {"precision": None, "recall": None, "f1": None, "iou": None}
            worst_event_f1 = {"event_id": -1, "f1": None, "recall": None, "precision": None}
            all_background_events = []
            all_permanent_water_events = []
            all_flood_events = []
            zero_flood_events = []
            zero_flood_on_flood_positive_events = []

        collapse_audit = {
            "run_all_background": bool(len(evaluable_rows) > 0 and len(all_background_events) == len(evaluable_rows)),
            "run_all_permanent_water": bool(len(evaluable_rows) > 0 and len(all_permanent_water_events) == len(evaluable_rows)),
            "run_all_flood": bool(len(evaluable_rows) > 0 and len(all_flood_events) == len(evaluable_rows)),
            "run_zero_flood_predicted": bool(len(evaluable_rows) > 0 and len(zero_flood_events) == len(evaluable_rows)),
            "all_background_events": all_background_events,
            "all_permanent_water_events": all_permanent_water_events,
            "all_flood_events": all_flood_events,
            "zero_flood_events": zero_flood_events,
            "zero_flood_on_flood_positive_events": zero_flood_on_flood_positive_events,
            "unevaluable_events": [r["event_id"] for r in unevaluable_rows],
        }

        # Completeness audit
        expected_counts = expected_counts or EXPECTED_TTA_DEV_COUNTS
        coverage_audit = {}
        for eid, exp_cnt in expected_counts.items():
            obs_cnt = events_data.get(eid, {}).get("sample_count", 0)
            coverage_audit[eid] = {
                "expected": exp_cnt,
                "observed": obs_cnt,
                "matched": (obs_cnt == exp_cnt),
            }

        total_exp_samples = sum(expected_counts.values())
        counts_verified = (
            all(c["matched"] for c in coverage_audit.values())
            and (len(observed_sample_ids) == total_exp_samples)
            and (len(duplicate_sample_ids) == 0)
        )
        quarantine_ok = len(accessed_events & FORBIDDEN_SEALED_TEST_EVENTS) == 0

        # Technical validity rules:
        # Require all six events to be technically evaluable for each seed before declaring a complete six-event baseline.
        # If an event has zero valid pixels:
        # - retain its sample and completeness records;
        # - mark technical validity as failed;
        # - report the number of evaluable events;
        # - label any reduced-event macro average as diagnostic only.
        exp_event_count = len(expected_counts) if expected_counts else 6
        all_events_evaluable = (len(evaluable_rows) == len(event_table) == exp_event_count) and (len(unevaluable_rows) == 0)
        technical_validity_passed = (
            counts_verified
            and quarantine_ok
            and (pre_hash == post_hash)
            and all_events_evaluable
        )

        technical_validity = {
            "passed": technical_validity_passed,
            "sample_counts_verified": counts_verified,
            "quarantine_sealed_test_zero_leakage": quarantine_ok,
            "model_immutability_verified": (pre_hash == post_hash),
            "all_events_evaluable": all_events_evaluable,
            "evaluable_events_count": len(evaluable_rows),
            "total_events_count": len(event_table),
            "failure_reasons": [] if technical_validity_passed else [
                *(["sample_count_or_duplicate_mismatch"] if not counts_verified else []),
                *(["sealed_test_quarantine_violation"] if not quarantine_ok else []),
                *(["model_parameter_or_buffer_mutation"] if pre_hash != post_hash else []),
                *(["unevaluable_zero_valid_pixel_events_present"] if not all_events_evaluable else []),
            ],
        }

        macro_dict = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "iou": macro_iou,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_iou": macro_iou,
            "evaluable_events_count": len(evaluable_rows),
            "total_events_count": len(event_table),
            "is_complete_six_event_macro": bool(all_events_evaluable),
            "macro_type": "complete_six_event" if all_events_evaluable else "reduced_event_diagnostic_only",
        }

        return {
            "num_events": len(event_table),
            "total_evaluated_samples": len(observed_sample_ids),
            "total_samples_observed": len(observed_sample_ids),
            "expected_total_samples": total_exp_samples,
            "sample_counts_verified": counts_verified,
            "duplicates_detected": len(duplicate_sample_ids),
            "quarantine_sealed_test_zero_leakage": quarantine_ok,
            "pre_inference_model_hash": pre_hash,
            "post_inference_model_hash": post_hash,
            "model_immutability_verified": (pre_hash == post_hash),
            "technical_validity": technical_validity,
            "technical_validity_passed": technical_validity_passed,
            "all_events_evaluable": all_events_evaluable,
            "evaluable_events_count": len(evaluable_rows),
            "elapsed_seconds": elapsed_sec,
            "macro_metrics": macro_dict,
            "equal_event_macro": macro_dict,
            "pooled_metrics": {
                "confusion": pooled_conf,
                "precision": pooled_m["precision"],
                "recall": pooled_m["recall"],
                "f1": pooled_m["f1"],
                "iou": pooled_m["iou"],
                "pooled_precision": pooled_m["precision"],
                "pooled_recall": pooled_m["recall"],
                "pooled_f1": pooled_m["f1"],
                "pooled_iou": pooled_m["iou"],
            },
            "worst_event": worst_event_f1,
            "events_with_zero_predicted_flood": zero_flood_events,
            "events_all_background": all_background_events,
            "events_all_permanent_water": all_permanent_water_events,
            "events_all_flood": all_flood_events,
            "zero_flood_on_flood_positive_events": zero_flood_on_flood_positive_events,
            "collapse_audit": collapse_audit,
            "event_table": event_table,
            "event_metrics": event_table,
            "coverage_audit": coverage_audit,
            "sample_valid_pixel_counts": sample_valid_pixel_counts,
            "state_hash": post_hash,
        }


def aggregate_three_seed_metrics(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes arithmetic mean and sample standard deviation (ddof=1) across seeds without pooling."""
    if len(seed_summaries) != 3:
        raise ValueError(f"Expected exactly 3 seed summaries, got {len(seed_summaries)}")

    def _get_val(summary: Dict[str, Any], group: str, key: str) -> Optional[float]:
        grp = summary.get(group)
        if grp is None and group == "macro_metrics":
            grp = summary.get("equal_event_macro")
        elif grp is None and group == "equal_event_macro":
            grp = summary.get("macro_metrics")
        if grp is None:
            return None

        val = grp.get(key)
        if val is None:
            alt_key = f"macro_{key}" if not key.startswith("macro_") else key.replace("macro_", "")
            val = grp.get(alt_key)
        if val is None:
            alt_key = f"pooled_{key}" if not key.startswith("pooled_") else key.replace("pooled_", "")
            val = grp.get(alt_key)
        return float(val) if val is not None else None

    def _stat(vals: List[Optional[float]]) -> Dict[str, Any]:
        valid_vals = [v for v in vals if v is not None]
        if not valid_vals:
            return {}
        m = float(np.mean(valid_vals))
        s = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
        return {
            "mean": m,
            "std": s,
            "sample_std": s,
            "formatted": f"{m:.4f} +/- {s:.4f}",
        }

    macro_f1s = [_get_val(s, "macro_metrics", "f1") for s in seed_summaries]
    macro_recalls = [_get_val(s, "macro_metrics", "recall") for s in seed_summaries]
    macro_precisions = [_get_val(s, "macro_metrics", "precision") for s in seed_summaries]
    macro_ious = [_get_val(s, "macro_metrics", "iou") for s in seed_summaries]

    pooled_f1s = [_get_val(s, "pooled_metrics", "f1") for s in seed_summaries]
    pooled_recalls = [_get_val(s, "pooled_metrics", "recall") for s in seed_summaries]
    pooled_precisions = [_get_val(s, "pooled_metrics", "precision") for s in seed_summaries]
    pooled_ious = [_get_val(s, "pooled_metrics", "iou") for s in seed_summaries]

    worst_f1s = [float(s["worst_event"]["f1"]) for s in seed_summaries if "worst_event" in s]

    # Per-event statistics across seeds
    first_tbl = seed_summaries[0].get("event_table") or seed_summaries[0].get("event_metrics", [])
    event_ids = [r["event_id"] for r in first_tbl]
    event_stats = {}
    for eid in event_ids:
        f1_list = []
        rec_list = []
        prec_list = []
        iou_list = []
        for s in seed_summaries:
            tbl = s.get("event_table") or s.get("event_metrics", [])
            match = next((r for r in tbl if r["event_id"] == eid), None)
            if match:
                f1_list.append(float(match["f1"]) if match.get("f1") is not None else None)
                rec_list.append(float(match["recall"]) if match.get("recall") is not None else None)
                prec_list.append(float(match["precision"]) if match.get("precision") is not None else None)
                iou_list.append(float(match["iou"]) if match.get("iou") is not None else None)
        event_stats[eid] = {
            "f1": _stat(f1_list),
            "recall": _stat(rec_list),
            "precision": _stat(prec_list),
            "iou": _stat(iou_list),
        }

    macro_summary = {
        "macro_f1": _stat(macro_f1s),
        "macro_recall": _stat(macro_recalls),
        "macro_precision": _stat(macro_precisions),
        "macro_iou": _stat(macro_ious),
    }

    all_seeds_technically_valid = all(
        s.get("technical_validity", {}).get("passed", False) or s.get("technical_validity_passed", False)
        for s in seed_summaries
    )
    all_seeds_complete_six = all(
        s.get("equal_event_macro", {}).get("is_complete_six_event_macro", False)
        for s in seed_summaries
    )
    is_complete_six_event_baseline = bool(all_seeds_technically_valid and all_seeds_complete_six)
    macro_type = "complete_six_event" if is_complete_six_event_baseline else "reduced_event_diagnostic_only"

    return {
        "num_seeds": 3,
        "aggregation_rule": "arithmetic_mean_and_sample_std_ddof_1_no_inter_seed_pooling",
        "is_complete_six_event_baseline": is_complete_six_event_baseline,
        "macro_type": macro_type,
        "all_seeds_technically_valid": all_seeds_technically_valid,
        "macro_metrics": {
            "f1": _stat(macro_f1s),
            "recall": _stat(macro_recalls),
            "precision": _stat(macro_precisions),
            "iou": _stat(macro_ious),
        },
        "macro_summary": macro_summary,
        "pooled_metrics": {
            "f1": _stat(pooled_f1s),
            "recall": _stat(pooled_recalls),
            "precision": _stat(pooled_precisions),
            "iou": _stat(pooled_ious),
        },
        "worst_event_f1": _stat(worst_f1s) if worst_f1s else {},
        "per_event_stats": event_stats,
    }
