"""ChangeGuard Week 12 Matched BN No-Update Diagnostic Runner.

Governing Config: configs/changeguard/week12_bn_diagnostic_dev6_v1.json
Evidence Level:   EXPLORATORY

Evaluates candidate BN behavior (module.train(), track_running_stats=False, running_mean/var=None)
with strictly ZERO backward passes and ZERO optimizer updates, verifying parameter immutability.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.evaluation_dataset import KuroEvaluationDataset
from tools.run_week12_ablation import MockSNUNet


class MatchedBNNoUpdateAdapter:
    """Evaluates in candidate BN context with ZERO backward passes and ZERO optimizer updates."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        # Candidate BN context: eval globally, BN modules train() with no running stats tracking
        self.model.eval()
        self.model.requires_grad_(False)

        self.bn_module_names: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                self.bn_module_names.append(name)

        # Snapshot initial parameters to verify immutability
        self._initial_params = {
            name: param.detach().clone() for name, param in self.model.named_parameters()
        }

        self.step_count = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0

    def adapt_and_predict(
        self,
        xA: torch.Tensor,
        xB: torch.Tensor,
        sar_valid_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None,
        event_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        xA = xA.to(self.device)
        xB = xB.to(self.device)

        if sar_valid_mask is not None:
            mask = sar_valid_mask.to(self.device)
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask_bool = mask > 0
            valid_count = int(valid_mask_bool.sum().item())
        else:
            valid_count = int(xA.shape[0] * xA.shape[2] * xA.shape[3])

        # Strictly no gradients, zero backward pass
        with torch.no_grad():
            logits = self.model(xA, xB)
            pred_class = logits.argmax(dim=1)

        # Assert zero parameter drift
        for name, param in self.model.named_parameters():
            if not torch.equal(param, self._initial_params[name]):
                raise RuntimeError(f"Parameter mutation detected in {name} during no-update diagnostic forward!")

        self.step_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primary_prediction": {
                "predicted_class": pred_class,
                "logits": logits.detach(),
            },
            "adaptation_metadata": {
                "action": "bn_eval_no_update",
                "update_applied": False,
                "loss": None,
                "valid_pixels": valid_count,
                "selected_pixels": 0,
                "selected_pixel_fraction": 0.0,
                "failure_reason": None,
                "update_time_ms": elapsed_ms,
            },
        }


class Week12BNDiagnosticRunner:
    """Orchestrates the single-condition matched_bn_no_update_v1 exploratory diagnostic."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cpu",
        synthetic_mock: bool = False,
        allow_overwrite: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.synthetic_mock = bool(synthetic_mock)
        self.allow_overwrite = bool(allow_overwrite)
        self.device_str = str(device).lower()

        # Outputs
        out_cfg = self.cfg.get("outputs", {})
        if output_dir is not None:
            self.output_dir = output_dir.resolve()
        else:
            self.output_dir = (REPO_ROOT / out_cfg.get("experiment_dir", "outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1")).resolve()

        self.log_dir = (REPO_ROOT / out_cfg.get("log_dir", "logs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1")).resolve()

        # Authorization & Status Validation
        self.status = self.cfg.get("status", "")
        self.approval_status = self.cfg.get("approval_status", "")
        self.evidence_level = self.cfg.get("evidence_level", "")
        auth = self.cfg.get("execution_authorizations", {})
        self.real_data_authorized = bool(auth.get("real_data_execution_authorized", False))
        self.single_gpu_authorized = bool(auth.get("gpu_execution_authorized", False))

        if not self.synthetic_mock:
            if self.status != "APPROVED_FOR_WEEK12_BN_DIAGNOSTIC_EXECUTION":
                raise PermissionError(f"Real-data execution rejected: status '{self.status}' is not approved.")
            if self.approval_status != "APPROVED_FOR_EXECUTION":
                raise PermissionError("approval_status must be 'APPROVED_FOR_EXECUTION'.")
            if not self.real_data_authorized or not self.single_gpu_authorized:
                raise PermissionError("real_data_execution_authorized and gpu_execution_authorized must be true.")

            # Device syntax validation
            cuda_pattern = re.compile(r"^cuda:(0|[1-9]\d*)$")
            if not cuda_pattern.match(self.device_str):
                raise ValueError(
                    f"Real diagnostic execution strictly requires device format 'cuda:<int>', got '{self.device_str}'."
                )

        # Dataset metadata
        d_cfg = self.cfg.get("dataset", {})
        self.canonical_preevent: str = d_cfg.get("canonical_preevent", "pre_event_2")
        self.sar_mean = tuple(d_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = tuple(d_cfg.get("sar_std", [0.0427, 0.0215]))
        self.sar_clamp_max = float(d_cfg.get("sar_clamp_max", 0.15))

        raw_root = d_cfg.get("dataset_root", "../datasets/KuroSiwo")
        raw_manifest = d_cfg.get("manifest_path", "../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz")
        self.dataset_root = self._resolve_path(raw_root)
        self.manifest_path = self._resolve_path(raw_manifest)
        self.target_events = list(d_cfg.get("development_events", [279, 437, 520, 559, 1111003, 1111008]))
        self.forbidden_events = set(d_cfg.get("forbidden_sealed_test_events", []))
        for ev in self.target_events:
            if ev in self.forbidden_events:
                raise ValueError(f"Target event {ev} is in forbidden_events!")

        # Diagnostic specification
        diag_cfg = self.cfg.get("diagnostic_specification", {})
        self.checkpoint_seed = int(diag_cfg.get("checkpoint_seed", 0))
        self.stream_order = diag_cfg.get("stream_order", "order_a")
        self.event_sequence = [int(e) for e in diag_cfg.get("event_sequence", self.target_events)]
        self.reset_policy = diag_cfg.get("reset_mode", "reset_per_event")
        self.batch_size = int(diag_cfg.get("batch_size", 16))
        self.condition_specs = diag_cfg.get("conditions", [])

        # Model configuration
        m_cfg = self.cfg.get("model", {})
        self.in_channels = int(m_cfg.get("in_channels", 2))
        self.num_classes = int(m_cfg.get("num_classes", 3))
        self.base_channel = int(m_cfg.get("base_channel", 32))

        # Source checkpoint metadata
        src_cfg = self.cfg.get("source_checkpoints", {})
        seed_info = src_cfg.get("seeds", {}).get(str(self.checkpoint_seed), {})
        self.ckpt_rel_path = seed_info.get("checkpoint_path", "")
        self.ckpt_expected_sha256 = seed_info.get("expected_sha256", "")
        self.manifest_sha256 = None

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        if (REPO_ROOT / p).exists():
            return (REPO_ROOT / p).resolve()
        return (REPO_ROOT.parent / p).resolve()

    def validate_preflight(self) -> Dict[str, Any]:
        ckpt_path_str = "MOCKED_SYNTHETIC"
        ckpt_exists = True
        ckpt_hash_valid = True
        ckpt_error = None
        self.ckpt_actual_sha256 = "MOCKED_SYNTHETIC"
        dataset_root_valid = True
        manifest_path_valid = True

        if not self.synthetic_mock:
            abs_ckpt = REPO_ROOT / self.ckpt_rel_path
            ckpt_path_str = str(abs_ckpt)
            if not abs_ckpt.is_file():
                ckpt_exists = False
                ckpt_error = f"Checkpoint file missing: {abs_ckpt}"
            else:
                h = hashlib.sha256()
                with open(abs_ckpt, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
                self.ckpt_actual_sha256 = h.hexdigest()
                ckpt_hash_valid = (self.ckpt_actual_sha256 == self.ckpt_expected_sha256)

            if not self.dataset_root.is_dir():
                dataset_root_valid = False
            if not self.manifest_path.is_file():
                manifest_path_valid = False
            else:
                hm = hashlib.sha256()
                with open(self.manifest_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        hm.update(chunk)
                self.manifest_sha256 = hm.hexdigest()

            if not (ckpt_exists and ckpt_hash_valid and dataset_root_valid and manifest_path_valid):
                raise RuntimeError(
                    f"Preflight validation failed: ckpt_exists={ckpt_exists}, ckpt_hash_valid={ckpt_hash_valid}, "
                    f"dataset_root_valid={dataset_root_valid}, manifest_path_valid={manifest_path_valid}"
                )

        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.allow_overwrite:
                raise FileExistsError(
                    f"Output directory {self.output_dir} already exists and contains files. "
                    "Re-running requires explicit allow_overwrite authorization."
                )

        return {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "evidence_level": self.evidence_level,
            "target_events": self.target_events,
            "forbidden_events": sorted(list(self.forbidden_events)),
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_path": ckpt_path_str,
            "checkpoint_exists": ckpt_exists,
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "checkpoint_hash_valid": ckpt_hash_valid,
            "checkpoint_error": ckpt_error,
            "dataset_root": str(self.dataset_root) if not self.synthetic_mock else "MOCKED_SYNTHETIC",
            "manifest_path": str(self.manifest_path) if not self.synthetic_mock else "MOCKED_SYNTHETIC",
            "manifest_sha256": self.manifest_sha256,
            "dataset_root_valid": dataset_root_valid,
            "manifest_path_valid": manifest_path_valid,
            "output_dir": str(self.output_dir),
            "output_dir_clean": not (self.output_dir.exists() and any(self.output_dir.iterdir())),
            "allow_overwrite": self.allow_overwrite,
            "synthetic_mock": self.synthetic_mock,
            "device": self.device_str,
        }

    def _instantiate_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        if self.synthetic_mock:
            model = MockSNUNet(in_channels=self.in_channels, num_classes=self.num_classes, base_channels=8)
            model.eval()
            model.to(self.device_str)
            return model, {"mock": True}
        else:
            ckpt_info = self.cfg["source_checkpoints"]["seeds"][str(self.checkpoint_seed)]
            ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
            wrapper, meta = SNUNetWrapper.load_from_checkpoint(
                checkpoint_path=str(ckpt_path),
                in_channels=self.in_channels,
                out_ch=self.num_classes,
                base_channel=self.base_channel,
                device=self.device_str,
            )
            return wrapper, meta

    def _resolve_samples_for_event(self, event_id: int, event_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not event_items:
            raise ValueError(f"No samples found for event {event_id}.")
        sorted_items = sorted(event_items, key=lambda x: str(x["sample_id"]))
        seen_ids = set()
        unique_items = []
        for it in sorted_items:
            sid = it["sample_id"]
            if sid in seen_ids:
                raise ValueError(f"Duplicate sample ID {sid} encountered in event {event_id}.")
            seen_ids.add(sid)
            unique_items.append(it)
        return unique_items

    def run_diagnostic(
        self,
        mock_records_by_event: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        preflight_info = self.validate_preflight()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write initial manifest
        initial_manifest = {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.status,
            "approval_status": self.approval_status,
            "evidence_level": self.evidence_level,
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_expected_sha256": self.ckpt_expected_sha256,
            "checkpoint_actual_sha256": self.ckpt_actual_sha256,
            "manifest_sha256": self.manifest_sha256,
            "event_sequence": self.target_events,
            "condition": "matched_bn_no_update_v1",
            "device": self.device_str,
            "start_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(self.output_dir / "diagnostic_initial_manifest.json", "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        if self.synthetic_mock and mock_records_by_event is not None:
            all_records_offline = []
            all_records_adapt = []
            for ev, recs in mock_records_by_event.items():
                for r in recs:
                    all_records_offline.append(r)
                    all_records_adapt.append({
                        "sample_id": r["sample_id"],
                        "event_id": r["event_id"],
                        "xA": r["xA"],
                        "xB": r["xB"],
                        "valid_mask": r["valid_mask"],
                    })
            adapt_dataset = KuroEvaluationDataset(records=all_records_adapt, allowed_events=self.target_events, load_labels=False)
            offline_dataset = KuroEvaluationDataset(records=all_records_offline, allowed_events=self.target_events, load_labels=True)
        else:
            adapt_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.target_events,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=False,
            )
            offline_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.target_events,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=True,
            )

        samples_by_event: Dict[int, List[Dict[str, Any]]] = {ev: [] for ev in self.target_events}
        for idx, item in enumerate(adapt_dataset.samples):
            ev = int(item["event_id"])
            if ev in samples_by_event:
                item_with_idx = dict(item)
                item_with_idx["dataset_index"] = idx
                samples_by_event[ev].append(item_with_idx)

        sorted_samples_by_event = {
            ev: self._resolve_samples_for_event(ev, items)
            for ev, items in samples_by_event.items()
        }

        cond_id = "matched_bn_no_update_v1"
        cond_dir = self.output_dir / cond_id
        cond_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running Diagnostic Condition: {cond_id} ...")
        event_summaries = []
        t0_cond = time.perf_counter()

        for ev_idx, event_id in enumerate(self.target_events):
            ev_items = sorted_samples_by_event[event_id]
            event_dir = cond_dir / f"event_{event_id}"
            event_dir.mkdir(parents=True, exist_ok=True)

            # Clean per-event model re-instantiation
            model, _ = self._instantiate_model()
            adapter = MatchedBNNoUpdateAdapter(model=model, device=self.device_str)

            batches = [ev_items[i : i + self.batch_size] for i in range(0, len(ev_items), self.batch_size)]
            frozen_predictions: Dict[str, np.ndarray] = {}
            step_records: List[Dict[str, Any]] = []
            completed_sample_ids: List[str] = []

            progress_file = event_dir / "adaptation_progress.json"

            try:
                for b_idx, b_items in enumerate(batches):
                    b_ids = [it["sample_id"] for it in b_items]

                    b_indices = [it["dataset_index"] for it in b_items]
                    batch_data = [adapt_dataset[idx] for idx in b_indices]

                    xA = torch.stack([d["xA"] for d in batch_data]).to(self.device_str)
                    xB = torch.stack([d["xB"] for d in batch_data]).to(self.device_str)
                    m = torch.stack([d["valid_mask"] for d in batch_data]).to(self.device_str)

                    step_res = adapter.adapt_and_predict(
                        xA, xB, sar_valid_mask=m, sample_ids=b_ids, event_ids=[event_id] * len(b_ids)
                    )

                    preds = step_res["primary_prediction"]["predicted_class"].cpu().numpy()
                    for idx_s, sid in enumerate(b_ids):
                        frozen_predictions[sid] = preds[idx_s]

                    completed_sample_ids.extend(b_ids)
                    adapt_meta = step_res["adaptation_metadata"]

                    rec = {
                        "step_index": b_idx + 1,
                        "event_id": event_id,
                        "condition_id": cond_id,
                        "batch_size": len(b_items),
                        "sample_ids": b_ids,
                        "action": adapt_meta["action"],
                        "update_applied": adapt_meta["update_applied"],
                        "loss": adapt_meta["loss"],
                        "valid_pixels": adapt_meta["valid_pixels"],
                        "selected_pixels": adapt_meta["selected_pixels"],
                        "selected_pixel_fraction": adapt_meta["selected_pixel_fraction"],
                        "failure_reason": adapt_meta["failure_reason"],
                        "batch_time_ms": adapt_meta["update_time_ms"],
                    }
                    step_records.append(rec)

                    # Durable per-batch progress
                    progress_snapshot = {
                        "status": "RUNNING_DIAGNOSTIC",
                        "condition_id": cond_id,
                        "event_id": event_id,
                        "batches_completed": b_idx + 1,
                        "total_expected_batches": len(batches),
                        "samples_completed": len(completed_sample_ids),
                        "updates_applied": adapter.num_updates_applied,
                        "updates_skipped": adapter.num_updates_skipped,
                        "updates_failed": adapter.num_updates_failed,
                        "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(progress_file, "w", encoding="utf-8") as pf:
                        json.dump(progress_snapshot, pf, indent=2)

                # Persist predictions and step records strictly before offline GT access
                pred_file = event_dir / "frozen_predictions.npz"
                step_file = event_dir / "step_records.json"
                np.savez_compressed(pred_file, **frozen_predictions)
                with open(step_file, "w", encoding="utf-8") as sf:
                    json.dump(step_records, sf, indent=2)

            except Exception as e:
                err_record = {
                    "status": "FAILED",
                    "condition_id": cond_id,
                    "event_id": event_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "batches_completed": len(step_records),
                    "failed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                with open(event_dir / "diagnostic_failure.json", "w", encoding="utf-8") as ff:
                    json.dump(err_record, ff, indent=2)
                raise

            # Offline Double-Masked Ground Truth Evaluation
            ev_tp = 0
            ev_fp = 0
            ev_fn = 0
            ev_tn = 0
            ev_valid_pixels = 0

            for b_items in batches:
                b_indices = [it["dataset_index"] for it in b_items]
                b_ds_items = [offline_dataset[idx] for idx in b_indices]
                batch_labels = [d["label"] for d in b_ds_items]
                batch_masks = [d["valid_mask"] for d in b_ds_items]
                batch_ids = [it["sample_id"] for it in b_items]

                for sid, gt_lbl, s_mask in zip(batch_ids, batch_labels, batch_masks):
                    p_cls = frozen_predictions[sid]
                    if isinstance(gt_lbl, torch.Tensor):
                        gt_arr = gt_lbl.cpu().numpy()
                    else:
                        gt_arr = np.asarray(gt_lbl)
                    if isinstance(s_mask, torch.Tensor):
                        mask_arr = s_mask.cpu().numpy()
                    else:
                        mask_arr = np.asarray(s_mask)

                    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                        mask_arr = mask_arr.squeeze(0)

                    # Double mask: sensory valid (1) AND annotated (gt != 3)
                    eval_mask = (mask_arr == 1) & (gt_arr != 3)
                    n_eval = int(np.sum(eval_mask))
                    ev_valid_pixels += n_eval

                    if n_eval > 0:
                        gt_flood = (gt_arr == 2)
                        pred_flood = (p_cls == 2)

                        tp = int(np.sum(pred_flood & gt_flood & eval_mask))
                        fp = int(np.sum(pred_flood & (~gt_flood) & eval_mask))
                        fn = int(np.sum((~pred_flood) & gt_flood & eval_mask))
                        tn = int(np.sum((~pred_flood) & (~gt_flood) & eval_mask))

                        assert tp + fp + fn + tn == n_eval
                        ev_tp += tp
                        ev_fp += fp
                        ev_fn += fn
                        ev_tn += tn

            ev_precision = ev_tp / (ev_tp + ev_fp) if (ev_tp + ev_fp) > 0 else 0.0
            ev_recall = ev_tp / (ev_tp + ev_fn) if (ev_tp + ev_fn) > 0 else 0.0
            ev_f1 = (2 * ev_precision * ev_recall) / (ev_precision + ev_recall) if (ev_precision + ev_recall) > 0 else 0.0
            ev_iou = ev_tp / (ev_tp + ev_fp + ev_fn) if (ev_tp + ev_fp + ev_fn) > 0 else 0.0

            ev_metrics = {
                "condition_id": cond_id,
                "event_id": event_id,
                "sample_count": len(ev_items),
                "batch_sizes": [len(b) for b in batches],
                "valid_pixels": ev_valid_pixels,
                "tp": ev_tp,
                "fp": ev_fp,
                "fn": ev_fn,
                "tn": ev_tn,
                "precision": ev_precision,
                "recall": ev_recall,
                "f1": ev_f1,
                "iou": ev_iou,
                "updates_applied": adapter.num_updates_applied,
                "updates_skipped": adapter.num_updates_skipped,
                "updates_failed": adapter.num_updates_failed,
            }
            with open(event_dir / "event_metrics.json", "w", encoding="utf-8") as mf:
                json.dump(ev_metrics, mf, indent=2)

            event_summaries.append(ev_metrics)
            print(f"  Event {event_id} ({len(ev_items)} samples): F1={ev_f1:.4f}, IoU={ev_iou:.4f}, Updates={adapter.num_updates_applied}/{len(batches)}")

        elapsed_cond = time.perf_counter() - t0_cond

        # Pooled metrics
        pooled_tp = sum(e["tp"] for e in event_summaries)
        pooled_fp = sum(e["fp"] for e in event_summaries)
        pooled_fn = sum(e["fn"] for e in event_summaries)
        pooled_tn = sum(e["tn"] for e in event_summaries)
        pooled_vp = sum(e["valid_pixels"] for e in event_summaries)

        pooled_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        pooled_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        pooled_f1 = (2 * pooled_prec * pooled_rec) / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = pooled_tp / (pooled_tp + pooled_fp + pooled_fn) if (pooled_tp + pooled_fp + pooled_fn) > 0 else 0.0

        macro_f1 = float(np.mean([e["f1"] for e in event_summaries]))
        macro_iou = float(np.mean([e["iou"] for e in event_summaries]))
        macro_prec = float(np.mean([e["precision"] for e in event_summaries]))
        macro_rec = float(np.mean([e["recall"] for e in event_summaries]))

        cond_summary = {
            "condition_id": cond_id,
            "evidence_level": self.evidence_level,
            "events_evaluated": self.target_events,
            "total_samples": sum(e["sample_count"] for e in event_summaries),
            "total_batches": sum(len(e["batch_sizes"]) for e in event_summaries),
            "pooled_metrics": {
                "tp": pooled_tp,
                "fp": pooled_fp,
                "fn": pooled_fn,
                "tn": pooled_tn,
                "valid_pixels": pooled_vp,
                "precision": pooled_prec,
                "recall": pooled_rec,
                "f1": pooled_f1,
                "iou": pooled_iou,
            },
            "macro_metrics": {
                "macro_f1": macro_f1,
                "macro_iou": macro_iou,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
            },
            "adaptation_behavior": {
                "updates_applied": 0,
                "updates_skipped": 0,
                "updates_failed": 0,
                "elapsed_seconds": elapsed_cond,
            },
            "event_rows": event_summaries,
        }
        with open(cond_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(cond_summary, f, indent=2)

        # Final completion status
        completion_status = {
            "status": "COMPLETED",
            "evidence_level": self.evidence_level,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_cond,
        }
        with open(self.output_dir / "completion_status.json", "w", encoding="utf-8") as f:
            json.dump(completion_status, f, indent=2)

        print("\n================================================================================")
        print(" ChangeGuard Week 12 Matched BN Diagnostic Completed Successfully!")
        print(f" Macro F1: {macro_f1:.4f} | Pooled F1: {pooled_f1:.4f} | Elapsed: {elapsed_cond:.2f}s")
        print("================================================================================")

        return cond_summary


def parse_args():
    parser = argparse.ArgumentParser(description="ChangeGuard Week 12 Matched BN Diagnostic Runner")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/changeguard/week12_bn_diagnostic_dev6_v1.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--synthetic_mock", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = Week12BNDiagnosticRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        synthetic_mock=args.synthetic_mock,
        allow_overwrite=args.allow_overwrite,
    )

    if args.validate_only:
        preflight = runner.validate_preflight()
        print("PREFLIGHT PASSED:", json.dumps(preflight, indent=2))
        sys.exit(0)

    runner.run_diagnostic()


if __name__ == "__main__":
    main()
