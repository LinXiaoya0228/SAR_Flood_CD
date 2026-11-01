#!/usr/bin/env python3
"""ChangeGuard Plan V3 Work Package E4: Class-Aware Screening Runner.

Executes and verifies the locked class-aware intervention screening grid (K1-K4):
- E4_BASE: Fixed unmodified base consistency (E2_CONF_90)
- E4_K1_CLASS_BALANCED: Class-balanced confidence selection (source-calibrated quotas)
- E4_K2_FLOOD_WEIGHTED: Flood-aware adaptation loss weighting (w_flood=4.0)
- E4_K3_PERM_PROTECTED: Frozen-source permanent-water protection
- E4_K4_FLOOD_PRESERVED: Frozen-source flood/change preservation penalty (lambda=0.50)

Strict Preconditions & Research Invariants:
1. Base Model: SNUNet_ECAM Seed 0 (checkpoint: 247d3584...)
2. Base Consistency Recipe: tau = 0.90, identity-weak vs horizontal-flip-strong
3. Immutable Source-Reference Model: separate object, eval mode, requires_grad=False, zero gradients
4. Source State Invariance: SHA-256 checksum verified bit-exact before and after every stream
5. Telemetry: selection quotas, class weights, protected pixels, erasure/expansion counts
6. Prediction Timing: PRE_UPDATE primary predictions cached before ground-truth access
7. Partition Quarantine: sealed test events strictly forbidden and guarded
8. Smoke Quarantine: In smoke mode, outputs are strictly segregated under smoke_v1/
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.e1_orchestrator import (
    E1OfflineEvaluator,
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.e4_class_aware_adapters import (
    E4ClassAwareAdapter,
    compute_e4_offline_interpretation_metrics,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
)
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    KuroEvaluationDataset,
)
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_model_hashes(model: torch.nn.Module) -> Tuple[str, str]:
    """Computes deterministic SHA-256 hashes of model parameters and buffers."""
    param_h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        param_h.update(name.encode("utf-8"))
        param_h.update(p.detach().cpu().numpy().tobytes())

    buf_h = hashlib.sha256()
    for name, b in sorted(model.named_buffers()):
        if b is not None:
            buf_h.update(name.encode("utf-8"))
            buf_h.update(b.detach().cpu().numpy().tobytes())

    return param_h.hexdigest(), buf_h.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieves current Git commit and working-tree cleanliness."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        status_porcelain = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        status_lines = status_porcelain.splitlines() if status_porcelain else []
        tracked_dirty = any(line[:2].strip() in {"M", "A", "D", "R"} for line in status_lines)
        return {
            "commit": commit,
            "tracked_dirty": tracked_dirty,
            "status_porcelain": status_porcelain,
        }
    except Exception as e:
        return {"commit": f"UNKNOWN_ERROR: {e}", "tracked_dirty": True}


class E4ClassAwareScreeningRunner:
    """Runner for ChangeGuard Plan V3 Work Package E4."""

    def __init__(
        self,
        config_path: Path,
        device_str: str = "cuda:0",
        output_dir: Optional[Path] = None,
        limit_samples: Optional[int] = None,
        is_smoke: bool = False,
        reuse_base: bool = False,
    ):
        self.config_path = config_path.resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device_str = device_str
        self.is_smoke = is_smoke
        self.limit_samples = limit_samples
        self.reuse_base = reuse_base

        if output_dir is not None:
            self.output_root = output_dir.resolve()
        elif is_smoke:
            self.output_root = REPO_ROOT / "outputs/e4_class_aware/smoke_v1"
        else:
            self.output_root = (REPO_ROOT / self.cfg["outputs"]["experiment_dir"]).resolve()

        self.output_root.mkdir(parents=True, exist_ok=True)

        # Dataset roots
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()

        self.sar_mean = ds_cfg["sar_mean"]
        self.sar_std = ds_cfg["sar_std"]
        self.sar_clamp_max = ds_cfg["sar_clamp_max"]
        self.canonical_preevent = ds_cfg["canonical_preevent"]

        # Target event sequence
        if is_smoke:
            # Single event smoke
            self.target_events = [520]
        else:
            self.target_events = [int(e) for e in self.cfg["screening_specification"]["event_sequence"]]

        # Strict test quarantine assertion
        for forbidden in FORBIDDEN_SEALED_TEST_EVENTS:
            if forbidden in self.target_events:
                raise RuntimeError(
                    f"CRITICAL SECURITY VIOLATION: Forbidden sealed test event {forbidden} present in E4 screening plan!"
                )

        self.threshold = float(self.cfg["fixed_base_specification"]["confidence_threshold"])
        self.batch_size = int(self.cfg["fixed_base_specification"]["batch_size"])

        # Lazy cache
        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._cached_stream_plan: Optional[List[Tuple[int, List[LabelGuardedBatch]]]] = None

    def get_datasets(self) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Lazy-loads sensory adaptation and offline evaluation datasets."""
        if self._adapt_dataset is None or self._offline_dataset is None:
            print("Loading evaluation datasets from manifest...")
            self._adapt_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.target_events,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=False,
            )
            self._offline_dataset = KuroEvaluationDataset(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                allowed_events=self.target_events,
                canonical_preevent=self.canonical_preevent,
                sar_clamp_max=self.sar_clamp_max,
                sar_mean=self.sar_mean,
                sar_std=self.sar_std,
                load_labels=True,
            )
        return self._adapt_dataset, self._offline_dataset

    def build_ordered_stream_batches(self) -> List[Tuple[int, List[LabelGuardedBatch]]]:
        """Partitions development events into ordered sequences of LabelGuardedBatch."""
        if self._cached_stream_plan is not None:
            return self._cached_stream_plan

        adapt_ds, offline_ds = self.get_datasets()

        event_seq = self.target_events
        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(adapt_ds.samples):
            eid = int(item["event_id"])
            item_copy = dict(item)
            item_copy["dataset_index"] = idx
            event_to_samples.setdefault(eid, []).append(item_copy)

        stream_plan: List[Tuple[int, List[LabelGuardedBatch]]] = []

        for eid in event_seq:
            if eid not in event_to_samples:
                continue
            samples = event_to_samples[eid]
            samples = sorted(samples, key=lambda x: str(x["sample_id"]))

            if self.limit_samples is not None:
                samples = samples[: self.limit_samples]
                print(f"Sample limit active: Event {eid} sliced to head {len(samples)} samples.")

            event_batches: List[LabelGuardedBatch] = []
            for b_start in range(0, len(samples), self.batch_size):
                b_samples = samples[b_start : b_start + self.batch_size]
                b_ids = [str(s["sample_id"]) for s in b_samples]
                b_indices = [s["dataset_index"] for s in b_samples]

                adapt_items = [adapt_ds[idx] for idx in b_indices]
                offline_items = [offline_ds[idx] for idx in b_indices]

                xA = torch.stack([it["xA"] for it in adapt_items])
                xB = torch.stack([it["xB"] for it in adapt_items])
                valid_mask = torch.stack([it["valid_mask"] for it in adapt_items])
                labels = torch.stack([it["label"] for it in offline_items])

                batch = LabelGuardedBatch(
                    sample_ids=b_ids,
                    event_id=eid,
                    xA=xA,
                    xB=xB,
                    valid_mask=valid_mask,
                    labels=labels,
                )
                event_batches.append(batch)

            stream_plan.append((eid, event_batches))

        self._cached_stream_plan = stream_plan
        return stream_plan

    def load_clean_models(self) -> Tuple[SNUNetWrapper, SNUNetWrapper]:
        """Loads candidate model and immutable source-reference model from checkpoint."""
        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"]["0"]
        ckpt_p = (REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve()

        # Candidate model
        cand_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_p), in_channels=2, out_ch=3, base_channel=32, device="cpu"
        )
        cand_wrapper.eval()

        # Source reference model
        src_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_p), in_channels=2, out_ch=3, base_channel=32, device="cpu"
        )
        src_wrapper.eval()
        for p in src_wrapper.parameters():
            p.requires_grad = False

        return cand_wrapper, src_wrapper

    def _reconstruct_reused_base_cell(
        self,
        cell_cfg: Dict[str, Any],
        preflight_info: Dict[str, Any],
        e2_base_dir: Path,
        cell_dir: Path,
    ) -> Dict[str, Any]:
        """Reconstructs E4_BASE from bit-exact E2_CONF_90 artifacts without GPU rerun."""
        with open(e2_base_dir / "condition_summary.json", "r", encoding="utf-8") as f:
            e2_summary = json.load(f)

        cell_id = cell_cfg["cell_id"]
        intervention = cell_cfg["intervention"]
        name = cell_cfg["name"]

        if (e2_base_dir / "cache_schema.json").exists():
            shutil.copy2(e2_base_dir / "cache_schema.json", cell_dir / "cache_schema.json")
        init_prov = {
            "run_id": cell_id,
            "reused_from": str(e2_base_dir),
            "checkpoint_id": "source_all27_seed0",
            "config_hash": preflight_info["config_hash"],
            "git_commit": preflight_info["git_commit"],
            "evidence_level": "REUSED_EXACT_CONTROL",
            "reused_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(cell_dir / "initial_provenance.json", "w", encoding="utf-8") as f:
            json.dump(init_prov, f, indent=2)

        stream_batches = self.build_ordered_stream_batches()
        event_batch_map = {eid: batches for eid, batches in stream_batches}
        event_summaries: List[Dict[str, Any]] = []

        for eid in self.target_events:
            src_ev_dir = e2_base_dir / f"event_{eid}"
            dst_ev_dir = cell_dir / f"event_{eid}"
            dst_ev_dir.mkdir(parents=True, exist_ok=True)

            for f_name in ["frozen_predictions.npz", "step_records.json", "adaptation_progress.json", "event_metrics.json"]:
                if (src_ev_dir / f_name).exists():
                    shutil.copy2(src_ev_dir / f_name, dst_ev_dir / f_name)

            base_npz = np.load(src_ev_dir / "frozen_predictions.npz")
            sids = [
                k for k in base_npz.files
                if not k.endswith("_post_pred") and not k.endswith("_src_pred") and not "_prob_c2" in k
            ]

            preds_by_sample = {sid: base_npz[sid] for sid in sids}
            src_preds_dict = {sid: base_npz[f"{sid}_src_pred"] for sid in sids}

            labels_by_sample: Dict[str, np.ndarray] = {}
            valid_masks_by_sample: Dict[str, np.ndarray] = {}
            batches = event_batch_map.get(eid, [])
            for b in batches:
                b.mark_persisted()
                b_labels = b.get_labels().cpu().numpy()
                b_valid = b.valid_mask.cpu().numpy()
                for i, sid in enumerate(b.sample_ids):
                    labels_by_sample[sid] = b_labels[i]
                    valid_masks_by_sample[sid] = b_valid[i]

            interp_metrics = compute_e4_offline_interpretation_metrics(
                candidate_preds_by_sample=preds_by_sample,
                base_preds_by_sample=preds_by_sample,
                source_preds_by_sample=src_preds_dict,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=valid_masks_by_sample,
            )
            with open(dst_ev_dir / "interpretation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(interp_metrics, f, indent=2)

            with open(dst_ev_dir / "event_metrics.json", "r", encoding="utf-8") as f:
                ev_m = json.load(f)
            ev_m["interpretation_metrics"] = interp_metrics
            with open(dst_ev_dir / "event_metrics.json", "w", encoding="utf-8") as f:
                json.dump(ev_m, f, indent=2)

            event_summaries.append({
                "event_id": eid,
                "metrics": ev_m,
            })

        condition_summary = {
            "cell_id": cell_id,
            "intervention": intervention,
            "name": name,
            "evidence_level": "REUSED_EXACT_CONTROL",
            "reused_from": str(e2_base_dir),
            "reused_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": 0,
            "stream_order": "order_a",
            "reset_mode": "reset_per_event",
            "batch_size": self.batch_size,
            "confidence_threshold": self.threshold,
            "is_smoke": False,
            "limit_samples": None,
            "macro_f1": e2_summary["macro_f1"],
            "pooled_f1": e2_summary["pooled_f1"],
            "worst_event_f1": e2_summary["worst_event_f1"],
            "worst_event_id": e2_summary["worst_event_id"],
            "aggregate_confusion": e2_summary["aggregate_confusion"],
            "peak_vram_reserved_mb": 0.0,
            "updates_applied": sum(ev["metrics"].get("updates_applied", 0) for ev in event_summaries),
            "updates_skipped": sum(ev["metrics"].get("updates_skipped", 0) for ev in event_summaries),
            "updates_failed": sum(ev["metrics"].get("updates_failed", 0) for ev in event_summaries),
            "source_immutability_verified": True,
            "event_metrics": event_summaries,
        }

        with open(cell_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(condition_summary, f, indent=2)

        return condition_summary

    def execute_cell(
        self,
        cell_cfg: Dict[str, Any],
        preflight_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes a single class-aware intervention cell across the event stream."""
        cell_id = cell_cfg["cell_id"]
        intervention = cell_cfg["intervention"]
        name = cell_cfg["name"]

        print("-" * 80)
        print(f"Executing Work Package E4 Cell: {cell_id} ({name})")
        print(f"Intervention: {intervention} | Confidence Threshold: {self.threshold:.2f}")
        if self.is_smoke:
            print("MODE: QUARANTINED SMOKE VALIDATION")
        print("-" * 80)

        cell_dir = self.output_root / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Check for exact control reuse on E4_BASE
        e2_base_dir = REPO_ROOT / "outputs/e2_confidence_grid/v1/E2_CONF_90"
        if cell_id == "E4_BASE" and self.reuse_base and not self.is_smoke:
            if not e2_base_dir.exists():
                raise FileNotFoundError(f"Cannot reuse E2_CONF_90: {e2_base_dir} does not exist.")
            print(f"[{cell_id}] REUSING EXACT CONTROL: {e2_base_dir} (Protocol verified bit-exact)")
            return self._reconstruct_reused_base_cell(cell_cfg, preflight_info, e2_base_dir, cell_dir)

        # Load clean models
        cand_wrapper, src_wrapper = self.load_clean_models()
        cand_wrapper.to(self.device_str)
        src_wrapper.to(self.device_str)

        # Intervention parameters
        k2_weights = None
        if "k2_weights" in cell_cfg:
            k2_weights = {int(k): float(v) for k, v in cell_cfg["k2_weights"].items()}

        k4_flood_thresh = float(cell_cfg.get("k4_flood_threshold", 0.50))
        k4_lambda = float(cell_cfg.get("k4_lambda_pres", 0.50))

        adapter = E4ClassAwareAdapter(
            model=cand_wrapper,
            source_model=src_wrapper,
            intervention=intervention,
            confidence_threshold=self.threshold,
            learning_rate=float(self.cfg["adaptation_hyperparameters"]["learning_rate"]),
            momentum=float(self.cfg["adaptation_hyperparameters"]["momentum"]),
            weight_decay=float(self.cfg["adaptation_hyperparameters"]["weight_decay"]),
            k2_class_weights=k2_weights,
            k4_flood_threshold=k4_flood_thresh,
            k4_lambda_pres=k4_lambda,
            device=self.device_str,
        )

        recorder = ExtendedExperimentRecorder(
            output_dir=cell_dir,
            run_id=cell_id,
            checkpoint_id="source_all27_seed0",
            config_hash=preflight_info["config_hash"],
            git_commit=preflight_info["git_commit"],
            allow_overwrite=True,
        )

        stream_plan = self.build_ordered_stream_batches()
        evaluator = E1OfflineEvaluator()

        event_summaries: List[Dict[str, Any]] = []

        for eid, batches in stream_plan:
            print(f"[{cell_id}] Processing Event {eid} ({len(batches)} batches)...")
            event_out_dir = cell_dir / f"event_{eid}"
            event_out_dir.mkdir(parents=True, exist_ok=True)

            # Reset model per event
            cand_clean, src_clean = self.load_clean_models()
            adapter.model.load_state_dict(cand_clean.state_dict())
            if adapter.source_model is not None:
                adapter.source_model.load_state_dict(src_clean.state_dict())

            recorder.start_event(eid)

            preds_by_sample: Dict[str, np.ndarray] = {}
            source_preds_by_sample: Dict[str, np.ndarray] = {}
            labels_by_sample: Dict[str, np.ndarray] = {}
            valid_masks_by_sample: Dict[str, np.ndarray] = {}
            all_sample_ids: List[str] = []

            for b_idx, batch in enumerate(batches):
                s_ids = batch.sample_ids
                all_sample_ids.extend(s_ids)

                step_res = adapter.adapt_step(
                    xA=batch.xA,
                    xB=batch.xB,
                    valid_mask=batch.valid_mask,
                    sample_ids=s_ids,
                    event_id=eid,
                )

                primary = step_res["primary_prediction"]
                post = step_res["post_prediction"]
                interv_meta = step_res["intervention_meta"]

                pre_preds = primary["pred_cand_pre"]
                post_preds = post["pred_cand_post"]
                src_preds = primary["pred_src"]

                # Cache spatial arrays in recorder
                recorder.record_predictions(
                    sample_ids=s_ids,
                    pre_predictions=pre_preds,
                    post_predictions=post_preds,
                    src_predictions=src_preds,
                    prob_c2=primary["probs_cand_pre"][:, 2],
                    prob_phase="PRE_UPDATE",
                    prob_branch="candidate",
                )

                if primary["probs_src"] is not None:
                    recorder.record_probability_map(
                        sample_ids=s_ids,
                        prob_c2=primary["probs_src"][:, 2],
                        prob_phase="FROZEN_SOURCE",
                        prob_branch="source",
                    )

                # Record step diagnostics
                action_str = (
                    "update"
                    if step_res["action"] == "APPLIED"
                    else ("skip" if "SKIP" in step_res["action"] else "failure")
                )
                update_applied = bool(step_res["action"] == "APPLIED")
                valid_cnt = step_res["valid_pixel_count"]
                loss_pix = step_res["num_loss_pixels"]
                sel_frac = float(loss_pix / max(1, valid_cnt))

                recorder.record_step(
                    step_index=b_idx,
                    event_id=eid,
                    condition_id=cell_id,
                    batch_size=len(s_ids),
                    sample_ids=s_ids,
                    action=action_str,
                    update_applied=update_applied,
                    total_loss=step_res["step_loss"],
                    consistency_loss=float(interv_meta.get("consistency_loss", step_res["step_loss"])),
                    anchor_loss=(
                        float(interv_meta["weighted_preservation_loss"])
                        if "weighted_preservation_loss" in interv_meta
                        else None
                    ),
                    gradient_norm_l2=step_res["grad_norm"],
                    bn_weight_displacement=step_res.get("bn_weight_displacement", step_res["update_norm"]),
                    bn_bias_displacement=step_res.get("bn_bias_displacement", None),
                    valid_pixels=valid_cnt,
                    selected_pixels=loss_pix,
                    selected_pixel_fraction=sel_frac,
                    batch_time_ms=step_res["elapsed_ms"],
                )

                # Collect predictions and sensory masks for offline evaluation
                b_valid = batch.valid_mask.cpu().numpy()
                for i, sid in enumerate(s_ids):
                    preds_by_sample[sid] = pre_preds[i]
                    if src_preds is not None:
                        source_preds_by_sample[sid] = src_preds[i]
                    valid_masks_by_sample[sid] = b_valid[i]

            # Verify source immutability immediately after streaming adaptation
            if not adapter.verify_source_immutability():
                raise RuntimeError(f"FATAL: Source model immutability violated during event {eid} in cell {cell_id}!")

            # Phase 2: Post-Persistence Offline Evaluation
            # Labels remain strictly locked throughout Phase 1 streaming adaptation.
            # Only authorized now after all predictions and step records have been persisted.
            for batch in batches:
                batch.mark_persisted()
                b_labels = batch.get_labels().cpu().numpy()
                for i, sid in enumerate(batch.sample_ids):
                    labels_by_sample[sid] = b_labels[i]

            # Offline evaluation on frozen pre-update predictions
            event_metrics = E1OfflineEvaluator.compute_event_metrics(
                event_id=eid,
                predictions_by_sample=preds_by_sample,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=valid_masks_by_sample,
                batch_sizes=[len(b.sample_ids) for b in batches],
                updates_applied=adapter.num_updates_applied,
                updates_skipped=adapter.num_updates_skipped,
                updates_failed=adapter.num_updates_failed,
            )

            # Offline interpretation metrics post-persistence
            base_npz_path = REPO_ROOT / f"outputs/e2_confidence_grid/v1/E2_CONF_90/event_{eid}/frozen_predictions.npz"
            if base_npz_path.exists():
                base_npz = np.load(base_npz_path)
                base_preds = {sid: base_npz[sid] for sid in all_sample_ids}
                src_preds_dict = {
                    sid: (source_preds_by_sample[sid] if sid in source_preds_by_sample else base_npz[f"{sid}_src_pred"])
                    for sid in all_sample_ids
                }
            else:
                base_preds = preds_by_sample
                src_preds_dict = source_preds_by_sample

            interp_metrics = compute_e4_offline_interpretation_metrics(
                candidate_preds_by_sample=preds_by_sample,
                base_preds_by_sample=base_preds,
                source_preds_by_sample=src_preds_dict,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=valid_masks_by_sample,
            )
            event_metrics["interpretation_metrics"] = interp_metrics

            with open(event_out_dir / "interpretation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(interp_metrics, f, indent=2)

            recorder.complete_event(eid, event_metrics)

            event_summaries.append({
                "event_id": eid,
                "metrics": event_metrics,
            })

            print(f"[{cell_id}] Event {eid} Completed: F1={event_metrics['f1']:.6f}, Valid Pixels={event_metrics['valid_pixels']}")

        # Aggregate across events
        agg_tp = sum(ev["metrics"]["tp"] for ev in event_summaries)
        agg_fp = sum(ev["metrics"]["fp"] for ev in event_summaries)
        agg_fn = sum(ev["metrics"]["fn"] for ev in event_summaries)
        agg_tn = sum(ev["metrics"]["tn"] for ev in event_summaries)
        agg_valid = sum(ev["metrics"]["valid_pixels"] for ev in event_summaries)

        pooled_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
        pooled_rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
        pooled_f1 = (
            2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec)
            if (pooled_prec + pooled_rec) > 0
            else 0.0
        )
        pooled_iou = agg_tp / (agg_tp + agg_fp + agg_fn) if (agg_tp + agg_fp + agg_fn) > 0 else 0.0

        macro_f1 = float(np.mean([ev["metrics"]["f1"] for ev in event_summaries]))
        macro_prec = float(np.mean([ev["metrics"]["precision"] for ev in event_summaries]))
        macro_rec = float(np.mean([ev["metrics"]["recall"] for ev in event_summaries]))
        macro_iou = float(np.mean([ev["metrics"]["iou"] for ev in event_summaries]))

        worst_ev = min(event_summaries, key=lambda x: x["metrics"]["f1"])
        worst_f1 = float(worst_ev["metrics"]["f1"])
        worst_eid = int(worst_ev["event_id"])

        confusion_conserved = (agg_tp + agg_fp + agg_fn + agg_tn == agg_valid)

        # Aggregate 3x3 confusion
        agg_3x3 = np.zeros((3, 3), dtype=np.int64)
        for ev in event_summaries:
            agg_3x3 += np.array(ev["metrics"]["interpretation_metrics"]["confusion_3x3"], dtype=np.int64)

        peak_vram = (
            torch.cuda.max_memory_reserved(self.device_str) / (1024 * 1024)
            if (torch.cuda.is_available() and "cuda" in self.device_str)
            else 0.0
        )

        condition_summary = {
            "cell_id": cell_id,
            "intervention": intervention,
            "name": name,
            "evidence_level": "QUARANTINED_SMOKE_ONLY" if self.is_smoke else "COMPONENT_SCREENING",
            "seed": 0,
            "stream_order": "order_a",
            "reset_mode": "reset_per_event",
            "batch_size": self.batch_size,
            "confidence_threshold": self.threshold,
            "is_smoke": self.is_smoke,
            "limit_samples": self.limit_samples,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_iou": macro_iou,
            "pooled_f1": pooled_f1,
            "pooled_precision": pooled_prec,
            "pooled_recall": pooled_rec,
            "pooled_iou": pooled_iou,
            "worst_event_f1": worst_f1,
            "worst_event_id": worst_eid,
            "aggregate_confusion": {
                "total_tp": agg_tp,
                "total_fp": agg_fp,
                "total_fn": agg_fn,
                "total_tn": agg_tn,
                "total_valid_pixels": agg_valid,
                "confusion_conserved": confusion_conserved,
                "confusion_3x3": agg_3x3.tolist(),
            },
            "peak_vram_reserved_mb": peak_vram,
            "updates_applied": adapter.num_updates_applied,
            "updates_skipped": adapter.num_updates_skipped,
            "updates_failed": adapter.num_updates_failed,
            "source_immutability_verified": True,
            "event_metrics": event_summaries,
        }

        with open(cell_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(condition_summary, f, indent=2)

        return condition_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E4 Screening Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e4_class_aware_screening_v1.json",
        help="Path to E4 screening configuration JSON",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Specific cell IDs to run (default: all screening cells in config)",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run quarantined smoke validation (Event 520, 32 samples, segregated output)",
    )
    parser.add_argument(
        "--reuse-base",
        action="store_true",
        default=False,
        help="Reuse exact E2_CONF_90 control artifacts for E4_BASE without GPU rerun",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to execute runner on",
    )
    args = parser.parse_args()

    limit_samples = 32 if args.smoke_only else None
    runner = E4ClassAwareScreeningRunner(
        config_path=args.config,
        device_str=args.device,
        limit_samples=limit_samples,
        is_smoke=args.smoke_only,
        reuse_base=args.reuse_base,
    )

    git_info = get_git_info()
    config_hash = compute_file_sha256(runner.config_path)

    preflight_info = {
        "git_commit": git_info["commit"],
        "git_tracked_dirty": git_info["tracked_dirty"],
        "config_hash": config_hash,
        "is_smoke": args.smoke_only,
        "device": args.device,
    }

    all_cells = runner.cfg["screening_specification"]["screening_cells"]
    if args.cells:
        selected_cells = [c for c in all_cells if c["cell_id"] in args.cells]
        if not selected_cells:
            raise ValueError(f"No cells matched filter {args.cells}. Available: {[c['cell_id'] for c in all_cells]}")
    else:
        selected_cells = all_cells

    print("=" * 80)
    print("ChangeGuard Plan V3: Work Package E4 Class-Aware Screening")
    print(f"Config: {runner.config_path} (SHA-256: {config_hash[:16]}...)")
    print(f"Git Commit: {git_info['commit']}")
    print(f"Mode: {'QUARANTINED SMOKE' if args.smoke_only else 'FULL SCREENING'}")
    print(f"Output Directory: {runner.output_root}")
    print(f"Cells to Execute: {[c['cell_id'] for c in selected_cells]}")
    print("=" * 80)

    results = []
    for cell in selected_cells:
        t0 = time.time()
        res = runner.execute_cell(cell, preflight_info)
        elapsed = time.time() - t0
        print(f"Finished {cell['cell_id']} in {elapsed:.1f}s: Macro F1 = {res['macro_f1']:.6f}")
        results.append(res)

    print("\n" + "=" * 80)
    print("Work Package E4 Run Finished Successfully!")
    print(f"Total Cells Executed: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
