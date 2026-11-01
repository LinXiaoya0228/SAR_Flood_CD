#!/usr/bin/env python3
"""ChangeGuard Plan V3 Work Package E3: Source-Anchor Screening Runner.

Executes and verifies the locked 9-cell source-anchor screening grid:
- Probability KL Anchor: lambda in {0.10, 0.50, 1.00}
- Class-Mean Centered Logit Anchor: lambda in {0.10, 0.50, 1.00}
- Penultimate ECAM Feature Anchor: lambda in {0.10, 0.50, 1.00}

Strict Preconditions & Research Invariants:
1. Base Model: SNUNet_ECAM Seed 0 (checkpoint: 247d3584...)
2. Base Consistency Recipe: tau = 0.90, identity-weak vs horizontal-flip-strong
3. Immutable Source-Reference Model: separate object, eval mode, requires_grad=False, zero gradients
4. Source State Invariance: SHA-256 checksum verified bit-exact before and after every stream
5. Expanded Gradient Telemetry: raw/weighted norms, weighted ratio, cosine conflict, update norm
6. Prediction Timing: PRE_UPDATE primary predictions cached before ground-truth access
7. Partition Quarantine: 10 sealed test events strictly forbidden and guarded
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
from src.changeguard.e3_anchor_adapters import (
    E3SourceAnchorAdapter,
    compute_offline_interpretation_metrics,
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


class E3SourceAnchorScreeningRunner:
    """Runner for ChangeGuard Plan V3 Work Package E3."""

    def __init__(
        self,
        config_path: Path,
        device_str: str = "cuda:0",
        output_dir: Optional[Path] = None,
        smoke_mode: bool = False,
        limit_samples: Optional[int] = None,
        target_events: Optional[List[int]] = None,
    ):
        self.config_path = config_path.resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device_str = device_str
        self.smoke_mode = bool(smoke_mode)
        self.limit_samples = int(limit_samples) if limit_samples is not None else None

        out_root = (
            output_dir
            if output_dir is not None
            else REPO_ROOT / self.cfg["outputs"]["experiment_dir"]
        )
        self.output_root = out_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.batch_size = int(self.cfg["screening_specification"].get("batch_size", 16))
        self.threshold = float(self.cfg["screening_specification"].get("confidence_threshold", 0.90))

        if target_events is not None:
            self.target_events = [int(e) for e in target_events]
        elif self.smoke_mode:
            self.target_events = [520]
            if self.limit_samples is None:
                self.limit_samples = 32
        else:
            self.target_events = [int(e) for e in self.cfg["screening_specification"]["event_sequence"]]

        self.sar_clamp_max = float(ds_cfg.get("sar_clamp_max", 0.15))
        self.sar_mean = list(ds_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = list(ds_cfg.get("sar_std", [0.0427, 0.0215]))
        self.canonical_preevent = str(ds_cfg.get("canonical_preevent", "pre_event_2"))

        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._cached_stream_plan: Optional[List[Tuple[int, List[LabelGuardedBatch]]]] = None

    def run_preflights(self) -> Dict[str, Any]:
        """Fail-closed preflight validation of code, assets, and partitions."""
        print("=" * 80)
        print("ChangeGuard Plan V3 E3: Executing Fail-Closed Preflights...")
        print("=" * 80)

        # 1. Sealed-test check
        for eid in self.target_events:
            if eid in FORBIDDEN_SEALED_TEST_EVENTS:
                raise RuntimeError(
                    f"CRITICAL RESEARCH INTEGRITY VIOLATION: Sealed test event {eid} in target events!"
                )

        # 2. Dataset files
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest path not found: {self.manifest_path}")

        # 3. Checkpoint
        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"]["0"]
        ckpt_p = (REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve()
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Source checkpoint for seed 0 not found: {ckpt_p}")
        actual_sha = compute_file_sha256(ckpt_p)
        exp_sha = ckpt_cfg["expected_sha256"]
        if actual_sha != exp_sha:
            raise ValueError(f"Checkpoint seed 0 SHA-256 mismatch: {actual_sha} != {exp_sha}")

        # 4. Git info
        git_info = get_git_info()
        print(f"Git Commit: {git_info['commit']}")
        print(f"Tracked Dirty: {git_info['tracked_dirty']}")
        print(f"Target Events: {self.target_events}")
        print(f"Smoke Mode: {self.smoke_mode} (limit={self.limit_samples})")
        print(f"Device: {self.device_str}")
        print("All fail-closed preflights PASSED cleanly.")
        print("=" * 80)

        return {
            "git_commit": git_info["commit"],
            "checkpoint_sha256": actual_sha,
            "config_hash": compute_file_sha256(self.config_path),
        }

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

        # 1. Candidate model
        cand_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_p), in_channels=2, out_ch=3, base_channel=32, device="cpu"
        )

        # 2. Source model (distinct object)
        src_wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_p), in_channels=2, out_ch=3, base_channel=32, device="cpu"
        )

        src_wrapper.eval()
        src_wrapper.requires_grad_(False)

        return cand_wrapper, src_wrapper

    def execute_cell(
        self,
        cell_cfg: Dict[str, Any],
        preflight_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes a single anchor cell across the event stream."""
        cell_id = cell_cfg["cell_id"]
        family = cell_cfg["family"]
        lambda_val = float(cell_cfg["lambda_val"])
        name = cell_cfg["name"]

        print("-" * 80)
        print(f"Executing Work Package E3 Cell: {cell_id} ({name})")
        print(f"Family: {family} | Lambda: {lambda_val:.2f} | Confidence Threshold: {self.threshold:.2f}")
        print("-" * 80)

        cell_dir = self.output_root / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Load clean models
        cand_wrapper, src_wrapper = self.load_clean_models()
        cand_wrapper.to(self.device_str)
        src_wrapper.to(self.device_str)

        adapter = E3SourceAnchorAdapter(
            model=cand_wrapper,
            source_model=src_wrapper,
            family=family,
            lambda_val=lambda_val,
            learning_rate=float(self.cfg["adaptation_hyperparameters"]["learning_rate"]),
            momentum=float(self.cfg["adaptation_hyperparameters"]["momentum"]),
            weight_decay=float(self.cfg["adaptation_hyperparameters"]["weight_decay"]),
            confidence_threshold=self.threshold,
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
        start_wall_time = time.time()

        # Track VRAM
        if torch.cuda.is_available() and "cuda" in self.device_str:
            torch.cuda.reset_peak_memory_stats()

        for eid, batches in stream_plan:
            print(f"[{cell_id}] Processing Event {eid} ({len(batches)} batches)...")
            event_out_dir = recorder.start_event(eid)

            adapter.reset_to_source()
            adapter.verify_source_immutability()

            # Adaptation loop
            for b_idx, batch in enumerate(batches):
                adapt_res = adapter.adapt_and_predict(
                    xA=batch.xA,
                    xB=batch.xB,
                    sar_valid_mask=batch.valid_mask,
                    sample_ids=batch.sample_ids,
                    event_ids=[eid] * len(batch.sample_ids),
                )

                primary_pred = adapt_res["primary_prediction"]
                step_telem = adapt_res["step_telemetry"]

                # Cache primary PRE_UPDATE spatial predictions
                probs_c2 = primary_pred["probabilities"][:, 2, :, :] if primary_pred["probabilities"].shape[1] > 2 else None
                recorder.record_predictions(
                    sample_ids=batch.sample_ids,
                    pre_predictions=primary_pred["predicted_class"],
                    prob_c2=probs_c2,
                    prob_phase="PRE_UPDATE",
                    prob_branch="candidate",
                )

                recorder.record_step(
                    step_index=step_telem["step_index"],
                    event_id=eid,
                    condition_id=cell_id,
                    batch_size=len(batch.sample_ids),
                    sample_ids=batch.sample_ids,
                    action=step_telem["action"],
                    update_applied=step_telem["update_applied"],
                    total_loss=step_telem["loss"],
                    consistency_loss=step_telem["consistency_loss"],
                    anchor_loss=step_telem["anchor_loss"],
                    gradient_norm_l2=step_telem["gradient_norm_total"],
                    bn_weight_displacement=step_telem["bn_weight_displacement"],
                    bn_bias_displacement=step_telem["bn_bias_displacement"],
                    valid_pixels=step_telem["valid_pixels"],
                    selected_pixels=step_telem["selected_pixels"],
                    selected_pixel_fraction=step_telem["selected_pixel_fraction"],
                    failure_reason=step_telem["failure_reason"],
                    batch_time_ms=step_telem["batch_time_ms"],
                    loss_anc_weighted=step_telem["anchor_loss_weighted"],
                    gradient_norm_consistency=step_telem["gradient_norm_consistency"],
                    gradient_norm_anchor_raw=step_telem["gradient_norm_anchor_raw"],
                    gradient_norm_anchor_weighted=step_telem["gradient_norm_anchor_weighted"],
                    gradient_norm_ratio_weighted=step_telem["gradient_norm_ratio_weighted"],
                    gradient_cosine_conflict=step_telem["gradient_cosine_conflict"],
                    update_norm_l2=step_telem["update_norm_l2"],
                    prob_disagreement_pre=step_telem["prob_disagreement_pre"],
                )

            # Verify source immutability at stream end
            adapter.verify_source_immutability()

            # Offline evaluation with double masking
            all_sample_ids = []
            labels_by_sample = {}
            valid_masks_by_sample = {}
            for b in batches:
                b.mark_persisted()
                lbls = b.get_labels().cpu().numpy()
                vms = b.valid_mask.cpu().numpy()
                for i, sid in enumerate(b.sample_ids):
                    all_sample_ids.append(sid)
                    labels_by_sample[sid] = lbls[i]
                    valid_masks_by_sample[sid] = vms[i]

            preds_by_sample = {sid: recorder.frozen_spatial_arrays[sid] for sid in all_sample_ids}

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

            # Offline interpretation metrics strictly post-persistence
            base_npz_path = REPO_ROOT / f"outputs/e2_confidence_grid/v1/E2_CONF_90/event_{eid}/frozen_predictions.npz"
            if base_npz_path.exists():
                base_npz = np.load(base_npz_path)
                base_preds = {sid: base_npz[sid] for sid in all_sample_ids}
                src_preds = {sid: base_npz[f"{sid}_src_pred"] for sid in all_sample_ids}
                interp_metrics = compute_offline_interpretation_metrics(
                    candidate_preds_by_sample=preds_by_sample,
                    base_preds_by_sample=base_preds,
                    source_preds_by_sample=src_preds,
                    labels_by_sample=labels_by_sample,
                    valid_masks_by_sample=valid_masks_by_sample,
                )
                event_metrics["interpretation_metrics"] = interp_metrics
                interp_path = event_out_dir / "interpretation_metrics.json"
                with open(interp_path, "w", encoding="utf-8") as f:
                    json.dump(interp_metrics, f, indent=2)
            else:
                interp_metrics = None

            # Complete event in recorder
            recorder.complete_event(eid, event_metrics)

            event_summaries.append({
                "event_id": eid,
                "metrics": event_metrics,
                "step_count": adapter.step_count,
                "updates_applied": adapter.num_updates_applied,
                "updates_skipped": adapter.num_updates_skipped,
                "updates_failed": adapter.num_updates_failed,
            })

            print(
                f"[{cell_id}] Event {eid} Complete: F1 = {event_metrics['f1']:.4f}, "
                f"Precision = {event_metrics['precision']:.4f}, Recall = {event_metrics['recall']:.4f}"
            )

        elapsed_sec = time.time() - start_wall_time
        peak_vram_alloc_mb = 0.0
        peak_vram_res_mb = 0.0
        if torch.cuda.is_available() and "cuda" in self.device_str:
            peak_vram_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            peak_vram_res_mb = float(torch.cuda.max_memory_reserved() / (1024 * 1024))

        adapter.close()
        del adapter
        del cand_wrapper
        del src_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute aggregate condition summary
        f1_list = [ev["metrics"]["f1"] for ev in event_summaries]
        mean_macro_f1 = float(np.mean(f1_list))
        worst_event_f1 = float(np.min(f1_list))

        tot_tp = sum(ev["metrics"]["tp"] for ev in event_summaries)
        tot_fp = sum(ev["metrics"]["fp"] for ev in event_summaries)
        tot_fn = sum(ev["metrics"]["fn"] for ev in event_summaries)
        tot_tn = sum(ev["metrics"]["tn"] for ev in event_summaries)
        tot_valid = sum(ev["metrics"]["valid_pixels"] for ev in event_summaries)

        pooled_prec = float(tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
        pooled_rec = float(tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
        pooled_f1 = (
            float(2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec))
            if (pooled_prec + pooled_rec) > 0
            else 0.0
        )

        # Aggregate offline interpretation metrics if available
        agg_interp = None
        if all("interpretation_metrics" in ev["metrics"] and ev["metrics"]["interpretation_metrics"] is not None for ev in event_summaries):
            agg_valid = sum(ev["metrics"]["interpretation_metrics"]["total_valid_pixels"] for ev in event_summaries)
            agg_disagree_src = sum(ev["metrics"]["interpretation_metrics"]["source_candidate_disagreement"]["count"] for ev in event_summaries)
            agg_disagree_3class = sum(ev["metrics"]["interpretation_metrics"]["source_candidate_disagreement"]["three_class_count"] for ev in event_summaries)
            agg_agree_src = sum(ev["metrics"]["interpretation_metrics"]["source_agreement"]["count"] for ev in event_summaries)
            agg_agree_base = sum(ev["metrics"]["interpretation_metrics"]["base_consistency_agreement"]["count"] for ev in event_summaries)

            agg_harm_rec = sum(ev["metrics"]["interpretation_metrics"]["harmful_base_error_recovery"]["recovered_count"] for ev in event_summaries)
            agg_harm_denom = sum(ev["metrics"]["interpretation_metrics"]["harmful_base_error_recovery"]["denominator_harmful_base_errors"] for ev in event_summaries)

            agg_ben_ret = sum(ev["metrics"]["interpretation_metrics"]["beneficial_adaptation_retention"]["retained_count"] for ev in event_summaries)
            agg_ben_denom = sum(ev["metrics"]["interpretation_metrics"]["beneficial_adaptation_retention"]["denominator_beneficial_adaptations"] for ev in event_summaries)

            agg_ben_loss = sum(ev["metrics"]["interpretation_metrics"]["beneficial_adaptation_loss"]["lost_count"] for ev in event_summaries)

            agg_reg_base = sum(ev["metrics"]["interpretation_metrics"]["regression_from_base"]["regressed_count"] for ev in event_summaries)
            agg_base_corr_denom = sum(ev["metrics"]["interpretation_metrics"]["regression_from_base"]["denominator_base_correct"] for ev in event_summaries)

            agg_corr_base = sum(ev["metrics"]["interpretation_metrics"]["correction_over_base"]["corrected_count"] for ev in event_summaries)
            agg_base_wrong_denom = sum(ev["metrics"]["interpretation_metrics"]["correction_over_base"]["denominator_base_wrong"] for ev in event_summaries)

            prob_disagrees = []
            for ev in event_summaries:
                ev_id = ev["event_id"]
                step_file = cell_dir / f"event_{ev_id}" / "step_records.json"
                if step_file.exists():
                    with open(step_file) as sf:
                        st_data = json.load(sf)
                        for st in st_data:
                            if st.get("prob_disagreement_pre") is not None:
                                prob_disagrees.append(st["prob_disagreement_pre"])
            mean_prob_disagree = float(np.mean(prob_disagrees)) if prob_disagrees else 0.0

            agg_interp = {
                "total_valid_pixels": agg_valid,
                "mean_prob_disagreement_pre": mean_prob_disagree,
                "source_candidate_disagreement": {
                    "count": agg_disagree_src,
                    "rate_rel_valid": float(agg_disagree_src / max(1, agg_valid)),
                    "three_class_count": agg_disagree_3class,
                    "three_class_rate_rel_valid": float(agg_disagree_3class / max(1, agg_valid)),
                },
                "source_agreement": {
                    "count": agg_agree_src,
                    "rate_rel_valid": float(agg_agree_src / max(1, agg_valid)),
                },
                "base_consistency_agreement": {
                    "count": agg_agree_base,
                    "rate_rel_valid": float(agg_agree_base / max(1, agg_valid)),
                },
                "harmful_base_error_recovery": {
                    "recovered_count": agg_harm_rec,
                    "denominator_harmful_base_errors": agg_harm_denom,
                    "recovery_rate_rel_harmful": float(agg_harm_rec / max(1, agg_harm_denom)),
                    "rate_rel_valid": float(agg_harm_rec / max(1, agg_valid)),
                },
                "beneficial_adaptation_retention": {
                    "retained_count": agg_ben_ret,
                    "denominator_beneficial_adaptations": agg_ben_denom,
                    "retention_rate_rel_beneficial": float(agg_ben_ret / max(1, agg_ben_denom)),
                    "rate_rel_valid": float(agg_ben_ret / max(1, agg_valid)),
                },
                "beneficial_adaptation_loss": {
                    "lost_count": agg_ben_loss,
                    "denominator_beneficial_adaptations": agg_ben_denom,
                    "loss_rate_rel_beneficial": float(agg_ben_loss / max(1, agg_ben_denom)),
                    "rate_rel_valid": float(agg_ben_loss / max(1, agg_valid)),
                },
                "regression_from_base": {
                    "regressed_count": agg_reg_base,
                    "denominator_base_correct": agg_base_corr_denom,
                    "regression_rate_rel_base_correct": float(agg_reg_base / max(1, agg_base_corr_denom)),
                    "rate_rel_valid": float(agg_reg_base / max(1, agg_valid)),
                },
                "correction_over_base": {
                    "corrected_count": agg_corr_base,
                    "denominator_base_wrong": agg_base_wrong_denom,
                    "correction_rate_rel_base_wrong": float(agg_corr_base / max(1, agg_base_wrong_denom)),
                    "rate_rel_valid": float(agg_corr_base / max(1, agg_valid)),
                },
                "scientific_guardrails": {
                    "source_agreement_is_not_correctness": True,
                    "approaching_source_not_sufficient_evidence_of_success": True,
                    "aggregate_f1_improvement_may_suppress_adaptation": True,
                    "beneficial_adaptation_losses_reported_alongside_recovered_errors": True,
                },
            }

        condition_summary = {
            "cell_id": cell_id,
            "name": name,
            "family": family,
            "lambda_val": lambda_val,
            "confidence_threshold": self.threshold,
            "seed": 0,
            "order": "order_a",
            "reset_mode": "reset_per_event",
            "batch_size": self.batch_size,
            "num_events": len(event_summaries),
            "macro_f1": mean_macro_f1,
            "worst_event_macro_f1": worst_event_f1,
            "pooled_f1": pooled_f1,
            "pooled_precision": pooled_prec,
            "pooled_recall": pooled_rec,
            "confusion_totals": {
                "TP": int(tot_tp),
                "FP": int(tot_fp),
                "FN": int(tot_fn),
                "TN": int(tot_tn),
                "valid": int(tot_valid),
            },
            "confusion_conserved": bool((tot_tp + tot_fp + tot_fn + tot_tn) == tot_valid),
            "aggregate_interpretation_metrics": agg_interp,
            "elapsed_seconds": elapsed_sec,
            "peak_vram_allocated_mb": peak_vram_alloc_mb,
            "peak_vram_reserved_mb": peak_vram_res_mb,
            "event_results": event_summaries,
        }

        with open(cell_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(condition_summary, f, indent=2)

        print(
            f"Cell {cell_id} Done in {elapsed_sec:.1f}s | Mean Macro F1: {mean_macro_f1:.4f} | "
            f"Worst Event F1: {worst_event_f1:.4f} | Peak VRAM Alloc: {peak_vram_alloc_mb:.1f} MB | "
            f"Peak VRAM Res: {peak_vram_res_mb:.1f} MB"
        )
        print("-" * 80)
        return condition_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E3 Source-Anchor Screening Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e3_anchor_screening_v1.json",
        help="Path to E3 screening config JSON",
    )
    parser.add_argument(
        "--cell-id",
        type=str,
        default=None,
        help="Single cell ID to run (e.g. E3_PROB_05)",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["prob", "logit", "feat"],
        help="Filter cells by family",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=None,
        help="Filter cells by lambda value (e.g. 0.50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run adaptation on",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Run smoke mode (sliced events / small samples)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit number of samples per event (for bounded smoke tests)",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of development events to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override (useful for smoke quarantine)",
    )
    args = parser.parse_args()

    runner = E3SourceAnchorScreeningRunner(
        config_path=args.config,
        device_str=args.device,
        output_dir=args.output_dir,
        smoke_mode=args.smoke_mode,
        limit_samples=args.limit_samples,
        target_events=args.events,
    )

    preflight_info = runner.run_preflights()

    with open(runner.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    all_cells = cfg["screening_specification"]["screening_cells"]
    target_cells = []

    for cell in all_cells:
        if args.cell_id is not None and cell["cell_id"] != args.cell_id:
            continue
        if args.family is not None and cell["family"] != args.family:
            continue
        if args.lambda_val is not None and abs(cell["lambda_val"] - args.lambda_val) > 1e-5:
            continue
        target_cells.append(cell)

    if not target_cells:
        print("No matching cells found to execute.")
        return

    print(f"Target Cells to Execute ({len(target_cells)}): {[c['cell_id'] for c in target_cells]}")

    results = []
    for cell in target_cells:
        res = runner.execute_cell(cell, preflight_info)
        results.append(res)

    print("\n" + "=" * 80)
    print("ALL REQUESTED E3 CELLS COMPLETED.")
    for r in results:
        print(
            f"  {r['cell_id']:<15} | Macro F1: {r['macro_f1']:.4f} | "
            f"Worst Event: {r['worst_event_macro_f1']:.4f} | Peak VRAM: {r['peak_vram_reserved_mb']:.1f} MB"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
