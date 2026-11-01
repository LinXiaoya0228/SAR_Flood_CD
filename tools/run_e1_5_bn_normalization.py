#!/usr/bin/env python3
"""ChangeGuard Plan V3 E1.5 Matched BN Normalization Confirmation Runner.

Executes and mathematically validates the 8-stream matched BN no-update matrix:
- 6 primary confirmation streams: Seeds 0, 1, 2 x Orders A, B (batch size 16)
- 2 batch-size sensitivity diagnostics: Seed 0, Order A (batch sizes 1 and 4; 16 shared)

Strict Plan V3 Requirements:
1. Zero access to sealed test partition ([205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]).
2. Strict PRE_UPDATE primary prediction timing; POST_UPDATE predictions are diagnostics only.
3. Model parameters held bit-exactly fixed: zero optimizer, zero backward pass, zero parameter updates.
4. BN running stats accumulation disabled (track_running_stats=False, running buffers None).
5. LabelGuardedBatch: zero labels accessed during forward pass; labels unlocked only after persistence.
6. Mathematical confusion conservation (TP + FP + FN + TN == valid_pixels) on every event.
7. Full 3x3 source-to-BN prediction transition matrix and source vs BN error shift decomposition.
8. Resumable execution: completed streams with verified manifests and outputs are safely skipped.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1MatchedBNNoUpdateAdapter,
)
from src.changeguard.e1_orchestrator import (
    E1OfflineEvaluator,
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
)
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
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


def compute_model_param_hash(model: nn.Module) -> str:
    """Computes SHA-256 digest over concatenated bytes of all model parameters."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def compute_model_buffer_hash(model: nn.Module) -> str:
    """Computes SHA-256 digest over concatenated bytes of all model buffers."""
    h = hashlib.sha256()
    for name, buf in sorted(model.named_buffers()):
        h.update(name.encode("utf-8"))
        if buf is not None:
            h.update(buf.detach().cpu().numpy().tobytes())
        else:
            h.update(b"NONE")
    return h.hexdigest()


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


class E1BNNormalizationRunner:
    """Runner for ChangeGuard Plan V3 Work Package E1.5."""

    def __init__(
        self,
        config_path: Path,
        output_root: Path,
        log_root: Path,
        device: str = "cuda:0",
    ):
        self.config_path = config_path.resolve()
        self.output_root = output_root.resolve()
        self.log_root = log_root.resolve()
        self.device_str = device

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)

        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.split_path = (REPO_ROOT / ds_cfg["split_path"]).resolve()
        self.target_events = [int(e) for e in ds_cfg["target_events"]]

        self.sar_mean = ds_cfg["sar_mean"]
        self.sar_std = ds_cfg["sar_std"]
        self.sar_clamp_max = ds_cfg["sar_clamp_max"]
        self.canonical_preevent = ds_cfg["canonical_preevent"]

        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._cached_stream_plans: Dict[str, List[Tuple[int, List[LabelGuardedBatch]]]] = {}

    def run_preflights(self) -> Dict[str, Any]:
        """Fail-closed preflight validation of all code, assets, and partitions."""
        print("=" * 80)
        print("ChangeGuard Plan V3 E1.5: Executing Fail-Closed Preflights...")
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
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split path not found: {self.split_path}")

        manifest_hash = compute_file_sha256(self.manifest_path)
        exp_manifest = self.cfg["dataset"]["manifest_expected_sha256"]
        if manifest_hash != exp_manifest:
            raise ValueError(f"Manifest SHA-256 mismatch: {manifest_hash} != {exp_manifest}")

        split_hash = compute_file_sha256(self.split_path)
        exp_split = self.cfg["dataset"]["split_expected_sha256"]
        if split_hash != exp_split:
            raise ValueError(f"Split SHA-256 mismatch: {split_hash} != {exp_split}")

        # 3. Checkpoints
        ckpt_info = {}
        for s_str, s_cfg in self.cfg["source_checkpoints"]["seeds"].items():
            s_idx = int(s_str)
            p = (REPO_ROOT / s_cfg["checkpoint_path"]).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Source checkpoint for seed {s_idx} not found: {p}")
            actual_sha = compute_file_sha256(p)
            exp_sha = s_cfg["expected_sha256"]
            if actual_sha != exp_sha:
                raise ValueError(f"Checkpoint seed {s_idx} SHA-256 mismatch: {actual_sha} != {exp_sha}")
            ckpt_info[s_idx] = {"path": str(p), "sha256": actual_sha}

        # 4. Git info
        git_info = get_git_info()
        print(f"Git Commit: {git_info['commit']}")
        print(f"Tracked Dirty: {git_info['tracked_dirty']}")
        print(f"Target Events: {self.target_events}")
        print("All fail-closed preflights PASSED cleanly.")
        print("=" * 80)

        return {
            "git_commit": git_info["commit"],
            "manifest_sha256": manifest_hash,
            "split_sha256": split_hash,
            "checkpoints": ckpt_info,
        }

    def get_datasets(self) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Lazy-loads and caches evaluation datasets."""
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

    def is_stream_completed(self, stream_dir: Path) -> bool:
        """Verifies if a stream was already completed with valid manifests and event outputs."""
        if not stream_dir.is_dir():
            return False
        summary_path = stream_dir / "stream_summary.json"
        manifest_path = stream_dir / "manifest.json"
        if not (summary_path.is_file() and manifest_path.is_file()):
            return False

        # Verify all target events are present and completed
        for eid in self.target_events:
            ev_dir = stream_dir / f"event_{eid}"
            ev_metrics = ev_dir / "event_metrics.json"
            ev_npz = ev_dir / "frozen_predictions.npz"
            if not (ev_metrics.is_file() and ev_npz.is_file()):
                return False
        return True

    def build_ordered_stream_batches(
        self,
        order_id: str,
        batch_size: int,
    ) -> List[Tuple[int, List[LabelGuardedBatch]]]:
        """Partitions development events into ordered sequences of LabelGuardedBatch."""
        cache_key = f"{order_id}_bs{batch_size}"
        if cache_key in self._cached_stream_plans:
            return self._cached_stream_plans[cache_key]

        adapt_ds, offline_ds = self.get_datasets()

        o_cfg = self.cfg["stream_orders"].get(order_id)
        if o_cfg is None:
            o_cfg = self.cfg["stream_orders"]["order_a"]

        event_seq = o_cfg["event_sequence"]
        within_sort = o_cfg["within_event_ordering"]

        target_seq = [e for e in event_seq if e in self.target_events]

        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(adapt_ds.samples):
            eid = int(item["event_id"])
            item_copy = dict(item)
            item_copy["dataset_index"] = idx
            event_to_samples.setdefault(eid, []).append(item_copy)

        stream_plan: List[Tuple[int, List[LabelGuardedBatch]]] = []

        for eid in target_seq:
            if eid not in event_to_samples:
                continue
            samples = event_to_samples[eid]
            if within_sort == "deterministic_sorted_sample_id_ascending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))
            elif within_sort == "deterministic_sorted_sample_id_descending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)
            else:
                raise ValueError(f"Unknown within_sort ordering: {within_sort}")

            # Partition into batches of batch_size
            event_batches: List[LabelGuardedBatch] = []
            for i in range(0, len(samples), batch_size):
                batch_items = samples[i : i + batch_size]
                b_ids = [str(s["sample_id"]) for s in batch_items]
                b_indices = [s["dataset_index"] for s in batch_items]

                adapt_data = [adapt_ds[idx] for idx in b_indices]
                xA = torch.stack([d["xA"] for d in adapt_data]).to(self.device_str)
                xB = torch.stack([d["xB"] for d in adapt_data]).to(self.device_str)
                valid_mask = torch.stack([d["valid_mask"] for d in adapt_data]).to(self.device_str)

                offline_data = [offline_ds[idx] for idx in b_indices]
                labels = torch.stack([d["label"] for d in offline_data]).to(self.device_str)

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

        self._cached_stream_plans[cache_key] = stream_plan
        return stream_plan

    def execute_stream(
        self,
        stream_spec: Dict[str, Any],
        preflight_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes a single E1.5 matched BN no-update stream."""
        sid = stream_spec["stream_id"]
        seed = int(stream_spec["seed"])
        order = stream_spec["order"]
        batch_size = int(stream_spec["batch_size"])
        reset_mode = stream_spec["reset_mode"]
        stream_dir = self.output_root / sid

        print("-" * 80)
        print(f"Executing Stream: {sid} (Seed {seed}, Order {order}, BatchSize {batch_size})")
        print("-" * 80)

        # 1. Check idempotency
        if self.is_stream_completed(stream_dir):
            print(f"Stream {sid} is already fully completed. Skipping re-execution.")
            with open(stream_dir / "stream_summary.json", "r", encoding="utf-8") as f:
                return json.load(f)

        stream_dir.mkdir(parents=True, exist_ok=True)

        # 2. Instantiate models
        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"][str(seed)]
        ckpt_path = (REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve()
        candidate_model, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path), in_channels=2, out_ch=3, base_channel=32, device=self.device_str
        )
        source_model, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path), in_channels=2, out_ch=3, base_channel=32, device=self.device_str
        )

        # Check zero storage aliasing
        assert candidate_model is not source_model, "Storage aliasing: model and source_model are identical object!"
        for p1, p2 in zip(candidate_model.parameters(), source_model.parameters()):
            assert p1.data_ptr() != p2.data_ptr(), "Parameter storage alias detected between candidate and source!"

        adapter = E1MatchedBNNoUpdateAdapter(
            model=candidate_model,
            source_model=source_model,
            device=self.device_str,
        )

        # Snapshot parameter and buffer hashes before execution
        param_hash_before = compute_model_param_hash(adapter.model)
        buffer_hash_before = compute_model_buffer_hash(adapter.model)
        src_param_hash_before = compute_model_param_hash(adapter.source_model)
        src_buffer_hash_before = compute_model_buffer_hash(adapter.source_model)

        recorder = ExtendedExperimentRecorder(
            output_dir=stream_dir,
            run_id=sid,
            checkpoint_id=ckpt_cfg["expected_sha256"],
            config_hash=preflight_info["manifest_sha256"],
            git_commit=preflight_info["git_commit"],
            allow_overwrite=False,
        )

        stream_plan = self.build_ordered_stream_batches(order, batch_size=batch_size)
        start_time = time.perf_counter()

        event_summaries: Dict[str, Any] = {}
        stream_transition_matrix = np.zeros((3, 3), dtype=np.int64)

        # Execute event stream
        for eid, batches in stream_plan:
            recorder.start_event(eid)
            batch_sizes: List[int] = []
            pre_preds_by_sample: Dict[str, np.ndarray] = {}
            src_preds_by_sample: Dict[str, np.ndarray] = {}
            labels_by_sample: Dict[str, np.ndarray] = {}
            masks_by_sample: Dict[str, np.ndarray] = {}

            event_trans_matrix = np.zeros((3, 3), dtype=np.int64)

            for step_idx, batch in enumerate(batches, start=1):
                batch_sizes.append(batch.batch_size)
                inputs = batch.get_sensory_inputs()
                xA = inputs["xA"]
                xB = inputs["xB"]
                valid_mask = inputs["valid_mask"]
                s_ids = inputs["sample_ids"]

                step_res = adapter.adapt_and_predict_step(
                    xA=xA,
                    xB=xB,
                    valid_mask=valid_mask,
                    sample_ids=s_ids,
                    event_id=eid,
                )

                primary = step_res["primary_prediction"]
                post = step_res["post_prediction"]
                adapt_meta = step_res["adaptation_metadata"]
                risk = step_res["risk_evidence"]

                pre_preds = primary["pred_cand_pre"]
                src_preds = primary["pred_src"]

                # Cache spatial predictions
                recorder.record_predictions(
                    sample_ids=s_ids,
                    pre_predictions=pre_preds,
                    post_predictions=post["pred_cand_post"],
                    src_predictions=src_preds,
                    prob_c2=primary["probs_cand_pre"][:, 2],
                    prob_phase="PRE_UPDATE",
                    prob_branch="candidate",
                )
                recorder.record_probability_map(
                    sample_ids=s_ids,
                    prob_c2=primary["probs_src"][:, 2],
                    prob_phase="FROZEN_SOURCE",
                    prob_branch="source",
                )

                # Record step telemetry
                recorder.record_step(
                    step_index=step_idx,
                    event_id=eid,
                    condition_id=self.cfg["condition"]["condition_id"],
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

                # Unlock labels only after step telemetry is persisted
                batch.mark_persisted()

                # Store for offline evaluation
                batch_labels = batch.get_labels().detach().cpu().numpy()
                pre_preds_np = pre_preds.detach().cpu().numpy()
                src_preds_np = src_preds.detach().cpu().numpy()
                v_mask_np = valid_mask.detach().cpu().numpy()

                for i, sid_item in enumerate(s_ids):
                    pre_preds_by_sample[sid_item] = pre_preds_np[i]
                    src_preds_by_sample[sid_item] = src_preds_np[i]
                    labels_by_sample[sid_item] = batch_labels[i]
                    masks_by_sample[sid_item] = v_mask_np[i]

                    # Transition matrix contribution
                    v_eval = (v_mask_np[i] == 1) & (batch_labels[i] != 3)
                    y_src = src_preds_np[i][v_eval]
                    y_cand = pre_preds_np[i][v_eval]
                    for r_c in range(3):
                        for c_c in range(3):
                            event_trans_matrix[r_c, c_c] += int(np.sum((y_src == r_c) & (y_cand == c_c)))

            # Offline ground truth evaluation for candidate
            event_metrics = E1OfflineEvaluator.compute_event_metrics(
                event_id=eid,
                predictions_by_sample=pre_preds_by_sample,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=masks_by_sample,
                batch_sizes=batch_sizes,
                updates_applied=adapter.num_updates_applied,
                updates_skipped=adapter.num_updates_skipped,
                updates_failed=adapter.num_updates_failed,
            )

            # Offline ground truth evaluation for frozen source reference
            source_metrics = E1OfflineEvaluator.compute_event_metrics(
                event_id=eid,
                predictions_by_sample=src_preds_by_sample,
                labels_by_sample=labels_by_sample,
                valid_masks_by_sample=masks_by_sample,
                batch_sizes=batch_sizes,
                updates_applied=0,
                updates_skipped=0,
                updates_failed=0,
            )

            # Append transition and delta metrics
            stream_transition_matrix += event_trans_matrix
            changed_pixels = int(np.sum(event_trans_matrix) - np.trace(event_trans_matrix))
            valid_px = event_metrics["valid_pixels"]
            changed_rate = float(changed_pixels / max(valid_px, 1))

            event_metrics["source_reference"] = {
                "f1": source_metrics["f1"],
                "iou": source_metrics["iou"],
                "precision": source_metrics["precision"],
                "recall": source_metrics["recall"],
                "tp": source_metrics["tp"],
                "fp": source_metrics["fp"],
                "fn": source_metrics["fn"],
                "tn": source_metrics["tn"],
            }
            event_metrics["delta_vs_source"] = {
                "delta_f1": float(event_metrics["f1"] - source_metrics["f1"]),
                "delta_tp": int(event_metrics["tp"] - source_metrics["tp"]),
                "delta_fp": int(event_metrics["fp"] - source_metrics["fp"]),
                "delta_fn": int(event_metrics["fn"] - source_metrics["fn"]),
                "delta_tn": int(event_metrics["tn"] - source_metrics["tn"]),
            }
            event_metrics["source_to_bn_transition_matrix"] = {
                f"t{r}{c}": int(event_trans_matrix[r, c]) for r in range(3) for c in range(3)
            }
            event_metrics["changed_pixels_vs_source"] = changed_pixels
            event_metrics["changed_pixel_rate_vs_source"] = changed_rate

            recorder.complete_event(eid, event_metrics)
            event_summaries[str(eid)] = event_metrics
            print(
                f"  Event {eid} ({len(batches)} batches, {event_metrics['sample_count']} samples): "
                f"F1={event_metrics['f1']:.4f} (Source={source_metrics['f1']:.4f}, Δ={event_metrics['delta_vs_source']['delta_f1']:+.4f}), "
                f"ChangedRate={changed_rate:.4f}"
            )

        elapsed_sec = time.perf_counter() - start_time

        # 4. Strict Immutability Assertions
        param_hash_after = compute_model_param_hash(adapter.model)
        buffer_hash_after = compute_model_buffer_hash(adapter.model)
        src_param_hash_after = compute_model_param_hash(adapter.source_model)
        src_buffer_hash_after = compute_model_buffer_hash(adapter.source_model)

        assert param_hash_after == param_hash_before, "CRITICAL: Candidate parameters mutated during execution!"
        assert buffer_hash_after == buffer_hash_before, "CRITICAL: Candidate buffers mutated during execution!"
        assert src_param_hash_after == src_param_hash_before, "CRITICAL: Source parameters mutated during execution!"
        assert src_buffer_hash_after == src_buffer_hash_before, "CRITICAL: Source buffers mutated during execution!"

        is_exact, max_w_disp, max_b_disp = adapter.verify_immutability()
        assert is_exact, "CRITICAL: Parameter tensor comparison failed immutability check!"
        assert max_w_disp == 0.0 and max_b_disp == 0.0, "CRITICAL: Non-zero BN affine displacement!"
        assert adapter.num_updates_applied == 0, "CRITICAL: Updates applied must be 0!"
        assert adapter.num_updates_skipped == 0, "CRITICAL: Updates skipped must be 0!"
        assert adapter.num_updates_failed == 0, "CRITICAL: Updates failed must be 0!"

        # Aggregate stream metrics
        event_f1s = [m["f1"] for m in event_summaries.values()]
        event_ious = [m["iou"] for m in event_summaries.values()]
        event_precs = [m["precision"] for m in event_summaries.values()]
        event_recs = [m["recall"] for m in event_summaries.values()]

        tot_tp = sum(m["tp"] for m in event_summaries.values())
        tot_fp = sum(m["fp"] for m in event_summaries.values())
        tot_fn = sum(m["fn"] for m in event_summaries.values())
        tot_tn = sum(m["tn"] for m in event_summaries.values())
        tot_valid = sum(m["valid_pixels"] for m in event_summaries.values())

        tot_src_tp = sum(m["source_reference"]["tp"] for m in event_summaries.values())
        tot_src_fp = sum(m["source_reference"]["fp"] for m in event_summaries.values())
        tot_src_fn = sum(m["source_reference"]["fn"] for m in event_summaries.values())
        tot_src_tn = sum(m["source_reference"]["tn"] for m in event_summaries.values())

        p_prec = float(tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
        p_rec = float(tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
        p_f1 = float(2 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) > 0 else 0.0
        p_iou = float(tot_tp / (tot_tp + tot_fp + tot_fn)) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

        p_src_prec = float(tot_src_tp / (tot_src_tp + tot_src_fp)) if (tot_src_tp + tot_src_fp) > 0 else 0.0
        p_src_rec = float(tot_src_tp / (tot_src_tp + tot_src_fn)) if (tot_src_tp + tot_src_fn) > 0 else 0.0
        p_src_f1 = float(2 * p_src_prec * p_src_rec / (p_src_prec + p_src_rec)) if (p_src_prec + p_src_rec) > 0 else 0.0

        tot_changed_pixels = int(np.sum(stream_transition_matrix) - np.trace(stream_transition_matrix))
        tot_changed_rate = float(tot_changed_pixels / max(tot_valid, 1))

        summary: Dict[str, Any] = {
            "stream_id": sid,
            "method": "matched_bn_no_update_pre_update_v1",
            "seed": seed,
            "order": order,
            "batch_size": batch_size,
            "reset_mode": reset_mode,
            "elapsed_seconds": elapsed_sec,
            "event_count": len(event_summaries),
            "macro_f1": float(np.mean(event_f1s)),
            "macro_iou": float(np.mean(event_ious)),
            "macro_precision": float(np.mean(event_precs)),
            "macro_recall": float(np.mean(event_recs)),
            "pooled_f1": p_f1,
            "pooled_iou": p_iou,
            "pooled_precision": p_prec,
            "pooled_recall": p_rec,
            "worst_event_f1": float(min(event_f1s)),
            "total_valid_pixels": tot_valid,
            "confusion": {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn, "tn": tot_tn},
            "source_reference_pooled": {
                "f1": p_src_f1,
                "tp": tot_src_tp,
                "fp": tot_src_fp,
                "fn": tot_src_fn,
                "tn": tot_src_tn,
            },
            "delta_pooled_vs_source": {
                "delta_f1": float(p_f1 - p_src_f1),
                "delta_tp": int(tot_tp - tot_src_tp),
                "delta_fp": int(tot_fp - tot_src_fp),
                "delta_fn": int(tot_fn - tot_src_fn),
                "delta_tn": int(tot_tn - tot_src_tn),
            },
            "stream_source_to_bn_transition_matrix": {
                f"t{r}{c}": int(stream_transition_matrix[r, c]) for r in range(3) for c in range(3)
            },
            "total_changed_pixels_vs_source": tot_changed_pixels,
            "total_changed_pixel_rate_vs_source": tot_changed_rate,
            "immutability_verification": {
                "param_hash_before": param_hash_before,
                "param_hash_after": param_hash_after,
                "buffer_hash_before": buffer_hash_before,
                "buffer_hash_after": buffer_hash_after,
                "source_param_hash": src_param_hash_after,
                "source_buffer_hash": src_buffer_hash_after,
                "is_bit_exact": is_exact,
                "max_bn_weight_displacement": max_w_disp,
                "max_bn_bias_displacement": max_b_disp,
                "optimizer_present": False,
                "backward_calls": 0,
                "updates_applied": 0,
                "updates_skipped": 0,
                "updates_failed": 0,
            },
            "events": event_summaries,
        }

        # Save stream summary
        with open(stream_dir / "stream_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Build stream manifest
        files_to_hash = [stream_dir / "stream_summary.json"]
        for eid in self.target_events:
            files_to_hash.append(stream_dir / f"event_{eid}/event_metrics.json")
            files_to_hash.append(stream_dir / f"event_{eid}/frozen_predictions.npz")
            files_to_hash.append(stream_dir / f"event_{eid}/step_records.json")

        manifest = {
            "stream_id": sid,
            "method": "matched_bn_no_update_pre_update_v1",
            "seed": seed,
            "order": order,
            "batch_size": batch_size,
            "reset_mode": reset_mode,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "macro_f1": summary["macro_f1"],
            "pooled_f1": summary["pooled_f1"],
            "worst_event_f1": summary["worst_event_f1"],
            "files": {str(p.relative_to(stream_dir)): compute_file_sha256(p) for p in files_to_hash if p.is_file()},
        }
        with open(stream_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"Completed Stream {sid}: Macro F1 = {summary['macro_f1']:.4f}, "
            f"Pooled F1 = {summary['pooled_f1']:.4f}, Worst Event F1 = {summary['worst_event_f1']:.4f}"
        )
        return summary

    def run_queue(self) -> List[Dict[str, Any]]:
        """Executes the full 8-stream E1.5 queue sequentially."""
        preflight_info = self.run_preflights()
        streams = self.cfg["grid"]["streams"]
        results: List[Dict[str, Any]] = []

        print("=" * 80)
        print(f"Starting ChangeGuard Plan V3 E1.5 Queue: {len(streams)} streams on {self.device_str}...")
        print("=" * 80)

        for idx, s in enumerate(streams, start=1):
            print(f"\n[E1.5 Queue {idx}/{len(streams)}] Starting stream: {s['stream_id']}")
            res = self.execute_stream(s, preflight_info)
            results.append(res)
            torch.cuda.empty_cache()
            gc.collect()

        # Build master coverage table
        self.generate_master_tables(results)
        print("=" * 80)
        print("ChangeGuard Plan V3 E1.5 Queue Completed Successfully!")
        print("=" * 80)
        return results

    def generate_master_tables(self, results: List[Dict[str, Any]]) -> None:
        """Generates machine-readable JSON and human-readable Markdown master tables."""
        master_json_path = self.output_root / "master_coverage_table.json"
        master_md_path = self.output_root / "master_coverage_table.md"

        # Simplified table for master table
        master_rows = []
        for r in results:
            master_rows.append({
                "stream_id": r["stream_id"],
                "method": r["method"],
                "seed": r["seed"],
                "order": r["order"],
                "batch_size": r["batch_size"],
                "reset_mode": r["reset_mode"],
                "macro_f1": r["macro_f1"],
                "pooled_f1": r["pooled_f1"],
                "worst_event_f1": r["worst_event_f1"],
                "total_valid_pixels": r["total_valid_pixels"],
                "total_changed_pixels_vs_source": r.get("total_changed_pixels_vs_source", 0),
                "total_changed_pixel_rate_vs_source": r.get("total_changed_pixel_rate_vs_source", 0.0),
                "delta_pooled_f1_vs_source": r.get("delta_pooled_vs_source", {}).get("delta_f1", 0.0),
                "status": "EXECUTED_NEW",
            })

        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(master_rows, f, indent=2)

        md_lines = [
            "# ChangeGuard Plan V3: Work Package E1.5 Master Coverage Table\n\n",
            f"**Generated:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n",
            f"**Condition:** `matched_bn_no_update_pre_update_v1`\n",
            f"**Total Streams:** {len(master_rows)}\n\n",
            "| Row | Stream ID | Seed | Order | Batch Size | Macro F1 | Pooled F1 | Worst-Event F1 | Changed Pixel Rate | ΔF1 vs Source |\n",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        ]

        for i, row in enumerate(master_rows, 1):
            md_lines.append(
                f"| {i} | `{row['stream_id']}` | {row['seed']} | `{row['order']}` | {row['batch_size']} | "
                f"{row['macro_f1']:.4f} | {row['pooled_f1']:.4f} | {row['worst_event_f1']:.4f} | "
                f"{row['total_changed_pixel_rate_vs_source']:.4f} | {row['delta_pooled_f1_vs_source']:+.4f} |\n"
            )

        with open(master_md_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)

        print(f"Master tables written to:\n  {master_json_path}\n  {master_md_path}")


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E1.5 Matched BN Normalization Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e1_5_bn_normalization_v1.json",
        help="Path to E1.5 configuration JSON",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs/e1_bn_normalization/v1",
        help="Root directory for E1.5 outputs",
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=REPO_ROOT / "logs/e1_bn_normalization/v1",
        help="Root directory for E1.5 execution logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for execution (e.g. 'cuda:0' or 'cpu')",
    )

    args = parser.parse_args()
    runner = E1BNNormalizationRunner(
        config_path=args.config,
        output_root=args.output_root,
        log_root=args.log_root,
        device=args.device,
    )
    runner.run_queue()


if __name__ == "__main__":
    main()
