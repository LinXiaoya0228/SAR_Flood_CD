"""ChangeGuard Extended Experiment Recorder for Plan V3.

Adheres strictly to docs/CACHE_AND_MEASUREMENT_CONTRACT.md:
1. Exact Provenance: Tracks checkpoint_id, run_id, config_hash, git_commit, event_id, step_index.
2. Distinct Predictions: Distinguishes PRE_UPDATE primary predictions from POST_UPDATE diagnostics.
3. Probability Maps: Persists float16 Class-2 flood probability maps with explicit branch and phase.
4. Telemetry & State: Tracks gradient L2 norms, BN affine displacements, and per-class selection counts.
5. Missing Values: Uses explicit None/null values, never fabricated zeros.
6. Durable Progress: Atomic writes for step_records, progress, predictions, failures, and metrics.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import uuid

import numpy as np
import torch
import torch.nn as nn


def _json_default(obj: Any) -> Any:
    """Handle numpy types and non-finite values for clean JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def compute_gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> Optional[float]:
    """Compute Euclidean L2 norm of parameter gradients.
    
    Returns None if no parameter has a populated gradient.
    """
    total_sq_norm = 0.0
    has_grad = False
    for p in parameters:
        if p.grad is not None:
            has_grad = True
            grad_data = p.grad.detach()
            if not torch.isfinite(grad_data).all():
                return None
            param_sq = float(torch.sum(grad_data.double() ** 2).item())
            total_sq_norm += param_sq
    if not has_grad:
        return None
    return float(math.sqrt(total_sq_norm))


def compute_bn_affine_displacement(
    current_model: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute L2 displacement between current and source BatchNorm affine parameters.
    
    Returns:
        (weight_displacement_l2, bias_displacement_l2)
        Each is None if no matching BN parameters exist.
    """
    current_state = current_model.state_dict()
    weight_sq_sum = 0.0
    bias_sq_sum = 0.0
    has_weight = False
    has_bias = False

    for name, param in current_state.items():
        if name not in source_state_dict:
            continue
        src_param = source_state_dict[name].to(param.device)

        # Identify BN affine parameters by convention
        if "bn" in name.lower() or "batchnorm" in name.lower():
            if name.endswith(".weight") or name.endswith("_weight"):
                diff = (param.double() - src_param.double()).view(-1)
                weight_sq_sum += float(torch.sum(diff ** 2).item())
                has_weight = True
            elif name.endswith(".bias") or name.endswith("_bias"):
                diff = (param.double() - src_param.double()).view(-1)
                bias_sq_sum += float(torch.sum(diff ** 2).item())
                has_bias = True

    w_disp = float(math.sqrt(weight_sq_sum)) if has_weight else None
    b_disp = float(math.sqrt(bias_sq_sum)) if has_bias else None
    return w_disp, b_disp


def compute_per_class_pixel_counts(
    predictions: Union[np.ndarray, torch.Tensor],
    selection_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[int, int]:
    """Count predictions per class (0, 1, 2) subject to selection and validity masks."""
    if isinstance(predictions, torch.Tensor):
        preds_np = predictions.detach().cpu().numpy()
    else:
        preds_np = np.asarray(predictions)

    mask = np.ones_like(preds_np, dtype=bool)
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            mask &= valid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask &= np.asarray(valid_mask, dtype=bool)
    if selection_mask is not None:
        if isinstance(selection_mask, torch.Tensor):
            mask &= selection_mask.detach().cpu().numpy().astype(bool)
        else:
            mask &= np.asarray(selection_mask, dtype=bool)

    active_preds = preds_np[mask]
    return {
        0: int(np.sum(active_preds == 0)),
        1: int(np.sum(active_preds == 1)),
        2: int(np.sum(active_preds == 2)),
    }


def verify_quantization_drift(
    full_precision: Union[np.ndarray, torch.Tensor],
    quantized_float16: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Measure maximum and mean absolute difference from float16 quantization."""
    if isinstance(full_precision, torch.Tensor):
        fp = full_precision.detach().cpu().numpy().astype(np.float64)
    else:
        fp = np.asarray(full_precision, dtype=np.float64)

    if isinstance(quantized_float16, torch.Tensor):
        q = quantized_float16.detach().cpu().numpy().astype(np.float64)
    else:
        q = np.asarray(quantized_float16, dtype=np.float64)

    abs_diff = np.abs(fp - q)
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
    }


CACHE_SCHEMA_V1: Dict[str, Any] = {
    "schema_version": "2.1.0",
    "contract_document": "docs/CACHE_AND_MEASUREMENT_CONTRACT.md",
    "storage_architecture": {
        "primary_authoritative_format": "batch_atomic_npz_shards",
        "shard_directory": "evidence_shards",
        "shard_pattern": "batch_{step_index:06d}.npz",
        "index_file": "evidence_index.json",
        "optional_convenience_aggregate": "frozen_predictions.npz",
    },
    "supported_contexts": [
        "source",      # source_frozen_running_stats (FROZEN_SOURCE)
        "bn_control",  # matched_bn_no_update_bs16 (DIAGNOSTIC_BN)
        "candidate",   # candidate_e2_conf90 (PRE_UPDATE)
    ],
    "supported_pairs": [
        "real",        # pre2/post
        "null_pre2",   # pre2/pre2
        "null_post",   # post/post
    ],
    "timing_semantics": {
        "prelabel_evidence_elapsed_ms": "Time in milliseconds from batch start through model forward passes, temporal probes, and evidence shard staging immediately before durable shard write and label unlock.",
        "batch_total_elapsed_ms": "Total time in milliseconds from batch start through candidate adaptation update and offline evaluation completion.",
        "timing_ms_legacy_alias": "Alias in evidence_index.json shards mapping to prelabel_evidence_elapsed_ms.",
    },
    "key_patterns": {
        "sensory_mask": {
            "pattern": "{sample_id}_sensory_mask",
            "dtype": "uint8",
            "shape": [224, 224],
            "description": "Sensory observation validity mask from MK0_MNA == 1 (1=valid, 0=nodata)",
        },
        "context_pair_prediction": {
            "pattern": "{sample_id}_{context}_{pair}_pred",
            "dtype": "uint8",
            "shape": [224, 224],
            "permitted_contexts": ["source", "bn_control", "candidate"],
            "permitted_pairs": ["real", "null_pre2", "null_post"],
            "description": "Discrete class map in {0, 1, 2} for declared context and pair",
        },
        "context_pair_probabilities": {
            "pattern": "{sample_id}_{context}_{pair}_probs",
            "dtype": "float16",
            "shape": [3, 224, 224],
            "permitted_contexts": ["source", "bn_control", "candidate"],
            "permitted_pairs": ["real", "null_pre2", "null_post"],
            "description": "Full three-class softmax probability tensor in [0.0, 1.0]",
        },
        "candidate_real_strong_probabilities": {
            "pattern": "{sample_id}_candidate_real_strong_probs",
            "dtype": "float16",
            "shape": [3, 224, 224],
            "description": "Aligned strong-view (horizontal flip) candidate probabilities on real pair",
        },
        "primary_prediction_legacy_alias": {
            "pattern": "{sample_id}",
            "maps_to": "{sample_id}_candidate_real_pred",
            "dtype": "uint8",
            "shape": [224, 224],
            "phase": "PRE_UPDATE",
            "branch": "candidate",
            "description": "Primary discrete class map in {0, 1, 2}",
        },
        "source_prediction_legacy_alias": {
            "pattern": "{sample_id}_src_pred",
            "maps_to": "{sample_id}_source_real_pred",
            "dtype": "uint8",
            "shape": [224, 224],
            "phase": "FROZEN_SOURCE",
            "branch": "source",
            "description": "Immutable source model discrete class map in {0, 1, 2}",
        },
        "probability_c2_legacy_alias": {
            "pattern": "{sample_id}_{phase}_{branch}_prob_c2",
            "dtype": "float16",
            "shape": [224, 224],
            "description": "Half-precision Class-2 flood probability map in [0.0, 1.0]",
        },
    },
}


class ExtendedExperimentRecorder:
    """Three-tier experiment recorder implementing ChangeGuard Plan V3 cache contract."""

    VALID_ACTIONS = {"update", "skip", "rollback", "fallback_source", "failure"}
    VALID_PROB_PHASES = {"PRE_UPDATE", "POST_UPDATE", "FROZEN_SOURCE"}
    VALID_PROB_BRANCHES = {"candidate", "source"}

    def __init__(
        self,
        output_dir: Union[str, Path],
        run_id: str,
        checkpoint_id: str,
        config_hash: str,
        git_commit: str,
        allow_overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.checkpoint_id = str(checkpoint_id)
        self.config_hash = str(config_hash)
        self.git_commit = str(git_commit)
        self.allow_overwrite = bool(allow_overwrite)

        # Artifact protection: fail-fast if output_dir already has completed results
        if not self.allow_overwrite and self.output_dir.exists():
            completion_file = self.output_dir / "completion_status.json"
            if completion_file.exists():
                raise FileExistsError(
                    f"Output directory {self.output_dir} already contains completed run artifacts. "
                    "Overwriting is prohibited (allow_overwrite=False)."
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write cache schema specification atomically
        self.cache_schema_path = self.output_dir / "cache_schema.json"
        _atomic_write_json(self.cache_schema_path, CACHE_SCHEMA_V1)

        # Write initial provenance manifest atomically
        self.initial_provenance_path = self.output_dir / "initial_provenance.json"
        _atomic_write_json(
            self.initial_provenance_path,
            {
                "run_id": self.run_id,
                "checkpoint_id": self.checkpoint_id,
                "config_hash": self.config_hash,
                "git_commit": self.git_commit,
                "output_dir": str(self.output_dir),
                "created_at_utc": str(np.datetime64("now")),
            },
        )

        # Current event state
        self.current_event_id: Optional[int] = None
        self.current_event_dir: Optional[Path] = None
        self.current_shards_dir: Optional[Path] = None
        self.current_shards_index: List[Dict[str, Any]] = []
        self.current_evidence_index_path: Optional[Path] = None
        self.current_step_records: List[Dict[str, Any]] = []
        self.seen_sample_ids: Set[str] = set()
        self.seen_sample_ids_ordered: List[str] = []
        self.sample_step_map: Dict[str, int] = {}
        self.frozen_spatial_arrays: Dict[str, np.ndarray] = {}

    def start_event(self, event_id: int) -> Path:
        """Initialize state and directory for a new event stream."""
        self.current_event_id = int(event_id)
        self.current_event_dir = self.output_dir / f"event_{event_id}"
        self.current_event_dir.mkdir(parents=True, exist_ok=True)
        self.current_shards_dir = self.current_event_dir / "evidence_shards"
        self.current_shards_dir.mkdir(parents=True, exist_ok=True)
        self.current_shards_index = []
        self.current_evidence_index_path = self.current_event_dir / "evidence_index.json"
        self.current_step_records = []
        self.seen_sample_ids = set()
        self.seen_sample_ids_ordered = []
        self.sample_step_map = {}
        self.frozen_spatial_arrays = {}

        # Initialize adaptation progress
        progress_path = self.current_event_dir / "adaptation_progress.json"
        _atomic_write_json(
            progress_path,
            {
                "event_id": self.current_event_id,
                "status": "IN_PROGRESS",
                "steps_completed": 0,
                "samples_completed": 0,
                "completed_sample_ids": [],
            },
        )

        # Initialize evidence index atomically
        _atomic_write_json(
            self.current_evidence_index_path,
            {
                "event_id": self.current_event_id,
                "schema_version": "2.1.0",
                "shard_format": "batch_atomic_npz_v1",
                "total_shards": 0,
                "total_samples": 0,
                "completed_sample_ids": [],
                "shards": [],
            },
        )
        return self.current_event_dir

    def record_batch_evidence_shard(
        self,
        step_index: int,
        sample_ids: Sequence[str],
        arrays: Dict[str, Union[np.ndarray, torch.Tensor]],
        contexts: Optional[Sequence[str]] = None,
        timing_ms: Optional[float] = None,
        prelabel_evidence_elapsed_ms: Optional[float] = None,
    ) -> Path:
        """Atomically persist a bounded batch evidence shard before ground-truth unlock.

        Writes arrays to a temporary file, flushes/syncs to disk, atomically renames to
        final shard name `evidence_shards/batch_{step_index:06d}.npz`, verifies readability
        via np.load, records SHA-256 in `evidence_index.json`, and bounds process RAM to 1 batch.
        """
        if self.current_event_id is None or self.current_event_dir is None or self.current_shards_dir is None or self.current_evidence_index_path is None:
            raise RuntimeError("Cannot record evidence shard before start_event() has been called.")

        if not sample_ids:
            raise ValueError("Cannot record evidence shard with empty sample_ids.")

        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample IDs within the same batch: {sample_ids}")

        # Check for cross-step sample ID collision
        for sid in sample_ids:
            if sid in self.sample_step_map and self.sample_step_map[sid] != int(step_index):
                raise ValueError(
                    f"Sample ID collision detected: '{sid}' was already recorded in step {self.sample_step_map[sid]}."
                )

        shard_name = f"batch_{int(step_index):06d}.npz"
        shard_path = self.current_shards_dir / shard_name
        if shard_path.exists():
            raise FileExistsError(f"Shard collision: {shard_path} already exists! Overwrite is strictly forbidden.")

        if not arrays:
            raise ValueError("Cannot record empty arrays dictionary in evidence shard.")

        # Convert to numpy and collect array telemetry
        np_arrays: Dict[str, np.ndarray] = {}
        dtypes: Dict[str, str] = {}
        shapes: Dict[str, List[int]] = {}
        for k, v in arrays.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
            else:
                arr = np.asarray(v)
            np_arrays[k] = arr
            dtypes[k] = str(arr.dtype)
            shapes[k] = list(arr.shape)

        # Write to atomic temporary file
        tmp_shard_name = f"batch_{int(step_index):06d}_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}.npz"
        tmp_shard_path = self.current_shards_dir / tmp_shard_name
        try:
            np.savez_compressed(tmp_shard_path, **np_arrays)

            # Flush and sync to disk
            with open(tmp_shard_path, "rb") as f:
                os.fsync(f.fileno())

            # Atomic rename to final shard
            os.replace(tmp_shard_path, shard_path)

            # Sync directory
            try:
                dir_fd = os.open(str(self.current_shards_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass

            # Compute SHA-256 hash of final shard on disk
            sha256 = hashlib.sha256()
            with open(shard_path, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            shard_hash = sha256.hexdigest()

            # Defense-in-depth: verify readability and key integrity
            with np.load(shard_path) as loaded:
                loaded_keys = set(loaded.files)
                expected_keys = set(np_arrays.keys())
                if loaded_keys != expected_keys:
                    raise RuntimeError(
                        f"Corrupted shard {shard_path}: key mismatch. "
                        f"Missing: {expected_keys - loaded_keys}, unexpected: {loaded_keys - expected_keys}"
                    )
                # Verify decompressability
                test_k = next(iter(expected_keys))
                _ = loaded[test_k].shape

        except Exception as exc:
            # Clean up temporary file if left behind
            if tmp_shard_path.exists():
                try:
                    tmp_shard_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to durably write or verify evidence shard {shard_path}: {exc}") from exc

        # Update sample tracking
        for sid in sample_ids:
            self.sample_step_map[sid] = int(step_index)
            if sid not in self.seen_sample_ids:
                self.seen_sample_ids.add(sid)
                self.seen_sample_ids_ordered.append(sid)

        # Append shard record to index
        elapsed_val = prelabel_evidence_elapsed_ms if prelabel_evidence_elapsed_ms is not None else timing_ms
        prelabel_ms = float(elapsed_val) if elapsed_val is not None and math.isfinite(elapsed_val) else None
        shard_meta = {
            "step_index": int(step_index),
            "shard_file": f"evidence_shards/{shard_name}",
            "sha256": shard_hash,
            "sample_ids": list(sample_ids),
            "batch_size": len(sample_ids),
            "array_count": len(np_arrays),
            "dtypes": dtypes,
            "shapes": shapes,
            "contexts": list(contexts) if contexts else [],
            "prelabel_evidence_elapsed_ms": prelabel_ms,
            "timing_ms": prelabel_ms,  # Legacy alias mapping to prelabel_evidence_elapsed_ms
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.current_shards_index.append(shard_meta)

        # Atomically update evidence index
        index_payload = {
            "event_id": self.current_event_id,
            "schema_version": "2.1.0",
            "shard_format": "batch_atomic_npz_v1",
            "total_shards": len(self.current_shards_index),
            "total_samples": len(self.seen_sample_ids_ordered),
            "completed_sample_ids": list(self.seen_sample_ids_ordered),
            "shards": self.current_shards_index,
        }
        _atomic_write_json(self.current_evidence_index_path, index_payload)

        return shard_path

    def record_step(
        self,
        step_index: int,
        event_id: int,
        condition_id: str,
        batch_size: int,
        sample_ids: Sequence[str],
        action: str,
        update_applied: bool,
        total_loss: Optional[float] = None,
        consistency_loss: Optional[float] = None,
        anchor_loss: Optional[float] = None,
        gradient_norm_l2: Optional[float] = None,
        bn_weight_displacement: Optional[float] = None,
        bn_bias_displacement: Optional[float] = None,
        valid_pixels: Optional[int] = None,
        selected_pixels: Optional[int] = None,
        selected_pixels_c0: Optional[int] = None,
        selected_pixels_c1: Optional[int] = None,
        selected_pixels_c2: Optional[int] = None,
        selected_pixel_fraction: Optional[float] = None,
        mean_entropy: Optional[float] = None,
        mean_max_confidence: Optional[float] = None,
        mean_prob_c2: Optional[float] = None,
        null_score_s_null: Optional[float] = None,
        batch_time_ms: Optional[float] = None,
        batch_total_elapsed_ms: Optional[float] = None,
        prelabel_evidence_elapsed_ms: Optional[float] = None,
        peak_vram_mb: Optional[float] = None,
        failure_reason: Optional[str] = None,
        **extra_telemetry: Any,
    ) -> None:
        """Record progressive batch step telemetry and update progress atomically."""
        if self.current_event_id is None or self.current_event_dir is None:
            raise RuntimeError("Cannot record step before start_event() has been called.")
        if int(event_id) != self.current_event_id:
            raise ValueError(f"Step event_id {event_id} does not match active event {self.current_event_id}.")
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {self.VALID_ACTIONS}.")
        if int(batch_size) != len(sample_ids):
            raise ValueError(f"batch_size {batch_size} does not match len(sample_ids) {len(sample_ids)}.")

        # Check for sample ID collision within the event
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample IDs within the same batch: {sample_ids}")

        for sid in sample_ids:
            if sid in self.sample_step_map and self.sample_step_map[sid] != int(step_index):
                raise ValueError(
                    f"Sample ID collision detected: '{sid}' was already recorded in step {self.sample_step_map[sid]}."
                )
            self.sample_step_map[sid] = int(step_index)
            if sid not in self.seen_sample_ids:
                self.seen_sample_ids.add(sid)
                self.seen_sample_ids_ordered.append(sid)

        total_elapsed = batch_total_elapsed_ms if batch_total_elapsed_ms is not None else batch_time_ms

        # Assemble record (explicit None for uncomputed measurements)
        step_record = {
            "step_index": int(step_index),
            "event_id": int(event_id),
            "condition_id": str(condition_id),
            "batch_size": int(batch_size),
            "sample_ids": list(sample_ids),
            "action": str(action),
            "update_applied": bool(update_applied),
            "total_loss": float(total_loss) if total_loss is not None and math.isfinite(total_loss) else None,
            "consistency_loss": float(consistency_loss) if consistency_loss is not None and math.isfinite(consistency_loss) else None,
            "anchor_loss": float(anchor_loss) if anchor_loss is not None and math.isfinite(anchor_loss) else None,
            "gradient_norm_l2": float(gradient_norm_l2) if gradient_norm_l2 is not None and math.isfinite(gradient_norm_l2) else None,
            "bn_weight_displacement": float(bn_weight_displacement) if bn_weight_displacement is not None and math.isfinite(bn_weight_displacement) else None,
            "bn_bias_displacement": float(bn_bias_displacement) if bn_bias_displacement is not None and math.isfinite(bn_bias_displacement) else None,
            "valid_pixels": int(valid_pixels) if valid_pixels is not None else None,
            "selected_pixels": int(selected_pixels) if selected_pixels is not None else None,
            "selected_pixels_c0": int(selected_pixels_c0) if selected_pixels_c0 is not None else None,
            "selected_pixels_c1": int(selected_pixels_c1) if selected_pixels_c1 is not None else None,
            "selected_pixels_c2": int(selected_pixels_c2) if selected_pixels_c2 is not None else None,
            "selected_pixel_fraction": float(selected_pixel_fraction) if selected_pixel_fraction is not None and math.isfinite(selected_pixel_fraction) else None,
            "mean_entropy": float(mean_entropy) if mean_entropy is not None and math.isfinite(mean_entropy) else None,
            "mean_max_confidence": float(mean_max_confidence) if mean_max_confidence is not None and math.isfinite(mean_max_confidence) else None,
            "mean_prob_c2": float(mean_prob_c2) if mean_prob_c2 is not None and math.isfinite(mean_prob_c2) else None,
            "null_score_s_null": float(null_score_s_null) if null_score_s_null is not None and math.isfinite(null_score_s_null) else None,
            "batch_time_ms": float(total_elapsed) if total_elapsed is not None and math.isfinite(total_elapsed) else None,
            "batch_total_elapsed_ms": float(total_elapsed) if total_elapsed is not None and math.isfinite(total_elapsed) else None,
            "prelabel_evidence_elapsed_ms": float(prelabel_evidence_elapsed_ms) if prelabel_evidence_elapsed_ms is not None and math.isfinite(prelabel_evidence_elapsed_ms) else None,
            "peak_vram_mb": float(peak_vram_mb) if peak_vram_mb is not None and math.isfinite(peak_vram_mb) else None,
            "failure_reason": str(failure_reason) if failure_reason is not None else None,
        }

        # Any additional expanded telemetry fields (e.g. Work Package E3 gradient decompositions)
        for k, v in extra_telemetry.items():
            if v is not None and isinstance(v, (float, np.floating)):
                step_record[k] = float(v) if math.isfinite(v) else None
            else:
                step_record[k] = v

        self.current_step_records.append(step_record)

        # Progressively update step records and progress on disk
        step_records_path = self.current_event_dir / "step_records.json"
        _atomic_write_json(step_records_path, self.current_step_records)

        progress_path = self.current_event_dir / "adaptation_progress.json"
        _atomic_write_json(
            progress_path,
            {
                "event_id": self.current_event_id,
                "status": "IN_PROGRESS",
                "steps_completed": len(self.current_step_records),
                "samples_completed": len(self.seen_sample_ids_ordered),
                "completed_sample_ids": list(self.seen_sample_ids_ordered),
            },
        )

    def record_predictions(
        self,
        sample_ids: Sequence[str],
        pre_predictions: Union[np.ndarray, torch.Tensor],
        post_predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prob_c2: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prob_phase: Optional[str] = None,
        prob_branch: Optional[str] = None,
        src_predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Cache spatial discrete predictions and float16 Class-2 probabilities.
        
        Args:
            sample_ids: List of sample IDs in this batch.
            pre_predictions: PRE_UPDATE primary discrete predictions (B, H, W).
            post_predictions: Optional POST_UPDATE diagnostic discrete predictions (B, H, W).
            prob_c2: Optional Class-2 flood probability maps (B, H, W).
            prob_phase: Declared phase of prob_c2 ('PRE_UPDATE', 'POST_UPDATE', 'FROZEN_SOURCE').
            prob_branch: Declared branch of prob_c2 ('candidate', 'source').
            src_predictions: Optional FROZEN_SOURCE discrete predictions (B, H, W).
        """
        if self.current_event_id is None:
            raise RuntimeError("Cannot record predictions before start_event() has been called.")

        B = len(sample_ids)
        if isinstance(pre_predictions, torch.Tensor):
            pre_np = pre_predictions.detach().cpu().numpy()
        else:
            pre_np = np.asarray(pre_predictions)

        if pre_np.shape[0] != B:
            raise ValueError(f"pre_predictions batch size {pre_np.shape[0]} does not match sample count {B}.")

        post_np = None
        if post_predictions is not None:
            if isinstance(post_predictions, torch.Tensor):
                post_np = post_predictions.detach().cpu().numpy()
            else:
                post_np = np.asarray(post_predictions)
            if post_np.shape[0] != B:
                raise ValueError(f"post_predictions batch size {post_np.shape[0]} does not match sample count {B}.")

        src_np = None
        if src_predictions is not None:
            if isinstance(src_predictions, torch.Tensor):
                src_np = src_predictions.detach().cpu().numpy()
            else:
                src_np = np.asarray(src_predictions)
            if src_np.shape[0] != B:
                raise ValueError(f"src_predictions batch size {src_np.shape[0]} does not match sample count {B}.")

        prob_np = None
        if prob_c2 is not None:
            if prob_phase not in self.VALID_PROB_PHASES:
                raise ValueError(
                    f"Ambiguous or missing prob_phase '{prob_phase}'. Must be one of {self.VALID_PROB_PHASES}."
                )
            if prob_branch not in self.VALID_PROB_BRANCHES:
                raise ValueError(
                    f"Ambiguous or missing prob_branch '{prob_branch}'. Must be one of {self.VALID_PROB_BRANCHES}."
                )
            if isinstance(prob_c2, torch.Tensor):
                prob_np = prob_c2.detach().cpu().numpy()
            else:
                prob_np = np.asarray(prob_c2)
            if prob_np.shape[0] != B:
                raise ValueError(f"prob_c2 batch size {prob_np.shape[0]} does not match sample count {B}.")

        # Store arrays in-memory for compressed packaging
        for i, sid in enumerate(sample_ids):
            # PRE_UPDATE primary prediction (uint8)
            self.frozen_spatial_arrays[sid] = pre_np[i].astype(np.uint8)

            # POST_UPDATE diagnostic prediction (uint8)
            if post_np is not None:
                self.frozen_spatial_arrays[f"{sid}_post_pred"] = post_np[i].astype(np.uint8)

            # FROZEN_SOURCE prediction (uint8)
            if src_np is not None:
                self.frozen_spatial_arrays[f"{sid}_src_pred"] = src_np[i].astype(np.uint8)

            # Float16 Class-2 probability map
            if prob_np is not None:
                p_c2 = prob_np[i]
                # Clamp to [0.0, 1.0] before float16 storage to guarantee bounded range
                p_c2_clamped = np.clip(p_c2, 0.0, 1.0)
                array_key = f"{sid}_{prob_phase}_{prob_branch}_prob_c2"
                self.frozen_spatial_arrays[array_key] = p_c2_clamped.astype(np.float16)

    def record_probability_map(
        self,
        sample_ids: Sequence[str],
        prob_c2: Union[np.ndarray, torch.Tensor],
        prob_phase: str,
        prob_branch: str,
    ) -> None:
        """Cache additional Class-2 probability maps with explicit phase and branch.
        
        Guarantees unambiguous keys adhering to Plan V3 cache contract:
        array_key = f"{sample_id}_{prob_phase}_{prob_branch}_prob_c2"
        """
        if self.current_event_id is None:
            raise RuntimeError("Cannot record probability map before start_event() has been called.")
        if prob_phase not in self.VALID_PROB_PHASES:
            raise ValueError(f"Ambiguous or missing prob_phase '{prob_phase}'. Must be one of {self.VALID_PROB_PHASES}.")
        if prob_branch not in self.VALID_PROB_BRANCHES:
            raise ValueError(f"Ambiguous or missing prob_branch '{prob_branch}'. Must be one of {self.VALID_PROB_BRANCHES}.")

        B = len(sample_ids)
        if isinstance(prob_c2, torch.Tensor):
            prob_np = prob_c2.detach().cpu().numpy()
        else:
            prob_np = np.asarray(prob_c2)
        if prob_np.shape[0] != B:
            raise ValueError(f"prob_c2 batch size {prob_np.shape[0]} does not match sample count {B}.")

        for i, sid in enumerate(sample_ids):
            p_c2 = prob_np[i]
            p_c2_clamped = np.clip(p_c2, 0.0, 1.0)
            array_key = f"{sid}_{prob_phase}_{prob_branch}_prob_c2"
            self.frozen_spatial_arrays[array_key] = p_c2_clamped.astype(np.float16)

    def complete_event(
        self,
        event_id: int,
        metrics: Dict[str, Any],
        expected_sample_count: Optional[int] = None,
        produce_convenience_aggregate: bool = True,
    ) -> Path:
        """Finalize event: verify coverage, write metrics, and optionally produce convenience aggregate."""
        if self.current_event_id is None or self.current_event_dir is None:
            raise RuntimeError("Cannot complete event before start_event() has been called.")
        if int(event_id) != self.current_event_id:
            raise ValueError(f"Event ID {event_id} does not match active event {self.current_event_id}.")

        # Verify sample coverage if expected count was declared
        if expected_sample_count is not None:
            if len(self.seen_sample_ids_ordered) != int(expected_sample_count):
                raise RuntimeError(
                    f"Sample coverage violation for event {event_id}: "
                    f"completed {len(self.seen_sample_ids_ordered)} samples, expected {expected_sample_count}."
                )

        # Verify confusion-cell conservation invariant if confusion cells are supplied
        if "tp" in metrics and "fp" in metrics and "fn" in metrics and "tn" in metrics and "valid_pixels" in metrics:
            cell_sum = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
            if cell_sum != metrics["valid_pixels"]:
                raise ValueError(
                    f"Confusion-cell sum ({cell_sum}) does not equal valid_pixels ({metrics['valid_pixels']})."
                )

        # Optional convenience aggregate (shards in evidence_shards/ remain authoritative)
        npz_path = self.current_event_dir / "frozen_predictions.npz"
        tmp_npz_path = self.current_event_dir / "frozen_predictions_tmp.npz"
        if self.current_shards_index and produce_convenience_aggregate:
            aggregate: Dict[str, np.ndarray] = {}
            for s_info in self.current_shards_index:
                s_path = self.current_event_dir / s_info["shard_file"]
                with np.load(s_path) as loaded:
                    for k in loaded.files:
                        aggregate[k] = loaded[k]
            np.savez_compressed(tmp_npz_path, **aggregate)
            os.replace(tmp_npz_path, npz_path)
        elif self.frozen_spatial_arrays:
            np.savez_compressed(tmp_npz_path, **self.frozen_spatial_arrays)
            os.replace(tmp_npz_path, npz_path)

        # Write event_metrics.json atomically
        metrics_path = self.current_event_dir / "event_metrics.json"
        _atomic_write_json(metrics_path, metrics)

        # Mark adaptation progress as complete
        progress_path = self.current_event_dir / "adaptation_progress.json"
        _atomic_write_json(
            progress_path,
            {
                "event_id": self.current_event_id,
                "status": "COMPLETED",
                "steps_completed": len(self.current_step_records),
                "samples_completed": len(self.seen_sample_ids_ordered),
                "completed_sample_ids": list(self.seen_sample_ids_ordered),
            },
        )
        return metrics_path

    def record_raw_array(
        self,
        key: str,
        array: Union[np.ndarray, torch.Tensor],
        dtype: Optional[Union[str, np.dtype]] = None,
    ) -> None:
        """Cache arbitrary named spatial array under the declared contract."""
        if self.current_event_id is None:
            raise RuntimeError("Cannot record array before start_event() has been called.")
        if isinstance(array, torch.Tensor):
            arr_np = array.detach().cpu().numpy()
        else:
            arr_np = np.asarray(array)
        if dtype is not None:
            arr_np = arr_np.astype(dtype)
        self.frozen_spatial_arrays[key] = arr_np

    def record_failure(self, error_message: str, phase: str = "adaptation", details: Optional[Dict[str, Any]] = None) -> Path:
        """Durable failure recording on unhandled exception or numerical failure."""
        target_dir = self.current_event_dir if self.current_event_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        failure_path = target_dir / "failure.json"
        payload = {
            "run_id": self.run_id,
            "event_id": self.current_event_id,
            "status": "FAILED",
            "phase": str(phase),
            "error_message": str(error_message),
            "details": details or {},
            "timestamp_utc": str(np.datetime64("now")),
        }
        _atomic_write_json(failure_path, payload)
        return failure_path

    def complete_run(self, condition_summary: Optional[Dict[str, Any]] = None) -> Path:
        """Finalize entire multi-event adaptation run."""
        completion_path = self.output_dir / "completion_status.json"
        _atomic_write_json(
            completion_path,
            {
                "run_id": self.run_id,
                "checkpoint_id": self.checkpoint_id,
                "config_hash": self.config_hash,
                "git_commit": self.git_commit,
                "status": "COMPLETED_RUN",
                "completed_at_utc": str(np.datetime64("now")),
            },
        )
        if condition_summary is not None:
            summary_path = self.output_dir / "condition_summary.json"
            _atomic_write_json(summary_path, condition_summary)

        return completion_path

