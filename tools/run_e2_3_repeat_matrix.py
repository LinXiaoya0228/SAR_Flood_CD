"""ChangeGuard Plan V3 Work Package E2.3: Promotion Repeat Matrix Runner.

Executes and verifies the locked repeat matrix:
1. E2_REP_CONF_90: Retained Base Consistency Recipe (Threshold 0.90, Horizontal Flip strong)
2. E2_REP_CONF_NONE: Decisive Component Contrast (No Confidence Filter / C1, Horizontal Flip strong)

Protocol:
- Base Model: SNUNet_ECAM Seed 1 (checkpoint: 72c4be78...)
- Presentation Order: Order B ([1111008, 1111003, 559, 520, 437, 279], descending sample IDs)
- Reset Mode: reset_per_event
- Batch Size: 16 (tail processed as-is)
- Target Events: All 6 development events (N = 2,826 samples)
- Zero access to sealed-test partition ([205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]).
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
from src.changeguard.e2_adapters import E2AugmentationAdapter
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


class E23RepeatMatrixRunner:
    """Runner for ChangeGuard Plan V3 Work Package E2.3."""

    def __init__(
        self,
        config_path: Path,
        device_str: str = "cuda:0",
        output_dir: Optional[Path] = None,
        smoke_mode: bool = False,
    ):
        self.config_path = config_path.resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device_str = device_str
        self.smoke_mode = smoke_mode

        if output_dir is not None:
            self.output_root = output_dir.resolve()
        else:
            self.output_root = REPO_ROOT / "outputs" / "e2_repeat_matrix" / "v1"

        self.output_root.mkdir(parents=True, exist_ok=True)

        if self.smoke_mode:
            self.target_events = [520]
        else:
            self.target_events = list(self.cfg["screening_specification"]["event_sequence"])

        self.checkpoint_seed = int(self.cfg["screening_specification"]["checkpoint_seed"])
        self.stream_order = str(self.cfg["screening_specification"]["stream_order"])
        self.reset_mode = str(self.cfg["screening_specification"]["reset_mode"])
        self.batch_size = int(self.cfg["screening_specification"]["batch_size"])

        self.dataset_root = (REPO_ROOT / self.cfg["dataset"]["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / self.cfg["dataset"]["manifest_path"]).resolve()
        self.canonical_preevent = self.cfg["dataset"]["canonical_preevent"]
        self.sar_clamp_max = float(self.cfg["dataset"]["sar_clamp_max"])
        self.sar_mean = tuple(self.cfg["dataset"]["sar_mean"])
        self.sar_std = tuple(self.cfg["dataset"]["sar_std"])

        seed_info = self.cfg["source_checkpoints"]["seeds"][str(self.checkpoint_seed)]
        self.checkpoint_path = (REPO_ROOT / seed_info["checkpoint_path"]).resolve()
        self.expected_checkpoint_sha256 = seed_info["expected_sha256"]

        self.repeat_cells = self.cfg["screening_specification"]["repeat_cells"]

        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._cached_stream_plan: Optional[List[Tuple[int, List[LabelGuardedBatch]]]] = None

    def execute_preflight_checks(self) -> Dict[str, Any]:
        """Fail-closed preflight verifications."""
        print("=" * 80)
        print("ChangeGuard Plan V3 E2.3: Executing Fail-Closed Preflights...")
        print("=" * 80)

        # 1. Checkpoint existence and cryptographic checksum
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Source checkpoint not found: {self.checkpoint_path}")

        actual_ck_sha256 = compute_file_sha256(self.checkpoint_path)
        if actual_ck_sha256 != self.expected_checkpoint_sha256:
            raise ValueError(
                f"Seed {self.checkpoint_seed} checkpoint SHA-256 mismatch!\n"
                f"  Expected: {self.expected_checkpoint_sha256}\n"
                f"  Actual:   {actual_ck_sha256}"
            )

        # 2. Strict Sealed-Test Quarantine Audit
        target_set = set(self.target_events)
        forbidden_set = set(FORBIDDEN_SEALED_TEST_EVENTS)
        overlap = target_set.intersection(forbidden_set)
        if overlap:
            raise PermissionError(
                f"CRITICAL SECURITY VIOLATION: Target events contain sealed test events: {sorted(list(overlap))}"
            )

        # 3. Git info
        git_info = get_git_info()
        print(f"Git Commit: {git_info['commit']}")
        print(f"Tracked Dirty: {git_info['tracked_dirty']}")
        print(f"Checkpoint Seed: {self.checkpoint_seed} ({actual_ck_sha256[:8]}...)")
        print(f"Stream Order: {self.stream_order}")
        print(f"Target Events: {self.target_events}")
        print(f"Device: {self.device_str}")
        print("All fail-closed preflights PASSED cleanly.")
        print("=" * 80)

        return {
            "git_commit": git_info["commit"],
            "checkpoint_sha256": actual_ck_sha256,
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
        """Partitions development events into ordered sequences of LabelGuardedBatch (Order B)."""
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
            # Order B within-event: descending sample_id
            samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)

            if self.smoke_mode and eid == 520:
                samples = samples[:50]
                print(f"Smoke mode active: Event 520 sliced to head {len(samples)} samples.")

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

    def execute_repeat_cell(
        self,
        cell_spec: Dict[str, Any],
        preflight_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes one repeat cell across the target event sequence."""
        cell_id = cell_spec["cell_id"]
        cell_name = cell_spec["name"]
        tau = float(cell_spec["threshold"])
        aug_type = str(cell_spec.get("augmentation_type", "horizontal_flip"))
        role = cell_spec.get("role", "REPEAT_STREAM")

        cell_dir = self.output_root / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f"Executing Work Package E2.3 Repeat Cell: {cell_id} ({cell_name})")
        print(f"Role: {role} | Threshold: {tau:.2f} | Augmentation: {aug_type} | Seed: {self.checkpoint_seed} | Order: {self.stream_order}")
        print("-" * 80)

        recorder = ExtendedExperimentRecorder(
            output_dir=cell_dir,
            run_id=cell_id,
            checkpoint_id=f"source_all27_seed{self.checkpoint_seed}",
            config_hash=preflight_info["config_hash"],
            git_commit=preflight_info["git_commit"],
            allow_overwrite=False,
        )

        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            in_channels=2,
            out_ch=3,
            base_channel=32,
            device=self.device_str,
        )

        param_hash_before, buf_hash_before = compute_model_hashes(wrapper)

        adapter = E2AugmentationAdapter(
            model=wrapper,
            aug_type=aug_type,
            learning_rate=float(self.cfg["optimization"]["lr"]),
            weight_decay=float(self.cfg["optimization"]["weight_decay"]),
            confidence_threshold=tau,
            device=self.device_str,
        )

        orchestrator = E1Orchestrator(
            recorder=recorder,
            adapter=adapter,
            condition_id=cell_id,
            seed=self.checkpoint_seed,
            reset_mode=self.reset_mode,
        )

        stream_plan = self.build_ordered_stream_batches()
        start_time = time.perf_counter()

        event_summaries: Dict[str, Any] = {}

        if "cuda" in self.device_str and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for eid, batches in stream_plan:
            ev_start = time.perf_counter()
            print(f"  Streaming Event {eid} ({len(batches)} batches)...", end="", flush=True)

            event_metrics = orchestrator.run_event_stream(event_id=eid, batches=batches)
            ev_elapsed = time.perf_counter() - ev_start

            tp = event_metrics["tp"]
            fp = event_metrics["fp"]
            fn = event_metrics["fn"]
            tn = event_metrics["tn"]
            valid = event_metrics["valid_pixels"]
            cell_sum = tp + fp + fn + tn
            if cell_sum != valid:
                raise ValueError(
                    f"Confusion matrix conservation violation in Event {eid}: {cell_sum} != {valid}"
                )

            step_records_path = cell_dir / f"event_{eid}" / "step_records.json"
            with open(step_records_path, "r", encoding="utf-8") as f:
                steps = json.load(f)

            ev_valid_pixels = sum(s.get("valid_pixels", 0) for s in steps)
            ev_selected_pixels = sum(s.get("selected_pixels", 0) for s in steps)
            ev_selected_c0 = sum(s.get("selected_pixels_c0", 0) for s in steps)
            ev_selected_c1 = sum(s.get("selected_pixels_c1", 0) for s in steps)
            ev_selected_c2 = sum(s.get("selected_pixels_c2", 0) for s in steps)

            updated_steps = [s for s in steps if s.get("update_applied")]
            skipped_steps = [s for s in steps if s.get("action") == "skip"]
            failed_steps = [s for s in steps if s.get("action") == "failure"]
            zero_sel_steps = [s for s in steps if s.get("selected_pixels") == 0]

            loss_vals = [s["total_loss"] for s in updated_steps if s.get("total_loss") is not None]
            grad_norms = [s["gradient_norm_l2"] for s in updated_steps if s.get("gradient_norm_l2") is not None]

            ev_summary = {
                "event_id": eid,
                "sample_count": event_metrics["sample_count"],
                "batch_count": len(batches),
                "valid_pixels": valid,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "f1": event_metrics["f1"],
                "precision": event_metrics["precision"],
                "recall": event_metrics["recall"],
                "iou": event_metrics["iou"],
                "confusion_matrix_3x3": {
                    "c00": event_metrics["c00_land_land"],
                    "c01": event_metrics["c01_land_perm"],
                    "c02": event_metrics["c02_land_flood_FP"],
                    "c10": event_metrics["c10_perm_land"],
                    "c11": event_metrics["c11_perm_perm"],
                    "c12": event_metrics["c12_perm_flood_FP"],
                    "c20": event_metrics["c20_flood_land_FN"],
                    "c21": event_metrics["c21_flood_perm_FN"],
                    "c22": event_metrics["c22_flood_flood_TP"],
                },
                "selection_telemetry": {
                    "total_valid_pixels": ev_valid_pixels,
                    "total_selected_pixels": ev_selected_pixels,
                    "selected_pixel_fraction": (
                        float(ev_selected_pixels / ev_valid_pixels) if ev_valid_pixels > 0 else 0.0
                    ),
                    "selected_pixels_c0": ev_selected_c0,
                    "selected_pixels_c1": ev_selected_c1,
                    "selected_pixels_c2": ev_selected_c2,
                    "fraction_valid_c0": (
                        float(ev_selected_c0 / ev_valid_pixels) if ev_valid_pixels > 0 else 0.0
                    ),
                    "fraction_valid_c1": (
                        float(ev_selected_c1 / ev_valid_pixels) if ev_valid_pixels > 0 else 0.0
                    ),
                    "fraction_valid_c2": (
                        float(ev_selected_c2 / ev_valid_pixels) if ev_valid_pixels > 0 else 0.0
                    ),
                    "class_distribution_c0": (
                        float(ev_selected_c0 / ev_selected_pixels) if ev_selected_pixels > 0 else 0.0
                    ),
                    "class_distribution_c1": (
                        float(ev_selected_c1 / ev_selected_pixels) if ev_selected_pixels > 0 else 0.0
                    ),
                    "class_distribution_c2": (
                        float(ev_selected_c2 / ev_selected_pixels) if ev_selected_pixels > 0 else 0.0
                    ),
                },
                "batch_lifecycle": {
                    "batches_considered": len(batches),
                    "batches_updated": len(updated_steps),
                    "batches_skipped": len(skipped_steps),
                    "batches_failed": len(failed_steps),
                    "zero_selection_batches": len(zero_sel_steps),
                },
                "loss_telemetry": {
                    "mean": float(np.mean(loss_vals)) if loss_vals else None,
                    "min": float(np.min(loss_vals)) if loss_vals else None,
                    "max": float(np.max(loss_vals)) if loss_vals else None,
                },
                "gradient_telemetry": {
                    "mean_l2_norm": float(np.mean(grad_norms)) if grad_norms else None,
                    "min_l2_norm": float(np.min(grad_norms)) if grad_norms else None,
                    "max_l2_norm": float(np.max(grad_norms)) if grad_norms else None,
                },
                "runtime_seconds": ev_elapsed,
            }

            event_summaries[str(eid)] = ev_summary
            print(f" done in {ev_elapsed:.1f}s | F1: {event_metrics['f1']:.4f} | Sel%: {ev_summary['selection_telemetry']['selected_pixel_fraction']*100:.1f}%")

        param_hash_after, buf_hash_after = compute_model_hashes(wrapper)
        total_cell_runtime = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if "cuda" in self.device_str and torch.cuda.is_available() else 0.0

        # Aggregate metrics
        macro_f1 = float(np.mean([s["f1"] for s in event_summaries.values()]))

        agg_tp = sum(s["tp"] for s in event_summaries.values())
        agg_fp = sum(s["fp"] for s in event_summaries.values())
        agg_fn = sum(s["fn"] for s in event_summaries.values())
        agg_tn = sum(s["tn"] for s in event_summaries.values())
        agg_valid = sum(s["valid_pixels"] for s in event_summaries.values())
        pooled_f1 = float(2 * agg_tp / (2 * agg_tp + agg_fp + agg_fn)) if (2 * agg_tp + agg_fp + agg_fn) > 0 else 0.0

        worst_eid_str = min(event_summaries.keys(), key=lambda k: event_summaries[k]["f1"])
        worst_f1 = float(event_summaries[worst_eid_str]["f1"])
        worst_eid = int(worst_eid_str)

        # Aggregate selection
        agg_valid_sel = sum(s["selection_telemetry"]["total_valid_pixels"] for s in event_summaries.values())
        agg_sel = sum(s["selection_telemetry"]["total_selected_pixels"] for s in event_summaries.values())
        agg_c0 = sum(s["selection_telemetry"]["selected_pixels_c0"] for s in event_summaries.values())
        agg_c1 = sum(s["selection_telemetry"]["selected_pixels_c1"] for s in event_summaries.values())
        agg_c2 = sum(s["selection_telemetry"]["selected_pixels_c2"] for s in event_summaries.values())

        # Aggregate loss/gradient
        all_losses = []
        all_grads = []
        for s in event_summaries.values():
            if s["loss_telemetry"]["mean"] is not None:
                all_losses.append(s["loss_telemetry"]["mean"])
            if s["gradient_telemetry"]["mean_l2_norm"] is not None:
                all_grads.append(s["gradient_telemetry"]["mean_l2_norm"])

        total_samples_all = sum(s["sample_count"] for s in event_summaries.values())

        cell_summary = {
            "cell_id": cell_id,
            "cell_name": cell_name,
            "role": role,
            "confidence_threshold": tau,
            "seed": self.checkpoint_seed,
            "stream_order": self.stream_order,
            "reset_mode": self.reset_mode,
            "batch_size": self.batch_size,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_f1": worst_f1,
            "worst_event_id": worst_eid,
            "aggregate_confusion": {
                "total_tp": agg_tp,
                "total_fp": agg_fp,
                "total_fn": agg_fn,
                "total_tn": agg_tn,
                "total_valid_pixels": agg_valid,
                "total_sample_count": total_samples_all,
                "confusion_conserved": (agg_tp + agg_fp + agg_fn + agg_tn) == agg_valid,
            },
            "aggregate_selection": {
                "total_valid_pixels": agg_valid_sel,
                "total_selected_pixels": agg_sel,
                "selected_pixel_fraction": float(agg_sel / agg_valid_sel) if agg_valid_sel > 0 else 0.0,
                "selected_pixels_c0": agg_c0,
                "selected_pixels_c1": agg_c1,
                "selected_pixels_c2": agg_c2,
                "fraction_valid_c0": float(agg_c0 / agg_valid_sel) if agg_valid_sel > 0 else 0.0,
                "fraction_valid_c1": float(agg_c1 / agg_valid_sel) if agg_valid_sel > 0 else 0.0,
                "fraction_valid_c2": float(agg_c2 / agg_valid_sel) if agg_valid_sel > 0 else 0.0,
                "class_distribution_c0": float(agg_c0 / agg_sel) if agg_sel > 0 else 0.0,
                "class_distribution_c1": float(agg_c1 / agg_sel) if agg_sel > 0 else 0.0,
                "class_distribution_c2": float(agg_c2 / agg_sel) if agg_sel > 0 else 0.0,
            },
            "aggregate_lifecycle": {
                "total_batches_considered": sum(s["batch_lifecycle"]["batches_considered"] for s in event_summaries.values()),
                "total_batches_updated": sum(s["batch_lifecycle"]["batches_updated"] for s in event_summaries.values()),
                "total_batches_skipped": sum(s["batch_lifecycle"]["batches_skipped"] for s in event_summaries.values()),
                "total_batches_failed": sum(s["batch_lifecycle"]["batches_failed"] for s in event_summaries.values()),
                "total_zero_selection_batches": sum(s["batch_lifecycle"]["zero_selection_batches"] for s in event_summaries.values()),
            },
            "aggregate_optimization": {
                "overall_loss_mean": float(np.mean(all_losses)) if all_losses else None,
                "overall_grad_norm_mean": float(np.mean(all_grads)) if all_grads else None,
            },
            "system_telemetry": {
                "total_runtime_seconds": total_cell_runtime,
                "peak_vram_mb": peak_vram,
                "param_hash_before": param_hash_before,
                "param_hash_after": param_hash_after,
                "buf_hash_before": buf_hash_before,
                "buf_hash_after": buf_hash_after,
            },
            "events": event_summaries,
        }

        with open(cell_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(cell_summary, f, indent=2)

        # Write cell manifest
        cell_manifest = {
            "cell_id": cell_id,
            "cell_name": cell_name,
            "role": role,
            "protocol_version": self.cfg["protocol_version"],
            "seed": self.checkpoint_seed,
            "stream_order": self.stream_order,
            "reset_mode": self.reset_mode,
            "batch_size": self.batch_size,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_f1": worst_f1,
            "worst_event_id": worst_eid,
            "total_runtime_seconds": total_cell_runtime,
            "events_evaluated": [int(k) for k in event_summaries.keys()],
        }
        with open(cell_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(cell_manifest, f, indent=2)

        print(f"Cell {cell_id} COMPLETE: Macro F1 = {macro_f1:.4f} | Pooled F1 = {pooled_f1:.4f} | Sel% = {cell_summary['aggregate_selection']['selected_pixel_fraction']*100:.2f}%")
        print("=" * 80)
        return cell_summary

    def generate_master_repeat_tables(
        self,
        cell_summaries: List[Dict[str, Any]],
    ) -> Tuple[Path, Path]:
        """Generates master repeat JSON and Markdown tables."""
        master_json_path = self.output_root / "master_repeat_matrix.json"
        master_md_path = self.output_root / "master_repeat_matrix.md"

        master_record = {
            "protocol_version": self.cfg["protocol_version"],
            "work_package": self.cfg["work_package"],
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_cells": len(cell_summaries),
            "screening_specification": {
                "checkpoint_seed": self.checkpoint_seed,
                "stream_order": self.stream_order,
                "reset_mode": self.reset_mode,
                "batch_size": self.batch_size,
            },
            "cells": cell_summaries,
        }

        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(master_record, f, indent=2)

        # Generate Markdown
        md_lines = [
            "# ChangeGuard Plan V3 Work Package E2.3: Master Repeat Matrix Table",
            "",
            f"**Generated:** {master_record['generated_at_utc']}  ",
            f"**Base Checkpoint:** SNUNet_ECAM Seed {self.checkpoint_seed} (`{self.expected_checkpoint_sha256[:8]}...`)  ",
            f"**Evaluation Partition:** Development 6 Events ($N=2,826$ patches)  ",
            f"**Stream Protocol:** Order B (`[1111008, 1111003, 559, 520, 437, 279]`), `{self.reset_mode}`, Batch Size {self.batch_size}  ",
            f"**Decision Threshold:** $\\theta = 0.50$ (Uniform, non-tuned across all evaluations)  ",
            "",
            "---",
            "",
            "## 1. Summary Performance & Telemetry Table",
            "",
            "| Cell ID | Role / Strategy | Threshold | Macro F1 | Pooled F1 | Worst Ev F1 (Min/ID) | Sel Pixel % | Sel c0% | Sel c1% | Sel c2% | Batches (Upd/Skip/Tot) | Loss (Mean) | Grad Norm | Peak VRAM | Runtime |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for c in cell_summaries:
            cid = c["cell_id"]
            role = c.get("role", "REPEAT")
            tau = c["confidence_threshold"]
            mf1 = f"{c['macro_f1']:.4f}"
            pf1 = f"{c['pooled_f1']:.4f}"
            wf1 = f"{c['worst_event_f1']:.4f} (#{c['worst_event_id']})"
            sel = f"{c['aggregate_selection']['selected_pixel_fraction']*100:.2f}%"
            c0 = f"{c['aggregate_selection']['class_distribution_c0']*100:.1f}%"
            c1 = f"{c['aggregate_selection']['class_distribution_c1']*100:.1f}%"
            c2 = f"{c['aggregate_selection']['class_distribution_c2']*100:.1f}%"
            batches = f"{c['aggregate_lifecycle']['total_batches_updated']}/{c['aggregate_lifecycle']['total_batches_skipped']}/{c['aggregate_lifecycle']['total_batches_considered']}"
            loss = f"{c['aggregate_optimization']['overall_loss_mean']:.5f}" if c["aggregate_optimization"]["overall_loss_mean"] is not None else "N/A"
            grad = f"{c['aggregate_optimization']['overall_grad_norm_mean']:.5f}" if c["aggregate_optimization"]["overall_grad_norm_mean"] is not None else "N/A"
            vram = f"{c['system_telemetry']['peak_vram_mb']:.1f} MB" if c["system_telemetry"]["peak_vram_mb"] is not None else "N/A"
            rt = f"{c['system_telemetry']['total_runtime_seconds']:.1f}s"

            md_lines.append(
                f"| **{cid}** | {role} | $\\tau={tau:.2f}$ | {mf1} | {pf1} | {wf1} | {sel} | {c0} | {c1} | {c2} | {batches} | {loss} | {grad} | {vram} | {rt} |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 2. Event-Level Macro F1 Breakdown",
            "",
            "| Cell ID | Event 1111008 | Event 1111003 | Event 559 | Event 520 | Event 437 | Event 279 | Macro F1 | Pooled F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for c in cell_summaries:
            cid = c["cell_id"]
            evs = c["events"]
            e1008 = f"{evs.get('1111008', {}).get('f1', 0.0):.4f}"
            e1003 = f"{evs.get('1111003', {}).get('f1', 0.0):.4f}"
            e559 = f"{evs.get('559', {}).get('f1', 0.0):.4f}"
            e520 = f"{evs.get('520', {}).get('f1', 0.0):.4f}"
            e437 = f"{evs.get('437', {}).get('f1', 0.0):.4f}"
            e279 = f"{evs.get('279', {}).get('f1', 0.0):.4f}"
            mf1 = f"**{c['macro_f1']:.4f}**"
            pf1 = f"**{c['pooled_f1']:.4f}**"
            md_lines.append(
                f"| **{cid}** | {e1008} | {e1003} | {e559} | {e520} | {e437} | {e279} | {mf1} | {pf1} |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 3. Comparison with Discovery Screen (Seed 0, Order A)",
            "",
            "| Configuration | Seed 0 / Order A Macro F1 | Seed 1 / Order B Macro F1 | Delta Macro F1 | Seed 0 / Order A Pooled F1 | Seed 1 / Order B Pooled F1 | Delta Pooled F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Retained Base (tau=0.90)** | 0.5733 | {cell_summaries[0]['macro_f1']:.4f} | {cell_summaries[0]['macro_f1'] - 0.57331846:+.4f} | 0.4419 | {cell_summaries[0]['pooled_f1']:.4f} | {cell_summaries[0]['pooled_f1'] - 0.44191174:+.4f} |",
            f"| **Decisive Contrast (tau=0.00)** | 0.5733 | {cell_summaries[1]['macro_f1']:.4f} | {cell_summaries[1]['macro_f1'] - 0.57326982:+.4f} | 0.4416 | {cell_summaries[1]['pooled_f1']:.4f} | {cell_summaries[1]['pooled_f1'] - 0.44160649:+.4f} |",
            "",
            "---",
            "",
            "## 4. Methodological Invariants",
            "1. **Seed & Order Invariance Check:** Verifies whether confidence filtering (tau=0.90 vs tau=0.00) exhibits consistent behavior across model seeds (Seed 1) and reverse streaming orders (Order B).",
            "2. **Conservation of Confusion:** Verified identically for all events (TP + FP + FN + TN = valid_pixels).",
            "3. **Zero Test Access:** The 10 sealed-test events were strictly quarantined.",
            "",
        ])

        with open(master_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"Master Repeat Matrix Tables Generated Successfully:")
        print(f"  JSON: {master_json_path} (SHA-256: {compute_file_sha256(master_json_path)})")
        print(f"  Markdown: {master_md_path} (SHA-256: {compute_file_sha256(master_md_path)})")
        print("=" * 80)
        return master_json_path, master_md_path

    def run(self) -> None:
        """Executes all locked repeat cells."""
        preflight_info = self.execute_preflight_checks()

        cell_summaries = []
        for cell_spec in self.repeat_cells:
            summary = self.execute_repeat_cell(cell_spec, preflight_info)
            cell_summaries.append(summary)

        self.generate_master_repeat_tables(cell_summaries)


def main():
    parser = argparse.ArgumentParser(description="Run ChangeGuard Plan V3 Work Package E2.3 Repeat Matrix")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "changeguard" / "e2_3_repeat_matrix_v1.json",
        help="Path to locked E2.3 repeat matrix configuration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for model forward/backward (default: cuda:0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory for repeat artifacts",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test on Event 520 (head 50 samples)",
    )

    args = parser.parse_args()

    if args.smoke:
        print("Notice: Running in SMOKE MODE on Event 520 (head 50 samples).")

    runner = E23RepeatMatrixRunner(
        config_path=args.config,
        device_str=args.device,
        output_dir=args.output_dir,
        smoke_mode=args.smoke,
    )
    runner.run()


if __name__ == "__main__":
    main()

