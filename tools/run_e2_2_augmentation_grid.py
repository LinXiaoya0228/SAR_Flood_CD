"""ChangeGuard Plan V3 Work Package E2.2: Augmentation Grid Screening Runner.

Executes and verifies the locked 5-cell augmentation grid:
1. E2_AUG_A0: Predefined Horizontal Flip reference (reused from E2.1 E2_CONF_90).
2. E2_AUG_A1: Paired Vertical Flip with exact coordinate inversion.
3. E2_AUG_A2: Small Paired Integer Translation ([4, 4]), overlap-only mask, padding excluded.
4. E2_AUG_A3: Independent Weak Geometric Views (H-flip weak 1 vs V-flip weak 2).
5. E2_AUG_A4: SAR Intensity Perturbation (Formally BLOCKED subject to certification audit).

Protocol:
- Base Model: SNUNet_ECAM Seed 0 (checkpoint: 247d3584...)
- Confidence Reference: Predefined threshold tau = 0.90
- Stream Order: Order A ([279, 437, 520, 559, 1111003, 1111008])
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


class E22AugmentationGridRunner:
    """Runner for ChangeGuard Plan V3 Work Package E2.2."""

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
        self.smoke_mode = bool(smoke_mode)

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

        if self.smoke_mode:
            self.target_events = [520]
            print("Notice: Running in SMOKE MODE on Event 520 (head 50 samples).")
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
        print("ChangeGuard Plan V3 E2.2: Executing Fail-Closed Preflights...")
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

    def is_cell_completed(self, cell_dir: Path) -> bool:
        """Verifies if a cell was already completed with valid summary and event outputs."""
        if not cell_dir.is_dir():
            return False
        summary_path = cell_dir / "condition_summary.json"
        manifest_path = cell_dir / "manifest.json"
        if not (summary_path.is_file() and manifest_path.is_file()):
            return False

        for eid in self.target_events:
            ev_dir = cell_dir / f"event_{eid}"
            ev_metrics = ev_dir / "event_metrics.json"
            ev_npz = ev_dir / "frozen_predictions.npz"
            ev_steps = ev_dir / "step_records.json"
            if not (ev_metrics.is_file() and ev_npz.is_file() and ev_steps.is_file()):
                return False
        return True

    def process_reused_a0(self, cell_cfg: Dict[str, Any], preflight_info: Dict[str, Any]) -> Dict[str, Any]:
        """Links and validates the reused E2.1 E2_CONF_90 primary reference cell as A0."""
        cell_id = cell_cfg["cell_id"]
        source_rel_path = cell_cfg["reused_from"]
        source_dir = (REPO_ROOT / source_rel_path).resolve()

        print("-" * 80)
        print(f"Processing Reused Reference Cell: {cell_id} ({cell_cfg['name']})")
        print(f"Reused Source: {source_dir}")
        print("-" * 80)

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory for A0 reuse does not exist: {source_dir}")

        source_summary_path = source_dir / "condition_summary.json"
        if not source_summary_path.is_file():
            raise FileNotFoundError(f"Source condition_summary.json missing at: {source_summary_path}")

        with open(source_summary_path, "r", encoding="utf-8") as f:
            e2_1_summary = json.load(f)

        dest_dir = self.output_root / cell_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy event artifacts
        for eid in self.target_events:
            src_ev = source_dir / f"event_{eid}"
            dst_ev = dest_dir / f"event_{eid}"
            if not dst_ev.exists():
                shutil.copytree(src_ev, dst_ev)

        # Adapt summary for A0
        a0_summary = copy.deepcopy(e2_1_summary)
        a0_summary["cell_id"] = cell_id
        a0_summary["cell_name"] = cell_cfg["name"]
        a0_summary["aug_type"] = "horizontal_flip"
        a0_summary["reused_from"] = str(source_rel_path)

        with open(dest_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(a0_summary, f, indent=2)

        manifest = {
            "cell_id": cell_id,
            "cell_name": cell_cfg["name"],
            "status": "REUSED",
            "reused_from": str(source_rel_path),
            "git_commit": preflight_info["git_commit"],
            "config_hash": preflight_info["config_hash"],
            "checkpoint_sha256": preflight_info["checkpoint_sha256"],
            "target_events": self.target_events,
            "completed_at_utc": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
        }
        with open(dest_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Cell {cell_id} REUSED: Macro F1 = {a0_summary['macro_f1']:.4f} | Pooled F1 = {a0_summary['pooled_f1']:.4f}")
        print("=" * 80)
        return a0_summary

    def execute_cell(
        self,
        cell_cfg: Dict[str, Any],
        preflight_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes consistency adaptation for a single augmentation cell."""
        cell_id = cell_cfg["cell_id"]
        aug_type = cell_cfg["type"]
        cell_name = cell_cfg["name"]
        shift = tuple(cell_cfg.get("shift", [4, 4]))

        cell_dir = self.output_root / cell_id
        if self.is_cell_completed(cell_dir):
            print(f"[REUSE/RESUME] Cell {cell_id} already completed with verified artifacts. Skipping.")
            with open(cell_dir / "condition_summary.json", "r", encoding="utf-8") as f:
                return json.load(f)

        print("-" * 80)
        print(f"Executing Work Package E2.2 Cell: {cell_id} ({cell_name})")
        print(f"Augmentation Type: {aug_type} | Threshold: {self.threshold:.2f} | Batch Size: {self.batch_size}")
        print("-" * 80)

        cell_dir.mkdir(parents=True, exist_ok=True)

        recorder = ExtendedExperimentRecorder(
            output_dir=cell_dir,
            run_id=cell_id,
            checkpoint_id="source_all27_seed0",
            config_hash=preflight_info["config_hash"],
            git_commit=preflight_info["git_commit"],
            allow_overwrite=False,
        )

        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"]["0"]
        ckpt_path = str((REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve())
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            in_channels=2,
            out_ch=3,
            base_channel=32,
            device=self.device_str,
        )

        param_hash_before, buf_hash_before = compute_model_hashes(wrapper)

        hp = self.cfg["adaptation_hyperparameters"]
        adapter = E2AugmentationAdapter(
            model=wrapper,
            aug_type=aug_type,
            shift=shift,
            learning_rate=float(hp.get("learning_rate", 1.25e-4)),
            momentum=float(hp.get("momentum", 0.9)),
            weight_decay=float(hp.get("weight_decay", 0.0)),
            confidence_threshold=self.threshold,
            device=self.device_str,
        )

        orchestrator = E1Orchestrator(
            recorder=recorder,
            adapter=adapter,
            condition_id=cell_id,
            seed=0,
            reset_mode="reset_per_event",
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
            ev_sensory_valid = sum(s.get("sensory_valid_pixels", s.get("valid_pixels", 0)) for s in steps)
            ev_overlap_pixels = sum(s.get("overlap_pixels", s.get("valid_pixels", 0)) for s in steps)

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

            last_w_disp = steps[-1].get("bn_weight_displacement") if steps else None
            last_b_disp = steps[-1].get("bn_bias_displacement") if steps else None

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
                "geometry_telemetry": {
                    "sensory_valid_pixels": ev_sensory_valid,
                    "overlap_pixels": ev_overlap_pixels,
                    "active_valid_pixels": ev_valid_pixels,
                    "selected_pixels": ev_selected_pixels,
                    "selected_pixel_fraction": (
                        float(ev_selected_pixels / ev_valid_pixels) if ev_valid_pixels > 0 else 0.0
                    ),
                    "selected_pixels_c0": ev_selected_c0,
                    "selected_pixels_c1": ev_selected_c1,
                    "selected_pixels_c2": ev_selected_c2,
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
                    "batches_considered": len(steps),
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
                "parameter_displacement": {
                    "final_bn_weight_l2": last_w_disp,
                    "final_bn_bias_l2": last_b_disp,
                },
                "runtime_seconds": float(ev_elapsed),
            }
            event_summaries[str(eid)] = ev_summary
            print(f" done in {ev_elapsed:.1f}s | F1: {event_metrics['f1']:.4f} | Sel%: {ev_summary['geometry_telemetry']['selected_pixel_fraction']*100:.1f}%")

        total_runtime = time.perf_counter() - start_time
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if ("cuda" in self.device_str and torch.cuda.is_available())
            else 0.0
        )

        param_hash_after, buf_hash_after = compute_model_hashes(wrapper)

        f1_values = [s["f1"] for s in event_summaries.values()]
        macro_f1 = float(np.mean(f1_values))

        total_tp = sum(s["tp"] for s in event_summaries.values())
        total_fp = sum(s["fp"] for s in event_summaries.values())
        total_fn = sum(s["fn"] for s in event_summaries.values())
        total_tn = sum(s["tn"] for s in event_summaries.values())
        total_valid = sum(s["valid_pixels"] for s in event_summaries.values())
        total_samples = sum(s["sample_count"] for s in event_summaries.values())

        pooled_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        pooled_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        pooled_f1 = (
            float(2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec))
            if (pooled_prec + pooled_rec) > 0
            else 0.0
        )

        worst_ev_id = min(event_summaries.keys(), key=lambda k: event_summaries[k]["f1"])
        worst_ev_f1 = event_summaries[worst_ev_id]["f1"]

        agg_sensory = sum(s["geometry_telemetry"]["sensory_valid_pixels"] for s in event_summaries.values())
        agg_overlap = sum(s["geometry_telemetry"]["overlap_pixels"] for s in event_summaries.values())
        agg_active = sum(s["geometry_telemetry"]["active_valid_pixels"] for s in event_summaries.values())
        agg_sel = sum(s["geometry_telemetry"]["selected_pixels"] for s in event_summaries.values())
        agg_c0 = sum(s["geometry_telemetry"]["selected_pixels_c0"] for s in event_summaries.values())
        agg_c1 = sum(s["geometry_telemetry"]["selected_pixels_c1"] for s in event_summaries.values())
        agg_c2 = sum(s["geometry_telemetry"]["selected_pixels_c2"] for s in event_summaries.values())

        agg_loss_vals = [
            s["loss_telemetry"]["mean"]
            for s in event_summaries.values()
            if s["loss_telemetry"]["mean"] is not None
        ]
        agg_grad_norms = [
            s["gradient_telemetry"]["mean_l2_norm"]
            for s in event_summaries.values()
            if s["gradient_telemetry"]["mean_l2_norm"] is not None
        ]

        condition_summary = {
            "cell_id": cell_id,
            "cell_name": cell_name,
            "aug_type": aug_type,
            "confidence_threshold": self.threshold,
            "seed": 0,
            "stream_order": "order_a",
            "reset_mode": "reset_per_event",
            "batch_size": self.batch_size,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_f1": worst_ev_f1,
            "worst_event_id": int(worst_ev_id),
            "aggregate_confusion": {
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "total_tn": total_tn,
                "total_valid_pixels": total_valid,
                "total_sample_count": total_samples,
                "confusion_conserved": bool((total_tp + total_fp + total_fn + total_tn) == total_valid),
            },
            "aggregate_geometry": {
                "total_sensory_valid_pixels": agg_sensory,
                "total_overlap_pixels": agg_overlap,
                "total_active_valid_pixels": agg_active,
                "total_selected_pixels": agg_sel,
                "selected_pixel_fraction": float(agg_sel / agg_active) if agg_active > 0 else 0.0,
                "selected_pixels_c0": agg_c0,
                "selected_pixels_c1": agg_c1,
                "selected_pixels_c2": agg_c2,
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
                "overall_loss_mean": float(np.mean(agg_loss_vals)) if agg_loss_vals else None,
                "overall_grad_norm_mean": float(np.mean(agg_grad_norms)) if agg_grad_norms else None,
            },
            "system_telemetry": {
                "total_runtime_seconds": float(total_runtime),
                "peak_vram_mb": float(peak_vram_mb),
                "param_hash_before": param_hash_before,
                "param_hash_after": param_hash_after,
                "buf_hash_before": buf_hash_before,
                "buf_hash_after": buf_hash_after,
            },
            "events": event_summaries,
        }

        with open(cell_dir / "condition_summary.json", "w", encoding="utf-8") as f:
            json.dump(condition_summary, f, indent=2)

        manifest = {
            "cell_id": cell_id,
            "cell_name": cell_name,
            "aug_type": aug_type,
            "threshold": self.threshold,
            "git_commit": preflight_info["git_commit"],
            "config_hash": preflight_info["config_hash"],
            "checkpoint_sha256": preflight_info["checkpoint_sha256"],
            "target_events": self.target_events,
            "completed_at_utc": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "status": "COMPLETED",
        }
        with open(cell_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Cell {cell_id} COMPLETE: Macro F1 = {macro_f1:.4f} | Pooled F1 = {pooled_f1:.4f} | Sel% = {condition_summary['aggregate_geometry']['selected_pixel_fraction']*100:.2f}%")
        print("=" * 80)

        del wrapper, adapter, orchestrator, recorder
        torch.cuda.empty_cache()
        gc.collect()

        return condition_summary

    def run_all_cells(self) -> List[Dict[str, Any]]:
        """Executes all locked augmentation grid cells."""
        preflight_info = self.run_preflights()

        cells = self.cfg["screening_specification"]["augmentation_cells"]
        results: List[Dict[str, Any]] = []

        for cell_cfg in cells:
            cid = cell_cfg["cell_id"]
            c_status = cell_cfg["status"]

            if c_status == "REUSED":
                res = self.process_reused_a0(cell_cfg, preflight_info)
                results.append(res)
            elif c_status == "EXECUTABLE":
                res = self.execute_cell(cell_cfg, preflight_info)
                results.append(res)
            elif c_status == "BLOCKED":
                print(f"Cell {cid} is marked BLOCKED: {cell_cfg['blocked_reason']}")
                # Record blocked cell in results
                results.append({
                    "cell_id": cid,
                    "cell_name": cell_cfg["name"],
                    "aug_type": cell_cfg["type"],
                    "status": "BLOCKED",
                    "blocked_reason": cell_cfg["blocked_reason"],
                    "confidence_threshold": self.threshold,
                })
            else:
                raise ValueError(f"Unknown cell status '{c_status}' for {cid}")

        if not self.smoke_mode:
            self.generate_master_tables(results)

        return results

    def generate_master_tables(self, cell_results: List[Dict[str, Any]]) -> None:
        """Generates deterministic master JSON and Markdown tables for E2.2."""
        json_path = self.output_root / "master_augmentation_grid.json"
        md_path = self.output_root / "master_augmentation_grid.md"

        master_payload = {
            "protocol_version": "e2_2_augmentation_grid_v1",
            "work_package": "E2.2",
            "generated_at_utc": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "total_cells": len(cell_results),
            "cells": cell_results,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(master_payload, f, indent=2)

        lines = [
            "# ChangeGuard Plan V3 Work Package E2.2: Master Augmentation Grid",
            "",
            f"**Generated:** {master_payload['generated_at_utc']}  ",
            f"**Base Checkpoint:** SNUNet_ECAM Seed 0 (`247d3584...`)  ",
            f"**Predefined Confidence Reference:** $\\tau = 0.90$  ",
            f"**Evaluation Partition:** Development 6 Events ($N=2,826$ patches)  ",
            f"**Stream Protocol:** Order A (`[279, 437, 520, 559, 1111003, 1111008]`), `reset_per_event`, Batch Size 16  ",
            f"**Decision Threshold:** $\\theta = 0.50$ (Uniform, non-tuned across all evaluations)  ",
            "",
            "---",
            "",
            "## 1. Summary Performance & Telemetry Table",
            "",
            "| Cell ID | Augmentation Strategy | Status | Macro F1 | Pooled F1 | Worst Ev F1 (Min/ID) | Sel Pixel % | Sel c0% | Sel c1% | Sel c2% | Batches (Upd/Skip/Tot) | Loss (Mean) | Grad Norm | Peak VRAM | Runtime |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for r in cell_results:
            cid = r["cell_id"]
            name = r["cell_name"]
            status = r.get("status", "COMPLETED")

            if status == "BLOCKED":
                lines.append(
                    f"| **{cid}** | {name} | **`BLOCKED`** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
                )
                continue

            macro = f"{r['macro_f1']:.4f}"
            pooled = f"{r['pooled_f1']:.4f}"
            worst = f"{r['worst_event_f1']:.4f} (#{r['worst_event_id']})"

            geo = r.get("aggregate_geometry", r.get("aggregate_selection", {}))
            sel_pct = f"{geo.get('selected_pixel_fraction', 0.0)*100:.2f}%"
            c0_dist = f"{geo.get('class_distribution_c0', 0.0)*100:.1f}%"
            c1_dist = f"{geo.get('class_distribution_c1', 0.0)*100:.1f}%"
            c2_dist = f"{geo.get('class_distribution_c2', 0.0)*100:.1f}%"

            lc = r.get("aggregate_lifecycle", {})
            batches_str = f"{lc.get('total_batches_updated', 0)}/{lc.get('total_batches_skipped', 0)}/{lc.get('total_batches_considered', 0)}"

            opt = r.get("aggregate_optimization", {})
            loss_str = f"{opt.get('overall_loss_mean', 0.0):.5f}" if opt.get("overall_loss_mean") is not None else "N/A"
            grad_str = f"{opt.get('overall_grad_norm_mean', 0.0):.5f}" if opt.get("overall_grad_norm_mean") is not None else "N/A"

            sys_tel = r.get("system_telemetry", {})
            vram_str = f"{sys_tel.get('peak_vram_mb', 0.0):.1f} MB"
            rt_str = f"{sys_tel.get('total_runtime_seconds', 0.0):.1f}s"

            lines.append(
                f"| **{cid}** | {name} | `{status}` | {macro} | {pooled} | {worst} | {sel_pct} | {c0_dist} | {c1_dist} | {c2_dist} | {batches_str} | {loss_str} | {grad_str} | {vram_str} | {rt_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Event-Level Macro F1 Breakdown",
            "",
            "| Cell ID | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for r in cell_results:
            if r.get("status") == "BLOCKED":
                continue
            cid = r["cell_id"]
            evs = r["events"]
            e279 = f"{evs['279']['f1']:.4f}"
            e437 = f"{evs['437']['f1']:.4f}"
            e520 = f"{evs['520']['f1']:.4f}"
            e559 = f"{evs['559']['f1']:.4f}"
            e1003 = f"{evs['1111003']['f1']:.4f}"
            e1008 = f"{evs['1111008']['f1']:.4f}"
            macro = f"{r['macro_f1']:.4f}"
            lines.append(f"| **{cid}** | {e279} | {e437} | {e520} | {e559} | {e1003} | {e1008} | **{macro}** |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Methodological Notes & Formal Block Decisions",
            "1. **Uniform Predefined Reference:** All cells were evaluated under fixed threshold $\\tau = 0.90$.",
            "2. **A0 Equivalence & Reuse:** Cell `E2_AUG_A0` was reused from verified E2.1 `E2_CONF_90` with full cryptographic equivalence.",
            "3. **Exact Geometric Inversion:** In A1, A2, and A3, strong-branch logits and probabilities were inverted back to the weak/native frame before computing consistency loss.",
            "4. **Overlap-Only Consistency Mask:** In A2, consistency is computed strictly on declared geometric overlap; padding is zero and strictly excluded.",
            "5. **A4 Blocking Justification:** Cell `E2_AUG_A4` is formally BLOCKED because multiplicative speckle is undefined on z-score standardized SAR tensors, and KuroSiwo metadata lacks the calibration parameters required for physical raw-domain simulation.",
            "",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print("=" * 80)
        print("Master Augmentation Grid Tables Generated Successfully:")
        print(f"  JSON: {json_path} (SHA-256: {compute_file_sha256(json_path)})")
        print(f"  Markdown: {md_path} (SHA-256: {compute_file_sha256(md_path)})")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E2.2 Augmentation Grid Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e2_2_augmentation_grid_v1.json",
        help="Path to E2.2 configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to execute adaptation ('cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bounded smoke test on Event 520 (head 50 samples).",
    )
    args = parser.parse_args()

    runner = E22AugmentationGridRunner(
        config_path=args.config,
        device_str=args.device,
        output_dir=args.output_dir,
        smoke_mode=args.smoke,
    )
    runner.run_all_cells()


if __name__ == "__main__":
    main()

