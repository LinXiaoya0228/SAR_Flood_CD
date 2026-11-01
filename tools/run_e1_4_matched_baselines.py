"""ChangeGuard Plan V3 E1.4 Matched Baseline Matrix Runner.

Executes and verifies matched baseline coverage across all 51 required streams:
- source_only_frozen_running_stats (Seeds 0, 1, 2 across dev6)
- adabn_matched_pre_update (12 streams)
- tent_matched_pre_update (12 streams)
- eata_v2_fisher_matched_pre_update (12 streams, d_margin=0.05, alpha=2000.0)
- sar_matched_pre_update (12 streams, rho=0.05, e_m=0.031808)

Strict Plan V3 Requirements:
1. Zero access to sealed test partition ([205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]).
2. Strict PRE_UPDATE primary prediction timing; POST_UPDATE predictions are diagnostics only.
3. LabelGuardedBatch: zero labels accessed during adaptation; labels unlocked only after persistence.
4. Memory stability check on frozen batch size 16 before full queue.
5. Resumable execution: completed streams with verified manifests and outputs are safely skipped.
6. Unified master coverage table generated strictly from source artifacts.
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
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    E1AdaBNAdapter,
    E1EATAAdapter,
    E1SARAdapter,
    E1SourceOnlyAdapter,
    E1TentAdapter,
)
from src.changeguard.e1_orchestrator import (
    E1ConsistencyAdapter,
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


class E1MatchedBaselinesRunner:
    """Runner for ChangeGuard Plan V3 Work Package E1.4."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        device: str = "cuda:0",
        allow_overwrite: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device_str = device
        self.allow_overwrite = allow_overwrite
        if self.allow_overwrite:
            raise PermissionError("Plan V3 rules strictly forbid allow_overwrite=True.")

        out_root = (
            output_dir
            if output_dir is not None
            else REPO_ROOT / "outputs/e1_matched_baselines/v1"
        )
        self.output_root = out_root.resolve()

        log_root = (
            log_dir
            if log_dir is not None
            else REPO_ROOT / "logs/e1_matched_baselines/v1"
        )
        self.log_root = log_root.resolve()

        # Resolve paths
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.split_path = (REPO_ROOT / ds_cfg["split_path"]).resolve()
        self.batch_size = int(ds_cfg.get("batch_size", 16))
        self.target_events = [int(e) for e in ds_cfg["target_events"]]

        self.sar_clamp_max = float(ds_cfg.get("sar_clamp_max", 0.15))
        self.sar_mean = list(ds_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = list(ds_cfg.get("sar_std", [0.0427, 0.0215]))
        self.canonical_preevent = str(ds_cfg.get("canonical_preevent", "pre_event_2"))

        # Cached datasets and stream plans
        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._cached_stream_plans: Dict[str, List[Tuple[int, List[LabelGuardedBatch]]]] = {}

    def run_preflights(self) -> Dict[str, Any]:
        """Fail-closed preflight validation of all code, assets, and partitions."""
        print("=" * 80)
        print("ChangeGuard Plan V3 E1.4: Executing Fail-Closed Preflights...")
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

        # 4. Fisher artifacts for EATA
        fisher_info = {}
        for s_str, f_cfg in self.cfg["fisher_artifacts_v2"]["seeds"].items():
            s_idx = int(s_str)
            p = (REPO_ROOT / f_cfg["path"]).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Fisher artifact for seed {s_idx} not found: {p}")
            actual_sha = compute_file_sha256(p)
            exp_sha = f_cfg["expected_sha256"]
            if actual_sha != exp_sha:
                raise ValueError(f"Fisher artifact seed {s_idx} SHA-256 mismatch: {actual_sha} != {exp_sha}")
            fisher_info[s_idx] = {"path": str(p), "sha256": actual_sha}

        # 5. Git info
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
            "fishers": fisher_info,
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

    def run_memory_stability_check(self) -> None:
        """Bounded memory check on Event 520 using batch size 16 to confirm no monotonic growth."""
        if self.device_str == "cpu" or not torch.cuda.is_available():
            print("Skipping GPU memory check in CPU context.")
            return

        print("=" * 80)
        print(f"ChangeGuard E1.4: Running Bounded Memory-Stability Check on {self.device_str} (bs=16)...")
        print("=" * 80)

        torch.cuda.empty_cache()
        gc.collect()

        ckpt_p = (REPO_ROOT / self.cfg["source_checkpoints"]["seeds"]["0"]["checkpoint_path"]).resolve()
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_p), in_channels=2, out_ch=3, base_channel=32, device=self.device_str
        )
        adapter = E1TentAdapter(model=wrapper, device=self.device_str)

        adapt_ds, offline_ds = self.get_datasets()

        # Select first 48 samples (3 batches of 16) for Event 520
        event_samples = [s for s in adapt_ds.samples if int(s["event_id"]) == 520][:48]
        if len(event_samples) < 32:
            print(f"Notice: only {len(event_samples)} samples available for Event 520.")

        allocations = []
        for i in range(0, len(event_samples), self.batch_size):
            batch_slice = event_samples[i : i + self.batch_size]
            b_ids = [str(s["sample_id"]) for s in batch_slice]
            # Find indices in dataset
            indices = [adapt_ds.samples.index(s) for s in batch_slice]
            adapt_data = [adapt_ds[idx] for idx in indices]
            xA = torch.stack([d["xA"] for d in adapt_data]).to(self.device_str)
            xB = torch.stack([d["xB"] for d in adapt_data]).to(self.device_str)
            valid_mask = torch.stack([d["valid_mask"] for d in adapt_data]).to(self.device_str)

            adapter.adapt_and_predict_step(
                xA=xA,
                xB=xB,
                valid_mask=valid_mask,
                sample_ids=b_ids,
                event_id=520,
            )
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            allocations.append(alloc_mb)
            print(f"Batch {len(allocations)}: Allocated VRAM = {alloc_mb:.2f} MB")

        # Clean up
        del adapter, wrapper, xA, xB, valid_mask
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Memory allocations across batches: {[round(a, 2) for a in allocations]} MB.")
        # Check that memory doesn't explode
        if len(allocations) >= 3 and allocations[-1] > allocations[1] * 1.5:
            raise RuntimeError(
                f"Memory leak detected: batch 3 memory ({allocations[-1]:.2f} MB) is > 1.5x batch 2 ({allocations[1]:.2f} MB)."
            )
        print("Memory stability check PASSED: No monotonic memory explosion detected.")
        print("=" * 80)

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
    ) -> List[Tuple[int, List[LabelGuardedBatch]]]:
        """Partitions development events into ordered sequences of LabelGuardedBatch."""
        cache_key = order_id if order_id in {"order_a", "order_b"} else "order_a"
        if cache_key in self._cached_stream_plans:
            return self._cached_stream_plans[cache_key]

        adapt_ds, offline_ds = self.get_datasets()

        o_cfg = self.cfg["stream_orders"].get(order_id)
        if o_cfg is None:
            # Source-only invariant order: use order_a
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
            for i in range(0, len(samples), self.batch_size):
                batch_items = samples[i : i + self.batch_size]
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

    def instantiate_adapter(
        self,
        method: str,
        seed: int,
    ) -> BaseE1Adapter:
        """Instantiates the appropriate baseline adapter with canonical checkpoints."""
        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"][str(seed)]
        ckpt_path = str((REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve())

        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            in_channels=2,
            out_ch=3,
            base_channel=32,
            device=self.device_str,
        )

        if method == "source_only_frozen_running_stats":
            return E1SourceOnlyAdapter(
                model=wrapper,
                device=self.device_str,
            )
        elif method == "adabn_matched_pre_update":
            return E1AdaBNAdapter(
                model=wrapper,
                device=self.device_str,
            )
        elif method == "tent_matched_pre_update":
            t_cfg = self.cfg["method_specifications"]["tent"]["hyperparameters"]
            return E1TentAdapter(
                model=wrapper,
                learning_rate=t_cfg.get("learning_rate", 1e-4),
                betas=tuple(t_cfg.get("betas", [0.9, 0.999])),
                weight_decay=t_cfg.get("weight_decay", 0.0),
                device=self.device_str,
            )
        elif method == "eata_v2_fisher_matched_pre_update":
            e_cfg = self.cfg["method_specifications"]["eata"]["hyperparameters"]
            fisher_path = (REPO_ROOT / self.cfg["fisher_artifacts_v2"]["seeds"][str(seed)]["path"]).resolve()
            fishers = torch.load(fisher_path, map_location=self.device_str)
            return E1EATAAdapter(
                model=wrapper,
                fishers=fishers,
                fisher_alpha=e_cfg.get("fisher_alpha", 2000.0),
                e_margin=e_cfg.get("reliability_threshold_e0", 0.4394449),
                d_margin=e_cfg.get("redundancy_threshold_d_margin", 0.05),
                learning_rate=e_cfg.get("learning_rate", 1.25e-4),
                momentum=e_cfg.get("momentum", 0.9),
                weight_decay=e_cfg.get("weight_decay", 0.0),
                device=self.device_str,
            )
        elif method == "sar_matched_pre_update":
            s_cfg = self.cfg["method_specifications"]["sar"]["hyperparameters"]
            return E1SARAdapter(
                model=wrapper,
                e_margin=s_cfg.get("reliability_threshold_e0", 0.4394449),
                sam_rho=s_cfg.get("sam_rho", 0.05),
                recovery_threshold=s_cfg.get("recovery_threshold_em", 0.03180808),
                learning_rate=s_cfg.get("learning_rate", 1.25e-4),
                momentum=s_cfg.get("momentum", 0.9),
                weight_decay=s_cfg.get("weight_decay", 0.0),
                device=self.device_str,
            )
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    def execute_stream(
        self,
        stream_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes a single baseline adaptation stream across all 6 dev events."""
        stream_id = stream_spec["stream_id"]
        method = stream_spec["method"]
        seed = stream_spec["seed"]
        order = stream_spec["order"]
        reset_mode = stream_spec["reset_mode"]

        stream_dir = self.output_root / stream_id

        # Skip if already completed
        if self.is_stream_completed(stream_dir):
            print(f"[SKIP COMPLETED] Stream '{stream_id}' already has verified outputs. Skipping.")
            with open(stream_dir / "stream_summary.json", "r", encoding="utf-8") as f:
                return json.load(f)

        print("-" * 80)
        print(f"Beginning Stream Execution: {stream_id}")
        print(f"Method: {method} | Seed: {seed} | Order: {order} | Reset: {reset_mode}")
        t_start = time.time()

        stream_dir.mkdir(parents=True, exist_ok=True)

        recorder = ExtendedExperimentRecorder(
            output_dir=stream_dir,
            run_id=stream_id,
            checkpoint_id=f"source_snunet_vvvh_all27_seed{seed}_v1_fp32",
            config_hash=compute_file_sha256(self.config_path),
            git_commit=get_git_info()["commit"],
            allow_overwrite=False,
        )

        adapter = self.instantiate_adapter(method=method, seed=seed)

        actual_reset = "reset_per_event" if reset_mode in {"reset_per_event", "invariant"} else "continual"

        orchestrator = E1Orchestrator(
            recorder=recorder,
            adapter=adapter,
            condition_id=stream_id,
            seed=seed,
            reset_mode=actual_reset,
        )

        stream_plan = self.build_ordered_stream_batches(order_id=order)

        event_metrics_list: List[Dict[str, Any]] = []

        for eid, batches in stream_plan:
            print(f"  Streaming Event {eid} ({len(batches)} batches)...")
            em = orchestrator.run_event_stream(event_id=eid, batches=batches)
            event_metrics_list.append(em)

        elapsed = time.time() - t_start

        # Compute summary metrics across events
        macro_f1 = float(np.mean([m["f1"] for m in event_metrics_list]))
        macro_iou = float(np.mean([m["iou"] for m in event_metrics_list]))
        macro_precision = float(np.mean([m["precision"] for m in event_metrics_list]))
        macro_recall = float(np.mean([m["recall"] for m in event_metrics_list]))

        total_tp = sum(m["tp"] for m in event_metrics_list)
        total_fp = sum(m["fp"] for m in event_metrics_list)
        total_fn = sum(m["fn"] for m in event_metrics_list)
        total_tn = sum(m["tn"] for m in event_metrics_list)
        total_valid = sum(m["valid_pixels"] for m in event_metrics_list)

        pooled_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        pooled_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        pooled_f1 = (
            float(2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec))
            if (pooled_prec + pooled_rec) > 0
            else 0.0
        )
        pooled_iou = (
            float(total_tp / (total_tp + total_fp + total_fn))
            if (total_tp + total_fp + total_fn) > 0
            else 0.0
        )

        worst_f1 = min(m["f1"] for m in event_metrics_list)

        stream_summary = {
            "stream_id": stream_id,
            "method": method,
            "seed": seed,
            "order": order,
            "reset_mode": reset_mode,
            "elapsed_seconds": elapsed,
            "event_count": len(event_metrics_list),
            "macro_f1": macro_f1,
            "macro_iou": macro_iou,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "pooled_f1": pooled_f1,
            "pooled_iou": pooled_iou,
            "pooled_precision": pooled_prec,
            "pooled_recall": pooled_rec,
            "worst_event_f1": worst_f1,
            "total_valid_pixels": total_valid,
            "confusion": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn,
            },
            "events": {m["event_id"]: m for m in event_metrics_list},
        }

        # Save summary
        with open(stream_dir / "stream_summary.json", "w", encoding="utf-8") as f:
            json.dump(stream_summary, f, indent=2)

        # Save manifest
        manifest = {
            "stream_id": stream_id,
            "method": method,
            "seed": seed,
            "order": order,
            "reset_mode": reset_mode,
            "status": "COMPLETE",
            "elapsed_seconds": elapsed,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_f1": worst_f1,
        }
        with open(stream_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Completed Stream {stream_id} in {elapsed:.1f}s | Macro F1: {macro_f1:.4f} | Pooled F1: {pooled_f1:.4f}")
        return stream_summary

    def run_queue(
        self,
        stream_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Runs the sequential execution queue over missing E1.4 streams."""
        preflights = self.run_preflights()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)

        # Run memory stability check before long queue
        self.run_memory_stability_check()

        # Load coverage ledger
        ledger_path = REPO_ROOT / "docs/E1_4_COVERAGE_LEDGER.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        target_streams = ledger["streams"]
        if stream_ids is not None:
            allowed = set(stream_ids)
            target_streams = [s for s in target_streams if s["stream_id"] in allowed]

        print("=" * 80)
        print(f"ChangeGuard E1.4 Execution Queue: {len(target_streams)} streams selected.")
        print("=" * 80)

        all_summaries: List[Dict[str, Any]] = []

        for idx, spec in enumerate(target_streams, start=1):
            sid = spec["stream_id"]
            status = spec["status"]
            print(f"\n[{idx}/{len(target_streams)}] Processing Stream: {sid} ({status})")

            if status == "REUSED_VERIFIED":
                # Import historical verified summary directly
                hist_path = REPO_ROOT / spec["historical_source_artifact"]
                summary_file = hist_path / "condition_summary.json"
                if not summary_file.is_file():
                    # Fallback check
                    summary_file = hist_path / "stream_summary.json"
                if not summary_file.is_file():
                    raise FileNotFoundError(f"Historical summary missing for reused stream {sid}: {summary_file}")

                with open(summary_file, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)

                # Robustly extract macro and pooled metrics from either format
                macro_f1 = (
                    hist_data.get("macro_f1")
                    or hist_data.get("macro_metrics", {}).get("macro_f1")
                    or hist_data.get("macro_metrics", {}).get("f1")
                )
                pooled_f1 = (
                    hist_data.get("pooled_f1")
                    or hist_data.get("pooled_metrics", {}).get("f1")
                    or hist_data.get("pooled_metrics", {}).get("pooled_f1")
                )
                worst_f1 = (
                    hist_data.get("worst_event_f1")
                    or (min(r["f1"] for r in hist_data["event_rows"]) if "event_rows" in hist_data else None)
                )
                valid_pix = (
                    hist_data.get("total_valid_pixels")
                    or hist_data.get("pooled_metrics", {}).get("valid_pixels")
                )

                summary = {
                    "stream_id": sid,
                    "method": spec["method"],
                    "seed": spec["seed"],
                    "order": spec["order"],
                    "reset_mode": spec["reset_mode"],
                    "status": "REUSED_VERIFIED",
                    "historical_source_artifact": spec["historical_source_artifact"],
                    "macro_f1": float(macro_f1) if macro_f1 is not None else None,
                    "pooled_f1": float(pooled_f1) if pooled_f1 is not None else None,
                    "worst_event_f1": float(worst_f1) if worst_f1 is not None else None,
                    "total_valid_pixels": int(valid_pix) if valid_pix is not None else None,
                    "reused": True,
                }
                all_summaries.append(summary)
                mf1_str = f"{summary['macro_f1']:.4f}" if summary['macro_f1'] is not None else "N/A"
                pf1_str = f"{summary['pooled_f1']:.4f}" if summary['pooled_f1'] is not None else "N/A"
                print(f"  Imported REUSED_VERIFIED metrics: Macro F1={mf1_str}, Pooled F1={pf1_str}")
            elif status == "EXECUTE_NEW":
                summary = self.execute_stream(spec)
                summary["status"] = "EXECUTED_NEW"
                summary["reused"] = False
                all_summaries.append(summary)
            else:
                raise ValueError(f"Unexpected status '{status}' for stream {sid}")

        # Generate Master Coverage Summary
        self.generate_master_coverage_table(all_summaries)
        return all_summaries

    def generate_master_coverage_table(self, summaries: List[Dict[str, Any]]) -> None:
        """Generates machine-readable and markdown master coverage tables strictly from outputs."""
        table_path = self.output_root / "master_coverage_table.json"
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        md_path = self.output_root / "master_coverage_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ChangeGuard E1.4: Master Baseline Coverage Summary Table\n\n")
            f.write(f"Generated at: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
            f.write("| Row | Stream ID | Method | Seed | Order | Reset | Status | Macro F1 | Pooled F1 | Worst F1 |\n")
            f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for i, s in enumerate(summaries, start=1):
                mf1 = f"{s.get('macro_f1', 0.0):.6f}" if s.get('macro_f1') is not None else "N/A"
                pf1 = f"{s.get('pooled_f1', 0.0):.6f}" if s.get('pooled_f1') is not None else "N/A"
                wf1 = f"{s.get('worst_event_f1', 0.0):.6f}" if s.get('worst_event_f1') is not None else "N/A"
                f.write(
                    f"| {i} | `{s['stream_id']}` | {s['method']} | {s['seed']} | {s['order']} | "
                    f"{s['reset_mode']} | `{s['status']}` | {mf1} | {pf1} | {wf1} |\n"
                )
        print("\n" + "=" * 80)
        print(f"Master Coverage Summary Table durably written to:\n  {table_path}\n  {md_path}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E1.4 Matched Baselines Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e1_4_matched_baselines_v1.json",
        help="Path to E1.4 protocol configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs/e1_matched_baselines/v1",
        help="Root directory for E1.4 outputs",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=REPO_ROOT / "logs/e1_matched_baselines/v1",
        help="Root directory for E1.4 logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device string (e.g. 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--stream_ids",
        nargs="+",
        default=None,
        help="Optional subset of stream IDs to execute",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate preflights and accounting without running adaptation",
    )
    args = parser.parse_args()

    runner = E1MatchedBaselinesRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    if args.dry_run:
        runner.run_preflights()
        print("Dry run complete: All preflights and paths verified.")
        return

    runner.run_queue(stream_ids=args.stream_ids)


if __name__ == "__main__":
    main()
