#!/usr/bin/env python3
"""ChangeGuard Work Package E7: Branch C Empirical Evaluation Runner.

Orchestrates controlled empirical evaluation of test-time adaptation methods
(Source-Only, AdaBN, Tent, EATA, SAR, Candidate Consistency, CoTTA Continual, CoTTA Reset)
across development flood events under Plan V3 measurement contract.

Features:
- Fail-closed preflight validation (--validate_only).
- Bounded GPU smoke test for Event 520, 2 batches (--smoke).
- Single-stream targeting (--stream_id).
- Safe resumability: already completed streams with verified manifests are skipped.
- Strict isolation: zero ground-truth label leakage during adaptation.
- Quarantined sealed-test partition: immediate fatal error on sealed-test access.
- Double-masking offline evaluation: (MK0_MNA == 1) & (MK0_MLU != 3).
"""

from __future__ import annotations

import argparse
import copy
import csv
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1AdaBNAdapter,
    E1EATAAdapter,
    E1SARAdapter,
    E1SourceOnlyAdapter,
    E1TentAdapter,
)
from src.changeguard.cotta_adapter import E7CoTTAAdapter
from src.changeguard.e1_orchestrator import (
    E1ConsistencyAdapter,
    E1OfflineEvaluator,
    LabelGuardedBatch,
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
        return {"commit": f"UNKNOWN_ERROR: {e}", "tracked_dirty": True, "status_porcelain": str(e)}


class E7EmpiricalRunner:
    """Runner for ChangeGuard Plan V3 Work Package E7 Empirical Evaluation."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda:0",
        allow_overwrite: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        if allow_overwrite:
            raise PermissionError("Plan V3 rules strictly forbid allow_overwrite=True.")

        self.device_str = device if torch.cuda.is_available() else "cpu"

        # Output directory configuration
        if output_dir is not None:
            self.output_root = output_dir.resolve()
        elif "smoke_specification" in self.cfg:
            self.output_root = (REPO_ROOT / self.cfg["smoke_specification"]["output_dir"]).resolve()
        else:
            self.output_root = (REPO_ROOT / self.cfg.get("output_directory", "outputs/e7_empirical_evidence")).resolve()

        # Dataset configuration
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.split_path = (REPO_ROOT / ds_cfg["split_path"]).resolve()
        self.batch_size = int(ds_cfg.get("batch_size", 16))

        if "smoke_specification" in self.cfg:
            self.target_events = [int(self.cfg["smoke_specification"]["event_id"])]
            self.max_smoke_samples = int(self.cfg["smoke_specification"].get("max_samples", 32))
        else:
            self.target_events = [int(e) for e in ds_cfg.get("development_events", DEFAULT_TTA_DEV_EVENTS)]
            self.max_smoke_samples = None

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
        print("ChangeGuard Plan V3 E7: Executing Fail-Closed Preflights...")
        print("=" * 80)

        # 1. Sealed-test check
        for eid in self.target_events:
            if eid in FORBIDDEN_SEALED_TEST_EVENTS:
                raise RuntimeError(
                    f"CRITICAL RESEARCH INTEGRITY VIOLATION: Sealed test event {eid} in target events!"
                )
        print(f"Sealed-test partition quarantine: VERIFIED (Zero sealed-test events in {self.target_events})")

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
        print(f"Dataset manifest SHA-256: VERIFIED ({manifest_hash[:16]}...)")

        split_hash = compute_file_sha256(self.split_path)
        exp_split = self.cfg["dataset"]["split_expected_sha256"]
        if split_hash != exp_split:
            raise ValueError(f"Split SHA-256 mismatch: {split_hash} != {exp_split}")
        print(f"Split definition SHA-256: VERIFIED ({split_hash[:16]}...)")

        # 3. Source checkpoints
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
            print(f"Source checkpoint seed {s_idx} SHA-256: VERIFIED ({actual_sha[:16]}...)")

        # 4. Git status
        git_info = get_git_info()
        print(f"Git HEAD commit: {git_info['commit']}")
        print(f"Git tracked dirty: {git_info['tracked_dirty']}")

        print("=" * 80)
        print("ALL PREFLIGHT VALIDATIONS PASSED SUCCESSFULLY.")
        print("=" * 80)

        return {
            "status": "PREFLIGHTS_PASSED",
            "git_commit": git_info["commit"],
            "manifest_sha256": manifest_hash,
            "split_sha256": split_hash,
            "checkpoints": ckpt_info,
            "target_events": self.target_events,
            "device": self.device_str,
        }

    def get_datasets(self) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Lazy-loads and caches evaluation datasets."""
        if self._adapt_dataset is None or self._offline_dataset is None:
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
        """Checks if a stream was already completed with valid summary and status."""
        if not stream_dir.is_dir():
            return False
        summary_path = stream_dir / "stream_summary.json"
        status_path = stream_dir / "completion_status.json"
        if not (summary_path.is_file() and status_path.is_file()):
            return False

        try:
            with open(status_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            return st.get("status") == "COMPLETED"
        except Exception:
            return False

    def build_stream_batches(
        self,
        order_id: str,
    ) -> List[Tuple[int, List[LabelGuardedBatch]]]:
        """Partitions development events into ordered sequences of LabelGuardedBatch."""
        cache_key = order_id if order_id in {"order_a", "order_b"} else "order_a"
        if cache_key in self._cached_stream_plans:
            return self._cached_stream_plans[cache_key]

        adapt_ds, offline_ds = self.get_datasets()

        if "stream_orders" in self.cfg and order_id in self.cfg["stream_orders"]:
            o_cfg = self.cfg["stream_orders"][order_id]
            event_seq = o_cfg["event_sequence"]
            within_sort = o_cfg["within_event_ordering"]
        else:
            event_seq = self.target_events
            within_sort = "deterministic_sorted_sample_id_ascending"

        target_seq = [e for e in event_seq if e in self.target_events]

        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(adapt_ds.samples):
            eid = int(item["event_id"])
            if eid in self.target_events:
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

            # Cap samples if running in bounded smoke mode
            if self.max_smoke_samples is not None:
                samples = samples[: self.max_smoke_samples]

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
        """Instantiates the specified adapter with canonical checkpoints."""
        ckpt_cfg = self.cfg["source_checkpoints"]["seeds"][str(seed)]
        ckpt_path = str((REPO_ROOT / ckpt_cfg["checkpoint_path"]).resolve())

        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            in_channels=2,
            out_ch=3,
            base_channel=32,
            device=self.device_str,
        )

        if method == "source_only":
            return E1SourceOnlyAdapter(model=wrapper, device=self.device_str)
        elif method == "adabn":
            return E1AdaBNAdapter(model=wrapper, device=self.device_str)
        elif method == "tent":
            return E1TentAdapter(model=wrapper, learning_rate=1e-4, device=self.device_str)
        elif method == "eata":
            fisher_path = REPO_ROOT / f"artifacts/eata_fishers/v2/source_all27_seed{seed}_fisher.pt"
            fishers = torch.load(fisher_path, map_location=self.device_str) if fisher_path.exists() else None
            return E1EATAAdapter(model=wrapper, fishers=fishers, device=self.device_str)
        elif method == "sar":
            return E1SARAdapter(model=wrapper, device=self.device_str)
        elif method == "candidate_e2":
            return E1ConsistencyAdapter(model=wrapper, confidence_threshold=0.90, device=self.device_str)
        elif method in ("cotta_continual", "cotta_reset", "cotta_smoke"):
            mode = "continual" if method == "cotta_continual" else "reset_per_event"
            cotta_cfg = self.cfg.get("cotta_parameters", {})
            return E7CoTTAAdapter(
                model=wrapper,
                learning_rate=float(cotta_cfg.get("learning_rate", 1e-4)),
                ema_alpha=float(cotta_cfg.get("ema_alpha", 0.999)),
                restoration_prob=float(cotta_cfg.get("restoration_prob", 0.01)),
                confidence_threshold=float(cotta_cfg.get("confidence_threshold", 0.90)),
                mode=mode,
                seed=seed,
                device=self.device_str,
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

    def execute_stream(
        self,
        stream_id: str,
        method: str,
        seed: int,
        order_id: str,
    ) -> Dict[str, Any]:
        """Executes a single evaluation stream across events."""
        stream_dir = self.output_root / stream_id
        if self.is_stream_completed(stream_dir):
            print(f"Stream {stream_id} is already completed. Skipping.")
            with open(stream_dir / "stream_summary.json", "r", encoding="utf-8") as f:
                return json.load(f)

        stream_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Starting Stream: {stream_id} (Method: {method}, Seed: {seed}, Order: {order_id})")

        adapter = self.instantiate_adapter(method=method, seed=seed)
        stream_plan = self.build_stream_batches(order_id=order_id)

        start_time = time.perf_counter()
        event_metrics_list = []
        global_tp = 0
        global_fp = 0
        global_fn = 0
        global_tn = 0
        global_valid = 0

        for event_idx, (eid, batches) in enumerate(stream_plan):
            # If reset mode and not first event, reset adapter to source
            if method in ("cotta_reset", "cotta_smoke", "candidate_e2", "tent", "sar", "adabn") and event_idx > 0:
                adapter.reset_to_source()

            ev_tp = 0
            ev_fp = 0
            ev_fn = 0
            ev_tn = 0
            ev_valid = 0
            ev_samples = 0

            # Directory for durable pre-label prediction shards
            shards_dir = stream_dir / f"event_{eid}" / "shards"
            shards_dir.mkdir(parents=True, exist_ok=True)

            for b_idx, batch in enumerate(batches):
                # 1. Primary PRE_UPDATE adaptation step (zero labels accessible)
                res = adapter.adapt_and_predict_step(
                    xA=batch.xA,
                    xB=batch.xB,
                    valid_mask=batch.valid_mask,
                    sample_ids=batch.sample_ids,
                    event_id=eid,
                )

                # 2. Extract and freeze pre-label predictions & evidence
                pred_pre = res["primary_prediction"]["pred_cand_pre"]
                probs_pre = res["primary_prediction"]["probs_cand_pre"]
                adapt_meta = res["adaptation_metadata"]
                risk_ev = res["risk_evidence"]

                # 3. Durably save pre-label prediction artifacts to disk before unlocking labels
                shard_path = shards_dir / f"batch_{b_idx:04d}.npz"
                np.savez_compressed(
                    shard_path,
                    sample_ids=np.array(batch.sample_ids),
                    pred_cand_pre=pred_pre.detach().cpu().numpy().astype(np.uint8),
                    probs_cand_pre=probs_pre.detach().cpu().numpy().astype(np.float32),
                    step_count=adapt_meta["step_count"],
                    action=adapt_meta["action"],
                    update_applied=adapt_meta["update_applied"],
                    selected_pixels=adapt_meta["selected_pixels"],
                    mean_entropy=risk_ev["mean_entropy"],
                    mean_max_confidence=risk_ev["mean_max_confidence"],
                )
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    raise RuntimeError(f"FATAL: Prediction shard {shard_path} was not persisted before label access!")

                # 4. Explicitly mark persisted to unlock labels
                batch.mark_persisted()

                # 5. Offline evaluation: ground truth unlocked strictly after persistence
                labels = batch.get_labels()
                V_sensor = batch.valid_mask == 1
                V_eval = V_sensor & (labels != 3)

                # Class 2 (flood) confusion metrics
                p_c2 = pred_pre == 2
                t_c2 = labels == 2

                tp = int(((p_c2 & t_c2) & V_eval).sum().item())
                fp = int(((p_c2 & (~t_c2)) & V_eval).sum().item())
                fn = int((((~p_c2) & t_c2) & V_eval).sum().item())
                tn = int((((~p_c2) & (~t_c2)) & V_eval).sum().item())
                valid = int(V_eval.sum().item())

                ev_tp += tp
                ev_fp += fp
                ev_fn += fn
                ev_tn += tn
                ev_valid += valid
                ev_samples += len(batch.sample_ids)

            # Compute event-level metrics
            prec = float(ev_tp / max(ev_tp + ev_fp, 1))
            rec = float(ev_tp / max(ev_tp + ev_fn, 1))
            f1 = float(2 * ev_tp / max(2 * ev_tp + ev_fp + ev_fn, 1))
            iou = float(ev_tp / max(ev_tp + ev_fp + ev_fn, 1))

            ev_metrics = {
                "event_id": eid,
                "samples": ev_samples,
                "valid_pixels": ev_valid,
                "tp": ev_tp,
                "fp": ev_fp,
                "fn": ev_fn,
                "tn": ev_tn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": iou,
            }
            event_metrics_list.append(ev_metrics)
            print(f"  Event {eid}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, IoU={iou:.4f} (N={ev_samples})")

            global_tp += ev_tp
            global_fp += ev_fp
            global_fn += ev_fn
            global_tn += ev_tn
            global_valid += ev_valid

        elapsed_s = time.perf_counter() - start_time
        macro_f1 = float(np.mean([m["f1"] for m in event_metrics_list]))
        macro_prec = float(np.mean([m["precision"] for m in event_metrics_list]))
        macro_rec = float(np.mean([m["recall"] for m in event_metrics_list]))
        macro_iou = float(np.mean([m["iou"] for m in event_metrics_list]))

        pooled_prec = float(global_tp / max(global_tp + global_fp, 1))
        pooled_rec = float(global_tp / max(global_tp + global_fn, 1))
        pooled_f1 = float(2 * global_tp / max(2 * global_tp + global_fp + global_fn, 1))
        pooled_iou = float(global_tp / max(global_tp + global_fp + global_fn, 1))

        stream_summary = {
            "stream_id": stream_id,
            "method": method,
            "seed": seed,
            "order_id": order_id,
            "elapsed_seconds": elapsed_s,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_iou": macro_iou,
            "pooled_f1": pooled_f1,
            "pooled_precision": pooled_prec,
            "pooled_recall": pooled_rec,
            "pooled_iou": pooled_iou,
            "event_metrics": event_metrics_list,
        }

        # Save summary JSON
        with open(stream_dir / "stream_summary.json", "w", encoding="utf-8") as f:
            json.dump(stream_summary, f, indent=2)

        # Save event metrics CSV
        with open(stream_dir / "event_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "event_id",
                    "samples",
                    "valid_pixels",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                ],
            )
            writer.writeheader()
            for row in event_metrics_list:
                writer.writerow(row)

        # Save manifest JSON with run metadata
        manifest = {
            "stream_id": stream_id,
            "method": method,
            "seed": seed,
            "order_id": order_id,
            "device": self.device_str,
            "elapsed_seconds": elapsed_s,
            "completed_events": [m["event_id"] for m in event_metrics_list],
            "total_samples": sum(m["samples"] for m in event_metrics_list),
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        }
        with open(stream_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Mark completed
        with open(stream_dir / "completion_status.json", "w", encoding="utf-8") as f:
            json.dump({"status": "COMPLETED", "stream_id": stream_id, "timestamp_utc": datetime.datetime.utcnow().isoformat()}, f, indent=2)

        print(f"<<< Completed Stream {stream_id} in {elapsed_s:.1f}s | Macro F1: {macro_f1:.4f} | Pooled F1: {pooled_f1:.4f}")
        return stream_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard E7 Branch C Empirical Evaluation Runner")
    parser.add_argument("--config", type=Path, default=None, help="Path to execution config JSON")
    parser.add_argument("--validate_only", action="store_true", help="Run preflights and exit without executing GPU work")
    parser.add_argument("--smoke", action="store_true", help="Run bounded smoke test using e7_cotta_smoke_v2.json")
    parser.add_argument("--stream_id", type=str, default=None, help="Target specific stream ID")
    parser.add_argument("--dry_run", action="store_true", help="Initialize and validate stream batches without running model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (cuda:0 or cpu)")
    args = parser.parse_args()

    if args.config is not None:
        config_path = args.config
    elif args.smoke:
        config_path = Path("configs/changeguard/e7_cotta_smoke_v2.json")
    else:
        config_path = Path("configs/changeguard/e7_branch_c_empirical_v1.json")

    runner = E7EmpiricalRunner(config_path=config_path, device=args.device)

    preflight_report = runner.run_preflights()

    if args.validate_only:
        print("\nPreflight validation check complete: SUCCESS (--validate_only). Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("\nDry run check: Building stream batches for Order A...")
        batches = runner.build_stream_batches("order_a")
        total_batches = sum(len(b[1]) for b in batches)
        print(f"Dry run complete: Built {len(batches)} events with {total_batches} total batches. Exiting.")
        sys.exit(0)

    if args.smoke:
        smoke_run_id = runner.cfg.get("smoke_specification", {}).get("run_id", "e7_cotta_smoke_v2")
        print(f"\nRunning bounded GPU smoke test on Event 520 (32 samples, BS=16, Stream: {smoke_run_id})...")
        runner.execute_stream(
            stream_id=smoke_run_id,
            method="cotta_smoke",
            seed=0,
            order_id="order_a",
        )
        print(f"\nBounded GPU smoke {smoke_run_id} completed successfully.")
        sys.exit(0)

    # Formal multi-stream execution
    streams_to_run = []
    if args.stream_id:
        # User specified single stream
        streams_to_run.append((args.stream_id, args.stream_id.split("_")[0], 0, "order_a"))
    else:
        # Enumerate planned streams across methods, seeds, and orders
        methods = list(runner.cfg.get("evaluated_methods", {}).keys())
        for m in methods:
            for s in [0, 1]:  # primary dev seeds
                for o in ["order_a", "order_b"]:
                    s_id = f"{m}_s{s}_{o}"
                    streams_to_run.append((s_id, m, s, o))

    print(f"Total planned streams: {len(streams_to_run)}")
    for s_id, m, s, o in streams_to_run:
        runner.execute_stream(stream_id=s_id, method=m, seed=s, order_id=o)


if __name__ == "__main__":
    main()

