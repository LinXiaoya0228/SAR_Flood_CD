#!/usr/bin/env python3
"""ChangeGuard Supplementary Evaluation Runner: R2 + Tent / CoTTA.

Executes test-time adaptation (Tent, CoTTA) initialized from frozen all-source R2 checkpoints
across the 10 historical Kuro Siwo target events.

Integrity & Scientific Invariants:
1. Cryptographic SHA-256 verification of R2 checkpoints before execution.
2. PRE_UPDATE prediction convention: predictions recorded before updating parameters on each batch.
3. Physical label isolation: predictions durably persisted to .npz before ground-truth labels are read.
4. CoTTA initialization: student, teacher, source anchor, and restoration targets all initialized
   from the SAME R2 checkpoint. Non-aliasing strictly verified.
5. Reset-per-event contract: complete reset to source checkpoint at every event boundary.
6. Exact matching: identical 10 events, sample sequence, batch size (16), and sensory-valid masks.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1TentAdapter,
)
from src.changeguard.cotta_adapter import E7CoTTAAdapter
from src.data.evaluation_dataset import KuroEvaluationDataset
from src.evaluation.metrics import (
    EventMetricAccumulator,
    compute_confusion_counts,
    compute_flood_metrics,
)
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 checksum of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class R2TTABatch:
    """Encapsulates sensory data and enforces deferred ground-truth access."""

    def __init__(
        self,
        sample_ids: List[str],
        xA: torch.Tensor,
        xB: torch.Tensor,
        valid_mask: torch.Tensor,
        dataset: KuroEvaluationDataset,
        sample_indices: List[int],
    ):
        self.sample_ids = sample_ids
        self.xA = xA
        self.xB = xB
        self.valid_mask = valid_mask
        self._dataset = dataset
        self._sample_indices = sample_indices
        self._is_persisted = False

    def mark_persisted(self) -> None:
        self._is_persisted = True

    def get_labels(self) -> torch.Tensor:
        if not self._is_persisted:
            raise PermissionError(
                "CRITICAL RESEARCH INTEGRITY VIOLATION: Labels accessed before primary predictions were persisted!"
            )
        labels = []
        for idx in self._sample_indices:
            raw_item = self._dataset[idx]
            labels.append(raw_item["label"])
        return torch.stack(labels, dim=0)


def load_r2_model(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    """Loads SNUNet_ECAM from validated R2 checkpoint."""
    wrapper, _ = SNUNetWrapper.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        in_channels=2,
        out_ch=3,
        base_channel=32,
        device=device,
    )
    return wrapper


def load_r2_adapter(
    method_name: str,
    model: nn.Module,
    device: str = "cpu",
    seed: int = 42,
) -> BaseE1Adapter:
    """Instantiates Tent or CoTTA adapter initialized from the R2 model."""
    if method_name == "tent":
        return E1TentAdapter(model=model, learning_rate=1e-4, device=device)
    elif method_name == "cotta":
        return E7CoTTAAdapter(
            model=model,
            learning_rate=1e-4,
            ema_alpha=0.999,
            restoration_prob=0.01,
            confidence_threshold=0.90,
            mode="reset_per_event",
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported adaptation method: {method_name}")


class R2TTAEvaluationRunner:
    """Execution engine for R2 + TTA supplementary evaluation."""

    def __init__(
        self,
        config_path: Path,
        ledger_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda:0",
        max_batches_per_event: Optional[int] = None,
    ):
        self.config_path = config_path.resolve()
        self.ledger_path = ledger_path.resolve()
        self.device_str = device
        self.max_batches_per_event = max_batches_per_event

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

        if output_dir is not None:
            self.output_dir = output_dir.resolve()
        else:
            rel_out = self.config.get("output_destinations", {}).get(
                "output_directory", "outputs/supplementary_r2_tta"
            )
            self.output_dir = (REPO_ROOT / rel_out).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_events = sorted(list(self.config.get("target_events", [])))
        self._validate_checkpoints()

    def _validate_checkpoints(self) -> None:
        """Verifies cryptographic SHA-256 checksums of the R2 source checkpoints."""
        print(">>> Validating cryptographic checksums of R2 source checkpoints...")
        ckpts = self.config["source_checkpoints"]["SNUNet_ECAM"]
        for seed_key, info in ckpts.items():
            p = REPO_ROOT / info["path"]
            if not p.is_file():
                raise FileNotFoundError(f"R2 checkpoint file not found: {p}")
            actual_hash = compute_file_sha256(p)
            expected_hash = info["sha256"]
            if actual_hash != expected_hash:
                raise ValueError(
                    f"CRITICAL R2 CHECKPOINT CORRUPTION: {p} hash mismatch!\n"
                    f"Expected: {expected_hash}\nActual:   {actual_hash}"
                )
            print(f"  ✓ SNUNet_ECAM R2 [{seed_key}]: SHA-256 verified ({actual_hash[:16]}...)")

    def run_dry_run(self) -> Dict[str, Any]:
        """Prints execution plan and accounting summary."""
        print("\n================================================================================")
        print("ChangeGuard R2 + TTA Supplementary Evaluation: Dry-Run Summary")
        print("================================================================================")
        total_streams = len(self.ledger["streams"])
        num_events = len(self.target_events)
        total_rows = total_streams * num_events

        summary = {
            "matrix_id": self.config["matrix_id"],
            "status": self.config.get("status"),
            "total_physical_streams": total_streams,
            "target_events": self.target_events,
            "num_target_events": num_events,
            "total_expected_event_rows": total_rows,
            "methods": self.config["evaluated_methods"]["SNUNet_ECAM"],
            "deployment_mode": self.config["deployment_mode"],
            "prediction_timing": self.config["prediction_timing"],
            "batch_size": self.config["runtime_and_hardware"]["batch_size"],
            "target_gpus": self.config["runtime_and_hardware"]["target_gpus"],
        }
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary

    def _build_event_batches(
        self,
        eval_ds: KuroEvaluationDataset,
        event_id: int,
        batch_size: int,
        max_batches: Optional[int] = None,
    ) -> List[R2TTABatch]:
        """Builds batch structures for an event on demand."""
        indices = [
            i for i, s in enumerate(eval_ds.samples) if int(s["event_id"]) == event_id
        ]
        batches = []
        for b_start in range(0, len(indices), batch_size):
            b_indices = indices[b_start : b_start + batch_size]
            s_ids = [eval_ds.samples[i]["sample_id"] for i in b_indices]
            xA_list = []
            xB_list = []
            vm_list = []
            for i in b_indices:
                item = eval_ds[i]
                xA_list.append(item["xA"])
                xB_list.append(item["xB"])
                vm_list.append(item["valid_mask"])

            b = R2TTABatch(
                sample_ids=s_ids,
                xA=torch.stack(xA_list, dim=0),
                xB=torch.stack(xB_list, dim=0),
                valid_mask=torch.stack(vm_list, dim=0),
                dataset=eval_ds,
                sample_indices=b_indices,
            )
            batches.append(b)
            if max_batches and len(batches) >= max_batches:
                break
        return batches

    def run_stream(
        self,
        stream_entry: Dict[str, Any],
        eval_ds: KuroEvaluationDataset,
        events: List[int],
        device: Optional[str] = None,
        max_batches_per_event: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes a single R2+TTA evaluation stream across specified events."""
        stream_id = stream_entry["stream_id"]
        arch = stream_entry["architecture"]
        method = stream_entry["method"]
        seed = stream_entry["seed"]
        target_device = device or stream_entry.get("assigned_gpu", self.device_str)

        stream_dir = self.output_dir / stream_id
        summary_path = stream_dir / "stream_summary.json"
        if summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("status") == "COMPLETED" and cached.get("macro_f1") is not None:
                    print(f"Stream {stream_id} already completed. Skipping.")
                    return cached
            except Exception:
                pass

        stream_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Stream {stream_entry['stream_index']}/6: {stream_id} ---")
        print(f"Arch: {arch} | Source: R2 | Method: {method} | Seed: {seed} | Device: {target_device}")

        ckpt_path = REPO_ROOT / stream_entry["checkpoint_path"]
        model = load_r2_model(ckpt_path, device=target_device)
        adapter = load_r2_adapter(method, model, device=target_device, seed=seed)

        # For CoTTA: explicitly verify that student, teacher, and source models are disjoint
        # and initialized from R2
        if method == "cotta":
            print("  Verifying CoTTA non-aliasing and R2 initialization...")
            cotta_adapter: E7CoTTAAdapter = adapter  # type: ignore
            cotta_adapter._assert_non_aliasing()
            print("  ✓ CoTTA non-aliasing verified.")

        accumulator = EventMetricAccumulator()
        t0 = time.time()
        if torch.cuda.is_available() and "cuda" in target_device:
            torch.cuda.reset_peak_memory_stats(torch.device(target_device))

        event_summaries = []
        batch_size = self.config["runtime_and_hardware"]["batch_size"]

        total_updates_applied = 0
        total_updates_skipped = 0
        total_updates_failed = 0

        for event_idx, eid in enumerate(events):
            if event_idx > 0:
                adapter.reset_to_source()

            batches = self._build_event_batches(
                eval_ds,
                event_id=eid,
                batch_size=batch_size,
                max_batches=max_batches_per_event,
            )

            shards_dir = stream_dir / f"event_{eid}" / "shards"
            shards_dir.mkdir(parents=True, exist_ok=True)

            ev_tp = 0
            ev_fp = 0
            ev_fn = 0
            ev_tn = 0
            ev_valid = 0
            ev_samples = 0
            ev_updates = 0
            ev_skips = 0
            ev_fails = 0

            for b_idx, batch in enumerate(batches):
                xA = batch.xA.to(target_device)
                xB = batch.xB.to(target_device)
                vm = batch.valid_mask.to(target_device)

                pre_applied = adapter.num_updates_applied
                pre_skipped = adapter.num_updates_skipped
                pre_failed = adapter.num_updates_failed

                # PRE_UPDATE adaptation step
                res = adapter.adapt_and_predict_step(
                    xA=xA,
                    xB=xB,
                    valid_mask=vm,
                    sample_ids=batch.sample_ids,
                    event_id=eid,
                )

                if adapter.num_updates_applied > pre_applied:
                    ev_updates += 1
                if adapter.num_updates_skipped > pre_skipped:
                    ev_skips += 1
                if adapter.num_updates_failed > pre_failed:
                    ev_fails += 1

                pred_pre = res["primary_prediction"]["pred_cand_pre"]
                probs_pre = res["primary_prediction"]["probs_cand_pre"]

                # Durably persist prediction shard BEFORE label access
                shard_path = shards_dir / f"batch_{b_idx:04d}.npz"
                np.savez_compressed(
                    shard_path,
                    sample_ids=np.array(batch.sample_ids),
                    pred_cand_pre=pred_pre.detach().cpu().numpy().astype(np.uint8),
                    probs_cand_pre=probs_pre.detach().cpu().numpy().astype(np.float32),
                )
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"FATAL: Shard {shard_path} was not persisted before label access!"
                    )

                # Unlock labels
                batch.mark_persisted()
                labels = batch.get_labels()

                labels_np = labels.cpu().numpy()
                preds_np = pred_pre.detach().cpu().numpy()
                vm_np = batch.valid_mask.cpu().numpy()

                accumulator.update(
                    event_ids=[eid] * len(batch.sample_ids),
                    pred_classes=preds_np,
                    labels=labels_np,
                    valid_masks=vm_np,
                )

                for i in range(len(batch.sample_ids)):
                    c = compute_confusion_counts(preds_np[i], labels_np[i], vm_np[i])
                    ev_tp += c["tp"]
                    ev_fp += c["fp"]
                    ev_fn += c["fn"]
                    ev_tn += c["tn"]
                    ev_valid += c["valid_pixels"]
                    ev_samples += 1

            total_updates_applied += ev_updates
            total_updates_skipped += ev_skips
            total_updates_failed += ev_fails

            ev_counts = {
                "tp": ev_tp,
                "fp": ev_fp,
                "fn": ev_fn,
                "tn": ev_tn,
                "valid_pixels": ev_valid,
            }
            ev_metrics = compute_flood_metrics(ev_counts)
            event_summaries.append({
                "event_id": eid,
                "samples": ev_samples,
                "confusion": ev_counts,
                "flood_f1": ev_metrics["f1"],
                "flood_iou": ev_metrics["iou"],
                "flood_recall": ev_metrics["recall"],
                "flood_precision": ev_metrics["precision"],
                "updates_applied": ev_updates,
                "updates_skipped": ev_skips,
                "updates_failed": ev_fails,
            })
            print(
                f"  Event {eid:7d} ({ev_samples:4d} samples) | "
                f"F1: {ev_metrics['f1']:.4f} | IoU: {ev_metrics['iou']:.4f} | "
                f"Updates: {ev_updates} | Skips: {ev_skips} | Fails: {ev_fails}"
            )

        elapsed = time.time() - t0
        peak_vram_mb = 0.0
        if torch.cuda.is_available() and "cuda" in target_device:
            peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(target_device)) / (1024 * 1024)

        summary_all = accumulator.compute_summary()
        event_table = accumulator.compute_event_table()

        stream_summary = {
            "stream_id": stream_id,
            "architecture": arch,
            "source_recipe": "R2",
            "method": method,
            "seed": seed,
            "assigned_gpu": target_device,
            "elapsed_seconds": elapsed,
            "peak_vram_mb": peak_vram_mb,
            "macro_f1": summary_all.get("macro_f1"),
            "worst_event_f1": (summary_all.get("worst_event_f1") or {}).get("f1"),
            "macro_iou": summary_all.get("macro_iou"),
            "pooled_f1": (summary_all.get("pooled_metrics") or {}).get("f1"),
            "total_updates_applied": total_updates_applied,
            "total_updates_skipped": total_updates_skipped,
            "total_updates_failed": total_updates_failed,
            "event_table": event_table,
            "event_summaries": event_summaries,
            "status": "COMPLETED",
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stream_summary, f, indent=2)

        print(
            f"  ✓ Complete: Macro F1: {stream_summary['macro_f1']:.4f} | "
            f"Elapsed: {elapsed:.1f}s | Peak VRAM: {peak_vram_mb:.1f}MB"
        )
        return stream_summary

    def execute_smoke(
        self,
        smoke_event: int = 520,
        max_batches: int = 2,
        device: str = "cuda:0",
    ) -> Dict[str, Any]:
        """Runs infrastructure smoke on declared subset of one event to verify flow and estimate runtime."""
        print("\n================================================================================")
        print("ChangeGuard R2 + TTA: Infrastructure Smoke Run")
        print(f"Smoke Event: {smoke_event} | Max Batches: {max_batches} | Device: {device}")
        print("================================================================================")

        eval_ds = KuroEvaluationDataset(
            manifest_path=REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
            dataset_root=REPO_ROOT.parent / "datasets/KuroSiwo",
            allowed_events=[smoke_event],
            load_labels=True,
            allow_sealed_test=False,
        )

        smoke_out = self.output_dir / "smoke"
        smoke_out.mkdir(parents=True, exist_ok=True)

        smoke_results = {}
        for method in ["tent", "cotta"]:
            stream_entry = {
                "stream_index": 0,
                "stream_id": f"smoke_r2_{method}_s0",
                "architecture": "SNUNet_ECAM",
                "source_recipe": "R2",
                "method": method,
                "seed": 0,
                "checkpoint_path": self.config["source_checkpoints"]["SNUNet_ECAM"]["seed_0"]["path"],
                "assigned_gpu": device,
            }
            # Save original output dir, temporarily point to smoke_out
            orig_out = self.output_dir
            self.output_dir = smoke_out
            try:
                res = self.run_stream(
                    stream_entry=stream_entry,
                    eval_ds=eval_ds,
                    events=[smoke_event],
                    device=device,
                    max_batches_per_event=max_batches,
                )
                smoke_results[method] = res
            finally:
                self.output_dir = orig_out

        print("\n✓ Smoke test successfully completed for both R2+Tent and R2+CoTTA!")
        return smoke_results

    def execute_formal(
        self,
        seed: Optional[int] = None,
        stream_indices: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs formal evaluation across the 10 historical Kuro Siwo target events."""
        print("\n================================================================================")
        print("ChangeGuard R2 + TTA: Formal Six-Stream Evaluation")
        print(f"Target Events: {self.target_events}")
        print("================================================================================")

        eval_ds = KuroEvaluationDataset(
            manifest_path=REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
            dataset_root=REPO_ROOT.parent / "datasets/KuroSiwo",
            allowed_events=self.target_events,
            load_labels=True,
            allow_sealed_test=True,
        )
        print(f"Dataset loaded: {len(eval_ds)} total samples across {len(self.target_events)} events.")

        streams = self.ledger["streams"]
        if seed is not None:
            streams = [s for s in streams if s["seed"] == seed]
        if stream_indices is not None:
            streams = [s for s in streams if s["stream_index"] in stream_indices]

        print(f"Executing {len(streams)} streams on device {device or self.device_str}...")

        results = {}
        for stream_entry in streams:
            target_dev = device or stream_entry.get("assigned_gpu", self.device_str)
            res = self.run_stream(
                stream_entry=stream_entry,
                eval_ds=eval_ds,
                events=self.target_events,
                device=target_dev,
                max_batches_per_event=None,
            )
            results[stream_entry["stream_id"]] = res

        print("\n================================================================================")
        print(f"✓ All {len(streams)} streams finished on {device or self.device_str}!")
        print("================================================================================")
        return results


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard R2 + TTA Evaluation Runner.")
    parser.add_argument(
        "--mode",
        choices=["validate_only", "dry_run", "smoke", "execute_formal"],
        default="dry_run",
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/supplementary_r2_tta/r2_tta_evaluation_matrix_v1.json",
        help="Path to evaluation matrix config.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "configs/supplementary_r2_tta/r2_tta_evaluation_ledger_v1.json",
        help="Path to evaluation ledger.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter streams by seed (0, 1, 2).",
    )
    parser.add_argument(
        "--stream_indices",
        type=int,
        nargs="+",
        default=None,
        help="Filter streams by stream indices.",
    )
    parser.add_argument(
        "--smoke_event",
        type=int,
        default=520,
        help="Event ID for smoke mode.",
    )
    parser.add_argument(
        "--smoke_batches",
        type=int,
        default=2,
        help="Number of batches for smoke mode.",
    )
    args = parser.parse_args()

    runner = R2TTAEvaluationRunner(
        config_path=args.config,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.mode == "validate_only":
        print("\n✓ R2 Checkpoint cryptographic verification passed.")
    elif args.mode == "dry_run":
        runner.run_dry_run()
    elif args.mode == "smoke":
        runner.execute_smoke(
            smoke_event=args.smoke_event,
            max_batches=args.smoke_batches,
            device=args.device,
        )
    elif args.mode == "execute_formal":
        runner.execute_formal(
            seed=args.seed,
            stream_indices=args.stream_indices,
            device=args.device,
        )


if __name__ == "__main__":
    main()

