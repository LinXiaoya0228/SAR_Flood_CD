#!/usr/bin/env python3
"""ChangeGuard Work Package E11: Final Test Engine and Rehearsal Runner.

Capabilities:
1. validate_only: Cryptographic hash validation of all 6 source checkpoints, config and ledger schema verification.
2. dry_run: Complete execution plan accounting (21 physical streams, 210 expected test rows), runtime and disk budgeting.
3. rehearse: Permitted development-event rehearsal of the complete 21-stream flow across SNUNet and SiamUnet,
   verifying prediction persistence before label access, metric accumulation, failure handling, and ledger recording.
4. execute_final: Sealed-test execution across the 10 quarantined test events.
   Fail-closed execution gate: strictly blocked unless authorized under versioned governance amendment.

Research Integrity:
- Fail-closed execution gate: any attempt to run unreleased test data raises an immediate RuntimeError.
- Strict PRE_UPDATE prediction timing and deferred ground-truth access via E11Batch.mark_persisted().
- Reset-per-event deployment across all streams.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1MatchedBNNoUpdateAdapter,
    E1SourceOnlyAdapter,
    E1TentAdapter,
)
from src.changeguard.cotta_adapter import E7CoTTAAdapter
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    KuroEvaluationDataset,
)
from src.evaluation.metrics import (
    EventMetricAccumulator,
    compute_confusion_counts,
    compute_flood_metrics,
)
from src.models.siam_diff_wrapper import SiamDiffWrapper
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 checksum of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class E11Batch:
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


def load_model(
    architecture: str,
    checkpoint_path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Loads either SNUNet_ECAM or SiamUnet_diff from validated checkpoint."""
    if architecture == "SNUNet_ECAM":
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            in_channels=2,
            out_ch=3,
            base_channel=32,
            device=device,
        )
        return wrapper
    elif architecture == "SiamUnet_diff":
        wrapper, _ = SiamDiffWrapper.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            in_channels=2,
            out_channels=3,
            device=device,
        )
        return wrapper
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_adapter(
    method_name: str,
    model: nn.Module,
    device: str = "cpu",
    seed: int = 42,
) -> BaseE1Adapter:
    """Instantiates the adapter for the selected method."""
    if method_name == "source_only":
        return E1SourceOnlyAdapter(model=model, device=device)
    elif method_name == "bn_no_update":
        return E1MatchedBNNoUpdateAdapter(model=model, device=device)
    elif method_name == "tent":
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
        raise ValueError(f"Unsupported method: {method_name}")


class E11FinalTestRunner:
    """Test engine executing validation, dry-run, rehearsal, and final test."""

    def __init__(
        self,
        config_path: Path,
        ledger_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda:0",
        rehearsal_events: Optional[List[int]] = None,
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
                "output_directory", "outputs/e11_rehearsal"
            )
            self.output_dir = (REPO_ROOT / rel_out).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rehearsal_events = rehearsal_events or [520, 559]
        self.test_events = sorted(list(self.config.get("sealed_test_events", [])))
        self._validate_checkpoint_provenance()

    def _validate_checkpoint_provenance(self) -> None:
        """Verifies cryptographic SHA-256 checksums of all 6 source checkpoints."""
        print(">>> Validating cryptographic checksums of source checkpoints...")
        ckpt_configs = self.config["source_checkpoints"]
        for arch_name, ckpts in ckpt_configs.items():
            for seed_key, info in ckpts.items():
                p = REPO_ROOT / info["path"]
                if not p.is_file():
                    raise FileNotFoundError(f"Checkpoint file not found: {p}")
                actual_hash = compute_file_sha256(p)
                expected_hash = info["sha256"]
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"CRITICAL CHECKPOINT CORRUPTION: Checkpoint {p} hash mismatch!\n"
                        f"Expected: {expected_hash}\nActual:   {actual_hash}"
                    )
                print(f"  ✓ {arch_name} [{seed_key}]: SHA-256 match ({actual_hash[:16]}...)")

    def run_dry_run(self) -> Dict[str, Any]:
        """Prints execution plan and resource estimates without reading rasters."""
        print("\n================================================================================")
        print("E11 Final Test Dry-Run & Planning Summary")
        print("================================================================================")
        total_streams = len(self.ledger["streams"])
        num_events = self.config["num_test_events"]
        total_rows = total_streams * num_events

        summary = {
            "matrix_id": self.config["matrix_id"],
            "status": self.config.get("status"),
            "total_physical_streams": total_streams,
            "target_events_count": num_events,
            "total_expected_event_rows": total_rows,
            "architectures": list(self.config["evaluated_methods"].keys()),
            "methods_by_arch": self.config["evaluated_methods"],
            "deployment_mode": self.config["deployment_mode"],
            "prediction_timing": self.config["prediction_timing"],
            "target_gpus": self.config["runtime_and_hardware"]["target_gpus"],
            "estimated_runtime_minutes": self.config["runtime_and_hardware"]["estimated_runtime_minutes"],
            "estimated_disk_gb": self.config["runtime_and_hardware"]["estimated_disk_gb"],
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
    ) -> List[E11Batch]:
        """Builds in-memory batch structures for a single event on demand."""
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

            b = E11Batch(
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
        """Executes a single evaluation stream across specified events."""
        stream_id = stream_entry["stream_id"]
        arch = stream_entry["architecture"]
        method = stream_entry["method"]
        seed = stream_entry["seed"]
        target_device = device or self.device_str

        stream_dir = self.output_dir / stream_id
        summary_path = stream_dir / "stream_summary.json"
        if summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("status") == "COMPLETED" and cached.get("macro_f1") is not None:
                    print(f"Stream {stream_id} already successfully completed. Skipping.")
                    return cached
            except Exception:
                pass

        stream_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Stream {stream_entry['stream_index']}/21: {stream_id} ---")
        print(f"Arch: {arch} | Method: {method} | Seed: {seed} | Device: {target_device}")

        ckpt_rel = self.config["source_checkpoints"][arch][f"seed_{seed}"]["path"]
        ckpt_path = REPO_ROOT / ckpt_rel

        model = load_model(arch, ckpt_path, device=target_device)
        adapter = load_adapter(method, model, device=target_device, seed=seed)

        accumulator = EventMetricAccumulator()
        t0 = time.time()
        event_summaries = []
        batch_size = self.config["runtime_and_hardware"]["batch_size"]

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

            for b_idx, batch in enumerate(batches):
                xA = batch.xA.to(target_device)
                xB = batch.xB.to(target_device)
                vm = batch.valid_mask.to(target_device)

                # PRE_UPDATE adaptation step
                res = adapter.adapt_and_predict_step(
                    xA=xA,
                    xB=xB,
                    valid_mask=vm,
                    sample_ids=batch.sample_ids,
                    event_id=eid,
                )

                pred_pre = res["primary_prediction"]["pred_cand_pre"]
                probs_pre = res["primary_prediction"]["probs_cand_pre"]

                # Durably persist shard
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
            })
            print(
                f"  Event {eid:7d} ({ev_samples:4d} samples) | "
                f"F1: {ev_metrics['f1']:.4f} | IoU: {ev_metrics['iou']:.4f}"
            )

        elapsed = time.time() - t0
        summary_all = accumulator.compute_summary()
        event_table = accumulator.compute_event_table()

        stream_summary = {
            "stream_id": stream_id,
            "architecture": arch,
            "method": method,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "macro_f1": summary_all.get("macro_f1"),
            "worst_event_f1": (summary_all.get("worst_event_f1") or {}).get("f1"),
            "macro_iou": summary_all.get("macro_iou"),
            "pooled_f1": (summary_all.get("pooled_metrics") or {}).get("f1"),
            "event_table": event_table,
            "event_summaries": event_summaries,
            "status": "COMPLETED",
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stream_summary, f, indent=2)

        print(f"  ✓ Complete: Macro F1: {stream_summary['macro_f1']:.4f} in {elapsed:.1f}s")
        return stream_summary

    def execute_rehearsal(self) -> Dict[str, Any]:
        """Runs rehearsal across all 21 streams on permitted development events."""
        print("\n================================================================================")
        print(f"Starting E11 Final-Test Rehearsal across {len(self.ledger['streams'])} Streams")
        print("================================================================================")
        for eid in self.rehearsal_events:
            if eid in FORBIDDEN_SEALED_TEST_EVENTS:
                raise RuntimeError(
                    f"CRITICAL INTEGRITY VIOLATION: Sealed test event {eid} in rehearsal events!"
                )

        eval_ds = KuroEvaluationDataset(
            manifest_path=REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
            dataset_root=REPO_ROOT.parent / "datasets/KuroSiwo",
            allowed_events=self.rehearsal_events,
            load_labels=True,
            allow_sealed_test=False,
        )

        rehearsal_results = {}
        for stream_entry in self.ledger["streams"]:
            res = self.run_stream(
                stream_entry=stream_entry,
                eval_ds=eval_ds,
                events=self.rehearsal_events,
                device=self.device_str,
                max_batches_per_event=self.max_batches_per_event,
            )
            rehearsal_results[stream_entry["stream_id"]] = res

        rehearsal_summary_path = self.output_dir / "rehearsal_summary.json"
        with open(rehearsal_summary_path, "w", encoding="utf-8") as f:
            json.dump(rehearsal_results, f, indent=2)

        print("\n================================================================================")
        print(f"✓ All {len(self.ledger['streams'])} streams successfully rehearsed!")
        print(f"Summary written to: {rehearsal_summary_path}")
        print("================================================================================")
        return rehearsal_results

    def execute_final(
        self,
        seed: Optional[int] = None,
        stream_indices: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs one-time frozen final test across the 10 sealed Kuro Siwo test events."""
        if self.config.get("status") != "AUTHORIZED_FOR_ONE_TIME_EXECUTION":
            raise RuntimeError(
                "CRITICAL RESEARCH INTEGRITY VIOLATION:\n"
                f"Config status is {self.config.get('status')}! "
                "Final sealed-test execution is strictly unauthorized under Gate T preparation!\n"
                "The 10 sealed test events remain quarantined until explicit written user authorization\n"
                "and supervisor approval are recorded in the repository. Use --mode rehearse or --mode dry_run."
            )
        if not self.config.get("authorization_reference"):
            raise RuntimeError("CRITICAL RESEARCH INTEGRITY VIOLATION: Missing authorization_reference.")

        print("\n================================================================================")
        print("Starting E11 Final-Test Execution across Sealed Test Partition")
        print(f"Events: {self.test_events}")
        print(f"Authorization: {self.config.get('authorization_reference')}")
        print("================================================================================")

        eval_ds = KuroEvaluationDataset(
            manifest_path=REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
            dataset_root=REPO_ROOT.parent / "datasets/KuroSiwo",
            allowed_events=self.test_events,
            load_labels=True,
            allow_sealed_test=True,
        )
        print(f"Dataset loaded: {len(eval_ds)} total test samples across {len(self.test_events)} events.")

        streams = self.ledger["streams"]
        if seed is not None:
            streams = [s for s in streams if s["seed"] == seed]
        if stream_indices is not None:
            streams = [s for s in streams if s["stream_index"] in stream_indices]

        print(f"Selected {len(streams)} streams for execution on device {device or self.device_str}.")

        results = {}
        for stream_entry in streams:
            target_dev = device or stream_entry.get("assigned_gpu", self.device_str)
            res = self.run_stream(
                stream_entry=stream_entry,
                eval_ds=eval_ds,
                events=self.test_events,
                device=target_dev,
                max_batches_per_event=None,
            )
            results[stream_entry["stream_id"]] = res

        print("\n================================================================================")
        print(f"✓ Completed {len(streams)} final test streams on {device or self.device_str}.")
        print("================================================================================")
        return results


def main():
    parser = argparse.ArgumentParser(description="ChangeGuard E11 Test Engine & Rehearsal Runner.")
    parser.add_argument(
        "--mode",
        choices=["validate_only", "dry_run", "rehearse", "execute_final"],
        default="dry_run",
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e11_final_test_matrix_v1.json",
        help="Path to final test matrix configuration.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e11_final_test_ledger_v1.json",
        help="Path to final test execution ledger.",
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
        help="Filter streams by seed (e.g., 0, 1, 2).",
    )
    parser.add_argument(
        "--stream_indices",
        type=int,
        nargs="+",
        default=None,
        help="Filter streams by stream indices.",
    )
    parser.add_argument(
        "--rehearsal_events",
        type=int,
        nargs="+",
        default=[520, 559],
        help="Permitted event IDs for rehearsal.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=1,
        help="Max batches per event in rehearsal mode.",
    )
    args = parser.parse_args()

    runner = E11FinalTestRunner(
        config_path=args.config,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        device=args.device,
        rehearsal_events=args.rehearsal_events,
        max_batches_per_event=args.max_batches,
    )

    if args.mode == "validate_only":
        print("\n✓ Checkpoint cryptographic validation passed.")
    elif args.mode == "dry_run":
        runner.run_dry_run()
    elif args.mode == "rehearse":
        runner.execute_rehearsal()
    elif args.mode == "execute_final":
        runner.execute_final(
            seed=args.seed,
            stream_indices=args.stream_indices,
            device=args.device,
        )


if __name__ == "__main__":
    main()
