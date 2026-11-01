#!/usr/bin/env python3
"""ChangeGuard Work Package E10: Unified Evaluation Engine for Robustness Extensions.

Supports:
1. dev_eval_second_backbone:
   - Evaluates source-trained SiamUnet_diff checkpoints across 6 development events
     ([279, 437, 520, 559, 1111003, 1111008], 2826 samples) for source_only, bn_no_update, tent.
   - Evaluates 3 random seeds (0, 1, 2) in reset_per_event mode under PRE_UPDATE timing.
   - Post-persistence deferred ground-truth disk read.
2. synthetic_shift:
   - Evaluates multiplicative backscatter calibration shifts s in {0.0, 0.05, 0.10, 0.20}
     on SNUNet and SiamUnet_diff under identical PRE_UPDATE and deferred label access.

Research Integrity:
- Zero access to sealed-test events ([205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]).
- Fail-closed isolation assertions.
- Double-masking validity: (MK0_MNA == 1) & (MK0_MLU != 3).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1MatchedBNNoUpdateAdapter,
    E1SourceOnlyAdapter,
    E1TentAdapter,
)
from src.changeguard.synthetic_shift import MultiplicativeBackscatterShift, SUPPORTED_SEVERITIES
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    TOTAL_EXPECTED_TTA_DEV_SAMPLES,
    KuroEvaluationDataset,
)
from src.evaluation.metrics import (
    EventMetricAccumulator,
    compute_confusion_counts,
    compute_flood_metrics,
    compute_macro_metrics,
    compute_pooled_metrics,
)
from src.models.siam_diff_wrapper import SiamDiffWrapper
from src.models.snunet_wrapper import SNUNetWrapper


class E10Batch:
    """Encapsulates a batch with deferred ground-truth access."""

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
                "CRITICAL RESEARCH INTEGRITY VIOLATION: Labels accessed before predictions persisted!"
            )
        # Read labels from dataset
        labels = []
        for idx in self._sample_indices:
            raw_item = self._dataset[idx]
            labels.append(raw_item["label"])
        return torch.stack(labels, dim=0)


def load_model(
    architecture: str,
    checkpoint_path: Path,
    device: str = "cuda:0",
) -> nn.Module:
    """Loads either SNUNet_ECAM or SiamUnet_diff from checkpoint."""
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
    device: str = "cuda:0",
) -> BaseE1Adapter:
    """Instantiates adapter for the specified method."""
    if method_name == "source_only":
        return E1SourceOnlyAdapter(model=model, device=device)
    elif method_name == "bn_no_update":
        return E1MatchedBNNoUpdateAdapter(model=model, device=device)
    elif method_name == "tent":
        return E1TentAdapter(model=model, learning_rate=1e-4, device=device)
    else:
        raise ValueError(f"Unsupported E10 method: {method_name}")


class E10EvaluationRunner:
    """Runner for E10 second backbone and synthetic shift evaluations."""

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda:1",
        batch_size: int = 16,
    ):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_str = device
        self.batch_size = batch_size

        # Verify zero overlap with sealed test
        for eid in DEFAULT_TTA_DEV_EVENTS:
            if eid in FORBIDDEN_SEALED_TEST_EVENTS:
                raise RuntimeError("CRITICAL RESEARCH INTEGRITY VIOLATION: Sealed test event in dev events!")

        # Initialize base evaluation dataset
        self.eval_dataset = KuroEvaluationDataset(
            manifest_path=REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
            dataset_root=REPO_ROOT.parent / "datasets/KuroSiwo",
            allowed_events=DEFAULT_TTA_DEV_EVENTS,
            load_labels=True,
        )

        # Build batches per event
        self.event_batches: Dict[int, List[E10Batch]] = {}
        self._build_event_batches()

    def _build_event_batches(self) -> None:
        samples_by_event: Dict[int, List[int]] = {eid: [] for eid in DEFAULT_TTA_DEV_EVENTS}
        for idx in range(len(self.eval_dataset)):
            eid = int(self.eval_dataset.samples[idx]["event_id"])
            if eid in samples_by_event:
                samples_by_event[eid].append(idx)

        for eid, indices in samples_by_event.items():
            expected = EXPECTED_TTA_DEV_COUNTS[eid]
            if len(indices) != expected:
                raise RuntimeError(f"Event {eid} sample count mismatch: expected {expected}, got {len(indices)}")

            batches = []
            for b_start in range(0, len(indices), self.batch_size):
                b_indices = indices[b_start : b_start + self.batch_size]
                s_ids = [self.eval_dataset.samples[i]["sample_id"] for i in b_indices]
                xA_list = []
                xB_list = []
                vm_list = []
                for i in b_indices:
                    item = self.eval_dataset[i]
                    xA_list.append(item["xA"])
                    xB_list.append(item["xB"])
                    vm_list.append(item["valid_mask"])

                b = E10Batch(
                    sample_ids=s_ids,
                    xA=torch.stack(xA_list, dim=0),
                    xB=torch.stack(xB_list, dim=0),
                    valid_mask=torch.stack(vm_list, dim=0),
                    dataset=self.eval_dataset,
                    sample_indices=b_indices,
                )
                batches.append(b)
            self.event_batches[eid] = batches

    def run_stream(
        self,
        stream_id: str,
        architecture: str,
        checkpoint_path: Path,
        method: str,
        seed: int,
        severity_factor: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs a single evaluation stream across the 6 dev events."""
        stream_dir = self.output_dir / stream_id
        summary_path = stream_dir / "stream_summary.json"
        if summary_path.exists():
            print(f"Stream {stream_id} already exists. Skipping.")
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)

        stream_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n================================================================================")
        print(f"Starting Stream: {stream_id}")
        print(f"Arch: {architecture} | Method: {method} | Seed: {seed} | Severity: {severity_factor}")
        print(f"================================================================================")

        model = load_model(architecture, checkpoint_path, device=self.device_str)
        adapter = load_adapter(method, model, device=self.device_str)
        shifter = MultiplicativeBackscatterShift(severity_factor=severity_factor)

        accumulator = EventMetricAccumulator()
        t0 = time.time()
        event_summaries = []

        for event_idx, eid in enumerate(DEFAULT_TTA_DEV_EVENTS):
            # Reset adapter on new event (reset_per_event mode)
            if event_idx > 0:
                adapter.reset_to_source()

            batches = self.event_batches[eid]
            shards_dir = stream_dir / f"event_{eid}" / "shards"
            shards_dir.mkdir(parents=True, exist_ok=True)

            ev_tp = 0
            ev_fp = 0
            ev_fn = 0
            ev_tn = 0
            ev_valid = 0
            ev_samples = 0

            for b_idx, batch in enumerate(batches):
                # Apply synthetic shift if non-zero
                xA = shifter.transform_normalized_tensor(batch.xA).to(self.device_str)
                xB = shifter.transform_normalized_tensor(batch.xB).to(self.device_str)
                vm = batch.valid_mask.to(self.device_str)

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

                # Durably persist prediction shard before accessing labels
                shard_path = shards_dir / f"batch_{b_idx:04d}.npz"
                np.savez_compressed(
                    shard_path,
                    sample_ids=np.array(batch.sample_ids),
                    pred_cand_pre=pred_pre.detach().cpu().numpy().astype(np.uint8),
                    probs_cand_pre=probs_pre.detach().cpu().numpy().astype(np.float32),
                )
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    raise RuntimeError(f"FATAL: Shard {shard_path} was not persisted before label access!")

                # Mark persisted to unlock labels
                batch.mark_persisted()

                # Read ground truth
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

                # Batch level stats
                for i in range(len(batch.sample_ids)):
                    c = compute_confusion_counts(preds_np[i], labels_np[i], vm_np[i])
                    ev_tp += c["tp"]
                    ev_fp += c["fp"]
                    ev_fn += c["fn"]
                    ev_tn += c["tn"]
                    ev_valid += c["valid_pixels"]
                    ev_samples += 1

            ev_counts = {"tp": ev_tp, "fp": ev_fp, "fn": ev_fn, "tn": ev_tn, "valid_pixels": ev_valid}
            ev_metrics = compute_flood_metrics(ev_counts)
            ev_summary = {
                "event_id": eid,
                "samples": ev_samples,
                "confusion": ev_counts,
                "flood_precision": ev_metrics["precision"],
                "flood_recall": ev_metrics["recall"],
                "flood_f1": ev_metrics["f1"],
                "flood_iou": ev_metrics["iou"],
            }
            event_summaries.append(ev_summary)
            print(
                f"  Event {eid:7d} ({ev_samples:4d} samples) | "
                f"F1: {ev_metrics['f1']:.4f} | Recall: {ev_metrics['recall']:.4f} | "
                f"IoU: {ev_metrics['iou']:.4f}"
            )

        elapsed = time.time() - t0
        summary_all = accumulator.compute_summary()
        event_table = accumulator.compute_event_table()

        stream_summary = {
            "stream_id": stream_id,
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path),
            "method": method,
            "seed": seed,
            "severity_factor": severity_factor,
            "elapsed_seconds": elapsed,
            "macro_f1": summary_all.get("macro_f1"),
            "worst_event_f1": (summary_all.get("worst_event_f1") or {}).get("f1"),
            "macro_iou": summary_all.get("macro_iou"),
            "macro_recall": summary_all.get("macro_recall"),
            "pooled_f1": (summary_all.get("pooled_metrics") or {}).get("f1"),
            "event_table": event_table,
            "event_summaries": event_summaries,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stream_summary, f, indent=2)

        print(
            f">>> Stream Complete: {stream_id} ({elapsed:.1f}s) | "
            f"Macro F1: {stream_summary['macro_f1']:.4f} | "
            f"Worst F1: {stream_summary['worst_event_f1']:.4f} | "
            f"Macro IoU: {stream_summary['macro_iou']:.4f}"
        )
        return stream_summary


def main():
    parser = argparse.ArgumentParser(description="E10 evaluation runner.")
    parser.add_argument(
        "--mode",
        choices=["dev_eval_second_backbone", "synthetic_shift", "all"],
        default="all",
        help="Evaluation mode.",
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="CUDA device.")
    parser.add_argument(
        "--arch",
        choices=["SNUNet_ECAM", "SiamUnet_diff", "all"],
        default="all",
        help="Filter by architecture for synthetic_shift.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Filter by seed(s).",
    )
    parser.add_argument(
        "--severities",
        type=str,
        nargs="+",
        default=None,
        help="Filter by severity name(s).",
    )
    args = parser.parse_args()

    # Load matrix configs
    dev_cfg_path = REPO_ROOT / "configs/changeguard/e10_second_backbone_dev_matrix.json"
    with open(dev_cfg_path, "r", encoding="utf-8") as f:
        dev_cfg = json.load(f)

    shift_cfg_path = REPO_ROOT / "configs/changeguard/e10_synthetic_shift_matrix.json"
    with open(shift_cfg_path, "r", encoding="utf-8") as f:
        shift_cfg = json.load(f)

    if args.mode in ("dev_eval_second_backbone", "all"):
        print("\n================================================================================")
        print("Starting E10 Phase 4: Second Backbone Development-Event Evaluation")
        print("================================================================================")
        dev_out = REPO_ROOT / dev_cfg["output_directory"]
        dev_runner = E10EvaluationRunner(output_dir=dev_out, device=args.device)

        methods = dev_cfg["methods"]
        seeds = dev_cfg["seeds"] if args.seeds is None else args.seeds
        ckpt_map = dev_cfg["checkpoints"]

        for seed in seeds:
            ckpt_path = REPO_ROOT / ckpt_map[f"seed_{seed}"]
            for method in methods:
                stream_id = f"siam_diff_{method}_seed{seed}"
                dev_runner.run_stream(
                    stream_id=stream_id,
                    architecture="SiamUnet_diff",
                    checkpoint_path=ckpt_path,
                    method=method,
                    seed=seed,
                    severity_factor=0.0,
                )

    if args.mode in ("synthetic_shift", "all"):
        print("\n================================================================================")
        print("Starting E10 Phase 6: Synthetic Domain-Shift Stress Test")
        print("================================================================================")
        shift_out = REPO_ROOT / shift_cfg["output_directory"]
        shift_runner = E10EvaluationRunner(output_dir=shift_out, device=args.device)

        severities = shift_cfg["severities"]
        if args.severities is not None:
            severities = {k: v for k, v in severities.items() if k in args.severities}

        methods = shift_cfg["methods"]
        seeds = shift_cfg["seeds"] if args.seeds is None else args.seeds
        arch_cfg = shift_cfg["architectures"]
        if args.arch != "all":
            arch_cfg = {k: v for k, v in arch_cfg.items() if k == args.arch}

        for arch_name, a_info in arch_cfg.items():
            ckpts = a_info["checkpoints"]
            prefix = "snunet" if arch_name == "SNUNet_ECAM" else "siam_diff"

            for sev_name, sev_factor in severities.items():
                for seed in seeds:
                    ckpt_path = REPO_ROOT / ckpts[f"seed_{seed}"]
                    for method in methods:
                        stream_id = f"shift_{prefix}_{method}_{sev_name}_seed{seed}"
                        shift_runner.run_stream(
                            stream_id=stream_id,
                            architecture=arch_name,
                            checkpoint_path=ckpt_path,
                            method=method,
                            seed=seed,
                            severity_factor=sev_factor,
                        )


if __name__ == "__main__":
    main()


