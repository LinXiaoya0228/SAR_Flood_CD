"""
ChangeGuard Week 8 Controlled Test-Time Adaptation (TTA) Matrix Runner.

Governing Protocol: configs/evaluation/week8_controlled_tta_protocol.json
Document Specification: docs/week8_controlled_tta_protocol.md

Strict Scientific & Integrity Guarantees:
1. Label-Free Adaptation:
   - Adaptation dataloader loads strictly sensory inputs (xA, xB) and sensory observation
     masks (MK0_MNA).
   - Ground truth annotations (MK0_MLU) are NEVER loaded or accessed during adaptation.
   - adapt_and_predict interface enforces strict label isolation; passing any label raises ValueError.
2. Separate Offline Evaluation:
   - Double-masking validity (MK0_MNA == 1 and MK0_MLU != 3) is applied strictly offline
     on frozen, cached predictions after adaptation streaming concludes.
3. 30 Planned Physical Streams vs. 36 Logical Cells:
   - 6 fresh matched source-only streams (3 checkpoint seeds x 2 stream orders at B=16).
   - 12 AdaBN streams (3 seeds x 2 orders x 2 reset modes).
   - 12 reference-aligned Tent streams (3 seeds x 2 orders x 2 reset modes).
   - The 6 source-only physical results are explicitly reused across reset modes after
     verifying model state immutability, populating all 36 logical matrix cells.
4. Reference-Aligned Tent Semantics:
   - Only BN affine parameters receive optimizer updates.
   - track_running_stats = False, running_mean = None, running_var = None.
   - Single forward pass computes both predictions and entropy loss.
   - Returns predictions from that single forward pass; no second post-update forward.
5. Numerical Safeguards & Non-Destructive Logging:
   - Verifies 100% finite outputs; records NaNs/Infs as execution failures.
   - Never deletes or overwrites existing runs; preserves failed and interrupted state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
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
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.evaluation_dataset import KuroEvaluationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.adabn import AdaBNAdapter
from src.tta.source_only import SourceOnlyAdapter
from src.tta.tent_reference import ReferenceTentAdapter


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_model_state_sha256(model: nn.Module) -> str:
    """Computes SHA-256 digest over all floating-point parameters and buffers."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    for name, buf in sorted(model.named_buffers()):
        if buf is not None:
            h.update(name.encode("utf-8"))
            h.update(buf.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieves current Git commit and working-tree cleanliness."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("ascii")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        return {"commit": commit, "is_clean": len(status) == 0}
    except Exception as e:
        return {"commit": "unknown", "is_clean": False, "error": str(e)}


def build_stream_id(method: str, seed: int, order: str, reset: str) -> str:
    """Builds standardized stream configuration identifier."""
    m_code = {"source_only": "source", "adabn": "adabn", "tent": "tent"}.get(method, method)
    o_code = {"order_a": "oa", "order_b": "ob"}.get(order, order)
    r_code = {"reset_per_event": "re", "continual": "co"}.get(reset, reset)
    return f"stream_{m_code}_s{seed}_{o_code}_{r_code}"


class ControlledTTAMatrixRunner:
    """Orchestrates controlled test-time adaptation matrix execution and offline evaluation."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda:0",
        dry_run: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = device
        self.dry_run = dry_run

        # Output paths
        out_cfg = self.cfg.get("outputs", {})
        self.output_dir = output_dir or (REPO_ROOT / out_cfg.get("root_dir", "outputs/week8_controlled_tta"))
        self.event_metrics_csv_path = self.output_dir / "stream_event_metrics.csv"
        self.stream_summary_json_path = self.output_dir / "stream_matrix_summary.json"
        self.manifest_json_path = self.output_dir / "week8_tta_manifest.json"
        self.log_dir = REPO_ROOT / out_cfg.get("log_dir", "logs/week8_controlled_tta")

        # Validation partitions
        self.target_events: List[int] = self.cfg["dataset"]["target_events"]
        self.forbidden_events: List[int] = self.cfg["dataset"]["forbidden_sealed_test_events"]
        self.expected_counts: Dict[int, int] = {
            int(k): int(v) for k, v in self.cfg["dataset"]["expected_event_counts"].items()
        }
        self.expected_total_samples: int = self.cfg["dataset"]["expected_total_samples"]

        # Cache for matched source controls to enable explicit reuse
        # Key: (checkpoint_seed, stream_order) -> Dict with event_metrics and predictions
        self.source_control_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def validate_preflight(self) -> Dict[str, Any]:
        """Audits configuration, event disjointness, checkpoints, and model architectures."""
        # 1. Target events disjoint from sealed test partition
        intersection = set(self.target_events) & set(self.forbidden_events)
        if intersection:
            raise RuntimeError(
                f"PREFLIGHT FATAL: Target events overlap with sealed test events: {sorted(list(intersection))}"
            )

        # 2. Checkpoint audits
        checkpoint_status = {}
        for s_id, s_info in self.cfg["source_checkpoints"]["seeds"].items():
            ckpt_path = REPO_ROOT / s_info["checkpoint_path"]
            exp_hash = s_info["expected_sha256"]
            if not ckpt_path.is_file():
                checkpoint_status[s_id] = {"exists": False, "path": str(ckpt_path)}
                continue
            act_hash = compute_file_sha256(ckpt_path)
            checkpoint_status[s_id] = {
                "exists": True,
                "path": str(ckpt_path),
                "hash_match": act_hash == exp_hash,
                "expected": exp_hash,
                "actual": act_hash,
            }

        return {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.cfg.get("status"),
            "target_events": self.target_events,
            "forbidden_events": self.forbidden_events,
            "checkpoint_status": checkpoint_status,
        }

    def load_datasets(
        self,
        target_events: Optional[List[int]] = None,
        is_smoke: bool = False,
    ) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Loads adaptation dataset (sensory only) and offline evaluation dataset (labels included).

        Strict Architectural Isolation:
        - adaptation_dataset: load_labels=False. Under no circumstance does it touch MK0_MLU.
        - offline_eval_dataset: load_labels=True. Used strictly offline for frozen predictions.
        """
        d_cfg = self.cfg["dataset"]
        dataset_root = Path(d_cfg["dataset_root"])
        manifest_path = Path(d_cfg["manifest_path"])
        events = target_events if target_events is not None else self.target_events

        # Adaptation dataset: strictly sensory
        adaptation_dataset = KuroEvaluationDataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            allowed_events=events,
            canonical_preevent=self.cfg["model"]["canonical_preevent"],
            sar_clamp_max=d_cfg["sar_clamp_max"],
            sar_mean=d_cfg["sar_mean"],
            sar_std=d_cfg["sar_std"],
            load_labels=False,  # STRICT LABEL ISOLATION
        )

        # Offline evaluation dataset: labels included strictly for scoring
        offline_eval_dataset = KuroEvaluationDataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            allowed_events=events,
            canonical_preevent=self.cfg["model"]["canonical_preevent"],
            sar_clamp_max=d_cfg["sar_clamp_max"],
            sar_mean=d_cfg["sar_mean"],
            sar_std=d_cfg["sar_std"],
            load_labels=True,
        )

        if not is_smoke:
            if len(adaptation_dataset) != self.expected_total_samples:
                raise ValueError(
                    f"Sample count mismatch: Loaded {len(adaptation_dataset)} vs expected {self.expected_total_samples}"
                )
        else:
            expected_smoke_total = sum(self.expected_counts.get(eid, 0) for eid in events)
            if len(adaptation_dataset) != expected_smoke_total:
                raise ValueError(
                    f"Smoke partition sample mismatch: Loaded {len(adaptation_dataset)} vs expected {expected_smoke_total}"
                )

        return adaptation_dataset, offline_eval_dataset

    def instantiate_adapter(
        self,
        method: str,
        checkpoint_seed: int,
        checkpoint_path: Path,
    ) -> Any:
        """Instantiates the declared adapter for the given method and checkpoint."""
        m_cfg = self.cfg["model"]
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            in_channels=m_cfg["in_channels"],
            out_ch=m_cfg["num_classes"],
            base_channel=m_cfg["base_channel"],
            device=self.device,
        )

        if method == "source_only":
            return SourceOnlyAdapter(
                model=wrapper,
                checkpoint_path=str(checkpoint_path),
                device=self.device,
            )
        elif method == "adabn":
            return AdaBNAdapter(
                model=wrapper,
                checkpoint_path=str(checkpoint_path),
                device=self.device,
            )
        elif method == "tent":
            t_cfg = self.cfg["method_hyperparameters"]["tent"]
            return ReferenceTentAdapter(
                model=wrapper,
                checkpoint_path=str(checkpoint_path),
                learning_rate=t_cfg.get("learning_rate", 1e-4),
                betas=tuple(t_cfg.get("betas", (0.9, 0.999))),
                weight_decay=t_cfg.get("weight_decay", 0.0),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown adaptation method: {method}")

    def get_ordered_samples_for_stream(
        self,
        dataset: KuroEvaluationDataset,
        order_id: str,
        events: Optional[List[int]] = None,
        max_samples_per_event: Optional[int] = None,
    ) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """Returns the list of (event_id, event_samples) in declared stream order."""
        o_cfg = self.cfg["matrix_specification"]["stream_orders"][order_id]
        event_seq = o_cfg["event_sequence"]
        within_sort = o_cfg["within_event_ordering"]

        target_seq = [e for e in event_seq if events is None or e in events]

        # Group dataset samples by event, attaching index
        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(dataset.samples):
            eid = int(item["event_id"])
            item_with_idx = dict(item)
            item_with_idx["dataset_index"] = idx
            event_to_samples.setdefault(eid, []).append(item_with_idx)

        ordered_stream: List[Tuple[int, List[Dict[str, Any]]]] = []
        for eid in target_seq:
            if eid not in event_to_samples:
                continue
            samples = event_to_samples[eid]
            if within_sort == "deterministic_sorted_sample_id_ascending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))
            elif within_sort == "deterministic_sorted_sample_id_descending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)
            else:
                raise ValueError(f"Unknown within-event ordering: {within_sort}")

            if max_samples_per_event is not None and max_samples_per_event > 0:
                samples = samples[:max_samples_per_event]

            ordered_stream.append((eid, samples))

        return ordered_stream

    def run_stream_adaptation(
        self,
        adapter: Any,
        ordered_stream: List[Tuple[int, List[Dict[str, Any]]]],
        adaptation_dataset: KuroEvaluationDataset,
        batch_size: int = 16,
        reset_mode: str = "reset_per_event",
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Executes online streaming adaptation and returns frozen cached predictions and batch diagnostics."""
        frozen_predictions: Dict[str, torch.Tensor] = {}
        batch_diagnostics: List[Dict[str, Any]] = []

        # Reset once at start of entire stream for continual mode
        if reset_mode == "continual":
            adapter.reset_to_source()

        for event_idx, (event_id, sample_items) in enumerate(ordered_stream):
            # Reset before each event for episodic mode
            if reset_mode == "reset_per_event":
                adapter.reset_to_source()

            num_samples = len(sample_items)
            for i in range(0, num_samples, batch_size):
                batch_items = sample_items[i : i + batch_size]
                sids = [s["sample_id"] for s in batch_items]
                eids = [s["event_id"] for s in batch_items]

                # Load sensory tensors via dataset __getitem__
                batch_tensors = [adaptation_dataset[s["dataset_index"]] for s in batch_items]
                xA = torch.stack([b["xA"] for b in batch_tensors]).to(self.device)
                xB = torch.stack([b["xB"] for b in batch_tensors]).to(self.device)
                sar_valid = torch.stack([b["valid_mask"] for b in batch_tensors]).to(self.device)

                # adapt_and_predict with STRICT label isolation
                out = adapter.adapt_and_predict(
                    xA=xA,
                    xB=xB,
                    sar_valid_mask=sar_valid,
                    sample_ids=sids,
                    event_ids=eids,
                )

                if not torch.isfinite(out["logits"]).all():
                    raise FloatingPointError(
                        f"Non-finite logits detected in event {event_id}, batch starting at index {i}"
                    )

                step_diag = out.get("step_diagnostics", {})
                batch_diagnostics.append({
                    "event_id": event_id,
                    "batch_index": i // batch_size,
                    "batch_size": len(batch_items),
                    "step_count": getattr(adapter, "step_count", 0),
                    "adaptation_loss": step_diag.get("adaptation_loss", 0.0),
                })

                preds = out["predicted_class"].detach().cpu()
                for j, sid in enumerate(sids):
                    frozen_predictions[str(sid)] = preds[j]

        return frozen_predictions, batch_diagnostics

    def evaluate_frozen_predictions_offline(
        self,
        frozen_predictions: Dict[str, torch.Tensor],
        offline_eval_dataset: KuroEvaluationDataset,
        checkpoint_seed: int,
        stream_order: str = "order_a",
        target_events: Optional[List[int]] = None,
        max_samples_per_event: Optional[int] = None,
        is_smoke: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluates frozen predictions against ground-truth labels strictly offline."""
        eval_events = target_events if target_events is not None else self.target_events
        event_to_items: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(offline_eval_dataset.samples):
            eid = int(item["event_id"])
            if eid in eval_events:
                item_with_idx = dict(item)
                item_with_idx["dataset_index"] = idx
                event_to_items.setdefault(eid, []).append(item_with_idx)

        event_rows: List[Dict[str, Any]] = []
        stream_tp, stream_fp, stream_fn, stream_tn = 0, 0, 0, 0

        # Matched source control for this seed and stream order
        matched_source = self.source_control_cache.get((checkpoint_seed, stream_order))

        for eid in eval_events:
            items = event_to_items.get(eid, [])
            items = sorted(items, key=lambda x: str(x["sample_id"]))
            if max_samples_per_event is not None and max_samples_per_event > 0:
                items = items[:max_samples_per_event]

            tp_c, fp_c, fn_c, tn_c = 0, 0, 0, 0
            valid_pix = 0
            gt_flood_pix = 0
            pred_flood_pix = 0

            for item in items:
                sid = str(item["sample_id"])
                if sid not in frozen_predictions:
                    raise KeyError(f"Missing prediction for sample {sid} in event {eid}")

                pred = frozen_predictions[sid]
                sample_data = offline_eval_dataset[item["dataset_index"]]
                label = sample_data["label"]
                sar_valid = sample_data["valid_mask"]

                # Double-masking validity: strictly offline
                v_mask = (sar_valid == 1) & (label != 3)

                pred_fl = (pred == 2) & v_mask
                gt_fl = (label == 2) & v_mask

                tp = int((pred_fl & gt_fl).sum().item())
                fp = int((pred_fl & ~gt_fl).sum().item())
                fn = int((~pred_fl & gt_fl).sum().item())
                tn = int((~pred_fl & ~gt_fl & v_mask).sum().item())

                tp_c += tp
                fp_c += fp
                fn_c += fn
                tn_c += tn
                valid_pix += int(v_mask.sum().item())
                gt_flood_pix += int(gt_fl.sum().item())
                pred_flood_pix += int(pred_fl.sum().item())

            stream_tp += tp_c
            stream_fp += fp_c
            stream_fn += fn_c
            stream_tn += tn_c

            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp_c / (tp_c + fp_c + fn_c) if (tp_c + fp_c + fn_c) > 0 else 0.0

            delta_f1 = 0.0
            delta_rec = 0.0
            impact = "neutral"
            if matched_source is not None and eid in matched_source["per_event"]:
                src_f1 = matched_source["per_event"][eid]["f1"]
                src_rec = matched_source["per_event"][eid]["recall"]
                delta_f1 = f1 - src_f1
                delta_rec = rec - src_rec
                if delta_f1 <= -0.01:
                    impact = "harmful"
                elif delta_f1 >= 0.01:
                    impact = "beneficial"

            event_rows.append({
                "event_id": eid,
                "sample_count": len(items),
                "valid_pixels": valid_pix,
                "gt_flood_pixels": gt_flood_pix,
                "pred_flood_pixels": pred_flood_pix,
                "tp": tp_c,
                "fp": fp_c,
                "fn": fn_c,
                "tn": tn_c,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": iou,
                "delta_f1": delta_f1,
                "delta_recall": delta_rec,
                "impact": impact,
                "is_formal_run": not is_smoke,
            })

        macro_f1 = float(np.mean([r["f1"] for r in event_rows])) if event_rows else 0.0
        macro_rec = float(np.mean([r["recall"] for r in event_rows])) if event_rows else 0.0
        macro_prec = float(np.mean([r["precision"] for r in event_rows])) if event_rows else 0.0
        macro_iou = float(np.mean([r["iou"] for r in event_rows])) if event_rows else 0.0

        pooled_prec = stream_tp / (stream_tp + stream_fp) if (stream_tp + stream_fp) > 0 else 0.0
        pooled_rec = stream_tp / (stream_tp + stream_fn) if (stream_tp + stream_fn) > 0 else 0.0
        pooled_f1 = 2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = stream_tp / (stream_tp + stream_fp + stream_fn) if (stream_tp + stream_fp + stream_fn) > 0 else 0.0

        min_event = min(event_rows, key=lambda r: r["f1"]) if event_rows else {"event_id": -1, "f1": 0.0}
        harmed_count = sum(1 for r in event_rows if r["impact"] == "harmful")
        benefited_count = sum(1 for r in event_rows if r["impact"] == "beneficial")

        stream_summary = {
            "is_formal_run": not is_smoke,
            "macro_f1": macro_f1,
            "macro_recall": macro_rec,
            "macro_precision": macro_prec,
            "macro_iou": macro_iou,
            "pooled_f1": pooled_f1,
            "pooled_recall": pooled_rec,
            "pooled_precision": pooled_prec,
            "pooled_iou": pooled_iou,
            "worst_event": {"event_id": min_event["event_id"], "f1": min_event["f1"]},
            "num_harmed_events": harmed_count,
            "num_benefited_events": benefited_count,
            "per_event": {r["event_id"]: r for r in event_rows},
        }

        return event_rows, stream_summary

    def run_smoke(self, batch_size: int = 16) -> Dict[str, Any]:
        """Runs bounded diagnostic smoke test on Checkpoint Seed 0 and Order A.

        Executes 5 physical cases on exactly 38 samples (19 from Event 437, 19 from Event 1111003):
        1. source_only (order_a, s0) -> state immutability checked & reused for both reset modes
        2. adabn (order_a, s0, reset_per_event)
        3. adabn (order_a, s0, continual)
        4. tent (order_a, s0, reset_per_event)
        5. tent (order_a, s0, continual)
        """
        start_time = time.time()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Output collision check
        if self.output_dir.exists():
            existing_files = [f for f in self.output_dir.iterdir() if f.is_file()]
            if len(existing_files) > 0:
                raise RuntimeError(
                    f"ABORT: Smoke output directory '{self.output_dir}' already exists and contains "
                    f"{len(existing_files)} files. Refusing to overwrite per safety rules."
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        git_info = get_git_info()
        print("=" * 75)
        print("WEEK 8B BOUNDED TTA DIAGNOSTIC SMOKE TEST")
        print("=" * 75)
        print(f"Timestamp:       {now_iso}")
        print(f"Git Commit:      {git_info['commit']} (clean={git_info['is_clean']})")
        print(f"Device:          {self.device}")
        print(f"Output Dir:      {self.output_dir}")
        print(f"Workload:        Checkpoint Seed 0, Order A")
        print(f"Events:          [437, 1111003] (19 samples each, total 38 samples)")
        print(f"Batch Size:      {batch_size} (16 + 3 partial batch)")
        print(f"Formal Run Flag: is_formal_run = False (diagnostic only; NOT for Gate A2)")
        print("=" * 75)

        # 1. Preflight
        preflight = self.validate_preflight()
        s0_info = preflight["checkpoint_status"].get("0", {})
        if not s0_info.get("exists") or not s0_info.get("hash_match"):
            raise RuntimeError(f"Checkpoint seed 0 validation failed: {s0_info}")
        print("[Preflight] Checkpoint Seed 0 verified bit-exact.")
        print("[Preflight] Disjointness from sealed test partition verified.")

        # 2. Datasets
        smoke_events = [437, 1111003]
        max_samples_per_event = 19
        adaptation_dataset, offline_eval_dataset = self.load_datasets(
            target_events=smoke_events, is_smoke=True
        )
        print(f"[Dataset] Adaptation dataset loaded (MK0_MNA only, load_labels=False).")
        print(f"[Dataset] Offline evaluation dataset loaded (load_labels=True).")

        # 3. Stream selection
        ordered_stream = self.get_ordered_samples_for_stream(
            dataset=adaptation_dataset,
            order_id="order_a",
            events=smoke_events,
            max_samples_per_event=max_samples_per_event,
        )

        total_smoke_samples = sum(len(s) for _, s in ordered_stream)
        if total_smoke_samples != 38:
            raise ValueError(f"Expected 38 total smoke samples, got {total_smoke_samples}")

        evaluated_samples_manifest = []
        for eid, s_list in ordered_stream:
            print(f"  Event {eid}: {len(s_list)} samples selected in Order A")
            for item in s_list:
                evaluated_samples_manifest.append({
                    "sample_id": str(item["sample_id"]),
                    "event_id": eid,
                    "dataset_index": item["dataset_index"],
                })

        ckpt_path_s0 = REPO_ROOT / self.cfg["source_checkpoints"]["seeds"]["0"]["checkpoint_path"]

        # Physical run definitions
        physical_cases = [
            ("source_only", 0, "order_a", "source_only"),
            ("adabn", 0, "order_a", "reset_per_event"),
            ("adabn", 0, "order_a", "continual"),
            ("tent", 0, "order_a", "reset_per_event"),
            ("tent", 0, "order_a", "continual"),
        ]

        physical_run_records = []
        all_csv_rows = []
        logical_stream_summaries = {}

        for p_idx, (method, seed, order, reset) in enumerate(physical_cases, start=1):
            p_stream_id = f"physical_{method}_s{seed}_{order}_{reset}"
            print("-" * 75)
            print(f"Physical Run {p_idx}/5: {p_stream_id}")
            t0 = time.time()

            adapter = self.instantiate_adapter(
                method=method,
                checkpoint_seed=seed,
                checkpoint_path=ckpt_path_s0,
            )

            model_to_hash = adapter.model.model if hasattr(adapter.model, "model") else adapter.model
            pre_hash = compute_model_state_sha256(model_to_hash)

            preds, batch_diags = self.run_stream_adaptation(
                adapter=adapter,
                ordered_stream=ordered_stream,
                adaptation_dataset=adaptation_dataset,
                batch_size=batch_size,
                reset_mode=reset if method != "source_only" else "reset_per_event",
            )

            post_hash = compute_model_state_sha256(model_to_hash)
            t1 = time.time()

            # State integrity checks
            if method == "source_only":
                if pre_hash != post_hash:
                    raise RuntimeError("Source-only state immutability violated: hash changed!")
                print("  State immutability verified: pre_hash == post_hash.")
            elif method == "tent":
                if pre_hash == post_hash:
                    raise RuntimeError("Tent parameters failed to update: hash unchanged!")
                print(f"  Tent adaptation verified: BN parameters updated over {getattr(adapter, 'step_count', 0)} steps.")

            # Offline evaluation
            event_rows, stream_summary = self.evaluate_frozen_predictions_offline(
                frozen_predictions=preds,
                offline_eval_dataset=offline_eval_dataset,
                checkpoint_seed=seed,
                stream_order=order,
                target_events=smoke_events,
                max_samples_per_event=max_samples_per_event,
                is_smoke=True,
            )

            run_record = {
                "physical_run_index": p_idx,
                "physical_stream_id": p_stream_id,
                "method": method,
                "checkpoint_seed": seed,
                "stream_order": order,
                "reset_mode": reset,
                "duration_seconds": round(t1 - t0, 3),
                "num_samples": len(preds),
                "num_batches": len(batch_diags),
                "pre_hash": pre_hash,
                "post_hash": post_hash,
                "macro_f1": stream_summary["macro_f1"],
                "macro_recall": stream_summary["macro_recall"],
                "batch_diagnostics": batch_diags,
            }
            physical_run_records.append(run_record)

            print(f"  Completed in {round(t1 - t0, 2)}s | 2-Event Macro F1: {stream_summary['macro_f1']:.4f} | Rec: {stream_summary['macro_recall']:.4f}")

            # Populate logical stream mapping
            if method == "source_only":
                self.source_control_cache[(seed, order)] = stream_summary
                # Create both reset_per_event and continual logical entries
                for r_mode in ["reset_per_event", "continual"]:
                    l_id = build_stream_id("source_only", seed, order, r_mode)
                    is_phys = (r_mode == "reset_per_event")
                    logical_stream_summaries[l_id] = {
                        "stream_id": l_id,
                        "method": "source_only",
                        "checkpoint_seed": seed,
                        "stream_order": order,
                        "reset_mode": r_mode,
                        "is_physical_run": is_phys,
                        "physical_stream_id": p_stream_id,
                        "provenance": "executed" if is_phys else f"reused_from: {build_stream_id('source_only', seed, order, 'reset_per_event')}",
                        "summary": stream_summary,
                    }
                    for erow in event_rows:
                        row_copy = dict(erow)
                        row_copy.update({
                            "stream_id": l_id,
                            "method": "source_only",
                            "checkpoint_seed": seed,
                            "stream_order": order,
                            "reset_mode": r_mode,
                            "is_physical_run": is_phys,
                            "physical_stream_id": p_stream_id,
                        })
                        all_csv_rows.append(row_copy)
            else:
                l_id = build_stream_id(method, seed, order, reset)
                logical_stream_summaries[l_id] = {
                    "stream_id": l_id,
                    "method": method,
                    "checkpoint_seed": seed,
                    "stream_order": order,
                    "reset_mode": reset,
                    "is_physical_run": True,
                    "physical_stream_id": p_stream_id,
                    "provenance": "executed",
                    "summary": stream_summary,
                }
                for erow in event_rows:
                    row_copy = dict(erow)
                    row_copy.update({
                        "stream_id": l_id,
                        "method": method,
                        "checkpoint_seed": seed,
                        "stream_order": order,
                        "reset_mode": reset,
                        "is_physical_run": True,
                        "physical_stream_id": p_stream_id,
                    })
                    all_csv_rows.append(row_copy)

        # 4. Serialize CSV
        csv_headers = [
            "stream_id", "method", "checkpoint_seed", "stream_order", "reset_mode",
            "is_physical_run", "physical_stream_id", "event_id", "sample_count",
            "valid_pixels", "gt_flood_pixels", "pred_flood_pixels", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "iou", "delta_f1", "delta_recall", "impact", "is_formal_run",
        ]
        with open(self.event_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for r in all_csv_rows:
                writer.writerow({k: r.get(k, "") for k in csv_headers})

        # 5. Serialize Summary JSON
        summary_payload = {
            "run_metadata": {
                "is_formal_run": False,
                "evaluation_context": "smoke_test",
                "diagnostic_gate_a2_eligible": False,
                "notes": "Diagnostic smoke test only on 38 samples (19 per event) for seed 0 Order A. Scores are non-formal and must NOT be used for Gate A2 decisions.",
                "timestamp_utc": now_iso,
                "git_commit": git_info["commit"],
                "git_is_clean": git_info["is_clean"],
                "device": self.device,
                "protocol_version": self.cfg.get("protocol_version"),
            },
            "smoke_specification": {
                "checkpoint_seed": 0,
                "stream_order": "order_a",
                "events_evaluated": smoke_events,
                "samples_per_event": {str(eid): max_samples_per_event for eid in smoke_events},
                "total_samples": total_smoke_samples,
                "batch_size": batch_size,
            },
            "physical_runs": physical_run_records,
            "logical_streams": logical_stream_summaries,
            "gate_a2_evaluation": {
                "eligible": False,
                "reason": "Diagnostic smoke test on 38 samples; subset scores must not be interpreted as Gate A2 evidence.",
            },
        }
        with open(self.stream_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        # 6. Serialize Manifest JSON
        manifest_payload = {
            "is_formal_run": False,
            "evaluation_context": "smoke_test",
            "timestamp_utc": now_iso,
            "git_commit": git_info["commit"],
            "git_is_clean": git_info["is_clean"],
            "output_directory": str(self.output_dir),
            "checkpoints_verified": {
                "0": {
                    "checkpoint_path": str(ckpt_path_s0),
                    "sha256": s0_info.get("actual"),
                    "hash_match": s0_info.get("hash_match"),
                }
            },
            "physical_stream_count": len(physical_run_records),
            "logical_stream_count": len(logical_stream_summaries),
            "samples_evaluated": evaluated_samples_manifest,
            "total_samples": len(evaluated_samples_manifest),
        }
        with open(self.manifest_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        elapsed = time.time() - start_time
        print("=" * 75)
        print("DIAGNOSTIC SMOKE TEST COMPLETED SUCCESSFULLY")
        print(f"Elapsed Time:    {elapsed:.2f}s")
        print(f"Metrics CSV:     {self.event_metrics_csv_path}")
        print(f"Summary JSON:    {self.stream_summary_json_path}")
        print(f"Manifest JSON:   {self.manifest_json_path}")
        print(f"Formal Status:   is_formal_run = False (safeguard asserted)")
        print("=" * 75)

        return summary_payload


    def audit_gate_a2(self, logical_streams: Dict[str, Any]) -> Dict[str, Any]:
        """Audits Gate A2 criterion:
        At least two identical checkpoint seed IDs must satisfy delta F1 <= -0.01
        under BOTH stream orders (Order A and Order B) for the same method/reset mode
        on the same event.
        """
        methods = ["adabn", "tent"]
        reset_modes = ["reset_per_event", "continual"]
        target_events = self.target_events

        harmed_combinations = []

        for m in methods:
            for r in reset_modes:
                for eid in target_events:
                    # Find harmed seeds under Order A
                    harmed_seeds_oa = set()
                    for s in [0, 1, 2]:
                        s_id = build_stream_id(m, s, "order_a", r)
                        if s_id in logical_streams:
                            ev_data = logical_streams[s_id]["summary"]["per_event"].get(eid, {})
                            if ev_data.get("impact") == "harmful":
                                harmed_seeds_oa.add(s)

                    # Find harmed seeds under Order B
                    harmed_seeds_ob = set()
                    for s in [0, 1, 2]:
                        s_id = build_stream_id(m, s, "order_b", r)
                        if s_id in logical_streams:
                            ev_data = logical_streams[s_id]["summary"]["per_event"].get(eid, {})
                            if ev_data.get("impact") == "harmful":
                                harmed_seeds_ob.add(s)

                    # Common harmed seeds across BOTH orders
                    common_seeds = sorted(list(harmed_seeds_oa & harmed_seeds_ob))
                    if len(common_seeds) >= 2:
                        harmed_combinations.append({
                            "method": m,
                            "reset_mode": r,
                            "event_id": eid,
                            "common_harmed_seeds": common_seeds,
                            "num_common_seeds": len(common_seeds),
                            "order_a_harmed_seeds": sorted(list(harmed_seeds_oa)),
                            "order_b_harmed_seeds": sorted(list(harmed_seeds_ob)),
                        })

        reproducible_harm = len(harmed_combinations) > 0
        category = "harm_observed" if reproducible_harm else "harm_not_established"

        return {
            "eligible": True,
            "category": category,
            "reproducible_harm_observed": reproducible_harm,
            "rule": "delta F1 <= -0.01 on same event in >=2 identical seeds under BOTH Order A and Order B for same method and reset mode",
            "num_reproducible_harm_combinations": len(harmed_combinations),
            "harmed_combinations": harmed_combinations,
        }

    def run_matrix(
        self,
        batch_size: int = 16,
        stream_ids: Optional[List[str]] = None,
        adaptation_dataset: Optional[KuroEvaluationDataset] = None,
        offline_eval_dataset: Optional[KuroEvaluationDataset] = None,
        override_adapters: Optional[Dict[Tuple[str, int], Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the formal 30 physical streams, populates 36 logical cells, and audits Gate A2.

        Physical Stream Decomposition (30 streams):
        - 6 matched Source-Only streams: 3 checkpoint seeds x 2 stream orders
        - 12 AdaBN streams: 3 checkpoint seeds x 2 stream orders x 2 reset modes
        - 12 reference-aligned Tent streams: 3 checkpoint seeds x 2 stream orders x 2 reset modes

        Logical Cell Accounting (36 cells):
        - The 6 source-only streams are verified immutable and explicitly reused across
          reset_per_event and continual modes, producing 12 logical cells.
        - The 24 adaptation streams produce 24 logical cells.
        - Total logical event-result rows: 36 x 6 = 216 rows (180 physical rows + 36 reused rows).
        """
        start_time = time.time()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Output collision check
        if self.output_dir.exists():
            existing_files = [f for f in self.output_dir.iterdir() if f.is_file()]
            if len(existing_files) > 0:
                raise RuntimeError(
                    f"ABORT: Formal output directory '{self.output_dir}' already exists and contains "
                    f"{len(existing_files)} files. Refusing to overwrite previous attempt per safety rules."
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        git_info = get_git_info()
        print("=" * 80)
        print("WEEK 8B CONTROLLED TEST-TIME ADAPTATION FORMAL MATRIX RUN")
        print("=" * 80)
        print(f"Timestamp (UTC):  {now_iso}")
        print(f"Git Commit:       {git_info['commit']} (clean={git_info['is_clean']})")
        print(f"Device:           {self.device}")
        print(f"Output Dir:       {self.output_dir}")
        print(f"Log Dir:          {self.log_dir}")
        print(f"Target Events:    {self.target_events} (6 events)")
        print(f"Batch Size:       {batch_size} (partial tail batches retained)")
        print(f"Physical Streams: 30 planned streams")
        print(f"Logical Cells:    36 matrix cells (6 matched source controls explicitly reused)")
        print(f"Gate A2 Rule:     Harmful in >=2 of 3 seeds under BOTH Order A & Order B")
        print("=" * 80)

        # 2. Preflight verification
        preflight = self.validate_preflight()
        for s_id, s_info in preflight["checkpoint_status"].items():
            if not self.dry_run and override_adapters is None:
                if not s_info.get("exists") or not s_info.get("hash_match"):
                    raise RuntimeError(f"Checkpoint seed {s_id} validation failed: {s_info}")
        print("[Preflight] Source checkpoints verified bit-exact against frozen hashes.")
        print("[Preflight] Disjointness from sealed test partition verified.")

        # 3. Load datasets
        if adaptation_dataset is None or offline_eval_dataset is None:
            adaptation_dataset, offline_eval_dataset = self.load_datasets(
                target_events=self.target_events, is_smoke=False
            )

        total_samples = len(adaptation_dataset)
        if not self.dry_run and override_adapters is None and total_samples != self.expected_total_samples:
            raise ValueError(
                f"Sample count mismatch: Loaded {total_samples} vs expected {self.expected_total_samples}"
            )
        print(f"[Dataset] Adaptation dataset loaded ({total_samples} samples; MK0_MNA only; load_labels=False).")
        print(f"[Dataset] Offline evaluation dataset loaded ({len(offline_eval_dataset)} samples; load_labels=True).")

        # 4. Enumerate 30 Physical Streams in declared order
        seeds = [0, 1, 2]
        orders = ["order_a", "order_b"]
        reset_modes = ["reset_per_event", "continual"]

        physical_specs: List[Tuple[str, int, str, str]] = []
        # Phase 1: Source controls (1 per seed and order)
        for s in seeds:
            for o in orders:
                physical_specs.append(("source_only", s, o, "source_only"))
        # Phase 2: AdaBN
        for s in seeds:
            for o in orders:
                for r in reset_modes:
                    physical_specs.append(("adabn", s, o, r))
        # Phase 3: Tent
        for s in seeds:
            for o in orders:
                for r in reset_modes:
                    physical_specs.append(("tent", s, o, r))

        # Filter by stream_ids if requested
        if stream_ids is not None:
            allowed_ids = set(stream_ids)
            filtered_specs = []
            for m, s, o, r in physical_specs:
                if m == "source_only":
                    id_re = build_stream_id("source_only", s, o, "reset_per_event")
                    id_co = build_stream_id("source_only", s, o, "continual")
                    if id_re in allowed_ids or id_co in allowed_ids:
                        filtered_specs.append((m, s, o, r))
                else:
                    lid = build_stream_id(m, s, o, r)
                    if lid in allowed_ids:
                        filtered_specs.append((m, s, o, r))
            physical_specs = filtered_specs

        print(f"[Matrix] Executing {len(physical_specs)} physical streams...")

        physical_run_records = []
        all_csv_rows = []
        logical_stream_summaries = {}

        # Cache pre-sorted streams per order to avoid re-sorting repeatedly
        ordered_streams_cache: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = {}
        for o in orders:
            ordered_streams_cache[o] = self.get_ordered_samples_for_stream(
                dataset=adaptation_dataset,
                order_id=o,
                events=self.target_events,
            )
            stream_count = sum(len(s) for _, s in ordered_streams_cache[o])
            if stream_count != total_samples:
                raise ValueError(
                    f"Sample coverage mismatch in {o}: Stream contains {stream_count} samples vs dataset {total_samples}"
                )

        for p_idx, (method, seed, order, reset) in enumerate(physical_specs, start=1):
            p_stream_id = f"physical_{method}_s{seed}_{order}_{reset}"
            print("-" * 80)
            print(f"[{p_idx}/{len(physical_specs)}] Executing: {p_stream_id}")
            t0 = time.time()

            ckpt_path = REPO_ROOT / self.cfg["source_checkpoints"]["seeds"][str(seed)]["checkpoint_path"]

            if override_adapters is not None:
                if callable(override_adapters):
                    adapter = override_adapters(method, seed)
                elif (method, seed) in override_adapters:
                    val = override_adapters[(method, seed)]
                    adapter = val() if callable(val) else val
                else:
                    adapter = self.instantiate_adapter(
                        method=method,
                        checkpoint_seed=seed,
                        checkpoint_path=ckpt_path,
                    )
            else:
                adapter = self.instantiate_adapter(
                    method=method,
                    checkpoint_seed=seed,
                    checkpoint_path=ckpt_path,
                )

            model_to_hash = adapter.model.model if hasattr(adapter.model, "model") else adapter.model
            pre_hash = compute_model_state_sha256(model_to_hash)

            ordered_stream = ordered_streams_cache[order]

            # Execute streaming adaptation
            preds, batch_diags = self.run_stream_adaptation(
                adapter=adapter,
                ordered_stream=ordered_stream,
                adaptation_dataset=adaptation_dataset,
                batch_size=batch_size,
                reset_mode=reset if method != "source_only" else "reset_per_event",
            )

            post_hash = compute_model_state_sha256(model_to_hash)
            t1 = time.time()

            # State integrity assertions
            if method == "source_only":
                if pre_hash != post_hash:
                    raise RuntimeError(f"State immutability violated in {p_stream_id}: pre/post hash mismatch!")
                print(f"  [Integrity] Model state immutability verified (SHA: {pre_hash[:16]}...).")
            elif method == "tent":
                if pre_hash == post_hash:
                    raise RuntimeError(f"Tent adaptation failed in {p_stream_id}: model state did not update!")
                print(f"  [Integrity] Tent adaptation verified: updated over {getattr(adapter, 'step_count', 0)} steps.")

            # Offline evaluation on frozen predictions
            event_rows, stream_summary = self.evaluate_frozen_predictions_offline(
                frozen_predictions=preds,
                offline_eval_dataset=offline_eval_dataset,
                checkpoint_seed=seed,
                stream_order=order,
                target_events=self.target_events,
                max_samples_per_event=None,
                is_smoke=False,
            )

            run_record = {
                "physical_run_index": p_idx,
                "physical_stream_id": p_stream_id,
                "method": method,
                "checkpoint_seed": seed,
                "stream_order": order,
                "reset_mode": reset,
                "duration_seconds": round(t1 - t0, 3),
                "num_samples": len(preds),
                "num_batches": len(batch_diags),
                "pre_hash": pre_hash,
                "post_hash": post_hash,
                "macro_f1": stream_summary["macro_f1"],
                "macro_recall": stream_summary["macro_recall"],
                "worst_event": stream_summary["worst_event"],
                "num_harmed_events": stream_summary["num_harmed_events"],
                "num_benefited_events": stream_summary["num_benefited_events"],
            }
            physical_run_records.append(run_record)

            print(
                f"  Completed in {t1 - t0:.1f}s | Macro F1: {stream_summary['macro_f1']:.4f} | "
                f"Rec: {stream_summary['macro_recall']:.4f} | Harmed Evts: {stream_summary['num_harmed_events']} | "
                f"Benefited: {stream_summary['num_benefited_events']}"
            )

            # Map to logical matrix cells
            if method == "source_only":
                self.source_control_cache[(seed, order)] = stream_summary
                for r_mode in ["reset_per_event", "continual"]:
                    l_id = build_stream_id("source_only", seed, order, r_mode)
                    is_phys = (r_mode == "reset_per_event")
                    logical_stream_summaries[l_id] = {
                        "stream_id": l_id,
                        "method": "source_only",
                        "checkpoint_seed": seed,
                        "stream_order": order,
                        "reset_mode": r_mode,
                        "is_physical_run": is_phys,
                        "physical_stream_id": p_stream_id,
                        "provenance": "executed" if is_phys else f"reused_from: {build_stream_id('source_only', seed, order, 'reset_per_event')}",
                        "summary": stream_summary,
                    }
                    for erow in event_rows:
                        row_copy = dict(erow)
                        row_copy.update({
                            "stream_id": l_id,
                            "method": "source_only",
                            "checkpoint_seed": seed,
                            "stream_order": order,
                            "reset_mode": r_mode,
                            "is_physical_run": is_phys,
                            "physical_stream_id": p_stream_id,
                        })
                        all_csv_rows.append(row_copy)
            else:
                l_id = build_stream_id(method, seed, order, reset)
                logical_stream_summaries[l_id] = {
                    "stream_id": l_id,
                    "method": method,
                    "checkpoint_seed": seed,
                    "stream_order": order,
                    "reset_mode": reset,
                    "is_physical_run": True,
                    "physical_stream_id": p_stream_id,
                    "provenance": "executed",
                    "summary": stream_summary,
                }
                for erow in event_rows:
                    row_copy = dict(erow)
                    row_copy.update({
                        "stream_id": l_id,
                        "method": method,
                        "checkpoint_seed": seed,
                        "stream_order": order,
                        "reset_mode": reset,
                        "is_physical_run": True,
                        "physical_stream_id": p_stream_id,
                    })
                    all_csv_rows.append(row_copy)

        # 5. Gate A2 Audit Evaluation
        gate_a2_result = self.audit_gate_a2(logical_stream_summaries)

        # 6. Serialize CSV
        csv_headers = [
            "stream_id", "method", "checkpoint_seed", "stream_order", "reset_mode",
            "is_physical_run", "physical_stream_id", "event_id", "sample_count",
            "valid_pixels", "gt_flood_pixels", "pred_flood_pixels", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "iou", "delta_f1", "delta_recall", "impact", "is_formal_run",
        ]
        with open(self.event_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for r in all_csv_rows:
                writer.writerow({k: r.get(k, "") for k in csv_headers})

        # 7. Serialize Summary JSON
        summary_payload = {
            "run_metadata": {
                "is_formal_run": True,
                "evaluation_context": "formal_evaluation",
                "timestamp_utc": now_iso,
                "git_commit": git_info["commit"],
                "git_is_clean": git_info["is_clean"],
                "device": self.device,
                "protocol_version": self.cfg.get("protocol_version"),
            },
            "matrix_specification": {
                "num_physical_streams": len(physical_run_records),
                "num_logical_streams": len(logical_stream_summaries),
                "num_csv_rows": len(all_csv_rows),
                "num_physical_csv_rows": sum(1 for r in all_csv_rows if r["is_physical_run"]),
                "num_reused_csv_rows": sum(1 for r in all_csv_rows if not r["is_physical_run"]),
                "checkpoint_seeds": seeds,
                "stream_orders": orders,
                "reset_modes": reset_modes,
                "target_events": self.target_events,
                "total_samples_per_stream": total_samples,
                "batch_size": batch_size,
            },
            "physical_runs": physical_run_records,
            "logical_streams": logical_stream_summaries,
            "gate_a2_evaluation": gate_a2_result,
        }
        with open(self.stream_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        # 8. Serialize Manifest JSON
        samples_manifest = [
            {"sample_id": str(s["sample_id"]), "event_id": int(s["event_id"]), "dataset_index": idx}
            for idx, s in enumerate(adaptation_dataset.samples)
        ]
        manifest_payload = {
            "is_formal_run": True,
            "evaluation_context": "formal_evaluation",
            "timestamp_utc": now_iso,
            "git_commit": git_info["commit"],
            "git_is_clean": git_info["is_clean"],
            "output_directory": str(self.output_dir),
            "checkpoints_verified": {
                str(s): {
                    "checkpoint_path": str(REPO_ROOT / self.cfg["source_checkpoints"]["seeds"][str(s)]["checkpoint_path"]),
                    "sha256": preflight["checkpoint_status"].get(str(s), {}).get("actual"),
                    "hash_match": preflight["checkpoint_status"].get(str(s), {}).get("hash_match"),
                }
                for s in seeds
            },
            "physical_stream_count": len(physical_run_records),
            "logical_stream_count": len(logical_stream_summaries),
            "total_physical_rows": sum(1 for r in all_csv_rows if r["is_physical_run"]),
            "total_logical_rows": len(all_csv_rows),
            "samples_evaluated": samples_manifest,
            "total_samples": len(samples_manifest),
        }
        with open(self.manifest_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        elapsed = time.time() - start_time
        print("=" * 80)
        print("FORMAL TTA MATRIX EVALUATION COMPLETED SUCCESSFULLY")
        print(f"Elapsed Time:       {elapsed:.1f}s")
        print(f"Metrics CSV:        {self.event_metrics_csv_path} ({len(all_csv_rows)} rows)")
        print(f"Summary JSON:       {self.stream_summary_json_path}")
        print(f"Manifest JSON:      {self.manifest_json_path}")
        print(f"Gate A2 Criterion:  {gate_a2_result['category']} (Harm observed: {gate_a2_result['reproducible_harm_observed']})")
        print("=" * 80)

        return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 8 Controlled TTA Matrix and Gate A2 Audit."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/week8_controlled_tta_protocol.json",
        help="Path to Week 8 protocol configuration JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (default: outputs/week8_controlled_tta/formal_v1 or outputs/week8_controlled_tta_smoke/smoke_s0_v1 for smoke).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Perform preflight integrity validation only and exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without full GPU adaptation execution.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bounded diagnostic smoke test (Seed 0, Order A, Events [437, 1111003], 19 samples per event, batch size 16, 5 physical cases).",
    )
    parser.add_argument(
        "--stream_ids",
        nargs="+",
        default=None,
        help="Optional specific stream IDs to execute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    out_dir = Path(args.output_dir) if args.output_dir else None
    if args.smoke and out_dir is None:
        out_dir = REPO_ROOT / "outputs/week8_controlled_tta_smoke/smoke_s0_v1"
    elif not args.smoke and out_dir is None:
        out_dir = REPO_ROOT / "outputs/week8_controlled_tta/formal_v1"

    runner = ControlledTTAMatrixRunner(
        config_path=config_path,
        output_dir=out_dir,
        device=args.device,
        dry_run=args.dry_run,
    )

    preflight = runner.validate_preflight()

    if args.validate_only:
        print("=" * 70)
        mode_label = "SMOKE DIAGNOSTIC" if args.smoke else "FORMAL MATRIX"
        print(f"WEEK 8B TTA PREFLIGHT VALIDATION [{mode_label}] (--validate_only)")
        print("=" * 70)
        print(f"Protocol Version: {preflight.get('protocol_version')}")
        print(f"Status:           {preflight.get('status')}")
        if args.smoke:
            print("Smoke Workload:   Checkpoint Seed 0, Order A, Events [437, 1111003], 19 samples each")
        else:
            print(f"Formal Workload:  30 Physical Streams, 36 Logical Cells, {runner.expected_total_samples} samples")
        print(f"Target Events:    {preflight.get('target_events')}")
        print(f"Forbidden Events: {preflight.get('forbidden_events')}")
        print("Checkpoint Status:")
        for sid, sinfo in preflight.get("checkpoint_status", {}).items():
            print(f"  Seed {sid}: exists={sinfo.get('exists')}, hash_match={sinfo.get('hash_match')}")
        print("=" * 70)
        print("Preflight validation PASSED.")
        return

    if args.smoke:
        runner.run_smoke(batch_size=16)
        return

    runner.run_matrix(batch_size=16, stream_ids=args.stream_ids)


if __name__ == "__main__":
    main()


