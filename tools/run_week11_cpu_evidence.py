"""ChangeGuard Week 11A CPU Evidence Consistency Runner Interface.

Governing Protocol: docs/protocols/week11_evidence_consistency_protocol_v1.md
Governing Config: configs/changeguard/week11_evidence_consistency_v1.json

Strict Authorizations & Fail-Closed Safeguards:
1. Real-Data Access:
   - Checked against config['execution_authorizations']['real_data_execution_authorized'].
   - Must be an explicit JSON boolean (bool).
   - If false, any attempt to access or evaluate real Kuro Siwo data raises PermissionError.
2. GPU Access:
   - Checked against config['execution_authorizations']['gpu_execution_authorized'].
   - Must be an explicit JSON boolean (bool).
   - Under APPROVED_FOR_CPU_IMPLEMENTATION, any device other than 'cpu' or enabling GPU
     raises PermissionError before model or device access (R05).
3. Label-Free Separation:
   - Adaptation forward pass strictly uses (xA, xB, sar_valid_mask).
   - Offline evaluation of simulated ground truth happens entirely after prediction caching.
4. Synthetic Pipeline Validation:
   - Supports fully deterministic synthetic batch streams for verifying adaptation mechanics,
     evidence indicator logging, sample tracking, provenance disclosure, and offline metric
     accumulation on CPU (R04).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MockSNUNet(nn.Module):
    """Synthetic proxy model matching SNUNet multi-temporal interface for CPU testing."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)


class Week11CPURunner:
    """Fail-closed runner interface for Week 11A CPU implementation and synthetic validation."""

    def __init__(self, config_path: Path, device: str = "cpu"):
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = str(device).lower()
        self._enforce_authorizations()

    def _enforce_authorizations(self) -> None:
        """Enforces protocol execution boundaries with strict boolean type checking (R05)."""
        auth = self.cfg.get("execution_authorizations", {})
        if "real_data_execution_authorized" not in auth:
            raise KeyError("Missing required authorization flag 'real_data_execution_authorized'")
        if "gpu_execution_authorized" not in auth:
            raise KeyError("Missing required authorization flag 'gpu_execution_authorized'")

        real_auth = auth["real_data_execution_authorized"]
        gpu_auth = auth["gpu_execution_authorized"]

        if not isinstance(real_auth, bool):
            raise TypeError(
                f"Invalid type for 'real_data_execution_authorized': expected JSON boolean (bool), "
                f"got {type(real_auth).__name__} ({real_auth!r})"
            )
        if not isinstance(gpu_auth, bool):
            raise TypeError(
                f"Invalid type for 'gpu_execution_authorized': expected JSON boolean (bool), "
                f"got {type(gpu_auth).__name__} ({gpu_auth!r})"
            )

        self.real_data_authorized = real_auth
        self.gpu_authorized = gpu_auth

        status = self.cfg.get("status")
        if status != "APPROVED_FOR_CPU_IMPLEMENTATION":
            raise ValueError(
                f"Unsupported configuration status '{status}'. "
                "This CPU runner strictly requires status == 'APPROVED_FOR_CPU_IMPLEMENTATION'."
            )

        if self.device != "cpu":
            raise PermissionError(
                f"GPU EXECUTION NOT AUTHORIZED: Requested device '{self.device}'. "
                f"Under status '{status}', execution is strictly restricted to CPU."
            )
        if self.real_data_authorized:
            raise PermissionError(
                f"REAL DATA EXECUTION NOT AUTHORIZED: Status '{status}' forbids "
                "real_data_execution_authorized=true."
            )
        if self.gpu_authorized:
            raise PermissionError(
                f"GPU EXECUTION NOT AUTHORIZED: Status '{status}' forbids "
                "gpu_execution_authorized=true."
            )

    def validate_preflight(self) -> Dict[str, Any]:
        """Validates configuration structure, event boundaries, and security parameters."""
        status = self.cfg.get("status")
        if status != "APPROVED_FOR_CPU_IMPLEMENTATION":
            raise ValueError(
                f"Unsupported configuration status '{status}'. "
                "This CPU runner strictly requires status == 'APPROVED_FOR_CPU_IMPLEMENTATION'."
            )
        protocol_version = self.cfg.get("protocol_version")
        dev_events = self.cfg.get("events", {}).get("development", [])
        sealed_events = self.cfg.get("events", {}).get("quarantined_sealed_test", [])

        # Check for event partition disjointness
        overlap = set(dev_events).intersection(set(sealed_events))
        if overlap:
            raise ValueError(f"FATAL: Development and sealed test events overlap: {overlap}")

        return {
            "protocol_version": protocol_version,
            "status": status,
            "real_data_execution_authorized": self.real_data_authorized,
            "gpu_execution_authorized": self.gpu_authorized,
            "development_events": dev_events,
            "sealed_test_events_quarantined": len(sealed_events),
            "config_sha256": compute_file_sha256(self.config_path),
        }

    def run_real_stream(self, *args: Any, **kwargs: Any) -> None:
        """Fail-closed barrier protecting real dataset access."""
        if not self.real_data_authorized:
            raise PermissionError(
                "REAL DATA EXECUTION NOT AUTHORIZED: 'real_data_execution_authorized' is false in "
                f"{self.config_path.name}. Week 11A is restricted to CPU implementation and synthetic validation."
            )
        raise NotImplementedError("Real data stream execution is not yet unlocked.")

    def run_synthetic_pipeline(
        self,
        num_batches: int = 4,
        batch_size: int = 16,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Runs a complete synthetic pipeline validating adaptation, evidence logging, and offline metrics."""
        torch.manual_seed(seed)

        # 1. Instantiate synthetic model and adapter
        model = MockSNUNet(
            in_channels=self.cfg["model"].get("in_channels", 2),
            num_classes=self.cfg["model"].get("num_classes", 3),
        )

        h_cfg = self.cfg.get("adaptation_hyperparameters", {})
        c_cfg = self.cfg.get("consistency_views", {}).get("confidence_masking", {})

        adapter = EvidenceConsistencyAdapter(
            model=model,
            learning_rate=h_cfg.get("learning_rate", 1.25e-4),
            momentum=h_cfg.get("momentum", 0.9),
            weight_decay=h_cfg.get("weight_decay", 0.0),
            confidence_threshold=c_cfg.get("threshold", 0.90),
            device=self.device,
        )

        # Honest synthetic provenance disclosure (R04)
        provenance_metadata = {
            "checkpoint_seed": "not_applicable_synthetic",
            "model_architecture": "MockSNUNet_ECAM_proxy",
            "stream_order": "deterministic_synthetic_stream",
            "reset_mode": "not_applicable_synthetic",
            "data_mode": "synthetic_only",
        }

        # 2. Generate synthetic batches with diverse conditions
        step_records: List[Dict[str, Any]] = []
        all_sample_ids: List[str] = []
        batch_to_samples: Dict[str, List[str]] = {}

        # Offline metric accumulation (TP, FP, FN, TN for flood class 2)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_valid_pixels = 0

        synthetic_event_id = 520

        for b_idx in range(num_batches):
            # Simulate a tail batch on the last iteration
            cur_bs = batch_size if b_idx < num_batches - 1 else max(1, batch_size // 2)
            xA = torch.randn(cur_bs, 2, 32, 32)
            xB = torch.randn(cur_bs, 2, 32, 32)

            # Sensory observation mask (mostly valid, some invalid)
            valid_mask = (torch.rand(cur_bs, 32, 32) > 0.05).long()

            # Batch 2 simulates an ambiguous/empty selection condition
            if b_idx == 2:
                # Scaled down to guarantee low confidence below threshold
                xA = xA * 0.01
                xB = xB * 0.01

            sample_ids = [f"synth_event_520_sample_{b_idx * batch_size + i:04d}" for i in range(cur_bs)]
            event_ids = [synthetic_event_id] * cur_bs
            all_sample_ids.extend(sample_ids)
            batch_to_samples[f"batch_{b_idx}"] = list(sample_ids)

            # --- ADAPTATION STEP (Label-Free) ---
            # Strictly NO labels passed to adapter
            result = adapter.adapt_and_predict(
                xA=xA,
                xB=xB,
                sar_valid_mask=valid_mask,
                sample_ids=sample_ids,
                event_ids=event_ids,
            )

            primary_pred = result["primary_prediction"]
            meta = result["adaptation_metadata"]
            evidence = result["risk_evidence"]

            # --- OFFLINE EVALUATION (Separate Phase) ---
            # Generate simulated ground truth for offline accumulation
            # 0: non-water, 1: perm-water, 2: flood, 3: ignore/invalid
            simulated_gt = torch.randint(0, 3, (cur_bs, 32, 32))
            pred_class = primary_pred["predicted_class"]

            eval_mask = (valid_mask == 1) & (simulated_gt != 3)
            pred_flood = (pred_class == 2) & eval_mask
            gt_flood = (simulated_gt == 2) & eval_mask

            tp = int((pred_flood & gt_flood).sum().item())
            fp = int((pred_flood & ~gt_flood).sum().item())
            fn = int((~pred_flood & gt_flood).sum().item())
            tn = int((~pred_flood & ~gt_flood & eval_mask).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            total_valid_pixels += int(eval_mask.sum().item())

            # Full persisted record distinguishing batch aggregates from per-sample fields (R04, N02)
            record = {
                "batch_index": b_idx,
                "batch_size": cur_bs,
                "event_id": synthetic_event_id,
                "sample_ids": "|".join(sample_ids),
                "action": meta["action"],
                "update_applied": meta["update_applied"],
                "failure_reason": meta["failure_reason"] if meta.get("failure_reason") else "none",
                "loss": meta["loss"],  # Python None preserved for actual JSON null (N02)
                "selected_pixels": meta["selected_pixels"],
                "valid_pixels": meta["valid_pixels"],
                "predicted_flood_pixels": primary_pred["predicted_flood_pixels"],
                "update_time_ms": meta["update_time_ms"],
                "transform_applied": meta["transform_applied"],
                "candidate_bn_context": evidence["bn_context"]["candidate_bn"],
                "source_bn_context": evidence["bn_context"]["source_bn"],
                "primary_prediction_timing": evidence["timing_contexts"]["primary_prediction"],
                "source_disagreement_timing": evidence["timing_contexts"]["source_disagreement_rate"],
                "null_probes_timing": evidence["timing_contexts"]["null_probes"],
                "entropy_norm": evidence["entropy_norm"],
                "max_confidence": evidence["max_confidence"],
                "source_disagreement_rate": evidence["source_disagreement_rate"],
                "source_disagreement_rate_pre_update": evidence["source_disagreement_rate_pre_update"],
                "null_score_primary": evidence["null_score_primary"],
                "null_score_primary_post_update": evidence["null_score_primary_post_update"],
                "null_pre2_delta": evidence["null_pre2_delta"],
                "null_post_delta": evidence["null_post_delta"],
                "cand_null_pre2_prob": evidence["cand_null_pre2_prob"],
                "src_null_pre2_prob": evidence["src_null_pre2_prob"],
                "cand_null_post_prob": evidence["cand_null_post_prob"],
                "src_null_post_prob": evidence["src_null_post_prob"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            step_records.append(record)

        # Calculate final offline metrics
        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = (2 * precision * recall) / max(1e-8, precision + recall) if (precision + recall) > 0 else 0.0
        iou = total_tp / max(1, total_tp + total_fp + total_fn)

        summary = {
            "provenance_metadata": provenance_metadata,
            "num_batches": num_batches,
            "total_samples": len(all_sample_ids),
            "unique_samples": len(set(all_sample_ids)),
            "all_sample_ids": all_sample_ids,
            "batch_to_samples": batch_to_samples,
            "num_updates_applied": adapter.num_updates_applied,
            "num_updates_skipped": adapter.num_updates_skipped,
            "num_updates_failed": adapter.num_updates_failed,
            "final_step_count": adapter.step_count,
            "offline_evaluation": {
                "total_valid_pixels": total_valid_pixels,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn,
                "flood_precision": precision,
                "flood_recall": recall,
                "flood_f1": f1,
                "flood_iou": iou,
            },
            "step_records": step_records,
        }

        # Save artifacts if output directory is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "synthetic_run_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            # CSV-specific copy where None is represented textually as "null" (N02)
            csv_records = [
                {k: ("null" if v is None else v) for k, v in rec.items()}
                for rec in step_records
            ]

            with open(output_dir / "synthetic_step_metrics.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_records[0].keys()))
                writer.writeheader()
                writer.writerows(csv_records)

        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChangeGuard Week 11A CPU Evidence Consistency Runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/changeguard/week11_evidence_consistency_v1.json",
        help="Path to Week 11 evidence consistency config.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run preflight validation and authorization checks without processing data.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic pipeline simulation for offline testing.",
    )
    parser.add_argument(
        "--num_synthetic_batches",
        type=int,
        default=4,
        help="Number of synthetic batches to run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for synthetic data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save synthetic run results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target compute device (must be 'cpu' under CPU implementation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = REPO_ROOT / args.config

    runner = Week11CPURunner(config_path=config_path, device=args.device)
    preflight = runner.validate_preflight()

    print("=" * 70)
    print("ChangeGuard Week 11A CPU Evidence Consistency Runner")
    print("=" * 70)
    print(f"Protocol Version: {preflight['protocol_version']}")
    print(f"Status:           {preflight['status']}")
    print(f"Config SHA-256:   {preflight['config_sha256']}")
    print(f"Real Data Auth:   {preflight['real_data_execution_authorized']}")
    print(f"GPU Auth:         {preflight['gpu_execution_authorized']}")
    print(f"Development Evts: {preflight['development_events']}")
    print(f"Sealed Evts Quarn:{preflight['sealed_test_events_quarantined']}")
    print("=" * 70)

    if args.validate_only:
        print("[PREFLIGHT OK] Authorization and configuration bounds verified.")
        return

    if args.synthetic:
        out_dir = Path(args.output_dir) if args.output_dir else None
        print(f"\nRunning synthetic pipeline ({args.num_synthetic_batches} batches, BS={args.batch_size})...")
        summary = runner.run_synthetic_pipeline(
            num_batches=args.num_synthetic_batches,
            batch_size=args.batch_size,
            output_dir=out_dir,
        )
        print(f"[SYNTHETIC COMPLETE] Total samples: {summary['total_samples']} (Unique: {summary['unique_samples']})")
        print(
            f"Updates Applied: {summary['num_updates_applied']}, "
            f"Skipped: {summary['num_updates_skipped']}, "
            f"Failed: {summary['num_updates_failed']}"
        )
        print(f"Offline Flood F1: {summary['offline_evaluation']['flood_f1']:.4f}, IoU: {summary['offline_evaluation']['flood_iou']:.4f}")
        if out_dir:
            print(f"Artifacts saved to: {out_dir}")
        return

    # If neither validate_only nor synthetic is requested, attempt real stream (which fails closed)
    print("\nAttempting real data execution...")
    runner.run_real_stream()


if __name__ == "__main__":
    main()
