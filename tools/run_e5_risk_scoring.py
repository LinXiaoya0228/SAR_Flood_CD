#!/usr/bin/env python3
"""ChangeGuard Plan V3 Work Package E5: Temporal Risk Evidence & Generic-Risk Runner.

Evaluates state-neutral bi-temporal null probes (s_null) and generic risk discriminators:
1. Contexts:
   - source_frozen_running_stats: Immutable source checkpoint, eval() mode, frozen running stats
   - matched_bn_no_update_bs16: Matched normalization context control
   - candidate_e2_conf90: Base consistency recipe (tau=0.90, horizontal flip), PRE_UPDATE timing
2. Probes:
   - Primary: s_null = 0.5 * sum_{z in {pre2/pre2, post/post}} max(0, delta_null(z))
   - Secondary: s_flood_expansion, s_flood_disappearance
   - Generic: predictive entropy, max confidence, source-to-candidate KL, disagreement,
     flood pixel shifts, augmentation inconsistency, BN affine displacements
3. Safety:
   - Strictly sensory-valid mask (MK0_MNA) for all online probes and adaptation.
   - Ground truth (MK0_MLU) accessible strictly offline after predictions and actions are frozen.
   - Sealed-test partition strictly quarantined.
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
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.e1_orchestrator import (
    E1OfflineEvaluator,
    LabelGuardedBatch,
)
from src.changeguard.e5_risk_probes import (
    E5RiskProbeEvaluator,
    assert_model_non_aliasing,
    compute_flood_change_probes,
    compute_generic_risk_evidence,
    compute_null_pair_risk_score,
)
from src.changeguard.recorder import (
    CACHE_SCHEMA_V1,
    ExtendedExperimentRecorder,
    compute_bn_affine_displacement,
    compute_gradient_l2_norm,
)
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    KuroEvaluationDataset,
)
from src.models.snunet_wrapper import SNUNetWrapper

EXPECTED_CHECKPOINT_SHA256 = "247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f"
EXPECTED_CHECKPOINT_SEED1_SHA256 = "72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a"

CANONICAL_CHECKPOINTS: Dict[int, Dict[str, str]] = {
    0: {
        "checkpoint_path": "outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt",
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
    },
    1: {
        "checkpoint_path": "outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt",
        "expected_sha256": EXPECTED_CHECKPOINT_SEED1_SHA256,
    },
}

CANONICAL_STREAM_ORDERS: Dict[str, Dict[str, Any]] = {
    "order_a": {
        "event_sequence": [279, 437, 520, 559, 1111003, 1111008],
        "within_event_ordering": "deterministic_sorted_sample_id_ascending",
    },
    "order_b": {
        "event_sequence": [1111008, 1111003, 559, 520, 437, 279],
        "within_event_ordering": "deterministic_sorted_sample_id_descending",
    },
}

SCIENTIFIC_INPUT_FILES = [
    "configs/changeguard/e5_temporal_risk_v1.json",
    "docs/e5_preflight_and_contracts.md",
    "src/changeguard/e5_risk_probes.py",
    "src/changeguard/recorder.py",
    "src/data/evaluation_dataset.py",
    "tools/run_e5_risk_scoring.py",
]


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a physical file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class E5RiskScoringRunner:
    """Orchestrates E5 temporal risk scoring and generic risk evidence collection."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        checkpoint_seed: Optional[int] = None,
        stream_order: Optional[str] = None,
        produce_convenience_aggregate: Optional[bool] = None,
        device: str = "cpu",
        is_smoke: bool = False,
        limit_samples: Optional[int] = None,
        allow_dirty: bool = False,
    ):
        self.config_path = config_path.resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # Fail-closed config status check
        status = self.cfg.get("status")
        if status != "LOCKED_FOR_EXECUTION":
            raise ValueError(
                f"Config status '{status}' is not LOCKED_FOR_EXECUTION. Execution refused."
            )

        self.device = torch.device(device)
        self.is_smoke = bool(is_smoke)
        self.allow_dirty = bool(allow_dirty)

        spec = self.cfg.get(
            "smoke_specification" if self.is_smoke else "screening_specification", {}
        )

        # Checkpoint seed resolution and validation
        if checkpoint_seed is not None:
            self.checkpoint_seed = int(checkpoint_seed)
        else:
            self.checkpoint_seed = int(spec.get("checkpoint_seed", 0))

        seeds_cfg = self.cfg.get("source_checkpoints", {}).get("seeds", {})
        seed_key = str(self.checkpoint_seed)
        if seed_key not in seeds_cfg and self.checkpoint_seed not in CANONICAL_CHECKPOINTS:
            raise ValueError(
                f"Unsupported checkpoint seed: {self.checkpoint_seed}. "
                f"Configured seeds: {list(seeds_cfg.keys())}, canonical seeds: {sorted(list(CANONICAL_CHECKPOINTS.keys()))}"
            )

        # Stream order resolution and validation
        if stream_order is not None:
            self.stream_order = str(stream_order)
        else:
            self.stream_order = str(spec.get("stream_order", "order_a"))

        if self.stream_order not in CANONICAL_STREAM_ORDERS:
            raise ValueError(
                f"Unsupported stream_order: {self.stream_order}. "
                f"Allowed stream orders: {sorted(list(CANONICAL_STREAM_ORDERS.keys()))}"
            )

        # Output directory resolution
        if output_dir is not None:
            self.output_dir = output_dir.resolve()
        elif self.is_smoke:
            smoke_rel = self.cfg.get("smoke_specification", {}).get(
                "output_dir", "outputs/e5_temporal_evidence/smoke_v4/"
            )
            self.output_dir = (REPO_ROOT / smoke_rel).resolve()
        else:
            formal_rel = self.cfg.get("screening_specification", {}).get(
                "output_dir", "outputs/e5_temporal_evidence/formal_s0_oa_v1/"
            )
            self.output_dir = (REPO_ROOT / formal_rel).resolve()

        # Run ID resolution
        if run_id is not None:
            self.run_id = str(run_id)
        elif self.is_smoke:
            self.run_id = str(
                self.cfg.get("smoke_specification", {}).get("run_id", "candidate_e2_conf90")
            )
        else:
            self.run_id = str(
                self.cfg.get("screening_specification", {}).get(
                    "run_id", "e5_temporal_risk_s0_oa_v1"
                )
            )

        # Produce convenience aggregate resolution
        # Formal screening defaults to False to prevent OOM on Event 279 (2,381 samples) and duplicate storage
        if produce_convenience_aggregate is not None:
            self.produce_convenience_aggregate = bool(produce_convenience_aggregate)
        elif self.is_smoke:
            self.produce_convenience_aggregate = bool(
                self.cfg.get("smoke_specification", {}).get("produce_convenience_aggregate", True)
            )
        else:
            self.produce_convenience_aggregate = bool(
                self.cfg.get("screening_specification", {}).get("produce_convenience_aggregate", False)
            )

        # Target event selection
        if self.is_smoke:
            smoke_evt = int(self.cfg["smoke_specification"]["smoke_event"])
            self.target_events = [smoke_evt]
            self.limit_samples = (
                limit_samples
                if limit_samples is not None
                else int(self.cfg["smoke_specification"]["sample_limit"])
            )
            self.batch_size = int(self.cfg["smoke_specification"]["batch_size"])
        else:
            cfg_seq = [
                int(e)
                for e in self.cfg["screening_specification"]["event_sequence"]
            ]
            self.target_events = cfg_seq
            self.limit_samples = limit_samples
            self.batch_size = int(
                self.cfg["screening_specification"]["batch_size"]
            )

        # Sealed test quarantine guard (strictly checked before canonical sequence validation)
        for forbidden in FORBIDDEN_SEALED_TEST_EVENTS:
            if forbidden in self.target_events:
                raise RuntimeError(
                    f"CRITICAL SECURITY VIOLATION: Sealed-test event {forbidden} "
                    f"present in E5 target events!"
                )

        if not self.is_smoke:
            canonical_seq = CANONICAL_STREAM_ORDERS[self.stream_order]["event_sequence"]
            if stream_order is not None:
                self.target_events = list(canonical_seq)
            elif "stream_order" in self.cfg.get("screening_specification", {}):
                if self.target_events != canonical_seq:
                    raise ValueError(
                        f"Config event_sequence {self.target_events} does not match canonical sequence {canonical_seq} for {self.stream_order}"
                    )

        # Dataset parameters
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (
            (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
            if not Path(ds_cfg["dataset_root"]).is_absolute()
            else Path(ds_cfg["dataset_root"])
        )
        self.manifest_path = (
            (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
            if not Path(ds_cfg["manifest_path"]).is_absolute()
            else Path(ds_cfg["manifest_path"])
        )
        self.sar_mean = tuple(float(x) for x in ds_cfg["sar_mean"])
        self.sar_std = tuple(float(x) for x in ds_cfg["sar_std"])
        self.sar_clamp_max = float(ds_cfg["sar_clamp_max"])
        self.canonical_preevent = str(ds_cfg["canonical_preevent"])

        # Candidate parameters
        cand_cfg = self.cfg["contexts"]["candidate_e2_conf90"]
        self.threshold = float(cand_cfg["confidence_threshold"])
        self.lr = float(cand_cfg["learning_rate"])
        self.betas = tuple(float(x) for x in cand_cfg["betas"])
        self.weight_decay = float(cand_cfg["weight_decay"])

        # Cache
        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None

    def _get_checkpoint_info(self) -> Tuple[Path, str]:
        """Resolves checkpoint path and expected SHA-256 for current checkpoint seed."""
        seed_key = str(self.checkpoint_seed)
        seeds_cfg = self.cfg.get("source_checkpoints", {}).get("seeds", {})
        if seed_key in seeds_cfg:
            ckpt_cfg = seeds_cfg[seed_key]
            rel_path = ckpt_cfg["checkpoint_path"]
            expected_sha256 = ckpt_cfg.get("expected_sha256")
        elif self.checkpoint_seed in CANONICAL_CHECKPOINTS:
            canon = CANONICAL_CHECKPOINTS[self.checkpoint_seed]
            rel_path = canon["checkpoint_path"]
            expected_sha256 = canon["expected_sha256"]
        else:
            raise ValueError(f"Checkpoint seed '{self.checkpoint_seed}' not configured")

        ckpt_path = (REPO_ROOT / rel_path).resolve()
        if not expected_sha256:
            expected_sha256 = (
                EXPECTED_CHECKPOINT_SHA256
                if self.checkpoint_seed == 0
                else EXPECTED_CHECKPOINT_SEED1_SHA256
            )

        return ckpt_path, expected_sha256

    def run_preflight_checks(self) -> Dict[str, Any]:
        """Runs preflight verification of hashes, environment, and quarantine invariants."""
        ckpt_path, expected_sha256 = self._get_checkpoint_info()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        actual_ckpt_sha256 = compute_file_sha256(ckpt_path)
        if actual_ckpt_sha256 != expected_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch! Expected {expected_sha256}, got {actual_ckpt_sha256}"
            )

        config_sha256 = compute_file_sha256(self.config_path)

        # Git commit check
        git_status_raw = ""
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
                )
                .decode("utf-8")
                .strip()
            )
            git_status_raw = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT
                )
                .decode("utf-8")
                .strip()
            )
            is_dirty = bool(git_status_raw)
        except Exception:
            git_commit = "unknown"
            is_dirty = True

        # Enforce clean Git state on mandatory scientific inputs unless explicitly permitted
        dirty_scientific_inputs = []
        if is_dirty:
            for line in git_status_raw.splitlines():
                path_part = line[3:].strip()
                for p in SCIENTIFIC_INPUT_FILES:
                    if p in path_part:
                        dirty_scientific_inputs.append(path_part)
            if dirty_scientific_inputs and not self.allow_dirty:
                raise RuntimeError(
                    f"Preflight integrity violation: Relevant scientific input files are dirty in working tree: {dirty_scientific_inputs}. "
                    "Canonical runs must start from a clean Git commit! Pass --allow_dirty only for local development."
                )

        # Compute and verify hashes of all declared scientific inputs
        scientific_inputs_sha256 = {}
        for rel_p in SCIENTIFIC_INPUT_FILES:
            full_p = REPO_ROOT / rel_p
            if full_p.is_file():
                scientific_inputs_sha256[rel_p] = compute_file_sha256(full_p)
            else:
                raise FileNotFoundError(f"Missing mandatory scientific input file: {full_p}")

        return {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": actual_ckpt_sha256,
            "checkpoint_seed": self.checkpoint_seed,
            "stream_order": self.stream_order,
            "config_path": str(self.config_path),
            "config_sha256": config_sha256,
            "scientific_inputs_sha256": scientific_inputs_sha256,
            "git_commit": git_commit,
            "git_dirty": is_dirty,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda
            if torch.cuda.is_available()
            else None,
            "device": str(self.device),
            "target_events": self.target_events,
            "is_smoke": self.is_smoke,
            "batch_size": self.batch_size,
            "sample_limit": self.limit_samples,
        }

    def load_clean_models(self) -> Tuple[SNUNetWrapper, SNUNetWrapper, SNUNetWrapper]:
        """Loads candidate model, immutable source model, and matched BN control model from clean checkpoint."""
        ckpt_path, _ = self._get_checkpoint_info()

        # 1. Candidate model: adaptive BN affine parameters
        cand_model, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path), in_channels=2, out_ch=3, base_channel=32, device=str(self.device)
        )
        for p in cand_model.parameters():
            p.requires_grad = False
        for m in cand_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

        # 2. Source reference model: immutable, separate memory, requires_grad=False, eval()
        src_model, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path), in_channels=2, out_ch=3, base_channel=32, device=str(self.device)
        )
        for p in src_model.parameters():
            p.requires_grad = False
        src_model.eval()

        # 3. Matched BN No-Update Control: separate memory, requires_grad=False, current-batch stats
        bn_model, _ = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path), in_channels=2, out_ch=3, base_channel=32, device=str(self.device)
        )
        for p in bn_model.parameters():
            p.requires_grad = False
        bn_model.eval()
        for m in bn_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.track_running_stats = False

        # Non-aliasing assertions across all pairs
        assert_model_non_aliasing(cand_model, src_model)
        assert_model_non_aliasing(cand_model, bn_model)
        assert_model_non_aliasing(src_model, bn_model)

        return cand_model, src_model, bn_model

    def get_datasets(
        self,
    ) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Lazy loads sensory adaptation dataset and offline evaluation dataset."""
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

    def build_ordered_stream_batches(
        self,
    ) -> List[Tuple[int, List[LabelGuardedBatch]]]:
        """Partitions target events into deterministic ordered sequences of LabelGuardedBatch."""
        adapt_ds, offline_ds = self.get_datasets()

        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(adapt_ds.samples):
            eid = int(item["event_id"])
            item_copy = dict(item)
            item_copy["dataset_index"] = idx
            event_to_samples.setdefault(eid, []).append(item_copy)

        within_sort = "deterministic_sorted_sample_id_ascending"
        if self.stream_order == "order_b":
            within_sort = "deterministic_sorted_sample_id_descending"
        elif "within_event_ordering" in self.cfg.get("screening_specification", {}):
            within_sort = self.cfg["screening_specification"]["within_event_ordering"]

        stream_plan: List[Tuple[int, List[LabelGuardedBatch]]] = []
        for eid in self.target_events:
            if eid not in event_to_samples:
                continue
            samples = event_to_samples[eid]
            if within_sort == "deterministic_sorted_sample_id_ascending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))
            elif within_sort == "deterministic_sorted_sample_id_descending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)
            else:
                raise ValueError(f"Unknown within_sort ordering: {within_sort}")

            if self.limit_samples is not None:
                samples = samples[: self.limit_samples]

            event_batches: List[LabelGuardedBatch] = []
            for b_start in range(0, len(samples), self.batch_size):
                b_samples = samples[b_start : b_start + self.batch_size]
                b_ids = [str(s["sample_id"]) for s in b_samples]
                b_indices = [s["dataset_index"] for s in b_samples]

                adapt_items = [adapt_ds[idx] for idx in b_indices]
                offline_items = [offline_ds[idx] for idx in b_indices]

                xA = torch.stack([it["xA"] for it in adapt_items])
                xB = torch.stack([it["xB"] for it in adapt_items])
                valid_mask = torch.stack(
                    [it["valid_mask"] for it in adapt_items]
                )
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

        return stream_plan

    def execute_streaming_evaluation(self) -> Dict[str, Any]:
        """Executes full streaming probe evaluation across target events."""
        preflight_info = self.run_preflight_checks()

        # Enforce fresh output directory
        if self.output_dir.exists():
            existing_files = list(self.output_dir.glob("*"))
            if existing_files:
                raise RuntimeError(
                    f"Fresh output directory violation: {self.output_dir} is not empty! Refusing to overwrite."
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stream_batches = self.build_ordered_stream_batches()
        all_sample_ids_ordered: List[str] = []
        event_expected_sample_ids: Dict[int, List[str]] = {}
        for eid, b_list in stream_batches:
            evt_sids: List[str] = []
            for b in b_list:
                evt_sids.extend(b.sample_ids)
            event_expected_sample_ids[eid] = evt_sids
            all_sample_ids_ordered.extend(evt_sids)

        if self.is_smoke:
            deterministic_sample_ids = all_sample_ids_ordered[:32]
        else:
            deterministic_sample_ids = all_sample_ids_ordered

        deterministic_sample_ids_hash = hashlib.sha256(
            "".join(deterministic_sample_ids).encode("utf-8")
        ).hexdigest()

        event_contracts: Dict[str, Dict[str, Any]] = {}
        for eid, sids in event_expected_sample_ids.items():
            event_contracts[str(eid)] = {
                "expected_event_sample_count": len(sids),
                "expected_event_sample_ids": list(sids),
                "expected_event_sample_ids_sha256": hashlib.sha256(
                    "".join(sids).encode("utf-8")
                ).hexdigest(),
            }

        runner_sha256 = compute_file_sha256(Path(__file__).resolve())
        probes_sha256 = compute_file_sha256(REPO_ROOT / "src/changeguard/e5_risk_probes.py")
        protocol_path = REPO_ROOT / "docs/e5_preflight_and_contracts.md"
        protocol_sha256 = compute_file_sha256(protocol_path) if protocol_path.exists() else None

        # Write initial manifest prior to any model forward passes
        initial_manifest = {
            "work_package": "E5",
            "title": "Temporal Risk Evidence & Generic-Risk Preflight Manifest",
            "protocol_version": "e5_temporal_risk_v1",
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "exact_command": sys.argv,
            "git_commit": preflight_info["git_commit"],
            "git_dirty": preflight_info["git_dirty"],
            "code_provenance": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": runner_sha256,
                "probes_path": str((REPO_ROOT / "src/changeguard/e5_risk_probes.py").resolve()),
                "probes_sha256": probes_sha256,
                "protocol_path": str(protocol_path.resolve()) if protocol_path.exists() else None,
                "protocol_sha256": protocol_sha256,
                "checkpoint_path": preflight_info["checkpoint_path"],
                "checkpoint_sha256": preflight_info["checkpoint_sha256"],
                "config_path": preflight_info["config_path"],
                "config_sha256": preflight_info["config_sha256"],
                "scientific_inputs_sha256": preflight_info["scientific_inputs_sha256"],
            },
            "runtime": {
                "python_version": preflight_info["python_version"],
                "torch_version": preflight_info["torch_version"],
                "cuda_available": preflight_info["cuda_available"],
                "cuda_version": preflight_info["cuda_version"],
                "device": preflight_info["device"],
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() and self.device.type == "cuda"
                    else None
                ),
            },
            "dataset_provenance": {
                "dataset_root": str(self.dataset_root),
                "manifest_path": str(self.manifest_path),
                "sensory_mask_component": "MK0_MNA",
                "label_component": "MK0_MLU",
                "online_mask_definition": "valid = (MK0_MNA == 1)",
                "offline_evaluation_mask": "valid = (MK0_MNA == 1) & (MK0_MLU != 3)",
            },
            "contexts": [
                "source_frozen_running_stats",
                "matched_bn_no_update_bs16",
                "candidate_e2_conf90",
            ],
            "pair_types": [
                "pre2/post",
                "pre2/pre2",
                "post/post",
            ],
            "run_id": self.run_id,
            "checkpoint_seed": self.checkpoint_seed,
            "stream_order": self.stream_order,
            "checkpoint_id": f"source_all27_seed{self.checkpoint_seed}",
            "target_events": self.target_events,
            "batch_size": self.batch_size,
            "sample_limit": self.limit_samples,
            "is_smoke": self.is_smoke,
            "deterministic_sample_ids": deterministic_sample_ids,
            "deterministic_sample_ids_hash": deterministic_sample_ids_hash,
            "event_contracts": event_contracts,
            "authoritative_shard_policy": "batch_atomic_disk_shards_primary",
            "aggregate_cache_policy": (
                "convenience_aggregate_present"
                if self.produce_convenience_aggregate
                else "disabled_for_bounded_memory"
            ),
            "zero_sealed_test_declaration": "strictly_quarantined_zero_access",
            "status": "INITIALIZED",
        }
        with open(self.output_dir / "initial_manifest.json", "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        recorder = ExtendedExperimentRecorder(
            output_dir=self.output_dir,
            run_id=self.run_id,
            checkpoint_id=f"source_all27_seed{self.checkpoint_seed}",
            config_hash=preflight_info["config_sha256"],
            git_commit=preflight_info["git_commit"],
        )

        all_step_csv_rows: List[Dict[str, Any]] = []
        event_metrics_all: List[Dict[str, Any]] = []

        total_batches_processed = 0
        start_time = time.time()

        try:
            for eid, batches in stream_batches:
                print(
                    f"\n>>> Starting streaming evaluation on Event {eid} ({len(batches)} batches)..."
                )

                # Reset candidate, source, and BN control models per event
                cand_model, src_model, bn_model = self.load_clean_models()
                cand_model.train()

                # Optimizer on BN affine parameters
                trainable_params = [p for p in cand_model.parameters() if p.requires_grad]
                optimizer = optim.Adam(
                    trainable_params,
                    lr=self.lr,
                    betas=self.betas,
                    weight_decay=self.weight_decay,
                )

                evaluator = E5RiskProbeEvaluator(
                    candidate_model=cand_model,
                    source_model=src_model,
                    bn_control_model=bn_model,
                    device=self.device,
                )

                # Immutable expected-event contract established before processing begins
                expected_event_sample_ids = event_expected_sample_ids[eid]
                expected_event_sample_count = len(expected_event_sample_ids)
                expected_event_sample_ids_sha256 = hashlib.sha256(
                    "".join(expected_event_sample_ids).encode("utf-8")
                ).hexdigest()

                # Verify event contract matches initial manifest
                manifest_contract = initial_manifest["event_contracts"][str(eid)]
                if manifest_contract["expected_event_sample_count"] != expected_event_sample_count:
                    raise RuntimeError(
                        f"Event {eid} contract mismatch on sample count: "
                        f"{expected_event_sample_count} != {manifest_contract['expected_event_sample_count']}"
                    )
                if manifest_contract["expected_event_sample_ids"] != expected_event_sample_ids:
                    raise RuntimeError(
                        f"Event {eid} contract mismatch on ordered sample IDs!"
                    )

                processed_event_sample_ids: List[str] = []

                recorder.start_event(eid)

                event_tp = 0
                event_fp = 0
                event_fn = 0
                event_tn = 0
                event_valid = 0

                for b_idx, batch in enumerate(batches):
                    total_batches_processed += 1
                    b_start_time = time.time()

                    # Step 1: Sensory inputs only (strictly free of ground truth)
                    sensory = batch.get_sensory_inputs()
                    s_ids = sensory["sample_ids"]
                    processed_event_sample_ids.extend(s_ids)
                    xA = sensory["xA"].to(self.device)
                    xB = sensory["xB"].to(self.device)
                    valid_mask = sensory["valid_mask"].to(self.device)

                    # Step 2-5: State-neutral probe evaluation across all 3 contexts and 3 pairs PRE_UPDATE
                    probe_res = evaluator.evaluate_batch_probes_state_neutral(
                        xA=xA,
                        xB=xB,
                        valid_mask=valid_mask,
                    )

                    cand_pred_real = probe_res["cand_pred_real"].cpu().numpy()
                    src_pred_real = probe_res["src_pred_real"].cpu().numpy()

                    # Step 6: Assemble bounded batch spatial arrays for atomic sharding
                    batch_arrays: Dict[str, np.ndarray] = {}
                    for s_i, sid in enumerate(s_ids):
                        # 1. Sensory valid mask (MK0_MNA == 1)
                        batch_arrays[f"{sid}_sensory_mask"] = valid_mask[s_i].cpu().numpy().astype(np.uint8)

                        # 2. Hard predictions (uint8, 224x224) for 3 pairs x 3 contexts
                        batch_arrays[f"{sid}_source_real_pred"] = probe_res["src_pred_real"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_bn_control_real_pred"] = probe_res["bn_pred_real"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_candidate_real_pred"] = probe_res["cand_pred_real"][s_i].cpu().numpy().astype(np.uint8)

                        batch_arrays[f"{sid}_source_null_pre2_pred"] = probe_res["src_pred_null_pre2"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_bn_control_null_pre2_pred"] = probe_res["bn_pred_null_pre2"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_candidate_null_pre2_pred"] = probe_res["cand_pred_null_pre2"][s_i].cpu().numpy().astype(np.uint8)

                        batch_arrays[f"{sid}_source_null_post_pred"] = probe_res["src_pred_null_post"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_bn_control_null_post_pred"] = probe_res["bn_pred_null_post"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_candidate_null_post_pred"] = probe_res["cand_pred_null_post"][s_i].cpu().numpy().astype(np.uint8)

                        # 3. Full three-class probability maps (float16, 3x224x224)
                        batch_arrays[f"{sid}_source_real_probs"] = probe_res["src_probs_real"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_bn_control_real_probs"] = probe_res["bn_probs_real"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_candidate_real_probs"] = probe_res["cand_probs_real"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_candidate_real_strong_probs"] = probe_res["cand_probs_strong_aligned"][s_i].cpu().numpy().astype(np.float16)

                        batch_arrays[f"{sid}_source_null_pre2_probs"] = probe_res["src_probs_null_pre2"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_bn_control_null_pre2_probs"] = probe_res["bn_probs_null_pre2"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_candidate_null_pre2_probs"] = probe_res["cand_probs_null_pre2"][s_i].cpu().numpy().astype(np.float16)

                        batch_arrays[f"{sid}_source_null_post_probs"] = probe_res["src_probs_null_post"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_bn_control_null_post_probs"] = probe_res["bn_probs_null_post"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_candidate_null_post_probs"] = probe_res["cand_probs_null_post"][s_i].cpu().numpy().astype(np.float16)

                        # 4. Backward-compatible legacy aliases for E1-E4 evaluation tools
                        batch_arrays[f"{sid}"] = probe_res["cand_pred_real"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_src_pred"] = probe_res["src_pred_real"][s_i].cpu().numpy().astype(np.uint8)
                        batch_arrays[f"{sid}_PRE_UPDATE_candidate_prob_c2"] = probe_res["cand_prob_c2"][s_i].cpu().numpy().astype(np.float16)
                        batch_arrays[f"{sid}_FROZEN_SOURCE_source_prob_c2"] = probe_res["src_prob_c2"][s_i].cpu().numpy().astype(np.float16)

                    # Step 7: Atomically persist evidence shard to disk before label unlock
                    shard_contexts = [
                        "source_frozen_running_stats",
                        "matched_bn_no_update_bs16",
                        "candidate_e2_conf90",
                    ]
                    prelabel_evidence_elapsed_ms = (time.time() - b_start_time) * 1000.0
                    shard_path = recorder.record_batch_evidence_shard(
                        step_index=b_idx,
                        sample_ids=s_ids,
                        arrays=batch_arrays,
                        contexts=shard_contexts,
                        prelabel_evidence_elapsed_ms=prelabel_evidence_elapsed_ms,
                        timing_ms=prelabel_evidence_elapsed_ms,
                    )

                    # Step 7: Record step telemetry before update
                    extra_telem = {
                        "s_null": probe_res["s_null"],
                        "delta_null_pre2": probe_res["delta_null_pre2"],
                        "delta_null_post": probe_res["delta_null_post"],
                        "cand_null_pre2_mean": probe_res["cand_null_pre2_mean"],
                        "cand_null_post_mean": probe_res["cand_null_post_mean"],
                        "s_flood_expansion": probe_res["s_flood_expansion"],
                        "s_flood_disappearance": probe_res["s_flood_disappearance"],
                        "source_to_candidate_kl": probe_res["source_to_candidate_kl"],
                        "hard_disagreement_rate": probe_res["hard_disagreement_rate"],
                        "delta_flood_pixels": probe_res["delta_flood_pixels"],
                        "abs_delta_flood_pixels": probe_res["abs_delta_flood_pixels"],
                        "aug_inconsistency": probe_res["aug_inconsistency"],
                        "probe_available": probe_res["probe_available"],
                        "source_norm_context": probe_res["source_norm_context"],
                        "bn_control_norm_context": probe_res["bn_control_norm_context"],
                        "candidate_norm_context": probe_res["candidate_norm_context"],
                        # Real Pair Comparative
                        "src_mean_c2_real": probe_res["src_mean_c2_real"],
                        "bn_mean_c2_real": probe_res["bn_mean_c2_real"],
                        "cand_mean_c2_real": probe_res["cand_mean_c2_real"],
                        "disagree_src_bn_real": probe_res["disagree_src_bn_real"],
                        "disagree_bn_cand_real": probe_res["disagree_bn_cand_real"],
                        "disagree_src_cand_real": probe_res["disagree_src_cand_real"],
                        "diff_c2_src_bn_real": probe_res["diff_c2_src_bn_real"],
                        "diff_c2_bn_cand_real": probe_res["diff_c2_bn_cand_real"],
                        "diff_c2_src_cand_real": probe_res["diff_c2_src_cand_real"],
                        # Null Pre2 Comparative
                        "src_mean_c2_pre2": probe_res["src_mean_c2_pre2"],
                        "bn_mean_c2_pre2": probe_res["bn_mean_c2_pre2"],
                        "cand_mean_c2_pre2": probe_res["cand_mean_c2_pre2"],
                        "disagree_src_bn_pre2": probe_res["disagree_src_bn_pre2"],
                        "disagree_bn_cand_pre2": probe_res["disagree_bn_cand_pre2"],
                        "disagree_src_cand_pre2": probe_res["disagree_src_cand_pre2"],
                        "diff_c2_src_bn_pre2": probe_res["diff_c2_src_bn_pre2"],
                        "diff_c2_bn_cand_pre2": probe_res["diff_c2_bn_cand_pre2"],
                        "diff_c2_src_cand_pre2": probe_res["diff_c2_src_cand_pre2"],
                        # Null Post Comparative
                        "src_mean_c2_post": probe_res["src_mean_c2_post"],
                        "bn_mean_c2_post": probe_res["bn_mean_c2_post"],
                        "cand_mean_c2_post": probe_res["cand_mean_c2_post"],
                        "disagree_src_bn_post": probe_res["disagree_src_bn_post"],
                        "disagree_bn_cand_post": probe_res["disagree_bn_cand_post"],
                        "disagree_src_cand_post": probe_res["disagree_src_cand_post"],
                        "diff_c2_src_bn_post": probe_res["diff_c2_src_bn_post"],
                        "diff_c2_bn_cand_post": probe_res["diff_c2_bn_cand_post"],
                        "diff_c2_src_cand_post": probe_res["diff_c2_src_cand_post"],
                    }

                    # Step 8: Perform Candidate Adaptation Update (Weak-Strong Consistency)
                    optimizer.zero_grad()
                    # Weak pass
                    weak_logits = cand_model(xA, xB)
                    weak_probs = F.softmax(weak_logits, dim=1)
                    weak_conf, weak_pseudo = weak_probs.max(dim=1)

                    # Strong pass (horizontal flip)
                    xA_strong = torch.flip(xA, dims=[-1])
                    xB_strong = torch.flip(xB, dims=[-1])
                    strong_logits = cand_model(xA_strong, xB_strong)
                    strong_aligned_logits = torch.flip(strong_logits, dims=[-1])

                    # Active selection mask: valid & conf >= tau
                    v_active = (valid_mask == 1) & (weak_conf >= self.threshold)
                    sel_count = int(v_active.sum().item())

                    if sel_count > 0:
                        cons_loss = F.cross_entropy(
                            strong_aligned_logits.permute(0, 2, 3, 1)[v_active],
                            weak_pseudo[v_active],
                        )
                        cons_loss.backward()
                        grad_norm = compute_gradient_l2_norm(trainable_params)
                        optimizer.step()
                        step_loss = float(cons_loss.item())
                        update_applied = True
                        action_str = "update"
                    else:
                        step_loss = 0.0
                        grad_norm = 0.0
                        update_applied = False
                        action_str = "skip"

                    # Step 9: Offline Ground-Truth Evaluation Post-Persistence
                    # Defense-in-depth: explicitly verify shard exists on disk before unlocking labels
                    if not shard_path.is_file():
                        raise RuntimeError(
                            f"Critical persistence violation: shard {shard_path} is missing from disk before label unlock!"
                        )
                    batch.mark_persisted()
                    gt_labels = batch.get_labels().cpu().numpy()
                    v_mask_np = valid_mask.cpu().numpy()

                    # Free batch array memory immediately to bound RAM to single-batch
                    del batch_arrays

                    for s_i in range(len(s_ids)):
                        eval_mask = (v_mask_np[s_i] == 1) & (gt_labels[s_i] != 3)
                        p_c = cand_pred_real[s_i]
                        l_c = gt_labels[s_i]

                        tp = int(((p_c == 2) & (l_c == 2) & eval_mask).sum())
                        fp = int(((p_c == 2) & (l_c != 2) & eval_mask).sum())
                        fn = int(((p_c != 2) & (l_c == 2) & eval_mask).sum())
                        tn = int(((p_c != 2) & (l_c != 2) & eval_mask).sum())

                        event_tp += tp
                        event_fp += fp
                        event_fn += fn
                        event_tn += tn
                        event_valid += int(eval_mask.sum())

                    # Step 10: Finalize batch timing and record step telemetry
                    batch_total_elapsed_ms = (time.time() - b_start_time) * 1000.0
                    peak_vram = (
                        float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
                        if self.device.type == "cuda"
                        else None
                    )

                    recorder.record_step(
                        step_index=b_idx,
                        event_id=eid,
                        condition_id="candidate_e2_conf90",
                        batch_size=len(s_ids),
                        sample_ids=s_ids,
                        action=action_str,
                        update_applied=update_applied,
                        total_loss=step_loss,
                        consistency_loss=step_loss,
                        gradient_norm_l2=grad_norm,
                        bn_weight_displacement=probe_res["bn_weight_displacement"],
                        bn_bias_displacement=probe_res["bn_bias_displacement"],
                        valid_pixels=probe_res["valid_pixels"],
                        selected_pixels=sel_count,
                        selected_pixel_fraction=(
                            float(sel_count / probe_res["valid_pixels"])
                            if probe_res["valid_pixels"] > 0
                            else 0.0
                        ),
                        mean_entropy=probe_res["mean_entropy"],
                        mean_max_confidence=probe_res["mean_max_confidence"],
                        null_score_s_null=probe_res["s_null"],
                        batch_time_ms=batch_total_elapsed_ms,
                        batch_total_elapsed_ms=batch_total_elapsed_ms,
                        prelabel_evidence_elapsed_ms=prelabel_evidence_elapsed_ms,
                        peak_vram_mb=peak_vram,
                        extra_telemetry=extra_telem,
                    )

                    # Flatten step record for summary CSV
                    csv_row = {
                        "event_id": eid,
                        "step_index": b_idx,
                        "batch_size": len(s_ids),
                        "valid_pixels": probe_res["valid_pixels"],
                        "prelabel_evidence_elapsed_ms": prelabel_evidence_elapsed_ms,
                        "batch_total_elapsed_ms": batch_total_elapsed_ms,
                        "s_null": probe_res["s_null"] if probe_res["s_null"] is not None else "",
                        "delta_null_pre2": probe_res["delta_null_pre2"] if probe_res["delta_null_pre2"] is not None else "",
                        "delta_null_post": probe_res["delta_null_post"] if probe_res["delta_null_post"] is not None else "",
                        "s_flood_expansion": probe_res["s_flood_expansion"] if probe_res["s_flood_expansion"] is not None else "",
                        "s_flood_disappearance": probe_res["s_flood_disappearance"] if probe_res["s_flood_disappearance"] is not None else "",
                        "mean_entropy": probe_res["mean_entropy"] if probe_res["mean_entropy"] is not None else "",
                        "mean_max_confidence": probe_res["mean_max_confidence"] if probe_res["mean_max_confidence"] is not None else "",
                        "source_to_candidate_kl": probe_res["source_to_candidate_kl"] if probe_res["source_to_candidate_kl"] is not None else "",
                        "hard_disagreement_rate": probe_res["hard_disagreement_rate"] if probe_res["hard_disagreement_rate"] is not None else "",
                        "delta_flood_pixels": probe_res["delta_flood_pixels"],
                        "abs_delta_flood_pixels": probe_res["abs_delta_flood_pixels"],
                        "aug_inconsistency": probe_res["aug_inconsistency"] if probe_res["aug_inconsistency"] is not None else "",
                        "bn_weight_displacement": probe_res["bn_weight_displacement"] if probe_res["bn_weight_displacement"] is not None else "",
                        "bn_bias_displacement": probe_res["bn_bias_displacement"] if probe_res["bn_bias_displacement"] is not None else "",
                        "probe_available": probe_res["probe_available"],
                        "update_applied": update_applied,
                        "step_loss": step_loss,
                        "grad_norm": grad_norm,
                        # BN Diagnostic Comparisons: Real Pair
                        "src_mean_c2_real": probe_res["src_mean_c2_real"] if probe_res["src_mean_c2_real"] is not None else "",
                        "bn_mean_c2_real": probe_res["bn_mean_c2_real"] if probe_res["bn_mean_c2_real"] is not None else "",
                        "cand_mean_c2_real": probe_res["cand_mean_c2_real"] if probe_res["cand_mean_c2_real"] is not None else "",
                        "disagree_src_bn_real": probe_res["disagree_src_bn_real"] if probe_res["disagree_src_bn_real"] is not None else "",
                        "disagree_bn_cand_real": probe_res["disagree_bn_cand_real"] if probe_res["disagree_bn_cand_real"] is not None else "",
                        "disagree_src_cand_real": probe_res["disagree_src_cand_real"] if probe_res["disagree_src_cand_real"] is not None else "",
                        "diff_c2_src_bn_real": probe_res["diff_c2_src_bn_real"] if probe_res["diff_c2_src_bn_real"] is not None else "",
                        "diff_c2_bn_cand_real": probe_res["diff_c2_bn_cand_real"] if probe_res["diff_c2_bn_cand_real"] is not None else "",
                        "diff_c2_src_cand_real": probe_res["diff_c2_src_cand_real"] if probe_res["diff_c2_src_cand_real"] is not None else "",
                        # BN Diagnostic Comparisons: Null Pre2 Pair
                        "src_mean_c2_pre2": probe_res["src_mean_c2_pre2"] if probe_res["src_mean_c2_pre2"] is not None else "",
                        "bn_mean_c2_pre2": probe_res["bn_mean_c2_pre2"] if probe_res["bn_mean_c2_pre2"] is not None else "",
                        "cand_mean_c2_pre2": probe_res["cand_mean_c2_pre2"] if probe_res["cand_mean_c2_pre2"] is not None else "",
                        "disagree_src_bn_pre2": probe_res["disagree_src_bn_pre2"] if probe_res["disagree_src_bn_pre2"] is not None else "",
                        "disagree_bn_cand_pre2": probe_res["disagree_bn_cand_pre2"] if probe_res["disagree_bn_cand_pre2"] is not None else "",
                        "disagree_src_cand_pre2": probe_res["disagree_src_cand_pre2"] if probe_res["disagree_src_cand_pre2"] is not None else "",
                        "diff_c2_src_bn_pre2": probe_res["diff_c2_src_bn_pre2"] if probe_res["diff_c2_src_bn_pre2"] is not None else "",
                        "diff_c2_bn_cand_pre2": probe_res["diff_c2_bn_cand_pre2"] if probe_res["diff_c2_bn_cand_pre2"] is not None else "",
                        "diff_c2_src_cand_pre2": probe_res["diff_c2_src_cand_pre2"] if probe_res["diff_c2_src_cand_pre2"] is not None else "",
                        # BN Diagnostic Comparisons: Null Post Pair
                        "src_mean_c2_post": probe_res["src_mean_c2_post"] if probe_res["src_mean_c2_post"] is not None else "",
                        "bn_mean_c2_post": probe_res["bn_mean_c2_post"] if probe_res["bn_mean_c2_post"] is not None else "",
                        "cand_mean_c2_post": probe_res["cand_mean_c2_post"] if probe_res["cand_mean_c2_post"] is not None else "",
                        "disagree_src_bn_post": probe_res["disagree_src_bn_post"] if probe_res["disagree_src_bn_post"] is not None else "",
                        "disagree_bn_cand_post": probe_res["disagree_bn_cand_post"] if probe_res["disagree_bn_cand_post"] is not None else "",
                        "disagree_src_cand_post": probe_res["disagree_src_cand_post"] if probe_res["disagree_src_cand_post"] is not None else "",
                        "diff_c2_src_bn_post": probe_res["diff_c2_src_bn_post"] if probe_res["diff_c2_src_bn_post"] is not None else "",
                        "diff_c2_bn_cand_post": probe_res["diff_c2_bn_cand_post"] if probe_res["diff_c2_bn_cand_post"] is not None else "",
                        "diff_c2_src_cand_post": probe_res["diff_c2_src_cand_post"] if probe_res["diff_c2_src_cand_post"] is not None else "",
                    }
                    all_step_csv_rows.append(csv_row)

                # End of event in recorder
                precision = float(event_tp / max(1, event_tp + event_fp))
                recall = float(event_tp / max(1, event_tp + event_fn))
                flood_f1 = (
                    float(2 * precision * recall / (precision + recall))
                    if (precision + recall) > 0
                    else 0.0
                )

                ev_metric_dict = {
                    "event_id": eid,
                    "tp": event_tp,
                    "fp": event_fp,
                    "fn": event_fn,
                    "tn": event_tn,
                    "valid_pixels": event_valid,
                    "flood_precision": precision,
                    "flood_recall": recall,
                    "flood_f1": flood_f1,
                }
                # Comprehensive fail-closed coverage checks before complete_event()
                # 1. Processed count equals expected count
                if len(processed_event_sample_ids) != expected_event_sample_count:
                    raise RuntimeError(
                        f"Coverage count mismatch for Event {eid}: processed {len(processed_event_sample_ids)} "
                        f"samples, expected {expected_event_sample_count}!"
                    )

                # 2. Processed IDs have no duplicates
                if len(set(processed_event_sample_ids)) != len(processed_event_sample_ids):
                    raise RuntimeError(
                        f"Coverage duplicate violation for Event {eid}: processed stream contains duplicate sample IDs!"
                    )

                # 3. Processed ordered IDs equal initial manifest's ordered IDs for this event
                if processed_event_sample_ids != expected_event_sample_ids:
                    raise RuntimeError(
                        f"Coverage sequence violation for Event {eid}: processed sample sequence does not match "
                        f"initial manifest expected sequence!"
                    )

                # 4. Evidence-index IDs match the same ordered list
                evidence_index_sids: List[str] = []
                for s_info in recorder.current_shards_index:
                    evidence_index_sids.extend(s_info["sample_ids"])
                if evidence_index_sids != expected_event_sample_ids:
                    raise RuntimeError(
                        f"Coverage evidence-index mismatch for Event {eid}: shard sample sequence does not match "
                        f"initial manifest expected sequence!"
                    )

                # 5. Pass frozen expected_event_sample_count and configured convenience aggregate flag to complete_event
                recorder.complete_event(
                    eid,
                    ev_metric_dict,
                    expected_sample_count=expected_event_sample_count,
                    produce_convenience_aggregate=self.produce_convenience_aggregate,
                )
                event_metrics_all.append(ev_metric_dict)

                print(
                    f"Completed Event {eid}: Valid Pixels={event_valid}, Flood F1={flood_f1:.4f}"
                )
        except Exception as e:
            recorder.record_failure(str(e))
            raise

        # Write Step Risk Summary CSV
        if all_step_csv_rows:
            csv_path = self.output_dir / "e5_step_risk_summary.csv"
            fieldnames = list(all_step_csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_step_csv_rows)

        # Write Run Manifest and completion status
        elapsed_sec = time.time() - start_time
        macro_f1 = float(np.mean([m["flood_f1"] for m in event_metrics_all])) if event_metrics_all else 0.0

        manifest_data = {
            "work_package": "E5",
            "title": "Temporal Risk Evidence & Generic-Risk Evaluation",
            "governing_document": "docs/MASTER_EXECUTION_PLAN_V3.md",
            "protocol_version": "e5_temporal_risk_v1",
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_sec,
            "total_batches_processed": total_batches_processed,
            "preflight_provenance": preflight_info,
            "event_metrics": event_metrics_all,
            "macro_flood_f1": macro_f1,
            "status": "COMPLETED_SMOKE" if self.is_smoke else "COMPLETED_STREAM",
        }
        with open(self.output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        recorder.complete_run(
            condition_summary={
                "macro_flood_f1": macro_f1,
                "elapsed_seconds": elapsed_sec,
                "total_batches_processed": total_batches_processed,
                "status": manifest_data["status"],
            }
        )

        print(f"\n>>> Streaming evaluation complete in {elapsed_sec:.1f}s. Outputs saved to {self.output_dir}")
        return manifest_data


def main():
    parser = argparse.ArgumentParser(
        description="ChangeGuard Work Package E5: Temporal Risk Evidence Runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e5_temporal_risk_v1.json",
        help="Path to E5 protocol JSON config",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Explicit output directory override",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to execute on (cuda or cpu)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bounded quarantined smoke on Event 520 (head 32 samples)",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run preflight checks and assertions only without executing model passes",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Explicit run ID override",
    )
    parser.add_argument(
        "--checkpoint_seed",
        type=int,
        default=None,
        help="Source checkpoint seed override (0 or 1)",
    )
    parser.add_argument(
        "--stream_order",
        type=str,
        default=None,
        choices=["order_a", "order_b"],
        help="Stream arrival order override (order_a or order_b)",
    )
    parser.add_argument(
        "--produce_convenience_aggregate",
        action="store_true",
        default=None,
        help="Force generation of convenience monolithic frozen_predictions.npz",
    )
    parser.add_argument(
        "--no_convenience_aggregate",
        action="store_true",
        help="Disable generation of convenience monolithic frozen_predictions.npz for bounded memory",
    )
    parser.add_argument(
        "--allow_dirty",
        action="store_true",
        help="Allow running with uncommitted Git modifications (for development/validation only)",
    )
    args = parser.parse_args()

    produce_aggregate = None
    if args.no_convenience_aggregate:
        produce_aggregate = False
    elif args.produce_convenience_aggregate:
        produce_aggregate = True

    runner = E5RiskScoringRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        checkpoint_seed=args.checkpoint_seed,
        stream_order=args.stream_order,
        produce_convenience_aggregate=produce_aggregate,
        device=args.device,
        is_smoke=args.smoke,
        allow_dirty=args.allow_dirty,
    )

    if args.validate_only:
        print("Running preflight validation checks...")
        preflight = runner.run_preflight_checks()
        print("Preflight validation successful:")
        print(json.dumps(preflight, indent=2))
        sys.exit(0)

    runner.execute_streaming_evaluation()


if __name__ == "__main__":
    main()
