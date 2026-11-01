"""
ChangeGuard Week 9 Stronger Test-Time Adaptation (TTA) Matrix Runner.

Governing Protocol: configs/evaluation/week9_stronger_tta_protocol.json
Document Specification: docs/week9_stronger_tta_protocol.md

Evaluates:
- EATA (ICML 2022): Reliable entropy filtering + cosine redundancy filtering + Fisher regularization.
- SAR-TTA (ICLR 2023 Oral): Reliable entropy filtering + two-stage SAM + model recovery.

Strict Scientific & Integrity Guarantees:
1. Label-Free Adaptation:
   - Adaptation dataloader loads strictly sensory inputs (xA, xB) and sensory observation masks (MK0_MNA via valid_mask).
   - Ground truth annotations (MK0_MLU) are NEVER loaded or accessed during adaptation.
2. Separate Offline Evaluation:
   - Double-masking validity (valid_mask == 1 and MK0_MLU != 3) is applied strictly offline
     on frozen, cached predictions after adaptation concludes.
3. 24 Planned New Physical Streams:
   - 12 EATA streams (3 seeds x 2 orders x 2 reset modes).
   - 12 SAR streams (3 seeds x 2 orders x 2 reset modes).
4. Explicit Reference to Week 8 Controls:
   - Matching Week 8 Source-Only (12 cells from 6 physical), AdaBN (12 cells), and Tent (12 cells)
     are referenced/imported directly without claiming they were newly executed.
5. Numerical Safeguards & Non-Destructive Logging:
   - Verifies finite outputs, tracks filtering/update/recovery counters.
   - Refuses accidental overwrites; preserves initial manifest and incremental partial progress.
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
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.evaluation_dataset import KuroEvaluationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.eata_adapter import EATAAdapter
from src.tta.sar_adapter import SARAdapter


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


class StrongerTTAMatrixRunner:
    """Orchestrates the 24 new physical streams for Week 9 stronger TTA baselines."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda:0",
        allow_overwrite: bool = False,
        synthetic_data: bool = False,
    ):
        self.config_path = config_path.resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.explicit_output_dir = output_dir.resolve() if output_dir is not None else None
        self.output_dir: Optional[Path] = self.explicit_output_dir
        self.log_dir = (REPO_ROOT / self.cfg.get("outputs", {}).get("log_dir", "logs/week9_stronger_tta")).resolve()

        self.device = device
        self.allow_overwrite = allow_overwrite
        self.synthetic_data = synthetic_data

        d_cfg = self.cfg["dataset"]
        self.target_events = d_cfg["target_events"]
        self.expected_counts = {int(k): v for k, v in d_cfg["expected_event_counts"].items()}
        self.expected_total_samples = d_cfg["expected_total_samples"]
        self.forbidden_events = d_cfg["forbidden_sealed_test_events"]
        self.batch_size = self.cfg.get("stream_execution_policy", {}).get("batch_size", 16)

    def validate_preflight(self, is_smoke: bool = False, fail_closed: bool = True) -> Dict[str, Any]:
        """Audits configuration, event disjointness, checkpoints, and Fisher matrices.

        Fails closed on any discrepancy: missing files, hash mismatches, or invalid artifacts.
        """
        intersection = set(self.target_events) & set(self.forbidden_events)
        if intersection:
            raise RuntimeError(
                f"PREFLIGHT FATAL: Target events overlap with sealed test events: {sorted(list(intersection))}"
            )

        target_seeds = [0] if is_smoke else [0, 1, 2]
        checkpoint_status = {}
        for s_id in [0, 1, 2]:
            s_str = str(s_id)
            s_info = self.cfg["source_checkpoints"]["seeds"][s_str]
            ckpt_path = REPO_ROOT / s_info["checkpoint_path"]
            exp_hash = s_info["expected_sha256"]

            if not ckpt_path.is_file():
                checkpoint_status[s_str] = {"exists": False, "path": str(ckpt_path)}
                if fail_closed and not self.synthetic_data and s_id in target_seeds:
                    raise FileNotFoundError(f"Checkpoint for seed {s_id} missing at {ckpt_path}")
                continue

            act_hash = compute_file_sha256(ckpt_path)
            hash_match = (act_hash == exp_hash)
            checkpoint_status[s_str] = {
                "exists": True,
                "path": str(ckpt_path),
                "hash_match": hash_match,
                "expected": exp_hash,
                "actual": act_hash,
            }
            if fail_closed and not self.synthetic_data and s_id in target_seeds and not hash_match:
                raise ValueError(
                    f"Checkpoint seed {s_id} hash mismatch: actual {act_hash} vs expected {exp_hash}"
                )

        # Check Week 8 controls reference
        w8_cfg = self.cfg["matrix_specification"].get("reused_baseline_controls", {})
        w8_csv = REPO_ROOT / w8_cfg.get("week8_metrics_csv", "")
        w8_exists = w8_csv.is_file()
        w8_hash_match = False
        if w8_exists:
            w8_hash_match = (compute_file_sha256(w8_csv) == w8_cfg.get("expected_sha256"))

        w8_status = {
            "exists": w8_exists,
            "path": str(w8_csv),
            "hash_match": w8_hash_match,
        }

        # Check EATA Fisher artifacts
        fisher_status = {}
        fisher_paths = self.cfg.get("fisher_information_protocol", {}).get("artifact_paths", {})
        for s_id in [0, 1, 2]:
            s_str = str(s_id)
            f_rel = fisher_paths.get(s_str)
            if f_rel is None:
                fisher_status[s_str] = {"configured": False}
                continue
            f_path = (REPO_ROOT / f_rel).resolve()
            meta_path = f_path.with_suffix(".json")
            f_exists = f_path.is_file()
            m_exists = meta_path.is_file()
            f_valid = False
            error_msg = None

            if f_exists:
                try:
                    loaded = torch.load(f_path, map_location="cpu", weights_only=False)
                    f_dict = loaded.get("fisher", loaded) if isinstance(loaded, dict) else loaded
                    if not isinstance(f_dict, dict) or len(f_dict) == 0:
                        raise ValueError(f"Fisher artifact {f_path} is empty or not a dict")

                    # Validate sidecar metadata (mandatory for all Fisher artifacts)
                    if not m_exists:
                        raise FileNotFoundError(
                            f"Missing required Fisher sidecar metadata at {meta_path}. "
                            "Every Fisher artifact must have an accompanying .json sidecar."
                        )
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    if meta.get("seed") != s_id:
                        raise ValueError(f"Fisher metadata seed mismatch: {meta.get('seed')} vs {s_id}")
                    if meta.get("artifact_sha256") and meta.get("artifact_sha256") != compute_file_sha256(f_path):
                        raise ValueError("Fisher artifact SHA-256 does not match sidecar metadata")
                    if not self.synthetic_data:
                        exp_ckpt_sha = self.cfg.get("source_checkpoints", {}).get("seeds", {}).get(s_str, {}).get("expected_sha256")
                        if exp_ckpt_sha and meta.get("checkpoint_sha256") and meta.get("checkpoint_sha256") != exp_ckpt_sha:
                            raise ValueError(f"Fisher checkpoint SHA-256 mismatch: {meta.get('checkpoint_sha256')} vs {exp_ckpt_sha}")

                    # Validate split config hash against canonical configuration
                    exp_split_sha = self.cfg.get("source_data_configuration", {}).get("split_config_sha256")
                    if not exp_split_sha:
                        raise ValueError("Missing canonical 'split_config_sha256' in protocol 'source_data_configuration'")
                    if meta.get("split_config_sha256") != exp_split_sha:
                        raise ValueError(f"Fisher split SHA-256 mismatch: {meta.get('split_config_sha256')} vs {exp_split_sha}")

                    # W9-N03: Validate manifest hash, total candidates, and expected selection digest if configured
                    exp_manifest_sha = self.cfg.get("source_data_configuration", {}).get("manifest_sha256")
                    if exp_manifest_sha and meta.get("manifest_sha256") and meta.get("manifest_sha256") != exp_manifest_sha:
                        raise ValueError(f"Fisher manifest SHA-256 mismatch: {meta.get('manifest_sha256')} vs {exp_manifest_sha}")

                    is_v2_protocol = (self.cfg.get("protocol_version") == "week9_stronger_tta_v2")
                    exp_total_patches = self.cfg.get("source_data_configuration", {}).get("total_source_patches")
                    sel_meta_dict = meta.get("selection_metadata") if isinstance(meta.get("selection_metadata"), dict) else meta
                    if is_v2_protocol and exp_total_patches is not None and sel_meta_dict.get("total_source_candidates") is not None:
                        if sel_meta_dict.get("total_source_candidates") != exp_total_patches:
                            raise ValueError(
                                f"Fisher candidate population mismatch: artifact has {sel_meta_dict.get('total_source_candidates')} "
                                f"vs protocol configured {exp_total_patches}"
                            )

                    exp_sel_sha = self.cfg.get("fisher_information_protocol", {}).get("selected_sample_ids_sha256")
                    if exp_sel_sha and meta.get("selected_sample_ids_sha256") != exp_sel_sha:
                        raise ValueError(
                            f"Fisher selection digest mismatch against protocol: {meta.get('selected_sample_ids_sha256')} vs {exp_sel_sha}"
                        )

                    # Validate selected sample IDs and deterministic digest
                    selected_ids = meta.get("selected_sample_ids")
                    if not isinstance(selected_ids, list) or len(selected_ids) == 0:
                        raise ValueError("Fisher sidecar metadata missing valid 'selected_sample_ids' list")
                    expected_ids_digest = hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest()
                    if meta.get("selected_sample_ids_sha256") != expected_ids_digest:
                        raise ValueError("Fisher metadata selected_sample_ids_sha256 does not match selected_sample_ids content")

                    # Validate declared estimator, loss reduction, and BN context
                    sel_meta = meta.get("selection_metadata") if isinstance(meta.get("selection_metadata"), dict) else meta
                    estimator = sel_meta.get("estimator")
                    if estimator not in {"per_patch_squared_gradient", "exact_per_sample"}:
                        raise ValueError(f"Unapproved Fisher estimator in metadata: {estimator}")

                    loss_red = sel_meta.get("loss_reduction")
                    if loss_red is not None and loss_red != "mean_over_valid_pixels":
                        raise ValueError(f"Unapproved Fisher loss reduction in metadata: {loss_red}")

                    bn_ctx = sel_meta.get("bn_context")
                    if bn_ctx is not None and bn_ctx != "eval_mode_frozen_running_stats":
                        raise ValueError(f"Unapproved Fisher BN context in metadata: {bn_ctx}")

                    # W9-R04: Require and strictly validate producer payload metadata bound to sidecar and configuration
                    if not isinstance(loaded, dict) or "metadata" not in loaded or not isinstance(loaded["metadata"], dict):
                        raise ValueError(f"Fisher artifact {f_path} missing required 'metadata' dictionary")
                    p_meta = loaded["metadata"]

                    p_seed = p_meta.get("checkpoint_seed")
                    if p_seed is None:
                        p_seed = p_meta.get("seed")
                    if p_seed is None:
                        raise ValueError(f"Fisher payload metadata missing 'checkpoint_seed' in {f_path}")
                    if p_seed != s_id:
                        raise ValueError(f"Fisher payload metadata seed mismatch: {p_seed} vs requested {s_id}")
                    if meta.get("seed") is not None and p_seed != meta.get("seed"):
                        raise ValueError(f"Fisher payload seed ({p_seed}) does not match sidecar seed ({meta.get('seed')})")

                    p_ckpt_sha = p_meta.get("checkpoint_sha256")
                    if not p_ckpt_sha:
                        raise ValueError(f"Fisher payload metadata missing 'checkpoint_sha256' in {f_path}")
                    if meta.get("checkpoint_sha256") and p_ckpt_sha != meta.get("checkpoint_sha256"):
                        raise ValueError(f"Fisher payload checkpoint SHA-256 ({p_ckpt_sha}) does not match sidecar ({meta.get('checkpoint_sha256')})")
                    if not self.synthetic_data:
                        exp_ckpt_sha = self.cfg.get("source_checkpoints", {}).get("seeds", {}).get(s_str, {}).get("expected_sha256")
                        if exp_ckpt_sha and p_ckpt_sha != exp_ckpt_sha:
                            raise ValueError(f"Fisher payload checkpoint SHA-256 ({p_ckpt_sha}) does not match configured expected SHA-256 ({exp_ckpt_sha})")

                    p_sel_meta = p_meta.get("selection_metadata")
                    if not isinstance(p_sel_meta, dict):
                        raise ValueError(f"Fisher payload metadata missing 'selection_metadata' dictionary in {f_path}")

                    p_ids_sha = p_sel_meta.get("selected_sample_ids_sha256")
                    if not p_ids_sha:
                        raise ValueError(f"Fisher payload selection_metadata missing 'selected_sample_ids_sha256' in {f_path}")
                    if p_ids_sha != expected_ids_digest:
                        raise ValueError(f"Fisher payload selection_metadata digest ({p_ids_sha}) does not match sidecar ({expected_ids_digest})")

                    p_ids = p_sel_meta.get("selected_sample_ids")
                    if p_ids is not None:
                        if not isinstance(p_ids, list):
                            raise ValueError(f"Fisher payload selection_metadata 'selected_sample_ids' must be a list in {f_path}")
                        if p_ids != selected_ids:
                            raise ValueError("Fisher payload selected_sample_ids list does not match sidecar selected_sample_ids")
                        p_calc_sha = hashlib.sha256(json.dumps(p_ids).encode()).hexdigest()
                        if p_ids_sha != p_calc_sha:
                            raise ValueError("Fisher payload selected_sample_ids content does not match its selected_sample_ids_sha256")

                    p_split_sha = p_sel_meta.get("split_config_sha256")
                    if not p_split_sha:
                        raise ValueError(f"Fisher payload selection_metadata missing 'split_config_sha256' in {f_path}")
                    if p_split_sha != exp_split_sha:
                        raise ValueError(f"Fisher payload split SHA-256 ({p_split_sha}) does not match canonical split SHA-256 ({exp_split_sha})")
                    if meta.get("split_config_sha256") and p_split_sha != meta.get("split_config_sha256"):
                        raise ValueError("Fisher payload split SHA-256 does not match sidecar split SHA-256")

                    if exp_manifest_sha and p_sel_meta.get("manifest_sha256") and p_sel_meta.get("manifest_sha256") != exp_manifest_sha:
                        raise ValueError(f"Fisher payload manifest SHA-256 mismatch: {p_sel_meta.get('manifest_sha256')} vs {exp_manifest_sha}")

                    if is_v2_protocol and exp_total_patches is not None and p_sel_meta.get("total_source_candidates") is not None:
                        if p_sel_meta.get("total_source_candidates") != exp_total_patches:
                            raise ValueError(
                                f"Fisher payload candidate population mismatch: artifact has {p_sel_meta.get('total_source_candidates')} "
                                f"vs protocol configured {exp_total_patches}"
                            )

                    p_estimator = p_sel_meta.get("estimator")
                    if p_estimator not in {"per_patch_squared_gradient", "exact_per_sample"}:
                        raise ValueError(f"Unapproved Fisher estimator in payload selection_metadata: {p_estimator}")

                    p_loss_red = p_sel_meta.get("loss_reduction")
                    if p_loss_red is not None and p_loss_red != "mean_over_valid_pixels":
                        raise ValueError(f"Unapproved Fisher loss reduction in payload selection_metadata: {p_loss_red}")

                    p_bn_ctx = p_sel_meta.get("bn_context")
                    if p_bn_ctx is not None and p_bn_ctx != "eval_mode_frozen_running_stats":
                        raise ValueError(f"Unapproved Fisher BN context in payload selection_metadata: {p_bn_ctx}")

                    # Determine expected BN parameters for this seed
                    expected_bn_params = {}
                    if self.synthetic_data:
                        mock_m = MockSNUNet()
                        for m_name, mod in mock_m.named_modules():
                            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                for p_name, param in mod.named_parameters():
                                    if p_name in ["weight", "bias"]:
                                        expected_bn_params[f"{m_name}.{p_name}"] = param
                    else:
                        ckpt_p = REPO_ROOT / self.cfg["source_checkpoints"]["seeds"][s_str]["checkpoint_path"]
                        if ckpt_p.is_file():
                            ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                            st = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))
                            for k, v in st.items():
                                clean_k = k[7:] if k.startswith("module.") else k
                                parts = clean_k.split(".")
                                if len(parts) >= 2 and parts[-1] in ["weight", "bias"] and any(p.startswith("bn") for p in parts):
                                    expected_bn_params[clean_k] = v

                    if expected_bn_params:
                        if set(f_dict.keys()) != set(expected_bn_params.keys()):
                            if self.synthetic_data:
                                # In synthetic data mode, real KuroSiwo Fisher artifacts on disk are not applicable to MockSNUNet
                                fisher_status[s_str] = {
                                    "configured": True,
                                    "exists": True,
                                    "valid": True,
                                    "synthetic_bypassed": True,
                                }
                                continue
                            raise ValueError(
                                f"Fisher parameter keys {set(f_dict.keys())} do not match model BN parameters {set(expected_bn_params.keys())}"
                            )

                    for name, item in f_dict.items():
                        if not isinstance(item, (tuple, list)) or len(item) != 2:
                            raise ValueError(f"Fisher entry for {name} must be a (f_diag, p_init) tuple")
                        f_diag, p_init = item
                        if not torch.all(torch.isfinite(f_diag)) or not torch.all(f_diag >= 0.0):
                            raise ValueError(f"Fisher diagonal for {name} contains non-finite or negative values")
                        if not torch.all(torch.isfinite(p_init)):
                            raise ValueError(f"Source anchor parameter for {name} contains non-finite values")
                        if expected_bn_params and name in expected_bn_params:
                            exp_p = expected_bn_params[name]
                            if f_diag.shape != exp_p.shape or p_init.shape != exp_p.shape:
                                raise ValueError(f"Shape mismatch for {name}: Fisher {f_diag.shape} vs expected {exp_p.shape}")
                            if not self.synthetic_data and not torch.equal(p_init, exp_p.cpu()):
                                raise ValueError(f"Source anchor parameter for {name} does not match checkpoint parameter values")

                    f_valid = True
                except Exception as exc:
                    error_msg = str(exc)

            fisher_status[s_str] = {
                "exists": f_exists,
                "path": str(f_path),
                "valid": f_valid,
                "error": error_msg,
            }

            if fail_closed and s_id in target_seeds:
                if "eata" in self.cfg["matrix_specification"]["methods"]:
                    if not self.synthetic_data and not f_exists:
                        raise FileNotFoundError(
                            f"Required Fisher artifact for seed {s_id} not found at {f_path}. "
                            "EATA adaptation requires precomputed source Fisher information."
                        )
                    if f_exists and not f_valid:
                        raise ValueError(
                            f"Fisher artifact for seed {s_id} at {f_path} is invalid: {error_msg}"
                        )

        return {
            "protocol_version": self.cfg.get("protocol_version"),
            "status": self.cfg.get("status"),
            "target_events": self.target_events,
            "forbidden_events": self.forbidden_events,
            "checkpoint_status": checkpoint_status,
            "week8_controls_status": w8_status,
            "fisher_status": fisher_status,
        }

    def instantiate_adapter(
        self,
        method: str,
        checkpoint_seed: int,
        checkpoint_path: Path,
    ) -> Any:
        """Instantiates EATA or SAR adapter for the given method and checkpoint."""
        m_cfg = self.cfg["model"]
        if self.synthetic_data:
            wrapper = MockSNUNet()
        else:
            wrapper, _ = SNUNetWrapper.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                in_channels=m_cfg["in_channels"],
                out_ch=m_cfg["num_classes"],
                base_channel=m_cfg["base_channel"],
                device=self.device,
            )
        wrapper.to(self.device)

        h_cfg = self.cfg["method_hyperparameters"].get(method, {})

        if method == "eata":
            fishers = None
            fisher_paths = self.cfg.get("fisher_information_protocol", {}).get("artifact_paths", {})
            f_rel = fisher_paths.get(str(checkpoint_seed))
            if f_rel is not None:
                f_path = REPO_ROOT / f_rel
                if f_path.is_file():
                    loaded = torch.load(f_path, map_location=self.device, weights_only=False)
                    fishers = loaded.get("fisher", loaded) if isinstance(loaded, dict) else loaded

            if fishers is None and not self.synthetic_data:
                raise ValueError(
                    f"EATA adaptation strictly requires precomputed Fisher information. "
                    f"Artifact missing or failed to load for seed {checkpoint_seed}: {f_rel}"
                )

            if fishers is not None:
                expected_keys = {
                    f"{m_name}.{p_name}"
                    for m_name, mod in wrapper.named_modules()
                    if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d))
                    for p_name, param in mod.named_parameters()
                    if p_name in ["weight", "bias"]
                }
                if set(fishers.keys()) != expected_keys:
                    if self.synthetic_data:
                        fishers = None
                    else:
                        raise ValueError(
                            f"Fisher parameter keys {set(fishers.keys())} do not match model BN parameters {expected_keys}"
                        )

            return EATAAdapter(
                model=wrapper,
                checkpoint_path=str(checkpoint_path),
                fishers=fishers,
                fisher_alpha=h_cfg.get("fisher_alpha", 2000.0),
                e_margin=h_cfg.get("reliability_threshold_e0"),
                d_margin=h_cfg.get("redundancy_threshold_d_margin", 0.05),
                learning_rate=h_cfg.get("learning_rate", 1.25e-4),
                momentum=h_cfg.get("momentum", 0.9),
                weight_decay=h_cfg.get("weight_decay", 0.0),
                device=self.device,
            )
        elif method == "sar":
            return SARAdapter(
                model=wrapper,
                checkpoint_path=str(checkpoint_path),
                margin_e0=h_cfg.get("reliability_threshold_e0"),
                reset_constant_em=h_cfg.get("recovery_threshold_em"),
                sam_rho=h_cfg.get("sam_rho", 0.05),
                learning_rate=h_cfg.get("learning_rate", 1.25e-4),
                momentum=h_cfg.get("momentum", 0.9),
                weight_decay=h_cfg.get("weight_decay", 0.0),
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported method for Week 9: {method}")

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

        # Group dataset samples by event attaching dataset_index
        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(dataset.samples):
            eid = int(item["event_id"])
            if events is not None and eid not in events:
                continue
            item_with_idx = dict(item)
            item_with_idx["dataset_index"] = idx
            event_to_samples.setdefault(eid, []).append(item_with_idx)

        ordered_stream: List[Tuple[int, List[Dict[str, Any]]]] = []
        for eid in target_seq:
            samples = event_to_samples.get(eid, [])
            if within_sort == "deterministic_sorted_sample_id_ascending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))
            elif within_sort == "deterministic_sorted_sample_id_descending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)
            else:
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))

            if max_samples_per_event is not None and max_samples_per_event > 0:
                samples = samples[:max_samples_per_event]

            ordered_stream.append((eid, samples))

        return ordered_stream

    def build_matrix_streams(self, is_smoke: bool = False) -> List[Dict[str, Any]]:
        """Constructs the manifest of streams (4 streams for smoke, 24 for formal)."""
        methods = self.cfg["matrix_specification"]["methods"]
        seeds = [0] if is_smoke else self.cfg["matrix_specification"]["checkpoint_seeds"]
        orders = ["order_a"] if is_smoke else ["order_a", "order_b"]
        reset_modes = ["reset_per_event", "continual"]
        smoke_events = self.cfg.get("smoke_test_specification", {}).get("events", [437, 1111003])

        streams = []
        for method in methods:
            for seed in seeds:
                for order in orders:
                    for reset_mode in reset_modes:
                        stream_id = f"stream_{method}_s{seed}_{order}_{reset_mode}"
                        order_cfg = self.cfg["matrix_specification"]["stream_orders"][order]
                        event_sequence = smoke_events if is_smoke else order_cfg["event_sequence"]

                        ckpt_info = self.cfg["source_checkpoints"]["seeds"][str(seed)]
                        streams.append({
                            "stream_id": stream_id,
                            "method": method,
                            "checkpoint_seed": seed,
                            "checkpoint_path": ckpt_info["checkpoint_path"],
                            "stream_order": order,
                            "event_sequence": event_sequence,
                            "reset_mode": reset_mode,
                            "is_physical_run": True,
                        })

        return streams

    def run_physical_stream(
        self,
        stream_spec: Dict[str, Any],
        adaptation_dataset: KuroEvaluationDataset,
        offline_eval_dataset: KuroEvaluationDataset,
        is_smoke: bool = False,
        event_callback: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Executes a single adaptation stream and returns event rows and batch diagnostics."""
        method = stream_spec["method"]
        seed = stream_spec["checkpoint_seed"]
        order = stream_spec["stream_order"]
        reset_mode = stream_spec["reset_mode"]
        event_sequence = stream_spec["event_sequence"]
        ckpt_path = REPO_ROOT / stream_spec["checkpoint_path"]

        adapter = self.instantiate_adapter(method, seed, ckpt_path)

        # In continual mode, reset model to source once at stream start and move to device
        if reset_mode == "continual":
            adapter.reset_to_source()

        ordered_stream = self.get_ordered_samples_for_stream(
            dataset=adaptation_dataset,
            order_id=order,
            events=event_sequence,
            max_samples_per_event=19 if is_smoke else None,
        )

        stream_event_results = []
        stream_batch_diagnostics = []
        total_samples_evaluated = 0

        for event_idx, (event_id, sample_items) in enumerate(ordered_stream, start=1):
            # Completeness assertion: empty events are strictly forbidden
            if len(sample_items) == 0:
                raise ValueError(
                    f"Completeness violation: Event {event_id} has 0 samples in stream {stream_spec['stream_id']}!"
                )

            # W9-R05: Validate sample ID uniqueness
            s_ids = [str(s["sample_id"]) for s in sample_items]
            if len(s_ids) != len(set(s_ids)):
                raise ValueError(
                    f"Integrity violation: Event {event_id} has duplicate sample IDs in stream {stream_spec['stream_id']}: "
                    f"total {len(s_ids)} vs unique {len(set(s_ids))}"
                )

            if not is_smoke and not self.synthetic_data:
                expected_count = self.expected_counts.get(event_id)
                if len(sample_items) != expected_count:
                    raise ValueError(
                        f"Completeness violation: Event {event_id} sample count {len(sample_items)} "
                        f"does not match expected {expected_count}!"
                    )
            elif is_smoke:
                if len(sample_items) != 19:
                    raise ValueError(
                        f"Smoke test violation: Event {event_id} has {len(sample_items)} samples (expected 19)!"
                    )

            if reset_mode == "reset_per_event":
                adapter.reset_to_source()

            cached_predictions = {}
            event_start_time = time.time()
            current_event_diagnostics = []

            # Batch processing loop
            batch_size = self.batch_size if not self.synthetic_data else 4
            num_samples = len(sample_items)

            for i in range(0, num_samples, batch_size):
                batch_items = sample_items[i : i + batch_size]
                b_ids = [s["sample_id"] for s in batch_items]
                b_indices = [s["dataset_index"] for s in batch_items]

                # Load sensory data via standard KuroEvaluationDataset __getitem__
                batch_tensors = [adaptation_dataset[idx] for idx in b_indices]
                xA = torch.stack([d["xA"] for d in batch_tensors]).to(self.device)
                xB = torch.stack([d["xB"] for d in batch_tensors]).to(self.device)
                m = torch.stack([d["valid_mask"] for d in batch_tensors]).to(self.device)

                # Pure label-isolated forward/adapt
                step_res = adapter.adapt_and_predict(xA, xB, sar_valid_mask=m, sample_ids=b_ids)
                preds = step_res["predicted_class"].cpu().numpy()

                for idx, s_id in enumerate(b_ids):
                    s_id_str = str(s_id)
                    if s_id_str in cached_predictions:
                        raise ValueError(f"Integrity violation: duplicate prediction generated for {s_id_str} in event {event_id}!")
                    cached_predictions[s_id_str] = preds[idx]

                # Collect batch diagnostics
                diag = step_res.get("step_diagnostics", {})
                diag_record = {
                    "stream_id": stream_spec["stream_id"],
                    "event_id": event_id,
                    "batch_start": i,
                    "batch_size": len(batch_items),
                    **diag,
                }
                stream_batch_diagnostics.append(diag_record)
                current_event_diagnostics.append(diag_record)

            event_elapsed = time.time() - event_start_time
            total_samples_evaluated += len(sample_items)

            # Offline Evaluation strictly on cached predictions
            metrics = self._evaluate_offline(cached_predictions, sample_items, offline_eval_dataset)

            event_row = {
                "stream_id": stream_spec["stream_id"],
                "method": method,
                "checkpoint_seed": seed,
                "stream_order": order,
                "reset_mode": reset_mode,
                "event_id": event_id,
                "event_position": event_idx,
                "sample_count": len(sample_items),
                "elapsed_seconds": round(event_elapsed, 2),
                "flood_f1": metrics["flood_f1"],
                "flood_iou": metrics["flood_iou"],
                "flood_precision": metrics["flood_precision"],
                "flood_recall": metrics["flood_recall"],
                "valid_pixels": metrics["valid_pixels"],
                "num_adaptation_steps": getattr(adapter, "num_adaptation_steps", 0),
                "num_reliable_samples": getattr(adapter, "num_reliable_samples", 0),
                "num_active_samples": getattr(adapter, "num_active_samples", 0),
                "num_redundant_samples": getattr(adapter, "num_redundant_samples", 0),
                "num_zero_valid_patches": getattr(adapter, "num_zero_valid_patches", 0),
                "num_recoveries": getattr(adapter, "num_recoveries", 0),
                "is_physical_run": True,
                "provenance": "executed",
            }
            stream_event_results.append(event_row)

            # W9-R09: Flush completed event row and diagnostics immediately
            if event_callback is not None:
                event_callback(event_row, current_event_diagnostics)

        expected_total = 38 if is_smoke else (self.expected_total_samples if not self.synthetic_data else len(ordered_stream[0][1]) * len(ordered_stream))
        if not self.synthetic_data and total_samples_evaluated != expected_total:
            raise ValueError(
                f"Stream total sample coverage mismatch: evaluated {total_samples_evaluated} vs expected {expected_total}"
            )

        return stream_event_results, stream_batch_diagnostics

    def _evaluate_offline(
        self,
        predictions: Dict[str, np.ndarray],
        sample_items: List[Dict[str, Any]],
        offline_dataset: Any,
    ) -> Dict[str, Any]:
        """Offline scoring with double-masking validity: valid_mask == 1 & MK0_MLU != 3."""
        pred_keys = set(predictions.keys())
        expected_keys = set(str(item["sample_id"]) for item in sample_items)
        if pred_keys != expected_keys:
            raise ValueError(f"Offline evaluation key mismatch: {len(pred_keys)} predictions vs {len(expected_keys)} expected samples")

        tp, fp, fn, tn = 0, 0, 0, 0
        total_valid = 0

        for item in sample_items:
            s_id = str(item["sample_id"])
            pred = predictions[s_id]
            sample = offline_dataset[item["dataset_index"]]

            gt = sample["label"].numpy() if torch.is_tensor(sample["label"]) else sample["label"]
            mna = sample["valid_mask"].squeeze().numpy() if torch.is_tensor(sample["valid_mask"]) else sample["valid_mask"]

            valid = (mna == 1) & (gt != 3)
            total_valid += int(np.sum(valid))

            p_flood = (pred == 2) & valid
            g_flood = (gt == 2) & valid

            tp += int(np.sum(p_flood & g_flood))
            fp += int(np.sum(p_flood & ~g_flood))
            fn += int(np.sum(~p_flood & g_flood))
            tn += int(np.sum(~p_flood & ~g_flood & valid))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            "flood_f1": round(float(f1), 4),
            "flood_iou": round(float(iou), 4),
            "flood_precision": round(float(precision), 4),
            "flood_recall": round(float(recall), 4),
            "valid_pixels": total_valid,
        }

    def run_matrix(self, is_smoke: bool = False) -> Dict[str, Any]:
        """Executes the full 24-stream matrix or 4-stream smoke matrix."""
        # 1. Determine output directory
        if self.explicit_output_dir is not None:
            self.output_dir = self.explicit_output_dir
        else:
            rel_dir = self.cfg["outputs"].get("smoke_dir" if is_smoke else "formal_dir")
            self.output_dir = (REPO_ROOT / rel_dir).resolve()

        # 2. Overwrite protection
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not is_smoke:
                raise FileExistsError(
                    f"CRITICAL SAFETY VIOLATION: Formal output directory '{self.output_dir}' "
                    "already exists and is non-empty. Overwriting formal runs is strictly prohibited."
                )
            elif not self.allow_overwrite:
                raise FileExistsError(
                    f"Output directory '{self.output_dir}' already exists and is non-empty. "
                    "Use --allow_overwrite to explicitly proceed for smoke runs."
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 3. Preflight check with fail-closed guarantee
        preflight = self.validate_preflight(is_smoke=is_smoke, fail_closed=True)

        streams = self.build_matrix_streams(is_smoke=is_smoke)
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        git_info = get_git_info()

        # 4. Pre-freeze initial manifest before executing heavy workloads
        manifest_path = self.output_dir / "week9_tta_manifest.json"
        csv_path = self.output_dir / "stream_event_metrics.csv"
        diag_path = self.output_dir / "batch_diagnostics.jsonl"

        initial_manifest = {
            "status": "running",
            "start_timestamp_utc": start_iso,
            "git_info": git_info,
            "is_smoke": is_smoke,
            "is_formal_run": not is_smoke,
            "total_planned_streams": len(streams),
            "planned_stream_ids": [s["stream_id"] for s in streams],
            "output_directory": str(self.output_dir),
            "config_path": str(self.config_path),
            "config_sha256": compute_file_sha256(self.config_path),
            "command_args": list(sys.argv),
            "runner_settings": {
                "batch_size": self.batch_size,
                "device": self.device,
                "allow_overwrite": self.allow_overwrite,
                "synthetic_data": self.synthetic_data,
            },
            "preflight_summary": {
                "checkpoints": preflight["checkpoint_status"],
                "fishers": preflight["fisher_status"],
                "week8_controls": preflight.get("week8_controls_status"),
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(initial_manifest, f, indent=2)

        print(f"=== Starting Week 9 Matrix Execution ({'SMOKE (Diagnostic)' if is_smoke else 'FORMAL'}) ===")
        print(f"Planned Streams:  {len(streams)}")
        print(f"Output Directory: {self.output_dir}")

        # 5. Incremental Persistence Setup & Dataset Loading inside try...except
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = None
        diag_file = open(diag_path, "w", encoding="utf-8")

        all_results = []
        all_diagnostics = []
        start_time = time.time()
        s_id = "initialization"

        def on_event_completed(row: Dict[str, Any], diags: List[Dict[str, Any]]) -> None:
            nonlocal csv_writer
            all_results.append(row)
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()

            for d in diags:
                all_diagnostics.append(d)
                diag_file.write(json.dumps(d) + "\n")
            diag_file.flush()

        try:
            # Load datasets
            if not self.synthetic_data:
                events_to_load = self.cfg.get("smoke_test_specification", {}).get("events", [437, 1111003]) if is_smoke else self.target_events
                d_cfg = self.cfg["dataset"]
                adapt_ds = KuroEvaluationDataset(
                    dataset_root=Path(d_cfg["dataset_root"]),
                    manifest_path=Path(d_cfg["manifest_path"]),
                    allowed_events=events_to_load,
                    canonical_preevent=self.cfg["model"]["canonical_preevent"],
                    sar_clamp_max=d_cfg["sar_clamp_max"],
                    sar_mean=d_cfg["sar_mean"],
                    sar_std=d_cfg["sar_std"],
                    load_labels=False,  # STRICT LABEL ISOLATION
                )
                eval_ds = KuroEvaluationDataset(
                    dataset_root=Path(d_cfg["dataset_root"]),
                    manifest_path=Path(d_cfg["manifest_path"]),
                    allowed_events=events_to_load,
                    canonical_preevent=self.cfg["model"]["canonical_preevent"],
                    sar_clamp_max=d_cfg["sar_clamp_max"],
                    sar_mean=d_cfg["sar_mean"],
                    sar_std=d_cfg["sar_std"],
                    load_labels=True,   # Strictly offline scoring
                )
            else:
                # Synthetic mock dataset conforming to KuroEvaluationDataset interface
                target_evs = [437, 1111003] if is_smoke else self.target_events
                mock_records = []
                for eid in target_evs:
                    cnt = 19 if is_smoke else 17
                    for i in range(cnt):
                        mock_records.append({
                            "sample_id": f"patch_{eid}_{i:04d}",
                            "event_id": eid,
                            "xA": torch.randn(2, 32, 32),
                            "xB": torch.randn(2, 32, 32),
                            "valid_mask": torch.ones(32, 32, dtype=torch.uint8),
                            "label": torch.zeros((32, 32), dtype=torch.long),
                        })
                adapt_ds = KuroEvaluationDataset.from_mock_records(mock_records, allowed_events=target_evs)
                eval_ds = KuroEvaluationDataset.from_mock_records(mock_records, allowed_events=target_evs)

            # 6. Stream Execution with incremental event-level persistence
            for s_idx, stream_spec in enumerate(streams, start=1):
                s_id = stream_spec["stream_id"]
                print(f"[{s_idx}/{len(streams)}] Executing {s_id} ...")

                self.run_physical_stream(
                    stream_spec=stream_spec,
                    adaptation_dataset=adapt_ds,
                    offline_eval_dataset=eval_ds,
                    is_smoke=is_smoke,
                    event_callback=on_event_completed,
                )

            if not csv_file.closed:
                csv_file.close()
            if not diag_file.closed:
                diag_file.close()

            elapsed = time.time() - start_time
            end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # 7. Final completion validation
            s_id = "final_validation"
            expected_streams = len(streams)
            expected_rows = expected_streams * (2 if is_smoke else 6)
            if len(all_results) != expected_rows:
                raise ValueError(
                    f"Integrity check failed: Produced {len(all_results)} event rows vs expected {expected_rows}!"
                )

            summary_payload = {
                "status": "completed",
                "is_smoke": is_smoke,
                "is_formal_run": not is_smoke,
                "total_physical_streams": expected_streams,
                "total_event_rows": len(all_results),
                "elapsed_seconds": round(elapsed, 2),
                "start_timestamp_utc": start_iso,
                "end_timestamp_utc": end_iso,
                "output_csv": str(csv_path),
                "output_diagnostics": str(diag_path),
                "reused_controls": "Referencing Week 8 formal_v1 outputs without re-execution",
            }

            with open(self.output_dir / "stream_matrix_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)

            final_manifest = {
                **initial_manifest,
                "status": "completed",
                "end_timestamp_utc": end_iso,
                "elapsed_seconds": round(elapsed, 2),
                "total_physical_streams": expected_streams,
                "total_event_rows": len(all_results),
                "summary": summary_payload,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(final_manifest, f, indent=2)

            print(f"Matrix run successfully completed in {elapsed:.1f}s. Saved to {self.output_dir}")
            return final_manifest

        except (Exception, KeyboardInterrupt) as exc:
            if not csv_file.closed:
                csv_file.close()
            if not diag_file.closed:
                diag_file.close()

            # Record failure state immediately in manifest, preserving initial run context
            failure_manifest = {
                **initial_manifest,
                "status": "failed",
                "failure_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "failed_stream": s_id,
                "error": str(exc) if str(exc) else type(exc).__name__,
                "failure_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "completed_event_rows": len(all_results),
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(failure_manifest, f, indent=2)
            raise


class MockSNUNet(nn.Module):
    """Tiny mock model with BN for synthetic runner tests."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChangeGuard Week 9 Stronger TTA Matrix Runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/week9_stronger_tta_protocol.json",
        help="Path to Week 9 protocol JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for metrics and artifacts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bounded diagnostic smoke test (4 streams).",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Perform preflight validation and exit.",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting non-empty output directory (permitted ONLY for smoke runs).",
    )
    parser.add_argument(
        "--synthetic_data",
        action="store_true",
        help="Run on synthetic data for CPU integration testing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional adaptation batch size check. Must match canonical protocol setting (16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g. cuda:0 or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    runner = StrongerTTAMatrixRunner(
        config_path=config_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
        allow_overwrite=args.allow_overwrite,
        synthetic_data=args.synthetic_data,
    )

    # Reject conflicting formal CLI overrides rather than silently changing the frozen setting
    if args.batch_size is not None and args.batch_size != runner.batch_size and not runner.synthetic_data:
        raise ValueError(
            f"Conflicting batch size override rejected: CLI specified --batch_size={args.batch_size}, "
            f"but protocol config freezes canonical adaptation batch_size={runner.batch_size}. "
            "Formal execution strictly prohibits overriding frozen protocol settings."
        )

    # Formal execution strictly forbids --allow_overwrite
    if not args.smoke and args.allow_overwrite and not runner.synthetic_data:
        raise ValueError(
            "Overwriting formal run outputs is strictly prohibited. "
            "--allow_overwrite is only permitted with diagnostic --smoke runs."
        )

    if args.validate_only:
        preflight = runner.validate_preflight(is_smoke=args.smoke, fail_closed=False)
        print("=== Week 9 Stronger TTA Preflight Audit ===")
        print(json.dumps(preflight, indent=2))
        print("Preflight validation complete.")
        return

    runner.run_matrix(is_smoke=args.smoke)


if __name__ == "__main__":
    main()
