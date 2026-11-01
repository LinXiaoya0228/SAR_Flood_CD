#!/usr/bin/env python3
"""ChangeGuard Work Package E8: Confirmation Matrix Orchestration Runner.

Orchestrates the frozen Plan V3 Work Package E8 confirmation matrix:
- Reuses verified completed cells for Source-Only (3 streams), BN-No-Update (6 streams),
  AdaBN (12 streams), Tent (12 streams), EATA (12 streams), and SAR (12 streams).
- Executes new cells for CoTTA Continual (6 streams) and CoTTA Reset (6 streams).
- Total: 69 physical streams across 3 seeds (0, 1, 2), 2 stream orders (Order A, Order B),
  and 6 development events (414 event-result rows total).

Features:
- Fail-closed preflight validation (--validate_only).
- Dry-run matrix audit (--dry_run).
- Single-stream execution (--stream_id).
- Safe resumability: already completed streams with verified manifests are skipped.
- Strict isolation: zero ground-truth label leakage during adaptation.
- Quarantined sealed-test partition: immediate fatal error on sealed-test access.
- Double-masking offline evaluation: (MK0_MNA == 1) & (MK0_MLU != 3).
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    E1MatchedBNNoUpdateAdapter,
    E1SARAdapter,
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
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably finalize a directory entry after an atomic replace."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON through fsync + atomic replace without exposing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    """Write a compressed NPZ shard through fsync + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@dataclass(frozen=True)
class DeferredBatchSpec:
    """Metadata-only batch plan; deliberately contains no GT tensor."""

    event_id: int
    sample_ids: Tuple[str, ...]
    dataset_indices: Tuple[int, ...]


class DeferredLabelLoader:
    """Loads GT only when explicitly invoked after a prediction shard is durable."""

    def __init__(self, offline_dataset: KuroEvaluationDataset):
        self.dataset = offline_dataset
        self.index_by_sample_id = {
            str(item["sample_id"]): idx for idx, item in enumerate(offline_dataset.samples)
        }
        if len(self.index_by_sample_id) != len(offline_dataset.samples):
            raise ValueError("Offline dataset contains duplicate sample IDs.")
        self.index_count = 0
        self.read_batches: List[Tuple[str, ...]] = []

    def load_exact(self, sample_ids: Sequence[str], event_id: int) -> torch.Tensor:
        """Read and validate exactly the requested labels, preserving request order."""
        requested = tuple(str(s) for s in sample_ids)
        labels: List[torch.Tensor] = []
        observed: List[str] = []
        for sample_id in requested:
            if sample_id not in self.index_by_sample_id:
                raise KeyError(f"Offline label sample ID not found: {sample_id}")
            record = self.dataset[self.index_by_sample_id[sample_id]]
            self.index_count += 1
            observed_id = str(record["sample_id"])
            observed_event = int(record["event_id"])
            if observed_id != sample_id or observed_event != int(event_id):
                raise RuntimeError(
                    "Deferred label identity mismatch: "
                    f"requested ({event_id}, {sample_id}), observed ({observed_event}, {observed_id})"
                )
            observed.append(observed_id)
            labels.append(record["label"] if torch.is_tensor(record["label"]) else torch.as_tensor(record["label"]))
        if tuple(observed) != requested:
            raise RuntimeError("Deferred label order does not match persisted prediction order.")
        self.read_batches.append(requested)
        return torch.stack(labels).long()


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


class E8ConfirmationRunner:
    """Orchestrates Work Package E8 confirmation matrix execution and reuse."""

    def __init__(
        self,
        config_path: Path,
        ledger_path: Optional[Path] = None,
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

        self.ledger_path = (
            ledger_path.resolve()
            if ledger_path is not None
            else (REPO_ROOT / self.cfg.get("execution_ledger_path", "configs/changeguard/e8_execution_ledger_v1.json")).resolve()
        )
        if not self.ledger_path.is_file():
            raise FileNotFoundError(f"Execution ledger not found: {self.ledger_path}")

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

        self.device_str = device if torch.cuda.is_available() else "cpu"

        # Output directory configuration
        if output_dir is not None:
            self.output_root = output_dir.resolve()
        else:
            self.output_root = (REPO_ROOT / self.cfg.get("output_directory", "outputs/e8_confirmation")).resolve()

        # Dataset configuration
        ds_cfg = self.cfg["dataset"]
        self.dataset_root = (REPO_ROOT / ds_cfg["dataset_root"]).resolve()
        self.manifest_path = (REPO_ROOT / ds_cfg["manifest_path"]).resolve()
        self.split_path = (REPO_ROOT / ds_cfg["split_path"]).resolve()
        self.batch_size = int(ds_cfg.get("batch_size", 16))

        self.target_events = [int(e) for e in ds_cfg.get("development_events", DEFAULT_TTA_DEV_EVENTS)]
        self.sar_clamp_max = float(ds_cfg.get("sar_clamp_max", 0.15))
        self.sar_mean = list(ds_cfg.get("sar_mean", [0.0953, 0.0264]))
        self.sar_std = list(ds_cfg.get("sar_std", [0.0427, 0.0215]))
        self.canonical_preevent = str(ds_cfg.get("canonical_preevent", "pre_event_2"))

        # Cached datasets and stream plans
        self._adapt_dataset: Optional[KuroEvaluationDataset] = None
        self._offline_dataset: Optional[KuroEvaluationDataset] = None
        self._deferred_label_loader: Optional[DeferredLabelLoader] = None
        self._cached_stream_plans: Dict[str, List[Tuple[int, List[DeferredBatchSpec]]]] = {}
        self._preflight_info: Optional[Dict[str, Any]] = None

    def planned_cells(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Resolve explicit v1 ledgers or rule-based v2 ledgers into physical cells."""
        if "source_ledger_path" not in self.ledger:
            return list(self.ledger.get("reused_cells", [])), list(self.ledger.get("execute_new_cells", []))

        source_path = (REPO_ROOT / self.ledger["source_ledger_path"]).resolve()
        with open(source_path, "r", encoding="utf-8") as f:
            source_ledger = json.load(f)

        certified_ids = set(self.ledger["certification_plan"]["certified_reuse_stream_ids"])
        source_reused = list(source_ledger.get("reused_cells", []))
        source_new = list(source_ledger.get("execute_new_cells", []))
        known_ids = {c["stream_id"] for c in source_reused + source_new}
        if not certified_ids.issubset(known_ids):
            raise ValueError(f"Unknown certified reuse IDs: {sorted(certified_ids - known_ids)}")

        reused: List[Dict[str, Any]] = []
        execute_new: List[Dict[str, Any]] = []
        classifications = self.ledger["certification_plan"]["stream_classifications"]
        for cell in source_reused:
            sid = cell["stream_id"]
            classification = classifications.get(sid, self.ledger["certification_plan"]["default_historical_classification"])
            if sid in certified_ids:
                copied = copy.deepcopy(cell)
                copied["status"] = "REUSE_CERTIFIED"
                copied["certification_status"] = "CERTIFIED_POST_PERSISTENCE_GT_READ"
                reused.append(copied)
            else:
                copied = copy.deepcopy(cell)
                copied["status"] = "EXECUTE_NEW"
                copied["completion_status"] = "PENDING_EXECUTION"
                copied["target_output_directory"] = f"outputs/e8_confirmation_v2/{sid}"
                copied["v1_certification_status"] = classification
                copied.pop("artifact_directory", None)
                execute_new.append(copied)

        for cell in source_new:
            copied = copy.deepcopy(cell)
            copied["status"] = "EXECUTE_NEW"
            copied["completion_status"] = "PENDING_EXECUTION"
            copied["target_output_directory"] = f"outputs/e8_confirmation_v2/{copied['stream_id']}"
            copied["v1_certification_status"] = "COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ"
            execute_new.append(copied)

        if len(reused) != int(self.ledger["matrix_cell_summary"]["certified_reused_physical_streams"]):
            raise ValueError("Resolved certified reuse count does not match v2 ledger.")
        if len(execute_new) != int(self.ledger["matrix_cell_summary"]["execute_new_physical_streams"]):
            raise ValueError("Resolved v2 rerun count does not match v2 ledger.")
        return reused, execute_new

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _resolve_contract_path(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

    def validate_authorized_matrix_contract(self) -> None:
        """Validate the immutable, human-authorized E8 v2 execution envelope."""
        if "authorized_scope" not in self.cfg:
            return

        auth = self.cfg.get("execution_authorizations", {})
        if self.cfg.get("status") != "AUTHORIZED_PENDING_SMOKE_V3":
            raise PermissionError("E8 v2 config is not in its authorized pending-smoke state.")
        if self.ledger.get("status") != "AUTHORIZED_PENDING_SMOKE_V3":
            raise PermissionError("E8 v2 ledger is not in its authorized pending-smoke state.")
        if auth.get("real_data_execution_authorized") is not True or auth.get("preparation_only") is not False:
            raise PermissionError("E8 v2 real-data execution is not authorized by the committed config.")
        if auth.get("formal_launch_is_conditional_on_verified_smoke_v3") is not True:
            raise PermissionError("E8 v2 must remain conditional on verified smoke v3 evidence.")
        if self.ledger.get("v2_execution", {}).get("launch_authorized") is not True:
            raise PermissionError("E8 v2 formal launch is not authorized by the committed ledger.")
        if self.ledger.get("v2_execution", {}).get("authorization_is_not_sufficient_without_smoke") is not True:
            raise PermissionError("Ledger does not enforce the mandatory smoke-first gate.")

        expected = {
            "certified_reused_physical_streams": 2,
            "execute_new_physical_streams": 67,
            "total_physical_streams": 69,
            "total_event_rows": 414,
        }
        scope = self.cfg["authorized_scope"]
        cfg_summary = self.cfg["matrix_cell_summary"]
        ledger_summary = self.ledger["matrix_cell_summary"]
        for key, value in expected.items():
            if int(scope.get(key, -1)) != value:
                raise ValueError(f"Authorized E8 scope mismatch for {key}: expected {value}.")
            if int(cfg_summary.get(key, -1)) != value:
                raise ValueError(f"E8 config matrix count mismatch for {key}: expected {value}.")
            if int(ledger_summary.get(key, -1)) != value:
                raise ValueError(f"E8 ledger matrix count mismatch for {key}: expected {value}.")

        expected_events = [279, 437, 520, 559, 1111003, 1111008]
        if list(scope.get("development_event_ids", [])) != expected_events:
            raise ValueError("Authorized development-event scope differs from the frozen six events.")
        if self.target_events != expected_events:
            raise ValueError("Configured development events differ from the authorized six events.")
        if set(self.target_events) & FORBIDDEN_SEALED_TEST_EVENTS:
            raise RuntimeError("Authorized matrix contains a sealed-test event.")
        if scope.get("source_checkpoint_seeds") != [0, 1, 2] or sorted(self.cfg["source_checkpoints"]["seeds"]) != ["0", "1", "2"]:
            raise ValueError("Authorized source-checkpoint seed scope must be exactly 0/1/2.")
        if scope.get("orders") != ["order_a", "order_b"] or list(self.cfg["stream_orders"]) != ["order_a", "order_b"]:
            raise ValueError("Authorized stream orders differ from frozen Order A/Order B.")

        reused, execute_new = self.planned_cells()
        if len(reused) != 2 or len(execute_new) != 67:
            raise ValueError("Resolved matrix must contain exactly 2 certified reuses and 67 execute-new streams.")
        if any(c.get("certification_status") != "CERTIFIED_POST_PERSISTENCE_GT_READ" for c in reused):
            raise ValueError("Every reused E8 v2 stream must carry certified post-persistence GT status.")
        if any(c.get("status") != "REUSE_CERTIFIED" for c in reused):
            raise ValueError("Every reused E8 v2 stream must be explicitly marked REUSE_CERTIFIED.")
        if any(c.get("status") != "EXECUTE_NEW" for c in execute_new):
            raise ValueError("Every non-reused E8 v2 stream must be explicitly marked EXECUTE_NEW.")
        if len({c["stream_id"] for c in reused + execute_new}) != 69:
            raise ValueError("Resolved E8 v2 stream IDs are not exactly 69 unique identities.")
        if {c["reset_mode"] for c in reused + execute_new} != set(scope.get("reset_modes", [])):
            raise ValueError("Resolved reset modes differ from the authorized frozen modes.")
        if any(c.get("events") != expected_events for c in reused + execute_new):
            raise ValueError("A resolved E8 v2 stream differs from the frozen six-event scope.")
        historical = [c for c in execute_new if c["method"] not in {"cotta_continual", "cotta_reset"}]
        cotta = [c for c in execute_new if c["method"] in {"cotta_continual", "cotta_reset"}]
        if len(historical) != 55 or any(
            c.get("v1_certification_status") != "GT_PRELOADED_BUT_GUARDED — NOT CERTIFIED" for c in historical
        ):
            raise ValueError("The 55 uncertified historical v1 attempts are not preserved accurately.")
        if len(cotta) != 12 or any(
            c.get("v1_certification_status")
            != "COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ"
            for c in cotta
        ):
            raise ValueError("The 12 uncertified CoTTA v1 attempts are not preserved accurately.")

        expected_output = (REPO_ROOT / "outputs/e8_confirmation_v2").resolve()
        expected_log = (REPO_ROOT / "logs/e8_confirmation/e8_confirmation_v2.log").resolve()
        if self._resolve_contract_path(self.cfg["retry_paths"]["output_directory"]) != expected_output:
            raise ValueError("E8 v2 output path differs from the authorized fresh path.")
        if self._resolve_contract_path(self.cfg["retry_paths"]["log_path"]) != expected_log:
            raise ValueError("E8 v2 log path differs from the authorized fresh path.")
        if self.cfg["authorized_scope"].get("overwrite_allowed") is not False:
            raise PermissionError("E8 v2 overwrite prohibition is not active.")

    def validate_smoke_launch(self, smoke_config_path: Path, require_fresh_output: bool = True) -> Dict[str, Any]:
        """Authorize only the frozen Event-520/seed-0/two-batch smoke attempt."""
        smoke_path = smoke_config_path.resolve()
        smoke = self._load_json(smoke_path)
        if smoke.get("status") != "AUTHORIZED_NOT_EXECUTED":
            raise PermissionError("E7 smoke v3 is not in its authorized unexecuted state.")
        if smoke.get("launch_authorized") is not True:
            raise PermissionError("E7 smoke v3 is not authorized.")
        if smoke.get("overwrite_allowed") is not False or smoke.get("authorization", {}).get("no_overwrite") is not True:
            raise PermissionError("E7 smoke v3 overwrite prohibition is not active.")
        scope = smoke.get("authorization", {}).get("scope", {})
        expected_scope = {
            "partition": "development_only",
            "event_ids": [520],
            "seed": 0,
            "stream_id": "cotta_reset_s0_oa",
            "expected_batches": 2,
            "expected_samples": 32,
            "sealed_test_access": 0,
            "scientific_evidence": False,
        }
        if scope != expected_scope:
            raise ValueError("E7 smoke v3 scope differs from the authorized bounded contract.")
        if smoke.get("event_ids") != [520] or int(smoke.get("max_samples", -1)) != 32:
            raise ValueError("E7 smoke v3 event/sample bound differs from the frozen contract.")
        if int(smoke.get("batch_size", -1)) != 16 or smoke.get("prediction_timing") != "PRE_UPDATE":
            raise ValueError("E7 smoke v3 batch size or prediction timing differs from the frozen contract.")
        if smoke.get("stream_id") != "cotta_reset_s0_oa" or smoke.get("sealed_test_access") != "FORBIDDEN":
            raise ValueError("E7 smoke v3 stream identity or sealed-test policy is invalid.")
        if smoke.get("config") != str(self.config_path.relative_to(REPO_ROOT)):
            raise ValueError("E7 smoke v3 points to a different E8 config.")
        if smoke.get("ledger") != str(self.ledger_path.relative_to(REPO_ROOT)):
            raise ValueError("E7 smoke v3 points to a different E8 ledger.")
        authorization = smoke["authorization"]
        if authorization.get("authorized_by") != "human_project_owner" or not authorization.get("authorized_at"):
            raise PermissionError("E7 smoke v3 lacks the human project-owner authorization record.")
        if self._resolve_contract_path(authorization["fresh_output_directory"]) != self._resolve_contract_path(
            smoke["output_directory"]
        ):
            raise ValueError("E7 smoke v3 authorized output path does not match its execution path.")
        if self._resolve_contract_path(authorization["fresh_log_path"]) != self._resolve_contract_path(smoke["log_path"]):
            raise ValueError("E7 smoke v3 authorized log path does not match its execution path.")

        self.validate_authorized_matrix_contract()
        expected_output = self._resolve_contract_path(smoke["output_directory"])
        if self.output_root != expected_output:
            raise ValueError("Smoke runner output root differs from the authorized v3 path.")
        if require_fresh_output and expected_output.exists():
            raise FileExistsError(f"Authorized smoke output path must be fresh: {expected_output}")
        git_info = get_git_info()
        if git_info["tracked_dirty"] or str(git_info["commit"]).startswith("UNKNOWN_ERROR"):
            raise RuntimeError("Authorized smoke requires committed code and a clean tracked worktree.")
        return smoke

    @staticmethod
    def _assert_finite_payload(value: Any, field: str = "payload") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                E8ConfirmationRunner._assert_finite_payload(child, f"{field}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                E8ConfirmationRunner._assert_finite_payload(child, f"{field}[{index}]")
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"Non-finite value in smoke evidence: {field}")

    def verify_smoke_completion(self) -> Dict[str, Any]:
        """Verify smoke v3 evidence and every hash before formal E8 execution."""
        dependency = self.ledger.get("mandatory_launch_dependency")
        if not dependency:
            raise PermissionError("E8 ledger has no mandatory smoke-v3 launch dependency.")
        smoke_path = self._resolve_contract_path(dependency["smoke_config_path"])
        smoke = self._load_json(smoke_path)
        stream_dir = self._resolve_contract_path(smoke["output_directory"]) / smoke["stream_id"]
        required_paths = {
            "initial": stream_dir / "initial_manifest.json",
            "completion": stream_dir / "completion_status.json",
            "manifest": stream_dir / "manifest.json",
            "summary": stream_dir / "stream_summary.json",
        }
        declared_paths = {
            "initial": self._resolve_contract_path(dependency["initial_manifest_artifact"]),
            "completion": self._resolve_contract_path(dependency["completion_artifact"]),
            "manifest": self._resolve_contract_path(dependency["final_manifest_artifact"]),
        }
        if any(declared_paths[key] != required_paths[key] for key in declared_paths):
            raise ValueError("Ledger smoke dependency artifact paths do not match the authorized smoke output.")
        missing = [str(path) for path in required_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("WAITING_FOR_SMOKE_V3: missing completion evidence: " + ", ".join(missing))

        initial = self._load_json(required_paths["initial"])
        completion = self._load_json(required_paths["completion"])
        manifest = self._load_json(required_paths["manifest"])
        summary = self._load_json(required_paths["summary"])
        initial_sha = compute_file_sha256(required_paths["initial"])
        for terminal in (completion, manifest):
            if terminal.get("status") != dependency["required_status"]:
                raise RuntimeError("Smoke v3 terminal status is not successful.")
            if terminal.get("initial_manifest_sha256") != initial_sha:
                raise ValueError("Smoke v3 terminal record does not reference the immutable initial manifest hash.")
            if int(terminal.get("total_samples", -1)) != int(dependency["required_samples"]):
                raise ValueError("Smoke v3 did not complete exactly 32 samples.")
            if int(terminal.get("total_batches", -1)) != int(dependency["required_batches"]):
                raise ValueError("Smoke v3 did not complete exactly two batches.")
            if terminal.get("completed_events") != dependency["required_event_ids"]:
                raise ValueError("Smoke v3 completed-event scope is invalid.")

        expected_hashes = {
            "config_sha256": compute_file_sha256(self.config_path),
            "ledger_sha256": compute_file_sha256(self.ledger_path),
            "runner_sha256": compute_file_sha256(Path(__file__).resolve()),
            "adapter_sha256": compute_file_sha256(REPO_ROOT / "src/changeguard/cotta_adapter.py"),
            "checkpoint_sha256": self.cfg["source_checkpoints"]["seeds"]["0"]["expected_sha256"],
            "split_sha256": self.cfg["dataset"]["split_expected_sha256"],
            "dataset_manifest_sha256": self.cfg["dataset"]["manifest_expected_sha256"],
            "execution_contract_sha256": compute_file_sha256(smoke_path),
        }
        for key, expected in expected_hashes.items():
            if initial.get(key) != expected:
                raise ValueError(f"Smoke v3 {key} does not match the currently authorized artifact.")
        if initial.get("event_ids") != [520] or initial.get("seed") != 0:
            raise ValueError("Smoke v3 initial manifest event or seed is outside authorization.")
        if initial.get("expected_total_samples") != 32 or initial.get("expected_total_batches") != 2:
            raise ValueError("Smoke v3 initial manifest does not freeze the 32-sample/two-batch scope.")
        if initial.get("gt_or_gt_derived_mask_entered_adaptation") is not False:
            raise RuntimeError("Smoke v3 does not prove adapter label isolation.")
        if initial.get("sealed_test_rejection_status") != "VERIFIED_ZERO_SEALED_EVENTS":
            raise RuntimeError("Smoke v3 does not prove zero sealed-test access.")
        if initial.get("git_tracked_dirty") is not False:
            raise RuntimeError("Smoke v3 was not executed from a clean tracked worktree.")
        try:
            initial_timestamp = datetime.datetime.fromisoformat(initial["start_timestamp_utc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Smoke v3 initial manifest has no valid pre-execution timestamp.") from exc

        verification = completion.get("label_io_verification", {})
        required_verification = {
            "zero_gt_reads_before_first_prediction_persistence": True,
            "all_label_reads_after_matching_prediction_persistence": True,
            "gt_or_gt_derived_mask_entered_adaptation": False,
            "sample_identity_verified": True,
            "sealed_test_access": 0,
            "finite_outputs_verified": True,
            "failed_persistence_count": 0,
        }
        if any(verification.get(key) != value for key, value in required_verification.items()):
            raise RuntimeError("Smoke v3 label-I/O verification did not pass every mandatory check.")
        if int(completion.get("deferred_gt_reads", -1)) != 32 or int(summary.get("deferred_gt_reads", -1)) != 32:
            raise ValueError("Smoke v3 deferred GT read accounting is not exactly 32.")

        audits = sorted(stream_dir.glob("event_520/label_io_audits/batch_*.json"))
        if len(audits) != 2:
            raise ValueError("Smoke v3 must contain exactly two label-I/O audit records.")
        audited_reads = 0
        for audit_path in audits:
            audit = self._load_json(audit_path)
            if audit.get("label_read_after_persistence") is not True or audit.get("sample_identity_match") is not True:
                raise RuntimeError("Smoke v3 contains a failed label-timing or identity audit.")
            try:
                persistence_timestamp = datetime.datetime.fromisoformat(audit["persistence_timestamp_utc"])
                label_timestamp = datetime.datetime.fromisoformat(audit["label_read_timestamp_utc"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Smoke v3 contains an invalid label-I/O timestamp.") from exc
            if not initial_timestamp < persistence_timestamp < label_timestamp:
                raise RuntimeError("Smoke timestamps do not prove manifest-before-persistence-before-label ordering.")
            shard = Path(audit["prediction_shard"])
            if not shard.is_file() or shard.stat().st_size == 0:
                raise FileNotFoundError("Smoke prediction shard is absent or empty before GT evidence.")
            if compute_file_sha256(shard) != audit.get("prediction_shard_sha256"):
                raise ValueError("Smoke prediction shard hash does not match its label-I/O audit.")
            audited_reads += int(audit.get("label_reads", 0))
        if audited_reads != 32:
            raise ValueError("Smoke label-I/O audits do not account for exactly 32 deferred reads.")
        if (stream_dir / "failure.json").exists():
            raise RuntimeError("Smoke v3 has a failure record and cannot authorize E8 v2.")
        self._assert_finite_payload(summary, "smoke_summary")
        return {
            "status": "SMOKE_V3_VERIFIED",
            "stream_dir": str(stream_dir),
            "initial_manifest_sha256": initial_sha,
            "samples": 32,
            "batches": 2,
        }

    def validate_formal_launch(self, require_fresh_output: bool = True) -> Dict[str, Any]:
        """Require authorization, exact counts, fresh output, and verified smoke evidence."""
        self.validate_authorized_matrix_contract()
        expected_output = self._resolve_contract_path(self.cfg["retry_paths"]["output_directory"])
        if self.output_root != expected_output:
            raise ValueError("Formal runner output root differs from the authorized E8 v2 path.")
        if require_fresh_output and expected_output.exists():
            raise FileExistsError(f"Authorized E8 v2 output path must be fresh: {expected_output}")
        git_info = get_git_info()
        if git_info["tracked_dirty"] or str(git_info["commit"]).startswith("UNKNOWN_ERROR"):
            raise RuntimeError("Formal E8 v2 requires committed code and a clean tracked worktree.")
        return self.verify_smoke_completion()

    def run_preflights(self) -> Dict[str, Any]:
        """Fail-closed preflight validation of all code, assets, partitions, and ledger."""
        print("=" * 80)
        print("ChangeGuard Plan V3 E8: Executing Fail-Closed Preflights...")
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

        fisher_info: Dict[int, Dict[str, str]] = {}
        for s_str, f_cfg in self.cfg.get("fisher_artifacts_v2", {}).get("seeds", {}).items():
            s_idx = int(s_str)
            p = (REPO_ROOT / f_cfg["path"]).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Fisher artifact for seed {s_idx} not found: {p}")
            actual_sha = compute_file_sha256(p)
            if actual_sha != f_cfg["expected_sha256"]:
                raise ValueError(f"Fisher seed {s_idx} SHA-256 mismatch: {actual_sha} != {f_cfg['expected_sha256']}")
            fisher_info[s_idx] = {"path": str(p), "sha256": actual_sha}

        # 4. Ledger validation
        reused_cells, execute_new_cells = self.planned_cells()
        expected_reused = int(self.ledger.get("matrix_cell_summary", {}).get("certified_reused_physical_streams", 57))
        expected_new = int(self.ledger.get("matrix_cell_summary", {}).get("execute_new_physical_streams", 12))
        if len(reused_cells) != expected_reused:
            raise ValueError(f"Expected exactly {expected_reused} reused cells, found {len(reused_cells)}")
        if len(execute_new_cells) != expected_new:
            raise ValueError(f"Expected exactly {expected_new} execute-new cells, found {len(execute_new_cells)}")

        # Verify all reused cells exist on disk with valid manifests or summaries
        verified_reused = 0
        for cell in reused_cells:
            art_dir = (REPO_ROOT / cell["artifact_directory"]).resolve()
            if not art_dir.is_dir():
                raise FileNotFoundError(f"Reused artifact directory not found: {art_dir}")
            m_path = art_dir / "manifest.json"
            cs_path = art_dir / "condition_summary.json"
            if not (m_path.is_file() or cs_path.is_file()):
                raise FileNotFoundError(f"Reused cell manifest or condition summary not found: {art_dir}")
            verified_reused += 1
        print(f"Ledger Reused Cells: VERIFIED ({verified_reused}/{expected_reused} on disk with manifests/summaries)")

        allowed_methods = set(self.cfg["evaluated_methods"])
        new_methods = set(c["method"] for c in execute_new_cells)
        if not new_methods.issubset(allowed_methods):
            raise ValueError(f"Unexpected execute-new methods: {sorted(new_methods - allowed_methods)}")
        print(f"Ledger Execute-New Cells: VERIFIED ({expected_new} versioned streams)")

        smoke_dependency_status = "NOT_APPLICABLE"
        if "authorized_scope" in self.cfg:
            self.validate_authorized_matrix_contract()
            try:
                self.verify_smoke_completion()
                smoke_dependency_status = "SMOKE_V3_VERIFIED"
                print("Formal launch dependency: VERIFIED (E7 smoke v3)")
            except FileNotFoundError as exc:
                if not str(exc).startswith("WAITING_FOR_SMOKE_V3"):
                    raise
                smoke_dependency_status = "WAITING_FOR_SMOKE_V3"
                print("Formal launch dependency: WAITING_FOR_SMOKE_V3 (validation may continue; execution may not)")

        # 5. Git status
        git_info = get_git_info()
        print(f"Git HEAD commit: {git_info['commit']}")
        print(f"Git tracked dirty: {git_info['tracked_dirty']}")

        print("=" * 80)
        print("ALL E8 PREFLIGHT VALIDATIONS PASSED SUCCESSFULLY.")
        print("=" * 80)

        result = {
            "status": "PREFLIGHTS_PASSED",
            "git_commit": git_info["commit"],
            "manifest_sha256": manifest_hash,
            "split_sha256": split_hash,
            "checkpoints": ckpt_info,
            "fishers": fisher_info,
            "reused_count": verified_reused,
            "execute_new_count": len(execute_new_cells),
            "target_events": self.target_events,
            "device": self.device_str,
            "smoke_dependency_status": smoke_dependency_status,
        }
        self._preflight_info = result
        return result

    def get_datasets(self) -> Tuple[KuroEvaluationDataset, KuroEvaluationDataset]:
        """Create sensory-only and deferred-label datasets without indexing either."""
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

    def get_deferred_label_loader(self) -> DeferredLabelLoader:
        """Return the sole E8 GT I/O gateway; construction performs metadata access only."""
        if self._deferred_label_loader is None:
            _, offline_ds = self.get_datasets()
            self._deferred_label_loader = DeferredLabelLoader(offline_ds)
        return self._deferred_label_loader

    def is_stream_completed(self, stream_dir: Path) -> bool:
        """Checks if a stream was already completed with valid summary and status."""
        manifest_path = stream_dir / "manifest.json"
        summary_path = stream_dir / "stream_summary.json"
        status_path = stream_dir / "completion_status.json"

        if not (manifest_path.is_file() and summary_path.is_file() and status_path.is_file()):
            return False

        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
            return status_data.get("status") == "COMPLETED"
        except Exception:
            return False

    def build_stream_batches(
        self,
        order_id: str,
        max_samples_per_event: Optional[int] = None,
        event_subset: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, List[DeferredBatchSpec]]]:
        """Build a metadata-only plan without indexing the offline or sensory dataset."""
        subset_key = tuple(int(e) for e in event_subset) if event_subset is not None else None
        cache_key = f"{order_id}_{max_samples_per_event}_{subset_key}"
        if cache_key in self._cached_stream_plans:
            return self._cached_stream_plans[cache_key]

        adapt_ds, _ = self.get_datasets()

        if "stream_orders" in self.cfg and order_id in self.cfg["stream_orders"]:
            o_cfg = self.cfg["stream_orders"][order_id]
            event_seq = [int(e) for e in o_cfg["event_sequence"]]
            within_sort = o_cfg["within_event_ordering"]
        else:
            event_seq = self.target_events
            within_sort = "deterministic_sorted_sample_id_ascending"

        allowed_subset = set(int(e) for e in event_subset) if event_subset is not None else set(self.target_events)
        target_seq = [e for e in event_seq if e in self.target_events and e in allowed_subset]

        event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(adapt_ds.samples):
            eid = int(item["event_id"])
            if eid in self.target_events:
                item_copy = dict(item)
                item_copy["dataset_index"] = idx
                event_to_samples.setdefault(eid, []).append(item_copy)

        stream_plan: List[Tuple[int, List[DeferredBatchSpec]]] = []

        for eid in target_seq:
            if eid not in event_to_samples:
                continue
            samples = event_to_samples[eid]
            if within_sort == "deterministic_sorted_sample_id_ascending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]))
            elif within_sort == "deterministic_sorted_sample_id_descending":
                samples = sorted(samples, key=lambda x: str(x["sample_id"]), reverse=True)

            if max_samples_per_event is not None and max_samples_per_event > 0:
                samples = samples[:max_samples_per_event]

            event_batches: List[DeferredBatchSpec] = []
            for i in range(0, len(samples), self.batch_size):
                batch_items = samples[i : i + self.batch_size]
                b_ids = [str(s["sample_id"]) for s in batch_items]
                b_indices = [s["dataset_index"] for s in batch_items]

                batch = DeferredBatchSpec(
                    event_id=eid,
                    sample_ids=tuple(b_ids),
                    dataset_indices=tuple(b_indices),
                )
                event_batches.append(batch)

            stream_plan.append((eid, event_batches))

        self._cached_stream_plans[cache_key] = stream_plan
        return stream_plan

    def instantiate_model(self, seed: int) -> torch.nn.Module:
        """Instantiates SNUNet model from source checkpoint."""
        s_cfg = self.cfg["source_checkpoints"]["seeds"][str(seed)]
        ckpt_path = str((REPO_ROOT / s_cfg["checkpoint_path"]).resolve())

        m_cfg = self.cfg["model"]
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            in_channels=int(m_cfg.get("in_channels", 2)),
            out_ch=int(m_cfg.get("num_classes", 3)),
            base_channel=int(m_cfg.get("base_channel", 32)),
            device=self.device_str,
        )
        return wrapper

    def instantiate_adapter(self, method_name: str, model: torch.nn.Module, seed: int = 42) -> BaseE1Adapter:
        """Instantiates adapter for the specified method."""
        m_cfg = self.cfg["evaluated_methods"].get(method_name)
        if m_cfg is None:
            raise ValueError(f"Method {method_name} not configured in evaluated_methods.")

        adapter_class_name = m_cfg["adapter_class"]

        if adapter_class_name == "E7CoTTAAdapter":
            mode = "continual" if method_name == "cotta_continual" else "reset_per_event"
            return E7CoTTAAdapter(
                model=model,
                learning_rate=float(m_cfg.get("learning_rate", 1e-4)),
                ema_alpha=float(m_cfg.get("ema_alpha", 0.999)),
                restoration_prob=float(m_cfg.get("restoration_prob", 0.01)),
                confidence_threshold=float(m_cfg.get("confidence_threshold", 0.90)),
                mode=mode,
                seed=seed,
                device=self.device_str,
            )
        if adapter_class_name == "E1SourceOnlyAdapter":
            return E1SourceOnlyAdapter(model=model, device=self.device_str)
        if adapter_class_name == "E1MatchedBNNoUpdateAdapter":
            return E1MatchedBNNoUpdateAdapter(model=model, device=self.device_str)
        if adapter_class_name == "E1AdaBNAdapter":
            return E1AdaBNAdapter(model=model, device=self.device_str)
        if adapter_class_name == "E1TentAdapter":
            return E1TentAdapter(
                model=model,
                learning_rate=float(m_cfg["learning_rate"]),
                betas=tuple(m_cfg.get("betas", [0.9, 0.999])),
                weight_decay=float(m_cfg.get("weight_decay", 0.0)),
                device=self.device_str,
            )
        if adapter_class_name == "E1EATAAdapter":
            fisher_cfg = self.cfg["fisher_artifacts_v2"]["seeds"][str(seed)]
            fishers = torch.load((REPO_ROOT / fisher_cfg["path"]).resolve(), map_location=self.device_str)
            return E1EATAAdapter(
                model=model,
                fishers=fishers,
                fisher_alpha=float(m_cfg["fisher_alpha"]),
                e_margin=float(m_cfg["reliability_threshold_e0"]),
                d_margin=float(m_cfg["redundancy_threshold_d_margin"]),
                learning_rate=float(m_cfg["learning_rate"]),
                momentum=float(m_cfg.get("momentum", 0.9)),
                weight_decay=float(m_cfg.get("weight_decay", 0.0)),
                device=self.device_str,
            )
        if adapter_class_name == "E1SARAdapter":
            return E1SARAdapter(
                model=model,
                e_margin=float(m_cfg["reliability_threshold_e0"]),
                sam_rho=float(m_cfg["sam_rho"]),
                recovery_threshold=float(m_cfg["recovery_threshold_em"]),
                learning_rate=float(m_cfg["learning_rate"]),
                momentum=float(m_cfg.get("momentum", 0.9)),
                weight_decay=float(m_cfg.get("weight_decay", 0.0)),
                device=self.device_str,
            )
        raise NotImplementedError(f"Unsupported E8 v2 adapter class: {adapter_class_name}")

    def execute_stream(
        self,
        stream_id: str,
        method_name: str,
        seed: int,
        order_id: str,
        reset_mode: str,
        max_samples_per_event: Optional[int] = None,
        event_subset: Optional[Sequence[int]] = None,
        execution_contract_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute one stream with physical GT reads deferred until shard durability."""
        stream_dir = self.output_root / stream_id
        if self.is_stream_completed(stream_dir):
            print(f"Skipping already completed stream: {stream_id}")
            with open(stream_dir / "stream_summary.json", "r", encoding="utf-8") as f:
                return json.load(f)

        if stream_dir.exists() and any(stream_dir.iterdir()):
            raise RuntimeError(
                f"Refusing unsafe partial resume for {stream_id}. Preserve the interrupted directory "
                "and use a fresh versioned stream/output identity; no prediction, update, or GT read was repeated."
            )

        stream_dir.mkdir(parents=True, exist_ok=True)

        model = self.instantiate_model(seed)
        adapter = self.instantiate_adapter(method_name, model, seed=seed)
        stream_plan = self.build_stream_batches(
            order_id,
            max_samples_per_event=max_samples_per_event,
            event_subset=event_subset,
        )
        label_loader = self.get_deferred_label_loader()
        label_reads_at_start = label_loader.index_count
        expected_counts = {str(eid): sum(len(b.sample_ids) for b in batches) for eid, batches in stream_plan}
        expected_batches = {str(eid): len(batches) for eid, batches in stream_plan}
        if execution_contract_path is not None:
            execution_contract_path = execution_contract_path.resolve()
            execution_contract = self._load_json(execution_contract_path)
            scope = execution_contract["authorization"]["scope"]
            if [eid for eid, _ in stream_plan] != scope["event_ids"]:
                raise ValueError("Execution plan events differ from the authorized smoke contract.")
            if sum(expected_counts.values()) != int(scope["expected_samples"]):
                raise ValueError("Execution plan sample count differs from the authorized smoke contract.")
            if sum(expected_batches.values()) != int(scope["expected_batches"]):
                raise ValueError("Execution plan batch count differs from the authorized smoke contract.")
        checkpoint_cfg = self.cfg["source_checkpoints"]["seeds"][str(seed)]
        checkpoint_path = (REPO_ROOT / checkpoint_cfg["checkpoint_path"]).resolve()
        adapter_path = REPO_ROOT / (
            "src/changeguard/cotta_adapter.py"
            if self.cfg["evaluated_methods"][method_name]["adapter_class"] == "E7CoTTAAdapter"
            else "src/changeguard/baseline_adapters.py"
        )
        git_info = get_git_info()
        start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        initial_manifest = {
            "protocol_version": self.cfg["protocol_version"],
            "run_id": stream_id,
            "stream_id": stream_id,
            "git_head": git_info["commit"],
            "git_tracked_dirty": git_info["tracked_dirty"],
            "config_path": str(self.config_path),
            "config_sha256": compute_file_sha256(self.config_path),
            "ledger_path": str(self.ledger_path),
            "ledger_sha256": compute_file_sha256(self.ledger_path),
            "execution_contract_path": str(execution_contract_path) if execution_contract_path else None,
            "execution_contract_sha256": (
                compute_file_sha256(execution_contract_path) if execution_contract_path else None
            ),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": compute_file_sha256(Path(__file__).resolve()),
            "adapter_path": str(adapter_path.resolve()),
            "adapter_sha256": compute_file_sha256(adapter_path.resolve()),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": compute_file_sha256(checkpoint_path),
            "split_path": str(self.split_path),
            "split_sha256": compute_file_sha256(self.split_path),
            "dataset_manifest_path": str(self.manifest_path),
            "dataset_manifest_sha256": compute_file_sha256(self.manifest_path),
            "event_ids": [eid for eid, _ in stream_plan],
            "expected_sample_counts": expected_counts,
            "expected_batch_counts": expected_batches,
            "expected_total_samples": sum(expected_counts.values()),
            "expected_total_batches": sum(expected_batches.values()),
            "prediction_timing": "PRE_UPDATE",
            "reset_mode": reset_mode,
            "order": order_id,
            "method": method_name,
            "seed": seed,
            "device": self.device_str,
            "sealed_test_rejection_status": "VERIFIED_ZERO_SEALED_EVENTS",
            "adapter_input_contract": ["xA", "xB", "valid_mask_MK0_MNA", "sample_ids", "event_id"],
            "gt_or_gt_derived_mask_entered_adaptation": False,
            "label_io_policy": (
                "GT dataset indexing and label raster reads occur only after the matching atomic prediction "
                "shard is fsync'd, atomically renamed, reopened, and identity-verified."
            ),
            "start_timestamp_utc": start_timestamp,
        }
        initial_manifest_path = stream_dir / "initial_manifest.json"
        atomic_write_json(initial_manifest_path, initial_manifest)
        initial_manifest_sha256 = compute_file_sha256(initial_manifest_path)

        start_time = time.time()
        print(f"\n>>> Starting Stream: {stream_id} (Method: {method_name}, Seed: {seed}, Order: {order_id})")
        event_metrics_list: List[Dict[str, Any]] = []
        completed_events: List[int] = []
        total_samples = global_tp = global_fp = global_fn = global_tn = global_valid = 0
        total_updates = total_skips = total_failures = total_recoveries = total_restorations = 0
        current_batch: Optional[DeferredBatchSpec] = None
        current_shard: Optional[Path] = None
        first_prediction_persisted = False
        zero_gt_reads_before_first_prediction_persistence = True
        all_label_reads_after_matching_prediction_persistence = True

        try:
            adapt_ds, _ = self.get_datasets()
            for event_id, batches in stream_plan:
                if reset_mode == "reset_per_event":
                    adapter.reset_to_source()

                event_dir = stream_dir / f"event_{event_id}"
                event_shards_dir = event_dir / "shards"
                event_audits_dir = event_dir / "label_io_audits"
                event_shards_dir.mkdir(parents=True, exist_ok=True)
                event_audits_dir.mkdir(parents=True, exist_ok=True)
                ev_tp = ev_fp = ev_fn = ev_tn = ev_valid = event_samples = 0

                for b_idx, batch in enumerate(batches):
                    current_batch = batch
                    sensory_records = [adapt_ds[idx] for idx in batch.dataset_indices]
                    observed_ids = tuple(str(r["sample_id"]) for r in sensory_records)
                    if observed_ids != batch.sample_ids:
                        raise RuntimeError("Sensory sample order differs from the metadata-only stream plan.")
                    xA = torch.stack([r["xA"] for r in sensory_records]).to(self.device_str)
                    xB = torch.stack([r["xB"] for r in sensory_records]).to(self.device_str)
                    sensory_valid_mask = torch.stack([r["valid_mask"] for r in sensory_records]).to(self.device_str)
                    if any("label" in r and getattr(adapt_ds, "load_labels", False) for r in sensory_records):
                        raise RuntimeError("Adaptation dataset unexpectedly exposed GT labels.")

                    res = adapter.adapt_and_predict_step(
                        xA=xA,
                        xB=xB,
                        valid_mask=sensory_valid_mask,
                        sample_ids=batch.sample_ids,
                        event_id=event_id,
                    )
                    pred_cand_pre = res["primary_prediction"]["pred_cand_pre"]
                    probs_cand_pre = res["primary_prediction"]["probs_cand_pre"]
                    if not torch.isfinite(pred_cand_pre.float()).all() or not torch.isfinite(probs_cand_pre).all():
                        raise RuntimeError("Non-finite prediction or probability detected before persistence.")
                    adapt_meta = res.get("adaptation_metadata", {})
                    risk_ev = res.get("risk_evidence", {})

                    current_shard = event_shards_dir / f"batch_{b_idx:04d}.npz"
                    if not first_prediction_persisted and label_loader.index_count != label_reads_at_start:
                        zero_gt_reads_before_first_prediction_persistence = False
                        raise RuntimeError("GT was read before the first prediction shard was persisted.")
                    atomic_write_npz(
                        current_shard,
                        sample_ids=np.array(batch.sample_ids),
                        sensory_valid_mask=sensory_valid_mask.detach().cpu().numpy().astype(np.uint8),
                        pred_cand_pre=pred_cand_pre.detach().cpu().numpy().astype(np.uint8),
                        probs_cand_pre=probs_cand_pre.detach().cpu().numpy().astype(np.float32),
                        step_count=adapt_meta.get("step_count", b_idx + 1),
                        action=adapt_meta.get("action", "unknown"),
                        update_applied=adapt_meta.get("update_applied", False),
                        selected_pixels=adapt_meta.get("selected_pixels", 0),
                        mean_entropy=risk_ev.get("mean_entropy", 0.0),
                        mean_max_confidence=risk_ev.get("mean_max_confidence", 0.0),
                    )
                    if not current_shard.is_file() or current_shard.stat().st_size == 0:
                        raise IOError(f"Prediction shard was not durably finalized: {current_shard}")
                    with np.load(current_shard, allow_pickle=False) as persisted:
                        persisted_ids = tuple(str(x) for x in persisted["sample_ids"].tolist())
                    if persisted_ids != batch.sample_ids:
                        raise RuntimeError("Persisted prediction IDs do not match planned batch IDs.")
                    persistence_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    shard_sha256 = compute_file_sha256(current_shard)
                    first_prediction_persisted = True

                    label_reads_before_batch = label_loader.index_count
                    labels = label_loader.load_exact(batch.sample_ids, event_id).to(self.device_str)
                    label_read_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    if label_loader.index_count - label_reads_before_batch != len(batch.sample_ids):
                        raise RuntimeError("Deferred label-read count does not match persisted sample count.")

                    V_eval = (sensory_valid_mask == 1) & (labels != 3)
                    p_c2 = pred_cand_pre == 2
                    t_c2 = labels == 2
                    tp = int(((p_c2 & t_c2) & V_eval).sum().item())
                    fp = int(((p_c2 & (~t_c2)) & V_eval).sum().item())
                    fn = int((((~p_c2) & t_c2) & V_eval).sum().item())
                    tn = int((((~p_c2) & (~t_c2)) & V_eval).sum().item())
                    valid = int(V_eval.sum().item())
                    if tp + fp + fn + tn != valid:
                        raise RuntimeError("Batch confusion cells do not conserve offline-valid pixels.")

                    audit_record = {
                        "event_id": event_id,
                        "batch_index": b_idx,
                        "sample_ids": list(batch.sample_ids),
                        "prediction_shard": str(current_shard),
                        "prediction_shard_sha256": shard_sha256,
                        "persistence_timestamp_utc": persistence_timestamp,
                        "label_read_timestamp_utc": label_read_timestamp,
                        "label_reads": len(batch.sample_ids),
                        "sample_identity_match": True,
                        "label_read_after_persistence": label_read_timestamp > persistence_timestamp,
                    }
                    if not audit_record["label_read_after_persistence"]:
                        all_label_reads_after_matching_prediction_persistence = False
                        raise RuntimeError("GT read timestamp is not strictly after shard persistence timestamp.")
                    atomic_write_json(event_audits_dir / f"batch_{b_idx:04d}.json", audit_record)

                    ev_tp += tp; ev_fp += fp; ev_fn += fn; ev_tn += tn; ev_valid += valid
                    event_samples += len(batch.sample_ids)
                    action = str(adapt_meta.get("action", "unknown"))
                    total_updates += int(bool(adapt_meta.get("update_applied", False)))
                    total_skips += int(action == "skip")
                    total_failures += int(action in {"failed", "rollback"})
                    total_recoveries += int(bool(adapt_meta.get("recovery_applied", False)))
                    total_restorations += int(adapt_meta.get("restored_elements", 0) or 0)

                ev_prec = float(ev_tp / max(ev_tp + ev_fp, 1))
                ev_rec = float(ev_tp / max(ev_tp + ev_fn, 1))
                ev_f1 = float(2 * ev_tp / max(2 * ev_tp + ev_fp + ev_fn, 1))
                ev_iou = float(ev_tp / max(ev_tp + ev_fp + ev_fn, 1))
                ev_record = {
                    "event_id": event_id, "samples": event_samples, "valid_pixels": ev_valid,
                    "tp": ev_tp, "fp": ev_fp, "fn": ev_fn, "tn": ev_tn,
                    "precision": ev_prec, "recall": ev_rec, "f1": ev_f1, "iou": ev_iou,
                }
                event_metrics_list.append(ev_record)
                completed_events.append(event_id)
                total_samples += event_samples
                global_tp += ev_tp; global_fp += ev_fp; global_fn += ev_fn; global_tn += ev_tn; global_valid += ev_valid
                print(f"  Event {event_id}: F1={ev_f1:.4f}, Prec={ev_prec:.4f}, Rec={ev_rec:.4f}, IoU={ev_iou:.4f} (N={event_samples})")

            if total_samples != sum(expected_counts.values()):
                raise RuntimeError(f"Completed sample count {total_samples} != expected {sum(expected_counts.values())}")
            if label_loader.index_count - label_reads_at_start != total_samples:
                raise RuntimeError("Terminal deferred GT read count does not match completed sample count.")

            elapsed = time.time() - start_time
            macro_f1 = float(np.mean([m["f1"] for m in event_metrics_list]))
            macro_iou = float(np.mean([m["iou"] for m in event_metrics_list]))
            macro_prec = float(np.mean([m["precision"] for m in event_metrics_list]))
            macro_rec = float(np.mean([m["recall"] for m in event_metrics_list]))
            pooled_prec = float(global_tp / max(global_tp + global_fp, 1))
            pooled_rec = float(global_tp / max(global_tp + global_fn, 1))
            pooled_f1 = float(2 * global_tp / max(2 * global_tp + global_fp + global_fn, 1))
            pooled_iou = float(global_tp / max(global_tp + global_fp + global_fn, 1))
            summary_data = {
                "stream_id": stream_id, "method": method_name, "seed": seed, "order_id": order_id,
                "reset_mode": reset_mode, "elapsed_seconds": elapsed,
                "macro_f1": macro_f1, "macro_precision": macro_prec, "macro_recall": macro_rec, "macro_iou": macro_iou,
                "pooled_f1": pooled_f1, "pooled_precision": pooled_prec, "pooled_recall": pooled_rec, "pooled_iou": pooled_iou,
                "updates_applied": total_updates, "updates_skipped": total_skips,
                "updates_failed": total_failures, "recoveries": total_recoveries,
                "stochastic_restored_elements": total_restorations,
                "deferred_gt_reads": label_loader.index_count - label_reads_at_start,
                "event_metrics": event_metrics_list,
            }
            atomic_write_json(stream_dir / "stream_summary.json", summary_data)
            csv_path = stream_dir / "event_metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(event_metrics_list[0].keys()))
                writer.writeheader(); writer.writerows(event_metrics_list); f.flush(); os.fsync(f.fileno())
            _fsync_directory(stream_dir)

            final_manifest = {
                "stream_id": stream_id,
                "status": "COMPLETED",
                "initial_manifest_path": str(initial_manifest_path),
                "initial_manifest_sha256": initial_manifest_sha256,
                "completed_events": completed_events,
                "total_samples": total_samples,
                "total_batches": sum(expected_batches.values()),
                "deferred_gt_reads": label_loader.index_count - label_reads_at_start,
                "label_io_verification": {
                    "zero_gt_reads_before_first_prediction_persistence": zero_gt_reads_before_first_prediction_persistence,
                    "all_label_reads_after_matching_prediction_persistence": all_label_reads_after_matching_prediction_persistence,
                    "gt_or_gt_derived_mask_entered_adaptation": False,
                    "sample_identity_verified": True,
                    "sealed_test_access": 0,
                    "finite_outputs_verified": True,
                    "failed_persistence_count": 0,
                },
                "elapsed_seconds": elapsed,
                "terminal_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            atomic_write_json(stream_dir / "manifest.json", final_manifest)
            atomic_write_json(stream_dir / "completion_status.json", final_manifest)
            if compute_file_sha256(initial_manifest_path) != initial_manifest_sha256:
                raise RuntimeError("Immutable initial manifest changed during execution.")
            print(f"<<< Completed Stream {stream_id} in {elapsed:.1f}s | Macro F1: {macro_f1:.4f} | Pooled F1: {pooled_f1:.4f}")
            return summary_data
        except BaseException as exc:
            failure = {
                "status": "FAILED",
                "stream_id": stream_id,
                "initial_manifest_path": str(initial_manifest_path),
                "initial_manifest_sha256": initial_manifest_sha256,
                "event_id": current_batch.event_id if current_batch else None,
                "sample_ids": list(current_batch.sample_ids) if current_batch else [],
                "prediction_shard": str(current_shard) if current_shard else None,
                "deferred_gt_reads_this_invocation": label_loader.index_count - label_reads_at_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "terminal_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            atomic_write_json(stream_dir / "failure.json", failure)
            atomic_write_json(stream_dir / "completion_status.json", failure)
            raise
        finally:
            del model
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E8 Confirmation Matrix Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v2.json",
        help="Path to confirmation matrix JSON configuration.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "configs/changeguard/e8_execution_ledger_v2.json",
        help="Path to execution ledger JSON.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run fail-closed preflights and ledger validation, then exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Display exact planned cell enumeration, reuse vs execute-new statuses, and exit.",
    )
    parser.add_argument(
        "--stream_id",
        type=str,
        default=None,
        help="Execute single specific stream by ID.",
    )
    parser.add_argument(
        "--execute_new_only",
        action="store_true",
        help="Execute all pending execute-new streams from ledger.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional override path for output root directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2-batch smoke test on Event 520.",
    )
    parser.add_argument(
        "--smoke_config",
        type=Path,
        default=None,
        help="Committed smoke authorization contract; required with --smoke.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    args = parser.parse_args()

    runner = E8ConfirmationRunner(
        config_path=args.config,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.validate_only:
        runner.run_preflights()
        print("\nPreflight validation check complete: SUCCESS (--validate_only). Exiting.")
        sys.exit(0)

    if args.dry_run:
        runner.run_preflights()
        reused, new_cells = runner.planned_cells()
        print("\n" + "=" * 80)
        print("ChangeGuard E8 Confirmation Matrix Dry-Run Audit")
        print("=" * 80)
        print(f"Total Physical Streams Planned: {len(reused) + len(new_cells)}")
        print(f"  - Certified Reused Streams:   {len(reused)} ({len(reused) * 6} event rows)")
        print(f"  - Execute-New Streams:        {len(new_cells)} ({len(new_cells) * 6} event rows)")
        print(f"Total Event-Result Rows:        {(len(reused) + len(new_cells)) * 6}")
        print("\nExecute-New Stream List:")
        for i, c in enumerate(new_cells, 1):
            print(f"  {i:2d}. {c['stream_id']}: method={c['method']}, seed={c['seed']}, order={c['order']}, reset={c['reset_mode']}")
        print("=" * 80)
        print("Dry-run audit complete: SUCCESS (--dry_run). Exiting.")
        sys.exit(0)

    runner.run_preflights()

    if args.smoke:
        if args.stream_id is None or args.execute_new_only:
            parser.error("--smoke requires exactly one --stream_id and cannot be combined with --execute_new_only.")
        if args.smoke_config is None:
            parser.error("--smoke requires --smoke_config for authorization and provenance.")
        runner.validate_smoke_launch(args.smoke_config, require_fresh_output=True)
    elif args.stream_id is not None or args.execute_new_only:
        runner.validate_formal_launch(require_fresh_output=True)
    else:
        parser.error("Choose --validate_only, --dry_run, --stream_id, or --execute_new_only.")

    if args.stream_id is not None:
        _, new_cells = runner.planned_cells()
        target_cell = next((c for c in new_cells if c["stream_id"] == args.stream_id), None)
        if target_cell is None:
            raise ValueError(f"Stream ID {args.stream_id} not found in execute-new ledger cells.")
        max_samples = 32 if args.smoke else None
        runner.execute_stream(
            stream_id=target_cell["stream_id"],
            method_name=target_cell["method"],
            seed=int(target_cell["seed"]),
            order_id=target_cell["order"],
            reset_mode=target_cell["reset_mode"],
            max_samples_per_event=max_samples,
            event_subset=[520] if args.smoke else None,
            execution_contract_path=args.smoke_config if args.smoke else None,
        )
    elif args.execute_new_only:
        _, new_cells = runner.planned_cells()
        max_samples = 32 if args.smoke else None
        print(f"\nExecuting {len(new_cells)} pending streams from ledger...")
        for i, c in enumerate(new_cells, 1):
            print(f"\n[{i}/{len(new_cells)}] Processing {c['stream_id']}...")
            runner.execute_stream(
                stream_id=c["stream_id"],
                method_name=c["method"],
                seed=int(c["seed"]),
                order_id=c["order"],
                reset_mode=c["reset_mode"],
                max_samples_per_event=max_samples,
                event_subset=[520] if args.smoke else None,
            )


if __name__ == "__main__":
    main()
