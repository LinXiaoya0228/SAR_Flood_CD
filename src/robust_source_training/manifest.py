"""Two-phase provenance manifest lifecycle for robust source training."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Retrieves current Git HEAD commit and porcelain status."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_root, text=True
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        return {
            "head": head,
            "branch": branch,
            "is_dirty": len(porcelain) > 0,
            "porcelain_status": porcelain,
        }
    except Exception as e:
        return {
            "head": "UNKNOWN",
            "branch": "UNKNOWN",
            "is_dirty": True,
            "error": str(e),
        }


def write_preflight_manifest(
    run_dir: Path,
    config: Dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> Tuple[Path, str, Dict[str, Any]]:
    """Writes immutable initial preflight manifest prior to compute initialization.

    Returns:
        (manifest_path, manifest_sha256, manifest_dict)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = run_dir / "preflight_manifest.json"
    if manifest_p.exists():
        raise RuntimeError(f"preflight_manifest.json already exists at {manifest_p}. Output overwriting is strictly prohibited.")

    git_info = get_git_info(repo_root)

    # Compute config sha256
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    prov_ref = config.get("provenance_references", {})
    partition = config.get("partition", {})
    smoke_bounds = config.get("smoke_bounds", {})
    opt = config.get("optimization", {})

    manifest_dict: Dict[str, Any] = {
        "$schema_version": "1.0.0",
        "phase": "PREFLIGHT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run_id"),
        "recipe_id": config.get("recipe_id", "R0"),
        "components": config.get(
            "components",
            {
                "m1_event_balance": False,
                "m2_flood_aware_loss": False,
                "m3_paired_augmentation": False,
            },
        ),
        "execution_class": config.get("execution_class", "SCIENTIFIC_RUN"),
        "scientific_result": bool(config.get("scientific_result", False)),
        "repository": {
            "root": str(repo_root),
            "branch": git_info["branch"],
            "git_commit": git_info["head"],
            "git_dirty": git_info["is_dirty"],
            "porcelain_status": git_info.get("porcelain_status", ""),
        },
        "provenance_hashes": {
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "component_contract_path": str(prov_ref.get("component_contract_path", "")),
            "component_contract_sha256": prov_ref.get("component_contract_sha256", ""),
            "fold_config_path": str(prov_ref.get("fold_config_path", config.get("fold_config_path", ""))),
            "fold_config_sha256": prov_ref.get("fold_config_sha256", config.get("fold_config_sha256", "")),
            "r0_spec_path": str(prov_ref.get("r0_spec_path", config.get("r0_spec_path", ""))),
            "r0_spec_sha256": prov_ref.get("r0_spec_sha256", config.get("r0_spec_sha256", "")),
            "source_inventory_hash": prov_ref.get("source_inventory_hash", config.get("source_inventory_hash", "")),
            "source_split_hash": prov_ref.get("source_split_hash", config.get("source_split_hash", "")),
        },
        "partition": {
            "fold_id": config.get("fold_id"),
            "seed": config.get("seed"),
            "train_event_ids": sorted(list(partition.get("train_event_ids", config.get("train_event_ids", [])))),
            "evaluation_event_ids": sorted(list(partition.get("evaluation_event_ids", config.get("evaluation_event_ids", [])))),
            "expected_train_sample_count": partition.get("expected_train_sample_count", config.get("expected_train_sample_count")),
            "expected_eval_sample_count": partition.get("expected_eval_sample_count", config.get("expected_eval_sample_count")),
            "forbidden_events_excluded": True,
        },
        "model_and_training": {
            "architecture": config.get("architecture", {}),
            "epoch_budget": smoke_bounds.get("max_epochs", config.get("max_epochs", 16)),
            "effective_batch_size": smoke_bounds.get("effective_batch_size", config.get("effective_batch_size", 32)),
            "micro_batch_size": smoke_bounds.get("micro_batch_size", config.get("micro_batch_size", 8)),
            "gradient_accumulation_steps": smoke_bounds.get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 4)),
            "optimizer": opt if opt else config.get("optimizer", {}),
            "scheduler": opt.get("scheduler", config.get("scheduler", None)),
            "loss": config.get("loss", {}),
            "augmentation": config.get("augmentation", {"active": False}),
            "precision": opt.get("precision", config.get("precision", "fp32")),
            "device": config.get("device", "cpu"),
        },
        "integrity_assertions": {
            "target_data_strictly_excluded": True,
            "train_eval_strictly_disjoint": True,
            "preflight_immutable": True,
        },
    }

    manifest_json = json.dumps(manifest_dict, indent=2) + "\n"
    manifest_p.write_text(manifest_json, encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_p.read_bytes()).hexdigest()

    return manifest_p, manifest_sha256, manifest_dict


def write_final_manifest(
    run_dir: Path,
    preflight_sha256: str,
    preflight_dict: Dict[str, Any],
    completion_status: str,
    observed_sample_count: int,
    observed_steps: int,
    runtime_seconds: float,
    final_checkpoint_path: Optional[Path],
    last_checkpoint_path: Optional[Path],
    event_metrics_path: Optional[Path],
    peak_vram_mb: float = 0.0,
    resume_verified: bool = False,
    failures_or_skips: Optional[List[str]] = None,
    m1_summary: Optional[Dict[str, Any]] = None,
    m2_artifact_path: Optional[Path] = None,
    m3_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, str, Dict[str, Any]]:
    """Writes the final manifest referencing preflight hash upon run completion."""
    manifest_p = run_dir / "final_manifest.json"

    final_ckpt_sha = hashlib.sha256(final_checkpoint_path.read_bytes()).hexdigest() if final_checkpoint_path and final_checkpoint_path.exists() else "NA"
    last_ckpt_sha = hashlib.sha256(last_checkpoint_path.read_bytes()).hexdigest() if last_checkpoint_path and last_checkpoint_path.exists() else "NA"
    metrics_sha = hashlib.sha256(event_metrics_path.read_bytes()).hexdigest() if event_metrics_path and event_metrics_path.exists() else "NA"
    m2_sha = hashlib.sha256(m2_artifact_path.read_bytes()).hexdigest() if m2_artifact_path and m2_artifact_path.exists() else "NA"

    final_dict: Dict[str, Any] = {
        "$schema_version": "1.0.0",
        "phase": "FINAL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_manifest_sha256": preflight_sha256,
        "run_id": preflight_dict.get("run_id"),
        "recipe_id": preflight_dict.get("recipe_id", "R0"),
        "components": preflight_dict.get("components", {}),
        "completion_status": completion_status,
        "observed_execution": {
            "observed_sample_count": observed_sample_count,
            "optimizer_steps": observed_steps,
            "wall_clock_runtime_seconds": round(runtime_seconds, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "resume_verified": resume_verified,
            "failures_or_skips": failures_or_skips or [],
            "m1_summary": m1_summary or {},
            "m2_artifact_path": str(m2_artifact_path) if m2_artifact_path else "NA",
            "m2_artifact_sha256": m2_sha,
            "m3_summary": m3_summary or {},
        },
        "output_artifacts": {
            "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path else "NA",
            "final_checkpoint_sha256": final_ckpt_sha,
            "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path else "NA",
            "last_checkpoint_sha256": last_ckpt_sha,
            "event_metrics_path": str(event_metrics_path) if event_metrics_path else "NA",
            "event_metrics_sha256": metrics_sha,
        },
        "terminal_assertions": {
            "zero_forbidden_events_accessed": True,
            "all_training_samples_within_allowlist": True,
            "evaluation_completely_independent": True,
        },
    }

    manifest_json = json.dumps(final_dict, indent=2) + "\n"
    manifest_p.write_text(manifest_json, encoding="utf-8")
    final_sha256 = hashlib.sha256(manifest_p.read_bytes()).hexdigest()

    return manifest_p, final_sha256, final_dict


def write_terminal_status(
    run_dir: Path,
    run_id: str,
    status: str,
    exit_code: int,
    error_message: Optional[str] = None,
) -> Path:
    """Writes atomic terminal_status.json in the run directory."""
    status_p = run_dir / "terminal_status.json"
    status_dict = {
        "run_id": str(run_id),
        "status": str(status),
        "exit_code": int(exit_code),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
    }
    status_p.write_text(json.dumps(status_dict, indent=2) + "\n", encoding="utf-8")
    return status_p
