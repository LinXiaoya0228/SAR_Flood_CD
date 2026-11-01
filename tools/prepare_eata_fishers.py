"""
Preparation utility for EATA Fisher Information Matrices on approved source training events.

Governing Protocol: configs/evaluation/week9_stronger_tta_protocol.json
Document Specification: docs/week9_stronger_tta_protocol.md

Strict Scientific & Data Isolation Principles:
1. Source-Only Data Isolation:
   - Data points are sampled strictly from the 27 approved source training events defined in
     configs/splits/kuro_source_all27_split_v1.json:
     [118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498,
      502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011].
   - Target development events [279, 437, 520, 559, 1111003, 1111008] and sealed test events
     are strictly forbidden and never accessed.
2. Unsupervised / Empirical Fisher:
   - Evaluated strictly using model-generated pseudo-labels (c_hat = argmax p) masked by sensory validity (MK0_MNA == 1).
   - Ground truth annotation rasters (MK0_MLU) are NEVER loaded or accessed.
3. Deterministic Selection:
   - Exactly N = 2,000 source patches sampled without replacement using selection seed 2026.
   - The same selected sample IDs are evaluated identically across checkpoint seeds 0, 1, and 2.
4. Exact Accumulation Formula:
   - True empirical Fisher is the mean of squared per-sample gradients:
     F_k = (1 / N) * sum_{i=1}^N (d l_i / d theta_k)^2.
   - Avoids the distortion of squared batch-mean gradients (which include cross-sample covariance terms).
5. Immutable Output Provenance:
   - Outputs saved with complete metadata recording selected sample IDs, hashes, loss reduction,
     and accumulation formula.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_deterministic_source_sample_ids(
    manifest_path: Path,
    split_config_path: Path,
    selection_seed: int = 2026,
    sample_size: int = 2000,
    expected_split_sha256: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Selects deterministic sample IDs without replacement from approved source events."""
    manifest_path = Path(manifest_path)
    split_config_path = Path(split_config_path)

    act_split_sha = compute_file_sha256(split_config_path)
    if expected_split_sha256 is not None and act_split_sha != expected_split_sha256:
        raise ValueError(
            f"Split config SHA-256 mismatch: actual {act_split_sha} vs expected {expected_split_sha256}"
        )

    with open(split_config_path, "r") as f:
        split_data = json.load(f)

    approved_events = set(split_data.get("source_all_events", []))
    if not approved_events:
        raise ValueError(f"No source_all_events found in {split_config_path}")

    # Load KuroSiwo manifest
    if str(manifest_path).endswith(".gz"):
        with gzip.open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
    else:
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)

    manifest_sha = compute_file_sha256(manifest_path)

    FORBIDDEN_SEALED_TEST_EVENTS = {
        205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
    }

    # Filter candidate sample IDs belonging to approved source events
    candidate_ids = []
    for sample_id, meta in manifest.items():
        # KuroSiwo metadata format is meta["info"]["actid"] or top-level event_id
        if isinstance(meta, dict) and "info" in meta and isinstance(meta["info"], dict) and "actid" in meta["info"]:
            event_id = int(meta["info"]["actid"])
        elif isinstance(meta, dict):
            event_id = meta.get("event_id", meta.get("event"))
            if event_id is not None:
                event_id = int(event_id)
        else:
            continue

        if event_id in FORBIDDEN_SEALED_TEST_EVENTS:
            continue
        if event_id in approved_events:
            candidate_ids.append(str(sample_id))

    candidate_ids = sorted(candidate_ids)
    total_candidates = len(candidate_ids)

    if total_candidates < sample_size:
        raise ValueError(
            f"Insufficient candidate samples ({total_candidates}) for sample_size {sample_size}"
        )

    # Deterministic selection without replacement
    rng = random.Random(selection_seed)
    selected_sample_ids = sorted(rng.sample(candidate_ids, sample_size))
    selected_ids_sha = hashlib.sha256(json.dumps(selected_sample_ids).encode()).hexdigest()

    metadata = {
        "selection_seed": selection_seed,
        "sample_size": sample_size,
        "total_source_candidates": total_candidates,
        "source_candidate_ids_sha256": hashlib.sha256(json.dumps(candidate_ids).encode()).hexdigest(),
        "approved_source_events": sorted(list(approved_events)),
        "split_config_path": str(split_config_path),
        "split_config_sha256": act_split_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "selected_sample_ids": selected_sample_ids,
        "selected_sample_ids_sha256": selected_ids_sha,
        "estimator": "per_patch_squared_gradient",
        "loss_reduction": "mean_over_valid_pixels",
        "bn_context": "eval_mode_frozen_running_stats",
        "preprocessing": {
            "sar_mean": [0.0953, 0.0264],
            "sar_std": [0.0427, 0.0215],
            "sar_clamp_max": 0.15,
        },
    }

    return selected_sample_ids, metadata


def compute_empirical_fisher_for_checkpoint(
    model: nn.Module,
    sample_ids: List[str],
    dataloader_fn: Any,
    device: str = "cpu",
    accumulation_formula: str = "exact_per_sample",
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    """Computes empirical diagonal Fisher Information Matrix over BN affine parameters.

    Returns:
        fishers: {param_name: (fisher_diagonal_tensor, initial_param_copy)}
        stats: execution statistics and metadata
    """
    if accumulation_formula not in ["exact_per_sample", "per_patch"]:
        raise ValueError(
            f"Unsupported accumulation_formula: '{accumulation_formula}'. "
            "Only 'exact_per_sample' ('per_patch') is approved by protocol."
        )

    model.eval()
    model.to(device)

    # Collect BN affine parameters and save initial parameter copies
    fisher_accum: Dict[str, torch.Tensor] = {}
    param_copies: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p_name, param in module.named_parameters():
                if p_name in ["weight", "bias"]:
                    full_name = f"{name}.{p_name}"
                    fisher_accum[full_name] = torch.zeros_like(param.data, device=device)
                    param_copies[full_name] = param.data.detach().clone().cpu()

    # Anchor parameter finite validation
    for name, p_copy in param_copies.items():
        if not torch.all(torch.isfinite(p_copy)):
            raise ValueError(f"Initial parameter anchor for {name} contains non-finite values!")

    # Configure gradients: only BN affine params require grad
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p_name, param in module.named_parameters():
                if p_name in ["weight", "bias"]:
                    param.requires_grad = True

    total_samples = len(sample_ids)
    start_time = time.time()
    evaluated_count = 0
    skipped_zero_valid_count = 0
    yielded_ids: List[str] = []

    # Dataloader yields (sample_id, xA, xB, sar_valid_mask or valid_mask)
    # Strictly zero ground truth labels are yielded or accessed
    for batch in dataloader_fn(sample_ids):
        b_sid = batch.get("sample_id", batch.get("sample_ids"))
        if b_sid is not None:
            if isinstance(b_sid, list):
                yielded_ids.extend([str(s) for s in b_sid])
            else:
                yielded_ids.append(str(b_sid))

        xA = batch["xA"].to(device)
        xB = batch["xB"].to(device)
        sar_valid_mask = batch.get("sar_valid_mask", batch.get("valid_mask"))
        if sar_valid_mask is not None:
            sar_valid_mask = sar_valid_mask.to(device)

        batch_size = xA.size(0)

        # Loop over individual samples to compute true per-sample squared gradients
        for b in range(batch_size):
            x_a_single = xA[b : b + 1]
            x_b_single = xB[b : b + 1]
            mask_single = (
                sar_valid_mask[b : b + 1] if sar_valid_mask is not None else None
            )

            if mask_single is not None:
                m = (
                    mask_single.squeeze(1).float()
                    if mask_single.dim() == 4
                    else mask_single.float()
                )
                v = torch.sum(m)
                if v == 0:
                    # Explicitly exclude zero-valid sensory patches
                    skipped_zero_valid_count += 1
                    continue
                model.zero_grad()
                logits = model(x_a_single, x_b_single)
                pseudo_targets = torch.argmax(logits, dim=1)
                ce_loss = F.cross_entropy(logits, pseudo_targets, reduction="none")
                loss = torch.sum(ce_loss * m) / v
            else:
                model.zero_grad()
                logits = model(x_a_single, x_b_single)
                pseudo_targets = torch.argmax(logits, dim=1)
                ce_loss = F.cross_entropy(logits, pseudo_targets, reduction="none")
                loss = torch.mean(ce_loss)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss ({loss.item()}) encountered during Fisher computation at patch {evaluated_count + skipped_zero_valid_count + 1}!"
                )

            loss.backward()

            for name, param in model.named_parameters():
                if name in fisher_accum and param.grad is not None:
                    if not torch.all(torch.isfinite(param.grad)):
                        raise ValueError(
                            f"Non-finite gradient encountered for {name} during Fisher computation at patch {evaluated_count + skipped_zero_valid_count + 1}!"
                        )
                    fisher_accum[name] += param.grad.detach() ** 2

            evaluated_count += 1

    total_processed = evaluated_count + skipped_zero_valid_count
    if total_processed != total_samples:
        raise ValueError(
            f"Incomplete sample processing: processed {total_processed} patches vs expected {total_samples}!"
        )

    if yielded_ids and len(yielded_ids) == total_samples and yielded_ids != [str(s) for s in sample_ids]:
        raise ValueError("Yielded sample IDs did not match expected selected sample IDs!")

    if evaluated_count == 0:
        raise ValueError("All candidate patches had zero sensory-valid pixels; cannot compute Fisher matrix!")

    # Normalize accumulated squared gradients by the number of evaluated patches
    for name in fisher_accum:
        fisher_accum[name] = fisher_accum[name] / evaluated_count
        if not torch.all(torch.isfinite(fisher_accum[name])) or not torch.all(fisher_accum[name] >= 0.0):
            raise ValueError(f"Accumulated Fisher for {name} contains non-finite or negative values!")

    elapsed = time.time() - start_time

    # Construct final dictionary {name: (fisher_tensor_cpu, initial_param_cpu)}
    fishers_result = {}
    for name in fisher_accum:
        fishers_result[name] = (
            fisher_accum[name].detach().cpu(),
            param_copies[name],
        )

    stats = {
        "total_samples_requested": total_samples,
        "total_samples_processed": total_processed,
        "num_evaluated_patches": evaluated_count,
        "num_zero_valid_patches": skipped_zero_valid_count,
        "normalization_convention": "mean_over_evaluated_patches",
        "normalization_denominator": evaluated_count,
        "elapsed_seconds": elapsed,
        "accumulation_formula": accumulation_formula,
        "num_adapted_parameters": len(fishers_result),
    }

    return fishers_result, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChangeGuard EATA Fisher Information Matrix Preparation Utility."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/week9_stronger_tta_protocol.json",
        help="Path to Week 9 protocol JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Source checkpoint seed (0, 1, or 2).",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=2026,
        help="Random seed for deterministic sample selection.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of source patches to sample without replacement.",
    )
    parser.add_argument(
        "--accumulation_formula",
        type=str,
        choices=["exact_per_sample", "per_patch"],
        default="exact_per_sample",
        help="Fisher accumulation formula (only exact_per_sample / per_patch is supported).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Prepare Fisher matrices for all three source checkpoint seeds (0, 1, 2).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify plan, sample IDs, and paths without executing model forward/backward.",
    )
    parser.add_argument(
        "--synthetic_test",
        action="store_true",
        help="Run lightweight synthetic end-to-end CPU verification test.",
    )
    return parser.parse_args()


def run_synthetic_test() -> None:
    """Lightweight synthetic CPU test verifying Fisher computation mechanics."""
    print("Running synthetic EATA Fisher preparation test...")
    # Create tiny mock model with BN
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(3)

        def forward(self, xA, xB):
            x = torch.cat([xA, xB], dim=1)
            x = F.relu(self.bn1(self.conv1(x)))
            return self.bn2(self.conv2(x))

    model = MockModel()

    # Synthetic batch generator
    def mock_dataloader(sample_ids):
        for s_id in sample_ids:
            yield {
                "sample_id": s_id,
                "xA": torch.randn(1, 2, 32, 32),
                "xB": torch.randn(1, 2, 32, 32),
                "sar_valid_mask": torch.ones(1, 1, 32, 32),
            }

    sample_ids = [f"patch_{i:04d}" for i in range(10)]
    fishers, stats = compute_empirical_fisher_for_checkpoint(
        model, sample_ids, mock_dataloader, device="cpu", accumulation_formula="exact_per_sample"
    )

    assert "bn1.weight" in fishers
    assert "bn1.bias" in fishers
    assert "bn2.weight" in fishers
    assert "bn2.bias" in fishers
    assert stats["num_evaluated_patches"] == 10

    # Ensure Fisher values are non-negative and finite
    for name, (f_diag, p_init) in fishers.items():
        assert torch.all(f_diag >= 0.0), f"Negative Fisher value in {name}"
        assert torch.all(torch.isfinite(f_diag)), f"Non-finite Fisher value in {name}"
        assert f_diag.shape == p_init.shape, f"Shape mismatch in {name}"

    print(f"Synthetic Fisher test PASSED across {len(fishers)} BN affine parameters.")


def main() -> None:
    import datetime
    args = parse_args()

    if args.synthetic_test:
        run_synthetic_test()
        return

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    source_cfg = protocol.get("source_data_configuration", {})
    manifest_path = Path(source_cfg.get("manifest_path", ""))
    split_config_path = Path(source_cfg.get("split_config_path", ""))

    print("=== EATA Fisher Information Preparation Preflight ===")
    print(f"Protocol:             {config_path}")
    print(f"Split Config:         {split_config_path}")
    print(f"Manifest:             {manifest_path}")
    print(f"Selection Seed:       {args.selection_seed}")
    print(f"Sample Size:          {args.sample_size}")
    print(f"Accumulation Formula: {args.accumulation_formula}")
    print(f"Device:               {args.device}")

    if not manifest_path.is_file() or not split_config_path.is_file():
        print("NOTICE: Manifest or split config not found at specified paths.")
        if args.dry_run:
            print("Dry run complete (verified arguments and configuration structure).")
            return
        raise FileNotFoundError("Raw source data files required for execution.")

    expected_split_sha = source_cfg.get("split_config_sha256")
    if not expected_split_sha:
        raise ValueError(
            "Missing canonical 'split_config_sha256' in protocol 'source_data_configuration'"
        )
    selected_ids, meta = get_deterministic_source_sample_ids(
        manifest_path=manifest_path,
        split_config_path=split_config_path,
        selection_seed=args.selection_seed,
        sample_size=args.sample_size,
        expected_split_sha256=expected_split_sha,
    )

    # W9-N03: Verify manifest hash, candidate count, and selection digest against configuration
    exp_manifest_sha = source_cfg.get("manifest_sha256")
    if exp_manifest_sha and meta.get("manifest_sha256") != exp_manifest_sha:
        raise ValueError(
            f"Manifest SHA-256 mismatch: actual {meta.get('manifest_sha256')} vs expected {exp_manifest_sha}"
        )

    exp_total_patches = source_cfg.get("total_source_patches")
    if exp_total_patches is not None and meta.get("total_source_candidates") != exp_total_patches:
        raise ValueError(
            f"Candidate count mismatch: actual {meta.get('total_source_candidates')} vs expected {exp_total_patches}"
        )

    exp_cand_sha = source_cfg.get("source_candidate_ids_sha256")
    if exp_cand_sha and meta.get("source_candidate_ids_sha256") != exp_cand_sha:
        raise ValueError(
            f"Candidate IDs SHA-256 mismatch: actual {meta.get('source_candidate_ids_sha256')} vs expected {exp_cand_sha}"
        )

    exp_sel_sha = protocol.get("fisher_information_protocol", {}).get("selected_sample_ids_sha256")
    if exp_sel_sha and meta.get("selected_sample_ids_sha256") != exp_sel_sha:
        raise ValueError(
            f"Selection digest mismatch against protocol: actual {meta.get('selected_sample_ids_sha256')} vs expected {exp_sel_sha}"
        )

    print(f"Selected {len(selected_ids)} source samples deterministically.")
    print(f"First 5 sample IDs: {selected_ids[:5]}")
    print(f"Last 5 sample IDs: {selected_ids[-5:]}")

    seeds_to_run = [0, 1, 2] if args.all_seeds else [args.seed]

    f_paths = protocol.get("fisher_information_protocol", {}).get("artifact_paths", {})
    storage_dir = protocol.get("fisher_information_protocol", {}).get("storage_dir", "artifacts/eata_fishers")

    # W9-N01: Check destination paths before performing any heavy raster loading or model computation
    dest_paths = {}
    for seed in seeds_to_run:
        out_rel = f_paths.get(str(seed), f"{storage_dir}/source_all27_seed{seed}_fisher.pt")
        out_path = (REPO_ROOT / out_rel).resolve()
        meta_path = out_path.with_suffix(".json")
        if out_path.exists() or meta_path.exists():
            raise FileExistsError(
                f"Destination Fisher artifact already exists for seed {seed}: {out_path} (or {meta_path}). "
                "Overwriting existing Fisher artifacts is strictly prohibited."
            )
        dest_paths[seed] = (out_path, meta_path)

    # Verify all target checkpoint files and hashes before doing heavy work
    ckpt_cfgs = protocol.get("source_checkpoints", {}).get("seeds", {})
    verified_ckpts = {}
    for seed in seeds_to_run:
        s_info = ckpt_cfgs.get(str(seed))
        if not s_info:
            raise KeyError(f"Seed {seed} not configured in protocol source_checkpoints")
        c_path = REPO_ROOT / s_info["checkpoint_path"]
        if not c_path.is_file():
            if args.dry_run:
                print(f"[Dry Run] Checkpoint seed {seed} path {c_path} not found.")
                continue
            raise FileNotFoundError(f"Checkpoint for seed {seed} not found at {c_path}")

        act_sha = compute_file_sha256(c_path)
        exp_sha = s_info["expected_sha256"]
        if act_sha != exp_sha:
            raise ValueError(
                f"Checkpoint seed {seed} hash mismatch: actual {act_sha} vs expected {exp_sha}"
            )
        verified_ckpts[seed] = (c_path, act_sha)
        print(f"[Verified] Checkpoint seed {seed}: {c_path.name} (SHA-256 match)")

    if args.dry_run:
        print("Dry run requested: preflight validations passed; stopping before model execution.")
        return

    # Construct real dataloader for selected source patches
    from src.data.evaluation_dataset import KuroEvaluationDataset

    dataset_root = Path(source_cfg["dataset_root"])
    source_dataset = KuroEvaluationDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        allowed_events=meta["approved_source_events"],
        canonical_preevent=protocol["model"]["canonical_preevent"],
        sar_clamp_max=protocol["dataset"]["sar_clamp_max"],
        sar_mean=protocol["dataset"]["sar_mean"],
        sar_std=protocol["dataset"]["sar_std"],
        load_labels=False,  # STRICT LABEL ISOLATION: Fisher NEVER touches ground truth
    )

    # Filter source dataset strictly to the deterministically selected candidate sample IDs
    selected_id_set = set(selected_ids)
    source_dataset.samples = [
        s for s in source_dataset.samples if str(s["sample_id"]) in selected_id_set
    ]
    source_dataset.samples.sort(key=lambda s: str(s["sample_id"]))
    if len(source_dataset.samples) != len(selected_ids):
        raise ValueError(
            f"Dataset filtered sample count ({len(source_dataset.samples)}) does not match selected IDs ({len(selected_ids)})"
        )

    def dataloader_fn(sample_ids_list: List[str]):
        for item in source_dataset:
            yield {
                "sample_id": item["sample_id"],
                "xA": item["xA"].unsqueeze(0),
                "xB": item["xB"].unsqueeze(0),
                "sar_valid_mask": item["valid_mask"].unsqueeze(0),
            }

    m_cfg = protocol["model"]

    for seed in seeds_to_run:
        c_path, act_sha = verified_ckpts[seed]
        print(f"\n--- Computing Fisher for Checkpoint Seed {seed} ---")
        wrapper, _ = SNUNetWrapper.load_from_checkpoint(
            checkpoint_path=str(c_path),
            in_channels=m_cfg["in_channels"],
            out_ch=m_cfg["num_classes"],
            base_channel=m_cfg["base_channel"],
            device=args.device,
        )

        fishers, stats = compute_empirical_fisher_for_checkpoint(
            model=wrapper,
            sample_ids=selected_ids,
            dataloader_fn=dataloader_fn,
            device=args.device,
            accumulation_formula=args.accumulation_formula,
        )

        out_path, meta_path = dest_paths[seed]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "fisher": fishers,
            "metadata": {
                "checkpoint_seed": seed,
                "checkpoint_path": str(c_path),
                "checkpoint_sha256": act_sha,
                "selection_metadata": meta,
                "stats": stats,
                "accumulation_formula": args.accumulation_formula,
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }

        # W9-N01: Atomic writing via temporary files to avoid corrupted partial outputs
        tmp_pt = out_path.with_suffix(".pt.tmp")
        tmp_json = out_path.with_suffix(".json.tmp")
        try:
            torch.save(payload, tmp_pt)
            fisher_sha = compute_file_sha256(tmp_pt)
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump({
                    "seed": seed,
                    "artifact_path": str(out_path),
                    "artifact_sha256": fisher_sha,
                    "checkpoint_path": str(c_path),
                    "checkpoint_sha256": act_sha,
                    "split_config_path": str(split_config_path),
                    "split_config_sha256": meta["split_config_sha256"],
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": meta["manifest_sha256"],
                    "selected_sample_ids": selected_ids,
                    "selected_sample_ids_sha256": meta["selected_sample_ids_sha256"],
                    "stats": stats,
                    "selection_metadata": meta,
                }, f, indent=2)
            os.replace(tmp_pt, out_path)
            os.replace(tmp_json, meta_path)
            print(f"Saved Fisher artifact for Seed {seed}: {out_path} (SHA-256: {fisher_sha})")
        finally:
            if tmp_pt.exists():
                tmp_pt.unlink()
            if tmp_json.exists():
                tmp_json.unlink()


if __name__ == "__main__":
    main()

