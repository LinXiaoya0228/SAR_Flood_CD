#!/usr/bin/env python3
"""Controlled Source-Only Baseline Evaluation CLI for ChangeGuard.

Protocol: Week 7A Source-Only Baseline Evaluation
Governing Config: configs/evaluation/source_baseline_tta_dev_protocol.json

Guarantees:
1. Pure source-only evaluation: model.eval(), torch.no_grad().
2. Zero optimizer steps, zero backward passes, zero batch-norm running statistics adaptation.
3. Cryptographic parameter and buffer hashing before and after inference to verify state immutability.
4. Sealed test events [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013] strictly forbidden.
5. Deterministic sample error visualization (2 fixed samples per event across all seeds).
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.data.evaluation_dataset import (
    KuroEvaluationDataset,
    FORBIDDEN_SEALED_TEST_EVENTS,
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    TOTAL_EXPECTED_TTA_DEV_SAMPLES,
)
from src.evaluation.evaluator import SourceModelEvaluator, aggregate_three_seed_metrics


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Retrieves current Git commit and working tree cleanliness."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        is_clean = len(status) == 0
        return {"commit": commit, "is_clean": is_clean, "status_summary": status[:500]}
    except Exception as e:
        return {"commit": "unknown", "is_clean": False, "error": str(e)}


def validate_preflight(config_path: Path) -> Dict[str, Any]:
    """Validates configuration, checkpoint existence, SHA-256 hashes, and model architecture."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Validate target events disjointness from sealed test partition
    target_events = set(cfg["dataset"]["target_events"])
    forbidden = set(cfg["dataset"]["forbidden_sealed_test_events"])
    if not target_events.isdisjoint(forbidden):
        intersection = sorted(list(target_events & forbidden))
        raise RuntimeError(
            f"Preflight validation failed: Target events include sealed test events: {intersection}"
        )

    # Validate expected sample counts
    expected_counts = {int(k): int(v) for k, v in cfg["dataset"]["expected_event_counts"].items()}
    total_expected = sum(expected_counts.values())
    if total_expected != cfg["dataset"]["expected_total_samples"]:
        raise ValueError(
            f"Total sample count mismatch in config: sum={total_expected} vs "
            f"expected_total_samples={cfg['dataset']['expected_total_samples']}"
        )

    # Validate dataset files
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    dataset_exists = dataset_root.is_dir()
    manifest_exists = manifest_path.is_file()

    # Validate checkpoints
    checkpoint_results: Dict[str, Any] = {}
    for seed_key, ckpt_info in cfg["checkpoints"].items():
        ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
        expected_hash = ckpt_info["expected_sha256"]
        if not ckpt_path.is_file():
            checkpoint_results[seed_key] = {
                "path": str(ckpt_path),
                "exists": False,
                "hash_match": False,
                "error": "File does not exist",
            }
            continue

        actual_hash = compute_file_sha256(ckpt_path)
        hash_match = actual_hash == expected_hash
        checkpoint_results[seed_key] = {
            "path": str(ckpt_path),
            "exists": True,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_match": hash_match,
        }

    # Verify model instantiation and forward pass with dummy tensor on CPU
    m_cfg = cfg["model"]
    dummy_model = SNUNetWrapper(
        in_channels=m_cfg["in_channels"],
        out_ch=m_cfg["num_classes"],
        base_channel=m_cfg["base_channel"],
    )
    xA_dummy = torch.zeros(1, m_cfg["in_channels"], 64, 64)
    xB_dummy = torch.zeros(1, m_cfg["in_channels"], 64, 64)
    with torch.no_grad():
        out_dummy = dummy_model(xA_dummy, xB_dummy)
    forward_ok = out_dummy.shape == (1, m_cfg["num_classes"], 64, 64)

    return {
        "config_path": str(config_path),
        "protocol_version": cfg.get("protocol_version"),
        "target_events": sorted(list(target_events)),
        "forbidden_sealed_test_events": sorted(list(forbidden)),
        "target_events_disjoint_from_sealed": True,
        "dataset_root_exists": dataset_exists,
        "manifest_path_exists": manifest_exists,
        "checkpoint_results": checkpoint_results,
        "architecture_forward_test_passed": forward_ok,
    }


def render_error_visualization(
    sample_id: str,
    event_id: int,
    xA: np.ndarray,
    xB: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    save_path: Path,
):
    """Renders a 5-panel deterministic error visualization figure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Double-masking
    v_mask = (valid_mask == 1) & (gt != 3)

    # Confusion categories for target flood class 2
    tp = (pred == 2) & (gt == 2) & v_mask
    fp = (pred == 2) & (gt != 2) & v_mask
    fn = (pred != 2) & (gt == 2) & v_mask
    tn = (pred != 2) & (gt != 2) & v_mask
    inv = ~v_mask

    # Error RGB map
    # TP = Green [0, 200, 0]
    # FP = Red [230, 30, 30]
    # FN = Yellow [240, 180, 0]
    # TN = Dark Gray [40, 40, 40]
    # Invalid = Light Gray [220, 220, 220]
    H, W = gt.shape
    error_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    error_rgb[tn] = [40, 40, 40]
    error_rgb[tp] = [40, 200, 40]
    error_rgb[fp] = [230, 40, 40]
    error_rgb[fn] = [240, 180, 0]
    error_rgb[inv] = [220, 220, 220]

    # Ground truth RGB
    # 0 = Background [50, 50, 50]
    # 1 = Perm Water [30, 100, 220]
    # 2 = Flood [40, 200, 40]
    # 3 = Invalid [220, 220, 220]
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    gt_rgb[gt == 0] = [50, 50, 50]
    gt_rgb[gt == 1] = [30, 100, 220]
    gt_rgb[gt == 2] = [40, 200, 40]
    gt_rgb[gt == 3] = [220, 220, 220]
    gt_rgb[valid_mask == 0] = [220, 220, 220]

    # Prediction RGB
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    pred_rgb[pred == 0] = [50, 50, 50]
    pred_rgb[pred == 1] = [30, 100, 220]
    pred_rgb[pred == 2] = [40, 200, 40]
    pred_rgb[inv] = [220, 220, 220]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))

    # Pre-event VV
    im0 = axes[0].imshow(xA[0], cmap="gray")
    axes[0].set_title("Pre-event SAR (VV)")
    axes[0].axis("off")

    # Post-event VV
    im1 = axes[1].imshow(xB[0], cmap="gray")
    axes[1].set_title("Post-event SAR (VV)")
    axes[1].axis("off")

    # Ground truth
    axes[2].imshow(gt_rgb)
    axes[2].set_title("Ground Truth\n(Grn=Flood, Blu=Perm)")
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(pred_rgb)
    axes[3].set_title("Prediction\n(Grn=Flood, Blu=Perm)")
    axes[3].axis("off")

    # Error Map
    axes[4].imshow(error_rgb)
    tp_c = int(np.count_nonzero(tp))
    fp_c = int(np.count_nonzero(fp))
    fn_c = int(np.count_nonzero(fn))
    p_f1 = (2 * tp_c) / (2 * tp_c + fp_c + fn_c + 1e-7) if (tp_c + fp_c + fn_c) > 0 else 0.0
    axes[4].set_title(f"Error Map (F1: {p_f1:.3f})\nGrn:TP, Red:FP, Ylw:FN")
    axes[4].axis("off")

    plt.suptitle(
        f"Event {event_id} | Sample {sample_id} | TP: {tp_c}, FP: {fp_c}, FN: {fn_c}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(args: argparse.Namespace):
    """Main evaluation routine across requested seeds."""
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    # Step 1: Preflight validation
    preflight = validate_preflight(config_path)

    if args.validate_only:
        print("=" * 70)
        print("WEEK 7A SOURCE-ONLY EVALUATION PREFLIGHT AUDIT (--validate_only)")
        print("=" * 70)
        print(f"Protocol: {preflight.get('protocol_version')}")
        print(f"Config path: {preflight.get('config_path')}")
        print(f"Target events (6): {preflight.get('target_events')}")
        print(f"Sealed test quarantine disjoint: {preflight.get('target_events_disjoint_from_sealed')}")
        print(f"Dataset root exists: {preflight.get('dataset_root_exists')}")
        print(f"Manifest path exists: {preflight.get('manifest_path_exists')}")
        print(f"Model architecture forward test: {'PASSED' if preflight.get('architecture_forward_test_passed') else 'FAILED'}")
        print("\nCheckpoint verification:")
        all_ckpts_ok = True
        for k, v in preflight["checkpoint_results"].items():
            status = "MATCH" if v.get("hash_match") else "MISMATCH / ERROR"
            if not v.get("hash_match"):
                all_ckpts_ok = False
            print(f"  {k}: {status} (path: {v.get('path')})")
        print("=" * 70)
        if not all_ckpts_ok or not preflight.get("architecture_forward_test_passed"):
            sys.exit(1)
        print("PREFLIGHT AUDIT PASSED: Evaluation setup is valid and frozen.")
        sys.exit(0)

    # Step 2: Initialize Dataset and DataLoader
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    target_events = [int(e) for e in cfg["dataset"]["target_events"]]
    expected_counts = {int(k): int(v) for k, v in cfg["dataset"]["expected_event_counts"].items()}
    output_dir = Path(args.output_dir or cfg["output"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading KuroSiwo evaluation dataset for events {target_events}...")
    dataset = KuroEvaluationDataset(
        dataset_root=cfg["dataset"]["dataset_root"],
        manifest_path=cfg["dataset"]["manifest_path"],
        allowed_events=target_events,
        canonical_preevent=cfg["model"]["canonical_preevent"],
        sar_clamp_max=cfg["dataset"]["sar_clamp_max"],
        sar_mean=tuple(cfg["dataset"]["sar_mean"]),
        sar_std=tuple(cfg["dataset"]["sar_std"]),
        load_labels=True,
    )
    print(f"Loaded {len(dataset)} evaluation samples. Expected: {cfg['dataset']['expected_total_samples']}.")
    if len(dataset) != cfg["dataset"]["expected_total_samples"]:
        raise ValueError(
            f"Sample count mismatch: Loaded {len(dataset)} vs expected {cfg['dataset']['expected_total_samples']}"
        )

    # Determine fixed deterministic visualization sample IDs (2 per event: index 0 and index N // 2)
    event_to_samples: Dict[int, List[Dict[str, Any]]] = {}
    for item in dataset.samples:
        eid = int(item["event_id"])
        event_to_samples.setdefault(eid, []).append(item)

    vis_sample_ids: Set[str] = set()
    for eid in sorted(event_to_samples.keys()):
        s_list = sorted(event_to_samples[eid], key=lambda x: str(x["sample_id"]))
        n = len(s_list)
        if n > 0:
            vis_sample_ids.add(str(s_list[0]["sample_id"]))
            vis_sample_ids.add(str(s_list[n // 2]["sample_id"]))
    print(f"Selected {len(vis_sample_ids)} deterministic samples across {len(event_to_samples)} events for visualization.")

    dl_cfg = cfg["dataloader"]
    batch_size = args.batch_size or dl_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else dl_cfg["num_workers"]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and dl_cfg.get("pin_memory", True),
        drop_last=False,
    )

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Determine seeds to evaluate
    seeds_to_run = args.seeds if args.seeds is not None else cfg["seeds"]
    seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds_to_run:
        seed_key = f"seed{seed}"
        if seed_key not in cfg["checkpoints"]:
            raise KeyError(f"Seed {seed} not configured in checkpoint list!")

        ckpt_info = cfg["checkpoints"][seed_key]
        ckpt_path = REPO_ROOT / ckpt_info["checkpoint_path"]
        print(f"\n{'=' * 60}")
        print(f"EVALUATING MODEL A (SEED {seed})")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        # Checkpoint hash verification
        ckpt_hash = compute_file_sha256(ckpt_path)
        if ckpt_hash != ckpt_info["expected_sha256"]:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch for seed {seed}!\n"
                f"Expected: {ckpt_info['expected_sha256']}\n"
                f"Actual:   {ckpt_hash}"
            )

        # Load model strictly
        wrapper, ckpt_meta = SNUNetWrapper.load_from_checkpoint(
            str(ckpt_path),
            in_channels=cfg["model"]["in_channels"],
            out_ch=cfg["model"]["num_classes"],
            base_channel=cfg["model"]["base_channel"],
            device=str(device),
        )

        evaluator = SourceModelEvaluator(wrapper, device=device)
        res = evaluator.evaluate(data_loader, expected_counts=expected_counts)

        # Save event-level metrics CSV
        csv_path = output_dir / f"source_baseline_seed{seed}_event_metrics.csv"
        fieldnames = [
            "event_id", "sample_count", "invalid_only_samples_count", "valid_pixels",
            "gt_flood_pixels", "pred_flood_pixels", "pred_flood_fraction",
            "pred_background_pixels", "pred_permanent_water_pixels",
            "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "iou",
            "is_evaluable", "evaluation_status"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in res["event_metrics"]:
                cleaned_row = {k: ("" if v is None else v) for k, v in row.items()}
                writer.writerow(cleaned_row)
        print(f"Saved event metrics CSV: {csv_path}")

        # Save summary JSON
        summary_payload = {
            "protocol_version": cfg.get("protocol_version"),
            "seed": seed,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "training_commit": ckpt_meta.get("git_commit"),
            "pre_inference_model_hash": res["pre_inference_model_hash"],
            "post_inference_model_hash": res["post_inference_model_hash"],
            "model_immutability_verified": res["model_immutability_verified"],
            "quarantine_sealed_test_zero_leakage": res["quarantine_sealed_test_zero_leakage"],
            "elapsed_seconds": res["elapsed_seconds"],
            "total_evaluated_samples": res["total_evaluated_samples"],
            "expected_total_samples": res["expected_total_samples"],
            "sample_counts_verified": res["sample_counts_verified"],
            "technical_validity": res.get("technical_validity", {}),
            "technical_validity_passed": res.get("technical_validity_passed", False),
            "evaluable_events_count": res.get("evaluable_events_count", len(res["event_metrics"])),
            "all_events_evaluable": res.get("all_events_evaluable", True),
            "sample_valid_pixel_counts": res.get("sample_valid_pixel_counts", {}),
            "equal_event_macro": res["equal_event_macro"],
            "pooled_metrics": res["pooled_metrics"],
            "worst_event": res["worst_event"],
            "collapse_audit": res.get("collapse_audit", {}),
            "event_metrics": res["event_metrics"],
        }
        json_path = output_dir / f"source_baseline_seed{seed}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(summary_payload, f_json, indent=2)
        print(f"Saved summary JSON: {json_path}")

        # Generate deterministic visualizations for fixed samples
        if cfg["output"].get("visualization", {}).get("enabled", True):
            vis_dir = output_dir / "visualizations" / f"seed{seed}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generating deterministic error visualizations under {vis_dir}...")

            # Run a quick pass to extract and render visualizations for vis_sample_ids
            # We can use dataset indexed access for exact visual samples
            wrapper.eval()
            for idx, item in enumerate(dataset.samples):
                sid = str(item["sample_id"])
                if sid in vis_sample_ids:
                    sample_tensor = dataset[idx]
                    xA = sample_tensor["xA"].unsqueeze(0).to(device)
                    xB = sample_tensor["xB"].unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = wrapper(xA, xB)
                        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

                    xA_np = sample_tensor["xA"].numpy()
                    xB_np = sample_tensor["xB"].numpy()
                    gt_np = sample_tensor["label"].numpy()
                    vm_np = sample_tensor["valid_mask"].numpy()
                    eid = int(item["event_id"])

                    fig_path = vis_dir / f"event_{eid}_{sid}.png"
                    render_error_visualization(
                        sample_id=sid,
                        event_id=eid,
                        xA=xA_np,
                        xB=xB_np,
                        gt=gt_np,
                        pred=pred,
                        valid_mask=vm_np,
                        save_path=fig_path,
                    )

        seed_results[seed] = summary_payload

    # Step 3: Multi-seed aggregation if all 3 seeds were evaluated
    if len(seed_results) == 3 and {0, 1, 2}.issubset(seed_results.keys()):
        print("\nAggregating three-seed results (Seed 0, 1, 2)...")
        seed_list = [seed_results[0], seed_results[1], seed_results[2]]
        agg_summary = aggregate_three_seed_metrics(seed_list)

        agg_json_path = output_dir / "source_baseline_three_seed_summary.json"
        with open(agg_json_path, "w", encoding="utf-8") as f_agg:
            json.dump(agg_summary, f_agg, indent=2)
        print(f"Saved three-seed summary: {agg_json_path}")

    # Step 4: Write run manifest
    git_info = get_git_info(REPO_ROOT)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": cfg.get("protocol_version"),
        "git_commit": git_info["commit"],
        "git_clean": git_info["is_clean"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_path": str(config_path),
        "config_sha256": compute_file_sha256(config_path),
        "evaluated_seeds": seeds_to_run,
        "artifacts_generated": [
            str(p.relative_to(REPO_ROOT)) for p in output_dir.glob("**/*") if p.is_file()
        ],
    }
    manifest_path = output_dir / "source_baseline_evaluation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)
    print(f"Saved evaluation manifest: {manifest_path}")
    print("\nSource baseline evaluation completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen source model baseline on TTA development events.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/source_baseline_tta_dev_protocol.json",
        help="Path to evaluation protocol configuration JSON",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to evaluate (default: [0, 1, 2] from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Dataloader workers override",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory override",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Perform preflight verification of config, checkpoints, and model without running evaluation",
    )
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
