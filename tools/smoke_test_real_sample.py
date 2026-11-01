"""Deterministic single-sample smoke test on official SNUNet checkpoint and Kuro Siwo validation data."""

import sys
import json
from pathlib import Path
from collections import Counter
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import compute_confusion_counts, compute_flood_metrics


def run_smoke_test(config_path: str = "configs/baseline_config.json"):
    config_file = PROJECT_ROOT / config_path
    with open(config_file, "r") as f:
        cfg = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 6: DETERMINISTIC REAL-SAMPLE SMOKE TEST")
    print("=" * 60)

    # Instantiate dataset
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=cfg["validation_event_ids"],
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
    )

    # Deterministically select first sample from event 279
    sample_index = None
    for idx in range(len(dataset)):
        if dataset.samples[idx]["event_id"] == 279:
            sample_index = idx
            break

    if sample_index is None:
        raise RuntimeError("No sample found for event 279!")

    sample = dataset[sample_index]
    sample_id = sample["sample_id"]
    event_id = sample["event_id"]

    print(f"\n1. Selected Sample ID: {sample_id}")
    print(f"   Event ID: {event_id} (Index in dataset: {sample_index})")

    print("\n2. Component Paths:")
    for comp, path in sample["component_paths"].items():
        print(f"   {comp:8s}: {path}")

    print("\n3. Raw and Normalized Shapes:")
    print(f"   xA shape (pre-event 3ch):  {tuple(sample['xA'].shape)}")
    print(f"   xB shape (post-event 3ch): {tuple(sample['xB'].shape)}")
    print(f"   label shape:               {tuple(sample['label'].shape)}")
    print(f"   valid_mask shape:          {tuple(sample['valid_mask'].shape)}")

    print("\n4. Raw vs Normalized Channel Statistics:")
    print("   Raw (min, max, mean, std):")
    for comp, stats in sample["raw_stats"].items():
        print(f"     {comp:8s}: min={stats[0]:.4f}, max={stats[1]:.4f}, mean={stats[2]:.4f}, std={stats[3]:.4f}")
    
    xA_np = sample["xA"].numpy()
    xB_np = sample["xB"].numpy()
    print("   Normalized xA (pre-event branch):")
    channel_names = ["VV", "VH", "DEM"]
    for c in range(3):
        print(f"     Ch {c} ({channel_names[c]}): min={xA_np[c].min():.4f}, max={xA_np[c].max():.4f}, mean={xA_np[c].mean():.4f}, std={xA_np[c].std():.4f}")
    print("   Normalized xB (post-event branch):")
    for c in range(3):
        print(f"     Ch {c} ({channel_names[c]}): min={xB_np[c].min():.4f}, max={xB_np[c].max():.4f}, mean={xB_np[c].mean():.4f}, std={xB_np[c].std():.4f}")

    # Label value counts
    label_np = sample["label"].numpy()
    valid_np = sample["valid_mask"].numpy()
    raw_label_counts = dict(Counter(label_np.flatten().tolist()))
    double_valid_mask = (valid_np == 1) & (label_np != cfg["ignore_index"])
    valid_pixel_count = int(np.count_nonzero(double_valid_mask))
    valid_label_counts = dict(Counter(label_np[double_valid_mask].tolist()))

    print(f"\n5. Raw Label-Value Counts: {raw_label_counts}")
    print(f"6. Valid Pixel Counts:     {valid_pixel_count} / {label_np.size} ({valid_pixel_count / label_np.size * 100:.2f}%)")
    print(f"   Filtered Valid Label Counts: {valid_label_counts}")

    # Load Model Strictly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n7. Loading official checkpoint strictly on device '{device}': {cfg['checkpoint_path']}")
    model, metadata = SNUNetWrapper.load_from_checkpoint(
        cfg["checkpoint_path"],
        in_channels=3,
        out_ch=3,
        base_channel=32,
        device=device,
    )
    safe_meta = {k: v for k, v in metadata.items() if not k.endswith("_dict")}
    print(f"   Checkpoint metadata: {safe_meta}")

    # Forward pass
    print("\n8. Executing exactly one frozen forward pass...")
    xA = sample["xA"].unsqueeze(0).to(device)
    xB = sample["xB"].unsqueeze(0).to(device)

    out = model.predict(xA, xB)
    logits = out["logits"].squeeze(0).cpu()
    probs = out["probabilities"].squeeze(0).cpu()
    pred_class = out["predicted_class"].squeeze(0).cpu().numpy()
    flood_prob = out["flood_probability"].squeeze(0).cpu().numpy()

    print(f"9. Model Output Logits Shape: {tuple(logits.shape)}")
    logits_finite = bool(torch.isfinite(logits).all())
    probs_finite = bool(torch.isfinite(probs).all())
    print(f"10. Finite Check: Logits Finite: {logits_finite}, Probabilities Finite: {probs_finite}")

    # Class predictions
    pred_counts = dict(Counter(pred_class.flatten().tolist()))
    total_pixels = pred_class.size
    pred_proportions = {k: v / total_pixels for k, v in pred_counts.items()}
    print(f"11. Predicted Counts:      {pred_counts}")
    print(f"    Predicted Proportions: { {k: f'{v*100:.2f}%' for k, v in pred_proportions.items()} }")

    # Binary flood metrics on valid pixels only
    confusion = compute_confusion_counts(pred_class, label_np, valid_np)
    metrics = compute_flood_metrics(confusion)
    print("\n12. Binary Flood Metrics on Valid Pixels:")
    print(f"    TP: {confusion['tp']}, FP: {confusion['fp']}, FN: {confusion['fn']}, TN: {confusion['tn']}")
    print(f"    GT Flood Pixels: {confusion['gt_flood_pixels']}, Pred Flood Pixels: {confusion['pred_flood_pixels']}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    IoU:       {metrics['iou']:.4f}")

    # Sanity checks
    warnings = []
    if len(pred_counts) == 1:
        warnings.append("WARNING: All pixels predicted as a single class (collapse)!")
    if not logits_finite or not probs_finite:
        warnings.append("FATAL: Non-finite logits or probabilities encountered!")
    if valid_pixel_count == 0:
        warnings.append("FATAL: Zero valid pixels in sample!")
    if pred_proportions.get(2, 0) > 0.8:
        warnings.append("WARNING: Flood prediction covers more than 80% of the image!")
    if label_np.shape != pred_class.shape:
        warnings.append("FATAL: Shape mismatch between ground truth and prediction!")

    print("\nSanity Check Results:")
    if not warnings:
        print("    ALL SANITY CHECKS PASSED.")
    else:
        for w in warnings:
            print(f"    {w}")

    # 13. Visualizations
    print("\n13. Saving multi-panel visualization...")
    cmap_label = ListedColormap(["black", "deepskyblue", "magenta", "gray"])

    # Helpers to display SAR cleanly
    def norm_disp(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr)
        p2, p98 = np.percentile(finite, [2, 98])
        if p98 <= p2:
            return np.zeros_like(arr)
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    pre_vv_raw = sample["raw_stats"]["sl2_vv"]  # canonical is pre2
    # Load raw arrays for visualization
    with rasterio.open(sample["component_paths"]["SL2_IVV"]) as s:
        disp_pre_vv = s.read(1)
    with rasterio.open(sample["component_paths"]["SL2_IVH"]) as s:
        disp_pre_vh = s.read(1)
    with rasterio.open(sample["component_paths"]["MS1_IVV"]) as s:
        disp_post_vv = s.read(1)
    with rasterio.open(sample["component_paths"]["MS1_IVH"]) as s:
        disp_post_vh = s.read(1)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: SAR inputs
    axes[0, 0].imshow(norm_disp(disp_pre_vv), cmap="gray")
    axes[0, 0].set_title(f"Pre-event VV (SL2)\nEvent {event_id}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(norm_disp(disp_pre_vh), cmap="gray")
    axes[0, 1].set_title("Pre-event VH (SL2)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(norm_disp(disp_post_vv), cmap="gray")
    axes[0, 2].set_title("Post-event VV (MS1)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(norm_disp(disp_post_vh), cmap="gray")
    axes[0, 3].set_title("Post-event VH (MS1)")
    axes[0, 3].axis("off")

    # Row 2: Labels and Valid Mask
    axes[1, 0].imshow(label_np, cmap=cmap_label, vmin=0, vmax=3)
    axes[1, 0].set_title("GT 3-Class Label\n(0:NW, 1:PW, 2:Flood, 3:Invalid)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(valid_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Valid Mask (MK0_MNA)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(double_valid_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Double Valid Mask\n(valid_mask==1 & label!=3)")
    axes[1, 2].axis("off")

    # DEM
    with rasterio.open(sample["component_paths"]["MK0_DEM"]) as s:
        disp_dem = s.read(1)
    axes[1, 3].imshow(disp_dem, cmap="terrain")
    axes[1, 3].set_title("Elevation (DEM)")
    axes[1, 3].axis("off")

    # Row 3: Predictions and Metrics
    axes[2, 0].imshow(pred_class, cmap=cmap_label, vmin=0, vmax=3)
    axes[2, 0].set_title("Predicted 3-Class Label")
    axes[2, 0].axis("off")

    im_prob = axes[2, 1].imshow(flood_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2, 1].set_title("Predicted Flood Probability")
    axes[2, 1].axis("off")
    fig.colorbar(im_prob, ax=axes[2, 1], fraction=0.046, pad=0.04)

    target_flood = ((label_np == 2) & double_valid_mask).astype(np.uint8)
    axes[2, 2].imshow(target_flood, cmap="gray", vmin=0, vmax=1)
    axes[2, 2].set_title("Binary Flood Target (GT==2)")
    axes[2, 2].axis("off")

    pred_flood = ((pred_class == 2) & double_valid_mask).astype(np.uint8)
    axes[2, 3].imshow(pred_flood, cmap="gray", vmin=0, vmax=1)
    axes[2, 3].set_title(f"Binary Flood Pred\n(F1={metrics['f1']:.3f}, IoU={metrics['iou']:.3f})")
    axes[2, 3].axis("off")

    fig.suptitle(f"ChangeGuard Smoke Test - Event {event_id} - Sample {sample_id[:16]}...", fontsize=14, fontweight="bold")
    plt.tight_layout()

    vis_path = output_dir / f"smoke_test_event_{event_id}_{sample_id[:8]}.png"
    plt.savefig(vis_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    Visualization saved: {vis_path}")

    # 14. Save JSON summary
    summary_path = output_dir / "smoke_test_summary.json"
    summary_data = {
        "event_id": event_id,
        "sample_id": sample_id,
        "device": device,
        "shapes": {
            "xA": list(sample["xA"].shape),
            "xB": list(sample["xB"].shape),
            "label": list(sample["label"].shape),
            "valid_mask": list(sample["valid_mask"].shape),
            "output_logits": list(logits.shape),
        },
        "raw_stats": sample["raw_stats"],
        "valid_pixels": valid_pixel_count,
        "total_pixels": int(label_np.size),
        "raw_label_counts": raw_label_counts,
        "valid_label_counts": valid_label_counts,
        "predicted_counts": pred_counts,
        "predicted_proportions": pred_proportions,
        "confusion": confusion,
        "metrics": metrics,
        "warnings": warnings,
        "visualization_path": str(vis_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"14. JSON summary saved: {summary_path}")

    return summary_data


if __name__ == "__main__":
    run_smoke_test()
