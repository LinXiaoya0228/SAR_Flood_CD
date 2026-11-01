"""Evaluation of the frozen source model on the 6 Kuro Siwo validation events."""

import sys
import json
import time
import logging
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import EventMetricAccumulator


def run_val_baseline(config_path: str = "configs/baseline_config.json"):
    config_file = PROJECT_ROOT / config_path
    with open(config_file, "r") as f:
        cfg = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "val_baseline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("val_baseline")

    logger.info("=" * 60)
    logger.info("PHASE 7: FROZEN SIX-EVENT VALIDATION BASELINE")
    logger.info("=" * 60)
    logger.info(f"Official Repo Commit: {cfg['official_repo_commit']}")
    logger.info(f"Validation Events:    {cfg['validation_event_ids']}")
    logger.info(f"Sealed Test Events:   {cfg['sealed_test_event_ids']}")
    logger.info(f"Checkpoint Path:      {cfg['checkpoint_path']}")
    logger.info(f"Channel Order:        {cfg['channel_order']} (Third channel status: {cfg['third_channel_status']})")
    logger.info(f"Canonical Pre-event:  {cfg['canonical_preevent']}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Computation Device:   {device}")
    if device == "cuda":
        logger.info(f"GPU Name:             {torch.cuda.get_device_name(0)}")

    # Load dataset
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

    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    logger.info(f"Total Validation Samples: {len(dataset)} across {len(cfg['validation_event_ids'])} events")
    logger.info(f"DataLoader: Batch Size={batch_size}, Num Workers={num_workers}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Load strictly frozen model
    logger.info("Loading official checkpoint strictly...")
    model, metadata = SNUNetWrapper.load_from_checkpoint(
        cfg["checkpoint_path"],
        in_channels=3,
        out_ch=3,
        base_channel=32,
        device=device,
    )
    safe_meta = {k: v for k, v in metadata.items() if not k.endswith("_dict")}
    logger.info(f"Checkpoint Metadata: {safe_meta}")
    model.freeze()

    # Metric Accumulator
    accumulator = EventMetricAccumulator()

    start_time = time.time()
    logger.info("Starting evaluation across validation batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            event_ids = batch["event_id"].tolist()
            xA = batch["xA"].to(device)
            xB = batch["xB"].to(device)
            labels = batch["label"]
            valid_masks = batch["valid_mask"]

            out = model.predict(xA, xB)
            pred_classes = out["predicted_class"]

            accumulator.update(event_ids, pred_classes, labels, valid_masks)

    elapsed_time = time.time() - start_time
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds ({len(dataset)/elapsed_time:.1f} samples/sec)")

    # 1. Event-Level Table
    event_rows = accumulator.compute_event_table()
    csv_path = output_dir / "val_event_metrics.csv"
    csv_headers = [
        "event_id", "sample_count", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for r in event_rows:
            writer.writerow(r)

    logger.info(f"\nSaved event metrics CSV: {csv_path}")
    logger.info("\n" + "=" * 90)
    logger.info(f"{'Event ID':10s} | {'Samples':7s} | {'Valid Pixels':12s} | {'GT Flood':10s} | {'Pred Flood':10s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'IoU':6s}")
    logger.info("-" * 90)
    for r in event_rows:
        logger.info(
            f"{r['event_id']:<10d} | {r['sample_count']:<7d} | {r['valid_pixels']:<12d} | "
            f"{r['gt_flood_pixels']:<10d} | {r['pred_flood_pixels']:<10d} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['iou']:.4f}"
        )
    logger.info("=" * 90)

    # 2. Macro & Pooled Summary
    summary = accumulator.compute_summary()
    summary["preprocessing_status"] = "inferred (provisional baseline: pre_event_2 + post + DEM v1 stats)"
    summary["checkpoint_provenance_status"] = "inferred (reconstructed from Kuro Siwo commit 043a36a)"
    summary["official_repo_commit"] = cfg["official_repo_commit"]
    summary["evaluation_duration_seconds"] = float(elapsed_time)

    json_path = output_dir / "val_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved validation summary JSON: {json_path}")
    logger.info("Summary Results:")
    logger.info(f"  Macro Precision: {summary['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:    {summary['macro_recall']:.4f}")
    logger.info(f"  Macro F1:        {summary['macro_f1']:.4f}")
    logger.info(f"  Macro IoU:       {summary['macro_iou']:.4f}")
    logger.info(f"  Median F1:       {summary['median_f1']:.4f}")
    logger.info(f"  Median IoU:      {summary['median_iou']:.4f}")
    logger.info(f"  Pooled F1:       {summary['pooled_metrics']['f1']:.4f}")
    logger.info(f"  Pooled IoU:      {summary['pooled_metrics']['iou']:.4f}")
    logger.info(f"  Worst Event F1:  Event {summary['worst_event_f1']['event_id']} (F1={summary['worst_event_f1']['f1']:.4f})")
    logger.info(f"  Events with Zero Predicted Flood: {summary['events_with_zero_predicted_flood']} / {summary['num_events']}")
    logger.info("=" * 60)

    return summary, event_rows


if __name__ == "__main__":
    run_val_baseline()

