"""Evaluation of the frozen source model on the 6 Kuro Siwo validation events

with corrected DEM NoData policy (kuro_dem_sanitize_v1).
Produces:
1. outputs/day2_corrected/val_event_metrics_corrected.csv
2. outputs/day2_corrected/summary.json
3. outputs/day2_corrected/raw_vs_corrected_baseline.csv
"""

import sys
import json
import time
import logging
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.metrics import EventMetricAccumulator, compute_flood_metrics


def run_val_baseline_corrected(
    config_path: str = "configs/baseline_config.json",
    output_dir_str: str = "outputs/day2_corrected"
):
    config_file = PROJECT_ROOT / config_path
    with open(config_file, "r") as f:
        cfg = json.load(f)

    output_dir = PROJECT_ROOT / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "val_baseline_corrected.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("val_baseline_corrected")

    logger.info("=" * 70)
    logger.info("CORRECTED CANONICAL BASELINE EVALUATION (kuro_dem_sanitize_v1)")
    logger.info("=" * 70)
    logger.info(f"Official Repo Commit: {cfg['official_repo_commit']}")
    logger.info(f"Validation Events:    {cfg['validation_event_ids']}")
    logger.info(f"Sealed Test Events:   {cfg['sealed_test_event_ids']}")
    logger.info(f"Checkpoint Path:      {cfg['checkpoint_path']}")
    logger.info(f"Canonical Pre-event:  {cfg['canonical_preevent']}")
    logger.info(f"Preprocessing Vers:   kuro_dem_sanitize_v1")
    logger.info(f"DEM NoData Policy:    < -1000.0 or non-finite -> normalized 0.0 (training mean)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Computation Device:   {device}")
    if device == "cuda":
        logger.info(f"GPU Name:             {torch.cuda.get_device_name(0)}")

    # Instantiate dataset with corrected policy
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
        dem_nodata_threshold=-1000.0,
        dem_replacement_value=0.0,
        preprocessing_version="kuro_dem_sanitize_v1",
    )

    batch_size = cfg.get("batch_size", 16)
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

    # Load frozen checkpoint strictly
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

    accumulator = EventMetricAccumulator()
    dem_invalid_eval_counts = {eid: 0 for eid in cfg["validation_event_ids"]}
    dem_invalid_total_counts = {eid: 0 for eid in cfg["validation_event_ids"]}

    start_time = time.time()
    logger.info("Starting evaluation across validation batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating Corrected")):
            event_ids = batch["event_id"].tolist()
            xA = batch["xA"].to(device)
            xB = batch["xB"].to(device)
            labels = batch["label"]
            valid_masks = batch["valid_mask"]
            dem_inv_masks = batch["dem_invalid_mask"]

            out = model.predict(xA, xB)
            pred_classes = out["predicted_class"]

            accumulator.update(event_ids, pred_classes, labels, valid_masks)

            # Count DEM-invalid pixels
            for i in range(len(event_ids)):
                eid = event_ids[i]
                eval_mask = (valid_masks[i] == 1) & (labels[i] != 3)
                dem_inv_eval = int((dem_inv_masks[i] & eval_mask).sum().item())
                dem_inv_tot = int(dem_inv_masks[i].sum().item())
                dem_invalid_eval_counts[eid] += dem_inv_eval
                dem_invalid_total_counts[eid] += dem_inv_tot

    elapsed_time = time.time() - start_time
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds ({len(dataset)/elapsed_time:.1f} samples/sec)")

    # 1. Save Event Metrics Corrected CSV
    base_rows = accumulator.compute_event_table()
    corrected_rows = []
    for r in base_rows:
        eid = r["event_id"]
        row_dict = dict(r)
        row_dict["dem_invalid_pixels"] = dem_invalid_eval_counts[eid]
        row_dict["total_dem_invalid_pixels"] = dem_invalid_total_counts[eid]
        corrected_rows.append(row_dict)

    csv_path = output_dir / "val_event_metrics_corrected.csv"
    csv_headers = [
        "event_id", "sample_count", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dem_invalid_pixels"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        for r in corrected_rows:
            writer.writerow(r)

    logger.info(f"\nSaved corrected event metrics CSV: {csv_path}")
    logger.info("\n" + "=" * 105)
    logger.info(f"{'Event ID':10s} | {'Samples':7s} | {'Valid Pixels':12s} | {'GT Flood':10s} | {'Pred Flood':10s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'IoU':6s} | {'DEM Inv Pix':11s}")
    logger.info("-" * 105)
    for r in corrected_rows:
        logger.info(
            f"{r['event_id']:<10d} | {r['sample_count']:<7d} | {r['valid_pixels']:<12d} | "
            f"{r['gt_flood_pixels']:<10d} | {r['pred_flood_pixels']:<10d} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['iou']:.4f} | "
            f"{r['dem_invalid_pixels']:<11d}"
        )
    logger.info("=" * 105)

    # 2. Save Summary JSON
    summary = accumulator.compute_summary()
    summary["preprocessing_version"] = "kuro_dem_sanitize_v1"
    summary["preprocessing_policy"] = "DEM values < -1000.0 or non-finite replaced with training mean elevation (normalized 0.0)"
    summary["canonical_preevent"] = cfg["canonical_preevent"]
    summary["dem_nodata_threshold"] = -1000.0
    summary["dem_replacement_value_normalized"] = 0.0
    summary["event_dem_invalid_pixels"] = dem_invalid_eval_counts
    summary["event_dem_invalid_total_pixels"] = dem_invalid_total_counts
    summary["evaluation_duration_seconds"] = float(elapsed_time)

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary JSON: {json_path}")

    # 3. Build Raw vs Corrected Comparison Table
    raw_csv_path = PROJECT_ROOT / "outputs/day2/val_event_metrics.csv"
    raw_rows_by_id = {}
    if raw_csv_path.is_file():
        with open(raw_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows_by_id[int(row["event_id"])] = {
                    k: (float(v) if "." in v else int(v)) for k, v in row.items()
                }

    comparison_rows = []
    for c_row in corrected_rows:
        eid = c_row["event_id"]
        r_row = raw_rows_by_id.get(eid, {})

        raw_prec = float(r_row.get("precision", np.nan))
        raw_rec = float(r_row.get("recall", np.nan))
        raw_f1 = float(r_row.get("f1", np.nan))
        raw_iou = float(r_row.get("iou", np.nan))
        raw_pred = int(r_row.get("pred_flood_pixels", 0))

        cor_prec = float(c_row["precision"])
        cor_rec = float(c_row["recall"])
        cor_f1 = float(c_row["f1"])
        cor_iou = float(c_row["iou"])
        cor_pred = int(c_row["pred_flood_pixels"])

        delta_prec = cor_prec - raw_prec
        delta_rec = cor_rec - raw_rec
        delta_f1 = cor_f1 - raw_f1
        delta_iou = cor_iou - raw_iou
        delta_pred = cor_pred - raw_pred
        pct_change_pred = (delta_pred / raw_pred * 100) if raw_pred > 0 else 0.0

        comp = {
            "event_id": eid,
            "sample_count": c_row["sample_count"],
            "dem_invalid_pixels": c_row["dem_invalid_pixels"],
            "raw_precision": raw_prec,
            "corrected_precision": cor_prec,
            "delta_precision": delta_prec,
            "raw_recall": raw_rec,
            "corrected_recall": cor_rec,
            "delta_recall": delta_rec,
            "raw_f1": raw_f1,
            "corrected_f1": cor_f1,
            "delta_f1": delta_f1,
            "raw_iou": raw_iou,
            "corrected_iou": cor_iou,
            "delta_iou": delta_iou,
            "raw_pred_flood_pixels": raw_pred,
            "corrected_pred_flood_pixels": cor_pred,
            "delta_pred_flood_pixels": delta_pred,
            "pct_change_pred_flood": pct_change_pred,
        }
        comparison_rows.append(comp)

    comp_csv_path = output_dir / "raw_vs_corrected_baseline.csv"
    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        for r in comparison_rows:
            writer.writerow(r)

    logger.info(f"\nSaved raw vs corrected comparison CSV: {comp_csv_path}")
    logger.info("\n" + "=" * 125)
    logger.info(f"{'Event ID':10s} | {'Raw F1':7s} | {'Corr F1':7s} | {'Delta F1':8s} | {'Raw IoU':7s} | {'Corr IoU':8s} | {'Delta IoU':9s} | {'Raw Pred':9s} | {'Corr Pred':9s} | {'Delta Pred':10s}")
    logger.info("-" * 125)
    for r in comparison_rows:
        logger.info(
            f"{r['event_id']:<10d} | {r['raw_f1']:.4f}  | {r['corrected_f1']:.4f}  | "
            f"{r['delta_f1']:+7.4f}  | {r['raw_iou']:.4f}  | {r['corrected_iou']:.4f}   | "
            f"{r['delta_iou']:+8.4f}  | {r['raw_pred_flood_pixels']:<9d} | {r['corrected_pred_flood_pixels']:<9d} | "
            f"{r['delta_pred_flood_pixels']:+10d}"
        )
    logger.info("=" * 125)

    return summary, corrected_rows, comparison_rows


if __name__ == "__main__":
    run_val_baseline_corrected()
