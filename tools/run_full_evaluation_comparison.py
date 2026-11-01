"""Comprehensive and unified baseline evaluation runner with strict completeness verification.

Evaluates both:
1. official-compatible raw baseline (unhandled DEM NoData, original behavior)
2. corrected canonical baseline (kuro_dem_sanitize_v1)

Guarantees:
- Exactly 2,826 samples evaluated per mode across all 6 validation events
- Identical DataLoader configuration, batch size (16), shuffle=False, worker count (4)
- Strict completeness validation (raises loudly on any count mismatch, duplicate, or missing sample ID)
- Produces:
  * outputs/day2_corrected/raw_full_val_event_metrics.csv
  * outputs/day2_corrected/corrected_full_val_event_metrics.csv
  * outputs/day2_corrected/raw_vs_corrected_full_baseline.csv
  * outputs/day2_corrected/evaluation_completeness.json
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
from src.evaluation.metrics import EventMetricAccumulator
from src.evaluation.completeness import CompletenessTracker


EXPECTED_VAL_COUNTS = {
    279: 2381,
    437: 169,
    520: 92,
    559: 42,
    1111003: 19,
    1111008: 123,
}
TOTAL_EXPECTED_SAMPLES = sum(EXPECTED_VAL_COUNTS.values())  # 2826


def evaluate_single_mode(
    mode_name: str,
    preprocessing_version: str,
    cfg: dict,
    model: SNUNetWrapper,
    device: str,
    logger: logging.Logger,
):
    logger.info("\n" + "=" * 80)
    logger.info(f"STARTING FULL EVALUATION: {mode_name} (preprocessing_version={preprocessing_version})")
    logger.info("=" * 80)

    # 1. Dataset instantiation
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
        preprocessing_version=preprocessing_version,
    )

    # 2. Extract and assert expected samples from manifest
    expected_samples_per_event = {}
    for eid in cfg["validation_event_ids"]:
        e_samples = [s["sample_id"] for s in dataset.samples if s["event_id"] == eid]
        assert len(e_samples) == EXPECTED_VAL_COUNTS[eid], (
            f"Manifest sample count mismatch for event {eid}: expected {EXPECTED_VAL_COUNTS[eid]}, got {len(e_samples)}"
        )
        expected_samples_per_event[eid] = e_samples

    assert len(dataset) == TOTAL_EXPECTED_SAMPLES, (
        f"Total dataset length mismatch: expected {TOTAL_EXPECTED_SAMPLES}, got {len(dataset)}"
    )

    # 3. Setup DataLoader and Completeness Tracker
    batch_size = cfg.get("batch_size", 16)
    num_workers = cfg.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    tracker = CompletenessTracker(expected_samples_per_event, allow_partial=False)
    accumulator = EventMetricAccumulator()
    dem_invalid_eval_counts = {eid: 0 for eid in cfg["validation_event_ids"]}
    dem_invalid_total_counts = {eid: 0 for eid in cfg["validation_event_ids"]}

    # 4. Evaluation Loop
    start_time = time.time()
    logger.info(f"Running inference over {len(dataset)} samples ({len(loader)} batches)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {mode_name}"):
            event_ids = batch["event_id"].tolist()
            sample_ids = batch["sample_id"]
            xA = batch["xA"].to(device)
            xB = batch["xB"].to(device)
            labels = batch["label"]
            valid_masks = batch["valid_mask"]
            dem_inv_masks = batch["dem_invalid_mask"]

            out = model.predict(xA, xB)
            pred_classes = out["predicted_class"]

            accumulator.update(event_ids, pred_classes, labels, valid_masks)

            # Valid pixel count per sample for completeness tracking
            batch_valid_pix = []
            for i in range(len(event_ids)):
                eid = event_ids[i]
                eval_mask = (valid_masks[i] == 1) & (labels[i] != 3)
                v_count = int(eval_mask.sum().item())
                batch_valid_pix.append(v_count)

                dem_inv_eval = int((dem_inv_masks[i] & eval_mask).sum().item())
                dem_inv_tot = int(dem_inv_masks[i].sum().item())
                dem_invalid_eval_counts[eid] += dem_inv_eval
                dem_invalid_total_counts[eid] += dem_inv_tot

            tracker.update(event_ids, sample_ids, batch_valid_pix)

    elapsed_time = time.time() - start_time
    logger.info(f"Completed {mode_name} in {elapsed_time:.2f}s ({len(dataset)/elapsed_time:.1f} samples/sec)")

    # 5. Strict completeness verification
    logger.info("Executing strict completeness assertion...")
    completeness_report = tracker.validate_or_raise()
    logger.info("Completeness assertion PASSED: 100% of samples processed, 0 duplicates, 0 missing.")

    # 6. Build event rows
    base_rows = accumulator.compute_event_table()
    rows = []
    for r in base_rows:
        eid = r["event_id"]
        row_dict = dict(r)
        row_dict["dem_invalid_pixels"] = dem_invalid_eval_counts[eid]
        row_dict["total_dem_invalid_pixels"] = dem_invalid_total_counts[eid]
        rows.append(row_dict)

    summary = accumulator.compute_summary()
    summary["mode_name"] = mode_name
    summary["preprocessing_version"] = preprocessing_version
    summary["elapsed_seconds"] = float(elapsed_time)
    summary["dem_invalid_pixels_eval"] = dem_invalid_eval_counts
    summary["dem_invalid_pixels_total"] = dem_invalid_total_counts

    return rows, summary, completeness_report


def main():
    config_file = PROJECT_ROOT / "configs/baseline_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    output_dir = PROJECT_ROOT / "outputs/day2_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run_full_evaluation_comparison.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("eval_comparison")

    logger.info("================================================================================")
    logger.info("UNIFIED FULL VALIDATION COMPARISON: RAW VS CORRECTED CANONICAL BASELINE")
    logger.info("================================================================================")
    logger.info(f"Official Repo Commit: {cfg['official_repo_commit']}")
    logger.info(f"Validation Events:    {cfg['validation_event_ids']}")
    logger.info(f"Sealed Test Events:   {cfg['sealed_test_event_ids']}")
    logger.info(f"Checkpoint Path:      {cfg['checkpoint_path']}")
    logger.info(f"Canonical Pre-event:  {cfg['canonical_preevent']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Computation Device:   {device}")
    if device == "cuda":
        logger.info(f"GPU Name:             {torch.cuda.get_device_name(0)}")

    # Load frozen checkpoint once
    logger.info("Loading official checkpoint strictly...")
    model, metadata = SNUNetWrapper.load_from_checkpoint(
        cfg["checkpoint_path"],
        in_channels=3,
        out_ch=3,
        base_channel=32,
        device=device,
    )
    model.freeze()

    # Run 1: Official-Compatible Raw Baseline
    raw_rows, raw_summary, raw_completeness = evaluate_single_mode(
        mode_name="official-compatible raw baseline",
        preprocessing_version="kuro_raw",
        cfg=cfg,
        model=model,
        device=device,
        logger=logger,
    )

    # Run 2: Corrected Canonical Baseline
    corr_rows, corr_summary, corr_completeness = evaluate_single_mode(
        mode_name="corrected canonical baseline",
        preprocessing_version="kuro_dem_sanitize_v1",
        cfg=cfg,
        model=model,
        device=device,
        logger=logger,
    )

    # 1. Save raw full metrics CSV
    raw_csv_path = output_dir / "raw_full_val_event_metrics.csv"
    csv_headers = [
        "event_id", "sample_count", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dem_invalid_pixels"
    ]
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        for r in raw_rows:
            writer.writerow(r)
    logger.info(f"\nSaved raw full event metrics: {raw_csv_path}")

    # 2. Save corrected full metrics CSV
    corr_csv_path = output_dir / "corrected_full_val_event_metrics.csv"
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        for r in corr_rows:
            writer.writerow(r)
    logger.info(f"Saved corrected full event metrics: {corr_csv_path}")

    # 3. Save evaluation completeness JSON
    completeness_path = output_dir / "evaluation_completeness.json"
    completeness_payload = {
        "raw_baseline_completeness": raw_completeness,
        "corrected_baseline_completeness": corr_completeness,
        "validation_events": cfg["validation_event_ids"],
        "expected_counts": EXPECTED_VAL_COUNTS,
        "total_expected_samples": TOTAL_EXPECTED_SAMPLES,
    }
    with open(completeness_path, "w") as f:
        json.dump(completeness_payload, f, indent=2)
    logger.info(f"Saved evaluation completeness JSON: {completeness_path}")

    # 4. Build fair comparison table
    raw_by_id = {r["event_id"]: r for r in raw_rows}
    comparison_rows = []
    for c_row in corr_rows:
        eid = c_row["event_id"]
        r_row = raw_by_id[eid]

        raw_prec = float(r_row["precision"])
        raw_rec = float(r_row["recall"])
        raw_f1 = float(r_row["f1"])
        raw_iou = float(r_row["iou"])
        raw_pred = int(r_row["pred_flood_pixels"])

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

    comp_csv_path = output_dir / "raw_vs_corrected_full_baseline.csv"
    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        for r in comparison_rows:
            writer.writerow(r)
    logger.info(f"Saved fair raw vs corrected full baseline comparison: {comp_csv_path}")

    # Print summary tables to stdout / log
    logger.info("\n" + "=" * 125)
    logger.info("FINAL FAIR FULL BASELINE COMPARISON TABLE (All 2,826 Samples Verified)")
    logger.info("=" * 125)
    logger.info(f"{'Event ID':10s} | {'Samples':7s} | {'DEM Inv':8s} | {'Raw F1':7s} | {'Corr F1':7s} | {'Delta F1':8s} | {'Raw IoU':7s} | {'Corr IoU':8s} | {'Delta IoU':9s} | {'Raw Pred':9s} | {'Corr Pred':9s} | {'Delta Pred':10s}")
    logger.info("-" * 125)
    for r in comparison_rows:
        logger.info(
            f"{r['event_id']:<10d} | {r['sample_count']:<7d} | {r['dem_invalid_pixels']:<8d} | "
            f"{r['raw_f1']:.4f}  | {r['corrected_f1']:.4f}  | {r['delta_f1']:+7.4f}  | "
            f"{r['raw_iou']:.4f}  | {r['corrected_iou']:.4f}   | {r['delta_iou']:+8.4f}  | "
            f"{r['raw_pred_flood_pixels']:<9d} | {r['corrected_pred_flood_pixels']:<9d} | {r['delta_pred_flood_pixels']:+10d}"
        )
    logger.info("=" * 125)
    logger.info(f"Macro F1:  Raw={raw_summary['macro_f1']:.4f} -> Corrected={corr_summary['macro_f1']:.4f} (Delta={corr_summary['macro_f1'] - raw_summary['macro_f1']:+.4f})")
    logger.info(f"Macro IoU: Raw={raw_summary['macro_iou']:.4f} -> Corrected={corr_summary['macro_iou']:.4f} (Delta={corr_summary['macro_iou'] - raw_summary['macro_iou']:+.4f})")
    logger.info(f"Pooled F1: Raw={raw_summary['pooled_metrics']['f1']:.4f} -> Corrected={corr_summary['pooled_metrics']['f1']:.4f} (Delta={corr_summary['pooled_metrics']['f1'] - raw_summary['pooled_metrics']['f1']:+.4f})")
    logger.info(f"Pooled IoU: Raw={raw_summary['pooled_metrics']['iou']:.4f} -> Corrected={corr_summary['pooled_metrics']['iou']:.4f} (Delta={corr_summary['pooled_metrics']['iou'] - raw_summary['pooled_metrics']['iou']:+.4f})")
    logger.info("=" * 125)


if __name__ == "__main__":
    main()
