"""Day 7 offline evaluation: joins frozen label-free risk artifacts with validation labels.

Completely decoupled from risk generation:
- Loads frozen risk features and prediction artifacts.
- Joins with ground-truth validation labels offline.
- Evaluates Spearman correlations, AUROC, AUPRC, LOEO, and confounder analyses.
"""

from __future__ import annotations

import sys
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper


def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_confusion_counts(pred: np.ndarray, label: np.ndarray, valid_mask: np.ndarray) -> Dict[str, int]:
    valid = (valid_mask == 1) & (label != 3)
    p_flood = (pred == 2) & valid
    gt_flood = (label == 2) & valid

    tp = int(np.sum(p_flood & gt_flood))
    fp = int(np.sum(p_flood & (~gt_flood)))
    fn = int(np.sum((~p_flood) & gt_flood))
    tn = int(np.sum((~p_flood) & (~gt_flood)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def run_offline_evaluation(
    cfg_path: Path,
    gate_b_root: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    risk_dir = gate_b_root / "risk_generation"
    pred_dir = gate_b_root / "predictions"
    eval_dir = gate_b_root / "offline_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    val_events = cfg["validation_event_ids"]
    seeds = cfg["seeds"]

    print("=" * 80)
    print("DAYS 6–7: GATE B OFFLINE OUTCOME EVALUATION")
    print(f"Risk Directory: {risk_dir}")
    print(f"Pred Directory: {pred_dir}")
    print("=" * 80)

    # 1. Verify existence, hashes, and timestamps of frozen artifacts
    artifact_provenance: Dict[str, Any] = {}
    for seed in seeds:
        run_id = f"gate_b_tent_lr1e-4_seed{seed}"
        r_file = risk_dir / f"{run_id}_risk_features.json"
        p_file = pred_dir / f"{run_id}_predictions.pt"

        if not r_file.exists() or not p_file.exists():
            raise FileNotFoundError(f"Missing required artifact for {run_id}: {r_file} or {p_file}")

        artifact_provenance[run_id] = {
            "risk_file": str(r_file),
            "risk_sha256": compute_file_sha256(r_file),
            "risk_mtime": time.ctime(r_file.stat().st_mtime),
            "pred_file": str(p_file),
            "pred_sha256": compute_file_sha256(p_file),
            "pred_mtime": time.ctime(p_file.stat().st_mtime),
        }
        print(f"Verified {run_id}: Risk SHA={artifact_provenance[run_id]['risk_sha256'][:16]}..., Pred SHA={artifact_provenance[run_id]['pred_sha256'][:16]}...")

    # 2. Load ground-truth labels and valid masks into memory
    print("\nLoading validation dataset ground truth labels...")
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=val_events,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        preprocessing_version=cfg["preprocessing_version"],
    )

    gt_data: Dict[str, Dict[str, Any]] = {}
    for item in dataset:
        sid = item["sample_id"]
        gt_data[sid] = {
            "event_id": item["event_id"],
            "label": item["label"].numpy(),
            "valid_mask": item["valid_mask"].numpy(),
            "xA": item["xA"],
            "xB": item["xB"],
        }

    # 3. Compute immutable source baseline predictions for each sample
    print("Computing immutable source baseline predictions...")
    source_model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    source_model.eval()

    source_predictions: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for sid, d in gt_data.items():
            xA = d["xA"].unsqueeze(0).to(device)
            xB = d["xB"].unsqueeze(0).to(device)
            logits = source_model(xA, xB)
            source_predictions[sid] = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # 4. Join risk features with hidden outcomes
    print("\nJoining label-free risk features with hidden empirical outcomes...")
    joined_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        run_id = f"gate_b_tent_lr1e-4_seed{seed}"
        r_file = risk_dir / f"{run_id}_risk_features.json"
        p_file = pred_dir / f"{run_id}_predictions.pt"

        with open(r_file, "r", encoding="utf-8") as f:
            risk_records = json.load(f)
        adapted_preds = torch.load(p_file)

        # Process event by event to compute cumulative prefix metrics
        for eid in val_events:
            event_records = [r for r in risk_records if r["event_id"] == eid]
            event_records.sort(key=lambda r: r["batch_idx"])

            cum_tp, cum_fp, cum_fn = 0, 0, 0
            cum_src_tp, cum_src_fp, cum_src_fn = 0, 0, 0

            # Compute final event metrics for final outcome
            event_sids = [sid for r in event_records for sid in r["sample_ids"]]
            # Verify no duplicate sample IDs within event
            if len(event_sids) != len(set(event_sids)):
                raise ValueError(f"FATAL JOIN ERROR: Duplicate sample IDs detected in event {eid}!")

            tot_e_tp, tot_e_fp, tot_e_fn = 0, 0, 0
            tot_src_tp, tot_src_fp, tot_src_fn = 0, 0, 0
            for sid in event_sids:
                if sid not in gt_data:
                    raise ValueError(f"FATAL JOIN ERROR: Sample {sid} missing from ground-truth data!")
                if sid not in adapted_preds:
                    raise ValueError(f"FATAL JOIN ERROR: Sample {sid} missing from adapted predictions!")
                p_arr = adapted_preds[sid].numpy() if isinstance(adapted_preds[sid], torch.Tensor) else adapted_preds[sid]
                s_arr = source_predictions[sid]
                l_arr = gt_data[sid]["label"]
                v_arr = gt_data[sid]["valid_mask"]

                cnt = compute_confusion_counts(p_arr, l_arr, v_arr)
                s_cnt = compute_confusion_counts(s_arr, l_arr, v_arr)
                tot_e_tp += cnt["tp"]; tot_e_fp += cnt["fp"]; tot_e_fn += cnt["fn"]
                tot_src_tp += s_cnt["tp"]; tot_src_fp += s_cnt["fp"]; tot_src_fn += s_cnt["fn"]

            _, _, final_adapt_f1 = compute_metrics_from_counts(tot_e_tp, tot_e_fp, tot_e_fn)
            _, final_adapt_rec, _ = compute_metrics_from_counts(tot_e_tp, tot_e_fp, tot_e_fn)
            _, _, final_src_f1 = compute_metrics_from_counts(tot_src_tp, tot_src_fp, tot_src_fn)
            _, final_src_rec, _ = compute_metrics_from_counts(tot_src_tp, tot_src_fp, tot_src_fn)

            final_delta_f1 = final_adapt_f1 - final_src_f1
            final_delta_recall = final_adapt_rec - final_src_rec

            for r in event_records:
                b_sids = r["sample_ids"]
                b_tp, b_fp, b_fn = 0, 0, 0
                b_src_tp, b_src_fp, b_src_fn = 0, 0, 0

                for sid in b_sids:
                    p_arr = adapted_preds[sid].numpy() if isinstance(adapted_preds[sid], torch.Tensor) else adapted_preds[sid]
                    s_arr = source_predictions[sid]
                    l_arr = gt_data[sid]["label"]
                    v_arr = gt_data[sid]["valid_mask"]

                    cnt = compute_confusion_counts(p_arr, l_arr, v_arr)
                    s_cnt = compute_confusion_counts(s_arr, l_arr, v_arr)
                    b_tp += cnt["tp"]; b_fp += cnt["fp"]; b_fn += cnt["fn"]
                    b_src_tp += s_cnt["tp"]; b_src_fp += s_cnt["fp"]; b_src_fn += s_cnt["fn"]

                cum_tp += b_tp; cum_fp += b_fp; cum_fn += b_fn
                cum_src_tp += b_src_tp; cum_src_fp += b_src_fp; cum_src_fn += b_src_fn

                _, cur_rec, cur_f1 = compute_metrics_from_counts(cum_tp, cum_fp, cum_fn)
                _, src_rec, src_f1 = compute_metrics_from_counts(cum_src_tp, cum_src_fp, cum_src_fn)

                delta_f1 = cur_f1 - src_f1
                delta_rec = cur_rec - src_rec

                # Primary harm definition (delta F1 <= -0.01)
                is_harmful_01 = 1 if delta_f1 <= -0.01 else 0
                is_harmful_00 = 1 if delta_f1 <= 0.0 else 0
                is_harmful_02 = 1 if delta_f1 <= -0.02 else 0
                is_harmful_10 = 1 if delta_f1 <= -0.10 else 0
                is_beneficial = 1 if delta_f1 >= 0.01 else 0

                row = dict(r)
                # Joined hidden outcomes (stored only in offline evaluation)
                row["hidden_source_f1"] = src_f1
                row["hidden_adapted_f1"] = cur_f1
                row["hidden_delta_f1"] = delta_f1
                row["hidden_source_recall"] = src_rec
                row["hidden_adapted_recall"] = cur_rec
                row["hidden_delta_recall"] = delta_rec
                row["hidden_final_event_delta_f1"] = final_delta_f1
                row["hidden_final_event_delta_recall"] = final_delta_recall
                row["hidden_is_harmful_primary"] = is_harmful_01
                row["hidden_is_harmful_thresh00"] = is_harmful_00
                row["hidden_is_harmful_thresh02"] = is_harmful_02
                row["hidden_is_harmful_thresh10"] = is_harmful_10
                row["hidden_is_beneficial"] = is_beneficial

                joined_rows.append(row)

    df_joined = pd.DataFrame(joined_rows)
    joined_csv_path = eval_dir / "joined_risk_and_hidden_outcomes.csv"
    df_joined.to_csv(joined_csv_path, index=False)
    print(f"Saved joined evaluation dataset ({len(df_joined)} rows) to: {joined_csv_path}")

    # 5. Statistical Evaluation for Every Indicator
    print("\nRunning statistical evaluations (Spearman, AUROC, AUPRC, LOEO, Confounders)...")
    
    indicators_to_evaluate = [
        # Generic indicators
        "mean_prediction_entropy",
        "mean_max_confidence",
        "kl_divergence_source_adapted",
        "source_disagreement_fraction",
        "bn_affine_drift_l2",
        "bn_running_stat_drift_l2",
        "intermediate_feature_drift_l2",
        "predicted_flood_area",
        "adapted_to_source_area_ratio",
        "augmentation_inconsistency",
        # Change-specific null indicators
        "null_pre2_pre2_area_increase",
        "null_pre2_pre2_activation_rate",
        "null_pre2_pre2_area_ratio",
        "null_pre2_pre2_aug_inconsistency",
        "null_pre2_pre2_spatial_concentration",
        "null_pre1_pre1_area_increase",
        "null_post_post_area_increase",
        "stability_pre1_pre2_area_increase",
        # Change-disappearance indicators
        "disappear_fraction_source_erased",
        "disappear_area_ratio",
        "disappear_prob_ratio",
        "disappear_feature_norm_ratio",
        "disappear_disagreement_fraction",
        # Predeclared composite signals
        "predeclared_signal_null_growth",
        "predeclared_signal_flood_erasure",
        "predeclared_change_risk_score",
    ]

    summary_metrics_list: List[Dict[str, Any]] = []

    y_harm = df_joined["hidden_is_harmful_primary"].values
    y_harm_00 = df_joined["hidden_is_harmful_thresh00"].values
    y_harm_02 = df_joined["hidden_is_harmful_thresh02"].values
    y_harm_10 = df_joined["hidden_is_harmful_thresh10"].values
    y_delta_f1 = df_joined["hidden_delta_f1"].values
    y_delta_rec = df_joined["hidden_delta_recall"].values

    for ind in indicators_to_evaluate:
        if ind not in df_joined.columns:
            continue
        vals = df_joined[ind].values

        # Spearman correlation (with delta F1 and delta Recall)
        # Note: Risk is positively oriented, so higher risk -> lower delta F1 (negative spearman with delta F1)
        r_f1, p_f1 = spearmanr(vals, y_delta_f1)
        r_rec, p_rec = spearmanr(vals, y_delta_rec)

        # AUROC and AUPRC for predicting harmful states (delta F1 <= -0.01)
        # For risk score, higher score predicts harm (label=1)
        try:
            auroc = float(roc_auc_score(y_harm, vals))
        except Exception:
            auroc = float("nan")

        try:
            auprc = float(average_precision_score(y_harm, vals))
        except Exception:
            auprc = float("nan")

        try:
            auroc_00 = float(roc_auc_score(y_harm_00, vals))
        except Exception:
            auroc_00 = float("nan")

        try:
            auroc_02 = float(roc_auc_score(y_harm_02, vals))
        except Exception:
            auroc_02 = float("nan")

        try:
            auroc_10 = float(roc_auc_score(y_harm_10, vals))
        except Exception:
            auroc_10 = float("nan")

        # Leave-one-event-out (LOEO) AUROC
        loeo_aurocs: List[float] = []
        for holdout_eid in val_events:
            sub = df_joined[df_joined["event_id"] != holdout_eid]
            sub_y = sub["hidden_is_harmful_primary"].values
            sub_v = sub[ind].values
            if len(np.unique(sub_y)) > 1:
                loeo_aurocs.append(float(roc_auc_score(sub_y, sub_v)))
        mean_loeo_auroc = float(np.mean(loeo_aurocs)) if loeo_aurocs else float("nan")

        # Event-wise Spearman correlations
        event_spearmans: Dict[str, float] = {}
        for eid in val_events:
            sub_e = df_joined[df_joined["event_id"] == eid]
            if len(sub_e) > 2 and len(np.unique(sub_e[ind].values)) > 1:
                r_e, _ = spearmanr(sub_e[ind].values, sub_e["hidden_delta_f1"].values)
                event_spearmans[str(eid)] = float(r_e)
            else:
                event_spearmans[str(eid)] = float("nan")

        # Confounder checks: correlation of indicator with event size, source area, dummy 559
        src_area = df_joined["source_predicted_flood_area"].values
        is_559 = (df_joined["event_id"] == 559).astype(float).values
        r_src_area, _ = spearmanr(vals, src_area)
        r_559, _ = spearmanr(vals, is_559)

        summary_metrics_list.append({
            "indicator": ind,
            "spearman_delta_f1": float(r_f1),
            "spearman_p_value_f1": float(p_f1),
            "spearman_delta_recall": float(r_rec),
            "spearman_p_value_recall": float(p_rec),
            "auroc_harmful_primary_01": auroc,
            "auprc_harmful_primary_01": auprc,
            "auroc_thresh_00": auroc_00,
            "auroc_thresh_02": auroc_02,
            "auroc_thresh_10": auroc_10,
            "mean_loeo_auroc": mean_loeo_auroc,
            "confounder_corr_source_area": float(r_src_area),
            "confounder_corr_is_event_559": float(r_559),
            "event_spearmans_f1": event_spearmans,
        })

    df_eval_summary = pd.DataFrame(summary_metrics_list)
    summary_csv_path = eval_dir / "indicator_evaluation_summary.csv"
    df_eval_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved indicator evaluation summary to: {summary_csv_path}")

    # 6. Synthesize Gate B Findings
    print("\n" + "=" * 80)
    print("GATE B INDICATOR EVALUATION RESULTS")
    print("=" * 80)
    for row in summary_metrics_list:
        print(
            f"{row['indicator']:35s} | "
            f"Spearman(dF1): {row['spearman_delta_f1']:7.4f} | "
            f"AUROC: {row['auroc_harmful_primary_01']:6.4f} | "
            f"AUPRC: {row['auprc_harmful_primary_01']:6.4f} | "
            f"LOEO: {row['mean_loeo_auroc']:6.4f}"
        )

    # Compile full Gate B evaluation JSON
    eval_json_path = eval_dir / "gate_b_evaluation.json"
    eval_report = {
        "protocol_hash": cfg.get("protocol_hash", compute_file_sha256(cfg_path)),
        "artifact_provenance": artifact_provenance,
        "total_batches_evaluated": len(df_joined),
        "total_harmful_batches_primary": int(np.sum(y_harm)),
        "total_beneficial_batches": int(np.sum(df_joined["hidden_is_beneficial"])),
        "indicator_evaluations": summary_metrics_list,
        "evaluation_timestamp": time.ctime(),
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved complete Gate B evaluation JSON to: {eval_json_path}")

    return eval_report


if __name__ == "__main__":
    cfg_p = PROJECT_ROOT / "configs/gate_b_risk_protocol.json"
    root_p = PROJECT_ROOT / "outputs/day6_7_gate_b"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_offline_evaluation(cfg_p, root_p, device=dev)
