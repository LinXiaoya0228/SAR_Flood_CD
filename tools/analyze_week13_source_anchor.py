"""ChangeGuard Week 13 Source-Anchored Evidence Consistency Analysis Script.

Analyzes Condition C3 vs existing:
1. source_only (Week 12)
2. matched_bn_no_update_v1 (Week 12 BN diagnostic)
3. matched_tent_pre_update_v1 (Week 12)
4. C2_consistency_conf90 (Week 12)
5. C3_anchored_consistency_v1 (Week 13)

Strict Invariants:
- Uses saved artifacts only without rerunning inference or rereading GT.
- Confusion cell conservation: TP + FP + FN + TN == valid_pixels.
- Reports per-event, event-macro, pooled, and worst-event F1.
- Reports separate consistency loss, anchor loss, and runtime.
- Computes pixel-level divergence across saved frozen predictions.
- Quantifies both harm reduction and loss of previously beneficial adaptation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WEEK12_DIR = REPO_ROOT / "outputs/week12_ablation/approved_dev6_s0_v1"
BN_DIAG_DIR = REPO_ROOT / "outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1"
WEEK13_DIR = REPO_ROOT / "outputs/week13_source_anchor/approved_dev6_s0_v1"
ANALYSIS_DIR = REPO_ROOT / "outputs/week13_source_anchor/analysis_v1"

EVENTS = [279, 437, 520, 559, 1111003, 1111008]
CONDITIONS = [
    "source_only",
    "matched_bn_no_update_v1",
    "matched_tent_pre_update_v1",
    "C2_consistency_conf90",
    "C3_anchored_consistency_v1",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_analysis() -> Dict[str, Any]:
    print("================================================================================")
    print(" ChangeGuard Week 13 Source-Anchor Verification & Comparative Analysis")
    print("================================================================================")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Verification of Week 13 Run Integrity
    # -------------------------------------------------------------------------
    init_manifest_path = WEEK13_DIR / "anchor_manifest.json"
    comp_status_path = WEEK13_DIR / "completion_status.json"

    assert init_manifest_path.is_file(), f"Missing {init_manifest_path}"
    assert comp_status_path.is_file(), f"Missing {comp_status_path}"

    with open(init_manifest_path, "r", encoding="utf-8") as f:
        init_man = json.load(f)
    with open(comp_status_path, "r", encoding="utf-8") as f:
        comp_stat = json.load(f)

    assert comp_stat["status"] == "COMPLETED", f"Status not COMPLETED: {comp_stat['status']}"
    assert init_man["event_sequence"] == EVENTS, f"Event sequence mismatch: {init_man['event_sequence']}"
    assert init_man["checkpoint_seed"] == 0
    assert init_man["condition"] == "C3_anchored_consistency_v1"
    assert init_man["anchor_lambda"] == 0.50
    assert init_man["confidence_threshold"] == 0.90

    print("✓ Run status: COMPLETED")
    print(f"✓ Total elapsed time: {comp_stat['elapsed_seconds']:.2f}s")
    print(f"✓ Checkpoint SHA256: {init_man['checkpoint_actual_sha256']}")
    print(f"✓ Manifest SHA256: {init_man['manifest_sha256']}")

    # -------------------------------------------------------------------------
    # 2. Load Event Metrics & Predictions Across All 5 Conditions
    # -------------------------------------------------------------------------
    event_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {c: {} for c in CONDITIONS}
    preds_by_cond: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {c: {} for c in CONDITIONS}
    loss_stats_by_event: Dict[int, Dict[str, Any]] = {}

    all_cons_losses: List[float] = []
    all_anc_losses: List[float] = []
    all_total_losses: List[float] = []
    all_step_times: List[float] = []

    for ev in EVENTS:
        # Load C3
        ev_dir_c3 = WEEK13_DIR / f"C3_anchored_consistency_v1/event_{ev}"
        with open(ev_dir_c3 / "event_metrics.json", "r", encoding="utf-8") as f:
            em_c3 = json.load(f)
        event_metrics["C3_anchored_consistency_v1"][ev] = em_c3
        npz_c3 = np.load(ev_dir_c3 / "frozen_predictions.npz")
        preds_by_cond["C3_anchored_consistency_v1"][ev] = {k: npz_c3[k] for k in npz_c3.files}

        with open(ev_dir_c3 / "step_records.json", "r", encoding="utf-8") as f:
            steps_c3 = json.load(f)

        ev_cons_losses = []
        ev_anc_losses = []
        ev_total_losses = []
        ev_times = []
        for st in steps_c3:
            if st.get("loss") is not None:
                ev_total_losses.append(st["loss"])
                all_total_losses.append(st["loss"])
            if st.get("consistency_loss") is not None:
                ev_cons_losses.append(st["consistency_loss"])
                all_cons_losses.append(st["consistency_loss"])
            if st.get("anchor_loss") is not None:
                ev_anc_losses.append(st["anchor_loss"])
                all_anc_losses.append(st["anchor_loss"])
            if st.get("batch_time_ms") is not None:
                ev_times.append(st["batch_time_ms"])
                all_step_times.append(st["batch_time_ms"])

        loss_stats_by_event[ev] = {
            "mean_total_loss": float(np.mean(ev_total_losses)) if ev_total_losses else 0.0,
            "mean_cons_loss": float(np.mean(ev_cons_losses)) if ev_cons_losses else 0.0,
            "mean_anc_loss": float(np.mean(ev_anc_losses)) if ev_anc_losses else 0.0,
            "mean_step_time_ms": float(np.mean(ev_times)) if ev_times else 0.0,
            "total_step_time_s": float(np.sum(ev_times) / 1000.0) if ev_times else 0.0,
            "num_steps": len(steps_c3),
        }

        # Load BN diag
        ev_dir_bn = BN_DIAG_DIR / f"matched_bn_no_update_v1/event_{ev}"
        with open(ev_dir_bn / "event_metrics.json", "r", encoding="utf-8") as f:
            event_metrics["matched_bn_no_update_v1"][ev] = json.load(f)
        npz_bn = np.load(ev_dir_bn / "frozen_predictions.npz")
        preds_by_cond["matched_bn_no_update_v1"][ev] = {k: npz_bn[k] for k in npz_bn.files}

        # Load Week 12 conditions
        for c in ["source_only", "matched_tent_pre_update_v1", "C2_consistency_conf90"]:
            ev_dir_w12 = WEEK12_DIR / f"{c}/event_{ev}"
            with open(ev_dir_w12 / "event_metrics.json", "r", encoding="utf-8") as f:
                event_metrics[c][ev] = json.load(f)
            npz_w12 = np.load(ev_dir_w12 / "frozen_predictions.npz")
            preds_by_cond[c][ev] = {k: npz_w12[k] for k in npz_w12.files}

    # Verify coverage and confusion conservation across all conditions
    for c in CONDITIONS:
        tot_cells = sum(
            event_metrics[c][ev]["tp"] + event_metrics[c][ev]["fp"] +
            event_metrics[c][ev]["fn"] + event_metrics[c][ev]["tn"]
            for ev in EVENTS
        )
        tot_valid = sum(event_metrics[c][ev]["valid_pixels"] for ev in EVENTS)
        assert tot_cells == tot_valid == 135859080, f"Conservation violation in {c}: {tot_cells} vs {tot_valid}"
        for ev in EVENTS:
            ref_keys = list(preds_by_cond["source_only"][ev].keys())
            c_keys = list(preds_by_cond[c][ev].keys())
            assert ref_keys == c_keys, f"Sample mismatch in {c} event {ev}"

    print("✓ Full 6-event coverage and identical sample keys verified across all 5 conditions.")
    print("✓ Confusion conservation (TP + FP + FN + TN == 135,859,080) verified for all conditions.")

    # -------------------------------------------------------------------------
    # 3. Compute Summary Metrics Across All 5 Conditions
    # -------------------------------------------------------------------------
    summary_rows = []
    for c in CONDITIONS:
        ev_list = [event_metrics[c][ev] for ev in EVENTS]
        macro_f1 = float(np.mean([m["f1"] for m in ev_list]))
        macro_iou = float(np.mean([m["iou"] for m in ev_list]))
        macro_prec = float(np.mean([m["precision"] for m in ev_list]))
        macro_rec = float(np.mean([m["recall"] for m in ev_list]))

        tot_tp = sum(m["tp"] for m in ev_list)
        tot_fp = sum(m["fp"] for m in ev_list)
        tot_fn = sum(m["fn"] for m in ev_list)
        tot_tn = sum(m["tn"] for m in ev_list)
        tot_valid = sum(m["valid_pixels"] for m in ev_list)

        pooled_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        pooled_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        pooled_f1 = 2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = tot_tp / (tot_tp + tot_fp + tot_fn) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

        worst_m = min(ev_list, key=lambda x: x["f1"])
        tot_updates = sum(m.get("updates_applied", 0) for m in ev_list)

        summary_rows.append({
            "condition_id": c,
            "macro_f1": macro_f1,
            "pooled_f1": pooled_f1,
            "worst_event_id": worst_m["event_id"],
            "worst_event_f1": worst_m["f1"],
            "macro_iou": macro_iou,
            "pooled_iou": pooled_iou,
            "macro_precision": macro_prec,
            "pooled_precision": pooled_prec,
            "macro_recall": macro_rec,
            "pooled_recall": pooled_rec,
            "total_tp": tot_tp,
            "total_fp": tot_fp,
            "total_fn": tot_fn,
            "total_tn": tot_tn,
            "total_valid_pixels": tot_valid,
            "total_updates": tot_updates,
        })

    with open(ANALYSIS_DIR / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # -------------------------------------------------------------------------
    # 4. Per-Event Granular Metrics & Differences
    # -------------------------------------------------------------------------
    per_event_rows = []
    for ev in EVENTS:
        m_src = event_metrics["source_only"][ev]
        m_bn = event_metrics["matched_bn_no_update_v1"][ev]
        m_tent = event_metrics["matched_tent_pre_update_v1"][ev]
        m_c2 = event_metrics["C2_consistency_conf90"][ev]
        m_c3 = event_metrics["C3_anchored_consistency_v1"][ev]

        for cid, m in [
            ("source_only", m_src),
            ("matched_bn_no_update_v1", m_bn),
            ("matched_tent_pre_update_v1", m_tent),
            ("C2_consistency_conf90", m_c2),
            ("C3_anchored_consistency_v1", m_c3),
        ]:
            diff_vs_src = m["f1"] - m_src["f1"]
            diff_vs_bn = m["f1"] - m_bn["f1"]
            diff_vs_c2 = m["f1"] - m_c2["f1"]
            per_event_rows.append({
                "event_id": ev,
                "condition_id": cid,
                "f1": m["f1"],
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
                "valid_pixels": m["valid_pixels"],
                "pred_flood_pixels": m.get("pred_flood_pixels", m["tp"] + m["fp"]),
                "gt_flood_pixels": m.get("gt_flood_pixels", m["tp"] + m["fn"]),
                "diff_vs_source_f1": diff_vs_src,
                "diff_vs_bn_no_update_f1": diff_vs_bn,
                "diff_vs_c2_f1": diff_vs_c2,
            })

    with open(ANALYSIS_DIR / "per_event_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_event_rows)

    # -------------------------------------------------------------------------
    # 5. Loss & Runtime Profile for C3
    # -------------------------------------------------------------------------
    loss_rows = []
    for ev in EVENTS:
        ls = loss_stats_by_event[ev]
        loss_rows.append({
            "event_id": ev,
            "batches": ls["num_steps"],
            "mean_total_loss": ls["mean_total_loss"],
            "mean_consistency_loss": ls["mean_cons_loss"],
            "mean_anchor_loss": ls["mean_anc_loss"],
            "mean_step_time_ms": ls["mean_step_time_ms"],
            "total_step_time_s": ls["total_step_time_s"],
        })

    loss_rows.append({
        "event_id": "ALL",
        "batches": len(all_total_losses),
        "mean_total_loss": float(np.mean(all_total_losses)),
        "mean_consistency_loss": float(np.mean(all_cons_losses)),
        "mean_anchor_loss": float(np.mean(all_anc_losses)),
        "mean_step_time_ms": float(np.mean(all_step_times)),
        "total_step_time_s": float(np.sum(all_step_times) / 1000.0),
    })

    with open(ANALYSIS_DIR / "loss_and_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loss_rows)

    # -------------------------------------------------------------------------
    # 6. Pixel Divergence Analysis (C3 vs C2, C3 vs BN, C3 vs Source)
    # -------------------------------------------------------------------------
    pixel_diff_rows = []
    tot_px_c3_vs_src = 0
    tot_px_c3_vs_bn = 0
    tot_px_c3_vs_c2 = 0
    total_grid_pixels = 0

    for ev in EVENTS:
        keys = list(preds_by_cond["source_only"][ev].keys())
        ev_grid_pixels = len(keys) * 224 * 224
        total_grid_pixels += ev_grid_pixels

        diff_c3_vs_src = 0
        diff_c3_vs_bn = 0
        diff_c3_vs_c2 = 0

        for k in keys:
            p_src = preds_by_cond["source_only"][ev][k]
            p_bn = preds_by_cond["matched_bn_no_update_v1"][ev][k]
            p_c2 = preds_by_cond["C2_consistency_conf90"][ev][k]
            p_c3 = preds_by_cond["C3_anchored_consistency_v1"][ev][k]

            diff_c3_vs_src += int(np.sum(p_c3 != p_src))
            diff_c3_vs_bn += int(np.sum(p_c3 != p_bn))
            diff_c3_vs_c2 += int(np.sum(p_c3 != p_c2))

        tot_px_c3_vs_src += diff_c3_vs_src
        tot_px_c3_vs_bn += diff_c3_vs_bn
        tot_px_c3_vs_c2 += diff_c3_vs_c2

        pixel_diff_rows.append({
            "event_id": ev,
            "samples": len(keys),
            "total_pixels": ev_grid_pixels,
            "diff_c3_vs_src": diff_c3_vs_src,
            "frac_c3_vs_src": diff_c3_vs_src / ev_grid_pixels,
            "diff_c3_vs_bn": diff_c3_vs_bn,
            "frac_c3_vs_bn": diff_c3_vs_bn / ev_grid_pixels,
            "diff_c3_vs_c2": diff_c3_vs_c2,
            "frac_c3_vs_c2": diff_c3_vs_c2 / ev_grid_pixels,
        })

    pixel_diff_rows.append({
        "event_id": "TOTAL",
        "samples": 2826,
        "total_pixels": total_grid_pixels,
        "diff_c3_vs_src": tot_px_c3_vs_src,
        "frac_c3_vs_src": tot_px_c3_vs_src / total_grid_pixels,
        "diff_c3_vs_bn": tot_px_c3_vs_bn,
        "frac_c3_vs_bn": tot_px_c3_vs_bn / total_grid_pixels,
        "diff_c3_vs_c2": tot_px_c3_vs_c2,
        "frac_c3_vs_c2": tot_px_c3_vs_c2 / total_grid_pixels,
    })

    with open(ANALYSIS_DIR / "pixel_differences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pixel_diff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pixel_diff_rows)

    # -------------------------------------------------------------------------
    # 7. SHA256 Manifest of Analysis Artifacts
    # -------------------------------------------------------------------------
    sha_lines = []
    for p in sorted(ANALYSIS_DIR.iterdir()):
        if p.is_file() and p.name != "artifact_sha256.txt":
            sha_lines.append(f"{sha256_file(p)}  {p.name}\n")
    with open(ANALYSIS_DIR / "artifact_sha256.txt", "w", encoding="utf-8") as f:
        f.writelines(sha_lines)

    print("✓ Analysis files successfully written to", ANALYSIS_DIR)
    return {
        "summary": summary_rows,
        "per_event": per_event_rows,
        "loss": loss_rows,
        "pixel_diffs": pixel_diff_rows,
    }


if __name__ == "__main__":
    run_analysis()

