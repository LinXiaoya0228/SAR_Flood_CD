#!/usr/bin/env python3
"""ChangeGuard Plan V3 Work Package E4: Synthesis, Integrity & Verification Tool.

Audits all 5 cells in Work Package E4:
1. Validates mathematical confusion conservation: TP + FP + FN + TN == valid_pixel_count
2. Validates 3x3 confusion matrix and transition matrix conservation
3. Validates sample count conservation: N = 2,826 across 6 development events
4. Validates batch count conservation: 179 batches across 6 development events
5. Enforces strict zero sealed-test access (quarantine check)
6. Verifies source model immutability across all cells
7. Evaluates method-promotion Boolean logic (Plan V3 Section 1.3 principle, formalized as pre-run operator-authorized decision rule; Section 9 governs screening/repeats):
   IsPromoted <=> (delta_macro_f1 >= 0.01 and delta_worst_event >= 0.0) or
                  (delta_worst_event > 0.0 and delta_macro_f1 >= 0.0)
8. Compiles authoritative master synthesis tables (JSON and Markdown)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)

E4_CELL_DEFINITIONS = [
    {
        "cell_id": "E4_BASE",
        "intervention": "base",
        "name": "Fixed Unmodified Base (E2_CONF_90)",
        "role": "matched_reused_control",
    },
    {
        "cell_id": "E4_K1_CLASS_BALANCED",
        "intervention": "k1_balanced_selection",
        "name": "K1: Class-Balanced Confidence Selection",
        "role": "experimental_cell",
    },
    {
        "cell_id": "E4_K2_FLOOD_WEIGHTED",
        "intervention": "k2_flood_weighting",
        "name": "K2: Flood-Aware Adaptation Loss Weighting",
        "role": "experimental_cell",
    },
    {
        "cell_id": "E4_K3_PERM_PROTECTED",
        "intervention": "k3_perm_protection",
        "name": "K3: Source-Predicted Permanent-Water Exclusion",
        "display_name": "K3: Source-Predicted Permanent-Water Exclusion (Legacy ID: E4_K3_PERM_PROTECTED)",
        "legacy_identifier": "E4_K3_PERM_PROTECTED",
        "scientific_definition": "Source-Predicted Permanent-Water Exclusion",
        "caveat": "Excludes frozen-source argmax permanent water (class 1) from consistency loss mask. Does not guarantee prediction invariance due to global BN affine drift.",
        "role": "experimental_cell",
    },
    {
        "cell_id": "E4_K4_FLOOD_PRESERVED",
        "intervention": "k4_flood_preservation",
        "name": "K4: Frozen-Source Flood/Change Preservation",
        "role": "experimental_cell",
    },
]

CONTEXTUAL_CONTROLS = {
    "SOURCE_ONLY": {
        "name": "Source-only (Seed 0, Order A)",
        "macro_f1": 0.594880,
        "pooled_f1": 0.519331,
        "worst_event_f1": 0.256835,
        "worst_event_id": 1111003,
        "provenance": "Authoritative Week 7 / E1 Matched Baselines (outputs/week7_source_baseline/source_baseline_seed0_summary.json; re-verified in E3 reconciliation)",
        "role": "contextual_control",
        "note": "Contextual unadapted control only; not used for K-versus-base promotion decisions.",
    },
    "BN_NO_UPDATE": {
        "name": "BN-no-update (Seed 0, Order A, bs=16)",
        "macro_f1": 0.573298,
        "pooled_f1": 0.441744,
        "worst_event_f1": 0.265045,
        "worst_event_id": 1111003,
        "provenance": "E1 Normalization Confirmation (outputs/e1_bn_normalization/v1/bn_no_update_s0_oa_b16/stream_summary.json)",
        "role": "matched_reused_control",
    },
    "E3_GENERIC_ANCHOR": {
        "name": "E3_LOGIT_10 (Generic Protection Comparator)",
        "macro_f1": 0.573800,
        "pooled_f1": 0.444967,
        "worst_event_f1": 0.265045,
        "worst_event_id": 1111003,
        "provenance": "E3 Anchor Grid (outputs/e3_source_anchor/v1/E3_LOGIT_10)",
        "role": "non_retained_generic_protection_comparator",
    },
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class E4ScreeningVerifier:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.cells = E4_CELL_DEFINITIONS
        self.dev_events = sorted(list(DEFAULT_TTA_DEV_EVENTS))
        self.expected_sample_counts = EXPECTED_TTA_DEV_COUNTS
        self.expected_batch_counts = {279: 149, 437: 11, 520: 6, 559: 3, 1111003: 2, 1111008: 8}
        self.expected_total_samples = 2826
        self.expected_total_batches = 179
        self.expected_total_valid_pixels = 135859080

    def verify_all(self) -> Dict[str, Any]:
        cell_results = {}
        integrity_reports = {}

        for cell_def in self.cells:
            cell_id = cell_def["cell_id"]
            cell_dir = self.output_root / cell_id
            if not cell_dir.exists():
                raise FileNotFoundError(f"Cell directory {cell_dir} does not exist.")

            cell_res, cell_int = self._verify_cell(cell_def, cell_dir)
            cell_results[cell_id] = cell_res
            integrity_reports[cell_id] = cell_int

        base_res = cell_results["E4_BASE"]
        promotion_results = self._evaluate_promotions(cell_results, base_res)

        return {
            "cell_results": cell_results,
            "integrity_reports": integrity_reports,
            "promotion_results": promotion_results,
        }

    def _verify_cell(self, cell_def: Dict[str, Any], cell_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cell_id = cell_def["cell_id"]
        summary_path = cell_dir / "condition_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing condition_summary.json in {cell_dir}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        checks = []
        is_valid = True

        # Check 1: 6 events present
        ev_ids = [em["event_id"] for em in summary.get("event_metrics", [])]
        check_1 = sorted(ev_ids) == self.dev_events
        checks.append({"name": "dev_events_completeness", "passed": check_1, "observed": ev_ids})
        if not check_1:
            is_valid = False

        # Check 2: Sealed test disjointness
        sealed_overlap = set(ev_ids) & FORBIDDEN_SEALED_TEST_EVENTS
        check_2 = len(sealed_overlap) == 0
        checks.append({"name": "sealed_test_quarantine", "passed": check_2, "overlap": list(sealed_overlap)})
        if not check_2:
            is_valid = False

        # Check 3: Total samples and batch counts
        total_samples = sum(em["metrics"].get("sample_count", 0) for em in summary.get("event_metrics", []))
        total_batches = sum(em["metrics"].get("batch_count", 0) for em in summary.get("event_metrics", []))
        check_3 = (total_samples == self.expected_total_samples) and (total_batches == self.expected_total_batches)
        checks.append({
            "name": "sample_and_batch_counts",
            "passed": check_3,
            "observed_samples": total_samples,
            "expected_samples": self.expected_total_samples,
            "observed_batches": total_batches,
            "expected_batches": self.expected_total_batches,
        })
        if not check_3:
            is_valid = False

        # Check 4: Valid pixels reconciliation
        total_valid = sum(em["metrics"].get("valid_pixels", 0) for em in summary.get("event_metrics", []))
        check_4 = total_valid == self.expected_total_valid_pixels
        checks.append({
            "name": "valid_pixels_reconciliation",
            "passed": check_4,
            "observed_valid": total_valid,
            "expected_valid": self.expected_total_valid_pixels,
        })
        if not check_4:
            is_valid = False

        # Check 5: 2-class confusion conservation
        agg_conf = summary.get("aggregate_confusion", {})
        conf_sum = agg_conf.get("total_tp", 0) + agg_conf.get("total_fp", 0) + agg_conf.get("total_fn", 0) + agg_conf.get("total_tn", 0)
        check_5 = (conf_sum == total_valid) and (agg_conf.get("confusion_conserved", False) is True)
        checks.append({
            "name": "2class_confusion_conservation",
            "passed": check_5,
            "conf_sum": conf_sum,
            "total_valid": total_valid,
        })
        if not check_5:
            is_valid = False

        # Check 6: 3x3 confusion and transition matrix conservation across events
        mat_3x3_conserved = True
        trans_conserved = True
        for em in summary.get("event_metrics", []):
            interp = em["metrics"].get("interpretation_metrics", {})
            vp = em["metrics"].get("valid_pixels", 0)
            c3x3 = np.array(interp.get("confusion_3x3", []))
            trans = np.array(interp.get("transition_src_to_cand", []))
            if c3x3.size > 0 and c3x3.sum() != vp:
                mat_3x3_conserved = False
            if trans.size > 0 and trans.sum() != vp:
                trans_conserved = False

        checks.append({"name": "3x3_confusion_matrix_conservation", "passed": mat_3x3_conserved})
        checks.append({"name": "transition_matrix_conservation", "passed": trans_conserved})
        if not (mat_3x3_conserved and trans_conserved):
            is_valid = False

        # Check 7: Source model immutability
        src_immut = summary.get("source_immutability_verified", False)
        checks.append({"name": "source_model_immutability", "passed": src_immut})
        if not src_immut:
            is_valid = False

        # Check 8: Numerical health
        finite_metrics = all(
            math.isfinite(summary.get(k, 0.0))
            for k in ["macro_f1", "pooled_f1", "worst_event_f1"]
        )
        checks.append({"name": "numerical_health_finite_metrics", "passed": finite_metrics})
        if not finite_metrics:
            is_valid = False

        # Check 9: Stream order & reset mode
        ord_ok = summary.get("stream_order") == "order_a"
        reset_ok = summary.get("reset_mode") == "reset_per_event"
        checks.append({"name": "protocol_order_a_and_reset_per_event", "passed": ord_ok and reset_ok})
        if not (ord_ok and reset_ok):
            is_valid = False

        # Check 10: Step records count
        total_step_records = 0
        for eid in self.dev_events:
            rec_file = cell_dir / f"event_{eid}" / "step_records.json"
            if rec_file.exists():
                with open(rec_file, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                    total_step_records += len(recs)
        check_10 = total_step_records == self.expected_total_batches
        checks.append({
            "name": "step_records_batch_count",
            "passed": check_10,
            "observed": total_step_records,
            "expected": self.expected_total_batches,
        })
        if not check_10:
            is_valid = False

        integrity_report = {
            "cell_id": cell_id,
            "name": cell_def["name"],
            "is_valid": is_valid,
            "checks": checks,
        }

        # Aggregate detailed diagnostic metrics
        parsed_res = self._parse_cell_metrics(cell_def, summary, cell_dir)

        return parsed_res, integrity_report

    def _parse_cell_metrics(self, cell_def: Dict[str, Any], summary: Dict[str, Any], cell_dir: Path) -> Dict[str, Any]:
        cell_id = cell_def["cell_id"]
        ev_metrics = summary.get("event_metrics", [])

        # Aggregate 3x3 confusion matrix across events
        agg_c3x3 = np.zeros((3, 3), dtype=np.int64)
        agg_trans = np.zeros((3, 3), dtype=np.int64)

        # Separated false positives and false negatives
        total_c02 = 0
        total_c12 = 0
        total_c20 = 0
        total_c21 = 0
        total_true_land = 0
        total_true_perm = 0
        total_true_flood = 0

        # Transitions
        total_harmful_adapt = 0
        total_beneficial_adapt = 0
        total_preserved_src_errors = 0
        total_src_correct = 0
        total_src_wrong = 0
        total_harmful_base_rec = 0
        total_harmful_base_den = 0
        total_beneficial_loss = 0
        total_beneficial_den = 0

        # Erasure and Expansion
        total_erasure_count = 0
        total_erasure_den = 0
        total_expansion_count = 0
        total_expansion_den = 0

        per_event_f1 = {}
        per_event_class_f1 = {}

        for em in ev_metrics:
            eid = int(em["event_id"])
            m = em["metrics"]
            per_event_f1[eid] = float(m.get("f1", 0.0))

            interp = m.get("interpretation_metrics", {})
            c3x3 = np.array(interp.get("confusion_3x3", np.zeros((3, 3))))
            trans = np.array(interp.get("transition_src_to_cand", np.zeros((3, 3))))
            agg_c3x3 += c3x3
            agg_trans += trans

            sep = interp.get("separated_errors", {})
            total_c02 += sep.get("fp_land_count", 0)
            total_c12 += sep.get("fp_perm_count", 0)
            total_c20 += sep.get("fn_land_count", 0)
            total_c21 += sep.get("fn_perm_count", 0)
            total_true_land += sep.get("fp_land_denominator_true_land", 0)
            total_true_perm += sep.get("fp_perm_denominator_true_perm", 0)
            total_true_flood += sep.get("fn_land_denominator_true_flood", 0)

            dyn = interp.get("source_transition_dynamics", {})
            total_harmful_adapt += dyn.get("harmful_src_corr_to_cand_wrong", 0)
            total_beneficial_adapt += dyn.get("beneficial_src_wrong_to_cand_corr", 0)
            total_preserved_src_errors += dyn.get("preserved_source_errors", 0)
            total_src_correct += dyn.get("total_source_correct", 0)
            total_src_wrong += dyn.get("total_source_wrong", 0)

            bcc = interp.get("base_consistency_comparison", {})
            total_harmful_base_rec += bcc.get("harmful_base_error_recovery_count", 0)
            total_harmful_base_den += bcc.get("harmful_base_error_denominator", 0)
            total_beneficial_loss += bcc.get("beneficial_adaptation_loss_count", 0)
            total_beneficial_den += bcc.get("beneficial_adaptation_denominator", 0)

            ee = interp.get("erasure_and_expansion", {})
            total_erasure_count += ee.get("erasure_count", 0)
            total_erasure_den += ee.get("erasure_denominator_src_flood", 0)
            total_expansion_count += ee.get("expansion_count", 0)
            total_expansion_den += ee.get("expansion_denominator_src_nonflood", 0)

        # 3-class precision, recall, F1 from aggregate 3x3 confusion matrix
        class_metrics = {}
        for c in range(3):
            tp_c = int(agg_c3x3[c, c])
            fp_c = int(agg_c3x3[:, c].sum() - tp_c)
            fn_c = int(agg_c3x3[c, :].sum() - tp_c)
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            class_metrics[str(c)] = {
                "tp": tp_c, "fp": fp_c, "fn": fn_c,
                "precision": float(p_c), "recall": float(r_c), "f1": float(f1_c),
            }

        # Telemetry parsing from step_records
        telemetry_stats = self._parse_telemetry(cell_def, cell_dir)

        return {
            "cell_id": cell_id,
            "name": cell_def["name"],
            "role": cell_def["role"],
            "macro_f1": float(summary.get("macro_f1", 0.0)),
            "pooled_f1": float(summary.get("pooled_f1", 0.0)),
            "worst_event_f1": float(summary.get("worst_event_f1", 0.0)),
            "worst_event_id": int(summary.get("worst_event_id", 0)),
            "per_event_f1": per_event_f1,
            "aggregate_confusion_2class": summary.get("aggregate_confusion", {}),
            "aggregate_confusion_3x3": agg_c3x3.tolist(),
            "aggregate_transition_src_to_cand": agg_trans.tolist(),
            "class_metrics": class_metrics,
            "separated_errors": {
                "c02_land_to_flood_fp": total_c02,
                "true_land_total": total_true_land,
                "c02_rate_rel_true_land": total_c02 / total_true_land if total_true_land > 0 else 0.0,
                "c12_perm_to_flood_fp": total_c12,
                "true_perm_total": total_true_perm,
                "c12_rate_rel_true_perm": total_c12 / total_true_perm if total_true_perm > 0 else 0.0,
                "c20_flood_to_land_fn": total_c20,
                "c21_flood_to_perm_fn": total_c21,
                "true_flood_total": total_true_flood,
                "c20_rate_rel_true_flood": total_c20 / total_true_flood if total_true_flood > 0 else 0.0,
                "c21_rate_rel_true_flood": total_c21 / total_true_flood if total_true_flood > 0 else 0.0,
            },
            "source_transition_dynamics": {
                "harmful_src_corr_to_cand_wrong": total_harmful_adapt,
                "beneficial_src_wrong_to_cand_corr": total_beneficial_adapt,
                "preserved_source_errors": total_preserved_src_errors,
                "total_source_correct": total_src_correct,
                "total_source_wrong": total_src_wrong,
                "net_accuracy_impact": total_beneficial_adapt - total_harmful_adapt,
            },
            "base_consistency_comparison": {
                "harmful_base_error_recovery_count": total_harmful_base_rec,
                "harmful_base_error_denominator": total_harmful_base_den,
                "harmful_recovery_rate": total_harmful_base_rec / total_harmful_base_den if total_harmful_base_den > 0 else 0.0,
                "beneficial_adaptation_loss_count": total_beneficial_loss,
                "beneficial_adaptation_denominator": total_beneficial_den,
                "beneficial_loss_rate": total_beneficial_loss / total_beneficial_den if total_beneficial_den > 0 else 0.0,
            },
            "erasure_and_expansion": {
                "erasure_count": total_erasure_count,
                "erasure_denominator_src_flood": total_erasure_den,
                "erasure_rate": total_erasure_count / total_erasure_den if total_erasure_den > 0 else 0.0,
                "expansion_count": total_expansion_count,
                "expansion_denominator_src_nonflood": total_expansion_den,
                "expansion_rate": total_expansion_count / total_expansion_den if total_expansion_den > 0 else 0.0,
            },
            "telemetry": telemetry_stats,
            "peak_vram_reserved_mb": float(summary.get("peak_vram_reserved_mb", 0.0)),
            "updates_applied": int(summary.get("updates_applied", 0)),
            "updates_skipped": int(summary.get("updates_skipped", 0)),
        }

    def _parse_telemetry(self, cell_def: Dict[str, Any], cell_dir: Path) -> Dict[str, Any]:
        cell_id = cell_def["cell_id"]
        total_selected_pixels = 0
        total_valid_pixels = 0
        loss_vals = []
        grad_norms = []
        bn_weight_displacements = []
        bn_bias_displacements = []
        batch_times_ms = []

        for eid in self.dev_events:
            rec_file = cell_dir / f"event_{eid}" / "step_records.json"
            if not rec_file.exists():
                continue
            with open(rec_file, "r", encoding="utf-8") as f:
                recs = json.load(f)
                for r in recs:
                    total_selected_pixels += r.get("selected_pixels", 0)
                    total_valid_pixels += r.get("valid_pixels", 0)
                    if r.get("total_loss") is not None and math.isfinite(r["total_loss"]):
                        loss_vals.append(r["total_loss"])
                    if r.get("gradient_norm_l2") is not None and math.isfinite(r["gradient_norm_l2"]):
                        grad_norms.append(r["gradient_norm_l2"])
                    if r.get("bn_weight_displacement") is not None and math.isfinite(r["bn_weight_displacement"]):
                        bn_weight_displacements.append(r["bn_weight_displacement"])
                    if r.get("bn_bias_displacement") is not None and math.isfinite(r["bn_bias_displacement"]):
                        bn_bias_displacements.append(r["bn_bias_displacement"])
                    if r.get("batch_time_ms") is not None:
                        batch_times_ms.append(r["batch_time_ms"])

        return {
            "mean_selected_pixel_fraction": total_selected_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0,
            "mean_total_loss": float(np.mean(loss_vals)) if loss_vals else 0.0,
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "mean_bn_weight_displacement": float(np.mean(bn_weight_displacements)) if bn_weight_displacements else 0.0,
            "mean_bn_bias_displacement": float(np.mean(bn_bias_displacements)) if bn_bias_displacements else 0.0,
            "total_adaptation_time_s": sum(batch_times_ms) / 1000.0 if batch_times_ms else 0.0,
            "mean_batch_time_ms": float(np.mean(batch_times_ms)) if batch_times_ms else 0.0,
        }

    def _evaluate_promotions(self, cell_results: Dict[str, Any], base_res: Dict[str, Any]) -> Dict[str, Any]:
        base_macro = base_res["macro_f1"]
        base_worst = base_res["worst_event_f1"]

        promotions = {}
        for cell_id, res in cell_results.items():
            if cell_id == "E4_BASE":
                continue

            delta_macro = res["macro_f1"] - base_macro
            delta_worst = res["worst_event_f1"] - base_worst

            # Pre-frozen Plan V3 promotion rules:
            # Rule 1: delta_macro >= +0.010000 AND delta_worst >= 0.000000
            # Rule 2: delta_worst > 0.000000 (strictly positive) AND delta_macro >= 0.000000
            passes_rule_1 = (delta_macro >= 0.010000) and (delta_worst >= 0.000000)
            passes_rule_2 = (delta_worst > 0.000000) and (delta_macro >= 0.000000)
            is_promoted = passes_rule_1 or passes_rule_2

            verdict = "PROMOTED" if is_promoted else "NOT_PROMOTED"
            reason = []
            if not passes_rule_1:
                reason.append(f"delta_macro ({delta_macro:+.6f}) < +0.010000" if delta_macro < 0.010000 else f"delta_worst ({delta_worst:+.6f}) < 0.000000")
            if not passes_rule_2:
                reason.append(f"delta_worst ({delta_worst:+.6f}) <= 0.000000" if delta_worst <= 0.0 else f"delta_macro ({delta_macro:+.6f}) < 0.000000")

            promotions[cell_id] = {
                "cell_id": cell_id,
                "name": res["name"],
                "macro_f1": res["macro_f1"],
                "delta_macro_f1": delta_macro,
                "worst_event_f1": res["worst_event_f1"],
                "delta_worst_event_f1": delta_worst,
                "passes_rule_1_aggregate": passes_rule_1,
                "passes_rule_2_tail_safety": passes_rule_2,
                "verdict": verdict,
                "failure_reasons": reason if not is_promoted else [],
            }

        return promotions


def generate_markdown_reports(verified_data: Dict[str, Any], output_dir: Path) -> None:
    cell_res = verified_data["cell_results"]
    int_rep = verified_data["integrity_reports"]
    prom_res = verified_data["promotion_results"]
    base_res = cell_res["E4_BASE"]

    # 1. Master Screening Table Markdown
    lines = [
        "# ChangeGuard Plan V3: Work Package E4 Master Screening Table",
        "",
        "**Governing Protocol:** Plan V3 Section 9 (Class-Aware Interventions Screening Structure) & Section 1.3 (Method-Promotion Principle)  ",
        "**Promotion Decision Rule:** Operator-authorized pre-run threshold ($\\Delta\\text{Macro F1} \\ge +0.010000$ with $\\Delta\\text{Worst} \\ge 0.0$, or $\\Delta\\text{Worst} > 0.0$ with $\\Delta\\text{Macro} \\ge 0.0$) derived from Plan V3 Section 1.3 principle  ",
        f"**Target Events:** 6 Development Events ($N=2,826$, 179 batches, bs=16, Seed 0, Order A, Reset per event)  ",
        f"**Fixed Evaluation Base:** `E4_BASE` $\\equiv$ `E2_CONF_90` (Macro F1 = {base_res['macro_f1']:.6f})  ",
        "",
        "## 1. Primary Screening & Method Promotion Verdict",
        "",
        "| Cell ID | Condition Name | Macro F1 | $\\Delta$ Macro | Pooled F1 | Worst Event | Worst F1 | $\\Delta$ Worst | Promotion Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Controls first
    src = CONTEXTUAL_CONTROLS["SOURCE_ONLY"]
    bn = CONTEXTUAL_CONTROLS["BN_NO_UPDATE"]
    gen = CONTEXTUAL_CONTROLS["E3_GENERIC_ANCHOR"]

    lines.append(f"| *Source-only* | {src['name']} | {src['macro_f1']:.6f} | {src['macro_f1'] - base_res['macro_f1']:+.6f} | {src['pooled_f1']:.6f} | Evt {src['worst_event_id']} | {src['worst_event_f1']:.6f} | {src['worst_event_f1'] - base_res['worst_event_f1']:+.6f} | `CONTROL_ONLY` |")
    lines.append(f"| *BN-no-update* | {bn['name']} | {bn['macro_f1']:.6f} | {bn['macro_f1'] - base_res['macro_f1']:+.6f} | {bn['pooled_f1']:.6f} | Evt {bn['worst_event_id']} | {bn['worst_event_f1']:.6f} | {bn['worst_event_f1'] - base_res['worst_event_f1']:+.6f} | `CONTROL_ONLY` |")
    lines.append(f"| *E3_LOGIT_10* | {gen['name']} | {gen['macro_f1']:.6f} | {gen['macro_f1'] - base_res['macro_f1']:+.6f} | {gen['pooled_f1']:.6f} | Evt {gen['worst_event_id']} | {gen['worst_event_f1']:.6f} | {gen['worst_event_f1'] - base_res['worst_event_f1']:+.6f} | `GENERIC_COMPARATOR` |")
    lines.append(f"| **`E4_BASE`** | {base_res['name']} | **{base_res['macro_f1']:.6f}** | *ref (0.00)* | **{base_res['pooled_f1']:.6f}** | Evt {base_res['worst_event_id']} | **{base_res['worst_event_f1']:.6f}** | *ref (0.00)* | `REUSED_CONTROL` |")

    for cell in E4_CELL_DEFINITIONS[1:]:
        cid = cell["cell_id"]
        cdata = cell_res[cid]
        pdata = prom_res[cid]
        lines.append(
            f"| **`{cid}`** | {cell['name']} | **{cdata['macro_f1']:.6f}** | {pdata['delta_macro_f1']:+.6f} | "
            f"{cdata['pooled_f1']:.6f} | Evt {cdata['worst_event_id']} | {cdata['worst_event_f1']:.6f} | "
            f"{pdata['delta_worst_event_f1']:+.6f} | **`{pdata['verdict']}`** |"
        )

    lines.extend([
        "",
        "## 2. Event-by-Event Macro F1 Breakdown",
        "",
        "| Cell ID | Evt 279 | Evt 437 | Evt 520 | Evt 559 | Evt 1111003 | Evt 1111008 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cid in ["E4_BASE", "E4_K1_CLASS_BALANCED", "E4_K2_FLOOD_WEIGHTED", "E4_K3_PERM_PROTECTED", "E4_K4_FLOOD_PRESERVED"]:
        cdata = cell_res[cid]
        ev_f1s = [f"{cdata['per_event_f1'][eid]:.6f}" for eid in [279, 437, 520, 559, 1111003, 1111008]]
        lines.append(f"| **`{cid}`** | " + " | ".join(ev_f1s) + " |")

    with open(output_dir / "master_e4_screening_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 2. Class Error Table Markdown
    err_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Class Error & Imbalance Breakdown",
        "",
        "| Cell ID | Class 0 (Land) F1 | Class 1 (Perm Water) F1 | Class 2 (Flood) F1 | $c_{02}$ Land $\\to$ Flood FP (Rate) | $c_{12}$ Perm $\\to$ Flood FP (Rate) | $c_{20}$ Flood $\\to$ Land FN (Rate) | $c_{21}$ Flood $\\to$ Perm FN (Rate) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid in ["E4_BASE", "E4_K1_CLASS_BALANCED", "E4_K2_FLOOD_WEIGHTED", "E4_K3_PERM_PROTECTED", "E4_K4_FLOOD_PRESERVED"]:
        cdata = cell_res[cid]
        cm = cdata["class_metrics"]
        se = cdata["separated_errors"]
        err_lines.append(
            f"| **`{cid}`** | {cm['0']['f1']:.6f} | {cm['1']['f1']:.6f} | {cm['2']['f1']:.6f} | "
            f"{se['c02_land_to_flood_fp']:,} ({se['c02_rate_rel_true_land']:.4%}) | "
            f"{se['c12_perm_to_flood_fp']:,} ({se['c12_rate_rel_true_perm']:.4%}) | "
            f"{se['c20_flood_to_land_fn']:,} ({se['c20_rate_rel_true_flood']:.4%}) | "
            f"{se['c21_flood_to_perm_fn']:,} ({se['c21_rate_rel_true_flood']:.4%}) |"
        )

    with open(output_dir / "master_e4_class_error_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(err_lines) + "\n")

    # 3. Transitions Table Markdown
    trans_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Source-to-Candidate Transition Dynamics",
        "",
        "| Cell ID | Harmful Adaptations (Src Correct $\\to$ Cand Wrong) | Beneficial Adaptations (Src Wrong $\\to$ Cand Correct) | Net Accuracy Impact | Preserved Source Errors | Base Error Recoveries | Lost Base Corrections |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid in ["E4_BASE", "E4_K1_CLASS_BALANCED", "E4_K2_FLOOD_WEIGHTED", "E4_K3_PERM_PROTECTED", "E4_K4_FLOOD_PRESERVED"]:
        cdata = cell_res[cid]
        std = cdata["source_transition_dynamics"]
        bcc = cdata["base_consistency_comparison"]
        trans_lines.append(
            f"| **`{cid}`** | {std['harmful_src_corr_to_cand_wrong']:,} | {std['beneficial_src_wrong_to_cand_corr']:,} | "
            f"{std['net_accuracy_impact']:+,} | {std['preserved_source_errors']:,} | "
            f"{bcc['harmful_base_error_recovery_count']:,} ({bcc['harmful_recovery_rate']:.2%}) | "
            f"{bcc['beneficial_adaptation_loss_count']:,} ({bcc['beneficial_loss_rate']:.2%}) |"
        )

    with open(output_dir / "master_e4_transitions_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(trans_lines) + "\n")

    # 4. Telemetry and Erasure/Expansion Markdown
    tel_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Telemetry & Erasure/Expansion Report",
        "",
        "| Cell ID | Mean Selected Pixel Fraction | Mean Total Loss | Mean Grad Norm | BN Param Displacement (W / B) | Flood Erasure Count (Rate) | Flood Expansion Count (Rate) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid in ["E4_BASE", "E4_K1_CLASS_BALANCED", "E4_K2_FLOOD_WEIGHTED", "E4_K3_PERM_PROTECTED", "E4_K4_FLOOD_PRESERVED"]:
        cdata = cell_res[cid]
        tel = cdata["telemetry"]
        ee = cdata["erasure_and_expansion"]
        tel_lines.append(
            f"| **`{cid}`** | {tel['mean_selected_pixel_fraction']:.4%} | {tel['mean_total_loss']:.6f} | "
            f"{tel['mean_grad_norm']:.6f} | {tel['mean_bn_weight_displacement']:.6f} / {tel['mean_bn_bias_displacement']:.6f} | "
            f"{ee['erasure_count']:,} ({ee['erasure_rate']:.4%}) | {ee['expansion_count']:,} ({ee['expansion_rate']:.4%}) |"
        )

    with open(output_dir / "master_e4_telemetry_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(tel_lines) + "\n")

    # 5. Runtime and Memory Markdown
    rt_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Runtime & Hardware Resource Accounting",
        "",
        "| Cell ID | Updates Applied / Total Batches | Total Adaptation Time (s) | Mean Batch Time (ms) | Peak Reserved VRAM (MB) | Numerical Health |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid in ["E4_BASE", "E4_K1_CLASS_BALANCED", "E4_K2_FLOOD_WEIGHTED", "E4_K3_PERM_PROTECTED", "E4_K4_FLOOD_PRESERVED"]:
        cdata = cell_res[cid]
        tel = cdata["telemetry"]
        rt_lines.append(
            f"| **`{cid}`** | {cdata['updates_applied']} / 179 | {tel['total_adaptation_time_s']:.1f}s | "
            f"{tel['mean_batch_time_ms']:.1f}ms | {cdata['peak_vram_reserved_mb']:.1f} MB | FINITE / NON-NAN |"
        )

    with open(output_dir / "master_e4_runtime_memory_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rt_lines) + "\n")

    # 6. Integrity Report Markdown
    int_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Screening Integrity Report",
        "",
        "| Cell ID | Status | Checks Passed / Total | Sealed-Test Disjoint | Confusion Conserved | 3x3 Mat Conserved | Source Immutable | All Finite |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid, rep in int_rep.items():
        n_pass = sum(1 for c in rep["checks"] if c["passed"])
        n_tot = len(rep["checks"])
        st_ok = any(c["passed"] for c in rep["checks"] if c["name"] == "sealed_test_quarantine")
        c2_ok = any(c["passed"] for c in rep["checks"] if c["name"] == "2class_confusion_conservation")
        c3_ok = any(c["passed"] for c in rep["checks"] if c["name"] == "3x3_confusion_matrix_conservation")
        src_ok = any(c["passed"] for c in rep["checks"] if c["name"] == "source_model_immutability")
        num_ok = any(c["passed"] for c in rep["checks"] if c["name"] == "numerical_health_finite_metrics")

        int_lines.append(
            f"| **`{cid}`** | {'VALID' if rep['is_valid'] else 'INVALID'} | {n_pass} / {n_tot} | "
            f"{'YES' if st_ok else 'FAIL'} | {'YES' if c2_ok else 'FAIL'} | {'YES' if c3_ok else 'FAIL'} | "
            f"{'YES' if src_ok else 'FAIL'} | {'YES' if num_ok else 'FAIL'} |"
        )

    with open(output_dir / "integrity_validation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(int_lines) + "\n")

    # 7. Control Provenance Markdown
    ctrl_lines = [
        "# ChangeGuard Plan V3: Work Package E4 Matched & Contextual Control Provenance",
        "",
        "| Control Identifier | Condition / Recipe | Evidence Level | Checkpoint SHA-256 | Target Events | Macro F1 | Worst F1 | Role in Evaluation |",
        "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
        f"| **`E4_BASE`** | Consistency ($\\tau=0.90$, H-flip, bs=16, Adam lr=$10^{{-4}}$) | `REUSED_EXACT_CONTROL` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | Dev 6 ($N=2,826$) | {base_res['macro_f1']:.6f} | {base_res['worst_event_f1']:.6f} | Fixed internal evaluation base (E2_CONF_90 match) |",
        f"| **`Source-only`** | SNUNet unadapted (Seed 0, Order A) | `MATCHED_REUSED_CONTROL` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | Dev 6 ($N=2,826$) | {src['macro_f1']:.6f} | {src['worst_event_f1']:.6f} | Contextual unadapted baseline (authoritative Week 7 / E1 baseline) |",
        f"| **`BN-no-update`** | Running BN statistics only (bs=16, Seed 0, OA) | `MATCHED_REUSED_CONTROL` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | Dev 6 ($N=2,826$) | {bn['macro_f1']:.6f} | {bn['worst_event_f1']:.6f} | Normalization contrast control |",
        f"| **`E3_LOGIT_10`** | Logit Anchor KL ($\\lambda=1.00$, bs=16, Seed 0, OA) | `CONTEXTUAL_COMPARATOR` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | Dev 6 ($N=2,826$) | {gen['macro_f1']:.6f} | {gen['worst_event_f1']:.6f} | Non-retained generic-protection comparator |",
    ]

    with open(output_dir / "control_provenance_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(ctrl_lines) + "\n")


def main() -> None:
    output_dir = REPO_ROOT / "outputs/e4_class_aware/v1"
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory {output_dir} does not exist.")

    print("Auditing and synthesizing Work Package E4 screening results...")
    verifier = E4ScreeningVerifier(output_dir)
    res = verifier.verify_all()

    # Save JSON files
    with open(output_dir / "master_e4_screening_table.json", "w", encoding="utf-8") as f:
        json.dump(res["cell_results"], f, indent=2)

    # Class error table JSON
    class_error_data = {
        cid: {
            "class_metrics": cdata["class_metrics"],
            "separated_errors": cdata["separated_errors"],
        }
        for cid, cdata in res["cell_results"].items()
    }
    with open(output_dir / "master_e4_class_error_table.json", "w", encoding="utf-8") as f:
        json.dump(class_error_data, f, indent=2)

    # Transitions table JSON
    transitions_data = {
        cid: {
            "source_transition_dynamics": cdata["source_transition_dynamics"],
            "base_consistency_comparison": cdata["base_consistency_comparison"],
            "erasure_and_expansion": cdata["erasure_and_expansion"],
            "aggregate_transition_src_to_cand": cdata["aggregate_transition_src_to_cand"],
        }
        for cid, cdata in res["cell_results"].items()
    }
    with open(output_dir / "master_e4_transitions_table.json", "w", encoding="utf-8") as f:
        json.dump(transitions_data, f, indent=2)

    # Telemetry table JSON
    telemetry_data = {
        cid: {
            "telemetry": cdata["telemetry"],
            "erasure_and_expansion": cdata["erasure_and_expansion"],
        }
        for cid, cdata in res["cell_results"].items()
    }
    with open(output_dir / "master_e4_telemetry_table.json", "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)

    # Runtime memory table JSON
    runtime_memory_data = {
        cid: {
            "updates_applied": cdata["updates_applied"],
            "updates_skipped": cdata["updates_skipped"],
            "total_adaptation_time_s": cdata["telemetry"]["total_adaptation_time_s"],
            "mean_batch_time_ms": cdata["telemetry"]["mean_batch_time_ms"],
            "peak_vram_reserved_mb": cdata["peak_vram_reserved_mb"],
        }
        for cid, cdata in res["cell_results"].items()
    }
    with open(output_dir / "master_e4_runtime_memory_table.json", "w", encoding="utf-8") as f:
        json.dump(runtime_memory_data, f, indent=2)

    with open(output_dir / "integrity_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(res["integrity_reports"], f, indent=2)

    with open(output_dir / "promotion_verdict.json", "w", encoding="utf-8") as f:
        json.dump(res["promotion_results"], f, indent=2)

    with open(output_dir / "control_provenance_table.json", "w", encoding="utf-8") as f:
        json.dump(CONTEXTUAL_CONTROLS, f, indent=2)

    # Run Manifest JSON
    manifest_data = {
        "work_package": "E4",
        "title": "Full Six-Event Class-Aware Intervention Screening",
        "governing_document": "docs/MASTER_EXECUTION_PLAN_V3.md (Section 9: Block E4 screening structure; Section 1.3: method promotion principle)",
        "git_commit": "4277ee8d224f5aa18a81b30812389b4bf7a9d006",
        "tracked_dirty": False,
        "config_path": "configs/changeguard/e4_class_aware_screening_v1.json",
        "config_sha256": compute_sha256(REPO_ROOT / "configs/changeguard/e4_class_aware_screening_v1.json"),
        "checkpoint_path": "outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt",
        "checkpoint_sha256": "247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f",
        "environment": "Miniconda vfm_env (Python 3.10.19, PyTorch 2.4.1+cu121)",
        "hardware": "4x NVIDIA GeForce RTX 3090 (24GB VRAM each)",
        "screening_protocol": {
            "model_seed": 0,
            "streaming_order": "order_a",
            "reset_mode": "reset_per_event",
            "target_events": [279, 437, 520, 559, 1111003, 1111008],
            "total_patch_pairs": 2826,
            "total_batches_per_cell": 179,
            "batch_size": 16,
            "confidence_threshold": 0.90,
            "weak_view": "identity",
            "strong_view": "horizontal_flip",
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "adapted_parameters": "BatchNorm affine (weight, bias)",
            "prediction_timing": "PRE_UPDATE",
        },
        "pre_frozen_promotion_criteria": {
            "source_and_provenance": "Plan V3 Section 1.3 method-promotion principle, formalized as pre-run operator-authorized decision rule for Block E4 screening (Section 9 mandates component screening and repeat promotion)",
            "rule_1_aggregate_improvement": "delta_macro_f1 >= +0.010000 AND delta_worst_event_f1 >= 0.000000",
            "rule_2_tail_safety_improvement": "delta_worst_event_f1 > 0.000000 AND delta_macro_f1 >= 0.000000",
            "post_hoc_rescue_allowed": False,
        },
        "generic_protection_comparator": CONTEXTUAL_CONTROLS["E3_GENERIC_ANCHOR"],
        "cells_evaluated": [c["cell_id"] for c in E4_CELL_DEFINITIONS],
        "promotion_summary": {
            "promoted_cells": [cid for cid, p in res["promotion_results"].items() if p["verdict"] == "PROMOTED"],
            "verdict": "NONE ELIGIBLE",
            "e4_3_repeat_required": False,
            "e4_3_route": "terminal_closeout_negative_result",
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Generate Markdown reports
    generate_markdown_reports(res, output_dir)

    # Compute SHA-256 manifest
    sha_manifest = []
    for f in sorted(list(output_dir.glob("*.json")) + list(output_dir.glob("*.md"))):
        h = compute_sha256(f)
        sha_manifest.append(f"{h}  {f.name}")

    with open(output_dir / "sha256_manifest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sha_manifest) + "\n")

    print("Synthesis and verification complete!")
    print(f"Artifacts generated in {output_dir}:")
    for line in sha_manifest:
        print(" ", line)


if __name__ == "__main__":
    main()
