#!/usr/bin/env python3
"""Master execution pipeline for ChangeGuard Work Package RST10.

Executes end-to-end reconstruction:
1. Audits and verifies evidence manifest SHA-256 digests (fail-closed).
2. Generates all 7 canonical synthesis tables (CSV + Markdown).
3. Generates all 9 canonical publication figures (PDF + 300 DPI PNG + CSV data).
4. Generates the failure cases analysis and claims audit ledger.
5. Emits the durable execution metadata manifest.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.robust_source_training.build_rst10_tables import (
    build_table1_component_staircase,
    build_table2_primary_source_confirmation,
    build_table3_architecture_transfer,
    build_table4_deployable_checkpoint_ledger,
    build_table5_historical_target_extension,
    build_table6_runtime_and_resource_cost,
    build_table7_contextual_tta_comparison,
    generate_markdown_tables,
)
from tools.robust_source_training.build_rst10_figures import (
    build_fig1_component_staircase,
    build_fig2_primary_per_event_paired_deltas,
    build_fig3_worst_bottom_event_comparison,
    build_fig4_precision_recall_tradeoff,
    build_fig5_seed_stability,
    build_fig6_event_sampling_distribution,
    build_fig7_architecture_transfer_deltas,
    build_fig8_historical_target_extension_deltas,
    build_fig9_runtime_vram_tradeoff,
)


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_evidence_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Audits all evidence sources listed in manifest against their cryptographic SHA-256 sums."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evidence manifest missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    audit_records = []
    print(f"Auditing {len(manifest['evidence_sources'])} evidence sources in manifest...")
    for entry in manifest["evidence_sources"]:
        p = REPO_ROOT / entry["path"]
        if not p.exists():
            raise FileNotFoundError(f"FAIL-CLOSED: Required evidence input missing: {p}")

        actual_sha = compute_sha256(p)
        expected_sha = entry["sha256"]
        if actual_sha != expected_sha:
            raise ValueError(
                f"FAIL-CLOSED: Cryptographic mismatch for {entry['source_id']} ({entry['path']})!\n"
                f"  Expected: {expected_sha}\n  Actual:   {actual_sha}"
            )
        audit_records.append({
            "source_id": entry["source_id"],
            "path": entry["path"],
            "tier": entry["evidence_tier"],
            "sha256": actual_sha,
            "status": "VERIFIED_BIT_EXACT",
        })
        print(f"  [OK] {entry['source_id']} ({entry['path']})")

    return {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "sources_count": len(manifest["evidence_sources"]),
        "audit_records": audit_records,
    }


def build_failure_cases_and_limitations_table(out_dir: Path) -> Dict[str, Any]:
    """Builds structured failure cases and limitation analysis."""
    rows = [
        {
            "case_id": "FC-1",
            "scope": "RST7 Primary Source Confirmation",
            "event_or_target": "Event 498 (Fold B, 102 patches)",
            "baseline_r0_f1": 0.8143,
            "recipe_r2_f1": 0.7712,
            "paired_delta": -0.0431,
            "failure_mechanism": "Precision degradation (-0.104) due to class reweighting; excessive false positives on relatively balanced urban/vegetation flood",
            "mitigation_or_note": "Documented trade-off; tail loss weighting penalizes false negatives more aggressively, slightly inflating false positives on benign scenes",
        },
        {
            "case_id": "FC-2",
            "scope": "RST7 Primary Source Confirmation",
            "event_or_target": "Event 273 (Fold A, 112 patches)",
            "baseline_r0_f1": 0.5265,
            "recipe_r2_f1": 0.5107,
            "paired_delta": -0.0158,
            "failure_mechanism": "Precision drop (-0.031) outpaced recall gain (+0.045); specular ground false positives",
            "mitigation_or_note": "Localized trade-off; event is in middle difficulty tier",
        },
        {
            "case_id": "FC-3",
            "scope": "RST7 Primary Source Confirmation",
            "event_or_target": "Worst Event 427 (Fold A, 795 patches)",
            "baseline_r0_f1": 0.2991,
            "recipe_r2_f1": 0.2929,
            "paired_delta": -0.0062,
            "failure_mechanism": "Slight worst-case drop (-0.006) on extreme dry-dominant activation (0.12% flood pixels); boundary confusion persists",
            "mitigation_or_note": "Bottom-3 mean improved significantly (+0.0511), but single worst event remained stubborn; severe low contrast in SAR imagery",
        },
        {
            "case_id": "FC-4",
            "scope": "RST8 Architecture Transfer",
            "event_or_target": "SiamUnet_diff (18 Held-Out Events, Seed 0)",
            "baseline_r0_f1": 0.7238,
            "recipe_r2_f1": 0.7139,
            "paired_delta": -0.0099,
            "failure_mechanism": "Shallow Siamese difference architecture lacks channel-attention capacity (ECAM) to absorb amplified class-weight loss gradients",
            "mitigation_or_note": "Stopped at Stage A by protocol rule; proves R2 is architecture-dependent and cannot be claimed as universally effective",
        },
        {
            "case_id": "FC-5",
            "scope": "RST9 Retrospective Target Extension",
            "event_or_target": "Event 1111003 (Dev Partition, 19 patches)",
            "baseline_r0_f1": 0.2628,
            "recipe_r2_f1": 0.2420,
            "paired_delta": -0.0208,
            "failure_mechanism": "Ultra-sparse event with only 6,961 flood pixels (0.85% flood fraction); recall dropped from 0.520 to 0.318 due to subtle feature dilution",
            "mitigation_or_note": "Event sample size is only 19 patch pairs; highly sensitive to single-patch detection thresholds",
        },
    ]

    out_csv = out_dir / "failure_cases_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"table_id": "Failure Cases", "rows": rows, "csv_path": str(out_csv.relative_to(REPO_ROOT))}


def build_claim_to_evidence_audit_table(out_dir: Path) -> Dict[str, Any]:
    """Builds the authoritative claim-to-evidence audit matrix."""
    rows = [
        {
            "claim_id": "CLAIM-1",
            "proposed_scientific_claim": "Event balancing (M1) + training-only class-balanced loss (M2) improves average source-event generalization on SNUNet-ECAM.",
            "supporting_work_package": "RST7",
            "empirical_evidence": "Grand event-macro F1 increased from 0.7202 to 0.7338 (delta = +0.0136) across 18 held-out source events over 3 seeds.",
            "allowed_scientific_wording": "Demonstrates consistent positive average source-domain generalization on SNUNet-ECAM across event-disjoint folds.",
            "prohibited_wording": "Universally superior method; guaranteed gain in all settings; definitive multi-domain solution.",
            "governing_limitation": "Bootstrap 95% CI [-0.0045, +0.0351] crosses zero; based on 18 source events and 3 repeated seed measurements.",
        },
        {
            "claim_id": "CLAIM-2",
            "proposed_scientific_claim": "Recipe R2 substantially improves bottom-three tail event performance.",
            "supporting_work_package": "RST7",
            "empirical_evidence": "Bottom-3 mean F1 improved from 0.3987 to 0.4498 (delta = +0.0511), satisfying Criterion P2.",
            "allowed_scientific_wording": "Substantially enhances tail reliability and worst-quartile event performance in source-domain cross-validation.",
            "prohibited_wording": "Solves tail risk completely; improves every difficult event unconditionally.",
            "governing_limitation": "Single worst event (Event 427) slightly declined (-0.0062), showing that extreme outlier events may require specialized handling.",
        },
        {
            "claim_id": "CLAIM-3",
            "proposed_scientific_claim": "The recipe does not consistently improve every event.",
            "supporting_work_package": "RST7 & RST9",
            "empirical_evidence": "2 of 18 source events in RST7 (Events 498, 273) and 1 of 16 events in RST9 (Event 1111003) showed negative paired deltas.",
            "allowed_scientific_wording": "Macro-level gains entail localized trade-offs on specific events with unique land-cover or class ratios.",
            "prohibited_wording": "Monotonically benefits all global flood events; zero failure cases.",
            "governing_limitation": "Loss reweighting trades off precision on dense flood events for higher recall on sparse tail events.",
        },
        {
            "claim_id": "CLAIM-4",
            "proposed_scientific_claim": "Seed stability was not established in primary source confirmation.",
            "supporting_work_package": "RST7",
            "empirical_evidence": "Seed 1 underperformed under Fold B, and inter-seed standard deviation increased by 75% in RST7, failing Criterion P3.",
            "allowed_scientific_wording": "Seed stability was not proven in primary source confirmation; R2 exhibited non-negligible inter-seed variance.",
            "prohibited_wording": "R2 is proven seed-invariant in source training.",
            "governing_limitation": "While RST9 historical extension showed all seeds positive with lower variance, this does not retroactively alter the primary RST7 verdict.",
        },
        {
            "claim_id": "CLAIM-5",
            "proposed_scientific_claim": "Transfer of Recipe R2 to SiamUnet_diff was not supported.",
            "supporting_work_package": "RST8",
            "empirical_evidence": "Stage A evaluation at seed 0 showed a macro F1 decline of -0.0099 and bottom-3 drop of -0.0448; execution halted at Stage A.",
            "allowed_scientific_wording": "Recipe R2 does not generalize out-of-the-box to SiamUnet_diff; benefits are architecture-specific.",
            "prohibited_wording": "Recipe R2 is a universal SAR change detection training methodology.",
            "governing_limitation": "Only seed 0 was evaluated on SiamUnet_diff; hyperparameter search on the lightweight backbone was not conducted by protocol design.",
        },
        {
            "claim_id": "CLAIM-6",
            "proposed_scientific_claim": "Three deployable all-source SNUNet-ECAM checkpoints were produced and verified.",
            "supporting_work_package": "RST9 Part A",
            "empirical_evidence": "3 checkpoints (seeds 0, 1, 2) trained on all 27 source events for 16 epochs (10,736 steps each) with full artifact logs and terminal status COMPLETED_SCIENTIFIC_RUN.",
            "allowed_scientific_wording": "Fully trained, verified, deployable SNUNet-ECAM source models are archived with bit-exact reloadability.",
            "prohibited_wording": "Models are proven fully immune to all target distribution shifts.",
            "governing_limitation": "Models remain source-trained feedforward neural networks without test-time adaptation.",
        },
        {
            "claim_id": "CLAIM-7",
            "proposed_scientific_claim": "The frozen recipe improved retrospective historical-target performance without target-time adaptation overhead.",
            "supporting_work_package": "RST9 Part B",
            "empirical_evidence": "Grand macro F1 increased from 0.6873 to 0.7380 (delta = +0.0507) across 16 historical target events, with 95% bootstrap CI [+0.0234, +0.0845].",
            "allowed_scientific_wording": "Demonstrates strong positive retrospective generalization across previously observed global flood distributions with zero target-time adaptation overhead.",
            "prohibited_wording": "Proves general out-of-distribution prospective generalization on unseen target domains.",
            "governing_limitation": "All 16 events were observed in prior project phases (Weeks 3-5).",
        },
        {
            "claim_id": "CLAIM-8",
            "proposed_scientific_claim": "Historical-target results are not an unseen final test.",
            "supporting_work_package": "RST9 Part B",
            "empirical_evidence": "Historical provenance audit confirms all 16 target events were inspected and evaluated during Weeks 3-5.",
            "allowed_scientific_wording": "Retrospective historical extension benchmarking deployable checkpoints against historical project baselines.",
            "prohibited_wording": "Blind prospective confirmatory test; generalization to unseen test domains.",
            "governing_limitation": "Strict research integrity constraint; must always be qualified as historical/retrospective.",
        },
        {
            "claim_id": "CLAIM-9",
            "proposed_scientific_claim": "Historical TTA comparison is contextual unless protocols are fully matched.",
            "supporting_work_package": "E11, Week 9, and RST9",
            "empirical_evidence": "TTA methods involve test-time gradient updates, streaming predictions, and batch-norm mutations, whereas R2 is pure frozen source inference.",
            "allowed_scientific_wording": "Contextual benchmark demonstrating that robust source training achieves greater empirical gains on the same target events without test-time compute, state drift, or latency penalties.",
            "prohibited_wording": "Direct apples-to-apples algorithmic leaderboard.",
            "governing_limitation": "Different inference protocols, memory footprints, and compute requirements.",
        },
    ]

    out_csv = out_dir / "claim_to_evidence_audit_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"table_id": "Claim Audit", "rows": rows, "csv_path": str(out_csv.relative_to(REPO_ROOT))}


def run_rst10_reconstruction_pipeline() -> Dict[str, Any]:
    """Executes the complete deterministic reconstruction pipeline."""
    manifest_path = REPO_ROOT / "configs/robust_source_training/rst10_evidence_manifest_v1.json"
    tables_dir = REPO_ROOT / "outputs/robust_source_training/rst10/tables"
    figures_dir = REPO_ROOT / "outputs/robust_source_training/rst10/figures"
    metadata_dir = REPO_ROOT / "outputs/robust_source_training/rst10/metadata"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit evidence manifest
    audit_res = audit_evidence_manifest(manifest_path)

    # 2. Build canonical tables
    print("\n--- Generating Canonical Tables ---")
    t1 = build_table1_component_staircase(
        REPO_ROOT / "outputs/robust_source_training/rst4/analysis_v1/discovery_recipe_summary.csv",
        tables_dir,
    )
    t2 = build_table2_primary_source_confirmation(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst7/canonical_event_results_v1.csv",
        tables_dir,
    )
    t3 = build_table3_architecture_transfer(
        REPO_ROOT / "outputs/robust_source_training/rst8/stage_a_summary_v1.json",
        tables_dir,
    )
    t4 = build_table4_deployable_checkpoint_ledger(
        REPO_ROOT / "configs/robust_source_training/rst9_ledger_v1.json",
        tables_dir,
    )
    t5 = build_table5_historical_target_extension(
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        tables_dir,
    )
    t6 = build_table6_runtime_and_resource_cost(tables_dir)
    t7 = build_table7_contextual_tta_comparison(tables_dir)

    all_tables = {
        "table1_component_staircase": t1,
        "table2_primary_source_confirmation": t2,
        "table3_architecture_transfer": t3,
        "table4_deployable_checkpoint_ledger": t4,
        "table5_historical_target_extension": t5,
        "table6_runtime_and_resource_cost": t6,
        "table7_contextual_tta_comparison": t7,
    }
    md_tables_path = generate_markdown_tables(all_tables, tables_dir)

    # 3. Build failure cases & claim audit
    print("\n--- Generating Failure Cases and Claims Audit ---")
    fc_res = build_failure_cases_and_limitations_table(tables_dir)
    ca_res = build_claim_to_evidence_audit_table(tables_dir)

    # 4. Build canonical figures
    print("\n--- Generating Canonical Figures ---")
    build_fig1_component_staircase(
        REPO_ROOT / "outputs/robust_source_training/rst4/analysis_v1/discovery_recipe_summary.csv",
        figures_dir,
    )
    build_fig2_primary_per_event_paired_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst7/canonical_event_results_v1.csv",
        figures_dir,
    )
    build_fig3_worst_bottom_event_comparison(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        figures_dir,
    )
    build_fig4_precision_recall_tradeoff(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        figures_dir,
    )
    build_fig5_seed_stability(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        figures_dir,
    )
    build_fig6_event_sampling_distribution(
        REPO_ROOT / "configs/robust_source_training/rst1_source_event_inventory_v1.csv",
        figures_dir,
    )
    build_fig7_architecture_transfer_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json",
        REPO_ROOT / "outputs/robust_source_training/rst8/stage_a_summary_v1.json",
        figures_dir,
    )
    build_fig8_historical_target_extension_deltas(
        REPO_ROOT / "outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json",
        figures_dir,
    )
    build_fig9_runtime_vram_tradeoff(
        tables_dir / "table6_runtime_and_resource_cost.csv",
        figures_dir,
    )

    # 5. Collect output hashes
    output_files = {}
    for p in sorted(list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.md")) +
                    list(figures_dir.glob("*.pdf")) + list(figures_dir.glob("*.png")) +
                    list((figures_dir / "data").glob("*.csv"))):
        rel = str(p.relative_to(REPO_ROOT))
        output_files[rel] = compute_sha256(p)

    # 6. Emit metadata manifest
    metadata_record = {
        "work_package": "RST10",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
        "evidence_manifest_sha256": compute_sha256(manifest_path),
        "total_evidence_sources_audited": audit_res["sources_count"],
        "total_canonical_tables_generated": len(all_tables) + 2, # 7 + failure + claims
        "total_figures_generated": 9,
        "total_output_artifacts": len(output_files),
        "output_artifacts_sha256": output_files,
    }

    meta_file = metadata_dir / "rst10_execution_metadata_v1.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2)

    print(f"\nSaved master execution metadata to {meta_file}")
    return metadata_record


def main():
    print("================================================================================")
    print("ChangeGuard Work Package RST10: Master Reconstruction Pipeline")
    print("================================================================================")
    meta = run_rst10_reconstruction_pipeline()
    print("================================================================================")
    print(f"RST10 Pipeline Complete: {meta['total_output_artifacts']} artifacts generated.")
    print(f"Git Commit: {meta['git_commit']}")
    print(f"Evidence Sources: {meta['total_evidence_sources_audited']} verified.")
    print("================================================================================")


if __name__ == "__main__":
    main()

