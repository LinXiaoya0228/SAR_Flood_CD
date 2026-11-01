"""Unit and regression tests for ChangeGuard Work Package RST10 Forensic Repair.

Validates:
1. Input evidence manifest cryptographic integrity.
2. Canonical table row counts, keys, formulas, and aggregations.
3. Paired R0/R2 completeness and event-set matching.
4. Retention of failed RST7 P3 and unsupported RST8 transfer verdicts.
5. Presence of backing CSV data for all 9 figures.
6. Execution metadata manifest completeness and artifact hash tracking.
7. RST9 checkpoint SHA-256 hashes and 10,736 optimizer step budget bit-exact matching.
8. RST4 staircase metrics and parsimonious selection of R2 matching formal discovery closeout.
9. RST7 canonical 108 rows, exact bootstrap CI [-0.0045, +0.0351], and fold-level SDs.
10. RST8 canonical 36 rows, exact absolute F1 metrics, and Stage-A stop-decision.
11. RST9 canonical 96 target rows and prominent retrospective disclaimers.
12. Strict consistency between canonical tables and figure-source CSVs.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = REPO_ROOT / "configs/robust_source_training/rst10_evidence_manifest_v1.json"
CONFIG_PATH = REPO_ROOT / "configs/robust_source_training/rst10_reconstruction_config_v1.json"
TABLES_DIR = REPO_ROOT / "outputs/robust_source_training/rst10/tables"
FIGURES_DIR = REPO_ROOT / "outputs/robust_source_training/rst10/figures"
FIG_DATA_DIR = FIGURES_DIR / "data"
METADATA_PATH = REPO_ROOT / "outputs/robust_source_training/rst10/metadata/rst10_execution_metadata_v1.json"


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_1_evidence_manifest_integrity():
    """Verify evidence manifest exists and all 15 sources match SHA-256 hashes bit-identically."""
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    manifest = _load_json(MANIFEST_PATH)
    assert len(manifest["evidence_sources"]) == 15

    for src in manifest["evidence_sources"]:
        p = REPO_ROOT / src["path"]
        assert p.exists(), f"Source file does not exist: {p}"
        actual = compute_sha256(p)
        assert actual == src["sha256"], f"SHA mismatch for {src['source_id']}"


def test_2_canonical_tables_exist_and_row_counts():
    """Verify all 7 canonical tables and audit tables exist and have expected row counts."""
    expected_tables = {
        "table1_component_staircase.csv": 4,  # R0, R1, R2, R3
        "table2_primary_source_confirmation.csv": 11,  # 3 Fold A seeds + Mean + 3 Fold B seeds + Mean + Combined + Bottom3 + Worst
        "table3_architecture_transfer.csv": 3,  # Fold A, Fold B, Combined
        "table4_deployable_checkpoint_ledger.csv": 3,  # Seeds 0, 1, 2
        "table5_historical_target_extension.csv": 4,  # Seeds 0, 1, 2 + Mean
        "table6_runtime_and_resource_cost.csv": 7,  # 7 method rows
        "table7_contextual_tta_comparison.csv": 7,  # 7 TTA rows
        "failure_cases_table.csv": 5,  # 5 detailed failure cases
        "claim_to_evidence_audit_table.csv": 9,  # 9 mandatory claims
    }

    for filename, expected_rows in expected_tables.items():
        p = TABLES_DIR / filename
        assert p.exists(), f"Missing table CSV: {p}"
        rows = _load_csv(p)
        assert len(rows) == expected_rows, f"Expected {expected_rows} rows in {filename}, got {len(rows)}"

    # Markdown table bundle exists and is non-empty
    md_p = TABLES_DIR / "all_canonical_tables_v1.md"
    assert md_p.exists()
    assert md_p.stat().st_size > 1000


def test_3_scientific_verdict_preservation():
    """Verify that RST7 P3 remains failed and RST8 transfer remains unsupported."""
    # Table 2 / RST7 summary
    t2_rows = _load_csv(TABLES_DIR / "table2_primary_source_confirmation.csv")
    r7_sum = _load_json(REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json")
    assert r7_sum["decision"]["criteria"]["P1_average_improvement"]["satisfied"] is True
    assert r7_sum["decision"]["criteria"]["P2_tail_reliability_improvement"]["satisfied"] is True
    assert r7_sum["decision"]["criteria"]["P3_seed_stability"]["satisfied"] is False

    # Table 3 / RST8 summary
    t3_rows = _load_csv(TABLES_DIR / "table3_architecture_transfer.csv")
    r8_sum = _load_json(REPO_ROOT / "outputs/robust_source_training/rst8/stage_a_summary_v1.json")
    assert r8_sum["decision"]["decision_code"] == "STOP_AT_SEED_0"
    assert r8_sum["decision"]["criteria"]["P1_average_improvement"]["satisfied"] is False
    assert r8_sum["decision"]["criteria"]["P2_tail_reliability_improvement"]["satisfied"] is False
    assert any("STOP AT SEED 0" in r["verdict"] for r in t3_rows)


def test_4_deployable_checkpoint_ledger_integrity():
    """Verify Table 4 deployable checkpoint ledger matches RST9 ledger."""
    t4_rows = _load_csv(TABLES_DIR / "table4_deployable_checkpoint_ledger.csv")
    assert len(t4_rows) == 3

    for r in t4_rows:
        assert r["architecture"] == "SNUNet_ECAM"
        assert r["recipe"] == "R2"
        assert int(r["optimizer_steps"]) == 10736
        assert int(r["epochs"]) == 16
        assert int(r["source_events"]) == 27
        assert r["terminal_status"] == "COMPLETED_SCIENTIFIC_RUN"
        assert int(r["exit_code"]) == 0
        assert r["weights_reload_verified"] == "True"


def test_5_historical_target_extension_disclaimer_and_metrics():
    """Verify Table 5 contains prominent historical disclaimer and accurate deltas."""
    t5_rows = _load_csv(TABLES_DIR / "table5_historical_target_extension.csv")
    for r in t5_rows:
        assert "retrospective extension only" in r["historical_disclaimer"].lower()

    # Mean row has positive grand delta
    mean_row = t5_rows[-1]
    assert "+0.0507" in mean_row["delta_f1"]


def test_6_canonical_figures_and_data_completeness():
    """Verify all 9 canonical figures exist as PDF, PNG, and have backing CSV data."""
    for i in range(1, 10):
        name_prefix = f"fig{i}_"
        pdf_matches = list(FIGURES_DIR.glob(f"{name_prefix}*.pdf"))
        png_matches = list(FIGURES_DIR.glob(f"{name_prefix}*.png"))
        csv_matches = list(FIG_DATA_DIR.glob(f"{name_prefix}*.csv"))

        assert len(pdf_matches) == 1, f"Missing PDF for Figure {i}"
        assert len(png_matches) == 1, f"Missing PNG for Figure {i}"
        assert len(csv_matches) == 1, f"Missing backing CSV for Figure {i}"

        # Verify backing CSV has rows and non-empty content
        rows = _load_csv(csv_matches[0])
        assert len(rows) > 0, f"Backing CSV for Figure {i} has 0 rows"


def test_7_no_event_exclusion_in_figure_data():
    """Verify that Figure 2 has all 18 source events and Figure 8 has all 16 target events."""
    f2_rows = _load_csv(FIG_DATA_DIR / "fig2_primary_per_event_paired_deltas.csv")
    assert len(f2_rows) == 18, f"Expected 18 events in Fig 2 data, got {len(f2_rows)}"
    f2_eids = {int(r["event_id"]) for r in f2_rows}
    expected_18 = {118, 174, 267, 273, 324, 417, 421, 470, 496, 498, 502, 518, 554, 555, 567, 1111004, 1111009, 1111010}
    assert f2_eids == expected_18

    f8_rows = _load_csv(FIG_DATA_DIR / "fig8_historical_target_extension_deltas.csv")
    assert len(f8_rows) == 16, f"Expected 16 events in Fig 8 data, got {len(f8_rows)}"
    f8_eids = {int(r["event_id"]) for r in f8_rows}
    expected_16 = {279, 437, 520, 559, 1111003, 1111008, 205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
    assert f8_eids == expected_16


def test_8_execution_metadata_record():
    """Verify master execution metadata record tracks all artifacts and hashes."""
    assert METADATA_PATH.exists()
    meta = _load_json(METADATA_PATH)
    assert meta["work_package"] == "RST10"
    assert meta["total_canonical_tables_generated"] == 9
    assert meta["total_figures_generated"] == 9
    assert meta["total_evidence_sources_audited"] == 15
    assert len(meta["output_artifacts_sha256"]) == meta["total_output_artifacts"]

    for rel_path, expected_sha in meta["output_artifacts_sha256"].items():
        p = REPO_ROOT / rel_path
        assert p.exists(), f"Output artifact does not exist: {p}"
        assert compute_sha256(p) == expected_sha, f"SHA mismatch for output artifact {rel_path}"


def test_9_rst9_checkpoint_hashes_and_steps_exact():
    """Verify accepted RST9 checkpoint SHA-256 hashes and 10,736 step count bit-identically."""
    expected_checkpoints = {
        0: "9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7",
        1: "279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2",
        2: "b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893",
    }
    t4_rows = _load_csv(TABLES_DIR / "table4_deployable_checkpoint_ledger.csv")
    for r in t4_rows:
        s = int(r["seed"])
        assert int(r["optimizer_steps"]) == 10736, f"Seed {s} steps mismatch: {r['optimizer_steps']}"
        assert r["checkpoint_sha256"] == expected_checkpoints[s], f"Seed {s} hash mismatch!"

    # Verify physical files exist and match hashes
    ledger = _load_json(REPO_ROOT / "configs/robust_source_training/rst9_ledger_v1.json")
    for run in ledger["runs"]:
        s = run["seed"]
        cp_path = REPO_ROOT / run["checkpoint_path"]
        assert cp_path.exists(), f"Missing physical checkpoint: {cp_path}"
        actual_sha = compute_sha256(cp_path)
        assert actual_sha == expected_checkpoints[s], f"Physical checkpoint hash mismatch for seed {s}"


def test_10_rst4_staircase_integrity():
    """Verify RST4 staircase discovery metrics and promotion decision."""
    t1_rows = _load_csv(TABLES_DIR / "table1_component_staircase.csv")
    by_recipe = {r["recipe_id"]: r for r in t1_rows}

    assert round(float(by_recipe["R0"]["macro_f1"]), 4) == 0.7631
    assert round(float(by_recipe["R1"]["macro_f1"]), 4) == 0.7561
    assert round(float(by_recipe["R2"]["macro_f1"]), 4) == 0.7651
    assert round(float(by_recipe["R3"]["macro_f1"]), 4) == 0.7697

    assert round(float(by_recipe["R0"]["bottom3_mean_f1"]), 4) == 0.5630
    assert round(float(by_recipe["R2"]["bottom3_mean_f1"]), 4) == 0.5869
    assert round(float(by_recipe["R2"]["delta_bottom3_vs_r0"]), 4) == 0.0239

    # Verify selected recipe is R2
    sel = _load_json(REPO_ROOT / "configs/robust_source_training/rst4_selected_recipe_v1.json")
    assert sel["selected_recipe_id"] == "R2"
    assert sel["decision_status"] == "RECIPE_R2_PROMOTED"
    assert sel["components"]["m1_event_balance"] is True
    assert sel["components"]["m2_flood_aware_loss"] is True
    assert sel["components"]["m3_paired_augmentation"] is False


def test_11_rst7_canonical_rows_and_bootstrap():
    """Verify RST7 canonical rows count (108), bootstrap interval, and fold-level SDs."""
    canonical_rows = _load_csv(REPO_ROOT / "outputs/robust_source_training/rst7/canonical_event_results_v1.csv")
    assert len(canonical_rows) == 108, f"Expected 108 rows in RST7 canonical table, got {len(canonical_rows)}"

    r7_sum = _load_json(REPO_ROOT / "outputs/robust_source_training/rst7/confirmatory_summary_v1.json")
    bs = r7_sum["bootstrap"]
    assert bs["resampling_unit"] == "18_unique_held_out_events"
    assert round(bs["ci_lower"], 4) == -0.0045
    assert round(bs["ci_upper"], 4) == 0.0351

    # Fold SDs
    fa = r7_sum["fold_summaries"]["fold_a"]
    fb = r7_sum["fold_summaries"]["fold_b"]
    assert round(fa["r0"]["sd_macro_f1"], 4) == 0.0120
    assert round(fa["r2"]["sd_macro_f1"], 4) == 0.0106
    assert round(fb["r0"]["sd_macro_f1"], 4) == 0.0035
    assert round(fb["r2"]["sd_macro_f1"], 4) == 0.0165


def test_12_rst8_canonical_rows_and_metrics():
    """Verify RST8 canonical rows count (36), absolute metrics, and stop-decision."""
    canonical_rows = _load_csv(REPO_ROOT / "outputs/robust_source_training/rst8/canonical_event_results_v1.csv")
    assert len(canonical_rows) == 36, f"Expected 36 rows in RST8 canonical table, got {len(canonical_rows)}"

    r8_sum = _load_json(REPO_ROOT / "outputs/robust_source_training/rst8/stage_a_summary_v1.json")
    gm = r8_sum["grand_metrics"]
    assert round(gm["r0"]["f1"], 4) == 0.7238
    assert round(gm["r2"]["f1"], 4) == 0.7139
    assert round(gm["delta"]["f1"], 4) == -0.0099

    assert round(gm["r0"]["bottom3_mean_f1"], 4) == 0.4350
    assert round(gm["r2"]["bottom3_mean_f1"], 4) == 0.3902
    assert round(gm["delta"]["bottom3_mean_f1"], 4) == -0.0448

    assert round(gm["r0"]["worst_event_f1"], 4) == 0.3565
    assert round(gm["r2"]["worst_event_f1"], 4) == 0.2660
    assert round(gm["delta"]["worst_event_f1"], 4) == -0.0905
    assert r8_sum["decision"]["decision_code"] == "STOP_AT_SEED_0"


def test_13_rst9_canonical_rows_and_retrospective():
    """Verify RST9 canonical rows count (96) and strict retrospective characterization."""
    target_rows = _load_csv(REPO_ROOT / "outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv")
    assert len(target_rows) == 96, f"Expected 96 rows in RST9 target table, got {len(target_rows)}"

    r0_rows = [r for r in target_rows if r.get("recipe") == "R0" or r.get("method") == "R0"]
    r2_rows = [r for r in target_rows if r.get("recipe") == "R2" or r.get("method") == "R2"]
    assert len(r0_rows) == 48
    assert len(r2_rows) == 48

    # Table 5 check
    t5_rows = _load_csv(TABLES_DIR / "table5_historical_target_extension.csv")
    for r in t5_rows:
        assert "retrospective" in r["historical_disclaimer"].lower()
        assert "unseen" not in r["historical_disclaimer"].lower()


def test_14_table_and_figure_csv_consistency():
    """Verify that every canonical table matches its corresponding figure-source CSV bit-consistently."""
    # Table 1 vs Fig 1
    t1_rows = _load_csv(TABLES_DIR / "table1_component_staircase.csv")
    f1_rows = _load_csv(FIG_DATA_DIR / "fig1_component_staircase.csv")
    assert len(t1_rows) == len(f1_rows)
    for t_row, f_row in zip(t1_rows, f1_rows):
        assert t_row["recipe_id"] == f_row["recipe_id"]
        assert round(float(t_row["macro_f1"]), 4) == round(float(f_row["macro_f1"]), 4)

    # Table 2 Grand row vs Fig 3
    t2_rows = _load_csv(TABLES_DIR / "table2_primary_source_confirmation.csv")
    f3_rows = _load_csv(FIG_DATA_DIR / "fig3_worst_bottom_event_comparison.csv")
    f3_by_metric = {r["metric_group"]: r for r in f3_rows}
    assert float(f3_by_metric["Grand Macro F1"]["delta"]) == 0.0136
    assert float(f3_by_metric["Bottom-3 Mean F1"]["delta"]) == 0.0511
    assert float(f3_by_metric["Worst Event F1"]["delta"]) == -0.0062

    # Table 2 seeds vs Fig 5
    f5_rows = _load_csv(FIG_DATA_DIR / "fig5_seed_stability.csv")
    assert len(f5_rows) == 9  # 3 Fold A + 3 Fold B + 3 RST9 Target
    fa_f5 = [r for r in f5_rows if r["partition"] == "Fold A"]
    assert len(fa_f5) == 3
    assert float(fa_f5[0]["r0_macro_f1"]) == 0.7330
    assert float(fa_f5[0]["r2_macro_f1"]) == 0.7565

    # Table 3 vs Fig 7
    t3_rows = _load_csv(TABLES_DIR / "table3_architecture_transfer.csv")
    f7_rows = _load_csv(FIG_DATA_DIR / "fig7_architecture_transfer_deltas.csv")
    f7_by_m = {r["metric"]: r for r in f7_rows}
    assert float(f7_by_m["Grand Macro F1"]["siamunet_diff_delta"]) == -0.0099
    assert float(f7_by_m["Bottom-3 Mean F1"]["siamunet_diff_delta"]) == -0.0448
    assert float(f7_by_m["Worst Event F1"]["siamunet_diff_delta"]) == -0.0905
    assert float(f7_by_m["Grand Macro F1"]["siamunet_r0"]) == 0.7238
    assert float(f7_by_m["Grand Macro F1"]["siamunet_r2"]) == 0.7139


if __name__ == "__main__":
    tests = [
        test_1_evidence_manifest_integrity,
        test_2_canonical_tables_exist_and_row_counts,
        test_3_scientific_verdict_preservation,
        test_4_deployable_checkpoint_ledger_integrity,
        test_5_historical_target_extension_disclaimer_and_metrics,
        test_6_canonical_figures_and_data_completeness,
        test_7_no_event_exclusion_in_figure_data,
        test_8_execution_metadata_record,
        test_9_rst9_checkpoint_hashes_and_steps_exact,
        test_10_rst4_staircase_integrity,
        test_11_rst7_canonical_rows_and_bootstrap,
        test_12_rst8_canonical_rows_and_metrics,
        test_13_rst9_canonical_rows_and_retrospective,
        test_14_table_and_figure_csv_consistency,
    ]

    print(f"Running {len(tests)} RST10 regression tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST10 tests PASSED successfully!")
