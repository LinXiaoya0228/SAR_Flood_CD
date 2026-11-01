# Reproduction Guide: Regenerating Paper Tables and Figures

This guide provides one-step commands to regenerate all 7 synthesis tables and 9 publication figures from published compact CSV/JSON result artifacts without requiring access to GPUs or raw large datasets.

---

## 1. Regenerating Canonical Synthesis Tables

Run `build_rst10_tables.py` from the repository root:

```bash
python tools/robust_source_training/build_rst10_tables.py
```

### Generated Tables (written to `results/tables/`):
- `table1_component_staircase.csv`: 4-arm discovery screening (R0, R1, R2, R3).
- `table2_primary_source_confirmation.csv`: Confirmatory evaluation on SNUNet-ECAM across Folds A & B (Seeds 0, 1, 2).
- `table3_architecture_transfer.csv`: SiamUnet_diff vs. SNUNet-ECAM at Seed 0.
- `table4_deployable_checkpoint_ledger.csv`: Full-source All-27 model provenance and hashes.
- `table5_historical_target_extension.csv`: Historical Kuro Siwo 10-target evaluation.
- `table6_runtime_and_resource_cost.csv`: Training and inference compute benchmarks.
- `table7_contextual_tta_comparison.csv`: R2 vs Tent and CoTTA adaptation.
- `all_canonical_tables_v1.md`: Consolidated Markdown table report.

---

## 2. Regenerating Publication Figures

Run `build_rst10_figures.py` from the repository root:

```bash
python tools/robust_source_training/build_rst10_figures.py
```

### Generated Figures (written to `results/figures/` in PDF and PNG):
- `fig1_component_staircase.{pdf,png}`
- `fig2_primary_per_event_paired_deltas.{pdf,png}` (Figure 1 in IGARSS 2027 paper)
- `fig3_worst_bottom_event_comparison.{pdf,png}`
- `fig4_precision_recall_tradeoff.{pdf,png}`
- `fig5_seed_stability.{pdf,png}`
- `fig6_event_sampling_distribution.{pdf,png}`
- `fig7_architecture_transfer_deltas.{pdf,png}`
- `fig8_historical_target_extension_deltas.{pdf,png}`
- `fig9_runtime_vram_tradeoff.{pdf,png}`

---

## 3. Regenerating Supplementary Figures

### Offline R2 vs TTA Comparison
```bash
python tools/robust_source_training/run_r2_vs_tta_offline_comparison.py
```
Output: `outputs/robust_source_training/rst12_tta_comparison/r2_vs_tta_comparison_figure.png`

### External GEOID-Flood Evaluation Comparison
```bash
python scripts/external_evaluation/analyze_geoid_nan_corrected.py
```
Output: `outputs/external_evaluation/geoid_nan_corrected_v1/geoid_r2_vs_r0_comparison_figure_nan_corrected.png`

