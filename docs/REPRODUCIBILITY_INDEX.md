# ChangeGuard Claim-to-Artifact Reproducibility Index

This document establishes the canonical mapping between all reported tables, figures, and empirical claims in the ChangeGuard research project (including the IEEE IGARSS 2027 conference paper draft and Final Year Project report) and their underlying configuration files, execution scripts, raw source artifacts, and scope.

---

## 1. Experimental Lineage & Status Overview

| Work Package / Lineage | Description | Models & Seeds | Status | Primary Output Directory |
|---|---|---|---|---|
| **RST4 Discovery** | 4-arm component staircase (R0, R1, R2, R3) | SNUNet-ECAM, Fold C (9 events), Seed 0 | **Completed** | `results/canonical/discovery_recipe_summary.csv` |
| **RST6 & RST7 Confirmatory** | Pre-registered 18-event confirmation across 2 disjoint folds | SNUNet-ECAM, Folds A & B (18 events), Seeds 0, 1, 2 (12 runs) | **Completed** | `results/canonical/confirmatory_summary_v1.json`, `results/canonical/canonical_event_results_v1.csv` |
| **RST8 Architecture Transfer** | Cross-backbone evaluation on SiamUnet_diff | SiamUnet_diff, Folds A & B, Seed 0 (4 runs) | **Completed** (Screening) | `results/canonical/stage_a_summary_v1.json` |
| **RST8 Supplementary Replication** | Supplementary 3-seed replication on SiamUnet_diff | SiamUnet_diff, Folds A & B, Seeds 0, 1, 2 (12 runs) | **Completed** | `results/canonical/rst8_full_3seed_summary_v1.json`, `results/canonical/rst8_canonical_event_results_full_3seed_v1.csv` |
| **RST9 All-27 Deployment** | Full-source models trained on all 27 events | SNUNet-ECAM, All 27 events, Seeds 0, 1, 2 (3 runs) | **Completed** | `results/canonical/rst9_ledger_v1.json` |
| **RST10 Synthesis** | Consolidated tables and figures for reporting | Cross-study canonical outputs | **Completed** | `results/tables/`, `results/figures/` |
| **M2_ONLY Ablation** | Class weighting without event-balanced sampling | SNUNet-ECAM, Folds A & B, Seeds 0, 1, 2 (6 runs) | **Completed** | `results/canonical/m2_only_ablation_summary.json` |
| **Historical Target & TTA** | Retrospective evaluation on 10 Kuro Siwo target events | SNUNet-ECAM All-27 (R0, R2) vs Tent, CoTTA | **Completed** | `results/canonical/r2_vs_tta_summary.json` |
| **R2 + TTA Supplementary Evaluation** | Bounded supplementary evaluation of Tent & CoTTA on frozen R2 source models | SNUNet-ECAM All-27 R2 (Seeds 0, 1, 2) + Tent / CoTTA (6 streams, 60 event rows) | **Completed** | `results/canonical/r2_tta_analysis_summary_v1.json`, `results/canonical/r2_tta_canonical_event_results_v1.csv` |
| **External GEOID-Flood** | Zero-shot generalization on 1,301 tiles across 9 EMS events | SNUNet-ECAM All-27 (R0, R2), Seeds 0, 1, 2 (6 runs) | **Completed** (NaN-Corrected) | `results/canonical/geoid_nan_corrected_summary.json` |

---

## 2. Manuscript Tables (IGARSS 2027 & FYP Report)

### Table 1: Primary Confirmatory Results on SNUNet-ECAM
- **Manuscript Location:** IGARSS 2027 Table 1 (`papers/igarss2027/tables/table1_primary_confirmation.tex`), FYP Report Table 2.
- **Scope:** 18 held-out source events evaluated across two 9-event folds (Fold A and Fold B) and 3 initialization seeds (0, 1, 2); 12 full training runs.
- **Key Results:**
  - Grand Macro F1: R0 = $0.7202 \pm 0.0044$, R2 = $0.7338 \pm 0.0132$ ($\Delta = +0.0136$, 95% event bootstrap CI `[-0.0045, +0.0351]`, P1 passed).
  - Bottom-3 Event Tail Mean F1: R0 = $0.3987 \pm 0.0298$, R2 = $0.4498 \pm 0.0248$ ($\Delta = +0.0511$, P2 passed).
  - Seed Variance Ratio: $+75.1\%$ ($\Delta\text{Var} > 0$, P3 failed).
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:**
  - Canonical CSV: `results/tables/table2_primary_source_confirmation.csv`
  - Input Summary: `results/canonical/confirmatory_summary_v1.json`
  - Canonical Per-Event Results: `results/canonical/canonical_event_results_v1.csv`
- **Governing Configs:** `configs/robust_source_training/rst5_confirmation/rst6_r0_snunet_*.json`, `rst6_r2_snunet_*.json`

### Table 2: Architecture Transfer to SiamUnet_diff (Seed 0 Screening)
- **Manuscript Location:** IGARSS 2027 Table 2 (`papers/igarss2027/tables/table2_architecture_transfer.tex`), FYP Report Table 3.
- **Scope:** SiamUnet_diff vs. SNUNet-ECAM at Seed 0 across 18 source events (Folds A & B); 4 runs.
- **Key Results:**
  - SiamUnet_diff Grand Macro F1: R0 = $0.7238$, R2 = $0.7139$ ($\Delta = -0.0099$, negative transfer).
  - SiamUnet_diff Bottom-3 Tail Mean F1: R0 = $0.4350$, R2 = $0.3902$ ($\Delta = -0.0448$, tail degradation).
  - Decision: Triggered pre-registered stopping rule at Seed 0.
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:**
  - Canonical CSV: `results/tables/table3_architecture_transfer.csv`
  - Input Summary: `results/canonical/stage_a_summary_v1.json`
  - Per-Event Records: `results/canonical/rst8_canonical_event_results_v1.csv`
- **Governing Configs:** `configs/robust_source_training/rst8_siamunet_diff/rst8_*_siamunet_diff_*_s0_v1.json`

### Table 3 (FYP Table 1): Component Staircase (Discovery Screening)
- **Scope:** 4-arm ablation on Discovery Split Group C (9 events), Seed 0; SNUNet-ECAM.
- **Key Results (from `results/canonical/discovery_recipe_summary.csv`):**
  - R0 (Matched Baseline): Event-Macro F1 $0.7631$, Bottom-3 F1 $0.5630$.
  - R1 (M1 Event-Balanced Sampling): Event-Macro F1 $0.7561$ ($\Delta = -0.0070$), Bottom-3 F1 $0.5426$ ($\Delta = -0.0204$).
  - R2 (M1 + M2 Inverse-Square-Root Class Weights): Event-Macro F1 $0.7651$ ($\Delta = +0.0020$), Bottom-3 F1 $0.5869$ ($\Delta = +0.0239$, $+2.39\text{ pp}$ on tail).
  - R3 (M1 + M2 + M3 SAR Augmentation): Event-Macro F1 $0.7697$ ($\Delta = +0.0065$), Bottom-3 F1 $0.5824$ ($\Delta = +0.0194$).
- **Archived Selection Rationale:** Under the pre-registered promotion rule (`configs/robust_source_training/rst4_promotion_rule_v1.json`), both R2 and R3 qualified for promotion (R2 qualified via Route D2 for tail recovery, $\Delta_{\text{bottom-3}} \ge +0.020$; R3 qualified via Route D1 for overall F1 gain). Top candidate event-macro F1 was R3 ($0.7697$). The pre-registered parsimony rule specified a $0.005$ tolerance window: $[0.7697 - 0.005, 0.7697] = [0.7647, 0.7697]$. Because R2 ($0.7651$) fell within $0.005$ of R3 ($0.7697$), the simpler recipe (parsimony preference R1 $\to$ R2 $\to$ R3) was promoted, eliminating the training augmentation jitter of M3 while achieving the highest tail disaster recovery ($+2.39\text{ pp}$ on bottom-3 events).
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:** `results/tables/table1_component_staircase.csv`, `results/canonical/discovery_recipe_summary.csv`
- **Governing Configs:** `configs/robust_source_training/rst4_*.json`

### Table 4 (FYP Table 4): Deployable Checkpoint Ledger (All-27 Source Models)
- **Scope:** Full-source SNUNet-ECAM models trained on all 27 events across Seeds 0, 1, 2 (16 epochs, 10,736 optimizer steps each).
- **Key Content:** Run ID, SHA-256 checkpoint hash, parameter count ($12,034,819$), peak VRAM ($4,913$ MB), training runtime ($3.2$–$3.4$ hours).
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:** `results/tables/table4_deployable_checkpoint_ledger.csv`, `results/canonical/rst9_ledger_v1.json`

### Table 5 (FYP Table 5): Historical Kuro Siwo Target Extension
- **Scope:** All-27 R0 vs R2 models evaluated on historical Kuro Siwo target events (16 epochs, 3 seeds).
- **Key Results:**
  - All 16 Historical Events (Grand Macro F1): R0 = $0.6873 \pm 0.0223$, R2 = $0.7380 \pm 0.0059$ ($\Delta = +0.0507$, +5.07 pp, 95% bootstrap CI `[+0.0234, +0.0845]`).
  - 10 Former Test Events (TTA Benchmark Subset): R0 = $0.7453$, R2 = $0.8086$ ($\Delta = +0.0633$, +6.33 pp).
  - 6 Historical Dev Events: R0 = $0.5906$, R2 = $0.6202$ ($\Delta = +0.0296$).
  - Historical Tail (Bottom-3 Events): R0 = $0.3889$, R2 = $0.4607$ ($\Delta = +0.0719$, +7.19 pp).
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:** `results/tables/table5_historical_target_extension.csv`, `results/canonical/rst9_comparative_analysis_v1.json`

### Table 6 (FYP Table 6): Runtime and Resource Cost Benchmarks
- **Scope:** Training throughput, peak memory, evaluation latency per patch and per tile across all architectures.
- **Generating Script:** `tools/robust_source_training/build_rst10_tables.py`
- **Tracked Source Artifacts:** `results/tables/table6_runtime_and_resource_cost.csv`

### Table 7 (FYP Table 7): Contextual Test-Time Adaptation (TTA) Comparison
- **Scope:** Offline comparison of frozen source-training models against test-time adaptation methods (Tent, CoTTA, Source Anchor) across 10 Kuro Siwo target events.
- **Key Results:**
  - Macro F1 on 10 Target Events: Source Anchor (R0) = $0.7453$, Tent = $0.7628$, CoTTA = $0.7669$, R2 (Source-Trained) = $0.8086$.
  - R2 vs Tent Delta: $+0.0458$ (95% CI `[+0.0274, +0.0648]`).
  - R2 vs CoTTA Delta: $+0.0417$ (95% CI `[+0.0260, +0.0585]`).
  - Operational characteristics: R2 operates in standard feed-forward mode without test-time parameter updates, backpropagation passes, or adaptation buffers.
- **Generating Script:** `tools/robust_source_training/run_r2_vs_tta_offline_comparison.py`
- **Tracked Source Artifacts:** `results/tables/table7_contextual_tta_comparison.csv`, `results/canonical/r2_vs_tta_summary.json`, `results/canonical/r2_vs_tta_event_averaged_paired.csv`

### Supplementary Table S1: M2_ONLY Weighting Ablation
- **Scope:** 6 runs (Folds A & B, Seeds 0, 1, 2) evaluating R0 uniform patch sampling + M2 inverse-square-root class weighting.
- **Key Results:**
  - Grand Macro F1: R0 = $0.7202$, M2_ONLY = $0.7126$ ($\Delta = -0.0076$ vs R0, 95% CI `[-0.0312, +0.0172]`).
  - Adding Event Balancing (R2 vs M2_ONLY): R2 = $0.7338$, $\Delta = \mathbf{+0.0212}$ (95% CI `[+0.0010, +0.0442]`, $p = 0.0181$).
  - Bottom-3 Tail F1: R0 = $0.3987$, M2_ONLY = $0.4270$ ($\Delta = +0.0283$ vs R0), R2 = $0.4498$ ($\Delta = +0.0511$ vs R0, $+0.0228$ vs M2_ONLY).
  - Finding: Class weighting alone without event balancing slightly reduces overall macro F1; combining event-balanced sampling with class weighting (R2) produces statistically significant overall gains and substantial tail recovery.
- **Generating Script:** `tools/robust_source_training/analyze_m2_only_ablation.py`
- **Tracked Source Artifacts:** `results/canonical/m2_only_ablation_summary.json`, `results/canonical/m2_only_ablation_canonical_records.csv`, `results/canonical/m2_only_ablation_per_event_paired.csv`

### Supplementary Table S2: External Zero-Shot Evaluation on GEOID-Flood
- **Scope:** 6 frozen SNUNet-ECAM all-27 checkpoints (R0 seeds 0/1/2, R2 seeds 0/1/2) evaluated on 1,301 paired S1-GRD tiles ($873,090,528$ valid pixels) across 9 Copernicus Emergency Management Service (EMS) activations under verified input transform parity (NaN filled with 0.15).
- **Key Results:**
  - 9-Event Macro F1: R0 = $0.5492 \pm 0.0326$, R2 = $0.5475 \pm 0.0167$ ($\Delta = -0.0017$, 95% event bootstrap CI `[-0.0238, +0.0217]`). The paired difference is not statistically distinguishable from zero at the 95% confidence level. *(An interval spanning zero indicates that a non-zero effect was not detected in this sample; it does not demonstrate equivalence or statistical parity between recipes).*
  - Precision / Recall Shift: Precision R0 = $0.4972$, R2 = $0.5439$ ($\Delta = \mathbf{+0.0466}$, +4.66 pp precision gain); Recall R0 = $0.6519$, R2 = $0.5994$ ($\Delta = -0.0525$).
  - Grouped 8-Event Sensitivity (merging EMSR861 & EMSR864 Iberia floods): $\Delta = -0.0003$ (95% CI `[-0.0250, +0.0247]`).
  - Pooled Dataset F1 (873M pixels): R0 = $0.7154$, R2 = $0.7156$ ($\Delta = +0.0002$).
- **Generating Script:** `scripts/external_evaluation/analyze_geoid_nan_corrected.py`
- **Tracked Source Artifacts:** `results/canonical/geoid_nan_corrected_summary.json`, `results/canonical/geoid_event_breakdown_nan_corrected.csv`, `results/canonical/full_precision_54_runs_confusion_metrics_nan_corrected.csv`, `results/canonical/geoid_seed_metrics_nan_corrected.csv`

### Supplementary Study S3: SiamUnet_diff 3-Seed Replication
- **Scope:** Full 3-seed evaluation (Seeds 0, 1, 2) on SiamUnet_diff across Folds A & B (12 runs total, 108 event records).
- **Status:** **Completed**
- **Key Results:**
  - Grand Macro F1: R0 = $0.7248 \pm 0.0063$, R2 = $0.7184 \pm 0.0042$, Paired Mean $\Delta = \mathbf{-0.0064 \pm 0.0056}$ (-0.64 pp).
  - 95% Paired Event Bootstrap CI: `[-0.0240, +0.0114]` ($p = 0.7546$).
  - Tail Reliability: Bottom-3 Mean F1 drops from $0.4637$ to $0.4214$ ($\Delta = -0.0423$).
  - Event Counts (±0.010 threshold): 5 Benefit, 4 Neutral, 9 Harm.
  - Scientific Conclusion: Rejects universal transfer; confirms ChangeGuard gains are architecture-specific to dense multi-scale attention backbones (SNUNet-ECAM) and do not benefit lightweight difference decoders.
- **Governing Configs:** `configs/robust_source_training/rst8_siamunet_diff/*.json`
- **Analysis Tool:** `tools/robust_source_training/run_rst8_full_3seed_analysis.py`
- **Tracked Source Artifacts:** `results/canonical/rst8_full_3seed_summary_v1.json`, `results/canonical/rst8_canonical_event_results_full_3seed_v1.csv`, `results/canonical/rst8_paired_event_deltas_full_3seed_v1.csv`, `results/canonical/rst8_bootstrap_distribution_full_3seed_v1.json`

### Supplementary Study S4: TTA on Robust Source Models (R2 + Tent / CoTTA)
- **Scope:** Bounded supplementary evaluation applying Tent and CoTTA directly to frozen Recipe R2 source checkpoints across 10 Kuro Siwo target events and 3 seeds (6 physical streams, 60 newly evaluated event-result rows).
- **Status:** **Completed**
- **Six-Condition Macro F1:**
  - Fixed R0: $0.7453 \pm 0.0345$
  - R0 + Tent: $0.7628 \pm 0.0147$
  - R0 + CoTTA: $0.7669 \pm 0.0132$
  - Fixed R2: $\mathbf{0.8086 \pm 0.0071}$
  - R2 + Tent: $0.8012 \pm 0.0019$ ($\Delta = \mathbf{-0.0074}$, 95% CI `[-0.0165, +0.0005]`, 0 benefit / 7 neutral / 3 harm)
  - R2 + CoTTA: $0.8049 \pm 0.0022$ ($\Delta = \mathbf{-0.0038}$, 95% CI `[-0.0136, +0.0044]`, 1 benefit / 7 neutral / 2 harm)
- **Scientific Conclusion:** Neither Tent nor CoTTA provides additional performance benefit over deploying the unadapted frozen R2 checkpoint. Test-time adaptation slightly degrades performance or remains statistically indistinguishable, while incurring substantial runtime and VRAM overhead.
- **Governing Configs:** `configs/supplementary_r2_tta/r2_tta_evaluation_matrix_v1.json`, `configs/supplementary_r2_tta/r2_tta_evaluation_ledger_v1.json`
- **Runner & Analysis Tools:** `scripts/run_r2_tta_parallel_evaluation.sh`, `tools/supplementary_r2_tta/run_r2_tta_evaluation.py`, `tools/supplementary_r2_tta/analyze_r2_tta_evaluation.py`
- **Tracked Source Artifacts:** `results/canonical/r2_tta_analysis_summary_v1.json`, `results/canonical/r2_tta_analysis_report.md`, `results/canonical/r2_tta_canonical_event_results_v1.csv`, `results/canonical/r2_tta_paired_event_deltas_v1.csv`, `docs/supplementary_r2_tta/r2_tta_evaluation_closeout_v1.md`

---

## 3. Publication Figures

| Figure | Title & Description | Generating Script | Input Artifact | Output Path |
|---|---|---|---|---|
| **Fig 1** | Component staircase progression (RST4 Discovery) | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/discovery_recipe_summary.csv` | `results/figures/fig1_component_staircase.{pdf,png}` |
| **Fig 2** | Primary per-event paired F1 deltas on SNUNet-ECAM (R2 $-$ R0) | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/canonical_event_results_v1.csv` | `results/figures/fig2_primary_per_event_paired_deltas.{pdf,png}` |
| **Fig 3** | Worst and bottom-3 event comparison (R0 vs R2) | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/confirmatory_summary_v1.json` | `results/figures/fig3_worst_bottom_event_comparison.{pdf,png}` |
| **Fig 4** | Precision vs. Recall trade-off across 18 confirmation events | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/confirmatory_summary_v1.json` | `results/figures/fig4_precision_recall_tradeoff.{pdf,png}` |
| **Fig 5** | Seed stability & variance comparison across Folds A and B | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/confirmatory_summary_v1.json` | `results/figures/fig5_seed_stability.{pdf,png}` |
| **Fig 6** | Event sampling probability distribution (M0 uniform patch vs M1 uniform event) | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/rst1_source_event_inventory_v1.csv` | `results/figures/fig6_event_sampling_distribution.{pdf,png}` |
| **Fig 7** | Architecture transfer deltas on SiamUnet_diff (Seed 0) | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/stage_a_summary_v1.json` | `results/figures/fig7_architecture_transfer_deltas.{pdf,png}` |
| **Fig 8** | Historical target extension per-event deltas | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/rst9_comparative_analysis_v1.json` | `results/figures/fig8_historical_target_extension_deltas.{pdf,png}` |
| **Fig 9** | Runtime throughput vs. peak VRAM tradeoff | `tools/robust_source_training/build_rst10_figures.py` | `results/canonical/rst9_ledger_v1.json` | `results/figures/fig9_runtime_vram_tradeoff.{pdf,png}` |
