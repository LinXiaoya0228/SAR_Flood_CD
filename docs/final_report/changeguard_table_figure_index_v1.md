# ChangeGuard Table and Figure Provenance Index

**Document ID:** `docs/final_report/changeguard_table_figure_index_v1.md`  
**Date:** 2026-10-28  
**Governing Authority:** `outputs/robust_source_training/rst10/metadata/rst10_execution_metadata_v1.json`  
**Status:** `FROZEN INDEX — RST11`

This document provides the authoritative inventory, cryptographic file paths, backing data paths, captions, evidence scopes, and applicable scientific limitations for all tables and figures incorporated into the Final Year Project report and the IGARSS 2027 manuscript.

---

## Part 1: Canonical Synthesis Tables

### Table 1: Component Discovery Staircase (RST4)
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table1_component_staircase.csv`
- **Source Data Path:** `outputs/robust_source_training/rst4/analysis_v1/discovery_recipe_summary.csv`
- **Identifier:** Table 1
- **Caption:** Systematic component discovery staircase evaluating the additive contributions of event-balanced sampling (M1), training-only class-balanced loss (M2), and synchronized paired SAR augmentation (M3) on discovery Group C (9 events, seed 0).
- **Evidence Scope:** Exploratory discovery fold (Group C, 9 events, 6,495 patches, single seed 0).
- **Applicable Limitation:** Serves as a pre-registered screening filter; quantitative rankings in discovery motivated parsimonious selection of R2, which was subsequently tested on independent confirmatory folds.

### Table 2: Primary Event-Disjoint Confirmation (RST7)
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table2_primary_source_confirmation.csv`
- **Source Data Path:** `outputs/robust_source_training/rst7/confirmatory_summary_v1.json` & `canonical_event_results_v1.csv`
- **Identifier:** Table 2
- **Caption:** Primary event-disjoint confirmatory evaluation comparing frozen baseline Recipe R0 and robust Recipe R2 across 18 unique held-out events in Folds A and B across random seeds 0, 1, 2 on SNUNet-ECAM.
- **Evidence Scope:** Confirmatory primary evidence (12 formal runs, 108 event evaluations across 18 held-out source events).
- **Applicable Limitation:** Paired event-cluster bootstrap 95% confidence interval spans `[-0.0045, +0.0351]`, touching zero at $\alpha=0.05$. Criterion P3 failed due to increased inter-seed variance in Fold B ($0.0035 \to 0.0165$).

### Table 3: Cross-Backbone Architecture Transfer (RST8)
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table3_architecture_transfer.csv`
- **Source Data Path:** `outputs/robust_source_training/rst8/stage_a_summary_v1.json` & `canonical_event_results_v1.csv`
- **Identifier:** Table 3
- **Caption:** Cross-backbone architecture transfer evaluation testing Recipe R2 on the lightweight Siamese difference backbone (`SiamUnet_diff`, 1.35M parameters) across the identical 18 held-out confirmatory events at seed 0.
- **Evidence Scope:** Secondary boundary evaluation (4 formal runs, 36 event evaluations, seed 0).
- **Applicable Limitation:** Transfer was not supported ($\Delta_{\text{macro}} = -0.0099$, $\Delta_{\text{B3}} = -0.0448$). Development was stopped at Stage A per protocol. Findings establish an architectural scope boundary for Recipe R2 on SNUNet-ECAM.

### Table 4: Final Deployable Checkpoint Ledger (RST9)
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table4_deployable_checkpoint_ledger.csv`
- **Source Data Path:** `configs/robust_source_training/rst9_ledger_v1.json`
- **Identifier:** Table 4
- **Caption:** Execution tracking, step budget reconciliation, checkpoint SHA-256 digests, and hardware resource metrics for the three deployable SNUNet-ECAM models trained on all 27 source events.
- **Evidence Scope:** Production source model training (27 source events, 21,462 patches, 16 fixed epochs, 10,736 optimizer steps each, seeds 0, 1, 2).
- **Applicable Limitation:** Checkpoints represent pure source-trained models; they do not perform online adaptation during inference.

### Table 5: Retrospective Historical-Target Extension (RST9)
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table5_historical_target_extension.csv`
- **Source Data Path:** `outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json` & `outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv`
- **Identifier:** Table 5
- **Caption:** Retrospective historical target extension comparing matched baseline R0 and deployable R2 models across 16 previously observed historical flood events (6 development, 10 former test) over seeds 0, 1, 2.
- **Evidence Scope:** Retrospective historical extension (96 event evaluations, 16 historical events).
- **Applicable Limitation:** All 16 events were analyzed during earlier project phases (Weeks 3–5). This evaluation is strictly retrospective and must not be described as an unseen prospective test.

### Table 6: Resource Cost and Runtime Footprint
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table6_runtime_and_resource_cost.csv`
- **Source Data Path:** `configs/robust_source_training/rst9_ledger_v1.json` & E11 execution logs
- **Identifier:** Table 6
- **Caption:** Computational complexity, training budgets, inference throughput, and target-time adaptation overhead across source-trained models and historical test-time adaptation strategies.
- **Evidence Scope:** Benchmark profiling on NVIDIA RTX 3090 GPUs.
- **Applicable Limitation:** Inference throughput measured at batch size 8; target-time overhead for R2 is exactly 0.0s because no test-time gradient or statistics updates are required.

### Table 7: Contextual Comparison with Test-Time Adaptation
- **Artifact Path:** `outputs/robust_source_training/rst10/tables/table7_contextual_tta_comparison.csv`
- **Source Data Path:** `outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json` & `outputs/robust_source_training/rst10/tables/table5_historical_target_extension.csv`
- **Identifier:** Table 7
- **Caption:** Contextual comparison on the 10 former test events between pure source-trained Recipe R2 and historical unsupervised test-time adaptation methods (AdaBN, Tent, CoTTA).
- **Evidence Scope:** Contextual historical comparison across differing execution paradigms.
- **Applicable Limitation:** Online TTA performs test-time gradient updates, whereas R2 is a pure source model; the comparison illustrates operational differences rather than an identical execution leaderboard.

---

## Part 2: Canonical Publication Figures

### Figure 1: Component Discovery Staircase
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig1_component_staircase.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig1_component_staircase.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig1_component_staircase.csv`
- **Identifier:** Figure 1
- **Caption:** Progression of event-macro F1 and bottom-3 tail F1 across discovery recipes R0, R1, R2, and R3 on Group C.
- **Limitation:** Discovery fold screening on seed 0 only.

### Figure 2: Primary Confirmatory Event-Level Paired Deltas
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig2_primary_per_event_paired_deltas.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig2_primary_per_event_paired_deltas.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig2_primary_per_event_paired_deltas.csv`
- **Identifier:** Figure 2
- **Caption:** Per-event paired differences ($\Delta = \text{F1}_{\text{R2}} - \text{F1}_{\text{R0}}$) across all 18 held-out source events in Folds A and B, displaying individual seed runs and mean deltas.
- **Limitation:** Displays heterogeneous response across global events; 9 events gained, 9 declined or remained neutral.

### Figure 3: Worst and Bottom-Event Comparison
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig3_worst_bottom_event_comparison.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig3_worst_bottom_event_comparison.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig3_worst_bottom_event_comparison.csv`
- **Identifier:** Figure 3
- **Caption:** Tail reliability vs. average generalization in RST7 primary confirmation, illustrating that Bottom-3 Mean F1 improved by +0.0511 while Worst Event F1 slightly dropped by -0.0062.
- **Limitation:** Emphasizes that tail improvements are concentrated in the bottom quartile rather than guaranteeing single worst-case protection.

### Figure 4: Precision-Recall Trade-off Dynamics
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig4_precision_recall_tradeoff.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig4_precision_recall_tradeoff.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig4_precision_recall_tradeoff.csv`
- **Identifier:** Figure 4
- **Caption:** Precision vs. recall shifts comparing source confirmation (RST7, recall-dominated trade-off) and retrospective target extension (RST9, simultaneous precision and recall gains).
- **Limitation:** Demonstrates that class reweighting gradients trade off false alarm rates on dense flood events for sensitivity on sparse tail events.

### Figure 5: Multi-Seed Stability Across Folds and Partitions
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig5_seed_stability.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig5_seed_stability.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig5_seed_stability.csv`
- **Identifier:** Figure 5
- **Caption:** Multi-seed stability across random seeds 0, 1, 2: Panel A (Fold A, SD $0.0120 \to 0.0106$), Panel B (Fold B, SD $0.0035 \to 0.0165$, P3 failed), and Panel C (Retrospective Target, SD $0.0223 \to 0.0059$).
- **Limitation:** Visualizes the explicit empirical failure of Criterion P3 in the primary source confirmation.

### Figure 6: Source Event Geographic Imbalance and Equalization
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig6_event_sampling_distribution.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig6_event_sampling_distribution.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig6_event_sampling_distribution.csv`
- **Identifier:** Figure 6
- **Caption:** Distribution of raw patch counts across the 27 source events (21 to 7,532 patches) and their equalization to ~795 draws/epoch under M1 event-balanced sampling.
- **Limitation:** Demonstrates sampling rebalancing without modifying the underlying raw sensor data.

### Figure 7: Architecture-Dependent Negative Transfer
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig7_architecture_transfer_deltas.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig7_architecture_transfer_deltas.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig7_architecture_transfer_deltas.csv`
- **Identifier:** Figure 7
- **Caption:** Comparison of Recipe R2 paired deltas on SNUNet-ECAM (positive confirmation) versus SiamUnet_diff (negative transfer across average, tail, and worst-case metrics).
- **Limitation:** Establishes the clear architectural limitation of Recipe R2; does not prove the internal mechanistic cause.

### Figure 8: Retrospective Historical Target Extension Deltas
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig8_historical_target_extension_deltas.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig8_historical_target_extension_deltas.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig8_historical_target_extension_deltas.csv`
- **Identifier:** Figure 8
- **Caption:** Retrospective paired differences across all 16 historical target events, categorized by development (6) and former test (10) events.
- **Limitation:** Strictly retrospective extension on previously observed historical benchmarks.

### Figure 9: Operational Computational and Memory Footprint
- **Artifact Path (PDF):** `outputs/robust_source_training/rst10/figures/fig9_runtime_vram_tradeoff.pdf`
- **Artifact Path (PNG):** `outputs/robust_source_training/rst10/figures/fig9_runtime_vram_tradeoff.png`
- **Backing Data:** `outputs/robust_source_training/rst10/figures/data/fig9_runtime_vram_tradeoff.csv`
- **Identifier:** Figure 9
- **Caption:** Throughput (samples/s) versus peak VRAM (GB) comparing pure source inference with online test-time adaptation methods.
- **Limitation:** Demonstrates zero target-time adaptation overhead for Recipe R2 relative to test-time backpropagation methods.
