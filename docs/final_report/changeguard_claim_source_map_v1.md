# ChangeGuard Writing-Time Claim Register & Evidence Trace

**Document ID:** `docs/final_report/changeguard_claim_source_map_v1.md`  
**Date:** 2026-10-28  
**Governing Authority:** `outputs/robust_source_training/rst10/tables/claim_to_evidence_audit_table.csv` & `docs/robust_source_training/rst10_final_analysis_and_figures_v1.md`  
**Status:** `FROZEN WRITING REGISTER — RST11`

This claim register establishes the permitted scope, exact quantitative numbers, canonical source artifacts, required limitations, and strictly prohibited interpretations for all empirical claims in the Final Year Project (FYP) dissertation and the IGARSS 2027 manuscript.

---

## The 13 Primary Scientific Claims

### Claim 1: Generic TTA Inconsistency under Controlled Paired-SAR Protocol
- **Exact Wording:** Generic unsupervised test-time adaptation (AdaBN, Tent, EATA, SAR, CoTTA) was not consistently reliable under the controlled multi-temporal paired-SAR flood change detection protocol.
- **Evidence Scope:** Controlled Phase E1–E11 experiments across 6 development events and 10 sealed test events.
- **Canonical Artifacts:**
  - `outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json`
  - `outputs/week9_stronger_tta/analysis_v2/table1_5method_macro_summary.csv`
  - `outputs/robust_source_training/rst10/tables/table7_contextual_tta_comparison.csv`
- **Table / Figure Reference:** Table 7, Figure 9.
- **Permitted Interpretation:** Standard unsupervised TTA methods suffer from severe instability on satellite SAR streams due to batch normalization context mismatch, speckle statistics, and severe temporal class imbalance.
- **Required Limitation:** Finding is specific to multi-temporal paired SAR flood detection and the evaluated methods; it does not constitute a universal mathematical disproof of TTA across all computer vision tasks.
- **Prohibited Stronger Interpretation:** Claiming that "TTA can never work in remote sensing" or "all adaptation methods are universally flawed."

---

### Claim 2: Severe Source Event Sample Imbalance
- **Exact Wording:** The 27 source events exhibited severe geographic sample imbalance, spanning a 358.7× disparity between the smallest event (Event 496: 21 patches) and largest event (Event 1111009: 7,532 patches).
- **Evidence Scope:** All 27 training events in the source partition (21,462 total patches).
- **Canonical Artifacts:**
  - `configs/robust_source_training/rst1_source_event_inventory_v1.csv`
  - `outputs/robust_source_training/rst10/figures/data/fig6_event_sampling_distribution.csv`
- **Table / Figure Reference:** Figure 6.
- **Permitted Interpretation:** Uniform patch-level random sampling implicitly biases gradient updates toward massive single-region flood events, neglecting small, localized inundations.
- **Required Limitation:** The patch count distribution is an intrinsic feature of the geographical footprints in the KuroSiwo benchmark, not an artificial preprocessing artifact.
- **Prohibited Stronger Interpretation:** Claiming that dataset imbalance alone explains all model generalization failures.

---

### Claim 3: Formulation of Recipe R2
- **Exact Wording:** Recipe R2 combines event-balanced sampling (M1: uniform draw across events with replacement to ~795 patches/event/epoch) and training-only inverse-square-root class-balanced cross-entropy loss (M2), omitting paired augmentation (M3).
- **Evidence Scope:** Discovered in RST4 on Group C (9 events, seed 0) and frozen in RST5.
- **Canonical Artifacts:**
  - `configs/robust_source_training/rst4_selected_recipe_v1.json`
  - `configs/robust_source_training/rst5_frozen_recipe_r2_v1.json`
  - `outputs/robust_source_training/rst10/tables/table1_component_staircase.csv`
- **Table / Figure Reference:** Table 1, Figure 1.
- **Permitted Interpretation:** A simple target-free recipe that equalizes event influence and mitigates majority-class background suppression during training only, adding zero runtime or adaptation overhead at deployment.
- **Required Limitation:** Neither event-balanced sampling nor inverse-square-root loss is an algorithmic invention of this project; the novelty lies in their controlled synthesis and empirical validation for cross-event SAR generalization.
- **Prohibited Stronger Interpretation:** Claiming algorithmic novelty of M1 or M2, or calling M2 "flood-aware class weighting."

---

### Claim 4: Primary Confirmatory Average Generalization Gain (P1)
- **Exact Wording:** On the primary architecture (SNUNet-ECAM), Recipe R2 improved grand event-macro F1 by +0.0136 (+1.36 percentage points, from 0.7202 to 0.7338) across 18 unique held-out events in the pre-registered event-disjoint confirmation matrix (RST7).
- **Evidence Scope:** 12 accepted formal confirmatory runs (Folds A & B, seeds 0, 1, 2; 108 event evaluation records).
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst7/confirmatory_summary_v1.json`
  - `outputs/robust_source_training/rst7/canonical_event_results_v1.csv`
  - `outputs/robust_source_training/rst10/tables/table2_primary_source_confirmation.csv`
- **Table / Figure Reference:** Table 2, Figure 2, Figure 3.
- **Permitted Interpretation:** Pre-registered Criterion P1 ($\Delta \ge +0.010$) was formally satisfied, supported by gains across both Fold A (+0.0083) and Fold B (+0.0189).
- **Required Limitation:** The gain is modest (+1.36 pp) and accompanied by a paired bootstrap 95% confidence interval that crosses zero (`[-0.0045, +0.0351]`).
- **Prohibited Stronger Interpretation:** Stating that the improvement is "statistically definitive" or "massive."

---

### Claim 5: Primary Confirmatory Tail Reliability Gain (P2)
- **Exact Wording:** Recipe R2 substantially enhanced lower-tail reliability on SNUNet-ECAM, lifting the bottom-three event mean F1 by +0.0511 (+5.11 percentage points, from 0.3987 to 0.4498, a +12.82% relative gain).
- **Evidence Scope:** Bottom-3 events identified per fold in the 18-event confirmation matrix.
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst7/confirmatory_summary_v1.json`
  - `outputs/robust_source_training/rst10/tables/table2_primary_source_confirmation.csv`
- **Table / Figure Reference:** Table 2, Figure 3.
- **Permitted Interpretation:** Pre-registered Criterion P2 ($\Delta_{\text{B3}} \ge +0.020$) was satisfied, demonstrating that equalizing event influence disproportionately benefits difficult, data-sparse flood regimes.
- **Required Limitation:** Single worst-event F1 slightly declined by -0.0062 (Event 1111010: 0.2991 to 0.2929), showing that extreme low-backscatter outliers require specialized physical handling.
- **Prohibited Stronger Interpretation:** Claiming that R2 eliminates all tail risk or guarantees improvements on every individual outlier.

---

### Claim 6: Formal Satisfaction of Criteria P1 and P2
- **Exact Wording:** Under the pre-registered confirmatory decision rules (`rst5_confirmation_decision_rule_v1.json`), Criteria P1 (Average Generalization Gain) and P2 (Tail Reliability Gain) passed.
- **Evidence Scope:** Confirmatory decision record in RST7.
- **Canonical Artifacts:**
  - `configs/robust_source_training/rst5_confirmation_decision_rule_v1.json`
  - `outputs/robust_source_training/rst7/confirmatory_summary_v1.json`
- **Table / Figure Reference:** Table 2.
- **Permitted Interpretation:** The primary scientific hypothesis—that event-balanced robust source training improves average and tail cross-event generalization—is supported.
- **Required Limitation:** Decision rules were pre-registered and evaluated strictly as designed; no post-hoc metric alteration was allowed.
- **Prohibited Stronger Interpretation:** Conflating passing P1/P2 with unconditional statistical proof.

---

### Claim 7: Failure of Criterion P3 (Seed Stability)
- **Exact Wording:** Criterion P3 (Seed Stability) was not satisfied in the primary source confirmation because across-seed standard deviation increased from 0.0035 to 0.0165 in Fold B, resulting in an overall inter-seed variance increase of 75%.
- **Evidence Scope:** Multi-seed evaluations across seeds 0, 1, 2 in Folds A and B.
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst7/confirmatory_summary_v1.json`
  - `outputs/robust_source_training/rst10/tables/table2_primary_source_confirmation.csv`
- **Table / Figure Reference:** Table 2, Figure 5 (Panel A and Panel B).
- **Permitted Interpretation:** While Fold A seed SD decreased slightly ($0.0120 \to 0.0106$), Fold B exhibited seed divergence (Seed 1 reached 0.7283 while Seed 2 reached 0.6956), proving that R2 does not guarantee seed variance reduction in source training.
- **Required Limitation:** Must be transparently disclosed as a failed pre-registered criterion.
- **Prohibited Stronger Interpretation:** Claiming that P3 passed, or using RST9 retrospective target results to claim that P3 was retroactively satisfied.

---

### Claim 8: Primary Bootstrap Interval Crosses Zero
- **Exact Wording:** The 10,000-resample paired event-cluster bootstrap 95% confidence interval for the primary confirmatory gain spans `[-0.0045, +0.0351]` ($p = 0.0779$), touching slightly below zero.
- **Evidence Scope:** 18 unique held-out confirmatory events as independent resampling units.
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst7/bootstrap_distribution_v1.json`
  - `outputs/robust_source_training/rst7/confirmatory_summary_v1.json`
- **Table / Figure Reference:** Table 2, Figure 2.
- **Permitted Interpretation:** The central tendency of Recipe R2 is positive (92.2% of bootstrap resamples $> 0$), but conservative statistical reporting acknowledges that zero is not excluded at the 95% level.
- **Required Limitation:** High geographic heterogeneity across the 18 global events contributes to interval dispersion.
- **Prohibited Stronger Interpretation:** Stating the result is statistically significant at $\alpha = 0.05$.

---

### Claim 9: Non-Transferability to SiamUnet_diff (RST8)
- **Exact Wording:** When transferred to the lightweight Siamese difference architecture (`SiamUnet_diff`, 1.35M parameters) at seed 0 in RST8, Recipe R2 did not improve performance, causing a -0.0099 drop in macro F1, a -0.0448 drop in bottom-3 F1, and a -0.0905 collapse in worst-event F1.
- **Evidence Scope:** 4 formal runs covering 18 held-out source events (36 canonical rows) on SiamUnet_diff seed 0.
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst8/stage_a_summary_v1.json`
  - `outputs/robust_source_training/rst8/canonical_event_results_v1.csv`
  - `outputs/robust_source_training/rst10/tables/table3_architecture_transfer.csv`
- **Table / Figure Reference:** Table 3, Figure 7.
- **Permitted Interpretation:** Recipe R2 exhibits an explicit architectural limitation: its empirical benefits are established on `SNUNet_ECAM` but do not transfer out-of-the-box to `SiamUnet_diff`. Work package was stopped cleanly at Stage A (seed 0).
- **Required Limitation:** We did not run an extensive hyperparameter search on SiamUnet_diff, nor did we isolate the precise mechanistic cause (e.g. lack of ECAM channel attention); we report only the empirical lack of transfer under the matched protocol.
- **Prohibited Stronger Interpretation:** Claiming that Recipe R2 is a universal change detection training method, or asserting that we "proved" why SiamUnet_diff failed.

---

### Claim 10: Production of Three Deployable SNUNet-ECAM Checkpoints
- **Exact Wording:** ChangeGuard produced three fully trained, verified deployable SNUNet-ECAM source model checkpoints trained on all 27 source events for 16 fixed epochs (10,736 optimizer steps each) under Recipe R2.
- **Evidence Scope:** Seeds 0, 1, and 2 trained on 21,462 source patches.
- **Canonical Artifacts:**
  - `configs/robust_source_training/rst9_ledger_v1.json`
  - `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s{0,1,2}_v1/final_checkpoint.pt`
  - `outputs/robust_source_training/rst10/tables/table4_deployable_checkpoint_ledger.csv`
- **Table / Figure Reference:** Table 4.
- **Permitted Interpretation:** Checkpoints are bit-exact, reloadable, and archived with verified SHA-256 digests (`9611df27...`, `279dc73c...`, `b057a018...`).
- **Required Limitation:** These models are pure source-trained feedforward neural networks; they do not perform online adaptation.
- **Prohibited Stronger Interpretation:** Claiming models are immune to all possible distribution shifts.

---

### Claim 11: Retrospective Historical-Target Performance Gain
- **Exact Wording:** In retrospective evaluation across the 16 historical target events, Recipe R2 deployable models achieved a grand event-macro F1 of 0.7380 (+0.0507 over baseline 0.6873, 95% bootstrap CI `[+0.0234, +0.0845]`), with 14 of 16 events showing positive paired deltas.
- **Evidence Scope:** 96 canonical evaluation records (48 R0, 48 R2) across 16 historical events over 3 seeds.
- **Canonical Artifacts:**
  - `outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json`
  - `outputs/robust_source_training/rst9_inference/all_historical_target_metrics.csv`
  - `outputs/robust_source_training/rst10/tables/table5_historical_target_extension.csv`
- **Table / Figure Reference:** Table 5, Figure 8.
- **Permitted Interpretation:** Strong empirical support that robust source training improves generalization on challenging global flood scenes without requiring online target-time updates.
- **Required Limitation:** Must be strictly designated as a retrospective historical extension, as these events were observed in earlier project phases (Weeks 3–5).
- **Prohibited Stronger Interpretation:** Claiming this is a prospective test on unseen data.

---

### Claim 12: Historical Target Events Are Not Unseen Prospective Tests
- **Exact Wording:** All 16 target events evaluated in RST9 were observed and analyzed during earlier project phases; they constitute a retrospective historical benchmark and must not be described as an unseen prospective test set.
- **Evidence Scope:** Project governance audit across Weeks 1–11 and RST0.
- **Canonical Artifacts:**
  - `configs/robust_source_training/rst0_historical_protection_v1.json`
  - `docs/robust_source_training/rst10_final_analysis_and_figures_v1.md`
- **Table / Figure Reference:** Table 5, Claim Audit Table (Claim C-8).
- **Permitted Interpretation:** Upholds strict research integrity standards by clearly distinguishing retrospective benchmarking from double-blind confirmation.
- **Required Limitation:** Absolute barrier; zero exceptions permitted in any manuscript or dissertation chapter.
- **Prohibited Stronger Interpretation:** Describing target events as "blind test," "unseen validation," or "prospective evaluation."

---

### Claim 13: Contextual Nature of Historical TTA Comparisons
- **Exact Wording:** The comparison between Recipe R2 and historical test-time adaptation methods on the 10 former test events (R2: 0.8086 vs CoTTA: 0.7669, Tent: 0.7628, AdaBN: 0.7599, Baseline: 0.7453) is contextual because operational protocols differ fundamentally.
- **Evidence Scope:** Contextual comparison across Work Packages E11, Week 9, and RST9.
- **Canonical Artifacts:**
  - `outputs/e11_final_test_v1/analysis/e11_consolidated_summary.json`
  - `outputs/robust_source_training/rst10/tables/table7_contextual_tta_comparison.csv`
- **Table / Figure Reference:** Table 7, Figure 9.
- **Permitted Interpretation:** Illustrates that target-free robust source training can achieve superior empirical performance to complex online adaptation methods while eliminating target-time adaptation overhead (0.0 ms latency, 0.0 MB adaptation memory, zero state drift).
- **Required Limitation:** TTA methods perform online updates during inference, whereas R2 is a pure source model; the comparison is contextual, not an identical execution protocol.
- **Prohibited Stronger Interpretation:** Presenting the comparison as an unconstrained single-leaderboard competition.
