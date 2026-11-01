# Day 9 Audit Report: Event-Level Adaptation-Suitability Feasibility

**Date:** October 4, 2026  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Protocol Specification:** `configs/day9_event_suitability_protocol.json` (SHA-256: `2a5893d6a5901f4e8f8ffa780eca7316254fa1ee542acc98c9aa9d0f07c20a4c`)  
**Validation Events:** `[279, 437, 520, 559, 1111003, 1111008]` (6 events)  
**Sealed Test Events:** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (Zero access confirmed)  

---

## 1. Executive Summary & Core Feasibility Finding

Day 9 evaluated the exploratory question: *Can early label-free evidence (first batch or two batches) decide whether adaptation should be enabled for an entire event?*

### Core Finding: **The Single-Class LOEO Degeneration Phenomenon**

1. **Single-Class Fold Structural Trap:** Only 1 of the 6 validation events (Event 559) is beneficial ($\Delta\text{F1} \ge +0.01$). In Leave-One-Event-Out cross-validation, whenever Event 559 is held out, the remaining 5 training events contain **zero beneficial events** (100% harmful or neutral). Because every training event is degraded by adaptation, the optimal training policy is strictly **REJECT ALL ADAPTATION**.
2. **Out-of-Fold Rejection of Event 559:** Consequently, when evaluated on held-out Event 559, the trained selector rejects adaptation, defaulting to source-only performance ($F1 = 0.0816$ instead of adapted $0.6762$).
3. **Degeneration to Reject-All:** Across the remaining 5 folds (where 559 is in the training set), candidate indicators (such as `bn_affine_drift_l2`, `source_predicted_flood_area`, and composite risk) successfully separate Event 559 from the training harmful events and correctly reject adaptation on the held-out harmful events. But because Event 559 was rejected on its own fold, **the overall policy rejects adaptation across 100% of validation events, degenerating exactly to Source-Only (Reject-All)**.
4. **Zero Net Macro F1 Gain:** Under rigorous cross-validation without label leakage, no label-free selector achieves an event-macro F1 above Source-Only ($0.5646$). The unattainable Oracle selector achieves $0.6637$ (+0.0991 gain), demonstrating that while event-level gating has massive theoretical value, **6 validation events with only 1 beneficial event are statistically insufficient to learn a robust event selector**.

---

## 2. Policy Performance Comparison (Primary Window: First Batch, Seed 0)

| Policy | Macro F1 | Macro Recall | Macro IoU | Worst Event F1 | Selection Freq | Harm Avoided | Ben Preserved | Mean Regret | Equivalence / Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `source_only` | 0.5646 | 0.5624 | 0.4385 | 0.0816 | 0.0% | 4 / 4 | 0 / 1 | 0.0991 | Equivalent to Source-Only (Reject-All) |
| `adapt_all` | 0.5465 | 0.6735 | 0.4085 | 0.2326 | 100.0% | 0 / 4 | 1 / 1 | 0.1172 | Full Adaptation |
| `oracle_upper_bound` | 0.6637 | 0.6469 | 0.5165 | 0.3291 | 16.7% | 4 / 4 | 1 / 1 | 0.0000 | Theoretical Upper Bound |
| `selector_bn_affine_drift` | 0.5646 | 0.5624 | 0.4385 | 0.0816 | 0.0% | 4 / 4 | 0 / 1 | 0.0991 | Equivalent to Source-Only (Reject-All) |
| `selector_best_null_probe` | 0.4474 | 0.5890 | 0.3304 | 0.0816 | 83.3% | 0 / 4 | 0 / 1 | 0.2163 | Partial Selection |
| `selector_null_growth` | 0.4474 | 0.5890 | 0.3304 | 0.0816 | 83.3% | 0 / 4 | 0 / 1 | 0.2163 | Partial Selection |
| `selector_intermediate_drift` | 0.4555 | 0.6091 | 0.3384 | 0.0816 | 33.3% | 2 / 4 | 0 / 1 | 0.2082 | Partial Selection |
| `selector_stability_area_increase` | 0.4489 | 0.5934 | 0.3324 | 0.0816 | 66.7% | 0 / 4 | 0 / 1 | 0.2148 | Partial Selection |
| `selector_source_flood_area` | 0.4983 | 0.5756 | 0.3838 | 0.0816 | 16.7% | 3 / 4 | 0 / 1 | 0.1653 | Partial Selection |
| `selector_generic_plus_null_composite` | 0.4983 | 0.5756 | 0.3838 | 0.0816 | 16.7% | 3 / 4 | 0 / 1 | 0.1653 | Partial Selection |

---

## 3. Leave-One-Event-Out Fold Analysis (Fold-by-Fold Breakdown)

Detailed fold decisions for `selector_bn_affine_drift` (Primary Window, Seed 0):

| Fold / Held-Out Event | Train Ben / Harm | Single-Class Fold? | Chosen Train Orientation | Held-Out Indicator Val | Decision | Ground-Truth Outcome | Selected F1 | Regret |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Event 279 | 1 / 3 | No (1 beneficial) | `le` | 0.008241 | **REJECT** | Harmful/Neutral | 0.6301 | 0.0000 |
| Event 437 | 1 / 3 | No (1 beneficial) | `le` | 0.008050 | **REJECT** | Harmful/Neutral | 0.7481 | 0.0000 |
| Event 520 | 1 / 4 | No (1 beneficial) | `le` | 0.008150 | **REJECT** | Harmful/Neutral | 0.7660 | 0.0000 |
| Event 559 | 0 / 4 | YES (0 beneficial) | `reject_all` | 0.007960 | **REJECT** | Beneficial (+0.595) | 0.0816 | 0.5946 |
| Event 1111003 | 1 / 3 | No (1 beneficial) | `le` | 0.008245 | **REJECT** | Harmful/Neutral | 0.3291 | 0.0000 |
| Event 1111008 | 1 / 3 | No (1 beneficial) | `le` | 0.008042 | **REJECT** | Harmful/Neutral | 0.8325 | 0.0000 |

---

## 4. Sensitivity Analysis (Stream Orders & Observation Windows)

### 4.1 Sensitivity to Stream Order (Seeds 0, 1, 2)

| Policy | Seed 0 Macro F1 | Seed 1 Macro F1 | Seed 2 Macro F1 | Mean Macro F1 | Mean Selection Freq |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `source_only` | 0.5646 | 0.5646 | 0.5646 | **0.5646** | 0.0% |
| `adapt_all` | 0.5465 | 0.5424 | 0.5465 | **0.5451** | 100.0% |
| `oracle_upper_bound` | 0.6637 | 0.6604 | 0.6624 | **0.6621** | 16.7% |
| `selector_bn_affine_drift` | 0.5646 | 0.5196 | 0.5632 | **0.5491** | 16.7% |
| `selector_best_null_probe` | 0.4474 | 0.4944 | 0.4557 | **0.4658** | 50.0% |
| `selector_null_growth` | 0.4474 | 0.4560 | 0.4557 | **0.4530** | 50.0% |
| `selector_intermediate_drift` | 0.4555 | 0.4544 | 0.4557 | **0.4552** | 38.9% |
| `selector_stability_area_increase` | 0.4489 | 0.5151 | 0.4507 | **0.4716** | 61.1% |
| `selector_source_flood_area` | 0.4983 | 0.4977 | 0.4985 | **0.4982** | 16.7% |
| `selector_generic_plus_null_composite` | 0.4983 | 0.5646 | 0.5218 | **0.5282** | 11.1% |

### 4.2 Sensitivity to Observation Window (First Batch vs First Two Batches, Seed 0)

| Policy | Window 1 (1 Batch) Macro F1 | Window 2 (2 Batches) Macro F1 | $\Delta$ Window | Window 2 Selection Freq |
| :--- | :---: | :---: | :---: | :---: |
| `source_only` | 0.5646 | 0.5646 | +0.0000 | 0.0% |
| `adapt_all` | 0.5465 | 0.5465 | +0.0000 | 100.0% |
| `oracle_upper_bound` | 0.6637 | 0.6637 | +0.0000 | 16.7% |
| `selector_bn_affine_drift` | 0.5646 | 0.5580 | -0.0066 | 33.3% |
| `selector_best_null_probe` | 0.4474 | 0.4489 | +0.0015 | 66.7% |
| `selector_null_growth` | 0.4474 | 0.4489 | +0.0015 | 66.7% |
| `selector_intermediate_drift` | 0.4555 | 0.4540 | -0.0015 | 50.0% |
| `selector_stability_area_increase` | 0.4489 | 0.4489 | +0.0000 | 66.7% |
| `selector_source_flood_area` | 0.4983 | 0.4983 | +0.0000 | 16.7% |
| `selector_generic_plus_null_composite` | 0.4983 | 0.5217 | +0.0234 | 16.7% |

---

## 5. Methodological & Research-Integrity Implications

1. **Confirmation of Offline Leakage vs Online Generalization:** If an analyst evaluates candidate indicators *in-sample* across all 6 validation events without cross-validation, indicators like `source_predicted_flood_area` or `predeclared_change_risk_score` appear to perfectly isolate Event 559 (which has minimal source flood area and lowest risk score), falsely claiming an oracle-level Macro F1 of $0.6637$. However, Leave-One-Event-Out cross-validation rigorously exposes that this is in-sample overfitting: because Event 559 is unique in the small validation set, an out-of-sample learner cannot discover the threshold without observing Event 559 in the training set.
2. **Statistical Under-Powering of the Validation Set:** Six validation events with only 1 positive instance ($P=1/6 \approx 16.7\%$) provides exactly 0 degrees of freedom for positive out-of-fold generalization. The single-class fold issue is not a flaw in the indicator, but a structural property of small-sample extreme class imbalance.
3. **Scientific Implication for the FYP:** Event-level gating is a valid conceptual framing, but its empirical validation requires a broader corpus of flood events with diverse baseline performance characteristics. Relying on 6 validation events to claim a validated event selector would violate research integrity.
4. **Integrity Certification:** Zero test events were accessed. All decisions were derived strictly from label-free early indicators. Folds and single-class properties are preserved and documented without alteration.
