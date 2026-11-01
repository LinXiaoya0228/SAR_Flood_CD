# ChangeGuard V3: Requirement Coverage Ledger

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md)  
**Active Work Package:** `E0` (Inventory & Planning/Measurement Gap Closure)  
**Status:** Implementer-Verified Baseline Ledger  

---

## 1. Executive Summary & Purpose

This ledger maps every research requirement, baseline comparison, component evaluation, and diagnostic question from ChangeGuard Master Plans V1 and V2 to:
1. Completed empirical evidence;
2. Active work package assignments under Plan V3 (Blocks `E0` through `E11`);
3. Explicit applicability decisions or exclusions.

Per the Plan V3 governing principles:
- **No Scientific Dilution:** No experiment or confirmation requirement is dropped merely due to upcoming tool transitions (e.g., November 8).
- **Completion vs. Success:** A completed run does not imply a successful method; negative outcomes are formally archived and analyzed.
- **Factor Boundaries:** Historical single-seed screens (e.g., Week 12 C1/C2 or Week 13 C3) do not constitute multi-factor confirmation.

---

## 2. Requirement Coverage Matrix

### 2.1 Foundation & Source Training (Weeks 0–6)

| Req ID | Historical Scope | Description & Protocol | Execution Status | Analysis Status | Matching Status | Eligible V3 Block | Missing Measurements / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **REQ-SRC-01** | Pilot / Weeks 0–3 | Official KuroSiwo checkpoint evaluation | Completed | Completed | `HISTORICAL_REF` | Historical Motivation | Input channel provenance incomplete; retained as secondary reference. |
| **REQ-SRC-02** | Weeks 1–3 | Model A (`[VV, VH]`) vs Model B (`[VV, VH, slope]`) inner-split 3-seed A/B comparison | Completed | Completed | `VERIFIED` | Foundation | 22 inner-train / 5 inner-val events. Flipped sign across seeds; Model A selected as primary architecture. |
| **REQ-SRC-03** | Week 5 | Source training recipe review & epoch lock ($E_{\text{final}} = \text{median}(8, 16, 32) = 16$) | Completed | Completed | `VERIFIED` | Foundation | Approved in `docs/week5_source_recipe_review.md`. 16 epochs, FP32, Adam lr=1e-3, constant LR. |
| **REQ-SRC-04** | Week 6 | All-source Model A (`[VV, VH]`) training on all 27 source events (21,462 patches) for Seeds 0, 1, 2 | Completed | Completed | `VERIFIED` | Foundation | All 3 seeds trained and verified (`outputs/week5_source_all27/`). Backed up in external storage. |
| **REQ-SRC-05** | Week 5/6 Scope | All-source Model B (`[VV, VH, slope]`) training on all 27 events | Not Executed | Not Executed | `EXCLUDED_BY_DESIGN` | Foundation / Robustness | Frozen protocol explicitly selected Model A as primary. Model B retained strictly as source-inner ablation. If an all-27 slope generalization claim is ever made, a training task must be registered. Currently, no such claim is made. |

---

### 2.2 Baseline Matrix & Historical TTA (Weeks 7–9)

| Req ID | Historical Scope | Description & Protocol | Execution Status | Analysis Status | Matching Status | Eligible V3 Block | Missing Measurements / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **REQ-BASE-01** | Week 7 | Source-only baseline on 6 development events (Seeds 0, 1, 2) | Completed | Completed | `REUSE_VERIFIED` | `E1` / `E8` | Seed 0 fully matched to current PRE_UPDATE protocol. Seeds 1 and 2 require ledger verification or replay in E1. |
| **REQ-BASE-02** | Week 8 | Controlled AdaBN baseline (Seeds 0, 1, 2; orders a/b; reset modes) | Completed | Completed | `REUSE_UNVERIFIED` | `E1` / `E8` | Historical Week 8 run used online batch statistics without saving spatial prediction arrays or pooled confusion counts. Requires protocol matching verification. |
| **REQ-BASE-03** | Week 8 | Controlled Tent baseline (Seeds 0, 1, 2; orders a/b; reset modes) | Completed | Completed | `REUSE_UNVERIFIED` | `E1` / `E8` | Historical Week 8 Tent used POST_UPDATE predictions. Week 12 implemented matched PRE_UPDATE Tent for Seed 0; multi-seed / multi-order PRE_UPDATE Tent required in E1/E8. |
| **REQ-BASE-04** | Week 9 | EATA baseline ($d_{\text{margin}}=0.05$, Fisher regularization, Seeds 0, 1, 2) | Completed | Completed | `REUSE_UNVERIFIED` | `E1` / `E8` | Week 9 ran on all 3 seeds with V2 Fishers, but lacked spatial prediction caching and pooled confusion counts. Requires matching verification. |
| **REQ-BASE-05** | Week 9 | SAR baseline (Entropy threshold + reliable sample selection, Seeds 0, 1, 2) | Completed | Completed | `REUSE_UNVERIFIED` | `E1` / `E8` | Same as REQ-BASE-04: batch summary diagnostics saved, but spatial prediction arrays absent. |

---

### 2.3 Adaptation Components & Diagnostics (Weeks 11–14)

| Req ID | Historical Scope | Description & Protocol | Execution Status | Analysis Status | Matching Status | Eligible V3 Block | Missing Measurements / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **REQ-COMP-01** | Week 11 | Real-data smoke infrastructure & failure-handling semantics | Completed | Completed | `VERIFIED` | `E1` | Event 520, head 50 samples, batches [16,16,16,2]. Double-masking, fail-fast verified. |
| **REQ-COMP-02** | Week 12 | Candidate BN normalization diagnostic (`matched_bn_no_update_v1`) | Completed | Completed | `VERIFIED` | `E1` | Seed 0, dev 6 events. Proves batch BN statistics without updates degrade F1 similarly to Tent/C1/C2. Repeats on seeds 1/2 required. |
| **REQ-COMP-03** | Week 12 | Confidence filtering grid (no-conf vs 0.90 vs finite sensitivity) | Partial | Completed (2 cells) | `PARTIAL` | `E2` | Only no-conf (`C1`) and 0.90 (`C2`) executed. Thresholds 0.70 and 0.95 remain to be evaluated in E2. |
| **REQ-COMP-04** | Week 12 | Augmentation comparison (Weak vs Strong views) | Partial | Completed (1 cell) | `PARTIAL` | `E2` | Identity-weak vs horizontal-flip strong (`A0`) evaluated. A1 (vertical flip), A2 (paired translation), A3 (independent views), A4 (SAR intensity perturbation) open. |
| **REQ-COMP-05** | Week 13 | Source probability anchor sensitivity ($\lambda \in \{0.10, 0.50, 1.00\}$) | Partial | Completed (1 cell) | `PARTIAL` | `E3` | Only $\lambda=0.50$ (`C3`) evaluated. $\lambda=0.10$ and $\lambda=1.00$ remain open in E3. |
| **REQ-COMP-06** | Week 13 | Source logit and feature anchors | Not Executed | Not Executed | `OPEN` | `E3` | Pre-registered in V1/V3. Bounded logit anchor and feature anchor comparisons scheduled in E3. |
| **REQ-COMP-07** | Week 14 | Offline $3 \times 3$ class error diagnosis | Completed | Completed | `VERIFIED` | `E0` / `E4` | Full $3 \times 3$ confusion decomposition completed across all 6 dev events and 6 conditions. Established that Event 279 FP surge is $97.4\%$ dry land misclassified as flood. |
| **REQ-COMP-08** | Week 14 | Four bounded class-aware intervention experiments (K1–K4) | Not Executed | Not Executed | `OPEN` | `E4` | K1 (class-balanced selection), K2 (flood-aware loss sampling), K3 (permanent-water protection), K4 (flood preservation) scheduled in E4. |

---

### 2.4 Evidence, Controller & Decision Logic (Weeks 15–17)

| Req ID | Historical Scope | Description & Protocol | Execution Status | Analysis Status | Matching Status | Eligible V3 Block | Missing Measurements / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **REQ-CTRL-01** | Week 15 | Temporal / paired null evidence ($s_{\text{null}}$ over pre2/pre2 and post/post) | Partial | Partial | `OPEN` | `E5` | Interfaces exist, but phase-correct discriminant evaluation and incremental value over generic risk remain open. |
| **REQ-CTRL-02** | Week 15 | Generic risk evidence (entropy, confidence, source KL, BN drift) | Partial | Partial | `OPEN` | `E5` | Evaluated partially in Week 9, but needs unified PRE_UPDATE evaluation against $s_{\text{null}}$ under matched calibration budgets. |
| **REQ-CTRL-03** | Week 16 | Feasibility Gate E decision (Branch A: method; Branch B: stable TTA; Branch C: empirical failure study) | Not Executed | Not Executed | `PLANNED` | `E6` | Scheduled Nov 30 – Dec 6. Governs transition to policy or empirical route. |
| **REQ-CTRL-04** | Week 16 | Closest baseline applicability analysis (CoTTA, PA-TTA, CIM-TTDA) | Not Executed | Not Executed | `PLANNED` | `E6` | Matrix of required assumptions, licenses, and adaptations for paired SAR segmentation. |
| **REQ-CTRL-05** | Week 17 | Batch-level decision policy & state rollback semantics | Partial | Partial | `OPEN` | `E7` | State rollback primitives implemented; full online accept/reject policy and real smoke scheduled in E7. |

---

### 2.5 Confirmation, Ablations, Robustness & Final Evaluation (Weeks 18–20 / E8–E11)

| Req ID | Historical Scope | Description & Protocol | Execution Status | Analysis Status | Matching Status | Eligible V3 Block | Missing Measurements / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **REQ-CONF-01** | Week 18 | Leave-One-Event-Out (LOEO) cross-event evaluation (6 folds) | Not Executed | Not Executed | `PLANNED` | `E8` | Scheduled Dec 14–20. Evaluates causal U-calibrated policy across held-out development events. |
| **REQ-CONF-02** | Week 19 | Full factor coverage (Seeds 0, 1, 2; Orders a, b; Reset modes) | Not Executed | Not Executed | `PLANNED` | `E8` | Scheduled Dec 21 – Jan 3. 12 streams per adaptive method. |
| **REQ-CONF-03** | Week 20 | Component removal ablations | Not Executed | Not Executed | `PLANNED` | `E9` | Scheduled Dec 21 – Jan 3. Decisive ablations of retained components. |
| **REQ-ROB-01** | Section 15 | Second lightweight backbone architecture | Not Executed | Not Executed | `PLANNED` | `E10` | Scheduled Jan 4–31. Training recipe selection and source training on 27 events. |
| **REQ-ROB-02** | Section 15 | Synthetic shift stress-test family | Not Executed | Not Executed | `PLANNED` | `E10` | Scheduled Jan 4–31. Physically valid SAR perturbation across clean and ordered severities. |
| **REQ-FINAL-01** | Section 16 | Gate T release & frozen final test on 10 reserved events | Not Executed | Not Executed | `QUARANTINED` | `E11` | Scheduled March 1–7, 2027. Strictly quarantined until Gate T approval. |
| **REQ-DELIV-01** | Section 16 | FYP Draft Final Report, Final Report, Demonstration, Oral, DR-NTU | In Progress | Drafts Exist | `SCHEDULED` | `E11` / Delivery | March 25 (Draft), April 9 (Final Report), April 12–16 (Demo), May 10–12 (Oral), May 19 (DR-NTU). |

---

## 3. Approved Slope Scope Resolution

In accordance with Section 5.1 of Plan V3:
1. **Source-Inner Study:** Weeks 1–3 completed a rigorous, 3-seed controlled comparison on the 22/5 inner split. Model B (`[VV, VH, slope]`) flipped performance sign relative to Model A across seeds (Seed 0: $+0.0153$, Seed 1: $-0.0011$, Seed 2: $-0.0198$).
2. **Architecture Lock:** In `docs/week5_source_recipe_review.md`, Model A (`[VV, VH]`) was formally locked as the primary source architecture.
3. **All-27 Training:** Week 6 trained only Model A across all 27 events (Seeds 0, 1, 2). All three checkpoints are verified and backed up.
4. **Registered Scope:** No all-27 checkpoint exists for Model B (`[VV, VH, slope]`). Because the approved research scope does not claim channel generalization to slope on downstream events, no retraining task is required at this stage. If future robustness blocks require a 3-channel baseline, a dedicated source training task will be registered prior to evaluation.

