# ChangeGuard Plan V3: Work Package E1.6 Closeout Report
## Multi-Seed Baseline Synthesis, Normalization Assessment & Block E1 Terminal Closure

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Documents:**  
- [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 1.1, Section 3.5, Section 6.1–6.6)  
- [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  
- [`docs/MATCHED_REUSE_LEDGER.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MATCHED_REUSE_LEDGER.md)  
- [`docs/e1_4_matched_baselines_closeout.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e1_4_matched_baselines_closeout.md)  
- [`docs/e1_5_bn_normalization_closeout.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e1_5_bn_normalization_closeout.md)  
**Execution Environment:** Python `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`)  
**Audit Tool:** [`tools/verify_e1_synthesis.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/verify_e1_synthesis.py)  
**Verification Provenance:** Implementer Self-Verification (offline, CPU-only statistical synthesis; zero GPU compute, zero raster re-reads, zero model forward passes)  
**Integrity Guarantee:** 354 / 354 event evaluations mathematically conserved ($TP + FP + FN + TN == \text{valid\_pixels}$). Zero access to sealed test partition.

---

## 1. Executive Summary & Purpose

Work Package **E1.6** completes the final synthesis and block-level closeout for **Block E1** of the ChangeGuard Master Execution Plan V3. It consolidates all experimental evidence generated across:
- **E1.1–E1.3:** The extended three-tier experiment recorder, fail-closed label persistence guard, and real-data execution pipeline.
- **E1.4:** The 51-stream matched baseline coverage matrix (Source-Only, AdaBN, Tent, SAR, EATA across Seeds 0, 1, 2, Orders A, B, and reset modes `reset_per_event` and `continual`).
- **E1.5:** The 8-stream matched BatchNorm normalization confirmation and bounded batch-size sensitivity grid ($bs \in \{1, 4, 16\}$).

### Governing Scientific Mandates:
1. **Multi-Seed Grounding:** No substantive conclusion in Plan V3 rests solely on a single screening seed (Seed 0 / Order A). All baseline comparisons are synthesized across three checkpoint seeds and two chronological event sequences.
2. **Standard Benchmark Freeze:** Locks the authoritative, immutable baseline benchmark table against which all proposed ChangeGuard interventions (confidence filtering, test-time augmentations, source anchors, class-aware safeguards, and temporal risk gates in Blocks E2–E11) will be measured.
3. **Strict Guardrails & Prohibited Claims:** Enforces all scientific boundaries regarding causality, additivity, source calibration, and descriptive thresholds.

---

## 2. Master Unified Baseline Performance Benchmark

Table 2.1 presents the complete, synthesized method-level performance across all 59 streams ($N = 2,826$ development samples, $135,859,080$ evaluated valid pixels per stream). All values are deterministically derived from [`outputs/e1_synthesis/v1/master_baseline_table.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e1_synthesis/v1/master_baseline_table.json).

### Table 2.1: Unified Cross-Method Baseline Benchmark

| Method Condition | Streams | Macro F1 (Mean ± SD) | Pooled F1 (Mean ± SD) | Worst Event F1 (Min / Mean) | ΔMacro F1 vs Source | Harm / Neutral / Benefit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`source_only_frozen_running_stats`** | 3 | **0.5906** ± 0.0037 | **0.5012** ± 0.0221 | 0.2525 / 0.2628 | — | — |
| **`matched_bn_no_update_bs16`** | 6 | **0.5793** ± 0.0147 | **0.4815** ± 0.0804 | 0.2529 / 0.2668 | -0.0113 | 4 / 0 / 2 |
| **`adabn_matched_pre_update`** | 12 | **0.5793** ± 0.0140 | **0.4815** ± 0.0766 | 0.2529 / 0.2668 | -0.0113 | 8 / 0 / 4 |
| **`eata_v2_fisher_matched_pre_update`** | 12 | **0.5793** ± 0.0140 | **0.4815** ± 0.0766 | 0.2529 / 0.2667 | -0.0113 | 8 / 0 / 4 |
| **`sar_matched_pre_update`** | 12 | **0.5801** ± 0.0143 | **0.4893** ± 0.0776 | 0.2522 / 0.2656 | -0.0105 | 8 / 0 / 4 |
| **`tent_matched_pre_update`** | 12 | **0.5824** ± 0.0153 | **0.5078** ± 0.0778 | 0.2451 / 0.2636 | -0.0082 | 8 / 0 / 4 |
| *Sensitivity Diagnostic ($bs=4$)* | 1 | 0.5734 ± 0.0000 | 0.4373 ± 0.0000 | 0.2750 / 0.2750 | -0.0215 | 1 / 0 / 0 |
| *Sensitivity Diagnostic ($bs=1$)* | 1 | 0.5678 ± 0.0000 | 0.4074 ± 0.0000 | 0.2731 / 0.2731 | -0.0271 | 1 / 0 / 0 |

> [!NOTE] Classification Threshold Clarification:
> The **Harm / Neutral / Benefit** counts categorize stream Macro F1 relative to each run's matched source seed baseline:
> - **Harmful:** $\Delta\text{Macro F1} < -0.005$ ($-0.5\%$ F1 vs matched source seed)
> - **Neutral:** $|\Delta\text{Macro F1}| \le 0.005$
> - **Beneficial:** $\Delta\text{Macro F1} > +0.005$ ($+0.5\%$ F1 vs matched source seed)  
> *Non-claim Assertion:* This threshold ($\pm 0.005$) is strictly an **exploratory descriptive summary of variance**. It does **not** represent a binding statistical test, hypothesis rejection, or Gate M / Gate C criterion.

---

## 3. Core Scientific Syntheses

### 3.1 Normalization Shift Replicates Adaptation Collapse
A central question across Weeks 12–14 and Block E1 was whether standard test-time adaptation collapse (as seen in Tent and AdaBN) requires active gradient descent on weights, or whether it stems primarily from the normalization context.

The unified E1 benchmark provides definitive empirical clarity:
1. **Numerical Equivalence of BN No-Update and AdaBN:**  
   Across all 6 primary confirmation streams of `matched_bn_no_update_bs16` (where all model weights are held bit-exactly fixed, zero optimizer steps occur, and zero running stats are tracked), the multi-seed mean Macro F1 is **0.5793** and Pooled F1 is **0.4815**.  
   Across all 12 streams of `adabn_matched_pre_update` (which updates moving-average running statistics), the multi-seed mean Macro F1 is **0.5793** and Pooled F1 is **0.4815**.
2. **Minimal Separation from Optimization Baselines:**  
   Standard `tent_matched_pre_update` (which performs continuous gradient descent on affine $\gamma, \beta$ parameters via entropy minimization) achieves a mean Macro F1 of **0.5824** ($\Delta = +0.0031$ vs BN no-update) and Pooled F1 of **0.5078**.  
   Standard `sar_matched_pre_update` achieves **0.5801** ($\Delta = +0.0008$ vs BN no-update).  
   Standard `eata_v2_fisher` achieves **0.5793** (applying zero parameter updates across all development events due to entropy margin filtering, thus behaving identically to AdaBN).
3. **Scientific Phrasing Compliance:**  
   > **Established Conclusion:** Changing the BatchNorm inference context to current-batch statistics at unchanged weights reproduces the observed degradation direction and substantial magnitude under the declared experimental settings.

### 3.2 Monotonic Sensitivity to Test-Time Batch Size
Evaluating `matched_bn_no_update` on Seed 0 Order A across $bs \in \{1, 4, 16\}$ under identical sample sequence reveals:
- Decreasing batch size monotonically increases prediction disagreement against source:
  - $bs = 16$: Disagreement Rate = **1.69%** (2,302,309 changed pixels) | Pooled F1 = **0.4417**
  - $bs = 4$: Disagreement Rate = **1.92%** (2,609,493 changed pixels) | Pooled F1 = **0.4373**
  - $bs = 1$: Disagreement Rate = **2.75%** (3,740,398 changed pixels) | Pooled F1 = **0.4074**
- **False Flood Surge Mechanism:** At $bs=1$, single-image spatial statistics exhibit high domain variance, converting 1,728,259 dry land pixels into false flood ($t_{02}$), inflating total predicted flood pixels by $+41.7\%$ over source.
- *Benchmark Integrity:* While batch size 1 forward passes execute without runtime failure in PyTorch `BatchNorm2d`, small batch sizes severely aggravate test-time distribution shift. Plan V3 maintains $bs=16$ as the locked primary evaluation benchmark.

### 3.3 Event-Level Heterogeneity & Trade-offs
Adaptation performance is highly non-uniform across the 6 development events:
- **Event 279 (Large Rural Flood, 2,381 samples):** Source F1 ranges from 0.3339 to 0.3769. Under Seeds 0 and 2, current-batch normalization and parameter updates trigger catastrophic false positive flood expansion ($\Delta\text{F1} \approx -0.076$ to $-0.096$).
- **Events 559 and 1111003 (Small Focal Events, 42 and 19 samples):** Current-batch normalization and entropy minimization provide minor positive adjustments ($\Delta\text{F1} \approx +0.008$ to $+0.038$), but their small sample weights fail to compensate for Event 279's collapse.
- **Seed 1 Variance:** Seed 1 source model had an unusually poor baseline on Event 279 (F1 = 0.3339, 1.49M false positives). Current-batch statistics pruned false floods on Event 279 (raising F1 to 0.4583), creating the appearance of net benefit (+0.1063 pooled) despite degrading performance on clean events (Events 437, 559, 1111008). Multi-seed evaluation was critical to prevent false promotion of Seed 1 artifacts.

---

## 4. Strict Enforcement of Prohibited Claims & Scientific Guardrails

In compliance with Plan V3 Section 6.6, the following boundaries are formally established:

| Boundary Domain | Prohibited Claim | Formally Approved Scientific Position |
| :--- | :--- | :--- |
| **Causal Attribution** | Claiming mathematical causality or assigning an additive percentage (e.g. "BN causes 85% of degradation"). | **Nonlinear Metric Reality:** F1 is non-additive. Changing BN inference context at unchanged weights reproduces the observed degradation direction and substantial magnitude under the declared experimental settings. |
| **Tent Failure Mode** | Claiming Tent degrades *solely* or *primarily* because of BatchNorm. | **Coupled Dynamics:** Tent applies gradient updates to affine parameters on top of current-batch statistics. While BN shift reproduces comparable collapse, parameter drift also alters boundaries. |
| **Source Calibration** | Claiming source model probabilities are well-calibrated. | **Uncalibrated Baseline:** Source calibration has not been formally established. Source probabilities are empirical references, not calibrated truth. |
| **Threshold Status** | Treating $\pm 0.005$ delta categorization as binding Gate M/C criteria. | **Descriptive Exploratory Threshold:** $\pm 0.005$ is solely an informal descriptive summary of variance across runs. |
| **Backup Status** | Claiming local disk copies in `/usr1` are verified off-server backups. | **Local Storage Only:** Checkpoints have a verified local copy on host `gpu22`, but no off-server/cloud backup has been performed. |
| **Screening Scope** | Extrapolating general method claims from Seed 0 / Order A alone. | **Multi-Factor Requirement:** Confirmation strictly requires all 3 seeds and 2 event evaluation orders. |

---

## 5. Comprehensive Audit & Cryptographic Checksums

The integrity verifier [`tools/verify_e1_synthesis.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/verify_e1_synthesis.py) executed deterministically under Python `vfm_env` and verified:
1. **Total Streams Audited:** 59 / 59 (51 from E1.4 + 8 from E1.5).
2. **Total Event Evaluations Audited:** 354 / 354 ($59 \times 6$).
3. **Confusion Conservation:** 354 / 354 evaluations verified exact ($TP + FP + FN + TN == \text{valid\_pixels}$).
4. **Sample Count Precision:** 354 / 354 evaluations match expected sample counts bit-for-bit ($2826$ samples per stream).
5. **Numerical Recomputation:** 354 / 354 evaluations confirm precision, recall, F1, and IoU match raw confusion counts bit-for-bit.
6. **Zero Sealed-Test Access:** Verified 100% disjoint from quarantined test IDs (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

### Synthesis Artifact Checksums:
- `tools/verify_e1_synthesis.py`: `d954fbda6ba6713919e13d9692dd9093284ff0d0246a48d2fa0c5b523e198ce4`
- `outputs/e1_synthesis/v1/master_baseline_table.json`: `8ae49139201cbc32bf0cc3e0199551f4840299d347c520b217ac490612e2476d`
- `outputs/e1_synthesis/v1/master_baseline_table.md`: `6ab4b1201a614f6ce79546cefe2409ea19b6e89d4e27dc212dad4209e9ab522b`

---

## 6. Acceptance Criteria Scorecard & Terminal E1 Sign-Off

| # | Work Package E1.6 Acceptance Criterion | Status | Verification Evidence |
| :---: | :--- | :---: | :--- |
| 1 | CPU-only, offline execution with zero GPU compute or raster re-reads | **PASS** | `verify_e1_synthesis.py` executed purely on saved JSON/NPZ metadata. |
| 2 | Full coverage of all 59 streams (51 E1.4 + 8 E1.5) | **PASS** | 59 streams audited and indexed in master table. |
| 3 | Mathematical confusion conservation across all 354 event evaluations | **PASS** | 354 / 354 verified exact ($TP+FP+FN+TN == \text{valid\_pixels}$). |
| 4 | Master synthesis tables generated in JSON and Markdown | **PASS** | `outputs/e1_synthesis/v1/master_baseline_table.{json,md}` generated. |
| 5 | Strict enforcement of prohibited-claim guardrails | **PASS** | Section 4 explicitly documents all boundaries and phrasing rules. |
| 6 | SHA-256 checksums pinned for all synthesis artifacts | **PASS** | Section 5 records exact cryptographic digests. |
| 7 | Zero access to quarantined sealed-test partition | **PASS** | Fail-closed preflight and verifier assertions validated. |
| 8 | Registry and dashboard tracking documents updated | **PASS** | `EXPERIMENT_REGISTRY.md` and `CURRENT_STATUS.md` updated. |
| 9 | Clean git commit on branch `main` | **PASS** | Tracked working tree committed clean. |
| 10 | Sequential execution policy preserved (E2 not started) | **PASS** | E2 remains INACTIVE pending user instruction. |

### Terminal Block E1 Closure Declaration:
All technical, empirical, and documentation requirements for Work Packages **E1.1, E1.2, E1.3, E1.4, E1.5, and E1.6** have been completely fulfilled and self-verified. 

**Work Package `E1.6` Status:** **`COMPLETED`**  
**Block `E1` Status:** **`COMPLETED`**  
**Next Work Package:** **`E2` (Confidence and Augmentation Comparison)** (`READY`, `NOT STARTED`)
