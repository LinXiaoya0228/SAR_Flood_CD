# ChangeGuard Plan V3 Work Package E2.1: Confidence Grid Screening Closeout Report

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `167b172da2241ac2b6e7795a7cfc1ace224823d4`  
**Work Package:** `E2.1` (Confidence Grid Screening)  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) Section 7.1  
**Execution Context:** Logical GPU 0 (`cuda:0`, NVIDIA GeForce RTX 3090, 24 GiB VRAM), Python `vfm_env`  
**Status:** **`COMPLETED`** (Implementer Self-Verification)

---

## 1. Executive Summary & Work Package Scope

Work Package E2.1 executed the locked, predefined four-cell confidence threshold grid for test-time weak-strong consistency adaptation under Master Execution Plan V3.

The evaluation was performed under a strict one-factor screening protocol on Checkpoint Seed 0 across all six Development events ($N = 2,826$ samples):
- **Stream Sequence:** Order A (`[279, 437, 520, 559, 1111003, 1111008]`)
- **Adaptation Protocol:** `reset_per_event`, batch size 16 (tail processed as-is)
- **Base Architecture:** SNUNet_ECAM Seed 0 (`247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`)
- **Adaptation Scope:** Trainable BatchNorm affine parameters (`weight`, `bias`) optimized with SGD ($\text{lr}=1.25\times 10^{-4}$, $\mu=0.9$, $\text{wd}=0.0$)
- **Augmentation Scheme:** Weak view = Identity, Strong view = Horizontal Flip (`torch.flip(..., dims=[-1])`)
- **Loss:** One-way KL divergence from detached weak view probabilities to realigned strong log-probabilities over selected sensory-valid pixels
- **Decision Threshold:** Uniform $\theta = 0.50$ (non-tuned across all evaluations)
- **Quarantine:** Zero access to the 10 reserved sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 2. Acceptance Scorecard (10 / 10 Criteria Satisfied)

| Criterion | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **AC-01** | **All 4 Predefined Cells Accounted For** | Exactly 4 locked cells evaluated: `E2_CONF_NONE` ($\tau=0.0$), `E2_CONF_70` ($\tau=0.70$), `E2_CONF_90` ($\tau=0.90$, primary reference), `E2_CONF_95` ($\tau=0.95$). | **PASS** |
| **AC-02** | **Explicit Reuse Equivalence Audit** | Documented in `docs/e2_1_preflight_and_reuse_audit.md`. Week 12 C1/C2 audited across 13 dimensions; marked non-reusable for primary tables due to missing extended recorder telemetry and re-executed fresh. Primary metrics replicated bit-exactly. | **PASS** |
| **AC-03** | **Full Development Coverage** | All 4 cells completed all 6 development events ($N=2,826$ patches each; 24 total event evaluations). | **PASS** |
| **AC-04** | **Conservation Invariants Verified** | Confusion conservation ($TP + FP + FN + TN == \text{valid\_pixels}$) passed for **24 / 24 evaluations (100.0%)**. Sample counts verified ($N=2,826$ per stream). | **PASS** |
| **AC-05** | **Selection & Class Consistency** | $c_0 + c_1 + c_2 == \text{selected\_pixels}$ and class distribution sums equal 1.0 across all events and stream aggregates. | **PASS** |
| **AC-06** | **Telemetry Completeness** | Step-level loss values, gradient L2 norms, BN affine displacements, batch lifecycles, runtime, and peak VRAM recorded and persisted. | **PASS** |
| **AC-07** | **Zero Sealed-Test Access** | Verified: exactly 0 samples from forbidden sealed-test events accessed. | **PASS** |
| **AC-08** | **No Post-Hoc Threshold Introduction** | Verified: only the four pre-locked grid thresholds were evaluated. Zero post-hoc thresholds introduced. | **PASS** |
| **AC-09** | **Deterministic Verification Passed** | `tools/verify_e2_1_integrity.py` executed cleanly with 0 errors across all 24 evaluations. | **PASS** |
| **AC-10** | **Clean Commit & Hashes Pinned** | All E2.1 assets committed with clean working tree and SHA-256 digests recorded. | **PASS** |

---

## 3. Master Confidence Grid Results & Benchmark Summary

All metrics are evaluated under the Plan V3 measurement contract on Development 6 events ($N=2,826$ patches, uniform non-tuned decision threshold $\theta = 0.50$):

```
---------------------------------------------------------------------------------------------------------------------------------------------------
Cell ID         | Tau    | Macro F1  | Pooled F1 | Worst Ev F1 (Min/ID) | Sel %    | Selected Class Dist (c0/c1/c2) | Batches (Upd/Skip/Tot) | Loss Mean
---------------------------------------------------------------------------------------------------------------------------------------------------
E2_CONF_NONE    | None   | 0.5733    | 0.4416    | 0.2650 (#1111003)    | 100.00 % | 94.3% / 2.6% / 3.1%            | 179 / 0 / 179          | 0.00511  
E2_CONF_70      | 0.70   | 0.5733    | 0.4418    | 0.2650 (#1111003)    | 95.75  % | 95.9% / 1.9% / 2.2%            | 179 / 0 / 179          | 0.00511  
E2_CONF_90      | 0.90   | 0.5733    | 0.4419    | 0.2650 (#1111003)    | 88.79  % | 97.2% / 1.5% / 1.3%            | 179 / 0 / 179          | 0.00508  
E2_CONF_95      | 0.95   | 0.5733    | 0.4419    | 0.2650 (#1111003)    | 83.91  % | 97.8% / 1.3% / 0.9%            | 179 / 0 / 179          | 0.00503  
---------------------------------------------------------------------------------------------------------------------------------------------------
```

### Event-Level F1 Breakdown Across Thresholds

| Cell ID | Threshold ($\tau$) | Event 279 | Event 437 | Event 520 | Event 559 | Event 1111003 | Event 1111008 | Macro F1 | Pooled F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E2_CONF_NONE`** | None (0.0) | 0.3001 | 0.6344 | 0.6996 | 0.6924 | 0.2650 | 0.8480 | **0.5733** | **0.4416** |
| **`E2_CONF_70`** | 0.70 | 0.3003 | 0.6344 | 0.6996 | 0.6924 | 0.2650 | 0.8480 | **0.5733** | **0.4418** |
| **`E2_CONF_90`** | 0.90 (Primary) | 0.3004 | 0.6344 | 0.6996 | 0.6924 | 0.2650 | 0.8480 | **0.5733** | **0.4419** |
| **`E2_CONF_95`** | 0.95 | 0.3004 | 0.6344 | 0.6996 | 0.6924 | 0.2650 | 0.8480 | **0.5733** | **0.4419** |

---

## 4. Key Scientific & Methodological Findings

1. **Exact Reproduction of Historical Baseline:**  
   The newly executed `E2_CONF_NONE` and `E2_CONF_90` cells reproduce the historical Week 12 C1 and C2 primary confusion matrices and metrics bit-exactly across all 135,859,080 sensory-valid pixels:
   - `E2_CONF_NONE`: $TP=1,334,989$, $FP=2,911,192$, $FN=464,887$, $TN=131,148,012$ (Exact match to Week 12 C1).
   - `E2_CONF_90`: $TP=1,334,864$, $FP=2,906,575$, $FN=465,012$, $TN=131,152,629$ (Exact match to Week 12 C2).

2. **Severe Asymmetric Flood Filtering:**  
   As $\tau$ increases from $0.0 \to 0.95$, the proportion of predicted Flood ($c_2$) pixels in the active loss mask drops monotonically from $3.1\% \to 2.2\% \to 1.3\% \to 0.9\%$. Because the network exhibits lower confidence near water/land interfaces, confidence thresholding predominantly discards flood boundary pixels while retaining high-confidence background land ($97.8\%$ of selected pixels at $\tau=0.95$).

3. **Invariance of Performance Metrics:**  
   Despite discarding $16.09\%$ of total sensory pixels ($100.0\% \to 83.91\%$), Macro F1 remains practically constant across all thresholds ($\Delta \text{Macro F1} \le +0.00005$), Pooled F1 varies by only $+0.00030$ ($0.44161 \to 0.44191$), and Worst-Event F1 is strictly invariant at $0.2650$ (Event 1111003).

4. **Zero Compute Savings:**  
   Fewer selected pixels do not reduce inference or training compute. Both weak and strong forward passes are required for every batch regardless of $\tau$. All 179 batches contained $\ge 1$ selected pixel and applied updates; zero batches were skipped.

5. **Degradation Relative to Non-Adapted Baselines:**  
   All confidence cells achieve Macro F1 of $\approx 0.5733$ and Pooled F1 of $\approx 0.4419$. This remains substantially lower than the frozen source checkpoint (Macro F1 = 0.5949, Pooled F1 = 0.5193) and lower than matched BN-no-update (Macro F1 = 0.5793, Pooled F1 = 0.4815). Confidence thresholding alone does not resolve test-time adaptation collapse on SAR flood data.

---

## 5. Artifact Inventory & Cryptographic Checksums

```
c71598f20ba51ea7bd05f6c6346fa72893f730dcfc9d6feb1f19e3b540bcd743  outputs/e2_confidence_grid/v1/master_confidence_grid.json
415ba89de9666b76ef5d835eb5913fe072a5d50e57487a6f597993651704b442  outputs/e2_confidence_grid/v1/master_confidence_grid.md
1cb9431e2193b2a5ef59f2a2ba783300a793c5d7a8dca963df087d3a05fe88eb  configs/changeguard/e2_1_confidence_grid_v1.json
cbf81bbcdba0ce022062ec717596ae6a524e937d1d2b78d8a7c205bb39a83d3b  tools/run_e2_1_confidence_grid.py
a7b6cf26fffe2db94d0360a768d0674c93549646b9736fe5c4eaae902c650ee4  tools/verify_e2_1_integrity.py
bf96ad6a52668b81373bc51b9e2621ce3575cce8309fe69ce5fcad4306497f1f  tests/test_e2_1_confidence_grid.py
59d09d3b25f0e3860bb4a2fa4ae499bb8e5fe7d6117aa98160a2b005fe45da9e  docs/e2_1_preflight_and_reuse_audit.md
```

---

## 6. Work Package Status & Handoff

- **Work Package `E2.1`:** **`COMPLETED`**
- **Next Work Package:** **`E2.2` (Strong Augmentation Strategy Exploration)** (`READY`, `NOT STARTED`)
- **Promotion Status:** No recipe is promoted yet. Subtask E2.2 will evaluate augmentation variants (A1: Vertical Flip, A2: Integer Translation, A3: Independent Weak Views) relative to the A0 (Horizontal Flip) reference.
