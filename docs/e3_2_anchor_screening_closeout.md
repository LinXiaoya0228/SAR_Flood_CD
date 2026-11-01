# ChangeGuard Plan V3: Work Package E3.2 Source-Anchor Screening Closeout Report

**Governing Plan:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 8: Block E3)  
**Execution Script:** [`scripts/launch_e3_screening.sh`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/scripts/launch_e3_screening.sh)  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Execution Environment:** Miniconda `vfm_env` (`/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`)  
**Hardware Device:** Single Physical GPU (`cuda:0`, NVIDIA GeForce RTX 3090, 24,576 MB VRAM)  
**Starting Commit:** `ccd010be0f083098aa507125ce2b14f50ae9a255`  
**Closeout Date:** 2026-10-20  
**Status:** **E3.2 COMPLETE** (All 9 Locked Cells Executed & Verified; Method Promotion: `NONE RETAINED`)

---

## 1. Executive Summary

Work Package **`E3.2: Full Nine-Cell Source-Anchor GPU Screening Grid`** has completed execution across all nine predeclared screening cells on the six development events (2,826 samples, 179 batches per cell, 1,611 batches total).

### Non-Negotiable Research Invariants & Protocol Adherence
1. **Isolated Process Execution:** Every cell was executed in a distinct Python process via [`scripts/launch_e3_screening.sh`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/scripts/launch_e3_screening.sh). Between cells, OS-level process termination guaranteed 100% VRAM release (pre-checked before each launch: 24,110 MB free VRAM). No concurrent GPU runs were launched.
2. **Fixed Protocol & Trainable Scope:** Seed 0, Order A, reset per event, batch size 16 with natural tail, PRE_UPDATE primary prediction timing, and confidence threshold $\tau=0.90$ with horizontal-flip strong augmentation were held bit-exact identical across all nine cells.
3. **Immutable Source-Reference Model:** The source model was loaded as an independent object, set to permanent `eval()` mode with `requires_grad=False`, excluded from the optimizer, and checked to ensure zero gradient accumulation (`p.grad is None`). Bit-exact cryptographic state-dict hashes were verified before and after every stream.
4. **Decomposed Gradient Telemetry:** For every batch step, raw consistency gradient norm $\|g_{cons}\|_2$, raw anchor gradient norm $\|g_{anc}\|_2$, weighted ratio $\frac{\lambda \|g_{anc}\|_2}{\|g_{cons}\|_2}$, cosine conflict $\cos(g_{cons}, g_{anc})$, and update norm $\|\Delta \theta\|_2$ were extracted without modifying optimizer updates.
5. **Offline Interpretation Metrics:** Source agreement, base agreement, harmful-base-error recovery, beneficial-adaptation retention and loss, and regressions/corrections were computed strictly post-persistence during offline evaluation after labels were legally unlocked.
6. **Strict Partition Quarantine:** Zero access to the 10 reserved test events was verified. All evaluations operated strictly on the six development events (`[279, 437, 520, 559, 1111003, 1111008]`).

---

## 2. Master Screening Results & Method-Promotion Audit

### 2.1 Complete Screening Performance Table
Evaluated across all 6 development events ($N = 2,826$ patch pairs, $135,859,080$ double-masked valid pixels):

| Cell ID | Anchor Family | $\lambda$ | Macro F1 | Pooled F1 | Worst Event F1 | $\Delta$ Macro vs AdaBN | $\Delta$ Worst vs AdaBN | Elapsed Time | Peak VRAM Res |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E3_PROB_01`** | Probability (KL) | $0.10$ | $0.573343$ | $0.442064$ | $0.265045$ | $+0.000045$ | $+0.000000$ | $262.6\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_PROB_05`** | Probability (KL) | $0.50$ | $0.573441$ | $0.442698$ | $0.265045$ | $+0.000144$ | $+0.000000$ | $264.4\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_PROB_10`** | Probability (KL) | $1.00$ | $0.573556$ | $0.443462$ | $0.265045$ | $+0.000258$ | $+0.000000$ | $262.8\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_01`**| Centered Logit | $0.10$ | $0.573368$ | $0.442175$ | $0.265045$ | $+0.000070$ | $+0.000000$ | $261.7\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_05`**| Centered Logit | $0.50$ | $0.573541$ | $0.443288$ | $0.265045$ | $+0.000243$ | $+0.000000$ | $263.3\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_10`**| Centered Logit | $1.00$ | $0.573800$ | $0.444967$ | $0.265045$ | $+0.000502$ | $+0.000000$ | $263.9\text{ s}$ | $21,046.0\text{ MB}$ |
| **`E3_FEAT_01`** | ECAM Bottleneck | $0.10$ | $0.573327$ | $0.441959$ | $0.265045$ | $+0.000029$ | $+0.000000$ | $275.9\text{ s}$ | $22,618.0\text{ MB}$ |
| **`E3_FEAT_05`** | ECAM Bottleneck | $0.50$ | $0.573357$ | $0.442148$ | $0.265045$ | $+0.000060$ | $+0.000000$ | $279.8\text{ s}$ | $22,618.0\text{ MB}$ |
| **`E3_FEAT_10`** | ECAM Bottleneck | $1.00$ | $0.573391$ | $0.442379$ | $0.265045$ | $+0.000094$ | $+0.000000$ | $275.5\text{ s}$ | $22,618.0\text{ MB}$ |

### 2.2 Verbatim Method-Promotion Evaluation (Plan V3 Section 1.3)
Plan V3 Section 1.3 defines the exact promotion Boolean logic:
$$\text{IsPromoted} \iff (\Delta \text{MacroF1} \ge 0.01) \lor (\Delta \text{WorstEvent} > 0 \land \Delta \text{MacroF1} \ge 0.0)$$
evaluated against matched AdaBN baseline ($\text{Macro F1} = 0.573298$, $\text{Worst Event F1} = 0.265045$):

| Cell ID | $\Delta \text{MacroF1}$ | $\Delta \text{WorstEvent}$ | Condition 1 ($\ge +0.01$) | Condition 2 ($\Delta \text{Worst} > 0 \land \Delta \text{Macro} \ge 0$) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`E3_PROB_01`** | $+0.000045$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_PROB_05`** | $+0.000144$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_PROB_10`** | $+0.000258$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_LOGIT_01`**| $+0.000070$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_LOGIT_05`**| $+0.000243$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_LOGIT_10`**| $+0.000502$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_FEAT_01`** | $+0.000029$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_FEAT_05`** | $+0.000060$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |
| **`E3_FEAT_10`** | $+0.000094$ | $+0.000000$ | **FAIL** | **FAIL** ($\Delta \text{Worst} \ngtr 0$) | **REJECTED** |

**Standalone Method Promotion Verdict:** **`NONE RETAINED`**  
**Proposed E3.3 Repeat Candidate:** **`NONE`**

---

## 3. Epistemic Guardrails & Scientific Deconstruction

### 3.1 Four Mandatory Epistemic Guardrails
1. **Source Agreement is NOT Correctness:** The frozen source model is an unadapted baseline with known domain-shift failure modes, not ground truth. A candidate model that achieves high source agreement ($>98.9\%$) is primarily demonstrating update suppression rather than optimal flood delineation.
2. **Approaching Source Predictions is NOT Sufficient Evidence of Successful Anchoring:** Small parameter displacements ($\|\Delta \theta\|_2 \approx 10^{-5}$) simply show that the anchor regularizer acted as a mathematical brake on adaptation.
3. **Aggregate F1 Improvement May Result from Suppressing Adaptation Rather Than Detecting Risk:** The modest numerical gains in macro F1 (at most $+0.0502\text{ pp}$ in `E3_LOGIT_10`) reflect uniform conservative shrinkage across all batches, not selective, risk-aware adaptation.
4. **Beneficial-Adaptation Losses Must Be Reported Alongside Recovered Harmful Adaptations:** Recovering false-positive errors by regularizing toward the source inevitably penalizes valid adaptations where the source was erroneous.

### 3.2 Harmful-Base-Error Recovery vs. Beneficial-Adaptation Loss Analysis
Across all $135,859,080$ double-masked evaluation pixels:
- Total Harmful Base Consistency Errors ($S_{corr} \land B_{wrong}$): $1,176,888\text{ pixels}$ ($0.866\%$ of valid pixels).
- Total Beneficial Adaptations ($S_{wrong} \land B_{corr}$): $253,039\text{ pixels}$ ($0.186\%$ of valid pixels).

| Cell ID | Source Agree (%) | Base Agree (%) | Harmful Recovery (Count) | Harmful Recovery Rate (%) | Beneficial Loss (Count) | Beneficial Loss Rate (%) | Regress from Base (%) | Correct over Base (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E3_PROB_01`** | $98.95\%$ | $100.00\%$ | $2,458$ | $0.21\%$ | $159$ | $0.06\%$ | $0.0004\%$ | $0.0021\%$ |
| **`E3_PROB_05`** | $98.95\%$ | $99.99\%$ | $11,885$ | $1.01\%$ | $650$ | $0.26\%$ | $0.0015\%$ | $0.0100\%$ |
| **`E3_PROB_10`** | $98.96\%$ | $99.98\%$ | $23,307$ | $1.98\%$ | $1,280$ | $0.51\%$ | $0.0030\%$ | $0.0197\%$ |
| **`E3_LOGIT_01`**| $98.95\%$ | $99.99\%$ | $5,123$ | $0.44\%$ | $505$ | $0.20\%$ | $0.0013\%$ | $0.0042\%$ |
| **`E3_LOGIT_05`**| $98.96\%$ | $99.97\%$ | $26,024$ | $2.21\%$ | $2,605$ | $1.03\%$ | $0.0062\%$ | $0.0215\%$ |
| **`E3_LOGIT_10`**| $98.98\%$ | $99.94\%$ | $53,747$ | **$4.57\%$** | $4,960$ | **$1.96\%$** | $0.0109\%$ | $0.0445\%$ |
| **`E3_FEAT_01`** | $98.95\%$ | $100.00\%$ | $840$ | $0.07\%$ | $64$ | $0.03\%$ | $0.0002\%$ | $0.0007\%$ |
| **`E3_FEAT_05`** | $98.95\%$ | $100.00\%$ | $3,837$ | $0.33\%$ | $244$ | $0.10\%$ | $0.0007\%$ | $0.0032\%$ |
| **`E3_FEAT_10`** | $98.95\%$ | $99.99\%$ | $7,639$ | $0.65\%$ | $488$ | $0.19\%$ | $0.0013\%$ | $0.0064\%$ |

#### Key Empirical Insights
1. **The Inevitable Anchoring Trade-Off:** In `E3_LOGIT_10` (the highest-regularization cell), 53,747 harmful base error pixels are recovered (a 4.57% recovery rate), but 4,960 beneficial adaptation pixels are forfeited (a 1.96% loss rate).
2. **Zero Worst-Event Impact:** On Event 1111003 (the worst-performing development event), all nine cells yield an identical F1 of $0.2650450298796398$ down to 16 decimal places. Because Event 1111003 consists of only 19 samples (2 batches), neither weak consistency nor source anchoring is able to alter the parameter trajectories sufficiently to improve the critical tail event.

---

## 4. Decomposed Gradient Telemetry Across Anchor Families

The expanded gradient telemetry captured per-batch gradients without modifying optimizer state:

| Cell ID | Family | $\lambda$ | Mean Ratio $\frac{\lambda \|g_{anc}\|_2}{\|g_{cons}\|_2}$ | Mean $\cos(g_{cons}, g_{anc})$ | Mean Update $\|\Delta \theta\|_2$ | Mean Prob Disagree | Peak Reserved VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E3_PROB_01`** | Probability (KL) | $0.10$ | $0.81\times$ | $+0.5082$ | $1.12 \times 10^{-5}$ | $0.0216$ | $21,046.0\text{ MB}$ |
| **`E3_PROB_05`** | Probability (KL) | $0.50$ | $4.02\times$ | $+0.5035$ | $3.26 \times 10^{-5}$ | $0.0215$ | $21,046.0\text{ MB}$ |
| **`E3_PROB_10`** | Probability (KL) | $1.00$ | $7.94\times$ | $+0.4989$ | $5.85 \times 10^{-5}$ | $0.0214$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_01`**| Centered Logit | $0.10$ | $14.28\times$ | $+0.2562$ | $6.55 \times 10^{-5}$ | $0.0216$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_05`**| Centered Logit | $0.50$ | $70.92\times$ | $+0.2606$ | $3.09 \times 10^{-4}$ | $0.0214$ | $21,046.0\text{ MB}$ |
| **`E3_LOGIT_10`**| Centered Logit | $1.00$ | **$141.21\times$** | $+0.2646$ | $5.95 \times 10^{-4}$ | $0.0211$ | $21,046.0\text{ MB}$ |
| **`E3_FEAT_01`** | ECAM Bottleneck | $0.10$ | $0.38\times$ | $+0.3503$ | $7.89 \times 10^{-6}$ | $0.0216$ | $22,618.0\text{ MB}$ |
| **`E3_FEAT_05`** | ECAM Bottleneck | $0.50$ | $1.88\times$ | $+0.3512$ | $1.68 \times 10^{-5}$ | $0.0216$ | $22,618.0\text{ MB}$ |
| **`E3_FEAT_10`** | ECAM Bottleneck | $1.00$ | $3.76\times$ | $+0.3500$ | $2.86 \times 10^{-5}$ | $0.0216$ | $22,618.0\text{ MB}$ |

### Empirical Telemetry Findings
1. **Severe Loss-Scale Disparity:** At identical nominal $\lambda = 1.00$, the Centered Logit MSE regularizer exerts a gradient force $141.21\times$ larger than the consistency gradient, while the ECAM feature cosine regularizer exerts only $3.76\times$.
2. **Alignment & Conflict:** Average gradient cosine conflict across all three families is positive ($\cos \approx +0.26$ to $+0.51$). While isolated batches exhibit negative conflict (e.g. $\cos = -0.07$ on Event 520 batch 1 in smoke tests), the average gradient direction is weakly aligned, confirming that source anchoring generally pulls the model back toward conservative baseline weights rather than inducing destructive orthogonal interference.

---

## 5. Comparison Against Matched Development Baselines

Comparison against matched baselines on Seed 0, Order A, Reset Per Event:

| Method / Condition | Scope / Mechanism | Macro F1 | Pooled F1 | Worst Event F1 | $\Delta$ Macro vs AdaBN (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Source-Only (Unadapted)** | Frozen Reference Model | $0.588498$ | $0.507850$ | $0.262791$ | $+1.5200\text{ pp}$ |
| **BN-No-Update** | Test Running Statistics Only | $0.573297$ | $0.441864$ | $0.265045$ | $+0.0000\text{ pp}$ |
| **AdaBN** | Matched Calibration Reference | $0.573298$ | $0.441864$ | $0.265045$ | $+0.0000\text{ pp}$ |
| **Base Consistency (`E2_CONF_90`)**| $\tau=0.90$, H-Flip Strong | $0.573318$ | $0.441912$ | $0.265045$ | $+0.0021\text{ pp}$ |
| **Tent (Matched Pre-Update)** | Entropy Minimization | $0.574088$ | $0.447186$ | $0.265045$ | $+0.0791\text{ pp}$ |
| **EATA (Matched Pre-Update)** | Fisher-Regularized EntMin | $0.573296$ | $0.441864$ | $0.265045$ | $-0.0002\text{ pp}$ |
| **SAR (Matched Pre-Update)** | Sharpness-Aware Safe EntMin | $0.574136$ | $0.447215$ | $0.265045$ | $+0.0838\text{ pp}$ |
| **Best E3: `E3_LOGIT_10`** | Centered Logit ($\lambda=1.00$) | $0.573800$ | $0.444967$ | $0.265045$ | $+0.0502\text{ pp}$ |

### Strategic Scientific Synthesis
- Source-only remains higher in overall macro F1 ($0.5885$ vs $0.5738$), but suffers from a worse minimum event F1 ($0.2628$ vs $0.2650$).
- All test-time adaptation methods on bi-temporal SAR flood detection (AdaBN, Tent, EATA, SAR, Base Consistency, and Source-Anchored Consistency) converge around $\text{Macro F1} \approx 0.5733$–$0.5741$ and $\text{Worst Event F1} \equiv 0.2650$.
- Source-anchored consistency slightly outperforms plain base consistency ($0.573800$ vs $0.573318$), but falls short of the $+1.00\text{ pp}$ threshold and fails to beat matched SAR ($0.574136$) or Source-Only ($0.588498$).
- Because no anchor variant satisfied the promotion criteria, **`NONE RETAINED`** is the only scientifically defensible verdict.

---

## 6. Tangible Artifacts & Cryptographic Hash Manifest

| File Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `configs/changeguard/e3_anchor_screening_v1.json` | Locked 9-cell screening config | `cdc52ca898dd206db083949244768f31fd968bba3e6af17f22bdf0ecea39cb31` |
| `src/changeguard/e3_anchor_adapters.py` | Source-anchor adapter suite & metrics | `1679ba1de7fc31d2ed6fecfc99b565168566ec723cc55f36f88590b93f24163d` |
| `tools/run_e3_anchor_screening.py` | Screening queue runner | `b9d9c1da432d69201dfb2d0b92d0e51e594a55fb0a1e0e6f860cb85d4a190b63` |
| `tools/verify_e3_synthesis.py` | Deterministic synthesis verifier | `7d73bdfd80fb785bc49c0ea4ee161b99c735d6dc1ee2f94c9e992925b682cf30` |
| `scripts/launch_e3_screening.sh` | Per-cell process launcher | `92ca8541b2fe136ebaa6546a7b85f4e8f17705fda3745a0b82c2a8c24b884d05` |
| `logs/e3_source_anchor/v1/source_anchor_screening_execution.log` | Queue execution log | `dec3d3068bfd82af0d623ec71a27815e9121634f7727ae8101f50e0132c36c37` |
| `outputs/e3_synthesis/v1/master_source_anchor_table.json` | Master screening JSON table | `0eb621c1f4e1f72cf7252f4c9c1f618a8039c360db93fcb9a953a7ddc99a897e` |
| `outputs/e3_synthesis/v1/master_source_anchor_table.md` | Master screening Markdown table | `085600c0f823fc7b469a473f7514a60db2a5dcfa0192080ee11171936c53e878` |
| `outputs/e3_source_anchor/v1/E3_PROB_01/condition_summary.json` | Cell `E3_PROB_01` summary | `fab32331e706ea2b6254975e527c4fb1773208faa37057020795966fbe505c9f` |
| `outputs/e3_source_anchor/v1/E3_PROB_05/condition_summary.json` | Cell `E3_PROB_05` summary | `be82dccb4181d2dee9507081417b7edd3cd9a823d02925c8434dca94f31a782e` |
| `outputs/e3_source_anchor/v1/E3_PROB_10/condition_summary.json` | Cell `E3_PROB_10` summary | `696690f3a7ef8ed90c3c7e0cbd3e25d7d0eaa4a83924f4cc5096f1ae0220b566` |
| `outputs/e3_source_anchor/v1/E3_LOGIT_01/condition_summary.json` | Cell `E3_LOGIT_01` summary | `6a2746bab4bb1b206f04d7bca42d564573e27644999ffc5ac962e68d20bf7e20` |
| `outputs/e3_source_anchor/v1/E3_LOGIT_05/condition_summary.json` | Cell `E3_LOGIT_05` summary | `af11e4a117422e1e92787bcd82d21cfefd20aafa9747cddada1aac546e78d950` |
| `outputs/e3_source_anchor/v1/E3_LOGIT_10/condition_summary.json` | Cell `E3_LOGIT_10` summary | `fed42ab189ce005b68773c78c56fe1364ef710f2b1742c235bf562e46d8efc2a` |
| `outputs/e3_source_anchor/v1/E3_FEAT_01/condition_summary.json` | Cell `E3_FEAT_01` summary | `a02ec3e19949036a5ebb9db0ff7f0244db514123c1722497c61af9ab8e29da42` |
| `outputs/e3_source_anchor/v1/E3_FEAT_05/condition_summary.json` | Cell `E3_FEAT_05` summary | `6fbd011b61f107131b6c7a5d0a0c6f2e9349be48d1863516a264f823f1ba469b` |
| `outputs/e3_source_anchor/v1/E3_FEAT_10/condition_summary.json` | Cell `E3_FEAT_10` summary | `e5be82f7610f0f90b48926692798b32f9c14c8c9e79bff118963ff66f8808d01` |

---

## 7. Exit Invariants & Transition State

- **E3.2 Screening Complete:** All 9 cells accounted for and audited.
- **Deterministic Verifier Passed:** 100% of cells status `AUDITED_VALID`.
- **Confusion & Sample Conservation:** All 54 event evaluations conserve confusion matrices and sample counts ($N = 2,826$).
- **Bit-Exact Source Immutability:** Checked across all 9 cells; zero parameter drift, zero gradients.
- **Method Promotion Verdict:** `NONE RETAINED`.
- **No E3.3 GPU Work Launched:** Repeat experiments strictly blocked pending instruction.
