# ChangeGuard Work Package RST9: Final Source-Model Training and Historical-Target Extension Closeout

**Document ID:** `docs/robust_source_training/rst9_final_source_model_closeout_v1.md`  
**Execution Date (UTC):** 2026-10-28  
**Repository Root:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Documents:**
- `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
- `docs/robust_source_training/rst9_final_source_model_protocol_v1.md`
- `configs/robust_source_training/rst9_matrix_v1.json`
- `configs/robust_source_training/rst9_ledger_v1.json`
- `configs/robust_source_training/rst9_all27_class_weights_v1.json`

**Terminal Scientific Status:** `RST9 COMPLETE — DEPLOYABLE MODELS AND POSITIVE HISTORICAL EXTENSION`

---

## 1. Executive Summary & Deliverables

Work Package RST9 represents the culmination of the ChangeGuard Robust Source-Training (RST) route. Following the formal confirmation of Recipe R2 on SNUNet-ECAM in Work Package RST7, RST9 executed two tightly bounded, non-overlapping phases:

1. **Part A (Deployable All-Source Model Training):** Trained three production-grade SNUNet-ECAM models across the entire authoritative 27-event source domain using frozen Recipe R2 (M1 event-balanced sampling + M2 training-only class-balanced loss, M3 paired SAR augmentation disabled). All three random seeds ($s \in \{0, 1, 2\}$) completed exactly 16 fixed epochs (10,736 optimizer steps) with zero non-finite values, zero failures, and bit-exact reloadability.
2. **Part B (Historical-Target Source-Only Extension):** Evaluated all three newly trained R2 deployable models alongside three matched historical R0 baseline models across the 16 historical target events (6 development events and 10 former test events, totaling 18,722 SAR patch pairs and 898,250,328 valid pixels). Evaluation was strictly frozen, non-adaptive (`model.eval()`, `torch.no_grad()`, zero optimizer steps, zero BN running statistic updates).

### Primary Findings

- **Grand Event-Macro F1:** Improved from $0.6873 \pm 0.0223$ (R0) to $\mathbf{0.7380 \pm 0.0059}$ (R2), achieving an aggregate gain of $\mathbf{+0.0507}$ ($+5.07$ percentage points). Criterion P1 is **satisfied**.
- **Bottom-3 Tail Reliability:** Improved from $0.3889 \pm 0.0614$ (R0) to $\mathbf{0.4607 \pm 0.0171}$ (R2), an aggregate gain of $\mathbf{+0.0719}$ ($+7.19$ percentage points). Criterion P2 is **satisfied**.
- **Seed Stability (Historical Extension):** All three seeds demonstrated substantial positive deltas: Seed 0 ($+0.0346$), Seed 1 ($+0.0721$), Seed 2 ($+0.0453$). Inter-seed standard deviation dropped by 74% (from $0.0223$ down to $0.0059$). Note: While all three paired seed deltas are positive on the historical target extension, this does not retroactively alter the primary source-domain RST7 confirmation outcome where P3 failed.
- **Bootstrap Statistical Significance:** 10,000-resample paired event-cluster bootstrapping yielded a mean delta of $+0.0505$ with 95% confidence interval $[\mathbf{+0.0234}, \mathbf{+0.0845}]$, strictly bounded above zero ($p < 0.001$).
- **Event-Level Distribution:** 14 out of 16 events improved under R2, with dramatic gains on extreme flood-frequency tail events (Event 562: $+0.2191$, Event 1111002: $+0.1843$, Event 279: $+0.0899$, Event 411: $+0.0599$).
- **Comparison to Historical Test-Time Adaptation:** On the 10 former test events, Recipe R2 achieved a Test Macro F1 of $\mathbf{0.8086}$ ($+0.0633$ over baseline $0.7453$). In comparison, the historical unsupervised test-time adaptation methods from E11 achieved $0.7599$ (`bn_no_update`), $0.7628$ (`tent`), and $0.7669$ (`cotta`), all with 95% bootstrap intervals crossing zero. R2 robust source training delivered nearly triple the empirical gain of online adaptation with zero target-time adaptation overhead, zero adaptation memory overhead, and zero adaptation fragility.

---

## 2. Checkpoint Provenance & Optimizer-Step Reconciliation

### 2.1 Cryptographic Artifact Ledger

| Run Identifier | Method | Seed | Checkpoint Path | Checkpoint SHA-256 | Peak VRAM | Wall Runtime | Status |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| `rst9_r2_snunet_all27_s0_v1` | R2 | 0 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1/final_checkpoint.pt` | `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7` | 4,913 MB | 11,682.7s | `COMPLETED_SCIENTIFIC_RUN` |
| `rst9_r2_snunet_all27_s1_v1` | R2 | 1 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s1_v1/final_checkpoint.pt` | `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2` | 4,913 MB | 12,247.6s | `COMPLETED_SCIENTIFIC_RUN` |
| `rst9_r2_snunet_all27_s2_v1` | R2 | 2 | `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s2_v1/final_checkpoint.pt` | `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893` | 4,913 MB | 11,564.5s | `COMPLETED_SCIENTIFIC_RUN` |
| `r0_snunet_all27_s0` (Matched) | R0 | 0 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | N/A | Week 5 Baseline | `AUDITED_COMPATIBLE` |
| `r0_snunet_all27_s1` (Matched) | R0 | 1 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` | N/A | Week 5 Baseline | `AUDITED_COMPATIBLE` |
| `r0_snunet_all27_s2` (Matched) | R0 | 2 | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` | N/A | Week 5 Baseline | `AUDITED_COMPATIBLE` |

### 2.2 Mathematical Reconciliation of Optimizer Steps

The training budget for RST9 all-source training is governed by the frozen runner implementation in `src/robust_source_training/runner.py`:

1. **Source Dataset Dimension:** Exactly 21,462 patches across the 27 source events.
2. **Microbatching:** With microbatch size 8, an epoch comprises $\lceil 21,462 / 8 \rceil = 2,683$ microbatches.
3. **Gradient Accumulation:** Gradient accumulation factor is 4. This produces $\lfloor 2,683 / 4 \rfloor = 670$ full 4-microbatch accumulation steps ($2,680$ microbatches).
4. **Epoch-Local Remainder Flush:** The remaining 3 microbatches reach the epoch boundary (`b_idx + 1 == len(epoch_loader)`). The runner triggers a partial-step accumulation flush: gradients are checked for finiteness, `optimizer.step()` is invoked, `optimizer.zero_grad()` resets the accumulator to zero, and `global_step` increments by 1.
5. **Exact Step Calculation:**
   $$\text{Steps per epoch} = 670 + 1 = 671 \quad \implies \quad \text{Total Steps (16 epochs)} = 671 \times 16 = \mathbf{10,736}$$
6. **Integrity Verification:**
   - Accumulator resets to zero at the boundary of every epoch; zero gradient leakage occurs across epochs.
   - All three seeds executed identical accumulation schedules and completed exactly 10,736 steps.
   - The earlier preliminary monitoring estimate of 10,720 steps arose from integer division without partial-step flushing ($670 \times 16 = 10,720$). The actual runner behavior is verified, correct, and uniform across all runs.

---

## 3. Historical Target Evaluation Accounting & Results

### 3.1 Strict Historical Status Declaration

> [!IMPORTANT]
> **Retrospective / Historical Status Constraint:**
> All 16 events evaluated below were observed during earlier project phases (Weeks 3–5) for preliminary analysis and baseline tuning. In accordance with strict research integrity protocols, **these 16 events must never be characterized as an unseen confirmatory test set**. This evaluation represents a pure retrospective historical extension to benchmark the deployable source models against previous project milestones.

### 3.2 Master Benchmark Comparison (R0 vs R2)

Evaluation was performed with microbatch size 32 on single GPUs in `model.eval()` mode with zero gradient updates or label access prior to prediction emission. Valid pixel masking and Class 2 positive-class semantics were preserved identically.

| Metric Level | Historical Baseline R0 (Mean ± SD) | Deployed Recipe R2 (Mean ± SD) | Paired Delta ($\Delta_{\text{R2} - \text{R0}}$) | Relative Gain | Criterion Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Grand Macro F1 (16 Events)** | $0.6873 \pm 0.0223$ | $\mathbf{0.7380 \pm 0.0059}$ | $\mathbf{+0.0507}$ | $+7.38\%$ | **P1: Satisfied** |
| **Bottom-3 Tail Mean F1** | $0.3889 \pm 0.0614$ | $\mathbf{0.4607 \pm 0.0171}$ | $\mathbf{+0.0719}$ | $+18.49\%$ | **P2: Satisfied** |
| **Worst Event F1 (1111003)** | $0.2628 \pm 0.0142$ | $0.2420 \pm 0.0626$ | $-0.0208$ | $-7.91\%$ | Trade-off |
| **Development Macro F1 (6 Events)** | $0.5906 \pm 0.0037$ | $\mathbf{0.6202 \pm 0.0085}$ | $\mathbf{+0.0296}$ | $+5.01\%$ | Sub-partition |
| **Former Test Macro F1 (10 Events)** | $0.7453 \pm 0.0345$ | $\mathbf{0.8086 \pm 0.0072}$ | $\mathbf{+0.0633}$ | $+8.50\%$ | Sub-partition |
| **Grand Macro IoU** | $0.5485 \pm 0.0252$ | $\mathbf{0.6068 \pm 0.0072}$ | $\mathbf{+0.0583}$ | $+10.63\%$ | Supporting |
| **Grand Macro Precision** | $0.7329 \pm 0.0174$ | $\mathbf{0.7679 \pm 0.0184}$ | $\mathbf{+0.0350}$ | $+4.78\%$ | Supporting |
| **Grand Macro Recall** | $0.7058 \pm 0.0270$ | $\mathbf{0.7460 \pm 0.0185}$ | $\mathbf{+0.0402}$ | $+5.70\%$ | Supporting |

---

## 4. Per-Seed Performance & Seed-Stability Audit

To prevent seed-selection bias, all three seeds are evaluated and reported without exclusions:

| Seed | Metric | Matched Baseline R0 | Deployed Recipe R2 | Paired Delta ($\Delta$) | Direction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Seed 0** | Grand Macro F1 | 0.7009 | **0.7355** | $\mathbf{+0.0346}$ | Positive |
| **Seed 0** | Dev Macro F1 (6 events) | 0.5949 | **0.6110** | $+0.0162$ | Positive |
| **Seed 0** | Test Macro F1 (10 events) | 0.7645 | **0.8102** | $+0.0457$ | Positive |
| **Seed 0** | Bottom-3 Mean F1 | 0.4354 | **0.4426** | $+0.0071$ | Positive |
| **Seed 0** | Worst Event F1 (1111003) | 0.2569 | 0.1710 | $-0.0859$ | Trade-off |
| **Seed 1** | Grand Macro F1 | 0.6616 | **0.7337** | $\mathbf{+0.0721}$ | Positive |
| **Seed 1** | Dev Macro F1 (6 events) | 0.5884 | **0.6218** | $+0.0334$ | Positive |
| **Seed 1** | Test Macro F1 (10 events) | 0.7055 | **0.8009** | $+0.0953$ | Positive |
| **Seed 1** | Bottom-3 Mean F1 | 0.3191 | **0.4629** | $+0.1438$ | Positive |
| **Seed 1** | Worst Event F1 (1111003) | 0.2790 | 0.2881 | $+0.0091$ | Positive |
| **Seed 2** | Grand Macro F1 | 0.6994 | **0.7447** | $\mathbf{+0.0453}$ | Positive |
| **Seed 2** | Dev Macro F1 (6 events) | 0.5885 | **0.6279** | $+0.0393$ | Positive |
| **Seed 2** | Test Macro F1 (10 events) | 0.7659 | **0.8149** | $+0.0489$ | Positive |
| **Seed 2** | Bottom-3 Mean F1 | 0.4120 | **0.4767** | $+0.0647$ | Positive |
| **Seed 2** | Worst Event F1 (1111003) | 0.2525 | 0.2670 | $+0.0145$ | Positive |

### Audit of Criterion P3 (Seed Stability)
In the earlier RST7 source confirmation, Criterion P3 failed because Seed 1 underperformed under cross-validation. In this full-source deployment training:
- Seed 0: $\Delta = +0.0346 > 0$
- Seed 1: $\Delta = +0.0721 > 0$
- Seed 2: $\Delta = +0.0453 > 0$
All three random seeds exhibit substantial positive improvements. Furthermore, the variance across seeds was dramatically compressed under R2: the baseline R0 variance ($\text{SD} = 0.0223$, range $0.6616$–$0.7009$) was reduced to $\text{SD} = 0.0059$ (range $0.7337$–$0.7447$).

> [!NOTE]
> **Distinction on Criterion P3:** In the primary event-disjoint source confirmation (RST7), P3 failed because Seed 1 underperformed under cross-validation. In this retrospective historical-target extension (RST9), all three paired seed deltas are positive ($\Delta > 0$) and R2 inter-seed variance was reduced. However, this positive retrospective target outcome does not retroactively alter the primary RST7 source-domain P3 verdict.

---

## 5. Granular 16-Event Comparative Performance Table

The following table presents the 3-seed averaged performance for each of the 16 historical target events, sorted by event partition and identifier:

| Event ID | Partition | Total Samples | R0 Precision | R2 Precision | R0 Recall | R2 Recall | R0 F1 | R2 F1 | $\Delta_{\text{F1}}$ | Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | Dev | 2,381 | 0.2422 | 0.3213 | 0.6917 | 0.7458 | 0.3586 | **0.4484** | $\mathbf{+0.0899}$ | Major Benefit |
| **437** | Dev | 169 | 0.6691 | 0.6695 | 0.7242 | 0.7889 | 0.6948 | **0.7233** | $+0.0285$ | Benefit |
| **520** | Dev | 92 | 0.7286 | 0.8288 | 0.7060 | 0.6774 | 0.7161 | **0.7454** | $+0.0293$ | Benefit |
| **559** | Dev | 42 | 0.8850 | 0.8742 | 0.5369 | 0.5862 | 0.6682 | **0.7007** | $+0.0325$ | Benefit |
| **1111003** | Dev | 19 | 0.1762 | 0.2189 | 0.5196 | 0.3177 | 0.2628 | 0.2420 | $-0.0208$ | Decline |
| **1111008** | Dev | 123 | 0.9080 | 0.8950 | 0.7884 | 0.8317 | 0.8432 | **0.8616** | $+0.0184$ | Benefit |
| **205** | Former Test | 130 | 0.9418 | 0.9381 | 0.5281 | 0.5891 | 0.6745 | **0.7234** | $+0.0489$ | Benefit |
| **277** | Former Test | 1,834 | 0.7925 | 0.8284 | 0.9111 | 0.9283 | 0.8476 | **0.8754** | $+0.0278$ | Benefit |
| **321** | Former Test | 432 | 0.7384 | 0.7717 | 0.7162 | 0.7295 | 0.7271 | **0.7493** | $+0.0222$ | Benefit |
| **411** | Former Test | 149 | 0.7761 | 0.8637 | 0.7919 | 0.8222 | 0.7819 | **0.8418** | $\mathbf{+0.0599}$ | Benefit |
| **445** | Former Test | 361 | 0.7175 | 0.7449 | 0.9397 | 0.9555 | 0.8137 | **0.8369** | $+0.0232$ | Benefit |
| **561** | Former Test | 69 | 0.8649 | 0.8458 | 0.8575 | 0.8684 | 0.8610 | 0.8565 | $-0.0045$ | Neutral |
| **562** | Former Test | 77 | 0.8566 | 0.9218 | 0.5380 | 0.7879 | 0.6299 | **0.8490** | $\mathbf{+0.2191}$ | Huge Benefit |
| **1111002** | Former Test | 32 | 0.8189 | 0.8841 | 0.5772 | 0.8120 | 0.6593 | **0.8436** | $\mathbf{+0.1843}$ | Huge Benefit |
| **1111007** | Former Test | 4,134 | 0.8496 | 0.8884 | 0.6711 | 0.7106 | 0.7496 | **0.7896** | $+0.0400$ | Benefit |
| **1111013** | Former Test | 8,678 | 0.7062 | 0.7428 | 0.7186 | 0.7046 | 0.7085 | **0.7209** | $+0.0124$ | Benefit |

### Summary of Benefit / Harm Counts
- **Benefited Events ($\Delta_{\text{F1}} > +0.01$):** 14 / 16 (87.5%)
- **Neutral Events ($|\Delta_{\text{F1}}| \le 0.01$):** 1 / 16 (Event 561: $-0.0045$)
- **Degraded Events ($\Delta_{\text{F1}} < -0.01$):** 1 / 16 (Event 1111003: $-0.0208$, a micro-event with only 19 patch pairs)

---

## 6. Contextual Comparison with Historical Test-Time Adaptation

In Work Package E11 (`docs/e11_final_test_closeout.md`), test-time adaptation was evaluated on the exact same 10 former test events for SNUNet-ECAM. This comparison is contextual because test-time adaptation operates with online test-batch updates and differing prediction timings, whereas Recipe R2 is a pure source-trained model:

| Adaptation Strategy | Execution Stage | Test Macro F1 (Mean ± SD) | $\Delta$ vs Frozen R0 Baseline | 95% Bootstrap CI | Zero Crossed? | Target-Time Adaptation Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical R0 Source-Only** | Pre-test Anchor | $0.7453 \pm 0.0345$ | Ref ($0.0000$) | N/A | N/A | None (Pure Source Inference) |
| **`bn_no_update` (Batch Normalization)** | Test-time inference | $0.7599 \pm 0.0158$ | $+0.0146$ | $[-0.0148, +0.0560]$ | **Yes** ($p = 0.45$) | Low (batch re-estimation) |
| **`tent` (Entropy Minimization)** | Test-time adaptation | $0.7628 \pm 0.0147$ | $+0.0175$ | $[-0.0119, +0.0586]$ | **Yes** ($p = 0.33$) | High (online backprop) |
| **`cotta` (Continual TTA)** | Test-time adaptation | $0.7669 \pm 0.0132$ | $+0.0216$ | $[-0.0077, +0.0634]$ | **Yes** ($p = 0.21$) | Very High (teacher-student + cache) |
| **ChangeGuard Recipe R2 (RST9)** | **Source-Only Deployment** | $\mathbf{0.8086 \pm 0.0072}$ | $\mathbf{+0.0633}$ | $[\mathbf{+0.0312}, \mathbf{+0.0984}]$ | **No** ($p < 0.001$) | **Zero target-time adaptation overhead** |

### Scientific Significance for the FYP
1. **Source Training Dominates Test-Time Adaptation:** R2 robust source training outperforms the best test-time adaptation method (CoTTA) by nearly $+0.042$ F1 points ($+4.2$ percentage points) on the identical 10 target flood events.
2. **True Statistical Reliability:** While all online TTA methods produced 95% bootstrap confidence intervals crossing zero, R2 delivers a strictly positive bootstrap confidence interval bounded comfortably away from zero.
3. **Operational Viability:** Test-time adaptation requires gradient updates during deployment, introduces non-deterministic state drifting, and incurs significant latency/VRAM penalties. Recipe R2 requires **zero target-time adaptation overhead**, running pure feed-forward inference with complete determinism.

---

## 7. Scope of Claims, Prohibited Interpretations, and Route Conclusion

### 7.1 Confirmed Scientific Contributions
1. **Source-Domain Robustness Established:** Event-balanced sampling (M1) combined with training-only class-balanced loss (M2) significantly mitigates flood-class imbalance and geographical domain shift on SNUNet-ECAM.
2. **Retrospective Target Performance:** When trained across the full 27 source events, R2 source models generalize with substantially higher accuracy and stability across previously observed global flood distributions than standard source models.

### 7.2 Prohibited Claims and Limitations
1. **Not a Prospective Test:** The 16 target events were observed in earlier project iterations and cannot be described as an unseen prospective test.
2. **Architecture Specificity:** Work Package RST8 demonstrated that Recipe R2 does not transfer to `SiamUnet_diff` without further architectural adaptation. R2 is frozen exclusively for `SNUNet_ECAM`.
3. **Worst-Case Trade-off:** Event 1111003 (a very sparse event with only 19 samples) showed a slight drop in F1 ($-0.0208$), illustrating that tail reweighting can occasionally depress recall on ultra-small activations where flood pixels are heavily overshadowed.

### 7.3 Authorization Status for Downstream Phases
- Work Package RST9 is formally closed and complete.
- RST9 concludes model development and formal training. RST10 analysis, figure generation and reporting are authorized. No further method tuning or unrestricted training search is authorized.

================================================================================  
FINAL TERMINAL VERDICT:  
`RST9 COMPLETE — DEPLOYABLE MODELS AND POSITIVE HISTORICAL EXTENSION`  
================================================================================
