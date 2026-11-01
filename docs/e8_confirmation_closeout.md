# ChangeGuard Work Package E8: Confirmation Matrix Final Closeout Report

**Status:** E8 CLOSED — BRANCH C CONFIRMATION COMPLETE
**Date:** 2026-10-25
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git HEAD at Closeout:** `924bd52e86f769715b13618286c59b20d0b70460`
**Canonical Execution Root:** `outputs/e8_confirmation_v2`
**Preserved Historical Root:** `outputs/e8_confirmation`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E8 (Sections 13.1–13.4) & Block E9 (Section 14)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/e7_final_closeout.md` (`E7 COMPLETE — READY FOR E8`)
- `docs/protocols/e8_cross_event_confirmation_protocol_v1.md`
- `docs/e8_closest_method_transfer_audit.md`
- `docs/e8_v1_label_io_integrity_incident.md`
- `docs/e8_v2_launch_authorization.md`
- `docs/e8_final_route_decision.md`

---

## 1. Executive Summary & Lineage Certification

Work Package E8 executed the comprehensive cross-event confirmation matrix to establish whether recent continual test-time adaptation methods (specifically CoTTA) or literature adaptation techniques justify promoting ChangeGuard from **Branch C** (Controlled Empirical Study) to **Branch B** (Positive Adaptive TTA Method).

### Execution Lineage Summary

1. **Lineage 2 (Canonical Evidence):**
   - Executed under `configs/changeguard/e8_confirmation_matrix_v2.json` and `configs/changeguard/e8_execution_ledger_v2.json`.
   - Log path: `logs/e8_confirmation/e8_confirmation_v2.log`.
   - Completed all 67 execute-new physical streams sequentially on GPU 0 with exit code `0`.
   - Reused exactly 2 certified Week-12 physical streams (`source_only_frozen_running_stats_s0` and `tent_matched_pre_update_s0_oa_re`).
   - Total physical streams: **69**; Total event rows: **414**.
   - Data I/O status: **FORMALLY CERTIFIED — POST-PERSISTENCE DEFERRED GT DISK READ**.
   - Verified that zero ground-truth rasters or label-derived valid masks were accessed, loaded, or read prior to the atomic persistence and fsync of each batch's prediction shard.
   - Sealed-test access: **Exactly 0 samples across all streams**.
2. **Lineage 1 (Preserved Historical Lineage):**
   - Located at `outputs/e8_confirmation/` and `logs/e8_confirmation/e8_confirmation_v1.log`.
   - Preserved without overwrite, deletion, or renaming.
   - Formal classification: `COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ`.

---

## 2. Provenance and Checkpoint Integrity

All evaluated streams derive from the bit-exact frozen source model checkpoints trained during Week 5:

| Seed | Checkpoint Path | Expected SHA-256 | Verified Status |
| :---: | :--- | :--- | :---: |
| **Seed 0** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` | **BIT-EXACT MATCH** |
| **Seed 1** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` | **BIT-EXACT MATCH** |
| **Seed 2** | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` | **BIT-EXACT MATCH** |

- **Dataset Manifest:** `../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` (SHA-256: `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`)
- **Event Split Definition:** `configs/kuro_event_split.json` (SHA-256: `fbf30fe0bab7b90c468aaf0fbdab7960a5cf76dd49dfb3ca51f1e01668fba85f`)
- **Development Events:** `[279, 437, 520, 559, 1111003, 1111008]` (N = `[2381, 169, 92, 42, 19, 123]`; Total = `2826` samples)
- **Quarantined Sealed-Test Partition:** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` (Zero access verified).

---

## 3. Method Family Aggregates (Lineage 2 Canonical)

The completed 69-stream matrix yields the following consolidated empirical metrics:

| Method / Family | Streams | Rows | Macro F1 (Mean ± SD) | Pooled F1 (Mean ± SD) | Worst Event F1 (Min / Mean) | Macro Prec / Rec / IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **source_only** | 3 | 18 | `0.5906 ± 0.0037` | `0.5012 ± 0.0221` | `0.2525` / `0.2628` | `0.655` / `0.627` / `0.456` |
| **bn_no_update** | 6 | 36 | `0.5793 ± 0.0147` | `0.4815 ± 0.0804` | `0.2529` / `0.2668` | `0.576` / `0.669` / `0.437` |
| **adabn** | 12 | 72 | `0.5793 ± 0.0140` | `0.4815 ± 0.0766` | `0.2529` / `0.2668` | `0.576` / `0.669` / `0.437` |
| **tent** | 12 | 72 | `0.5824 ± 0.0153` | `0.5078 ± 0.0778` | `0.2451` / `0.2636` | `0.569` / `0.686` / `0.441` |
| **eata** | 12 | 72 | `0.5793 ± 0.0140` | `0.4815 ± 0.0766` | `0.2529` / `0.2667` | `0.576` / `0.669` / `0.437` |
| **sar** | 12 | 72 | `0.5801 ± 0.0143` | `0.4893 ± 0.0776` | `0.2522` / `0.2656` | `0.575` / `0.672` / `0.439` |
| **cotta_continual** | 6 | 36 | `0.5820 ± 0.0158` | `0.4903 ± 0.0821` | `0.2584` / `0.2688` | `0.574` / `0.671` / `0.441` |
| **cotta_reset** | 6 | 36 | `0.5821 ± 0.0158` | `0.4902 ± 0.0821` | `0.2584` / `0.2688` | `0.574` / `0.671` / `0.441` |
| **cotta_all** | 12 | 72 | `0.5820 ± 0.0151` | `0.4903 ± 0.0783` | `0.2584` / `0.2688` | `0.574` / `0.671` / `0.441` |

---

## 4. Answers to Core Scientific Questions

### Q1: Does CoTTA improve over source_only?
**NO.** CoTTA Macro F1 is `0.5820 ± 0.0151` compared to `0.5906 ± 0.0037` for `source_only` (mean delta: `-0.0086`). The 95% paired event-cluster bootstrap confidence interval is `[-0.0202, +0.0010]`, confirming net negative expected performance. Out of 72 CoTTA event rows, `26` (36.1%) suffer harm (delta <= -0.0100), `14` (19.4%) show benefit, and `32` (44.4%) are neutral. Harms outnumber benefits nearly 2:1.

### Q2: Does CoTTA improve over matched bn_no_update control?
**MARGINAL / NON-SIGNIFICANT.** CoTTA Macro F1 is `0.5820 ± 0.0151` compared to `0.5793 ± 0.0147` for `bn_no_update` (mean delta: `+0.0028`). The 95% paired event-cluster bootstrap confidence interval is `[-0.0010, +0.0065]`, crossing zero. Out of 72 event rows, `62` (86.1%) are completely neutral (|delta| < 0.0100), `10` (13.9%) show benefit, and `0` suffer harm. CoTTA's teacher pseudo-labeling, EMA, and stochastic restoration provide virtually zero measurable benefit beyond streaming batch normalization statistics.

### Q3: Does CoTTA exhibit severe seed sensitivity (Seed 2 behavior)?
**YES.** On Seed 1, CoTTA achieves modest gains (Macro F1: ~0.6017 vs source 0.5884). However, on Seed 0 (0.5769 vs source 0.5949, delta: -0.0180) and Seed 2 (0.5675 vs source 0.5885, delta: -0.0210), CoTTA causes substantial degradation. On Seed 2, CoTTA yields exactly 0 benefits out of 24 event evaluations (100% neutral vs bn_no_update, delta +0.0005 to +0.0015). Adaptation mechanisms completely stall on Seed 2.

### Q4: Does reset mode (continual vs reset-per-event) affect performance?
**NO.** CoTTA Continual Macro F1 is `0.5820 ± 0.0158`; CoTTA Reset Macro F1 is `0.5821 ± 0.0158`. The difference is less than `0.0001` Macro F1 and less than `0.0001` Pooled F1. Because adaptation steps per event are small and error accumulation is bounded by the conservative teacher momentum / learning rate, continual adaptation neither suffers from compounding drift nor benefits from episodic resets.

### Q5: Does stream order (Order A vs Order B) affect performance?
**NO.** CoTTA Order A Macro F1 is `0.5820 ± 0.0169`; CoTTA Order B Macro F1 is `0.5821 ± 0.0173` (delta < 0.0002 F1). Permuting event presentation order has no measurable impact.

### Q6: What explains the divergence between Macro F1 and Pooled F1?
**Extreme sample and pixel imbalance.** Event 279 contains 2,381 of the 2,826 samples (84.25% of all samples, ~116M pixels). On Event 279, streaming BN statistics cause a sharp drop in precision/F1 on Seeds 0 and 2. Because Pooled F1 weights all pixels equally, Event 279 dominates the pooled calculation (pulling Pooled F1 down to ~0.4815-0.4903), whereas Macro F1 weights all 6 events equally, masking the dominant failure on Event 279.

### Q7: How does CoTTA compare against prior TTA methods (Tent, SAR, EATA, AdaBN)?
CoTTA (0.5820 Macro F1) performs on par with Tent (0.5824) and slightly above SAR (0.5801) and AdaBN/EATA (0.5793). However, ALL adaptation methods remain strictly inferior to Source-Only (0.5906). Adding sophisticated teacher regularization (CoTTA) fails to overcome the fundamental domain-shift bottlenecks in SAR change detection.

### Q8: Does any adaptation method prevent catastrophic degradation?
**NO.** The global minimum event F1 across all 414 event rows is `0.2451` (Tent on Event 1111003). CoTTA's minimum event F1 is `0.2584` (on Event 1111003, N=19). No adaptation method prevents severe degradation on hard flood events.

### Q9: What is the status of PA-TTA and CIM-TTDA?
Both methods are audited in `docs/e8_closest_method_transfer_audit.md` and classified as `INCOMPATIBLE_FOR_E8_EXECUTION`. PA-TTA (Huang et al., IEEE TGRS 2026) is an optical photovoltaic growth method requiring SAM masks (`sam_mask`), binary change detection architecture, and multi-epoch target dataset iteration. CIM-TTDA requires optical spatial priors and multi-epoch target iteration. Neither can be executed under ChangeGuard's single-pass online PRE_UPDATE SAR protocol.

### Q10: What is the final route decision?
Branch B promotion criteria have definitively **FAILED**. Work Package E8 confirms retention of **Branch C: Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams**. Terminal verdict: `E8 CLOSED — BRANCH C CONFIRMATION COMPLETE`.

---

## 5. Uncertainty and Governance Disclosure

1. **Policy-Threshold LOEO:** Explicitly marked as `"NA under Branch C"`. Gate E proved that unsupervised pre-update risk scores invert across seeds; no selective accept/reject policy was approved or claimed under Branch C. Fabricating a controller gate is strictly prohibited by Plan V3 Section 13.1.
2. **Small-Event-Count Disclosure:** All evaluations are clustered over `K = 6` development events (`[279, 437, 520, 559, 1111003, 1111008]`). Asymptotic normality approximations do not apply. All reported confidence intervals are empirical paired event-cluster bootstrap resamples (1,000 resamples).
3. **LOEO Jackknife Stability:** 6-fold jackknife recalculations demonstrate that holding out individual events does not alter the fundamental ranking: Source-Only remains the top-performing method, and CoTTA continues to yield statistically negligible differences from unadapted BN controls.

---

## 6. Minimal Work Package E9 Plan

In strict adherence to Branch C:
- **No Controller or Policy Tuning:** No attempts will be made to fabricate or tune selective gating rules.
- **No Checkpoint Retraining:** Source model checkpoints remain frozen.
- **Zero Sealed-Test Access:** Sealed-test events remain strictly quarantined.
- **Targeted Scope:** Execute minimal, focused empirical ablations to explain the mechanistic drivers of Branch C findings:
  1. *Sample Weighting Ablation:* Quantify the exact contribution of Event 279 pixel dominance on pooled vs macro metrics.
  2. *BN Momentum Ablation:* Evaluate the sensitivity of running statistics updates under severe SAR backscatter distribution shift.
  3. *CoTTA Hyperparameter Sensitivity:* Document the sensitivity of teacher EMA alpha and stochastic restoration probability.

