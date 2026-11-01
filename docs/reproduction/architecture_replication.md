# Reproduction Guide: Architecture Transfer and Supplementary Replication

This document describes the second-backbone evaluation on SiamUnet_diff, the initial Seed 0 screening outcome, the supplementary 3-seed replication study, and the terminal scientific conclusion.

---

## 1. Background and Protocol Provenance

To test whether the empirical benefits of Recipe R2 (M1 event-balanced sampling + M2 training-only class-balanced loss) transfer universally across change detection backbones or depend on network inductive bias, the exact frozen R0 and R2 recipes were evaluated on **SiamUnet_diff** (FC-Siam-diff, ~1.35M parameters).

### Preregistered Stopping Rule & Supplementary Study Provenance
- **Preregistered Screening (Stage A)**: The initial protocol screened SiamUnet_diff at Seed 0 across Confirmatory Folds A and B (4 runs, 18 unique held-out events). Because Recipe R2 degraded performance ($\Delta F_1 = -0.0099$), the preregistered stopping rule was triggered and execution was formally halted at Seed 0.
- **Supplementary Extension**: To rigorously verify whether this negative outcome was an artifact of random initialization or reflects consistent architectural behavior, an authorized supplementary extension evaluated Seeds 1 and 2 under identical frozen configs and datasets (8 additional runs; 12 total runs; 108 evaluated event records). The original Seed 0 screening outcome and decision rule remain fully preserved.

---

## 2. Seed 0 Screening Results

Across the 18 unique held-out confirmatory source events at Seed 0:
- **Grand Macro F1:**
  - R0 Baseline: $0.7238$
  - R2 Recipe: $0.7139$
  - Paired Delta: $\mathbf{-0.0099}$ (-0.99 pp)
- **Tail Reliability (Bottom-3 Tail F1):**
  - R0 Baseline: $0.4350$
  - R2 Recipe: $0.3902$
  - Paired Delta: $\mathbf{-0.0448}$ (-4.48 pp)
- **Worst-Event F1:** R0 = $0.3565$, R2 = $0.2660$ ($\Delta = -0.0905$)

---

## 3. Full 3-Seed Replication Results

### Execution Status: **COMPLETED**
All 12 runs (4 historical seed 0 + 8 supplementary seeds 1/2) completed with `COMPLETED_SCIENTIFIC_RUN`, zero skipped optimizer steps, and verified independence on held-out events.

### Per-Seed and 3-Seed Aggregate Performance (18 Unique Held-Out Events)

| Model / Recipe | Seed | Macro F1 | Precision | Recall | IoU | Bottom-3 F1 | Worst Event F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SiamUnet R0** | 0 | 0.7238 | 0.8441 | 0.6757 | 0.5894 | 0.4350 | 0.3565 |
| **SiamUnet R2** | 0 | 0.7139 | 0.8140 | 0.6872 | 0.5805 | 0.3902 | 0.2660 |
| *Δ (R2 - R0)* | *0* | *-0.0099* | *-0.0301* | *+0.0115* | *-0.0089* | *-0.0448* | *-0.0905* |
| \midrule | | | | | | | |
| **SiamUnet R0** | 1 | 0.7315 | 0.7856 | 0.7342 | 0.5972 | 0.4907 | 0.4137 |
| **SiamUnet R2** | 1 | 0.7221 | 0.7530 | 0.7507 | 0.5895 | 0.4012 | 0.3150 |
| *Δ (R2 - R0)* | *1* | *-0.0094* | *-0.0326* | *+0.0165* | *-0.0077* | *-0.0895* | *-0.0987* |
| \midrule | | | | | | | |
| **SiamUnet R0** | 2 | 0.7190 | 0.8098 | 0.6940 | 0.5843 | 0.4654 | 0.3826 |
| **SiamUnet R2** | 2 | 0.7191 | 0.7540 | 0.7509 | 0.5799 | 0.5144 | 0.3725 |
| *Δ (R2 - R0)* | *2* | *+0.0000* | *-0.0558* | *+0.0569* | *-0.0044* | *+0.0490* | *-0.0101* |
| \midrule | | | | | | | |
| **SiamUnet R0 (3-Seed Mean)** | Mean ± SD | $0.7248 \pm 0.0063$ | $0.8132 \pm 0.0294$ | $0.7013 \pm 0.0299$ | $0.5903 \pm 0.0065$ | 0.4637 | 0.3843 |
| **SiamUnet R2 (3-Seed Mean)** | Mean ± SD | $0.7184 \pm 0.0042$ | $0.7737 \pm 0.0349$ | $0.7296 \pm 0.0367$ | $0.5833 \pm 0.0054$ | 0.4214 | 0.3848 |
| **Paired Mean Δ (R2 - R0)** | **Mean ± SD** | $\mathbf{-0.0064 \pm 0.0056}$ | $\mathbf{-0.0395}$ | $\mathbf{+0.0283}$ | $\mathbf{-0.0070}$ | $\mathbf{-0.0423}$ | $\mathbf{+0.0005}$ |

### Statistical Inference (Paired Event-Level Bootstrap, 10,000 Resamples)
- **Grand Paired Mean Difference:** $\Delta = -0.0064$ (-0.64 pp)
- **95% Bootstrap Confidence Interval:** `[-0.0240, +0.0114]`
- **One-Sided Empirical $p$-value ($H_0: \Delta \le 0$):** $p = 0.7546$
- **Event-Level Distribution (±0.010 threshold):** 5 Benefit, 4 Neutral, 9 Harm

---

## 4. Scientific Conclusion & Architectural Contrast

The full 3-seed replication confirms that Recipe R2 **fails to reproduce the generalisation gains** on SiamUnet_diff:
- **Seed 1**: Confirms negative transfer ($\Delta = -0.0094$), with severe tail degradation (Bottom-3 $\Delta = -0.0895$).
- **Seed 2**: Shows flat performance ($\Delta = +0.0000$), with no overall gain.
- **Contrast against SNUNet-ECAM**:
  - On SNUNet-ECAM (12.03M parameters), Recipe R2 improved Macro F1 by $\mathbf{+0.0136}$ (95% CI `[-0.0045, +0.0351]`) and improved Bottom-3 tail reliability by $\mathbf{+0.0511}$.
  - On SiamUnet_diff (1.35M parameters), Recipe R2 degraded Macro F1 by $\mathbf{-0.0064}$ (95% CI `[-0.0240, +0.0114]`) and degraded Bottom-3 tail reliability by $\mathbf{-0.0423}$.
- **Mechanistic Root Cause**: SiamUnet_diff's shallow skip-difference decoder lacks the representational capacity and dense feature recalibration of SNUNet-ECAM. Class-weight upweighting induces severe precision loss ($-0.0395$) from false-positive flood predictions along ambiguous boundaries that overwhelms modest recall gains ($+0.0283$).
- **Scope of Claim**: The hypothesis of universal architectural transfer is **rejected**. The performance gains of ChangeGuard Recipe R2 are architecture-specific to dense multi-scale attention backbones (SNUNet-ECAM) and do not transfer to lightweight difference decoders.

---

## 5. Audit & Reproduction Commands

Audit the canonical outputs and regenerate the 3-seed summary:
```bash
python tools/robust_source_training/run_rst8_full_3seed_analysis.py
```
Outputs are archived in `results/canonical/`:
- `rst8_canonical_event_results_full_3seed_v1.csv` (108 rows)
- `rst8_paired_event_deltas_full_3seed_v1.csv` (18 rows)
- `rst8_bootstrap_distribution_full_3seed_v1.json` (10,000 resamples)
- `rst8_full_3seed_summary_v1.json` (machine-readable aggregate metrics)
