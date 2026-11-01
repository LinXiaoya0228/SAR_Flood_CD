# ChangeGuard Work Package E5.2: Final Temporal-Risk Closeout Report

> [!IMPORTANT]
> **Work Package E5.2 Status: COMPLETE & VERIFIED**
> - **Execution Commit**: `81f6f3af35fbe6d4d15e76c708d8b8fbffa64f45` (`feat(e5): refine offline temporal-risk analyzer and audit wording`)
> - **Worktree State**: `CLEAN` (`git_dirty: false`)
> - **Governing Plans & Contracts**:
>   - [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md)
>   - [`docs/e5_preflight_and_contracts.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_preflight_and_contracts.md)
>   - [`docs/e5_2c_analysis_contract.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_2c_analysis_contract.md)
>   - [`docs/e5_2c_seed0_ordera_analysis.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/e5_2c_seed0_ordera_analysis.md)
> - **Formal Source Collection**: [`outputs/e5_temporal_evidence/formal_s0_oa_v1/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/formal_s0_oa_v1/) (`COMPLETED_RUN`, 179 shards verified bit-exact)
> - **Canonical Analysis Directory**: [`outputs/e5_temporal_evidence/analysis_s0_oa_v2/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v2/) (`COMPLETED_ANALYSIS`, exactly 7 canonical artifacts)
> - **Zero Sealed-Test Access**: Strictly verified quarantined zero access across all 10 test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).
> - **Terminal Verdict**: `E5.2 COMPLETE — READY TO START E5.3`

---

## 1. Executive Summary & Verification Verdict

Work Package E5.2 ("Formal Temporal-Risk Evidence Collection and Offline Analysis") evaluated adaptation risks on streaming test-time adaptation for ChangeGuard. The entire lifecycle—spanning formal GPU evidence collection (`E5.2B`), audit and analysis runner implementation (`E5.2C1`), baseline analysis execution (`E5.2C2`), audit and wording checkpointing (`E5.2C3 v1`), and clean committed headless regeneration (`E5.2C3 v2`)—has executed cleanly to completion.

### 1.1 Complete Lifecycle Summary
1. **Formal Collection (`E5.2B`)**: Run `e5_temporal_risk_s0_oa_v1` completed on physical GPU 0 with zero OOM, capturing 2,826 samples across 179 authoritative `.npz` batch shards across the six development events (`[279, 437, 520, 559, 1111003, 1111008]`).
2. **Analysis Refinement & Headless Freezing (`E5.2C3`)**: The offline temporal-risk analyzer [`tools/analyze_e5_temporal_risk.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/analyze_e5_temporal_risk.py) and documentation were reviewed, audited, and committed at HEAD `81f6f3af35fbe6d4d15e76c708d8b8fbffa64f45`.
3. **Clean Regeneration (`v2`)**: Executed deterministically on CPU in detached tmux session `cg_e5_2c_analysis_v2` without modifying any scientific inputs. All 5 output CSV files reproduced bit-for-bit identical hashes to v1, the redundant summary file was eliminated, and `analysis_manifest.json` formally certified `COMPLETED_ANALYSIS` from a clean git worktree.
4. **Regression Verification**: All 49 focused unit and contract tests (11 analysis tests + 38 E5 risk tests) passed cleanly with zero failures.

---

## 2. Canonical Artifact Inventory & Cryptographic Hashes

The authoritative analysis output directory is [`outputs/e5_temporal_evidence/analysis_s0_oa_v2/`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v2/). It contains **exactly seven canonical files** with zero temporary or duplicate files:

| File Name | File Size (Bytes) | Cryptographic SHA-256 Hash | Purpose / Contractual Mapping |
| :--- | :---: | :--- | :--- |
| **`analysis_manifest.json`** | 23,077 | `71e4f40657a18f2722986f352e04612f127b1a5f700ef403b503633b3937194d` | Execution manifest, environment, Git commit, shard hash verification, zero-access declaration |
| **`analysis_summary.json`** | 68,095 | `c529058fbca34f0491dff29b5a1469fd7bfc815fc091a98a68d24d954b4e9ffa` | Comprehensive machine-readable analysis results and statistical metrics |
| **`batch_outcomes.csv`** | 54,581 | `4363bc2601673d61a14a847b3df30300d89b100c8428f458bc32cd5b65f74ff5` | Granular per-batch outcomes, risk scores, confusion cells, and signed $\Delta \text{F1}$ ($N=179$) |
| **`event_outcomes.csv`** | 1,137 | `db100f7b2c69997f1da9a7b72cd8becfff43d5ad1ae58f9e5c9a1dd20c0dc9b7` | Event-level F1 outcomes across Source, BN-no-update, and Candidate ($N=6$) |
| **`rejection_budget_summary.csv`** | 7,182 | `0bc13deb6d24d34e0fc68abf97f163465b469cb646ce6858899a0ec26b6b814a` | Rejection curve evaluation across budgets $b \in [0.05, 0.30]$ using source fallback |
| **`signal_batch_grouped_summary.csv`** | 2,076 | `eb2f7149b9942ad72b23214c7d8d1ba6bc4f467ed3bb52db73003d91b6c0f44f` | Grouped batch-level Spearman $\rho$, pooled AUROC, and pooled AUPRC |
| **`signal_event_summary.csv`** | 1,627 | `7c405304e7006701c78db35a2fbd1f6d56659f448489d1f179fa21ba6e25fc9d` | Event-level Spearman $\rho$, AUROC, and AUPRC against $\Delta \text{F1}$ |

### 2.1 Elimination of v1 Redundancy
In the initial v1 run, a duplicate file `e5_risk_analysis_summary.json` was generated because of a historical config key. In v2, the analyzer code was aligned to output only `analysis_summary.json`. The five CSV files in `analysis_s0_oa_v2/` match the hashes of `analysis_s0_oa_v1/` bit-for-bit, proving zero numerical drift between the initial run and the clean committed reproduction.

---

## 3. Formal Scientific Results & Key Findings

### 3.1 Primary Event Outcomes ($N = 6$ Development Events)
*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v2/event_outcomes.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v2/event_outcomes.csv)

| Event ID | Geography | Samples | Shards | Valid Pixels | Source F1 | BN Control F1 | Candidate F1 | $\Delta \text{F1}_{\text{Cand-Src}}$ | $\Delta \text{F1}_{\text{Cand-BN}}$ | Outcome Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **279** | Spain / Ebro | 2,381 | 149 | 116,199,247 | 0.3769 | 0.3758 | 0.3224 | **-0.0545** | -0.0533 | **Harmful** |
| **437** | UK / Severn | 169 | 11 | 7,859,430 | 0.6994 | 0.6925 | 0.6352 | **-0.0642** | -0.0573 | **Harmful** |
| **520** | USA / Florence | 92 | 6 | 3,455,248 | 0.7138 | 0.7106 | 0.6997 | **-0.0140** | -0.0108 | **Harmful** |
| **559** | USA / Midwest | 42 | 3 | 1,843,520 | 0.6726 | 0.6841 | 0.6923 | **+0.0197** | +0.0082 | **Beneficial** |
| **1111003** | Bangladesh / Ganges | 19 | 2 | 815,490 | 0.2568 | 0.2665 | 0.2650 | **+0.0082** | -0.0014 | **Neutral** |
| **1111008** | Myanmar / Sagaing | 123 | 8 | 5,686,145 | 0.8499 | 0.8496 | 0.8479 | **-0.0019** | -0.0016 | **Neutral** |

- **Macro Candidate F1** (unweighted event mean): **0.5771**
- **Harm Prevalence**: $\pi_{\text{harm}} = 3 / 6 = 0.500$ (50.0%)
- **Normalization Control Insight**: Across the three harmful events (279, 437, 520), the BN-no-update control exhibited negligible change relative to Source reference ($\Delta \text{F1}_{\text{BN-Src}} \in [-0.0069, -0.0011]$), whereas Candidate adaptation produced severe degradation. This demonstrates that performance degradation is associated with adaptation updates rather than normalization shifts alone.

### 3.2 Signal Discrimination Ranking
- **Top Generic Indicator**: `source_to_candidate_kl`
  - Event-level: Spearman $\rho = -0.8857$, $\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$ (vs 0.5000 baseline)
  - Grouped batch-level: Mean Spearman $\rho = -0.2329$, Pooled $\text{AUROC} = 0.6530$, Pooled $\text{AUPRC} = 0.5765$ (vs 0.4078 baseline)
- **Top Change-Specific Indicator**: `s_flood_expansion`
  - Event-level: Spearman $\rho = -0.7714$, $\text{AUROC} = 1.0000$, $\text{AUPRC} = 1.0000$ (vs 0.5000 baseline)
  - Grouped batch-level: Mean Spearman $\rho = -0.1360$, Pooled $\text{AUROC} = 0.6929$, Pooled $\text{AUPRC} = 0.6221$ (vs 0.4078 baseline)
- **Top Fine-Grained Batch Indicator**: `abs_delta_flood_pixels`
  - Grouped batch-level: Mean Spearman $\rho = -0.1736$, Pooled $\text{AUROC} = 0.7190$, Pooled $\text{AUPRC} = 0.6457$
- **Uninformative Indicators**:
  - Model confidence metrics (`mean_max_confidence`, `mean_entropy`, `aug_inconsistency`) yielded near-random event separation ($\text{AUROC} \approx 0.444$).
  - Batch normalization displacement (`bn_weight_displacement`) is uninformative at step 0 within individual batches before parameter drift accumulates.

### 3.3 Metric-Semantics Audit & Rejection Budget Performance
*Source file:* [`outputs/e5_temporal_evidence/analysis_s0_oa_v2/rejection_budget_summary.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e5_temporal_evidence/analysis_s0_oa_v2/rejection_budget_summary.csv)

1. **Pooled Pixel Baseline F1 (0.4578)**: Reconstructed overall valid pixel true positives over $2\text{TP} + \text{FP} + \text{FN}$ across all 135,858,070 valid pixels in all 179 batches. (Heavily weighted by Event 279, which contains 85.5% of all valid pixels and has an F1 of 0.3224).
2. **Rejection Fallback Mechanism**: High-risk candidate batches ($\text{score} > \tau_b$) are safely replaced with their **exact source confusion counts** ($\text{TP}_{\text{src}, b}, \text{FP}_{\text{src}, b}, \text{FN}_{\text{src}, b}$).
3. **Harmful Unit Accounting (73 units)**: Evaluated at candidate batch outcome resolution ($\Delta \text{F1}_{\text{batch}} \le -0.010$).
4. **Exemplary Rejection Curve (`s_flood_expansion`)**:
   - At $b = 0.10$: Rejects 18 batches; blocks 12 / 73 harmful batch outcomes (16.4% harm averted) with **0 / 18 lost beneficial outcomes (0.0% loss)**; net pooled pixel F1 increases from **0.4578 to 0.4882**.
   - At $b = 0.20$: Rejects 36 batches; blocks 24 / 73 harmful batch outcomes (**32.9% harm averted**) with **1 / 18 lost beneficial outcomes (5.6% loss)**; net pooled pixel F1 increases from **0.4578 to 0.5000**.
   - At $b = 0.30$: Rejects 54 batches; blocks 36 / 73 harmful batch outcomes (**49.3% harm averted**) with **1 / 18 lost beneficial outcomes (5.6% loss)**; net pooled pixel F1 increases from **0.4578 to 0.5064**.

### 3.4 Calibration & Composite Rule Status
As required by the analysis contract, the multivariate combinations:
- Generic-only combination
- Generic + change-specific combination

are explicitly designated:
```text
NOT EVALUATED — CALIBRATION RULE NOT PREDECLARED
```
No post-hoc weighting or speculative classifier fitting was performed.

---

## 4. Verification & Integrity Checklist

| Check | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Git Commit** | Analysis run from clean HEAD `81f6f3af35fbe6d4d15e76c708d8b8fbffa64f45` | Working tree clean (`git_dirty: false`) | **PASS** |
| **Input Shard Hashes** | All 179 formal shards match `evidence_index.json` | 179 / 179 SHA-256 hashes bit-exact | **PASS** |
| **Unit Test Coverage** | 11 offline analysis unit tests in `test_e5_temporal_risk_analysis.py` | Ran in 30.259s, 11 passed | **PASS** |
| **Contract Test Coverage** | 38 temporal risk contract tests in `test_e5_risk_contracts.py` | Ran in 11.681s, 38 passed | **PASS** |
| **Headless Output Validation** | Exactly 7 canonical artifacts in `analysis_s0_oa_v2/` | Verified, no extra or duplicate files | **PASS** |
| **Numerical Consistency** | Identical scientific results between v1 and v2 | All 5 CSV hashes match bit-for-bit | **PASS** |
| **Zero Sealed-Test Access** | Quarantined test events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` | Verified zero read, zero load, zero access | **PASS** |
| **GPU Inactivity** | No GPU compute launched during E5.2C closeout | Verified CPU execution only | **PASS** |

---

## 5. Epistemic Scope, Remaining Limitations & Next Steps

1. **Exploratory Descriptive Screening**: Six development events provide descriptive insight into risk indicators under stream conditions. They do not constitute asymptotic or general population-level proof.
2. **Streaming Cell Specificity**: These findings represent a single cell: **Seed 0, Stream Arrival Order A**.
3. **Requirement for Replication (E5.3)**: To verify that these risk signals are robust to temporal ordering and parameter initialization, repeat seed and stream order evidence must be collected in **E5.3** under identical bounded-memory contracts.
4. **No Premature Deployment Claims**: These offline results demonstrate the diagnostic efficacy of pre-update signals for selective fallback under simulation, but do not yet authorize an unmonitored production deployment gate.

---

## 6. Formal Terminal Verdict

```text
E5.2 COMPLETE — READY TO START E5.3
```

