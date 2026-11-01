# ChangeGuard Plan V3: Work Package E1.4 Matched Baseline Coverage Ledger

**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 3.4 & Section 6.4)  
**Measurement Contract:** [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  
**Machine-Readable Ledger:** [`docs/E1_4_COVERAGE_LEDGER.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/E1_4_COVERAGE_LEDGER.json)  
**Provenance:** Implementer Self-Verification (Python `vfm_env`)  

---

## 1. Executive Summary & Baseline Stream Accounting

Work Package **E1.4** establishes the definitive, matched baseline coverage under the unified Plan V3 measurement contract for all comparative adaptation claims. 

### Stream Inventory & Classification Standards:
1. **`source_only_frozen_running_stats`**: Evaluated strictly **once per seed** (Seeds 0, 1, 2) across the 6 development events ($N=2,826$ samples). Source evaluation uses frozen training batch normalization statistics and zero parameter updates; it is mathematically and empirically invariant to sample ordering and reset modes. It is **never multiplied into fake order/reset runs**.
2. **`adabn_matched_pre_update`**: 12 streams covering Seeds 0, 1, 2 $\times$ Orders A, B $\times$ Reset modes (`reset_per_event`, `continual`). Evaluated under candidate batch statistics with zero gradient updates.
3. **`tent_matched_pre_update`**: 12 streams covering Seeds 0, 1, 2 $\times$ Orders A, B $\times$ Reset modes. Primary predictions obtained in `PRE_UPDATE` phase; diagnostic predictions obtained in `POST_UPDATE` phase.
4. **`eata_v2_fisher_matched_pre_update`**: 12 streams covering Seeds 0, 1, 2 $\times$ Orders A, B $\times$ Reset modes. Evaluated with canonical V2 Fisher diagonal artifacts, redundancy margin $d_{\text{margin}} = 0.05$, and reliability threshold $E_0 = 0.40 \ln(3) \approx 0.4394449$.
5. **`sar_matched_pre_update`**: 12 streams covering Seeds 0, 1, 2 $\times$ Orders A, B $\times$ Reset modes. Evaluated with two-stage SAM ($\rho = 0.05$), reliability threshold $E_0$, and model recovery threshold $e_m = 0.2 \ln(3)/\ln(1000) \approx 0.03180808$.
6. **`matched_bn_no_update` Diagnostic Note**: The exploratory matched BN no-update condition from Week 12 is preserved as an **E1.5 normalization diagnostic reference**, and is **NOT expanded during E1.4**.

### Stream Accounting Summary:
- **Total Required Stream Configurations:** **51**
  - Source-Only: 3
  - AdaBN: 12
  - Tent: 12
  - EATA: 12
  - SAR: 12
- **Reused Without Modification (`REUSED_VERIFIED`):** **2**
  1. `source_only_frozen_running_stats_s0` (from `outputs/week12_ablation/approved_dev6_s0_v1/source_only`)
  2. `tent_matched_pre_update_s0_oa_re` (from `outputs/week12_ablation/approved_dev6_s0_v1/matched_tent_pre_update_v1`)
- **Executed New & Verified (`EXECUTED_NEW`):** **49**
  - Source-Only: 2 (Seeds 1, 2)
  - AdaBN: 12 (all 12 streams)
  - Tent: 11 (all except Seed 0 Order A Reset)
  - EATA: 12 (all 12 streams)
  - SAR: 12 (all 12 streams)
- **Blocked Streams (`BLOCKED`):** **0**
- **Excluded / Not Applicable (`NOT_APPLICABLE`):** 0
- **Failed Streams (`FAILED`):** **0**
- **Audit Verdict:** ✅ **PASSED (Deterministic Verification Complete)**

---

## 2. Strict Historical Reuse Test & Missing Field Dispositions

In strict compliance with Plan V3 Section 3.4 and the user directive:
- **Zero Fabrication Policy:** For reused historical cells, fields that were not recorded historically are serialized as native `null` with explicit reasons (`NOT_RECORDED_HISTORICALLY` or `NOT_APPLICABLE`). Under no circumstances are missing values substituted with fabricated zeros or reconstructed from rounded aggregates.
- **Matched Replay Requirement:** Historical runs from Weeks 7, 8, and 9 lacked spatial prediction caches (`frozen_predictions.npz`) or used `POST_UPDATE` predictions for primary evaluation. Rather than attempting to patch or misrepresent older outputs, these 49 missing streams are scheduled for matched execution under the unified E1 measurement contract.

### Reused Stream Evidence & Audit:

| Stream ID | Method | Checkpoint SHA-256 | Source Artifact Path | Invariance & Audit Status | Missing Fields Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `source_only_frozen_running_stats_s0` | Source-Only | `247d3584...` | `outputs/week12_ablation/approved_dev6_s0_v1/source_only` | Verified identical sample sequence (2,826 samples across dev6), batch size 16, double-masking, finite parameters. | `spatial_prob_c2_maps`: `null` (`NOT_RECORDED_HISTORICALLY`)<br>`post_update_predictions`: `null` (`NOT_APPLICABLE`)<br>`cache_schema_v1`: `null` (`NOT_RECORDED_HISTORICALLY`) |
| `tent_matched_pre_update_s0_oa_re` | Tent | `247d3584...` | `outputs/week12_ablation/approved_dev6_s0_v1/matched_tent_pre_update_v1` | Verified identical sample sequence, Adam lr=1e-4, PRE_UPDATE primary predictions cached in `frozen_predictions.npz`, step records and event metrics complete. | `spatial_prob_c2_maps`: `null` (`NOT_RECORDED_HISTORICALLY`)<br>`post_update_predictions`: `null` (`NOT_RECORDED_HISTORICALLY`)<br>`cache_schema_v1`: `null` (`NOT_RECORDED_HISTORICALLY`) |

---

## 3. Complete 51-Stream Coverage Ledger

| Row | Stream ID | Method | Seed | Order | Reset Mode | Status | Macro F1 | Pooled F1 | Worst-Event F1 | Audit Notes |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `source_only_frozen_running_stats_s0` | `source_only_frozen_running_stats` | 0 | `order_a_invariant` | `invariant` | `REUSED_VERIFIED` | 0.5949 | 0.5193 | 0.2568 | Reused from Week 12 (verified) |
| 2 | `source_only_frozen_running_stats_s1` | `source_only_frozen_running_stats` | 1 | `order_a_invariant` | `invariant` | `EXECUTED_NEW` | 0.5884 | 0.4766 | 0.2790 | Executed in E1.4 (verified) |
| 3 | `source_only_frozen_running_stats_s2` | `source_only_frozen_running_stats` | 2 | `order_a_invariant` | `invariant` | `EXECUTED_NEW` | 0.5885 | 0.5078 | 0.2525 | Executed in E1.4 (verified) |
| 4 | `adabn_matched_pre_update_s0_oa_re` | `adabn_matched_pre_update` | 0 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5733 | 0.4417 | 0.2650 | Executed in E1.4 (verified) |
| 5 | `adabn_matched_pre_update_s0_oa_co` | `adabn_matched_pre_update` | 0 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5733 | 0.4417 | 0.2650 | Executed in E1.4 (verified) |
| 6 | `adabn_matched_pre_update_s0_ob_re` | `adabn_matched_pre_update` | 0 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5742 | 0.4410 | 0.2724 | Executed in E1.4 (verified) |
| 7 | `adabn_matched_pre_update_s0_ob_co` | `adabn_matched_pre_update` | 0 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5742 | 0.4410 | 0.2724 | Executed in E1.4 (verified) |
| 8 | `adabn_matched_pre_update_s1_oa_re` | `adabn_matched_pre_update` | 1 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5977 | 0.5829 | 0.2803 | Executed in E1.4 (verified) |
| 9 | `adabn_matched_pre_update_s1_oa_co` | `adabn_matched_pre_update` | 1 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5977 | 0.5829 | 0.2803 | Executed in E1.4 (verified) |
| 10 | `adabn_matched_pre_update_s1_ob_re` | `adabn_matched_pre_update` | 1 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5977 | 0.5858 | 0.2741 | Executed in E1.4 (verified) |
| 11 | `adabn_matched_pre_update_s1_ob_co` | `adabn_matched_pre_update` | 1 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5977 | 0.5858 | 0.2741 | Executed in E1.4 (verified) |
| 12 | `adabn_matched_pre_update_s2_oa_re` | `adabn_matched_pre_update` | 2 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5672 | 0.4208 | 0.2560 | Executed in E1.4 (verified) |
| 13 | `adabn_matched_pre_update_s2_oa_co` | `adabn_matched_pre_update` | 2 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5672 | 0.4208 | 0.2560 | Executed in E1.4 (verified) |
| 14 | `adabn_matched_pre_update_s2_ob_re` | `adabn_matched_pre_update` | 2 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5656 | 0.4164 | 0.2529 | Executed in E1.4 (verified) |
| 15 | `adabn_matched_pre_update_s2_ob_co` | `adabn_matched_pre_update` | 2 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5656 | 0.4164 | 0.2529 | Executed in E1.4 (verified) |
| 16 | `tent_matched_pre_update_s0_oa_re` | `tent_matched_pre_update` | 0 | `order_a` | `reset_per_event` | `REUSED_VERIFIED` | 0.5741 | 0.4469 | 0.2650 | Reused from Week 12 (verified) |
| 17 | `tent_matched_pre_update_s0_oa_co` | `tent_matched_pre_update` | 0 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5752 | 0.4648 | 0.2561 | Executed in E1.4 (verified) |
| 18 | `tent_matched_pre_update_s0_ob_re` | `tent_matched_pre_update` | 0 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5790 | 0.4704 | 0.2724 | Executed in E1.4 (verified) |
| 19 | `tent_matched_pre_update_s0_ob_co` | `tent_matched_pre_update` | 0 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5807 | 0.4795 | 0.2721 | Executed in E1.4 (verified) |
| 20 | `tent_matched_pre_update_s1_oa_re` | `tent_matched_pre_update` | 1 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.6035 | 0.6096 | 0.2803 | Executed in E1.4 (verified) |
| 21 | `tent_matched_pre_update_s1_oa_co` | `tent_matched_pre_update` | 1 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5952 | 0.6053 | 0.2632 | Executed in E1.4 (verified) |
| 22 | `tent_matched_pre_update_s1_ob_re` | `tent_matched_pre_update` | 1 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.6035 | 0.6129 | 0.2740 | Executed in E1.4 (verified) |
| 23 | `tent_matched_pre_update_s1_ob_co` | `tent_matched_pre_update` | 1 | `order_b` | `continual` | `EXECUTED_NEW` | 0.6050 | 0.6197 | 0.2736 | Executed in E1.4 (verified) |
| 24 | `tent_matched_pre_update_s2_oa_re` | `tent_matched_pre_update` | 2 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5705 | 0.4445 | 0.2560 | Executed in E1.4 (verified) |
| 25 | `tent_matched_pre_update_s2_oa_co` | `tent_matched_pre_update` | 2 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5615 | 0.4420 | 0.2451 | Executed in E1.4 (verified) |
| 26 | `tent_matched_pre_update_s2_ob_re` | `tent_matched_pre_update` | 2 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5694 | 0.4442 | 0.2528 | Executed in E1.4 (verified) |
| 27 | `tent_matched_pre_update_s2_ob_co` | `tent_matched_pre_update` | 2 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5707 | 0.4537 | 0.2526 | Executed in E1.4 (verified) |
| 28 | `eata_v2_fisher_matched_pre_update_s0_oa_re` | `eata_v2_fisher_matched_pre_update` | 0 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5733 | 0.4418 | 0.2650 | Executed in E1.4 (verified) |
| 29 | `eata_v2_fisher_matched_pre_update_s0_oa_co` | `eata_v2_fisher_matched_pre_update` | 0 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5733 | 0.4418 | 0.2650 | Executed in E1.4 (verified) |
| 30 | `eata_v2_fisher_matched_pre_update_s0_ob_re` | `eata_v2_fisher_matched_pre_update` | 0 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5742 | 0.4410 | 0.2724 | Executed in E1.4 (verified) |
| 31 | `eata_v2_fisher_matched_pre_update_s0_ob_co` | `eata_v2_fisher_matched_pre_update` | 0 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5742 | 0.4410 | 0.2723 | Executed in E1.4 (verified) |
| 32 | `eata_v2_fisher_matched_pre_update_s1_oa_re` | `eata_v2_fisher_matched_pre_update` | 1 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5977 | 0.5829 | 0.2803 | Executed in E1.4 (verified) |
| 33 | `eata_v2_fisher_matched_pre_update_s1_oa_co` | `eata_v2_fisher_matched_pre_update` | 1 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5977 | 0.5829 | 0.2803 | Executed in E1.4 (verified) |
| 34 | `eata_v2_fisher_matched_pre_update_s1_ob_re` | `eata_v2_fisher_matched_pre_update` | 1 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5977 | 0.5858 | 0.2740 | Executed in E1.4 (verified) |
| 35 | `eata_v2_fisher_matched_pre_update_s1_ob_co` | `eata_v2_fisher_matched_pre_update` | 1 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5977 | 0.5858 | 0.2740 | Executed in E1.4 (verified) |
| 36 | `eata_v2_fisher_matched_pre_update_s2_oa_re` | `eata_v2_fisher_matched_pre_update` | 2 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5672 | 0.4209 | 0.2560 | Executed in E1.4 (verified) |
| 37 | `eata_v2_fisher_matched_pre_update_s2_oa_co` | `eata_v2_fisher_matched_pre_update` | 2 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5672 | 0.4209 | 0.2560 | Executed in E1.4 (verified) |
| 38 | `eata_v2_fisher_matched_pre_update_s2_ob_re` | `eata_v2_fisher_matched_pre_update` | 2 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5656 | 0.4164 | 0.2529 | Executed in E1.4 (verified) |
| 39 | `eata_v2_fisher_matched_pre_update_s2_ob_co` | `eata_v2_fisher_matched_pre_update` | 2 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5656 | 0.4164 | 0.2529 | Executed in E1.4 (verified) |
| 40 | `sar_matched_pre_update_s0_oa_re` | `sar_matched_pre_update` | 0 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5741 | 0.4472 | 0.2650 | Executed in E1.4 (verified) |
| 41 | `sar_matched_pre_update_s0_oa_co` | `sar_matched_pre_update` | 0 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5739 | 0.4472 | 0.2610 | Executed in E1.4 (verified) |
| 42 | `sar_matched_pre_update_s0_ob_re` | `sar_matched_pre_update` | 0 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5752 | 0.4473 | 0.2724 | Executed in E1.4 (verified) |
| 43 | `sar_matched_pre_update_s0_ob_co` | `sar_matched_pre_update` | 0 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5754 | 0.4488 | 0.2722 | Executed in E1.4 (verified) |
| 44 | `sar_matched_pre_update_s1_oa_re` | `sar_matched_pre_update` | 1 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5997 | 0.5921 | 0.2803 | Executed in E1.4 (verified) |
| 45 | `sar_matched_pre_update_s1_oa_co` | `sar_matched_pre_update` | 1 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5962 | 0.5907 | 0.2739 | Executed in E1.4 (verified) |
| 46 | `sar_matched_pre_update_s1_ob_re` | `sar_matched_pre_update` | 1 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5996 | 0.5954 | 0.2740 | Executed in E1.4 (verified) |
| 47 | `sar_matched_pre_update_s1_ob_co` | `sar_matched_pre_update` | 1 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5999 | 0.5965 | 0.2740 | Executed in E1.4 (verified) |
| 48 | `sar_matched_pre_update_s2_oa_re` | `sar_matched_pre_update` | 2 | `order_a` | `reset_per_event` | `EXECUTED_NEW` | 0.5681 | 0.4275 | 0.2560 | Executed in E1.4 (verified) |
| 49 | `sar_matched_pre_update_s2_oa_co` | `sar_matched_pre_update` | 2 | `order_a` | `continual` | `EXECUTED_NEW` | 0.5656 | 0.4270 | 0.2522 | Executed in E1.4 (verified) |
| 50 | `sar_matched_pre_update_s2_ob_re` | `sar_matched_pre_update` | 2 | `order_b` | `reset_per_event` | `EXECUTED_NEW` | 0.5668 | 0.4256 | 0.2529 | Executed in E1.4 (verified) |
| 51 | `sar_matched_pre_update_s2_ob_co` | `sar_matched_pre_update` | 2 | `order_b` | `continual` | `EXECUTED_NEW` | 0.5670 | 0.4268 | 0.2528 | Executed in E1.4 (verified) |
---

## 4. Preserved Normalization Diagnostic (Work Package E1.5 Scope)

- `matched_bn_no_update_s0_oa_re`: Completed in Week 12 (`outputs/week12_ablation/approved_bn_diagnostic_dev6_s0_v1/matched_bn_no_update_v1`). Preserved as an authoritative normalization contrast. Extension across Seeds 1 and 2, Order B, and batch-size sensitivity $\in \{1, 4, 16\}$ is assigned strictly to **Work Package E1.5**.

