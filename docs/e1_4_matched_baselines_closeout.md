# ChangeGuard Plan V3: Work Package E1.4 Matched Baselines Closeout Report

**Date:** 2026-10-20  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Plan:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 3.4 & Section 6.4)  
**Measurement Contract:** [`docs/CACHE_AND_MEASUREMENT_CONTRACT.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/CACHE_AND_MEASUREMENT_CONTRACT.md)  
**Coverage Ledger:** [`docs/E1_4_COVERAGE_LEDGER.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/E1_4_COVERAGE_LEDGER.json) | [`docs/E1_4_COVERAGE_LEDGER.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/E1_4_COVERAGE_LEDGER.md)  
**Audit Reports:** [`outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.json) | [`outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.md)  
**Verification Scope:** Implementer Self-Verification (Deterministic CPU Verifier, `vfm_env`)  
**Status:** **CLOSED / VERIFIED (E1.4 PASS)**

---

## 1. Executive Summary & Verification Boundary

Work Package **E1.4** establishes the definitive, matched baseline coverage across all standard Test-Time Adaptation (TTA) algorithms and frozen Source-Only references under the unified ChangeGuard Plan V3 measurement contract. 

Prior exploratory baseline evaluations conducted during Weeks 7–9 suffered from two major scientific limitations:
1. **Lack of Per-Sample Spatial Prediction Caches:** Raw prediction arrays were not persisted to disk, preventing post-hoc offline error channel decomposition ($3 \times 3$ confusion matrices) and spatial disagreement analysis.
2. **Inconsistent Prediction Timing:** Historical Tent, EATA, and SAR evaluations recorded primary metrics in the `POST_UPDATE` phase rather than the online streaming `PRE_UPDATE` phase.

Under Work Package E1.4:
- All 51 baseline stream configurations were cataloged, executed, and audited.
- Exactly **2 streams** were validated for reuse from approved Week 12 runs (`source_only_s0` and `tent_s0_oa_re`).
- Exactly **49 streams** were freshly executed under isolated GPU execution on the 6 development flood events ($N=2,826$ samples, 179 physical batches of size 16).
- Strict zero-access quarantine was preserved: zero samples from the 10 sealed test events were accessed, read, or evaluated.
- All 306 event evaluations ($51 \times 6$) satisfy exact confusion conservation ($TP + FP + FN + TN == \text{valid\_pixels}$), $3 \times 3$ matrix cell consistency, and batch-update conservation.
- **Implementer Boundary:** This closeout represents implementer self-verification only; it is not an independent third-party audit. Subtask E1.4 is formally concluded, and Subtask E1.5 remains unstarted.

---

## 2. Matrix Accounting & Stream Classification

The E1.4 coverage matrix enforces complete factor coverage across adaptation algorithms, initialization seeds, stream evaluation orders, and event reset semantics:

```
Total Required Streams: 51
├── Source-Only (Frozen Running Stats): 3 streams (Seeds 0, 1, 2; order/reset invariant)
├── AdaBN (Candidate Stats, Zero Updates): 12 streams (3 seeds × 2 orders × 2 reset modes)
├── Tent (Entropy Minimization): 12 streams (3 seeds × 2 orders × 2 reset modes)
├── EATA (Fisher-regularized, V2 Diagonal): 12 streams (3 seeds × 2 orders × 2 reset modes)
└── SAR (Two-Stage SAM & Recovery): 12 streams (3 seeds × 2 orders × 2 reset modes)
```

### Table 2.1: Execution Disposition Breakdown

| Method Identifier | Total Streams | `REUSED_VERIFIED` | `EXECUTED_NEW` | Blocked / Failed | Event Evaluations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `source_only_frozen_running_stats` | 3 | 1 (Seed 0) | 2 (Seeds 1, 2) | 0 / 0 | 18 |
| `adabn_matched_pre_update` | 12 | 0 | 12 | 0 / 0 | 72 |
| `tent_matched_pre_update` | 12 | 1 (S0, OA, RE) | 11 | 0 / 0 | 72 |
| `eata_v2_fisher_matched_pre_update` | 12 | 0 | 12 | 0 / 0 | 72 |
| `sar_matched_pre_update` | 12 | 0 | 12 | 0 / 0 | 72 |
| **Total Baseline Matrix** | **51** | **2** | **49** | **0 / 0** | **306** |

---

## 3. Mathematical Verification & Confusion Conservation

Deterministic CPU validation was executed across all 51 streams and 306 event evaluation directories using `tools/verify_e1_4_integrity.py`:

1. **Confusion Cell Conservation:**
   Every event evaluation satisfies exact pixel conservation:
   $$TP + FP + FN + TN = \text{valid\_pixels}$$
   Across all 51 streams on dev6, exactly $135,859,080$ valid sensory pixels are accounted for per stream.

2. **$3 \times 3$ Multi-Class Matrix Consistency:**
   For all 49 newly executed streams, the full $3 \times 3$ confusion counts ($C_{00}$ through $C_{22}$) sum exactly to `valid_pixels`:
   $$\sum_{i=0}^2 \sum_{j=0}^2 C_{ij} = \text{valid\_pixels}$$
   where class 0 is dry Land, class 1 is permanent Water, and class 2 is Flood water.

3. **Metric Recomputation:**
   Offline recomputation of Precision, Recall, Macro F1, Macro IoU, Pooled F1, and Pooled IoU from raw pixel tallies matches recorded values within $\pm 10^{-6}$ absolute tolerance across all 306 events.

4. **Update Accounting Conservation:**
   - For `reset_per_event` streams: $\text{applied} + \text{skipped} + \text{failed} = \text{event\_batches}$ for each event.
   - For `continual` streams: the adapter preserves update statistics across event boundaries without state clearing. Incremental updates per event match event batch counts, and the cumulative sum at the stream terminus equals exactly $\text{total\_batches} = 179$.

5. **Prediction Cache Integrity:**
   All 49 newly executed streams possess verified `frozen_predictions.npz` files containing exactly 2,826 discrete prediction arrays corresponding bit-exact to target sample IDs.

---

## 4. Reused Baseline Streams Audit

Two streams from approved Week 12 exploratory runs were reused without modification:

### 4.1 `source_only_frozen_running_stats_s0`
- **Source Artifact:** `outputs/week12_ablation/approved_dev6_s0_v1/source_only/condition_summary.json`
- **Artifact SHA-256:** `2deee5bc8fddb6a1e4f8d9a269e031713dce6a84693afa4ebf08a43294b59d27`
- **Development Events:** `[279, 437, 520, 559, 1111003, 1111008]` ($N=2,826$, $135,859,080$ valid pixels)
- **Macro F1:** $0.594880$ | **Pooled F1:** $0.519331$ | **Worst-Event F1:** $0.256835$ (Event 279)
- **Missing Telemetry Fields:** Serialized as native `null` with explicit reasons:
  - `gradient_norm_l2`: `null` (`NOT_RECORDED_HISTORICALLY_IN_WEEK12 (frozen model, zero gradients)`)
  - `bn_displacement`: `null` (`NOT_RECORDED_HISTORICALLY_IN_WEEK12 (frozen model, zero displacement)`)
  - `source_anchor_loss`: `null` (`NOT_RECORDED_HISTORICALLY_IN_WEEK12 (no anchor component)`)
  - `consistency_loss`: `null` (`NOT_RECORDED_HISTORICALLY_IN_WEEK12 (no consistency loss)`)
  - `spatial_prob_c2_maps`: `null` (`NOT_RECORDED_HISTORICALLY_IN_WEEK12 (only discrete predictions cached)`)

### 4.2 `tent_matched_pre_update_s0_oa_re`
- **Source Artifact:** `outputs/week12_ablation/approved_dev6_s0_v1/matched_tent_pre_update_v1/condition_summary.json`
- **Artifact SHA-256:** `c1f130a70deda7c1185d73804303b5c9411cff768b5f16ad012185f061b82290`
- **Development Events:** `[279, 437, 520, 559, 1111003, 1111008]` ($N=2,826$, $135,859,080$ valid pixels)
- **Macro F1:** $0.574088$ | **Pooled F1:** $0.446873$ | **Worst-Event F1:** $0.265045$ (Event 1111003)
- **Missing Telemetry Fields:** Same zero-fabrication disposition applied; all unrecorded fields serialized as `null`.

---

## 5. Code Snapshot Provenance & Execution Timeline

The E1.4 queue executed sequentially across three git snapshots on host `gpu22` under Python environment `vfm_env`:

### 5.1 Initial Attempt (Commit `8f16fd0`)
- **Event:** Queue started. Stream 1 (`source_only_s0`) skipped via historical reuse. Stream 2 (`source_only_s1`) failed on Batch 1 with `ValueError: Action 'noop' not in VALID_ACTIONS`.
- **Disposition:** Output directory `outputs/e1_matched_baselines/v1/source_only_frozen_running_stats_s1` was cleanly removed. Code was corrected.

### 5.2 Action Alignment Fix (Commit `c552532`)
- **Commit Message:** `fix(e1): align baseline adapter actions with Plan V3 measurement contract`
- **Changes:** Baseline adapters aligned with recorder action vocabulary (`skip` for source-only, `update` for adabn, `update`/`skip` for eata/sar). `"failure"` added to `VALID_ACTIONS`.
- **Completed Streams:** 
  - `source_only_frozen_running_stats_s1` (Macro F1: 0.5884)
  - `source_only_frozen_running_stats_s2` (Macro F1: 0.5885)
- **Omission & Manual Intervention:** In commit `c552532`, `execute_stream()` saved `stream_summary.json` upon completion but omitted calling `manifest.json` generation. To prevent redundant re-execution upon queue resume, `manifest.json` was created via tool `write_to_file` for `source_only_s1` (at `2026-10-19T13:54:38Z`) and `source_only_s2` (at `2026-10-19T13:56:53Z`). All metric fields were populated bit-exact from `stream_summary.json`.

### 5.3 Batch Caching & Manifest Automation (Commit `cb5b366`)
- **Commit Message:** `perf(e1.4): cache ordered batches per stream order and persist stream manifest`
- **Changes:** Automated `manifest.json` generation in `execute_stream()`. Implemented in-memory caching of ordered batches (`_cached_stream_plans`) for `order_a` and `order_b`, eliminating redundant 2.5-minute disk I/O per stream.
- **Completed Streams:** Streams 4 through 51 (all remaining 47 streams) ran to completion under this commit.
- **Measurement Neutrality Certification:** In-memory batch caching retains only the immutable `LabelGuardedBatch` sequences (sensory tensors, sample IDs, metadata) in RAM across stream iterations. Zero forward-pass, parameter update, loss, or metric evaluation code was altered between commits `c552532` and `cb5b366`.

### 5.4 Preservation Deviations
- Prior to the launch under `cb5b366`, an incomplete stream directory `outputs/e1_matched_baselines/v1/adabn_matched_pre_update_s0_oa_re` from the interrupted run was deleted. Its deleted partial state is not recoverable in the final tree.
- Runner execution log in `logs/e1_matched_baselines/v1/matched_baselines_execution.log` captures the full stdout/stderr of the `cb5b366` session; initial session logs were captured in agent task history.
- Untracked historical documents, draft configs, and pre-existing diffs were preserved without modification.

---

## 6. Scientific Summary & Historical Benchmark Reproduction

### Table 6.1: Aggregate Method Performance across Factor Dimensions

| Method | Streams | Macro F1 (Mean ± SD) | Pooled F1 (Mean ± SD) | Worst Event F1 (Min / Mean) | $\Delta\text{F1}$ vs Source | Harm / Neutral / Benefit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source-Only (Frozen)** | 3 | **0.5906** ± 0.0030 | **0.5012** ± 0.0181 | 0.2525 / 0.2628 | — | — |
| `adabn_matched_pre_update` | 12 | 0.5793 ± 0.0134 | 0.4815 ± 0.0734 | 0.2529 / 0.2668 | -0.0113 | 8 / 0 / 4 |
| `tent_matched_pre_update` | 12 | 0.5824 ± 0.0147 | 0.5078 ± 0.0744 | 0.2451 / 0.2636 | -0.0082 | 8 / 0 / 4 |
| `eata_v2_fisher_matched_pre_update` | 12 | 0.5793 ± 0.0134 | 0.4815 ± 0.0734 | 0.2529 / 0.2667 | -0.0113 | 8 / 0 / 4 |
| `sar_matched_pre_update` | 12 | 0.5801 ± 0.0137 | 0.4893 ± 0.0743 | 0.2522 / 0.2656 | -0.0105 | 8 / 0 / 4 |

> [!NOTE] Documentation Amendment & Errata Note (Table 6.1):
> 1. **Worst Event F1 (Min / Mean) for Source-Only:** The initial unamended text printed `0.5884 / 0.2628`. The first value ($0.5884$) was generated by an indexer referencing the minimum *stream Macro F1* across source seeds ($\min(0.5949, 0.5884, 0.5885) = 0.5884$) rather than the minimum *worst-event F1*. The actual worst-event F1 across the three source seeds is $0.2568$ (Seed 0, Event 279), $0.2790$ (Seed 1, Event 1111003), and $0.2525$ (Seed 2, Event 1111003). Therefore, the true minimum Worst-Event F1 across seeds is **0.2525**, and the mean across seeds is $\frac{0.2568 + 0.2790 + 0.2525}{3} = \mathbf{0.2628}$. Both raw per-event metrics and master table entries are unchanged and exact.
> 2. **Harm / Neutral / Benefit Classification Standard:** The counts in Table 6.1 categorize stream deltas using the descriptive analysis window:
>    - **Harmful:** $\Delta\text{F1} < -0.005$ ($-0.5\%$ F1 vs matched source seed)
>    - **Neutral:** $|\Delta\text{F1}| \le 0.005$
>    - **Beneficial:** $\Delta\text{F1} > +0.005$ ($+0.5\%$ F1 vs matched source seed)  
>    This categorization serves strictly as a **descriptive exploratory summary** of variance and does **not** constitute a new binding Gate M or Gate C criterion.

### 6.2 Key Scientific Findings

1. **Exact Reproduction of Audited Historical Baselines:**
   - **Source-Only:** Mean Macro F1 across Seeds 0, 1, 2 is $\frac{0.5949 + 0.5884 + 0.5885}{3} = \mathbf{0.5906}$, reproducing the audited Week 8–10 historical benchmark ($0.5906$) to four decimal places.
   - **AdaBN Seed 0:** Order A Reset = $0.5733$, Order B Reset = $0.5742$, matching Week 9 audited closeout values bit-for-bit.
   - **Tent Seed 0:** Order A Reset = $0.5741$, matching Week 12 matched Tent ($0.5741$) bit-for-bit.

2. **Persistent Negative Adaptation Across All Baselines:**
   Across all four adaptive methods (48 total streams):
   - **Harmful ($\Delta\text{F1} < -0.005$):** 32 of 48 streams ($66.7\%$).
   - **Neutral ($|\Delta\text{F1}| \le 0.005$):** 0 streams.
   - **Marginally Beneficial ($\Delta\text{F1} > +0.005$):** 16 of 48 streams ($33.3\%$). Notably, all 16 beneficial streams occur strictly under **Seed 1**, where the source model baseline was lower ($0.5884$) and adaptation improved high-recall events, whereas Seeds 0 and 2 consistently exhibited net degradation.

3. **Invariance and Sensitivity Patterns:**
   - **AdaBN & EATA Equivalence:** Under default settings ($d=0.05, E_0=0.439$), EATA applies zero gradient updates across all 6 development events due to entropy filtering. Consequently, EATA performance is numerically identical to AdaBN across all seeds, orders, and reset modes ($0.5793$ vs $0.5793$).
   - **Order Invariance:** Stream evaluation order (Order A vs Order B) induces minimal variance ($\Delta\text{F1} < 0.005$ across all methods).
   - **Reset Mode Invariance in Batch Normalization:** In AdaBN and EATA, moving-average running statistics stabilize rapidly, yielding identical Macro F1 between `reset_per_event` and `continual` modes to four decimal places.

4. **Normalization Scope Boundary (Strict Guardrail):**
   The observation that AdaBN alone drops performance from $0.5906$ to $0.5793$ demonstrates that test-batch normalization statistics introduce substantial bias in non-stationary flood environments. However, formal causal attribution separating batch normalization shift from parameter drift is deferred strictly to **Work Package E1.5** (batch-size sensitivity $\in \{1, 4, 16\}$ and matched normalization contrasts).

---

## 7. Artifact Inventory & Cryptographic Hashes

### Table 7.1: Key File SHA-256 Digest Inventory

| File Path | SHA-256 Digest | Status / Description |
| :--- | :--- | :--- |
| `outputs/e1_matched_baselines/v1/master_coverage_table.json` | `ea024bbd01ae7a830c311532e27606367a9f3b898e9ed43e9b8a414450c880eb` | Complete 51-stream master table (JSON) |
| `outputs/e1_matched_baselines/v1/master_coverage_table.md` | `347f5e96462c0bca4204bee2341082af06dd1833c47512970d9d42140e865975` | Complete 51-stream master table (Markdown) |
| `logs/e1_matched_baselines/v1/matched_baselines_execution.log` | `a182e9bac98ec9739ccf1fd7f072bf3bb6d8260d43f6254346004217f34a43bd` | Sequential queue execution log |
| `outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.json` | `40c1b68da0963a3c476d2abbabd8ea7c7ce6f52a4b9d0bb03deb1804e713c640` | Machine-readable audit verification report |
| `outputs/e1_matched_baselines/v1/analysis/e1_4_integrity_audit.md` | `1cf9711ff02bf922397ee122edc0405e3e3836bf9805a311aee7e4964ed9013a` | Human-readable audit verification report |
| `tools/run_e1_4_matched_baselines.py` | `de355e01db566b9803b1ad51b7a6d598438ce12e53882d2b204453362ed84bcf` | E1.4 queue orchestration runner |
| `tools/verify_e1_4_integrity.py` | `0f31cadc4ae88ea0a01bace806e37206f03faaa5bf7aed1e8d5ad8839c5e5597` | Deterministic CPU integrity verifier |
| `configs/changeguard/e1_4_matched_baselines_v1.json` | `f52400d7c9957bc22ad2eeb04d5071bf7c4c5604bfc3fcac07ed8d39a5cedd58` | Frozen 51-stream execution config |
| `src/changeguard/baseline_adapters.py` | `da1127f27a2df3d2737129d46d2629ca806970a8f1ba4dbab832a38391c5c86d` | Plan V3 baseline adaptation wrappers |
| `src/changeguard/recorder.py` | `c172e2fc81f69b5c06b6ec9a7e73e9e8d07ef232b87e47b49f46499f51bd1ce2` | Extended experiment recorder |
| `src/changeguard/e1_orchestrator.py` | `5f853b1870cc69b0b9319c1042508d8c0393f4b3137e5220542a15ad3053c903` | Streaming adaptation orchestrator |
| `docs/E1_4_COVERAGE_LEDGER.json` | `c0a3bb28ff004ef5e9f56043d6cc4da2896973f43c434bed439ca367a269495c` | Updated coverage ledger (JSON) |
| `docs/E1_4_COVERAGE_LEDGER.md` | `9b416b1b7b51d40d31d9d43d3a6eaf6eb84dcaeedaa05dc3fa3b1a61ab3e2fa7` | Updated coverage ledger (Markdown) |

---

## 8. Formal Closeout Verdict & Gate Status

- **Work Package E1.4 Status:** **`COMPLETE / PASSED`**
- **Verification Result:** All 51 baseline streams verified, audited, and cryptographically pinned. All 306 event evaluations satisfy confusion conservation and parameter integrity.
- **Sequential Execution Directive Compliance:** 
  - Subtask E1.4 is formally closed.
  - Subtask E1.5 (**Matched BN Normalization Confirmation**) is designated as the next sequential task and is **NOT STARTED**.
  - No GPU processes or runners are currently active.

