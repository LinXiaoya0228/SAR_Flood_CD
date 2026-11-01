# RST4 Discovery Staircase Closeout Report

## 1. Executive Summary

This report documents the completion of **RST4: Discovery Staircase Evaluation** under the ChangeGuard Robust Source-Training Route (`docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`).

RST4 froze and executed four full 16-epoch source-training runs on discovery fold C (Groups A+B for training, 18 events, 14,967 patches; Group C for evaluation, 9 events, 6,495 patches) across four NVIDIA RTX 3090 GPUs in parallel.

The discovery staircase systematically evaluated the additive contributions of:
- **R0**: Matched Baseline (uniform patch sampling, unweighted loss, no paired augmentation)
- **R1**: R0 + M1 (Event-Balanced Sampling)
- **R2**: R1 + M2 (Flood-Aware Class-Weighted Loss)
- **R3**: R2 + M3 (Synchronized Paired SAR Augmentation)

Based on the pre-registered promotion criteria in `configs/robust_source_training/rst4_promotion_rule_v1.json`:
- **R2** qualified for promotion under **Route D2 (Tail reliability improvement)**, achieving a **+0.023907** gain in bottom-3 event mean F1 (threshold >= +0.020) and a **+0.002012** gain in event-macro F1.
- **R3** qualified under **Route D1 (Mean improvement)**, achieving an event-macro F1 of 0.769657 (+0.006523 vs R0).
- Because R2's event-macro F1 (0.765146) is within 0.004511 (<= 0.005 bandwidth) of the top recipe R3, both R2 and R3 entered the parsimony candidate pool.
- Applying the pre-registered parsimony preference (`R1 -> R2 -> R3`), **R2** was selected for promotion. R2 delivers significant tail robustness and a notable precision improvement (+0.058954 over R0) while avoiding paired augmentation (M3), which caused severe degradation on event 497 (-0.240461 F1 reduction).

Deterministic Decision: **`RECIPE_R2_PROMOTED`** (Recipe R2 selected for confirmatory fold training in RST5).

---

## 2. Commit Lineage and Authoritative Artifacts

The robust source-training route lineage leading to RST4 closeout is as follows:

| Stage | Commit Hash | Description |
|---|---|---|
| **RST0** | `9c73e84e853458fcb765422e24a2599c70882ec9` | Route transition and historical freeze |
| **RST1** | `d3e3c052b24b4e49bf5b4768dfcf59426a7f83cc` | Data, input-domain, and matched-baseline audit |
| **RST2 Prelaunch** | `edf6b9e58f47785a867f107491398911fa401430` | Event-disjoint infrastructure freeze |
| **RST2 Closeout** | `35b48875a443c1d661c4c0384047a8f349704401` | Event-disjoint infrastructure closeout |
| **RST2 Addendum** | `b704596e792da869e5d4cbfe086c8a77a9404be1` | Disclosure of deleted diagnostic smoke attempts |
| **RST3 Impl** | `f0635171e54911e3b6eb41a5405e3ba9f0a2df3c` | Implement M1, M2, and M3 components |
| **RST3 Closeout** | `6bc10103e680a6d0ca0a647d6cb04c34cb35be4d` | Component smoke verification closeout |
| **RST3 Reconciliation** | `80ab1758c0ffec20593b4c10c14c51e06fa8325a` | Pre-discovery contract reconciliation |
| **RST4 Prelaunch Freeze** | `bdb40113b41249e8829062f775db7e3fca3b8980` | Discovery staircase protocol and matrix freeze |

---

## 3. Pre-Discovery Contract Reconciliation Record

Prior to launching discovery runs, two documentation and contract points were reconciled in `docs/robust_source_training/rst3_pre_discovery_contract_reconciliation_v1.md`:

1. **M2 Flood-Aware Loss Weighting Formula**:
   - The authoritative formula is:
     ```text
     p_c = valid_pixels_c / total_valid_pixels
     raw_weight_c = 1 / sqrt(max(p_c, 1e-12))
     normalized_weight_c = raw_weight_c / mean(raw_weight)
     clipped_weight_c = clip(normalized_weight_c, 0.5, 2.0)
     final_weight_c = clipped_weight_c / mean(clipped_weight_c)
     ```
   - Applied strictly to the cross-entropy term in `ControlledSourceLoss`; Dice loss remains unweighted.
   - For Groups A+B (14,967 patches, 737,344,057 valid pixels), the computed weights are `[0.492271, 1.666056, 0.841673]` with SHA-256 `1619bfdf0f717b738d4bf5cae51cbf074ff2f63b2a531c697dc2929305b0f62f`.
2. **M3 Paired SAR Augmentation Distribution**:
   - Reconciled to conservative uniform radiometric gain: `gain ~ Uniform(0.95, 1.05)`.
   - Synchronized spatial transforms (RandomHorizontalFlip, RandomVerticalFlip) applied identically across pre, post, and target masks.

---

## 4. Discovery Staircase Execution Environment

The four runs were executed concurrently across 4 GPUs with isolated tmux sessions and dedicated log paths:

| Recipe | Run ID | GPU Device | Seed | Max Epochs | Optimizer Steps | Initial State SHA-256 |
|---|---|---|---|---|---|---|
| **R0** | `rst4_r0_snunet_foldc_s0_v2` | `cuda:0` | 0 | 16 | 7,496 | `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1` |
| **R1** | `rst4_r1_snunet_foldc_s0_v2` | `cuda:1` | 0 | 16 | 7,488 | `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1` |
| **R2** | `rst4_r2_snunet_foldc_s0_v2` | `cuda:2` | 0 | 16 | 7,488 | `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1` |
| **R3** | `rst4_r3_snunet_foldc_s0_v2` | `cuda:3` | 0 | 16 | 7,488 | `71177eb1fb878c54aab309a61d61b4ca45a282c50c06eb037e7c917cfa3ffdb1` |

All four runs began from the **bit-identical initial model state** (`71177eb1fb...`) and trained on the exact same 18 source events from Groups A and B. Evaluation was performed strictly on the 9 unseen events of Group C (6,495 patches, 304,268,202 valid pixels).

---

## 5. Discovery Staircase Evaluation Results

### 5.1 Comprehensive Summary Table

| Metric | R0 (Baseline) | R1 (+M1) | R2 (+M1+M2) | R3 (+M1+M2+M3) |
|---|---|---|---|---|
| **Run ID** | `..._r0_..._v2` | `..._r1_..._v2` | `..._r2_..._v2` | `..._r3_..._v2` |
| **M1: Event-Balanced Sampling** | No | Yes | Yes | Yes |
| **M2: Flood-Aware Loss** | No | No | Yes | Yes |
| **M3: Paired SAR Augmentation** | No | No | No | Yes |
| **Event-Macro F1** | **0.763134** | 0.756085 | **0.765146** | **0.769657** |
| **Event-Macro IoU** | 0.644164 | 0.639089 | 0.643778 | 0.647546 |
| **Event-Macro Precision** | 0.771665 | 0.820078 | **0.830619** | 0.819130 |
| **Event-Macro Recall** | **0.779724** | 0.730677 | 0.731456 | 0.739951 |
| **Pooled F1** | 0.822338 | 0.825016 | 0.828784 | 0.839298 |
| **Pooled IoU** | 0.698280 | 0.702151 | 0.707627 | 0.723095 |
| **Worst-Event ID** | Event 427 | Event 427 | Event 427 | Event 427 |
| **Worst-Event F1** | 0.419705 | 0.325309 | 0.413698 | 0.481716 |
| **Bottom-3 Event Mean F1** | 0.562984 | 0.542608 | **0.586891** | 0.582430 |
| **Delta F1 vs R0** | 0.000000 | -0.007049 | **+0.002012** | **+0.006523** |
| **Delta Bottom-3 vs R0** | 0.000000 | -0.020375 | **+0.023907** | **+0.019446** |
| **Improved Events vs R0 (>=0.01)**| - | 2 | 1 | 6 |
| **Harmed Events vs R0 (<=-0.01)** | - | 3 | 2 | 1 |
| **Neutral Events vs R0 (<0.01)** | - | 4 | 6 | 2 |
| **Wall-Clock Runtime (s)** | 9,355.3 | 8,361.9 | 8,724.9 | 8,315.6 |
| **Peak VRAM (MB)** | 4,913.2 | 4,913.2 | 4,913.2 | 4,913.2 |
| **Final Checkpoint SHA-256** | `5c9725c845...` | `70d3d722c4...` | `da60c7ac29...` | `2f91e3a7f2...` |

---

### 5.2 Pairwise Component Effects

The incremental and composite differences from `component_differences.csv` are summarized below:

| Comparison | Target vs Baseline | Component Tested | Delta Macro F1 | Delta Macro IoU | Delta Precision | Delta Recall | Delta Worst F1 | Delta Bottom-3 F1 | Improved / Harmed / Neutral |
|---|---|---|---|---|---|---|---|---|---|
| **R1 - R0** | R1 vs R0 | **M1** (Event Balance) | -0.007049 | -0.005075 | +0.048413 | -0.049047 | -0.094396 | -0.020375 | 2 / 3 / 4 |
| **R2 - R1** | R2 vs R1 | **M2** (Flood-Aware Loss) | +0.009061 | +0.004689 | +0.010541 | +0.000779 | +0.088389 | +0.044283 | 4 / 1 / 4 |
| **R3 - R2** | R3 vs R2 | **M3** (Paired Augmentation) | +0.004511 | +0.003768 | -0.011489 | +0.008495 | +0.068018 | -0.004461 | 6 / 1 / 2 |
| **R2 - R0** | R2 vs R0 | **M1 + M2** vs Baseline | +0.002012 | -0.000386 | +0.058954 | -0.048268 | -0.006007 | +0.023907 | 1 / 2 / 6 |
| **R3 - R0** | R3 vs R0 | **Full Composite** vs Baseline| +0.006523 | +0.003382 | +0.047465 | -0.039773 | +0.062011 | +0.019446 | 6 / 1 / 2 |

#### Analysis of Component Impacts:
1. **Impact of M1 (Event-Balanced Sampling, R1 vs R0)**:
   - Alone, M1 caused a slight net drop in event-macro F1 (-0.0070) and bottom-3 mean (-0.0204).
   - Worst-event F1 on Event 427 fell by -0.0944 (from 0.4197 to 0.3253). Because Event 427 has 1,980 patches (the largest in Group C), under-sampling large source events without loss reweighting reduced training recall on large flood bodies.
   - However, precision improved markedly (+0.0484), showing reduced false alarms on background regions.
2. **Impact of M2 (Flood-Aware Loss, R2 vs R1)**:
   - Introducing M2 reversed the drawbacks of isolated event-balancing, lifting event-macro F1 by **+0.009061** and bottom-3 mean by **+0.044283**.
   - Worst-event F1 rebounded by **+0.088389** (from 0.3253 back to 0.4137).
   - Event 147 saw a substantial improvement from 0.5593 to 0.5905 (+0.0806 over R0).
   - Compared to R0, R2 achieved a notable **+0.023907** gain on the bottom-3 event mean F1 while improving precision by **+0.058954**.
3. **Impact of M3 (Paired SAR Augmentation, R3 vs R2)**:
   - R3 showed the highest overall event-macro F1 (0.769657) and helped Event 427 reach 0.4817.
   - However, M3 exhibited severe per-event downside risk: Event 497 suffered a catastrophic loss of **-0.240461** F1 (dropping from 0.832082 in R0 to 0.591621 in R3). The random radiometric gain jitter disrupted subtle speckle and backscatter contrast required for water delineation in this event.
   - Consequently, R3's bottom-3 gain (+0.019446) was lower than R2's (+0.023907).

---

### 5.3 Per-Event Breakdown Across Group C (9 Events)

| Event ID | Patch Count | Valid Pixels | R0 F1 | R1 F1 | R2 F1 | R3 F1 | R2 - R0 | R3 - R0 |
|---|---|---|---|---|---|---|---|---|
| **427** | 1,980 | 97,755,206 | 0.419705 | 0.325309 | 0.413698 | 0.481716 | -0.006007 | +0.062011 |
| **1111011** | 1,619 | 80,748,095 | 0.829828 | 0.837929 | 0.836834 | 0.849938 | +0.007006 | +0.020110 |
| **275** | 742 | 36,873,536 | 0.864072 | 0.855542 | 0.850113 | 0.865721 | -0.013959 | +0.001649 |
| **1111005** | 620 | 31,054,253 | 0.759422 | 0.743183 | 0.756520 | 0.788786 | -0.002902 | +0.029364 |
| **497** | 398 | 19,130,935 | 0.832082 | 0.844042 | 0.770666 | 0.591621 | -0.061416 | -0.240461 |
| **147** | 362 | 17,988,126 | 0.509824 | 0.559333 | 0.590455 | 0.673952 | +0.080631 | +0.164128 |
| **1111006** | 313 | 15,484,644 | 0.797912 | 0.777988 | 0.802239 | 0.816420 | +0.004327 | +0.018508 |
| **130** | 248 | 11,811,738 | 0.947969 | 0.953641 | 0.957783 | 0.959102 | +0.009814 | +0.011133 |
| **492** | 213 | 10,250,405 | 0.907391 | 0.907800 | 0.908003 | 0.899654 | +0.000612 | -0.007737 |
| **Event-Macro** | **6,495** | **304,268,202** | **0.763134** | **0.756085** | **0.765146** | **0.769657** | **+0.002012** | **+0.006523** |

---

## 6. Promotion Rule Evaluation and Decision Rationale

The promotion decision was evaluated using the deterministic rule specified in `configs/robust_source_training/rst4_promotion_rule_v1.json`:

1. **Eligibility Evaluation**:
   - **R1**:
     - Route D1: Delta F1 = -0.007049 (<= 0) -> Fail.
     - Route D2: Delta bottom-3 = -0.020375 (< 0.020) -> Fail.
     - Status: Ineligible.
   - **R2**:
     - Route D1: Delta F1 = +0.002012 (> 0), but improved (1) < harmed (2) -> Fail.
     - Route D2: Delta bottom-3 = +0.023907 (>= 0.020 threshold) and Delta F1 = +0.002012 (>= -0.005 threshold) -> **PASS**.
     - Status: **Eligible under Route D2**.
   - **R3**:
     - Route D1: Delta F1 = +0.006523 (> 0) and improved (6) >= harmed (1) -> **PASS**.
     - Route D2: Delta bottom-3 = +0.019446 (< 0.020 threshold) -> Fail.
     - Status: **Eligible under Route D1**.
2. **Top Candidate Determination**:
   - Top event-macro F1 among eligible candidates is R3: **0.769657**.
3. **0.005 Parsimony Bandwidth Pool**:
   - Bandwidth window: `[0.769657 - 0.005, 0.769657] = [0.764657, 0.769657]`.
   - R2's event-macro F1 is **0.765146**, which satisfies `0.765146 >= 0.764657` (gap of 0.004511 <= 0.005000).
   - Therefore, **both R2 and R3 enter the bandwidth pool**.
4. **Parsimony Preference (`R1 -> R2 -> R3`)**:
   - R1 is not in the pool.
   - Next simplest recipe in the hierarchy is **R2**.
   - R2 requires only M1 (event-balanced sampling) and M2 (training-only flood-aware loss reweighting), adding zero runtime latency and no training augmentation jitter.
   - Furthermore, R2 avoids the catastrophic degradation observed with M3 on Event 497 (-0.2405 F1).
5. **Deterministic Outcome**:
   - **`RECIPE_R2_PROMOTED`**

---

## 7. Selected Recipe Freeze Specification for RST5

The selected recipe **R2** is frozen in `configs/robust_source_training/rst4_selected_recipe_v1.json`:
- **Components**:
  - `m1_event_balance`: `true`
  - `m2_flood_aware_loss`: `true`
  - `m3_paired_augmentation`: `false`
- **M1 Configuration**:
  - Uniform event sampling with replacement across all allowed training events.
  - Epoch sample budget: 14,967 patches (468 optimizer steps at effective batch size 32).
- **M2 Configuration**:
  - Applied to `ControlledSourceLoss` cross-entropy term only (Dice loss unweighted).
  - Weights computed from Groups A+B: `[0.492271, 1.666056, 0.841673]`.
- **Confirmatory Rule**:
  - Strictly prohibited from adjusting, re-tuning, or screening hyperparameters on Groups A or B prior to confirmatory evaluation.

---

## 8. Record of Preserved Diagnostic and Failed Attempts

To uphold the strict research-integrity protocol requiring preservation of all failed attempts:
- **Aborted v1 Runs**:
  - An initial launch attempt of R0-R3 (`rst4_r{0,1,2,3}_snunet_foldc_s0_v1`) was launched and immediately detected within 1 minute as having an epoch count configuration mismatch (the runner checked top-level `max_epochs` instead of `training.max_epochs`, defaulting to 1 epoch).
  - The runs were cleanly terminated within 38 seconds before meaningful training occurred.
  - The directories and log files were **preserved intact** under `outputs/robust_source_training/rst4/rst4_r{0,1,2,3}_snunet_foldc_s0_v1` with a formal `terminal_status.json` recording `status: "ABORTED_PRE_DISCOVERY_CONFIG_MISMATCH"`.
  - The runner code was patched in `src/robust_source_training/runner.py` line 310 to inspect `train_cfg.get("max_epochs", ...)`, and the discovery runs were relaunched as version `v2`.

---

## 9. Research Integrity and Boundary Verification

1. **Target Event Isolation**:
   - None of the 16 forbidden target events (`[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`) were accessed, loaded, or evaluated at any point during RST4.
2. **Confirmatory Fold Isolation**:
   - Zero evaluation was performed on Groups A or B.
   - Evaluation in RST4 was strictly confined to Group C.
3. **Historical E0-E11 Protection**:
   - The two historical tracked-dirty E11 documents remained strictly untouched:
     - `docs/e11_final_scientific_conclusions.md`: `26832a93be8efd85ada522c39e9334a678c6b3860f3379efe479567d879b3f36`
     - `docs/e11_final_test_closeout.md`: `2e0d460ce088a8a6883ed5cdfadb61da01d9baaa3bf2b28249efcd6bcfd87ff8`
4. **Limitations of Seed-0 Discovery**:
   - As documented in the proposal, discovery fold C with seed 0 serves as a screening filter. Multi-seed confirmatory fold testing across independent splits (Folds A and B) in RST5 will rigorously establish statistical significance and variance.

---

## 10. Closeout Conclusion

RST4 is **`COMPLETED`**. Recipe **R2** is selected and frozen. The repository is ready for **RST5 (Multi-Fold Confirmatory Training and Evaluation)**.
