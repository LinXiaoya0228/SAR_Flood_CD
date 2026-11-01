# Week 3 Seed-0/Seed-1 Source Comparison & Seed-1 Closeout Report

**Date:** 2026-10-11
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Git Branch & HEAD:** `main` at commit `c9fcaa5d5380fdafed40d7239e05369ecf04b478` (`c9fcaa5`)
**Git Status:** Clean (`git status --porcelain` is empty)
**Governing Protocol:** `docs/week3_source_replication_protocol.md` (SHA-256: `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927`)
**Immutable Lock:** `configs/source/source_training_week3_replication_lock.json` (SHA-256: `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677`)

---

## 1. Epistemological Categorization & Executive Summary

To maintain research integrity and prevent over-interpretation, all statements in this report follow strict epistemological classification:
- **[FACT]:** Directly verified from immutable on-disk artifacts (manifests, CSVs, logs, checkpoint binaries, Git tree).
- **[CALC]:** Mathematical transformations and statistical summaries computed deterministically from verified facts.
- **[UNAVAILABLE]:** Properties that cannot be observed from preserved evidence.
- **[INTERPRETATION]:** Working scientific conclusions bounded by experimental scope, requiring explicit caveats and future validation.

### Executive Summary of Key Findings
1. **Successful Autonomous Completion [FACT]:** Both Week 3 seed-1 runs completed without intervention under detached `tmux` sessions. Both terminated via standard early stopping (`COMPLETED_EARLY_STOP`) without collapse or numerical failure.
2. **Exact Seed-1 Results [FACT]:**
   - **Model A Seed 1 (Radar Only `[VV, VH]`):** Best inner-validation event-macro F1 = **0.793257** at Epoch 32 (final Epoch 52).
   - **Model B Seed 1 (Radar + Slope `[VV, VH, slope]`):** Best inner-validation event-macro F1 = **0.792165** at Epoch 37 (final Epoch 57).
3. **Slope Effect Inconsistency Across Seeds [CALC, INTERPRETATION]:**
   - In **Seed 0**, Model B outperformed Model A by **+0.015316** (+1.53 pp) on event-macro F1.
   - In **Seed 1**, Model A outperformed Model B by **+0.001092** (+0.11 pp) on event-macro F1, and by **+0.019419** (+1.94 pp) on worst-event F1.
   - **Conclusion:** On the primary checkpoint selection metric (inner-validation event-macro F1) and on the worst-event metric, the sign of the slope effect flips across seeds. Topographic slope has not yet shown a stable event-macro or worst-event improvement across seeds.
4. **Data Isolation & Quarantine Compliance [FACT]:** Zero quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) and zero sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) were accessed.

---

## 2. Independent Verification of Week 3 Seed-1 Runs

### 2.1 Process Lifecycle & GPU Verification
- **Tmux Sessions:** Both detached sessions (`cg_w3_a_s1` and `cg_w3_b_s1`) terminated cleanly upon training completion [FACT].
- **Process Liveness:** No Week 3 Python training process (`tools/train_source_model.py`) remains active [FACT].
- **Active System GPU Audit:** At time of verification (2026-10-11 14:16 UTC), physical GPU 1 and GPU 2 were idle (18 MiB VRAM used by display server Xorg). Physical GPU 0 was active running an unrelated process: PID `3587447` owned by user `s126mdg22_02` executing `python vgg16/train_vgg16.py --train`. This process was not disturbed or terminated [FACT].

### 2.2 Run Integrity Matrix
| Verification Item | Model A Seed 1 (`[VV, VH]`) | Model B Seed 1 (`[VV, VH, slope]`) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Run ID** | `source_snunet_vvvh_seed1_v3_fp32` | `source_snunet_vvvh_slope_seed1_v3_fp32` | Verified [FACT] |
| **Output Directory** | `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32` | `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32` | Verified [FACT] |
| **Completion Status** | `COMPLETED_EARLY_STOP` | `COMPLETED_EARLY_STOP` | Verified [FACT] |
| **Failure Manifest** | Does NOT exist (`failure_manifest.json` absent) | Does NOT exist (`failure_manifest.json` absent) | Verified [FACT] |
| **Random Seed** | `1` (resolved from config, no CLI override) | `1` (resolved from config, no CLI override) | Verified [FACT] |
| **Git Commit & Tree** | `c9fcaa5d5380fdafed40d7239e05369ecf04b478` (`git_dirty: false`) | `c9fcaa5d5380fdafed40d7239e05369ecf04b478` (`git_dirty: false`) | Verified [FACT] |
| **Final Epoch** | Epoch 52 | Epoch 57 | Verified [FACT] |
| **Best Epoch** | Epoch 32 | Epoch 37 | Verified [FACT] |
| **Patience Exhausted** | 20 / 20 epochs without improvement | 20 / 20 epochs without improvement | Verified [FACT] |
| **Collapse Counter** | `0 / 3` across all 52 epochs | `0 / 3` across all 57 epochs | Verified [FACT] |
| **Numerical Integrity** | 100% finite (0 NaN, 0 Inf across train/val histories) | 100% finite (0 NaN, 0 Inf across train/val histories) | Verified [FACT] |
| **Best Checkpoint Exists** | `best_checkpoint.pt` (144,672,493 bytes) | `best_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Last Checkpoint Exists** | `last_checkpoint.pt` (144,672,493 bytes) | `last_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Checkpoint Loadability**| Loads cleanly in PyTorch (`model_state_dict` present) | Loads cleanly in PyTorch (`model_state_dict` present) | Verified [FACT] |
| **Train Samples Processed**| 19,178 samples (100% match) | 19,178 samples (100% match) | Verified [FACT] |
| **Val Samples Evaluated** | 2,284 samples (100% match) | 2,284 samples (100% match) | Verified [FACT] |
| **Accessed Train Events (22)**| `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]` | Same 22 events | Verified [FACT] |
| **Accessed Val Events (5)**| `[267, 275, 497, 518, 1111005]` | Same 5 events | Verified [FACT] |
| **TTA-Val Events Accessed**| 0 events (`no_tta_val_accessed: true`) | 0 events (`no_tta_val_accessed: true`) | Verified [FACT] |
| **Sealed Test Accessed** | 0 events (`no_sealed_test_accessed: true`) | 0 events (`no_sealed_test_accessed: true`) | Verified [FACT] |

---

## 3. Exact Seed-1 Quantitative Results

All values are read directly from `validation_history.csv` and `event_validation_metrics.csv` without console rounding [FACT].

### 3.1 Model A Seed 1 (Radar Only `[VV, VH]`)
- **Best Epoch:** 32 [FACT]
- **Final Epoch:** 52 [FACT]
- **Early-Stopping Trigger:** Patience 20/20 reached at Epoch 52 following peak at Epoch 32 [FACT]
- **Event-Macro Metrics (at Best Epoch 32):**
  - Event-Macro F1: **`0.793257`**
  - Macro Precision: **`0.776712`**
  - Macro Recall: **`0.844683`**
  - Macro IoU: **`0.660814`**
- **Pixel-Pooled Metrics (at Best Epoch 32):**
  - Pooled F1: **`0.796832`**
  - Pooled Precision: **`0.831854`**
  - Pooled Recall: **`0.764639`**
  - Pooled IoU: **`0.662278`**
- **Worst-Event Breakdown (at Best Epoch 32):**
  - Worst Event ID: **`267`**
  - Worst Event F1: **`0.733998`** (Precision: 0.596643, Recall: 0.953509, IoU: 0.579777)
- **All Inner-Validation Events at Best Epoch 32:**
  - Event 267: F1 = `0.733998`, IoU = `0.579777`
  - Event 275: F1 = `0.866220`, IoU = `0.764010`
  - Event 497: F1 = `0.848718`, IoU = `0.737194`
  - Event 518: F1 = `0.740068`, IoU = `0.587388`
  - Event 1111005: F1 = `0.777282`, IoU = `0.635700`
- **Checkpoint Hashes:**
  - `best_checkpoint.pt`: `985a766f8ddcfd14fdaf5ade6bc6257f7e21f5ffac2483f4d8ea95536d02ed5d`
  - `last_checkpoint.pt`: `2b1cd1b365e8bab36a77f23519b2e33e9ab57455bb9eb5488485766d2de91c70`

### 3.2 Model B Seed 1 (Radar + Slope `[VV, VH, slope]`)
- **Best Epoch:** 37 [FACT]
- **Final Epoch:** 57 [FACT]
- **Early-Stopping Trigger:** Patience 20/20 reached at Epoch 57 following peak at Epoch 37 [FACT]
- **Event-Macro Metrics (at Best Epoch 37):**
  - Event-Macro F1: **`0.792165`**
  - Macro Precision: **`0.770256`**
  - Macro Recall: **`0.853064`**
  - Macro IoU: **`0.659916`**
- **Pixel-Pooled Metrics (at Best Epoch 37):**
  - Pooled F1: **`0.802090`**
  - Pooled Precision: **`0.827668`**
  - Pooled Recall: **`0.778046`**
  - Pooled IoU: **`0.669574`**
- **Worst-Event Breakdown (at Best Epoch 37):**
  - Worst Event ID: **`267`**
  - Worst Event F1: **`0.714579`** (Precision: 0.569478, Recall: 0.958905, IoU: 0.555911)
- **All Inner-Validation Events at Best Epoch 37:**
  - Event 267: F1 = `0.714579`, IoU = `0.555911`
  - Event 275: F1 = `0.866349`, IoU = `0.764212`
  - Event 497: F1 = `0.847793`, IoU = `0.735799`
  - Event 518: F1 = `0.735177`, IoU = `0.581249`
  - Event 1111005: F1 = `0.796928`, IoU = `0.662411`
- **Checkpoint Hashes:**
  - `best_checkpoint.pt`: `b20dd72232e9022027d75f616748af8376d32dd8d5b83d039199e393bdaaeeef`
  - `last_checkpoint.pt`: `a4095e161b6461a50b749a73e26ac1a403cd25a8e3abb4b450a04113ebf592e6`

---

## 4. Matched Seed-0 vs Seed-1 Source Comparison

Machine-derived summaries are saved under `outputs/week3_source_replication/comparison_seed0_seed1/` [FACT]:
- `source_seed0_seed1_comparison.csv`
- `paired_differences.csv`
- `comparison_summary.json`

### 4.1 Four-Run Master Comparison Table [FACT]
| Model | Channels | Seed | Status | Best Ep | Final Ep | Event-Macro F1 | Pooled F1 | Worst-Event F1 (Ev ID) | Macro Prec | Macro Rec | Macro IoU | Pooled Prec | Pooled Rec | Pooled IoU | Best Checkpoint SHA-256 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A** | `[VV, VH]` | 0 | COMPLETED | 8 | 28 | 0.781741 | 0.791289 | 0.706420 (267) | 0.756831 | 0.848323 | 0.646084 | 0.818764 | 0.765599 | 0.654656 | `750360fca8fd...` |
| **Model B** | `[VV, VH, slope]` | 0 | COMPLETED | 15 | 35 | 0.797057 | 0.810865 | 0.718636 (267) | 0.769214 | 0.862601 | 0.666302 | 0.827708 | 0.794694 | 0.681895 | `321e80a07a2a...` |
| **Model A** | `[VV, VH]` | 1 | COMPLETED | 32 | 52 | 0.793257 | 0.796832 | 0.733998 (267) | 0.776712 | 0.844683 | 0.660814 | 0.831854 | 0.764639 | 0.662278 | `985a766f8ddc...` |
| **Model B** | `[VV, VH, slope]` | 1 | COMPLETED | 37 | 57 | 0.792165 | 0.802090 | 0.714579 (267) | 0.770256 | 0.853064 | 0.659916 | 0.827668 | 0.778046 | 0.669574 | `b20dd72232e9...` |

### 4.2 Paired Model Differences ($\Delta = \text{Model B} - \text{Model A}$) [CALC]
| Metric | Seed 0 $\Delta$ (B - A) | Seed 1 $\Delta$ (B - A) | Sign Consistency Across Seeds | Interpretation |
| :--- | :---: | :---: | :--- | :--- |
| **Event-Macro F1** | **`+0.015316`** (+1.53 pp) | **`-0.001092`** (-0.11 pp) | **INCONSISTENT (FLIPS)** | Seed 0 favored slope; Seed 1 favored radar only |
| **Pooled F1** | **`+0.019576`** (+1.96 pp) | **`+0.005258`** (+0.53 pp) | **CONSISTENT (+)** | Slope positive on pixel pool in both seeds |
| **Worst-Event F1** | **`+0.012216`** (+1.22 pp) | **`-0.019419`** (-1.94 pp) | **INCONSISTENT (FLIPS)** | Seed 0 favored slope; Seed 1 favored radar only on Ev 267 |
| **Macro Precision** | `+0.012383` | `-0.006456` | **INCONSISTENT (FLIPS)** | Seed 0 favored slope; Seed 1 favored radar only |
| **Macro Recall** | `+0.014278` | `+0.008381` | **CONSISTENT (+)** | Slope slightly improves macro recall in both seeds |
| **Macro IoU** | `+0.020218` | `-0.000898` | **INCONSISTENT (FLIPS)** | Seed 0 favored slope; Seed 1 favored radar only |
| **Pooled Precision** | `+0.008944` | `-0.004186` | **INCONSISTENT (FLIPS)** | Seed 0 favored slope; Seed 1 favored radar only |
| **Pooled Recall** | `+0.029095` | `+0.013407` | **CONSISTENT (+)** | Slope consistently improves pooled recall |
| **Pooled IoU** | `+0.027239` | `+0.007296` | **CONSISTENT (+)** | Slope consistently improves pooled IoU |

### 4.3 Aggregated Two-Seed Statistics ($N=2$, sample $s$ with $N-1=1$) [CALC]
| Metric | Model A Mean $\pm$ $s$ [Range] | Model B Mean $\pm$ $s$ [Range] | Paired Difference Mean $\pm$ $s$ [Range] |
| :--- | :---: | :---: | :---: |
| **Event-Macro F1** | `0.787499 ± 0.008143` [`0.011516`] | `0.794611 ± 0.003459` [`0.004892`] | `+0.007112 ± 0.011602` [`0.016408`] |
| **Pooled F1** | `0.794061 ± 0.003919` [`0.005543`] | `0.806477 ± 0.006205` [`0.008775`] | `+0.012417 ± 0.010124` [`0.014318`] |
| **Worst-Event F1** | `0.720209 ± 0.019501` [`0.027578`] | `0.716608 ± 0.002869` [`0.004057`] | `-0.003602 ± 0.022369` [`0.031635`] |
| **Macro Precision**| `0.766771 ± 0.014058` [`0.019881`] | `0.769735 ± 0.000737` [`0.001042`] | `+0.002964 ± 0.013321` [`0.018839`] |
| **Macro Recall** | `0.846503 ± 0.002574` [`0.003640`] | `0.857832 ± 0.006744` [`0.009537`] | `+0.011329 ± 0.004170` [`0.005897`] |
| **Macro IoU** | `0.653449 ± 0.010416` [`0.014730`] | `0.663109 ± 0.004516` [`0.006386`] | `+0.009660 ± 0.014931` [`0.021116`] |
| **Pooled Precision**| `0.825309 ± 0.009256` [`0.013090`] | `0.827688 ± 0.000028` [`0.000040`] | `+0.002379 ± 0.009284` [`0.013130`] |
| **Pooled Recall** | `0.765119 ± 0.000679` [`0.000960`] | `0.786370 ± 0.011772` [`0.016648`] | `+0.021251 ± 0.011093` [`0.015688`] |
| **Pooled IoU** | `0.658467 ± 0.005390` [`0.007622`] | `0.675735 ± 0.008712` [`0.012321`] | `+0.017267 ± 0.014102` [`0.019943`] |

---

## 5. Scientific Interpretation & Invariant Rules

### 5.1 Analysis of the Slope Effect
1. **Seed 0 Favored Slope [FACT]:** In seed 0, incorporating DEM slope increased event-macro F1 from 0.781741 to 0.797057 (+1.53 pp) and worst-event F1 from 0.706420 to 0.718636 (+1.22 pp).
2. **Seed 1 Favored Radar Only on Macro & Worst-Event F1 [FACT]:** In seed 1, radar only (Model A) achieved higher event-macro F1 (0.793257 vs 0.792165, a +0.11 pp advantage for Model A) and substantially higher worst-event F1 (0.733998 vs 0.714579, a +1.94 pp advantage for Model A).
3. **Slope Has Not Yet Shown Stable Event-Macro or Worst-Event Improvement [INTERPRETATION]:** Because the sign of the difference flips on the primary event-macro metric ($\Delta_{\text{macro F1}} = +0.015316$ in Seed 0 vs $-0.001092$ in Seed 1) and reverses substantially on the worst-event metric ($\Delta_{\text{worst F1}} = +0.012216$ in Seed 0 vs $-0.019419$ in Seed 1), the topographic slope feature has not yet shown a stable event-macro or worst-event improvement across seeds.
4. **Two-Seed Sample Inadequacy [INTERPRETATION]:** A sample of $N=2$ seeds is preliminary and statistically insufficient to support strong general robustness claims. The paired sample standard deviation ($\pm 0.011602$) exceeds the mean paired difference ($+0.007112$).

### 5.2 Mandatory Safeguards & Scope Invariants
- **Quarantine Invariant:** Neither Model A nor Model B may be declared the "better" or "selected" recipe on the basis of TTA-validation (`[279, 437, 520, 559, 1111003, 1111008]`) or sealed test (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) metrics. The quarantine remains absolute.
- **Pre-Registration of Seed 2:** Executing seed 2 is scientifically justified **strictly because seed 2 was preregistered in the immutable replication lock manifest (`source_training_week3_replication_lock.json`) before seed 1 completed**, and physical GPU compute resources remain available—**not** because an outcome favoring slope is being sought.
- **Reference Checkpoint Status:** The official KuroSiwo three-channel baseline checkpoint remains an external reference point, not an official internal reproduction.
- **Non-Exaggeration Rule:** Neither the success nor failure of the overall FYP thesis is established by these preliminary two-seed source training comparisons. The FYP investigation proceeds according to pre-registered empirical protocols.

---

## 6. Cryptographic Artifacts & File Inventory

All artifacts are preserved on disk. No generated file was manually edited [FACT].

### 6.1 Artifact Manifest Table
| Relative File Path | SHA-256 Checksum | File Size (Bytes) | Nature of Artifact |
| :--- | :--- | :---: | :--- |
| `docs/week3_source_replication_protocol.md` | `4df416c524f2bc2d5ca38e2df80705ef4a46ac263ee9c2e52b07e61e46efa927` | 9,667 | Frozen Protocol |
| `configs/source/source_training_week3_replication_lock.json` | `eb25a8f79406f6684f94fc30600e2b6833b9bd212dc4f3f1a3f8d9fc11a5d677` | 5,119 | Immutable Lock |
| `configs/source/snunet_vvvh_seed1_v2_fp32.json` | `fdcad256e5710588b193dadd902ab2442630c1c8f494c43bf424c4087ae8dccf` | 5,314 | Model A Config |
| `configs/source/snunet_vvvh_slope_seed1_v2_fp32.json` | `4845bcca17962aed26a4231b2a692d1d10834e8c548c5e9802759ec2e24b6f78` | 7,016 | Model B Config |
| `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` | 4,252 | Frozen Data Split |
| `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | 1,574 | Frozen Slope Stats |
| **Model A Seed 1** | | | |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/best_checkpoint.pt` | `985a766f8ddcfd14fdaf5ade6bc6257f7e21f5ffac2483f4d8ea95536d02ed5d` | 144,672,493 | Checkpoint (Ep 32) |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/last_checkpoint.pt` | `2b1cd1b365e8bab36a77f23519b2e33e9ab57455bb9eb5488485766d2de91c70` | 144,672,493 | Checkpoint (Ep 52) |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/checkpoint_hashes.json` | `0b5df4c5b982322693a82a3aa72d010e111d110b1dd2938be487f2dc99ed624d` | 330 | JSON Hashes |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/run_manifest.json` | `6ce13cf57e13826119880e0af131db70ad0e376215f1c4364e509b3a9ffa083c` | 4,079 | Execution Manifest |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/access_trace.json` | `fe7dd71cfa86e46a030ed659d459999ba13ab5bdcde1731299c091e9ced1a071` | 1,051 | Access Trace |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/train_history.csv` | `165ae002e936b571f2d1d0c666a9019872fc1cfd1002270f5af25f8051d951fd` | 3,857 | Train History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/validation_history.csv` | `bd08eddf66f7281b21e26411f24ccba0d04d5011dceaaa6e8a46c7ebd33d8f71` | 8,144 | Val History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/event_validation_metrics.csv` | `3dca1eec5cb97817667494e913a8f2473a197cb977fb9fac489c75ea0d954c8f` | 27,155 | Event Metrics CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/train.log` | `d57673ff1cf0a945d3fcbb6eaa6babe3c9993dfaf7f3567b2d18200c319a5ca0` | 76,891 | Console / Run Log |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/loss_curve.png` | `f450f1512093a564f8e08dfe69476e13945a09b7dd19107a42c8a5ab5627e052` | 87,107 | Loss Curve PNG |
| `outputs/week3_source_replication/source_snunet_vvvh_seed1_v3_fp32/f1_curve.png` | `ba2bf0de34f5a910f693ea569dbd6b525ee56db88a14f910d74d58e26761e3e4` | 106,556 | F1 Curve PNG |
| **Model B Seed 1** | | | |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/best_checkpoint.pt` | `b20dd72232e9022027d75f616748af8376d32dd8d5b83d039199e393bdaaeeef` | 144,676,013 | Checkpoint (Ep 37) |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/last_checkpoint.pt` | `a4095e161b6461a50b749a73e26ac1a403cd25a8e3abb4b450a04113ebf592e6` | 144,676,013 | Checkpoint (Ep 57) |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/checkpoint_hashes.json` | `c7b3a0a76ec3b6bd73fa51095525f4f9107f3d0033a13a038f026fd0ac9b430b` | 330 | JSON Hashes |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/run_manifest.json` | `a2c7132d3928d15df7494229bd5463ddcb8ea84661cad6d2ed29ba8b76c43145` | 4,251 | Execution Manifest |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/access_trace.json` | `e137b07d4e307047b0937c612c65193dac3dd1dbdcd785df5110af33d74cb65e` | 1,051 | Access Trace |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/train_history.csv` | `b63cb8e1fa0c293b451af4a6bc4e48e1df05588ab65a14d6084b863d1fbb8cd7` | 4,217 | Train History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/validation_history.csv` | `6e7c2fea9618a0adeb9ac5454479ab3b1218e907425520f65ea4026699c38d9d` | 8,902 | Val History CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/event_validation_metrics.csv` | `d245fd3e998eba1232a91c16b988ca137e189389df52286a0097ffef6cf82f0a` | 29,746 | Event Metrics CSV |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/train.log` | `d8b4de531f51995c20c4d1cef3be3b1a39277e538ad7dabe9c17806642921eeb` | 84,310 | Console / Run Log |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/loss_curve.png` | `dadb56968690b3f9ab40a98c3e6970d073590ab1b6cc8baf2845b42e47c0513c` | 90,482 | Loss Curve PNG |
| `outputs/week3_source_replication/source_snunet_vvvh_slope_seed1_v3_fp32/f1_curve.png` | `1725a9d80324fcc34d854ede113d65a1d2df3b2766ad65ce6ded1dca0ec91589` | 112,564 | F1 Curve PNG |
| **Two-Seed Comparison Artifacts** | | | |
| `outputs/week3_source_replication/comparison_seed0_seed1/source_seed0_seed1_comparison.csv` | `b5350e8a6416127c95850b3a5dfadb2e844f19790eaeadfdcccc56dbc63964d2` | 1,349 | Master Comparison CSV |
| `outputs/week3_source_replication/comparison_seed0_seed1/paired_differences.csv` | `a4d3812ab9eaa907b60eac9b3a31d5c5b40e13b919927ccc97932d7667c779d1` | 1,361 | Paired Diff CSV |
| `outputs/week3_source_replication/comparison_seed0_seed1/comparison_summary.json` | `91f00ee92b307e3972697d51a787617b7b1f7f9387f21361ac6e52976519711f` | 10,321 | Aggregated Stats JSON |
| **Parent Seed 0 Canonical Checkpoints** | | | |
| `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/best_checkpoint.pt` | `750360fca8fdbef0ecfda8529bfdc3d815f93dd04686e41c10b928087b79ca9e` | 144,672,429 | Parent Checkpoint (Ep 8) |
| `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2/best_checkpoint.pt` | `321e80a07a2af2e37e624ca97808efec3856488be6f71318f4b6b53f2cf3d57a` | 144,676,013 | Parent Checkpoint (Ep 15) |

---

## 7. Execution Commands & Provenance

### 7.1 Commands Executed During Verification & Comparison
1. Process and GPU verification:
   ```bash
   tmux ls
   ps aux | grep -E "train_source_model|cg_w3"
   nvidia-smi
   ps -p 3587447 -o user:20,pid,%cpu,%mem,start,time,command
   ```
2. Checkpoint SHA-256 verification and PyTorch deserialization test:
   ```bash
   python -c "
   import hashlib, json, torch
   from pathlib import Path
   # verified on best_checkpoint.pt and last_checkpoint.pt for Model A and B
   "
   ```
3. Event isolation and access trace audit:
   ```bash
   python -c "
   import json
   from pathlib import Path
   # verified train_events, val_events, zero TTA-val, zero sealed-test
   "
   ```
4. Machine generation of comparison summaries:
   ```bash
   python -c "
   # generated source_seed0_seed1_comparison.csv, paired_differences.csv, comparison_summary.json
   "
   ```
5. File hashing of all generated artifacts via SHA-256:
   ```bash
   python -c "
   import hashlib
   from pathlib import Path
   # hashed all 27 artifacts across docs, configs, outputs
   "
   ```

### 7.2 Git Status & Working Tree
```
On branch main
nothing to commit, working tree clean
```
Committed HEAD: `c9fcaa5d5380fdafed40d7239e05369ecf04b478` (`c9fcaa5`).

### 7.3 Bounded Next Step
The immediate, bounded next step is to execute the **pre-registered seed-2 replication runs**:
- Model A: `configs/source/snunet_vvvh_seed2_v2_fp32.json` (Seed 2, Run ID: `source_snunet_vvvh_seed2_v3_fp32`)
- Model B: `configs/source/snunet_vvvh_slope_seed2_v2_fp32.json` (Seed 2, Run ID: `source_snunet_vvvh_slope_seed2_v3_fp32`)
These runs will complete the pre-registered three-seed replication set ($N=3$), enabling a more reliable three-seed preliminary comparison before any test-time adaptation work begins.
