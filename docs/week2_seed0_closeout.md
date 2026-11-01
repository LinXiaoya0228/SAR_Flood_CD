# Week 2 Seed-0 Canonical Source Training Closeout Report

**Date:** 2026-10-10  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Git Branch & HEAD:** `main` at `f922a132b466f27648ecd44262a09035b90d85f1` (`f922a13`)  
**Git Status:** Clean (`git status --porcelain` is empty)  
**Governing Protocol:** `docs/source_training_protocol_v2.md` (SHA-256: `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144`)  
**Immutable Lock:** `configs/source/source_training_v2_lock.json` (SHA-256: `7139dbdd2031fd5654b4414299bf80e20295ebdba13bb507b64a59f10a50fbd3`)  

---

## 1. Epistemological Categorization of Findings

To ensure scientific integrity and eliminate ambiguity, every statement in this document belongs to one of four epistemological categories:

1. **Directly Verified Facts [FACT]:** Observations read directly from immutable disk artifacts (manifests, logs, CSVs, binaries, git commit tree).
2. **Calculations from Verified Facts [CALC]:** Mathematical transformations, differences, or metrics computed strictly from verified facts.
3. **Unavailable Information [UNAVAILABLE]:** Attributes or conclusions that cannot be observed or established from preserved evidence.
4. **Preliminary Interpretations [INTERPRETATION]:** Working scientific inferences subject to explicit caveats and future validation across multiple seeds.

---

## 2. Independent Run Completion Verification

Both canonical runs completed through uncorrupted early stopping under Protocol v2 (FP32).

### 2.1 Verification Matrix

| Verification Item | Model A: Radar Only `[VV, VH]` | Model B: Radar + Slope `[VV, VH, slope]` | Status |
| :--- | :--- | :--- | :--- |
| **Run ID** | `source_snunet_vvvh_seed0_v3_fp32` | `source_snunet_vvvh_slope_seed0_v3_fp32` | Verified [FACT] |
| **Output Directory** | `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2` | `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2` | Verified [FACT] |
| **Manifest Completion Status** | `COMPLETED_EARLY_STOP` | `COMPLETED_EARLY_STOP` | Verified [FACT] |
| **Absence of Failure Manifest** | `failure_manifest.json` does NOT exist | `failure_manifest.json` does NOT exist | Verified [FACT] |
| **Final Epoch** | Epoch 28 | Epoch 35 | Verified [FACT] |
| **Best Epoch** | Epoch 8 | Epoch 15 | Verified [FACT] |
| **Best Inner-Val Macro F1** | `0.781741` | `0.797057` | Verified [FACT] |
| **Early-Stopping Trigger** | Patience 20/20 exhausted at Epoch 28 | Patience 20/20 exhausted at Epoch 35 | Verified [FACT] |
| **Best Checkpoint Presence** | `best_checkpoint.pt` (144,672,429 bytes) | `best_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Last Checkpoint Presence** | `last_checkpoint.pt` (144,672,429 bytes) | `last_checkpoint.pt` (144,676,013 bytes) | Verified [FACT] |
| **Training Curve Integrity** | All recorded values finite (no NaN/Inf); overall convergence trend (strict monotonic decrease not required or established) | All recorded values finite (no NaN/Inf); overall convergence trend (strict monotonic decrease not required or established) | Verified [FACT] |
| **Validation Curve Integrity** | All recorded values finite (no NaN/Inf); overall convergence trend across all 28 epochs (strict monotonic decrease not required or established) | All recorded values finite (no NaN/Inf); overall convergence trend across all 35 epochs (strict monotonic decrease not required or established) | Verified [FACT] |
| **Expected Train Samples** | 19,178 samples (100% matched) | 19,178 samples (100% matched) | Verified [FACT] |
| **Expected Val Samples** | 2,284 samples (100% matched) | 2,284 samples (100% matched) | Verified [FACT] |
| **Accessed Train Events** | 22 events (disjoint from val) | 22 events (disjoint from val) | Verified [FACT] |
| **Accessed Val Events** | 5 events (`[267, 275, 497, 518, 1111005]`) | 5 events (`[267, 275, 497, 518, 1111005]`) | Verified [FACT] |
| **Quarantined TTA-Val Access**| 0 events accessed (`no_tta_val_accessed: true`) | 0 events accessed (`no_tta_val_accessed: true`) | Verified [FACT] |
| **Sealed-Test Access** | 0 events accessed (`no_sealed_test_accessed: true`) | 0 events accessed (`no_sealed_test_accessed: true`) | Verified [FACT] |

### 2.2 Event IDs Verification
- **Accessed Train Events (22):** `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]` [FACT].
- **Accessed Inner-Validation Events (5):** `[267, 275, 497, 518, 1111005]` [FACT].
- **Quarantined TTA-Validation Events (6 - Zero Access):** `[279, 437, 520, 559, 1111003, 1111008]` [FACT].
- **Sealed Test Events (10 - Zero Access):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` [FACT].
- Both runs verified `train_val_events_disjoint: true` and `train_val_samples_disjoint: true` [FACT].

### 2.3 Training and Validation Curve Characteristics
- **Finite Numerical Values:** All recorded train and validation values were finite [FACT].
- **No Nonfinite Anomalies:** No NaN/Inf was observed [FACT].
- **Overall Convergence Trend:** The curves showed an overall convergence trend [FACT].
- **Strict Monotonicity:** Strict monotonic decrease was not required or established [FACT].

---

## 3. Provenance & Cryptographic Verification

### 3.1 Environment & Git Provenance
- **Active Git Commit:** `f922a132b466f27648ecd44262a09035b90d85f1` [FACT].
- **Recorded Git Commit in Manifests:** `f922a132b466f27648ecd44262a09035b90d85f1` (both runs match exact committed HEAD) [FACT].
- **Recorded `git_dirty` State:** `false` (both runs launched from clean working tree) [FACT].
- **Python Environment:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` [FACT].
- **Hardware Device:** NVIDIA GeForce RTX 3090 (Model A on physical GPU 1, Model B on physical GPU 2) [FACT].

### 3.2 Protocol, Lock, and Configuration Hashes
Every configuration and dependency hash was validated against the one-way immutable lock:

| Component | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Governing Protocol v2** | `docs/source_training_protocol_v2.md` | `b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144` |
| **Immutable Lock Manifest** | `configs/source/source_training_v2_lock.json` | `7139dbdd2031fd5654b4414299bf80e20295ebdba13bb507b64a59f10a50fbd3` |
| **Model A Config** | `configs/source/snunet_vvvh_v2_fp32.json` | `528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3` |
| **Model B Config** | `configs/source/snunet_vvvh_slope_v2_fp32.json` | `e99e980a9afc0f72b526ccff5484e4aadaabece12c5a542bd20f94be74ef6b56` |
| **Frozen Split Manifest** | `configs/splits/kuro_source_inner_split_v1.json` | `7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3` |
| **Slope Normalization Stats**| `artifacts/week1/source_train_slope_statistics.json` | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` |

### 3.3 Checkpoint & Artifact Hashes

#### Model A (`source_snunet_vvvh_seed0_v3_fp32_attempt2`)
- `best_checkpoint.pt` (Epoch 8): `750360fca8fdbef0ecfda8529bfdc3d815f93dd04686e41c10b928087b79ca9e` [FACT]
- `last_checkpoint.pt` (Epoch 28): `c685f3c753b13e3db4ffa25c033c274ce22fa0e69c9f6a1979ce82a77faae2b8` [FACT]
- `run_manifest.json`: `3597700f0ecc698211d067f12199a935a4cd2544c3fded8cea8fec203c1ff336` [FACT]
- `checkpoint_hashes.json`: `f4475062c43d5b7ae8ed9f9b12ff6f40dff066869d097d72884e6545807bd06d` [FACT]
- `access_trace.json`: `fe7dd71cfa86e46a030ed659d459999ba13ab5bdcde1731299c091e9ced1a071` [FACT]
- `train_history.csv`: `23878b35fde2f108bfb6f5c61e088ba5224206707e3e9134a2c0b69d495d7ea6` [FACT]
- `validation_history.csv`: `8950722725a9623590a8bd2979b411563a705cae13a4fa77b591d29cb490fd29` [FACT]
- `event_validation_metrics.csv`: `221fac54da6ba4a6cfa4c26730e6166ad75252bcd7a2628b45f9b087714030ae` [FACT]
- `train.log`: `916c62712a24adea19f945a014b0b2e792a576be32d2285e22e9da04f41cbaff` [FACT]
- `loss_curve.png`: `1bb6528d3ef9bfb564d6b34ee58523a1421040a6bfb5b72a90a3d3ec599332e7` [FACT]
- `f1_curve.png`: `b840def2755f354f93f8fd8790c38d91ae310262a670fdb4ffb3d98e32f5a279` [FACT]

#### Model B (`source_snunet_vvvh_slope_seed0_v3_fp32_attempt2`)
- `best_checkpoint.pt` (Epoch 15): `321e80a07a2af2e37e624ca97808efec3856488be6f71318f4b6b53f2cf3d57a` [FACT]
- `last_checkpoint.pt` (Epoch 35): `4b9292cdd7f7f76acbfe4a35cb9f3798ec0daee39b563b7a3cdee90cfd4e4d60` [FACT]
- `run_manifest.json`: `6a678f110b26301db347b0b0301d830fd389caba9944b6cd46fdf0901362abcb` [FACT]
- `checkpoint_hashes.json`: `5d0efe03d5ef42efc9e5fcd552496802f20f86e56ce5863094841e85e3338be4` [FACT]
- `access_trace.json`: `e137b07d4e307047b0937c612c65193dac3dd1dbdcd785df5110af33d74cb65e` [FACT]
- `train_history.csv`: `1c102afe19c89ce4a84d2e0a2930ccf6d540795ef0b04002f7b2aae14be7ebfc` [FACT]
- `validation_history.csv`: `e81baa741c16c72942d87b1d1e275c8725970e1eafd5da619d0a9c3020c0af20` [FACT]
- `event_validation_metrics.csv`: `a2b3fcef4a78d2b2acf2e648c6a90ba047e459496d51518d88a17a820e5e0286` [FACT]
- `train.log`: `1a10ddc2ef6b4fad966a76aa4d83133db4fdf60d8d2569c6b624c014ed92beb0` [FACT]
- `loss_curve.png`: `f5bd38cdcd7f26e15fd109b8cf38d5b05773283be2a7a2493b35afad8fa31962` [FACT]
- `f1_curve.png`: `cd63b48c6b8ad12846cde51efd9e1148f7d9820236b2c342b3080d101442ef6e` [FACT]

### 3.4 Config Equivalence Assertion
A programmatic field-by-field assertion verified that `split_specification`, `optimization`, `training_budget`, `precision`, `evaluation`, `leakage_safeguards`, and `random_seed` are **100% IDENTICAL** between both configurations [FACT]. The only difference between Model A and Model B is the inclusion of the normalized physical metric slope channel in `input_specification` (`in_channels_per_branch`: 2 vs 3) [FACT].

---

## 4. Preliminary Seed-0 Scientific Comparison

The metrics below are extracted from the machine-generated comparison table `outputs/week2_source_seed0/week2_seed0_comparison.csv` (SHA-256: `fd25a7ea21a3b9d6f46f148084a75cda0bc0933d86e38566359b322760e0de92`).

### 4.1 Summary Metric Comparison

| Metric | Model A (Radar Only) | Model B (Radar + Slope) | Difference (Model B - Model A) | Epistemological Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Channels** | `[VV, VH]` (2 channels) | `[VV, VH, slope]` (3 channels) | +1 topographic channel | [FACT] |
| **Seed** | `0` | `0` | `0` (matched seed) | [FACT] |
| **Final Epoch** | 28 | 35 | +7 epochs | [FACT] / [CALC] |
| **Best Epoch** | 8 | 15 | +7 epochs | [FACT] / [CALC] |
| **Early Stopping Reason** | Patience 20/20 at Ep 28 | Patience 20/20 at Ep 35 | Matched criterion | [FACT] |
| **Best Inner-Val Macro F1** | `0.781741` | `0.797057` | **+0.015316** (+1.53 pp) | [FACT] / [CALC] |
| **Best Inner-Val Pooled F1** | `0.791289` | `0.810865` | **+0.019576** (+1.96 pp) | [FACT] / [CALC] |
| **Worst-Event ID** | Event 267 | Event 267 | Same hardest event | [FACT] |
| **Worst-Event F1** | `0.706420` | `0.718636` | **+0.012216** (+1.22 pp) | [FACT] / [CALC] |
| **Validation Loss (Total)** | `0.202707` | `0.158304` | **-0.044403** (-21.9%) | [FACT] / [CALC] |
| **Validation CE Loss** | `0.251363` | `0.197413` | **-0.053950** (-21.5%) | [FACT] / [CALC] |
| **Validation Dice Loss** | `0.154050` | `0.119196` | **-0.034854** (-22.6%) | [FACT] / [CALC] |
| **Inner-Val Macro Precision**| `0.756831` | `0.769214` | **+0.012383** | [FACT] / [CALC] |
| **Inner-Val Macro Recall** | `0.848323` | `0.862601` | **+0.014278** | [FACT] / [CALC] |
| **Inner-Val Macro IoU** | `0.646084` | `0.666302` | **+0.020218** | [FACT] / [CALC] |
| **Inner-Val Pooled Precision**| `0.818764` | `0.827708` | **+0.008944** | [FACT] / [CALC] |
| **Inner-Val Pooled Recall** | `0.765599` | `0.794694` | **+0.029095** | [FACT] / [CALC] |
| **Inner-Val Pooled IoU** | `0.654656` | `0.681895` | **+0.027239** | [FACT] / [CALC] |

### 4.2 Per-Event Validation Metrics at Selected Checkpoints

Inner validation comprises 5 disjoint events (`2,284` samples, `112,243,176` valid pixels):

| Event ID | Samples | Model A F1 (Ep 8) | Model B F1 (Ep 15) | F1 Difference | Model A IoU | Model B IoU | IoU Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **267** | 218 | `0.706420` | `0.718636` | **+0.012216** | `0.546097` | `0.560836` | **+0.014739** |
| **275** | 742 | `0.864562` | `0.868826` | **+0.004264** | `0.761435` | `0.768075` | **+0.006640** |
| **497** | 398 | `0.838762` | `0.840600` | **+0.001838** | `0.722300` | `0.725031` | **+0.002731** |
| **518** | 306 | `0.718903` | `0.744195` | **+0.025292** | `0.561162` | `0.592604` | **+0.031442** |
| **1111005** | 620 | `0.780060` | `0.813030` | **+0.032970** | `0.639425` | `0.684963` | **+0.045538** |
| **Macro** | **2,284** | **0.781741** | **0.797057** | **+0.015316** | **0.646084** | **0.666302** | **+0.020218** |

### 4.3 Essential Scientific Limitations & Protocol Statements

> [!WARNING]
> **STATISTICAL POWER LIMITATION ($N=1$ SEED):**  
> One random seed is strictly insufficient to establish that slope provides a statistically significant, stable, or generalizable improvement. The observed $+0.0153$ Macro F1 and $+0.0196$ Pooled F1 improvements on seed 0 represent preliminary single-seed evidence [INTERPRETATION]. Multi-seed replication (e.g. seeds 1 and 2 in future work) is required to determine confidence bounds.

> [!IMPORTANT]
> **ZERO TTA-VALIDATION CONTAMINATION:**  
> Neither Model A nor Model B accessed quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) or sealed-test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`). TTA-validation metrics were **NOT** computed, were **NOT** inspected, and were **NOT** used to compare or select the source model recipe [FACT].

> [!NOTE]
> **ABLATION STATUS & OFFICIAL THIRD CHANNEL:**  
> Model B is a controlled scientific ablation incorporating physical terrain gradient. Model B does **NOT** claim that slope was the third channel of the original KuroSiwo official pretrained checkpoint. The official third channel remains completely UNKNOWN [FACT].

---

## 5. Provenance of Historical Attempts in Week 2

To maintain full project provenance, the four stages of Week 2 source-model development are recorded below:

1. **Attempt 1: Canonical v1/v2 FP16/AMP Runs (Superceded & Preserved)**
   - Paths:
     - `outputs/week2_source_seed0/source_snunet_vvvh_seed0/`
     - `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0/`
   - Outcome: Model B experienced gradient overflow and NaN loss at Epoch 24. Later epochs developed numerical corruption. Both runs were disqualified as canonical source baselines. Their Epoch 20 checkpoints are preserved as noncanonical diagnostic artifacts only.
2. **Investigation: Numerical Diagnostic Experiments (Completed)**
   - Paths:
     - `outputs/week2_numerical_diagnostics/model_b_diagnostic_amp_20261008/`
     - `outputs/week2_numerical_diagnostics/model_b_diagnostic_fp32_retry_20261008/`
   - Outcome: Demonstrated that FP16/AMP backward gradients overflowed at Epoch 23, Batch 353 while forward activations were finite. In contrast, matched FP32 completed 11,990 batches through Epoch 25 with 0 NaN/Inf. Protocol v2 formally mandated FP32 for all canonical source runs.
3. **Attempt 2: Canonical v3 FP32 Attempt 1 (Software Failure - Preserved)**
   - Paths:
     - `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32/`
     - `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32/`
   - Outcome: Both runs computed 2,398 Epoch 1 training batches stably in FP32 with finite loss, but crashed during Epoch 1 checkpoint finalization due to an unpack bug (`NameError: name 'split_hash' is not defined` at `tools/train_source_model.py:745`). Software fix committed at `f922a13` with 16 automated safeguards.
4. **Attempt 3: Canonical v3 FP32 Attempt 2 (Completed Canonical Source Models)**
   - Paths:
     - `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v3_fp32_attempt2/`
     - `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v3_fp32_attempt2/`
   - Outcome: Both runs trained from scratch at seed 0 without interruption, cleanly terminating at Epochs 28 and 35 via patience-20 early stopping with uncorrupted best checkpoints.

---

## 6. Week 2 Completion Criteria Assessment & Readiness for Week 3

| Criterion | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **C1: Controlled Training** | Exact protocol match, frozen splits, FP32 numerical stability | Verified 100% finite across all epochs and batches | **PASSED** |
| **C2: Provenance Integrity** | Git commit recorded, clean working tree, lock SHA validated | Manifests match committed HEAD `f922a13`, `git_dirty: false` | **PASSED** |
| **C3: Zero Leakage** | Zero TTA-val or test events accessed | Checked in manifests and access traces | **PASSED** |
| **C4: Artifact Persistence** | Nonzero best & last checkpoints, full histories, curve plots | All artifacts preserved and verified on disk | **PASSED** |
| **C5: Objective Comparison** | Event-macro F1 comparison from preserved records only | Documented in `week2_seed0_comparison.csv` and report | **PASSED** |

### Scope of Week 3
Week 3 is:
- seed-1 source-training replication;
- preliminary multi-seed `[VV,VH]` versus `[VV,VH,slope]` comparison;
- no TTA.

### Remaining Blockers Before Week 3
- No scientific blocker remains;
- The Week 2 closeout document must be committed as a clean Git checkpoint before Week 3 is launched.

