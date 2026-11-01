# Week 2 Seed-0 Numerical-Validity Audit Report

**Date:** 2026-10-07  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22`  
**Git HEAD:** `5ca642ac92036e95f85bb447101ffe6b6e802d2c` (`main`, clean)  
**Status:** **DEFECT CONFIRMED (NUMERICAL INSTABILITY IN AMP / BATCHNORM)**  
**Executive Conclusion:** Week 2 formal source training is **NOT COMPLETE**. A corrected `v3` rerun with numerical guards and AMP stability protections is strictly required.

---

## 1. Executive Summary & Critical Distinctions

This independent numerical audit investigated the canonical Week 2 seed-0 runs:
- **Model A:** `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v2/`
- **Model B:** `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v2/`

Both training processes terminated early at Epoch 40 via the early stopping counter reaching patience (20 non-improving epochs following Epoch 20), writing `completion_status: "COMPLETED"` into their run manifests. However, numerical auditing revealed that training suffered catastrophic numerical collapse after Epoch 20:

| Dimension | Assessment | Operational Meaning |
| :--- | :--- | :--- |
| **Program Completion** | **Terminated (Normal Exit)** | Launcher process reached early-stopping threshold (patience 20) and exited with code 0. |
| **Numerical Validity** | **FAILED (Catastrophic Collapse)** | Nonfinite values (`NaN`/`Inf`) entered BatchNorm running statistics at Epoch 25 (Model A) and Epoch 24 (Model B). Post-collapse epochs produced NaN validation losses and 0 predicted flood pixels (serving as a clear collapse indicator on this known-positive inner-validation set). |
| **Checkpoint Validity** | **PARTIAL** | **`best_checkpoint.pt` (Epoch 20) is 100% finite, intact, and valid** across both models.<br>**`last_checkpoint.pt` (Epoch 40) is severely corrupted** with nonfinite BatchNorm running statistics and is **completely unusable**. |
| **Reproducibility** | **VERIFIED (Bit-level)** | Independent evaluation of both `best_checkpoint.pt` files reproduced the stored Epoch 20 metrics with exact bit-level pixel match (`Diff: 0`). |
| **Scientific Usability** | **PROVISIONAL / INCOMPLETE** | Stored best Macro F1 values (`0.7925` for Model A, `0.7857` for Model B at Epoch 20) are mathematically genuine snapshots, but later collapsed epochs cannot be counted as training. Seed 0 alone cannot establish scientific superiority, and the run must be rerun with numerical safeguards. |

### Explicit Declarations
1. **Model A stored best Macro F1:** `0.792522` (~`0.7925`) achieved at **Epoch 20**.
2. **Model B stored best Macro F1:** `0.785700` (~`0.7857`) achieved at **Epoch 20**.
3. These values were provisional in the logs and are now **independently verified as mathematically genuine representations of the model weights at Epoch 20**.
4. The later zero/NaN epochs (Epochs 25–40 in Model A, Epochs 24–40 in Model B) **cannot be treated as successful or legitimate training**.
5. **Seed 0 alone cannot establish superiority** of Model A over Model B or vice versa.

---

## 2. Preserved Evidence & Pre-Audit Verification

Before conducting any diagnostics or evaluations, all files in both canonical `v2` directories were inventoried, hashed (SHA-256), and compared with their respective `checkpoint_hashes.json`. All files remained strictly untouched and unmodified.

### Preserved Artifact Inventory

#### Model A (`source_snunet_vvvh_seed0_v2`)
- Output Path: [`outputs/week2_source_seed0/source_snunet_vvvh_seed0_v2`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week2_source_seed0/source_snunet_vvvh_seed0_v2)
- `access_trace.json`: `fe7dd71cfa86e46a030ed659d459999ba13ab5bdcde1731299c091e9ced1a071` (1,051 bytes)
- `best_checkpoint.pt`: `a30abc59e6026960593b378c588f5ce8021f05ed4b944464f10ab2ed5fb8fb18` (144,667,399 bytes)
  - Matches `checkpoint_hashes.json`: **MATCHED**
- `last_checkpoint.pt`: `eb699487c8feb587489e1ca666fd609bd0e7c41c8f2798ed8e4136957af61e5a` (144,667,399 bytes)
  - Matches `checkpoint_hashes.json`: **MATCHED**
- `train_history.csv`: `b0523aae91dd0ee12722e542c3de6c565a698af222642efaba71033396b44baa` (2,978 bytes)
- `validation_history.csv`: `62e4259efaaccd523f653fd293e0510aee6293cf0caa3da278ec01d58e041add` (5,874 bytes)
- `event_validation_metrics.csv`: `6e7ba64da5ce1c92ec896d882220e2477cc81228f1fe6cb405d46549f7b009bb` (19,658 bytes)
- `train.log`: `0a08867a0a0080cef36c24b1ce38fb7340eeb8705f0c417e9ca84948279f7fac` (76,580 bytes)
- `run_manifest.json`: `afa818f963d11019314dfe9229afc91d93ee4fe79f601b72c8d4807bd1d0b7d6` (3,499 bytes)

#### Model B (`source_snunet_vvvh_slope_seed0_v2`)
- Output Path: [`outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v2`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v2)
- `access_trace.json`: `e137b07d4e307047b0937c612c65193dac3dd1dbdcd785df5110af33d74cb65e` (1,051 bytes)
- `best_checkpoint.pt`: `3eb8f75a95b6b7a0e8e47e6e5ca0b075fb70f3aca0244c2c25d786ae15512856` (144,670,855 bytes)
  - Matches `checkpoint_hashes.json`: **MATCHED**
- `last_checkpoint.pt`: `d365c5a40cad562dcb6735c99467c3740ddfb3d2a9a09625100f0b2c4b233680` (144,670,855 bytes)
  - Matches `checkpoint_hashes.json`: **MATCHED**
- `train_history.csv`: `c4e74ac4de615bfd71d5837912537a0f6e2eb8d166a3c98d5a02dd63ad98daa3` (2,783 bytes)
- `validation_history.csv`: `35211442884321993f559e8a68e00d6254fa72ba10a6038d24255cdbf671c599` (5,850 bytes)
- `event_validation_metrics.csv`: `0b453b8983b337984aeb7c2aefbc9aef98be1d1111d977a65de7235c02d52aa9` (19,576 bytes)
- `train.log`: `5ba078b03c96c453d5f59233536ef4bf3d14377341bcd7d25a973d5afaf849cc` (75,463 bytes)
- `run_manifest.json`: `5ca46ea0cb90a1e15a5f98ff2885223d45b92a3b1c75387eea0cc230c3d30568` (3,571 bytes)

Machine-readable evidence catalog saved to: [`artifacts/week2_seed0_numerical_audit/v2_preserved_evidence.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/v2_preserved_evidence.json).

### Noncanonical v1 Attempts Confirmation
The two historical noncanonical `v1` directories remained completely untouched and excluded:
- `outputs/week2_source_seed0/source_snunet_vvvh_seed0_v1/` (mtime unchanged: 2026-10-07 14:13)
- `outputs/week2_source_seed0/source_snunet_vvvh_slope_seed0_v1/` (mtime unchanged: 2026-10-07 14:13)

---

## 3. Reconstructed Numerical Timeline

Analysis of `train_history.csv`, `validation_history.csv`, and `train.log` reveals the exact progression of the numerical failure:

### Numerical Failure Inception Points

| Metric Event | Model A (`vvvh`) | Model B (`vvvh_slope`) |
| :--- | :--- | :--- |
| **Best Valid Epoch** | **Epoch 20** (Macro F1 = `0.792522`) | **Epoch 20** (Macro F1 = `0.785700`) |
| **Last Fully Finite Epoch** | **Epoch 24** | **Epoch 23** |
| **First Nonfinite Train Total Loss** | **Epoch 25** | **Epoch 24** |
| **First Nonfinite Train CE** | **Epoch 25** | **Epoch 24** |
| **First Nonfinite Train Dice** | **Epoch 25** | **Epoch 24** |
| **First Nonfinite Validation Loss** | **Epoch 25** | **Epoch 24** |
| **First Zero Predicted Flood Pixels** | **Epoch 25** (`0` pixels) | **Epoch 24** (`0` pixels) |
| **First Zero Macro F1** | **Epoch 25** (`0.000000`) | **Epoch 24** (`0.000000`) |
| **Onset Dynamic** | **Sudden Step Function** | **Sudden Step Function** |

### Detailed Trajectory Surrounding Collapse

#### Model A Trajectory (Epochs 19–27)
```text
Epoch 19: Train Loss=0.1115 (CE=0.1198, Dice=0.1032) | Val Loss=0.2275 | Macro F1=0.7778 | Flood Pix=18,358,003
Epoch 20: Train Loss=0.1112 (CE=0.1191, Dice=0.1032) | Val Loss=0.1701 | Macro F1=0.7925 | Flood Pix=18,203,671 [BEST]
Epoch 21: Train Loss=0.1098 (CE=0.1175, Dice=0.1021) | Val Loss=0.1822 | Macro F1=0.7871 | Flood Pix=18,599,140
Epoch 22: Train Loss=0.1079 (CE=0.1162, Dice=0.0996) | Val Loss=0.2001 | Macro F1=0.7773 | Flood Pix=18,899,228
Epoch 23: Train Loss=0.1081 (CE=0.1162, Dice=0.0999) | Val Loss=0.1733 | Macro F1=0.7802 | Flood Pix=18,996,424
Epoch 24: Train Loss=0.1080 (CE=0.1156, Dice=0.1005) | Val Loss=0.1829 | Macro F1=0.7892 | Flood Pix=19,268,971
Epoch 25: Train Loss=nan    (CE=nan,    Dice=nan)    | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0 [COLLAPSE]
Epoch 26: Train Loss=0.1075 (CE=0.1150, Dice=0.1000) | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0
Epoch 27: Train Loss=0.1067 (CE=0.1138, Dice=0.0995) | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0
...
Epoch 40: Train Loss=0.1039 (CE=0.1118, Dice=0.0961) | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0 [EARLY STOP]
```

#### Model B Trajectory (Epochs 19–26)
```text
Epoch 19: Train Loss=0.1093 (CE=0.1186, Dice=0.0999) | Val Loss=0.1816 | Macro F1=0.7774 | Flood Pix=18,089,189
Epoch 20: Train Loss=0.1139 (CE=0.1230, Dice=0.1048) | Val Loss=0.1878 | Macro F1=0.7857 | Flood Pix=17,053,504 [BEST]
Epoch 21: Train Loss=0.1109 (CE=0.1190, Dice=0.1028) | Val Loss=0.1890 | Macro F1=0.7748 | Flood Pix=18,273,983
Epoch 22: Train Loss=0.1084 (CE=0.1167, Dice=0.1001) | Val Loss=0.1783 | Macro F1=0.7828 | Flood Pix=18,873,983
Epoch 23: Train Loss=0.1057 (CE=0.1136, Dice=0.0978) | Val Loss=0.1832 | Macro F1=0.7767 | Flood Pix=18,145,336
Epoch 24: Train Loss=nan    (CE=nan,    Dice=nan)    | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0 [COLLAPSE]
Epoch 25: Train Loss=0.1165 (CE=0.1254, Dice=0.1076) | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0
Epoch 26: Train Loss=0.1105 (CE=0.1184, Dice=0.1026) | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0
...
Epoch 40: Train Loss=nan    (CE=nan,    Dice=nan)    | Val Loss=nan    | Macro F1=0.0000 | Flood Pix=0 [EARLY STOP]
```

Complete timeline files saved to:
- [`artifacts/week2_seed0_numerical_audit/model_a_numerical_timeline.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/model_a_numerical_timeline.csv)
- [`artifacts/week2_seed0_numerical_audit/model_b_numerical_timeline.csv`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/model_b_numerical_timeline.csv)
- [`artifacts/week2_seed0_numerical_audit/numerical_timeline_summary.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/numerical_timeline_summary.json)

---

## 4. Checkpoint Tensor and Optimizer Integrity Audit

Every parameter, buffer, optimizer momentum state, and GradScaler state was audited for nonfinite elements across all four saved checkpoints.

### Detailed Checkpoint Audit Results

| Audit Property | Model A Best (`ep=20`) | Model A Last (`ep=40`) | Model B Best (`ep=20`) | Model B Last (`ep=40`) |
| :--- | :--- | :--- | :--- | :--- |
| **Filename** | `best_checkpoint.pt` | `last_checkpoint.pt` | `best_checkpoint.pt` | `last_checkpoint.pt` |
| **SHA-256** | `a30abc...fb8fb18` | `eb6994...af61e5a` | `3eb8f7...512856` | `d365c5...4b233680` |
| **Total Tensors** | 236 | 236 | 236 | 236 |
| **Tensors with NaN** | **0** | **8** | **0** | **11** |
| **Tensors with Inf** | **0** | **0** | **0** | **1** |
| **Nonfinite Elements** | **0** | **284** | **0** | **646** |
| **Corrupted BN Stats** | **0** | **8 (100% of nonfinite)** | **0** | **12 (100% of nonfinite)**|
| **Optimizer Nonfinite** | **0** | **0** | **0** | **0** |
| **GradScaler Scale** | `32768.0` | `16384.0` | `32768.0` | `0.0` (collapsed) |
| **Safe for Evaluation**| **YES (PROVED)** | **NO (UNUSABLE)** | **YES (PROVED)** | **NO (UNUSABLE)** |

### Exact Corrupted Tensors in `last_checkpoint.pt`
Audit revealed that **all nonfinite elements in `last_checkpoint.pt` reside exclusively in the running statistics of BatchNorm layers in the dense nested decoder**:

- **Model A `last_checkpoint.pt`:**
  - `model.conv1_3.bn1.running_mean`: 14 / 64 nonfinite elements
  - `model.conv1_3.bn1.running_var`: 14 / 64 nonfinite elements
  - `model.conv1_3.bn2.running_mean`: 64 / 64 nonfinite elements
  - `model.conv1_3.bn2.running_var`: 64 / 64 nonfinite elements
  - `model.conv0_4.bn1.running_mean`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn1.running_var`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn2.running_mean`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn2.running_var`: 32 / 32 nonfinite elements
- **Model B `last_checkpoint.pt`:**
  - `model.conv2_2.bn1.running_mean`: 3 / 128 nonfinite elements
  - `model.conv2_2.bn1.running_var`: 3 / 128 nonfinite elements
  - `model.conv2_2.bn2.running_mean`: 128 / 128 nonfinite elements
  - `model.conv2_2.bn2.running_var`: 128 / 128 nonfinite elements
  - `model.conv1_3.bn1.running_mean`: 64 / 64 nonfinite elements
  - `model.conv1_3.bn1.running_var`: 64 / 64 nonfinite elements
  - `model.conv1_3.bn2.running_mean`: 64 / 64 nonfinite elements
  - `model.conv1_3.bn2.running_var`: 64 / 64 nonfinite elements
  - `model.conv0_4.bn1.running_mean`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn1.running_var`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn2.running_mean`: 32 / 32 nonfinite elements
  - `model.conv0_4.bn2.running_var`: 32 / 32 nonfinite elements

Machine-readable integrity audit saved to: [`artifacts/week2_seed0_numerical_audit/checkpoint_integrity_audit.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/checkpoint_integrity_audit.json).

---

## 5. Independent Best-Checkpoint Re-Evaluation

Because both `best_checkpoint.pt` files passed all integrity checks (zero NaN, zero Inf, fully finite BN stats and weights), both were independently loaded in evaluation mode and re-evaluated over the 5 source-inner validation events (`[267, 275, 497, 518, 1111005]`, 2,284 patches) on GPU 1.

### Comparison: Re-Evaluated vs Stored Epoch 20 Records

#### Model A (`source_snunet_vvvh_seed0_v2`)
| Metric | Stored Epoch 20 | Re-Evaluated | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **Validation Loss Total** | `0.170089` | `0.170089` | `+3.93e-07` |
| **Validation Loss CE** | `0.208379` | `0.208379` | `+6.29e-08` |
| **Validation Loss Dice** | `0.131800` | `0.131800` | `-2.76e-07` |
| **Event-Macro Flood F1** | **`0.792522`** | **`0.792522`** | **`+1.74e-07`** |
| **Event-Macro Flood Recall** | `0.853301` | `0.853301` | `+1.06e-07` |
| **Event-Macro Flood Precision**| `0.767074` | `0.767074` | `+3.02e-07` |
| **Event-Macro Flood IoU** | `0.660419` | `0.660419` | `-3.81e-07` |
| **Pooled Flood F1** | `0.811063` | `0.811063` | `+3.68e-07` |
| **Predicted Flood Pixels** | **`18,203,671`** | **`18,203,671`** | **`0` (Exact)** |
| **Ground-Truth Flood Pixels** | **`18,946,970`** | **`18,946,970`** | **`0` (Exact)** |
| **Total Valid Pixels** | **`112,243,176`**| **`112,243,176`**| **`0` (Exact)** |

#### Model B (`source_snunet_vvvh_slope_seed0_v2`)
| Metric | Stored Epoch 20 | Re-Evaluated | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **Validation Loss Total** | `0.187795` | `0.187795` | `+2.25e-08` |
| **Validation Loss CE** | `0.237976` | `0.237976` | `+1.15e-07` |
| **Validation Loss Dice** | `0.137614` | `0.137614` | `-6.98e-08` |
| **Event-Macro Flood F1** | **`0.785700`** | **`0.785700`** | **`+1.32e-07`** |
| **Event-Macro Flood Recall** | `0.829528` | `0.829528` | `-3.50e-07` |
| **Event-Macro Flood Precision**| `0.782412` | `0.782412` | `+3.62e-07` |
| **Event-Macro Flood IoU** | `0.650800` | `0.650800` | `+1.71e-08` |
| **Pooled Flood F1** | `0.786158` | `0.786158` | `-3.51e-07` |
| **Predicted Flood Pixels** | **`17,053,504`** | **`17,053,504`** | **`0` (Exact)** |
| **Ground-Truth Flood Pixels** | **`18,946,970`** | **`18,946,970`** | **`0` (Exact)** |
| **Total Valid Pixels** | **`112,243,176`**| **`112,243,176`**| **`0` (Exact)** |

#### Event-by-Event Breakdown Comparison
| Event ID | Model A Stored F1 | Model A Re-Eval F1 | Model B Stored F1 | Model B Re-Eval F1 |
| :--- | :--- | :--- | :--- | :--- |
| **267** | `0.712306` | `0.712306` (`-4.5e-7`) | `0.711928` | `0.711928` (`-2.3e-8`) |
| **275** | `0.870620` | `0.870620` (`-4.6e-7`) | `0.868622` | `0.868622` (`-4.1e-7`) |
| **497** | `0.828859` | `0.828859` (`+5.0e-7`) | `0.835013` | `0.835013` (`+4.5e-7`) |
| **518** | `0.733111` | `0.733111` (`+3.1e-8`) | `0.748491` | `0.748491` (`+1.9e-8`) |
| **1111005** | `0.817715` | `0.817715` (`+2.6e-7`) | `0.764447` | `0.764447` (`-3.7e-7`) |

**Conclusion on Best Checkpoints:**  
All metrics reproduced identically within IEEE 754 string formatting rounding (`< 5e-7`), and all pixel confusion counts matched with exact `0` difference. The Epoch 20 best checkpoints are **independently valid and reproducible snapshots**.

Complete comparison data saved to:
- [`artifacts/week2_seed0_numerical_audit/best_checkpoints_reevaluation.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/best_checkpoints_reevaluation.json)
- [`artifacts/week2_seed0_numerical_audit/metric_reproduction_comparison.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/metric_reproduction_comparison.json)

---

## 6. Root Cause Investigation: Data, Loss, and BatchNorm Failure Mechanism

### 1. Dataset Integrity Scan (Diagnostic Only)
A complete scan was performed across all 19,178 training patches (22 events) and 2,284 validation patches (5 events).

> **Class Mapping Clarification:**  
> The KuroSiwo dataset class encoding is:  
> - `0 = No Water`  
> - `1 = Permanent Water`  
> - `2 = Flood`  
> - `3 = Invalid`  
> Any flood-presence scan must strictly evaluate `label == 2` (Flood), not `label == 1` (Permanent Water).

- **Corrected Scan Artifact:** Versioned superseding artifact [`artifacts/week2_seed0_numerical_audit/dataset_finiteness_scan_v2.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/artifacts/week2_seed0_numerical_audit/dataset_finiteness_scan_v2.json) supersedes the preliminary `train_data_finiteness_scan.json` (which preserved earlier logs).
- **Training scan scope:** 19,178 patches across 600 batches (duration: 87.91s).
  - Class distribution: 708,320,137 No Water (`0`), 67,154,836 Perm Water (`1`), 175,127,681 Flood (`2`), 11,672,674 Invalid (`3`).
  - Batches with zero flood pixels (`label == 2`): **0** (every batch contains positive flood pixels).
  - Nonfinite inputs: **0**.
  - All-invalid batches (`target == 3`): **0**.
- **Validation scan scope:** 2,284 patches across 72 batches (duration: 24.20s).
  - Class distribution: 85,612,851 No Water (`0`), 7,683,355 Perm Water (`1`), 18,946,970 Flood (`2`), 2,358,808 Invalid (`3`).
  - Total flood pixels: exactly `18,946,970`, matching `validation_history.csv` (`gt_flood_pixels: 18946970`).
  - Nonfinite inputs: **0**.
  - All-invalid batches: **0**.

The raw dataset itself contains **zero nonfinite values**. The failure is purely algorithmic/numerical.

### 2. Numerical Failure Causal Hypothesis
**Hypothesis Statement:**  
*FP16 forward activation overflow followed by BatchNorm running-statistics corruption is the strongest current hypothesis, but the first offending operation has not yet been instrumentally observed.*

The conceptual failure cascade under investigation:

```
[1] SNUNet Nested Concatenation (conv0_4 / conv1_3 / conv2_2)
    Dense concatenation of up to 6 feature maps creates high activation magnitudes.
                         ↓
[2] Potential PyTorch AMP Dynamic Range Overflow
    In float16, max representable value is 65,504.
    If accumulated feature activations in deep blocks exceed 65,504 → Inf.
                         ↓
[3] In-Place BatchNorm Statistics Corruption
    In model.train() mode, BatchNorm updates running_mean and running_var in-place during the forward pass:
        running_mean = (1 - m) * running_mean + m * batch_mean (Inf)
        running_var  = (1 - m) * running_var  + m * batch_var  (Inf - Inf = NaN)
    CRITICAL: Running statistics are buffers updated during FORWARD PASS,
    COMPLETELY OUTSIDE THE OPTIMIZER.
                         ↓
[4] GradScaler Failure to Protect BatchNorm
    GradScaler checks gradients AFTER backward pass.
    It catches Inf in gradients and skips optimizer.step().
    HOWEVER, BatchNorm buffers ALREADY contain NaNs from the forward pass!
    GradScaler does NOT revert BatchNorm buffers!
                         ↓
[5] Permanent Model Poisoning in Evaluation Mode
    In model.eval(), BatchNorm computes:
        y = (x - running_mean) / sqrt(running_var + eps)
    Because running_mean/var contain NaN, outputs become NaN.
    Softmax/argmax on NaN logits collapses to class 0 (no flood predicted).
    Val Loss = NaN, Pred Flood Pixels = 0, Macro F1 = 0.0.
    (Note: Zero predicted flood pixels serves as a collapse indicator on this
    known-positive inner-validation set, not as a universal definition of numerical error).
                         ↓
[6] Silent Early Stopping Swallowing
    Early stopping checks: cand_f1 > best_f1.
    0.0000 is treated as a routine non-improvement.
    Counter increments 1..20 and terminates cleanly with status "COMPLETED".
```

---

## 7. Fail-Fast Deficiencies and Smallest Required Protections

The investigation identified why the failure was not caught immediately:

1. **Lowercase `nan` was not surfaced as an error:**
   Python and PyTorch treat IEEE 754 `NaN` as a valid float. String formatting prints `nan` without throwing an exception. No assertion checked `math.isnan(loss)`.
2. **Training continued after nonfinite loss:**
   PyTorch AMP GradScaler is designed to silently skip the optimizer step and reduce the scale when nonfinite gradients occur. It assumes the next batch might be fine, unaware that BatchNorm buffers have been permanently corrupted.
3. **Manifest recorded `COMPLETED`:**
   Because early stopping triggered via `counter >= patience`, the loop exited cleanly without an unhandled exception.
4. **Early stopping accepted 0 / NaN as ordinary non-improvements:**
   `is_better_checkpoint` evaluated `cand_f1 - best_f1 > tol`. `0.0 - 0.7925 < 0` evaluated to `False`, merely incrementing the patience counter.

### Smallest Required Protections for Production Training

The following minimal protections must be incorporated before `v3`:

1. **Pre-Backward Loss Finiteness Guard:**
   Immediately after computing loss:
   ```python
   if not torch.isfinite(loss):
       raise FloatingPointError(f"Nonfinite loss detected at step {self.global_step}: total={loss.item()}, components={components}")
   ```
2. **Post-Forward Feature Map / BatchNorm Buffer Audit:**
   Audit model BatchNorm running statistics at the end of each training epoch:
   ```python
   for name, buf in self.model.named_buffers():
       if "running_" in name and not torch.isfinite(buf).all():
           raise FloatingPointError(f"BatchNorm buffer corruption in {name}")
   ```
3. **Validation Loss Finiteness Guard:**
   Raise an immediate failure if validation loss is NaN or Inf.
4. **Early Stopping Zero/NaN Guard:**
   Reject an epoch with zero predicted flood pixels or NaN F1 as an immediate fatal error (`FAILED_NUMERICAL`) rather than a patience non-improvement.
5. **AMP Stability via BFloat16 / Precision Configuration:**
   Evaluate `torch.bfloat16` (supported on RTX 3090, offering the same 8-bit exponent and dynamic range as float32) or FP32.  
   *(Crucial Note: Gradient clipping alone cannot prevent forward-pass activation overflow or undo BatchNorm buffer corruption; while it can bound gradients in the backward pass, it is not a sufficient standalone solution for forward-pass numerical instability).*
6. **Manifest Status `FAILED_NUMERICAL`:**
   Catch `FloatingPointError` in launcher and record `completion_status: "FAILED_NUMERICAL"`, preventing corrupted runs from claiming `COMPLETED`.

---

## 8. Compliance & Research Integrity Confirmations

During this audit:
- **Every command executed:** Recorded in audit scripts and logs.
- **Files accessed:** Strictly read-only access to source code, `outputs/week2_source_seed0/*`, `configs/source/*`, and `configs/splits/kuro_source_inner_split_v1.json`.
- **Events accessed:** Strictly the 22 approved source-inner train events and 5 source-inner validation events.
- **Quarantined TTA-validation events accessed:** **ZERO (0)** (Events `[279, 437, 520, 559, 1111003, 1111008]` were never accessed).
- **Sealed-test events accessed:** **ZERO (0)** (Events `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]` and `KuroV2_grid_dict_test_0_100.gz` were never opened).
- **Manually modified metrics:** **ZERO (0)**.
- **Git status:** Maintained clean relative to tracked files. No files committed.

---

## 9. Conclusion & Recommended Next Action

1. **Are best checkpoints independently usable?**
   **YES.** Both Model A and Model B `best_checkpoint.pt` (Epoch 20) are mathematically fully finite, contain zero corrupted parameters, and their performance (`0.7925` and `0.7857`) is independently reproduced with exact pixel-level accuracy.
2. **Are last checkpoints unusable?**
   **YES.** Both `last_checkpoint.pt` files contain corrupted BatchNorm running statistics, predict 0 flood pixels, and are completely unusable.
3. **Is the likely cause identified?**
   **HYPOTHESIS ADVANCED:** FP16 forward activation overflow followed by BatchNorm running-statistics corruption is the strongest current hypothesis, but the first offending operation has not yet been instrumentally observed.
4. **Is a corrected `v3` rerun required?**
   **YES.** A formal Week 2 source-model training cannot be declared complete when training collapsed at Epoch 24/25. A corrected `v3` rerun equipped with fail-fast guards and validated precision/stability protections is strictly required before proceeding to downstream tasks.

