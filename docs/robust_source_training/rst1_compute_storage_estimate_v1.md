# RST1 Compute and Storage Resource Estimates

**Document version:** 1.0.0
**Estimation date:** 2026-10-26
**Status:** Approved planning resource estimation for robust source training
**Host target:** `gpu22` (4 $\times$ NVIDIA GeForce RTX 3090 24GB, CUDA 13.0, Driver 580.173.02)
**Disk storage root:** `/usr1/home/s126mdg22_06/FYP_SARCD` (Available: 3.8 TB)

---

## 1. Measured Historical Baseline Values

The resource requirements are derived from actual measured execution metrics recorded during the Week 5 all-27 source training runs (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/train_history.csv`):

| Metric | Measured Historical Value | Source of Evidence |
| :--- | :--- | :--- |
| **All-27 Patch Count** | 21,462 patches | Verified on-disk and in manifest |
| **Micro-batch size** | 8 patches | Config `snunet_vvvh_all27_seed0_v1_fp32.json` |
| **Gradient accumulation steps** | 4 steps (Effective batch size = 32) | Config `snunet_vvvh_all27_seed0_v1_fp32.json` |
| **Mean epoch runtime (21,462 patches)** | **889.3 seconds** (14.82 minutes) | `train_history.csv` (16 epochs mean) |
| **Total 16-epoch training duration** | **14,228 seconds** (3.95 hours $\approx$ **4.0 GPU-hours**) | Measured total run time |
| **Peak GPU VRAM usage** | **4,914.88 MB** (~4.92 GB) | Measured in `train_history.csv` |
| **Checkpoint file size (single epoch)** | **144.67 MB** (`epoch_XXX_checkpoint.pt`) | Measured file on disk |
| **Full run checkpoint storage (16 epochs)** | **2.31 GB** (16 epoch ckpts + final + last) | Measured directory size |
| **Single model evaluation runtime (9 events)** | **~90 to 120 seconds** (1.5 to 2.0 minutes) | Measured in E11 sealed test execution |

---

## 2. Derived Training Run Duration Estimates

In the event-disjoint folds, training runs use 18 events rather than 27 events. Scaling linearly with patch count yields:

### 2.1 Discovery Runs (Trained on Groups A + B)
- **Training Patches:** $8,160 + 6,807 = 14,967$ patches ($69.74\%$ of all-27 corpus).
- **Optimizer Steps per Epoch:** $\lceil 14,967 / 32 \rceil = 468$ steps.
- **Estimated Epoch Duration:** $889.3 \times \frac{14,967}{21,462} \approx 620.2\text{ seconds}$ (10.34 minutes).
- **Estimated 16-Epoch Duration:** $16 \times 620.2\text{ s} \approx 9,923\text{ seconds} \approx$ **2.76 GPU-hours**.

### 2.2 Confirmation Fold A Runs (Trained on Groups B + C)
- **Training Patches:** $6,807 + 6,495 = 13,302$ patches ($61.98\%$ of all-27 corpus).
- **Optimizer Steps per Epoch:** $\lceil 13,302 / 32 \rceil = 416$ steps.
- **Estimated Epoch Duration:** $889.3 \times \frac{13,302}{21,462} \approx 551.2\text{ seconds}$ (9.19 minutes).
- **Estimated 16-Epoch Duration:** $16 \times 551.2\text{ s} \approx 8,819\text{ seconds} \approx$ **2.45 GPU-hours**.

### 2.3 Confirmation Fold B Runs (Trained on Groups A + C)
- **Training Patches:** $8,160 + 6,495 = 14,655$ patches ($68.28\%$ of all-27 corpus).
- **Optimizer Steps per Epoch:** $\lceil 14,655 / 32 \rceil = 458$ steps.
- **Estimated Epoch Duration:** $889.3 \times \frac{14,655}{21,462} \approx 607.3\text{ seconds}$ (10.12 minutes).
- **Estimated 16-Epoch Duration:** $16 \times 607.3\text{ s} \approx 9,717\text{ seconds} \approx$ **2.70 GPU-hours**.

---

## 3. Work Package Execution Budget

### 3.1 Discovery Stage (RST4)
- **Matrix:** 4 training runs (R0 baseline, R1 balanced, R2 balanced+loss, R3 full method) on Fold C (seed 0).
- **Total GPU-Hours:** $4 \text{ runs} \times 2.76 \text{ h} \approx$ **11.04 GPU-hours**.
- **Parallel Scheduling:** 4 runs distributed across GPUs 0, 1, 2, 3 in parallel.
- **Estimated Wall-Clock Time:** **~2.8 to 3.0 hours**.

### 3.2 Confirmatory Stage (RST6)
- **Matrix:** 12 training runs:
  - 6 runs on Fold A (R0 + Promoted Recipe $\times$ Seeds 0, 1, 2): $6 \times 2.45\text{ h} = 14.70\text{ GPU-hours}$.
  - 6 runs on Fold B (R0 + Promoted Recipe $\times$ Seeds 0, 1, 2): $6 \times 2.70\text{ h} = 16.20\text{ GPU-hours}$.
- **Total GPU-Hours:** $14.70 + 16.20 =$ **30.90 GPU-hours**.
- **Parallel Scheduling:** 12 runs distributed across 4 GPUs (3 sequential runs per GPU).
- **Estimated Wall-Clock Time:** $\max(3 \times 2.70\text{ h}, 3 \times 2.45\text{ h}) \approx$ **8.1 to 8.5 hours**.

### 3.3 Evaluation Overhead (RST4 & RST7)
- **Number of Evaluated Checkpoints:** 4 discovery + 12 confirmatory = 16 models.
- **Evaluation Time per Model:** ~2 minutes across 9 held-out events.
- **Total Evaluation GPU-Hours:** $16 \times 2\text{ min} \approx$ **0.53 GPU-hours** (~32 minutes total).

### 3.4 Optional Second-Backbone Verification (RST8)
- If triggered, `SiamUnet_diff` (1.35M parameters) trains ~3x faster per epoch than SNUNet:
  - 6 to 12 runs $\times$ ~0.9 GPU-hours $\approx$ **5.4 to 10.8 GPU-hours**.

### 3.5 Total Sequential & Parallel Compute Summary

| Phase | Runs | Total Sequential GPU-Hours | Wall-Clock on 4 RTX 3090s |
| :--- | :---: | :---: | :---: |
| **RST4 Discovery Staircase** | 4 | ~11.0 GPU-h | **~2.8 hours** |
| **RST6 Confirmatory Matrix** | 12 | ~30.9 GPU-h | **~8.2 hours** |
| **Evaluation Overhead** | 16 | ~0.5 GPU-h | **~0.2 hours** |
| **Primary Route Total** | **16 runs** | **~42.4 GPU-hours** | **~11.2 hours** |
| *RST8 Optional Second Backbone* | *12* | *~10.8 GPU-h* | *~2.7 hours* |
| *RST9 Optional Deployable Model* | *3 (all 27)* | *~12.0 GPU-h* | *~4.0 hours* |

---

## 4. Storage Budget and Disk Space Requirements

### 4.1 Checkpoint and Log Storage
- **Per-Run Storage (Saving All 16 Epochs):**
  - Checkpoints: $16 \times 144.67\text{ MB} + 2 \times 144.67\text{ MB} \approx 2.6\text{ GB}$.
  - Logs, history CSV, and manifest JSON: $\sim 2\text{ MB}$.
  - Total per run: $\approx 2.6\text{ GB}$.
- **Storage for 16 Primary Runs (RST4 + RST6):**
  $$16 \text{ runs} \times 2.6\text{ GB} = \mathbf{41.6\text{ GB}}$$
- **Secondary Extensions (RST8 + RST9):** $\sim 20\text{ GB}$.
- **Total Route Storage Requirement:** $\sim 65\text{ GB}$.
- **Safety Margin:** Budgeting 100 GB.
- **Filesystem Verification:**
  - Filesystem `/usr1`: **3.8 TB available** (46% used).
  - Storage margin: **$>38\times$ required disk capacity**.

---

## 5. Operational Safety Guidelines

1. **One Job Per GPU Rule:** Each RTX 3090 GPU possesses 24 GB VRAM. While a single SNUNet run occupies ~4.9 GB VRAM, **never co-locate two training processes on the same physical GPU**. Independent training avoids CUDA scheduling contention, memory spikes during backpropagation, and thermal throttling.
2. **GPU Allocation Map:**
   - `GPU 0` (`cuda:0`): Baseline R0 runs
   - `GPU 1` (`cuda:1`): Component runs (R1/R2)
   - `GPU 2` (`cuda:2`): Full method R3 runs
   - `GPU 3` (`cuda:3`): Spare worker for evaluations, recovery retries, or parallel seeds
3. **Session Management:** All long-running jobs must execute inside detached `tmux` sessions with stdout/stderr redirected to durable run logs under `logs/robust_source_training/<rst_block>/`.
