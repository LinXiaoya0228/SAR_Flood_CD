# Week 1 Protocol Amendment: Physical Microbatch & Memory Validation

**Document ID:** `docs/week1_batch_memory_amendment.md`  
**Status:** `CONFIRMED_AND_LOCKED`  
**Date:** 2026-10-06  
**Hardware:** Host `gpu22`, GPU 0 (NVIDIA GeForce RTX 3090 24GB, 24,124 MiB available VRAM)  
**Parent Protocol:** [`docs/source_training_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v1.md) (Section 8)  
**Raw Benchmark Artifact:** [`outputs/week1/memory_smoke_results.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/outputs/week1/memory_smoke_results.json)  

---

## 1. Executive Summary

As mandated by Section 8 of the frozen controlled source training protocol (`docs/source_training_protocol_v1.md`), physical microbatch sizes in candidate progression `[2, 4, 8, 16]` were empirically benchmarked on GPU 0 under Automatic Mixed Precision (AMP FP16) to establish the optimal physical microbatch size satisfying effective batch size 32 with exact accumulation divisibility (`32 % microbatch == 0`).

All four candidate microbatch configurations executed without OOM, numerical instability, or NaN/Inf gradients across both Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`).

Based on memory headroom, accumulation efficiency, and throughput stability, the official training configuration is locked as:
- **Physical Microbatch Size:** `8` (conservative standard) or `16` (fast alternative)
- **Primary Operational Choice:** `physical_microbatch_size = 8`, `gradient_accumulation_steps = 4`
- **Effective Batch Size:** Exactly `32` (`8 * 4 = 32`)
- **VRAM Headroom:** Over **87.2%** unused VRAM on RTX 3090, guaranteeing zero OOM risk during multi-epoch runs.

---

## 2. Empirical Benchmark Data

All tests were executed on GPU 0 using PyTorch AMP FP16 and Adam optimizer with effective batch size 32.

### 2.1 Model A (Controlled Radar Baseline: 2 Channels `[VV, VH]`)

| Microbatch Size | Accumulation Steps | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Headroom | Throughput (samples/s) | Status |
|:---------------:|:------------------:|:-------------------:|:------------------:|:-------------:|:----------------------:|:------:|
| 2 | 16 | 827.24 MiB | 928.00 MiB | 96.1% | 41.46 | PASS |
| 4 | 8 | 1,435.67 MiB | 1,576.00 MiB | 93.5% | 48.86 | PASS |
| **8** | **4** | **2,559.77 MiB** | **3,024.00 MiB** | **87.5%** | **49.86** | **OPTIMAL** |
| 16 | 2 | 4,991.43 MiB | 5,990.00 MiB | 75.2% | 49.54 | PASS |

### 2.2 Model B (Controlled Topographic Ablation: 3 Channels `[VV, VH, slope]`)

| Microbatch Size | Accumulation Steps | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Headroom | Throughput (samples/s) | Status |
|:---------------:|:------------------:|:-------------------:|:------------------:|:-------------:|:----------------------:|:------:|
| 2 | 16 | 831.91 MiB | 986.00 MiB | 95.9% | 42.59 | PASS |
| 4 | 8 | 1,437.66 MiB | 1,588.00 MiB | 93.4% | 47.93 | PASS |
| **8** | **4** | **2,654.01 MiB** | **3,068.00 MiB** | **87.3%** | **49.38** | **OPTIMAL** |
| 16 | 2 | 4,998.87 MiB | 6,020.00 MiB | 75.0% | 49.25 | PASS |

---

## 3. Analysis & Operational Selection

1. **Peak Memory Scaling:**
   Memory allocation scales almost linearly with microbatch size: ~830 MiB at $B=2$, ~1,437 MiB at $B=4$, ~2,654 MiB at $B=8$, and ~4,998 MiB at $B=16$. Even at $B=16$, only 6.0 GB of the available 24.1 GB is reserved, leaving 18.1 GB unutilized.
2. **Throughput Saturation:**
   Throughput plateaus at $B=8$ (~49.6 samples/s). Increasing microbatch size from 8 to 16 does not yield higher throughput (49.86 vs 49.54 samples/s) because GPU compute saturation is already achieved at $B=8$ for $224 \times 224$ patches.
3. **Accumulation Trade-Off:**
   $B=8$ with accumulation 4 provides an optimal balance: it reduces Python loop overhead 4x compared to $B=2$, while maintaining lower peak activation memory and negligible risk of fragmentation compared to $B=16$.
4. **Conclusion:**
   `physical_microbatch_size = 8` with `gradient_accumulation_steps = 4` is locked as the primary setting for all Week 1 and subsequent controlled source-model training runs.
