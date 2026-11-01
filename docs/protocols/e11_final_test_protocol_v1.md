# ChangeGuard Work Package E11: Final Test Protocol Specification

**Document Version:** `e11_final_test_protocol_v1`  
**Date:** 2026-10-25  
**Governing Plan:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E11 (Section 16)  
**Route:** Branch C — Controlled Empirical Study of Test-Time Adaptation on Multi-Temporal SAR Streams  
**Status:** `PREPARATION_ONLY — STRICTLY UNAUTHORIZED FOR REAL EXECUTION`  

---

## 1. Absolute Quarantine & Authorization Pre-requisites

> [!CAUTION]
> **CRITICAL RESEARCH INTEGRITY NOTICE:**  
> This protocol specifies the definitive configuration for the final sealed-test evaluation on the 10 reserved Kuro Siwo test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).  
> **Under no circumstances may this protocol be executed, nor may the sealed test partition be unpickled or read, without explicit written user authorization AND supervisor approval.**

### Prerequisites for Sealed Partition Release (Gate T):
1. Complete evidence-gap audit and claim freeze committed to repository HEAD.
2. Verified rehearsal on permitted non-test partitions passing 100% of accounting and integrity assertions.
3. Clean Git working directory with signed, versioned commit.
4. Formal recording of explicit user authorization and supervisor approval in repository records.

---

## 2. Experimental Scope & Matrix Formulation

The final test evaluation is structured to provide conclusive empirical closure for the Branch C research route. It tests whether the findings documented on development events replicate across 10 geographically independent, completely unobserved historical disaster events.

### 2.1 Model & Method Coverage (21 Physical Streams)
1. **SNUNet-ECAM (12.06M parameters):**
   - `source_only` (Seeds 0, 1, 2): Frozen source baseline with training running stats.
   - `bn_no_update` (Seeds 0, 1, 2): Matched test-batch statistics without parameter updates.
   - `tent` (Seeds 0, 1, 2): Unsupervised Shannon entropy minimization ($\eta = 1.0 \times 10^{-4}$).
   - `cotta` (Seeds 0, 1, 2): Continual test-time adaptation with weight-averaged teacher and stochastic restoration.
   - *SNUNet Subtotal:* 12 physical streams (120 event rows).
2. **SiamUnet_diff (1.35M parameters):**
   - `source_only` (Seeds 0, 1, 2): Frozen source baseline with training running stats.
   - `bn_no_update` (Seeds 0, 1, 2): Matched test-batch statistics without parameter updates.
   - `tent` (Seeds 0, 1, 2): Unsupervised Shannon entropy minimization ($\eta = 1.0 \times 10^{-4}$).
   - *SiamUnet Subtotal:* 9 physical streams (90 event rows).
3. **Total Final Evaluation Scope:** Exactly **21 physical streams** producing exactly **210 event-result rows**.

### 2.2 Operational Assumptions & Deployment Mode
- **Deployment Mode:** `reset_per_event` (full model restoration to source checkpoint prior to the first batch of each event).
- **Event Sequence:** Frozen Order A (canonical ascending event ID ordering).
- **Prediction Timing:** `PRE_UPDATE` primary protocol (predictions generated using model state before consuming current-batch gradient updates).
- **Label Access Timing:** Strict post-persistence deferred disk read (primary prediction shards must be durably committed before reading ground truth).
- **Masking:** Double-masking validity `valid = (valid_mask == 1) & (label != 3)`.

---

## 3. Cryptographic Provenance of Checkpoints

| Architecture | Seed | Training Recipe / Fixed Epochs | Expected Checkpoint Path | Expected SHA-256 Checksum |
| :--- | :---: | :---: | :--- | :--- |
| **SNUNet_ECAM** | 0 | Recipe V1 / 16 epochs | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt` | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| **SNUNet_ECAM** | 1 | Recipe V1 / 16 epochs | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt` | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` |
| **SNUNet_ECAM** | 2 | Recipe V1 / 16 epochs | `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt` | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` |
| **SiamUnet_diff** | 0 | Recipe A / 5 epochs | `outputs/e10_second_backbone/all27_seed0/final_checkpoint.pt` | `8e23da9a4d5f1ad01889075d6c98a7197ce295f77ea8d377f1ef26ac814b1532` |
| **SiamUnet_diff** | 1 | Recipe A / 5 epochs | `outputs/e10_second_backbone/all27_seed1/final_checkpoint.pt` | `a92f52153fda9e966d8f8e5ffe5d92c36783d1323d44a0c855d6464ae63e60ea` |
| **SiamUnet_diff** | 2 | Recipe A / 5 epochs | `outputs/e10_second_backbone/all27_seed2/final_checkpoint.pt` | `d96bb4e8d74b6e7d0163129d8990769f5e1650ed9d744cafa2a291638b251aab` |

---

## 4. Statistical Analysis Plan

### 4.1 Primary & Secondary Metrics
- **Primary Metric:** Event-Macro Flood F1 ($\frac{1}{10}\sum_{i=1}^{10} \text{F1}_i$).
- **Secondary Metrics:**
  - Worst-Event Flood F1 ($\min_i \text{F1}_i$)
  - Event-Macro Flood IoU ($\frac{1}{10}\sum_{i=1}^{10} \text{IoU}_i$)
  - Pooled Flood F1 (global confusion matrix micro-average)

### 4.2 Statistical Significance & Bootstrap Plan
- **Paired Event-Cluster Bootstrap:** 10,000 bootstrap resamples drawn at the *event cluster level* ($N=10$). For each resample, the paired performance delta $\Delta_{\text{TTA}} = \text{Macro F1}_{\text{TTA}} - \text{Macro F1}_{\text{source}}$ is evaluated.
- **Confidence Intervals:** 95% Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals reported for all aggregate metrics and paired differences.
- **Hypothesis Testing:** Two-sided paired Wilcoxon signed-rank test across the 10 event clusters.

---

## 5. Execution Logistics & Failure Recovery

### 5.1 Hardware & Runtime Specifications
- **Execution Target:** 3 dedicated NVIDIA RTX 3090 GPUs (24.5 GB VRAM each):
  - `cuda:0`: SNUNet Streams (Seed 0) + SiamUnet Streams (Seed 0) [7 streams]
  - `cuda:1`: SNUNet Streams (Seed 1) + SiamUnet Streams (Seed 1) [7 streams]
  - `cuda:2`: SNUNet Streams (Seed 2) + SiamUnet Streams (Seed 2) [7 streams]
- **Estimated Runtime:** ~45 minutes total (all 3 GPUs running concurrently).
- **Disk Budget:** ~35 GB for compressed prediction shards (`.npz`) and stream summaries.

### 5.2 Failure & Recovery Rules
1. **Strict No-Overwrite Policy:** If a stream summary or shard directory already exists, execution will refuse to overwrite without explicit command-line intervention.
2. **Atomic Artifact Writes:** Manifests, tables, and metrics must be written via atomic temporary file replacement (`.tmp` $\to$ target).
3. **Incomplete Stream Isolation:** If a hardware or process fault occurs, partial shards remain preserved, the stream is marked `EXECUTION_FAILED`, and recovery proceeds via versioned resumption.

