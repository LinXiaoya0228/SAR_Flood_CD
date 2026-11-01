# Controlled Source-Model Training Protocol Amendment Proposal (v2.0 Draft)

**Document Version:** 2.0.0-PROPOSAL  
**Status:** `APPROVED AND SUPERSEDED BY docs/source_training_protocol_v2.md`  
**Governing Approved Protocol:** [`docs/source_training_protocol_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v2.md)  
**Immutable Provenance Lock:** [`configs/source/source_training_v2_lock.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/source_training_v2_lock.json)  
**Project:** ChangeGuard FYP (Controlled Source-Model Phase)  
**Date:** 2026-10-08  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA RTX 3090 24GB)  
**Reference Frozen Protocol:** [`docs/source_training_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v1.md)  
**Diagnostic Evidence Base:** [`docs/week2_seed0_numerical_audit.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week2_seed0_numerical_audit.md) and [`docs/week2_seed0_causal_diagnostic.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week2_seed0_causal_diagnostic.md)  

---

## 1. Executive Summary & Purpose of Amendment

This document originally proposed a targeted amendment to the frozen source-training protocol (`v1.0`) to resolve the numerical breakdown discovered during Week 2 seed-0 canonical training (`source_snunet_vvvh_slope_seed0_v2` and `source_snunet_vvvh_seed0_v2`), where automatic mixed precision (AMP FP16) training suffered nonfinite backward gradient overflow and catastrophic BatchNorm running-statistics corruption.

> [!NOTE]
> **APPROVAL STATUS & PROVENANCE NOTICE:**  
> This proposal was **formally approved on 2026-10-08** and is now **SUPERSEDED BY [`docs/source_training_protocol_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v2.md)**.  
> This proposal file is retained permanently as research and protocol provenance.  
> 
> **Formal Approval Decisions:**  
> 1. **Both Model A and Model B use full single-precision (FP32)** for canonical seed-0 recovery runs from Epoch 1.  
> 2. **BFloat16 (BF16) is not part of the canonical recovery** (neither primary nor fallback for canonical runs; excluded from canonical recovery).  
> 3. **All other scientific settings remain strictly frozen** (Alternative A1 22/5 event split, Policy S2 metric slope normalization, Adam optimizer with lr=1e-3, betas=(0.9, 0.999), weight decay=0.0, batch size 32, max 150 epochs, patience 20, zero access to quarantined TTA-validation or sealed-test events).

---

## 2. Evidence Base & Rationale for Changes

### 2.1 Empirical Findings from Week 2 Seed-0 Diagnostics
1. **Instrumented Failure Point:** In Model B under AMP FP16, the earliest directly observed failure occurred at Epoch 23, Batch 353 (Step 13288), where backward gradients overflowed to `Inf`/`NaN` while `GradScaler` scale was 65,536.0, despite inputs, forward activations, logits, loss, and BatchNorm buffers being strictly finite.
2. **Post-Collapse Contamination:** In uninstrumented training (`v2`), nonfinite values contaminated BatchNorm running buffers (`running_mean`, `running_var`) by Epoch 24 (Model B) and Epoch 25 (Model A), resulting in NaN validation losses and complete predictive collapse (0 predicted flood pixels).
3. **Matched FP32 Stability:** Replaying Model B from the identical Epoch 20 checkpoint under full FP32 across the exact same sample/event sequence successfully crossed the failure batch with finite gradients and completed through Epoch 25 (11,990 batches) with **zero nonfinite tensors across all layers and buffers**.
4. **Vulnerabilities in Protocol v1 Implementation:**
   - No pre-backward or post-backward finiteness checks existed in the training loop.
   - BatchNorm running buffers were never audited at epoch boundaries.
   - Early stopping accepted NaN validation loss and 0 predicted flood pixels as ordinary non-improving epochs, silently exhausting patience (20 epochs) and recording `COMPLETED`.
   - `last_checkpoint.pt` was saved despite containing corrupted `NaN` BatchNorm buffers.

---

## 3. Specific Proposed Amendments vs. Frozen Protocol v1

Every scientific parameter and hyperparameter from Protocol v1 remains **strictly unchanged**, except for the explicit items proposed below:

| Protocol Dimension | Protocol v1.0 (Frozen) | Proposed Amendment (v2.0 Proposal) | Rationale |
| :--- | :--- | :--- | :--- |
| **Model A Execution** | AMP FP16 | **Unchanged (if canonical run verified valid)** | If Model A canonical run is verified valid, retain it; if corrupted, amend precision. |
| **Model B Precision** | AMP FP16 | **Full FP32 (`precision: "fp32"`)** | Eliminates GradScaler overflow; proven 100% stable through Epoch 25 in matched replay. |
| **BFloat16 Precision Status** | Not evaluated | **Future Efficiency Ablation Only** | BFloat16 must NOT replace FP32 without an independent bounded smoke test. |
| **Pre-Backward Loss Guard** | None | **Mandatory Assertion (`torch.isfinite(loss)`)** | Halts training immediately before computing nonfinite gradients. |
| **Gradient Guard** | None | **Mandatory Assertion (`torch.isfinite(grad)`)** | Verifies gradients after unscaling and before `optimizer.step()`. |
| **Forward & Buffer Guards** | None | **Mandatory Checks on inputs, logits, parameters, and BN buffers** | Audits all model parameters and BatchNorm buffers at each epoch boundary. |
| **Validation Loss Guard** | None | **Mandatory Assertion (`torch.isfinite(val_loss)`)** | Halts immediately if validation loss is nonfinite. |
| **Checkpoint Integrity Rule** | Best saved on Macro F1 | **Never save a corrupted checkpoint** | Both `best_checkpoint.pt` and `last_checkpoint.pt` must pass finiteness audits before writing. |
| **Run Status Classifications** | `COMPLETED` / `FAILED` | **`COMPLETED` / `FAILED_NUMERICAL` / `FAILED_COLLAPSE`** | Explicitly distinguishes raw numerical invalidity from degenerate predictive collapse. |

---

## 4. Detailed Specification of Proposed Safeguards

### 4.1 Precision Strategy: Primary FP32 Recovery & BF16 Ablation Boundary
- **Model B Recovery Precision:** Full single-precision (`torch.float32`).
  - GradScaler is completely disabled (`scaler = None`).
  - Dynamic scaling overflow is mathematically eliminated.
  - Runtime throughput on NVIDIA RTX 3090 is ~3.5 batches/sec (~5.5 hours for full 150 epochs), which is well within FYP execution limits.
- **BFloat16 Policy:**
  - `torch.bfloat16` features native hardware support on the RTX 3090 with an 8-bit exponent matching float32 dynamic range.
  - However, BFloat16 has lower precision (7-bit mantissa) and must **not** replace FP32 as the primary recovery protocol without an independent, bounded smoke test demonstrating numerical stability and convergence.

### 4.2 Comprehensive Finiteness Assertions
The training engine (`src/training/source_trainer.py`) must implement explicit runtime assertions:

1. **Input Finiteness:**
   ```python
   if not (torch.isfinite(xA).all() and torch.isfinite(xB).all()):
       raise FloatingPointError(f"Nonfinite input tensor at step {step}")
   ```
2. **Logits Finiteness:**
   ```python
   if not torch.isfinite(logits).all():
       raise FloatingPointError(f"Nonfinite model logits at step {step}")
   ```
3. **Loss Before Backward:**
   ```python
   if not torch.isfinite(loss):
       raise FloatingPointError(f"Nonfinite loss ({loss.item()}) before backward at step {step}")
   ```
4. **Gradients After Backward & Unscale:**
   ```python
   for name, param in self.model.named_parameters():
       if param.grad is not None and not torch.isfinite(param.grad).all():
           raise FloatingPointError(f"Nonfinite gradient in parameter '{name}' at step {step}")
   ```
5. **Model Parameters Audit:**
   ```python
   for name, param in self.model.named_parameters():
       if not torch.isfinite(param).all():
           raise FloatingPointError(f"Nonfinite weights in parameter '{name}' at step {step}")
   ```
6. **BatchNorm Running Buffer Audit:**
   At the end of each training epoch:
   ```python
   for name, buf in self.model.named_buffers():
       if "running_" in name and not torch.isfinite(buf).all():
           raise FloatingPointError(f"Corrupted BatchNorm running buffer in '{name}' at epoch {epoch}")
   ```
7. **Validation Outputs & Loss:**
   ```python
   if not torch.isfinite(val_loss):
       raise FloatingPointError(f"Nonfinite validation loss ({val_loss}) at epoch {epoch}")
   ```

### 4.3 Checkpoint Saving Invariants
- **Rule:** Never save a numerically corrupted checkpoint as `best_checkpoint.pt` or `last_checkpoint.pt`.
- **Pre-Save Audit:** Before writing any checkpoint file to disk, the trainer must execute a comprehensive finiteness scan over all state dict entries (`model_state_dict`, `optimizer_state_dict`, and buffer states). If any tensor contains `NaN` or `Inf`, saving is blocked, and an immediate fatal error is raised.

### 4.4 Formal Status Classification & Early Stopping Protection
The run manifest `completion_status` and early stopping logic must strictly implement three mutually exclusive classifications:

1. **`FAILED_NUMERICAL`:**
   - **Trigger:** Any occurrence of `NaN` or `Inf` in inputs, activations, logits, loss, gradients, model parameters, or BatchNorm running statistics.
   - **Action:** Training halts immediately. Process exits with code 2. Manifest status: `"FAILED_NUMERICAL"`.
2. **`FAILED_COLLAPSE`:**
   - **Trigger:** Model execution remains mathematically finite across all tensors, but model predictions exhibit degenerate predictive collapse—specifically predicting exactly **0 flood pixels** across the entire inner-validation set.
   - **Contextual Validity Rule:** Zero predicted flood pixels is **not universally invalid** (e.g. on a patch or event with zero ground-truth water, predicting zero flood pixels is correct). It serves strictly as a collapse criterion on the **predeclared, verified known-positive inner-validation set** (which contains 18,946,970 genuine flood pixels under `label == 2`).
   - **Action:** Training halts immediately. Process exits with code 3. Manifest status: `"FAILED_COLLAPSE"`.
3. **`COMPLETED`:**
   - **Trigger:** All required epochs (up to 150) and validation evaluations complete successfully without encountering either `FAILED_NUMERICAL` or `FAILED_COLLAPSE`.

---

## 5. Unchanged Core Scientific Parameters (Frozen from v1.0)

All foundational scientific settings from Protocol v1 remain **100% frozen and unmodified**:

- **Event Split:** Frozen Alternative A1 (22 inner-train events, 5 inner-val events) in [`configs/splits/kuro_source_inner_split_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/splits/kuro_source_inner_split_v1.json).
- **Protected Data Isolation:**
  - Quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`): **Permanently isolated (0 access)**.
  - Sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`): **Permanently sealed (0 access)**.
- **Data Preprocessing:** Sentinel-1 linear backscatter clamped to $[0.0, 0.15]$, z-score standardized ($\mu_{VV}=0.0953, \sigma_{VV}=0.0427; \mu_{VH}=0.0264, \sigma_{VH}=0.0215$).
- **Topographic Policy S2:** Physical cell-spacing slope derivation in degrees, metric CRS verification, train-only standardization ($\mu=1.4926^\circ, \sigma=2.3723^\circ$).
- **Sample Retention Policy:** Retain all 19,178 training patches without filtering or oversampling.
- **Architecture:** SNUNet with ECAM, `base_channel=32`, `out_ch=3`.
- **Optimization Hyperparameters:** Adam optimizer, learning rate $1.0 \times 10^{-3}$, weight decay $0.0$, betas $(0.9, 0.999)$, no learning rate scheduler.
- **Loss Formulation:** Balanced $0.5 \cdot L_{\text{CE}} + 0.5 \cdot L_{\text{Dice}}$ with ignore index 3 and smoothing $\epsilon=1.0$.
- **Batch Size & Accumulation:** Physical microbatch size 8, gradient accumulation steps 4 (effective batch size 32).
- **Training Budget:** Maximum 150 epochs, early stopping patience 20 epochs based on event-macro Flood F1.
- **Deterministic Retraining Rule:** Locked median retraining epoch across seeds `[42, 43, 44]` for final full-corpus production models.

---

## 6. Implementation & Review Checklist Before Formal v3 Launch

Before any formal v3 source-model training can be executed:
1. **User Review & Approval:** Formally approved on 2026-10-08 and codified into [`docs/source_training_protocol_v2.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v2.md).
2. **Trainer Implementation:** Implement the 7 runtime finiteness checks, pre-save checkpoint verification, and exit status classifications in `src/training/source_trainer.py`.
3. **Unit & Smoke Tests:** Add unit tests verifying that `FloatingPointError` is triggered when nonfinite tensors are injected and that status `"FAILED_NUMERICAL"` and `"FAILED_COLLAPSE"` are correctly written.
4. **Config Update:** Create dedicated FP32 recovery configs `configs/source/snunet_vvvh_v2_fp32.json` and `configs/source/snunet_vvvh_slope_v2_fp32.json`.
5. **Git Checkpoint:** Commit all implementation updates and tests under a clean Git tree before launching formal v3 runs.

