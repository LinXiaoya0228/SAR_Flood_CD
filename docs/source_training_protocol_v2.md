# Controlled Source-Model Training Protocol (v2.0 Approved)

**Document Version:** 2.0.0-APPROVED  
**Status:** `APPROVED — FROZEN FOR WEEK 2 FP32 SOURCE TRAINING RECOVERY`  
**Project:** ChangeGuard FYP (Controlled Source-Model Phase)  
**Approval Date:** 2026-10-08  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Host:** `gpu22` (4x NVIDIA GeForce RTX 3090 24GB)  
**Approved Split File:** [`configs/splits/kuro_source_inner_split_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/splits/kuro_source_inner_split_v1.json) (`7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3`)  
**Historical v1 Protocol:** [`docs/source_training_protocol_v1.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/source_training_protocol_v1.md) (`bc7c2ea8b9a8e62553c200bc1a9f3de326f39edde0f9a0a3fa4843ec61934efa`)  
**Diagnostic Evidence:** [`docs/week2_seed0_numerical_audit.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week2_seed0_numerical_audit.md) and [`docs/week2_seed0_causal_diagnostic.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week2_seed0_causal_diagnostic.md)  

---

## 1. Rationale for Protocol Amendment

### 1.1 Why Protocol v1 (AMP FP16) Was Amended
During Week 2 seed-0 canonical training under Protocol v1:
1. **Model B Collapse:** Training collapsed at Epoch 24 with `nan` training losses, `nan` validation losses, and 0 predicted flood pixels.
2. **Model A Collapse:** Training collapsed at Epoch 25 with `nan` validation losses and 0 predicted flood pixels.
3. **BatchNorm Contamination:** Corrupted `NaN`/`Inf` statistics entered BatchNorm running buffers (`running_mean`, `running_var`), rendering post-collapse weights and `last_checkpoint.pt` completely uninvertible and unusable.
4. **Early Stopping Masking:** The Protocol v1 launcher silently accepted `0.000000` Macro F1 as an ordinary non-improvement and incremented patience to 20, recording `COMPLETED` despite complete numerical and predictive collapse.
5. **Checkpoint Invalidation:** Both Model A and Model B canonical v2 runs are invalid because later epochs developed numerical corruption. Their finite Epoch 20 best checkpoints remain diagnostic/exploratory artifacts only and must not be presented as final canonical source models.

### 1.2 Direct Evidence vs. Inference
- **Directly Observed Evidence:** In bounded causal replay of Model B, the earliest directly observed failure was nonfinite scaled gradients in the FP16/AMP backward path at Epoch 23, Batch 353 (Step 13288), while recorded inputs, forward activations, logits, loss, and BatchNorm buffers were strictly finite. The nonfinite gradients occurred while `GradScaler` scale `65,536.0` was active.
- **Inference vs. Lower-Level Cause:** The co-occurrence of scale `65,536.0` and gradient overflow does not by itself prove that the scale value alone caused it; the lower-level cause of gradient overflow remains unresolved.
- **Empirical Resolution:** Replaying the identical checkpoint and exact sample/event sequence in full single-precision (FP32) completely avoided gradient overflow, crossing the failure point with finite gradients and completing all 11,990 batches through Epoch 25 with 100% finite tensors.
- **Protocol Decision:** To eliminate AMP-path gradient instability and GradScaler dynamics, both Model A and Model B formal recovery runs are amended to **full FP32 precision**.

---

## 2. Protected Event Boundaries & Research Integrity Safeguards

The strict research integrity rules established in Protocol v1 remain **permanently active and non-negotiable**:

1. **Quarantined TTA-Validation Events (Permanent Isolation):**
   ```text
   [279, 437, 520, 559, 1111003, 1111008]
   ```
   **Permanent Ban:** These 6 events must **never** be accessed or monitored during source training, validation, checkpoint selection, or smoke testing.
2. **Sealed-Test Events (Permanent Sealing):**
   ```text
   [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
   ```
   **Permanent Ban:** Opening, unpickling, reading, or inspecting `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` or any test data is strictly prohibited.
3. **Approved Source-Inner Split (Alternative A1):**
   - **Inner-Training Events ($N=22$):**
     `[118, 130, 147, 174, 273, 324, 417, 421, 427, 470, 492, 496, 498, 502, 554, 555, 567, 1111004, 1111006, 1111009, 1111010, 1111011]` (19,178 patches).
   - **Inner-Validation Events ($N=5$):**
     `[267, 275, 497, 518, 1111005]` (2,284 patches, 18,946,970 genuine flood pixels under `label == 2`).
   - Disjointness: `inner_train ∩ inner_val = ∅`.

---

## 3. Approved Recovery Models & Precision Specification

Both models are retrained from **Epoch 1** under full single-precision:

### 3.1 Precision Architecture
- **Precision Mode:** `torch.float32` (Full FP32).
- **Autocast:** Completely disabled (`torch.cuda.amp.autocast()` not used).
- **GradScaler:** Completely disabled (`scaler = None`).
- **BFloat16 Policy:** BFloat16 is **not approved** for canonical recovery runs. BF16 remains a future efficiency ablation and must not replace FP32 without an independent bounded smoke test.

### 3.2 Model Configurations
- **Model A (`snunet_vvvh_v2_fp32.json`):**
  - Path: `configs/source/snunet_vvvh_v2_fp32.json`
  - Architecture: `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
  - Inputs: Pre-event SAR `[VV, VH]` and Post-event SAR `[VV, VH]`.
  - Config ID: `model_a_radar_only_v2_fp32`.
  - Canonical Run ID: `source_snunet_vvvh_seed0_v3_fp32`.
  - Authoritative Hash: Stored in [`configs/source/source_training_v2_lock.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/source_training_v2_lock.json).
- **Model B (`snunet_vvvh_slope_v2_fp32.json`):**
  - Path: `configs/source/snunet_vvvh_slope_v2_fp32.json`
  - Architecture: `SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)`.
  - Inputs: Pre-event SAR `[VV, VH, slope]` and Post-event SAR `[VV, VH, slope]`.
  - Config ID: `model_b_topographic_ablation_v2_fp32`.
  - Canonical Run ID: `source_snunet_vvvh_slope_seed0_v3_fp32`.
  - Authoritative Hash: Stored in [`configs/source/source_training_v2_lock.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/source/source_training_v2_lock.json).

> [!NOTE]
> **Acyclic Provenance Policy:**  
> In accordance with the one-way immutable lock architecture, this governing protocol does not embed final configuration hashes. The authoritative SHA-256 digests of both configurations are immutably bound in `configs/source/source_training_v2_lock.json`.

---

## 4. Unchanged Core Scientific Parameters

All scientific settings from Protocol v1 remain **100% frozen and identical**:

1. **SAR Preprocessing:** Clamped to $[0.0, 0.15]$, nonfinite replacement with $0.15$, z-score normalization ($\mu_{VV}=0.0953, \sigma_{VV}=0.0427; \mu_{VH}=0.0264, \sigma_{VH}=0.0215$).
2. **Topographic Policy S2:** Physical metric pixel spacing, metric CRS verification, train-only normalization ($\mu=1.4926^\circ, \sigma=2.3723^\circ$, SHA-256 `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`).
3. **Sample Retention Policy:** All 19,178 inner-train patches retained without filtering or oversampling.
4. **Optimization Hyperparameters:** Adam optimizer, learning rate $1.0 \times 10^{-3}$, weight decay $0.0$, betas `(0.9, 0.999)`, no learning rate scheduler.
5. **Loss Formulation:** Unweighted $0.5 \times \text{CE} + 0.5 \times \text{Dice}$ with ignore index 3 and smoothing $\epsilon=1.0$.
6. **Effective Batch Size:** Effective batch size 32 via physical microbatch 8 and accumulation steps 4.
7. **Training Budget:** Maximum 150 epochs.
8. **Early Stopping Patience:** Patience 20 on inner-validation Event-Macro Flood F1.
9. **Tie-Breaking Rule:** Higher Macro Flood F1 $\to$ Higher Worst-Event Flood F1 $\to$ Higher Macro Flood Recall $\to$ Earlier Epoch.

---

## 5. Fail-Fast Safeguards & Status Semantics

### 5.1 Immediate `FAILED_NUMERICAL` Guards
Training halts immediately before an invalid optimizer step or corrupted checkpoint write if any nonfinite tensor is detected:
1. **Inputs:** `not (torch.isfinite(xA).all() and torch.isfinite(xB).all())`
2. **Logits:** `not torch.isfinite(logits).all()`
3. **Loss (Pre-Backward):** `not torch.isfinite(loss)`
4. **Gradients (Post-Backward, Pre-Step):** `not torch.isfinite(p.grad).all()`
5. **Model Parameters:** `not torch.isfinite(p).all()`
6. **BatchNorm Running Buffers:** `not torch.isfinite(buf).all()` across all `running_mean` and `running_var` buffers.
7. **Validation Outputs & Loss:** `not torch.isfinite(val_loss)` or nonfinite validation logits.

### 5.2 Predictive Collapse Distinction (`FAILED_COLLAPSE`)
Predictive collapse is strictly distinguished from numerical failure:
- **Evaluation Set:** Evaluated strictly on the predeclared known-positive inner-validation set (which contains 18,946,970 genuine flood pixels under `label == 2`).
- **Warm-up Period:** `collapse_warmup_epochs: 5`. Collapse is **not triggered** during the first 5 epochs.
- **Patience Counter:** `collapse_patience: 3`. After the 5-epoch warm-up, zero predicted flood pixels must occur for **3 consecutive validation passes** before declaring collapse.
- **Reset Rule:** The collapse counter resets to 0 whenever a validation pass produces nonzero predicted flood pixels.
- **Precedence Rule:** If any tensor is `NaN` or `Inf`, the status is `FAILED_NUMERICAL`, **not** `FAILED_COLLAPSE`.
- **Domain Clarification:** Zero predicted flood pixels is **not universally invalid** (on dry scenes without flood, zero predicted flood is correct). It serves strictly as a collapse criterion on the predeclared known-positive validation set.

### 5.3 Valid Completion Statuses
- **`COMPLETED_EARLY_STOP`:** Valid termination when inner-validation Macro F1 does not improve for 20 consecutive epochs while all tensors remain finite and non-collapsed.
- **`COMPLETED_MAX_EPOCH`:** Valid termination when all 150 epochs are completed without early stopping.

### 5.4 Checkpoint Integrity Invariant
- **Atomic Writing:** Checkpoints are written to `.tmp` files and committed via `os.replace` to prevent corrupted partial files.
- **Finiteness Audit:** Before saving `best_checkpoint.pt` or `last_checkpoint.pt`, all parameters and running buffers must pass finiteness verification. Corrupted models are **never saved as best or usable last checkpoints**.
- **Forensic Failure Manifest:** On failure, `failure_manifest.json` is written, recording failure status, epoch, batch, step, sample IDs, event IDs, tensor category, and timestamp.

---

## 6. Execution Gates

### Gate 1: Unit & Safeguard Test Gate
All automated tests verifying FP32 mode, finiteness assertions, failure manifest generation, collapse warm-up/patience, and hash guards must pass completely:
```bash
/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python -m pytest tests/
```

### Gate 2: Bounded Smoke-Test Gate
Prior to launching any formal full-budget runs:
- Execute 10–50 training batches in isolated smoke directories (`outputs/smoke_model_a_fp32/` and `outputs/smoke_model_b_fp32/`) using at most two idle GPUs.
- Verify forward/backward passes, finite loss, finite gradients, finite BatchNorm buffers, checkpoint save, checkpoint resume, and zero protected data access.
- Document results in [`docs/week2_fp32_recovery_smoke_closeout.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/week2_fp32_recovery_smoke_closeout.md).

### Gate 3: Formal Canonical Run Gate
Formal canonical training (`source_snunet_vvvh_seed0_v3_fp32` and `source_snunet_vvvh_slope_seed0_v3_fp32`) may only be launched:
- From a clean, committed Git checkpoint containing Protocol v2, versioned configs, safeguards, and smoke closeout.
- Inside dedicated, detached tmux sessions with stdout/stderr logged.
- With zero full-run directories pre-existing or overwritten.
