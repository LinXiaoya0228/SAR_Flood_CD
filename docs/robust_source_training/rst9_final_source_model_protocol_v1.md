# ChangeGuard RST9 Final Deployable Source Model Training and Historical-Target Extension Protocol

**Work Package:** RST9  
**Status:** FROZEN AND AUTHORIZED  
**Date:** 2026-10-28  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Documents:**
- `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
- `docs/research_integrity_protocol.md`
- `docs/robust_source_training/rst7_confirmatory_closeout_v1.md`
- `docs/robust_source_training/rst8_siamunet_diff_seed0_closeout_v1.md`

---

## 1. Executive Summary and Scope Boundaries

Following the confirmatory support for Recipe R2 on SNUNet-ECAM in RST7 and the bounded architecture stopping decision in RST8, Work Package RST9 executes the final deployment phase of the ChangeGuard robust source-training route.

RST9 consists of two strictly partitioned stages:
1. **Part A (Deployable Model Training):** Training the final Recipe R2 model on all 27 authoritative source events across seeds 0, 1, and 2 using the SNUNet-ECAM backbone.
2. **Part B (Historical-Target Extension):** Source-only zero-shot inference comparing the 3 R2 models against 3 matched historical R0 models across the 16 historical target events (6 dev events and 10 former test events).

### Strict Boundary Rules
- **Non-Confirmatory Status of Historical Targets:** All 16 historical target events were observed and evaluated in earlier project phases (Weeks 3-5). They must **never** be reported as an unseen confirmatory test set. Part B is strictly an *extension / comparability analysis* against past benchmarks.
- **Zero Target Access During Training:** No historical target image or label data is loaded, sampled, or accessed during Part A training, model selection, or tuning. Part A training operates exclusively on the 27 source events.
- **Recipe Freeze:** Recipe R2 is frozen: M1 (event-balanced sampling) enabled, M2 (training-only class-balanced loss) enabled, M3 (paired SAR augmentation) disabled. No hyperparameter alterations or additional components are permitted.

---

## 2. Part A: Deployable Source-Model Training Specification

### 2.1 Dataset and Partition
- **Training Population:** Exactly the 27 authoritative source events (21,462 patches).
  `[118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427, 470, 492, 496, 497, 498, 502, 518, 554, 555, 567, 1111004, 1111005, 1111006, 1111009, 1111010, 1111011]`
- **Evaluation Events During Training:** None (`[]`). Training runs produce final deployable checkpoints without intra-training validation splits.
- **Execution Class:** `ALL_SOURCE_TRAINING`.

### 2.2 Model Architecture
- **Backbone:** SNUNet-ECAM (`src/models/snunet_wrapper.py`).
- **Input Channels:** 2 (pre-flood VV/VH, post-flood VV/VH).
- **Output Channels:** 3 (Class 0: unflooded land, Class 1: permanent water, Class 2: flood inundated).
- **Parameters:** Exactly 12,034,819 parameters.

### 2.3 Optimization and Training Schedule
- **Epoch Budget:** Exactly 16 fixed epochs.
- **Micro-batch Size:** 8.
- **Gradient Accumulation Steps:** 4.
- **Effective Batch Size:** 32.
- **Optimizer Steps per Epoch:** $\lfloor 21,462 / 32 \rfloor = 670$ steps.
- **Total Optimizer Steps:** $16 \times 670 = 10,720$ steps.
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{weight\_decay} = 0.0$, initial $\text{lr} = 1 \times 10^{-3}$).
- **Seeds:** 0, 1, 2.

### 2.4 Loss Function and M2 Class Weights
- **Loss:** 0.5 Weighted Cross-Entropy + 0.5 Multi-Class Soft Dice ($\epsilon = 1.0$, ignore index 3).
- **M2 Frozen Weights Artifact:** `configs/robust_source_training/rst9_all27_class_weights_v1.json`
- **Artifact SHA-256:** `0a1f38bb285c40b91ce15d8fe2c024523ce9b640fd2351ef79da514a184bda85`
- **Population Pixel Statistics (all 21,462 patches):**
  - Class 0 (Dry land): 793,932,988 pixels (74.6988%)
  - Class 1 (Permanent water): 74,838,191 pixels (7.0413%)
  - Class 2 (Newly inundated flood): 194,074,651 pixels (18.2599%)
  - Total valid pixels: 1,062,845,830
- **Final Normalized Weights:** `[0.496316, 1.544550, 0.959134]`

---

## 3. Matched Baseline R0 Audit

The three historical all-27 source R0 checkpoints from Week 5 were audited and verified for exact scientific compatibility:
1. `outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`  
   SHA-256: `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`
2. `outputs/week5_source_all27/source_snunet_vvvh_all27_seed1_v1_fp32/final_checkpoint.pt`  
   SHA-256: `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a`
3. `outputs/week5_source_all27/source_snunet_vvvh_all27_seed2_v1_fp32/final_checkpoint.pt`  
   SHA-256: `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd`

All three checkpoints match the SNUNet-ECAM architecture, 27 source events training set, 16 epoch budget, Adam optimizer settings, and unweighted cross-entropy loss. These checkpoints are reused directly for matched R0 comparative inference.

---

## 4. Part B: Historical-Target Extension Specification

### 4.1 Target Event Allowlist (16 Events)
- **Dev Events (6):** `[279, 437, 520, 559, 1111003, 1111008]`
- **Former Test Events (10):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`

### 4.2 Inference Protocol
- **Evaluation Mode:** `model.eval()`, `torch.no_grad()`.
- **Positive Class:** Class 2 (newly inundated flood water).
- **Threshold:** Standard argmax ($\text{pred} = \arg\max(\text{logits}, \dim=1) == 2$).
- **Metrics Collected per Event:**
  - True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), True Negatives ($TN$), Valid Pixels.
  - Precision, Recall, $F_1$, IoU.
- **Aggregation:**
  - Grand Event-Macro $F_1$ across all 16 target events.
  - Dev-Macro $F_1$ (6 dev events).
  - Test-Macro $F_1$ (10 former test events).
  - Tail performance: Bottom-3 mean $F_1$ and Worst-Event $F_1$.
  - Pooled pixel metrics.

### 4.3 Total Inference Rows
- 3 R2 checkpoints $\times$ 16 events = 48 rows.
- 3 R0 checkpoints $\times$ 16 events = 48 rows.
- Total event-result rows: 96 rows.

---

## 5. Execution Infrastructure

- **Hardware Allocation:**
  - GPU 0: Reserved / Smokes / Inference.
  - GPU 1: `rst9_r2_snunet_all27_s0_v1` in tmux session `cg_rst9_gpu1`.
  - GPU 2: `rst9_r2_snunet_all27_s1_v1` in tmux session `cg_rst9_gpu2`.
  - GPU 3: `rst9_r2_snunet_all27_s2_v1` in tmux session `cg_rst9_gpu3`.
- **Recovery Policy:** Fresh-run only. If a training run is interrupted, it must be restarted from epoch 0 under clean conditions.

---

## 6. Terminal Verdict Strings

Upon completion of training, inference, and analysis, the report will conclude with exactly one authoritative verdict string:
- `RST9 COMPLETE — DEPLOYABLE MODELS AND POSITIVE HISTORICAL EXTENSION`
- `RST9 COMPLETE — DEPLOYABLE MODELS AND NEGATIVE HISTORICAL EXTENSION`
- `RST9 COMPLETE — DEPLOYABLE MODELS AND INCONCLUSIVE HISTORICAL EXTENSION`
- `RST9 BLOCKED`
