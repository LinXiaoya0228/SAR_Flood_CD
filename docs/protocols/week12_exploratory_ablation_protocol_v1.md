# ChangeGuard Week 12 Exploratory Ablation Protocol (v1)

**Protocol Identifier:** `week12_exploratory_ablation_dev6_v1`  
**Date:** 2026-10-19  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Governing Closeout:** [`docs/week11_closeout.md`](../week11_closeout.md)  
**Status:** **FROZEN FOR EXPLORATORY EXECUTION**  
**Evidence Level:** **EXPLORATORY (Single-Seed, Development-Only)**  

---

## 1. Research Objective & Scientific Scope

The objective of Week 12 is to execute a tightly controlled, minimal exploratory ablation isolating the specific contribution of confidence-thresholded weak–strong consistency adaptation against source-only and matched entropy-minimization (Tent) baselines.

### Critical Scope Boundaries:
1. **Exploratory Status:** All results generated under this protocol are strictly **EXPLORATORY**. They do not constitute three-seed confirmatory evidence or claimed generalization proof.
2. **Development Events Only:** Restricted strictly to the 6 approved development events: `[279, 437, 520, 559, 1111003, 1111008]`.
3. **Quarantine:** All 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain strictly forbidden and quarantined.
4. **Frozen Source Checkpoint:** Single seed 0 checkpoint (`outputs/week5_source_all27/source_snunet_vvvh_all27_seed0_v1_fp32/final_checkpoint.pt`), SHA-256 `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`. No source retraining.
5. **No Hyperparameter Search / No Tuning:** Frozen hyperparameters throughout; no post-hoc threshold searching or learning-rate sweeping.

---

## 2. Compared Experimental Conditions

Exactly four experimental conditions are evaluated. All conditions operate on identical dataloader pipelines, identical sample IDs, identical ascending within-event ordering, and identical batch boundaries ($B=16$ with natural tail batches).

### Condition 1: `source_only`
- **Description:** Non-adapted baseline. The model remains strictly in evaluation mode (`eval()`) with frozen BatchNorm running statistics.
- **Updates:** Zero optimizer updates applied ($\text{lr} = 0$, optimizer step bypassed).
- **Primary Predictions:** Untransformed forward pass $z = f(x_A, x_B)$.

### Condition 2: `matched_tent_pre_update_v1`
- **Description:** Entropy minimization baseline matched to ChangeGuard's architecture and optimizer conventions.
- **Trainable Parameters:** BatchNorm affine parameters only (`weight`, `bias`); all other layers frozen.
- **BatchNorm Policy:** Non-BN modules in `eval()`, BN modules in `train()` with `track_running_stats=False`.
- **Adaptation Loss:** Shannon entropy over sensory-valid pixels only:
  $$\mathcal{L}_{\text{Tent}} = -\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sum_{c=0}^2 p_{i, c} \ln(p_{i, c} + 10^{-12})$$
  where $\mathcal{V} = \{i : \text{MK0\_MNA}_i = 1\}$. Strictly zero ground-truth label access during adaptation.
- **Optimizer:** SGD, learning rate $1.25 \times 10^{-4}$, momentum $0.9$, weight decay $0.0$, FP32, 1 step per batch.
- **Primary Prediction Timing:** **PRE_UPDATE**. Predictions are captured from the forward pass *before* applying the gradient update for that batch.
  > [!IMPORTANT]
  > **Timing Distinction from Week 8 Tent:** In Week 8 (`src/tta/tent.py`), Tent was evaluated under transductive *post-update* timing (a second forward pass following `optimizer.step()`). In Week 12, all four conditions are unified under the strictly causal *pre-update* timing convention. To avoid misleading claims of exact equivalence to historical Week 8 results, this condition is designated `matched_tent_pre_update_v1`.

### Condition 3: `C1_consistency_no_conf` (Weak–Strong Consistency without Confidence Filter)
- **Description:** Paired weak–strong temporal consistency learning without confidence-based pixel selection.
- **Views:**
  - Weak view: Untransformed paired inputs $x_A, x_B$.
  - Strong view: Paired horizontal flip $\text{flip\_h}(x_A), \text{flip\_h}(x_B)$ with flipped sensory mask $\text{flip\_h}(\text{MK0\_MNA})$.
- **Target:** Detached weak predictive distribution:
  $$p_{\text{weak}} = \text{stop\_gradient}(\text{softmax}(z_{\text{weak}}))$$
- **Spatial Alignment:** Realignment of strong logits via horizontal flip inversion:
  $$z_{\text{strong}}^{\text{aligned}} = \text{flip\_h}(z_{\text{strong}})$$
- **Pixel Selection Mask:** Strictly sensory-valid pixels:
  $$\mathcal{M}_{\text{C1}} = \{i : \text{MK0\_MNA}_i = 1\}$$
  Includes 100% of sensory-valid pixels. No confidence thresholding.
- **Loss:** One-way Kullback-Leibler divergence:
  $$\mathcal{L}_{\text{C1}} = \frac{1}{|\mathcal{M}_{\text{C1}}|} \sum_{i \in \mathcal{M}_{\text{C1}}} D_{\text{KL}}\left(p_{\text{weak}, i} \parallel \text{softmax}(z_{\text{strong}, i}^{\text{aligned}})\right)$$
- **Optimizer:** SGD, learning rate $1.25 \times 10^{-4}$, momentum $0.9$, weight decay $0.0$, FP32, 1 step per batch.
- **Primary Prediction Timing:** **PRE_UPDATE** ($z_{\text{weak}}$).

### Condition 4: `C2_consistency_conf90` (Weak–Strong Consistency with Confidence $\ge 0.90$)
- **Description:** ChangeGuard core candidate: Paired weak–strong temporal consistency filtered by weak-target confidence.
- **Identical to C1 in all aspects, EXCEPT pixel selection:**
  $$\mathcal{M}_{\text{C2}} = \{i : \text{MK0\_MNA}_i = 1 \land \max_{c} p_{\text{weak}, i, c} \ge 0.90\}$$
- **Empty Selection Policy:** If $|\mathcal{M}_{\text{C2}}| = 0$, the optimizer step is skipped, parameters are preserved, and $\text{loss} = \text{None}$.
- **Computational Note:** Confidence masking filters which pixels enter the loss reduction; it does not eliminate the forward/backward passes. Runtime per batch and per event is explicitly logged to quantify efficiency.
- **Primary Prediction Timing:** **PRE_UPDATE** ($z_{\text{weak}}$).

---

## 3. Dataset, Stream & Ordering Specification

- **Dataset Root:** `../datasets/KuroSiwo`
- **Manifest:** `../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz` (`SHA-256: dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad`)
- **Event Sequence (`order_a`):**
  $$\text{Event Sequence} = [279, 437, 520, 559, 1111003, 1111008]$$
- **Within-Event Ordering:** Deterministic sorted alphanumeric ascending `sample_id`.
- **Sample Population by Event:**
  - Event 279: 2,381 samples (149 batches: 148 $\times 16$ + 1 $\times 13$)
  - Event 437: 169 samples (11 batches: 10 $\times 16$ + 1 $\times 9$)
  - Event 520: 92 samples (6 batches: 5 $\times 16$ + 1 $\times 12$)
  - Event 559: 42 samples (3 batches: 2 $\times 16$ + 1 $\times 10$)
  - Event 1111003: 19 samples (2 batches: 1 $\times 16$ + 1 $\times 3$)
  - Event 1111008: 123 samples (8 batches: 7 $\times 16$ + 1 $\times 11$)
  - **Total:** 2,826 samples per stream (179 batches per condition).
- **Reset Mode (`reset_per_event`):**
  - Before the first batch of every event, the model parameters, running buffers, and optimizer momentum/step counters are restored bit-identically to the source checkpoint.

---

## 4. Evaluation & Integrity Invariants

1. **Strict Offline Evaluation:** Ground-truth masks (`MK0_MLU`) are accessed strictly *after* all predictions and adaptation diagnostics for an event are persisted to disk.
2. **Canonical Double-Masking:**
   $$\text{Evaluation Mask} = (\text{MK0\_MNA} == 1) \land (\text{MK0\_MLU} \ne 3)$$
3. **Confusion Conservation:**
   $$\text{TP} + \text{FP} + \text{FN} + \text{TN} \equiv \text{valid\_pixels}$$
   Verified for every sample, step, and event.
4. **Durable Phase-Aware Progress:**
   - Atomic progress persistence (`adaptation_progress.json`) after every completed batch.
   - Separate recording of `selected_pixel_fraction` from `batches_updated`/`batches_skipped`.
5. **No Overwrite Policy:** The output directory must not exist prior to launch; `--allow_overwrite` is strictly prohibited.
