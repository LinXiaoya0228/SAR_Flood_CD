# ChangeGuard Week 11 Evidence Consistency Protocol (v1)

**Status:** `APPROVED_FOR_CPU_IMPLEMENTATION`  
**Execution Authorizations:**
- `real_data_execution_authorized: false`
- `gpu_execution_authorized: false`

---

## 1. Scope & Scientific Constraints

This protocol specifies the approved minimal candidate adaptation and risk-evidence module for ChangeGuard Week 11A CPU implementation.

### 1.1 Strict Invariants
1. **Model Architecture:** Controlled bi-temporal `SNUNet_ECAM` (VV/VH, 2 input channels, 3 output classes, base channel 32). No architecture changes, no auxiliary heads, no retraining of source weights.
2. **Trainable Scope:** Selected BatchNorm affine parameters (`weight` and `bias`) only. All non-BN weights and biases remain frozen (`requires_grad = False`).
3. **Hyperparameters:**
   - Single-step SGD with momentum: $\text{LR} = 1.25 \times 10^{-4}$, $\text{momentum} = 0.9$, $\text{weight\_decay} = 0.0$.
   - Batch size: 16 (FP32). 1 optimizer step per batch.
   - Confidence threshold: $\tau = 0.90$.
   - Zero source-anchor loss ($\alpha = 0$), zero EMA teacher, zero SAM, zero controller.
4. **Source Independence (R06):**
   - The frozen source model is an independent deepcopy ensuring zero whole-object or shared parameter/buffer storage aliasing with the candidate model.
   - Evaluated strictly in `eval()` mode with persistent running statistics.
5. **Candidate BN Policy:**
   - Active BatchNorm modules are set to `module.train()` with `track_running_stats = False`, `running_mean = None`, `running_var = None`.
   - Forward passes compute batch statistics without mutating persistent running buffers.
   - All non-BN modules remain in `eval()` mode.
6. **Input Validation & Normalization (N01):**
   - Spatial dimensions of $xA$ and $xB$ must match exactly.
   - Sensory valid masks with singleton channel `(B, 1, H, W)` are normalized to `(B, H, W)`. Unsupported shapes or spatial mismatches raise `ValueError` to prevent silent broadcasting.
7. **Numerical Safeguards (R02):**
   - Finite loss and gradient checks occur before optimizer stepping.
   - Post-step parameter finiteness is verified; any corrupting step triggers an immediate rollback to pre-step state and reports `action = "failure"`, `update_applied = False`, and `loss = None`.
8. **Loss Representation (N02):**
   - Skipped updates (empty selection, zero valid pixels, or numerical failure) represent loss as `None`, distinct from a computed `0.0`.
9. **Event Quarantine:** All 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) are strictly quarantined.

---

## 2. Consistency Adaptation Mechanism

The consistency mechanism uses a weak-strong pair:
1. **Weak View (Identity):**
   - Input: $(xA, xB)$ untransformed.
   - Primary evaluation predictions $p^w$ are cached from this pre-update forward pass.
2. **Strong View (Horizontal Flip with Spatial Realignment):**
   - Input: $xA^s = \text{flip}(xA, \text{dim}=-1)$, $xB^s = \text{flip}(xB, \text{dim}=-1)$, $\text{mask}^s = \text{flip}(\text{mask}, \text{dim}=-1)$.
   - Realignment: Strong logits are spatially inverted before loss computation: $\tilde{z}^s = \text{flip}(z^s, \text{dim}=-1)$.
3. **One-Way KL Divergence Loss:**
   - Weak probabilities $p^w$ are detached ($\nabla p^w = 0$).
   - Selection mask: $M_{i} = \mathbb{I}(\max_c p^w_{i,c} \ge 0.90) \land (\text{valid\_mask}_i == 1)$.
   - $\mathcal{L}_{\text{consistency}} = \frac{1}{\sum M_i} \sum_{i \in M} \sum_{c=0}^2 p^w_{i,c} \left(\ln(p^w_{i,c} + 10^{-12}) - \ln \tilde{p}^s_{i,c}\right)$.
   - If $\sum M_i == 0$ or valid pixels $== 0$, the optimizer step is skipped (`update_applied = False`, `action = "skip"`, `loss = None`).

---

## 3. Timing Disclosures (R03)

To resolve conflicting diagnostic declarations:
- **PRE_UPDATE Timing:**
  - Primary batch prediction: Cached from untransformed pre-update candidate pass $p^w$.
  - Sensory-valid normalized Shannon entropy: Computed on $p^w$.
  - Max confidence: Computed on $p^w$.
  - Source / Candidate Disagreement Rate: Computed comparing frozen source predictions with pre-update candidate predictions $p^w$.
- **POST_UPDATE Timing:**
  - Null evidence probes: Evaluated on `(pre2, pre2)` and `(post, post)` using candidate model after the optimizer step.

---

## 4. Evidence Recording & Probe Isolation

All risk indicators are recorded for observation only. **No gating, thresholding, or abstention is executed in this task.**

### 4.1 Recorded Risk Indicators
1. **Sensory-Valid Normalized Three-Class Shannon Entropy (PRE_UPDATE):**
   $$H_{\text{norm}}(p^w) = -\frac{1}{|V| \ln(3)} \sum_{i \in V} \sum_{c=0}^2 p^w_{i,c} \ln(p^w_{i,c} + 10^{-12}), \quad V = \{i : \text{valid\_mask}_i == 1\}$$
2. **Maximum Confidence (PRE_UPDATE):**
   $$C(p^w) = \frac{1}{|V|} \sum_{i \in V} \max_c p^w_{i,c}$$
3. **Source / Candidate Disagreement Rate (PRE_UPDATE):**
   $$D_{\text{pre}} = \frac{1}{|V|} \sum_{i \in V} \mathbb{I}(\arg\max p^{\text{src}}_i \neq \arg\max p^w_i)$$
4. **Both Null Pairs (`(pre2, pre2)` and `(post, post)`) (POST_UPDATE):**
   - Candidate evaluated under candidate BN context on `(xA, xA)` and `(xB, xB)`.
   - Frozen source evaluated under source BN context on `(xA, xA)` and `(xB, xB)`.
   - $\Delta_{\text{pre2}} = \max\left(0.0, C_{2, \text{pre2}}^{\text{cand}} - C_{2, \text{pre2}}^{\text{src}}\right)$
   - $\Delta_{\text{post}} = \max\left(0.0, C_{2, \text{post}}^{\text{cand}} - C_{2, \text{post}}^{\text{src}}\right)$
   - $\text{Score}_{\text{null}} = 0.5 \times (\Delta_{\text{pre2}} + \Delta_{\text{post}})$
   - *Context Disclosure:* Recorded alongside both branches' BN contexts.

### 4.2 State-Neutral Probe Isolation (R01)
- Source evaluation and null probe evaluations must strictly preserve candidate parameters, buffers, optimizer state, and PyTorch/CUDA RNG states.
- Handled with exception-safe `try...finally` restoration of RNG state and registered buffers across all probe and reference forwards.

### 4.3 Interface Exclusions & Calibration Mode U
- **Deferred Interface:** The source-referenced change disappearance/expansion interface is marked explicitly as **DEFERRED**.
- **Mode U Compliance:** Zero development ground truth is accessed during adaptation. No thresholds are fitted, no stream quantiles are computed, and no abstention is triggered.

---

## 5. Output Schema & Execution Safeguards

A dedicated runner interface ([`tools/run_week11_cpu_evidence.py`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/tools/run_week11_cpu_evidence.py)) enforces the CPU execution boundary.

### 5.1 Fail-Closed Authorization Guard (R05)
- Authorization flags in configuration must be actual JSON booleans (`bool`); string representations (e.g. `"false"`) or other malformed types raise `TypeError`.
- Under `APPROVED_FOR_CPU_IMPLEMENTATION`, attempting to select a non-CPU device or enabling GPU/real-data execution immediately raises `PermissionError` before model/device access.

### 5.2 Decoupled Output Schema (R04)
Every evaluation records:
1. `sample_ids` and `event_id` (actual sample IDs, synthetic event identity, and explicit batch-to-sample mapping).
2. Provenance metadata (honest synthetic values: `checkpoint_seed`, `stream_order`, `reset_mode`, `model_architecture`).
3. `valid_pixels`, `selected_pixels`, `predicted_flood_pixels`.
4. Pre-update observation indicators: `entropy_norm`, `max_confidence`, `source_disagreement_rate_pre_update`.
5. Post-update diagnostic indicators: `null_score_primary_post_update`, `null_pre2_delta`, `null_post_delta`.
6. Execution metadata: `action` (`update`, `skip`, `failure`), `update_applied`, `failure_reason`, `loss` (`None` when skipped/failed), `update_time_ms`, `transform_applied`, and candidate/source BN contexts.
7. *Offline Evaluation Block:* `tp`, `fp`, `fn`, `tn`, `flood_f1`, `flood_recall`, `flood_iou` evaluated strictly offline after prediction records are frozen.
