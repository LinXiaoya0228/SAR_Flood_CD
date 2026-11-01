# RST5 Prelaunch Terminology and Resume Addendum

## 1. Executive Summary

This addendum resolves two prelaunch requirements before entering RST6:
1. Formalizes the scientific terminology for Method Component M2 and Recipe R2.
2. Formally audits and tests the 13 required checkpoint, resume, and sampler-progression contract criteria.

---

## 2. Component M2 and Recipe R2 Terminology Clarification

### 2.1 Component M2 Scientific Description
Component M2 is scientifically described as:
```text
training-only inverse-square-root class-balanced cross-entropy
```

### 2.2 Rejection of "Preferential Flood Weighting" Mischaracterization
Component M2 must **not** be described as a loss that "always preferentially weights the flood class" or "specifically targets flood inundation with high weights." 

The class weights $w_c$ are determined strictly and objectively by inverse square-root class frequencies in the training events:
$$w_c = \text{clip}\left(\frac{1 / \sqrt{p_c}}{\frac{1}{3} \sum_{k} 1 / \sqrt{p_k}}, 0.5, 2.0\right)$$
normalized such that $\sum_{c=0}^2 w_c = 3.0$ (mean weight = 1.0).

Empirical evidence across the two confirmation folds proves that M2 does **not** preferentially weight flood water over other classes:
- **Confirmation Fold A** (Train B+C, 13,302 patches):
  - Proportions: $p_0 = 0.821312$ (Dry Ground), $p_1 = 0.067988$ (Permanent Water), $p_2 = 0.110700$ (Flood Water)
  - Resulting Weights: `[0.486491, 1.409167, 1.104342]`
  - Permanent water (Class 1) is the rarest class and receives the highest weight ($1.409$), higher than flood water ($1.104$).
- **Confirmation Fold B** (Train A+C, 14,655 patches):
  - Proportions: $p_0 = 0.687768$ (Dry Ground), $p_1 = 0.090354$ (Permanent Water), $p_2 = 0.221877$ (Flood Water)
  - Resulting Weights: `[0.543519, 1.499552, 0.956929]`
  - In Fold B, flood water is abundant ($22.2\%$), so its weight is actually clipped/normalized to $0.957$ (below average). Permanent water receives the highest weight ($1.500$).

### 2.3 Authoritative Recipe R2 Name
The formal scientific name of Recipe R2 is:
```text
Event-balanced sampling with training-only class-balanced loss
```
The internal codebase identifiers `M2` and `R2` are strictly retained to preserve provenance across configurations, manifests, logs, and Git history.

---

## 3. Resume Verification and Contract Audit

A rigorous audit of `src/robust_source_training/checkpoint.py`, `src/robust_source_training/runner.py`, and `src/robust_source_training/sampler.py` was conducted against the 13 required interruption/resume criteria:

| Criterion | Implementation Status | Technical Finding |
|---|---|---|
| **1. Model parameters** | **VERIFIED** | `model.state_dict()` saved and restored via `assert_states_equal()`. |
| **2. Model buffers** | **VERIFIED** | Non-parameter persistent buffers (e.g. BatchNorm running stats) are included in `state_dict`. |
| **3. Optimizer state** | **VERIFIED** | `optimizer.state_dict()` (Adam momentum and variance tensors) saved and restored bit-identically. |
| **4. Epoch** | **VERIFIED** | Integer `epoch` saved and reloaded. Runner sets `start_epoch = ckpt["epoch"] + 1`. |
| **5. Within-epoch batch position** | **DEFICIENT** | `save_rst_checkpoint()` does not record within-epoch batch index $b\_idx$. Runner resets to batch 0 on resume. |
| **6. Global optimizer step** | **VERIFIED** | `global_step` saved and restored to preserve total parameter update counters. |
| **7. Full RNG state** | **VERIFIED** | Python `random`, NumPy `np.random`, and PyTorch CPU/CUDA RNG states are serialized and restored. |
| **8. Sampler order/progression** | **DEFICIENT** | `checkpoint_dict` does not serialize sampler progression, draw index, or consumed sample indices. |
| **9. No duplicated post-resume samples** | **DEFICIENT** | Because within-epoch batch position is not saved, resuming an interrupted epoch from its start re-executes all previously seen microbatches in that epoch. |
| **10. No skipped post-resume samples** | **DEFICIENT** | Resuming at `start_epoch = epoch + 1` skips all remaining microbatches of an interrupted epoch. |
| **11. R0 patch-uniform continuation** | **DEFICIENT** | PyTorch `DataLoader(..., shuffle=True)` does not support arbitrary sample offset slicing. |
| **12. R2 event-balanced continuation** | **DEFICIENT** | `EventBalancedSampler` has `start_index` in `src/robust_source_training/sampler.py`, but `runner.py` never passes `start_index`, nor is `start_index` saved in checkpoints. |
| **13. Mismatch rejection** | **VERIFIED** | `load_rst_checkpoint()` fails closed on mismatched `config_hash`, `fold_hash`, or `train_event_ids`. |

---

## 4. Contract Violation & Pre-Launch Stop Trigger

The authoritative RST6 protocol establishes the explicit precondition:
> *"If sampler progression is not actually checkpointed or reproducible, stop before launch:*
> ```text
> RST6 BLOCKED -- RESUME CONTRACT INCOMPLETE
> ```"

### Finding:
While epoch-level checkpointing and model/optimizer/RNG restoration function correctly, **intra-epoch within-batch sampler progression is neither serialized nor reproducible in the current runner implementation**. Checkpoints in `runner.py` are saved only at run termination (`epoch == max_epochs`), without intermediate epoch or microbatch persistence.

In strict adherence to the research integrity protocol, formal confirmatory GPU training cannot be launched until this resume contract discrepancy is acknowledged and resolved.

**Status**:
```text
RST6 BLOCKED -- RESUME CONTRACT INCOMPLETE
```
