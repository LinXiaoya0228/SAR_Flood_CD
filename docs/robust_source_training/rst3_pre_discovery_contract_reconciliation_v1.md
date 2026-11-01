# RST3 Pre-Discovery Contract Reconciliation Record

**Date**: 2026-10-26  
**Status**: COMPLETED AND FROZEN BEFORE RST4 DISCOVERY  
**Scope**: Reconciliation of M2 loss formula documentation and M3 augmentation gain distribution prior to launching RST4 discovery staircase.

---

## 1. Executive Summary

Prior to launching any compute or evaluation under **RST4 (Discovery Staircase)**, a comprehensive pre-discovery audit of code, contracts, protocols, tests, and smoke artifacts was conducted. Two inconsistencies between documentation and code specifications were identified and reconciled:
1. **M2 Formula Inconsistency**: The RST3 closeout document recorded an inaccurate notation for M2 class weighting ($w'_c = \text{clip}(\sqrt{\min(p)/p_c}, 0.1, 10.0)$), whereas the implementation in `src/robust_source_training/loss.py` and the protocol in `docs/robust_source_training/rst3_component_protocol_v1.md` correctly implemented the frozen inverse square-root formula with unit-mean clipping and normalization, which produced the exact reported smoke weights `[0.474823, 0.960079, 1.565099]`.
2. **M3 Gain Distribution Reconciliation**: The M3 common-mode radiometric gain was reconciled to the approved conservative distribution $\text{gain} \sim \mathcal{U}(0.95, 1.05)$ across code, contract, protocol, and tests, replacing the previously drafted log-uniform $[0.90, 1.10]$ distribution.

Zero RST4 compute, training, or evaluation has taken place prior to this reconciliation.

---

## 2. Issue 1: M2 Formula Reconciliation

### A. Inconsistency Identified
In `docs/robust_source_training/rst3_closeout_v1.md`, the mathematical formula was written as:
$$w'_c = \text{clip}\left(\sqrt{\frac{\min_{k} p_k}{p_c}}, 0.1, 10.0\right), \quad w_c = \frac{w'_c}{\frac{1}{3} \sum_{k=0}^2 w'_k}$$
However, the recorded smoke weights on Events 273 and 518 ($N_{\text{valid}} = 2,976,830$; counts = `[2573977, 292708, 110145]`) were:
$$w = [0.474823, 0.960079, 1.565099]$$

### B. Actual Code Behavior
Inspection of `src/robust_source_training/loss.py` (lines 105-117) revealed that the code was executing the authoritative frozen recipe:
```python
p = counts_arr / float(total_valid)
raw_w = 1.0 / np.sqrt(np.maximum(p, float(epsilon)))
norm_w = raw_w / float(np.mean(raw_w))
clipped_w = np.clip(norm_w, float(clip_min), float(clip_max))
final_w = clipped_w / float(np.mean(clipped_w))
```
with `clip_min = 0.5, clip_max = 2.0, epsilon = 1e-12`.

Evaluating this exact sequence on the smoke counts yields:
- $p = [0.86467047, 0.09832876, 0.03700077]$
- $\text{raw\_w} = [1.07541152, 3.18903819, 5.19869841]$
- $\text{norm\_w} = [0.34092614, 1.01098646, 1.64808740]$
- $\text{clipped\_w} = [0.50000000, 1.01098646, 1.64808740]$ (clipping lower bound $0.5$ engaged)
- $\text{final\_w} = [0.47482271, 0.96007865, 1.56509864] \to [0.474823, 0.960079, 1.565099]$

Furthermore, in `RobustSourceLoss`:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \text{CE}_{\text{weighted}}(w) + 0.5 \cdot \text{Dice}_{\text{macro}}$$
where macro soft-Dice is completely unweighted across classes 0, 1, and 2.

### C. Resolution
The implementation in `src/robust_source_training/loss.py` was **correct** from the beginning. Only the documentation notation in `docs/robust_source_training/rst3_closeout_v1.md` was inaccurate. No loss code change was required, and no rerun of bounded component smokes was necessary. A dedicated regression test `test_m2_exact_operation_order_and_recorded_smoke_weights` was added to `tests/test_rst3_components.py` to lock this exact operation order.

---

## 3. Issue 2: M3 Gain Distribution Reconciliation

### A. Inconsistency Identified
The initial component contract (`rst3_component_contract_v1.json`) specified $\log(\text{gain}) \sim \text{Uniform}(\log(0.90), \log(1.10))$, whereas the pre-discovery physical consensus approved a conservative bounded transform $\text{gain} \sim \text{Uniform}(0.95, 1.05)$ ($1 \pm 0.05$).

### B. Approved Pre-Discovery Specification
For all RST4 runs, M3 common-mode radiometric gain is frozen as:
$$\text{gain} \sim \mathcal{U}(0.95, 1.05)$$
Invariants:
- Sampled once per paired sample from deterministic sample seed.
- Applied identically to pre-event and post-event backscatter.
- Applied identically to VV and VH channels.
- Applied in linear-amplitude space before clipping $[0.0, 0.15]$ and standardization.
- Disabled unconditionally during evaluation.

### C. Code and Contract Updates
1. `src/robust_source_training/augmentation.py`: Default `min_gain = 0.95, max_gain = 1.05`; sampling implemented as `gain = rng.uniform(self.min_gain, self.max_gain)`.
2. `src/robust_source_training/runner.py`: Default `min_gain = 0.95, max_gain = 1.05`.
3. `configs/robust_source_training/rst3_component_contract_v1.json`: Updated `radiometric_gain` to `{"distribution": "uniform", "min_gain": 0.95, "max_gain": 1.05}`.
4. `docs/robust_source_training/rst3_component_protocol_v1.md`: Stated $\text{gain} \sim \text{Uniform}(0.95, 1.05)$.
5. `tests/test_rst3_components.py`: Added `test_m3_conservative_uniform_gain_distribution` to verify 1,000 empirical draws fall strictly within $[0.95, 1.05]$ with mean $1.00 \pm 0.01$.

---

## 4. Evidence of Pre-Discovery Timing

- No training runs for RST4 have been launched.
- No outputs exist under `outputs/robust_source_training/rst4/`.
- No evaluation metrics on Group C discovery events have been computed or observed.
- This reconciliation is frozen and committed before any RST4 code execution.

---

## 5. Affected File Hashes

| File | SHA-256 Hash |
| :--- | :--- |
| `src/robust_source_training/augmentation.py` | `8dad74ae3ca0e06ceafae12047c7dbda034e098de310b2cc115b9a63a535745a` |
| `src/robust_source_training/runner.py` | `e15dbd49c6865b50495e0b16a663588dc5b987a89448a2ad6866756a9a4e7b59` |
| `configs/robust_source_training/rst3_component_contract_v1.json` | `8992fff8cd0b4796aa89a1cbb370444143785120e1ec0bcb2487ca2b606a865d` |
| `docs/robust_source_training/rst3_component_protocol_v1.md` | `c33b0c89700da22747b8e94695ba8ecb7bb4cb3af1279ec661ced32fc9bfc589` |
| `docs/robust_source_training/rst3_closeout_v1.md` | `186727f3bfb31d450205d67d8b35e08be0c9eefc6aa4c4cb21a8c56281f701a0` |
| `tests/test_rst3_components.py` | `7e4ef701626cf294583b3a02d0c7af5e328f6c664ccf813b932b735a38c0c7f9` |
| `docs/robust_source_training/rst3_pre_discovery_contract_reconciliation_v1.md` | *(This document)* |

---

## 6. Verdict

**`RST3 CONTRACT RECONCILIATION COMPLETE — AUTHORIZED FOR RST4 FREEZE`**

