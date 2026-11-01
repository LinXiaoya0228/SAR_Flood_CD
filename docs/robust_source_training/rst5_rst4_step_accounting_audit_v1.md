# RST4 Optimizer-Step Accounting and Gradient-Accumulation Audit

## 1. Executive Summary

This audit investigates the apparent discrepancy between the reported optimizer-step counts in RST4:
- Theoretical / Reported in config: **7,488** steps
- Observed in manifests / logs: **7,496** steps

### Audit Verdict:
1. **Zero Mismatch Across Recipes in Reality**:
   All four RST4 discovery runs (R0, R1, R2, R3) executed the **exact same number of parameter updates**: **7,496 optimizer steps**.
   All four runs observed the **exact same number of training samples**: **239,472 samples** (14,967 samples per epoch across 16 fixed epochs).
2. **Root Cause of the 7,496 vs 7,488 Discrepancy**:
   - The 7,488 figure was an *a priori* analytical estimate based on 468 steps per epoch (`14967 / 32 = 467.71875 -> 468` steps; `468 * 16 = 7488`).
   - The actual runner code in `src/robust_source_training/runner.py` tracked `total_microbatches` across the entire run without resetting the accumulation counter at epoch boundaries (`total_microbatches = 0` initialized once at runner entry).
   - In each epoch of 14,967 samples with microbatch size 8, there are 1,871 microbatches (`1870 * 8 = 14960`, plus 1 tail batch of 7 samples).
   - With `accum_steps = 4`, `1871 = 467 * 4 + 3`.
   - At the final batch of Epoch 1 (`b_idx = 1870`), the tail condition `or (b_idx + 1 == len(epoch_loader))` flushed the 3 accumulated microbatches as Step 468.
   - On the first batch of Epoch 2 (`b_idx = 0`), `total_microbatches` reached 1,872. Because `1872 % 4 == 0`, `optimizer.step()` was triggered after only 1 microbatch, creating an extra step at the boundary.
   - This boundary step occurred across 8 epoch transitions, yielding `467 * 16 + 16 + 8 = 7,496` optimizer steps in all four runs.
3. **Scientific Impact**:
   Because all four recipes (R0, R1, R2, R3) ran the identical runner and experienced the identical 7,496 steps and 239,472 samples, the discovery staircase comparison was **strictly matched, fair, and valid**.
4. **Resolution for Confirmatory Training (RST5 / RST6)**:
   The accumulation logic is sanitized in `src/robust_source_training/runner.py` by resetting the microbatch accumulation counter whenever `optimizer.step()` is invoked. This eliminates inter-epoch accumulation bleed, guaranteeing that both R0 and R2 execute identical, clean tail flushes with exactly 468 steps per epoch and zero cross-epoch contamination.

---

## 2. Quantitative Trace of RST4 Discovery Runs

Manifest inspection on disk (`outputs/robust_source_training/rst4/*/final_manifest.json`):

| Recipe ID | Run ID | Total Samples Observed | Total Microbatches | Total Optimizer Steps | Terminal Status |
|---|---|---|---|---|---|
| **R0** | `rst4_r0_snunet_foldc_s0_v2` | 239,472 | 29,936 | 7,496 | `COMPLETED_SCIENTIFIC_RUN` |
| **R1** | `rst4_r1_snunet_foldc_s0_v2` | 239,472 | 29,936 | 7,496 | `COMPLETED_SCIENTIFIC_RUN` |
| **R2** | `rst4_r2_snunet_foldc_s0_v2` | 239,472 | 29,936 | 7,496 | `COMPLETED_SCIENTIFIC_RUN` |
| **R3** | `rst4_r3_snunet_foldc_s0_v2` | 239,472 | 29,936 | 7,496 | `COMPLETED_SCIENTIFIC_RUN` |

### Detailed Epoch Breakdown:
- **Dataset Partition**: Groups A + B (18 events, 14,967 patches).
- **Microbatch Size**: 8.
- **Full Microbatches**: 1,870 batches of 8 samples = 14,960 samples.
- **Tail Microbatch**: 1 batch of 7 samples = 7 samples.
- **Microbatches per Epoch**: 1,871.
- **Total Microbatches in 16 Epochs**: `1871 * 16 = 29,936`.
- **Total Samples in 16 Epochs**: `14967 * 16 = 239,472`.

---

## 3. Mathematical Mechanism of the 8-Step Boundary Bleed

In `src/robust_source_training/runner.py`:
```python
total_microbatches = 0
for epoch in range(start_epoch, max_epochs + 1):
    for b_idx, batch in enumerate(epoch_loader):
        ...
        total_microbatches += 1
        if total_microbatches % accum_steps == 0 or (b_idx + 1 == len(epoch_loader)):
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
```

Tracing `total_microbatches % 4`:
- **Epoch 1**:
  - Batches 0 to 1867: 467 steps at multiples of 4 (`total_microbatches` 4, 8, ..., 1868).
  - Batches 1868, 1869, 1870: `total_microbatches` = 1869, 1870, 1871.
  - At `b_idx = 1870`, `b_idx + 1 == 1871`, so tail condition triggers Step 468 (accumulated 3 microbatches).
- **Epoch 2**:
  - At `b_idx = 0`, `total_microbatches` becomes 1,872.
  - Since `1872 % 4 == 0`, `total_microbatches % 4 == 0` evaluates to True!
  - Step 469 triggers immediately after only 1 microbatch.
  - Then batches 1 to 4 trigger after 4 microbatches (total 1,876), and so on.

The sequence of steps per epoch was:
`[468, 469, 469, 468, 468, 469, 469, 468, 468, 469, 469, 468, 468, 469, 469, 468]`
Sum = **7,496 steps**.

---

## 4. Confirmatory Tail-Gradient Sanitation for RST5 / RST6

To ensure completely isolated, predictable tail-gradient flushes per epoch during confirmatory training:
1. Maintain an epoch-local accumulation counter:
   `accum_microbatches = 0`
2. Increment `accum_microbatches += 1` on each microbatch.
3. Trigger condition:
   `if accum_microbatches == accum_steps or (b_idx + 1 == len(epoch_loader)):`
4. Reset on update:
   `optimizer.step()`
   `optimizer.zero_grad()`
   `accum_microbatches = 0`
   `global_step += 1`

With this update:
- Every epoch executes exactly `ceil(M / accum_steps)` steps.
- For 1,871 microbatches and `accum_steps = 4`, every epoch executes exactly 468 steps (`467` full 4-microbatch steps + `1` clean 3-microbatch tail flush).
- 16 epochs will execute exactly `468 * 16 = 7,488` optimizer steps.
- R0 and R2 will share bit-identical tail-gradient semantics across every epoch.

---

## 5. Audit Conclusion

The step accounting audit is **resolved**:
- No invalidation of RST4 discovery results: all four models received identical compute budgets, identical sample exposures, and identical update counts (7,496).
- The discrepancy was an analytical vs realized reporting discrepancy resulting from inter-epoch accumulator bleed.
- The sanitized accumulator implementation ensures exact, reproducible, and matched step budgets for RST5 confirmation.
