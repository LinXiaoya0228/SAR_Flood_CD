# RST6 R2 Class-Weights Audit and Invalidation Report (v1)

- **Date**: 2026-10-27 (Asia/Singapore)
- **Repository**: `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
- **Scope**: Audit of R2 confirmatory attempts following reported Fold B class-weight discrepancy; code hardening; invalidation of v1 attempts; preparation of versioned v2 retries.

---

## 1. Executive Summary and Incident Timeline

During the initial launch of the RST6 confirmatory matrix, four queues were started across GPUs 0, 1, 2, and 3. An interim status report to the user quoted:

```text
RST5 frozen Fold B class weights:
[0.543519, 1.499552, 0.956929]

Reported chat summary for Fold B:
[0.473523, 1.637568, 1.096417]
```

To preserve research integrity, both R2 queues (GPU 1 and GPU 3) were immediately and conservatively stopped during Epoch 1 (at Steps 56 and 57 respectively), prior to the completion of any epoch or the generation of any checkpoint. The two baseline R0 queues (GPU 0 and GPU 2) were authorized to continue uninterrupted.

---

## 2. Root Cause Analysis

A rigorous forensic comparison was conducted across disk artifacts, configuration files, manifests, logs, and runner source code:

1. **Runtime Artifact on Disk**:
   Inspection of `outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s0_v1/training_class_weights.json` revealed:
   ```json
   "final_weights_list": [0.543519, 1.499552, 0.956929]
   ```
   This is **100% byte-identical** (and SHA-256 identical) to the frozen RST5 artifact `configs/robust_source_training/rst5_foldb_class_weights_v1.json`.
2. **Runtime Log on Disk**:
   Inspection of `logs/robust_source_training/rst6/rst6_r2_snunet_foldb_s0_v1.log` (line 11) recorded:
   ```text
   [2026-10-27 15:49:04] [INFO] M2 class weights computed and saved to ...: [0.543519, 1.499552, 0.956929]
   ```
3. **Discrepancy Origin**:
   The string `[0.473523, 1.637568, 1.096417]` **never existed on disk, was never computed by code, and was never used by the training engine**. It originated entirely as an LLM formatting hallucination in the assistant's previous chat reply, written prior to tailing the Fold B log.
4. **Architectural Vulnerability Identified**:
   While the runtime calculation was empirically correct, `src/robust_source_training/runner.py` recomputed class weights dynamically from `train_ds.samples` without an explicit, fail-closed assertion verifying that the recomputed weights and aggregate counts matched the frozen artifact and SHA-256 hash declared in `provenance_references`.

---

## 3. Independent Verification of Pixel Counts and Weights

The per-class pixel counts and class-weight calculations for both confirmatory folds were independently verified from raw patch labels and masks across all 18 training events per fold:

### 3.1 Fold A (18 Training Events, 13,302 Patches)
- **Training Events**: `[130, 147, 267, 275, 324, 417, 427, 470, 492, 497, 502, 518, 555, 1111004, 1111005, 1111006, 1111010, 1111011]`
- **Valid Pixel Counts**:
  - Class 0 (dry ground / unflooded land): `538,262,742`
  - Class 1 (permanent water): `44,557,222`
  - Class 2 (newly inundated flood water -- evaluation positive class): `72,549,679`
  - Total Valid: `655,369,643`
- **Class Proportions**:
  - Class 0: `0.821312`
  - Class 1: `0.067988`
  - Class 2: `0.110700`
- **RST5 Frozen Formula Application**:
  - Raw inverse-square-root: `[1.103433, 3.835165, 3.005560]`
  - Normalized: `[0.416738, 1.448455, 1.135118]`
  - Clipped `[0.5, 2.0]`: `[0.500000, 1.448455, 1.135118]`
  - Final Normalized Weights: `[0.486491, 1.409167, 1.104342]`
  - **Match with `rst5_folda_class_weights_v1.json`**: Exact.

### 3.2 Fold B (18 Training Events, 14,655 Patches)
- **Training Events**: `[118, 130, 147, 174, 273, 275, 421, 427, 492, 496, 497, 498, 554, 567, 1111005, 1111006, 1111009, 1111011]`
- **Valid Pixel Counts**:
  - Class 0 (dry ground / unflooded land): `501,089,528`
  - Class 1 (permanent water): `65,829,766`
  - Class 2 (newly inundated flood water -- evaluation positive class): `161,653,831`
  - Total Valid: `728,573,125`
- **Class Proportions**:
  - Class 0: `0.687768`
  - Class 1: `0.090354`
  - Class 2: `0.221877`
- **RST5 Frozen Formula Application**:
  - Raw inverse-square-root: `[1.205810, 3.326790, 2.122969]`
  - Normalized: `[0.543519, 1.499552, 0.956929]`
  - Clipped `[0.5, 2.0]`: `[0.543519, 1.499552, 0.956929]`
  - Final Normalized Weights: `[0.543519, 1.499552, 0.956929]`
  - **Match with `rst5_foldb_class_weights_v1.json`**: Exact.

---

## 4. Implementation: Fail-Closed Pre-Compute Weight Assertion

To permanently eliminate any possibility of divergence or unnoticed discrepancy:
1. `src/robust_source_training/runner.py` was updated in `validate_only()` to verify that all confirmatory R2 configurations specify `class_weights_path` and `class_weights_sha256`, and that the file on disk matches the SHA-256 hash.
2. In `execute()`, M2 class-weight computation is now performed **strictly before model instantiation**.
3. A fail-closed assertion verifies that:
   - `m2_weights_list == frozen_weights_list` (exact floating-point match rounded to 6 decimals);
   - `m2_stats["aggregate_counts"] == frozen_cw_data["aggregate_counts"]`.
   If any discrepancy is detected, execution raises `ContractViolationError` and terminates before GPU memory allocation or model parameter initialization.

---

## 5. Invalidation and Permanent Preservation of R2 v1 Attempts

The two interrupted R2 v1 runs were preserved as negative/interrupted evidence in accordance with the RST6 Fresh-Run Recovery Policy:

1. `outputs/robust_source_training/rst6/rst6_r2_snunet_folda_s0_v1/`:
   - `terminal_status.json`: `{"status": "INTERRUPTED_RUN", "exit_code": 1}`
   - `invalidation_record_v1.md`: written and preserved.
2. `outputs/robust_source_training/rst6/rst6_r2_snunet_foldb_s0_v1/`:
   - `terminal_status.json`: `{"status": "INTERRUPTED_RUN", "exit_code": 1}`
   - `invalidation_record_v1.md`: written and preserved.

Neither directory will be modified or reused.

---

## 6. Versioned R2 v2 Retry Configurations

Six fresh, versioned configuration files have been prepared under `configs/robust_source_training/rst5_confirmation/`:

| Config File | Run ID | Fold | Seed | Output Root | Log File |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `rst6_r2_snunet_folda_s0_v2.json` | `rst6_r2_snunet_folda_s0_v2` | Fold A | 0 | `outputs/.../rst6_r2_snunet_folda_s0_v2` | `logs/.../rst6_r2_snunet_folda_s0_v2.log` |
| `rst6_r2_snunet_folda_s1_v2.json` | `rst6_r2_snunet_folda_s1_v2` | Fold A | 1 | `outputs/.../rst6_r2_snunet_folda_s1_v2` | `logs/.../rst6_r2_snunet_folda_s1_v2.log` |
| `rst6_r2_snunet_folda_s2_v2.json` | `rst6_r2_snunet_folda_s2_v2` | Fold A | 2 | `outputs/.../rst6_r2_snunet_folda_s2_v2` | `logs/.../rst6_r2_snunet_folda_s2_v2.log` |
| `rst6_r2_snunet_foldb_s0_v2.json` | `rst6_r2_snunet_foldb_s0_v2` | Fold B | 0 | `outputs/.../rst6_r2_snunet_foldb_s0_v2` | `logs/.../rst6_r2_snunet_foldb_s0_v2.log` |
| `rst6_r2_snunet_foldb_s1_v2.json` | `rst6_r2_snunet_foldb_s1_v2` | Fold B | 1 | `outputs/.../rst6_r2_snunet_foldb_s1_v2` | `logs/.../rst6_r2_snunet_foldb_s1_v2.log` |
| `rst6_r2_snunet_foldb_s2_v2.json` | `rst6_r2_snunet_foldb_s2_v2` | Fold B | 2 | `outputs/.../rst6_r2_snunet_foldb_s2_v2` | `logs/.../rst6_r2_snunet_foldb_s2_v2.log` |

All 6 configs preserve identical scientific settings (architecture, loss, optimizer, learning rate, effective batch size 32, 16 epochs, random seeds). All 6 currently have `launch_authorized: false` and are ready for explicit user authorization.

---

## 7. Status of R0 Baseline Runs

Baseline R0 runs continue without pause or modification:
- GPU 0 (`rst6_r0_snunet_folda_s0_v1`): Running normally.
- GPU 2 (`rst6_r0_snunet_foldb_s0_v1`): Running normally.

