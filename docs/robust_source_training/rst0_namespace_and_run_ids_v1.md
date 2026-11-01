# RST Namespace, Directory Structure, and Run-ID Conventions

**Document version:** 1.0.0
**Effective date:** 2026-10-26
**Status:** Approved convention for robust source training

---

## 1. Directory Structure and Namespaces

All files and execution artifacts generated for the robust source-training route must reside strictly within dedicated namespaces to prevent collision with historical E0–E11 assets.

```text
configs/robust_source_training/
docs/robust_source_training/
outputs/robust_source_training/<rst_block>/<run_id>/
logs/robust_source_training/<rst_block>/<run_id>.log
```

### Namespace Rules
1. **Planned directories only:** Directory paths in `outputs/` and `logs/` are created only at the moment a real, authorized run is launched. No empty directories are created in advance.
2. **Historical isolation:** No file under `configs/robust_source_training/` or `outputs/robust_source_training/` may point to or modify any historical path under `configs/changeguard/`, `outputs/e11_final_test_v1/`, or earlier historical experiment folders.

---

## 2. Standardized Run-ID Template

Every execution (training, evaluation, validation, or smoke run) must have a unique, immutable Run ID conforming to the following template:

```text
rst{block}_{purpose}_{backbone}_{fold}_{seed}_v{version}
```

### Token Definitions

| Token | Allowed Values / Format | Description |
| :--- | :--- | :--- |
| `{block}` | Integer `0` to `11` | The governing RST block (e.g., `2` for infrastructure tests, `4` for discovery, `6` for confirmation). |
| `{purpose}` | Short descriptive string (e.g., `smoke`, `baseline_r0`, `bal_r1`, `loss_r2`, `full_r3`, `eval`) | The specific objective or recipe of the run. |
| `{backbone}` | `snunet` or `siamunet` | Target network architecture (`snunet` for SNUNet-ECAM, `siamunet` for SiamUnet_diff). |
| `{fold}` | `fold_c` (discovery), `fold_a`, `fold_b` (confirmatory), `all27` (deployable), or `synthetic` | The event partition or fold being evaluated/trained. |
| `{seed}` | `seed0`, `seed1`, `seed2` | Fixed pseudorandom seed. |
| `{version}` | Integer starting at `1` (e.g., `v1`, `v2`) | Version number for tracking attempts and retries. |

### Illustrative Examples (No Run May Be Launched at RST0)

- Discovery Baseline on Fold C: `rst4_baseline_r0_snunet_fold_c_seed0_v1`
- Discovery Full Method on Fold C: `rst4_full_r3_snunet_fold_c_seed0_v1`
- Confirmatory Fold A Seed 1: `rst6_confirm_full_snunet_fold_a_seed1_v1`
- Retry Following Hardware Interruption: `rst6_confirm_full_snunet_fold_a_seed1_v2`

---

## 3. Operational Integrity Principles

### 3.1 Immutability of Run IDs
Once assigned to an execution attempt, a Run ID is permanent and immutable. It may never be reused, reassigned, or modified.

### 3.2 Strict No-Overwrite Policy
Under no circumstances may an existing output directory or log file be overwritten. If a run crashes, is interrupted, or fails validation:
- The partial artifacts and logs of that run must remain completely intact.
- Any subsequent attempt must increment the version token (e.g., from `_v1` to `_v2`) and record the predecessor run ID in its manifest.

### 3.3 Failed-Attempt Preservation
All failed, aborted, or divergent attempts must be preserved in place with their full log streams and diagnostic dumps. Negative or failed runs carry equal scientific importance and must never be deleted, hidden, or purged.

### 3.4 Terminal Status Files
Upon completion, termination, or failure, every run must write an atomic status file:
```text
outputs/robust_source_training/<rst_block>/<run_id>/terminal_status.json
```
This status file records the terminal exit code, final wall-clock timestamp, error trace (if applicable), and summary metrics.

### 3.5 Manifest-Before-Execution Rule
Before starting training or evaluation loops, every runner must serialize a complete `preflight_manifest.json` to the run directory. This manifest must record:
- Full Git commit hash and working tree status (`git status --porcelain`);
- Checksums (SHA-256) of all active configuration files, data split definitions, and initial checkpoint weights;
- Execution start time (UTC ISO 8601);
- Exact command-line invocation arguments;
- Python and PyTorch runtime environment specifications;
- Physical GPU hardware identity (device model, UUID, CUDA compute capability, and driver version).

### 3.6 Hardware Attribution
For all future GPU runs, the execution manifest and log must record the physical GPU identifier (e.g., `GPU-UUID`, `pci_bus_id`, `cuda:X`). Training jobs must not share physical GPUs.

### 3.7 English-Only Durable Repository Records
All newly created configuration files, documentation, code comments, durable logs, JSON keys, and commit messages must be strictly in English. No Chinese characters are permitted in new route artifacts.

---

## 4. Execution Lifecycle States

Every planned or executed run must be explicitly categorized into one of the following formal states:

| Lifecycle State | Definition and Operational Boundary |
| :--- | :--- |
| `DRAFT` | Initial specification; config is under development and not authorized for execution. |
| `APPROVED_NOT_EXECUTED` | Formal protocol and config are frozen and approved; waiting for execution resources. |
| `RUNNING` | Process is actively executing; output directory and preflight manifest exist. |
| `COMPLETED` | Process terminated with exit code 0; all required outputs and terminal status file exist. |
| `FAILED` | Process crashed, aborted, or terminated with non-zero exit code; logs preserved. |
| `INVALIDATED` | Run completed technically, but a post-run audit revealed protocol violation or data error. |

### Distinction Between `COMPLETED` and Scientifically Accepted
The state `COMPLETED` indicates purely operational and computational completion. **`COMPLETED` must never be used as a synonym for scientifically accepted, verified, or successful.** A method is scientifically accepted only when its final audited results satisfy the frozen promotion criteria (P1, P2, or P3).
