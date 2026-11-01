# ChangeGuard Research Integrity & Scientific Rigor Protocol

**Document Version:** 1.0.0  
**Effective Date:** 2026-09-25  
**Scope:** All experimental runs, model evaluations, and Test-Time Adaptation (TTA) algorithms within the ChangeGuard project.

---

## 1. Purpose and Philosophy

In test-time adaptation and change-detection benchmarks, unsupervised claims are easily compromised by subtle forms of data leakage, post-hoc hyperparameter tuning on test distributions, manual metric cherry-picking, and silent sample omissions.

This protocol establishes permanent, non-negotiable operational boundaries for the ChangeGuard project to guarantee absolute research integrity, transparency, and scientific reproducibility.

---

## 2. Mandatory Protocol Requirements

### 2.1 Data Boundaries
1. **Split Exclusivity**:
   - **Training Events**: May be used exclusively for source-model training, source parameter estimation, or computing training-set statistics.
   - **Validation Events** (`[279, 437, 520, 559, 1111003, 1111008]`): Available for diagnostic audits, pilot adaptation experiments, baseline evaluations, and development hyperparameter selection.
   - **Sealed Test Events** (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`): Must remain **strictly sealed and untouched**. No script, loader, evaluator, or diagnostic tool may load, query, pre-calculate, or inspect any raster or sample from sealed test events until all methods, architectures, loss functions, and hyperparameters are irrevocably frozen.
2. **Access Tracking**:
   - Every experiment must programmatically declare and verify the list of accessed event IDs.
   - Any execution attempting to access a sealed test event during development will fail immediately.

### 2.2 Strict Label Isolation
Test-time adaptation algorithms claim to be unsupervised. Under no circumstance may target ground truth influence model predictions or adaptations:
1. **Post-Inference Access Only**:
   - Target ground-truth labels and evaluation masks may be read from disk **only after all inference and adaptation steps for the sample or batch are complete**, purely for final metric calculation.
2. **Strict Prohibition in Adaptation Loops**:
   - Target labels must **never** be passed to:
     - The neural network forward pass;
     - Unsupervised adaptation loss functions (e.g. Tent entropy, consistency loss);
     - Pseudo-label filtering, selection, or thresholding logic;
     - Test-time safety probes or rollback controllers;
     - Online risk score computation;
     - Dynamic threshold selection or hyperparameter adaptation during a test run.
3. **Hyperparameter Selection**:
   - Validation labels may only be used **offline** to evaluate completed methods and select global hyperparameters.
   - Test-set labels must **never** be accessed for hyperparameter tuning.

### 2.3 Protocol Immutability & Run Traceability
1. **Pre-Execution Freezing**:
   - Every run must record the starting Git commit hash and whether working tree modifications exist.
   - All run configuration parameters must be recorded to a manifest JSON *prior* to evaluation.
   - Any post-hoc parameter change constitutes a distinct experiment and requires a new, unique `run_id`.
2. **Negative Results Preservation**:
   - Failed runs, catastrophic adaptation failures, neutral runs, and harmful adaptations must be preserved in archival logs and reported with equal scientific weight. Deleting or hiding unpromising runs is strictly forbidden.
3. **Automated Pipeline Generation**:
   - Results tables, CSVs, and JSON summaries must be generated directly from raw outputs via automated scripts.
   - Manual editing of generated CSV or JSON result files is prohibited.

### 2.4 Complete Evaluation & Non-Subsetting Guarantee
1. **Mandatory Completeness Tracking**:
   - Every event-level evaluation must employ [`CompletenessTracker`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/src/evaluation/completeness.py).
   - Evaluators must verify the processed sample count against the canonical manifest, track all processed sample IDs, and assert:
     - Expected sample count == Processed sample count
     - Duplicate sample count == 0
     - Missing sample count == 0
     - Unexpected sample count == 0
2. **Diagnostic Partial Runs**:
   - Any diagnostic tool that evaluates a subset of samples (e.g., fast descriptive statistics) must explicitly declare `allow_partial=True` and log the visible banner:
     `WARNING: PARTIAL DIAGNOSTIC ONLY — MUST NOT BE REPORTED AS FULL EVENT METRICS`
   - Partial diagnostic results must never be transcribed, summarized, or reported as full event-level performance metrics.

### 2.5 Reproducibility Manifest
Every evaluation and experiment must record a standardized run manifest containing:
- `run_id`: Unique identifier (e.g. `YYYYMMDD_HHMMSS_<method>_<split>`);
- `timestamp`: UTC ISO 8601 timestamp;
- `git_commit`: Active git HEAD commit hash;
- `git_dirty`: Boolean flag indicating uncommitted changes;
- `python_executable`: Full path to Python binary;
- `environment`: OS and environment details;
- `gpu_device`: GPU model, CUDA version, and driver info;
- `random_seed`: Seed for deterministic execution;
- `command`: Full CLI invocation string;
- `config_path`: Path to configuration file;
- `checkpoint_path`: Model weights path;
- `checkpoint_hash`: SHA-256 hash of the checkpoint file;
- `data_manifest`: Manifest pickle/CSV path;
- `event_ids`: Evaluated event IDs;
- `output_paths`: Paths to all generated artifacts;
- `completion_status`: Run status (`SUCCESS`, `FAILED`, `ABORTED`).

### 2.6 Repository Boundaries
1. **Third-Party Upstream Isolation**:
   - `third_party/KuroSiwo` must remain clean and unmodified. All adaptors, fixes, sanitizations, and controllers must reside entirely within `ChangeGuard`.
   - External benchmark repositories (`UrbanSARFloods`, `BRIGHT`) must never be modified.
2. **Dual-Repository Status Reporting**:
   - Every task report and submission must include `git status --short` for `ChangeGuard` as well as `git status --short` for `third_party/KuroSiwo`.
