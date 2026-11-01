# RST2 Event-Disjoint Infrastructure Protocol

**Document version:** 1.0.0
**Effective date:** 2026-10-26
**Status:** Approved operational protocol for RST2 infrastructure and smoke
**Governing proposal:** `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
**Governing specification:** `configs/robust_source_training/rst1_matched_r0_spec_v1.json`
**Event folds definition:** `configs/robust_source_training/rst1_event_folds_v1.json`

---

## 1. Purpose and Non-Scientific Scope

The objective of Work Package RST2 is to establish, verify, and freeze the software infrastructure required for event-disjoint source training and independent evaluation under the matched R0 baseline recipe.

RST2 implements:
- Strict fail-closed event allowlist validation preventing partition overlap and data leakage;
- Fold-restricted matched R0 dataset loading conforming to the proven linear-amplitude SAR domain;
- Deterministic data loading and worker RNG management;
- Event-independent evaluation producing per-event metrics and unweighted event-macro aggregations;
- Two-phase immutable manifest tracking (`preflight_manifest.json` and `final_manifest.json`);
- Full atomic checkpoint serialization with verified resume capability;
- A bounded, non-scientific real-data infrastructure smoke test.

### Non-Scientific Evidence Boundary
The real-data smoke executed under RST2 is purely an engineering and infrastructure verification. The resulting diagnostic loss values and metrics carry zero scientific weight and must never be cited as model performance or used to select methods or hyperparameters.

### Historical Checkpoint Invalidation for Folds
The historical full-source checkpoints (`outputs/week5_source_all27/`) were trained across all 27 source events. Because every candidate held-out fold event was seen during their optimization, **historical full-27 checkpoints cannot serve as fold-specific R0 baselines**. Fold-specific baselines must be trained from scratch strictly on the 18 authorized training events of each fold.

---

## 2. Event Partitioning and Allowlist Semantics

### 2.1 Explicit Allowlist Enforcement
Every training and evaluation run must explicitly configure two disjoint lists of numeric event IDs:
1. `train_event_ids`: Authorized source events for parameter optimization;
2. `evaluation_event_ids`: Authorized source events for held-out validation or testing.

The dataset and runner must reject any implicit fallback to "all events" when an allowlist is omitted.

### 2.2 Prohibited Events
The following sixteen Kuro Siwo events remain strictly quarantined:
```text
[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]
```
Any runner or dataset encountering an event in this set must abort immediately (`ResearchIntegrityViolation`).

### 2.3 Group C Discovery-Fold Mapping
For the primary discovery phase (RST4):
- `train_event_ids`: Union of Group A and Group B (18 events, 14,967 patches);
- `evaluation_event_ids`: Group C (9 events, 6,495 patches).

For the bounded RST2 infrastructure smoke, a small, outcome-blind subset of these groups is declared:
- Training: at least one event from Group A (e.g., Event 273) and one from Group B (e.g., Event 518);
- Evaluation: at least one event from Group C (e.g., Event 130).

---

## 3. Matched R0 Pipeline Specification

### 3.1 Model Initialization
- **Architecture:** `SNUNet_ECAM(in_channels=2, out_ch=3, base_channel=32)`.
- **Initialization:** Deterministic PyTorch default weight initialization governed by `torch.manual_seed(seed)`.

### 3.2 Input Channels and Pairing
- **Inputs:** Dual-polarization Sentinel-1 GRD SAR (`[VV, VH]`).
- **Temporal Pairing:** Pre-event observation $x_A$ (`pre_event_2`, files `SL2_IVV`, `SL2_IVH`) and post-event observation $x_B$ (`post_event`, files `MS1_IVV`, `MS1_IVH`).
- **Input Domain:** Linear amplitude backscatter clamped to $[0.0, 0.15]$, non-finite replacement with $0.15$, standardized by global constants ($\mu_{\text{VV}}=0.0953, \sigma_{\text{VV}}=0.0427, \mu_{\text{VH}}=0.0264, \sigma_{\text{VH}}=0.0215$).

### 3.3 Label Semantics and Masking
- Class 0: No Water (dry land);
- Class 1: Permanent Water;
- Class 2: Flood Water (positive target);
- Class 3: Invalid / Ignore.
- **Double Masking:** `label[valid_mask == 0] = 3`.
- Labels and valid masks never enter the neural network forward pass.

### 3.4 Sampling and Augmentation State
- **Sampling:** Patch-uniform random shuffle without replacement (`RandomSampler(replacement=False)`). Micro-batch size 8, gradient accumulation 4 (effective batch size 32).
- **Augmentation:** Disabled. Zero geometric or radiometric transforms.

### 3.5 Loss Function
- `ControlledSourceLoss` combining unweighted $0.5 \times \text{CrossEntropyLoss}(\text{ignore}=3) + 0.5 \times \text{SoftDiceLoss}(\epsilon=1.0, \text{ignore}=3)$ with reduction `batch`.

---

## 4. Execution, Manifest, and Checkpoint Lifecycle

### 4.1 Two-Phase Manifest Lifecycle
1. **Preflight Manifest (`preflight_manifest.json`):**
   - Written to the run directory *before* model instantiation, dataset loading, or GPU initialization.
   - Records starting Git commit hash, git dirty status, SHA-256 of configuration files, split definitions, R0 specification, random seed, hardware identity, and complete event allowlists.
   - Once written, this file is permanent and immutable.
2. **Final Manifest (`final_manifest.json`):**
   - Written upon completion of execution.
   - References the SHA-256 hash of the initial manifest.
   - Records observed sample counts, optimizer steps, checkpoint checksums, event metric paths, elapsed wall-clock runtime, peak VRAM, and terminal execution status.

### 4.2 Checkpoint and Resume Semantics
- Checkpoints are saved atomically as `last_checkpoint.pt`, `best_checkpoint.pt` (when validation is active), and `final_checkpoint.pt`.
- Every checkpoint dictionary must serialize:
  - `model_state_dict`: Full neural network weights and buffers;
  - `optimizer_state_dict`: Optimizer state (momentum buffers, step counts);
  - `scheduler_state_dict`: `None` or explicit `"NA"`;
  - `scaler_state_dict`: `None` or explicit `"NA"`;
  - `epoch`: Current epoch index;
  - `global_step`: Cumulative optimizer update step count;
  - `random_seed`: Seed of the run;
  - `rng_states`: PyTorch CPU, CUDA, NumPy, and Python random states;
  - `metadata`: Run ID, config hash, fold hash, train event allowlist, evaluation event allowlist.
- **Fail-Closed Resume:** Any resume attempt where configuration hashes, event allowlists, or architecture definitions disagree must fail closed with an explicit error.
- **Resume Verification:** Verification must instantiate a fresh model, load state, and assert exact weight equality and finite loss/gradient continuation.

### 4.3 Output Protection and Terminal Status
- Under no circumstances may an existing run directory be overwritten. Existing paths fail closed unless `--resume` is explicitly invoked on the same run ID.
- Upon process termination, the runner writes `terminal_status.json` containing:
  - `run_id`: Unique identifier;
  - `status`: Formal execution status (`COMPLETED_INFRASTRUCTURE_SMOKE`, `COMPLETED_SCIENTIFIC_RUN`, or `FAILED`);
  - `exit_code`: Numeric exit code (0 for success);
  - `timestamp_utc`: Completion timestamp.

---

## 5. Exit Criteria for RST2

Work Package RST2 is considered complete when:
1. All reusable modular contracts and data paths pass focused CPU unit testing (covering the 32 required test criteria);
2. A scoped prelaunch commit captures the reviewed code, tests, protocol, and smoke configuration;
3. The smoke configuration is authorized (`launch_authorized = true`);
4. The bounded infrastructure smoke executes cleanly in a detached tmux session, verifying multi-batch training, optimizer update, checkpoint saving, fresh-process reload, state equality, finite continuation, independent event evaluation, and terminal status `COMPLETED_INFRASTRUCTURE_SMOKE`;
5. Zero forbidden events were accessed;
6. Historical E0-E11 files and the two tracked-dirty documents remain untouched;
7. A formal RST2 closeout report is recorded and committed.

