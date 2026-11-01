# RST8 Supplementary Architecture Replication Protocol (SiamUnet_diff across Seeds 0, 1, 2)

## 1. Context, Motivation, and Preregistration Boundaries

### 1.1 Context and Historical Decision
Work Package **RST8** originally investigated whether the empirical gains of Recipe R2 (M1 event-balanced sampling + M2 training-only class-balanced CE + Dice), established on SNUNet-ECAM during RST6 and RST7, transfer to a second change detection architecture: **SiamUnet_diff** (FC-Siam-diff, Daudt et al., 2018).

At Stage A, RST8 evaluated seed 0 across Confirmation Folds A and B (4 training runs). The observed results showed a slight average reduction in event-macro F1 ($-0.010$) and tail degradation (Bottom-3 mean F1 $-0.045$). In accordance with the preregistered Stage-A continuation rule (`docs/robust_source_training/rst8_siamunet_diff_protocol_v1.md`, Section 4), formal multi-seed expansion was not triggered, and compute was stopped at seed 0 (`docs/robust_source_training/rst8_siamunet_diff_seed0_closeout_v1.md`).

### 1.2 Authorization of Supplementary Replication Study
This supplementary study authorizes an extension to initialization **seeds 1 and 2** across R0 and R2 on Confirmation Folds A and B (8 new training runs), completing a balanced 3-seed evaluation (12 total runs).

**Explicit Preregistration Record**:
- This extension was decided **strictly after observing seed 0 results**.
- It does **not** rewrite, modify, or overturn the original discovery, stopping decision, or preregistered status of RST8 Stage A.
- This is **not a search for a positive result**. All valid results across all three seeds (seeds 0, 1, and 2) will enter the final comparative analysis and be reported completely and objectively, regardless of magnitude or direction.

---

## 2. Experimental Architecture and Governing Invariants

### 2.1 Model Architecture: SiamUnet_diff
- **Architecture**: `SiamUnet_diff` (FC-Siam-diff) via `src/models/siam_diff_wrapper.py`.
- **Parameters**: Exactly **1,350,147 parameters** (~1.35M parameters, compared to 12.03M in SNUNet-ECAM).
- **Channels & Regularization**: `in_channels = 2` (dual-polarization SAR VV/VH), `out_ch = 3` (3 semantic classes), `dropout_p = 0.2` (active during training, disabled during evaluation).
- **Difference Mechanism**: Dual-stream encoder with shared weights; absolute feature differences $|\mathbf{z}_{\text{pre}} - \mathbf{z}_{\text{post}}|$ concatenated at each decoder stage.

### 2.2 Semantic Label Contract
- Class 0: Dry ground / unflooded land
- Class 1: Permanent water body
- Class 2: Newly inundated flood water (**positive evaluation class**)
- Class 3: Ignore / invalid data mask (strictly excluded from loss and evaluation metrics)

### 2.3 Training and Optimization Hyperparameters (Identical to Seed 0)
- **Optimizer**: Adam (`lr = 0.001`, `betas = [0.9, 0.999]`, `weight_decay = 0.0`).
- **Batching**: `micro_batch_size = 8`, `gradient_accumulation_steps = 4`, `effective_batch_size = 32`.
- **Training Budget**: 16 epochs.
  - Fold A: 13,302 training patches $\times$ 16 epochs / 32 effective batch size = **6,656 optimizer steps**.
  - Fold B: 14,655 training patches $\times$ 16 epochs / 32 effective batch size = **7,328 optimizer steps**.
- **Loss Functions**:
  - **Recipe R0**: Unweighted Cross-Entropy (0.5) + Dice Loss (0.5).
  - **Recipe R2**: Training-only class-balanced Cross-Entropy (0.5) + Dice Loss (0.5), using frozen fold class weights.
- **Class Weights Provenance**:
  - Fold A: `[0.613865, 1.487843, 0.898292]` (`configs/robust_source_training/rst5_folda_class_weights_v1.json`, SHA-256: `c3bbb6ba...`).
  - Fold B: `[0.543519, 1.499552, 0.956929]` (`configs/robust_source_training/rst5_foldb_class_weights_v1.json`, SHA-256: `d84e797f...`).
  - Derived strictly from each fold's training events; zero influence from target events or evaluation events.
- **Sampling & Augmentation**:
  - M1 event-balanced sampling active for R2; standard patch sampling for R0.
  - M3 paired augmentation disabled for both R0 and R2.
- **Checkpoint Selection**: Final checkpoint saved after 16 completed epochs (`final_checkpoint.pt`), with reload-verification asserting bit-identical state.

---

## 3. Experimental Matrix

### 3.1 Cell Definitions (12 Total Runs)

| Cell ID | Recipe | Fold | Seed | Status | Assigned GPU | Run ID |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `cell_01` | R0 | Fold A | 0 | **Existing (Verified)** | -- | `rst8_r0_siamunet_diff_folda_s0_v1` |
| `cell_02` | R2 | Fold A | 0 | **Existing (Verified)** | -- | `rst8_r2_siamunet_diff_folda_s0_v2` |
| `cell_03` | R0 | Fold B | 0 | **Existing (Verified)** | -- | `rst8_r0_siamunet_diff_foldb_s0_v2` |
| `cell_04` | R2 | Fold B | 0 | **Existing (Verified)** | -- | `rst8_r2_siamunet_diff_foldb_s0_v2` |
| `cell_05` | R0 | Fold A | 1 | **New Authorized** | GPU 0 (Queue 1) | `rst8_r0_siamunet_diff_folda_s1_v1` |
| `cell_06` | R0 | Fold A | 2 | **New Authorized** | GPU 0 (Queue 2) | `rst8_r0_siamunet_diff_folda_s2_v1` |
| `cell_07` | R2 | Fold A | 1 | **New Authorized** | GPU 1 (Queue 1) | `rst8_r2_siamunet_diff_folda_s1_v1` |
| `cell_08` | R2 | Fold A | 2 | **New Authorized** | GPU 1 (Queue 2) | `rst8_r2_siamunet_diff_folda_s2_v1` |
| `cell_09` | R0 | Fold B | 1 | **New Authorized** | GPU 2 (Queue 1) | `rst8_r0_siamunet_diff_foldb_s1_v1` |
| `cell_10` | R0 | Fold B | 2 | **New Authorized** | GPU 2 (Queue 2) | `rst8_r0_siamunet_diff_foldb_s2_v1` |
| `cell_11` | R2 | Fold B | 1 | **New Authorized** | GPU 3 (Queue 1) | `rst8_r2_siamunet_diff_foldb_s1_v1` |
| `cell_12` | R2 | Fold B | 2 | **New Authorized** | GPU 3 (Queue 2) | `rst8_r2_siamunet_diff_foldb_s2_v1` |

### 3.2 Partition Invariants
- **Fold A Evaluation**: Exactly 9 events (`118, 174, 273, 421, 496, 498, 554, 567, 1111009`).
- **Fold B Evaluation**: Exactly 9 events (`267, 324, 417, 470, 502, 518, 555, 1111004, 1111010`).
- **Total Unique Evaluated Events**: Exactly 18 held-out source events (yielding 108 evaluated event rows across the 12 runs).
- **Target Event Exclusion**: All 16 historical target events and all external datasets (GEOID-Flood) remain strictly excluded.

---

## 4. Execution and Resource Plan

- **Hardware**: 4 $\times$ NVIDIA GeForce RTX 3090 GPUs.
- **Concurrency**: Exactly one active training process per GPU, executed via detached `tmux` sessions.
- **Resource Footprint**:
  - VRAM per process: ~1,166 MiB peak.
  - Disk footprint per run: ~32 MB (checkpoints and logs), total ~256 MB.
  - Wall-clock time per process: ~35–48 minutes.
  - Total elapsed time with 2-job queues across 4 GPUs: ~75–96 minutes.
- **Failure Recovery Policy**: Fresh-run only; failed preflight attempts preserved with `terminal_status.json`; overwriting existing results strictly prohibited.

---

## 5. Statistical Analysis Procedure

Following the confirmatory methodology established in RST7:
1. **Per-Event Seed Averaging**: For each of the 18 unique held-out events, compute mean metric values across seeds 0, 1, 2 for R0 and R2:
   $$\bar{F}_{1}(e, \text{R0}) = \frac{1}{3} \sum_{s=0}^2 F_1(e, \text{R0}, s), \quad \bar{F}_{1}(e, \text{R2}) = \frac{1}{3} \sum_{s=0}^2 F_1(e, \text{R2}, s)$$
2. **Paired Per-Event Differences**:
   $$\Delta F_1(e) = \bar{F}_{1}(e, \text{R2}) - \bar{F}_{1}(e, \text{R0})$$
3. **Grand Macro Means**: Unweighted mean of the 18 per-event means across the entire source benchmark.
4. **Tail Reliability**:
   - Bottom-3 Mean F1 across the 18 events.
   - Single Worst-Event F1 across the 18 events.
5. **Paired Event-Level Bootstrap**: 10,000 resamples with replacement across the 18 unique events (`np.random.RandomState(42)`), reporting empirical 95% confidence interval and one-sided $p$-value.
6. **Cross-Seed Dispersion**: Sample standard deviation across the three seeds computed per fold.
7. **Benefit / Neutral / Harm Counts**: Evaluated against historical thresholds ($> +0.010$ Benefit, $< -0.010$ Harm, else Neutral).

