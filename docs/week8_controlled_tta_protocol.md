# Week 8 Controlled TTA Protocol & Gate A2 Audit Specification

**Document ID:** `docs/week8_controlled_tta_protocol.md`
**Configuration File:** `configs/evaluation/week8_controlled_tta_protocol.json`
**Date (UTC):** 2026-10-16
**Approval Reference:** `USER_DIRECTIVE_20261016_WEEK8B` (Implementation choices approved by User)
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Target Model:** Model A (`SNUNet_ECAM [VV, VH]`, 2 channels, base 32, FP32)
**Source Checkpoint Commit:** `6c63a5f8aa7ae2c608b145f0badc84b422b62ef1` (`outputs/week5_source_all27/`)
**Source Checkpoints:** Seeds 0, 1, and 2 (Epoch 16 `final_checkpoint.pt`)
**Target Events (6 TTA Development Events):** `[279, 437, 520, 559, 1111003, 1111008]` (2,826 samples, 135,859,080 valid pixels)
**Quarantined Events (10 Sealed Test Events):** `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`
**Status:** `FROZEN_APPROVED_FOR_IMPLEMENTATION`

---

## 1. Scientific Motivation & Context

Week 7 established the verified, multi-seed source-only baseline for Model A across the 6 TTA development events under batch size 32 ($0.5906 \pm 0.0037$ Macro F1, $0.5012 \pm 0.0221$ Pooled F1).

In Week 8, we conduct a controlled audit of standard unsupervised Test-Time Adaptation (TTA) algorithms—Adaptive Batch Normalization (AdaBN) and reference-aligned Test-Time Entropy Minimization (Tent)—to evaluate whether adaptation causes reproducible degradation, selective benefit, or stability under declared experimental conditions.

### Historical Context & Clean-Slate Evaluation
In the preliminary exploratory pilot (Day 8), Tent and AdaBN produced widespread degradation on the old official KuroSiwo reference model (`kuro_snunet_official.pt`, 3 channels including DEM). However, scientific rigor demands that we do not assume those failure modes automatically transfer to our autonomously trained, 2-channel, all-source FP32 models. Week 8 evaluates adaptation behavior directly on our verified source checkpoints.

### Correct Gate A2 Scope
Gate A2 is strictly an **empirical audit of whether harmful adaptation is reproducible under the declared experimental conditions**.
- It does **not** evaluate risk-prediction quality, controller effectiveness, or the final viability of the project topic.
- It does **not** mandate automatic controller "GO" or research topic-pivot decisions.
- Operational reproducibility criteria are formulated as an explicit pre-registered rule approved prior to execution.

---

## 2. Logical Stream Matrix & Matched Source-Only Control Plan

The evaluation framework specifies **36 logical matrix cells** across 3 methods, 3 checkpoint seeds, 2 stream orders, and 2 reset modes.

### Physical Execution Plan: 30 Physical Streams vs. 36 Logical Matrix Cells
To ensure exact scientific matching without unnecessary duplicate compute, we distinguish physical execution from logical matrix accounting:

1. **Fresh Matched Source-Only Controls (6 physical streams):**
   - Source-only baselines will be executed fresh under the exact same dataloader configuration ($B=16$), sample ordering, preprocessing, and offline evaluation conventions as the adaptation runs.
   - Evaluates all 3 checkpoint seeds $\times$ 2 stream orders = **6 physical source-only streams**.
   - Model state immutability (pre/post parameter and buffer hashes) will be cryptographically audited.
   - Because source-only inference updates neither parameters nor running buffers, reset mode has zero effect on its outputs. Once verified, these 6 stream results are explicitly reused across `reset_per_event` and `continual` modes, populating the 12 logical source-only matrix cells.
   - *Integrity Label:* Reused control rows are explicitly tagged as `reused_control_from_matched_order`.
2. **Adaptation Runs (24 physical streams):**
   - AdaBN: 3 seeds $\times$ 2 orders $\times$ 2 reset modes = **12 physical streams**.
   - Tent (Reference-Aligned): 3 seeds $\times$ 2 orders $\times$ 2 reset modes = **12 physical streams**.
3. **Accounting Totals:**
   - **Planned Physical Streams:** $6 + 12 + 12 = \mathbf{30}$ streams.
   - **Logical Matrix Cells:** $12 + 12 + 12 = \mathbf{36}$ stream cells.
   - **Logical Event-Result Rows:** $36 \times 6 = \mathbf{216}$ event rows.

### Distinguishing Checkpoint Seeds vs. Stream-Order Random Seeds
- **Checkpoint Seed ($s \in \{0, 1, 2\}$):** Indexes the source-trained model parameter weights resulting from source training initialization.
- **Stream Order ($\omega \in \{\text{order\_a}, \text{order\_b}\}$):** Indexes the sequence in which target events and within-event sample patches are presented to the adapter.

### Stream Order Definitions & Causal Scope
- **Order A (Canonical Chronological / Ascending Event ID):**
  - Event sequence: `[279, 437, 520, 559, 1111003, 1111008]`
  - Within-event sample order: Deterministic ascending sort by unique `sample_id` string.
- **Order B (Reversed Event Sequence / Descending Sample ID):**
  - Event sequence: `[1111008, 1111003, 559, 520, 437, 279]`
  - Within-event sample order: Deterministic descending sort by unique `sample_id` string.
- **Causal Scope of Order B:** Order B changes **both** event sequence and within-event sample order simultaneously. It evaluates **combined ordering sensitivity**, not their separate causal effects.

### Complete 36-Configuration Logical Ledger

| Config Index | Config ID | Execution Type | Method | Checkpoint Seed | Stream Order | Reset Mode |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- |
| **01** | `stream_source_s0_oa_re` | Physical Run | `source_only` | 0 | `order_a` | `reset_per_event` |
| **02** | `stream_source_s0_oa_co` | Reused Control | `source_only` | 0 | `order_a` | `continual` |
| **03** | `stream_source_s0_ob_re` | Physical Run | `source_only` | 0 | `order_b` | `reset_per_event` |
| **04** | `stream_source_s0_ob_co` | Reused Control | `source_only` | 0 | `order_b` | `continual` |
| **05** | `stream_source_s1_oa_re` | Physical Run | `source_only` | 1 | `order_a` | `reset_per_event` |
| **06** | `stream_source_s1_oa_co` | Reused Control | `source_only` | 1 | `order_a` | `continual` |
| **07** | `stream_source_s1_ob_re` | Physical Run | `source_only` | 1 | `order_b` | `reset_per_event` |
| **08** | `stream_source_s1_ob_co` | Reused Control | `source_only` | 1 | `order_b` | `continual` |
| **09** | `stream_source_s2_oa_re` | Physical Run | `source_only` | 2 | `order_a` | `reset_per_event` |
| **10** | `stream_source_s2_oa_co` | Reused Control | `source_only` | 2 | `order_a` | `continual` |
| **11** | `stream_source_s2_ob_re` | Physical Run | `source_only` | 2 | `order_b` | `reset_per_event` |
| **12** | `stream_source_s2_ob_co` | Reused Control | `source_only` | 2 | `order_b` | `continual` |
| **13** | `stream_adabn_s0_oa_re` | Physical Run | `adabn` | 0 | `order_a` | `reset_per_event` |
| **14** | `stream_adabn_s0_oa_co` | Physical Run | `adabn` | 0 | `order_a` | `continual` |
| **15** | `stream_adabn_s0_ob_re` | Physical Run | `adabn` | 0 | `order_b` | `reset_per_event` |
| **16** | `stream_adabn_s0_ob_co` | Physical Run | `adabn` | 0 | `order_b` | `continual` |
| **17** | `stream_adabn_s1_oa_re` | Physical Run | `adabn` | 1 | `order_a` | `reset_per_event` |
| **18** | `stream_adabn_s1_oa_co` | Physical Run | `adabn` | 1 | `order_a` | `continual` |
| **19** | `stream_adabn_s1_ob_re` | Physical Run | `adabn` | 1 | `order_b` | `reset_per_event` |
| **20** | `stream_adabn_s1_ob_co` | Physical Run | `adabn` | 1 | `order_b` | `continual` |
| **21** | `stream_adabn_s2_oa_re` | Physical Run | `adabn` | 2 | `order_a` | `reset_per_event` |
| **22** | `stream_adabn_s2_oa_co` | Physical Run | `adabn` | 2 | `order_a` | `continual` |
| **23** | `stream_adabn_s2_ob_re` | Physical Run | `adabn` | 2 | `order_b` | `reset_per_event` |
| **24** | `stream_adabn_s2_ob_co` | Physical Run | `adabn` | 2 | `order_b` | `continual` |
| **25** | `stream_tent_s0_oa_re` | Physical Run | `tent` | 0 | `order_a` | `reset_per_event` |
| **26** | `stream_tent_s0_oa_co` | Physical Run | `tent` | 0 | `order_a` | `continual` |
| **27** | `stream_tent_s0_ob_re` | Physical Run | `tent` | 0 | `order_b` | `reset_per_event` |
| **28** | `stream_tent_s0_ob_co` | Physical Run | `tent` | 0 | `order_b` | `continual` |
| **29** | `stream_tent_s1_oa_re` | Physical Run | `tent` | 1 | `order_a` | `reset_per_event` |
| **30** | `stream_tent_s1_oa_co` | Physical Run | `tent` | 1 | `order_a` | `continual` |
| **31** | `stream_tent_s1_ob_re` | Physical Run | `tent` | 1 | `order_b` | `reset_per_event` |
| **32** | `stream_tent_s1_ob_co` | Physical Run | `tent` | 1 | `order_b` | `continual` |
| **33** | `stream_tent_s2_oa_re` | Physical Run | `tent` | 2 | `order_a` | `reset_per_event` |
| **34** | `stream_tent_s2_oa_co` | Physical Run | `tent` | 2 | `order_a` | `continual` |
| **35** | `stream_tent_s2_ob_re` | Physical Run | `tent` | 2 | `order_b` | `reset_per_event` |
| **36** | `stream_tent_s2_ob_co` | Physical Run | `tent` | 2 | `order_b` | `continual` |

---

## 3. Mask Provenance & Strict Label Isolation

### 3.1 Provenance of KuroSiwo Masks
In the KuroSiwo dataset, two distinct raster masks exist:
1. **Sensory Validity Component (`MK0_MNA`):** A binary raster produced from sensor geometry that identifies valid SAR swath observations ($1 = \text{valid observation}$, $0 = \text{no-data / swath margin}$). This mask is an intrinsic sensory attribute available at deployment time without any ground-truth labels.
2. **Ground Truth Land-Cover / Flood Mask (`MK0_MLU`):** Ground-truth annotations where classes $0, 1, 2$ represent background, permanent water, and flood inundation, respectively, and class $3$ represents unannotated or invalid ground-truth pixels.

### 3.2 Adaptation vs. Offline Evaluation Masking Policy
- **Adaptation Policy (Strictly Label-Free):**
  Adaptation must receive **zero ground-truth labels and zero ground-truth-derived masks**.
  Under Tent, the entropy loss is computed strictly over the sensory observation mask:
  $$\text{Adaptation Valid Mask} = \{i \mid \text{MK0\_MNA}_i = 1\}$$
  Under no circumstance is `label != 3` evaluated or applied during adaptation.
- **Offline Evaluation Policy (Double-Masking):**
  Double-masking ($\text{MK0\_MNA} = 1 \land \text{MK0\_MLU} \ne 3$) is strictly confined to the offline evaluation phase after predictions have been generated and frozen. Offline evaluation must never influence parameter updates, stopping decisions, or checkpoint selection.

---

## 4. Algorithmic Specifications: Reference-Aligned Tent Implementation

### 4.1 Upstream Tent Reference Alignment
The primary matrix uses `ReferenceTentAdapter` (`src/tta/tent_reference.py`, identifier: `reference_tent_v1`), aligned with the official reference:
- **Reference:** Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021.
- **Upstream Source:** `https://github.com/DequanWang/tent/blob/master/tent.py` (master branch).
- **Semantics Implemented:**
  1. Only BatchNorm affine parameters ($\gamma, \beta$) receive gradients and optimizer updates. Non-BN parameters remain frozen.
  2. Running-statistics tracking is disabled (`track_running_stats = False`), and running mean and variance are set to `None` (`running_mean = None`, `running_var = None`). Normalization uses current batch statistics exclusively.
  3. **Single Forward Pass:** One forward pass computes both predictions and entropy loss.
  4. **Prediction Timing:** The predictions returned are those computed by that single forward pass. No second post-update forward is executed.
  5. One Adam step per batch ($lr = 1 \times 10^{-4}$, $\beta = (0.9, 0.999)$, weight decay $= 0.0$).
- **Preservation of Legacy Variant:** The historical two-forward Tent implementation is preserved intact in `src/tta/tent.py` and is **not** overwritten.

### 4.2 SAR-Specific Adaptations (Explicitly Documented)
1. **Spatial Masked Entropy:** Shannon entropy is computed over spatial pixels belonging to the sensory observation mask ($\text{MK0\_MNA} == 1$). Pixels outside valid swath coverage are excluded from the loss denominator.
2. **Dual-Input Branching:** Bi-temporal SAR inputs ($x_A$: pre-event, $x_B$: post-event).
3. **Strict Label Isolation:** Zero ground-truth labels (`MK0_MLU`) or label-derived masks enter adaptation.

### 4.3 State Restoration & Reset Mechanics
- **`reset_per_event` (Episodic Mode):**
  Before each event, running buffers are re-registered to allow strict loading of source checkpoints, `load_state_dict` restores original source weights and buffers, reference Tent configuration is reapplied, a fresh Adam optimizer is instantiated, and the step counter is zeroed.
- **`continual` Mode:**
  `reset_to_source()` is called only once at the beginning of the stream. Adapted parameters and optimizer momentum state persist across event boundaries.

### 4.4 Partial-Batch Execution Policy
- **Policy:** Process the final partial batch as-is (no dropped samples, no artificial padding).
- **Narrow Statement on Tail Batches:** All $2,826$ samples will be retained. Under $B=16$, the expected tail batch sizes across the 6 events are $13, 9, 12, 10, 3, 11$. No singleton ($N=1$) batch is expected to occur.
- **Mandatory Numerical Safeguard:** While singleton tail batches are not expected, actual execution strictly requires runtime numerical health checks (confirming absence of NaNs, Infs, or non-finite outputs).

---

## 5. Gate A2 Audit Scope & Operational Reproducibility Rule

### 5.1 Correct Scope of Gate A2
Gate A2 is exclusively an empirical check for the presence of reproducible harmful adaptation under the declared conditions.
- It does **not** establish whether risk scores are effective, whether a controller can prevent degradation, or the ultimate viability of the FYP topic.
- Automatic "controller GO" and research topic-pivot decisions are removed.

### 5.2 Approved Operational Reproducibility Rule
1. **Event-Level Metric Deltas:** Paired strictly against the matching source checkpoint seed:
   $$\Delta F_{1, e} = F_{1, \text{adapted}, e} - F_{1, \text{source}, e}(s)$$
2. **Event Classification:**
   - **Harmful Event:** $\Delta F_{1, e} \le -0.0100$ (drop $\ge 1.0$ percentage point).
   - **Beneficial Event:** $\Delta F_{1, e} \ge +0.0100$ (gain $\ge 1.0$ percentage point).
   - **Neutral Event:** $-0.0100 < \Delta F_{1, e} < +0.0100$.
3. **Operational Reproducibility Criterion:**
   > **Reproducible Harm is established if and only if:**
   > The same event is harmed ($\Delta F_{1, e} \le -0.0100$) for the same adaptation method and reset mode in **at least two of the three checkpoint seeds** under **both stream orders** (Order A and Order B).
4. **Separate Benefit Reporting:**
   Beneficial events ($\Delta F_{1, e} \ge +0.0100$) will be reported in separate summary tables. Observed benefit is **not** required to establish the existence of harmful adaptation.

### 5.3 Tri-State Outcome Classification
The Gate A2 audit will conclude in one of three distinct outcome states:
1. **Reproducible Harm Observed:** The operational reproducibility criterion is satisfied for at least one event.
2. **Reproducible Harm Not Established Under Tested Conditions:** The operational reproducibility criterion is not met under the tested conditions.
   - *Scientific Precaution:* Absence of observed harm under these specific conditions must **not** be inferred as universal safety across all possible hyperparameters or domains.
3. **Incomplete / Inconclusive Evidence:** The audit could not be fully evaluated due to numerical failures (NaN/Inf), process interruption, or execution defects.
   - *Numerical Safeguard Rule:* Numerical failures (NaN/Inf) must **not** be treated as valid measured performance degradation; they are execution defects.

---

## 6. Artifact Registry & Output Specification

All Week 8 outputs will be recorded under `outputs/week8_controlled_tta/`:
- `stream_event_metrics.csv`: 216 rows (36 logical stream configurations $\times$ 6 events), recording confusion counts, F1, Recall, Precision, IoU, paired $\Delta$ metrics, and reuse tags.
- `stream_matrix_summary.json`: Detailed 36-stream ledger recording stream macro/pooled metrics, worst-event F1, harmed/benefited event counts, and configuration metadata.
- `week8_tta_manifest.json`: Cryptographic manifest recording execution environment, git commit, checkpoint SHA-256 hashes, and execution logs.
- Execution logs preserved under `logs/week8_controlled_tta/`.

---

## 7. Status & Next Bounded Task

Week 8B formal matrix execution implementation (`run_matrix()`) and 15 synthetic CPU unit tests are **complete and verified**.
- Diagnostic smoke test verified on 38 samples (Seed 0, Order A).
- Formal runner supports 30 physical streams, 36 logical cells, 216 event rows, Gate A2 audit evaluation, and output collision protection.
- CLI entrypoint connected: `--smoke` executes diagnostic smoke; default invocation executes formal matrix; `--validate_only` runs preflight checks.

**Next Bounded Task:**
1. Commit the implementation checkpoint (`tools/run_controlled_tta_matrix.py`, `tests/test_controlled_tta_matrix.py`, `src/data/evaluation_dataset.py`, `docs/week8_smoke_provenance_note.md`, `docs/week8_controlled_tta_protocol.md`).
2. Await explicit user authorization before launching formal matrix evaluation on GPU in a detached tmux session.
