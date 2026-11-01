# RST3 Component Protocol: Robust Source Training Components

Date: 2026-10-26
Status: FROZEN PROTOCOL
Route: Robust Source Training (RST)
Authoritative Proposal: `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
Preceding Milestones: RST0 (Freeze), RST1 (Audit/Folds), RST2 (Infrastructure)

---

## 1. Scope and Objective

The objective of work package **RST3** is to implement, test, and verify the three robust source training components:
- **M1**: Event-balanced epoch sampling;
- **M2**: Training-only flood-aware loss objective;
- **M3**: Synchronized paired SAR data augmentation.

RST3 is an **engineering and verification** milestone. All GPU executions in RST3 are bounded infrastructure component smokes to verify numerical stability, loss convergence, gradient finiteness, checkpointing, and manifest generation. No scientific claims, recipe selection, or evaluation against the full discovery staircase (RST4) may occur during RST3.

---

## 2. Frozen Recipe Staircase

The robust source training route strictly enforces a 4-step frozen staircase:

| Recipe ID | M1: Event Balance | M2: Flood-Aware Loss | M3: Paired Augmentation | Scientific Purpose |
| :---: | :---: | :---: | :---: | :--- |
| **R0** | Off | Off | Off | Matched source baseline (RST2 reference) |
| **R1** | On | Off | Off | Isolates event representation rebalancing |
| **R2** | On | On | Off | Isolates class imbalance mitigation |
| **R3** | On | On | On | Composite full proposed robust recipe |

### Invariants:
1. **Independent Switches**: Each component has an explicit boolean toggle (`m1_event_balance`, `m2_flood_aware_loss`, `m3_paired_augmentation`).
2. **Strict Staircase Enforcement**: Only combinations corresponding to R0, R1, R2, and R3 are valid recipes. Arbitrary ad-hoc combinations (e.g. M3 without M1) are rejected in scientific validation.
3. **R0 Parity Mandate**: When all three switches are false, the runner must execute in bit-exact or justified numerical parity with the RST2 matched-R0 baseline.
4. **No Metric-Driven Recipe Choice**: RST3 smoke evaluation numbers must not be used to select or eliminate any recipe. All four recipes proceed unconditionally into the formal discovery staircase in RST4.

---

## 3. Component M1: Event-Balanced Epoch Sampler

### Motivation
In natural disaster datasets like Kuro Siwo / UrbanSARFloods, eligible patches are heavily concentrated in a small number of large flood events, causing standard uniform random sampling to over-represent large events while starving geographically distinct small events.

### Formal Semantics
For a training partition containing $E$ training events and $N$ total eligible patches:
1. **Epoch Compute Budget**: An epoch yields exactly $N$ sample draws, preserving identical epoch compute length and optimizer step count to R0.
2. **Event Quotas**: Event identities are drawn with approximately uniform frequency:
   - Each event receives at least $\lfloor N / E \rfloor$ draws.
   - The remainder $R = N \pmod E$ draws are distributed deterministically across events sorted by event ID.
3. **Queue Cycling Within Events**:
   - For each event, a deterministic shuffled queue of available patches is maintained.
   - Patches are consumed without replacement until the queue is exhausted.
   - When exhausted, the queue is deterministically reshuffled and recycled.
   - Small events are safely recycled without immediate back-to-back duplicate adjacency where possible.
4. **Seed Derivation**:
   The sampler seed is deterministically computed from:
   $$\text{sampler\_seed} = (\text{run\_seed} \times 1000003 + \text{epoch\_idx} \times 997 + 101) \pmod{2^{32}}$$
5. **Durable Logging**:
   Every epoch records an `M1EpochSummary`:
   - `expected_draws_per_event`, `realized_draws_per_event`;
   - `unique_samples_per_event`, `repeat_count_per_event`;
   - `min_draws`, `max_draws`;
   - `ordered_sample_id_digest` (SHA-256 of the exact sequence of yielded sample IDs);
   - `total_draws` (asserted equal to $N$).

---

## 4. Component M2: Training-Only Flood-Aware Loss

### Motivation
Newly inundated flood water (class 2) is rare compared to permanent water (class 1) and dry land (class 0). Unweighted loss optimization biases predictions toward background, causing high precision but severely depressed recall.

### Strict Data Boundary
Class statistics must be gathered **strictly and exclusively** from authorized training events (`train_event_ids`). Any attempt to access evaluation events, target events, or the 16 forbidden events during class counting triggers a fatal `ContractViolationError`.

### Frozen Class Weighting Formula
For valid training pixels where $(\text{valid\_mask} == 1) \land (\text{gt\_labels} \neq 3)$:
1. Compute class pixel counts $N_0, N_1, N_2$ and total valid pixels $N_{\text{total}} = N_0 + N_1 + N_2$.
2. Compute class proportions:
   $$p_c = \frac{N_c}{N_{\text{total}}}$$
3. Compute raw inverse square-root weights with numerical stabilizer $\epsilon = 10^{-12}$:
   $$\text{raw\_weight}_c = \frac{1}{\sqrt{\max(p_c, \epsilon)}}$$
4. Mean-normalize:
   $$\overline{\text{raw}} = \frac{1}{3} \sum_{k=0}^2 \text{raw\_weight}_k, \quad w_c = \frac{\text{raw\_weight}_c}{\overline{\text{raw}}}$$
5. Apply conservative clipping to $[0.5, 2.0]$ to prevent training destabilization:
   $$w_c^{\text{clipped}} = \min(\max(w_c, 0.5), 2.0)$$
6. Final mean-normalization to ensure unit average weight:
   $$\overline{w} = \frac{1}{3} \sum_{k=0}^2 w_c^{\text{clipped}}, \quad w_c^{\text{final}} = \frac{w_c^{\text{clipped}}}{\overline{w}}$$

### Loss Objective Formulation
The class weights $w_c^{\text{final}}$ are applied **only to the cross-entropy term**. The SoftDice term remains unweighted macro-Dice identical to R0:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \text{CE}_{\text{weighted}}(y, \hat{y}; w^{\text{final}}) + 0.5 \cdot \text{Dice}_{\text{macro}}(y, \hat{y})$$

For R0 and R1, weights are frozen at $[1.0, 1.0, 1.0]$. For R2 and R3, $w^{\text{final}}$ is applied.

---

## 5. Component M3: Synchronized Paired SAR Augmentation

### Motivation
Physical SAR sensors experience orbital variations, atmospheric effects, and seasonal radar response shifts. Augmentation must respect the physical properties of Sentinel-1 SLC/GRD backscatter without corrupting spatial alignment or cross-temporal change signals.

### Domain Placement Invariant
Stored SAR patches are linear amplitude backscatter clipped to $[0.0, 0.15]$ and standardized by frozen constants. M3 augmentations **must operate on the raw linear amplitude domain prior to clipping and standardization**.

### Synchronized Transforms
1. **Geometric Transforms**:
   Applied identically across pre-event image, post-event image, change label, and valid mask:
   - Horizontal Flip: $\text{Bernoulli}(0.5)$
   - Vertical Flip: $\text{Bernoulli}(0.5)$
   - Orthogonal Rotation: $k \times 90^\circ$ where $k \in \{0, 1, 2, 3\}$ uniformly.
2. **Radiometric Common-Mode Amplitude Gain**:
   A single conservative scaling factor applied identically across pre-event VV/VH and post-event VV/VH:
   $$\text{gain} \sim \text{Uniform}(0.95, 1.05)$$
   $$x_A' = x_A \times \text{gain}, \quad x_B' = x_B \times \text{gain}$$
3. **Subsequent Pipeline Invariance**:
   Following geometric and radiometric augmentation:
   - Values are clipped to $[0.0, 0.15]$.
   - Nonfinite values are sanitized matching R0.
   - Standard standardization with frozen mean/std constants is applied.

### Prohibitions:
- No independent radiometric gain between pre-event and post-event images.
- No independent gain between VV and VH channels.
- No augmentation during evaluation.
- No augmentation for R0, R1, or R2.

---

## 6. Real-Data Smoke Bounds (R1, R2, R3)

All RST3 executions are bounded smoke tests:
- Configuration parameters:
  - Architecture: `SNUNet_ECAM` (FP32)
  - Seed: 0
  - Partition: Train `[273, 518]`, Eval `[130]`
  - Maximum Epochs: 1
  - Microbatch Size: 8, Gradient Accumulation: 4 (effective batch size 32)
  - Train Steps: exactly 2 optimizer steps (8 microbatches)
  - Eval Batches: 2 batches (32 samples)
- Output Roots:
  - R1: `outputs/robust_source_training/rst3/rst3_r1_smoke_snunet_foldc_s0_v1`
  - R2: `outputs/robust_source_training/rst3/rst3_r2_smoke_snunet_foldc_s0_v1`
  - R3: `outputs/robust_source_training/rst3/rst3_r3_smoke_snunet_foldc_s0_v1`
- Terminal Status: `COMPLETED_COMPONENT_SMOKE`
- Zero failed directories deleted or wiped.
