# RST1 Sampler, Loss, and Augmentation Audit

**Document version:** 1.0.0
**Audit date:** 2026-10-26
**Status:** Approved technical audit for robust source training
**Audited files:**
- `src/data/source_dataset.py`
- `src/training/source_trainer.py`
- `src/evaluation/losses.py`
- `tools/train_source_model.py`
- `configs/source/snunet_vvvh_all27_seed0_v1_fp32.json`

---

## 1. Existing Source Sampler Audit (Task 3)

### 1.1 Dataset Length and Weighting
- **Dataset Length ($N$):** In the all-27 source configuration, `len(dataset) = 21,462` patches.
- **Event-to-Patch Weighting:** **Completely unweighted uniform patch sampling**.
  - Every patch in the dataset has identical selection probability $P(\text{patch}_i) = \frac{1}{21,462} \approx 4.66 \times 10^{-5}$.
  - The probability of sampling from event $e$ is directly proportional to its patch count $N_e$:
    $$P(\text{event}_e) = \frac{N_e}{\sum_{k=1}^{27} N_k} = \frac{N_e}{21,462}$$

### 1.2 Sampler and Shuffle Behavior
- **PyTorch DataLoader Implementation (`tools/train_source_model.py:669–676`):**
  ```python
  train_loader = DataLoader(
      train_dataset,
      batch_size=micro_bs,  # 8
      shuffle=True,
      num_workers=int(args.num_workers),
      pin_memory=True,
      worker_init_fn=seed_worker,
  )
  ```
- **Sampler Class:** Default `torch.utils.data.RandomSampler(train_dataset, replacement=False)`.
- **Replacement Policy:** Sampling **without replacement** within each epoch. Every patch is visited exactly once per epoch.

### 1.3 Batch Construction and Tail-Batch Handling
- **Micro-Batch Size:** $B_{\text{micro}} = 8$ patches.
- **Gradient Accumulation:** $S_{\text{accum}} = 4$ steps.
- **Effective Batch Size:** $B_{\text{effective}} = B_{\text{micro}} \times S_{\text{accum}} = 32$ patches.
- **Divisibility & Tail-Batch:**
  $$\frac{21,462}{8} = 2,682.75 \implies 2,682 \text{ full batches of 8} + 1 \text{ tail batch of 6}$$
  `drop_last=False` is the default. The final tail batch of 6 patches is accumulated and updated at epoch termination.

### 1.4 Worker Seeding and RNG
- **Worker Init Function:**
  ```python
  def seed_worker(worker_id):
      worker_seed = (seed + worker_id + 1) % (2**32)
      np.random.seed(worker_seed)
      random.seed(worker_seed)
  ```
  Guarantees deterministic epoch shuffling across worker processes given a fixed primary seed.

### 1.5 Severe Domination by Large Events
Because sampling is patch-uniform, optimizer trajectories are heavily dominated by the few geographic regions that contain large numbers of patches:
- **Event 1111009:** Contains 7,556 patches $\to$ **$35.21\%$** of all gradient updates.
- **Event 1111004:** Contains 4,086 patches $\to$ **$19.04\%$** of all gradient updates.
- **Top 4 Events (1111009, 1111004, 427, 1111011):** Account for **$71.01\%$** (15,241 patches) of all training batches.
- **Bottom 8 Events Combined (496, 174, 421, 554, 567, 498, 118, 273):** Total only 704 patches $\to$ **$3.28\%$** of all updates.
- **Extreme Disparity Ratio:** Event 1111009 contributes **$359.8\times$** more updates than Event 496 (21 patches).
This empirical imbalance provides the primary motivation for **Component M1 (Event-Balanced Sampling)**.

---

## 2. Existing Loss Audit (Task 4)

### 2.1 Formal Loss Formulation
The loss function is instantiated as `ControlledSourceLoss` in `src/evaluation/losses.py:92–153`.

$$\mathcal{L}_{\text{total}} = w_{\text{CE}} \cdot \mathcal{L}_{\text{CE}} + w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}$$

With frozen parameters:
$$w_{\text{CE}} = 0.5, \quad w_{\text{Dice}} = 0.5$$

### 2.2 Multiclass Cross-Entropy Component ($\mathcal{L}_{\text{CE}}$)
- **Implementation:** `torch.nn.CrossEntropyLoss(ignore_index=3, reduction='mean')`.
- **Class Weights:** Unweighted $[1.0, 1.0, 1.0]$.
- **Pixel-Level Formula:**
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{c=0}^2 \exp(z_{i, c})} \right)$$
  where $\mathcal{M} = \{i \mid y_i \neq 3\}$ is the set of valid pixels in the batch.
- **Ignore Index:** $3$ (invalid/masked pixels generate exactly zero gradient).

### 2.3 Multiclass Soft Dice Component ($\mathcal{L}_{\text{Dice}}$)
- **Implementation:** `SoftDiceLoss` (`src/evaluation/losses.py:13–90`).
- **Classes Included:** Classes $\{0, 1, 2\}$ (No Water, Permanent Water, Flood Water).
- **Class Weights:** Equal weighting ($1/3$ per class):
  $$\mathcal{L}_{\text{Dice}} = \frac{1}{3} \sum_{c=0}^2 \mathcal{L}_{\text{Dice}}^{(c)}$$
- **Per-Class Formula with Squared Denominator:**
  $$\mathcal{L}_{\text{Dice}}^{(c)} = 1.0 - \frac{2 \sum_{i \in \mathcal{M}} p_{i, c} t_{i, c} + \epsilon}{\sum_{i \in \mathcal{M}} p_{i, c}^2 + \sum_{i \in \mathcal{M}} t_{i, c}^2 + \epsilon}$$
  where:
  - $p_{i, c} = \text{Softmax}(z_i)_c$ is the predicted probability for class $c$;
  - $t_{i, c} = \mathbb{I}(y_i = c)$ is the one-hot binary ground truth;
  - $\epsilon = 1.0$ is the smooth numerical stabilizer.

### 2.4 Numerical Stability and Empty-Class Behavior
- **Smooth Epsilon ($\epsilon = 1.0$):**
  - If class $c$ is absent in the batch ($\sum t_{i, c} = 0$) and the model predicts near-zero probability ($\sum p_{i, c} \approx 0$):
    $$\mathcal{L}_{\text{Dice}}^{(c)} = 1.0 - \frac{0 + 1.0}{0 + 1.0} = 0.0$$
    The loss smoothly evaluates to $0.0$, eliminating false penalties on absent classes.
  - If class $c$ is absent but the model falsely predicts $p > 0$, the loss increases smoothly, penalizing false alarms.
  - Division by zero is mathematically impossible.
- **Batch-Level Reduction:** Reduction is computed across all valid pixels in the micro-batch (`reduction="batch"`), preventing high-variance division artifacts from patches containing very few valid pixels.

### 2.5 Independence from Target Data and Split Statistics
- No target data, development event labels, or former test event labels are accessed or referenced in any loss term.
- Class weights do not depend on global dataset-wide frequency heuristics; they remain strictly unweighted $1:1:1$.

---

## 3. Existing Augmentation Audit (Task 5)

### 3.1 Active Training Augmentations
- **State in R0:** **Strictly None**.
- `configs/source/snunet_vvvh_all27_seed0_v1_fp32.json` and upstream `data_config.json:13` set `"data_augmentations": false`.
- The dataset loader applies zero spatial transforms, zero rotations, zero flips, zero color jitter, and zero noise.

### 3.2 Evaluation Augmentation
- Evaluation is strictly deterministic and unaugmented. No test-time augmentation (TTA) or multi-scale evaluation is enabled in source evaluation.

### 3.3 Requirements for Future Component M3 (Paired SAR Augmentation)
When data augmentation is developed under RST3:
1. **Geometric Synchronization:** Pre-event ($x_A$), post-event ($x_B$), sensory mask ($M_{\text{valid}}$), and label ($y$) must undergo identical, synchronized spatial transformations (horizontal flip, vertical flip, valid 90-degree rotations).
2. **Domain-Correct Radiometric Transform:** Radiometric adjustments must be applied strictly to raw linear amplitudes prior to clamping and z-score normalization. Adding noise or gain to standardized tensors $(x-\mu)/\sigma$ distorts the physical SAR distribution and is prohibited.
3. **Switchable Isolation:** Component M3 must have a clean toggle allowing bit-exact reproduction of the unaugmented R0 baseline when disabled.
