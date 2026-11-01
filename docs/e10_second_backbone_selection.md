# ChangeGuard Plan V3 E10: Second Backbone Selection and Architectural Specification

**Document Version:** `1.0.0-FROZEN`  
**Date:** 2026-10-25  
**Governing Document:** `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E10 (Section 15.1)  
**Route:** Branch C — Controlled Empirical Study  
**Selected Backbone:** `SiamUnet_diff` (FC-Siam-diff; Daudt et al., IEEE ICIP 2018)  

---

## 1. Candidate Architecture Inventory & Audit

To establish whether the empirical findings of Work Packages E1–E9 (namely: that test-time adaptation provides no reliable improvement over frozen source checkpoints on multi-temporal SAR streams, and that feature normalization explains virtually all adaptation variation) generalize beyond the primary `SNUNet_ECAM` backbone, we inventoried and audited the available change-detection architectures in the project repository and `third_party/KuroSiwo/models/`.

| Candidate Architecture | Citation / Provenance | Parameter Count | Paired SAR Suitability | Target Output | Memory / Compute Feasibility | Selection Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`SiamUnet_diff` (FC-Siam-diff)** | Daudt et al., IEEE ICIP 2018 | **$\sim 1.35\text{M}$** | **Genuinely paired** dual-time Siamese difference encoder | Dense 3-class logits | Low ($\sim 2\text{GB}$ VRAM, fast training) | **SELECTED** |
| **`SiamUnet_conc` (FC-Siam-conc)** | Daudt et al., IEEE ICIP 2018 | $\sim 1.55\text{M}$ | Genuinely paired dual-time Siamese concatenation encoder | Dense 3-class logits | Low ($\sim 2.2\text{GB}$ VRAM) | **REJECTED** (concatenation does not explicitly enforce zero-change difference symmetry) |
| **`BIT_CD`** | Chen et al., IEEE TGRS 2022 | $\sim 11.5\text{M}$ | ResNet-18 + Bitemporal Transformer tokenizer | Dense logits | Moderate ($\sim 8\text{GB}$ VRAM, slow training) | **REJECTED** (heavyweight, designed primarily for optical ImageNet pre-training) |
| **`ChangeFormer`** | Bandara et al., IEEE IGARSS 2022 | $\sim 41.0\text{M}$ | Hierarchical Transformer Encoder-Decoder | Dense logits | High ($\sim 16\text{GB}$ VRAM, prohibitive training cost) | **REJECTED** (not lightweight, high parameter overhead) |
| **`SNUNet_ECAM`** | Fang et al., IEEE TGRS 2021 | $\sim 12.0\text{M}$ | Dense nested connectivity + ECAM attention | Dense 3-class logits | Baseline primary backbone | **PRIMARY (BASELINE)** |

---

## 2. Objective Selection Rationale

`SiamUnet_diff` (FC-Siam-diff) is selected as the sole second backbone for Work Package E10 based on the following objective scientific criteria:

1. **Defensible Canonical Provenance:** Published by Daudt, Le Saux, and Boulch (IEEE ICIP 2018), FC-Siam-diff is the seminal, most widely cited baseline architecture for deep learning-based bi-temporal change detection in the remote sensing literature.
2. **True Architectural Diversity:** Unlike `SNUNet_ECAM` (which utilizes deep nested dense skip connections with channel-attention modules), `SiamUnet_diff` uses a clean Siamese encoder with shared weights and multi-scale difference skip connections $|f_{\text{pre}} - f_{\text{post}}|$. This provides an authentic structural contrast for evaluating whether TTA findings depend on dense attention networks.
3. **Strict Pairwise SAR Compatibility:** Operates natively on 2-channel VV/VH polarimetric SAR inputs at $t_1$ and $t_2$. It requires no optical bands, no SAM masks, no external pre-training, and no elevation/slope channels.
4. **Lightweight Profile:** With only $\sim 1.35\text{M}$ trainable parameters, it trains cleanly within the project schedule and executes evaluation passes rapidly without risk of out-of-memory errors on NVIDIA RTX 3090 GPUs.
5. **Direct Metric Compatibility:** Generates dense 3-class segmentation logits `(B, 3, H, W)` directly compatible with `ControlledSourceLoss` and the ChangeGuard evaluation harness.

---

## 3. Implementation and Wrapper Specification

The upstream implementation in `third_party/KuroSiwo/models/siam_diff.py` terminates with `nn.LogSoftmax(dim=1)`. To ensure exact interface compatibility with `ControlledSourceLoss` (which expects unnormalized raw logits for combined Cross-Entropy and Soft Dice losses), we implement a clean wrapper `SiamDiffWrapper`:

- **Module Path:** `src/models/siam_diff_wrapper.py`.
- **Interface:**
  - `forward(xA, xB)`: Takes pre-event `xA: (B, 2, H, W)` and post-event `xB: (B, 2, H, W)`, and returns raw unnormalized logits `(B, 3, H, W)`.
  - `predict(xA, xB)`: Returns dictionary containing `logits`, `probabilities` (via softmax), `predicted_class` (argmax), and `flood_probability` (class 2).
  - `load_from_checkpoint(path)`: Robust loader supporting both raw model weights and wrapped checkpoints with strict metadata verification.

