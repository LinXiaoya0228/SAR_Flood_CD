# ChangeGuard Interim Report Outline

- **Document Version:** 1.0.0  
- **Project:** ChangeGuard FYP (NTU School of Electrical & Electronic Engineering)  
- **Interim Report Submission Deadline:** `2026-11-10`  
- **Target Manuscript Readiness:** March 2027  
- **Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  

---

## 1. Executive Overview & Problem Background
- **The Operational Challenge:** Remote sensing disaster response requires rapid, reliable satellite flood extent mapping. Optical satellites (Sentinel-2, Landsat) are obstructed by persistent cloud cover and storm precipitation during flood emergencies.
- **Why Synthetic Aperture Radar (SAR) Change Detection Matters:** Spaceborne active microwave sensors (Sentinel-1 C-band SAR) provide day-and-night, all-weather Earth Observation. Multi-temporal change detection isolates newly inundated flood waters by contrasting pre-disaster backscatter reference images against post-disaster passes.
- **The Out-of-Distribution Vulnerability:** Deep learning segmentation models trained on historical geographic events experience significant performance degradation when deployed on unseen geographic disaster streams due to differences in topography, soil moisture, surface roughness, and vegetation cover.

---

## 2. Benchmark Task and Label Semantics
- **KuroSiwo Benchmark Corpus:** 43 global disaster events across diverse biogeographic zones (Bountos et al., NeurIPS 2024).
- **Temporal Pairing:** Dual temporal inputs $(x_A, x_B) = (\text{pre\_event\_2}, \text{post\_event})$.
- **Four-Class Ground Truth Semantics:**
  - Class 0: `No Water` (dry background)
  - Class 1: `Permanent Water` (pre-existing lakes, reservoirs, river channels)
  - Class 2: `Flood Water` (newly inundated flood areas — target change class)
  - Class 3: `Invalid / NoData` (unmapped / missing satellite coverage)
- **Masking Protocol:** Double-masking rule $\text{valid} = (\text{valid\_mask} == 1) \land (\text{label} \ne 3)$, masking out pre-existing water and invalid pixels to isolate true flood change.

---

## 3. Related Work & Literature Foundations
- **Supervised SAR Flood Mapping:** Historical thresholding algorithms, Otsu methods, Siamese U-Nets (SNUNet, Siam-Conc, ChangeFormer).
- **Test-Time Adaptation (TTA) Paradigms:**
  - Fully test-time adaptation via entropy minimization (Tent; Wang et al., ICLR 2021).
  - Test-time normalization statistics updating (AdaBN; Li et al., 2018).
  - Anti-forgetting sample filtering (EATA; Niu et al., ICML 2022).
- **The Core TTA Assumption Failure in EO:** Standard computer vision TTA assumes balanced object recognition (e.g. ImageNet-C). In satellite change detection, severe class imbalance (flood $< 2\%$ of pixels) causes standard unsupervised entropy minimization to favor degenerate all-background solutions or massive false-positive over-predictions on speckle noise.

---

## 4. Dataset Completeness & Official Checkpoint Provenance Audit
*(Completed Work — Audited)*
- **Data Completeness Audit:** Confirmed 43 events on disk (31,707 train manifest records, 67,490 test manifest records, 782 GB total). Event 514 confirmed absent on disk (`configured_but_absent`).
- **Official Checkpoint Provenance Findings:**
  - Checkpoint `checkpoints/kuro_snunet_official.pt` has first convolution shape `(32, 3, 3, 3)`, requiring 3 channels per temporal branch.
  - Official repository documents only `["vv", "vh"]` in primary configuration, leaving the 3rd channel ambiguous (provisionally inferred as DEM from contradictory config keys and unmerged PRs).
  - Checkpoint metadata (epoch 13, loss 0.147) lacks training split provenance, data augmentation logs, and exact normalization parameters.
  - **Scientific Decision:** The official checkpoint is retained as a secondary external reference; it cannot serve as the sole primary model for rigorous academic claims.

---

## 5. Two-Week Pilot Methodology & Findings
*(Completed Work — Audited in `docs/two_week_pilot_complete_evidence_report.md`)*
- **Gate A Verification (Degradation Existence):** Tent (lr=1e-4) degraded macro F1 from 0.5646 to 0.5451 and slashes pooled F1 from 0.7020 to 0.3614. On Event 279, F1 collapsed from 0.6301 to 0.2316 ($\Delta = -0.3985$).
- **Gate B Outcome (Batch-Level Risk Signaling Falsification):**
  - Predeclared change-specific risk score achieved AUROC **0.5102** (near chance) and within-event macro Spearman $r_s = +0.0796$.
  - Generic batch norm drift achieved pooled AUROC 0.9501, but within-event correlation collapsed to **+0.0458** (confounded by stream length).
- **Day 8 Learning Rate Robustness:** Proved that TTA degradation is learning-rate invariant across $10^{-5}, 3\times 10^{-5}, 10^{-4}$ and AdaBN ($lr=0.0$).
- **Day 9 Event Suitability & The Single-Class Fold Phenomenon:** Oracle selector achieves Macro F1 0.6637, but out-of-sample LOEO cross-validation degenerates to Reject-All / Source-Only (0.5646) due to having only 1 beneficial validation event.
- **Strategic Recommendation:** `PIVOT_FAILURE_CHARACTERIZATION` (pivot from fragile heuristic controllers to systematic characterization of failure modes and safety boundaries).

---

## 6. Controlled Source-Model Architecture & Protocol
*(In-Progress Work — Week 0)*
- **Proposed 22/5 Source-Inner Split:**
  - 22 inner-train events (19,899 samples, 92.72% of source data).
  - 5 inner-val events (1,563 samples, 7.28% of source data: Events `[267, 496, 497, 518, 1111005]`).
  - Stratified across flood prevalence tertiles and permanent water variation.
- **Two Controlled Model Configurations:**
  - **Model A:** SNUNet with pure dual-polarization SAR `[VV, VH]` (2 channels per branch).
  - **Model B:** SNUNet with topographic ablation `[VV, VH, slope]` (3 channels per branch).
- **Gate S Acceptance Criteria:** Inner-validation Macro F1 $\ge 0.50$, recall $\ge 0.45$, dry false positive rate $\le 2.0\%$.

---

## 7. Execution Timeline: Weeks 1 to 4 Plan
*(Planned Work)*
- **Week 1 (Oct 12–18): Source Pipeline Implementation & Smoke Tests**
  - Modular dataset loader supporting both 2-channel and 3-channel input pipelines.
  - Multi-worker training harness supporting CE+Dice loss, AMP mixed precision, and checkpoint resume.
  - 50-batch smoke tests on GPU 0 confirming forward/backward pass and loss convergence.
- **Week 2 (Oct 19–25): Controlled Source-Model Training Execution**
  - Train Model A (`[VV, VH]`) and Model B (`[VV, VH, slope]`) across seeds 42, 43, 44 on the 22/5 split.
  - Select optimal recipe and fixed epoch schedule based on 5 inner-validation events.
- **Week 3 (Oct 26–Nov 1): Final Retraining & Baseline Freezing**
  - Retrain optimal configurations on all 27 source events using fixed epoch schedule.
  - Freeze final controlled checkpoints and record cryptographic hashes.
- **Week 4 (Nov 2–8): Interim Report Finalization & Video Recording**
  - Compile final experimental comparison tables between controlled source models and official reference.
  - Submit written Interim Report and upload presentation video by November 10 deadline.

---

## 8. Research Integrity Safeguards
- **Zero Test Set Contamination:** 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) strictly unaccessed.
- **Independent Validation Isolation:** TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) never used for source model selection.
- **Full Negative Results Reporting:** All failure modes, degraded runs, and rejected hypotheses retained in permanent version control.

---

## 9. Interim Presentation Video Storyboard (10-Minute Structure)
1. **Minute 0:00–1:30 | Motivation & Problem:** SAR Earth Observation during floods; failure of out-of-distribution transfer; why naive TTA collapses into background or false positives.
2. **Minute 1:30–3:30 | The KuroSiwo Benchmark & Data Auditing:** Task semantics, 43-event corpus, official checkpoint provenance gap, need for controlled baselines.
3. **Minute 3:30–6:00 | Pilot Findings (The Sensitivity-Calibration Paradox):** Gate A degradation evidence, Gate B falsification of batch-level rollback, why AdaBN collapses pooled F1, and the single-class fold barrier.
4. **Minute 6:00–8:00 | The Controlled Source Model Phase:** 22/5 source-inner split, Model A (`[VV, VH]`) vs Model B (`[VV, VH, slope]`) design, Gate S criteria.
5. **Minute 8:00–10:00 | Project Roadmap & Academic Contributions:** Timeline to March 2027 completion, publication targets, and safety boundaries in satellite remote sensing.
