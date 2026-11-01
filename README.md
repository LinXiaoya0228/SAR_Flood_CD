# ChangeGuard: Event-Balanced Source Training for Robust Cross-Event SAR Flood Change Detection

Official implementation and reproducibility package for:
> **"Event-Balanced Source Training for Robust Cross-Event SAR Flood Change Detection"**  
> Final Year Project (FYP) Dissertation.

---

## 1. Executive Summary & Research Question

Synthetic Aperture Radar (SAR) satellite imagery provides all-weather, day-and-night observation for flood mapping during humanitarian crises. However, deep neural networks deployed for cross-event SAR change detection exhibit notable performance volatility across unseen disaster events.

While recent research has explored **Test-Time Adaptation (TTA)** (e.g., Tent, CoTTA, EATA) to adapt models during deployment on unlabeled test data, in our benchmark on multi-temporal SAR flood change detection, standard test-time adaptation strategies exhibited substantial performance instability and frequent degradation in flood predictions under the severe foreground class imbalance characteristic of SAR flood scenes (where flood pixels often constitute $< 2\%$ of the image area). Under these conditions, unsupervised entropy minimization frequently suppressed positive flood predictions.

**The Central Question:**  
*Can an existing-component, target-free source-training strategy—specifically uniform event sampling ($\mathbf{M1}$) combined with training-only inverse-square-root class weighting ($\mathbf{M2}$)—improve cross-event generalization and tail disaster reliability in SAR flood change detection without requiring test-time parameter updates?*

### The ChangeGuard Strategy ($\mathbf{R2}$)
ChangeGuard combines established training components into a unified, target-free recipe:
1. **$\mathbf{M1}$ Event-Balanced Sampling:** Samples training patches uniformly across diverse disaster events rather than uniformly across the pooled patch archive. This prevents physically extensive flood events from disproportionately dominating gradient updates.
2. **$\mathbf{M2}$ Inverse-Square-Root Class Weighting:** Applies soft inverse-square-root class weights ($w_c = 1/\sqrt{f_c}$) strictly to the training loss (Cross-Entropy + Dice), balancing foreground gradient contribution without destabilizing false alarms.
3. **Target-Free Deployment:** At test time, the resulting model operates in standard feed-forward evaluation mode without test-time parameter updates, test-time backpropagation passes, or adaptation buffers.

---

## 2. Core Empirical Findings & Explicit Limitations

All hypotheses were evaluated through pre-registered decision rules and independent cross-validation protocols:

### A. Primary Confirmatory Evaluation on SNUNet-ECAM (18 Events, 3 Seeds, 12 Runs)
- **Criterion P1 (Macro F1 Gain $\ge +0.010$):** **PASSED**.  
  Grand Macro F1 improved from $0.7202 \pm 0.0044$ (R0) to $0.7338 \pm 0.0132$ (R2), achieving a paired gain of $\mathbf{+0.0136}$ (+1.36 pp, 95% event bootstrap CI: `[-0.0045, +0.0351]`).
- **Criterion P2 (Tail Reliability Gain $\ge +0.020$ on Bottom-3 Events):** **PASSED**.  
  Mean F1 on the lowest three disaster events rose from $0.3987 \pm 0.0298$ (R0) to $0.4498 \pm 0.0248$ (R2), a paired gain of $\mathbf{+0.0511}$ (+5.11 pp).
- **Criterion P3 (Seed Variance Reduction $\Delta\text{Var} \le 0$):** **FAILED**.  
  Between-seed variance increased by $\mathbf{+75.1\%}$ on the Grand Evaluation ($0.0132$ vs $0.0044$), showing that event-balanced sampling introduces modest initialization sensitivity.

### B. Supplementary Weighting-Only Ablation ($\text{M2\_ONLY}$, 6 Runs)
- Evaluating class weighting alone without event-balanced sampling yielded Grand Macro F1 of $0.7126$ ($\Delta = -0.0076$ vs R0) and Bottom-3 Tail F1 of $0.4270$ ($\Delta = +0.0283$ vs R0).
- Adding event-balanced sampling ($\mathbf{M1}$) on top of class weighting ($\mathbf{R2}$ vs $\mathbf{M2\_ONLY}$) produced a statistically significant gain of $\mathbf{+0.0212}$ (+2.12 pp, 95% bootstrap CI: `[+0.0010, +0.0442]`, $p = 0.0181$).
- **Finding:** Class weighting alone does not yield an overall macro F1 benefit over baseline R0; event-balanced sampling ($\mathbf{M1}$) accounts for the measured macro F1 and tail disaster recovery gains observed in R2 (+5.11 pp tail gain for R2 vs +2.83 pp for M2_ONLY, and $+0.0212$ gain for R2 vs M2_ONLY).

### C. Architecture Specificity on SiamUnet_diff
- **Seed 0 Screening:** When tested on SiamUnet_diff across all 18 events at Seed 0, R2 degraded performance relative to R0 (Grand Macro F1: $0.7139$ vs $0.7238$, $\Delta = -0.0099$; Bottom-3 Tail F1: $0.3902$ vs $0.4350$, $\Delta = -0.0448$), triggering the pre-registered stopping rule.
- **3-Seed Supplementary Replication:** A supplementary multi-seed study extending SiamUnet_diff to Seeds 1 and 2 (12 runs total, 108 evaluated event records) confirmed this outcome (*status: completed*). Across 3 seeds, Grand Macro F1 dropped from $0.7248 \pm 0.0063$ (R0) to $0.7184 \pm 0.0042$ (R2), yielding a paired mean delta of $\mathbf{-0.0064 \pm 0.0056}$ (95% paired event bootstrap CI `[-0.0240, +0.0114]`, $p = 0.7546$) and degrading Bottom-3 Tail F1 by $-0.0423$.
- **Finding:** The hypothesis of universal architectural transfer is rejected. ChangeGuard gains are architecture-specific to dense multi-scale attention backbones (SNUNet-ECAM) and do not transfer to lightweight difference decoders.

### D. Retrospective Kuro Siwo Target Evaluation
- **10-Event Test-Time Adaptation Benchmark:** On the 10 former Kuro Siwo test events evaluated in the TTA benchmark, R2 achieved Grand Target Macro F1 of $\mathbf{0.8086}$, outperforming the unadapted baseline R0 ($0.7453$, $\Delta = +0.0633$), Tent ($0.7628$, $\Delta = +0.0458$, 95% CI `[+0.0274, +0.0648]`), and CoTTA ($0.7669$, $\Delta = +0.0417$, 95% CI `[+0.0260, +0.0585]`), without requiring test-time parameter updates or adaptation latency.
- **16-Event Full Historical Target Extension:** Across all 16 historical target events, full-source R2 models achieved a Grand Macro F1 of $\mathbf{0.7380 \pm 0.0059}$ compared to $0.6873 \pm 0.0223$ for baseline R0 ($\Delta = +0.0507$, 95% event bootstrap CI `[+0.0234, +0.0845]`), with 14 of 16 events exhibiting positive paired gains.

### E. External Zero-Shot Generalization on GEOID-Flood (1,301 Tiles, 9 EMS Activations)
- Evaluated on $873,090,528$ valid pixels across 9 Copernicus Emergency Management Service (EMS) flood activations under verified input transform parity (NaN filled with 0.15).
- **Macro F1:** R0 = $0.5492 \pm 0.0326$, R2 = $0.5475 \pm 0.0167$ ($\Delta = -0.0017$, 95% event bootstrap CI `[-0.0238, +0.0217]`). The paired difference is not statistically distinguishable from zero at the 95% confidence level. *(Note: an interval spanning zero indicates that a non-zero effect was not detected in this sample; it does not demonstrate equivalence or statistical parity between recipes).*
- **Precision Shift:** R2 delivered a $\mathbf{+4.66\text{ pp}}$ gain in flood precision ($0.5439$ vs $0.4972$) with a corresponding reduction in recall ($0.5994$ vs $0.6519$).

---

## 3. Repository Structure

```text
SAR_Flood_CD/
├── configs/                       # Frozen experiment configuration files (JSON)
│   ├── robust_source_training/    # RST4 (discovery), RST5 (confirmation), RST8 (SiamUnet)
│   ├── ablation_m2_only/          # M2_ONLY supplementary ablation configs
│   └── external_evaluation/       # GEOID-Flood evaluation configs
├── docs/                          # Comprehensive technical and scientific documentation
│   ├── REPRODUCIBILITY_INDEX.md   # Paper Table & Figure Claim-to-Artifact Inventory
│   ├── CHECKPOINTS.md             # Model parameter counts, architectures, and SHA-256 hashes
│   └── reproduction/              # Step-by-step modular reproduction guides
│       ├── environment.md         # Environment setup and third-party dependencies
│       ├── datasets.md            # Dataset acquisition and directory layouts
│       ├── splits_and_labels.md   # Event splits and binary label semantics
│       ├── training.md            # Training R0, M2_ONLY, and R2 models
│       ├── evaluation.md          # Evaluation and bootstrap significance testing
│       ├── target_and_tta.md      # Kuro Siwo historical target evaluation
│       ├── geoid_external_eval.md # External GEOID-Flood evaluation and NaN parity
│       ├── regenerate_tables_figures.md # Table/figure one-click regeneration
│       └── architecture_replication.md  # SiamUnet_diff replication guide & findings
├── environment/                   # Conda environment and pip requirements files
│   ├── environment_sar_cd.yml     # Core training environment definition
│   ├── requirements_sar_cd.txt    # Core pip requirements
│   ├── environment_vfm_env.yml    # Geospatial/external evaluation environment
│   └── requirements_vfm_env.txt   # Geospatial pip requirements
├── papers/                        # Publication manuscripts, LaTeX tables, and vector assets
│   └── igarss2027/                # IEEE conference 4-page paper draft, tables, and figures
├── results/                       # Tracked compact canonical CSV/JSON results and figures
│   ├── canonical/                 # 18 compact canonical source data files
│   ├── tables/                    # All 7 synthesis tables in CSV and Markdown
│   └── figures/                   # All 9 publication figures (PDF & PNG) + comparisons
├── scripts/                       # External evaluation scripts and launch tools
├── src/                           # Core implementation source code
│   ├── data/                      # Dataset loaders (KuroSourceDataset, evaluation)
│   ├── models/                    # Architecture wrappers (SNUNet-ECAM, SiamUnet_diff)
│   ├── losses/                    # Standard and class-weighted CE + Dice loss functions
│   └── external_evaluation/       # GEOID-Flood dataset adapter
├── tests/                         # Automated unit tests and smoke verification
└── tools/                         # Offline analysis, smoke verification, and table builders
```

---

## 4. Quick-Start Guide & Availability Boundaries

### Available Components vs. Checkpoint Downloads
- **Immediately Reproducible without Large Downloads:** This public repository includes complete source code, frozen configs, automated tests, and tracked compact canonical data (`results/`). Anyone can clone the repository and immediately execute:
  - Unit tests and synthetic forward passes (`tools/smoke_reproducibility.py`),
  - Table regeneration from compact data (`tools/robust_source_training/build_rst10_tables.py`),
  - Publication figure rendering (`tools/robust_source_training/build_rst10_figures.py`),
  - Offline TTA and historical comparison tools (`tools/robust_source_training/run_r2_vs_tta_offline_comparison.py`).
- **Checkpoint Availability:** Pre-trained checkpoint binary weights (`.pt` files, totaling $> 3$ GB) are not hosted in this Git repository due to file-size constraints. Public distribution and download hosting for trained weights remain under institutional and author review. Full end-to-end retraining from scratch is supported using the provided configurations and public Kuro Siwo data.

### Step 1: Environment Installation
```bash
# Clone the repository
git clone https://github.com/LinXiaoya0228/SAR_Flood_CD.git
cd SAR_Flood_CD

# Create and activate conda environment
conda create -n sar_cd python=3.10 -y
conda activate sar_cd

# Install PyTorch and core scientific dependencies
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.9.2 tqdm
```

### Step 2: Third-Party Dependencies (KuroSiwo Model Architecture)
The official SNUNet-ECAM architecture is loaded from the peer-reviewed Kuro Siwo repository:
```bash
git clone https://github.com/Orion-AI-Lab/KuroSiwo.git third_party/KuroSiwo
cd third_party/KuroSiwo && git checkout 4347ed1 && cd ../..
```
*(Alternatively, set the `KURO_ROOT` environment variable pointing to your local KuroSiwo clone.)*

### Step 3: Portable Smoke Verification (No GPU / Private Data Required)
Verify environment integrity, configuration syntax, model forward passes, and metric computations:
```bash
python tools/smoke_reproducibility.py
```

### Step 4: Regenerate Paper Tables and Figures (10 Seconds)
Regenerate all 7 canonical synthesis tables and 9 publication figures from published compact results:
```bash
# Generate all LaTeX and CSV tables (written to results/tables/)
python tools/robust_source_training/build_rst10_tables.py

# Generate all publication vector figures (written to results/figures/)
python tools/robust_source_training/build_rst10_figures.py
```

---

## 5. Checkpoint Registry

Checkpoints are cryptographically verified using SHA-256 hashes to guarantee provenance. Due to file size constraints ($\approx 138$ MB per model, $> 3$ GB across the matrix), raw weights are preserved outside git history and are not distributed as direct downloads in this repository.

| Model Checkpoint | Architecture | Recipe | Events | SHA-256 Hash |
|---|---|---|---|---|
| `R0_all27_s0` | SNUNet-ECAM | R0 (Baseline) | All 27 | `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f` |
| `R0_all27_s1` | SNUNet-ECAM | R0 (Baseline) | All 27 | `72c4be7856fe4bb06560d61d3ab8e72780e81fdff84660f8dadfe2f8bd385d7a` |
| `R0_all27_s2` | SNUNet-ECAM | R0 (Baseline) | All 27 | `0d403d4e8885b8b64c99fc76850be6bf324b65f2d5474e2dcb7e72f22c03fedd` |
| `R2_all27_s0` | SNUNet-ECAM | R2 (ChangeGuard) | All 27 | `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7` |
| `R2_all27_s1` | SNUNet-ECAM | R2 (ChangeGuard) | All 27 | `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2` |
| `R2_all27_s2` | SNUNet-ECAM | R2 (ChangeGuard) | All 27 | `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893` |

*For complete listings of all 12 confirmatory checkpoints, 6 M2_ONLY ablation checkpoints, and SiamUnet_diff checkpoints, see [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).*

---

## 6. Detailed Reproduction Documentation

For full step-by-step procedures, refer to our modular guides:
- **[Reproducibility Index](docs/REPRODUCIBILITY_INDEX.md):** Complete Claim-to-Artifact mapping for all tables and figures.
- **[Environment Setup](docs/reproduction/environment.md):** Hardware, Conda environments, and KuroSiwo dependency setup.
- **[Dataset Acquisition](docs/reproduction/datasets.md):** Download instructions and directory layouts for Kuro Siwo and GEOID-Flood.
- **[Splits & Label Semantics](docs/reproduction/splits_and_labels.md):** Disjoint event definitions, fold mappings, and nodata masking.
- **[Training & Resumption](docs/reproduction/training.md):** Instructions for launching and resuming R0, M2_ONLY, and R2 runs.
- **[Source Evaluation](docs/reproduction/evaluation.md):** Event-macro metrics and paired bootstrap confidence intervals.
- **[Historical Target & TTA](docs/reproduction/target_and_tta.md):** Retrospective evaluation against Tent and CoTTA baselines.
- **[External GEOID-Flood](docs/reproduction/geoid_external_eval.md):** Evaluation workflow and NaN correction on GEOID-Flood.
- **[Regenerating Tables & Figures](docs/reproduction/regenerate_tables_figures.md):** Commands to regenerate all artifacts from compact data.
- **[Architecture Replication](docs/reproduction/architecture_replication.md):** Findings and reproduction workflow for the SiamUnet_diff 3-seed replication study.

---

## 7. Citation

If you use this codebase, models, or evaluation protocols in your research, please cite:

```bibtex
@inproceedings{changeguard2027igarss,
  author    = {Lin, Xiaoya and Collaborators},
  title     = {Event-Balanced Source Training for Robust Cross-Event {SAR} Flood Change Detection},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2027},
  note      = {Manuscript in preparation}
}

@thesis{lin2026changeguard,
  author    = {Lin, Xiaoya},
  title     = {Robust Cross-Event Deep Learning for Satellite {SAR} Flood Change Detection},
  school    = {Nanyang Technological University},
  year      = {2026},
  type      = {Final Year Project (FYP) Dissertation}
}
```

---

## 8. License and Attribution

- **Project Code & Checkpoints:** Project licensing terms and public checkpoint release decisions are reserved for determination by the author and institution.
- **Third-Party Code:** The official `SNUNet_ECAM` architecture implementation is authored by the Orion-AI-Lab / KuroSiwo contributors ([https://github.com/Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo)).
- **Datasets:** The Kuro Siwo dataset is provided under CC-BY 4.0; the GEOID-Flood dataset is provided under Copernicus Emergency Management Service data policies.
