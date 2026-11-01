# ChangeGuard Work Package RST11: Formal Review Report

**Date:** 2026-10-28  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Author:** ChangeGuard Research & Engineering Team  
**Governing Documents:**
- `docs/ROBUST_SOURCE_TRAINING_PROPOSAL_V1.md`
- `docs/research_integrity_protocol.md`
- Frozen RST10 Synthesis (`commit 55d1912`, `docs/robust_source_training/rst10_synthesis_v1.md`)

---

## 1. Executive Summary

Work Package **RST11** represents the final synthesis, writing, and dissemination package of the ChangeGuard project. All scientific investigation, exploratory screening (RST4), confirmatory cross-validation (RST6/RST7), architectural boundary analysis (RST8), deployable training and retrospective extension (RST9), and forensic reporting (RST10) were previously completed and frozen.

Under RST11, zero new training runs, zero inference executions, and zero hyperparameter modifications were authorized or conducted. The package focused exclusively on creating:
1. A rigorous, claim-level source map anchoring every statement in reproducible artifacts;
2. A comprehensive, 12-section Final Year Project (FYP) Dissertation Draft;
3. A 4-page IEEE conference paper draft for IGARSS 2027;
4. Complete reproducibility appendices, verified bibliographies, and table/figure indices;
5. An automated validation harness (`tools/validate_rst11_report.py`) enforcing numerical consistency, checkpoint integrity, and boundary compliance.

Both draft deliverables are complete, self-contained, and validated against canonical artifacts.

---

## 2. Deliverables Inventory

| Deliverable | Path | Format | Status | Details |
|---|---|---|---|---|
| **Claim Source Map** | `docs/final_report/changeguard_claim_source_map_v1.md` | Markdown | Complete | 13 primary claims mapped to exact canonical artifacts, tables, figures, permitted interpretations, and explicit prohibitions. |
| **Bibliography** | `docs/final_report/references.bib` | BibTeX | Complete | 13 verified references with real DOIs and URLs (KuroSiwo, SNUNet, SiamUnet, Tent, CoTTA, Class-Balanced Loss, Focal Loss, AdamW, etc.). Duplicate copy placed in `papers/igarss2027/references.bib`. |
| **Table & Figure Index** | `docs/final_report/changeguard_table_figure_index_v1.md` | Markdown | Complete | Indexes all 7 synthesis tables and 9 vector/raster figures with backing data paths, captions, evidence scopes, and limitations. |
| **Reproducibility Appendix** | `docs/final_report/changeguard_reproducibility_appendix_v1.md` | Markdown | Complete | Exhaustive hardware specs, Python virtual environment details, repository tree, SHA-256 checkpoint ledger, and CLI reproduction commands. |
| **FYP Full Report Draft** | `docs/final_report/changeguard_fyp_full_draft_v1.md` | Markdown | Complete | 12 complete sections, 5,631 words, zero placeholders. Fully covers project motivation, TTA failure modes, robust source formulation, confirmatory results, architecture limits, and retrospective benchmarks. |
| **IGARSS 2027 Paper Draft** | `papers/igarss2027/changeguard_igarss2027_draft_v1.tex` | LaTeX (IEEEtran) | Complete | 4-page IEEE conference format, 1,913 words across 6 compact sections, 2 formatted tables, 1 vector figure, full BibTeX citations. |
| **IGARSS Tables** | `papers/igarss2027/tables/` | LaTeX | Complete | `table1_primary_confirmation.tex` and `table2_architecture_transfer.tex`. |
| **IGARSS Checklist** | `papers/igarss2027/README.md` | Markdown | Complete | 4-page limit verification, document outline, asset index, and build instructions. |
| **Validation Tool** | `tools/validate_rst11_report.py` | Python 3 | Complete | Automated script scanning drafts for numbers, checkpoint hashes, retrospective labels, prohibited phrases, and broken citations. |

---

## 3. Scientific Fidelity and Numerical Audit

All quantitative statements in both the FYP Dissertation and the IGARSS paper drafts were cross-checked against the frozen RST10 synthesis tables and raw evaluation logs:

### 3.1 Primary Confirmatory Matrix (RST7, SNUNet-ECAM, 18 Source Events, Seeds 0, 1, 2)
- **Grand Macro F1:** $0.7202 \to 0.7338$ ($\mathbf{\Delta = +0.0136}$, $+1.36$ percentage points). **Criterion P1 Satisfied** ($\Delta \ge +0.010$).
- **Bottom-3 Tail F1:** $0.3987 \to 0.4498$ ($\mathbf{\Delta = +0.0511}$, $+5.11$ percentage points, $+12.82\%$ relative gain). **Criterion P2 Satisfied** ($\Delta \ge +0.020$).
- **Worst Single Event:** $0.2991 \to 0.2929$ ($\Delta = -0.0062$).
- **Paired 95% Bootstrap Confidence Interval:** `[-0.0045, +0.0351]` ($p = 0.0779$, crosses zero; reported openly as non-significant at $\alpha = 0.05$).
- **Seed Variance Analysis:**
  - Fold A across-seed SD: $0.0120 \to 0.0106$ ($-11.3\%$, variance reduction satisfied).
  - Fold B across-seed SD: $0.0035 \to 0.0165$ ($+368.5\%$, variance inflation).
  - Mean across-seed SD: $0.0078 \to 0.0136$ ($+75.1\%$ variance increase).
  - **Criterion P3 Formally Failed** and documented without obfuscation.

### 3.2 Architecture Transfer Boundary (RST8, SiamUnet_diff, 18 Events, Seed 0)
- **Grand Macro F1:** $0.7238 \to 0.7139$ ($\mathbf{\Delta = -0.0099}$).
- **Bottom-3 Tail F1:** $0.4350 \to 0.3902$ ($\mathbf{\Delta = -0.0448}$).
- **Worst Single Event:** $0.3565 \to 0.2660$ ($\mathbf{\Delta = -0.0905}$).
- **Outcome:** Negative transfer confirmed. Formal stop rule triggered at Seed 0; multi-seed expansion cancelled; claim strictly bounded to SNUNet-ECAM.

### 3.3 Deployable Models & Retrospective Historical Target Extension (RST9)
- **Training Budget:** 16 epochs, exactly 10,736 optimizer steps per seed across all 3 seeds.
- **Cryptographic Hashes Verified:**
  - Seed 0: `9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7`
  - Seed 1: `279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2`
  - Seed 2: `b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893`
- **Retrospective 16 Historical Target Events:**
  - Macro F1: $0.6873 \to 0.7380$ ($\mathbf{\Delta = +0.0507}$, 95% CI `[+0.0234, +0.0845]`).
- **Contextual Benchmark on 10 Former Test Events:**
  - Baseline R0: $0.7453 \pm 0.0345$
  - AdaBN: $0.7599 \pm 0.0158$ ($\Delta = +0.0146$)
  - Tent: $0.7628 \pm 0.0147$ ($\Delta = +0.0175$)
  - CoTTA: $0.7669 \pm 0.0132$ ($\Delta = +0.0216$)
  - **Recipe R2:** $0.8086 \pm 0.0072$ ($\mathbf{\Delta = +0.0633}$, 95% CI `[+0.0312, +0.0984]`).
- **Target-Time Operational Overhead:**
  - Additional latency: **0.0 s**
  - Additional memory: **0.0 MB**
  - Test-time parameter updates: **0**

---

## 4. Governance, Tone, and Integrity Verification

The text in both deliverables strictly enforces all project governance constraints:
1. **Central Story Alignment:** Both drafts articulate the exact governing narrative: Generic test-time adaptation was not consistently reliable for paired SAR flood change detection, motivating a target-free robust source-training strategy (event-balanced sampling + training-only class-balanced loss). The strategy improved average and lower-tail cross-event performance on SNUNet-ECAM without target-time updates, while its failure on SiamUnet_diff defines an explicit architectural boundary.
2. **Retrospective Labeling:** All 16 target events are consistently designated as "retrospective historical target events" and never mischaracterized as unseen prospective test events.
3. **Absence of Hyperbole:** Zero instances of promotional phrasing ("state of the art", "SOTA", "novel breakthrough", "guarantee", "universal").
4. **Transparent Negative Results:** Criteria P3 failure and SiamUnet_diff negative transfer are featured prominently in abstracts, summaries, and discussion sections.
5. **Preservation of Core Integrity Records:** `docs/e11_final_scientific_conclusions.md` and `docs/e11_final_test_closeout.md` remain completely unmodified.

---

## 5. Automated Validation Results

Execution of `python tools/validate_rst11_report.py`:
```text
======================================================================
ChangeGuard RST11 Report & Paper Validation
======================================================================
[1/6] Checking for unresolved placeholders...
[2/6] Checking prohibited phrasing and boundaries...
[3/6] Checking canonical metrics and checkpoint hashes...
[4/6] Checking BibTeX citation resolution...
[5/6] Checking LaTeX structure and environment closure...
[6/6] Checking referenced figure assets on disk...
----------------------------------------------------------------------
SUCCESS: All validation checks passed!
  - FYP Draft lines: 385, words: 5631
  - IGARSS Draft lines: 153, words: 1913
  - Canonical metrics: 100% matched with frozen RST10 synthesis
  - Checkpoint SHA-256 hashes: 100% verified
  - Zero prohibited phrases or unresolved placeholders
======================================================================
```

---

## 6. Conclusion and Readiness Statement

Work Package **RST11** is successfully completed. The outputs provide a complete, mathematically verified, and reproducible foundation for human examination of the Final Year Project and peer review of the IGARSS 2027 conference submission.
