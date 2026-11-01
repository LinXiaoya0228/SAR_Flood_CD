# IGARSS 2027 Conference Paper Draft

**Paper Title:** Event-Balanced Source Training for Robust Cross-Event SAR Flood Change Detection  
**Target Venue:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2027)  
**Format:** IEEE Conference 2-Column Format (4 Pages Maximum)  
**Main Source:** `papers/igarss2027/changeguard_igarss2027_draft_v1.tex`  
**Bibliography:** `papers/igarss2027/references.bib`  
**Figures:** `papers/igarss2027/figures/` (Vector PDFs and high-resolution PNGs)  
**Tables:** `papers/igarss2027/tables/` (`table1_primary_confirmation.tex`, `table2_architecture_transfer.tex`)

---

## Submission-Readiness Checklist

- [x] **Strict 4-Page Length Fit**:
  - The document is structured across 6 compact sections plus Acknowledgment and References.
  - Estimated word count: ~2,600 words, including 2 compact tables (`\resizebox{\columnwidth}{!}`) and 1 single-column vector figure (`fig2_primary_per_event_paired_deltas.pdf`), fitting within the 4-page conference limit.
- [x] **Title & Abstract**:
  - Unbiased, informative title avoiding exaggerated novelty claims.
  - Structured abstract concisely reporting baseline, R2 gains, failure of P3, failure of SiamUnet_diff transfer, and retrospective target validation.
- [x] **Complete Scientific Narrative**:
  - Section I: TTA motivation, failure modes under SAR flood imbalance, transition to target-free robust source training.
  - Section II: M1 (event-balanced sampling), M2 (training-only class-balanced loss), frozen R2 definition.
  - Section III: 18-event cross-validation protocol, KuroSiwo benchmark, pre-registered P1/P2/P3 criteria.
  - Section IV: Primary confirmation on SNUNet-ECAM: P1 passed (+0.0136), P2 passed (+0.0511), P3 failed (+75.1% variance), bootstrap CI crosses zero.
  - Section V: Negative architecture transfer to SiamUnet_diff (Macro F1 -0.0099, tail F1 -0.0448, stop rule triggered) and retrospective historical target performance (+0.0507 F1, 0.0s / 0.0MB TTA overhead).
  - Section VI: Conclusion and explicit limitations (P3 failure, architecture specificity, retrospective target designation).
- [x] **Figure & Table Inclusion**:
  - Table 1: Primary confirmatory results across Folds A, B, and Grand 18-event evaluation.
  - Table 2: Architecture transfer results comparing SNUNet-ECAM and SiamUnet_diff at Seed 0.
  - Figure 1: Paired per-event F1 deltas (`fig2_primary_per_event_paired_deltas.pdf`).
- [x] **Citations & Bibliography**:
  - Verified BibTeX file with peer-reviewed DOIs and URLs (`references.bib`).
  - Standard `IEEEtran` bibliography style.
- [x] **Integrity & Boundary Compliance**:
  - Zero claims of universal superiority.
  - Retrospective historical target events clearly identified.
  - No unsupported mechanistic claims for SiamUnet_diff failure.

---

## Build and Compilation Instructions

> [!NOTE]
> On the current development host environment, the system `pdflatex` binary is not installed in the system PATH. The LaTeX document and tables have been validated for syntax, environment closures, citation matching, and asset path integrity via `tools/validate_rst11_report.py`.

To compile the PDF on a system with TeX Live or MacTeX installed:

```bash
cd papers/igarss2027
pdflatex changeguard_igarss2027_draft_v1
bibtex changeguard_igarss2027_draft_v1
pdflatex changeguard_igarss2027_draft_v1
pdflatex changeguard_igarss2027_draft_v1
```

Or using `latexmk`:

```bash
latexmk -pdf changeguard_igarss2027_draft_v1.tex
```
