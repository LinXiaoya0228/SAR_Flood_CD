#!/usr/bin/env python3
"""Build professionally formatted Word (.docx) documents and rendered PDFs
for ChangeGuard FYP Report and IGARSS 2027 Paper.

Deliverables:
- exports/rst11_word/ChangeGuard_FYP_Report_Review_v1.docx
- exports/rst11_word/ChangeGuard_FYP_Report_Review_v1.pdf
- exports/rst11_word/ChangeGuard_IGARSS2027_Review_v1.docx
- exports/rst11_word/ChangeGuard_IGARSS2027_Review_v1.pdf
- exports/rst11_word/WORD_EXPORT_REVIEW_NOTES_v1.md
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
import pypdf

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = REPO_ROOT / "exports" / "rst11_word"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = REPO_ROOT / "papers" / "igarss2027" / "figures"
TABLES_DIR = REPO_ROOT / "outputs" / "robust_source_training" / "rst10" / "tables"


def set_cell_shading(cell, color_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="8" w:space="0" w:color="1F4E79"/>'
        f'<w:bottom w:val="{val}" w:sz="8" w:space="0" w:color="1F4E79"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def format_table_header(row, bg_color="1F4E79", text_color=RGBColor(255, 255, 255), font_size=Pt(9.0)):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for cell in row.cells:
        set_cell_shading(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = text_color
                r.font.size = font_size


def format_table_data_row(row, is_even=False, font_size=Pt(8.5)):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    bg = "F7F9FB" if is_even else "FFFFFF"
    for cell in row.cells:
        if is_even:
            set_cell_shading(cell, bg)
        set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.size = font_size


def add_figure_with_caption(doc, img_name: str, caption: str, width_in=5.8):
    img_path = FIGURES_DIR / img_name
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    if img_path.is_file():
        run_img.add_picture(str(img_path), width=Inches(width_in))
    else:
        run_img.add_text(f"[Figure missing: {img_name}]")

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def add_table_caption(doc, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(caption)
    r.font.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)


def convert_to_pdf(docx_path: Path, out_dir: Path) -> Path:
    cmd = [
        "/usr/bin/libreoffice",
        "--headless",
        "--convert-to", "pdf",
        str(docx_path),
        "--outdir", str(out_dir)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pdf_path = out_dir / (docx_path.stem + ".pdf")
    if not pdf_path.is_file():
        print(f"Warning: PDF conversion failed for {docx_path}:\n{res.stderr}")
    return pdf_path


def add_equation_paragraph(doc, eq_text: str, eq_num: str = ""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(eq_text)
    r.font.name = "Cambria Math"
    r.font.size = Pt(10.5)
    r.font.italic = True
    if eq_num:
        r_num = p.add_run(f"    ({eq_num})")
        r_num.font.name = "Times New Roman"
        r_num.font.italic = False
        r_num.font.size = Pt(9.5)


def build_fyp_report_docx() -> Path:
    print("Building ChangeGuard_FYP_Report_Review_v1.docx...")
    doc = docx.Document()

    # Base page settings: A4
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(5)

    # ==========================================
    # 1. COVER PAGE (Section 1)
    # ==========================================
    section.different_first_page_header_footer = True
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("ChangeGuard: Change-Preserving Safe Test-Time Adaptation for SAR Flood Mapping\n")
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    r_sub = p_title.add_run("Robust Source Training for Cross-Event SAR Flood Change Detection")
    r_sub.font.size = Pt(14)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(60)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.line_spacing = 1.4

    r = p_meta.add_run("Submitted by:\n")
    r.font.size = Pt(12)
    r.font.bold = True
    r = p_meta.add_run("Lin Xiaoya\n")
    r.font.size = Pt(13)
    r.font.bold = True
    r = p_meta.add_run("[Matriculation / Student ID — confirm]\n\n")
    r.font.size = Pt(11)
    r.font.italic = True

    r = p_meta.add_run("Supervised by:\n")
    r.font.size = Pt(12)
    r.font.bold = True
    r = p_meta.add_run("[Supervisor Name — confirm: Professor / Associate Professor]\n")
    r.font.size = Pt(12)
    r = p_meta.add_run("[Co-supervisor Name — confirm if applicable]\n\n")
    r.font.size = Pt(11)
    r.font.italic = True

    p_school = doc.add_paragraph()
    p_school.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_school.paragraph_format.space_before = Pt(80)
    p_school.paragraph_format.line_spacing = 1.3

    r = p_school.add_run("School of Electrical & Electronic Engineering\n")
    r.font.size = Pt(13)
    r.font.bold = True

    r = p_school.add_run(
        "A final year project report presented to Nanyang Technological University "
        "in partial fulfilment of the requirements for the Degree of Bachelor of Engineering\n\n"
    )
    r.font.size = Pt(11)

    r = p_school.add_run("Academic Year 2026/2027 (Submission 2027)")
    r.font.size = Pt(12)
    r.font.bold = True

    # ==========================================
    # 2. FRONT MATTER (Section 2, Roman numerals)
    # ==========================================
    s2 = doc.add_section(WD_SECTION.NEW_PAGE)
    s2.different_first_page_header_footer = False
    s2.header.is_linked_to_previous = False
    s2.footer.is_linked_to_previous = False

    sectPr2 = s2._sectPr
    pgNumType2 = OxmlElement('w:pgNumType')
    pgNumType2.set(qn('w:fmt'), 'lowerRoman')
    pgNumType2.set(qn('w:start'), '1')
    sectPr2.append(pgNumType2)

    footer2 = s2.footer.paragraphs[0]
    footer2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_f2 = footer2.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    run_f2._r.extend([f1, it, f2, f3])

    # 1. Abstract (Strictly <= 1 page, font 11pt, 1.15 line spacing)
    h_abs = doc.add_paragraph()
    h_abs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_abs.paragraph_format.space_before = Pt(8)
    h_abs.paragraph_format.space_after = Pt(12)
    r = h_abs.add_run("Abstract")
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p1 = doc.add_paragraph(
        "Automated flood disaster mapping from satellite Synthetic Aperture Radar (SAR) imagery requires robust change "
        "detection models capable of generalizing across disparate geographic regions, terrain types, and soil moisture "
        "conditions. While generic unsupervised test-time adaptation (TTA) has been proposed to overcome deployment domain "
        "shifts by updating model statistics or weights on streaming test data, our controlled multi-seed experiments "
        "across 16 global flood events demonstrate that standard TTA techniques—including batch normalization statistic "
        "adaptation (AdaBN), entropy minimization (Tent), and continual adaptation (CoTTA)—fail to provide statistically "
        "reliable performance gains in multi-temporal paired SAR flood mapping. Instead, unsupervised adaptation frequently "
        "causes localized performance degradation and catastrophic recall collapse due to extreme temporal class imbalance "
        "and non-stationary radar speckle dynamics."
    )
    p1.paragraph_format.line_spacing = 1.15
    p1.paragraph_format.space_after = Pt(4)
    p1.runs[0].font.size = Pt(11)

    p2 = doc.add_paragraph(
        "To address these vulnerabilities, this project develops and evaluates ChangeGuard, a target-free robust source-training "
        "strategy. ChangeGuard investigates the empirical hypothesis that cross-event generalization can be enhanced at training "
        "time without requiring target-domain model updates or test-time compute. Our analysis reveals severe geographic sample "
        "imbalance across the 27 source events in the global KuroSiwo benchmark, spanning a 358.7× disparity in available patches. "
        "We formulate a parsimonious training strategy—Recipe R2—which combines event-balanced sampling (M1) to equalize geographic "
        "representation with training-only inverse-square-root class-balanced cross-entropy loss (M2) to counter majority-class "
        "background suppression."
    )
    p2.paragraph_format.line_spacing = 1.15
    p2.paragraph_format.space_after = Pt(4)
    p2.runs[0].font.size = Pt(11)

    p3 = doc.add_paragraph(
        "In formal pre-registered event-disjoint confirmation across 18 held-out source events (Folds A and B, random seeds 0, 1, 2) "
        "on the primary SNUNet-ECAM architecture, Recipe R2 passed pre-registered criteria for average generalization gain "
        "(Criterion P1: grand event-macro F1 improved from 0.7202 to 0.7338, Δ = +0.0136) and tail reliability improvement "
        "(Criterion P2: bottom-three event mean F1 improved from 0.3987 to 0.4498, Δ = +0.0511, a +12.82% relative gain). "
        "However, pre-registered Criterion P3 (seed stability) was not satisfied because inter-seed variance increased under Fold B "
        "(standard deviation rose from 0.0035 to 0.0165), and the 10,000-resample paired event-cluster bootstrap 95% confidence interval "
        "crossed zero ([-0.0045, +0.0351], p = 0.0779). Crucially, when transferred to a lightweight Siamese difference backbone "
        "(SiamUnet_diff, 1.35M parameters), Recipe R2 failed to generalize (Δ = -0.0099 macro F1, Δ = -0.0448 bottom-3 F1), establishing "
        "an explicit architectural limitation. Finally, retrospective evaluation on 16 previously observed historical target events "
        "demonstrated that deployable R2 models achieved 0.7380 macro F1 (Δ = +0.0507, 95% CI [+0.0234, +0.0845]) with zero target-time "
        "adaptation overhead."
    )
    p3.paragraph_format.line_spacing = 1.15
    p3.paragraph_format.space_after = Pt(4)
    p3.runs[0].font.size = Pt(11)

    # 2. Acknowledgment
    doc.add_page_break()
    h_ack = doc.add_paragraph()
    h_ack.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_ack.paragraph_format.space_before = Pt(12)
    h_ack.paragraph_format.space_after = Pt(18)
    r = h_ack.add_run("Acknowledgments")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p = doc.add_paragraph(
        "I express sincere gratitude to the European Space Agency (ESA) and Copernicus Earth Observation Programme for "
        "providing the Sentinel-1 radar imagery, and to the authors of the KuroSiwo benchmark dataset for their curation of "
        "multi-sensor flood inundation annotations."
    )
    p = doc.add_paragraph(
        "I thank the School of Electrical & Electronic Engineering at Nanyang Technological University (NTU Singapore) "
        "for computational resources and research facilities supporting this Final Year Project."
    )
    p = doc.add_paragraph(
        "[Personal acknowledgments — confirm if desired: Mentors, lab colleagues, family, and friends who supported the "
        "completion of this study.]"
    )
    p.runs[0].font.italic = True
    p.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # 3. Table of Contents
    doc.add_page_break()
    h_toc = doc.add_paragraph()
    h_toc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_toc.paragraph_format.space_before = Pt(12)
    h_toc.paragraph_format.space_after = Pt(14)
    r = h_toc.add_run("Table of Contents")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    # Automatic TOC Word Field
    p_fld = doc.add_paragraph()
    p_fld.paragraph_format.space_after = Pt(2)
    run_toc_fld = p_fld.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'TOC \\o "1-3" \\h \\z \\u'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    run_toc_fld._r.extend([f1, it, f2, f3])

    toc_entries = [
        ("Abstract", "i"),
        ("Acknowledgments", "ii"),
        ("Table of Contents", "iii"),
        ("Acronyms", "v"),
        ("Symbols", "vi"),
        ("List of Figures", "vii"),
        ("List of Tables", "viii"),
        ("Chapter 1: Introduction", "1"),
        ("    1.1 Motivations", "1"),
        ("    1.2 Objectives and Scope", "2"),
        ("    1.3 Organisation of the Report", "2"),
        ("Chapter 2: Related Work / Literature Review", "3"),
        ("    2.1 Deep Learning for SAR Change Detection", "3"),
        ("    2.2 Test-Time Adaptation and Unsupervised Domain Adaptation", "3"),
        ("    2.3 Class Imbalance and Long-Tailed Learning", "4"),
        ("Chapter 3: Paired SAR Flood Change Detection", "5"),
        ("    3.1 Physical Principles of SAR Backscatter and Inundation", "5"),
        ("    3.2 Formulation of Bi-Temporal Change Detection", "5"),
        ("    3.3 Semantic Label Hierarchy and Class Semantics", "5"),
        ("Chapter 4: Dataset and Event-Disjoint Protocol", "6"),
        ("    4.1 KuroSiwo Global Benchmark", "6"),
        ("    4.2 Cryptographic Event-Disjoint Partitioning", "6"),
        ("    4.3 Evaluation Metrics and Statistical Framework", "6"),
        ("Chapter 5: Controlled Study of Test-Time Adaptation", "7"),
        ("    5.1 Experimental Protocol", "7"),
        ("    5.2 Evaluated TTA Algorithms", "7"),
        ("    5.3 Empirical Findings: Instability and Performance Collapse", "7"),
        ("Chapter 6: Motivation for Robust Source Training", "8"),
        ("    6.1 Discovery of Severe Geographic Source Imbalance", "8"),
        ("    6.2 Class Imbalance and Background Suppression", "9"),
        ("    6.3 Pre-Registered Hypothesis Criteria", "9"),
        ("Chapter 7: Event-Balanced Robust Source Training", "10"),
        ("    7.1 Component Formulation (R0, M1, M2, M3)", "10"),
        ("    7.2 The Discovery Staircase and Parsimonious Promotion of R2", "10"),
        ("Chapter 8: Experimental Results", "12"),
        ("    8.1 Primary Source Confirmation on SNUNet-ECAM (RST7)", "12"),
        ("    8.2 Per-Event Performance Breakdown", "13"),
        ("    8.3 Cross-Backbone Architecture Transfer Boundary (RST8)", "14"),
        ("Chapter 9: Ablation and Reliability Analysis", "16"),
        ("    9.1 Decomposed Contribution of Components", "16"),
        ("    9.2 Tail Robustness vs. Single Worst-Case Outlier Analysis", "16"),
        ("    9.3 Detailed Seed Variance Analysis", "17"),
        ("Chapter 10: Comparison with Test-Time Adaptation", "18"),
        ("    10.1 Contextual Retrospective Comparison (RST9)", "18"),
        ("    10.2 Operational Advantage Over Online Adaptation", "19"),
        ("Chapter 11: Limitations and Research Integrity", "22"),
        ("    11.1 Methodological Limitations", "22"),
        ("    11.2 Research Integrity Protocol Adherence", "22"),
        ("Chapter 12: Conclusion and Future Work", "23"),
        ("    12.1 Conclusion", "23"),
        ("    12.2 Recommendations for Future Work", "23"),
        ("Reflection on Learning Outcome Attainment", "24"),
        ("Acknowledging/Declaring the Use of GAI", "25"),
        ("References", "26"),
        ("Appendix A: Reproducibility Package and Checkpoint Ledger", "28"),
        ("Appendix B: Full Per-Event Confirmatory and Retrospective Data", "29"),
    ]

    for title, page_str in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(title)
        if title.startswith("Chapter") or title.startswith("Reflection") or title.startswith("Acknowledging") or title.startswith("References") or title.startswith("Appendix"):
            r.font.bold = True
        p.add_run(" " + "." * max(2, int(65 - len(title) * 0.9)) + " " + page_str)

    # 4. Acronyms
    doc.add_page_break()
    h_acro = doc.add_paragraph()
    h_acro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_acro.paragraph_format.space_before = Pt(12)
    h_acro.paragraph_format.space_after = Pt(14)
    r = h_acro.add_run("Acronyms")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    acronyms = [
        ("SAR", "Synthetic Aperture Radar"),
        ("TTA", "Test-Time Adaptation"),
        ("RST", "Robust Source Training"),
        ("ECAM", "Ensemble Channel Attention Module"),
        ("SNUNet", "Siamese Nested UNet for Change Detection"),
        ("SiamUnet", "Fully Convolutional Siamese UNet Architecture"),
        ("CE", "Cross-Entropy Loss"),
        ("CB-CE", "Class-Balanced Cross-Entropy Loss"),
        ("CI", "Confidence Interval"),
        ("SD", "Standard Deviation"),
        ("IoU", "Intersection over Union (Jaccard Index)"),
        ("VRAM", "Video Random Access Memory"),
        ("VV", "Vertical Transmit and Vertical Receive Polarization"),
        ("VH", "Vertical Transmit and Horizontal Receive Polarization"),
        ("DEM", "Digital Elevation Model"),
        ("GAI", "Generative Artificial Intelligence"),
        ("AdaBN", "Adaptive Batch Normalization"),
        ("Tent", "Test-Time Entropy Minimization"),
        ("CoTTA", "Continual Test-Time Adaptation"),
    ]

    t_acro = doc.add_table(rows=len(acronyms) + 1, cols=2)
    set_table_borders(t_acro)
    t_acro.rows[0].cells[0].paragraphs[0].add_run("Acronym")
    t_acro.rows[0].cells[1].paragraphs[0].add_run("Definition")
    format_table_header(t_acro.rows[0])
    for idx, (acr, defn) in enumerate(acronyms, start=1):
        row = t_acro.rows[idx]
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(acr); r0.font.bold = True
        row.cells[1].paragraphs[0].add_run(defn)
        format_table_data_row(row, is_even=(idx % 2 == 0))

    # 5. Symbols
    doc.add_page_break()
    h_sym = doc.add_paragraph()
    h_sym.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_sym.paragraph_format.space_before = Pt(12)
    h_sym.paragraph_format.space_after = Pt(14)
    r = h_sym.add_run("Symbols")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    symbols = [
        ("y ∈ {0, 1, 2}", "Ground-truth semantic pixel label (0: dry land, 1: permanent water, 2: flood)"),
        ("p_c", "Predicted class posterior probability for class c ∈ C"),
        ("w_c", "Inverse-square-root class loss weight for class c"),
        ("w_bar_c", "Normalized, clipped class weight applied in training cross-entropy"),
        ("f_c", "Empirical pixel frequency of class c measured strictly on training split"),
        ("C", "Set of active valid semantic classes {0, 1, 2}"),
        ("L_CE", "Standard unweighted multi-class cross-entropy loss"),
        ("L_CB", "Training-only class-balanced cross-entropy loss"),
        ("L_Dice", "Multi-class soft Dice loss"),
        ("Δ (Delta)", "Paired metric difference between Recipe R2 and Baseline R0 (R2 − R0)"),
        ("α", "Statistical hypothesis testing significance level (α = 0.05)"),
        ("σ", "Sample standard deviation across experimental initialization seeds"),
        ("Var", "Sample variance across experimental initialization seeds"),
    ]

    t_sym = doc.add_table(rows=len(symbols) + 1, cols=2)
    set_table_borders(t_sym)
    t_sym.rows[0].cells[0].paragraphs[0].add_run("Symbol")
    t_sym.rows[0].cells[1].paragraphs[0].add_run("Mathematical Definition / Usage")
    format_table_header(t_sym.rows[0])
    for idx, (sym, defn) in enumerate(symbols, start=1):
        row = t_sym.rows[idx]
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(sym); r0.font.bold = True
        row.cells[1].paragraphs[0].add_run(defn)
        format_table_data_row(row, is_even=(idx % 2 == 0))

    # 6. List of Figures
    doc.add_page_break()
    h_lof = doc.add_paragraph()
    h_lof.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_lof.paragraph_format.space_before = Pt(12)
    h_lof.paragraph_format.space_after = Pt(14)
    r = h_lof.add_run("List of Figures")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    # Native LOF field
    p_lof_fld = doc.add_paragraph()
    p_lof_fld.paragraph_format.space_after = Pt(2)
    run_lof_fld = p_lof_fld.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'TOC \\h \\z \\c "Figure"'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    run_lof_fld._r.extend([f1, it, f2, f3])

    figures_list = [
        ("Figure 1: Component Discovery Staircase on Group C (RST4)", "11"),
        ("Figure 2: Primary Confirmatory Paired Per-Event F1 Differences on SNUNet-ECAM (RST7)", "13"),
        ("Figure 3: Tail Event Robustness vs. Single Worst-Case Outlier Comparison", "17"),
        ("Figure 4: Grand Precision-Recall Trade-off Analysis Across Confirmatory Folds", "13"),
        ("Figure 5: Three-Panel Across-Seed Variance Distribution on Fold A, Fold B, and Historical Targets", "17"),
        ("Figure 6: Geographic Patch Count Distribution Across 27 KuroSiwo Source Events", "8"),
        ("Figure 7: Negative Cross-Backbone Architecture Transfer on SiamUnet_diff (RST8)", "15"),
        ("Figure 8: Retrospective Historical Target Generalization on 16 Previously Observed Events (RST9)", "19"),
        ("Figure 9: Test-Time Operational Latency and VRAM Footprint Trade-off vs. Online TTA", "21"),
    ]

    for title, page_str in figures_list:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        p.add_run(title)
        p.add_run(" " + "." * max(2, int(68 - len(title) * 0.9)) + " " + page_str)

    # 7. List of Tables
    doc.add_page_break()
    h_lot = doc.add_paragraph()
    h_lot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_lot.paragraph_format.space_before = Pt(12)
    h_lot.paragraph_format.space_after = Pt(14)
    r = h_lot.add_run("List of Tables")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    # Native LOT field
    p_lot_fld = doc.add_paragraph()
    p_lot_fld.paragraph_format.space_after = Pt(2)
    run_lot_fld = p_lot_fld.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'TOC \\h \\z \\c "Table"'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    run_lot_fld._r.extend([f1, it, f2, f3])

    tables_list = [
        ("Table 1: Component Discovery Staircase Performance on Discovery Fold C", "10"),
        ("Table 2: Primary Event-Disjoint Confirmation Results on SNUNet-ECAM (RST7)", "12"),
        ("Table 3: Cross-Backbone Architecture Transfer Evaluation on SiamUnet_diff (RST8)", "14"),
        ("Table 4: Cryptographically Verified Deployable Checkpoint Ledger (RST9)", "18"),
        ("Table 5: Retrospective Generalization Performance on 16 Historical Target Events", "18"),
        ("Table 6: Computational Complexity, Training Time, and Test Latency Resource Audit", "19"),
        ("Table 7: Contextual Performance Benchmark Against Historical TTA Algorithms", "20"),
    ]

    for title, page_str in tables_list:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        p.add_run(title)
        p.add_run(" " + "." * max(2, int(68 - len(title) * 0.9)) + " " + page_str)

    # ==========================================
    # 3. MAIN CHAPTERS (Section 3, Arabic numerals)
    # ==========================================
    s3 = doc.add_section(WD_SECTION.NEW_PAGE)
    s3.different_first_page_header_footer = False
    s3.header.is_linked_to_previous = False
    s3.footer.is_linked_to_previous = False

    sectPr3 = s3._sectPr
    pgNumType3 = OxmlElement('w:pgNumType')
    pgNumType3.set(qn('w:fmt'), 'decimal')
    pgNumType3.set(qn('w:start'), '1')
    sectPr3.append(pgNumType3)

    # Running Header
    header3 = s3.header.paragraphs[0]
    header3.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_h3 = header3.add_run("ChangeGuard: Robust Source Training for SAR Flood Change Detection")
    r_h3.font.size = Pt(8.5)
    r_h3.font.italic = True
    r_h3.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # Footer
    footer3 = s3.footer.paragraphs[0]
    footer3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_f3 = footer3.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    run_f3._r.extend([f1, it, f2, f3])

    def add_chapter_heading(num: int, title: str):
        if num > 1:
            doc.add_page_break()
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(10)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(f"Chapter {num}\n{title}")
        r.font.size = Pt(18)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    def add_sec_heading(title: str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    # CHAPTER 1: INTRODUCTION
    add_chapter_heading(1, "Introduction")
    add_sec_heading("1.1 Motivations")
    doc.add_paragraph(
        "Rapid flood disaster response depends critically on timely, accurate delineation of inundated areas from spaceborne "
        "Earth observation sensors. Optical imaging systems, while providing high spatial resolution, are frequently rendered "
        "ineffective during acute hydrological crises due to persistent cloud cover, precipitation, and severe weather. In contrast, "
        "spaceborne Synthetic Aperture Radar (SAR)—exemplified by the European Space Agency's Copernicus Sentinel-1 constellation—operates "
        "at microwave frequencies (C-band, λ ≈ 5.6 cm) that penetrate atmospheric clouds and precipitation, operating reliably day and night."
    )
    doc.add_paragraph(
        "In operational satellite flood mapping, the standard methodology relies on bi-temporal change detection. A pre-disaster "
        "acquisition (T1) capturing baseline hydrological conditions is paired with a post-disaster acquisition (T2) over the exact "
        "same geographic footprint. Because smooth, open water surfaces act as specular reflectors of microwave radiation, radar pulses "
        "scatter away from the sensor, resulting in characteristic low backscatter intensity (dark pixels). Conversely, dry land, rough "
        "vegetation, and built infrastructure exhibit diffuse backscattering, returning significantly higher energy to the radar receiver. "
        "By comparing pre- and post-event radar returns, deep neural networks aim to segment newly inundated flood water."
    )
    doc.add_paragraph(
        "Despite the clear physical contrast between calm water and rough soil, deep learning models trained on specific historical flood "
        "events often suffer severe performance degradation when deployed to new, unseen geographic regions. This performance degradation "
        "arises from pronounced cross-event distribution shifts: topographic heterogeneity (mountainous relief inducing radar layover, "
        "foreshortening, and shadow); dielectric soil moisture variations; wind-induced surface capillary waves that artificially brighten "
        "water backscatter; and acquisition geometry differences (local incidence angle variations between 29° and 46°)."
    )
    doc.add_paragraph(
        "To mitigate these distribution shifts without requiring real-time manual annotations from human analysts, unsupervised test-time "
        "adaptation (TTA) emerged as an appealing paradigm. TTA algorithms update model normalization statistics (e.g., AdaBN; Li et al., "
        "2018) or optimize affine parameters via self-supervised objectives such as entropy minimization (Tent; Wang et al., 2021) directly "
        "on unlabeled inference streams. However, operational deployment in disaster response imposes severe constraints: zero adaptation "
        "fragility (emergency responders require predictable, deterministic predictions); extreme class imbalance (newly inundated flood "
        "water typically covers less than 2% of scene pixels); and non-Gaussian speckle noise violating standard normalization assumptions. "
        "Under these conditions, unsupervised adaptation frequently collapses into trivial background prediction, wiping out flood detections entirely."
    )

    add_sec_heading("1.2 Objectives and Scope")
    doc.add_paragraph(
        "This Final Year Project investigates a fundamental research question: Can a target-free robust source-training strategy deliver "
        "consistent average and tail performance improvements across heterogeneous flood events without the operational risks, compute "
        "overhead, and instability of online test-time adaptation?"
    )
    doc.add_paragraph(
        "The project formulates and evaluates ChangeGuard under strict research integrity controls: (1) design modular source-training "
        "interventions addressing event-level geographic sample imbalance and pixel-level class imbalance; (2) pre-register clear quantitative "
        "confirmatory criteria (P1 for average generalization, P2 for tail reliability, P3 for across-seed stability); (3) evaluate cross-backbone "
        "generalization on multiple Siamese architectures (SNUNet-ECAM and SiamUnet_diff); and (4) benchmark deployable source models "
        "retrospectively against historical TTA baselines on previously observed disaster scenes."
    )

    add_sec_heading("1.3 Organisation of the Report")
    doc.add_paragraph(
        "The remainder of this report is organized as follows: Chapter 2 reviews related literature in satellite radar flood mapping, deep "
        "change detection architectures, unsupervised domain adaptation, test-time adaptation, and long-tailed learning. Chapter 3 formulates "
        "the paired SAR change detection problem and physical sensor principles. Chapter 4 details the KuroSiwo benchmark and our cryptographic "
        "event-disjoint cross-validation protocol. Chapter 5 presents our controlled empirical study of test-time adaptation failure modes. "
        "Chapter 6 motivates robust source training through an audit of geographic and class imbalances. Chapter 7 introduces the modular ChangeGuard "
        "formulation and exploratory component staircase. Chapter 8 reports the primary confirmatory results on SNUNet-ECAM and cross-backbone transfer "
        "on SiamUnet_diff. Chapter 9 presents detailed ablation and seed reliability analyses. Chapter 10 compares robust source training against online TTA. "
        "Chapter 11 details limitations and research integrity compliance. Chapter 12 concludes with recommendations for future research. Finally, "
        "post-chapter sections present reflections on learning outcome attainment, a declaration on the use of generative AI tools, complete references, "
        "and reproducibility appendices."
    )

    # CHAPTER 2: RELATED WORK
    add_chapter_heading(2, "Related Work / Literature Review")
    add_sec_heading("2.1 Deep Learning for SAR Change Detection")
    doc.add_paragraph(
        "Satellite-based radar flood mapping has evolved from classical radiometric thresholding and statistical distribution fitting "
        "(Martinis et al., 2015; Twele et al., 2016) to modern deep neural networks. Early operational processing chains relied on split-based "
        "thresholding of dual-polarization Sentinel-1 backscatter (VV and VH channels), where empirical thresholds were selected based on local image "
        "histograms. While computationally efficient, global thresholding struggles in urban environments, vegetated floodplains, and regions with "
        "high wind-induced wave roughening."
    )
    doc.add_paragraph(
        "Recent benchmarks have introduced large-scale annotated datasets to train supervised deep learning models for flood mapping, including "
        "Sen1Floods11 (Bonafilia et al., 2020) and KuroSiwo (Bountos et al., 2024). These benchmarks emphasize the superiority of multi-temporal "
        "change detection over single-image water segmentation, as comparing pre- and post-disaster passes allows models to distinguish permanent "
        "water bodies from disaster-induced inundations."
    )
    doc.add_paragraph(
        "Deep learning architectures for remote sensing change detection predominantly employ Siamese feature extraction networks: "
        "Fully Convolutional Siamese Networks (FC-Siam-diff / SiamUnet_diff; Daudt et al., 2018) concatenate absolute feature differences across "
        "multi-scale decoder blocks with 1.35M parameters; and densely connected Siamese networks with Ensemble Channel Attention Modules "
        "(SNUNet-ECAM; Fang et al., 2021) maintain multi-scale high-resolution feature representations with 12.03M parameters."
    )

    add_sec_heading("2.2 Test-Time Adaptation and Unsupervised Domain Adaptation")
    doc.add_paragraph(
        "In Earth observation, addressing geographic domain shifts has historically relied on Unsupervised Domain Adaptation (UDA), requiring "
        "simultaneous access to large pools of labeled source data and unlabeled target data during training (Hoffman et al., 2018). While effective, "
        "UDA is ill-suited for real-time disaster response because emergency responders cannot wait hours to retrain models on newly collected satellite swaths."
    )
    doc.add_paragraph(
        "To bypass target-time retraining, Test-Time Adaptation (TTA) modifies model parameters or statistics directly during inference: "
        "Test-Time Normalization (AdaBN; Li et al., 2018; Schneider et al., 2020) replaces frozen source batch normalization statistics with test-batch "
        "statistics; Entropy Minimization (Tent; Wang et al., 2021) optimizes affine scale and shift parameters by minimizing Shannon entropy; "
        "Anti-Forgetting Adaptation (EATA & SAR; Niu et al., 2022, 2023) introduces sample filtering based on entropy and Fisher information; and "
        "Continual Test-Time Adaptation (CoTTA; Wang et al., 2022) employs teacher-student exponential moving average (EMA) consistency."
    )

    add_sec_heading("2.3 Class Imbalance and Long-Tailed Learning")
    doc.add_paragraph(
        "The challenges of severe class imbalance in computer vision have motivated loss reweighting strategies, such as Focal Loss (Lin et al., 2017) "
        "and Class-Balanced Loss based on effective sample numbers (Cui et al., 2019). Similarly, group distributionally robust optimization "
        "(Group DRO; Sagawa et al., 2020) and balanced sampling (Buda et al., 2018) address disparities across sub-populations. In ChangeGuard, "
        "we build directly upon these foundational concepts, adapting them to the discrete geographic event structure of multi-temporal satellite SAR imagery."
    )

    # CHAPTER 3: PAIRED SAR FLOOD CHANGE DETECTION
    add_chapter_heading(3, "Paired SAR Flood Change Detection")
    add_sec_heading("3.1 Physical Principles of SAR Backscatter and Inundation")
    doc.add_paragraph(
        "The radar observations originate from the European Space Agency's Sentinel-1 constellation, operating at C-band (5.405 GHz, λ ≈ 5.6 cm) "
        "in Interferometric Wide (IW) swath mode with Ground Range Detected (GRD) format. Radar pulses are transmitted and received in dual-polarization: "
        "VV polarization (vertical transmit, vertical receive) provides high sensitivity to water-land dielectric boundaries due to distinct Fresnel "
        "reflection coefficients; and VH polarization (vertical transmit, horizontal receive) captures cross-polarized volume scattering from forest "
        "canopies and depolarizing urban infrastructure."
    )
    doc.add_paragraph(
        "Sensor data are processed in linear amplitude (A = sqrt(σ0)). During input ingestion, pixel values are clamped to the range [0.0, 0.15] "
        "to eliminate extreme non-physical speckle spikes and standardized using fixed global constants: μ_norm = [0.0953, 0.0264], "
        "σ_norm = [0.0427, 0.0215]."
    )

    add_sec_heading("3.2 Formulation of Bi-Temporal Change Detection")
    doc.add_paragraph(
        "Let X denote the space of paired satellite radar acquisitions. For any geographic flood event, the observation comprises an aligned pre-disaster "
        "image x_A ∈ R^(2×H×W) acquired at time T1 and a post-disaster image x_B ∈ R^(2×H×W) acquired at time T2 over an identical spatial grid of dimension "
        "H × W = 256 × 256. The model predicts a dense segmentation map y_hat ∈ {0, 1, 2}^(H×W)."
    )
    add_equation_paragraph(doc, "x = (x_VV, x_VH),    y_hat = f_θ(x_A, x_B)", "3.1")

    add_sec_heading("3.3 Semantic Label Hierarchy and Class Semantics")
    doc.add_paragraph(
        "The authoritative semantic label spaces are defined as: Class 0: unflooded land / dry ground (unchanged terrestrial surfaces); "
        "Class 1: permanent water (pre-existing water bodies present at both T1 and T2); Class 2: newly inundated flood water (the primary scientific "
        "evaluation positive class; surfaces dry at T1 and covered by water at T2); and Class 3: ignore / invalid mask (nodata pixels, permanent radar "
        "shadows, or out-of-swath regions). Class 3 is strictly excluded from all loss calculations, gradient backpropagation, and metric evaluations."
    )

    # CHAPTER 4: DATASET AND PROTOCOL
    add_chapter_heading(4, "Dataset and Event-Disjoint Protocol")
    add_sec_heading("4.1 KuroSiwo Global Benchmark")
    doc.add_paragraph(
        "Experiments are conducted on the global KuroSiwo benchmark (Bountos et al., 2024), which compiles 43 disaster events worldwide spanning "
        "diverse biomes, climatic conditions, and flood triggers (hurricanes, monsoons, dam collapses, snowmelts). The 43 events are partitioned into "
        "two permanent subsets: (1) Source Partition (27 events, 21,462 patches of size 256×256), used exclusively for supervised model training "
        "and confirmatory cross-validation; and (2) Historical Target Partition (16 events), subdivided into 6 development events and 10 former test "
        "events observed during earlier project phases (Weeks 3–5) serving strictly as a retrospective benchmark."
    )

    add_sec_heading("4.2 Cryptographic Event-Disjoint Partitioning")
    doc.add_paragraph(
        "To conduct rigorous exploratory discovery and confirmatory validation without data leakage, the 27 source events were partitioned into "
        "three mutually disjoint groups of 9 events each: Group A (9 events, 7,411 patches), Group B (9 events, 7,556 patches), and Group C "
        "(9 events, 6,495 patches). These groups define three experimental folds: Fold C (exploratory discovery, training on Groups A+B, evaluating on "
        "Group C); Fold A (primary confirmation, training on Groups B+C, evaluating on Group A across seeds 0, 1, 2); and Fold B (primary confirmation, "
        "training on Groups A+C, evaluating on Group B across seeds 0, 1, 2)."
    )

    add_sec_heading("4.3 Evaluation Metrics and Statistical Framework")
    doc.add_paragraph(
        "Because disaster response prioritizes newly inundated water, all primary evaluation metrics evaluate segmentation of Class 2: Precision, "
        "Recall, F1 score, and Intersection over Union (IoU). For an evaluation partition containing E distinct flood events, Event-Macro F1 is "
        "computed as the unweighted arithmetic mean across events:"
    )
    add_equation_paragraph(doc, "Macro F1 = (1 / E) * ∑_{e=1}^E F1_e", "4.1")
    doc.add_paragraph(
        "To evaluate lower-tail vulnerability and worst-case risk, we evaluate Bottom-3 Event Mean F1 (mean of the three lowest-scoring events in the "
        "partition) and Worst-Event F1 (min_{e} F1_e). Paired differences (Δ = R2 − R0) are evaluated with 10,000 paired cluster-bootstrap resamples, "
        "and across-seed stability is evaluated by computing sample standard deviation (σ) and variance (Var)."
    )

    # CHAPTER 5: CONTROLLED TTA STUDY
    add_chapter_heading(5, "Controlled Study of Test-Time Adaptation")
    add_sec_heading("5.1 Experimental Protocol")
    doc.add_paragraph(
        "To ensure operational realism, all TTA evaluations adhered to strict controls: (1) PRE_UPDATE prediction timing: the model predicts "
        "each incoming test batch using parameters θ_{t-1} before consuming the batch for self-supervised updates; (2) Deferred Ground-Truth Access: "
        "ground-truth labels were quarantined on disk and loaded into memory only after prediction tensors were persisted and hashed; and "
        "(3) Streaming Sequences: patches were evaluated in spatio-temporal acquisition order."
    )

    add_sec_heading("5.2 Evaluated TTA Algorithms")
    doc.add_paragraph(
        "The evaluation benchmarked five representative TTA strategies against frozen source inference: source_only (frozen feed-forward inference); "
        "bn_no_update / AdaBN (batch normalization running statistic re-estimation); Tent (entropy minimization updating affine scale and shift parameters; "
        "Wang et al., 2021); EATA (entropy-filtered adaptation with Fisher regularization; Niu et al., 2022); SAR (sharpness-aware entropy minimization; "
        "Niu et al., 2023); and CoTTA (continual adaptation with teacher-student EMA weight averaging; Wang et al., 2022)."
    )

    add_sec_heading("5.3 Empirical Findings: Instability and Performance Collapse")
    doc.add_paragraph(
        "In the formal sealed test evaluation across 10 events (15,896 samples), while point estimates for AdaBN (0.7599, +0.0146), Tent (0.7628, +0.0175), "
        "and CoTTA (0.7669, +0.0216) showed minor average shifts over source_only (0.7453), all 95% paired bootstrap confidence intervals crossed zero "
        "([-0.0148, +0.0560], [-0.0119, +0.0586], and [-0.0077, +0.0634], respectively; p > 0.20), showing no statistically significant benefit."
    )
    doc.add_paragraph(
        "More critically, on the lightweight Siamese difference backbone (SiamUnet_diff), TTA induced catastrophic failure: AdaBN degraded macro F1 by "
        "-5.69 pp (0.7105 to 0.6536) and Tent degraded macro F1 by -5.08 pp (0.7105 to 0.6597). On individual events (such as Event 279), Tent erased "
        "flood recall entirely, collapsing F1 from 0.6558 down to 0.2583 (-39.75 pp). The primary failure mechanism is batch normalization context "
        "corruption: extreme flood sparsity (<2% of pixels) corrupts internal normalization statistics when re-estimated on small inference batches (B=8), "
        "shifting effective decision thresholds and triggering massive false alarm surges."
    )

    # CHAPTER 6: MOTIVATION FOR ROBUST SOURCE TRAINING
    add_chapter_heading(6, "Motivation for Robust Source Training")
    add_sec_heading("6.1 Discovery of Severe Geographic Source Imbalance")
    doc.add_paragraph(
        "Auditing the 27 source events in rst1_source_event_inventory_v1.csv revealed a severe structural flaw in standard source training: Event 496 "
        "contains only 21 patches (0.10% of the dataset), whereas Event 1111009 contains 7,532 patches (35.09% of the dataset), representing a "
        "358.7× imbalance ratio. Three large events account for over 63% of all available training patches. Under standard uniform patch sampling (R0), "
        "stochastic gradient descent spends the vast majority of its updates fitting dominant regions, while data-sparse events are starved of updates."
    )
    add_figure_with_caption(doc, "fig6_event_sampling_distribution.png", "Figure 6: Geographic patch count distribution across 27 KuroSiwo source events showing the 358.7× disparity.")

    add_sec_heading("6.2 Class Imbalance and Background Suppression")
    doc.add_paragraph(
        "Compounding event-level imbalance is extreme pixel-level class imbalance: Class 0 (dry ground) occupies 73.4% of valid pixels, Class 1 "
        "(permanent water) occupies 15.6%, and Class 2 (newly inundated flood water) occupies only 11.0%. Standard unweighted cross-entropy loss allows "
        "the majority unflooded class to dominate backpropagation, suppressing flood recall on difficult events."
    )

    add_sec_heading("6.3 Pre-Registered Hypothesis Criteria")
    doc.add_paragraph(
        "To enforce objective scientific evaluation, confirmatory testing on SNUNet-ECAM across Folds A and B over seeds {0, 1, 2} was governed by three "
        "pre-registered hypotheses: Criterion P1 (Average Generalization): Grand event-macro F1 must improve by Δ ≥ +0.010 (+1.0 pp); Criterion P2 "
        "(Tail Reliability): Bottom-three event mean F1 must improve by Δ ≥ +0.020 (+2.0 pp); and Criterion P3 (Seed Stability): Across-seed variance "
        "on both folds must not increase (ΔVar ≤ 0)."
    )

    # CHAPTER 7: EVENT-BALANCED SOURCE TRAINING
    add_chapter_heading(7, "Event-Balanced Robust Source Training")
    add_sec_heading("7.1 Component Formulation (R0, M1, M2, M3)")
    doc.add_paragraph(
        "ChangeGuard investigates modular training modifications: Baseline Recipe R0 uses uniform patch sampling and unweighted cross-entropy plus "
        "Dice loss; Module 1 (M1) introduces a hierarchical two-stage sampler that first selects an event uniformly at random (e ~ Uniform(E_train)), "
        "then draws a patch uniformly within that event (p ~ Uniform(P_e)), equalizing per-event gradient exposure; Module 2 (M2) applies training-only "
        "normalized inverse-square-root class-balanced cross-entropy weights (w_c = 1 / sqrt(f_c + ε)), leaving test-time evaluation completely unbiased; "
        "and Module 3 (M3) applies synchronized geometric flips and paired radiometric gain jitter (g ~ Uniform(0.95, 1.05))."
    )
    add_equation_paragraph(doc, "w_raw,c = 1 / sqrt(max(p_c, 10^-12)),    w_norm,c = w_raw,c / ((1/3) * ∑ w_raw,k)", "7.1")
    add_equation_paragraph(doc, "L_total = L_WCE(w) + L_Dice(unweighted)", "7.2")

    add_sec_heading("7.2 The Discovery Staircase and Parsimonious Promotion of R2")
    doc.add_paragraph(
        "In Work Package RST4 on discovery Fold C (Group C evaluation, 9 events, seed 0), candidate components were evaluated sequentially: Baseline R0 "
        "achieved 0.7631 macro F1 and 0.5630 bottom-3 F1; R1 (R0+M1) achieved 0.7561 macro F1 and 0.5426 bottom-3 F1; R2 (R1+M2) achieved 0.7651 macro F1 "
        "and 0.5869 bottom-3 F1 (+0.0239 tail gain); and R3 (R2+M3) achieved 0.7697 macro F1 and 0.5824 bottom-3 F1. While R3 attained slightly higher "
        "average F1, it caused catastrophic degradation on Event 497 (F1 collapsed by -0.2405 from 0.8321 to 0.5916). Under the pre-registered parsimony "
        "rule (|R2 − R3| ≤ 0.005), Recipe R2 was promoted and frozen for formal confirmation."
    )

    add_table_caption(doc, "Table 1: Component Discovery Staircase Performance on Discovery Fold C (RST4, Seed 0).")
    t1 = doc.add_table(rows=5, cols=6)
    set_table_borders(t1)
    headers = ["Recipe", "Modules Included", "Macro F1", "Bottom-3 F1", "Worst Event F1", "Promotion Verdict"]
    for c_idx, h in enumerate(headers):
        t1.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t1.rows[0])
    rows_data_t1 = [
        ("R0 (Baseline)", "Standard uniform patch sampling + CE", "0.7631", "0.5630", "0.4578", "Baseline Anchor"),
        ("R1", "R0 + M1 (Event-balanced sampling)", "0.7561", "0.5426", "0.3634", "Tail degradation; reject"),
        ("R2 (Frozen)", "R1 + M2 (Class-balanced cross-entropy)", "0.7651", "0.5869", "0.4580", "Promoted under Parsimony"),
        ("R3", "R2 + M3 (Paired radiometric jitter)", "0.7697", "0.5824", "0.4497", "Event 497 collapse (-0.2405)"),
    ]
    for r_idx, rdata in enumerate(rows_data_t1, start=1):
        row = t1.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    add_figure_with_caption(doc, "fig1_component_staircase.png", "Figure 1: Component discovery staircase on Fold C showing macro and tail F1 progression across R0 through R3.")

    # CHAPTER 8: EXPERIMENTAL RESULTS
    add_chapter_heading(8, "Experimental Results")
    add_sec_heading("8.1 Primary Source Confirmation on SNUNet-ECAM (RST7)")
    doc.add_paragraph(
        "Primary confirmatory evaluation was conducted across 12 formal training runs on SNUNet-ECAM across 18 held-out source events in Folds A and B "
        "over seeds 0, 1, and 2. Table 2 summarizes the primary confirmation metrics."
    )

    add_table_caption(doc, "Table 2: Primary Event-Disjoint Confirmation Results on SNUNet-ECAM Across 18 Source Events (RST7, Seeds 0, 1, 2).")
    t2 = doc.add_table(rows=6, cols=7)
    set_table_borders(t2)
    headers_t2 = ["Split / Metric", "Baseline (R0)", "Recipe (R2)", "Delta (R2 - R0)", "95% Bootstrap CI", "Criterion", "Status"]
    for c_idx, h in enumerate(headers_t2):
        t2.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t2.rows[0])
    rows_data_t2 = [
        ("Fold A Macro F1", "0.7291 ± 0.0120", "0.7431 ± 0.0106", "+0.0140", "[-0.0075, +0.0380]", "P3 (Fold A)", "Passed (σ reduced)"),
        ("Fold B Macro F1", "0.7114 ± 0.0035", "0.7246 ± 0.0165", "+0.0132", "[-0.0183, +0.0468]", "P3 (Fold B)", "Failed (σ increased)"),
        ("Grand Macro F1 (18 Events)", "0.7202 ± 0.0044", "0.7338 ± 0.0132", "+0.0136 (+1.36 pp)", "[-0.0045, +0.0351]", "P1 (Δ ≥ +0.010)", "PASSED"),
        ("Bottom-3 Mean F1 (Tail)", "0.3987 ± 0.0298", "0.4498 ± 0.0248", "+0.0511 (+5.11 pp)", "N/A", "P2 (Δ ≥ +0.020)", "PASSED (+12.8%)"),
        ("Worst Single Event F1", "0.2991 ± 0.0315", "0.2929 ± 0.0305", "-0.0062 (-0.62 pp)", "N/A", "Descriptive Anchor", "Outlier stable"),
    ]
    for r_idx, rdata in enumerate(rows_data_t2, start=1):
        row = t2.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
            if c_idx == 6 and "PASSED" in val:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(0x00, 0x80, 0x00)
            elif c_idx == 6 and "Failed" in val:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    doc.add_paragraph(
        "Evaluation of Pre-Registered Criteria: Criterion P1 (Average Improvement ≥ +0.010): PASSED. Grand macro F1 improved by +0.0136 (+1.36 pp), "
        "distributed across Fold A (+0.0140) and Fold B (+0.0132). Criterion P2 (Tail Reliability Gain ≥ +0.020): PASSED. Bottom-3 mean F1 improved "
        "substantially from 0.3987 to 0.4498 (Δ = +0.0511, a +12.82% relative gain). Criterion P3 (Seed Stability): FAILED. On Fold A, seed standard "
        "deviation decreased from 0.0120 to 0.0106; however, on Fold B, across-seed standard deviation increased from 0.0035 to 0.0165 (+368.5% SD), "
        "yielding an aggregate +75.1% variance increase. The paired event-bootstrap 95% confidence interval spans [-0.0045, +0.0351] (p = 0.0779), "
        "crossing zero."
    )

    add_figure_with_caption(doc, "fig2_primary_per_event_paired_deltas.png", "Figure 2: Paired per-event F1 difference (Δ = R2 − R0) across 18 held-out source events averaged over seeds 0, 1, and 2. 12 of 18 events show positive improvement under R2.")

    add_sec_heading("8.2 Per-Event Performance Breakdown")
    doc.add_paragraph(
        "As visualized in Figure 2 and Figure 4: Massive recall-driven gains occurred on challenging data-sparse flood events: Event 118 gained +0.1150 F1 "
        "(0.4308 to 0.5459; recall rose +11.7 pp) and Event 470 gained +0.1140 F1 (0.4661 to 0.5801; recall rose +10.7 pp). Minor precision trade-offs "
        "occurred on high-baseline scenes (Event 554: -0.0322; Event 1111004: -0.0256) where class reweighting slightly increased false alarms. Zero events "
        "experienced dual drops in precision and recall."
    )
    add_figure_with_caption(doc, "fig4_precision_recall_tradeoff.png", "Figure 4: Grand Precision-Recall trade-off analysis across confirmatory folds on SNUNet-ECAM.")

    add_sec_heading("8.3 Cross-Backbone Architecture Transfer Boundary (RST8)")
    doc.add_paragraph(
        "To test whether R2 provides general inductive bias across backbones, Work Package RST8 deployed R2 on SiamUnet_diff (1.35M parameters) at seed 0 "
        "across the identical 18 confirmatory events. Table 3 presents the architecture transfer evaluation."
    )

    add_table_caption(doc, "Table 3: Architecture Transfer Results on SiamUnet_diff vs. SNUNet-ECAM at Seed 0 Across 18 Events (RST8).")
    t3 = doc.add_table(rows=7, cols=5)
    set_table_borders(t3)
    headers_t3 = ["Architecture / Metric", "Baseline (R0)", "Recipe (R2)", "Delta (R2 - R0)", "Outcome / Decision"]
    for c_idx, h in enumerate(headers_t3):
        t3.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t3.rows[0])
    rows_data_t3 = [
        ("SiamUnet_diff: Fold A Macro F1", "0.7275", "0.7201", "-0.0074", "Degradation"),
        ("SiamUnet_diff: Fold B Macro F1", "0.7201", "0.7077", "-0.0124", "Degradation"),
        ("SiamUnet_diff: Grand Macro F1", "0.7238", "0.7139", "-0.0099 (-0.99 pp)", "NEGATIVE TRANSFER"),
        ("SiamUnet_diff: Bottom-3 Mean F1", "0.4350", "0.3902", "-0.0448 (-4.48 pp)", "Severe Tail Collapse"),
        ("SiamUnet_diff: Worst Event F1", "0.3565", "0.2660", "-0.0905 (-9.05 pp)", "Worst-case degradation"),
        ("SNUNet-ECAM (Seed 0 Reference)", "0.7267", "0.7287", "+0.0020 (Tail +0.0536)", "Positive Reference"),
    ]
    for r_idx, rdata in enumerate(rows_data_t3, start=1):
        row = t3.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
            if "NEGATIVE" in val or "Collapse" in val:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    doc.add_paragraph(
        "On SiamUnet_diff, Recipe R2 degraded performance across average, tail, and worst-case metrics. Eleven of 18 events suffered declines. "
        "Under pre-registered protocol rules, negative transfer at Stage A triggered an immediate halt: multi-seed expansion was cancelled, "
        "computational resources were preserved, and the claim of Recipe R2 was explicitly scoped strictly to SNUNet-ECAM."
    )
    add_figure_with_caption(doc, "fig7_architecture_transfer_deltas.png", "Figure 7: Architecture transfer per-event deltas on SiamUnet_diff showing consistent performance degradation.")

    # CHAPTER 9: ABLATION AND RELIABILITY
    add_chapter_heading(9, "Ablation and Reliability Analysis")
    add_sec_heading("9.1 Decomposed Contribution of Components")
    doc.add_paragraph(
        "Comparing across discovery and confirmation demonstrates the complementary synergy of M1 and M2: (1) M1 alone (event balancing) equalizes "
        "geographic exposure, preventing dominant swaths from commandeering optimization, but under-samples large flood swaths and depresses recall "
        "if unweighted; (2) M2 alone (class-balanced loss) heavily penalizes false negatives on newly inundated pixels (w_flood = 0.8417 vs "
        "w_bg = 0.4923), lifting recall; and (3) The synergy of M1 + M2 (Recipe R2) ensures all geographic regimes contribute equally while protecting "
        "sparse flood pixels, yielding the peak tail gain of +0.0511 on Bottom-3 events."
    )

    add_sec_heading("9.2 Tail Robustness vs. Single Worst-Case Outlier Analysis")
    doc.add_paragraph(
        "Figure 3 highlights the distinction between lower-tail improvement and single worst-case protection: while bottom-3 mean F1 improved by +5.11 pp, "
        "the single worst event (Event 427) remained virtually unchanged (0.2991 to 0.2929, Δ = -0.0062). In Event 427, floodwater is dispersed across "
        "dense vegetation with virtually zero radar backscatter contrast. Loss reweighting cannot fabricate missing physical contrast, demonstrating that "
        "certain environmental outliers require multi-sensor optical fusion or auxiliary topography inputs."
    )
    add_figure_with_caption(doc, "fig3_worst_bottom_event_comparison.png", "Figure 3: Worst and bottom event comparison illustrating tail reliability gain vs persistent vegetation outlier (Event 427).")

    add_sec_heading("9.3 Detailed Seed Variance Analysis")
    doc.add_paragraph(
        "Figure 5 displays the three-panel across-seed variance distribution across Fold A, Fold B, and historical target events. In Fold A, across-seed "
        "standard deviation decreased from 0.0120 to 0.0106. In Fold B, across-seed standard deviation increased from 0.0035 to 0.0165 (+368.5%) because "
        "Seed 1 surged to 0.7283 while Seed 2 reached 0.6956. This fold-dependent variance demonstrates that event-balanced source training introduces "
        "sensitivity to initialization on certain event compositions, failing Criterion P3."
    )
    add_figure_with_caption(doc, "fig5_seed_stability.png", "Figure 5: Three-panel seed stability distribution comparing across-seed variance on Fold A, Fold B, and Historical Targets.")

    # CHAPTER 10: COMPARISON WITH TTA
    add_chapter_heading(10, "Comparison with Test-Time Adaptation")
    add_sec_heading("10.1 Contextual Retrospective Comparison (RST9)")
    doc.add_paragraph(
        "In Work Package RST9, three deployable SNUNet-ECAM models were trained on all 27 source events for 16 epochs (exactly 10,736 optimizer steps per "
        "seed) under Recipe R2 and evaluated retrospectively on the 16 historical target events. Table 4 records the cryptographically audited deployable "
        "checkpoint ledger."
    )

    add_table_caption(doc, "Table 4: Cryptographically Verified Deployable Checkpoint Ledger for SNUNet-ECAM (RST9, All 27 Source Events).")
    t4 = doc.add_table(rows=4, cols=6)
    set_table_borders(t4)
    headers_t4 = ["Run ID", "Seed", "Steps", "Runtime", "VRAM", "Checkpoint SHA-256 Digest"]
    for c_idx, h in enumerate(headers_t4):
        t4.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t4.rows[0])
    rows_data_t4 = [
        ("rst9_r2_snunet_all27_s0_v1", "Seed 0", "10,736", "3.23h", "4,913 MB", "9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7"),
        ("rst9_r2_snunet_all27_s1_v1", "Seed 1", "10,736", "3.34h", "4,913 MB", "279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2"),
        ("rst9_r2_snunet_all27_s2_v1", "Seed 2", "10,736", "3.21h", "4,913 MB", "b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893"),
    ]
    for r_idx, rdata in enumerate(rows_data_t4, start=1):
        row = t4.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
            if c_idx == 5:
                p.runs[0].font.size = Pt(7.5)
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    doc.add_paragraph(
        "Across all 16 historical target events, Recipe R2 achieved 0.7380 macro F1 compared to 0.6873 for baseline R0 (Δ = +0.0507, 95% CI [+0.0234, +0.0845]). "
        "On the 10 former test events, R2 achieved 0.8086 macro F1 (Δ = +0.0633, 95% CI [+0.0312, +0.0984]). Table 5 and Table 7 provide the detailed breakdown."
    )

    add_table_caption(doc, "Table 5: Retrospective Generalization on 16 Historical Target Events Across Seeds 0, 1, and 2.")
    t5 = doc.add_table(rows=5, cols=5)
    set_table_borders(t5)
    headers_t5 = ["Seed / Evaluation Set", "Baseline (R0)", "Recipe (R2)", "Delta (R2 - R0)", "Retrospective Qualification"]
    for c_idx, h in enumerate(headers_t5):
        t5.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t5.rows[0])
    rows_data_t5 = [
        ("Seed 0 (All 16 Events)", "0.7009", "0.7355", "+0.0346", "Observed historically in Weeks 3–5"),
        ("Seed 1 (All 16 Events)", "0.6616", "0.7337", "+0.0721", "Observed historically in Weeks 3–5"),
        ("Seed 2 (All 16 Events)", "0.6994", "0.7447", "+0.0453", "Observed historically in Weeks 3–5"),
        ("Mean ± SD (16 Events)", "0.6873 ± 0.0223", "0.7380 ± 0.0059", "+0.0507 (95% CI: [+0.0234, +0.0845])", "Retrospective Benchmark Extension"),
    ]
    for r_idx, rdata in enumerate(rows_data_t5, start=1):
        row = t5.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    add_figure_with_caption(doc, "fig8_historical_target_extension_deltas.png", "Figure 8: Retrospective historical target extension per-event paired F1 deltas across 16 events.")

    add_sec_heading("10.2 Operational Advantage Over Online Adaptation")
    doc.add_paragraph(
        "Table 6 and Table 7 contrast pure source-trained Recipe R2 against historical online TTA algorithms. Beyond accuracy, robust source training delivers "
        "immense operational advantages: zero target-time adaptation overhead (0.0 s latency, 0.0 MB memory), maintaining maximum throughput (64 samples/s) "
        "and baseline VRAM (4.9 GB), whereas CoTTA inflates VRAM to 8.95 GB and slows throughput by 6× (11 samples/s) due to online backpropagation. "
        "Furthermore, pure source models are deterministic and completely immune to streaming-order permutations."
    )

    add_table_caption(doc, "Table 6: Computational Complexity, Training Time, and Test Latency Resource Audit.")
    t6 = doc.add_table(rows=6, cols=6)
    set_table_borders(t6)
    headers_t6 = ["Architecture & Method", "Parameters", "Peak VRAM", "Inference Throughput", "Target-Time Overhead", "Operational Risk"]
    for c_idx, h in enumerate(headers_t6):
        t6.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t6.rows[0])
    rows_data_t6 = [
        ("SNUNet-ECAM: R0 (Baseline)", "12.03M", "4,913 MB", "~64 samples/s", "0.0 s (Zero overhead)", "Standard feed-forward"),
        ("SNUNet-ECAM: R2 (Ours)", "12.03M", "4,913 MB", "~64 samples/s", "0.0 s (Zero overhead)", "Standard feed-forward; robust"),
        ("SiamUnet_diff: R0", "1.35M", "3,124 MB", "~180 samples/s", "0.0 s (Zero overhead)", "Lightweight baseline"),
        ("SNUNet-ECAM: Tent (TTA)", "12.03M", "5,840 MB", "~22 samples/s", "High (Online test backprop)", "Risk of background collapse"),
        ("SNUNet-ECAM: CoTTA (TTA)", "12.03M", "8,950 MB", "~11 samples/s", "Very High (Teacher-student EMA)", "Severe memory & drift risk"),
    ]
    for r_idx, rdata in enumerate(rows_data_t6, start=1):
        row = t6.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    add_table_caption(doc, "Table 7: Contextual Performance Benchmark Against Historical TTA Algorithms on 10 Former Test Events.")
    t7 = doc.add_table(rows=6, cols=6)
    set_table_borders(t7)
    headers_t7 = ["Method / Strategy", "Category", "Test Macro F1", "Delta vs. R0", "95% Bootstrap CI", "Compatibility / Limitations"]
    for c_idx, h in enumerate(headers_t7):
        t7.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t7.rows[0])
    rows_data_t7 = [
        ("R0 Source-Only Baseline", "Source Anchor", "0.7453 ± 0.0345", "Ref (0.000)", "N/A", "Direct benchmark anchor"),
        ("AdaBN (bn_no_update)", "Test-Time Adaptation", "0.7599 ± 0.0158", "+0.0146", "[-0.0148, +0.0560]", "Crosses zero; non-deterministic batch size"),
        ("Tent (Entropy Minimization)", "Test-Time Adaptation", "0.7628 ± 0.0147", "+0.0175", "[-0.0119, +0.0586]", "Crosses zero; stream order dependency"),
        ("CoTTA (Continual TTA)", "Test-Time Adaptation", "0.7669 ± 0.0132", "+0.0216", "[-0.0077, +0.0634]", "Crosses zero; 6× slower throughput"),
        ("Recipe R2 (ChangeGuard)", "Robust Source", "0.8086 ± 0.0072", "+0.0633", "[+0.0312, +0.0984]", "Retrospective extension; zero target compute"),
    ]
    for r_idx, rdata in enumerate(rows_data_t7, start=1):
        row = t7.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0))

    add_figure_with_caption(doc, "fig9_runtime_vram_tradeoff.png", "Figure 9: Test-time operational latency and VRAM footprint trade-off demonstrating pure source efficiency.")

    # CHAPTER 11: LIMITATIONS AND INTEGRITY
    add_chapter_heading(11, "Limitations and Research Integrity")
    add_sec_heading("11.1 Methodological Limitations")
    doc.add_paragraph(
        "1. Architectural Dependence: Recipe R2 failed to transfer to SiamUnet_diff (Macro F1 Δ = -0.0099; Tail F1 Δ = -0.0448). While SNUNet-ECAM's "
        "dense attention modules accommodated class-weighted gradients smoothly, the lightweight differencing backbone suffered severe degradation. "
        "The benefits of R2 cannot be assumed for arbitrary change detection architectures."
    )
    doc.add_paragraph(
        "2. Seed Variability (P3 Failure): R2 exhibited increased across-seed variance in Fold B (standard deviation rose from 0.0035 to 0.0165). "
        "R2 does not provide invariant seed stability across different data splits."
    )
    doc.add_paragraph(
        "3. Single Benchmark Scope: All experiments were conducted on the KuroSiwo Sentinel-1 dataset. While geographically global (43 events across "
        "5 continents), performance on other SAR constellations (e.g., TerraSAR-X, ALOS-2 PALSAR-2) remains unverified."
    )
    doc.add_paragraph(
        "4. Retrospective Nature of Target Events: All 16 target events evaluated in RST9 were observed during earlier project phases. These results "
        "demonstrate strong historical reproducibility, but do not constitute a blind prospective test on newly collected global floods."
    )

    add_sec_heading("11.2 Research Integrity Protocol Adherence")
    doc.add_paragraph(
        "This project strictly adhered to pre-registered research integrity protocols: (1) No Event Dropping: every planned event in Folds A, B, and C was "
        "evaluated and incorporated into macro averages; zero unfavorable events were omitted; (2) Fail-Closed Verification: all 15 evidence inputs were audited "
        "bit-identically against SHA-256 digests; (3) Preserved Attempt Ledger: interrupted and failed attempts were preserved permanently with public "
        "invalidation notes; and (4) Deterministic Replication: the entire analysis pipeline is executable via run_rst10_pipeline.py."
    )

    # CHAPTER 12: CONCLUSION AND FUTURE WORK
    add_chapter_heading(12, "Conclusion and Future Work")
    add_sec_heading("12.1 Conclusion")
    doc.add_paragraph(
        "This Final Year Project investigated the problem of cross-event distribution shifts in satellite SAR flood change detection. Through controlled "
        "empirical evaluations, we demonstrated that generic unsupervised test-time adaptation methods suffer from acute instability, non-significant gains, "
        "and catastrophic recall collapse under operational satellite disaster conditions. In response, we developed and validated ChangeGuard, showing "
        "that a target-free robust source-training strategy combining event-balanced sampling (M1) and training-only class-balanced loss (M2) systematically "
        "improves average generalization (+1.36 pp F1) and lower-tail reliability (+5.11 pp Bottom-3 F1) on SNUNet-ECAM without target-time compute or "
        "adaptation fragility. Finally, by documenting an explicit negative transfer result on SiamUnet_diff, we established the boundaries of our findings, "
        "providing a transparent empirical baseline for operational Earth observation."
    )

    add_sec_heading("12.2 Recommendations for Future Work")
    doc.add_paragraph(
        "1. Prospective Real-Time Evaluation: Future investigations should evaluate frozen Recipe R2 models on newly occurring global disaster events "
        "in real time during emergency satellite tasking to validate operational generalization prospectively."
    )
    doc.add_paragraph(
        "2. Multi-Sensor Optical/SAR Fusion: For severe environmental outliers such as Event 427 (flooded dense vegetation), future architectures should "
        "integrate cloud-free pre-disaster optical imagery (Sentinel-2) with microwave radar to overcome radar backscatter ambiguity."
    )
    doc.add_paragraph(
        "3. Architecture-Adaptive Loss Balancing: To extend robust source training to lightweight backbones like SiamUnet_diff, future research should explore "
        "gradient-clipping or curvature-aware loss scaling tailored to shallow Siamese networks."
    )

    # ==========================================
    # 4. REQUIRED POST-CHAPTER SECTIONS
    # ==========================================
    doc.add_page_break()
    h_ref = doc.add_paragraph()
    h_ref.paragraph_format.space_before = Pt(18)
    h_ref.paragraph_format.space_after = Pt(12)
    h_ref.paragraph_format.keep_with_next = True
    r = h_ref.add_run("Reflection on Learning Outcome Attainment")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p_note = doc.add_paragraph("[Drafted from documented project activities — student verification required]")
    p_note.runs[0].font.italic = True
    p_note.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_paragraph(
        "This project provided an extensive capstone engineering experience across multiple accredited student learning outcomes:\n\n"
        "1. Problem Analysis: I analyzed the fundamental mathematical conflict between extreme temporal class imbalance (<2% flood pixels) "
        "and self-supervised test-time entropy minimization in disaster mapping. Rather than accepting the common literature assumption that TTA "
        "is universally beneficial, I identified the mathematical mechanism of background collapse and diagnosed a 358.7× geographic sample disparity "
        "in benchmark training data.\n\n"
        "2. Investigation: I designed and executed a multi-seed, event-disjoint confirmatory experimental protocol across 18 held-out source events and "
        "16 retrospective target events. I formulated pre-registered quantitative criteria (P1, P2, P3) and applied cluster-bootstrap statistical "
        "inference to prevent premature conclusions.\n\n"
        "3. Modern Tool Usage: I leveraged modern deep learning frameworks (PyTorch, torchvision), high-performance workstation hardware (NVIDIA RTX 3090), "
        "geospatial data pipelines for Sentinel-1 radar processing, Git version control, and automated regression testing suites (14 tests in test_rst10_pipeline.py).\n\n"
        "4. Ethics and Research Integrity: In strict adherence to academic honesty, I reported negative results transparently—including the failure of "
        "Criterion P3, the crossing of zero in bootstrap confidence intervals, and the negative transfer on SiamUnet_diff. I maintained immutable log ledgers, "
        "air-gapped test partitions, and preserved failed attempts.\n\n"
        "5. Project Management: I managed a complex multi-week research workflow across 12 structured work packages (RST0 through RST11), maintaining cryptographic "
        "manifests, detailed technical ledgers, and milestone deliverables on schedule."
    )

    doc.add_page_break()
    h_gai = doc.add_paragraph()
    h_gai.paragraph_format.space_before = Pt(18)
    h_gai.paragraph_format.space_after = Pt(12)
    h_gai.paragraph_format.keep_with_next = True
    r = h_gai.add_run("Acknowledging / Declaring the Use of Generative AI")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    doc.add_paragraph(
        "In accordance with Nanyang Technological University academic guidelines and research integrity policies, I declare the use of Generative "
        "Artificial Intelligence (GAI) and agentic coding tools during this Final Year Project:\n\n"
        "1. Tools Utilized: Google Antigravity (Advanced Agentic AI Assistant) and OpenAI Codex CLI assistants.\n\n"
        "2. Specific Areas of AI Assistance: (a) Code development and debugging: drafting repetitive data loader functions, regression test scripts, "
        "and figure plotting routines; (b) Verification script authoring: generating automated validation harnesses (e.g., validate_rst11_report.py) to "
        "audit numerical consistency; (c) Document formatting and LaTeX/OpenXML syntax generation: assisting with table formatting, stylesheet alignment, "
        "and BibTeX verification; and (d) Drafting assistance: refining technical prose and ensuring structural adherence to NTU report guidelines.\n\n"
        "3. Student Verification and Responsibility: All experimental designs, mathematical formulations, code logic, empirical data, and interpretations "
        "were reviewed and directed by the student researcher. Where independent human verification remains pending, statements are explicitly marked "
        "for final supervisory review. The student author retains full responsibility for the scientific integrity, accuracy, and originality of this report."
    )

    # References
    doc.add_page_break()
    h_bib = doc.add_paragraph()
    h_bib.paragraph_format.space_before = Pt(18)
    h_bib.paragraph_format.space_after = Pt(12)
    h_bib.paragraph_format.keep_with_next = True
    r = h_bib.add_run("References")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    references_list = [
        ("[1]", "N. I. Bountos, C. Angelokostas, D. Michail, and I. Papoutsis, \"KuroSiwo: A Dataset and Benchmark for Multisensor Flood Inundation Mapping,\" IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024. doi: 10.1109/LGRS.2024.3361208."),
        ("[2]", "S. Fang, K. Li, J. Shao, and Z. Li, \"SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,\" IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2021. doi: 10.1109/LGRS.2021.3049928."),
        ("[3]", "R. C. Daudt, B. Le Saux, and A. Boulch, \"Fully Convolutional Siamese Networks for Change Detection,\" in 2018 25th IEEE International Conference on Image Processing (ICIP), 2018, pp. 4063–4067. doi: 10.1109/ICIP.2018.8451652."),
        ("[4]", "D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, \"Tent: Fully Test-Time Adaptation by Entropy Minimization,\" in International Conference on Learning Representations (ICLR), 2021."),
        ("[5]", "Q. Wang, O. Fink, L. Van Gool, and D. Dai, \"Continual Test-Time Domain Adaptation,\" in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 7201–7211. doi: 10.1109/CVPR52688.2022.00706."),
        ("[6]", "S. Schneider, E. Rusak, L. Eck, P. Gehler, W. Brendel, and M. Bethge, \"Improving robustness against common corruptions by covariate shift adaptation,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 11539–11551, 2020."),
        ("[7]", "Y. Li, N. Wang, J. Shi, J. Liu, and X. Hou, \"Revisiting Batch Normalization For Practical Domain Adaptation,\" Pattern Recognition, vol. 80, pp. 109–117, 2018. doi: 10.1016/j.patcog.2018.03.005."),
        ("[8]", "S. Niu, J. Wu, Y. Zhang, Z. Chen, P. Zhao, and M. Tan, \"Efficient Test-Time Model Adaptation without Forgetting,\" in International Conference on Machine Learning (ICML), 2022, pp. 16888–16905."),
        ("[9]", "S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, \"Towards Stable Test-Time Adaptation in Dynamic Wild World,\" in International Conference on Learning Representations (ICLR), 2023."),
        ("[10]", "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, \"Class-Balanced Loss Based on Effective Number of Samples,\" in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268–9277. doi: 10.1109/CVPR.2019.00949."),
        ("[11]", "M. Buda, A. Maki, and M. A. Mazurowski, \"A systematic study of the class imbalance problem in convolutional neural networks,\" Neural Networks, vol. 106, pp. 249–259, 2018. doi: 10.1016/j.neunet.2018.07.011."),
        ("[12]", "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal Loss for Dense Object Detection,\" in IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988. doi: 10.1109/ICCV.2017.324."),
        ("[13]", "I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization,\" in International Conference on Learning Representations (ICLR), 2019."),
    ]

    for num, citation in references_list:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        r_num = p.add_run(num + " ")
        r_num.font.bold = True
        p.add_run(citation)

    # Appendices
    doc.add_page_break()
    h_app = doc.add_paragraph()
    h_app.paragraph_format.space_before = Pt(18)
    h_app.paragraph_format.space_after = Pt(12)
    h_app.paragraph_format.keep_with_next = True
    r = h_app.add_run("Appendix A: Reproducibility Package and Checkpoint Ledger")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    doc.add_paragraph(
        "Hardware Specifications: Workstation equipped with 1× NVIDIA GeForce RTX 3090 GPU (24,576 MB VRAM, Driver 535.183.01, CUDA 12.1), "
        "Intel Xeon Silver 4210R CPU @ 2.40GHz (20 logical threads), and 128 GB system RAM running Ubuntu Linux 22.04 LTS."
    )
    doc.add_paragraph(
        "Software Environment: Python 3.8.10, PyTorch 2.1.2+cu121, Torchvision 0.16.2+cu121, NumPy 1.24.4, SciPy 1.10.1, Matplotlib 3.7.5. "
        "Conda environment path: /usr1/home/s126mdg22_06/miniconda3/envs/sar_cd."
    )
    doc.add_paragraph(
        "Deterministic Execution Instructions:\n"
        "To reproduce the entire RST10 canonical synthesis, run the audited regression pipeline from the repository root:\n"
        "python tools/robust_source_training/run_rst10_pipeline.py\n"
        "To validate numerical consistency and text boundaries across reports:\n"
        "python tools/validate_rst11_report.py"
    )

    doc.add_page_break()
    h_appb = doc.add_paragraph()
    h_appb.paragraph_format.space_before = Pt(18)
    h_appb.paragraph_format.space_after = Pt(12)
    h_appb.paragraph_format.keep_with_next = True
    r = h_appb.add_run("Appendix B: Full Per-Event Confirmatory and Retrospective Data")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    doc.add_paragraph(
        "Complete event-level performance records across all 18 confirmatory source events (Folds A and B) and 16 historical target events "
        "are preserved in outputs/robust_source_training/rst10/tables/all_canonical_tables_v1.md. All primary evaluation metrics reflect Class 2 "
        "(newly inundated flood water) segmentation performance."
    )

    out_path = EXPORTS_DIR / "ChangeGuard_FYP_Report_Review_v1.docx"
    doc.save(str(out_path))
    print(f"Successfully saved {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def build_igarss_paper_docx() -> Path:
    print("Building ChangeGuard_IGARSS2027_Review_v1.docx...")
    doc = docx.Document()

    # Section 1: Title and Authors (US Letter, 1 column)
    s1 = doc.sections[0]
    s1.page_width = Inches(8.5)
    s1.page_height = Inches(11.0)
    s1.top_margin = Inches(0.75)
    s1.bottom_margin = Inches(1.0)
    s1.left_margin = Inches(0.625)
    s1.right_margin = Inches(0.625)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    normal_style.paragraph_format.line_spacing = 1.08
    normal_style.paragraph_format.space_after = Pt(4)

    # Notice Header
    p_notice = doc.add_paragraph()
    p_notice.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_notice.paragraph_format.space_before = Pt(0)
    p_notice.paragraph_format.space_after = Pt(6)
    r_not = p_notice.add_run("Review draft using the IEEE conference template; IGARSS 2027 Paper Kit verification pending.")
    r_not.font.size = Pt(8.5)
    r_not.font.italic = True
    r_not.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("Event-Balanced Source Training for Robust Cross-Event SAR Flood Change Detection")
    r_title.font.size = Pt(20)
    r_title.font.bold = True

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(12)
    r_a = p_auth.add_run("Anonymous Authors\n")
    r_a.font.size = Pt(11)
    r_a.font.bold = True
    r_aff = p_auth.add_run("IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2027) Submission Draft")
    r_aff.font.size = Pt(9.5)
    r_aff.font.italic = True

    # Section 2: 2 Columns for Abstract, Keywords, Body, Tables, Figures, References
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    s2.top_margin = Inches(0.75)
    s2.bottom_margin = Inches(1.0)
    s2.left_margin = Inches(0.625)
    s2.right_margin = Inches(0.625)

    sectPr2 = s2._sectPr
    cols = sectPr2.xpath('./w:cols')
    if not cols:
        cols = OxmlElement('w:cols')
        sectPr2.append(cols)
    else:
        cols = cols[0]
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '288')  # 0.2 inch = 288 twips

    # Abstract & Keywords in 2 columns
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_after = Pt(4)
    r_lbl = p_abs.add_run("Abstract—")
    r_lbl.font.bold = True
    r_lbl.font.size = Pt(9)
    r_txt = p_abs.add_run(
        "Synthetic Aperture Radar (SAR) imagery enables all-weather flood mapping, yet deep learning change detection "
        "models suffer substantial performance drops when deployed across geographically diverse flood events. Prior investigations "
        "indicated that online test-time adaptation (TTA) strategies—such as entropy minimization and test-time normalization "
        "updates—exhibit severe instabilities under extreme class imbalance and heterogeneous backscatter distributions in paired "
        "SAR change detection. In response, this paper investigates a target-free robust source-training strategy designed to improve "
        "cross-event generalization without requiring online test-time parameter modifications. The strategy combines event-balanced "
        "sampling across disparate flood acquisitions with training-only inverse-square-root class-balanced cross-entropy supervision. "
        "Across an 18-event confirmatory cross-validation protocol on the KuroSiwo benchmark using the SNUNet-ECAM architecture over three "
        "random seeds, the proposed frozen recipe (R2) improved grand event-macro F1 by +0.0136 (from 0.7202 to 0.7338) and bottom-three "
        "lower-tail event F1 by +0.0511 (from 0.3987 to 0.4498), satisfying pre-registered confirmatory criteria P1 and P2. However, "
        "seed-to-seed variance increased on Fold B (failing pre-registered criterion P3), and the paired event-bootstrap 95% confidence "
        "interval crossed zero ([-0.0045, +0.0351]). Furthermore, when tested on SiamUnet_diff, the recipe exhibited negative transfer "
        "(grand macro F1 -0.0099; bottom-three F1 -0.0448), establishing an architectural limitation. Retrospective evaluation on 16 historical "
        "target events confirmed consistent performance gains (+0.0507 macro F1) with zero test-time computational overhead."
    )
    r_txt.font.italic = True
    r_txt.font.size = Pt(9)

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_after = Pt(8)
    r_kwlbl = p_kw.add_run("Index Terms—")
    r_kwlbl.font.bold = True
    r_kwlbl.font.size = Pt(9)
    r_kwtxt = p_kw.add_run("Synthetic Aperture Radar (SAR), Flood Mapping, Change Detection, Domain Generalization, Class Imbalance, Robust Training.")
    r_kwtxt.font.italic = True
    r_kwtxt.font.size = Pt(9)

    def add_sec(title: str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(8.5)
        r.font.bold = True

    def add_subsec(title: str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(9.5)
        r.font.italic = True

    # I. INTRODUCTION
    add_sec("I. INTRODUCTION")
    doc.add_paragraph(
        "Satellite Synthetic Aperture Radar (SAR) offers reliable surface observations during severe flood emergencies due to its "
        "cloud-penetrating and day-and-night imaging capabilities [1]. Paired SAR flood change detection compares co-registered pre-event "
        "and post-event backscatter images to identify newly inundated flood pixels. Deep Siamese neural networks, such as SNUNet-ECAM [2] "
        "and SiamUnet [3], achieve high accuracy on identically distributed benchmarks. However, when deployed across distinct flood events "
        "differing in sensor incidence angles, soil moisture, surface roughness, and terrain morphology, these architectures experience acute "
        "performance degradation, particularly in lower-tail events [4]."
    )
    doc.add_paragraph(
        "To address cross-event domain shift, recent literature has explored test-time adaptation (TTA) algorithms, including entropy "
        "minimization [4], continual adaptation [5], and test-time normalization statistics adaptation [6], [7]. In prior controlled "
        "experiments within this project, generic TTA methods demonstrated pronounced vulnerability in paired SAR flood detection: extreme "
        "class imbalance (flood pixels frequently occupying under 2% of a tile) and non-stationary SAR speckle frequently led online unsupervised "
        "adaptation to collapse into degenerate majority-class predictions or produce high variance across batches."
    )
    doc.add_paragraph(
        "These findings motivated a fundamental shift from test-time parameter adaptation to target-free robust source training. Rather than "
        "relying on fragile test-time optimization under low-resource emergency constraints, we examine whether systematic modifications to the "
        "source training distribution—specifically event-balanced sampling and training-only class-balanced supervision—can strengthen cross-event robustness."
    )
    doc.add_paragraph(
        "This study makes three empirical contributions under a pre-registered confirmatory protocol:\n"
        "1) We demonstrate that the frozen robust source-training recipe (R2) improves cross-event macro F1 (+0.0136) and tail-event F1 (+0.0511) "
        "on SNUNet-ECAM across 18 source events and 3 seeds, satisfying primary criteria P1 and P2.\n"
        "2) We document transparently that the paired event-bootstrap 95% confidence interval crosses zero and seed stability (criterion P3) was not "
        "established, showing fold-dependent variance.\n"
        "3) We show that R2 fails to transfer to SiamUnet_diff (grand macro F1 Δ = -0.0099, tail F1 Δ = -0.0448), establishing an explicit architectural "
        "boundary and demonstrating that the strategy's benefits do not transfer across all backbones."
    )

    # II. METHOD
    add_sec("II. ROBUST SOURCE-TRAINING FORMULATION")
    doc.add_paragraph(
        "Standard source training (denoted R0) optimizes cross-entropy loss over uniformly sampled image patches across the training corpus:"
    )
    add_equation_paragraph(doc, "L_CE(y, p_hat) = - ∑_{c ∈ C} y_c log p_hat_c", "1")
    doc.add_paragraph(
        "where C = {0, 1, 2} represents unflooded land (0), permanent water (1), and newly inundated flood water (2). Under R0, larger flood events "
        "contribute disproportionately more patches than smaller localized floods, biasing feature representations toward dominant events [10], [11]."
    )
    doc.add_paragraph(
        "To address these vulnerabilities, we evaluate two modular modifications:\n"
        "Module 1 (M1) — Event-Balanced Sampling: M1 replaces uniform patch sampling with a hierarchical two-stage sampler. For each training batch, "
        "candidate events are sampled uniformly at random from the active training event set, and patches are drawn uniformly within the selected events. "
        "This ensures that every source flood event exerts equal gradient influence per epoch.\n"
        "Module 2 (M2) — Training-Only Class-Balanced Cross-Entropy: M2 applies inverse-square-root effective frequency weighting [10], [12] to the training loss:"
    )
    add_equation_paragraph(doc, "w_c = 1 / sqrt(f_c + ε),    w_bar_c = (w_c / ∑ w_k) * |C|", "2")
    doc.add_paragraph(
        "where f_c is the global pixel frequency of class c measured strictly on the source training split. Crucially, M2 is applied only during source training; "
        "inference thresholding and evaluation metrics remain completely unadjusted to avoid test-time bias."
    )
    doc.add_paragraph(
        "In earlier screening on an exploratory group of 9 events (RST4), an additional hard-patch mining module (M3) was evaluated alongside M1 and M2. "
        "Under pre-registered parsimony rules, M3 provided no meaningful gain over M2 (|F1_R3 − F1_R2| ≤ 0.005). Consequently, recipe R2 (combining M1 and M2 "
        "with M3 disabled) was frozen as the sole candidate for confirmation."
    )

    # III. PROTOCOL
    add_sec("III. CONFIRMATORY EXPERIMENTAL PROTOCOL")
    add_subsec("A. Dataset and Preprocessing")
    doc.add_paragraph(
        "Experiments use the KuroSiwo benchmark [1], comprising multi-temporal Sentinel-1 dual-polarization (VV/VH) SAR image pairs and high-resolution "
        "flood extent annotations across diverse global flood disasters. The source pool contains 18 events, structured into two balanced 9-event folds "
        "(Fold A and Fold B) for cross-validation. A separate set of 16 historical target events is designated strictly for retrospective evaluation. "
        "All images are preprocessed into 256×256 patches. The authoritative evaluation target is Class 2 (newly inundated flood water)."
    )

    add_subsec("B. Architectures and Training Pipeline")
    doc.add_paragraph(
        "The primary architecture is SNUNet-ECAM [2] (12.03M parameters). For architectural transfer testing, we employ SiamUnet_diff [3] (1.35M parameters). "
        "All models are trained for 16 epochs using AdamW [13] with a cosine annealing schedule (lr_max = 10^-4), batch size 16, weight decay 10^-4, and "
        "deterministic orthogonal augmentations."
    )

    add_subsec("C. Pre-Registered Evaluation Criteria")
    doc.add_paragraph(
        "Confirmatory testing on SNUNet-ECAM across Folds A and B over seeds {0, 1, 2} was governed by three pre-registered hypotheses:\n"
        "• Criterion P1 (Average Generalization): Grand event-macro F1 must improve by Δ ≥ +0.010.\n"
        "• Criterion P2 (Tail Robustness): Bottom-three event mean F1 must improve by Δ ≥ +0.020.\n"
        "• Criterion P3 (Seed Stability): Across-seed variance on both folds must not increase (ΔVar ≤ 0)."
    )

    # TABLE I (Full 10 rows matching LaTeX Table 1)
    add_table_caption(doc, "TABLE I: Primary Confirmatory Results on SNUNet-ECAM Across 18 Source Events (Seeds 0, 1, 2).")
    t_ig1 = doc.add_table(rows=10, cols=7)
    set_table_borders(t_ig1, sz="2")
    headers_ig1 = ["Split / Metric", "R0 (Base)", "R2 (Ours)", "Delta", "95% CI", "Criterion", "Status"]
    for c_idx, h in enumerate(headers_ig1):
        t_ig1.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t_ig1.rows[0], font_size=Pt(8.0))
    data_ig1 = [
        ("Fold A: Macro F1", "0.7291±0.012", "0.7431±0.011", "+0.0140", "[-0.0075, +0.0380]", "--", "--"),
        ("Fold A: Bottom-3 F1", "0.4632±0.046", "0.5134±0.021", "+0.0502", "--", "--", "--"),
        ("Fold A: Seed SD", "0.0120", "0.0106", "-0.0014", "--", "P3 (Fold A)", "Passed"),
        ("Fold B: Macro F1", "0.7114±0.004", "0.7246±0.017", "+0.0132", "[-0.0183, +0.0468]", "--", "--"),
        ("Fold B: Bottom-3 F1", "0.3343±0.027", "0.3862±0.035", "+0.0520", "--", "--", "--"),
        ("Fold B: Seed SD", "0.0035", "0.0165", "+0.0130", "--", "P3 (Fold B)", "Failed"),
        ("Grand Macro F1 (18 Ev)", "0.7202±0.004", "0.7338±0.013", "+0.0136", "[-0.0045, +0.0351]", "P1 (Δ≥+0.010)", "PASSED"),
        ("Bottom-3 Mean F1 (Tail)", "0.3987±0.030", "0.4498±0.025", "+0.0511", "--", "P2 (Δ≥+0.020)", "PASSED"),
        ("Seed Variance Ratio", "--", "--", "+75.1%", "--", "P3 (ΔVar≤0)", "FAILED"),
    ]
    for r_idx, rdata in enumerate(data_ig1, start=1):
        row = t_ig1.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
            if "PASSED" in val or "Passed" in val:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0), font_size=Pt(7.5))

    # FIGURE 1
    add_figure_with_caption(doc, "fig2_primary_per_event_paired_deltas.png", "Fig. 1: Paired per-event F1 difference (Δ = R2 − R0) on SNUNet-ECAM across 18 source events.", width_in=3.3)

    # IV. CONFIRMATORY RESULTS
    add_sec("IV. CONFIRMATORY EVALUATION ON SNUNET-ECAM")
    add_subsec("A. Average and Lower-Tail Gains (P1 and P2)")
    doc.add_paragraph(
        "Under R2, grand event-macro F1 across all 18 source events reached 0.7338, an improvement of +0.0136 over the R0 baseline (0.7202), "
        "successfully satisfying Criterion P1. Furthermore, R2 substantially improved performance on challenging lower-tail flood events: "
        "the mean F1 of the bottom three events increased from 0.3987 to 0.4498 (Δ = +0.0511), surpassing the Criterion P2 threshold of +0.020 "
        "by more than a factor of two. On Fold A, bottom-three F1 improved by +0.0502; on Fold B, it improved by +0.0520. As shown in Fig. 1, "
        "12 out of 18 events exhibited positive F1 gains."
    )

    add_subsec("B. Statistical Significance and Seed Variance (P3 Failure)")
    doc.add_paragraph(
        "Despite consistent point-estimate gains, rigorous statistical testing revealed important boundaries:\n"
        "1) Paired Event-Bootstrap CI: The 95% confidence interval for grand macro F1 difference spanned [-0.0045, +0.0351] (p = 0.0779). Because the "
        "interval crosses zero, the average gain cannot be claimed as statistically significant at the 95% confidence level.\n"
        "2) Seed Stability (P3): On Fold A, across-seed standard deviation decreased from 0.0120 to 0.0106 (meeting P3). However, on Fold B, across-seed "
        "standard deviation increased from 0.0035 to 0.0165 (+368.5% SD), yielding an aggregate variance increase of +75.1%. Consequently, Criterion P3 "
        "was formally failed. Event-balanced source training introduces sensitivity to initialization seeds on certain event distributions."
    )

    # TABLE II (Full 7 rows matching LaTeX Table 2)
    add_table_caption(doc, "TABLE II: Architecture Transfer Results on SiamUnet_diff vs. SNUNet-ECAM at Seed 0 Across 18 Events.")
    t_ig2 = doc.add_table(rows=7, cols=5)
    set_table_borders(t_ig2, sz="2")
    headers_ig2 = ["Architecture & Metric", "R0 Base", "R2 Recipe", "Delta", "Stage A Outcome"]
    for c_idx, h in enumerate(headers_ig2):
        t_ig2.rows[0].cells[c_idx].paragraphs[0].add_run(h)
    format_table_header(t_ig2.rows[0], font_size=Pt(8.0))
    data_ig2 = [
        ("SiamUnet: Fold A Macro F1", "0.7275", "0.7201", "-0.0074", "Degradation"),
        ("SiamUnet: Fold B Macro F1", "0.7201", "0.7077", "-0.0124", "Degradation"),
        ("SiamUnet: Grand Macro F1", "0.7238", "0.7139", "-0.0099", "NEGATIVE TRANSFER"),
        ("SiamUnet: Bottom-3 Mean F1", "0.4350", "0.3902", "-0.0448", "Severe Tail Drop"),
        ("SiamUnet: Worst Single Event", "0.3565", "0.2660", "-0.0905", "Worst-case Collapse"),
        ("SNUNet-ECAM (Reference)", "0.7267", "0.7287", "+0.0020", "Positive Reference"),
    ]
    for r_idx, rdata in enumerate(data_ig2, start=1):
        row = t_ig2.rows[r_idx]
        for c_idx, val in enumerate(rdata):
            p = row.cells[c_idx].paragraphs[0]
            p.add_run(val)
            if c_idx == 0:
                p.runs[0].font.bold = True
        format_table_data_row(row, is_even=(r_idx % 2 == 0), font_size=Pt(7.5))

    # FIGURE 2 (Architecture Transfer Deltas)
    add_figure_with_caption(doc, "fig7_architecture_transfer_deltas.png", "Fig. 2: Per-event deltas on SiamUnet_diff illustrating negative transfer across 11 of 18 events.", width_in=3.3)

    # V. ARCHITECTURE TRANSFER & TARGET SUPPORT
    add_sec("V. ARCHITECTURE TRANSFER AND TARGET SUPPORT")
    add_subsec("A. Negative Transfer to SiamUnet_diff")
    doc.add_paragraph(
        "To evaluate whether R2 provides general inductive bias across backbones, Work Package RST8 deployed R2 on SiamUnet_diff at seed 0. As presented in "
        "Table II and Fig. 2, R2 caused consistent performance degradation across both folds: grand macro F1 declined by -0.0099 (0.7238 to 0.7139); "
        "bottom-three F1 dropped by -0.0448 (0.4350 to 0.3902); and worst single event collapsed by -0.0905 (0.3565 to 0.2660). Under pre-registered "
        "protocol rules, negative transfer triggered an immediate halt at Stage A. This confirms that R2's benefits are architecture-specific to SNUNet-ECAM."
    )

    add_subsec("B. Retrospective Historical Target Support")
    doc.add_paragraph(
        "In Work Package RST9, three deployable SNUNet-ECAM models were trained on all 27 source events across seeds {0, 1, 2} (10,736 steps per seed). "
        "Across 16 historical target events, R2 achieved 0.7380 macro F1 vs. 0.6873 for baseline R0 (Δ = +0.0507, 95% CI [+0.0234, +0.0845]). On the 10 former "
        "test events, macro F1 improved by +0.0633 (0.7453 to 0.8086)."
    )

    add_subsec("C. Operational Efficiency vs. Test-Time Adaptation")
    doc.add_paragraph(
        "A decisive operational advantage of robust source training is the total absence of test-time computational overhead: test-time adaptation latency "
        "overhead: 0.0 s; additional VRAM overhead: 0.0 MB; online parameter updates: 0. Unlike online TTA (which multiplies inference latency by 3× to 6× and "
        "risks test-time collapse), R2 operates as a standard frozen network with deterministic inference."
    )

    # VI. CONCLUSION
    add_sec("VI. CONCLUSION AND LIMITATIONS")
    doc.add_paragraph(
        "This paper presented an empirical evaluation of target-free robust source training for cross-event SAR flood change detection. On SNUNet-ECAM, "
        "combining event-balanced sampling with training-only class-balanced supervision (R2) produced substantial improvements in grand macro F1 (+0.0136) "
        "and lower-tail event F1 (+0.0511), confirming pre-registered criteria P1 and P2."
    )
    doc.add_paragraph(
        "Equally important are the documented limitations: (1) R2 failed stability criterion P3 due to elevated seed variance on Fold B, and its confidence "
        "interval crossed zero; (2) R2 demonstrated severe negative transfer on SiamUnet_diff (Δ = -0.0099 macro F1, Δ = -0.0448 tail F1); and (3) Target events "
        "represent retrospective historical benchmarks. All code, checkpoints, and evaluation tables are made publicly available."
    )

    # ACKNOWLEDGMENT
    add_sec("ACKNOWLEDGMENT")
    doc.add_paragraph(
        "The authors acknowledge the European Space Agency and Copernicus program for Sentinel-1 SAR imagery and the KuroSiwo benchmark authors for dataset curation."
    )

    # REFERENCES
    add_sec("REFERENCES")
    refs_ig = [
        "[1] N. I. Bountos et al., \"KuroSiwo: A Dataset and Benchmark for Multisensor Flood Inundation Mapping,\" IEEE LGRS, vol. 21, pp. 1–5, 2024.",
        "[2] S. Fang et al., \"SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,\" IEEE LGRS, vol. 19, pp. 1–5, 2021.",
        "[3] R. C. Daudt et al., \"Fully Convolutional Siamese Networks for Change Detection,\" in IEEE ICIP, 2018, pp. 4063–4067.",
        "[4] D. Wang et al., \"Tent: Fully Test-Time Adaptation by Entropy Minimization,\" in ICLR, 2021.",
        "[5] Q. Wang et al., \"Continual Test-Time Domain Adaptation,\" in IEEE/CVF CVPR, 2022, pp. 7201–7211.",
        "[6] S. Schneider et al., \"Improving robustness against common corruptions by covariate shift adaptation,\" in NeurIPS, vol. 33, pp. 11539–11551, 2020.",
        "[7] Y. Li et al., \"Revisiting Batch Normalization For Practical Domain Adaptation,\" Pattern Recognition, vol. 80, pp. 109–117, 2018.",
        "[8] S. Niu et al., \"Efficient Test-Time Model Adaptation without Forgetting,\" in ICML, 2022, pp. 16888–16905.",
        "[9] S. Niu et al., \"Towards Stable Test-Time Adaptation in Dynamic Wild World,\" in ICLR, 2023.",
        "[10] Y. Cui et al., \"Class-Balanced Loss Based on Effective Number of Samples,\" in IEEE/CVF CVPR, 2019, pp. 9268–9277.",
        "[11] M. Buda et al., \"A systematic study of the class imbalance problem in convolutional neural networks,\" Neural Networks, vol. 106, pp. 249–259, 2018.",
        "[12] T.-Y. Lin et al., \"Focal Loss for Dense Object Detection,\" in IEEE ICCV, 2017, pp. 2980–2988.",
        "[13] I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization,\" in ICLR, 2019.",
    ]
    for ref_str in refs_ig:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(ref_str)
        r.font.size = Pt(7.5)

    out_path = EXPORTS_DIR / "ChangeGuard_IGARSS2027_Review_v1.docx"
    doc.save(str(out_path))
    print(f"Successfully saved {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def main() -> int:
    fyp_docx = build_fyp_report_docx()
    igarss_docx = build_igarss_paper_docx()

    print("\nConverting DOCX documents to PDF using LibreOffice...")
    fyp_pdf = convert_to_pdf(fyp_docx, EXPORTS_DIR)
    igarss_pdf = convert_to_pdf(igarss_docx, EXPORTS_DIR)

    print("\nVerifying PDF page counts:")
    if fyp_pdf.is_file():
        r_fyp = pypdf.PdfReader(str(fyp_pdf))
        print(f"  - FYP Report PDF: {len(r_fyp.pages)} pages ({fyp_pdf})")
    else:
        print(f"  - FYP Report PDF FAILED to generate!")

    if igarss_pdf.is_file():
        r_ig = pypdf.PdfReader(str(igarss_pdf))
        print(f"  - IGARSS 2027 PDF: {len(r_ig.pages)} pages ({igarss_pdf})")
    else:
        print(f"  - IGARSS 2027 PDF FAILED to generate!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
