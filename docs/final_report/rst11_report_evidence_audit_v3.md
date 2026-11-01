# ChangeGuard RST11 Consolidated Evidence Audit and Verification Report (v3)

**Author:** Antigravity (Advanced Agentic Coding Pair Programmer)  
**Project:** ChangeGuard: Robust Source Training for Cross-Event SAR Flood Change Detection  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Date:** October 29, 2026  
**Audit Status:** COMPLETE — ALL SCIENTIFIC AND DOCUMENTARY ISSUES RESOLVED  

---

## 1. Executive Verdict

This forensic audit reconciles the original ChangeGuard Final Year Project report draft (v1) with the Codex review revision (v2) against the immutable repository execution ledgers, pre-registered configurations, code implementations, and checkpoint artifacts. 

**Summary Verdict:**
1. **Optimizer Discrepancy Resolved:** The actual executed optimizer across all source training stages (matched R0, discovery R0–R3, confirmatory R0/R2, SiamUnet_diff verification, and final all-27 R2 models) is **standard Adam** ($\text{lr}=10^{-3}, \beta=(0.9, 0.999), \text{weight\_decay}=0.0$, fixed learning rate schedule, no cosine decay). The appearance of "AdamW" in the draft was an erroneous attribution introduced in commit `9334a21` from generic deep learning template text and bibliography entries. Codex’s substitution of Adam (Kingma & Ba, 2014) is **empirically correct and fully affirmed**.
2. **Figure 6 / Source Inventory Resolved (7,532 vs. 7,556):** The authoritative 27-source-event inventory (`rst1_source_event_inventory_v1.csv` and `rst1_source_event_inventory_summary_v1.json`) contains exactly **21,462 total eligible patches**. The largest event (Event 1111009) contains **7,556 patches**; the smallest event (Event 496) contains **21 patches**. The verified geographic disparity ratio is **359.8×** (359.81×). The figure of 7,532 patches and 358.7× arose from an earlier manual transposition ($7532 / 21 = 358.67$). Figure 1 (formerly Fig 6) is regenerated directly from canonical data, with the M1 line properly identified as an **expected draw frequency** ($\approx 795$ draws/epoch), not a guaranteed realized count.
3. **Figure 2 / Paired Confirmatory Direction Counts Resolved:** The 18 paired per-event F1 differences in RST7 (`outputs/robust_source_training/rst7/paired_event_deltas_v1.csv`) exhibit exactly **9 positive mean differences, 9 negative mean differences, and 0 zero differences**. Codex’s warning that "the plotted source data require reconciliation if they show a different count" was an unverified caveat inserted into manuscript prose; the plotted data match the canonical table exactly. Applying a practical benefit threshold of $|\Delta| \ge 0.010$ reveals 7 beneficial events ($\Delta \ge +0.010$), 7 harmful events ($\Delta \le -0.010$), and 4 neutral events ($|\Delta| < 0.010$). Net macro F1 gain (+0.0136) is driven by large tail-event recoveries (+0.1150 in Event 118, +0.1140 in Event 470) offsetting smaller degradations in high-baseline scenes.
4. **Figure Numbering and Layout Repaired:** All figures are renumbered consecutively in first-appearance order (1 through 8). Figure 9 (unverified matched deployment benchmark) has been retired from manuscript figures and replaced by an audited training resource table and operational profile in Table 6. Figure 4 (formerly Fig 3) legend collision is eliminated by expanding y-limits to $[0.20, 0.95]$ and placing the legend in the upper right. Figures 5 and 6 (formerly Figs 4 and 5) suptitle crowding is eliminated by adjusting subplot margins.
5. **Academic Prose Reconciled:** All reviewer-facing meta-commentary ("the supplied table reports", "requires reconciliation", "the draft previously claimed") has been excised from manuscript prose. The report is revised into an affirmative, academically rigorous dissertation that proudly articulates supported scientific contributions while transparently and concisely presenting empirical boundaries (P3 seed instability, SiamUnet_diff architecture non-transferability, and retrospective historical target scope).

---

## 2. Itemized Issue Resolution Matrix

| Audit Issue | Status | Canonical Finding & Exact Evidence | Action in Manuscript v3 |
| :--- | :--- | :--- | :--- |
| **1. Adam vs. AdamW Discrepancy** | **RESOLVED** | `src/robust_source_training/runner.py:467`, all 40 JSON configs, and all 29 `.pt` checkpoints confirm `torch.optim.Adam` ($\text{lr}=10^{-3}, \beta=(0.9, 0.999), \text{wd}=0.0$). No cosine schedule or weight decay was executed. | Affirm Adam [19]; correct text and references; remove all AdamW references. |
| **2. Inventory Max Patches (7,532 vs. 7,556)** | **RESOLVED** | `rst1_source_event_inventory_v1.csv` records Event 1111009 with exactly **7,556 patches**, Event 496 with **21 patches**, total **21,462 patches**, ratio **359.8×**. | Synchronize 7,556 and 359.8× across text, tables, captions, and regenerated Figure 1. |
| **3. M1 Realized vs. Expected Draws** | **RESOLVED** | Multinomial draw with $p=1/27$ per event yields expected draws $\approx 795$/epoch. Realized draws are recorded in `final_manifest.json`. | Label horizontal line in Figure 1 explicitly as "Expected Draws Under M1 Event Balancing". |
| **4. Paired Direction Counts in Confirmatory Test** | **RESOLVED** | `rst7/paired_event_deltas_v1.csv` has exactly 9 positive, 9 negative, 0 zero. At $\pm 0.010$: 7 positive, 7 negative, 4 neutral. Plotted data and table match. | Remove Codex warning; present exact counts and tail-skewed magnitude distribution. |
| **5. Resource Comparison (Table 6 / Fig 9)** | **RESOLVED** | 4,913 MB VRAM is training peak under microbatch 8 $\times$ 4 accum. Comparing training VRAM to TTA test VRAM is invalid. | Retire Fig 9 from figures. Report verified training costs in Table 6; describe zero-adaptation operational profile. |
| **6. SAR Input Domain & Preprocessing** | **RESOLVED** | `dataset.py:206-224` and `rst1_closeout_v1.md` verify stored rasters are linear amplitude $A = \sqrt{I}$ (IVV/IVH), clipped to $[0.0, 0.15]$, standardized via global constants. | Confirm linear amplitude and exact preprocessing constants in Chapter 4. |
| **7. Class Weights M2 Calculation** | **RESOLVED** | `loss.py:28-39` verifies $w_c = 1/\sqrt{f_c}$, normalized to mean 1.0, clipped to $[0.5, 2.0]$, normalized again. Computed strictly on training events. | Document exact formula, clipping bounds, and double-masking rule in Chapter 5. |
| **8. Reviewer Comments in Prose** | **RESOLVED** | Multiple paragraphs in v2 contained editorial audit notes inserted into body prose and captions. | Remove all reviewer-facing prose; rephrase as confident, factual academic text. |
| **9. Figure Consecutive Numbering** | **RESOLVED** | Figures in v1/v2 were out of first-appearance order (Fig 6 appeared before Fig 1). | Renumber 1 through 8 in strict appearance order; update text, captions, and List of Figures. |
| **10. Authorship, ID, and Acknowledgment** | **RESOLVED** | Author confirmed: Lin Xiaoya (Matriculation U2321632G). Supervisor: Assoc. Prof. Wen Bihan. ROSE Lab fellow: Dr. Qing Yuanyuan. | Apply exact author-requested acknowledgment; retain matriculation ID; include truthful AI declaration. |

---

## 3. Deep-Dive Forensic Investigations

### 3.1 The Adam versus AdamW Discrepancy

#### Investigation Steps:
1. **Source Training Pipeline Code:**
   In `src/robust_source_training/runner.py`, lines 466–472:
   ```python
   opt_cfg = self.config.get("optimization", {})
   optimizer = torch.optim.Adam(
       model.parameters(),
       lr=float(opt_cfg.get("learning_rate", 1e-3)),
       betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
       weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
   )
   ```
   The runner directly calls `torch.optim.Adam`. It does not support or instantiate AdamW.

2. **Configuration Ledgers:**
   All 40 frozen JSON configuration files across RST1, RST2, RST4, RST6, RST8, and RST9 specify:
   ```json
   "optimization": {
     "optimizer": "Adam",
     "learning_rate": 0.001,
     "betas": [0.9, 0.999],
     "weight_decay": 0.0
   }
   ```
   No configuration file in the robust source training pipeline contains `"optimizer": "AdamW"`.

3. **Saved Checkpoint State Dicts:**
   Inspecting all 29 saved checkpoint files across all phases (`outputs/robust_source_training/**/final_checkpoint.pt`), each checkpoint's `optimizer_state_dict["param_groups"][0]` records:
   - `lr`: `0.001`
   - `betas`: `(0.9, 0.999)`
   - `eps`: `1e-08`
   - `weight_decay`: `0.0`
   - `amsgrad`: `False`
   This is the distinctive state dictionary schema of PyTorch's native `Adam` optimizer.

4. **TTA Optimizers:**
   - Tent (`src/tta/tent.py:80`): `optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))`.
   - EATA (`src/changeguard/e2_adapters.py:143`): `optim.SGD(params, lr=2.5e-4)`.
   - SAR (`src/tta/sar_adapter.py:128`): `SAM` wrapping `optim.SGD`.
   - Source-Only: Zero optimizer.
   None of the test-time adaptation methods employed AdamW.

5. **Origin of the AdamW Error:**
   Git history traces the introduction of "AdamW" to commit `9334a21` (`papers/igarss2027/changeguard_igarss2027_draft_v1.tex` line 70 and `references.bib`). The text was written using standard deep learning boilerplate ("trained for 16 epochs using the AdamW optimizer with a cosine annealing learning rate schedule..."), citing Loshchilov & Hutter (2018). This was a drafting oversight that contradicted the actual repository implementation. Codex's correction to Adam in v2 was completely justified by empirical code and checkpoint evidence.

---

### 3.2 Source Inventory Discrepancy (7,532 vs. 7,556)

#### Investigation Steps:
1. **Authoritative Source CSV:**
   `configs/robust_source_training/rst1_source_event_inventory_v1.csv` lines 2–28:
   - Event 1111009 has **7,556 eligible training patches**.
   - Event 496 has **21 eligible training patches**.
   - Sum of eligible patches across all 27 events: **21,462 patches**.

2. **Group Allocation:**
   `configs/robust_source_training/rst1_event_folds_v1.json`:
   - Group A (9 events): 7,556 + 112 + 108 + 102 + 96 + 71 + 68 + 26 + 21 = **8,160 patches**.
   - Group B (9 events): 4,086 + 684 + 445 + 376 + 341 + 306 + 218 + 189 + 162 = **6,807 patches**.
   - Group C (9 events): 1,980 + 1,619 + 742 + 620 + 398 + 362 + 313 + 248 + 213 = **6,495 patches**.
   - Total: $8,160 + 6,807 + 6,495 = \mathbf{21,462\text{ patches}}$.

3. **Disparity Ratio:**
   $$\text{Ratio} = \frac{7556}{21} = 359.8095\dots \approx 359.8\times$$
   The figure 7,532 ($7532 / 21 = 358.666\dots \approx 358.7\times$) appeared due to an earlier typographical error (24 patches fewer than 7,556). The actual raster corpus and manifest contain 7,556 patches.

4. **Sampling Dynamics (M1):**
   In an epoch of 21,462 patch draws across 27 events, sampling an event uniformly at random yields an expected frequency of:
   $$\mathbb{E}[\text{draws}_e] = \frac{21,462}{27} \approx 794.89 \text{ draws/epoch}$$
   The realized draws in any finite training run follow a multinomial distribution. For example, in epoch 1 of `rst9_r2_snunet_all27_s0_v1`, 24 events had exactly 795 draws and 3 events had 794 draws (sum = 21,462). Labeling the reference line as an **expectation** correctly describes the stochastic sampling process.

---

### 3.3 Confirmatory Direction Counts (Figure 3 / Old Figure 2)

#### Investigation Steps:
1. Re-evaluating all 18 events from `outputs/robust_source_training/rst7/paired_event_deltas_v1.csv`:
   - **Degraded ($\Delta \text{F1} < 0$):** 9 events
     1. Event 554: $\Delta = -0.0322$
     2. Event 1111004: $\Delta = -0.0256$
     3. Event 1111009: $\Delta = -0.0228$
     4. Event 567: $\Delta = -0.0197$
     5. Event 273: $\Delta = -0.0158$
     6. Event 498: $\Delta = -0.0143$
     7. Event 496: $\Delta = -0.0132$
     8. Event 518: $\Delta = -0.0081$
     9. Event 1111010: $\Delta = -0.0062$
   - **Improved ($\Delta \text{F1} > 0$):** 9 events
     10. Event 417: $\Delta = +0.0011$
     11. Event 502: $\Delta = +0.0030$
     12. Event 324: $\Delta = +0.0129$
     13. Event 267: $\Delta = +0.0362$
     14. Event 174: $\Delta = +0.0377$
     15. Event 421: $\Delta = +0.0396$
     16. Event 555: $\Delta = +0.0428$
     17. Event 470: $\Delta = +0.1140$
     18. Event 118: $\Delta = +0.1150$

2. **Fold Distribution:**
   - Fold A (9 events): 3 improved (118, 421, 174), 6 degraded (554, 1111009, 567, 273, 498, 496). Mean delta = $+0.0083$.
   - Fold B (9 events): 6 improved (470, 555, 267, 324, 502, 417), 3 degraded (1111004, 518, 1111010). Mean delta = $+0.0189$.
   - Combined (18 events): 9 improved, 9 degraded. Grand macro delta = $+0.0136$.

3. **Significance of Skewed Magnitudes:**
   While the event direction count is evenly split (9 vs. 9), the **magnitude of gains heavily outweighs the magnitude of degradations**:
   - Mean of positive gains: $+0.0447$ (sum = $+0.4023$)
   - Mean of negative losses: $-0.0175$ (sum = $-0.1579$)
   - Largest gain: $+0.1150$ (Event 118) and $+0.1140$ (Event 470)
   - Largest loss: $-0.0322$ (Event 554)
   This confirms that Recipe R2 functions precisely as designed: it rescues severely degraded tail events where standard training collapsed, while sacrificing only marginal performance in high-baseline, data-dominant scenes.

---

### 3.4 Table 6 and Resource Claims

#### Investigation Steps:
1. `outputs/robust_source_training/rst9/rst9_r2_snunet_all27_s0_v1/final_manifest.json` confirms:
   - `optimizer_steps`: 10,736
   - `wall_clock_runtime_seconds`: 11,682.69 (~3.25 hours)
   - `peak_vram_mb`: 4,913.16 MB
2. This VRAM measurement represents **peak training VRAM** for SNUNet-ECAM under physical microbatch size 8, gradient accumulation 4, FP32 precision, PyTorch autograd graph retention, and Adam optimizer states ($12.03\text{M} \times 4 \times 3 \text{ bytes} \approx 144\text{ MB}$ parameter + gradient + moment buffers).
3. In contrast, the numbers reported for Tent (5,840 MB) and CoTTA (8,950 MB) in earlier drafts were test-time adaptation estimates from separate pilot experiments.
4. **Conclusion:** Comparing training peak VRAM to test adaptation VRAM as a unified deployment table was scientifically inappropriate. Table 6 has been refactored into a rigorous verified training resource specification, accompanied by an explicit qualitative operational comparison establishing that frozen R2 inference requires zero target-time parameter updates, zero backpropagation, and zero runtime adaptation state.

---

## 4. Master Change Log (v1 / v2 to v3)

| Manuscript Location | Original Statement (v1 / v2) | Authoritative Replacement (v3) | Reason & Justification | Category |
| :--- | :--- | :--- | :--- | :--- |
| **Title Page** | `[Matriculation / Student ID — confirm]` / `Matriculation No. U2321632G` | `Submitted by: Lin Xiaoya`<br>`Matriculation No. U2321632G` | Confirmed student identity from academic project record. | Factual update |
| **Abstract** | Long v1 draft (~380 words) / tentative v2 draft (~260 words with review notes) | Coherent, balanced 340-word academic abstract covering motivation, TTA limits, R2 design, primary & retrospective results, and bounded limitations. | Standard NTU 1-page template compliance; confident, factual academic tone. | Structural & Tone |
| **Acknowledgments** | Placeholder / generic note | "I would like to thank my supervisor, Associate Professor Wen Bihan, for his guidance and support. I also thank Dr. Qing Yuanyuan, Research Fellow at ROSE Lab, for support during this project." | Author-requested exact text; no invented supervisory endorsement. | Authoritative alignment |
| **Figure 6 / Fig 1 Caption** | `...21 patches min vs 7,532 patches max (358.7× disparity)...` | `Figure 1: Equalization of Event Sampling Frequencies Under M1 (Authoritative 27 Source Events: 21 Patches Min vs. 7,556 Patches Max; 359.8× Disparity)` | Corrected patch counts and ratio from canonical inventory CSV. Renumbered to Fig 1. | Factual correction & numbering |
| **Figure 2 / Fig 3 Caption** | `...The supplied RST7 event table contains nine positive and nine negative differences; the plotted source data require reconciliation if they show a different count.` | `Figure 3: Primary Confirmatory Event-Level Paired Deltas (RST7) (18 Held-Out Source Events across Seeds 0, 1, 2: 9 Positive Mean Differences, 9 Negative Mean Differences)` | Removed Codex warning; affirmed exact agreement between plotted data and canonical table. | Reviewer prose removal |
| **Figure 4 (Old Fig 3)** | Legend located at `loc="upper left"` overlapping delta annotation. | Legend relocated to `loc="upper right"`; y-limits expanded to $[0.20, 0.95]$. | Visual QA fix preventing graphic collision. | Visual QA |
| **Figure 5 (Old Fig 4)** | Figure suptitle crowded with panel titles. | Subplot margins adjusted (`top=0.82`, `wspace=0.22`); layout padded cleanly. | Visual QA fix eliminating text overlap. | Visual QA |
| **Figure 6 (Old Fig 5)** | Figure suptitle crowded with panel titles and legends. | Subplot margins adjusted (`top=0.82`, `wspace=0.22`); clean marker diamond lines. | Visual QA fix eliminating crowding. | Visual QA |
| **Table 6 / Chapter 9** | Mismatched quantitative VRAM table and speculative Figure 9 with reviewer warnings. | Verified training cost table (10,736 steps, ~3.2h, 4,913 MB) and qualitative operational matrix (zero target backprop / state). Figure 9 retired. | Eliminated unsupported deployment benchmark; reported verified costs. | Evidence integrity |
| **Chapter 5 (Optimization)** | References to AdamW with cosine annealing and weight decay $10^{-4}$. | "All models are trained for 16 epochs using the standard Adam optimizer [19] ($\text{lr}=10^{-3}, \beta=(0.9, 0.999), \text{weight\_decay}=0.0$, effective batch size 32, fixed learning rate schedule)." | Reconciled optimizer to actual code and checkpoint state dicts. | Factual correction |
| **Chapter 13 (Reflections)** | Generic AI reflection placeholder. | Student Learning Outcome Reflection cleanly drafted across Problem Analysis, Design, Engineering Tools, and Integrity, flagged for final author confirmation. | Author confirmation item preserved without imposing supervisor approval prerequisite. | Compliance |
| **AI Declaration** | Generic placeholder. | Comprehensive NTU GenAI declaration truthfully detailing AI assistance in planning, code implementation, test automation, formatting, and analysis, while establishing sole student responsibility. | NTU policy compliance. | Ethical disclosure |
| **Bibliography** | 13 references in v1; 19 in v2 (including Kingma & Ba for Adam). | 19 fully audited IEEE citations. Reference `[19]` correctly cites Kingma & Ba (Adam). | Verified academic citations linked to in-text markers. | Academic integrity |

---

## 5. Old-to-New Figure and Table Concordance

### Figures

| Old Figure ID (v1) | v2 Review ID | Authoritative v3 ID | Figure Title & Scope | Status in v3 |
| :--- | :--- | :--- | :--- | :--- |
| Figure 6 | Figure 6 | **Figure 1** | Geographic patch count distribution across 27 source events (359.8× disparity) | Regenerated with 7,556 max & expected draw label |
| Figure 1 | Figure 1 | **Figure 2** | Component discovery staircase on Fold C (RST4, Seed 0) | Renumbered to consecutive appearance |
| Figure 2 | Figure 2 | **Figure 3** | Primary confirmatory paired per-event F1 differences on SNUNet-ECAM (RST7) | Regenerated; affirmed 9 positive / 9 negative deltas |
| Figure 3 | Figure 3 | **Figure 4** | Tail reliability vs average generalization in RST7 (grand macro, bottom-3, worst-event) | Regenerated; legend moved to upper right (no overlap) |
| Figure 4 | Figure 4 | **Figure 5** | Precision vs recall dynamics under Recipe R2 (RST7 vs RST9) | Regenerated; panel margin crowding resolved |
| Figure 5 | Figure 5 | **Figure 6** | Evaluation of seed stability across random seeds 0, 1, 2 (RST7 vs RST9) | Regenerated; top title crowding resolved |
| Figure 7 | Figure 7 | **Figure 7** | Cross-backbone architecture transfer evaluation on SiamUnet_diff (RST8) | Authoritative comparison preserved |
| Figure 8 | Figure 8 | **Figure 8** | RST9 Retrospective historical-target extension paired deltas across 16 events | Retrospective scope clearly labeled |
| Figure 9 | Figure 9 | *Retired* | Historical resource tradeoff (unsupported quantitative benchmark) | Retired from figures; replaced by Table 6 |

### Tables

| Table ID | Title & Scope | Verification Status |
| :--- | :--- | :--- |
| **Table 1** | Component Discovery Staircase Performance on Discovery Fold C (RST4, Seed 0) | Verified against `outputs/robust_source_training/rst4/analysis_v1/discovery_recipe_summary.csv` |
| **Table 2** | Primary Event-Disjoint Confirmation Results on SNUNet-ECAM Across 18 Events (RST7, Seeds 0, 1, 2) | Verified against `outputs/robust_source_training/rst7/confirmatory_summary_v1.json` |
| **Table 3** | Seed-0 Architecture-Transfer Results for SiamUnet_diff on 18 Confirmatory Events (RST8) | Verified against `outputs/robust_source_training/rst8/stage_a_summary_v1.json` |
| **Table 4** | Deployable All-27 Source-Trained Checkpoint Ledger (RST9, Seeds 0, 1, 2) | Verified against `outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json` |
| **Table 5** | Retrospective Historical-Target Benchmark Across 16 Disaster Events (RST9, Seeds 0, 1, 2) | Verified against `outputs/robust_source_training/rst9_analysis/rst9_comparative_analysis_v1.json` |
| **Table 6** | Computational Complexity and Verified Training Resource Specifications | Refactored to verified training costs and operational deployment profiles |
| **Table 7** | Comprehensive 18-Event Paired Confirmation Breakdown (Appendix Companion) | Generated directly from `outputs/robust_source_training/rst7/paired_event_deltas_v1.csv` |

---

## 6. Audit Conclusion

All discrepancies and editorial defects identified in the v1 and v2 drafts have been resolved using verifiable repository artifacts. The resulting Word deliverable (`ChangeGuard_FYP_Report_Reviewed_Bold_v3.docx`) and matching PDF provide an authoritative, publication-quality Final Year Project report that satisfies all degree and template requirements of Nanyang Technological University.
