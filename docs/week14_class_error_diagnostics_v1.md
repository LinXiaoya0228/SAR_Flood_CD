# Week 14: Offline Class Error Diagnostics Report (v1)

**Status:** Completed Implementer Self-Verification  
**Date:** October 19, 2026  
**Repository:** `ChangeGuard`  
**Execution Context:** Offline Python evaluation via `tools/run_week14_offline_diagnosis.py` (`vfm_env`)  
**Scope:** 6 KuroSiwo development events (`[279, 437, 520, 559, 1111003, 1111008]`), 2,826 samples, 135,859,080 valid pixels  
**Conditions Evaluated:**
1. `source_only`
2. `matched_bn_no_update_v1`
3. `matched_tent_pre_update_v1`
4. `C1_consistency_no_conf`
5. `C2_consistency_conf90`
6. `C3_anchored_consistency_v1`

---

## 1. Executive Summary & Diagnostic Objectives

During Weeks 12 and 13, exploratory adaptation experiments revealed substantial performance degradation on specific events (notably Event 279 and Event 437) primarily triggered by replacing source Batch Normalization (BN) statistics with target batch statistics (`matched_bn_no_update_v1`), which active adaptation (Tent, consistency, anchored consistency) only marginally altered.

The primary objective of this Week 14 diagnostic is to perform an offline $3 \times 3$ confusion decomposition on frozen pre-update predictions across all six development events. Ground truth labels in KuroSiwo comprise:
- **Class 0:** Land / Background
- **Class 1:** Permanent Water
- **Class 2:** Flood Water (the positive change class of interest)
- *(Label 3 is unannotated / invalid and masked out alongside `MK0_MNA != 1`)*

Binary classification metrics collapse classes 0 and 1 into "Non-Flood" (Negative) and treat class 2 as "Flood" (Positive). This diagnostic answers two fundamental questions:
1. **False Positive Decomposition:** When the model erroneously predicts Flood, is the false alarm caused by confusion with permanent water (`c12`: Permanent Water $\to$ Flood) or dry land (`c02`: Land $\to$ Flood)?
2. **False Negative Decomposition:** When true flood is missed, is it misclassified as land (`c20`: Flood $\to$ Land) or as permanent water (`c21`: Flood $\to$ Permanent Water)?

### Key Empirical Findings:
1. **The Event 279 False Alarm Surge is Overwhelmingly Dry Land:** In Event 279, the addition of candidate batch BN statistics creates an increase of $+918,680$ false positive flood pixels relative to source-only ($+51.6\%$). **$97.37\%$ of this surge ($+894,507$ pixels) originates from dry Land being misclassified as Flood (`c02`)**, whereas permanent water confusion accounts for only $+24,173$ pixels ($2.63\%$).
2. **Per-Unit-Area Rates vs. Absolute Shares:** Although permanent water exhibits a higher false alarm rate per pixel ($8.17\%$ in source-only and $9.87\%$ in BN no-update for Event 279) than land ($1.46\%$ and $2.25\%$, respectively), land comprises $98.02\%$ of all valid pixels in Event 279. Consequently, **land false alarms constitute $94.78\%$ of all false positive pixels** in Event 279 under BN no-update.
3. **Macro Error Structure Across All 6 Events:** Across all 135.86 million evaluated pixels, Land $\to$ Flood false alarms account for **$93.63\%$** of total false alarms under BN no-update ($2,723,797$ out of $2,909,127$ FP), while Permanent Water $\to$ Flood accounts for only **$6.37\%$** ($185,330$ FP).
4. **False Negative Misses:** Missed flood pixels are predominantly classified as dry land ($78.94\%$ to $80.15\%$ of misses), with misses to permanent water accounting for the remaining $\sim 20\%$.
5. **C3 Source-Anchor Impact:** The C3 source anchor ($L_{\text{anchor}} = D_{\text{KL}}(p_{\text{adapt}} \parallel p_{\text{source}})$) provides a modest aggregate reduction of $14,436$ false positives compared to BN no-update ($14,141$ fewer land false alarms, $295$ fewer permanent water false alarms). However, this reduction offsets only $1.5\%$ of the $+938,955$ false alarm inflation caused by candidate BN statistics.

---

## 2. Methodology & Formal Metric Definitions

### 2.1 Confusion Matrix Structure ($3 \times 3$)

Let $C_{i, j}$ denote the number of valid pixels where the ground-truth class is $i \in \{0, 1, 2\}$ and the predicted class is $j \in \{0, 1, 2\}$.

$$\mathbf{C} = \begin{bmatrix}
C_{0,0} (\text{Land}\to\text{Land}) & C_{0,1} (\text{Land}\to\text{Perm}) & C_{0,2} (\text{Land}\to\text{Flood}) \\
C_{1,0} (\text{Perm}\to\text{Land}) & C_{1,1} (\text{Perm}\to\text{Perm}) & C_{1,2} (\text{Perm}\to\text{Flood}) \\
C_{2,0} (\text{Flood}\to\text{Land}) & C_{2,1} (\text{Flood}\to\text{Perm}) & C_{2,2} (\text{Flood}\to\text{Flood})
\end{bmatrix}$$

### 2.2 Conservation and Binary Equivalence Invariants

The evaluation pipeline strictly asserts the following mathematical invariants:
1. **Total Valid Pixels:**
   $$\sum_{i=0}^2 \sum_{j=0}^2 C_{i, j} = N_{\text{valid}} = 135,859,080$$
2. **Binary Positive (Flood) Classification:**
   $$\text{TP} = C_{2,2}$$
   $$\text{FP}_{\text{total}} = C_{0,2} + C_{1,2} = \text{FP}_{\text{land}} + \text{FP}_{\text{perm}}$$
   $$\text{FN}_{\text{total}} = C_{2,0} + C_{2,1} = \text{FN}_{\text{land}} + \text{FN}_{\text{perm}}$$
   $$\text{TN} = C_{0,0} + C_{0,1} + C_{1,0} + C_{1,1}$$

### 2.3 Explicit Denominators for Diagnostic Rates

To prevent ambiguity, all rates are reported with explicit mathematical denominators:
- **Land False Alarm Rate:** Fraction of true land mispredicted as flood:
  $$\text{FAR}_{\text{land}} = \frac{C_{0,2}}{N_{\text{gt\_land}}}, \quad \text{where } N_{\text{gt\_land}} = C_{0,0} + C_{0,1} + C_{0,2}$$
- **Permanent Water False Alarm Rate:** Fraction of true permanent water mispredicted as flood:
  $$\text{FAR}_{\text{perm}} = \frac{C_{1,2}}{N_{\text{gt\_perm}}}, \quad \text{where } N_{\text{gt\_perm}} = C_{1,0} + C_{1,1} + C_{1,2}$$
- **False Positive Origin Shares:** Relative composition of the false alarm pool:
  $$\text{Share}_{\text{FP, land}} = \frac{C_{0,2}}{\text{FP}_{\text{total}}}, \qquad \text{Share}_{\text{FP, perm}} = \frac{C_{1,2}}{\text{FP}_{\text{total}}}$$
- **Flood Miss Rates:** Fraction of true flood mispredicted as land or permanent water:
  $$\text{MissRate}_{\text{to\_land}} = \frac{C_{2,0}}{N_{\text{gt\_flood}}}, \qquad \text{MissRate}_{\text{to\_perm}} = \frac{C_{2,1}}{N_{\text{gt\_flood}}}, \quad \text{where } N_{\text{gt\_flood}} = C_{2,0} + C_{2,1} + C_{2,2}$$
- **False Negative Destination Shares:** Relative composition of missed flood pixels:
  $$\text{Share}_{\text{FN, land}} = \frac{C_{2,0}}{\text{FN}_{\text{total}}}, \qquad \text{Share}_{\text{FN, perm}} = \frac{C_{2,1}}{\text{FN}_{\text{total}}}$$

---

## 3. Aggregate Error Decomposition (All 6 Events Pooled)

Across the entire six-event development set ($N_{\text{valid}} = 135,859,080$ pixels; Ground Truth: $132,008,062$ Land [$97.17\%$], $2,051,142$ Permanent Water [$1.51\%$], $1,799,876$ Flood [$1.32\%$]):

| Condition | $\text{FP}_{\text{total}}$ | $\text{FP}_{\text{land}}$ | $\text{FP}_{\text{perm}}$ | $\text{Share}_{\text{FP, land}}$ | $\text{FAR}_{\text{land}}$ | $\text{FAR}_{\text{perm}}$ | $\text{FN}_{\text{total}}$ | $\text{FN}_{\text{land}}$ | $\text{FN}_{\text{perm}}$ | $\text{Share}_{\text{FN, land}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`source_only`** | 1,970,172 | 1,812,621 | 157,551 | 92.00% | 1.37% | 7.68% | 477,566 | 382,763 | 94,803 | 80.15% |
| **`matched_bn_no_update_v1`** | 2,909,127 | 2,723,797 | 185,330 | 93.63% | 2.06% | 9.04% | 464,939 | 367,028 | 97,911 | 78.94% |
| **`matched_tent_pre_update_v1`** | 2,832,369 | 2,648,542 | 183,827 | 93.51% | 2.01% | 8.96% | 467,065 | 369,516 | 97,549 | 79.11% |
| **`C1_consistency_no_conf`** | 2,911,192 | 2,725,826 | 185,366 | 93.63% | 2.06% | 9.04% | 464,887 | 366,971 | 97,916 | 78.94% |
| **`C2_consistency_conf90`** | 2,906,575 | 2,721,293 | 185,282 | 93.63% | 2.06% | 9.03% | 465,012 | 367,110 | 97,902 | 78.95% |
| **`C3_anchored_consistency_v1`** | 2,894,691 | 2,709,656 | 185,035 | 93.61% | 2.05% | 9.02% | 465,340 | 367,495 | 97,845 | 78.97% |

### Observations on Aggregate Channels:
1. **False Positives are Dominated by Land:** Under all conditions, over $92\%$ of all false positive pixels are dry land pixels predicted as flood. Permanent water false alarms never exceed $8.0\%$ of the false positive burden in aggregate.
2. **False Negatives are Dominated by Misses to Land:** When flood water is undetected, four out of five missed pixels ($78.9\%$ to $80.2\%$) are labeled as Land, meaning the model treats them as unflooded background terrain rather than confusing them with baseline water bodies.

---

## 4. Event-by-Event Error Decomposition & Heterogeneity

The six development events display marked structural heterogeneity in scene composition and error responses:

### 4.1 Event 279 (Large Scale Scene, Extreme Land Dominance)
- **Scene Composition:** Total Valid: $116,199,247$ ($85.5\%$ of entire dev set). Land: $113,902,340$ ($98.02\%$), Permanent Water: $1,428,210$ ($1.23\%$), Flood: $868,697$ ($0.75\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{FP}_{\text{land}} = 1,663,158$ ($93.44\%$ of FP), $\text{FP}_{\text{perm}} = 116,720$ ($6.56\%$ of FP). $\text{FN}_{\text{land}} = 224,934$ ($88.67\%$ of FN), $\text{FN}_{\text{perm}} = 28,752$. F1 = $0.3769$.
  - `matched_bn_no_update_v1`: $\text{FP}_{\text{land}} = 2,557,665$ ($94.78\%$ of FP), $\text{FP}_{\text{perm}} = 140,893$ ($5.22\%$ of FP). $\text{FN}_{\text{land}} = 209,394$, $\text{FN}_{\text{perm}} = 29,156$. F1 = $0.3003$.
  - `C3_anchored_consistency_v1`: $\text{FP}_{\text{land}} = 2,543,525$, $\text{FP}_{\text{perm}} = 140,595$. F1 = $0.3011$.
- **Detailed Origin of the False Positive Surge:**
  $$\Delta \text{FP}_{\text{total}} (\text{BN} - \text{Source}) = +918,680 \text{ pixels}$$
  $$\Delta \text{FP}_{\text{land}} = +894,507 \text{ pixels} \quad (\mathbf{97.37\%} \text{ of surge})$$
  $$\Delta \text{FP}_{\text{perm}} = +24,173 \text{ pixels} \quad (2.63\% \text{ of surge})$$
  *Conclusion:* The severe degradation in Event 279 is almost entirely driven by candidate batch statistics inducing spurious flood detections across vast expanses of dry land.

### 4.2 Event 437 (Low Flood Prevalence, High Land Dominance)
- **Scene Composition:** Total Valid: $7,859,430$. Land: $7,750,774$ ($98.62\%$), Permanent Water: $44,654$ ($0.57\%$), Flood: $64,002$ ($0.81\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{FP}_{\text{land}} = 25,977$, $\text{FP}_{\text{perm}} = 884$. F1 = $0.6994$.
  - `matched_bn_no_update_v1`: $\text{FP}_{\text{land}} = 40,786$ ($+14,809$), $\text{FP}_{\text{perm}} = 825$ ($-59$). F1 = $0.6344$.
- **Origin of Surge:** In Event 437, candidate BN statistics increased land false alarms by $+57.0\%$ ($+14,809$ pixels), while permanent water false alarms actually slightly decreased ($-59$ pixels). $100\%$ of the net false positive surge stems from dry land.

### 4.3 Event 520 (High Flood Prevalence, Significant Permanent Water)
- **Scene Composition:** Total Valid: $3,455,248$. Land: $2,722,051$ ($78.78\%$), Permanent Water: $403,954$ ($11.69\%$), Flood: $329,243$ ($9.53\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{FP}_{\text{land}} = 53,038$, $\text{FP}_{\text{perm}} = 31,739$ ($37.44\%$ share of FP). $\text{FN}_{\text{land}} = 40,806$, $\text{FN}_{\text{perm}} = 58,694$ ($58.99\%$ share of FN). F1 = $0.7138$.
  - `matched_bn_no_update_v1`: $\text{FP}_{\text{land}} = 57,735$ ($+4,697$), $\text{FP}_{\text{perm}} = 35,225$ ($+3,486$). $\text{FN}_{\text{land}} = 39,446$, $\text{FN}_{\text{perm}} = 62,666$ ($+3,972$). F1 = $0.6996$.
- **Origin of Shifts:** Event 520 has substantial permanent water presence. Here, permanent water confusion constitutes $37.9\%$ of false alarms, and $61.4\%$ of missed flood pixels are classified as permanent water (`c21`). Both land and permanent water error channels expanded moderately under BN no-update.

### 4.4 Event 559 (Moderate Flood Prevalence, Beneficial Adaptation Case)
- **Scene Composition:** Total Valid: $1,843,520$. Land: $1,792,096$ ($97.21\%$), Permanent Water: $752$ ($0.04\%$), Flood: $50,672$ ($2.75\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{TP} = 27,727$, $\text{FP}_{\text{land}} = 4,046$, $\text{FP}_{\text{perm}} = 5$, $\text{FN}_{\text{land}} = 22,934$, $\text{FN}_{\text{perm}} = 11$. F1 = $0.6726$.
  - `matched_bn_no_update_v1`: $\text{TP} = 29,050$ ($+1,323$), $\text{FP}_{\text{land}} = 4,179$ ($+133$), $\text{FP}_{\text{perm}} = 6$ ($+1$), $\text{FN}_{\text{land}} = 21,618$ ($-1,316$), $\text{FN}_{\text{perm}} = 4$. F1 = $0.6924$ ($+0.0199$).
- **Origin of Gain:** In Event 559, candidate BN statistics produced a net benefit: it recovered $1,323$ true positive flood pixels previously missed as dry land, while increasing false alarms by only $134$ pixels.

### 4.5 Event 1111003 (Very Low Prevalence Flood, Low False Alarm Regime)
- **Scene Composition:** Total Valid: $815,490$. Land: $792,978$ ($97.24\%$), Permanent Water: $15,551$ ($1.91\%$), Flood: $6,961$ ($0.85\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{TP} = 3,537$, $\text{FP}_{\text{land}} = 16,730$, $\text{FP}_{\text{perm}} = 315$, $\text{FN}_{\text{land}} = 3,174$, $\text{FN}_{\text{perm}} = 250$. F1 = $0.2568$.
  - `matched_bn_no_update_v1`: $\text{TP} = 3,149$ ($-388$), $\text{FP}_{\text{land}} = 13,410$ ($-3,320$), $\text{FP}_{\text{perm}} = 242$ ($-73$), $\text{FN}_{\text{land}} = 3,400$ ($+226$), $\text{FN}_{\text{perm}} = 412$ ($+162$). F1 = $0.2650$ ($+0.0082$).
- **Origin of Gain:** Unlike other events, candidate BN statistics reduced false alarms on both land ($-3,320$) and water ($-73$). Even though recall declined ($-388$ TP), the substantial false alarm reduction yielded a modest F1 increase ($+0.0082$).

### 4.6 Event 1111008 (High Prevalence Flood, Highly Stable)
- **Scene Composition:** Total Valid: $5,686,145$. Land: $5,047,823$ ($88.77\%$), Permanent Water: $158,021$ ($2.78\%$), Flood: $480,301$ ($8.45\%$).
- **Confusion Breakdown:**
  - `source_only`: $\text{TP} = 397,434$, $\text{FP}_{\text{land}} = 49,672$, $\text{FP}_{\text{perm}} = 7,888$, $\text{FN}_{\text{land}} = 75,803$, $\text{FN}_{\text{perm}} = 7,064$. F1 = $0.8499$.
  - `matched_bn_no_update_v1`: $\text{TP} = 396,392$ ($-1,042$), $\text{FP}_{\text{land}} = 50,022$ ($+350$), $\text{FP}_{\text{perm}} = 8,139$ ($+251$), $\text{FN}_{\text{land}} = 78,342$ ($+2,539$), $\text{FN}_{\text{perm}} = 5,567$ ($-1,497$). F1 = $0.8480$ ($-0.0018$).
- **Origin of Shifts:** Minimal variation across all adaptation variants ($< 0.002$ delta F1). The distribution of errors between land and water remained virtually static.

---

## 5. Comparative Analysis: C3 Source Anchor vs. BN No Update and Source

The source anchor constraint ($L_{\text{anchor}} = D_{\text{KL}}(p \parallel p_{\text{source}})$ with $\lambda_{\text{anchor}} = 0.5$) was introduced in Week 13 to restrain parameter drift and pull adapted predictions toward the source model.

### 5.1 Shift in Error Channels for C3 vs. `matched_bn_no_update_v1`

Across all events:
- $\Delta \text{FP}_{\text{land}} = -14,141$ pixels ($-0.52\%$)
- $\Delta \text{FP}_{\text{perm}} = -295$ pixels ($-0.16\%$)
- $\Delta \text{FN}_{\text{land}} = +467$ pixels ($+0.13\%$)
- $\Delta \text{FN}_{\text{perm}} = -66$ pixels ($-0.07\%$)
- $\Delta \text{F1} = +0.00095$ ($+0.095$ percentage points)

In Event 279 specifically:
- $\Delta \text{FP}_{\text{land}} = -14,140$ pixels
- $\Delta \text{FP}_{\text{perm}} = -298$ pixels
- $\Delta \text{FN}_{\text{land}} = +464$ pixels
- $\Delta \text{FN}_{\text{perm}} = -64$ pixels
- $\Delta \text{F1} = +0.00087$

### 5.2 Mechanistic Interpretation
1. **Directionally Consistent with Source Shrinkage:** The C3 anchor does successfully pull predictions toward the source distribution, reducing false alarms primarily on land ($-14,141$ pixels).
2. **Order-of-Magnitude Insufficiency:** While C3 reduces false alarms by $14,436$ pixels relative to BN no-update, BN no-update has $938,955$ more false alarms than source-only. Thus, the anchor rectifies only **$1.54\%$** of the false alarm deficit introduced by candidate batch normalization.
3. **Entropy Minimization vs. Anchoring:** By comparison, Tent's pure entropy minimization (`matched_tent_pre_update_v1`) reduced land false alarms by $75,255$ pixels relative to BN no-update, yet still suffered from over $835,000$ excess false alarms compared to source-only.

---

## 6. Critical Boundary: Offline Diagnosis vs. Online Method Promotion

A critical principle of rigorous machine learning research is to distinguish between **retrospective diagnostic analysis** and **online algorithm design**:

### 6.1 What Offline Diagnosis Establishes:
- Identifies the empirical error channels ($C_{i, j}$) responsible for observed metric drops.
- Quantifies that the primary vulnerability under candidate BN statistics is false alarms on dry land ($C_{0,2}$), rather than confusion between flood and permanent water ($C_{1,2}$).
- Demonstrates extreme event heterogeneity: flood prevalence varies tenfold ($0.75\%$ in Event 279 vs. $9.53\%$ in Event 520), and land false alarm rates shift non-uniformly across scenes.

### 6.2 What Offline Diagnosis Does NOT Authorize:
- **No Ground-Truth Leakage into Test-Time Inference:** Ground truth is strictly unavailable at test time. An online method cannot use the event's true land/water ratio or class proportions to tune thresholds, select hyperparameters, or calibrate outputs.
- **No Direct Justification for Ad-Hoc "Class-Aware" Online Variants:**
  - **Permanent Water Masking:** One might hypothesize that since permanent water has a higher false alarm rate per pixel ($7.7\% - 9.0\%$), masking permanent water would resolve the problem. However, the data proves this is false: permanent water false alarms represent only $6.37\%$ of total false alarms. Perfect removal of all permanent water false alarms would leave $93.63\%$ of false alarms intact.
  - **Global Threshold Adjustments:** Raising the flood classification threshold globally to suppress land false alarms would disproportionately harm recall on true flood events (such as Event 559 and Event 1111008).
  - **Prior-Based Reweighting:** Without an online, unsupervised mechanism to reliably estimate event-specific class distributions from unlabeled SAR backscatter, applying assumed class priors will introduce severe bias on anomalous scenes.

### 6.3 Missing Information & Necessary Future Evidence
To justify any online class-aware or confidence-calibrated adaptation strategy, future work must establish:
1. **Feature-Space Separation:** Can low-backscatter dry terrain (smooth soil, sand, tarmac) be distinguished from standing water in SAR latent representations without supervised labels?
2. **Unsupervised Prevalence Estimation:** Can the model estimate scene-level flood prevalence reliably from unlabeled test batches prior to adaptation?
3. **Multi-Seed and Multi-Event Generalization:** All findings to date reflect `seed0` with a specific batch order on 6 development events. Independent validation across alternative seeds and sealed events is essential before drawing architectural conclusions.

---

## 7. Artifact Summary & SHA-256 Checksums

The diagnostic outputs and tabular breakdowns are persisted in `outputs/week14_offline_diagnosis/`:

| Artifact File | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `outputs/week14_offline_diagnosis/class_confusion_3x3.csv` | Full $3 \times 3$ confusion matrix counts ($C_{00} \dots C_{22}$) per event and pooled | `923563ac22d6bf92799a133ddf699930d8106b1cd2a0ea081ceb312bd17d9f27` |
| `outputs/week14_offline_diagnosis/error_breakdown.csv` | Comprehensive breakdown of FP/FN counts, shares, and rates with explicit denominators | `eabb024a539b9275cad004c16272561afd04656b33ff3295c12a0e93066d4a88` |
| `outputs/week14_offline_diagnosis/comparative_shifts.csv` | Relative shifts in F1 and error channels vs. `source_only` and `matched_bn_no_update_v1` | `8ea5117f5e54182bdf99cca8211a2aeac6d5eff7ab584aee30007b28c13b1bf8` |
| `outputs/week14_offline_diagnosis/artifact_sha256.txt` | Inventory of SHA-256 digests for all generated CSV tables | `a9a5bdb4ae8cac5b8f192bf18c76e6273e79524c44d9f381b103f88aa8b01e5d` |
