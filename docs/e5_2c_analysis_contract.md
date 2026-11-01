# ChangeGuard Plan V3: Work Package E5.2C Offline Temporal Risk Analysis Contract

**Governing Document:** [`docs/MASTER_EXECUTION_PLAN_V3.md`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/docs/MASTER_EXECUTION_PLAN_V3.md) (Section 10)  
**Configuration File:** [`configs/changeguard/e5_temporal_risk_analysis_v1.json`](file:///usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard/configs/changeguard/e5_temporal_risk_analysis_v1.json)  
**Source Dataset:** Formal E5.2B Collection (`outputs/e5_temporal_evidence/formal_s0_oa_v1/`)  
**Analysis Output Directory:** `outputs/e5_temporal_evidence/analysis_s0_oa_v1/`  
**Status:** LOCKED FOR IMPLEMENTATION & CPU VERIFICATION  

---

## 1. Executive Context & Methodological Boundary

Work Package **`E5.2C`** defines the deterministic offline evaluation protocol for the completed formal temporal-risk evidence collection (`e5_temporal_risk_s0_oa_v1`).

### 1.1 Epistemic Role of the Screening Cell
- **Primary Cell:** Seed 0, Stream Order A (`s0_oa_v1`) across the six development events (`[279, 437, 520, 559, 1111003, 1111008]`).
- **Descriptive Screening:** This evaluation serves as an exploratory and descriptive screening of pre-update risk indicators against observed adaptation outcomes.
- **No Population Overclaiming:** With $N = 6$ development events, statistical claims of general population superiority are strictly prohibited. Exact descriptive outcomes, per-event distributions, and grouped batch curves are reported without treating batches as independent events.

---

## 2. Outcome Definitions & Evaluation Units

### 2.1 Primary Outcome Unit: Event
The fundamental unit of adaptation decision-making is the **event**:
$$\Delta \text{F1}_{\text{event}} = \text{F1}_{\text{candidate}}(\text{event}) - \text{F1}_{\text{source}}(\text{event})$$

Where $\text{F1}$ is the Class-2 (flood) F1 score computed from accumulated confusion cells $(\text{TP}, \text{FP}, \text{FN})$ over all valid pixels in the event:
$$\text{F1} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

Every event is classified into one of three predeclared regimes:
- **Harmful Event:** $\Delta \text{F1}_{\text{event}} \le -0.010$ (loss of $\ge 1.0$ percentage points).
- **Beneficial Event:** $\Delta \text{F1}_{\text{event}} \ge +0.010$ (gain of $\ge 1.0$ percentage points).
- **Neutral Event:** $|\Delta \text{F1}_{\text{event}}| < 0.010$.

#### Isolation of Normalization Context
In addition to candidate vs source differences, the offline analysis evaluates the **matched BN-no-update** control:
$$\Delta \text{F1}_{\text{BN}} = \text{F1}_{\text{BN\_control}}(\text{event}) - \text{F1}_{\text{source}}(\text{event})$$
This prevents misattributing performance shifts caused purely by normalization statistics shifts ($\mu_{\text{batch}}, \sigma^2_{\text{batch}}$) to weight adaptation drift.

### 2.2 Secondary Outcome Unit: Grouped Batch
For fine-grained temporal resolution:
- Each batch $b \in \{1, \dots, 179\}$ produces a matched candidate and source Class-2 F1 on valid pixels.
- The signed change is $\Delta \text{F1}_{\text{batch}} = \text{F1}_{\text{cand}, b} - \text{F1}_{\text{src}, b}$.
- The batch is classified as harmful ($\le -0.01$), beneficial ($\ge +0.01$), or neutral ($(-0.01, +0.01)$).
- **Grouped Discipline:** Batches must **never** be treated as 179 independent independent samples. All statistical uncertainty, rank correlations, and distributions must preserve event clustering.

---

## 3. Frozen Signal Definitions & Risk Directions

All risk indicators are extracted strictly **`PRE_UPDATE`** before each candidate batch update. Their risk directions are frozen in advance and must not be altered post-hoc:

| Indicator Key | Signal Name | Family | Declared Risk Direction | Mathematical Definition |
| :--- | :--- | :--- | :--- | :--- |
| `mean_entropy` | Predictive Entropy | Generic | **Higher $\rightarrow$ Greater Risk** | $-\frac{1}{|\Omega|} \sum_{i} \sum_{c} p_{i, c} \log(p_{i, c} + \epsilon)$ |
| `mean_max_confidence` | Maximum Confidence | Generic | **Lower $\rightarrow$ Greater Risk** | $\frac{1}{|\Omega|} \sum_{i} \max_c p_{i, c}$ |
| `source_to_candidate_kl` | Source-to-Candidate KL | Generic | **Higher $\rightarrow$ Greater Risk** | $\frac{1}{|\Omega|} \sum_{i} \sum_{c} p_{\text{cand}, c} \log \frac{p_{\text{cand}, c} + \epsilon}{p_{\text{src}, c} + \epsilon}$ |
| `hard_disagreement_rate` | Hard Disagreement Rate | Generic | **Higher $\rightarrow$ Greater Risk** | $\frac{1}{|\Omega|} \sum_i \mathbb{I}(\hat{y}_{\text{cand}, i} \ne \hat{y}_{\text{src}, i})$ |
| `abs_delta_flood_pixels` | Absolute Flood Shift | Generic | **Higher $\rightarrow$ Greater Risk** | $|\sum_i \mathbb{I}(\hat{y}_{\text{cand}, i}=2) - \sum_i \mathbb{I}(\hat{y}_{\text{src}, i}=2)|$ |
| `aug_inconsistency` | Augmentation Inconsistency | Generic | **Higher $\rightarrow$ Greater Risk** | $\frac{1}{2|\Omega|} \sum_i \sum_c |p_{\text{weak}, c} - p_{\text{strong}, c}|$ |
| `bn_weight_displacement` | BN Weight Drift | Generic | **Higher $\rightarrow$ Greater Risk** | $\|\gamma_{\text{cand}} - \gamma_{\text{src}}\|_2$ |
| `s_null` | Temporal Null Risk | Change-Specific | **Higher $\rightarrow$ Greater Risk** | $0.5 \sum_{z \in \{\text{pre2/pre2}, \text{post/post}\}} \max(0, \Delta_{\text{null}}(z))$ |
| `s_flood_expansion` | Source Flood Expansion | Change-Specific | **Higher $\rightarrow$ Greater Risk** | $\text{mean}_{M_{\text{sensory}}} \max(0, p_{\text{cand}}(\text{flood}) - p_{\text{src}}(\text{flood}))$ |
| `s_flood_disappearance` | Source Flood Disappearance | Change-Specific | **Higher $\rightarrow$ Greater Risk** | $\text{mean}_{M_{\text{sensory}}} \max(0, p_{\text{src}}(\text{flood}) - p_{\text{cand}}(\text{flood}))$ |

*Convention for Receiver Operating Characteristic Analysis:* For signals where lower values indicate greater risk (e.g. `mean_max_confidence`), the score is negated ($- \text{signal}$) so that higher scores consistently denote higher risk for AUROC/AUPRC calculation.

---

## 4. Evaluative Metrics & Statistical Methodology

### 4.1 Discriminative Metrics
1. **AUROC (Area Under the ROC Curve):**
   - Evaluates rank-order ability to discriminate harmful cases from non-harmful cases (beneficial + neutral).
   - If an evaluation grouping has zero harmful cases or zero non-harmful cases, AUROC is mathematically degenerate and must be marked **`NA`** (`NaN`), never 0.5 or 1.0.
2. **AUPRC (Area Under the Precision-Recall Curve):**
   - Evaluates precision across detection thresholds.
   - **No-Skill Benchmark:** Every reported AUPRC must be benchmarked against the empirical harm prevalence $\pi = P / N$.
   - If degenerate, marked **`NA`** (`NaN`).
3. **Spearman Rank Correlation:**
   - Evaluates monotonic association between the risk score and signed performance change $\Delta \text{F1}$.
   - Expected sign is negative (higher risk score associated with worse $\Delta \text{F1}$).

### 4.2 Rejection Budget Analysis
To simulate deployment safety interventions without high-capacity classifiers:
- Rejection budgets: $b \in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]$ (fraction of highest-risk batches rejected).
- At each budget $b$:
  - Threshold $\tau_b$ is set to the $(1 - b)$-quantile of the risk score.
  - Batches with score $> \tau_b$ are rejected (fallback to source model).
  - Metrics tracked:
    1. **Harm Averted:** Number and percentage of harmful updates successfully blocked.
    2. **Lost Benefits (Collateral Harm):** Number of beneficial updates incorrectly rejected.
    3. **Net F1 Realized:** Effective event F1 when rejected batches fall back to source model predictions.

---

## 5. Generic vs Change-Specific Comparison Rules

1. **Individual Signal Parity:**
   - Every generic signal and every change-specific signal must be evaluated across identical batches, identical rejection budgets, and identical evaluation units.
2. **Composite Rule Boundary:**
   - Single-signal descriptive evaluation is fully authorized.
   - Multivariate combinations (e.g. generic + change-specific composite classifiers) are marked **`awaiting_predeclared_multivariate_calibration_rule`**.
   - No post-hoc weighting or high-capacity classifier may be trained on the six events to declare artificial victory.

---

## 6. Implementation Architecture & File Contracts

The offline analysis implementation consists of:
- **`tools/analyze_e5_temporal_risk.py`**: Deterministic analysis engine.
- **`tests/test_e5_temporal_risk_analysis.py`**: Unit test suite verifying mathematical correctness, threshold boundaries, and fail-closed integrity.

### Input Constraints & Immutability:
- Reads strictly from `outputs/e5_temporal_evidence/formal_s0_oa_v1/`.
- Never writes, modifies, or deletes any artifact in the source directory.
- Recomputes metrics from authoritative shards (`batch_XXXXXX.npz`) and verifies float16 equivalence within declared tolerances ($\le 5 \times 10^{-4}$).
- Output is written strictly to `outputs/e5_temporal_evidence/analysis_s0_oa_v1/`.

