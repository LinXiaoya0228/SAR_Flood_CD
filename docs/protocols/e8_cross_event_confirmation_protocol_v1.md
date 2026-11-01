# ChangeGuard Work Package E8: Cross-Event Confirmation Protocol

**Protocol Identifier:** `e8_cross_event_confirmation_protocol_v1`
**Created Date:** 2026-10-21
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
**Governing Documents:**
- `docs/MASTER_EXECUTION_PLAN_V3.md`, Block E8 (Sections 13.1–13.4)
- `docs/e6_final_route_decision.md` (`E6 COMPLETE — BRANCH C`)
- `docs/e8_closest_method_transfer_audit.md`
- `configs/changeguard/e8_execution_ledger_v1.json`
- `configs/changeguard/e8_confirmation_matrix_v1.json`

---

## 1. Governing Scientific Scope & Objective

Work Package E8 conducts the **cross-event evaluation and full confirmation** of test-time adaptation methods on multi-temporal SAR streams under **Branch C: Controlled Empirical Study**.

Under Branch C governance:
1. **Evidence-Supported Working Branch:** Branch C is the primary empirical track, established by Gate E findings confirming that pre-update unsupervised risk scores lack directional cross-seed stability.
2. **Reconsideration Mandate:** Branch C is not irreversible. E8 rigorously evaluates whether the closest task-specific methods (notably CoTTA and audited nearest baselines) provide stable, repeatable performance improvements across independent events, seeds, and stream orders to justify a versioned amendment to Branch B.
3. **No Fabricated Controller:** No selective gate, accept/reject controller, or heuristic decision rule will be invented or claimed under Branch C.

---

## 2. Confirmation Matrix Specification & Exact Cell Enumeration

### 2.1 Evaluated Methods & Roles

| Method | Role in E8 Matrix | Implementation / Provenance | Status in E8 | Physical Streams | Event-Result Rows |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Source-Only** | Static reference control | `E1SourceOnlyAdapter` (E1.4) | **REUSED** | 3 | 18 |
| **BN-No-Update** | Matched normalization control ($B=16$) | `E1MatchedBNNoUpdateAdapter` (E1.5) | **REUSED** | 6 | 36 |
| **AdaBN** | Streaming normalization baseline | `E1AdaBNAdapter` (E1.4) | **REUSED** | 12 | 72 |
| **Tent** | Streaming entropy minimization | `E1TentAdapter` (E1.4) | **REUSED** | 12 | 72 |
| **EATA** | Regularized active sample selection | `E1EATAAdapter` (E1.4) | **REUSED** | 12 | 72 |
| **SAR** | Sharpness-Aware and Reliable TTA | `E1SARAdapter` (E1.4) | **REUSED** | 12 | 72 |
| **CoTTA (Continual)** | Nearest task-relevant TTA (continual) | `E7CoTTAAdapter` (E7/E8) | **EXECUTE_NEW** | 6 | 36 |
| **CoTTA (Reset)** | Nearest task-relevant TTA (episodic) | `E7CoTTAAdapter` (E7/E8) | **EXECUTE_NEW** | 6 | 36 |
| **TOTAL** | **Primary Confirmation Matrix** | — | — | **69** | **414** |

### 2.2 Factorial Breakdown

For each applicable adaptive method:
- **Source Checkpoint Seeds:** 0, 1, 2 (trained to epoch 16 on the 27 source events).
- **Stream Orders:**
  - Order A: chronological forward order (`[279, 437, 520, 559, 1111003, 1111008]`), ascending sample IDs within events.
  - Order B: reversed event order (`[1111008, 1111003, 559, 520, 437, 279]`), descending sample IDs within events.
- **Reset Semantics:**
  - `continual`: Parameters/buffers accumulate across all 6 development events sequentially.
  - `reset_per_event`: Parameters/buffers/RNG reset to initial source checkpoint bit-exactly at the start of each event.
  - `invariant`: Stateless per-batch or static parameters (Source-Only: 3 streams; BN-No-Update: 6 streams).
- **Development Events:** Exactly 6 events ($N = 2,826$ samples, 179 physical batches of BS=16).
- **Sealed Test Quarantine:** Strictly 0 access to the 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`).

---

## 3. Nearest-Method Transfer & Exclusions

In accordance with `docs/e8_closest_method_transfer_audit.md`:
1. **PA-TTA (Huang et al., IEEE TGRS 2026):** Categorized as **`NOT DIRECTLY COMPARABLE IN THE PRIMARY ONLINE MATRIX — RETAIN AS A SEPARATE TRANSDUCTIVE TASK-TRANSFER CANDIDATE`**. Upstream code consumes pre-computed `sam_mask` boundary guidance, is structured around binary change detection (`output_nc=2`), and performs multi-epoch transductive iteration in `trainer.py`. This setup is not directly comparable to ChangeGuard's causal online single-pass `PRE_UPDATE` streaming protocol. A separate transductive task-transfer protocol may be scheduled after E8 if required for final publication claims; no open-source license is present in the upstream repository root.
2. **CIM-TTDA (Chen et al., JGR-MLC / PIERS 2026):** Categorized as **`NOT DIRECTLY COMPARABLE IN THE PRIMARY ONLINE MATRIX (SEPARATE TRANSDUCTIVE CANDIDATE) — NO OFFICIAL CODE LOCATED`**. No official implementation was located during this audit. Published as a multi-epoch event-transductive water segmentation method rather than bi-temporal change detection, it is preserved for literature discussion and potential transductive comparison. Its constituent teacher EMA and parameter restoration principles are represented in the online track by CoTTA.
3. **Audit Framing & Baseline Role:**
   - E8 tests **CoTTA** as the strongest currently executable, fully compatible generic test-time adaptation baseline under causal online streaming.
   - **PA-TTA and CIM-TTDA are not claimed to have been defeated.** Their exclusion from the primary online table is strictly a protocol-comparability decision, not evidence that they fail on SAR change detection.
   - A separate event-transductive comparison may be scheduled after E8 if necessary for the final publication claims.
4. **TTA-Monitor (Schirmer et al., NeurIPS 2025), NOTE, RoTTA, ASR:** Categorized as `LITERATURE DISCUSSION ONLY` with full technical justifications recorded in the transfer audit.

---

## 4. Branch-C Interpretation of Cross-Event Evaluation (LOEO)

Under Plan V3, Leave-One-Event-Out (LOEO) was originally formulated as a cross-event policy threshold calibration procedure (calibrating an accept/reject gate on 5 events and evaluating on the 6th).

Because Gate E established that pre-update unsupervised risk scores lack directional stability across source seeds, **no selective policy was approved**:
1. **Policy-Threshold LOEO Status:** Marked strictly as **`NA under Branch C`**. No fabricated policy gate or threshold will be simulated.
2. **Approved Cross-Event Empirical Analyses:**
   - **LOEO Jackknife Sensitivity:** Systematically recalculates all condition-level summary statistics (Macro F1, rank order, win/loss rates) holding out each of the 6 events in turn to verify that empirical conclusions do not depend on any single event.
   - **Clustered Event Bootstrap:** Computes 95% paired bootstrap confidence intervals over the 6 development events (clustered by event, not pseudo-replicated over pixels or batches) to honestly reflect the true small-event sample size ($K = 6$).
   - **Paired Cross-Event Contrast:** Evaluates the exact delta $\Delta \text{F1}$ of each adaptive method relative to Source-Only and Matched BN-No-Update across all 6 events.

---

## 5. Required Statistical Outputs & Reporting Standards

The E8 empirical analysis engine must compute and report:
1. **Accuracy Metrics:**
   - Event-macro F1, IoU, Precision, Recall (unweighted mean across the 6 events).
   - Pooled F1, IoU, Precision, Recall (micro-average over all 135,859,080 valid pixels across the 6 events).
2. **Extreme Risk Measures:**
   - Condition worst-event F1: $\min_{e \in \text{dev6}} \text{F1}(e)$ for each stream.
   - Global minimum F1: lowest single event F1 observed across all conditions.
3. **Harm / Benefit / Neutral Accounting:**
   - Harm count ($H$): $\Delta \text{F1} \le -0.0100$ vs reference.
   - Benefit count ($B$): $\Delta \text{F1} \ge +0.0100$ vs reference.
   - Neutral count ($N$): $|\Delta \text{F1}| < 0.0100$ vs reference.
   - Denominator: Stated explicitly as 6 per stream, 18 per seed-aggregate, 72 per adaptive method.
4. **Operational & Cost Metrics:**
   - Total updates applied, updates skipped, nonfinite rollbacks, stochastic restorations.
   - Wall-clock runtime per stream and average per batch.
   - Peak VRAM allocated.
5. **Factor Sensitivities:**
   - Order sensitivity: $\Delta_{\text{order}} = \text{F1}(\text{Order B}) - \text{F1}(\text{Order A})$.
   - Reset sensitivity: $\Delta_{\text{reset}} = \text{F1}(\text{Continual}) - \text{F1}(\text{Reset-per-event})$.
   - Seed sensitivity: Sample standard deviation across Seeds 0, 1, 2.

---

## 6. Predeclared Evidence-Based Route Reconsideration

Upon completion of the full 69-stream E8 matrix and analysis:

1. **Stay Branch C (Primary Route):**
   - Condition: If no closest method (including CoTTA) yields stable, statistically significant improvement over Matched BN-No-Update across events, seeds, and stream orders, or if severe worst-event degradation ($\Delta \text{F1} \le -0.05$) persists.
   - Action: Finalize E8 empirical findings, proceed to Block E9 ablations and E10 publication packaging under Branch C.

2. **Move by Versioned Amendment to Branch B (Action-Intervention Route):**
   - Condition: If a specific task-relevant component (e.g., SSR or Teacher EMA in CoTTA) demonstrates consistent, repeatable gains over Matched BN-No-Update across all 3 seeds and both orders without producing severe worst-event harm on any event.
   - Action: Draft formal Plan V3 amendment specifying the empirical mechanism for selective parameter update intervention.

3. **Return to Branch A (Autonomous Pre-Action Route):**
   - Condition: Only if a pre-action, label-free change-specific signal demonstrates incremental value beyond matched generic evidence in preventing adaptation harm. (Existing failure observations alone are insufficient).

