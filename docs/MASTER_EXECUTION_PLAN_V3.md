# ChangeGuard: Integrated Research and Execution Plan V3

Date: 2026-10-19. Time zone: Asia/Singapore.

Planning horizon: 2026-10-20 to 2027-05-19.

Suggested repository location: docs/MASTER_EXECUTION_PLAN_V3.md.

Project: NTU Final-Year Project on paired SAR flood change detection and failure-aware test-time adaptation.

Working title: Failure-Aware Test-Time Adaptation for SAR Flood Change Detection: Event-Wise Characterization and Selective Adaptation.

## 1. The decision this version implements

November 8 is a possible change in available AI tools, not a deadline for scientific completion. Research, coding, debugging and experiments can continue with VS Code Codex after that date. No experiment, confirmation requirement or comparison is removed merely to reach the old "Week 20 by November 8" milestone.

This document combines the original complete execution plan with the later methodological corrections and the experiments already completed. It replaces their conflicting FUTURE schedules. It does not modify historical protocols, data splits, code versions, results or review records.

Use one active future-plan entry point in the repository. Keep V1/V2 as historical references and link them to V3 through a short adoption note. Existing frozen experiment protocols still govern their historical runs; V3 does not retroactively change their settings.

The original Week numbers remain reference labels for topics. Future scheduling uses actual dates and execution blocks E0-E11. "Week 13 exploratory run completed" does not mean that every anchor comparison in the original proposal has been completed.

The aim is to finish a defensible research cycle before the FYP deadlines: method development, fair comparison, confirmation, ablation, robustness, a frozen final evaluation and a complete report. It is not to complete the largest possible number of stage labels before a subscription changes.

All new repository documentation, protocols, manifests, experiment logs, terminal status messages, code comments and commit messages must be English. Preserve original historical evidence rather than translating it in place.

### 1.1 Source documents and evidence boundary

This plan reconciles:

1. Original ChangeGuard master execution plan V1, supplied on 2026-10-19. Source attachment SHA-256:
   A19837640A89D2B7C3690CD24BAD6450ED4203B63BE7C4D51487CB055355BB2D.
2. Accelerated research proposal V2, dated 2026-10-18. Source attachment SHA-256:
   D2D25CF81182E2971CB2FE67857456918D251C48CFD26CA084A8BC998692045C.
3. User-supplied execution and analysis records through reported commit 0bd5d32.

The present author has read those documents and execution summaries, but has not directly audited every file on gpu22. Reported remote completion is distinguished from a fresh direct verification.

This is a planning document. Exact future configs, dataset identities, formulas and launch matrices must be fixed in executable protocols before their runs. A plan does not itself prove that a future experiment ran.

### 1.2 What is retained from each version

From V1:

- controlled source training and its three checkpoints;
- strong TTA baselines and matched evaluation;
- confidence-threshold and augmentation comparisons;
- probability, logit and feature anchor questions;
- class balance, flood-aware selection, permanent-water protection and change preservation;
- temporal evidence, controller feasibility, cross-event validation;
- three-seed, multi-order confirmation, ablation, resource measurement and robustness;
- final test, reporting, demonstration, oral presentation and submission dates.

From V2:

- corrected novelty and evidence claims;
- strict source/development/test boundaries;
- explicit U/C calibration definitions;
- honest separation of exploration and confirmation;
- comparisons with generic risk/gates and reject-all;
- state-neutral probes and complete rollback semantics;
- matched acceptance/calibration budgets;
- evidence-based method promotion and legitimate negative outcomes;
- preservation of failed attempts and independent backups.

The V2 November 8 scope cuts are superseded. Later experiments are not automatically cancelled because Antigravity becomes unavailable.

### 1.3 Completion and success are separate

An experiment is complete when its declared matrix ran or its precise non-execution reason is documented, outputs are verified, and results are archived. A method is successful only when its claims are supported.

The original aspiration of approximately 1-2 percentage points of macro-F1 improvement, or meaningful worst-event improvement without an average loss, remains a method-promotion target. It is not a condition under which negative results may be hidden or a guarantee that an anchor must improve F1.

A required experiment cannot silently become "optional" after a disappointing result. A scientifically invalid transform or a method outside the information setting may be excluded with a written applicability argument. Such an item is recorded as excluded, not experimentally completed.

### 1.4 Amendment: Strict Sequential Execution and Single Active Block (2026-10-19)

As of 2026-10-19, the execution discipline is formally amended to enforce strict sequential execution across work packages:

1. **Strict Sequential Order:** Execution proceeds strictly from E0, to E1, through E11.
2. **Single Active Block:** At any given time, exactly one E block is active. A successor work package remains inactive until the current block's exit conditions and verification criteria are fully satisfied.
3. **Subtask Numbering:** Subtasks within each block use hierarchical numbering (e.g., E1.1, E1.2, E2.1).
4. **Historical Reuse Boundary:** Completed historical experiments (Weeks 0–14) remain cataloged in the reuse ledger for later comparative reuse, but their existence does not cause multiple future blocks to be considered simultaneously active.
5. **Forecast Nature of Calendar Windows:** Calendar windows are planning forecasts and schedule buffers. They do not permit skipping requirements, bypassing exit conditions, or declaring an incomplete block closed.

## 2. Actual position at the start of V3

### 2.1 Existing assets

| Asset | Reported state | Future treatment |
| --- | --- | --- |
| Official-checkpoint pilot | Completed; input provenance remains incomplete | Historical motivation and secondary reference |
| Source split and source training pipeline | Frozen with recorded corrections | Reuse; do not restart source training by default |
| VV/VH versus VV/VH/slope source comparison | Three-seed study reported | Preserve; do not assert slope is the official third channel |
| All-27 primary source models | Seeds 0/1/2, fixed 16 epochs, FP32 | Reuse all three; do not select the best development seed |
| Week 8/9 baseline studies | Source, AdaBN, Tent, EATA, SAR; accepted historical analyses | Reuse only where protocol matching is demonstrable |
| V2 EATA Fishers | Correct 21,462-sample source population; 2,000 selected samples | Reuse with checkpoint/selection binding |
| Week 11 implementation and real smoke | CPU tests and Event-520 head50 smoke completed | Engineering foundation; final infrastructure self-verified |
| Week 12 initial comparison | Source/Tent/C1/C2, seed0, six development events | Completed exploratory comparison |
| Matched BN no-update diagnostic | Seed0, six development events completed | Completed exploratory normalization diagnostic |
| Week 13 probability anchor | Lambda 0.50, seed0, six development events completed | One completed anchor experiment, not the full V1 anchor family |
| Week 14 class-error diagnosis | Cached-prediction offline diagnosis reported complete | Reuse; class-aware intervention experiments remain open |
| Interim report/video | Drafts exist; submission not established | Finish and submit by November 10 |
| Final sealed test | No authorized access reported | Remains sealed |

Reported milestone commits:

- e5cbbfc: Week 9 execution implementation.
- 9d45e4e: Week 11 smoke implementation/config checkpoint.
- 0a8b63c: Week 11 closeout and Week 12 initial ablation.
- 197fd31 and its predecessor: BN diagnostic implementation/config corrections.
- 5a217d8: Week 13 source-anchor implementation.
- 0bd5d32: Week 12/13 closeout and Week 14 offline diagnosis.

Exact full commit IDs and artifact identities must be resolved from the remote repository when preparing a run. Do not infer the current remote HEAD solely from this table.

### 2.2 What current results establish

Reported seed0, order_a, reset-per-event comparison:

| Condition | Event-macro Flood F1 | Pooled Flood F1 |
| --- | ---: | ---: |
| Source with frozen training BN statistics | 0.594880 | 0.519331 |
| Current-batch BN, no gradient update | 0.573298 | 0.441744 |
| Matched PRE_UPDATE Tent | 0.574088 | 0.446873 |
| C1 consistency, no confidence filter | 0.573270 | 0.441606 |
| C2 consistency, confidence 0.90 | 0.573318 | 0.441912 |
| C3 probability anchor, lambda 0.50 | 0.573441 | 0.442698 |

These comparisons show that:

- the current tested consistency and anchor settings have not beaten source;
- confidence filtering changed selected pixels without a material improvement;
- changing the BN prediction context without gradient updates reproduces similar aggregate deterioration;
- some events benefit and others deteriorate;
- the source anchor produced a small measured change, not a demonstrated main-method improvement.

They do not establish:

- an exact percentage of degradation caused by normalization;
- small parameter drift merely because class predictions barely changed;
- the failure of all anchor weights, augmentation families or adaptation methods;
- a working risk detector or selective controller;
- cross-seed or sealed-event generalization of the new components.

The historical multi-seed Week 9 table and this seed0 table have different averaging scopes and may have different prediction-timing details. Do not concatenate them into a supposedly matched table.

### 2.3 Open work that must not be hidden by stage labels

| Topic | Already available | Still required |
| --- | --- | --- |
| Confidence | No filter versus 0.90 | Finite threshold sensitivity, class-wise selection and repetition |
| Augmentation | Identity weak plus aligned horizontal-flip strong | Meaningful weak/strong comparison and other validated transforms |
| Anchor | One probability-anchor weight | Probability sensitivity; logit and feature comparisons |
| Class-aware design | Three-class error decomposition | Bounded class-aware intervention comparisons |
| Baselines | Historical EATA/SAR and matched seed0 Tent | Explicit compatibility ledger and new measurements where matching/cache fields are missing |
| BN mechanism | Matched no-update seed0 comparison | Repeated-seed/order checks and normalization controls |
| Change-related evidence | Probe interfaces and some logged scores | Timing-correct discriminative and incremental-value evaluation |
| Controller | State-management foundations | Evidence-supported policy, runtime tests and actual trajectories |
| Confirmation | Three source checkpoints and historical baseline matrix | New candidate/control confirmation across all three seeds |
| Robustness | Main SNUNet setting | Planned second-backbone and physically valid shift evaluation |
| Final evaluation | Reserved ten events | Frozen, approved March evaluation |

## 3. Fixed research and data contract

### 3.1 Questions

RQ1: Under which event, checkpoint, order, normalization and reset conditions does TTA help or harm?

RQ2: Does paired-change evidence add useful information beyond entropy, confidence, source disagreement, normalization context and parameter drift?

RQ3: Can that additional information improve an actual label-free policy while retaining beneficial adaptation?

RQ4: Do the findings persist across seeds, orders, a second architecture or shift, and final reserved events?

The tentative title may change after these questions are answered. A controller is not declared successful because it exists in code.

### 3.2 Data and checkpoints

- Source events: 27.
- Source training corpus: KuroV2_grid_dict.gz, 21,462 samples across those events.
- Development events: 279, 437, 520, 559, 1111003, 1111008.
- Development samples: 2,381; 169; 92; 42; 19; 123, respectively; total 2,826.
- Evaluation manifest: KuroV2_grid_dict_test_0_100.gz.
- Reserved test events: 205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013.
- Classes: 0 No Water, 1 Permanent Water, 2 Flood. Label 3 is invalid.
- Primary source: SNUNet_ECAM, VV/VH per date, pre2/post, three classes, FP32.
- Source-inner recipe selection has already occurred on source data. The final all-27 models use 16 epochs and seeds 0/1/2.
- Frozen source preprocessing/loss/training settings remain authoritative. Do not replace 16 epochs with 150 epochs merely because the historical planning document discussed the upstream setting.

The official checkpoint remains a provenance-limited reference. There is no new task to repeatedly reconstruct its third channel.

### 3.3 Label use and calibration

Source labels remain permitted for source training. During adaptation, probing and decisions, inputs are SAR, sensory validity and model state only. GT and label-derived masks cannot enter those paths.

Offline GT is read only after the corresponding predictions, risk scores and actions have been frozen and persisted. Offline exclusion uses sensory validity and label != 3.

U is the primary calibration track: numerical gate thresholds use source or explicitly permitted unlabeled calibration data, not development GT. The proposed-method design is development-informed and must be disclosed as such.

C is an optional, separately declared comparison: thresholds or risk parameters are selected using the other five development events' labels, then evaluated on the held-out event. It is described as development-label-calibrated and label-free at deployment.

The threshold sensitivity grid may be evaluated offline in U. Selecting its best threshold by development F1 and then claiming GT-free threshold calibration is not permitted. Retain the predeclared primary threshold, or disclose that selection under C.

No result from a development event is called final independent test evidence. LOEO cannot undo the fact that these six events have already informed project design.

### 3.4 Timing and normalization

Preserve the PRE_UPDATE primary prediction convention of the current matched studies.

The new recorder must also save POST_UPDATE diagnostics wherever an update is attempted. Every field has a phase, model state, normalization context and sample/batch identity.

Keep these distinct:

1. Effect of changing inference normalization with identical weights.
2. Effect of parameter updates under an unchanged normalization context.
3. Effect of choosing a different output branch.
4. Effect of accepting or rejecting updates on future model state.

A no-update current-batch BN model is not equivalent to frozen-source inference. Weight rollback alone does not restore source predictions if normalization behavior differs.

A score computed after an update can monitor a proposed/current state. It does not establish advance warning. An event score calculated after processing the entire event is retrospective and cannot decide how the already processed event was predicted.

### 3.5 Shared factors for confirmation

- Checkpoint seeds: 0, 1, 2.
- Orders: the exact frozen order_a and order_b definitions, including event and within-event sample sequences.
- Reset: reset_per_event and continual for each method claiming both.
- Batch size: 16, natural tails, unless an explicit secondary sensitivity protocol states otherwise.
- Same sample IDs, invalid-mask semantics, preprocessing, initial checkpoint, prediction timing and metric definitions across a matched comparison.
- Same evidence/calibration information for generic and change-related gates.

Seed0/order_a/reset-per-event remains the screening cell. Confirmation uses the complete declared factor set, not just the best seed.

## 4. One chronology from now to submission

Dates are planning windows, not permissions to declare incomplete work complete. A successor can start early when dependencies and evidence requirements are satisfied. November 8 does not freeze unfinished code. If work slips, use the stated buffers and revise the forecast openly.

| Calendar window | Main execution block | Tangible output and exit condition |
| --- | --- | --- |
| Oct 20-22 | E0: scope, matching and measurement contract | Requirement ledger, matched-reuse ledger, cache schema and future matrix fixed |
| Oct 23-25 | E1a: recorder and normalization checks | New recorder tested; one real bounded smoke; existing results preserved |
| Oct 26-Nov 1 | E1b: baseline and BN coverage; E2 begins | Matched controls assembled; BN finding checked beyond seed0; confidence/augmentation screen starts |
| Nov 2-8 | E2/E3: complete component implementation and screening | Confidence/augmentation matrix and anchor comparison progressing or completed; exact status and executable handoff |
| Nov 8 | Tool-availability checkpoint | Codex can locate code, verify a config, rebuild a table and continue a task; no research completion claim required |
| Nov 9-10 | Interim submission protection | Report/video final checks, actual submission and receipt |
| Nov 11-15 | E3 finish and transition to E4 | Complete any outstanding anchor cells; source-error and benefit-retention analysis |
| Nov 16-22 | E4: class-aware interventions | Four bounded intervention questions assessed, with matched comparisons |
| Nov 23-29 | E5: temporal and generic risk evidence | Timing-correct evidence screen and repeat-seed/order measurements |
| Nov 30-Dec 6 | E6: feasibility decision and method definition | Gate E decision; one method or empirical branch defined; closest-baseline applicability settled |
| Dec 7-13 | E7: policy implementation or empirical-study implementation | State/prediction semantics tested; real smoke; LOEO/confirmation scripts ready |
| Dec 14-20 | E8a: cross-event evaluation and confirmation starts | LOEO results, first multi-seed confirmation, December 20 conference readiness review |
| Dec 21-Jan 3 | E8b/E9: full confirmation and required ablations | Three seeds, two orders, declared reset modes, main table, uncertainty and ablations complete |
| Jan 3 | Final IGARSS submission decision | Complete scoped paper and stable evidence, or recorded NO-GO |
| Jan 4-8 | IGARSS submission buffer if GO; E10a begins | Author approval/submission; second-backbone and shift preparation starts |
| Jan 9-11 | Official IGARSS deadline Jan 11; E10a continues | No research/test shortcut; source-only robustness training proceeds |
| Jan 12-17 | E10a: robustness implementation/source training | Second-backbone source recipe selected using source only; shift protocol fixed |
| Jan 18-31 | E10b: robustness experiments | Second-backbone and shift comparisons, failure cases and resource analysis |
| Feb 1-7 | Remaining declared validation gaps | Required matrix cells and uncertainty analysis complete; no open correctness blocker |
| Feb 8-14 | Final method and paper-scope lock | Code/config/thresholds/claims frozen; no open-ended method search |
| Feb 15-21 | E11a: final-test rehearsal | Complete English paper draft with clearly pending final-test results; FYP non-test sections complete |
| Feb 22-28 | E11b: Gate T | Approved final evaluation matrix, scoped commit, backup and supervisor/user release |
| Mar 1-7 | Final reserved-event evaluation | Ten events, frozen methods, complete predictions/counts/manifests; no tuning |
| Mar 8-14 | Final analysis | Final tables, uncertainty, error figures and claim-to-evidence checks |
| Mar 15-21 | Complete FYP draft and paper | All chapters/results integrated; complete FYP draft and complete paper manuscript |
| Mar 22-24 | Draft review buffer | Numerical, citation, structure and formatting checks |
| Mar 25 | Draft Final Report deadline | Actual submission and receipt |
| Mar 26-Apr 4 | Final report revisions | Feedback or self-review addressed; no test-driven method redesign |
| Apr 5-8 | Final report proofing buffer | Final PDF, appendix, references and archived code/results |
| Apr 9 | Final Report deadline | Actual submission and receipt |
| Apr 10-11 | Demonstration rehearsal | Short inference example, static backup and offline rehearsal ready |
| Apr 12-16 | Project Demonstration | Working short demonstration and static offline backup |
| Apr 19-May 2 | Paper revision/extension and oral slides | Revise the existing manuscript for the selected venue; slides and questions/answers |
| May 3-9 | Oral rehearsal | Three complete timed rehearsals |
| May 10-12 | Oral Presentation | Presentation delivered; actual feedback recorded |
| May 13-18 | Repository/library-version preparation | Final corrections, copyright/permissions and archive checks |
| May 19 | DR-NTU full text deadline | Full text submitted and receipt archived |

The external FYP dates above come from the timetable supplied by the user. Check subsequent official departmental notices; do not assume a planning document proves a submission or approval occurred.

## 5. E0: close the scope and measurement gaps first

Window: Oct 20-22. Dependencies: existing reports and repository metadata.

### 5.1 Tasks

1. Resolve the actual current HEAD and authoritative report versions.
2. Map every V1/V2 requirement to completed evidence, future experiment ID, conditional decision or explicit exclusion.
3. Inspect actual historical and new config fields for reuse compatibility.
4. Inventory cached predictions, probabilities, risk scores, states and missing fields.
5. Define the shared recorder schema before expanding the experiment matrix.
6. Record primary metrics, U/C calibration scope, screening grids, factor counts and selection rules.
7. Verify independent backups of source checkpoints, Fisher artifacts, raw results and critical code.

Check the approved Week 5/6 slope scope explicitly. The completed source-inner A/B study does not automatically prove an all-27 slope checkpoint exists. If the frozen scope or intended channel-generalization claim requires that checkpoint and it is absent, register the single missing training/evaluation task; do not restart the completed A/B study or silently mark the requirement satisfied.

### 5.2 Required outputs

- docs/MASTER_EXECUTION_PLAN_V3.md: this plan, adopted into the repository without altering previous versions.
- docs/REQUIREMENT_COVERAGE.md: requirement -> evidence or future experiment.
- docs/MATCHED_REUSE_LEDGER.md: one entry per reused method/condition.
- docs/CACHE_AND_MEASUREMENT_CONTRACT.md: mandatory fields, timing and storage precision.
- docs/EXPERIMENT_REGISTRY.md: all scheduled cells with status.
- docs/CURRENT_STATUS.md: actual current task and next three actions.

These files have distinct purposes. Avoid copying full result tables into all of them.

### 5.3 Reuse decision

Reuse a historical cell only when checkpoint, sample set/order, batch/tails, preprocessing, sensory validity, prediction timing, reset and algorithm settings match.

Missing metadata produces REUSE_UNVERIFIED, not an automatic PASS. A necessary new measurement replay gets a new run ID and a reason. It does not overwrite or invalidate a valid old result.

Historical Week 9 missing pooled confusion counts remain a historical limitation. New matching runs export raw counts; do not reverse-engineer or invent absent old counts.

### 5.4 Exit

The outstanding confidence, augmentation, anchor, class-aware and confirmation cells are enumerated. No item disappears because November 8 is close. No blanket historical re-audit or source retraining is required.

## 6. E1: recorder, matched baselines and normalization confirmation

Window: Oct 23-Nov 1. Dependencies: E0 exit condition verified.

### 6.1 Subtask E1.1: Extended Recorder Implementation & Cache Contract

For every new scientifically relevant run, preserve:

- exact source checkpoint/config/code/environment identities;
- event/sample IDs, order, batch index and tail size;
- prediction-source branch and normalization context;
- sensory-valid and offline-evaluated pixel counts;
- PRE_UPDATE primary class predictions;
- POST_UPDATE class predictions for update-effect diagnostics;
- per-sample/batch probability summaries for all declared generic and change-related scores;
- source/candidate class-2 probability maps where spatial preservation analyses require them;
- both null-pair source/candidate probabilities or sufficient predeclared sums/counts;
- selected-pixel counts per predicted class and update/skip/failure/action counts;
- loss components, gradient norm and BN affine displacement when parameter-update claims are made;
- per-batch/event 3x3 confusion counts, TP/FP/FN/TN and prediction class counts;
- elapsed time, forward/probe counts and peak VRAM;
- durable initial/progress/failure/completion artifacts.

A hard class map cannot reconstruct entropy, KL or class probabilities. A scalar loss cannot reconstruct gradient norms or parameter displacement.

Calculate scores in declared precision. If probability maps are compressed or stored at reduced precision, save the original score summaries and document precision effects. Do not change risk formulas later and pretend an incompatible cache supports them.

Estimate storage before a broad run. Source maps may be shared by checkpoint/sample identity, but pointers must resolve to immutable archived content. Large files stay outside Git with hashes and verified backup.

### 6.2 Subtask E1.2: Integration and Synthetic Orchestration

Integrate the extended recorder into the runner and adapter orchestration pipeline. For the primary PRE_UPDATE study:

- score current candidate state before the update for an online decision about its current output;
- keep post-update probe scores as explicitly post-update diagnostics;
- evaluate update-effect labels only with corresponding pre/post predictions on the same batch;
- never assign an event harm label to every batch and call it per-update ground truth.

Verify the integrated runner end-to-end under synthetic CPU fixtures with mock data before launching any real execution.

### 6.3 Subtask E1.3: Bounded Real Smoke

Execute exactly one real bounded smoke on Event 520 (head 50 samples) under `vfm_env` using logical `cuda:0`. Verify durable writing of `step_records.json`, `frozen_predictions.npz` (with both PRE and POST predictions and float16 probabilities), and `event_metrics.json` without data loss or corruption.

### 6.4 Subtask E1.4: Matched Baseline Coverage

Required methods: source-only, matched BN-no-update, AdaBN, Tent, EATA and SAR.

For methods used in the final matched claims, cover seeds 0/1/2, order_a/order_b and both declared reset modes. A full adaptive method has 12 streams and 72 event-result rows. Source/no-update redundancies may be reused only after invariance/matching is demonstrated.

Historical cells can satisfy these requirements through the reuse ledger. Re-execute only genuinely mismatched cells or cells needing unavailable measurements for the specific new claim under the unified PRE_UPDATE protocol.

Keep the original EATA d_margin=0.05 baseline unchanged. A separately source-calibrated EATA variant, if justified, receives a new identity and the same fair calibration budget; do not silently repair its selection rate to favor a narrative.

### 6.5 Subtask E1.5: Normalization Confirmation & Sensitivity

Repeat the matched BN-no-update contrast beyond seed0. Report current-batch versus frozen-training-statistics behavior for all three checkpoints, matched orders and tails.

Include a frozen-training-statistics, zero-update reference and current-batch, zero-update reference in all interpretation. A frozen-statistics adaptation variant is a distinct algorithm and must be named separately if tested.

Analyze first-batch differences, per-event variation, batch-size sensitivity and full-stream changes. Do not give an additive causal percentage for a nonlinear metric.

Batch-size sensitivity is a bounded diagnostic at 1, 4 and 16 for the BN-no-update control, with the same sample coverage. Treat it as sensitivity, not as a reason to quietly change the primary batch16 benchmark.

### 6.6 Subtask E1.6: Analysis and Closeout

The recorder has focused tests and a real permitted smoke. Matching/reuse is explicit. The normalization finding is either repeated or its limits documented. New substantive claims are not supported solely by the seed0 screen. All artifacts and summary tables verified before proceeding to E2.

## 7. E2: restore the complete confidence and augmentation comparison

Window: Oct 26-Nov 8; finish later if needed without removing cells.

Dependencies: stable E1 recorder and matched controls.

### 7.1 Confidence grid

Use the same consistency objective and augmentation:

- no confidence filtering: all sensory-valid pixels;
- threshold 0.70;
- threshold 0.90, retaining the current predefined primary reference;
- threshold 0.95.

Existing C1 and C2 cells are reused if matched. No extra threshold is introduced after inspecting the grid.

Measure selected-pixel fraction per predicted class, number of updated batches, loss/gradient/update magnitude, class distributions, F1/recall/precision, runtime and memory.

Do not equate fewer selected pixels with fewer forwards or reliable compute savings.

### 7.2 Augmentation grid

At the predefined confidence reference:

| ID | Comparison | Required handling |
| --- | --- | --- |
| A0 | Current identity-weak / horizontal-flip-strong | Existing reference |
| A1 | Paired vertical flip | Invert outputs and validity into original coordinates |
| A2 | Small paired integer translation | Fixed displacement budget, overlap-only consistency mask, no label-dependent handling |
| A3 | Independent weak geometric views versus the declared stronger view | Make the weak-only comparison mathematically meaningful |
| A4 | One physically defensible SAR intensity/noise perturbation | Certify linear/dB/normalized input domain and order of operations first |

Identity compared with the identical deterministic prediction gives zero consistency loss; it is a sanity control, not evidence that weak augmentation fails.

For A4, derive the permitted perturbation from verified preprocessing/source statistics. Do not multiply dB values by synthetic speckle or use RGB color jitter. Select one mild fixed magnitude before development scoring. If the input-domain/semantic assumption cannot be justified, keep A4 blocked with its exact reason; do not implement an invalid transform to fill a table.

Paired geometry is synchronized across both dates. Strong outputs and masks return to the weak/original coordinate frame before loss. Crops or translations use declared overlap; no hidden GT validity.

### 7.3 Screening and repeat

Screen seed0/order_a/reset-per-event across all six development events. This is a one-factor comparison, not the full Cartesian product of thresholds and transforms.

Repeat the retained configuration and its decisive component contrast on seed1 and the other declared order before promotion. The final main comparison later covers all three seeds.

If a threshold is selected by development labels, declare that selection and calibration scope. Do not quietly call it source-only U calibration.

### 7.4 Exit

All confidence cells and valid augmentation questions are accounted for. Choose at most one main consistency recipe through the recorded rule, or record that none improves the controls. Preserve the full grid and all negative cells.

## 8. E3: the full bounded source-anchor comparison

Window: Nov 2-15, with Nov 9-10 protected for the interim submission.

Dependencies: fixed base consistency recipe and source branch.

### 8.1 Required families

1. Frozen-source probability KL anchor.
2. Frozen-source logit anchor with an explicit shift/scale convention.
3. One selected source-feature anchor with a fixed layer, geometry mapping and normalization.

Use a detached immutable source branch. Do not add learned projection heads or extra teachers merely to make the feature anchor fit.

For each family, use the predefined finite weights 0.10, 0.50 and 1.00. This is nine seed0 screening cells, with reuse of the existing probability/0.50 cell only where the base protocol matches exactly.

Equal numeric lambda values across different losses do not mean equal gradient strength. Record raw loss, weighted loss and gradient/update measures; any normalization/scaling is declared before running that family.

### 8.2 Controls and interpretation

Matched controls: source, BN-no-update, base consistency, Tent, EATA and SAR where information/protocol conditions match.

An initial single-cell anchor screen is not the full original anchor study. The generic-protection comparison is explicitly completed or recorded as absent; historical mismatches are not hidden.

Measure improvements, deterioration, source-disagreement changes and beneficial-adaptation losses. Source agreement is not correctness. A large anchor can approach no adaptation without detecting risk.

The scalar share of anchor loss is not its share of the gradient. Positive source-anchor loss can exist before any parameter update because source and candidate normalization contexts differ.

### 8.3 Exit

All three families and their bounded weight cells are assessed, with a repeat of any promoted family on another seed/order. A family may be rejected by negative evidence. Claims remain conditional on the tested definitions.

The original objective "an anchor reduces harm" is a research success target. If none does, report that outcome and proceed to the risk-evidence question without claiming successful anchoring.

## 9. E4: class-aware intervention experiments

Window: Nov 16-22. Dependencies: existing three-class diagnosis and a fixed base method.

The offline diagnosis is complete; the intervention study is not. The predominance of land-to-flood false positives motivates care, but does not by itself prove that every permanent-water-aware method is ineffective.

### 9.1 Required bounded conditions

| ID | Question | Implementation boundary |
| --- | --- | --- |
| K1 | Does class-balanced confidence selection help? | Select using predicted classes/confidence only; record per-class quotas and empty classes |
| K2 | Does flood-aware loss sampling help? | Sample or weight pixels inside the adaptation loss; preserve prediction coverage and primary stream order |
| K3 | Does source-predicted permanent-water protection help? | Use frozen-source class-1 evidence, not GT permanent-water masks |
| K4 | Does source-predicted flood/change preservation help? | Use source probabilities/uncertainty; measure both erasure and expansion |

Assess these separately before combining them. Freeze one simple definition and one initial setting per condition. Reuse source calibration for class quotas/weights where needed. No event-specific tuning.

Run seed0/order_a/reset-per-event on all six development events. Repeat a promoted intervention and its decisive control on another seed/order before including it in a main method.

Compare with the fixed unmodified base, source, BN-no-update and appropriate generic protection. Choosing a different base is recorded before the intervention results are inspected.

### 9.2 Required analysis

- False positives and false negatives by true class, with denominators.
- Per-class prediction and selection counts.
- Low-/high-flood and permanent-water-heavy descriptions used offline only.
- Whether source errors are being preserved.
- Lost beneficial corrections and source-fallback frequency.
- Runtime, memory and numerical health.

Do not describe the non-water class as physically low-backscatter terrain without corresponding measurements. Class imbalance and speckle remain hypotheses unless explicitly tested.

### 9.3 Exit

All four original class-aware questions have an experimental result or an explicit applicability exclusion. The existing offline diagnosis alone is not marked as completion of these intervention cells.

Promote only supported components. If all fail, retain the diagnosis and negative comparisons; do not add a fifth class-aware method simply to obtain a positive result.

## 10. E5: temporal evidence and generic-risk comparison

Window: Nov 23-29. Dependencies: E1 phase-correct recording; a frozen candidate and no-update controls.

### 10.1 Required inputs and controls

- Real pair: pre2/post.
- Identical-time pairs: pre2/pre2 and post/post.
- Frozen-source and current-candidate predictions/probabilities for each.
- A BN-no-update probe control to identify evidence explained by normalization context.
- Generic entropy, confidence, source KL/disagreement, predicted flood-area change, augmentation inconsistency and measured BN affine displacement.
- At most two main change-related families: null evidence and source-referenced change disappearance/expansion.

Primary null score:

    s_null = 0.5 * sum_z max(0, mean_valid p_candidate(flood | z)
                               - mean_valid p_source(flood | z))
    z in {pre2/pre2, post/post}

Higher is the predeclared risk direction. Zero-valid probes are unavailable, not automatically safe.

Retain the existing source/candidate context definition and add a declared matched-context/no-update diagnostic where necessary. Do not interpret a source-adjusted null score as pure weight drift when normalization differs.

Natural pre1/pre2 is an auxiliary question. If used, collect label-free behavior and validity/temporal metadata; never assign synthetic all-zero GT.

### 10.2 Units and outcomes

Primary scientific characterization is event-level. Describe at least the seed0 and one repeat-seed/order result.

For deployment policy feasibility, also assess batch-level decisions using scores available before that decision. Batch outcomes come from the corresponding frozen predictions versus the matched source, with GT added offline.

An update-effect study uses same-batch pre/post predictions and the declared timing. It must not be confused with the effect of the current normalization context or with advance forecasting.

### 10.3 Analysis

- AUROC, AUPRC and harm-prevalence/no-skill references.
- Signed performance differences and rank relationships.
- Event-balanced results and within-event behavior.
- Group-aware uncertainty, class-degenerate metrics recorded as NA.
- Generic-only versus generic-plus-change evidence.
- Comparable calibration/acceptance budgets.
- Compute, probe counts, runtime, peak VRAM and unavailable scores.
- Source-error-driven false alarms and rejection of beneficial changes.

Use harm <= -0.01 F1 and benefit >= +0.01 F1 for the primary event characterization, preserving historical definitions. Any sensitivity band is separate.

Do not treat hundreds of batches from six events as hundreds of independent events. Do not reverse a risk score's direction after evaluating the repeat cell and report that as predeclared success.

### 10.4 Exit

Every primary score has a formula, timing and recorded scope. Its result is positive, negative or inconclusive. The question is incremental risk information, not whether a null-pair implementation exists.

## 11. E6: feasibility decision and nearest-method completeness

Window: Nov 30-Dec 6. Dependencies: E2-E5 evidence.

### 11.1 Gate E

Review whether:

- harm and benefit coexist in relevant conditions;
- a usable score improves on no-skill and reasonable generic evidence;
- direction survives at least a second seed/order;
- incremental value remains with comparable calibration/acceptance;
- the signal is available before the declared action;
- overhead and false rejection are acceptable for the proposed claim.

Six development events impose substantial uncertainty. "Inconclusive" is a valid decision and is not relabeled PASS.

### 11.2 Method branch

A: change-related evidence has incremental value. Build the smallest policy using that evidence and matched generic controls.

B: a stable adaptation/normalization/anchor component improves fair controls, but change evidence does not. Develop that stable-adaptation claim and remove the unsupported temporal-risk contribution.

C: no method improves reasonable controls. Complete an empirical study with repeated normalization/seed/event evidence, strong baselines, robustness and final reserved-event evaluation.

The presence of disappointing average TTA performance does not itself rule out a selective policy when some conditions benefit. Conversely, an interesting failure mode does not itself justify a controller.

### 11.3 Closest baselines

Complete an applicability matrix for CoTTA, relevant source-restoration/risk-monitoring work, CIM-TTDA, PA-TTA and prediction-verification/rollback methods.

Record task, required information, training/retraining assumptions, prediction timing, available code, upstream version/license and deviations needed for dense paired segmentation.

CoTTA is retained as a scheduled secondary generic comparison, rather than silently deleted. Implement a bounded compatible version if its assumptions can be preserved; otherwise document a specific incompatibility. Extend it to the main confirmation if it is a relevant strong competitor for the final claim.

For a method paper, schedule at least one closest comparable protection/monitoring control beyond a weak generic score. Selection is based on methodological relevance, not which competitor is easiest to beat. Do not require reproducing every paper with a different task.

This work is not postponed until final writing, when a missing nearest baseline would trigger major reconstruction.

### 11.4 Exit

Create a one-page method/empirical-study definition, claim-to-experiment map and branch decision. No unsupported component enters the primary method merely because code already exists.

## 12. E7: implement the chosen policy or empirical branch

Window: Dec 7-13. Dependencies: E6.

### 12.1 Default policy semantics if branch A is supported

Use batch-level decisions with PRE_UPDATE primary predictions:

1. Enter with the last accepted candidate state.
2. Compute current candidate and immutable-source predictions.
3. Compute declared pre-decision generic/change scores without altering either branch.
4. Apply the frozen U-calibrated rule.
5. Accept: output the cached candidate prediction and apply the declared update for future batches.
6. Reject: output the frozen-source prediction and retain/restore the last accepted state; no successful update is counted.

This policy controls both current prediction source and future adaptation. Report their effects separately. It does not claim to detect the harmful effect of an update that has not yet happened.

Post-update trial/rollback is a separate optional policy if supported by timing-correct evidence. It needs its own action/output specification and matched comparators; it cannot silently replace the PRE_UPDATE primary convention.

### 12.2 Required controls and actions

- Source-only.
- Always-adapt.
- BN-no-update.
- Reject-all with source fallback.
- Generic gate.
- Generic-plus-change gate.
- Fixed random acceptance with declared random seeds/rates.
- Update-rejection-only versus output-fallback behavior.

Accept/reject is primary. The original reduce action remains a bounded secondary sensitivity: one declared smaller update scale, only if a policy already has useful intermediate-risk behavior. If scientifically inapplicable, record that reason; do not silently mark it implemented.

State restoration covers parameters, buffers, modes, optimizer/momentum, selector state, relevant RNG and EMA only where EMA exists. An absent EMA is NA.

### 12.3 Verification

Focused tests: actual action/update correspondence, complete reset, rollback after partial/failed updates, numerical failures, empty selections, tails, source independence, GT isolation, phase-correct persistence and interruption behavior.

Run one real bounded source/development smoke after the implementation passes. Use a new run ID, retain every failed attempt and verify a terminal state.

### 12.4 Empirical branch

If branch C is chosen, do not invent a policy. Use this block to implement the remaining normalization/order/checkpoint diagnostic experiments and publication-quality empirical analysis. The confirmation and robustness work below still applies.

### 12.5 Exit

The selected implementation is executable, its scope is frozen, a real smoke is verified and final comparison/LOEO runners have been tested. A declining-to-build decision must include its actual evidence and alternative experiment matrix.

## 13. E8: cross-event evaluation and full confirmation

Window: Dec 14-Jan 3. Dependencies: E7.

### 13.1 LOEO

For a policy claim, run six held-out-event folds.

- Main U track: calibrate from permitted source or the other five events' unlabeled evidence.
- No held-out future input or labels enters a strictly causal threshold.
- C track, if explicitly adopted: other-five-event GT may calibrate, with a separate name and report.
- State starts from the declared frozen initialization for each fold.
- The six-event formula-development history is disclosed.

Primary LOEO evaluates reset-per-event deployment. Continual cross-event experiments are a separate actual stream protocol with explicitly allowed prior/calibration data and initialization. Do not concatenate the six LOEO outputs and call them one real continual trajectory.

The numeric primary threshold/calibration rule is fixed before confirmation. A small sensitivity set may be reported in full. Do not choose a threshold using the held-out event's F1.

For U, predeclare one calibration statistic and the same small quantile-sensitivity budget for generic and change-related scores. For example, 0.75/0.90/0.95 calibration quantiles may be assessed, with 0.90 fixed as primary before outcome inspection. Quantiles come from the permitted calibration population, not from held-out-event performance. Matching realized acceptance by retrospectively ranking the entire target stream is a secondary descriptive analysis, not a causal deployment policy.

### 13.2 Full factor coverage

For the selected method, matched strong generic controls and decisive component controls:

- seeds 0/1/2;
- order_a/order_b;
- reset-per-event and continual where supported;
- all six development events;
- no best-seed selection or removal of failed conditions.

A method with all factors has 12 physical streams and 72 event-result rows. The number of independent events remains six.

Record the exact planned cell count before launching. Reuse compatible cells through the ledger; do not rerun every discarded exploratory combination.

Use seed0 for discovery and seed1 for the planned repeat screen. Freeze the selected new-method recipe before inspecting its seed2 results. Historical baseline exposure on seed2 is disclosed. All three seeds remain in the final table; seed2 is not dropped or used to repeatedly retune the method. This is checkpoint replication, not an independent-event test.

### 13.3 Report

- Event-macro and pooled F1, IoU, precision, recall.
- Condition worst-event and the single global minimum, separately defined.
- Per-event benefits/harms/neutral outcomes with the denominator.
- Actual accepted/rejected/reduced/skipped/failed updates.
- Harm prevented, beneficial adaptation retained and false rejection.
- Prediction-source usage and no-update behavior.
- Runtime/probe/VRAM cost.
- Per-seed results and sample SD across the three seeds.
- Paired event-cluster bootstrap intervals, with the small-event-count limitation.

Orders/reset conditions are repeated factors, not extra independent seeds. Pixel-level bootstrap does not establish event generalization.

### 13.4 Exit

The complete matrix is present, every omission has an explicit reason, and the main claims are supported or withdrawn. No main-method claim rests only on the seed0 exploratory table.

## 14. E9: required ablations and method freeze

Window: Dec 21-Jan 3, using E8 infrastructure.

### 14.1 Matrix follows actual claims

For each retained component, run the relevant removal:

- no confidence filtering;
- no consistency;
- no anchor;
- no temporal/null evidence;
- no controller;
- generic protection substituted;
- normalization control;
- reject-update-only versus source-fallback;
- same acceptance/calibration budget generic versus change-related policy.

If a component was rejected earlier, cite its complete negative experiment rather than inserting it into the final method solely to remove it again.

Core positive claims and decisive negative findings receive the repeat-seed/order support needed for their wording. Do not extend every exploratory grid to all factors.

### 14.2 Exit

The method/empirical scope, main factors, uncertainty, runtime, limitations and decisive ablations are fixed. Code/configs and generated tables can be reconstructed from archived inputs.

The method may still fail Gate M. That does not excuse changing a frozen threshold after looking at the final test.

## 15. E10: robustness and publication-strength extensions

Window: Jan 4-31. Planning/design may start earlier without changing the core matrix.

### 15.1 Required planned extension: second lightweight backbone

Choose one architecture with a defensible paired SAR implementation and compatible outputs. Freeze the choice by Jan 10 using methodological suitability and source-data feasibility, not final-test scores.

Select its training recipe using source-inner data, then train source checkpoints on the same 27 source events. The SNUNet's 16-epoch budget is not automatically appropriate for another architecture.

Use three source seeds. Evaluate source, the relevant strongest generic control and the selected method/empirical normalization contrast under the same information/timing assumptions.

This is new planned robustness training, not a redo of the existing SNUNet source models.

### 15.2 Required planned extension: one synthetic-shift family

Choose one physically interpretable shift after verifying input domain, preprocessing and masks. Freeze clean plus a small ordered set of severities before scoring.

Apply the same shift to all compared methods. Keep spatial semantics/labels valid. Use at least source, the relevant generic baseline and the proposed method or empirical contrast.

The synthetic-shift family is a controlled stress test, not proof of real-world shift coverage.

### 15.3 External data/cross-task extension

Retain this as a planned TGRS-strength extension, not an already completed FYP requirement.

Before acquisition/experiments, verify licenses, temporal pairing, SAR channels, annotation semantics, preprocessing and independent event identities. A post-event water dataset cannot silently become a three-class change dataset.

Select a genuine compatible dataset/task or document why no compatible external test can be executed within this FYP. A TGRS-level cross-dataset claim remains unsupported if this extension is not completed.

Do not consume or rename the ten reserved Kuro Siwo events as this external extension.

### 15.4 Exit

Second-backbone and synthetic-shift results are present or their real blockers are recorded and escalated before February. Tool expiry is not an exclusion reason. Wording of generalization claims matches the completed coverage.

## 16. E11: frozen final test and FYP delivery

### 16.1 Feb 1-7: close genuine gaps

Complete missing declared confirmation/ablation cells, figures and uncertainty. Only evidence-affecting corrections justify re-execution.

Do not start an unbounded new method search in response to disappointing January results.

### 16.2 Feb 8-14: freeze

Freeze the research route, source checkpoints, preprocessing, method formulas, calibration rules, thresholds, primary metrics, statistical plan and final test matrix.

Draft the complete paper/report with final-test placeholders explicitly marked pending. Do not fill them using development results.

### 16.3 Feb 15-28: final-test rehearsal and Gate T

Test the actual execution/recovery/analysis flow using synthetic, source or permitted development fixtures.

Prepare final commands, expected coverage/cell counts, code/config hashes, output/log destinations, failure preservation and analysis scripts before opening the reserved data.

By February 21, deliver a complete English paper draft with the final-test section explicitly pending and all other sections populated. By March 21, integrate the actual final-test analysis into a complete paper manuscript as well as the FYP draft. April paper work is revision or extension, not the first attempt to write a paper.

Final evaluation includes:

- source-only and matched BN-no-update;
- the strongest relevant generic TTA/protection controls selected before test;
- the chosen method, if one is supported;
- generic gate/reject-all and decisive ablations where policy claims require them;
- necessary empirical controls if the route is failure characterization.

For a method claiming both reset modes and two orders, the plan explicitly includes those modes/orders and all three checkpoints. A full method contributes 12 streams and 120 event-result rows on ten events; source redundancies may be reused only through declared invariance.

If a second-backbone final-test generalization claim is made, its frozen source/generic/main-method scope is included before release. Development-only robustness is labeled development-only.

Gate T requires frozen inputs, validation, required ablations, reproducibility package, user authorization and supervisor approval. No response is not approval. A negative/empirical route can pass Gate T without inventing a successful controller.

### 16.4 Mar 1-7: final evaluation

Run the approved matrix once as a research decision. It can contain multiple declared seeds/orders/methods; "one final evaluation" does not mean one arbitrary model run.

Preserve all attempts. Real bugs receive a versioned correction with the original evidence retained. Do not tune after viewing test scores.

### 16.5 Mar 8-May 19: fixed outputs

By March 14:

- final main/event tables and confidence intervals;
- ablation, normalization and error-channel analysis;
- risk discrimination and policy outcomes where applicable;
- qualitative examples, failures, runtime/VRAM and limitations.

By March 21:

- full report: Introduction, Related Work, Task/Data, Motivation, Method/Study Design, Protocol, Results, Ablation, Failure Analysis, Limitations, Integrity/Reproducibility, Conclusion.

March 22-24 is a review buffer. Submit the complete Draft Final Report by March 25.

Revise during March 26-April 4 using real feedback where available. If feedback is unavailable, use numerical/reproducibility/clarity checks without claiming supervisor endorsement.

Proof and submit the Final Report by April 9. Demonstrate during April 12-16 with a short repeatable example and static backup.

Prepare the paper and oral slides during April 19-May 2. Complete three rehearsals by May 9: timing, non-specialist questions, examiner simulation.

Present May 10-12. Address actual feedback, permissions and final formatting, then submit the DR-NTU full text by May 19.

## 17. IGARSS and publication schedule without sacrificing the FYP

Official pages checked on 2026-10-19:

- Submission portal opens November 10, 2026.
- Submission deadline is January 11, 2027.
- Full papers are four pages; accepted full papers are included in IEEE Xplore.
- Abstracts are 400-600 words and are not included in IEEE Xplore.

Sources:

- https://2027.ieeeigarss.org/important_dates.php
- https://2027.ieeeigarss.org/call_for_papers.php

November: write the task, protocol, existing empirical findings and related work alongside experiments. This also supports the FYP.

December 1-13: prepare a complete scoped four-page draft using available evidence, with uncompleted results explicitly pending.

December 20: provisional GO/HOLD/NO-GO. Require a clear claim, honest novelty positioning, matched main table, sufficient stability and a feasible completion path.

January 3: final decision after the scheduled confirmation. If stable evidence and a complete defensible paper are absent, record NO-GO. Do not replace missing confirmation with an optimistic forecast.

January 4-8: final author/supervisor approval, numerical/format checks and submission buffer. Recheck the live paper kit and deadline policies before submitting.

The reserved final test remains scheduled for March. A January paper may make only the claims supported by its actual development/internal cross-event or genuinely external validation. If its central claim needs independent validation that is unavailable, revise the claim or do not submit. No silent early opening of reserved test events.

An empirical paper does not automatically require a new algorithm, but it requires an important, well-supported finding. A method paper requires demonstrated incremental value. Neither route guarantees acceptance.

TGRS is a later evidence-based target. More modules or run counts do not establish novelty. A strong journal submission may need external/cross-task validation and additional analysis beyond the FYP; those requirements cannot be declared fulfilled by the current single-dataset screen.

## 18. How to work before and after November 8

### 18.1 Before November 8

Use available tools to implement/test the common recorder, matched orchestration, augmentation/anchor interfaces, analysis generators and reusable state-management primitives.

If the complete remaining experiment blocks can finish early, pull them forward only after their dependencies and full coverage requirements are met. Never reduce the factor matrix or relabel an unfinished stage to claim early completion.

Preserve the user's current choice of implementer self-verification. Historical independent reviews retain their original authorship/status. A self-check is not renamed independent review.

### 18.2 November 8 handoff checkpoint

Provide:

- CURRENT_STATUS with completed/open/failed items;
- this V3 plan and the requirement/experiment ledgers;
- tested runbook, environment and paths;
- exact active/queued commands and restoration rules;
- source/checkpoint/Fisher/config identities;
- immutable raw outputs, analysis scripts and hashes;
- preserved failed attempts and deviation records;
- at least one verified backup independent of the current working directory;
- a terminal-only rehearsal of table reconstruction and a synthetic task.

Unfinished science remains OPEN. Do not freeze an immature method or stop implementation just because Antigravity may disappear.

### 18.3 After November 8

VS Code Codex continues implementation, debugging, tests, execution, analysis and documentation.

Use one writer at a time. A separate session may inspect a pinned snapshot, but do not claim independent validation merely because the same tool opened a new conversation.

Scientific LOEO/cross-validation, three-seed repetition and held-out testing remain unchanged by AI-tool availability. Losing a second AI reviewer does not mean losing those experimental requirements.

Prefer executable checks and reproducible analysis over repeated AI assurances. Human decisions remain necessary for research claims, reserved-test release, external submission and material scope changes.

## 19. Operating rules that reduce later rework

### 19.1 Experiment lifecycle

1. Identify current code/config and relevant dirty files.
2. Freeze the next exact hypothesis, factors, primary metric and result directory.
3. Run focused tests for the changed behavior.
4. Use a real small smoke when a new data/state/execution path requires it.
5. Commit the scientific scope before substantive runs.
6. Launch long tasks in detached tmux and verify real progress.
7. Check counts, hashes, finite values, labels/timing and completion.
8. Generate analysis from raw artifacts.
9. Update the experiment ledger and backup.

Passing tests are not proof that the scientific hypothesis is true. Negative outcomes do not require automatic parameter search.

### 19.2 Corrections and valid reuse

- If code/config/data identity is unchanged and the result is complete, reuse it.
- If only prose/table transcription is wrong, regenerate the report/table without rerunning models.
- If a required probability/state was not saved, identify the smallest new measurement replay.
- If prediction timing changes, create a separate protocol and matched controls.
- If a bug affects predictions/labels/state, preserve old outputs, define the affected set and rerun only that set under a correction identity.
- Never delete a failed attempt to make the next output directory appear fresh.
- A hash inventory is not an independent backup.
- Do not change frozen config bytes just to clean whitespace.
- Do not use checkout/reset to discard changes without first identifying and preserving their ownership/content.

### 19.3 GPU, environment and progress

Use the validated vfm_env interpreter for scientific work. Record actual Python/PyTorch/CUDA/NumPy versions. Do not switch to sar_cd merely because an executable is missing.

Default to one clearly idle shared GPU. Add a second only under lab rules and an explicit scoped need. Never interfere with another user's process.

Use CUDA_VISIBLE_DEVICES and logical cuda:0. Capture the Python exit code, not just a shell/tee success. No real asset reads or CUDA initialization in CPU-only synthetic paths.

Progress should show method/event/batch, completed samples, update/skip/failure counts and terminal state. A completed tmux session normally disappears; an unmatched pgrep exits 1 and is not evidence of failed training.

If reliable full-state resume is unsupported, preserve the interrupted attempt and use a new attempt directory. Do not promise arbitrary pause/resume or keep a shared GPU occupied with SIGSTOP.

### 19.4 Human time

Plan two to three focused working days per week:

- Day A: define/implement and run focused verification.
- Day B: freeze/launch and verify initial progress.
- Day C: verify results, analyze, write and backup.

Use short intervening checks for long tasks. Reserve approximately one quarter of calendar capacity for shared-GPU waiting, debugging and revisions. More GPU cards do not replace research judgment.

Estimate each matrix as its required new stream count multiplied by observed runtime, with a measured overhead allowance. Current observed six-event runtimes range from roughly 3-8 minutes depending on condition; new probes, architecture, I/O and contention can change that substantially.

If a window slips, record actual completion and update the forecast. Protect November 10 and the March/April submissions. Do not silently cut scientific coverage.

## 20. Requirement crosswalk: nothing disappears

| Original topic | Current status | Scheduled closure |
| --- | --- | --- |
| Pilot/Week 0-3 source foundations | Reported completed | Historical evidence preserved; E0 verifies pointers only |
| Week 4/5 report and submission | Drafts; submission not established | November 9-10 actual submission |
| Week 5/6 source recipe and all-27 seeds | Reported completed | Reuse three controlled checkpoints |
| Week 7 source baseline | Historical completed | E1 checks protocol compatibility, not automatic retraining |
| Week 8/9 strong baseline matrix | Historical completed | E1 fills exact matching/measurement gaps |
| Week 10 route, information boundaries, claims | Documents and decisions exist | E0 consolidates authority/remaining requirements |
| Week 11 minimal consistency | Completed limited implementation/smoke | Reuse; new recorder verified in E1 |
| Week 12 confidence threshold sensitivity | Only no-mask/0.90 done | E2 finite grid |
| Week 12 meaningful augmentation comparison | Horizontal flip reference done | E2 validated view/intensity comparisons |
| Week 13 probability anchor sensitivity | Only lambda 0.50 done | E3 finite weights |
| Week 13 logit/feature anchors | Not yet completed | E3 bounded family comparison |
| Week 13 matched generic protection | Matching not yet fully established | E1/E3 and final E8 ledger |
| Week 14 error diagnosis | Offline diagnosis completed | Reuse existing tables |
| Week 14 four class-aware questions | Intervention cells open | E4 bounded comparisons |
| Week 15 temporal evidence | Interfaces/some logs exist | E5 discrimination, timing and incremental value |
| Week 16 minimum-method definition | Pending evidence | E6 |
| Week 17 accept/reject/rollback | Infrastructure partial | E7 if Gate E supports; explicit empirical branch otherwise |
| Week 17 reduce action | Not primary implementation | Bounded E7 secondary test if scientifically justified |
| Week 18 LOEO | Not completed | E8; U primary, C separately declared |
| Week 19 three seeds/two orders/reset coverage | Historical baseline coverage only | E8 for retained new method and matched controls |
| Week 20 removal ablations | Partial component screens | E9 for actual retained claims |
| Second backbone and shift robustness | Not completed | E10; not removed for November 8 |
| External data/cross-task | Not completed | E10 journal extension, with compatibility decision |
| Final freeze/test | Not authorized yet | E11 and March 1-7 |
| Final report, demonstration, oral, library deposit | Future | Fixed dated chronology in Section 4 |

No row is marked experimentally complete because a new file was merely created. Every excluded, deferred or unsuccessful item keeps its reason and claim consequence.

## 21. What the plan can and cannot promise

The plan is designed to avoid discovering basic comparison, cache or state-management gaps at the end. It preserves valid previous work and schedules confirmation as part of the main research, not as a surprise second pass after Week 30.

It cannot guarantee that no future bug or inconclusive result will require a targeted correction. Reproducibility includes honest correction. The relevant distinction is planned confirmation versus avoidable reconstruction caused by missing records or incompatible protocols.

The expected final outputs are:

1. A controlled, reproducible source/baseline study.
2. A completed bounded component/evidence investigation, including negative results.
3. An evidence-supported method or a rigorous empirical result.
4. Matched multi-seed/order confirmation, ablation and robustness.
5. A frozen independent final-event evaluation.
6. A complete FYP report, paper draft, demonstration and oral presentation.

November 8 changes who helps execute this plan. It does not change the evidence required to support the research.

