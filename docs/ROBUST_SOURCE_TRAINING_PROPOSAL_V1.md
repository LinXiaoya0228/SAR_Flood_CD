# Event-Balanced Robust Source Training for Paired SAR Flood Change Detection

## Research Proposal and Sequential Execution Plan

**Version:** 1.0  
**Date:** 2026-10-26  
**Project:** ChangeGuard FYP  
**Status:** Proposed new primary research route  
**Target implementation window:** 2026-10-26 to 2026-11-08  
**Target experimental completion:** January 2027  
**Target complete FYP draft:** February 2027  

---

## 1. Project transition

### 1.1 Completed work retained as motivation

The completed ChangeGuard E0–E11 study evaluated source-only inference and several test-time adaptation methods for paired Sentinel-1 SAR flood change detection.

The main findings were:

- TTA can improve some events but harm others.
- Performance is sensitive to source checkpoint, architecture, event, normalization context and evaluation protocol.
- BN-related effects explain a substantial part of the apparent adaptation gain.
- Generic pre-update risk scores did not reliably predict harmful adaptation across seeds and stream orders.
- The tested consistency, confidence filtering, source-anchor and temporal/null-probe mechanisms did not provide sufficient evidence for a reliable selective controller.
- TTA improvements did not transfer consistently between SNUNet-ECAM and SiamUnet_diff.

These findings remain valid historical evidence. Their code, outputs, manifests and conclusions must not be overwritten or retrospectively changed.

### 1.2 New primary route

The new route moves the intervention from test time to supervised source training.

Instead of attempting to decide whether an unlabeled target update is safe, the project will investigate whether a source model can be trained to generalize more reliably across unseen flood events without any target-time parameter update.

The proposed method is:

> **Event-Balanced Robust Source Training for Paired SAR Flood Change Detection**

The method will combine a small number of defensible source-training components:

1. event-balanced sampling;
2. a flood-aware supervised objective;
3. physically valid paired SAR augmentation.

The final method may retain only the components that produce repeatable improvements. Complexity will not be added merely to make the method appear more novel.

---

## 2. Research problem

Flood-event datasets are strongly imbalanced at several levels:

- some events contain far more image patches than others;
- permanent water, flooded area and background have very different frequencies;
- large events may dominate optimization;
- the flood class is often the class of greatest practical interest but may occupy relatively few valid pixels;
- SAR appearance varies between events because of acquisition conditions, land cover, geometry and environmental conditions.

A model trained by uniformly sampling patches can therefore optimize mainly for large events and common classes rather than for event-level generalization.

This produces the following research question:

> Can event-balanced supervised training, a flood-aware loss and physically justified paired SAR augmentation improve cross-event flood change detection without using target-event data or test-time model updates?

---

## 3. Hypothesis

The primary hypothesis is:

> Giving source events more equal influence during training will improve event-level generalization and reduce the domination of large events.

The secondary hypotheses are:

- a flood-aware supervised objective may improve flood recall and F1 without causing unacceptable precision loss;
- paired SAR augmentation may reduce sensitivity to moderate acquisition-related intensity variation;
- the combined recipe may improve worst-event performance or reduce variation across training seeds;
- source-only deployment may become more reliable than generic TTA while remaining simpler and safer.

These are testable hypotheses, not guaranteed results.

---

## 4. Intended contribution

The intended positive contribution is not merely a new trained checkpoint.

The intended contribution is:

> A reproducible, target-free source-training recipe for paired SAR flood change detection that improves average cross-event performance, worst-event reliability, or seed stability under event-disjoint evaluation.

Possible accepted forms of positive contribution are:

### Contribution A: average improvement

The proposed recipe improves mean event-level flood F1 over the matched original training recipe.

### Contribution B: reliability improvement

The proposed recipe substantially improves the lower-performing events or worst-event result while keeping average performance approximately unchanged.

### Contribution C: stability improvement

The proposed recipe reduces variability across source-training seeds while maintaining comparable average performance.

### Contribution D: simpler component finding

If event-balanced sampling alone provides the useful improvement and the other components do not, the simpler event-balanced recipe becomes the final contribution.

The project will not claim that a component is useful unless its contribution is supported by matched experiments.

---

## 5. Relationship to the completed TTA study

The previous study and the new method form one coherent FYP story:

1. Generic TTA was evaluated under controlled, multi-seed and event-wise conditions.
2. Its gains were found to be inconsistent and dependent on normalization, architecture and event.
3. Unsupervised risk signals were not sufficiently reliable for a safe controller.
4. The project therefore moves the robustness intervention to supervised source training.
5. The new method attempts to improve unseen-event generalization without relying on target-time adaptation.

The resulting thesis is not:

> TTA never works.

The intended thesis is:

> In paired SAR flood change detection, generic test-time adaptation is not reliably beneficial across events and architectures. Improving how source events and flood classes contribute during supervised training may provide a simpler and more stable alternative.

---

## 6. Method

### 6.1 Frozen task

The task remains paired SAR flood change detection using:

- pre-event and post-event SAR observations;
- VV and VH channels;
- the existing three-class semantic definition;
- the existing valid-pixel and label mapping rules;
- the existing SNUNet-ECAM implementation as the primary backbone.

No label semantics, channel definitions or event identities may be silently changed.

### 6.2 Component M1: event-balanced sampling

The original patch-level sampler allows events with more patches to contribute more optimizer updates.

The proposed event-balanced sampler will:

1. select a source event using a uniform event probability;
2. select a valid training patch from that event;
3. construct batches without allowing event size alone to determine event frequency;
4. record the realized sample count for every event and epoch.

The sampler must define:

- sampling with or without replacement;
- behavior for small events;
- number of samples per epoch;
- deterministic seed handling;
- worker-level random-state behavior;
- whether a batch may contain multiple events.

The primary implementation should remain simple and auditable.

### 6.3 Component M2: flood-aware supervised objective

The loss must use source labels only.

The first candidate is a controlled combination of:

- multiclass cross-entropy;
- multiclass Dice loss;
- a predeclared flood-class weighting or event-aware class balance term.

Any class weights must be computed only from the current training partition. Held-out event labels must not influence the weights.

The loss must not:

- use target development or former test labels;
- change weights according to held-out performance;
- ignore precision collapse;
- optimize only flood recall.

All loss terms and coefficients must be frozen before confirmatory folds are inspected.

### 6.4 Component M3: physically valid paired SAR augmentation

Spatial transforms must be applied identically to the pre-event and post-event images and to the corresponding labels.

Initial permitted geometric transformations are:

- horizontal flip;
- vertical flip;
- 90-degree rotations, only after confirming that the task does not depend on absolute image orientation.

Radiometric augmentation must be implemented in the correct input domain.

Before implementation, the pipeline must establish whether stored inputs are:

- linear backscatter;
- decibel values;
- normalized decibel values;
- standardized network inputs.

The primary radiometric transform must correspond to a defensible SAR acquisition variation. Arbitrary multiplicative speckle must not be applied directly to standardized or decibel inputs.

The first radiometric candidate should be a small common-mode gain or offset applied consistently to both times. A differential pre/post perturbation may only be introduced as a separately declared sensitivity experiment.

### 6.5 Final method selection

The final method is not required to contain all three components.

The selected recipe will be the smallest recipe that satisfies the frozen promotion criteria.

For example:

- if M1 improves results and M2/M3 do not, retain only M1;
- if M1+M2 improves the flood class but M3 adds nothing, retain M1+M2;
- if the full method performs best consistently, retain all three.

---

## 7. Experimental integrity and data policy

### 7.1 Previously accessed target events

The six historical development events and ten historical final-test events have already been evaluated during E0–E11.

They can no longer be described as completely unseen for the new method.

They must not be used to:

- tune the new loss;
- choose augmentation severity;
- select method components;
- choose epochs;
- select a checkpoint;
- change the promotion rule.

Later evaluation on these events must be described as a previously observed cross-domain extension, not a pristine final test.

### 7.2 Primary new evaluation population

The primary evaluation will use event-disjoint partitions constructed from the 27 labeled source events.

The split must be frozen before method outcomes are inspected.

The 27 events will be divided into three balanced groups:

- Group A: 9 events;
- Group B: 9 events;
- Group C: 9 events.

The grouping procedure should use only pre-existing metadata such as patch count and source-event identity. It must not optimize the groups according to model performance.

### 7.3 Discovery and confirmation roles

Group C is the designated discovery group.

During discovery:

- train on Groups A+B;
- evaluate on Group C;
- use seed 0;
- compare the predefined method staircase.

After the method is frozen:

- Confirmation Fold A: train on Groups B+C and evaluate on Group A;
- Confirmation Fold B: train on Groups A+C and evaluate on Group B;
- use source-training seeds 0, 1 and 2;
- do not alter the method after seeing Group A or Group B outcomes.

The 18 events in Groups A and B form the main confirmatory event population.

A complete three-fold summary may later include Group C, but Group C must remain identified as the discovery fold.

### 7.4 Checkpoint selection

No checkpoint may be selected using an outer held-out event group.

The primary comparison should use:

- the same fixed training budget for baseline and proposed recipes;
- the same architecture;
- the same optimizer family;
- the same checkpoint timing;
- the same training partition;
- the same seed;
- the same evaluation implementation.

If the epoch budget is changed, it must be selected using training-only or source-inner evidence and applied equally to the matched baseline and proposed method.

---

## 8. Experimental matrix

### 8.1 Discovery staircase

Run on seed 0 with Groups A+B for training and Group C for evaluation.

| ID | Event-balanced sampling | Flood-aware loss | Paired SAR augmentation |
|---|---:|---:|---:|
| R0 | No | No | No |
| R1 | Yes | No | No |
| R2 | Yes | Yes | No |
| R3 | Yes | Yes | Yes |

R0 must reproduce the original source recipe under the new event-disjoint training partition. It must not reuse a model trained on all 27 events.

This is a staircase ablation, not a full unrestricted hyperparameter search.

### 8.2 Confirmatory matrix

After the recipe is frozen, compare:

- matched R0 baseline;
- one selected robust recipe.

Run:

- Confirmation Fold A;
- Confirmation Fold B;
- seeds 0, 1 and 2.

This produces:

- 2 methods;
- 2 confirmatory folds;
- 3 seeds;
- 12 training runs;
- 18 unique held-out events;
- one result per held-out event and seed.

No failed seed or difficult event may be removed.

### 8.3 Optional robustness extensions

Only after the primary confirmation is complete:

- repeat the selected recipe on SiamUnet_diff;
- train the final deployable model on all 27 source events;
- evaluate the frozen recipe on the previously observed six development and ten former test events;
- compare source-only robustness with the historical TTA results.

These are secondary extensions and must not delay the primary confirmation.

---

## 9. Metrics

### 9.1 Primary metric

The primary metric is:

> Event-macro flood-class F1 over held-out events.

Each held-out event contributes equal weight regardless of patch count.

### 9.2 Secondary metrics

Report:

- event-macro IoU;
- event-macro precision;
- event-macro recall;
- pooled F1 and IoU;
- per-event F1;
- worst-event F1;
- lower-decile or bottom-event average;
- predicted flood area;
- seed mean and sample standard deviation;
- training runtime;
- peak VRAM;
- convergence and numerical failures.

### 9.3 Paired analysis

For each held-out event, compute:

\[
\Delta F1_e = F1_{robust,e} - F1_{baseline,e}.
\]

Report:

- median event difference;
- mean event difference;
- number of improved, neutral and harmed events;
- paired event-level bootstrap interval;
- results per seed;
- pooled results only as a secondary view.

Pixels and patches must not be treated as independent events.

---

## 10. Promotion criteria

The recipe is promoted if it satisfies at least one of the following predefined outcomes.

### Outcome P1: average performance

- confirmatory event-macro F1 improves by at least 0.01;
- the gain is not driven by only one event;
- no widespread precision or recall collapse occurs.

### Outcome P2: worst-event reliability

- worst-event or bottom-event-average F1 improves by at least 0.02;
- mean event-macro F1 decreases by no more than 0.005.

### Outcome P3: seed stability

- mean performance remains within 0.005 of the baseline;
- across-seed variability falls by at least 20%;
- the stability gain appears in both confirmatory folds.

The final report must state which outcome, if any, was satisfied.

These rules must not be changed after the confirmatory results are viewed.

---

## 11. Sequential execution plan

The work must proceed in the order below. A later block must not begin until the previous block reaches its exit condition.

---

### RST0 — Route transition and historical freeze

**Dates:** 2026-10-26

Tasks:

1. Record E0–E11 as completed historical work.
2. Record that the new route does not invalidate or overwrite the TTA study.
3. Mark the six development and ten former test events as previously observed.
4. Create a new output namespace for robust source training.
5. Confirm that all repository documents and generated records are written in English.
6. Record the new research question and contribution boundary.

Required outputs:

- this proposal;
- a transition note;
- a historical-output protection list;
- a new run-ID and directory convention.

Exit condition:

- historical outputs are protected;
- no target-event label is authorized for method development;
- the new route is clearly separated from E0–E11.

---

### RST1 — Data, input-domain and baseline audit

**Dates:** 2026-10-26 to 2026-10-27

Tasks:

1. Audit the 27 source events and patch counts.
2. Verify input storage and preprocessing:
   - linear or dB;
   - normalization;
   - invalid-pixel handling;
   - VV/VH channel order;
   - pre/post ordering.
3. Audit the existing source sampler.
4. Audit the existing loss.
5. Audit augmentation currently used by the original recipe.
6. Reconstruct the exact matched R0 training recipe.
7. Design the deterministic A/B/C event grouping procedure.
8. Estimate runtime and storage for all planned runs.

Required outputs:

- source-event inventory;
- input-domain report;
- matched-baseline specification;
- frozen A/B/C split;
- compute and storage estimate.

Exit condition:

- the original recipe can be reproduced exactly;
- the augmentation domain is known;
- the event split contains no overlap;
- no model training has yet used held-out fold labels.

---

### RST2 — Event-disjoint training and evaluation infrastructure

**Dates:** 2026-10-27 to 2026-10-29

Tasks:

1. Implement explicit train-event and evaluation-event allowlists.
2. Add fail-closed event-overlap checks.
3. Implement event-wise metrics and event-macro aggregation.
4. Ensure each held-out event is evaluated independently.
5. Store manifests containing:
   - Git commit;
   - config hash;
   - split hash;
   - checkpoint hash;
   - seed;
   - training events;
   - held-out events;
   - epoch budget;
   - optimizer and loss;
   - augmentation settings.
6. Add resume and checkpoint reload verification.
7. Add deterministic data-loader tests.

Required tests:

- no held-out event enters training;
- no event appears in both partitions;
- label mapping is unchanged;
- valid-pixel handling is unchanged;
- resume restores model, optimizer and scheduler;
- event-macro aggregation is correct;
- configuration mismatch fails closed.

Exit condition:

- a CPU synthetic test passes;
- a bounded real-data smoke completes;
- baseline training and evaluation can be reproduced from one frozen config.

---

### RST3 — Implement the three method components

**Dates:** 2026-10-29 to 2026-10-31

Tasks:

1. Implement event-balanced sampling.
2. Implement realized event-frequency logging.
3. Implement the flood-aware loss.
4. Compute any class statistics from training events only.
5. Implement synchronized paired geometric augmentation.
6. Implement the approved radiometric transform in the verified input domain.
7. Add independent switches for M1, M2 and M3.
8. Confirm that R0 remains bit-for-bit or numerically matched to the baseline path when all switches are disabled.

Required tests:

- deterministic event selection under fixed seeds;
- small-event behavior;
- no held-out statistics used;
- identical spatial transform for pre, post and label;
- label values remain valid;
- invalid pixels remain excluded;
- loss is finite;
- gradients are finite;
- R0 does not accidentally enable a new component.

Exit condition:

- all CPU tests pass;
- one short GPU smoke for R1–R3 passes;
- no NaN, Inf, empty-selection or shape failure remains.

---

### RST4 — Discovery staircase

**Dates:** 2026-10-31 to 2026-11-03

Training:

- train on Groups A+B;
- evaluate on Group C;
- seed 0;
- fixed training budget;
- run R0, R1, R2 and R3.

Tasks:

1. Launch independent runs in versioned tmux sessions.
2. Use at most one training process per GPU.
3. Preserve every failed run.
4. Validate all final checkpoints.
5. Produce the complete discovery table.
6. Report component-level changes, not only the best result.
7. Inspect precision, recall and worst-event behavior.
8. Freeze one selected recipe using the predefined promotion logic.

Exit condition:

- all four discovery recipes are complete;
- the selected recipe is explicitly recorded;
- no more than one recipe is promoted;
- confirmatory Groups A and B remain uninspected for new-method performance.

---

### RST5 — Recipe freeze and reproducibility handoff

**Dates:** 2026-11-03 to 2026-11-05

Tasks:

1. Freeze:
   - selected components;
   - loss coefficients;
   - augmentation severities;
   - epoch budget;
   - optimizer;
   - batch size;
   - checkpoint rule;
   - metrics;
   - promotion criteria.
2. Create confirmatory configs for Folds A and B and seeds 0/1/2.
3. Run validation-only and dry-run checks.
4. Test resume and interrupted-run recovery.
5. Create monitoring commands and closeout templates.
6. Commit the reviewed implementation and configs.
7. Write a self-contained continuation guide for work after 2026-11-08.

Exit condition:

- scientific settings cannot be silently overridden;
- confirmatory configurations are fixed;
- another agent can reproduce the planned runs without chat history.

---

### RST6 — Confirmatory launch

**Dates:** 2026-11-05 to 2026-11-08

Launch:

- R0 and selected robust recipe;
- Confirmation Folds A and B;
- seeds 0, 1 and 2;
- 12 total training runs.

Operational rules:

- distribute independent jobs across the four RTX 3090 GPUs;
- do not run two memory-heavy training jobs on one GPU;
- use detached tmux sessions;
- use durable logs;
- preserve partial checkpoints;
- record terminal completion or failure for every run.

Before 2026-11-08, aim to complete:

- all implementation;
- all configs;
- all tests;
- the discovery stage;
- the recipe freeze;
- launch of all confirmatory jobs;
- preferably the majority of confirmatory training.

Exit condition:

- all 12 jobs have either completed or have recoverable, documented status;
- no scientific setting changed after launch.

---

### RST7 — Confirmatory completion and analysis

**Dates:** 2026-11-09 to 2026-11-22

Tasks:

1. Complete any remaining training runs.
2. Verify every checkpoint and manifest.
3. Evaluate all 18 confirmatory held-out events.
4. Generate paired per-event comparisons.
5. Calculate seed mean and sample standard deviation.
6. Calculate paired event-bootstrap intervals.
7. Apply the frozen P1/P2/P3 decision.
8. Record positive, negative or inconclusive results honestly.
9. Do not start a new unrestricted hyperparameter search.

Exit condition:

- the primary contribution is supported or rejected;
- every event and seed is present;
- no result was removed because it was unfavorable.

---

### RST8 — Optional second-backbone verification

**Dates:** 2026-11-23 to 2026-12-06

Proceed only if the selected recipe satisfies P1, P2 or P3 on SNUNet-ECAM.

Tasks:

1. Apply R0 and the selected recipe to SiamUnet_diff.
2. Use the same confirmatory event groups.
3. Start with seed 0.
4. Expand to three seeds only if the direction is meaningful.
5. Report whether the contribution transfers across architecture.

If the method does not transfer, limit the claim to SNUNet-ECAM and report the architectural dependence.

Exit condition:

- the architecture scope of the claim is known;
- unnecessary full-factor runs are avoided.

---

### RST9 — Final deployable source model and historical-target extension

**Dates:** 2026-12-07 to 2026-12-20

Proceed after the source-domain method is frozen.

Tasks:

1. Train the selected recipe on all 27 source events using seeds 0/1/2.
2. Produce three final deployable checkpoints.
3. Save complete hashes and provenance.
4. Evaluate source-only inference on:
   - the six previously observed development events;
   - the ten previously observed former test events.
5. Compare with the historical source-only and TTA results.
6. Clearly label this as a previously observed cross-domain extension.
7. Do not tune the method using these results.

Exit condition:

- final checkpoints exist;
- the source-domain confirmatory claim remains separate from the historical-target extension;
- the deployable model requires no target-time update.

---

### RST10 — Final analysis and figures

**Dates:** 2026-12-21 to 2027-01-15

Required outputs:

1. Main event-disjoint confirmation table.
2. Component staircase table.
3. Per-event paired-difference figure.
4. Worst-event and bottom-event analysis.
5. Precision/recall trade-off figure.
6. Seed-stability figure.
7. Event sampling distribution before and after balancing.
8. Runtime and VRAM table.
9. Failure cases and limitations.
10. Comparison with the completed TTA study.

Exit condition:

- all figures are reconstructable from archived CSV/JSON files;
- no manually copied number is the only source of a reported result;
- claims match the completed evidence.

---

### RST11 — FYP writing and delivery

**Dates:** 2027-01-16 to 2027-02-28

Proposed report structure:

1. Introduction
2. Related Work
3. Paired SAR Flood Change Detection
4. Dataset and Event-Disjoint Protocol
5. Controlled Study of Test-Time Adaptation
6. Motivation for Robust Source Training
7. Event-Balanced Robust Source Training
8. Experimental Results
9. Ablation and Reliability Analysis
10. Comparison with Test-Time Adaptation
11. Limitations and Research Integrity
12. Conclusion

Central story:

> Controlled experiments showed that generic TTA was not consistently reliable for paired SAR flood change detection. The project therefore developed and evaluated a target-free robust source-training approach that gives source events more balanced influence and explicitly considers flood-class performance.

Exit condition:

- a complete English FYP draft exists by February;
- numerical tables are linked to archived outputs;
- negative TTA findings and positive source-training findings are clearly separated;
- novelty is described conservatively.

---

## 12. Compute plan

Available hardware:

- four NVIDIA RTX 3090 GPUs.

Recommended scheduling:

- GPU 0: baseline or Fold A seed 0;
- GPU 1: selected method or Fold A seed 1;
- GPU 2: Fold B runs;
- GPU 3: smoke, evaluation, failed-run recovery or remaining seed.

Rules:

- do not share one GPU between two full training processes;
- record physical GPU UUID;
- record peak VRAM;
- keep training and evaluation logs;
- do not overwrite outputs;
- use new run IDs for corrections;
- preserve failed attempts.

---

## 13. Scope control

The following are outside the primary route unless later evidence requires them:

- further TTA controller searches;
- new temporal/null risk signals;
- unrestricted loss grids;
- unrestricted augmentation grids;
- reinforcement learning;
- architectural redesign of SNUNet;
- unverified external datasets;
- repeated tuning on the six development or ten former test events;
- implementation of every related TTA paper.

The project should prioritize one complete positive contribution over several unfinished directions.

---

## 14. Risk management

### Risk 1: event balancing does not improve mean F1

Response:

- inspect worst-event and seed-stability outcomes;
- retain the component only if it satisfies P2 or P3;
- do not repeatedly alter the sampler using confirmatory results.

### Risk 2: flood-aware loss improves recall but damages precision

Response:

- report both effects;
- retain it only if event-level F1 or the predefined reliability outcome improves;
- do not claim success from recall alone.

### Risk 3: augmentation is harmful

Response:

- remove augmentation from the final recipe;
- retain its negative ablation as evidence;
- do not add more augmentation families without a new approved protocol.

### Risk 4: no component satisfies promotion criteria

Response:

- preserve the complete negative result;
- do not fabricate a positive method;
- the completed TTA and robust-training study still forms a rigorous empirical FYP;
- write the limitation clearly.

### Risk 5: confirmatory jobs are incomplete before 2026-11-08

Response:

- ensure code, tests, configs, commits, tmux commands and recovery instructions are complete before the Antigravity access deadline;
- continue the frozen runs afterward without changing the scientific protocol.

---

## 15. Claim boundaries

Permitted claim if successful:

> Event-balanced robust source training improved event-level generalization or reliability for paired Sentinel-1 SAR flood change detection under a frozen event-disjoint evaluation protocol.

Not permitted without additional evidence:

- the method solves SAR domain shift generally;
- the method always outperforms TTA;
- the method generalizes to all sensors or flood datasets;
- the method is the first robust SAR change-detection training method;
- the method is statistically proven across all real-world events;
- the previously accessed target events constitute an untouched final test.

---

## 16. Antigravity execution instructions

Before acting, Antigravity must read:

1. `PROJECT_PROPOSAL.md`;
2. `docs/MASTER_EXECUTION_PLAN_V3.md`;
3. the final E11 closeout and final-result records;
4. this proposal;
5. the current Git log and repository status.

Operational requirements:

- treat E0–E11 as immutable historical evidence;
- do not overwrite historical configs, runners, logs or outputs;
- create versioned files for this route;
- write repository documents in English;
- distinguish draft, approved, running, completed and failed states;
- do not infer approval merely because a config exists;
- do not access the six development or ten former test labels for method selection;
- do not launch a large experiment before its protocol, split and output paths are frozen;
- run focused tests before GPU execution;
- preserve every failed attempt;
- make scoped Git commits;
- produce a closeout for every RST block;
- do not advance to the next RST block until the current exit condition is satisfied.

The first action after reading this proposal is **RST0**, followed strictly by **RST1**. Antigravity must not jump directly to model training.