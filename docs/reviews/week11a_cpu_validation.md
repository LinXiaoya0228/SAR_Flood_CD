# Week 11B independent review of the Week 11A CPU implementation

Date: 2026-10-18. **CPU implementation: CHANGES REQUIRED.**

This is a narrowly scoped implementation review, not real-data execution approval or evidence of scientific effectiveness. The missing real-data runner is intentional and is **not** a CPU-stage defect. Weeks 0–9 were not reopened.

## Review identity, boundaries and stability

- Host: `gpu22`; working directory: `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`.
- Branch: `main`; HEAD: `e5cbbfc5154bc09ca9eaf48bffa976b596b954ba`.
- All seven requested files were read, including the untracked implementation/config/test files. These are working-tree contents, not an assumption that HEAD contains Week 11A. The full initial Git status is in `stability_start.json`.
- The seven hashes below were checked again after the probes and recorded in `stability_end.json`. No reviewed input or HEAD changed. Existing reports/evidence were preserved.
- Reviewer writes are limited to this new report and `outputs/week11_review_evidence/week11b_independent_v1/`. No implementation, scientific config, protocol, repository test, original artifact, or prior report was edited. Temporary authorization-config copies are diagnostic fixtures inside that evidence directory only.
- No real SAR/label files, source checkpoints or Fisher tensors were loaded. No GPU computation, staging, commits, repairs, historical suite or real-data experiments were performed. Independent probes also install failure sentinels on `torch.load` and GPU lazy initialization.

| Reviewed file | SHA-256 before and after (identical) |
| --- | --- |
| `docs/protocols/week11_evidence_consistency_protocol_v1.md` | `608cbe92d11627777c857e77575915b9c0a9a508c3c310b0188d78338dfbfca7` |
| `configs/changeguard/week11_evidence_consistency_v1.json` | `c0a4f5be443a026f233e5ec4468d6bf2cb01551456ef8bfe75f5791c21f9eb39` |
| `docs/week10_claim_to_evidence.md` | `fb0a764cf2bd44f32970484d618b442218d537980b17f10aeec0816d281b7bdc` |
| `src/changeguard/__init__.py` | `98963ab588cafbe022d405992d38ea253bdcccbeeac3669969f26c5fd3d041f0` |
| `src/changeguard/adapter.py` | `ddb0ed119840ec23ea785a05de6e2fed2e70968f8e17846dd4d731071f1f82f3` |
| `tools/run_week11_cpu_evidence.py` | `06c5814577565306f39d134d158c4a05a7e01198e5c4475cae82b53e1728f525` |
| `tests/test_week11_evidence_consistency.py` | `d38fee86f7ef8aa467ca5b7be97fdac9bf87fb613d99b3127fcba42c2bc91b37` |

## Executed evidence

Evidence directory: [`../../outputs/week11_review_evidence/week11b_independent_v1/`](../../outputs/week11_review_evidence/week11b_independent_v1/).

Python: `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` (3.10.20); PyTorch 2.13.0+cu130, CPU tensors only. Commands used `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, and logs capture both streams with `> file 2>&1`.

Before running the repository tests, I inspected their imports and bodies: the specified module uses randomly initialized proxy models, synthetic tensors and the approved config JSON; its runner integration passes no output directory. It does not load real assets.

| Focused execution | Result | Evidence |
| --- | --- | --- |
| `python -m unittest -v tests.test_week11_evidence_consistency` | 9 tests passed; exit 0 | `focused_tests.log` |
| `python outputs/week11_review_evidence/week11b_independent_v1/independent_probes.py` | Seven diagnostic groups completed; exit 0 means observations reproduced, **not** all contracts passed | `independent_probes.py`, `independent_probes.log`, `independent_probe_results.json` |
| CPU CLI `--validate_only`, original config | Accepted; exit 0 | `cli_preflight.log` |
| CLI with neither synthetic nor validate-only, original config | PermissionError before real-data loading; expected exit 1 | `cli_unauthorized_default.log` |
| CLI `--validate_only --device cuda:0`, temporary config with GPU flag string `"false"` | Incorrectly accepted with `GPU Auth: True`; exit 0; no GPU work | `cli_string_false_preflight.log` |

The independent script exercises actual adapter forwards/backward/SGD and the runner's adaptation → null probes → return → offline scoring chain. It does not merely restate the repository tests' assertions. Its retained synthetic outputs contain 40 samples in batches **16, 16, 8**, with **2 actual update calls and 1 skip**. Random proxy behavior is not a claim about real-data update frequency.

## Findings requiring correction or explicit contract resolution

### W11-R01 — BLOCKER: probe isolation does not implement the declared state contract

**Code:** [`adapter.py:295`](../../src/changeguard/adapter.py#L295), null forwards at lines 311–322, restoration at lines 329–332. Only RNG is saved; candidate/source state is not isolated or restored, and RNG restoration has no `finally`.

**Requirement:** [`protocol:100`](../protocols/week11_evidence_consistency_protocol_v1.md#L100) expressly requires preservation of parameters, buffers, optimizer/momentum and RNG, and says `no_grad()` alone is insufficient.

**Reproduction:** `independent_probes.py`, `probe_isolation()` (line 201); JSON key `probe_isolation`. A synthetic eval-compatible forward increments a registered buffer and consumes RNG. Both candidate and source counters change **0 → 2** during the null call. Success-path RNG is restored, but a synthetic forward exception leaves RNG changed. The ordinary MockSNUNet positive control preserves its state, including populated optimizer momentum; this does not establish the stronger general guarantee.

**Impact:** diagnostic-only forwards can alter later adaptation or the supposedly immutable source for models with forward-side state. Failure paths are not state-neutral either. This is a contract failure, not evidence that the real SNUNet already suffered buffer drift.

**Minimal correction:** isolate probe/reference forwards on independent state or save/restore the relevant model and RNG state with exception-safe restoration. Include forward-side buffer and exceptional-path regressions; preserve the intentional candidate/source BN context difference.

### W11-R02 — BLOCKER: nonfinite updates are reported as successfully applied

**Code:** [`adapter.py:214`](../../src/changeguard/adapter.py#L214), especially loss reduction at line 217 and unconditional backward/step/applied increment at lines 219–225.

**Reproduction:** `numerical_boundaries()` (line 290); JSON key `numerical_boundaries`. An independent model with finite parameters, inputs and logits (`±1e38`, FP32) makes the selected-loss sum overflow: the adapter returns **loss `inf`, update_applied `true`**. A separate, explicitly fault-injected infinite BN gradient produces nonfinite `bn1.weight` while still reporting an applied update. Normal-scale selected-path gradients, SGD state and parameters pass the positive control.

**Impact:** numerical failures can contaminate the candidate/optimizer and be logged as ordinary adaptation. The fault-injection and extreme-logit probes test boundaries; they do not assert that such failures occur on real SAR.

**Minimal correction:** reject or explicitly fail on nonfinite losses/gradients before stepping; verify candidate/optimizer finiteness and restore the pre-step state if an update corrupts it. Record numerical failure honestly, without lowering confidence thresholds, changing the optimizer, or labeling it as an ordinary successful update. A guarded failure is acceptable; silent acceptance is not.

### W11-R03 — NEEDS CONTRACT RESOLUTION: disagreement has conflicting declared timing

**Code:** [`adapter.py:248`](../../src/changeguard/adapter.py#L248) compares the frozen source with `pred_weak`, cached before the update at lines 171–181. It does not compute post-update candidate disagreement. Null evidence is post-update (line 253).

**Conflicting declarations:** [`protocol:86`](../protocols/week11_evidence_consistency_protocol_v1.md#L86) defines disagreement using pre-update `p^w`, which the implementation follows. However, [`protocol:124`](../protocols/week11_evidence_consistency_protocol_v1.md#L124) lists it among post-update diagnostic scores, and [`config:88`](../../configs/changeguard/week11_evidence_consistency_v1.json#L88) declares post-update diagnostic predictions.

**Reproduction:** `prediction_timing_and_source_alias()` (line 240); JSON key `diagnostic_timing`. A controlled optimizer-step sentinel changes the candidate class 0 → 2 while the source stays class 0. Primary predictions correctly remain class 0. Logged disagreement is **0**, whereas independently measured post-update disagreement is **1**. The forced transition tests timing only, not the size of a normal SGD update.

**Impact:** consumers cannot treat the mixed evidence record as uniformly post-update. This is an internally inconsistent contract, not grounds to silently declare the formula's pre-update interpretation mathematically wrong.

**Minimal resolution:** the protocol owner must choose and document the timing explicitly. If retaining the formula, identify disagreement as pre-update in the schema/config and persisted evidence. If post-update disagreement is intended, implement a separate state-neutral diagnostic prediction. Keep the primary pre-update metric unchanged. Resolve this before claiming full conformance; this reviewer has changed neither declaration nor code.

### W11-R04 — MAJOR, CPU acceptance blocker: persisted evidence loses the required identity and context

**Code:** [`run_week11_cpu_evidence.py:183`](../../tools/run_week11_cpu_evidence.py#L183) creates IDs and passes them at lines 189–195; the serialization record at lines 222–239 drops them. The summary at lines 247–265 retains counts only. Adapter BN-context disclosure at [`adapter.py:280`](../../src/changeguard/adapter.py#L280) is also dropped.

**Requirement:** [`protocol:118`](../protocols/week11_evidence_consistency_protocol_v1.md#L118) requires sample/event identity, seed/order/reset, pixel counts, timing/action/transform and separate offline evaluation. Context disclosure is required at protocol line 98.

**Reproduction:** JSON keys `runner_gt_and_coverage` and `persisted_schema`, plus saved `synthetic_output/`. Instrumentation independently captures **40 unique actual IDs**, but neither persisted JSON nor CSV contains those IDs or event IDs. Timing, transform, branch BN contexts and predicted flood counts are absent too. Aggregate confusion counts and update/skip totals are internally consistent; that does not repair missing identity records.

**Impact:** saved evidence cannot independently reconstruct coverage or connect risk observations to the represented samples and branch contexts. Count evidence alone is not a persisted sample-uniqueness proof.

**Minimal correction:** persist sample/event IDs and the relevant schema/context alongside batch evidence, with an explicit batch-to-sample mapping. Clearly distinguish batch aggregates from any per-sample fields; use honest synthetic/not-applicable values for checkpoint/order/reset metadata. Retain pre/post phase metadata and offline confusion counts. This does not require building a real-data runner or pretending the proxy has a real checkpoint.

### W11-R05 — BLOCKER: unsupported CPU-stage modes can pass preflight

**Code:** [`run_week11_cpu_evidence.py:85`](../../tools/run_week11_cpu_evidence.py#L85), especially `bool(...)` at lines 88–89 and the condition at line 91. Preflight at lines 98–118 does not enforce CPU-only status or boolean authorization types.

**Reproduction:** JSON key `authorizations`; `cli_string_false_preflight.log`. The unchanged approved config correctly rejects non-CPU devices and real execution. In a reviewer-owned temporary copy, `gpu_execution_authorized: "false"` becomes Python `True`; the actual CLI accepts `--device cuda:0 --validate_only` and prints **PREFLIGHT OK** while status is still **APPROVED_FOR_CPU_IMPLEMENTATION**. Boolean `true` also permits the non-CPU constructor under the CPU-only status. Only constructor/preflight were exercised for these configurations; no GPU initialization or tensors were used.

**Impact:** the CPU-stage boundary depends on permissive truthiness and flags rather than validating the declared execution stage. A malformed false value is treated as permission.

**Minimal correction:** strictly validate authorization types and enforce the CPU/synthetic stage independently of flag truthiness. Unsupported combinations must fail before model/device access. A future real/GPU execution path requires its own implemented and approved stage; enabling flags alone is insufficient.

### W11-R06 — MAJOR: explicit source-model aliasing breaks source immutability

**Code:** [`adapter.py:70`](../../src/changeguard/adapter.py#L70) accepts the caller's source object directly; candidate configuration at lines 78 and 86–104 then re-enables BN training/gradients and removes running buffers if the two branches share state.

**Reproduction:** JSON key `source_alias`. `Adapter(model, source_model=model)` is accepted. The resulting source is the candidate, its BN is training with trainable bias, and its running mean is `None`.

**Impact:** the public constructor permits a state in which the source-reference invariant is immediately false. The normal CPU runner uses the default deep-copy path and passes the immutability positive control; its results are not invalidated by this separate boundary case.

**Minimal correction:** reject shared module/parameter/buffer storage or create an independent source copy before candidate reconfiguration. Cover both whole-object and shared-storage aliasing. Correct this before claiming the constructor's unconditional frozen-source guarantee.

## Nonblocking interface/schema notes

- **W11-N01 — mask shape validation.** [`adapter.py:164`](../../src/changeguard/adapter.py#L164), selection at line 202: a synthetic `(B,1,H,W)` mask silently broadcasts against `(B,H,W)` confidence. With `B=2` and only one valid sample, metadata reports **48 selected pixels but 24 valid pixels**. The current runner supplies `(B,H,W)` and passes. Explicitly reject unsupported shapes or normalize a singleton channel before integration with a future loader; this unsupported-input boundary is not the sole basis for failing the current CPU runner.
- **W11-N02 — empty-loss representation.** [`adapter.py:208`](../../src/changeguard/adapter.py#L208) correctly performs no backward/optimizer step on empty selection, but emits numeric `0.0` where protocol line 43 says `None`. Align the representation so a skipped loss is distinguishable from a computed zero KL. No optimizer-mechanics failure was observed for empty/zero-valid batches.

## Positive checks and limits

| Requested check | Independently established result |
| --- | --- |
| BN-only scope and explicit branch behavior | Default path: only the two proxy BNs' affine weights/biases train; convolution/head stay frozen. Candidate batch-stat/no-tracking and source eval/frozen-running-stat behavior are explicit. Source parameters/buffers stay unchanged on the normal path. Explicit-source alias exception: R06. |
| Identity/horizontal-flip alignment | Actual first two forwards receive identity and paired horizontal flips. Strong logits are inverted before selection/loss. An asymmetric original-coordinate mask correctly selects after realignment; the unused flipped-mask local is not itself a spatial error. |
| Detached one-way KL, confidence and sensory validity | Actual weak-logit hook receives no backward gradient; strong branch does. Loss `8.567499776290788e-7` versus independent double-precision oracle `8.586636781132618e-7`. Boundary confidence exactly FP32 0.90 is included, 0.899 excluded, and an invalid high-confidence column excluded. No approved threshold was changed. |
| Empty selection and ordinary numerical health | Uniform-confidence and zero-valid probes at threshold 0.90 cause zero optimizer calls and two skips, with model/optimizer/RNG unchanged. Ordinary selected path has finite gradients and parameters and matches one SGD step including momentum. Nonfinite boundaries: R02. |
| Prediction timing | Actual primary tensors equal the first, pre-update forward and remain distinct from a forced post-update class change. Diagnostic contract conflict: R03. |
| Null metrics and interpretation | Both null pairs' class-2 probabilities and positive-delta average match independent calculations. Adapter exposes both BN contexts. Their difference includes normalization-context effects and is not a pure parameter-drift measurement. Full state neutrality: R01; persisted disclosure: R04. |
| Reset | After a real synthetic selected update, candidate parameters/configured buffers return exactly to initial candidate state, momentum is emptied, counters reset, and source remains unchanged. Candidate BN running buffers intentionally return to `None`, not source-eval buffers. |
| GT isolation through actual call chain | Runner passes only SAR tensors, sensory mask and IDs to adaptation. Sentinel GT creation occurs after adapter return/primary caching, never during adaptation or null forwards. A tensor-subclass sentinel observes 72 GT operations, all offline. The synthetic ignore-label-3 sentinel is excluded only offline. This goes beyond a forbidden-key test. |
| IDs, tails, confusions and counters | Independent capture proves 40 unique IDs, 16/16/8 batches. Offline `tp=1740`, `fp=3504`, `fn=11269`, `tn=22369`, sum **38882** equals evaluated valid pixels. Per-batch counts sum to totals; two update booleans plus one skip equal the adapter counters. Persisted identity is missing: R04. These are synthetic counts, not evidence about development coverage. |
| CPU authorization | Original config passes CPU preflight, denies GPU requests, and raises before real loading. Malformed/unsupported combinations: R05. |

Repository tests passed but do not establish every advertised guarantee. For example, tests 02/03 mostly reconstruct the loss/flip outside the adapter; test 05 checks parameters and normal-path RNG without stateful-buffer/exception coverage; the runner test checks totals without saved-ID recovery. The independent checks above supplement them without modifying repository tests.

The claim-to-evidence matrix was read for its hypothesis/claim boundaries. Its earlier empirical and literature statements were not re-audited here. Null-risk usefulness, harmful-update detection, controller gains and generalization remain **untested scientific hypotheses**. A low or zero update rate at the approved confidence setting would not itself prove a code defect or justify threshold tuning.

## Minimal next step and stage boundaries

1. Have the implementer correct **R01, R02, R04, R05 and R06**, and have the protocol owner resolve **R03** explicitly. Preserve approved scientific settings; address the small shape/loss-schema notes in the same narrow boundary work if appropriate. Freeze a new version and perform a targeted CPU re-review of the affected paths. No broad historical re-audit or real-data run is needed for these corrections.
2. Before any later real-data smoke, implement and review the deliberately absent execution infrastructure: controlled SNUNet/checkpoint identity loading; development-only manifest/ID resolution and sensory-mask loading with event filtering before raster access; isolated label-free adaptation/probe inputs; frozen primary prediction persistence and separately delayed offline scoring; event/order/reset scheduling, tail/coverage checks, provenance and output protection. Test that orchestration synthetically. `run_real_stream()` currently raises `NotImplementedError` even if permission is flipped (runner line 127); config changes alone cannot supply this infrastructure.
3. A future Week 11C GPU smoke needs a stable reviewed code/config version, a concrete bounded smoke plan and separate explicit real-data/GPU authorization. None is granted by this review. Scientific effectiveness remains to be tested only through a subsequently authorized experiment.

Evidence integrity and exact commands/exit statuses are recorded in `stability_start.json`, `stability_end.json`, `execution_record.json` and `evidence_manifest.json` in the independent evidence directory.
