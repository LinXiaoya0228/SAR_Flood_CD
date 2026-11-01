# Week 11A corrections: independent targeted CPU re-verification v2

2026-10-18. **CPU-stage acceptance: CHANGES REQUIRED.**

Scope is limited to **R01–R06 and N01–N02** from [the original independent review](week11a_cpu_validation.md), using [the corrections note](../week11a_cpu_corrections_v1.md), current code/config/protocol/tests, and new independent synthetic probes. The implementer's resolution table is not treated as verification. No Week 0–9 review or broad historical suite was repeated.

## Finding dispositions

| Finding | Verdict | Independent result |
| --- | --- | --- |
| **R01** | **PASS** | The original forward-counter/RNG fixture now preserves candidate/source parameters and buffers, optimizer with populated momentum, module modes and CPU RNG on success and candidate/source probe exceptions. Actual source-reference success/exception paths also preserve their entry state. |
| **R02** | **CHANGES REQUIRED — BLOCKER** | Loss/gradient rejection and parameter-triggered rollback pass. **Optimizer-only** nonfinite state is neither rejected nor restored and is reported as an applied update. |
| **R03** | **PASS** | Protocol, config, adapter and saved outputs consistently identify primary predictions/disagreement as **PRE_UPDATE** and null probes as **POST_UPDATE**. A forced class transition confirms the actual phases, not only labels. |
| **R04** | **PASS** | Saved JSON/CSV IDs and batch mapping match actual adapter calls; context, provenance, timing and counts are retained. Forty unique IDs in 16/16/8 batches are independently verified from persisted evidence. |
| **R05** | **CHANGES REQUIRED — BLOCKER** | Strict boolean types and the recognized CPU status behave correctly. Null/unknown status falls through to a permissive branch, allowing non-CPU preflight. |
| **R06** | **PASS** | Whole-object aliasing and distinct models sharing parameter/buffer storage both produce an independent, frozen internal source branch; actual adaptation leaves it unchanged. |
| **N01** | **PASS** | Singleton-channel masks normalize correctly; invalid dimensions, batch count or spatial shape are rejected. Original 48-selected/24-valid broadcast error is gone. |
| **N02** | **CHANGES REQUIRED — MINOR** | Adapter skipped losses are genuinely `None`, with no optimizer call. Persisted JSON still stores the string `"null"`, rather than JSON `null`. CSV text `null` is acceptable. This is not an additional execution blocker. |

**Remaining CPU blockers are R02 and R05.** N02 requires a small serialization correction but does not independently warrant rejecting otherwise correct CPU execution. The missing real-data runner remains a separate future stage and is not a finding here.

## Stable review identity and preserved inputs

Host `gpu22`; repository `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`; branch `main`; HEAD `e5cbbfc5154bc09ca9eaf48bffa976b596b954ba`. Week 11 files remain untracked working-tree inputs, so HEAD alone does not identify the implementation. The current changed inputs were already present when this turn began; they are not reviewer edits.

All nine input hashes are recorded before and after in `stability_start.json` / `stability_end.json` in the new evidence directory. No reviewed input or HEAD changed during the review. The original review and every file in its independent evidence directory were also hashed and remained unchanged. The five corrected file hashes match the corrections note's declared values:

| File | SHA-256, identical before/after |
| --- | --- |
| `src/changeguard/adapter.py` | `10e986a1f7fad6d3902f18cc474dbb81e102432718d6898443c92b90451a3c4b` |
| `tools/run_week11_cpu_evidence.py` | `eb1ff51c8149dac7c4acb4385a1ef024796e1bbe2e4d83b951e77d7fb500e398` |
| `tests/test_week11_evidence_consistency.py` | `e5abb70ede964f006e03da9b059ad47d892a121a82c938d994f769a5c92f4f08` |
| `configs/changeguard/week11_evidence_consistency_v1.json` | `21b3f40f4b836389e1f1114711b53a4e51ab2d018001721b79f350119e1d127f` |
| `docs/protocols/week11_evidence_consistency_protocol_v1.md` | `84aab300c10c74aa2548ba75de2cce57e3928a0f9dfa07cd1efb687d11c03faf` |

Additional pinned inputs: the original review, corrections note, unchanged `src/changeguard/__init__.py`, and unchanged claim-to-evidence document. The latter was hashed for continuity, not re-audited. No implementation, config, protocol, repository test or prior evidence was edited. Reviewer writes are only this new report and the new independent evidence directory. Temporary config copies are clearly isolated probe fixtures there; they do not change approved settings.

## Focused execution and independent evidence

Evidence: [`../../outputs/week11_review_evidence/week11b_independent_v2/`](../../outputs/week11_review_evidence/week11b_independent_v2/).

The current focused tests were inspected before execution: randomly initialized proxy models, synthetic tensors, config metadata and temporary output directories only. They contain no real asset loads. `TMPDIR` was directed inside the fresh review evidence directory. No real SAR/label/checkpoint/Fisher assets or GPU work were used; no staging, commits or automatic repair occurred.

Interpreter: `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`, Python 3.10.20, PyTorch 2.13.0+cu130, CPU tensors only. Environment: `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and review-local `TMPDIR`. Both output streams are retained with **`> file 2>&1`**.

| Execution | Observed result | Evidence |
| --- | --- | --- |
| `python -m unittest -v tests.test_week11_evidence_consistency` | **12 tests passed**, 8.022 s, exit 0 | `focused_tests.log` |
| `python outputs/week11_review_evidence/week11b_independent_v2/targeted_probes.py` | Seven diagnostic groups completed, exit 0; individual PASS/failure observations are explicit | `targeted_probes.py`, `targeted_probes.log`, `targeted_probe_results.json` |
| Actual CLI `--validate_only --device cuda:0` with isolated null-status config | Incorrectly accepted, **exit 0 / PREFLIGHT OK**; no GPU initialization or execution | `cli_unsupported_status.log`, `authorization_copies/case_15.json` |

The script imports only definitions from the preserved original independent probe file, reusing its exact `StatefulModel`, `LargeFiniteLogits`, `TimingModel`, high-confidence proxy and input fixtures. It does not run or modify the old script's main function or evidence. This directly reproduces the original boundaries against current code. Additional probes cover optimizer-only corruption and actual saved output types. `torch.load` and GPU lazy initialization are guarded against in the independent process. A diagnostic script exit of zero means observations were captured, not that every finding passed.

## Concrete remaining findings

### R02 — post-step optimizer-state finiteness is not checked

**Code evidence:** [`adapter.py:295`](../../src/changeguard/adapter.py#L295) claims to check candidate and optimizer state, but lines **296–299 inspect only parameters**. The successful branch at lines **315–319** increments applied updates without inspecting optimizer state. Pre-step optimizer snapshots exist at line 290 and restoration at line 304, but restoration is reached only when a parameter is nonfinite.

**Independent reproduction:** `targeted_probes.py`, `r02_numerical_boundaries()`; JSON key `findings.R02.cases.poststep_optimizer_only`. After one actual selected update populates momentum, the next real SGD step is wrapped to inject `inf` into momentum **after** the parameter update. Parameters remain finite. Observed result:

```text
parameters_finite = true
optimizer_finite = false
optimizer_restored = false
action = "update"
update_applied = true
failure_reason = null
applied_counter = 2
failed_counter = 0
```

This is a focused fault-injection test of the expressly requested optimizer-state invariant, not a claim that ordinary real-data SGD has exhibited this fault. It proves that optimizer-only corruption survives and is logged as success.

**What now passes:** the exact original finite-logit FP32 loss-overflow boundary is rejected before any optimizer call. Both `inf` and `nan` gradient injections are rejected with parameters and populated optimizer state unchanged. Parameter-only and combined parameter/momentum corruption are both detected and restored bit-exactly, with `action="failure"`, `loss=None`, and honest counters.

**Impact:** nonfinite optimizer state can enter later batches while the evidence claims a successful update. The existing repository R02 test covers loss and gradient faults, but not this optimizer-only boundary.

**Minimal correction:** inspect optimizer state for nonfinite values as well as parameters after stepping; route either failure through the existing parameter/optimizer rollback and failure-reporting path. Preserve approved SGD/scientific settings. Add a focused optimizer-only regression, including restoration of already-populated momentum and counters.

### R05 — unknown execution-stage metadata bypasses the CPU-only branch

**Code evidence:** [`run_week11_cpu_evidence.py:114`](../../tools/run_week11_cpu_evidence.py#L114) applies CPU-stage restrictions only when status equals `APPROVED_FOR_CPU_IMPLEMENTATION`. The **else branch at lines 132–137** accepts any other status if the GPU flag is true. `validate_preflight()` at **139–159** does not reject unsupported/null status.

**Independent reproduction:** JSON key `findings.R05`; isolated `authorization_copies/case_15.json` uses `status: null`, actual boolean `gpu_execution_authorized: true`, and device `cuda:0`. Case 16 uses `status: "UNSUPPORTED_STAGE"`. Both pass constructor/preflight. The actual CLI for case 15 prints:

```text
Status:           None
GPU Auth:         True
[PREFLIGHT OK] Authorization and configuration bounds verified.
```

It exits **0**. Only validate-only preflight was invoked; no GPU model allocation or computation occurred. These are temporary malformed/unsupported configs in the evidence directory, not edits to the approved config or claims that the unchanged approved command bypasses its gate.

**What now passes:** strings, integer flags and null flags are rejected; missing flags fail closed. True flags under the recognized CPU status are rejected. The original CPU status also rejects `cuda:0`, `mps` and an invalid device before model/device access. Fifteen such type/recognized-stage cases pass the independent check.

**Impact:** unsupported metadata is treated as an alternative authorized execution stage, despite this being a CPU-only runner with no implemented approved GPU stage. The original truthiness bug is fixed, but stage validation remains incomplete.

**Minimal correction:** reject missing/null/unknown status in this CPU runner, or otherwise constrain it to the explicitly implemented approved CPU stage before considering device flags. Do not use a permissive fallback for hypothetical future stages. Retain strict booleans. A later implemented and authorized real/GPU runner remains separate work.

### N02 — skipped losses are stringified before JSON persistence (MINOR)

**Code evidence:** [`run_week11_cpu_evidence.py:285`](../../tools/run_week11_cpu_evidence.py#L285) replaces `None` with string `"null"` in the shared record before [`json.dump` at line 352](../../tools/run_week11_cpu_evidence.py#L352).

**Reproduction:** `null_loss_output/synthetic_run_summary.json` was produced through the actual runner using a uniform-confidence proxy at the unchanged threshold **0.90**. Both steps skip. Reading JSON yields losses `['null', 'null']`, with Python types `['str', 'str']`, not `None`. The actual adapter correctly returns `None`, performs no optimizer step, and preserves state for empty and zero-valid selection.

**Impact/minimal correction:** the JSON null/type contract remains inconsistent, although action flags honestly distinguish skips and no numerical metric is fabricated. Preserve `None` in the structured JSON record, and apply a CSV-specific textual representation only when serializing CSV if desired. This is a minor serialization correction, separate from R02/R05 execution blockers.

## Verification details for closed findings

- **R01:** [`adapter.py:335`](../../src/changeguard/adapter.py#L335) source-reference execution and [`adapter.py:403`](../../src/changeguard/adapter.py#L403) null probes now restore registered buffers and RNG in `finally`. The exact original stateful fixture passes on normal null forwards and exceptions originating in either candidate or source. With optimizer momentum populated, parameter/buffer/optimizer/mode/RNG snapshots match before/after. Source-reference entry hooks verify its normal and exceptional paths independently. This verifies the CPU scope and original failing boundaries; it is not a GPU/CUDA test or a guarantee for arbitrary malicious forward implementations.
- **R03:** [`protocol:58`](../protocols/week11_evidence_consistency_protocol_v1.md#L58), [`config:88`](../../configs/changeguard/week11_evidence_consistency_v1.json#L88), and [`adapter.py:375`](../../src/changeguard/adapter.py#L375) now agree. The original controlled transition changes candidate class 0 → 2 while the source stays 0: primary class and recorded disagreement remain pre-update (0 and 0.0), independently measured post-update disagreement is 1.0, and both null-pair candidate probabilities equal the independently measured post-update class-2 probability **0.9503301978111267**. No confidence threshold or optimizer setting was changed.
- **R04:** [`runner:236`](../../tools/run_week11_cpu_evidence.py#L236) creates and maps IDs, [`runner:276`](../../tools/run_week11_cpu_evidence.py#L276) persists batch context, and [`runner:321`](../../tools/run_week11_cpu_evidence.py#L321) retains provenance and mappings. Actual adapter-call IDs equal JSON `all_sample_ids`, every `batch_to_samples` entry, and CSV pipe-delimited IDs. There are **40 unique persisted IDs**, batches **16/16/8**, event ID 520 explicitly synthetic, **2 updates / 1 skip / 0 failures**, and final step count 3. All per-batch actions/pixel counts/timings/transforms/BN contexts match actual adapter results. Confusions sum across rows: **TP 1740, FP 3504, FN 11269, TN 22372**, total **38885**, equal evaluated valid pixels. F1 recomputed from these counts matches saved F1. Checkpoint/order/reset/model metadata honestly describe a synthetic proxy. These values differ from the first review's ignore-label sentinel by three TN pixels because this targeted persistence probe uses the runner's normal synthetic labels; no original artifact is changed or compared as if the fixtures were identical.
- **R06:** [`adapter.py:76`](../../src/changeguard/adapter.py#L76) deep-copies the source before candidate configuration. Independent fixtures cover the same-object case **and** distinct models sharing a parameter allocation and BN buffer. Underlying storage pointers and module identities are disjoint after construction, source BNs remain eval with running statistics, and all source parameters/buffers remain unchanged after an actual selected update. This extends the repository test, which principally exercises the same-object case.
- **N01:** [`adapter.py:189`](../../src/changeguard/adapter.py#L189) normalizes `(B,1,H,W)` and checks batch/spatial dimensions. Original two-sample asymmetric validity now gives **24 selected / 24 valid**. Five malformed dimension/batch/spatial cases raise `ValueError` rather than broadcasting.

## Acceptance boundary

Correct **R02** and **R05**, fix the small **N02** JSON serialization issue, then pin a new version for another narrow CPU re-verification. No further review of the passed findings is needed unless affected by those changes. No changes were made by this reviewer.

The deliberately absent real-data runner remains outside CPU acceptance. This review does **not** authorize or launch real-data loading, checkpoint/Fisher work, GPU smoke or experiments; scientific effectiveness remains untested. Future execution infrastructure and authorization are separate stages, not substitutes for resolving the two CPU blockers above.
