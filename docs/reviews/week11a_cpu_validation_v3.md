# Week 11A targeted CPU re-verification v3

2026-10-18. **CPU-stage acceptance: PASS. Remaining CPU blockers: none.**

This review closes only **R02, R05 and N02** from [independent review v2](week11a_cpu_validation_v2.md), against the actual implementation described in [corrections delivery v2](../week11a_cpu_corrections_v2.md). Previously passed findings remain accepted; they were not reopened. No historical suite or Week 0–9 audit was repeated. Implementer declarations were checked against code and independent CPU reproductions, not treated as independent evidence.

| Finding | Verdict | Independently established correction |
| --- | --- | --- |
| **R02** | **PASS** | Optimizer-only Inf/NaN corruption with already-populated momentum is detected, all pre-step candidate/optimizer state is restored exactly, failures are counted honestly, and a subsequent ordinary update succeeds. |
| **R05** | **PASS** | Missing, null and unknown statuses are rejected by construction and direct `validate_preflight()` calls. Each actual CLI preflight rejects the former bypass; the approved CPU CLI remains accepted. |
| **N02** | **PASS** | Actual saved JSON has native null losses for skipped and failed steps; CSV retains textual `null`. Numeric losses remain numeric and CSV serialization does not mutate shared JSON records. |

## Review identity and stability

Host: `gpu22`. Repository: `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`. Branch: `main`. HEAD: `e5cbbfc5154bc09ca9eaf48bffa976b596b954ba`.

Relevant Week 11 files are untracked working-tree inputs; HEAD alone does not identify the reviewed implementation. All ten input hashes, HEAD and all **45 prior independent evidence files** were recorded before/after and remained unchanged. The previous reviews are also pinned unchanged. Full hashes and Git status are in `stability_start.json` and `stability_end.json` in the new evidence directory.

| Current relevant file | SHA-256 before/after, identical |
| --- | --- |
| `src/changeguard/adapter.py` | `4a410e382c8691e2fe8ad947614b53d0b6856c30fc6a3b0858e1f6e863032879` |
| `tools/run_week11_cpu_evidence.py` | `ce6ac897af1278807cfde47c621a89744b6e0eb489082d9683987e59aec4beb3` |
| `tests/test_week11_evidence_consistency.py` | `020ad3ecf71b97c9876e92ec6214571257bbbccbd5e7631b55b6e626116235ed` |
| `configs/changeguard/week11_evidence_consistency_v1.json` | `21b3f40f4b836389e1f1114711b53a4e51ab2d018001721b79f350119e1d127f` |
| `docs/protocols/week11_evidence_consistency_protocol_v1.md` | `84aab300c10c74aa2548ba75de2cce57e3928a0f9dfa07cd1efb687d11c03faf` |

The three corrected code/test hashes match the delivery. Approved scientific config and protocol hashes remain those accepted in v2. Other pinned inputs are the two prior independent reports, both corrections notes and package initializer. Code changes were already present at the start of this turn. This reviewer edited no implementation, config, protocol, repository test, original artifact or prior evidence. Writes are only this new report and its fresh independent evidence directory; diagnostic configuration copies are isolated there.

## Executed focused checks

New evidence: [`../../outputs/week11_review_evidence/week11b_independent_v3/`](../../outputs/week11_review_evidence/week11b_independent_v3/).

Interpreter: `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`, Python 3.10.20, PyTorch 2.13.0+cu130, CPU tensors only. Environment: `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; `TMPDIR` points inside the fresh evidence directory. Shell test/probe logs use **`> file 2>&1`**; child CLI logs capture stdout and stderr together.

Before execution, I inspected the selected test bodies and relevant fixtures/imports. They use random proxy models, synthetic tensors, config metadata and temporary files only. No real SAR images/labels, source checkpoints or Fisher tensors were loaded. Independent probes guard `torch.load` and GPU lazy initialization. No GPU computation, real-data execution, staging, commits, environment installation or repairs occurred.

Only these existing repository regressions were run:

```text
tests.test_week11_evidence_consistency.TestWeek11EvidenceConsistency.test_06_r02_numerical_failures_guarded_and_rolled_back
tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration.test_r05_strict_boolean_and_stage_enforcement
tests.test_week11_evidence_consistency.TestWeek11CPURunnerIntegration.test_r04_and_n02_persisted_json_null_and_csv_text_null
```

**Result: 3 tests passed in 1.052 seconds, exit 0** (`focused_tests.log`). The last selected test combines N02 with existing serialization assertions; this is not a separate reopening of R04. The full 12-test module and historical suites were not run.

`targeted_probes.py` then independently executed the actual adapter and runner. It reuses read-only original fixture definitions through the preserved v2 evidence module; neither old main function is invoked and no old evidence is written. **All three probe groups passed, exit 0**, with observations in `targeted_probe_results.json` and both output streams in `targeted_probes.log`.

## R02: optimizer-only failure and exact rollback

**Code:** recursive optimizer-state finiteness inspection at [`adapter.py:140`](../../src/changeguard/adapter.py#L140); actual post-step check at [`adapter.py:308`](../../src/changeguard/adapter.py#L308); parameter/optimizer restoration and failure reporting at lines 314–333. The formerly parameter-only condition now includes the optimizer state, including nested tensors and scalar floats.

**Independent reproduction:** `targeted_probes.py`, `r02_optimizer_only()`; JSON key `findings.R02.cases`.

For each of **Inf and NaN**, a real selected update at the unchanged confidence threshold **0.90** first creates finite SGD momentum. The next real SGD step is wrapped to corrupt only momentum after parameter updating. The wrapper confirms parameters are still finite while optimizer state is nonfinite, reproducing the exact v2 failure boundary rather than making parameter corruption trigger the guard indirectly.

Both cases now return:

```text
action = "failure"
update_applied = false
loss = null
failure_reason identifies non-finite optimizer state and restored pre-step state
applied = 1, failed = 1, skipped = 0, step_count = 2
```

Independent snapshots prove bit-exact restoration of **all candidate parameters, source state, populated optimizer state including momentum and parameter groups, registered buffers, module modes, gradients and CPU RNG**. Restoration retains four populated momentum entries rather than resetting to an empty optimizer. A subsequent unwrapped update succeeds with finite parameters/optimizer state and counters **applied=2, failed=1, skipped=0, step_count=3**.

The selected repository R02 regression also exercises the existing loss-overflow and infinite-gradient rejection paths. The independent checks supply the exact optimizer-state equality test and NaN case beyond finiteness alone. These are synthetic fault-injection boundaries, not claims about numerical failures on real SAR. **R02 is resolved.**

## R05: missing/null/unknown status through actual preflight

**Code:** [`run_week11_cpu_evidence.py:114`](../../tools/run_week11_cpu_evidence.py#L114) now rejects every status except `APPROVED_FOR_CPU_IMPLEMENTATION` before device handling; [`run_week11_cpu_evidence.py:137`](../../tools/run_week11_cpu_evidence.py#L137) independently validates status in `validate_preflight()`. The permissive fallback is removed. Existing strict boolean and CPU-only restrictions remain in place.

**Independent reproduction:** `r05_status_preflight()`; JSON key `findings.R05`, `status_fixtures/`, and the four retained CLI logs.

For a **missing status key**, **JSON null status**, and **`UNSUPPORTED_STAGE`**, reviewer-owned temporary config copies set the GPU flag to actual boolean `true` and request `cuda:0`, reproducing the old bypass. Each constructor now raises `ValueError` before any model/device access. To test actual preflight separately, a valid CPU runner's in-memory status is removed/changed, then its real `validate_preflight()` method is called; each also raises `ValueError`.

The actual CLI is then invoked for all three temporary configurations with `--device cuda:0 --validate_only`:

| Status fixture | CLI exit | Result |
| --- | --- | --- |
| Missing | **1** | `ValueError: Unsupported configuration status`; no PREFLIGHT OK |
| Null | **1** | Same fail-closed rejection |
| Unknown | **1** | Same fail-closed rejection |
| Original approved CPU config | **0** | PREFLIGHT OK |

Logs: `cli_missing_status.log`, `cli_null_status.log`, `cli_unknown_status.log`, `cli_valid_cpu_preflight.log`. No model or GPU tensors are constructed in these preflight commands. The selected repository regression additionally verifies strict boolean validation and rejection of a GPU device under the approved CPU status. **R05 is resolved.**

## N02: actual saved JSON nulls and separate CSV conversion

**Code:** [`run_week11_cpu_evidence.py:288`](../../tools/run_week11_cpu_evidence.py#L288) keeps `meta['loss']` unchanged in the shared record. JSON is saved at lines 354–355. CSV-specific copies at [`run_week11_cpu_evidence.py:357`](../../tools/run_week11_cpu_evidence.py#L357) convert `None` to text only for CSV output.

**Independent reproduction:** `n02_saved_types()`; JSON key `findings.N02.cases`. The real runner writes three separate synthetic evidence outputs:

| Output | Actions | JSON loss types after reading from disk | CSV behavior |
| --- | --- | --- | --- |
| `persisted_skip/` | 2 skips | `NoneType`, `NoneType` | Both losses are textual `null` |
| `persisted_failure/` | 2 numerical failures | `NoneType`, `NoneType` | Both losses are textual `null` |
| `persisted_mixed/` | 2 updates, 1 skip | `float`, `float`, `NoneType` | Numeric losses retained; skip is textual `null` |

Skips are made deterministic with a uniform-confidence synthetic proxy at unchanged threshold 0.90; failures use a high-confidence synthetic proxy and optimizer-only fault injection through actual adaptation. Mixed output uses the runner's ordinary randomly initialized proxy. Actual adapter metadata is captured and compared with saved JSON/CSV records.

Every skipped/failed JSON record contains **native `"loss": null`**, never `"loss": "null"`. Numeric losses remain numeric and agree with actual adapter output. The saved JSON equals the returned summary after CSV generation, proving the CSV-specific conversion does not mutate the shared structured records. Action counters remain consistent with the observed records. This verifies the affected serialization path without repeating the closed coverage/metric audit. **N02 is resolved.**

## Acceptance boundary

All three remaining findings are resolved. Combined with the unchanged prior accepted findings, **Week 11A CPU-stage implementation is accepted for the reviewed working-tree hashes**. No concrete CPU blocker remains in this targeted scope. Evidence file hashes and exact execution records are retained in `evidence_manifest.json` and `execution_record.json`.

This CPU pass is not experimental completion or evidence of scientific effectiveness. The deliberately absent real-data runner remains a separate future implementation stage, not a CPU defect. No real-data/GPU launch or authorization is granted by this review.
