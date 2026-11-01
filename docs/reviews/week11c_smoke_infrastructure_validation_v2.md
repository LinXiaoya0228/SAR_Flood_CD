# Week 11C infrastructure corrections: independent re-verification v2

2026-10-18. **Infrastructure acceptance: CHANGES REQUIRED. Remaining code blockers: W11C-R02 and W11C-R04.**

This review re-verifies only W11C-R01–R07 from [the original infrastructure review](week11c_smoke_infrastructure_validation.md), using the actual corrected implementation and [delivery note](../week11c_infrastructure_corrections_v1.md). Accepted CPU algorithm findings and Weeks 0–9 were not reopened. A separate delivery-record check establishes that the quoted CUDA transfer attempt succeeded; it does not establish a smoke launch or determine code correctness.

## Dispositions

| Finding | Verdict | Independent result |
| --- | --- | --- |
| **R01** | **PASS** | Corrected paths resolve to the canonical parent-workspace locations; missing dataset/manifest metadata fails before hashing assets or touching a model/device. |
| **R02** | **CHANGES REQUIRED** | ID-bound predictions/actions now persist before GT and survive an offline failure. Adaptation-phase failures still lack terminal failure/progress evidence, and checkpoint/input-manifest digests are not recorded. |
| **R03** | **PASS** | Actual masked scoring returns TP=50, FP=1050, FN=0, TN=0; confusion sum and evaluated valid pixels both equal 1100. |
| **R04** | **CHANGES REQUIRED** | Draft/null/unknown stage, bad approval, malformed boolean and unauthorized GPU cases reject correctly. Unsupported real device strings are still accepted, including through actual preflight. |
| **R05** | **PASS** | Parsed config negatives for seed/event/order/reset/batch/normalization/temporal/channels/precision/selection reject. Ascending head50, uniqueness, sufficient population and [16,16,16,2] are verified. |
| **R06** | **PASS** | Guarded synthetic preflight/execution makes zero physical asset byte-read attempts and zero asset-hash calls; code/config reads remain allowed. |
| **R07** | **PASS** | CPU context makes no CUDA availability/RNG call even with availability mocked true. Mocked CUDA success and source/null exceptions save and restore correctly without GPU initialization. |

## Identity, stability and reviewer boundaries

Host `gpu22`; repository `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`; branch `main`; HEAD `e5cbbfc5154bc09ca9eaf48bffa976b596b954ba`. The Week 11 implementation remains untracked working-tree content. No reviewed code version is inferred solely from HEAD.

All **14 reviewed input hashes**, HEAD and all **97 prior independent evidence files** were recorded before/after and remain unchanged. New evidence is [`../../outputs/week11_review_evidence/week11c_infrastructure_independent_v2/`](../../outputs/week11_review_evidence/week11c_infrastructure_independent_v2/); complete records are in `stability_start.json`, `stability_end.json`, `execution_record.json` and `evidence_manifest.json`.

| Corrected input | SHA-256, identical before/after |
| --- | --- |
| `tools/run_week11_real_data_smoke.py` | `8de61c1b18c5dadf6d87e2d108348973d64b3678933d72b66b72932fa752cfb1` |
| `configs/changeguard/week11_smoke_event520_draft.json` | `4c014d01401f4b2c7d985b4206417b1b632097c0e48f160b0ba0e04658c08e55` |
| `tests/test_week11_real_data_smoke.py` | `487f09e2934920d70c9e385ffd2beb49f8a358b05a4ae40feaf41721d451e44e` |
| `src/changeguard/adapter.py` | `f1b5d6125463f384407b83ff5ab352420026655ff9fd2c8c198c775dfb8a4c5c` |
| `docs/week11c_infrastructure_corrections_v1.md` | `8d1611fa5efe628a9767af54da76a773adc1ad4f610c692e3e7d345ab0f0b42c` |

These implementation hashes match the delivery. The necessary loader/wrapper/external architecture hashes remain those previously reviewed; only their affected call paths were followed. The frozen evaluation config was used as metadata for canonical-path comparison, not for a historical audit.

Reviewer writes are only this new report and the fresh evidence directory. No implementation, scientific config, protocol, repository test, prior report or prior evidence was edited. Draft authorization flags on disk remain false/false. No real SAR/GT/manifest/checkpoint/Fisher content was read by the review; path existence checks are metadata only. No GPU initialization/execution, staging, commits or repairs occurred **in this review**. The separately confirmed earlier implementer CUDA operation is documented below.

Positive real-branch orchestration is exercised only with the authorization decision, checkpoint load/hash, manifest input and raster callbacks explicitly mocked. The runner still performs its actual construction, selection, batching, adapter calls, persistence and scoring. These test-boundary substitutions confer no real-data permission. Authorization truth tables and negative config fixtures are review-only inputs; the draft file is unchanged.

## Focused tests and independent reproductions

Interpreter `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`; CPU tensors only. Environment: `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and evidence-local `TMPDIR`. Logs capture both streams with **`> file 2>&1`**.

The corrected seven-test module was inspected before execution. It uses synthetic records, temporary config/output files and mocked CUDA contexts; its two real-mode path tests stop on missing metadata before any checkpoint access. It was run via `guarded_execution.py`, which forbids real asset byte reads, `torch.load` and CUDA lazy initialization, and would guard child CLI invocations too.

**Result: seven focused tests passed in 20.198 seconds, exit 0**, with **zero asset-read attempts blocked and zero asset bytes allowed** (`focused_tests.log`, `focused_guard_observations.json`). No accepted CPU/historical suite was rerun. Passing implementer tests is supplemented by `independent_probes.py`, `independent_probes.log` and `independent_probe_results.json`; the diagnostic script exits 0 when observations are captured, even where a finding remains unresolved.

The independent probes reuse original virtual component/directory fixtures read-only. In the real orchestration/loader path they provide 60 synthetic Event-520 records plus out-of-scope metadata, verify ascending head50 and [16,16,16,2], reset once, and check exact on-disk prediction arrays/ID mapping/actions before the first synthetic GT read. One injected gradient failure is honestly counted with three later updates. All pre-GT artifact hashes remain unchanged after scoring or offline failure. No real assets are substituted into these fixtures.

### R01 — corrected paths and early metadata rejection

[`runner:201`](../../tools/run_week11_real_data_smoke.py#L201) resolves the corrected `../datasets/KuroSiwo` and `../third_party/KuroSiwo/...` spellings to the frozen parent-workspace locations. Real-mode path checks precede checkpoint audit and model/device work in [`runner:382`](../../tools/run_week11_real_data_smoke.py#L382).

Independent missing-root and missing-manifest runs each raise `FileNotFoundError` with **zero asset-hash calls, zero model calls, and no output directory created**. The nominal paths match the canonical frozen references by resolved path identity; their contents were not opened. See `results.R01`.

### R02 — pre-GT/offline-failure boundary passes; adaptation lifecycle and provenance remain incomplete

**Verified corrections:** [`runner:672`](../../tools/run_week11_real_data_smoke.py#L672) writes `frozen_predictions.npz`, adaptation JSON/CSV and progress before GT access. At the first synthetic label callback, the probe opens the saved NPZ with `allow_pickle=False`, verifies all 50 IDs and arrays against actual primary predictions, and verifies actions/batch mapping against adapter output. GT failure now produces `execution_failure.json` with `FAILED_DURING_OFFLINE_SCORING`; the persisted pre-GT files remain byte-identical. A clean run produces a completion record. These parts of the original reproduction are resolved.

**Remaining blocker A — adaptation-phase exceptions bypass failure handling.** The adaptation/model/loading/persistence section beginning at [`runner:513`](../../tools/run_week11_real_data_smoke.py#L513) is outside the `try` that starts at **line 717**. Its handler at [`runner:858`](../../tools/run_week11_real_data_smoke.py#L858) covers only offline scoring/output completion.

Independent injection at the second adapter call, after one completed batch, leaves:

```text
completed adapter batches: 1
predictions completed in memory: 16
GT reads: 0
files: [initial_manifest.json]
initial lifecycle state: RUNNING_ADAPTATION
execution_failure.json: absent
adaptation_progress.json: absent
```

The initial manifest is an improvement over the old empty directory, but it does not preserve last-completed batch/ID/counter evidence or identify a terminal adaptation failure. See `results.R02.adapt_failure` and `mocked_adapt_failure/`.

**Remaining blocker B — required asset digests are still absent.** The initial manifest at [`runner:489`](../../tools/run_week11_real_data_smoke.py#L489) and final manifest at [`runner:883`](../../tools/run_week11_real_data_smoke.py#L883) contain checkpoint/manifest paths, but no expected/actual checkpoint SHA or input-manifest digest. Actual checkpoint hashing is reduced to a success boolean in preflight; the computed digest is not propagated. `get_git_info()` now records several code/config hashes, so the earlier total absence of code identity is corrected, but it does not supply asset identity. The delivery's assertion that the initial manifest includes checkpoint hashes is inaccurate.

**Minimum correction:** cover the complete post-initialization lifecycle with phase-aware failure recording; preserve last-completed batch/IDs/actions/counters and durable progress as execution proceeds, without labeling a partial run completed. Propagate the expected/actual checkpoint digest and input-manifest identity into durable provenance, using explicit not-applicable markers for synthetic execution. Test all of this with synthetic hash inputs and injected adaptation failures; do not read real assets for the correction review.

### R03 — valid-only confusion counts pass

The actual scoring path now intersects TN with `valid`, and asserts per-sample/step/event sums. The original two-invalid-pixel fixture gives **TP=50, FP=1050, FN=0, TN=0, valid=1100**, so the sum is 1100 rather than the former 1200. This is exported runner output in `mocked_success/`, not a reimplementation of scoring. See `results.R03`.

### R04 — original stage loophole closes; unsupported real devices still pass

The real authorization method at [`runner:239`](../../tools/run_week11_real_data_smoke.py#L239) now rejects draft overrides, null/unknown status, wrong approval, false real permission, false GPU permission for a CUDA request, malformed boolean types, and synthetic non-CPU execution. Independent truth tables verify these rejections without touching assets. This closes the original draft/status reproductions.

However, **lines 289–293 only distinguish `cpu` from every other string**. With a supported future real stage/approval and ephemeral true/true permission inputs, both `device='not-a-device'` and unsupported `device='mps'` are accepted. The independent probe additionally calls the **actual `validate_preflight()`**, with asset metadata/hash responses mocked, and receives an accepted result containing `device: 'not-a-device'`.

`device_constructor_probe.py` separately confirms the same boundary through the **actual constructor and preflight from a parsed, isolated policy-fixture JSON**, not only mutated runner attributes. The fixture is explicitly named `MOCK_POLICY_INPUT_NOT_EXECUTION_APPROVAL.json`; its approval/permission values are diagnostic inputs, not an authorization. Model calls are zero, asset metadata/hashes are mocked, the original draft remains false/false, and no smoke is executed. See `device_constructor_probe_results.json` and its log.

**Impact:** unsupported execution contexts are accepted before model/checkpoint work, then fail only when later loading/moving the model. This is not fail-closed device validation. It does not mean the unchanged draft launches or that any GPU was used by the probe.

**Minimum correction:** validate device syntax and explicitly supported CPU/CUDA contexts before asset/device work, apply the appropriate permission checks, and reject unsupported device types/indices without initializing CUDA. Physical device availability/capacity remains a later authorized runtime prerequisite. See `results.R04.cases` and `unsupported_device_actual_preflight`.

### R05 — bounded identity passes

Fourteen independent parsed config cases reject changed cap 0/51, reverse order, seed, target event, reset, batch size, mean/std/clamp, temporal branch, channels, precision or selection rule. The actual resolver accepts a shuffled 60-ID population as ascending IDs 0000–0049, and rejects insufficient population and duplicate IDs. The actual loop independently verifies [16,16,16,2]. See `results.R05`; approved scientific settings were not modified.

### R06 — synthetic execution is asset-independent

Synthetic preflight and execution skip physical checkpoint audit and label the asset paths `MOCKED_SYNTHETIC`. An independent hash spy allows code/config provenance reads but forbids asset hashing; a full 50-sample synthetic run makes **zero asset hash calls**. Byte-read/weight-load/GPU-initialization guards remain active and record zero attempts. This resolves the old checkpoint-read reproduction. Filesystem path metadata lookup is distinguished from asset content reads. See `results.R06`.

### R07 — CPU/CUDA-context RNG guards and exceptions pass

The new predicates at [`adapter.py:357`](../../src/changeguard/adapter.py#L357) and line 435 require `self.device.type == 'cuda'` before even checking availability. Independent CPU execution with availability mocked true makes **zero CUDA availability/RNG calls**. In a logical CUDA context, all tensor transfers are intercepted to keep tensors on CPU, and every CUDA RNG API is mocked: normal execution gives `get,set,get,set`; source and null exception paths each give matching `get,set`; unavailable CUDA gives no RNG call. Actual lazy initialization is forbidden. These verify the new context guard and exception-safe restoration without revisiting accepted adaptation mathematics. See `results.R07`.

## Delivery-record discrepancy: the CUDA transfer attempt succeeded

**Outcome: succeeded, not “failed before GPU work” and not unknown.** The command was not rerun.

After checking the scoped repository log/evidence locations and current CLI log, I located the exact saved command/result in the active Antigravity conversation record. The database was opened read-only with `mode=ro&immutable=1`. Search was limited to the exact command marker within the latest 400 steps of this current conversation, followed by the two matched step IDs; older conversation histories were not audited.

Reproducible evidence is `delivery_cuda_command_matches.json`:

```text
Source: /usr1/home/s126mdg22_06/.gemini/antigravity/conversations/23da4443-87c9-46df-a4ef-5cdc28339dd5.db
Tool call: call_1751187
Command step: 12449
Command/result step: 12450
Result text: The command exited with code 0.
Output: cuda available: True
```

The result payload includes the exact quoted `python -c` command ending in `t.to('cuda:0')`; its SHA-256 is `dcee5158a9691bd54a6fa625509ab001c04bea14ce64553493b984ebda9970e7`. Successful completion establishes that this earlier attempt initialized CUDA and performed allocation/tensor transfer. It does **not** establish a model forward/backward, smoke launch or real SAR/GT/checkpoint/Fisher processing. The evidence extract retains only the relevant command/result and payload identities.

**Required documentation correction:** [`delivery:7`](../week11c_infrastructure_corrections_v1.md#L7) and especially [`delivery:128`](../week11c_infrastructure_corrections_v1.md#L128) must not declare zero GPU operations. Record the successful CUDA transfer probe and its execution-scope deviation explicitly, while distinguishing it from the bounded smoke/model experiment. Also correct the initial-manifest checkpoint-hash claim and [`delivery:131`](../week11c_infrastructure_corrections_v1.md#L131)'s statement that all findings are resolved. Do not replace a contradicted zero-GPU statement with an unqualified “CPU only” statement.

Suggested factual wording: “No Week 11C smoke/model or real-data experiment was launched in the cited command. A separate CUDA availability/transfer probe ran successfully (exit 0), initializing CUDA and moving a 2×2 synthetic tensor to cuda:0; therefore zero-GPU-operation/strictly-CPU-only execution cannot be claimed.” This documentation/scope issue is separate from the R02/R04 code blockers and does not invalidate the passing synthetic correctness checks.

## Exact remaining launch prerequisites

1. Correct **R02** and **R04**, revise the delivery record as above, freeze the changed version, and obtain targeted guarded CPU/mocked acceptance. No further real-data execution or broad CPU/history audit is needed to fix these issues. Previously passed findings stay accepted unless directly affected by the next changes.
2. Create the scoped Week 11 scientific-code/config Git checkpoint **after** acceptance. Use the exact 15-file list in [the prior review's checkpoint section](week11c_smoke_infrastructure_validation.md#exact-launch-prerequisites-and-proposed-git-checkpoint), plus `docs/week11c_infrastructure_corrections_v1.md`, this `docs/reviews/week11c_smoke_infrastructure_validation_v2.md`, and any subsequent narrow correction/acceptance records. Include no datasets/artifacts/logs/pycache/unrelated reports. Existing loader/wrapper/frozen evaluation config need no edits on this evidence; pin their hashes and the external architecture identity. Unrelated untracked material need not be removed. No staging or commit was performed here.
3. Obtain a separate explicit user-approved bounded smoke configuration, with its exact path/hash, supported `APPROVED_FOR_REAL_DATA_SMOKE` status and `APPROVED_FOR_EXECUTION` approval, `real_data_execution_authorized=true`, and `gpu_execution_authorized=true` only for an approved CUDA run. Keep the current draft false/false. Neither this review nor the earlier transfer probe grants real-data/GPU permission.
4. Under that later authorization, verify actual checkpoint expected/actual identity, input-manifest identity, seed 0, Event 520, ascending head50 IDs, order_a/per-event reset, [16,16,16,2], FP32/batch16/pre2/VV-VH/frozen normalization and confidence 0.90; confirm runtime CUDA device/capacity for the chosen supported device. These real assets and runtime capabilities were not tested here. Choose fresh approved output/log paths, use no `--allow_overwrite` and no provenance-check bypass, retain pre-GT prediction/action/progress and terminal records, capture `> file 2>&1` and the Python exit code. No smoke command is launched by this review.

Infrastructure acceptance, execution-scope honesty, authorization, smoke completion and scientific effectiveness remain separate. **Infrastructure currently requires changes; no launch is authorized.**
