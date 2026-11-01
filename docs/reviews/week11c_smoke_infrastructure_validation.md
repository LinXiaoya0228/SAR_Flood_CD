# Week 11C smoke infrastructure: independent review

2026-10-18. **Infrastructure: CHANGES REQUIRED.** No real-data/GPU launch or authorization is granted.

Scope: the new smoke runner, its draft configuration and focused tests, the CUDA-RNG change in the accepted adapter, and only the loader/model dependencies needed to trace this runner. [Week 11A CPU acceptance](week11a_cpu_validation_v3.md) is carried forward; its closed algorithm findings were not reopened. No Week 0–9 audit, real-data experiment or scientific-effectiveness assessment was performed.

## Stable review identity and boundaries

Host `gpu22`; repository `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`; branch `main`; HEAD `e5cbbfc5154bc09ca9eaf48bffa976b596b954ba`. New Week 11 files are untracked, so HEAD alone does not identify their contents. All 12 reviewed input hashes, HEAD and all 66 prior independent evidence files were checked before/after and remained unchanged. Full records are in `stability_start.json` and `stability_end.json`.

| Primary reviewed input | SHA-256 before and after, identical |
| --- | --- |
| `tools/run_week11_real_data_smoke.py` | `5f6ff62dbeba1ca5c92b447ac0961b4483c3844a224972c979ec8a1902d23d0a` |
| `configs/changeguard/week11_smoke_event520_draft.json` | `6ec2376d49174c1e264b24a1a5bee2b99c49c8fb6c1cd29a57a789d8ffe7f21e` |
| `tests/test_week11_real_data_smoke.py` | `cb873aebb4809eddaa7042551240ea130d1d328fe72798cb7d0e466be6dd5420` |
| `src/changeguard/adapter.py` | `2ca838b7ae9236c3677e00508ebd3a9f0d1faed5718fc465b83cae45c3e6754e` |

Necessary dependencies inspected and pinned: `src/data/evaluation_dataset.py`, `src/models/snunet_wrapper.py`, `src/changeguard/__init__.py`, and `../third_party/KuroSiwo/models/snunet.py`. The wrapper resolves that external architecture from the parent workspace. Its current dependency Git HEAD is `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`, with no modification to `models/snunet.py`. The accepted CPU config/protocol and frozen Week 9 v2 config were read only to compare existing model/checkpoint/preprocessing identities, not to re-audit earlier experiments.

Reviewer writes are only this report and the fresh evidence directory. No implementation, scientific config, protocol, repository test, original artifact, prior report or prior evidence was edited; no staging or commits occurred. Draft authorization flags on disk remain **false/false**. Authorization truth-table fixtures are ephemeral method inputs only. The positive real-branch probe explicitly mocks authorization and every physical asset boundary; it does not constitute an authorized real execution.

## Executed checks and asset protections

Evidence: [`../../outputs/week11_review_evidence/week11c_infrastructure_independent_v1/`](../../outputs/week11_review_evidence/week11c_infrastructure_independent_v1/).

Python `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python`; CPU tensors only; `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and evidence-local `TMPDIR`. Logs capture both streams using `> file 2>&1` or equivalent combined child-process capture.

The seven focused tests were inspected before execution. Their nominal synthetic preflight actually attempts to hash the existing source checkpoint (R06 below). Therefore I ran the specified module through **`guarded_execution.py`**, which denies real checkpoint/raster/manifest byte reads, denies `torch.load` and CUDA initialization, and applies the same guard to child CLI commands. **Seven tests passed in 17.652 seconds, exit 0**, with five checkpoint byte-read attempts blocked in the parent; zero asset bytes were allowed. These results are guarded CPU test results, not a claim that the unguarded suite is asset-independent. See `focused_tests.log` and `focused_guard_observations.json`.

Independent evidence:

- `independent_probes.py` / `independent_probes.log` / `independent_probe_results.json`: actual runner orchestration and actual loader construction/preprocessing/indexing, with in-memory metadata and raster callbacks, a CPU proxy replacing checkpoint loading, and mocked CUDA RNG APIs. Missing/mismatched checkpoint checks are separately exercised with synthetic hash answers. No original checkpoint, Fisher, SAR raster, annotation raster or real manifest content is read.
- The virtual metadata includes 60 Event-520 samples, another development event and a sealed event. The real loader filters before the first synthetic raster callback. The runner selects the ascending first 50 IDs and runs **[16,16,16,2]**, resets once, preserves its independent source, and records **3 actual applied updates / 1 injected numerical failure / 0 skips**. Actual adaptation inputs are verified as FP32 two-channel `[VV,VH]`, SL2/pre2 and MS1/post with the frozen normalization. Adapter/probe calls receive sensory validity and identities, not labels or label-derived validity.
- A GT callback asserts that all four adaptation calls have returned, captures the files present immediately before first GT access, and in a second run raises a deliberate GT failure. This tests actual ordering and failure artifacts, not merely forbidden key names.
- `additional_policy_probes.py` and `parsed_subset_policy_results.json` confirm unbounded/oversized/reversed selection from **parsed temporary config copies** whose authorization flags remain false/false. They do not rely only on mutating runner attributes.
- CUDA availability/state APIs are mocked for success and exceptions; no GPU is initialized. Installed PyTorch RNG source is retained in `torch_cuda_rng_implementation.txt` to document eager initialization behavior.

The first independent attempt stopped on a reviewer assertion comparing a resolved checkpoint symlink target with its unresolved spelling, after its mocked loop completed. The corrected comparison uses resolved identity. Its script, log and artifacts remain preserved as `independent_probes_attempt1.py`, `independent_probes_attempt1.log`, and `mock_real_success/`; this was a probe error, not an implementation defect. Final successful reproductions are under `attempt2/`. Diagnostic exit 0 means observations were reproduced, not that infrastructure passed.

## Concrete blockers

### W11C-R01 — configured dataset/manifest paths resolve to nonexistent locations

**Evidence:** [`draft config:29`](../../configs/changeguard/week11_smoke_event520_draft.json#L29) sets `datasets/KuroSiwo` and `third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz`. The real branch resolves both under `REPO_ROOT` in [`runner:376`](../../tools/run_week11_real_data_smoke.py#L376). Path metadata checks, without opening assets, show:

```text
ChangeGuard/datasets/KuroSiwo                                      absent
ChangeGuard/third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz absent
../datasets/KuroSiwo                                              exists
../third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz          exists
```

The existing frozen evaluation config points to the parent-workspace locations. Current preflight checks the checkpoint/output directory but not these required dataset paths. The real branch would load the model before the loader fails on its missing manifest ([loader:122](../../src/data/evaluation_dataset.py#L122)).

**Impact/minimum correction:** the delivered real path cannot execute as configured. Correct the draft to the intended frozen locations without changing data/preprocessing identity, and validate required path metadata before model/device work. Actual content identity verification remains part of a later authorized launch preflight; it was not performed here.

### W11C-R02 — predictions/actions are not persisted before GT; failure leaves no durable run evidence

**Evidence:** [`runner:444`](../../tools/run_week11_real_data_smoke.py#L444) stores detached primary predictions only in `frozen_predictions`; lines 451–476 keep action/diagnostic records in memory. GT is loaded at [`runner:498`](../../tools/run_week11_real_data_smoke.py#L498). `_save_outputs()` is called only after all GT scoring at lines 581–589. It does not save the prediction arrays at all.

**Reproduction:** `mocked_real_execution` reports **`files_before_first_gt: []`** and `prediction_files_present: false`. Its counterpart `mocked_real_failure` finishes all 50 predictions, raises at the first synthetic GT read, and leaves **an empty output directory**: no predictions/actions, initial manifest, progress checkpoint, failure status or completion record.

The in-memory order is label-isolated and primary predictions remain PRE_UPDATE; this finding is the missing required **durable pre-GT boundary**, not observed GT leakage into optimization.

**Impact/minimum correction:** write and close the ID-bound primary prediction/action records before corresponding GT access; keep those immutable while adding separate offline metrics. Write an initial provenance manifest and progress/failure/completion records so a crash preserves what ran. Include the relevant reviewed code identity (or pinned commit with scope checks), settings, checkpoint expected/actual hash and input-manifest identity. The present final manifest retains IDs/config hash/counters but only a checkpoint path, and HEAD plus `is_clean=false` cannot identify the current untracked implementation. A later scoped Git checkpoint is also required, below.

### W11C-R03 — invalid/unannotated pixels are counted as true negatives

**Evidence:** [`runner:507`](../../tools/run_week11_real_data_smoke.py#L507) masks `p_flood` and `g_flood`, but line **515** computes `tn = sum(~p_flood & ~g_flood)` without intersecting `valid`. Both operands are false outside validity, so every invalid pixel becomes a TN.

**Reproduction:** 50 synthetic 4×6 samples each have one sensory-invalid pixel and one distinct label-3 pixel. Actual scoring reports:

```text
evaluated valid pixels = 1100
TP=50, FP=1050, FN=0, TN=100
sum of confusion cells = 1200
correct valid TN = 0
```

See `results.mocked_real_execution.event_row`. This is an actual runner calculation, not an isolated arithmetic assertion. Flood F1/IoU happen not to use TN and are unaffected in this example; the exported confusion counts are nevertheless false.

**Impact/minimum correction:** restrict TN to the same offline validity mask; assert per-batch and event confusion-cell sums equal evaluated valid pixels, including independent sensory-invalid and label-3 cases. This correction belongs only to the new runner; no earlier results need to be reopened on this evidence.

### W11C-R04 — draft/unsupported authorization stages are ignored

**Evidence:** [`runner:163`](../../tools/run_week11_real_data_smoke.py#L163) validates boolean flags but never validates `status`, `approval_status` or a supported execution-stage combination. Those fields are merely copied into the final manifest.

**Reproduction:** calling the actual authorization method with ephemeral policy inputs and true/true flags accepts all of `DRAFT_SMOKE_PROTOCOL_EXECUTION_UNAUTHORIZED`, null status and `UNSUPPORTED_STAGE` for a requested `cuda:0` device. No asset/device calls are involved in this test. The unchanged on-disk draft false/false still correctly denies a normal real launch; that positive guard does not reject contradictory authorization states.

**Impact/minimum correction:** define and enforce supported stage/approval/boolean/device combinations, reject draft or missing/unknown stages for real execution before asset access, and retain the explicitly CPU-only synthetic path. Boolean toggles alone must not silently override the draft's execution-unauthorized status. This is a defect in the **new** runner, not a reopening of the accepted Week 11A CPU runner's R05 fix. No approval or flag change is made by this review.

### W11C-R05 — the declared bounded subset is not enforced

**Evidence:** constructor fields at [`runner:149`](../../tools/run_week11_real_data_smoke.py#L149) are accepted without bounded-smoke validation. `_resolve_samples_for_event()` at [`runner:284`](../../tools/run_week11_real_data_smoke.py#L284) supports reverse ordering; lines 292–294 apply no cap for `max_samples <= 0`. Preflight never checks those settings against the declared head-50 rule or expected batches.

**Reproduction from parsed temporary configurations:**

| Config setting | Accepted preflight / actual resolved subset |
| --- | --- |
| `max_samples=0`, order_a | **60** samples: cap disabled |
| `max_samples=51`, order_a | **51** samples |
| `max_samples=50`, order_b | Descending IDs `s520_0059` through `s520_0010` |

Every case still declares `deterministic_sorted_sample_id_ascending_head_50`. The default draft correctly produces 50 ascending IDs and [16,16,16,2], but unsupported configurations can execute a different population under the same smoke identity. Seed/reset/batch/model/preprocessing fields also lack a complete frozen-smoke consistency audit; the current draft's nominal values match the frozen settings, aside from R01 paths.

**Impact/minimum correction:** reject unsupported smoke settings before assets: seed 0, Event 520, approved pre2/VV-VH/FP32 settings, order_a, per-event reset, batch 16, explicit sorted-ID selection and a positive hard ceiling of 50. Verify the selected IDs and expected [16,16,16,2] coverage for the declared 50-sample smoke rather than merely printing a declared rule. Do not silently treat a malformed cap as unbounded or change scientific thresholds.

### W11C-R06 — synthetic preflight reads a real checkpoint

**Evidence:** [`runner:222`](../../tools/run_week11_real_data_smoke.py#L222) inspects checkpoint existence and calls `compute_file_sha256()` at line 233 regardless of `synthetic_mock`. Only the later requirement to succeed is conditional on real mode. Hashing reads the entire checkpoint file.

**Reproduction:** an independent checkpoint-hash spy observes the real configured checkpoint hash call during `synthetic_mock=True` preflight. The focused test guard additionally blocks five such byte-read attempts in the parent test process; the synthetic CLI child is guarded too. Zero real bytes were allowed during this review.

**Impact/minimum correction:** synthetic mode is not asset-independent as advertised; ordinary unguarded synthetic tests/preflight touch the real checkpoint. Skip physical checkpoint audit entirely for synthetic mode, mark identity not applicable/mocked honestly, and assert zero asset accesses in synthetic regressions. Retain fail-closed real-mode checkpoint checks, which pass mocked missing-file/hash-mismatch rejection tests.

### W11C-R07 — CUDA RNG snapshot guard can initialize CUDA during CPU-only execution

**Evidence:** new source-reference snapshot at [`adapter.py:356`](../../src/changeguard/adapter.py#L356) calls CUDA RNG APIs whenever `torch.cuda.is_available()` is true, regardless of the adapter's selected device. The null path has the same availability-only predicate at line 434. In installed PyTorch, `get_rng_state_all()` calls `get_rng_state()` for devices, and the latter explicitly calls `_lazy_init()`; the local implementation is retained as evidence.

**Reproduction:** on an adapter whose logical device is **CPU**, mocks returning availability=true observe `get,set,get,set` across source and null forwards. The availability=false case calls neither. Source and null exception cases correctly call matching `get,set` restoration; **exception safety passes**. No actual CUDA API initializes a GPU in these tests because every availability/state call is mocked and lazy initialization is forbidden.

**Impact/minimum correction:** CPU/synthetic execution on a GPU-visible host can initialize CUDA despite the CPU-only mode and false GPU authorization. Gate CUDA state work on an intended CUDA execution context as well as availability, preserving exception-safe restoration. This review only follows the new CUDA-reference change and its directly affected CPU/mock boundary; it does not reopen unrelated accepted adapter behavior.

## Checks that pass in the reviewed call chain

- **Nominal frozen model identity:** draft seed-0 checkpoint path and expected SHA `247d358431f5c54c3d3d3421b4de6c4738ad457509900f8ddb717f5eef21087f`, architecture/base 32/two channels/three classes, pre2 pairing and normalization match the frozen metadata. The real wrapper call receives `in_channels=2`, `out_ch=3`, `base_channel=32`; its strict state loading is traced without opening the checkpoint. Actual source weights/hash have **not** been revalidated here.
- **Real missing/mismatched checkpoint checks:** mocked real preflight rejects missing files and wrong hashes before the wrapper is called. The normal runner uses `fail_closed=True`; the optional `validate_preflight(fail_closed=False)` weakens only that check and is not used by normal CLI/run paths. It must not be used as launch validation.
- **Event and GT isolation:** target events must be [520]. The real loader filters manifest metadata before raster reads, and `__getitem__` rechecks events before opening components. Adaptation uses `load_labels=False` and `valid_mask` (MK0_MNA), not its dummy label or `valid` field. The offline dataset's early construction reads metadata only; actual GT access occurs after all four adaptation calls. Synthetic callbacks prove no out-of-event or outside-head-50 raster is read on the nominal path.
- **State/tails/failure accounting:** the actual adapter is constructed once and reset once. Its independent source state is unchanged; 50 unique IDs are handled with tail 2. One injected nonfinite gradient is honestly recorded as failure/null loss, with three later updates and four total steps. Failed adaptation is distinguishable in counters; this does not by itself certify a numerically healthy smoke.
- **Fresh output protection and explicit bypass:** nonempty output directories are rejected by default at [`runner:248`](../../tools/run_week11_real_data_smoke.py#L248). Constructor `allow_overwrite=False` and CLI `store_true` make **`--allow_overwrite` opt-in**, not a default or config-enabled path. It permits replacing existing outputs and must **not** be supplied for the smoke. `log_dir` is currently only resolved; an external command must capture both streams/exit status unless the implementer adds explicit logging. R02 still requires durable progress/failure evidence.

## Exact launch prerequisites and proposed Git checkpoint

1. Implementer resolves **W11C-R01–R07**, preserving the approved scientific settings. Freeze a new version and obtain targeted CPU/mocked re-verification of those corrections. No real-data run is needed to resolve them.
2. Create a scoped scientific-code/config checkpoint after that pass. A commit must identify the reviewed Week 11 implementation rather than relying on the current old HEAD and untracked files. Suggested exact contents, including the previously accepted CPU baseline needed by the new runner and its review trail:

```text
src/changeguard/__init__.py
src/changeguard/adapter.py
tools/run_week11_cpu_evidence.py
tools/run_week11_real_data_smoke.py
tests/test_week11_evidence_consistency.py
tests/test_week11_real_data_smoke.py
configs/changeguard/week11_evidence_consistency_v1.json
configs/changeguard/week11_smoke_event520_draft.json
docs/protocols/week11_evidence_consistency_protocol_v1.md
docs/week11a_cpu_corrections_v1.md
docs/week11a_cpu_corrections_v2.md
docs/reviews/week11a_cpu_validation.md
docs/reviews/week11a_cpu_validation_v2.md
docs/reviews/week11a_cpu_validation_v3.md
docs/reviews/week11c_smoke_infrastructure_validation.md
```

Include any subsequent narrowly scoped Week 11C correction note and acceptance report produced for the fixes. Existing loader/wrapper/frozen evaluation config are already tracked and need no edits/staging on this evidence. Record their hashes and the external architecture commit/hash in run provenance. Do not include artifacts, datasets, logs, unrelated untracked reports or pycache. Unrelated untracked material need not be deleted or committed; verify the reviewed scientific scope rather than requiring a globally empty Git status. **This is a proposed future checkpoint, not an instruction executed by this reviewer.**

3. Obtain separate explicit user authorization for a concrete, bounded Week 11C real-data/GPU smoke, with the supported approved status/config identity established after review. Do not modify the accepted CPU-only authorization protocol to imply approval, and do not treat this draft or boolean toggles alone as permission.
4. Under that future authorization, validate the actual seed-0 checkpoint digest, corrected target-manifest/dataset identity and deterministic Event-520 50-ID selection; confirm [16,16,16,2], FP32/batch16/threshold0.90 and the reviewed model/preprocessing. Runtime device availability and capacity remain untested. Use a **fresh run output directory**, no `--allow_overwrite`, durable pre-GT prediction/action records, complete manifest/progress/failure/completion records, and a fresh log capturing `> file 2>&1` plus the Python exit status. Exact approved run ID/output/log paths must be chosen before launch; none is selected or launched by this review.

Infrastructure approval, actual smoke completion and scientific effectiveness are separate decisions. **The current infrastructure does not pass; no execution is authorized.**
