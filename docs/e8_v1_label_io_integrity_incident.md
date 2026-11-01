# E8 v1 Label-I/O Integrity Incident and v2 Repair Record

**Date:** 2026-10-22
**Status:** E8 v1 preserved; not certified as formal evidence
**Affected computation:** `outputs/e8_confirmation/` and `logs/e8_confirmation/e8_confirmation_v1.log`
**Required classification:** `COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ`

## 1. Direct observation and impact

The E8 v1 launcher completed all 12 planned CoTTA streams with exit code 0. The output directories,
log, summaries, prediction shards and completion records remain unchanged and must not be deleted,
overwritten or promoted silently.

Independent code review found that `tools/run_e8_confirmation_matrix.py` v1 constructed the complete
stream plan before entering the prediction loop. `get_datasets()` created an online dataset with
`load_labels=False` and an offline dataset with `load_labels=True`. During `build_stream_batches()`,
the runner indexed both datasets. Indexing the offline dataset opened `MK0_MLU`, converted the GT
class raster to a tensor and stored it inside `LabelGuardedBatch`. Prediction, adaptation and shard
persistence occurred only later in `execute_stream()`.

`LabelGuardedBatch.get_labels()` remained locked until `mark_persisted()`, so the available evidence
does not establish that GT influenced CoTTA inference, pseudo-label selection, loss or updates.
Nevertheless, the physical disk read occurred too early. This violates
`docs/research_integrity_protocol.md`, which requires target labels to be read from disk only after
the applicable inference and adaptation work is complete and the pre-label evidence is durable.
An API guard around an already-loaded tensor is not sufficient.

## 2. Exact v1 data path

1. The compressed dataset manifest supplied sample IDs, event IDs and relative sample paths.
2. `KuroEvaluationDataset(load_labels=False)` exposed SAR inputs and the independent sensory-valid
   raster `MK0_MNA`; it did not locate or open `MK0_MLU`.
3. `KuroEvaluationDataset(load_labels=True)` exposed the same metadata but, when indexed, also opened
   `MK0_MLU` and computed a convenience `valid` mask containing the label-derived `label != 3` term.
4. E8 v1 batch construction indexed the online dataset and then the offline dataset before returning
   the stream plan. It placed SAR tensors, sensory-valid masks and GT tensors in `LabelGuardedBatch`.
5. `E7CoTTAAdapter.adapt_and_predict_step()` received only SAR tensors, sample IDs, event ID and
   `MK0_MNA`; it did not receive the stored GT tensor or the label-derived `valid` mask.
6. The adapter produced the declared PRE_UPDATE teacher prediction and executed its unsupervised
   update.
7. The runner wrote the prediction shard, then called `mark_persisted()` and `get_labels()`.
8. Metrics used `(MK0_MNA == 1) & (label != 3)`.

The integrity failure is step 4: GT disk I/O preceded steps 5–7 even though programmatic retrieval was
guarded until after persistence.

## 3. Repaired v2 path

`tools/run_e8_confirmation_matrix.py` now builds metadata-only `DeferredBatchSpec` objects. Each
batch follows this order:

1. Index the sensory-only dataset and load SAR plus `MK0_MNA`.
2. Run PRE_UPDATE prediction and the declared adaptation step without GT or GT-derived masks.
3. Atomically write, fsync, rename and reopen the prediction shard; verify its exact sample IDs.
4. Invoke `DeferredLabelLoader.load_exact()` for those sample IDs only.
5. Record a label-read timestamp strictly after the shard-persistence timestamp.
6. Apply class-3 exclusion and compute offline metrics.

A persistence exception writes `failure.json` and a failed `completion_status.json`; the label loader
is not called for that batch. Partial directories are not automatically resumed because full adaptive
state recovery is not implemented. A retry requires a fresh versioned output identity, preventing
duplicate predictions, updates or GT reads. Completed streams remain idempotently skippable.

Before any prediction, update or GT read, the runner atomically writes immutable
`initial_manifest.json` with Git, config, ledger, runner, adapter, checkpoint, split and dataset-manifest
hashes plus the complete expected execution contract. Terminal records reference its SHA-256 rather
than rewriting it.

## 4. Audit of the 57 proposed reusable streams

Classification is based on the producing runner code and artifacts, not documentation claims.

| Family | Streams | Certified | GT preloaded but guarded — not certified | Evidence |
| --- | ---: | ---: | ---: | --- |
| Source-only | 3 | 1 | 2 | Seed 0 came from Week-12; seeds 1/2 from E1.4. |
| Matched BN-no-update | 6 | 0 | 6 | E1.5 `build_ordered_stream_batches()` indexed `offline_ds` before execution. |
| AdaBN | 12 | 0 | 12 | E1.4 `build_ordered_stream_batches()` indexed `offline_ds` before execution. |
| Tent | 12 | 1 | 11 | Seed 0/order A/reset came from Week-12; the other 11 came from E1.4. |
| EATA | 12 | 0 | 12 | E1.4 preload path. |
| SAR | 12 | 0 | 12 | E1.4 preload path. |
| **Total** | **57** | **2** | **55** | Zero insufficient-evidence or incompatible classifications. |

The two certified physical streams are:

- `source_only_frozen_running_stats_s0`;
- `tent_matched_pre_update_s0_oa_re`.

Their producing Week-12 runner indexed `offline_dataset` only in the offline-scoring phase after each
event's `frozen_predictions.npz` and `step_records.json` had been written. Its initial manifest and
completion marker are present under `outputs/week12_ablation/approved_dev6_s0_v1/`.

The E1.4 runner produced the other 49 E1.4 streams and indexed `offline_ds` while constructing every
`LabelGuardedBatch`. The E1.5 runner used the same preload design for all six BN-control streams.
Therefore 55 historical streams require a versioned rerun under the repaired lifecycle.

## 5. E7 smoke v2 scope correction

The successful E7 smoke v2 runner also indexed `offline_ds` and loaded GT tensors during batch-plan
construction. Its status is therefore:

`USEFUL EXECUTION SMOKE; STRICT LABEL-I/O CERTIFICATION REVOKED`

It remains an infrastructure-only smoke and did not constitute scientific evidence. Failed smoke v1
and successful smoke v2 remain preserved. A bounded `e7_cotta_smoke_v3` using the repaired runner is
scheduled and must pass before E8 v2 is launched.

## 6. Versioned retry accounting and boundaries

E8 v2 retains the exact v1 scientific settings. It reuses the two certified Week-12 streams, reruns
55 uncertified historical streams and reruns all 12 E8 v1 CoTTA streams: **67 new physical streams,
402 new event rows**, for the same **69-stream/414-row** matrix.

Prepared paths:

- config: `configs/changeguard/e8_confirmation_matrix_v2.json`;
- ledger: `configs/changeguard/e8_execution_ledger_v2.json`;
- output root: `outputs/e8_confirmation_v2`;
- log: `logs/e8_confirmation/e8_confirmation_v2.log`;
- tmux session: `cg_e8_confirmation_v2`.

Neither E7 smoke v3 nor E8 v2 was launched during this repair. No sealed-test event was accessed.
No E8 scientific consolidation, route selection, threshold change or E9 work was performed.
