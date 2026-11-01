# E8 v2 Human Launch Authorization

**Authorization time:** 2026-10-22T10:19:13+08:00 (Asia/Singapore)

**Authorizing authority:** Human project owner

**Pre-authorization HEAD:** `6ea7c9469f11eaf2406f9125867314e25f2a3a9c`

## Scope

The human project owner authorizes the following work, in this order:

1. One bounded E7 CoTTA smoke v3 on development Event 520, seed 0, using exactly two
   batches and 32 samples. Its sole purpose is physical label-I/O verification. It is
   infrastructure evidence, not scientific evidence.
2. Only after that smoke completes successfully and its artifacts pass every frozen
   provenance and label-I/O check, the E8 v2 formal development evaluation: exactly two
   certified reused streams and 67 execute-new streams, producing a 69-stream and
   414-event-row final matrix.

The governing machine-readable contracts are:

- `configs/changeguard/e7_cotta_smoke_v3.json`;
- `configs/changeguard/e8_confirmation_matrix_v2.json`;
- `configs/changeguard/e8_execution_ledger_v2.json`.

## Mandatory smoke-first gate

Human authorization alone is not sufficient to launch formal E8 v2. The smoke must prove
that its immutable initial manifest preceded execution; no GT was read before the matching
prediction shard was durably persisted; neither GT nor a GT-derived mask entered adaptation;
32 of 32 samples in two batches completed; terminal status was successful; all current
config, ledger, runner, adapter, checkpoint, split and dataset hashes match; and sealed-test
access remained zero. Missing, failed, mismatched or incomplete evidence blocks E8 v2.

## Authorized paths and sessions

| Attempt | Output | Log | tmux session |
| --- | --- | --- | --- |
| E7 smoke v3 | `outputs/e7_cotta_smoke_v3` | `logs/e7_cotta_smoke/e7_cotta_smoke_v3.log` | `cg_e7_cotta_smoke_v3` |
| E8 confirmation v2 | `outputs/e8_confirmation_v2` | `logs/e8_confirmation/e8_confirmation_v2.log` | `cg_e8_confirmation_v2` |

Both attempts require fresh paths and prohibit overwrite.

## Prohibited actions

This authorization does not permit sealed-test access, source training, changes to model or
adaptation settings, threshold tuning, changes to seeds/orders/reset modes, overwriting any
historical artifact, scientific consolidation or route selection, or any E9 work. The only
authorized dataset events are the six frozen development events, with smoke v3 further
restricted to Event 520.
