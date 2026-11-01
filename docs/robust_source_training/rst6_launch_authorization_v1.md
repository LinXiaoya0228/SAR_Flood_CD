# RST6 Launch Authorization Document

## 1. Authorizing User Instruction

Formal execution authorization was granted by the user under **Option 1: Fresh-Run-Only Recovery**:
> *"Continue RST6 from the current blocked state using Option 1: fresh-run-only recovery... Change only: launch_authorized: true, preparation_only: false in the 12 frozen confirmation configs. Do not change any scientific field."*

---

## 2. Authorized Configuration Matrix & Cryptographic Hashes

All 12 configurations are located in `configs/robust_source_training/rst5_confirmation/`:

| Config File Name | SHA-256 Digest | Recipe | Fold | Seed |
|---|---|---|---|---|
| `rst6_r0_snunet_folda_s0_v1.json` | `69cb17c15d968c7f8f544e9dc4db8fbf686193da467fb41b4a75295cc049b069` | R0 | Fold A | 0 |
| `rst6_r0_snunet_folda_s1_v1.json` | `99ad32fa54b4b088a42e9b2212b27e2062afec8c1e5788c7234d0cbe5e7d7570` | R0 | Fold A | 1 |
| `rst6_r0_snunet_folda_s2_v1.json` | `2ef05ba235bebcec69be5e03775e60612e9fba88fd75056f3b37b18312f6cbfe` | R0 | Fold A | 2 |
| `rst6_r0_snunet_foldb_s0_v1.json` | `c4d1e36144b9f43fffdbb6f4ed2869f3d1419fd9a10b13c227494af2ccc52dbe` | R0 | Fold B | 0 |
| `rst6_r0_snunet_foldb_s1_v1.json` | `92556d3637d5afb07613ee5c35205c21a50df78f861426c30697ce4436684a9c` | R0 | Fold B | 1 |
| `rst6_r0_snunet_foldb_s2_v1.json` | `4531cfbceaad8b11fe247e0b8c9d7ca1159041425aeecbba6e7289aef86e3d7e` | R0 | Fold B | 2 |
| `rst6_r2_snunet_folda_s0_v1.json` | `b2ec6aba610a6b7d2409f85c945a634540550c65e3595b8bc67b4c9ad3ad9b9e` | R2 | Fold A | 0 |
| `rst6_r2_snunet_folda_s1_v1.json` | `16aa14b266fc97dcfe0003a34868ef60923ab8f624591e8898223afeb242beea` | R2 | Fold A | 1 |
| `rst6_r2_snunet_folda_s2_v1.json` | `3947e882e6abb052051d703e37b48e720edbd3c0d4481b8f765212b9f53ef02c` | R2 | Fold A | 2 |
| `rst6_r2_snunet_foldb_s0_v1.json` | `461e22271bed2e942bdaace556f925f62f2d91040e92eb98d52ca5820f22648d` | R2 | Fold B | 0 |
| `rst6_r2_snunet_foldb_s1_v1.json` | `7b0a6d8c57b257e6866023d82d9befa2efce3734adffd046be880383c589f957` | R2 | Fold B | 1 |
| `rst6_r2_snunet_foldb_s2_v1.json` | `e5f4e35d0b27aeb1f72e2c0a33c033677c611045978ae3ee89a611a441bebaea` | R2 | Fold B | 2 |

---

## 3. Scope of Modifications & Scientific Immutability

The following two fields were updated across all 12 configs:
- `launch_authorized: false -> true`
- `preparation_only: true -> false`

### Absolute Scientific Invariants Confirmed
Zero modifications were made to any of the following fields:
- Architecture: `SNUNetWrapper` (in=2, out=3, base_channel=32)
- Optimization: Adam ($\text{lr}=10^{-3}$, weight_decay=0.0)
- Epoch budget: exactly 16 epochs
- Batching: microbatch size 8, gradient accumulation 4 (effective batch 32)
- Training event partitions and evaluation event partitions
- Loss formulation and pre-computed fold class weights
- Seeds: exactly 0, 1, 2
- Output root paths and log paths

---

## 4. Four-GPU Queue Layout

Execution is mapped to the four available physical NVIDIA GeForce RTX 3090 GPUs in host `gpu22`:

| Physical GPU | Tmux Session Name | Method | Fold | Runs Executed Sequentially |
|---|---|---|---|---|
| **GPU 0** | `cg_rst6_folda_r0_v1` | R0 | Fold A | `rst6_r0_snunet_folda_s0_v1`, `s1`, `s2` |
| **GPU 1** | `cg_rst6_folda_r2_v1` | R2 | Fold A | `rst6_r2_snunet_folda_s0_v1`, `s1`, `s2` |
| **GPU 2** | `cg_rst6_foldb_r0_v1` | R0 | Fold B | `rst6_r0_snunet_foldb_s0_v1`, `s1`, `s2` |
| **GPU 3** | `cg_rst6_foldb_r2_v1` | R2 | Fold B | `rst6_r2_snunet_foldb_s0_v1`, `s1`, `s2` |

---

## 5. Research Integrity & Zero-Contamination Guarantees

1. **Zero Access to Forbidden Events**: None of the 16 forbidden historical target events (`[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`) appear in any training or evaluation partition.
2. **Deterministic Queue Sequentiality**: Each GPU queue runs exactly one process at a time. The next seed begins only after the preceding seed completes with exit code 0.
3. **Fresh-Run-Only Recovery**: No partial checkpoint resume is permitted for confirmatory runs. An interrupted run must be preserved as failed evidence and retried fresh under a versioned run ID (`<run_id>_v2`).
4. **Historical Document Protection**: Tracked-dirty historical documents `docs/e11_final_scientific_conclusions.md` and `docs/e11_final_test_closeout.md` remain byte-identically preserved.

---

## 6. Authorization Addendum: Versioned R2 v2 Confirmatory Retries

Following the forensic class-weight audit (`docs/robust_source_training/rst6_r2_class_weights_audit_and_invalidation_v1.md`) and implementation of fail-closed pre-compute class-weight assertions in `src/robust_source_training/runner.py`, the six versioned R2 v2 confirmatory runs are explicitly authorized:

- `rst6_r2_snunet_folda_s0_v2`
- `rst6_r2_snunet_folda_s1_v2`
- `rst6_r2_snunet_folda_s2_v2`
- `rst6_r2_snunet_foldb_s0_v2`
- `rst6_r2_snunet_foldb_s1_v2`
- `rst6_r2_snunet_foldb_s2_v2`

### 6.1 Scientific Invariance Verification
A systematic check confirmed that all six R2 v2 configs are identical to their R2 v1 counterparts across all scientific keys:
- `recipe_id`: `R2`
- `components`: `m1_event_balance: true`, `m2_flood_aware_loss: true`, `m3_paired_augmentation: false`
- `fold_id`: `confirmation_fold_a` / `confirmation_fold_b`
- `seed`: 0, 1, 2
- `max_epochs`: 16
- `partition`: 18 train events, 9 eval events
- `training`: microbatch 8, grad accum 4, effective batch size 32
- `architecture`: SNUNet_ECAM, base channel 32
- `optimization`: Adam, lr 1e-3, betas [0.9, 0.999]

The `src/robust_source_training/runner.py` commit adds only pre-compute provenance assertions and does not alter data loading, model initialization, sampler draw sequence, loss computation, optimizer updates, epoch budgets, or evaluation logic.

### 6.2 Updated Queue Mapping
- **GPU 1**: Tmux session `cg_rst6_folda_r2_v2` running `rst6_r2_snunet_folda_s0_v2`, `s1_v2`, `s2_v2`
- **GPU 3**: Tmux session `cg_rst6_foldb_r2_v2` running `rst6_r2_snunet_foldb_s0_v2`, `s1_v2`, `s2_v2`

