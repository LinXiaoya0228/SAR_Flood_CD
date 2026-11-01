# RST6 Confirmatory Run Closeout Template

> Note: This template must be populated upon the completion of each individual confirmatory training run in RST6.

## 1. Run Header & Execution Identity

- **Run ID**: `<RUN_ID>` (e.g., `rst6_r2_snunet_folda_s0_v1`)
- **Recipe ID**: `<RECIPE_ID>` (`R0` or `R2`)
- **Confirmatory Fold**: `<FOLD_ID>` (`confirmation_fold_a` or `confirmation_fold_b`)
- **Seed**: `<SEED>` (`0`, `1`, or `2`)
- **GPU Device**: `<DEVICE>` (e.g., `cuda:0` - NVIDIA GeForce RTX 3090)
- **Host**: `gpu22`
- **Execution Class**: `CONFIRMATORY_TRAINING`
- **Start Time UTC**: `YYYY-MM-DDTHH:MM:SSZ`
- **Completion Time UTC**: `YYYY-MM-DDTHH:MM:SSZ`
- **Total Wall-Clock Runtime**: `<SECONDS>` s (`<HOURS>` h)

---

## 2. Provenance & Artifact Integrity

- **Config Path**: `configs/robust_source_training/rst5_confirmation/<RUN_ID>.json`
- **Config SHA-256**: `<CONFIG_SHA256>`
- **Output Root**: `outputs/robust_source_training/rst6/<RUN_ID>/`
- **Log Path**: `logs/robust_source_training/rst6/<RUN_ID>.log`
- **Initial Model State SHA-256**: `<INITIAL_STATE_SHA256>`
- **Final Checkpoint SHA-256**: `<FINAL_CKPT_SHA256>`
- **Last Checkpoint SHA-256**: `<LAST_CKPT_SHA256>`
- **Event Metrics JSON SHA-256**: `<METRICS_SHA256>`

---

## 3. Training Execution Trace

- **Epochs Completed**: 16 / 16 (Fixed Budget)
- **Total Training Samples Observed**: `<TOTAL_SAMPLES>` (e.g., `212832` for Fold A, `234480` for Fold B)
- **Total Microbatches**: `<TOTAL_MICROBATCHES>`
- **Total Optimizer Steps**: `<TOTAL_STEPS>` (e.g., `6672` for Fold A, `7344` for Fold B)
- **Peak VRAM Allocated**: `<PEAK_VRAM>` MB
- **Resume Status**: `CLEAN_SINGLE_ATTEMPT` / `RESUMED_FROM_STEP_<STEP>`
- **Terminal Status**: `COMPLETED_SCIENTIFIC_RUN`

---

## 4. Confirmatory Event-Wise Evaluation Metrics

Evaluated strictly and offline on the 9 unseen events of the confirmatory fold:

| Event ID | Patch Count | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| `<EV_1>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_2>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_3>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_4>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_5>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_6>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_7>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_8>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `<EV_9>` | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Macro & Pooled Summary:
- **Event-Macro F1**: `<MACRO_F1>`
- **Event-Macro IoU**: `<MACRO_IOU>`
- **Event-Macro Precision**: `<MACRO_PREC>`
- **Event-Macro Recall**: `<MACRO_REC>`
- **Pooled F1**: `<POOLED_F1>`
- **Pooled IoU**: `<POOLED_IOU>`
- **Worst-Event ID**: `<WORST_EVENT_ID>` (F1 = `<WORST_EVENT_F1>`)
- **Bottom-3 Event Mean F1**: `<BOTTOM_3_F1>`

---

## 5. Mandatory Research-Integrity Assertions

- [ ] Zero access to the 16 forbidden target events (`[205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562, 1111002, 1111003, 1111007, 1111008, 1111013]`).
- [ ] Training samples strictly contained within authorized training partition.
- [ ] Evaluation was performed strictly offline after checkpoint persistence.
- [ ] Initial model state matches the paired baseline for this fold and seed.
- [ ] Output artifacts verified and persisted without collision or overwrite.
