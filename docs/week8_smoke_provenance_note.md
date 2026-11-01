# Week 8 TTA Diagnostic Smoke Provenance & Pre-Launch Audit Note

- **Document ID:** `docs/week8_smoke_provenance_note.md`
- **Date (UTC):** 2026-10-16
- **Status:** APPROVED PROVENANCE RECORD
- **Governing Protocol:** `configs/evaluation/week8_controlled_tta_protocol.json`
- **Target Model:** Model A (`SNUNet_ECAM [VV, VH]`, 2 channels, FP32)
- **Code HEAD at Smoke Completion:** `ca4f558fca0cdc31a5aaaaa9a72f5c0de752123d`

---

## 1. Resolution of the Sealed-Test Event List Discrepancy

During the initial Week 8 smoke execution summary, a prose passage referenced `[263, 269, 368, 415, 453, 520]` as quarantined sealed-test events. This note establishes the authoritative facts verified from repository files:

1. **The Incorrect Event List Appeared in Chat Prose Only:**
   - The string `[263, 269, 368, 415, 453, 520]` was an assistant reporting error confined strictly to the conversational text.
   - It does not exist in any executable runner, configuration file, test fixture, or data split file.
   - Event `520` is explicitly an authorized **development** event (92 samples), not a sealed test event.

2. **Authoritative Code and Configuration Split Definitions:**
   - **Quarantined Sealed-Test Partition (10 Events):**
     `[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`
     Enforced identically across:
     - `src/data/evaluation_dataset.py` (`FORBIDDEN_SEALED_TEST_EVENTS`)
     - `configs/evaluation/source_baseline_tta_dev_protocol.json` (`forbidden_sealed_test_events`)
     - `configs/evaluation/week8_controlled_tta_protocol.json` (`forbidden_sealed_test_events`)
     - `tools/train_source_model.py` (`FORBIDDEN_SEALED_TEST_EVENTS`)
     - `tools/eval_official_reference.py` (`FORBIDDEN_SEALED_TEST_EVENTS`)
     - `docs/week8_controlled_tta_protocol.md` (`Quarantined Events`)
   - **Approved TTA Development Partition (6 Events, 2,826 samples):**
     `[279, 437, 520, 559, 1111003, 1111008]`
     Enforced identically across:
     - `src/data/evaluation_dataset.py` (`DEFAULT_TTA_DEV_EVENTS`)
     - `configs/evaluation/source_baseline_tta_dev_protocol.json` (`target_events`)
     - `configs/evaluation/week8_controlled_tta_protocol.json` (`target_events`)
     - `docs/week8_controlled_tta_protocol.md` (`Target Events`)

3. **Smoke Physical Data Access Audit:**
   - The diagnostic smoke test strictly accessed only 38 samples across two authorized development events:
     - Event `437`: first 19 samples in deterministic sorted order
     - Event `1111003`: all 19 samples in deterministic sorted order
   - **Adaptation Phase:** Loaded sensory rasters (`SL1_IVV`, `SL1_IVH`) and observational validity mask (`MK0_MNA`) only (`load_labels=False`). No ground-truth label files (`MK0_MLU`) were opened.
   - **Offline Evaluation Phase:** Evaluated frozen predictions strictly offline using double-masking validity (`MK0_MNA == 1 & MK0_MLU != 3`) on those 38 samples only.
   - Zero rasters, labels, or metadata from quarantined sealed-test events were accessed.

---

## 2. Reconstruction of Smoke Attempts & Surviving Evidence

Five smoke attempts occurred during runner development and verification:

1. **Attempt 1 (HEAD `05dd33b`):**
   - **Failure Point:** `torch.load(checkpoint_path)` in `instantiate_adapter`.
   - **Root Cause:** PyTorch 2.6 default change to `weights_only=True` failed on numpy scalar reconstruction in the checkpoint dictionary.
   - **Outcome:** Process raised `_pickle.UnpicklingError` before inference.

2. **Attempt 2 (HEAD `023fa76`):**
   - **Failure Point:** `wrapper.model.load_state_dict(state_dict, strict=True)`.
   - **Root Cause:** Checkpoint state dict stored keys prefixed with `model.` from `SNUNetWrapper`, causing `strict=True` key mismatch against the unwrapped `model`.
   - **Outcome:** Process raised `RuntimeError` before inference.

3. **Attempt 3 (HEAD `503aa9d`):**
   - **Failure Point:** Invocation of `SNUNetWrapper.load_checkpoint(...)`.
   - **Root Cause:** Method name typo; the class method is `load_from_checkpoint`.
   - **Outcome:** Process raised `AttributeError` before inference.

4. **Attempt 4 (HEAD `da6eb3c`):**
   - **Failure Point:** Logging step diagnostics after Physical Run 1 (`source_only`).
   - **Root Cause:** `SourceOnlyAdapter` inherits from `BaseOnlineAdapter` and does not define a `step_count` attribute directly, raising `AttributeError` when accessed in `run_stream_adaptation`.
   - **Outcome:** Raised `AttributeError` during batch diagnostic logging.

5. **Attempt 5 (HEAD `ca4f558` — Final Successful Smoke):**
   - **Resolution:** Replaced direct attribute access with `getattr(adapter, "step_count", 0)`.
   - **Outcome:** Completed all 5 physical cases in 20.30 seconds.
   - **Verification:** State immutability confirmed on Source-Only; BN parameter updates and step resets confirmed on Tent; double-masking offline evaluation completed; all outputs verified finite.

### Surviving Evidence vs. Deleted Intermediate State
- Intermediate temporary directories and logs from aborted attempts 1–4 were cleaned with `rm -rf` prior to relaunching. Those intermediate failure logs were not preserved on disk and are reconstructed here from execution transcripts.
- **Surviving Original Evidence:**
  - `logs/week8_controlled_tta/smoke_s0_v1.log` (full log of successful Attempt 5)
  - `outputs/week8_controlled_tta_smoke/smoke_s0_v1/stream_event_metrics.csv` (12 metric rows)
  - `outputs/week8_controlled_tta_smoke/smoke_s0_v1/stream_matrix_summary.json` (full summary, timings, and hashes)
  - `outputs/week8_controlled_tta_smoke/smoke_s0_v1/week8_tta_manifest.json` (audit of the 38 evaluated samples)

---

## 3. Scope of the Diagnostic Smoke Evidence

The successful fifth smoke run serves exclusively as a technical validity check of:
1. Dataloader integration with sensory/label isolation flags (`load_labels=False` during adaptation);
2. Batch normalization behavior on partial tail batches ($N=3$);
3. GPU memory and execution stability on `cuda:0`;
4. Bit-exact parameter immutability for source baselines;
5. Proper parameter scope and step reset/continual logic for Tent and AdaBN;
6. Offline double-masking scoring.

All generated smoke artifacts explicitly record:
- `"is_formal_run": false`
- `"diagnostic_gate_a2_eligible": false`

Under no circumstances should the 38-sample metric scores from the smoke test be interpreted as scientific evidence for Gate A2 decisions.

---

## 4. Audit of Automatic Git Commits

During runner implementation and fixing, five commits were recorded:
- `05dd33b`: Implementation of reference Tent adapter, matrix runner, and smoke mode.
- `023fa76`: Addition of `weights_only=False` fallback in checkpoint loading.
  - *Provenance Note:* This commit used `git commit -am`, which unintentionally committed a one-line school affiliation update in `docs/interim_report_outline.md` (`School of Computer Science and Engineering` $\to$ `School of Electrical & Electronic Engineering`). No content was lost or damaged.
- `503aa9d`: Integration of `SNUNetWrapper.load_checkpoint`.
- `da6eb3c`: Correction of method call to `SNUNetWrapper.load_from_checkpoint`.
- `ca4f558`: Addition of safe `getattr` fallback for `adapter.step_count`.

In accordance with project integrity standards, Git history is preserved as-is without rebasing or rewriting.
