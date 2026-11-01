# ChangeGuard Week 1 Optimizer Protocol Correction & Verification Report

**Document:** `docs/week1_optimizer_protocol_correction.md`  
**Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`  
**Base Commit:** `af164ec69aff0105fc462d94417582be79ffd8a4` (Week 0 Protocol Freeze)  
**Branch:** `main`  
**Audit Date:** `2026-10-06`  
**Host:** `gpu22` (Linux 6.8.0-136-generic x86_64)  
**Python Runtime:** `/usr1/home/s126mdg22_06/miniconda3/envs/vfm_env/bin/python` (Python 3.10.16, PyTorch 2.6.0+cu124)  
**Compute Hardware:** GPU 0 (NVIDIA GeForce RTX 3090, 24,576 MiB VRAM)  
**Status:** `RESOLVED_AND_VERIFIED`

---

## 1. Executive Summary

This report establishes the complete provenance, audit, resolution, and empirical confirmation of the optimizer protocol for ChangeGuard Week 1.

### 1.1 Core Audit Finding
An exhaustive audit of Git history, on-disk configuration files, runtime Python modules, and historical output manifests proves that **the on-disk codebase and configuration files never experienced optimizer drift**:
1. At the Week 0 baseline commit (`af164ec69aff0105fc462d94417582be79ffd8a4`), both `configs/source/snunet_vvvh_v1.json` and `configs/source/snunet_vvvh_slope_v1.json` codified:
   ```json
   "optimization_protocol": {
     "optimizer": "adam",
     "learning_rate": 0.001,
     "weight_decay": 0.0,
     "betas": [0.9, 0.999],
     "lr_scheduler": null,
     "max_epochs": 150,
     "early_stopping_patience": 20
   }
   ```
2. The runtime trainer module (`src/training/source_trainer.py`) initialized:
   ```python
   self.optimizer = torch.optim.Adam(
       self.model.parameters(),
       lr=self.learning_rate,
       betas=self.betas,
       weight_decay=self.weight_decay,
   )
   ```
   with defaults `learning_rate=1e-3`, `weight_decay=0.0`, `betas=(0.9, 0.999)`, and `lr_scheduler=None`.
3. All historical smoke tests (`outputs/week1_smoke/` and `outputs/week1_correction/`) executed `torch.optim.Adam` with learning rate $1\times 10^{-3}$, weight decay $0.0$, betas $(0.9, 0.999)$, and no learning rate scheduler.
4. **Origin of Drift:** The statement asserting that the pipeline used *"AdamW, lr 1e-4, weight_decay 1e-4, cosine annealing"* existed strictly as **conversational prose drift** in an earlier session summary. No such code or configuration was ever committed or executed on disk.

### 1.2 Corrective Actions Taken
To permanently prevent any future divergence between documentation, configuration files, and runtime execution:
1. **Config Standardization:** Updated the `optimizer` field value from `"adam"` to `"Adam"` in both `configs/source/snunet_vvvh_v1.json` and `configs/source/snunet_vvvh_slope_v1.json` to match canonical PyTorch naming.
2. **Runtime Configuration Binding:** Added `SourceTrainer.from_config(model, config, output_dir, device, seed)` in `src/training/source_trainer.py` to directly instantiate the trainer from the configuration dictionary, ensuring that learning rate, weight decay, betas, scheduler, and gradient accumulation parameters are parsed directly from JSON into PyTorch objects.
3. **Automated Unit Testing:** Added `test_approved_optimizer_protocol_and_runtime_agreement` to `tests/test_source_configs.py` (totaling 12 configuration tests and 81 tests overall across the repository), asserting that both configuration files and instantiated trainers strictly match the approved Adam protocol.
4. **Empirical Final Protocol Smoke Confirmation:** Executed `tools/run_week1_final_protocol_smoke.py` on GPU 0 for 10 optimizer steps each on Model A and Model B, generating verified metrics, run manifests, and access traces.
5. **Canonical Artifact Preservation:** Stored the resulting JSON manifests and metrics under `artifacts/week1/final_protocol_*`.

---

## 2. Frozen Approved Optimization Protocol

The user-approved frozen source optimization protocol for ChangeGuard Week 1 and Week 2 is:

| Parameter | Approved Value | PyTorch Implementation |
|---|---|---|
| **Architecture** | SNUNet-CD (`SNUNet_ECAM`) | `src/models/snunet.py` |
| **Optimizer** | `Adam` | `torch.optim.Adam` (NOT `AdamW`) |
| **Learning Rate** | `1e-3` ($0.001$) | `lr=0.001` |
| **Weight Decay** | `0.0` | `weight_decay=0.0` |
| **Betas** | `(0.9, 0.999)` | `betas=(0.9, 0.999)` (PyTorch defaults) |
| **LR Scheduler** | `None` (`null`) | `self.lr_scheduler = None` |
| **Maximum Epochs** | `150` | `max_epochs: 150` |
| **Early Stopping Patience** | `20` epochs | Evaluated on 5-event inner-validation set |
| **Loss Function** | Masked CE + Soft Dice ($1:1$) | `CombinedLoss(ce_weight=1.0, dice_weight=1.0)` |
| **Physical Microbatch Size** | `8` | Confirmed locked in Week 1 memory audit |
| **Gradient Accumulation Steps** | `4` | `gradient_accumulation_steps: 4` |
| **Effective Optimization Batch** | `32` | $8 \times 4 = 32$ |
| **Precision** | FP16 Mixed Precision | `torch.cuda.amp.autocast()` + `GradScaler` |

---

## 3. Forensic Trace & Provenance Verification

### 3.1 Base Commit Inspection (`af164ec`)
Inspecting the configuration files at commit `af164ec` via Git:
```bash
git show af164ec:configs/source/snunet_vvvh_v1.json
git show af164ec:configs/source/snunet_vvvh_slope_v1.json
```
Both files explicitly contained:
```json
  "optimization_protocol": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "lr_scheduler": null,
    "max_epochs": 150,
    "early_stopping_patience": 20
  }
```
**Conclusion:** No commit in the repository ever specified `AdamW`, `lr=1e-4`, `weight_decay=1e-4`, or cosine annealing.

### 3.2 Runtime Code Inspection
`src/training/source_trainer.py` was authored during Week 1 with the following initialization:
```python
if optimizer_name.lower() == "adam":
    self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.learning_rate,
        betas=self.betas,
        weight_decay=self.weight_decay,
    )
elif optimizer_name.lower() == "adamw":
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.learning_rate,
        betas=self.betas,
        weight_decay=self.weight_decay,
    )
```
Because `optimizer_name` defaulted to `"adam"`, `self.learning_rate` defaulted to `1e-3`, `self.weight_decay` defaulted to `0.0`, and `self.betas` defaulted to `(0.9, 0.999)`, the trainer always instantiated `torch.optim.Adam`.

### 3.3 Output Manifest Inspection
The run manifest in `outputs/week1_correction/model_a/run_manifest.json` recorded:
- `config_sha256`: `71b0e4e20ae99130f16f39c049964a2789f2cf05d259fa190eb8784d5df68be9`
Inspecting this exact config on disk confirms `optimizer: "adam"`, `lr: 0.001`, `weight_decay: 0.0`, and `betas: [0.9, 0.999]`.

---

## 4. Code & Configuration Hardening

### 4.1 Canonical Capitalization in Configuration Files
Both source configurations were edited to standardize the optimizer key:
- `configs/source/snunet_vvvh_v1.json`: `"optimizer": "Adam"`
- `configs/source/snunet_vvvh_slope_v1.json`: `"optimizer": "Adam"`

### 4.2 Factory Method `SourceTrainer.from_config`
To ensure that configuration files directly govern model instantiation and training parameters, `SourceTrainer.from_config` was implemented in `src/training/source_trainer.py`:
```python
@classmethod
def from_config(
    cls,
    model: nn.Module,
    config: Dict[str, Any],
    output_dir: Union[str, Path],
    device: str = "cuda:0",
    seed: int = 42,
) -> "SourceTrainer":
    opt_cfg = config.get("optimization_protocol", {})
    batch_policy = config.get("batch_size_policy", {})
    loss_cfg = config.get("loss_function", {})

    return cls(
        model=model,
        optimizer_name=opt_cfg.get("optimizer", "Adam"),
        learning_rate=float(opt_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        lr_scheduler_name=opt_cfg.get("lr_scheduler", None),
        max_epochs=int(opt_cfg.get("max_epochs", 150)),
        early_stopping_patience=int(opt_cfg.get("early_stopping_patience", 20)),
        effective_batch_size=int(batch_policy.get("effective_batch_size", 32)),
        physical_microbatch_size=int(batch_policy.get("physical_microbatch_size", 8)),
        gradient_accumulation_steps=int(batch_policy.get("gradient_accumulation_steps", 4)),
        mixed_precision=bool(config.get("compute_resources", {}).get("mixed_precision", True)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        dice_smooth=float(loss_cfg.get("dice_smooth", 1.0)),
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
```

### 4.3 Automated Verification Test
The automated unit test `test_approved_optimizer_protocol_and_runtime_agreement` in `tests/test_source_configs.py` performs deep assertions on instantiated optimizers:
```python
def test_approved_optimizer_protocol_and_runtime_agreement(self):
    for path in [self.model_a_path, self.model_b_path]:
        with open(path, "r") as f:
            cfg = json.load(f)
        opt_cfg = cfg["optimization_protocol"]
        self.assertEqual(opt_cfg["optimizer"], "Adam")
        self.assertEqual(opt_cfg["learning_rate"], 1e-3)
        self.assertEqual(opt_cfg["weight_decay"], 0.0)
        self.assertEqual(opt_cfg["betas"], [0.9, 0.999])
        self.assertIsNone(opt_cfg["lr_scheduler"])

        # Instantiate dummy model and trainer via from_config
        dummy_model = nn.Conv2d(2, 2, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SourceTrainer.from_config(
                model=dummy_model,
                config=cfg,
                output_dir=tmpdir,
                device="cpu",
                seed=42,
            )
            self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
            self.assertNotIsInstance(trainer.optimizer, torch.optim.AdamW)
            self.assertIsNone(trainer.lr_scheduler)
            param_group = trainer.optimizer.param_groups[0]
            self.assertEqual(param_group["lr"], 1e-3)
            self.assertEqual(param_group["weight_decay"], 0.0)
            self.assertEqual(param_group["betas"], (0.9, 0.999))
```
This test runs and passes cleanly as part of the 81-test suite.

---

## 5. Final Protocol Smoke Confirmation (GPU 0)

To provide empirical evidence of pipeline readiness, `tools/run_week1_final_protocol_smoke.py` was executed on GPU 0 (`cuda:0`).

### 5.1 Execution Parameters
- **Script:** `tools/run_week1_final_protocol_smoke.py --model_type both --steps 10 --device cuda:0`
- **Steps:** 10 optimizer steps per model ($10 \times 4 = 40$ microbatches, 320 samples per model).
- **Physical Microbatch:** 8
- **Gradient Accumulation:** 4
- **Effective Batch:** 32
- **Mixed Precision:** FP16 autocast with GradScaler
- **Validation Subset:** 10 validation batches across all 5 approved inner-validation events (`[267, 275, 497, 518, 1111005]`).
- **Checkpoint Verification:** Evaluated both `checkpoint_latest.pt` and `checkpoint_best.pt`, confirmed checkpoint reload, and verified full RNG state restoration (CPU, CUDA, NumPy).

### 5.2 Model A Results (`[VV, VH]`)
- **Config SHA-256:** `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc`
- **Optimizer Verified:** `torch.optim.Adam(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)`, scheduler: `None`
- **Loss Progression:** Step 1: `12.4734` $\rightarrow$ Step 10: `1.1792` (Duration: `7.64s`)
- **Validation Macro F1:** `0.5088` across 5 inner-validation events
- **Validation Macro Precision:** `0.4096`
- **Validation Macro Recall:** `0.8279`
- **Validation Macro IoU:** `0.3765`
- **Output Artifacts:**
  - `artifacts/week1/final_protocol_model_a_smoke_metrics.json`
  - `artifacts/week1/final_protocol_model_a_run_manifest.json`
  - `artifacts/week1/final_protocol_model_a_access_trace.json`

### 5.3 Model B Results (`[VV, VH, slope]`)
- **Config SHA-256:** `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec`
- **Slope Normalization:** Mean: `1.4926` deg, Std: `2.3723` deg (from `artifacts/week1/source_train_slope_statistics.json`)
- **Optimizer Verified:** `torch.optim.Adam(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)`, scheduler: `None`
- **Loss Progression:** Step 1: `14.0554` $\rightarrow$ Step 10: `1.2397` (Duration: `6.93s`)
- **Validation Macro F1:** `0.4000` across 5 inner-validation events
- **Validation Macro Precision:** `0.3533`
- **Validation Macro Recall:** `0.8122`
- **Validation Macro IoU:** `0.2882`
- **Output Artifacts:**
  - `artifacts/week1/final_protocol_model_b_smoke_metrics.json`
  - `artifacts/week1/final_protocol_model_b_run_manifest.json`
  - `artifacts/week1/final_protocol_model_b_access_trace.json`

### 5.4 Valid-Mask Fairness Confirmation
Both Model A and Model B evaluated identical ground-truth flood pixel counts and valid-mask pixel counts across all 5 inner-validation events during the validation pass:
- Event 267: 752,640 valid pixels, 110,369 GT flood pixels
- Event 275: 852,992 valid pixels, 85,975 GT flood pixels
- Event 497: 626,023 valid pixels, 34,189 GT flood pixels
- Event 518: 382,515 valid pixels, 2,128 GT flood pixels
- Event 1111005: 1,354,752 valid pixels, 519,573 GT flood pixels

---

## 6. Configuration Hashes Comparison

| Configuration File | Week 0 Baseline (`af164ec`) | Pre-Correction Week 1 | Final Week 1 (Locked) |
|---|---|---|---|
| `configs/source/snunet_vvvh_v1.json` | `ae18249e7c6ceac4db9ec27ae8797c95eb308c58070ade09b28367593a4cf310` | `71b0e4e20ae99130f16f39c049964a2789f2cf05d259fa190eb8784d5df68be9` | `9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc` |
| `configs/source/snunet_vvvh_slope_v1.json` | `8523f7448bfc6a4b153f285f769140978abdca670557eb0f9cdcd3e74f63e03f` | `8894451ec9ddb2cbe005b535cb994f71a93863489381667634f1e56b823db2e8` | `f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec` |

### Differences from Week 0 Baseline:
1. `batch_size_policy`: `physical_microbatch_size` set to `8`, `gradient_accumulation_steps` set to `4`, `physical_microbatch_status` updated to `"CONFIRMED_LOCKED"`, and `accumulation_loss_semantics` rationale added.
2. `task_metadata`: `run_type` set to `"source_training"`.
3. `dem_slope_channel` (Model B): `required_statistics_artifact` path updated to `artifacts/week1/source_train_slope_statistics.json`, status set to `"GENERATED_AND_VERIFIED"`, and hash locked to `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04`.
4. `optimization_protocol.optimizer`: Capitalized to `"Adam"`.
All other parameters (`lr=0.001`, `weight_decay=0.0`, `betas=[0.9, 0.999]`, `lr_scheduler=null`, `max_epochs=150`, `early_stopping_patience=20`) are completely identical to the Week 0 baseline.

---

## 7. Canonical Week 1 Artifact Inventory (`artifacts/week1/`)

The 18 version-controlled small JSON audit, profiling, and manifest artifacts stored under `artifacts/week1/` are:

| File Path | Size | SHA-256 Hash | Purpose |
|---|---|---|---|
| `artifacts/week1/checkpoint_resume_validation.json` | 272 B | `f6acbaa2c6756b8fa0056bd468fb2cb866dd585bece4e096343605439f399460` | Checkpoint resume deterministic test |
| `artifacts/week1/loader_access_trace.json` | 668 B | `e86b52c1cbf434ddf3f0d0475e214af8bba1c1d901bb4ba5b9bc498e17c518ee` | DataLoader sample access trace |
| `artifacts/week1/memory_smoke_results.json` | 2,371 B | `c2da571fb39b5daea2fae88d9dd4edc235d2644a19e9ed3448ea48037dc91475` | Microbatch memory profiling |
| `artifacts/week1/model_a_access_trace.json` | 755 B | `81237dcb4094fe1e7da83ce44f7c5717e4138d58562a4c175568076afcc2bdea` | Model A corrected smoke access trace |
| `artifacts/week1/model_a_run_manifest.json` | 2,992 B | `b78181835cef5dc5c0bf2ea3e8a00aeb0b3c86287eaa44cc4dcab79350e70046` | Model A corrected smoke run manifest |
| `artifacts/week1/model_a_smoke_metrics.json` | 5,054 B | `d932176c9a58d1df4b957ed2ede34fa28675e2d467eb3c41789676fee8f60658` | Model A corrected smoke metrics |
| `artifacts/week1/model_b_access_trace.json` | 773 B | `c35067e83d97a52433d84d9561b636f17aca03b2fdd53e99852829d6ee307324` | Model B corrected smoke access trace |
| `artifacts/week1/model_b_run_manifest.json` | 3,020 B | `6d02ca969266fb26555b84e695e792b8f1022550fea85b04887be49816081187` | Model B corrected smoke run manifest |
| `artifacts/week1/model_b_smoke_metrics.json` | 5,046 B | `2446f844c4d09d3a4db54a0813d62af3f1f1ca9cf409ab32055bfde746024ac2` | Model B corrected smoke metrics |
| `artifacts/week1/slope_sensitivity_comparison.json` | 2,103 B | `2a5be9b9f00882c3b37ed08857f8a965f07d28ab7b1cef691939319d1d563dcb` | 5-rule DEM nodata & slope benchmark |
| `artifacts/week1/source_train_slope_statistics.json` | 1,574 B | `bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04` | 22-event slope mean and std |
| `artifacts/week1/split_isolation_audit.json` | 579 B | `c56ad8daef271eb04d66cc17377a3dd87639771249700369ae3b853cd8256114` | Train/val/quarantined event split audit |
| `artifacts/week1/final_protocol_model_a_access_trace.json` | 682 B | `0b29f5772a1d3ad45259e30717ac47c0286a1529ba0b5d411e49bd73748fde2f` | Final protocol Model A access trace |
| `artifacts/week1/final_protocol_model_a_run_manifest.json` | 2,962 B | `fe5ea9ade4aaf4e5496b3f4f40455c14dcd8586013c93546d046513b9fcf160c` | Final protocol Model A run manifest |
| `artifacts/week1/final_protocol_model_a_smoke_metrics.json` | 4,624 B | `38a2836df971db8e36a44411239f1ab3937a3602fa182627105294e3e93e40f5` | Final protocol Model A smoke metrics |
| `artifacts/week1/final_protocol_model_b_access_trace.json` | 718 B | `48775fe7d0712cd3ca846a212834fe1af670c1dd6fbbd35f04f036125c5ba1c2` | Final protocol Model B access trace |
| `artifacts/week1/final_protocol_model_b_run_manifest.json` | 3,012 B | `ecdfa80bee009f4793816780cc36131fc1a9ec0bbf4b514a0c918be8ac974225` | Final protocol Model B run manifest |
| `artifacts/week1/final_protocol_model_b_smoke_metrics.json` | 4,631 B | `14ebe6dc1bdc74c0bc2c14395f30e0a5c468b657d784f629275ec62d49f3d4ca` | Final protocol Model B smoke metrics |

---

## 8. Research Integrity Affirmation

It is certified that:
1. **Zero Results Selected on Drifted Parameters:** No model checkpoint, validation ranking, ablation comparison, or empirical conclusion in this project was selected or evaluated using parameters other than the approved Adam protocol ($lr=1\times 10^{-3}$, $\beta=(0.9, 0.999)$, $\text{weight\_decay}=0.0$, scheduler: none).
2. **Short Smoke Verification Only:** All GPU runs in Week 1 were short engineering smoke tests (10 to 25 steps) designed exclusively to verify numerical stability, memory consumption, checkpoint reloadability, and metric computations. No formal multi-epoch training has been conducted.
3. **Strict Data Boundary Adherence:** Zero samples from the 6 quarantined TTA-validation events (`[279, 437, 520, 559, 1111003, 1111008]`) were loaded during training or validation. Zero sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) were accessed.
