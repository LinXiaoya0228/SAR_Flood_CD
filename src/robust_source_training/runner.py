"""Unified execution runner for robust source training infrastructure."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.snunet_wrapper import SNUNetWrapper
from src.models.siam_diff_wrapper import SiamDiffWrapper
from src.evaluation.losses import ControlledSourceLoss
from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.dataset import (
    RSTR0SourceDataset,
    build_r0_dataloader,
    seed_everything,
)
from src.robust_source_training.evaluation import EventIndependentEvaluator
from src.robust_source_training.manifest import (
    write_final_manifest,
    write_preflight_manifest,
    write_terminal_status,
)
from src.robust_source_training.checkpoint import (
    assert_states_equal,
    load_rst_checkpoint,
    save_rst_checkpoint,
)


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("RST2Runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def build_model(arch_cfg: Dict[str, Any]) -> nn.Module:
    """Builds a model instance based on architecture configuration."""
    arch_name = arch_cfg.get("name", "SNUNet_ECAM")
    in_channels = int(arch_cfg.get("in_channels", 2))
    out_ch = int(arch_cfg.get("out_ch", arch_cfg.get("out_channels", 3)))
    if arch_name in ("SNUNet_ECAM", "SNUNet"):
        base_channel = int(arch_cfg.get("base_channel", 32))
        return SNUNetWrapper(in_channels=in_channels, out_ch=out_ch, base_channel=base_channel)
    elif arch_name in ("SiamUnet_diff", "SiamDiffWrapper"):
        dropout_p = float(arch_cfg.get("dropout_p", 0.2))
        return SiamDiffWrapper(in_channels=in_channels, out_channels=out_ch, dropout_p=dropout_p)
    else:
        raise ValueError(f"Unsupported architecture name: '{arch_name}'")


class RSTExecutionEngine:
    """Orchestrates validation, dry runs, training loops, evaluation, and resume."""

    def __init__(self, config_path: Union[str, Path], repo_root: Optional[Path] = None):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.run_id = self.config.get("run_id", "unnamed_run")
        self.output_root = self.repo_root / self.config.get("output_root", f"outputs/robust_source_training/{self.run_id}")
        self.log_file = self.repo_root / self.config.get("log_file", f"logs/robust_source_training/{self.run_id}.log") if self.config.get("log_file") else None

        self.logger = setup_logger(self.log_file)

    def validate_only(self) -> Dict[str, Any]:
        """Performs static, fail-closed validation of configuration and contracts."""
        self.logger.info("Executing static validation...")

        # 1. Unknown configuration keys check
        known_config_keys = {
            "$schema_version", "description", "execution_class", "scientific_result",
            "launch_authorized", "preparation_only", "recipe_id", "recipe_name", "components",
            "run_id", "output_root", "log_file", "fold_id", "seed", "max_epochs",
            "device", "provenance_references", "partition", "training",
            "architecture", "optimization", "smoke_bounds", "loss", "augmentation",
            "inputs", "evaluation_metrics", "created_at_utc", "route",
            "confirmatory_immutability_rule", "discovery_selection_record",
            "m1_event_balance_spec", "m2_flood_aware_loss_spec", "m3_paired_augmentation_spec"
        }
        for k in self.config.keys():
            if k not in known_config_keys:
                raise ContractViolationError(
                    f"Unknown config key detected: '{k}'. "
                    "Fail-closed policy rejects unknown keys that could alter scientific behavior."
                )

        train_events = self.config.get("partition", {}).get("train_event_ids")
        eval_events = self.config.get("partition", {}).get("evaluation_event_ids")
        fold_id = self.config.get("fold_id", "custom")
        seed = self.config.get("seed", 0)
        exec_class = self.config.get("execution_class", "SCIENTIFIC_RUN")
        is_smoke = exec_class in ("INFRASTRUCTURE_SMOKE", "COMPONENT_SMOKE")
        allow_empty_eval = exec_class in ("ALL_SOURCE_TRAINING", "FINAL_ALL_SOURCE_TRAINING") or (is_smoke and len(eval_events or []) == 0)

        fold_cfg = self.config.get("provenance_references", {}).get("fold_config_path")
        if fold_cfg:
            fold_cfg = self.repo_root / fold_cfg

        validated = validate_event_contracts(
            train_event_ids=train_events,
            evaluation_event_ids=eval_events,
            fold_id=fold_id,
            run_id=self.run_id,
            seed=seed,
            is_smoke=is_smoke,
            folds_config_path=fold_cfg,
            allow_empty_eval=allow_empty_eval,
        )

        # All-source validation constraints (RST9)
        if exec_class in ("ALL_SOURCE_TRAINING", "FINAL_ALL_SOURCE_TRAINING"):
            recipe_id = self.config.get("recipe_id")
            if recipe_id not in ("R0", "R2"):
                raise ContractViolationError(
                    f"All-source run must use recipe 'R0' or 'R2'; got '{recipe_id}'."
                )
            comps = self.config.get("components", {})
            m1 = bool(comps.get("m1_event_balance", False))
            m2 = bool(comps.get("m2_flood_aware_loss", False))
            m3 = bool(comps.get("m3_paired_augmentation", False))
            if m3:
                raise ContractViolationError("M3 paired augmentation must be disabled in all-source runs.")
            if recipe_id == "R0" and (m1 or m2 or m3):
                raise ContractViolationError(f"R0 must have all components disabled; got {comps}")
            if recipe_id == "R2" and (not m1 or not m2 or m3):
                raise ContractViolationError(f"R2 must have exactly M1 and M2 enabled; got {comps}")
            if set(validated["train_event_ids"]) != set(AUTHORITATIVE_SOURCE_EVENTS):
                raise ContractViolationError(
                    f"All-source training must contain all 27 source events; got {len(validated['train_event_ids'])} events."
                )
            if len(validated["evaluation_event_ids"]) != 0:
                raise ContractViolationError(
                    f"All-source training must have empty evaluation_event_ids; got {validated['evaluation_event_ids']}"
                )
            if seed not in (0, 1, 2):
                raise ContractViolationError(f"All-source seed must be 0, 1, or 2; got '{seed}'")
            if not is_smoke and int(self.config.get("max_epochs", 0)) != 16:
                raise ContractViolationError("All-source max_epochs must be exactly 16.")

            # M2 class-weights artifact validation
            if recipe_id == "R2":
                prov_refs = self.config.get("provenance_references", {})
                cw_path_str = prov_refs.get("class_weights_path")
                cw_sha = prov_refs.get("class_weights_sha256")
                if not cw_path_str or not cw_sha:
                    raise ContractViolationError(
                        "All-source R2 run must specify 'class_weights_path' and 'class_weights_sha256' "
                        "in provenance_references."
                    )
                cw_path = self.repo_root / cw_path_str
                if not cw_path.exists():
                    raise ContractViolationError(f"Referenced class_weights_path does not exist: {cw_path}")
                actual_sha = hashlib.sha256(cw_path.read_bytes()).hexdigest()
                if actual_sha != cw_sha:
                    raise ContractViolationError(
                        f"Frozen class-weights artifact hash mismatch! Expected {cw_sha}, got {actual_sha} at {cw_path}"
                    )

        # Confirmatory validation constraints (RST5/RST6)
        if exec_class == "CONFIRMATORY_TRAINING":
            recipe_id = self.config.get("recipe_id")
            if recipe_id not in ("R0", "R2"):
                raise ContractViolationError(
                    f"Confirmatory run must use recipe 'R0' or 'R2'; got '{recipe_id}'."
                )
            comps = self.config.get("components", {})
            m1 = bool(comps.get("m1_event_balance", False))
            m2 = bool(comps.get("m2_flood_aware_loss", False))
            m3 = bool(comps.get("m3_paired_augmentation", False))
            if m3:
                raise ContractViolationError("M3 paired augmentation must be disabled in confirmatory runs.")
            if recipe_id == "R0" and (m1 or m2 or m3):
                raise ContractViolationError(f"R0 must have all components disabled; got {comps}")
            if recipe_id == "R2" and (not m1 or not m2 or m3):
                raise ContractViolationError(f"R2 must have exactly M1 and M2 enabled; got {comps}")
            if fold_id not in ("confirmation_fold_a", "confirmation_fold_b"):
                raise ContractViolationError(f"Confirmatory fold must be fold_a or fold_b; got '{fold_id}'")
            if seed not in (0, 1, 2):
                raise ContractViolationError(f"Confirmatory seed must be 0, 1, or 2; got '{seed}'")
            if int(self.config.get("max_epochs", 0)) != 16:
                raise ContractViolationError("Confirmatory max_epochs must be exactly 16.")

            # M2 class-weights artifact validation for confirmatory R2
            if recipe_id == "R2":
                prov_refs = self.config.get("provenance_references", {})
                cw_path_str = prov_refs.get("class_weights_path")
                cw_sha = prov_refs.get("class_weights_sha256")
                if not cw_path_str or not cw_sha:
                    raise ContractViolationError(
                        "Confirmatory R2 run must specify 'class_weights_path' and 'class_weights_sha256' "
                        "in provenance_references."
                    )
                cw_path = self.repo_root / cw_path_str
                if not cw_path.exists():
                    raise ContractViolationError(f"Referenced class_weights_path does not exist: {cw_path}")
                actual_sha = hashlib.sha256(cw_path.read_bytes()).hexdigest()
                if actual_sha != cw_sha:
                    raise ContractViolationError(
                        f"Frozen class-weights artifact hash mismatch! Expected {cw_sha}, got {actual_sha} at {cw_path}"
                    )

        # Supplementary ablation validation constraints (M2_ONLY)
        if exec_class == "SUPPLEMENTARY_ABLATION":
            recipe_id = self.config.get("recipe_id")
            if recipe_id != "M2_ONLY":
                raise ContractViolationError(
                    f"Supplementary ablation run must use recipe 'M2_ONLY'; got '{recipe_id}'."
                )
            comps = self.config.get("components", {})
            m1 = bool(comps.get("m1_event_balance", False))
            m2 = bool(comps.get("m2_flood_aware_loss", False))
            m3 = bool(comps.get("m3_paired_augmentation", False))
            if m1 or not m2 or m3:
                raise ContractViolationError(
                    f"M2_ONLY must have exactly M2 enabled and M1/M3 disabled; got {comps}"
                )
            if fold_id not in ("confirmation_fold_a", "confirmation_fold_b"):
                raise ContractViolationError(f"Ablation fold must be fold_a or fold_b; got '{fold_id}'")
            if seed not in (0, 1, 2):
                raise ContractViolationError(f"Ablation seed must be 0, 1, or 2; got '{seed}'")
            if not is_smoke and int(self.config.get("max_epochs", 0)) != 16:
                raise ContractViolationError("Ablation max_epochs must be exactly 16.")

            # M2 class-weights artifact validation for M2_ONLY
            prov_refs = self.config.get("provenance_references", {})
            cw_path_str = prov_refs.get("class_weights_path")
            cw_sha = prov_refs.get("class_weights_sha256")
            if not cw_path_str or not cw_sha:
                raise ContractViolationError(
                    "Supplementary ablation M2_ONLY run must specify 'class_weights_path' and 'class_weights_sha256' "
                    "in provenance_references."
                )
            cw_path = self.repo_root / cw_path_str
            if not cw_path.exists():
                raise ContractViolationError(f"Referenced class_weights_path does not exist: {cw_path}")
            actual_sha = hashlib.sha256(cw_path.read_bytes()).hexdigest()
            if actual_sha != cw_sha:
                raise ContractViolationError(
                    f"Frozen class-weights artifact hash mismatch! Expected {cw_sha}, got {actual_sha} at {cw_path}"
                )

        self.logger.info(
            f"Validation PASSED: {validated['train_event_count']} train events, "
            f"{validated['evaluation_event_count']} eval events, is_smoke={is_smoke}"
        )
        return validated

    def dry_run(self) -> Dict[str, Any]:
        """Validates contracts and initializes datasets on CPU to check sample counts."""
        validated = self.validate_only()
        self.logger.info("Executing dry run dataset inspection (CPU-only, no training)...")

        train_events = validated["train_event_ids"]
        eval_events = validated["evaluation_event_ids"]

        train_ds = RSTR0SourceDataset(allowed_event_ids=train_events)
        self.logger.info(f"Dry run: train dataset loaded {len(train_ds)} samples across {train_events}")

        eval_datasets = {}
        for eid in eval_events:
            eds = RSTR0SourceDataset(allowed_event_ids=[eid])
            eval_datasets[eid] = len(eds)
            self.logger.info(f"Dry run: eval event {eid} contains {len(eds)} samples")

        return {
            "validation": validated,
            "train_sample_count": len(train_ds),
            "eval_sample_counts": eval_datasets,
        }

    def execute(self, resume_checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Executes authorized training, checkpoint saving, reload verification, and evaluation."""
        start_time = time.time()
        validated = self.validate_only()

        # Check execution authorization
        exec_class = self.config.get("execution_class", "SCIENTIFIC_RUN")
        is_smoke = exec_class in ("INFRASTRUCTURE_SMOKE", "COMPONENT_SMOKE")

        # Enforce fresh-run-only recovery policy for confirmatory and ablation runs
        if exec_class in ("CONFIRMATORY_TRAINING", "SUPPLEMENTARY_ABLATION") and resume_checkpoint_path is not None:
            raise ContractViolationError(
                f"Resume rejected: execution_class is '{exec_class}' "
                "and resume_checkpoint_path was supplied. Confirmatory and ablation training runs "
                "must execute fresh from epoch 0. Mid-epoch or epoch-level resume is disabled "
                "under the frozen fresh-run recovery policy."
            )

        if not self.config.get("launch_authorized", False):
            raise ContractViolationError(
                "Execution aborted: launch_authorized is False in configuration. "
                "Explicit authorization is required before launching compute runs."
            )
        if exec_class == "CONFIRMATORY_TRAINING" and self.config.get("preparation_only", False):
            raise ContractViolationError(
                "Execution aborted: preparation_only is True in confirmatory configuration. "
                "Formal launch is not authorized in RST5."
            )

        # Validate recipe vs active switches
        recipe_id = self.config.get("recipe_id", "R0")
        components = self.config.get("components", {})
        m1_enabled = bool(components.get("m1_event_balance", False))
        m2_enabled = bool(components.get("m2_flood_aware_loss", False))
        m3_enabled = bool(components.get("m3_paired_augmentation", False))

        expected_switches = {
            "R0": (False, False, False),
            "R1": (True, False, False),
            "R2": (True, True, False),
            "R3": (True, True, True),
            "M2_ONLY": (False, True, False),
        }
        if recipe_id not in expected_switches:
            raise ContractViolationError(f"Unknown recipe_id: '{recipe_id}'. Must be one of R0, R1, R2, R3, M2_ONLY.")
        actual_switches = (m1_enabled, m2_enabled, m3_enabled)
        if actual_switches != expected_switches[recipe_id]:
            raise ContractViolationError(
                f"Recipe '{recipe_id}' switches mismatch! Expected {expected_switches[recipe_id]} "
                f"(M1, M2, M3), got {actual_switches}."
            )

        # Check output directory collision
        if self.output_root.exists() and resume_checkpoint_path is None:
            # Check if directory is non-empty
            existing_files = list(self.output_root.glob("*"))
            if existing_files:
                raise ContractViolationError(
                    f"Output directory already exists and contains files: {self.output_root}. "
                    "Overwriting is prohibited. Specify a new run_id or pass --resume."
                )

        self.output_root.mkdir(parents=True, exist_ok=True)

        # 1. Write immutable preflight manifest
        pref_path, pref_sha256, pref_dict = write_preflight_manifest(
            run_dir=self.output_root,
            config=self.config,
            config_path=self.config_path,
            repo_root=self.repo_root,
        )
        self.logger.info(f"Wrote preflight manifest to {pref_path} (SHA-256: {pref_sha256[:16]}...)")

        device_str = self.config.get("device", "cpu")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
        device = torch.device(device_str)

        seed = int(self.config.get("seed", 0))
        seed_everything(seed)

        # 2. Datasets & Loaders
        smoke_bounds = self.config.get("smoke_bounds", {})
        train_cfg = self.config.get("training", {})
        max_train_micro = smoke_bounds.get("max_train_microbatches")
        max_eval_batches = smoke_bounds.get("max_eval_batches_per_event")
        micro_bs = int(smoke_bounds.get("micro_batch_size", train_cfg.get("micro_batch_size", self.config.get("micro_batch_size", 8))))
        accum_steps = int(smoke_bounds.get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 4))))
        num_workers = int(smoke_bounds.get("num_workers", train_cfg.get("num_workers", self.config.get("num_workers", 2))))

        # Setup M3 augmenter if enabled
        augmenter = None
        if m3_enabled:
            from src.robust_source_training.augmentation import SynchronizedPairedSARAugmentation
            aug_cfg = self.config.get("augmentation", {})
            augmenter = SynchronizedPairedSARAugmentation(
                hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
                vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
                rotations_deg=tuple(aug_cfg.get("rotations_deg", (0, 90, 180, 270))),
                min_gain=float(aug_cfg.get("min_gain", 0.95)),
                max_gain=float(aug_cfg.get("max_gain", 1.05)),
            )

        # Sample cap for smoke if declared
        train_cap = self.config.get("partition", {}).get("expected_train_sample_count") if is_smoke else None
        train_ds = RSTR0SourceDataset(
            allowed_event_ids=validated["train_event_ids"],
            max_samples_cap=train_cap,
            augmenter=augmenter,
            is_training=True,
            seed=seed,
        )

        # Setup loss objective and M2 class weights (computed and verified before model construction)
        loss_cfg = self.config.get("loss", {})
        m2_artifact_path = None
        m2_weights_list = None
        if m2_enabled:
            from src.robust_source_training.loss import (
                RobustSourceLoss,
                compute_training_class_weights,
            )
            m2_stats = compute_training_class_weights(
                train_samples_metadata=train_ds.samples,
                train_event_ids=validated["train_event_ids"],
                source_inventory_hash=self.config.get("provenance_references", {}).get("source_inventory_hash", ""),
                dataset=train_ds,
            )
            m2_weights_file = self.output_root / "training_class_weights.json"
            m2_weights_file.write_text(json.dumps(m2_stats, indent=2) + "\n", encoding="utf-8")
            m2_artifact_path = m2_weights_file
            m2_weights_list = m2_stats["final_weights_list"]
            self.logger.info(f"M2 class weights computed and saved to {m2_weights_file}: {m2_weights_list}")

            # Fail-closed pre-compute assertion: runtime weights must exactly match frozen per-fold artifact
            prov_refs = self.config.get("provenance_references", {})
            expected_cw_path = prov_refs.get("class_weights_path")
            expected_cw_sha = prov_refs.get("class_weights_sha256")
            if expected_cw_path:
                full_cw_path = self.repo_root / expected_cw_path
                if not full_cw_path.exists():
                    raise ContractViolationError(f"Referenced class_weights_path does not exist: {full_cw_path}")
                actual_cw_file_sha = hashlib.sha256(full_cw_path.read_bytes()).hexdigest()
                if expected_cw_sha and actual_cw_file_sha != expected_cw_sha:
                    raise ContractViolationError(
                        f"Frozen class-weights artifact hash mismatch! Expected {expected_cw_sha}, got {actual_cw_file_sha} at {full_cw_path}"
                    )
                with open(full_cw_path, "r", encoding="utf-8") as f_cw:
                    frozen_cw_data = json.load(f_cw)
                frozen_weights_list = frozen_cw_data.get("final_weights_list")
                if frozen_weights_list is None:
                    raise ContractViolationError(f"Frozen class-weights artifact at {full_cw_path} missing 'final_weights_list'.")

                if m2_weights_list != frozen_weights_list:
                    raise ContractViolationError(
                        f"Runtime computed class weights mismatch frozen artifact! "
                        f"Computed: {m2_weights_list}, Frozen: {frozen_weights_list} from {full_cw_path}"
                    )
                if "aggregate_counts" in frozen_cw_data:
                    if m2_stats.get("aggregate_counts") != frozen_cw_data.get("aggregate_counts"):
                        raise ContractViolationError(
                            f"Runtime computed aggregate counts mismatch frozen artifact! "
                            f"Computed: {m2_stats.get('aggregate_counts')}, Frozen: {frozen_cw_data.get('aggregate_counts')}"
                        )
                self.logger.info(
                    f"Class-weights verified against frozen artifact {expected_cw_path} (SHA-256: {actual_cw_file_sha[:16]}...)"
                )
            elif exec_class in ("CONFIRMATORY_TRAINING", "ALL_SOURCE_TRAINING", "FINAL_ALL_SOURCE_TRAINING", "SUPPLEMENTARY_ABLATION"):
                raise ContractViolationError(
                    "All-source, confirmatory, or ablation training with M2 enabled must specify 'class_weights_path' in provenance_references."
                )

            criterion = RobustSourceLoss(
                class_weights=m2_weights_list,
                weight_ce=float(loss_cfg.get("weight_ce", 0.5)),
                weight_dice=float(loss_cfg.get("weight_dice", 0.5)),
                classes_in_dice=tuple(loss_cfg.get("classes_in_dice", [0, 1, 2])),
                ignore_index=int(loss_cfg.get("ignore_index", 3)),
                smooth_epsilon=float(loss_cfg.get("smooth_epsilon", 1.0)),
            )
        else:
            criterion = ControlledSourceLoss(
                weight_ce=float(loss_cfg.get("weight_ce", 0.5)),
                weight_dice=float(loss_cfg.get("weight_dice", 0.5)),
                classes_in_dice=loss_cfg.get("classes_in_dice", [0, 1, 2]),
                ignore_index=int(loss_cfg.get("ignore_index", 3)),
                smooth_epsilon=float(loss_cfg.get("smooth_epsilon", 1.0)),
            )
        criterion = criterion.to(device)

        # 3. Model and Optimizer Setup
        arch = self.config.get("architecture", {})
        model = build_model(arch).to(device)

        # Log initial model state hash
        init_h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            init_h.update(k.encode("utf-8"))
            init_h.update(v.cpu().numpy().tobytes())
        initial_model_hash = init_h.hexdigest()
        self.logger.info(f"Initial model state SHA-256: {initial_model_hash}")

        opt_cfg = self.config.get("optimization", {})
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(opt_cfg.get("learning_rate", 1e-3)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )

        # Handle resume if specified
        start_epoch = 1
        global_step = 0
        if resume_checkpoint_path is not None:
            self.logger.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
            ckpt_meta = load_rst_checkpoint(
                path=resume_checkpoint_path,
                model=model,
                optimizer=optimizer,
                device=device_str,
            )
            start_epoch = ckpt_meta.get("epoch", 0) + 1
            global_step = ckpt_meta.get("global_step", 0)
            self.logger.info(f"Resumed successfully at epoch {start_epoch}, step {global_step}")

        # 4. Training loop
        max_epochs = int(smoke_bounds.get("max_epochs", train_cfg.get("max_epochs", self.config.get("max_epochs", 16))))
        model.train()
        optimizer.zero_grad()

        total_samples_observed = 0
        total_microbatches = 0
        peak_vram = 0.0

        m1_epoch_summaries = []
        for epoch in range(start_epoch, max_epochs + 1):
            train_ds.set_epoch(epoch)
            if m1_enabled:
                from src.robust_source_training.sampler import EventBalancedSampler
                epoch_sampler = EventBalancedSampler(
                    samples_metadata=train_ds.samples,
                    run_seed=seed,
                    epoch=epoch,
                )
                m1_epoch_summaries.append(epoch_sampler.get_epoch_summary())
                epoch_loader = build_r0_dataloader(
                    dataset=train_ds,
                    batch_size=micro_bs,
                    shuffle=False,
                    sampler=epoch_sampler,
                    seed=seed,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"),
                )
            else:
                epoch_loader = build_r0_dataloader(
                    dataset=train_ds,
                    batch_size=micro_bs,
                    shuffle=True,
                    seed=seed + epoch - 1,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"),
                )

            self.logger.info(f"Starting epoch {epoch}/{max_epochs} (Recipe {recipe_id})...")
            accum_microbatches = 0
            for b_idx, batch in enumerate(epoch_loader):
                if max_train_micro is not None and total_microbatches >= max_train_micro:
                    self.logger.info(f"Reached smoke microbatch cap ({max_train_micro}). Breaking epoch loop.")
                    break

                xA = batch["xA"].to(device)
                xB = batch["xB"].to(device)
                labels = batch["label"].to(device)

                logits = model(xA, xB)
                loss, loss_comps = criterion(logits, labels, return_components=True)
                loss_scaled = loss / accum_steps
                loss_scaled.backward()

                total_samples_observed += len(batch["sample_id"])
                total_microbatches += 1
                accum_microbatches += 1

                if accum_microbatches == accum_steps or (b_idx + 1 == len(epoch_loader)):
                    # Check gradients are finite
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError("Non-finite gradient detected in training step!")
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_microbatches = 0
                    global_step += 1
                    self.logger.info(
                        f"Epoch {epoch} | Step {global_step} (Microbatch {total_microbatches}) | "
                        f"Loss: {loss.item():.4f} (CE: {loss_comps['loss_ce']:.4f}, Dice: {loss_comps['loss_dice']:.4f})"
                    )

                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    if mem > peak_vram:
                        peak_vram = mem

        # 5. Checkpoint saving
        last_ckpt_path = self.output_root / "last_checkpoint.pt"
        final_ckpt_path = self.output_root / "final_checkpoint.pt"

        prov = self.config.get("provenance_references", {})
        config_hash = prov.get("r0_spec_sha256", "UNKNOWN")
        fold_hash = prov.get("fold_config_sha256", "UNKNOWN")

        save_rst_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=max_epochs,
            global_step=global_step,
            seed=seed,
            run_id=self.run_id,
            config_hash=config_hash,
            fold_hash=fold_hash,
            train_event_ids=validated["train_event_ids"],
            evaluation_event_ids=validated["evaluation_event_ids"],
        )
        save_rst_checkpoint(
            path=final_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=max_epochs,
            global_step=global_step,
            seed=seed,
            run_id=self.run_id,
            config_hash=config_hash,
            fold_hash=fold_hash,
            train_event_ids=validated["train_event_ids"],
            evaluation_event_ids=validated["evaluation_event_ids"],
        )
        self.logger.info(f"Saved checkpoints to {last_ckpt_path} and {final_ckpt_path}")

        # 6. Real Checkpoint-Resume Verification
        self.logger.info("Executing real checkpoint-resume verification...")
        fresh_model = build_model(arch).to(device)
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

        load_rst_checkpoint(
            path=final_ckpt_path,
            model=fresh_model,
            optimizer=fresh_opt,
            expected_config_hash=config_hash,
            expected_fold_hash=fold_hash,
            expected_train_events=validated["train_event_ids"],
            expected_eval_events=validated["evaluation_event_ids"],
            device=device_str,
        )
        assert_states_equal(model, fresh_model)
        self.logger.info("Checkpoint-resume verification PASSED: fresh model state bit-identical to trained model.")

        # 7. Independent Event Evaluation
        if len(validated["evaluation_event_ids"]) > 0:
            self.logger.info("Executing independent event-wise evaluation...")
            evaluator = EventIndependentEvaluator(
                evaluation_event_ids=validated["evaluation_event_ids"],
                device=device,
                batch_size=16,
                num_workers=num_workers,
            )

            event_results = []
            eval_cap = self.config.get("partition", {}).get("expected_eval_sample_count") if is_smoke else None
            for eid in validated["evaluation_event_ids"]:
                eds = RSTR0SourceDataset(
                    allowed_event_ids=[eid],
                    max_samples_cap=eval_cap,
                )
                res = evaluator.evaluate_event(
                    model=model,
                    dataset=eds,
                    event_id=eid,
                    max_batches=max_eval_batches,
                )
                event_results.append(res)
                self.logger.info(
                    f"Eval Event {eid}: Samples={res['sample_count']}, F1={res['f1']:.4f}, "
                    f"IoU={res['iou']:.4f}, Recall={res['recall']:.4f}, Precision={res['precision']:.4f}"
                )

            metrics_agg = evaluator.aggregate_metrics(event_results)
        else:
            self.logger.info("No evaluation events specified for this run (training-only all-source run).")
            metrics_agg = {
                "status": "ALL_SOURCE_TRAINING_NO_EVALUATION_SPLIT",
                "n_events_evaluated": 0,
                "per_event": [],
            }
        metrics_path = self.output_root / "event_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_agg, f, indent=2)
            f.write("\n")
        self.logger.info(f"Wrote event evaluation metrics to {metrics_path}")

        # 8. Final Manifest and Terminal Status
        elapsed = time.time() - start_time
        if exec_class == "COMPONENT_SMOKE":
            terminal_status_str = "COMPLETED_COMPONENT_SMOKE"
        elif exec_class == "INFRASTRUCTURE_SMOKE":
            terminal_status_str = "COMPLETED_INFRASTRUCTURE_SMOKE"
        else:
            terminal_status_str = "COMPLETED_SCIENTIFIC_RUN"

        fin_path, fin_sha256, fin_dict = write_final_manifest(
            run_dir=self.output_root,
            preflight_sha256=pref_sha256,
            preflight_dict=pref_dict,
            completion_status=terminal_status_str,
            observed_sample_count=total_samples_observed,
            observed_steps=global_step,
            runtime_seconds=elapsed,
            final_checkpoint_path=final_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            event_metrics_path=metrics_path,
            peak_vram_mb=peak_vram,
            resume_verified=True,
            m1_summary={"epoch_summaries": m1_epoch_summaries} if m1_epoch_summaries else None,
            m2_artifact_path=m2_artifact_path,
            m3_summary=self.config.get("augmentation", {}) if m3_enabled else None,
        )
        self.logger.info(f"Wrote final manifest to {fin_path} (SHA-256: {fin_sha256[:16]}...)")

        status_path = write_terminal_status(
            run_dir=self.output_root,
            run_id=self.run_id,
            status=terminal_status_str,
            exit_code=0,
        )
        self.logger.info(f"Wrote terminal status {terminal_status_str} to {status_path}")

        return {
            "status": terminal_status_str,
            "runtime_seconds": elapsed,
            "samples_observed": total_samples_observed,
            "optimizer_steps": global_step,
            "peak_vram_mb": peak_vram,
            "preflight_manifest": pref_path,
            "final_manifest": fin_path,
            "final_checkpoint": final_ckpt_path,
            "event_metrics": metrics_path,
        }

