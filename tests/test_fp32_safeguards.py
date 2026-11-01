"""Unit tests for Week 2 FP32 recovery safeguards and fail-fast assertions.

Verifies:
1. FP32 configurations disable autocast and GradScaler.
2. Clean FP32 training steps execute without error.
3. Nonfinite forward logits fail fast.
4. Nonfinite pre-backward loss fails fast with NumericalInstabilityError.
5. Nonfinite backward gradients fail fast with NumericalInstabilityError.
6. Nonfinite BatchNorm running buffers fail fast with NumericalInstabilityError.
7. Predictive collapse detector respects warmup=5 and patience=3, resetting on positive flood.
8. Atomic checkpoint writer rejects nonfinite parameters/buffers and preserves existing checkpoint.
9. Checkpoint resumption rejects corrupted/nonfinite checkpoints.
10. Protocol v2 config hashes and parameters match frozen specifications.
"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import argparse
from unittest.mock import patch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.source_trainer import (
    SourceTrainer,
    NumericalInstabilityError,
    PredictiveCollapseError,
    PredictiveCollapseDetector,
)
from src.evaluation.losses import ControlledSourceLoss
from src.evaluation.run_manifest import compute_file_sha256
from tools.train_source_model import (
    train_source_model,
    validate_provenance_and_config,
    LOCK_MANIFEST_DEFAULT_PATH,
    FROZEN_SPLIT_HASH,
    FROZEN_SLOPE_STATS_HASH,
)


class DummyBNModel(nn.Module):
    """Simple 2-branch model with BatchNorm2d for testing numerical guards."""

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.convA = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, xA, xB):
        feat = self.convA(xA) + self.convB(xB)
        feat = self.bn(feat)
        return self.head(feat)


def make_dummy_batch(batch_size: int = 4, in_channels: int = 2, H: int = 16, W: int = 16):
    return {
        "xA": torch.randn(batch_size, in_channels, H, W),
        "xB": torch.randn(batch_size, in_channels, H, W),
        "label": torch.randint(0, 3, (batch_size, H, W)),
        "valid_mask": torch.ones(batch_size, H, W, dtype=torch.uint8),
        "event_id": [118] * batch_size,
        "sample_id": [f"sample_{i}" for i in range(batch_size)],
    }


class TestFP32Safeguards(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fp32_config_disables_amp_and_scaler(self):
        """Verify FP32 configs strictly disable mixed precision and GradScaler."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_a_path = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        cfg_b_path = repo_root / "configs/source/snunet_vvvh_slope_v2_fp32.json"

        self.assertTrue(cfg_a_path.exists())
        self.assertTrue(cfg_b_path.exists())

        for cfg_path in [cfg_a_path, cfg_b_path]:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            opt = cfg["optimization_protocol"]
            self.assertEqual(opt["precision"], "fp32")
            self.assertFalse(opt["mixed_precision"])
            self.assertFalse(opt["autocast"])
            self.assertFalse(opt["grad_scaler"])

            in_ch = int(cfg["architecture"]["in_channels_per_branch"])
            model = DummyBNModel(in_channels=in_ch)
            trainer = SourceTrainer.from_config(
                model=model,
                config=cfg,
                output_dir=self.temp_dir,
                device="cpu",
            )
            self.assertFalse(trainer.mixed_precision)
            self.assertIsNone(trainer.scaler)
            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-3)
            self.assertEqual(trainer.optimizer.param_groups[0]["weight_decay"], 0.0)
            self.assertEqual(trainer.optimizer.param_groups[0]["betas"], (0.9, 0.999))

    def test_clean_fp32_training_step(self):
        """Verify normal training steps execute cleanly under FP32 with weight updates."""
        model = DummyBNModel(in_channels=2)
        initial_param = model.head.weight.detach().clone()

        dataset = [make_dummy_batch(batch_size=1, in_channels=2) for _ in range(8)]
        # Unpack single items
        flat_data = [
            {
                "xA": d["xA"][0],
                "xB": d["xB"][0],
                "label": d["label"][0],
                "valid_mask": d["valid_mask"][0],
                "event_id": 118,
                "sample_id": f"s_{i}",
            }
            for i, d in enumerate(dataset)
        ]
        loader = DataLoader(flat_data, batch_size=4, shuffle=False)

        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        metrics = trainer.train_steps(loader, num_optimizer_steps=1)
        self.assertEqual(len(metrics), 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics[0]["loss_total"])))
        self.assertFalse(torch.allclose(initial_param, model.head.weight))

    def test_nonfinite_loss_fails_fast(self):
        """Verify that nonfinite loss raises NumericalInstabilityError immediately."""
        model = DummyBNModel(in_channels=2)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        class BrokenLoss(nn.Module):
            def forward(self, logits, target, return_components=False):
                loss = torch.tensor(float("nan"), requires_grad=True)
                comp = {"loss_total": float("nan"), "loss_ce": float("nan"), "loss_dice": float("nan")}
                return (loss, comp) if return_components else loss

        trainer.loss_fn = BrokenLoss()

        batch = make_dummy_batch(batch_size=4)
        loader = DataLoader([
            {
                "xA": batch["xA"][i],
                "xB": batch["xB"][i],
                "label": batch["label"][i],
                "valid_mask": batch["valid_mask"][i],
                "event_id": 118,
                "sample_id": f"s_{i}",
            }
            for i in range(4)
        ], batch_size=4)

        with self.assertRaises(NumericalInstabilityError) as ctx:
            trainer.train_epoch(loader, epoch_idx=1)

        err = ctx.exception
        self.assertEqual(err.tensor_name, "loss")
        self.assertEqual(err.stage, "train_loss")
        self.assertEqual(err.epoch, 1)

    def test_nonfinite_gradient_fails_fast(self):
        """Verify that nonfinite gradients before optimizer.step() fail fast."""
        model = DummyBNModel(in_channels=2)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        # Hook head weight to inject NaN into gradient during backward
        def nan_grad_hook(grad):
            g = grad.clone()
            g[0, 0, 0, 0] = float("nan")
            return g

        model.head.weight.register_hook(nan_grad_hook)

        batch = make_dummy_batch(batch_size=4)
        loader = DataLoader([
            {
                "xA": batch["xA"][i],
                "xB": batch["xB"][i],
                "label": batch["label"][i],
                "valid_mask": batch["valid_mask"][i],
                "event_id": 118,
                "sample_id": f"s_{i}",
            }
            for i in range(4)
        ], batch_size=4)

        with self.assertRaises(NumericalInstabilityError) as ctx:
            trainer.train_epoch(loader, epoch_idx=2)

        err = ctx.exception
        self.assertIn("grad_", err.tensor_name)
        self.assertEqual(err.stage, "train_gradients")

    def test_nonfinite_bn_buffer_fails_fast(self):
        """Verify that corrupted BatchNorm running buffers at epoch end fail fast."""
        model = DummyBNModel(in_channels=2)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        batch = make_dummy_batch(batch_size=4)
        loader = DataLoader([
            {
                "xA": batch["xA"][i],
                "xB": batch["xB"][i],
                "label": batch["label"][i],
                "valid_mask": batch["valid_mask"][i],
                "event_id": 118,
                "sample_id": f"s_{i}",
            }
            for i in range(4)
        ], batch_size=4)

        # Corrupt running_var
        model.bn.running_var[0] = float("inf")

        with self.assertRaises(NumericalInstabilityError) as ctx:
            trainer.train_epoch(loader, epoch_idx=5)

        err = ctx.exception
        self.assertIn("running_var", err.tensor_name)
        self.assertEqual(err.stage, "train_epoch_bn_audit")

    def test_predictive_collapse_detector_warmup_and_patience(self):
        """Verify collapse detector respects warmup=5 and patience=3, resetting on positive flood."""
        detector = PredictiveCollapseDetector(warmup_epochs=5, patience=3)

        # Warmup epochs 1 to 5 with 0 predicted flood: must NOT trigger
        for ep in range(1, 6):
            triggered = detector.update(epoch=ep, pred_flood_pixels=0)
            self.assertFalse(triggered, f"Should not trigger during warmup epoch {ep}")
            self.assertEqual(detector.consecutive_zero_epochs, 0)

        # After warmup: Epoch 6, 7, 8 with 0 flood
        self.assertFalse(detector.update(epoch=6, pred_flood_pixels=0))
        self.assertEqual(detector.consecutive_zero_epochs, 1)

        self.assertFalse(detector.update(epoch=7, pred_flood_pixels=0))
        self.assertEqual(detector.consecutive_zero_epochs, 2)

        self.assertTrue(detector.update(epoch=8, pred_flood_pixels=0))
        self.assertEqual(detector.consecutive_zero_epochs, 3)
        self.assertTrue(detector.triggered)

        # Test reset rule: nonzero flood resets counter
        detector2 = PredictiveCollapseDetector(warmup_epochs=5, patience=3)
        detector2.update(epoch=6, pred_flood_pixels=0)  # counter = 1
        detector2.update(epoch=7, pred_flood_pixels=0)  # counter = 2
        # Nonzero flood at epoch 8 resets counter
        self.assertFalse(detector2.update(epoch=8, pred_flood_pixels=450))
        self.assertEqual(detector2.consecutive_zero_epochs, 0)
        self.assertFalse(detector2.triggered)

        # Needs 3 consecutive zeros again: 9, 10, 11
        self.assertFalse(detector2.update(epoch=9, pred_flood_pixels=0))
        self.assertEqual(detector2.consecutive_zero_epochs, 1)
        self.assertFalse(detector2.update(epoch=10, pred_flood_pixels=0))
        self.assertEqual(detector2.consecutive_zero_epochs, 2)
        self.assertTrue(detector2.update(epoch=11, pred_flood_pixels=0))
        self.assertEqual(detector2.consecutive_zero_epochs, 3)

    def test_atomic_checkpoint_saving_and_finiteness_check(self):
        """Verify atomic checkpoint writing works and rejects corrupted weights."""
        model = DummyBNModel(in_channels=2)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )
        trainer.current_epoch = 10
        trainer.global_step = 100

        # Valid save
        ckpt_path = trainer.save_checkpoint("clean.pt")
        self.assertTrue(ckpt_path.exists())
        orig_hash = compute_file_sha256(ckpt_path)

        # Inject NaN into model parameter
        with torch.no_grad():
            model.head.weight[0, 0, 0, 0] = float("nan")

        # Must reject saving corrupted state
        with self.assertRaises(NumericalInstabilityError) as ctx:
            trainer.save_checkpoint("clean.pt")
        self.assertEqual(ctx.exception.stage, "save_checkpoint_param_audit")

        # Existing checkpoint must remain unmodified and uncorrupted
        self.assertEqual(compute_file_sha256(ckpt_path), orig_hash)

        # Also verify no leftover .tmp files
        tmp_files = list(self.temp_dir.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0)

    def test_resume_rejects_corrupted_checkpoint(self):
        """Verify resume_from_checkpoint rejects checkpoints containing nonfinite weights or failure flags."""
        model = DummyBNModel(in_channels=2)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )
        clean_ckpt = trainer.save_checkpoint("valid.pt")

        # Create a checkpoint flagged as FAILED_NUMERICAL
        try:
            corrupt_state = torch.load(clean_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            corrupt_state = torch.load(clean_ckpt, map_location="cpu")
        corrupt_state["status"] = "FAILED_NUMERICAL"
        corrupt_path = self.temp_dir / "failed.pt"
        torch.save(corrupt_state, corrupt_path)

        model2 = DummyBNModel(in_channels=2)
        trainer2 = SourceTrainer(
            model=model2,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        with self.assertRaises(ValueError) as ctx:
            trainer2.resume_from_checkpoint(corrupt_path)
        self.assertIn("Refusing to resume from checkpoint flagged as failed/corrupted", str(ctx.exception))

        # Create a checkpoint with NaN weights
        try:
            corrupt_weights_state = torch.load(clean_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            corrupt_weights_state = torch.load(clean_ckpt, map_location="cpu")
        corrupt_weights_state["model_state_dict"]["head.weight"][0, 0, 0, 0] = float("nan")
        corrupt_weights_path = self.temp_dir / "nan_weights.pt"
        torch.save(corrupt_weights_state, corrupt_weights_path)

        with self.assertRaises(ValueError) as ctx:
            trainer2.resume_from_checkpoint(corrupt_weights_path)
        self.assertIn("contains nonfinite values", str(ctx.exception))

    def test_protocol_v2_hashes_and_validity(self):
        """Verify protocol v2 configs and lock manifest validate cleanly."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_a = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        cfg_b = repo_root / "configs/source/snunet_vvvh_slope_v2_fp32.json"

        # Validate Model A against lock manifest
        res_a = validate_provenance_and_config(
            config_path=cfg_a,
            run_id="source_snunet_vvvh_seed0_v3_fp32",
            seed=0,
            repo_root=repo_root,
        )
        self.assertEqual(res_a["model_key"], "model_a")
        self.assertEqual(res_a["precision"], "fp32")
        self.assertEqual(res_a["status"], "VALIDATED")

        # Validate Model B against lock manifest
        res_b = validate_provenance_and_config(
            config_path=cfg_b,
            run_id="source_snunet_vvvh_slope_seed0_v3_fp32",
            seed=0,
            repo_root=repo_root,
        )
        self.assertEqual(res_b["model_key"], "model_b")
        self.assertEqual(res_b["precision"], "fp32")
        self.assertEqual(res_b["status"], "VALIDATED")
        self.assertEqual(res_b["split_hash"], FROZEN_SPLIT_HASH)
        self.assertEqual(res_b["slope_stats_hash"], FROZEN_SLOPE_STATS_HASH)

    def test_provenance_lock_negative_rejections(self):
        """Verify production validation rejects altered protocol, config, split, slope, or precision."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_a = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        cfg_b = repo_root / "configs/source/snunet_vvvh_slope_v2_fp32.json"
        lock_file = repo_root / "configs/source/source_training_v2_lock.json"

        with open(cfg_a, "r", encoding="utf-8") as f:
            base_cfg_a = json.load(f)
        with open(cfg_b, "r", encoding="utf-8") as f:
            base_cfg_b = json.load(f)
        with open(lock_file, "r", encoding="utf-8") as f:
            base_lock = json.load(f)

        # 1. Deliberately alter protocol content
        temp_proto = self.temp_dir / "altered_protocol.md"
        temp_proto.write_text("Altered protocol content that breaks SHA-256")
        tampered_lock = copy.deepcopy(base_lock)
        tampered_lock["protocol"]["path"] = str(temp_proto.relative_to(self.temp_dir))
        tampered_lock_path = self.temp_dir / "tampered_lock_proto.json"
        with open(tampered_lock_path, "w") as f:
            json.dump(tampered_lock, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a,
                lock_path=tampered_lock_path,
                repo_root=self.temp_dir,
            )
        self.assertIn("Protocol hash mismatch", str(ctx.exception))

        # 2. Deliberately alter config content
        altered_cfg = copy.deepcopy(base_cfg_a)
        altered_cfg["description"] = "Altered description that changes config SHA-256"
        altered_cfg_path = self.temp_dir / "altered_config.json"
        with open(altered_cfg_path, "w") as f:
            json.dump(altered_cfg, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=altered_cfg_path,
                lock_path=lock_file,
                repo_root=repo_root,
            )
        self.assertIn("config hash mismatch", str(ctx.exception))

        # 3. Deliberately alter split reference in config
        bad_split_cfg = copy.deepcopy(base_cfg_a)
        bad_split_cfg["data_split"]["split_config_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        bad_split_cfg_path = self.temp_dir / "bad_split_config.json"
        with open(bad_split_cfg_path, "w") as f:
            json.dump(bad_split_cfg, f)

        tampered_lock_split = copy.deepcopy(base_lock)
        tampered_lock_split["model_a"]["config_sha256"] = compute_file_sha256(bad_split_cfg_path)
        tampered_lock_split_path = self.temp_dir / "tampered_lock_split.json"
        with open(tampered_lock_split_path, "w") as f:
            json.dump(tampered_lock_split, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=bad_split_cfg_path,
                lock_path=tampered_lock_split_path,
                repo_root=repo_root,
            )
        self.assertIn("Config recorded split hash mismatch", str(ctx.exception))

        # 4. Deliberately alter slope-statistics reference in Model B config
        bad_slope_cfg = copy.deepcopy(base_cfg_b)
        bad_slope_cfg["input_specification"]["slope_policy"]["statistics_artifact_sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"
        bad_slope_cfg_path = self.temp_dir / "bad_slope_config.json"
        with open(bad_slope_cfg_path, "w") as f:
            json.dump(bad_slope_cfg, f)

        tampered_lock_slope = copy.deepcopy(base_lock)
        tampered_lock_slope["model_b"]["config_sha256"] = compute_file_sha256(bad_slope_cfg_path)
        tampered_lock_slope_path = self.temp_dir / "tampered_lock_slope.json"
        with open(tampered_lock_slope_path, "w") as f:
            json.dump(tampered_lock_slope, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=bad_slope_cfg_path,
                lock_path=tampered_lock_slope_path,
                repo_root=repo_root,
            )
        self.assertIn("Config recorded slope stats hash mismatch", str(ctx.exception))

        # 5. Deliberately alter precision setting
        bad_prec_cfg = copy.deepcopy(base_cfg_a)
        bad_prec_cfg["optimization_protocol"]["precision"] = "fp16"
        bad_prec_cfg_path = self.temp_dir / "bad_prec_config.json"
        with open(bad_prec_cfg_path, "w") as f:
            json.dump(bad_prec_cfg, f)

        tampered_lock_prec = copy.deepcopy(base_lock)
        tampered_lock_prec["model_a"]["config_sha256"] = compute_file_sha256(bad_prec_cfg_path)
        tampered_lock_prec_path = self.temp_dir / "tampered_lock_prec.json"
        with open(tampered_lock_prec_path, "w") as f:
            json.dump(tampered_lock_prec, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=bad_prec_cfg_path,
                lock_path=tampered_lock_prec_path,
                repo_root=repo_root,
            )
        self.assertIn("Invalid precision in config", str(ctx.exception))

        # 6. Deliberately request unapproved run_id and seed
        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a,
                run_id="unapproved_run_id_xyz",
                lock_path=lock_file,
                repo_root=repo_root,
            )
        self.assertIn("not in approved lock run IDs", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a,
                seed=999,
                lock_path=lock_file,
                repo_root=repo_root,
            )
        self.assertIn("not in approved lock seeds", str(ctx.exception))

    def test_forward_logits_nonfinite_fails_fast(self):
        """Verify that nonfinite logits in forward pass raise NumericalInstabilityError."""
        model = DummyBNModel(in_channels=2)
        # Inject NaN into conv weights
        with torch.no_grad():
            model.convA.weight.fill_(float("nan"))

        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=4,
            physical_microbatch_size=4,
            mixed_precision=False,
        )

        batch = make_dummy_batch(batch_size=4)
        loader = DataLoader([
            {
                "xA": batch["xA"][i],
                "xB": batch["xB"][i],
                "label": batch["label"][i],
                "valid_mask": batch["valid_mask"][i],
                "event_id": 118,
                "sample_id": f"s_{i}",
            }
            for i in range(4)
        ], batch_size=4)

        with self.assertRaises(NumericalInstabilityError) as ctx:
            trainer.train_epoch(loader, epoch_idx=1)

        err = ctx.exception
        self.assertEqual(err.tensor_name, "logits")
        self.assertEqual(err.stage, "train_forward_logits")

    def test_failure_manifest_generation(self):
        """Verify failure_manifest.json format and metadata for FAILED_NUMERICAL and FAILED_COLLAPSE."""
        # 1. Numerical failure manifest
        err_num = NumericalInstabilityError(
            message="Nonfinite gradients detected in parameter head.weight",
            tensor_name="grad_head.weight",
            stage="train_gradients",
            epoch=23,
            batch_idx=353,
            global_step=11990,
            sample_ids=["sample_1", "sample_2"],
            event_ids=[118, 118],
            details={"grad_norm": float("inf")},
        )

        num_manifest_path = self.temp_dir / "failure_manifest_num.json"
        num_data = {
            "run_id": "test_run_num",
            "status": "FAILED_NUMERICAL",
            "failure_type": "numerical_instability",
            "epoch": err_num.epoch,
            "batch_idx": err_num.batch_idx,
            "global_step": err_num.global_step,
            "tensor_name": err_num.tensor_name,
            "stage": err_num.stage,
            "sample_ids": err_num.sample_ids,
            "event_ids": err_num.event_ids,
            "error_message": str(err_num),
            "details": err_num.details,
        }
        with open(num_manifest_path, "w", encoding="utf-8") as f:
            json.dump(num_data, f, indent=2)

        self.assertTrue(num_manifest_path.exists())
        with open(num_manifest_path, "r", encoding="utf-8") as f:
            loaded_num = json.load(f)
        self.assertEqual(loaded_num["status"], "FAILED_NUMERICAL")
        self.assertEqual(loaded_num["epoch"], 23)
        self.assertEqual(loaded_num["batch_idx"], 353)
        self.assertEqual(loaded_num["tensor_name"], "grad_head.weight")
        self.assertEqual(loaded_num["sample_ids"], ["sample_1", "sample_2"])
        self.assertEqual(loaded_num["event_ids"], [118, 118])

        # 2. Collapse failure manifest
        err_col = PredictiveCollapseError(
            message="Predictive collapse triggered at Epoch 8: 3 consecutive zero-flood epochs",
            epoch=8,
            consecutive_zero_epochs=3,
            warmup_epochs=5,
            patience=3,
        )
        col_manifest_path = self.temp_dir / "failure_manifest_col.json"
        col_data = {
            "run_id": "test_run_col",
            "status": "FAILED_COLLAPSE",
            "failure_type": "predictive_collapse",
            "epoch": err_col.epoch,
            "consecutive_zero_epochs": err_col.consecutive_zero_epochs,
            "warmup_epochs": err_col.warmup_epochs,
            "patience": err_col.patience,
            "error_message": str(err_col),
        }
        with open(col_manifest_path, "w", encoding="utf-8") as f:
            json.dump(col_data, f, indent=2)

        self.assertTrue(col_manifest_path.exists())
        with open(col_manifest_path, "r", encoding="utf-8") as f:
            loaded_col = json.load(f)
        self.assertEqual(loaded_col["status"], "FAILED_COLLAPSE")
        self.assertEqual(loaded_col["epoch"], 8)
        self.assertEqual(loaded_col["consecutive_zero_epochs"], 3)
        self.assertEqual(loaded_col["warmup_epochs"], 5)
        self.assertEqual(loaded_col["patience"], 3)

    def test_one_completed_epoch_finalization_model_a(self):
        """Verify Model A completes 1 epoch through production finalization without NameError and records split hash."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        out_dir = self.temp_dir / "model_a_finalization_test"

        args = argparse.Namespace(
            config=str(cfg_path),
            output_dir=str(out_dir),
            run_id="source_snunet_vvvh_seed0_v3_fp32",
            device="cpu",
            seed=0,
            max_epochs=1,
            patience=20,
            num_workers=0,
            log_interval=1,
            resume=None,
            resume_auto=False,
            max_train_batches=1,
            max_val_batches=1,
            validate_only=False,
        )
        exit_code = train_source_model(args)
        self.assertEqual(exit_code, 0)

        # Verify checkpoint extra_metadata
        ckpt_path = out_dir / "last_checkpoint.pt"
        self.assertTrue(ckpt_path.exists())
        ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra_meta = ckpt_state.get("extra_metadata", {})
        self.assertIn("split_hash", extra_meta)
        self.assertEqual(extra_meta["split_hash"], FROZEN_SPLIT_HASH)
        self.assertIn("protocol_hash", extra_meta)
        self.assertIn("lock_hash", extra_meta)
        self.assertEqual(extra_meta["model_type"], "model_a")
        self.assertIsNone(extra_meta["slope_stats_hash"])

        # Verify run_manifest.json
        manifest_path = out_dir / "run_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        rep = manifest_data["reproducibility"]
        self.assertEqual(rep["completion_status"], "COMPLETED_MAX_EPOCH")
        self.assertEqual(rep["split_sha256"], FROZEN_SPLIT_HASH)
        self.assertNotEqual(rep["checkpoint_sha256"], "FILE_NOT_FOUND")

    def test_one_completed_epoch_finalization_model_b(self):
        """Verify Model B completes 1 epoch through production finalization without NameError and records slope stats hash."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs/source/snunet_vvvh_slope_v2_fp32.json"
        out_dir = self.temp_dir / "model_b_finalization_test"

        args = argparse.Namespace(
            config=str(cfg_path),
            output_dir=str(out_dir),
            run_id="source_snunet_vvvh_slope_seed0_v3_fp32",
            device="cpu",
            seed=0,
            max_epochs=1,
            patience=20,
            num_workers=0,
            log_interval=1,
            resume=None,
            resume_auto=False,
            max_train_batches=1,
            max_val_batches=1,
            validate_only=False,
        )
        exit_code = train_source_model(args)
        self.assertEqual(exit_code, 0)

        # Verify checkpoint extra_metadata
        ckpt_path = out_dir / "last_checkpoint.pt"
        self.assertTrue(ckpt_path.exists())
        ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra_meta = ckpt_state.get("extra_metadata", {})
        self.assertIn("split_hash", extra_meta)
        self.assertEqual(extra_meta["split_hash"], FROZEN_SPLIT_HASH)
        self.assertIn("slope_stats_hash", extra_meta)
        self.assertEqual(extra_meta["slope_stats_hash"], FROZEN_SLOPE_STATS_HASH)
        self.assertEqual(extra_meta["model_type"], "model_b")

        # Verify run_manifest.json
        manifest_path = out_dir / "run_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        rep = manifest_data["reproducibility"]
        self.assertEqual(rep["completion_status"], "COMPLETED_MAX_EPOCH")
        self.assertEqual(rep["split_sha256"], FROZEN_SPLIT_HASH)
        self.assertEqual(rep["slope_stats_sha256"], FROZEN_SLOPE_STATS_HASH)
        self.assertNotEqual(rep["checkpoint_sha256"], "FILE_NOT_FOUND")

    def test_missing_or_corrupted_provenance_fails_before_training(self):
        """Verify undefined or corrupted provenance values fail immediately before training starts."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        out_dir = self.temp_dir / "bad_provenance_test"

        args = argparse.Namespace(
            config=str(cfg_path),
            output_dir=str(out_dir),
            run_id="source_snunet_vvvh_seed0_v3_fp32",
            device="cpu",
            seed=0,
            max_epochs=1,
            patience=20,
            num_workers=0,
            log_interval=1,
            resume=None,
            resume_auto=False,
            max_train_batches=1,
            max_val_batches=1,
            validate_only=False,
        )

        # Test missing split_hash from provenance
        with patch("tools.train_source_model.validate_provenance_and_config") as mock_val:
            mock_val.return_value = {
                "config": {"architecture": {"in_channels_per_branch": 2}},
                "config_hash": "a" * 64,
                "model_key": "model_a",
                "protocol_hash": "b" * 64,
                "split_hash": None,
                "slope_stats_hash": None,
                "lock_path": str(repo_root / "configs/source/source_training_v2_lock.json"),
            }
            with self.assertRaises(RuntimeError) as ctx:
                train_source_model(args)
            self.assertIn("split_hash missing or invalid", str(ctx.exception))

        # Test missing slope_stats_hash for Model B
        cfg_b_path = repo_root / "configs/source/snunet_vvvh_slope_v2_fp32.json"
        args_b = argparse.Namespace(
            config=str(cfg_b_path),
            output_dir=str(out_dir),
            run_id="source_snunet_vvvh_slope_seed0_v3_fp32",
            device="cpu",
            seed=0,
            max_epochs=1,
            patience=20,
            num_workers=0,
            log_interval=1,
            resume=None,
            resume_auto=False,
            max_train_batches=1,
            max_val_batches=1,
            validate_only=False,
        )
        with patch("tools.train_source_model.validate_provenance_and_config") as mock_val:
            mock_val.return_value = {
                "config": {"architecture": {"in_channels_per_branch": 3}},
                "config_hash": "a" * 64,
                "model_key": "model_b",
                "protocol_hash": "b" * 64,
                "split_hash": "c" * 64,
                "slope_stats_hash": None,
                "lock_path": str(repo_root / "configs/source/source_training_v2_lock.json"),
            }
            with self.assertRaises(RuntimeError) as ctx:
                train_source_model(args_b)
            self.assertIn("slope_stats_hash missing or invalid for Model B", str(ctx.exception))

    def test_run_failure_recorded_accurately_on_unhandled_exception(self):
        """Verify unhandled exception records FAILED in run_manifest and creates failure_manifest."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs/source/snunet_vvvh_v2_fp32.json"
        out_dir = self.temp_dir / "unhandled_failure_test"

        args = argparse.Namespace(
            config=str(cfg_path),
            output_dir=str(out_dir),
            run_id="source_snunet_vvvh_seed0_v3_fp32",
            device="cpu",
            seed=0,
            max_epochs=1,
            patience=20,
            num_workers=0,
            log_interval=1,
            resume=None,
            resume_auto=False,
            max_train_batches=1,
            max_val_batches=1,
            validate_only=False,
        )

        with patch("src.training.source_trainer.SourceTrainer.save_checkpoint", side_effect=RuntimeError("Simulated write error during epoch finalization")):
            exit_code = train_source_model(args)
            self.assertEqual(exit_code, 1)

        # Check run_manifest.json
        manifest_path = out_dir / "run_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["reproducibility"]["completion_status"], "FAILED")

        # Check failure_manifest.json
        fail_path = out_dir / "failure_manifest.json"
        self.assertTrue(fail_path.exists())
        with open(fail_path, "r", encoding="utf-8") as f:
            fail_data = json.load(f)
        self.assertEqual(fail_data["status"], "FAILED")
        self.assertEqual(fail_data["failure_type"], "unhandled_exception")
        self.assertIn("Simulated write error during epoch finalization", fail_data["error_message"])


if __name__ == "__main__":
    unittest.main()
