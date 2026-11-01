"""
Focused CPU verification tests for Week 9 Stronger TTA Baselines (EATA and SAR-TTA).

Tests verify:
1. Label and split isolation.
2. Fisher arithmetic, parameter-name matching, and source weight immutability.
3. EATA reliability filtering, redundancy filtering, entropy reweighting, Fisher regularization, and empty selection safety.
4. SAR two-stage SAM, safe perturbation restoration on aborted stage 2, configurable recovery threshold, and reset state.
5. Accuracy of selection, update, skip, and recovery counters.
6. 24-stream matrix enumeration and normal CLI runner execution on synthetic data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gzip
import pickle

from src.data.evaluation_dataset import KuroEvaluationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.eata_adapter import EATAAdapter
from src.tta.sar_adapter import SARAdapter
from src.tta.sam import SAM
from tools.prepare_eata_fishers import (
    compute_empirical_fisher_for_checkpoint,
    get_deterministic_source_sample_ids,
    compute_file_sha256,
)
from tools.run_stronger_tta_matrix import StrongerTTAMatrixRunner, MockSNUNet

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyMockSNUNet(nn.Module):
    """Tiny synthetic dual-input change detection model with BatchNorm for fast CPU tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.classifier = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.classifier(x)


class TestStrongerTTA(unittest.TestCase):
    """Focused CPU verification test suite for Week 9 TTA components."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        self.model = TinyMockSNUNet()
        self.ckpt_path = self.tmp_path / "mock_checkpoint.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "epoch": 16}, self.ckpt_path)
        self.wrapper = self.model

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_label_and_split_isolation(self):
        """Ensures adapters reject label inputs and verifies split partition isolation."""
        eata = EATAAdapter(self.wrapper, str(self.ckpt_path), device="cpu")
        sar = SARAdapter(self.wrapper, str(self.ckpt_path), device="cpu")

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        labels = torch.randint(0, 3, (2, 16, 16))

        # adapt_and_predict should reject any attempt to pass ground-truth labels
        with self.assertRaises((ValueError, TypeError)):
            eata.adapt_and_predict(xA, xB, labels=labels)

        with self.assertRaises((ValueError, TypeError)):
            sar.adapt_and_predict(xA, xB, label=labels)

        # Verify frozen split file isolation
        split_path = REPO_ROOT / "configs/splits/kuro_source_all27_split_v1.json"
        with open(split_path, "r") as f:
            split_data = json.load(f)

        source_events = set(split_data["source_all_events"])
        dev_events = set(split_data["tta_validation_events"])
        sealed_events = set(split_data["sealed_test_events"])

        self.assertEqual(len(source_events & dev_events), 0, "Source events overlap with dev events!")
        self.assertEqual(len(source_events & sealed_events), 0, "Source events overlap with sealed events!")
        self.assertEqual(len(dev_events & sealed_events), 0, "Dev events overlap with sealed events!")
        self.assertEqual(len(source_events), 27)

    def test_02_fisher_arithmetic_and_source_immutability(self):
        """Verifies empirical Fisher computation arithmetic and source weight immutability."""
        initial_state = copy.deepcopy(self.model.state_dict())

        def mock_loader(sample_ids):
            for s in sample_ids:
                yield {
                    "xA": torch.randn(1, 2, 16, 16),
                    "xB": torch.randn(1, 2, 16, 16),
                    "sar_valid_mask": torch.ones(1, 1, 16, 16),
                }

        sample_ids = [f"s_{i}" for i in range(5)]
        fishers, stats = compute_empirical_fisher_for_checkpoint(
            self.model, sample_ids, mock_loader, device="cpu", accumulation_formula="exact_per_sample"
        )

        # 1. Parameter matching: only BN weight and bias
        self.assertIn("bn1.weight", fishers)
        self.assertIn("bn1.bias", fishers)
        self.assertIn("bn2.weight", fishers)
        self.assertIn("bn2.bias", fishers)
        self.assertNotIn("conv1.weight", fishers)
        self.assertNotIn("classifier.weight", fishers)

        # 2. Arithmetic properties
        for name, (f_diag, p_init) in fishers.items():
            self.assertTrue(torch.all(f_diag >= 0.0), f"Negative Fisher in {name}")
            self.assertTrue(torch.all(torch.isfinite(f_diag)), f"Non-finite Fisher in {name}")
            self.assertEqual(f_diag.shape, p_init.shape)

        # 3. Model weights unchanged
        current_state = self.model.state_dict()
        for k in initial_state:
            self.assertTrue(torch.equal(initial_state[k], current_state[k]), f"Weight changed during Fisher: {k}")

    def test_03_eata_filtering_reweighting_and_empty_selection(self):
        """Tests EATA reliability filter, redundancy filter, reweighting, and empty selection."""
        # Case A: Empty selection handling: set e_margin very low so all samples are rejected
        eata_empty = EATAAdapter(self.wrapper, str(self.ckpt_path), e_margin=0.00001, device="cpu")
        initial_p = [p.clone() for p in eata_empty.model.parameters() if p.requires_grad]

        xA = torch.randn(4, 2, 16, 16)
        xB = torch.randn(4, 2, 16, 16)
        mask = torch.ones(4, 1, 16, 16)

        res = eata_empty.adapt_and_predict(xA, xB, sar_valid_mask=mask)

        # Should skip update cleanly without error
        self.assertEqual(res["step_diagnostics"]["num_active_in_batch"], 0)
        self.assertEqual(eata_empty.num_skipped_updates, 1)
        self.assertEqual(eata_empty.num_adaptation_steps, 0)

        current_p = [p for p in eata_empty.model.parameters() if p.requires_grad]
        for p_init, p_curr in zip(initial_p, current_p):
            self.assertTrue(torch.equal(p_init, p_curr), "Weights modified on empty selection!")

        # Case B: Active selection with standard threshold
        eata_active = EATAAdapter(self.wrapper, str(self.ckpt_path), e_margin=2.0, device="cpu")
        res2 = eata_active.adapt_and_predict(xA, xB, sar_valid_mask=mask)

        self.assertEqual(res2["step_diagnostics"]["num_reliable_in_batch"], 4)
        self.assertEqual(res2["step_diagnostics"]["num_active_in_batch"], 4)
        self.assertEqual(eata_active.num_adaptation_steps, 1)
        self.assertIsNotNone(eata_active.current_model_probs)
        self.assertEqual(eata_active.current_model_probs.shape, (3,))

    def test_04_sar_two_stage_sam_perturbation_restore_and_recovery(self):
        """Tests SAR two-stage SAM, safe perturbation restore, and configurable recovery."""
        sar = SARAdapter(
            self.wrapper,
            str(self.ckpt_path),
            margin_e0=2.0,
            reset_constant_em=0.03180808,
            device="cpu",
        )

        xA = torch.randn(4, 2, 16, 16)
        xB = torch.randn(4, 2, 16, 16)
        mask = torch.ones(4, 1, 16, 16)

        # Standard step
        res = sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        self.assertEqual(sar.num_adaptation_steps, 1)
        self.assertIsNotNone(sar.ema)

        # Test SAM perturbation restoration safety:
        # If SAM climbs to perturbed state, restore_perturbation restores parameters to unperturbed state
        params = [p for p in sar.model.parameters() if p.requires_grad]
        orig_data = [p.clone() for p in params]

        # Artificially trigger first_step
        loss = sum(p.sum() for p in params)
        sar.optimizer.zero_grad()
        loss.backward()
        sar.optimizer.first_step(zero_grad=True)

        # Verify parameters changed
        changed = any(not torch.equal(p, o) for p, o in zip(params, orig_data))
        self.assertTrue(changed, "Parameters not perturbed after first_step!")

        # Call restore_perturbation
        sar.optimizer.restore_perturbation(zero_grad=True)
        for p, o in zip(params, orig_data):
            self.assertTrue(torch.equal(p, o), "restore_perturbation failed to restore original weights!")

        # Test configurable recovery trigger
        # With default threshold ~0.0318, normal loss ~1.0 does not trigger recovery
        self.assertEqual(sar.num_recoveries, 0)

        # When reset_constant_em is set above current loss EMA, recovery triggers
        sar.reset_constant_em = 10.0
        sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        self.assertGreaterEqual(sar.num_recoveries, 1)

    def test_05_selection_and_update_counters(self):
        """Tests accuracy of tracking counters across multiple batches with redundant inputs."""
        eata = EATAAdapter(self.wrapper, str(self.ckpt_path), e_margin=2.0, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 1, 16, 16)

        for _ in range(3):
            eata.adapt_and_predict(xA, xB, sar_valid_mask=mask)

        # Batch 1 accepted (2 active); Batches 2 & 3 filtered out as redundant (4 redundant)
        self.assertEqual(eata.step_count, 3)
        self.assertEqual(eata.num_reliable_samples, 6)
        self.assertEqual(eata.num_redundant_samples, 4)
        self.assertEqual(eata.num_active_samples, 2)
        self.assertEqual(eata.num_adaptation_steps, 1)
        self.assertEqual(eata.num_skipped_updates, 2)

        # Reset to source restores all counters to zero
        eata.reset_to_source()
        self.assertEqual(eata.step_count, 0)
        self.assertEqual(eata.num_reliable_samples, 0)
        self.assertEqual(eata.num_redundant_samples, 0)
        self.assertEqual(eata.num_active_samples, 0)
        self.assertEqual(eata.num_adaptation_steps, 0)
        self.assertEqual(eata.num_skipped_updates, 0)

    def test_06_matrix_enumeration_and_synthetic_runner(self):
        """Verifies 24-stream matrix enumeration and runner execution on synthetic data."""
        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"

        runner = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=self.tmp_path / "synthetic_run",
            device="cpu",
            allow_overwrite=True,
            synthetic_data=True,
        )

        streams = runner.build_matrix_streams()
        self.assertEqual(len(streams), 24, f"Expected 24 streams, got {len(streams)}")

        methods = {s["method"] for s in streams}
        self.assertEqual(methods, {"eata", "sar"})

        seeds = {s["checkpoint_seed"] for s in streams}
        self.assertEqual(seeds, {0, 1, 2})

        orders = {s["stream_order"] for s in streams}
        self.assertEqual(orders, {"order_a", "order_b"})

        resets = {s["reset_mode"] for s in streams}
        self.assertEqual(resets, {"reset_per_event", "continual"})

        # Run smoke matrix on synthetic data (4 streams)
        manifest = runner.run_matrix(is_smoke=True)
        self.assertTrue(manifest["is_smoke"])
        self.assertEqual(manifest["total_physical_streams"], 4)
        self.assertTrue((self.tmp_path / "synthetic_run" / "stream_event_metrics.csv").is_file())
        self.assertTrue((self.tmp_path / "synthetic_run" / "week9_tta_manifest.json").is_file())

    def test_07_configuration_propagation_and_cli_batch_rejection(self):
        """Verifies canonical adaptation batch size is 16 and conflicting CLI overrides are rejected."""
        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # 1. Canonical configuration field exists and equals 16
        self.assertIn("stream_execution_policy", cfg)
        self.assertEqual(cfg["stream_execution_policy"]["batch_size"], 16)

        # 2. Separate Fisher batch size exists and is strictly separate
        self.assertIn("fisher_information_protocol", cfg)
        self.assertEqual(cfg["fisher_information_protocol"]["fisher_batch_size"], 1)

        # 3. Runner correctly reads canonical batch size 16
        runner = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=self.tmp_path / "cfg_test",
            device="cpu",
            synthetic_data=False,
        )
        self.assertEqual(runner.batch_size, 16)

    def test_08_zero_sensory_valid_mask_safety(self):
        """Verifies EATA and SAR safely skip adaptation on all-zero sensory valid masks without NaN."""
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        zero_mask = torch.zeros(2, 1, 16, 16)

        eata = EATAAdapter(self.wrapper, str(self.ckpt_path), device="cpu")
        sar = SARAdapter(self.wrapper, str(self.ckpt_path), device="cpu")

        out_eata = eata.adapt_and_predict(xA, xB, sar_valid_mask=zero_mask)
        self.assertEqual(eata.num_adaptation_steps, 0)
        self.assertEqual(eata.num_reliable_samples, 0)
        self.assertEqual(out_eata["step_diagnostics"]["num_reliable_in_batch"], 0)
        self.assertEqual(out_eata["step_diagnostics"]["num_active_in_batch"], 0)
        self.assertEqual(out_eata["step_diagnostics"]["num_zero_valid_in_batch"], 2)

        out_sar = sar.adapt_and_predict(xA, xB, sar_valid_mask=zero_mask)
        self.assertEqual(sar.num_adaptation_steps, 0)
        self.assertEqual(sar.num_reliable_samples, 0)
        self.assertEqual(out_sar["step_diagnostics"]["num_reliable_in_batch"], 0)
        self.assertEqual(out_sar["step_diagnostics"]["num_zero_valid_in_batch"], 2)

    def test_09_sam_stage2_robustness(self):
        """Tests that SAM restores unperturbed weights even if stage 2 has missing gradient or raises exception."""
        # 1. Missing stage 2 gradient for some parameters:
        p = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        q = nn.Parameter(torch.tensor(2.0, requires_grad=True))
        sam = SAM([p, q], torch.optim.SGD, lr=0.01)

        (p + q).backward()
        sam.first_step(zero_grad=True)
        self.assertTrue(sam.perturbed)

        # Only p receives gradient in stage 2; q has None
        p.backward()
        sam.second_step(zero_grad=True)
        self.assertFalse(sam.perturbed)
        # q should be restored to its unperturbed value (2.0)
        self.assertAlmostEqual(float(q.detach()), 2.0, places=5)

        # 2. Stage 2 forward raises an exception:
        sar = SARAdapter(self.wrapper, str(self.ckpt_path), margin_e0=2.0, device="cpu")
        initial_params = {n: p.clone() for n, p in sar.model.named_parameters() if p.requires_grad}

        # Mock second forward pass to raise RuntimeError
        orig_forward = sar.model.forward
        calls = [0]

        def faulty_forward(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("Synthetic Stage 2 Forward Failure")
            return orig_forward(*args, **kwargs)

        sar.model.forward = faulty_forward
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 1, 16, 16)

        with self.assertRaises(RuntimeError):
            sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)

        # Parameters should be fully restored to unperturbed weights
        self.assertFalse(sar.optimizer.perturbed)
        for n, p in sar.model.named_parameters():
            if p.requires_grad:
                self.assertTrue(torch.equal(p, initial_params[n]), f"Parameter {n} was not restored after exception!")

    def test_10_sar_repeated_recovery_counter_persistence(self):
        """Tests that SAR recovery resets model weights but preserves cumulative audit counters."""
        sar = SARAdapter(
            self.wrapper,
            str(self.ckpt_path),
            margin_e0=2.0,
            reset_constant_em=100.0,  # Ensure recovery triggers every step
            device="cpu",
        )
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 1, 16, 16)

        out1 = sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        self.assertTrue(out1["step_diagnostics"]["recovered_this_step"])
        self.assertEqual(sar.num_recoveries, 1)
        self.assertEqual(sar.step_count, 1)

        out2 = sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)
        self.assertTrue(out2["step_diagnostics"]["recovered_this_step"])
        self.assertEqual(sar.num_recoveries, 2)
        self.assertEqual(sar.step_count, 2)

    def test_11_fisher_accumulation_validation_and_source_id_extraction(self):
        """Tests rejection of unsupported batch_mean and verifies source ID extraction with sealed test guards."""
        def mock_loader(sample_ids):
            yield {
                "xA": torch.randn(1, 2, 16, 16),
                "xB": torch.randn(1, 2, 16, 16),
                "sar_valid_mask": torch.ones(1, 1, 16, 16),
            }

        # 1. Unsupported batch_mean raises ValueError immediately
        with self.assertRaises(ValueError):
            compute_empirical_fisher_for_checkpoint(
                self.model, ["s1"], mock_loader, device="cpu", accumulation_formula="batch_mean"
            )

        # 2. Source ID extraction handles nested info.actid and rejects sealed test events
        mock_manifest_path = self.tmp_path / "mock_manifest.gz"
        mock_manifest_data = {
            "s1": {"info": {"actid": 118}, "path": "path1"},
            "s2": {"info": {"actid": 128}, "path": "path2"},
            "s3": {"info": {"actid": 437}, "path": "path3"},  # sealed test event!
        }
        with gzip.open(mock_manifest_path, "wb") as f:
            pickle.dump(mock_manifest_data, f)

        mock_split_path = self.tmp_path / "mock_split.json"
        with open(mock_split_path, "w") as f:
            json.dump({"source_all_events": [118, 128]}, f)

        source_ids, stats = get_deterministic_source_sample_ids(
            mock_manifest_path, mock_split_path, sample_size=2
        )
        self.assertEqual(len(source_ids), 2)
        self.assertIn("s1", source_ids)
        self.assertIn("s2", source_ids)
        self.assertNotIn("s3", source_ids)

    def test_12_preflight_fail_closed_and_overwrite_protection(self):
        """Verifies runner fails closed on missing checkpoints/fishers and prevents formal directory overwrites."""
        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Point checkpoint and fisher to non-existent paths
        cfg["source_checkpoints"]["seeds"]["0"]["checkpoint_path"] = str(self.tmp_path / "missing_ckpt.pt")
        bad_config_path = self.tmp_path / "bad_config.json"
        with open(bad_config_path, "w") as f:
            json.dump(cfg, f)

        runner = StrongerTTAMatrixRunner(
            config_path=bad_config_path,
            output_dir=self.tmp_path / "fail_closed_test",
            device="cpu",
            synthetic_data=False,
        )

        with self.assertRaises((FileNotFoundError, ValueError)):
            runner.validate_preflight(is_smoke=False, fail_closed=True)

        # Overwrite protection for formal dir
        formal_dir = self.tmp_path / "existing_formal"
        formal_dir.mkdir(parents=True)
        (formal_dir / "existing_file.txt").write_text("existing")

        runner_formal = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=formal_dir,
            device="cpu",
            synthetic_data=True,
        )
        with self.assertRaises(FileExistsError):
            runner_formal.run_matrix(is_smoke=False)

    def test_13_runner_kuro_dataset_contract(self):
        """Verifies runner physical stream loop directly against KuroEvaluationDataset interface."""
        mock_records = [
            {
                "sample_id": f"s_{i:03d}",
                "event_id": 437,
                "xA": np.zeros((2, 16, 16), dtype=np.float32),
                "xB": np.zeros((2, 16, 16), dtype=np.float32),
                "valid_mask": np.ones((16, 16), dtype=np.uint8),
                "label": np.zeros((16, 16), dtype=np.int64),
            }
            for i in range(19)
        ] + [
            {
                "sample_id": f"s2_{i:03d}",
                "event_id": 1111003,
                "xA": np.zeros((2, 16, 16), dtype=np.float32),
                "xB": np.zeros((2, 16, 16), dtype=np.float32),
                "valid_mask": np.ones((16, 16), dtype=np.uint8),
                "label": np.zeros((16, 16), dtype=np.int64),
            }
            for i in range(19)
        ]

        ds = KuroEvaluationDataset.from_mock_records(mock_records, allowed_events=[437, 1111003])

        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        runner = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=self.tmp_path / "contract_test",
            device="cpu",
            synthetic_data=True,
        )
        spec = runner.build_matrix_streams(is_smoke=True)[0]
        event_rows, diags = runner.run_physical_stream(spec, ds, ds, is_smoke=True)

        self.assertEqual(len(event_rows), 2)
        self.assertEqual(event_rows[0]["sample_count"], 19)
        self.assertEqual(event_rows[1]["sample_count"], 19)
        self.assertIn("flood_f1", event_rows[0])

    def test_14_fisher_provenance_and_metadata_binding(self):
        """W9-R03: Verifies Fisher source selection captures full provenance and binds SHA-256 digests."""
        mock_manifest_path = self.tmp_path / "prov_manifest.gz"
        mock_manifest_data = {
            f"s_{i}": {"info": {"actid": 118}, "path": f"p_{i}"} for i in range(10)
        }
        with gzip.open(mock_manifest_path, "wb") as f:
            pickle.dump(mock_manifest_data, f)

        mock_split_path = self.tmp_path / "prov_split.json"
        with open(mock_split_path, "w") as f:
            json.dump({"source_all_events": [118]}, f)

        source_ids, meta = get_deterministic_source_sample_ids(
            mock_manifest_path, mock_split_path, sample_size=5, selection_seed=2026
        )
        self.assertEqual(len(source_ids), 5)
        self.assertIn("selected_sample_ids", meta)
        self.assertEqual(meta["selected_sample_ids"], source_ids)
        self.assertIn("selected_sample_ids_sha256", meta)
        self.assertIn("manifest_sha256", meta)
        self.assertIn("split_config_sha256", meta)
        self.assertEqual(meta["selection_seed"], 2026)

        # Split SHA mismatch triggers ValueError
        with self.assertRaises(ValueError):
            get_deterministic_source_sample_ids(
                mock_manifest_path,
                mock_split_path,
                sample_size=5,
                expected_split_sha256="wrong_hash_value_expected_to_fail",
            )

    def test_14b_fisher_cli_split_hash_enforcement(self):
        """W9-R03: Verifies prepare_eata_fishers.py CLI reads and enforces canonical split_config_sha256."""
        from tools.prepare_eata_fishers import main as prepare_main
        import sys

        mock_manifest = self.tmp_path / "cli_manifest.gz"
        with gzip.open(mock_manifest, "wb") as f:
            pickle.dump({f"s_{i}": {"info": {"actid": 118}} for i in range(5)}, f)

        correct_split = self.tmp_path / "cli_correct_split.json"
        correct_split.write_text(json.dumps({"source_all_events": [118]}))
        correct_sha = compute_file_sha256(correct_split)

        corrupt_split = self.tmp_path / "cli_corrupt_split.json"
        corrupt_split.write_text(json.dumps({"source_all_events": [118, 437]}))

        cli_cfg = {
            "source_data_configuration": {
                "manifest_path": str(mock_manifest),
                "split_config_path": str(corrupt_split),
                "split_config_sha256": correct_sha,  # Config specifies correct split hash!
            },
            "source_checkpoints": {
                "seeds": {
                    "0": {"checkpoint_path": str(mock_manifest), "expected_sha256": compute_file_sha256(mock_manifest)}
                }
            },
            "fisher_information_protocol": {
                "artifact_paths": {
                    "0": str(self.tmp_path / "cli_fisher0.pt")
                }
            }
        }
        cfg_file = self.tmp_path / "cli_test_cfg.json"
        cfg_file.write_text(json.dumps(cli_cfg))

        # CLI dry_run must reject the corrupt split whose hash does not match canonical split_config_sha256
        orig_argv = sys.argv
        sys.argv = ["prepare_eata_fishers.py", "--config", str(cfg_file), "--seed", "0", "--sample_size", "2", "--dry_run"]
        try:
            with self.assertRaises(ValueError) as ctx:
                prepare_main()
            self.assertIn("Split config SHA-256 mismatch", str(ctx.exception))

            # When pointed to the correct split matching split_config_sha256, CLI dry_run passes
            cli_cfg["source_data_configuration"]["split_config_path"] = str(correct_split)
            cfg_file.write_text(json.dumps(cli_cfg))
            prepare_main()  # Must not raise
        finally:
            sys.argv = orig_argv

    def test_15_fisher_preflight_destination_protection(self):
        """W9-N01: Verifies Fisher script destination preflight rejects existing files across seeds."""
        from tools.prepare_eata_fishers import main as prepare_main
        import sys

        # Create dummy existing destination file and config
        out_dir = self.tmp_path / "fishers_exist_test"
        out_dir.mkdir(parents=True)
        existing_file = out_dir / "snunet_s0_eata_fisher.pt"
        existing_file.write_text("existing content")

        manifest_file = self.tmp_path / "dummy_manifest.pkl"
        with open(manifest_file, "wb") as f:
            pickle.dump({"s1": {"event_id": 118}}, f)
        split_file = self.tmp_path / "dummy_split.json"
        split_file.write_text(json.dumps({"source_all_events": [118]}))
        split_sha = compute_file_sha256(split_file)

        dummy_cfg = {
            "source_data_configuration": {
                "manifest_path": str(manifest_file),
                "split_config_path": str(split_file),
                "split_config_sha256": split_sha,
            },
            "source_checkpoints": {
                "seeds": {
                    "0": {"checkpoint_path": str(manifest_file), "expected_sha256": compute_file_sha256(manifest_file)}
                }
            },
            "fisher_information_protocol": {
                "artifact_paths": {
                    "0": str(existing_file.relative_to(REPO_ROOT) if existing_file.is_relative_to(REPO_ROOT) else existing_file)
                }
            }
        }
        cfg_path = self.tmp_path / "dummy_fisher_cfg.json"
        cfg_path.write_text(json.dumps(dummy_cfg))

        test_args = [
            "prepare_eata_fishers.py",
            "--config", str(cfg_path),
            "--seed", "0",
            "--sample_size", "1",
            "--dry_run",
        ]
        orig_argv = sys.argv
        sys.argv = test_args
        try:
            with self.assertRaises(FileExistsError):
                prepare_main()
        finally:
            sys.argv = orig_argv

    def test_16_fisher_accumulation_denominator_and_finite_checks(self):
        """W9-N02: Verifies zero-valid exclusion, evaluated_count normalization, and non-finite failure."""
        # 1. Zero valid patch handling and normalization convention
        def mock_loader_with_zero_valid(sample_ids):
            for i, s in enumerate(sample_ids):
                # 1st sample is valid, 2nd sample has zero valid mask
                mask_val = 1.0 if i == 0 else 0.0
                yield {
                    "xA": torch.randn(1, 2, 16, 16),
                    "xB": torch.randn(1, 2, 16, 16),
                    "sar_valid_mask": torch.full((1, 1, 16, 16), mask_val),
                }

        sample_ids = ["s0", "s1"]
        fishers, stats = compute_empirical_fisher_for_checkpoint(
            self.model, sample_ids, mock_loader_with_zero_valid, device="cpu"
        )
        self.assertEqual(stats["total_samples_processed"], 2)
        self.assertEqual(stats["num_evaluated_patches"], 1)
        self.assertEqual(stats["num_zero_valid_patches"], 1)
        self.assertEqual(stats["normalization_convention"], "mean_over_evaluated_patches")

        # 2. Incomplete iteration raises ValueError
        def mock_loader_incomplete(sample_ids):
            yield {
                "xA": torch.randn(1, 2, 16, 16),
                "xB": torch.randn(1, 2, 16, 16),
                "sar_valid_mask": torch.ones(1, 1, 16, 16),
            }
            # Only yields 1 sample, expected 2

        with self.assertRaises(ValueError):
            compute_empirical_fisher_for_checkpoint(
                self.model, ["s0", "s1"], mock_loader_incomplete, device="cpu"
            )

        # 3. Non-finite loss/grad raises ValueError
        def mock_loader_nan(sample_ids):
            for s in sample_ids:
                yield {
                    "xA": torch.full((1, 2, 16, 16), float("nan")),
                    "xB": torch.randn(1, 2, 16, 16),
                    "sar_valid_mask": torch.ones(1, 1, 16, 16),
                }

        with self.assertRaises(ValueError):
            compute_empirical_fisher_for_checkpoint(
                self.model, ["s0"], mock_loader_nan, device="cpu"
            )

    def test_17_runner_preflight_fisher_validation(self):
        """W9-R04: Verifies runner preflight rejects invalid keys, shape mismatch, NaN anchors, and missing/invalid sidecars."""
        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Restrict methods to eata only and target seed 0 for targeted preflight test
        cfg["matrix_specification"]["methods"] = ["eata"]
        cfg["matrix_specification"]["checkpoint_seeds"] = [0]

        dummy_model = TinyMockSNUNet()
        dummy_ckpt_path = self.tmp_path / "dummy_ckpt.pt"
        torch.save({"model_state_dict": dummy_model.state_dict()}, dummy_ckpt_path)
        dummy_ckpt_sha = compute_file_sha256(dummy_ckpt_path)
        cfg["source_checkpoints"]["seeds"]["0"]["checkpoint_path"] = str(dummy_ckpt_path.relative_to(REPO_ROOT) if dummy_ckpt_path.is_relative_to(REPO_ROOT) else dummy_ckpt_path)
        cfg["source_checkpoints"]["seeds"]["0"]["expected_sha256"] = dummy_ckpt_sha

        canonical_split_sha = cfg.get("source_data_configuration", {}).get("split_config_sha256", "test_split_sha")

        # Expected BN keys in TinyMockSNUNet
        valid_fishers = {
            "bn1.weight": (torch.ones(16), dummy_model.bn1.weight.data.clone()),
            "bn1.bias": (torch.ones(16), dummy_model.bn1.bias.data.clone()),
            "bn2.weight": (torch.ones(8), dummy_model.bn2.weight.data.clone()),
            "bn2.bias": (torch.ones(8), dummy_model.bn2.bias.data.clone()),
        }

        # Subtest helper creating valid/invalid artifacts and sidecars
        def make_runner_with_artifacts(test_fishers, sidecar_dict=None, write_sidecar=True, payload_metadata=None):
            fisher_pt = self.tmp_path / f"test_fisher_{len(test_fishers)}_{id(test_fishers)}.pt"
            sample_ids = ["s0"]
            sample_ids_sha = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
            if payload_metadata is None:
                payload_metadata = {
                    "checkpoint_seed": 0,
                    "checkpoint_sha256": dummy_ckpt_sha,
                    "selection_metadata": {
                        "selected_sample_ids": sample_ids,
                        "selected_sample_ids_sha256": sample_ids_sha,
                        "split_config_sha256": canonical_split_sha,
                        "estimator": "per_patch_squared_gradient",
                        "bn_context": "eval_mode_frozen_running_stats",
                        "loss_reduction": "mean_over_valid_pixels",
                    },
                }
            if payload_metadata is False:
                torch.save({"fisher": test_fishers}, fisher_pt)
            else:
                torch.save({"fisher": test_fishers, "metadata": payload_metadata}, fisher_pt)

            meta_pt = fisher_pt.with_suffix(".json")
            if write_sidecar:
                if sidecar_dict is None:
                    sidecar_dict = {
                        "seed": 0,
                        "artifact_sha256": compute_file_sha256(fisher_pt),
                        "checkpoint_sha256": dummy_ckpt_sha,
                        "split_config_sha256": canonical_split_sha,
                        "selected_sample_ids": sample_ids,
                        "selected_sample_ids_sha256": sample_ids_sha,
                        "selection_metadata": {
                            "estimator": "per_patch_squared_gradient",
                            "bn_context": "eval_mode_frozen_running_stats",
                            "loss_reduction": "mean_over_valid_pixels",
                        },
                    }
                else:
                    if sidecar_dict.get("artifact_sha256") in ("dummy", "AUTO", None):
                        sidecar_dict["artifact_sha256"] = compute_file_sha256(fisher_pt)
                meta_pt.write_text(json.dumps(sidecar_dict))
            elif meta_pt.exists():
                meta_pt.unlink()

            cfg["fisher_information_protocol"]["artifact_paths"]["0"] = str(fisher_pt.relative_to(REPO_ROOT) if fisher_pt.is_relative_to(REPO_ROOT) else fisher_pt)
            sub_cfg_path = self.tmp_path / f"sub_cfg_{len(test_fishers)}_{id(test_fishers)}.json"
            sub_cfg_path.write_text(json.dumps(cfg))
            runner = StrongerTTAMatrixRunner(
                config_path=sub_cfg_path,
                output_dir=self.tmp_path / f"preflight_subtest_{len(test_fishers)}_{id(test_fishers)}",
                device="cpu",
                synthetic_data=False,
            )
            return runner

        # Positive control: valid tensors + matching sidecar passes preflight
        valid_runner = make_runner_with_artifacts(valid_fishers)
        res = valid_runner.validate_preflight(is_smoke=True, fail_closed=True)
        self.assertTrue(res["fisher_status"]["0"]["valid"])

        # 1. Missing BN key
        missing_fishers = copy.deepcopy(valid_fishers)
        del missing_fishers["bn2.bias"]
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(missing_fishers).validate_preflight(is_smoke=True, fail_closed=True)

        # 2. Shape mismatch
        mismatched_fishers = copy.deepcopy(valid_fishers)
        mismatched_fishers["bn1.weight"] = (torch.ones(32), dummy_model.bn1.weight.data.clone())
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(mismatched_fishers).validate_preflight(is_smoke=True, fail_closed=True)

        # 3. NaN anchor parameter
        nan_fishers = copy.deepcopy(valid_fishers)
        nan_fishers["bn1.bias"] = (torch.ones(16), torch.full((16,), float("nan")))
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(nan_fishers).validate_preflight(is_smoke=True, fail_closed=True)

        # 4. Anchor diverged from checkpoint weight
        diverged_fishers = copy.deepcopy(valid_fishers)
        diverged_fishers["bn1.weight"] = (torch.ones(16), dummy_model.bn1.weight.data.clone() + 5.0)
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(diverged_fishers).validate_preflight(is_smoke=True, fail_closed=True)

        # 5. Missing sidecar -> ValueError / FileNotFoundError
        with self.assertRaises((ValueError, FileNotFoundError)):
            make_runner_with_artifacts(valid_fishers, write_sidecar=False).validate_preflight(is_smoke=True, fail_closed=True)

        # 6. Wrong split config hash in sidecar -> ValueError
        wrong_split_sidecar = {
            "seed": 0,
            "artifact_sha256": "AUTO",
            "checkpoint_sha256": dummy_ckpt_sha,
            "split_config_sha256": "WRONG_SPLIT_HASH",
            "selected_sample_ids": ["s0"],
            "selected_sample_ids_sha256": hashlib.sha256(json.dumps(["s0"]).encode()).hexdigest(),
            "selection_metadata": {"estimator": "per_patch_squared_gradient"},
        }
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, sidecar_dict=wrong_split_sidecar).validate_preflight(is_smoke=True, fail_closed=True)

        # 7. Wrong estimator in sidecar -> ValueError
        wrong_est_sidecar = {
            "seed": 0,
            "artifact_sha256": "AUTO",
            "checkpoint_sha256": dummy_ckpt_sha,
            "split_config_sha256": canonical_split_sha,
            "selected_sample_ids": ["s0"],
            "selected_sample_ids_sha256": hashlib.sha256(json.dumps(["s0"]).encode()).hexdigest(),
            "selection_metadata": {"estimator": "unapproved_random_estimator"},
        }
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, sidecar_dict=wrong_est_sidecar).validate_preflight(is_smoke=True, fail_closed=True)

        # 8. Wrong selected_sample_ids_sha256 in sidecar -> ValueError
        wrong_ids_sidecar = {
            "seed": 0,
            "artifact_sha256": "AUTO",
            "checkpoint_sha256": dummy_ckpt_sha,
            "split_config_sha256": canonical_split_sha,
            "selected_sample_ids": ["s0"],
            "selected_sample_ids_sha256": "WRONG_IDS_DIGEST",
            "selection_metadata": {"estimator": "per_patch_squared_gradient"},
        }
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, sidecar_dict=wrong_ids_sidecar).validate_preflight(is_smoke=True, fail_closed=True)

        # 9. W9-R04: Missing payload metadata dictionary -> ValueError
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, payload_metadata=False).validate_preflight(is_smoke=True, fail_closed=True)

        # 10. W9-R04: Payload/sidecar selection disagreement -> ValueError
        # Sidecar has ["s_alt"] and matching artifact SHA, but payload has ["s0"]
        alt_ids = ["s_alt"]
        alt_sha = hashlib.sha256(json.dumps(alt_ids).encode()).hexdigest()
        disagreed_sidecar = {
            "seed": 0,
            "artifact_sha256": "AUTO",
            "checkpoint_sha256": dummy_ckpt_sha,
            "split_config_sha256": canonical_split_sha,
            "selected_sample_ids": alt_ids,
            "selected_sample_ids_sha256": alt_sha,
            "selection_metadata": {
                "estimator": "per_patch_squared_gradient",
                "bn_context": "eval_mode_frozen_running_stats",
                "loss_reduction": "mean_over_valid_pixels",
            },
        }
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, sidecar_dict=disagreed_sidecar).validate_preflight(is_smoke=True, fail_closed=True)

        # 11. W9-R04: Payload/sidecar seed and checkpoint disagreement -> ValueError
        # Payload has seed 1 and wrong checkpoint SHA, sidecar has seed 0 and matching artifact SHA
        mismatched_payload = {
            "checkpoint_seed": 1,
            "checkpoint_sha256": "WRONG_PAYLOAD_CKPT",
            "selection_metadata": {
                "selected_sample_ids": ["s0"],
                "selected_sample_ids_sha256": hashlib.sha256(json.dumps(["s0"]).encode()).hexdigest(),
                "split_config_sha256": canonical_split_sha,
                "estimator": "per_patch_squared_gradient",
                "bn_context": "eval_mode_frozen_running_stats",
                "loss_reduction": "mean_over_valid_pixels",
            },
        }
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, payload_metadata=mismatched_payload).validate_preflight(is_smoke=True, fail_closed=True)

        # 12. W9-R04: Payload unapproved estimator -> ValueError
        bad_est_payload = copy.deepcopy(mismatched_payload)
        bad_est_payload["checkpoint_seed"] = 0
        bad_est_payload["checkpoint_sha256"] = dummy_ckpt_sha
        bad_est_payload["selection_metadata"]["estimator"] = "unapproved_estimator"
        with self.assertRaises(ValueError):
            make_runner_with_artifacts(valid_fishers, payload_metadata=bad_est_payload).validate_preflight(is_smoke=True, fail_closed=True)

        # 13. W9-R04: Full producer-shaped payload passes preflight
        producer_shaped_payload = {
            "checkpoint_seed": 0,
            "checkpoint_path": str(dummy_ckpt_path),
            "checkpoint_sha256": dummy_ckpt_sha,
            "stats": {"num_evaluated_patches": 1, "accumulation_formula": "exact_per_sample"},
            "accumulation_formula": "exact_per_sample",
            "created_at_utc": "2026-10-18T00:00:00Z",
            "selection_metadata": {
                "selection_seed": 2026,
                "sample_size": 1,
                "total_source_candidates": 10,
                "approved_source_events": [118],
                "split_config_path": "configs/splits/kuro_source_all27_split_v1.json",
                "split_config_sha256": canonical_split_sha,
                "manifest_path": "manifest.gz",
                "manifest_sha256": "dummy_manifest_sha",
                "selected_sample_ids": ["s0"],
                "selected_sample_ids_sha256": hashlib.sha256(json.dumps(["s0"]).encode()).hexdigest(),
                "estimator": "per_patch_squared_gradient",
                "loss_reduction": "mean_over_valid_pixels",
                "bn_context": "eval_mode_frozen_running_stats",
                "preprocessing": {"sar_mean": [0.0953, 0.0264], "sar_std": [0.0427, 0.0215], "sar_clamp_max": 0.15},
            },
        }
        producer_runner = make_runner_with_artifacts(valid_fishers, payload_metadata=producer_shaped_payload)
        res_prod = producer_runner.validate_preflight(is_smoke=True, fail_closed=True)
        self.assertTrue(res_prod["fisher_status"]["0"]["valid"])

    def test_18_runner_unique_sample_ids_and_duplicate_rejection(self):
        """W9-R05: Verifies runner raises ValueError when duplicate sample IDs are encountered in an event."""
        # Create dataset with duplicate sample IDs in the same event
        duplicate_records = [
            {
                "sample_id": "dup_001",
                "event_id": 437,
                "xA": np.zeros((2, 16, 16), dtype=np.float32),
                "xB": np.zeros((2, 16, 16), dtype=np.float32),
                "valid_mask": np.ones((16, 16), dtype=np.uint8),
                "label": np.zeros((16, 16), dtype=np.int64),
            },
            {
                "sample_id": "dup_001",  # duplicate ID!
                "event_id": 437,
                "xA": np.zeros((2, 16, 16), dtype=np.float32),
                "xB": np.zeros((2, 16, 16), dtype=np.float32),
                "valid_mask": np.ones((16, 16), dtype=np.uint8),
                "label": np.zeros((16, 16), dtype=np.int64),
            },
        ]
        ds = KuroEvaluationDataset.from_mock_records(duplicate_records, allowed_events=[437])

        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        runner = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=self.tmp_path / "dup_test",
            device="cpu",
            synthetic_data=True,
        )
        spec = runner.build_matrix_streams(is_smoke=True)[0]
        with self.assertRaises(ValueError):
            runner.run_physical_stream(spec, ds, ds, is_smoke=True)

    def test_19_sar_nonfinite_gradient_step_refusal(self):
        """W9-R07: Verifies SAR records updated_this_step=False when SAM second step refuses non-finite gradients."""
        sar = SARAdapter(self.wrapper, str(self.ckpt_path), margin_e0=2.0, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        mask = torch.ones(2, 1, 16, 16)

        # Force SAM.second_step to receive non-finite gradient
        orig_second_step = sar.optimizer.second_step
        def mock_second_step(*args, **kwargs):
            # Inject NaN into gradient before checking
            for group in sar.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.fill_(float("nan"))
            return orig_second_step(*args, **kwargs)

        sar.optimizer.second_step = mock_second_step
        out = sar.adapt_and_predict(xA, xB, sar_valid_mask=mask)

        # Step should be skipped: num_adaptation_steps remains 0, updated_this_step is False
        self.assertFalse(out["step_diagnostics"]["updated_this_step"])
        self.assertEqual(sar.num_adaptation_steps, 0)
        self.assertEqual(sar.num_skipped_updates, 1)

    def test_20_runner_incremental_persistence_and_failure_manifest(self):
        """W9-R09: Verifies runner flushes completed event rows immediately and writes failure manifest on exceptions and interrupts."""
        config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        out_dir = self.tmp_path / "persist_test"
        runner = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=out_dir,
            device="cpu",
            synthetic_data=True,
        )

        # Inject an exception during stream 2 to simulate mid-run failure
        orig_run_stream = runner.run_physical_stream
        call_count = [0]
        def failing_run_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated mid-run GPU/process crash")
            return orig_run_stream(*args, **kwargs)

        runner.run_physical_stream = failing_run_stream

        with self.assertRaises(RuntimeError):
            runner.run_matrix(is_smoke=True)

        # Check that stream 1 event rows are preserved on disk in CSV
        csv_path = out_dir / "stream_event_metrics.csv"
        self.assertTrue(csv_path.is_file(), "CSV was not flushed before failure!")
        with open(csv_path, "r") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1, "CSV has no event rows!")

        # Check failure manifest for RuntimeError
        manifest_path = out_dir / "week9_tta_manifest.json"
        self.assertTrue(manifest_path.is_file(), "Manifest was not written on failure!")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("Simulated mid-run", manifest["error"])
        self.assertIn("config_sha256", manifest)
        self.assertIn("runner_settings", manifest)
        self.assertIn("command_args", manifest)
        self.assertGreater(manifest["completed_event_rows"], 0)

        # Test KeyboardInterrupt mid-stream
        out_dir_interrupt = self.tmp_path / "persist_interrupt_test"
        runner_int = StrongerTTAMatrixRunner(
            config_path=config_path,
            output_dir=out_dir_interrupt,
            device="cpu",
            synthetic_data=True,
        )
        call_count_int = [0]
        def interrupt_run_stream(*args, **kwargs):
            call_count_int[0] += 1
            if call_count_int[0] == 2:
                raise KeyboardInterrupt("Simulated user interrupt")
            return orig_run_stream(*args, **kwargs)

        runner_int.run_physical_stream = interrupt_run_stream
        with self.assertRaises(KeyboardInterrupt):
            runner_int.run_matrix(is_smoke=True)

        manifest_int_path = out_dir_interrupt / "week9_tta_manifest.json"
        self.assertTrue(manifest_int_path.is_file(), "Manifest not written on interrupt!")
        with open(manifest_int_path, "r") as f:
            manifest_int = json.load(f)
        self.assertEqual(manifest_int["status"], "failed")
        self.assertEqual(manifest_int["failure_type"], "KeyboardInterrupt")
        self.assertGreater(manifest_int["completed_event_rows"], 0)
        self.assertIn("config_sha256", manifest_int)

    def test_21_fisher_candidate_population_identity_and_split_isolation(self):
        """W9-N03: Verifies 21,462-sample candidate population identity and zero dev/test leakage."""
        m_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz"
        split_path = REPO_ROOT / "configs/splits/kuro_source_all27_split_v1.json"
        self.assertTrue(m_path.is_file(), f"Manifest missing at {m_path}")
        self.assertTrue(split_path.is_file(), f"Split missing at {split_path}")

        with open(split_path, "r") as f:
            split_data = json.load(f)
        source_events = set(split_data["source_all_events"])
        dev_events = set(split_data["tta_validation_events"])
        test_events = set(split_data["sealed_test_events"])

        with gzip.open(m_path, "rb") as f:
            manifest = pickle.load(f)

        candidate_ids = []
        candidate_event_map = {}
        for k, v in manifest.items():
            actid = int(v["info"]["actid"])
            if actid in source_events:
                sid = str(k)
                candidate_ids.append(sid)
                candidate_event_map[sid] = actid

        candidate_ids = sorted(candidate_ids)
        self.assertEqual(len(candidate_ids), 21462)
        cand_sha = hashlib.sha256(json.dumps(candidate_ids).encode()).hexdigest()
        self.assertEqual(cand_sha, "145aa6bb0ceb18f0ab4f5d4d1121d7d12ccd7ce3da600519e035904e1b9d6c7e")

        # Isolation verification
        cand_events = set(candidate_event_map.values())
        self.assertEqual(len(cand_events & dev_events), 0, "Development events leaked into candidates!")
        self.assertEqual(len(cand_events & test_events), 0, "Sealed test events leaked into candidates!")
        self.assertEqual(cand_events, source_events)

        # Deterministic sample 2000 seed 2026
        import random
        rng = random.Random(2026)
        selected_ids = sorted(rng.sample(candidate_ids, 2000))
        self.assertEqual(len(selected_ids), 2000)
        self.assertEqual(len(set(selected_ids)), 2000)
        self.assertTrue(all(sid in set(candidate_ids) for sid in selected_ids))

        sel_sha = hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest()
        self.assertEqual(sel_sha, "b21e6c9ea244262c85726a779e3c7b322774ac4f75f40ac36d66c92aa4a31a5c")

        sel_events = set(candidate_event_map[sid] for sid in selected_ids)
        self.assertEqual(len(sel_events), 27, "Not all 27 source events represented in sample")

    def test_22_fisher_candidate_population_rejection_of_out_of_corpus_ids(self):
        """W9-N03: Verifies runner preflight rejects artifacts with candidate population or manifest mismatch."""
        v2_config_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol_v2.json"
        with open(v2_config_path, "r") as f:
            cfg = json.load(f)

        cfg["matrix_specification"]["methods"] = ["eata"]
        cfg["matrix_specification"]["checkpoint_seeds"] = [0]

        dummy_model = TinyMockSNUNet()
        dummy_ckpt_path = self.tmp_path / "dummy_v2_ckpt.pt"
        torch.save({"model_state_dict": dummy_model.state_dict()}, dummy_ckpt_path)
        dummy_ckpt_sha = compute_file_sha256(dummy_ckpt_path)
        cfg["source_checkpoints"]["seeds"]["0"]["checkpoint_path"] = str(dummy_ckpt_path.relative_to(REPO_ROOT) if dummy_ckpt_path.is_relative_to(REPO_ROOT) else dummy_ckpt_path)
        cfg["source_checkpoints"]["seeds"]["0"]["expected_sha256"] = dummy_ckpt_sha

        canonical_split_sha = cfg["source_data_configuration"]["split_config_sha256"]
        canonical_manifest_sha = cfg["source_data_configuration"]["manifest_sha256"]
        canonical_sel_sha = cfg["fisher_information_protocol"]["selected_sample_ids_sha256"]

        # Valid mock Fisher tensors
        valid_fishers = {
            "bn1.weight": (torch.ones(16), dummy_model.bn1.weight.data.clone()),
            "bn1.bias": (torch.ones(16), dummy_model.bn1.bias.data.clone()),
            "bn2.weight": (torch.ones(8), dummy_model.bn2.weight.data.clone()),
            "bn2.bias": (torch.ones(8), dummy_model.bn2.bias.data.clone()),
        }

        def make_runner_with_v2_artifact(total_candidates=21462, manifest_sha=canonical_manifest_sha, sel_sha=canonical_sel_sha):
            f_pt = self.tmp_path / f"v2_fisher_{total_candidates}_{manifest_sha[:6]}.pt"
            sample_ids = ["s0"]
            meta_payload = {
                "checkpoint_seed": 0,
                "checkpoint_sha256": dummy_ckpt_sha,
                "selection_metadata": {
                    "total_source_candidates": total_candidates,
                    "manifest_sha256": manifest_sha,
                    "selected_sample_ids": sample_ids,
                    "selected_sample_ids_sha256": sel_sha,
                    "split_config_sha256": canonical_split_sha,
                    "estimator": "per_patch_squared_gradient",
                    "bn_context": "eval_mode_frozen_running_stats",
                    "loss_reduction": "mean_over_valid_pixels",
                },
            }
            torch.save({"fisher": valid_fishers, "metadata": meta_payload}, f_pt)
            f_json = f_pt.with_suffix(".json")
            sidecar_dict = {
                "seed": 0,
                "artifact_sha256": compute_file_sha256(f_pt),
                "checkpoint_sha256": dummy_ckpt_sha,
                "split_config_sha256": canonical_split_sha,
                "manifest_sha256": manifest_sha,
                "selected_sample_ids": sample_ids,
                "selected_sample_ids_sha256": sel_sha,
                "selection_metadata": meta_payload["selection_metadata"],
            }
            with open(f_json, "w") as mf:
                json.dump(sidecar_dict, mf)

            cfg_copy = copy.deepcopy(cfg)
            cfg_copy["fisher_information_protocol"]["artifact_paths"]["0"] = str(f_pt.relative_to(REPO_ROOT) if f_pt.is_relative_to(REPO_ROOT) else f_pt)
            runner = StrongerTTAMatrixRunner(config_path=v2_config_path, output_dir=self.tmp_path / "v2_val_test", device="cpu", synthetic_data=True)
            runner.cfg = cfg_copy
            return runner

        # Candidate population mismatch: 48,768 (v1) against v2 protocol expecting 21,462
        runner_mismatch = make_runner_with_v2_artifact(total_candidates=48768)
        with self.assertRaises(ValueError) as cm:
            runner_mismatch.validate_preflight()
        self.assertIn("candidate population mismatch", str(cm.exception).lower())

        # Manifest mismatch
        runner_bad_manifest = make_runner_with_v2_artifact(manifest_sha="0000000000000000000000000000000000000000000000000000000000000000")
        with self.assertRaises(ValueError) as cm:
            runner_bad_manifest.validate_preflight()
        self.assertIn("manifest sha-256 mismatch", str(cm.exception).lower())

        # Selection digest mismatch
        runner_bad_sel = make_runner_with_v2_artifact(sel_sha="1111111111111111111111111111111111111111111111111111111111111111")
        with self.assertRaises(ValueError) as cm:
            runner_bad_sel.validate_preflight()
        self.assertIn("selection digest mismatch", str(cm.exception).lower())

    def test_23_fisher_v1_v2_overlap_split_reproduction(self):
        """W9-N03: Reproduces exact 902/1098 overlap split between v1 attempt and true 21,462 corpus."""
        m1_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz"
        m2_path = REPO_ROOT.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"
        split_path = REPO_ROOT / "configs/splits/kuro_source_all27_split_v1.json"

        with open(split_path, "r") as f:
            split_data = json.load(f)
        source_events = set(split_data["source_all_events"])
        dev_events = set(split_data["tta_validation_events"])
        test_events = set(split_data["sealed_test_events"])

        with gzip.open(m1_path, "rb") as f:
            d1 = pickle.load(f)
        with gzip.open(m2_path, "rb") as f:
            d2 = pickle.load(f)

        cands_v1 = sorted([str(k) for k, v in d2.items() if int(v["info"]["actid"]) in source_events])
        cands_v2 = sorted([str(k) for k, v in d1.items() if int(v["info"]["actid"]) in source_events])
        self.assertEqual(len(cands_v1), 48768)
        self.assertEqual(len(cands_v2), 21462)

        import random
        rng_v1 = random.Random(2026)
        sel_v1 = sorted(rng_v1.sample(cands_v1, 2000))
        sel_v1_sha = hashlib.sha256(json.dumps(sel_v1).encode()).hexdigest()
        self.assertEqual(sel_v1_sha, "862c6a39e0bcdc2bcff259bcc4cf340d724c0312a5b98e774ed9967c3aec9600")

        cands_v2_set = set(cands_v2)
        in_corpus = [x for x in sel_v1 if x in cands_v2_set]
        out_corpus = [x for x in sel_v1 if x not in cands_v2_set]
        self.assertEqual(len(in_corpus), 902, "In-corpus overlap must be exactly 902")
        self.assertEqual(len(out_corpus), 1098, "Out-of-corpus count must be exactly 1098")

        # Verify all 1,098 out of corpus belong to source events and never to dev/test
        for sid in out_corpus:
            actid = int(d2[sid]["info"]["actid"])
            self.assertIn(actid, source_events)
            self.assertNotIn(actid, dev_events)
            self.assertNotIn(actid, test_events)

    def test_24_versioned_protocol_v2_validation_and_destination_isolation(self):
        """W9-N03: Verifies protocol v2 configuration, dry-run CLI preflight, and artifact isolation."""
        v2_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol_v2.json"
        self.assertTrue(v2_path.is_file())
        with open(v2_path, "r") as f:
            v2_cfg = json.load(f)

        # Confirm non-overwriting destination paths
        paths = v2_cfg["fisher_information_protocol"]["artifact_paths"]
        for seed_str, p_rel in paths.items():
            self.assertTrue(p_rel.startswith("artifacts/eata_fishers/v2/"), f"Path {p_rel} does not use v2 directory")
            target_p = REPO_ROOT / p_rel
            self.assertFalse(target_p.exists(), f"Target v2 artifact {target_p} already exists!")

        # Confirm v1 artifacts exist and are preserved
        v1_path = REPO_ROOT / "configs/evaluation/week9_stronger_tta_protocol.json"
        with open(v1_path, "r") as f:
            v1_cfg = json.load(f)
        v1_paths = v1_cfg["fisher_information_protocol"]["artifact_paths"]
        for seed_str, p_rel in v1_paths.items():
            orig_p = REPO_ROOT / p_rel
            self.assertTrue(orig_p.is_file(), f"Original v1 artifact {orig_p} missing!")

        # Test CLI preflight dry-run using real tools/prepare_eata_fishers.py
        import subprocess
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools/prepare_eata_fishers.py"),
            "--config", str(v2_path),
            "--dry_run"
        ]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0, f"Dry-run CLI failed: {res.stderr}")
        self.assertIn("preflight validations passed", res.stdout)
        self.assertIn("Selected 2000 source samples deterministically", res.stdout)


if __name__ == "__main__":
    unittest.main()



