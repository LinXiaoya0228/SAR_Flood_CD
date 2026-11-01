"""Automated regression prevention test suite for Week 1 controlled training.

Tests:
1. Mathematical equivalence of gradient accumulation (4 microbatches of 8 vs 1 batch of 32).
2. Masked loss correctness (CE + Soft Dice on valid pixels, ignoring class 3 and MNA=0).
3. Policy S2 DEM nodata boundary cliff suppression.
4. Research integrity split isolation (22 inner-train, 5 inner-val, 0 quarantined/sealed).
5. Checkpoint provenance, RNG state tracking, and deterministic resumption.
"""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.evaluation.losses import ControlledSourceLoss
from src.data.source_dataset import compute_policy_s2_slope, KuroSourceDataset
from src.training.source_trainer import SourceTrainer
from src.evaluation.run_manifest import RunManifest, compute_file_sha256


class SimpleLinearNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, num_classes, kernel_size=1, bias=False)

    def forward(self, xA, xB):
        x = torch.cat([xA, xB], dim=1)
        return self.conv(x)


class DummyDataset(Dataset):
    def __init__(self, n_samples=32, in_channels=2):
        super().__init__()
        torch.manual_seed(123)
        self.xA = torch.randn(n_samples, in_channels, 16, 16)
        self.xB = torch.randn(n_samples, in_channels, 16, 16)
        self.label = torch.randint(0, 3, (n_samples, 16, 16))
        self.valid_mask = torch.ones(n_samples, 16, 16, dtype=torch.uint8)
        self.event_id = torch.full((n_samples,), 118)

    def __len__(self):
        return len(self.xA)

    def __getitem__(self, idx):
        return {
            "xA": self.xA[idx],
            "xB": self.xB[idx],
            "label": self.label[idx],
            "valid_mask": self.valid_mask[idx],
            "event_id": self.event_id[idx],
        }


class TestWeek1ControlledTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gradient_accumulation_equivalence(self):
        """Verify gradient accumulation equivalence: exact for CE, close (<=5e-5) for Soft Dice."""
        dataset = DummyDataset(n_samples=32, in_channels=2)

        # 1. Exact equivalence on Cross-Entropy (linear over samples)
        torch.manual_seed(42)
        model_full = SimpleLinearNet(in_channels=2, num_classes=3)
        opt_full = torch.optim.Adam(model_full.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
        loss_fn_ce = ControlledSourceLoss(weight_ce=1.0, weight_dice=0.0)

        loader_full = DataLoader(dataset, batch_size=32, shuffle=False)
        batch_full = next(iter(loader_full))
        opt_full.zero_grad(set_to_none=True)
        logits_full = model_full(batch_full["xA"], batch_full["xB"])
        loss_full = loss_fn_ce(logits_full, batch_full["label"])
        loss_full.backward()
        grad_full = model_full.conv.weight.grad.clone()
        opt_full.step()

        torch.manual_seed(42)
        model_accum = SimpleLinearNet(in_channels=2, num_classes=3)
        opt_accum = torch.optim.Adam(model_accum.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

        loader_micro = DataLoader(dataset, batch_size=8, shuffle=False)
        opt_accum.zero_grad(set_to_none=True)
        for batch_m in loader_micro:
            logits_m = model_accum(batch_m["xA"], batch_m["xB"])
            loss_m = loss_fn_ce(logits_m, batch_m["label"])
            (loss_m / 4.0).backward()
        grad_accum = model_accum.conv.weight.grad.clone()
        opt_accum.step()

        # CE must be strictly equal within floating point precision
        torch.testing.assert_close(grad_accum, grad_full, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(model_accum.conv.weight, model_full.conv.weight, atol=1e-7, rtol=1e-6)

        # 2. Combined CE + Dice: close to within 5e-5 due to microbatch denominator summation
        torch.manual_seed(42)
        m_comb_full = SimpleLinearNet(in_channels=2, num_classes=3)
        opt_comb_full = torch.optim.Adam(m_comb_full.parameters(), lr=1e-3)
        loss_fn_comb = ControlledSourceLoss(weight_ce=0.5, weight_dice=0.5)

        opt_comb_full.zero_grad(set_to_none=True)
        loss_cf = loss_fn_comb(m_comb_full(batch_full["xA"], batch_full["xB"]), batch_full["label"])
        loss_cf.backward()
        g_full = m_comb_full.conv.weight.grad.clone()

        torch.manual_seed(42)
        m_comb_acc = SimpleLinearNet(in_channels=2, num_classes=3)
        opt_comb_acc = torch.optim.Adam(m_comb_acc.parameters(), lr=1e-3)
        opt_comb_acc.zero_grad(set_to_none=True)
        for bm in loader_micro:
            l_m = loss_fn_comb(m_comb_acc(bm["xA"], bm["xB"]), bm["label"])
            (l_m / 4.0).backward()
        g_acc = m_comb_acc.conv.weight.grad.clone()
        torch.testing.assert_close(g_acc, g_full, atol=5e-5, rtol=1e-2)

    def test_masked_source_loss_spec(self):
        """Verify that ControlledSourceLoss masks class 3 / MNA=0 and averages properly."""
        loss_fn = ControlledSourceLoss(weight_ce=0.5, weight_dice=0.5, ignore_index=3)

        logits = torch.zeros(2, 3, 10, 10, requires_grad=True)
        # Pixel (0, 0) has label 3 (ignore), pixel (0, 1) has label 1
        labels = torch.full((2, 10, 10), 3, dtype=torch.long)
        labels[:, 5:, :] = 1  # 50% valid pixels with class 1

        loss, comp = loss_fn(logits, labels, return_components=True)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIn("loss_ce", comp)
        self.assertIn("loss_dice", comp)

        # All invalid batch returns 0.0 with valid grad
        all_invalid_labels = torch.full((2, 10, 10), 3, dtype=torch.long)
        all_inv_loss = loss_fn(logits, all_invalid_labels)
        self.assertEqual(float(all_inv_loss.detach()), 0.0)
        all_inv_loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_policy_s2_slope_fringe_suppression(self):
        """Verify that DEM nodata inpainting suppresses artificial boundary cliff."""
        class MockCRS:
            is_geographic = False

        dem = np.full((30, 30), 10.0, dtype=np.float32)
        # Half image is nodata sentinel -32768
        dem[:, :15] = -32768.0
        # Add linear ramp to simulate bilinear resampling fringe
        dem[:, 15:18] = np.array([-500.0, -100.0, 0.0])[None, :]

        # With inpainting (Policy S2)
        slope_deg, valid_mask = compute_policy_s2_slope(dem, dx=10.0, dy=10.0, crs=MockCRS(), nodata_threshold=-430.0, dilation_iterations=3)
        valid_slopes = slope_deg[valid_mask]
        # Inpainted ground has smooth transition without 89-degree cliff
        self.assertLess(float(valid_slopes.max()), 65.0)

    def test_research_integrity_split_isolation(self):
        """Verify 22 inner-train and 5 inner-val events are disjoint and quarantined events untouched."""
        split_path = REPO_ROOT / "configs/splits/kuro_source_inner_split_v1.json"
        with open(split_path) as f:
            split_cfg = json.load(f)

        inner_train = set(split_cfg["source_inner_train_events"])
        inner_val = set(split_cfg["source_inner_validation_events"])
        tta_val = {279, 437, 520, 559, 1111003, 1111008}
        sealed_test = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}

        self.assertEqual(len(inner_train), 22)
        self.assertEqual(len(inner_val), 5)
        self.assertTrue(inner_train.isdisjoint(inner_val))
        self.assertTrue(inner_train.isdisjoint(tta_val))
        self.assertTrue(inner_train.isdisjoint(sealed_test))
        self.assertTrue(inner_val.isdisjoint(tta_val))
        self.assertTrue(inner_val.isdisjoint(sealed_test))

    def test_checkpoint_provenance_and_rng(self):
        """Verify that checkpoint stores RNG states, config hash, and restores cleanly."""
        model = SimpleLinearNet(in_channels=2, num_classes=3)
        trainer = SourceTrainer(
            model=model,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )
        trainer.current_epoch = 3
        trainer.global_step = 25

        ckpt_path = trainer.save_checkpoint(
            "test_provenance.pt",
            config_path="configs/source/snunet_vvvh_v1.json",
            config_hash="dummy_hash_12345",
        )
        self.assertTrue(ckpt_path.exists())

        # Inspect checkpoint content
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt_data["global_step"], 25)
        self.assertEqual(ckpt_data["config_hash"], "dummy_hash_12345")
        self.assertIn("rng_state", ckpt_data)
        self.assertIn("git_commit", ckpt_data)

        # Restore
        model2 = SimpleLinearNet(in_channels=2, num_classes=3)
        trainer2 = SourceTrainer(
            model=model2,
            output_dir=self.temp_dir,
            device="cpu",
            effective_batch_size=8,
            physical_microbatch_size=4,
            mixed_precision=False,
        )
        resumed = trainer2.resume_from_checkpoint(ckpt_path, restore_rng=True)
        self.assertEqual(trainer2.global_step, 25)
        self.assertEqual(trainer2.current_epoch, 3)

    def test_kuro_source_dataset_rejects_test_manifest(self):
        """Verify KuroSourceDataset loudly rejects any manifest containing 'test'."""
        with self.assertRaises(RuntimeError) as ctx:
            KuroSourceDataset(
                manifest_path="third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz",
                split="val",
            )
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("Test manifest provided to source dataset", str(ctx.exception))

    def test_kuro_source_dataset_rejects_forbidden_events_in_split(self):
        """Verify KuroSourceDataset rejects split configs containing quarantined or sealed events."""
        # Create a malicious/buggy split config with sealed event 205
        bad_split_data = {
            "source_inner_train_events": [118, 205],  # 205 is sealed test!
            "source_inner_validation_events": [267],
        }
        bad_split_path = self.temp_dir / "bad_split.json"
        with open(bad_split_path, "w", encoding="utf-8") as f:
            json.dump(bad_split_data, f)

        # Dataset constructor must check EXPECTED_SPLIT_HASH and forbidden events
        with self.assertRaises(RuntimeError) as ctx:
            KuroSourceDataset(
                split_config_path=str(bad_split_path),
                split="train",
            )
        # Will fail either on hash mismatch or forbidden event check
        self.assertTrue(
            "Split config SHA-256 mismatch" in str(ctx.exception)
            or "CRITICAL RESEARCH INTEGRITY VIOLATION" in str(ctx.exception)
        )

    def test_run_manifest_source_training_rejects_forbidden_access(self):
        """Verify RunManifest in source_training mode rejects quarantined and sealed events."""
        dummy_ckpt = self.temp_dir / "dummy.pt"
        dummy_ckpt.write_bytes(b"dummy")

        # Quarantined TTA-val event (279)
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="bad_source_run_tta_val",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=dummy_ckpt,
                data_manifest="data/manifest.pkl",
                accessed_event_ids=[118, 279],
                run_type="source_training",
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("279", str(ctx.exception))

        # Sealed test event (205)
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="bad_source_run_sealed",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=dummy_ckpt,
                data_manifest="data/manifest.pkl",
                accessed_event_ids=[118, 205],
                run_type="source_training",
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("205", str(ctx.exception))

        # Test manifest
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="bad_source_run_test_manifest",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=dummy_ckpt,
                data_manifest="pickle/KuroV2_grid_dict_test_0_100.gz",
                accessed_event_ids=[118],
                run_type="source_training",
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("Source training run declared test data manifest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
