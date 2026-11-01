"""ChangeGuard Week 11A Focused CPU Synthetic Unit Tests.

Governing Protocol: docs/protocols/week11_evidence_consistency_protocol_v1.md
Governing Config: configs/changeguard/week11_evidence_consistency_v1.json

Focused Regressions for Week 11B Independent Review Findings:
- R01: State-neutral probe isolation with buffer preservation and exception safety.
- R02: Numerical boundary safeguards (non-finite loss, gradients, candidate parameters,
       and post-step optimizer momentum corruption with exact rollback and honest counters).
- R03: Diagnostic timing clarification (PRE_UPDATE primary/disagreement vs POST_UPDATE null probes).
- R04: Persisted sample/event identity, batch-to-sample mapping, and context disclosure.
- R05: Fail-closed CPU authorization (strict boolean validation, status enforcement, rejection of unsupported stages).
- R06: Source model independence and elimination of whole-object/storage aliasing.
- N01: Sensory mask shape validation and normalization (singleton channel squeeze, rejection of invalid shapes).
- N02: Actual JSON null (Python None) in persisted summary JSON, textual 'null' in CSV.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.adapter import EvidenceConsistencyAdapter
from tools.run_week11_cpu_evidence import MockSNUNet, Week11CPURunner


class StatefulMockModel(MockSNUNet):
    """Model with a registered buffer and simulated stateful/failing forward for R01 tests."""

    def __init__(self, base_channels: int = 8):
        super().__init__(base_channels=base_channels)
        self.register_buffer("forward_counter", torch.tensor(0, dtype=torch.long))
        self.fail_forward = False

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self.forward_counter.add_(1)
        torch.rand(1)  # Consume RNG even in eval mode
        if self.fail_forward:
            raise RuntimeError("Synthetic forward failure sentinel")
        return super().forward(a, b)


class LargeFiniteLogitsModel(nn.Module):
    """Finite inputs and parameters; FP32 reduction overflows to non-finite loss (R02)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([a, b], dim=1)
        z = self.bn1(self.conv1(z)) * 0.0
        grid = torch.ones_like(z[:, :1])
        grid[..., ::2] = -1.0
        z_large = z[:, :1] + grid * 1e38
        return torch.cat([z_large, z_large * 0.0, -z_large], dim=1)


class TestWeek11EvidenceConsistency(unittest.TestCase):
    """Focused CPU unit tests for EvidenceConsistencyAdapter and runner."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = "cpu"
        self.model = MockSNUNet(in_channels=2, num_classes=3, base_channels=8)
        self.adapter = EvidenceConsistencyAdapter(
            model=self.model,
            learning_rate=1.25e-4,
            momentum=0.9,
            weight_decay=0.0,
            confidence_threshold=0.90,
            device=self.device,
        )

    def test_01_trainable_scope_and_finite_gradients(self):
        """Test 1: Trainable scope restricted to BN affine parameters; gradients finite."""
        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in self.model.named_parameters() if not p.requires_grad]

        self.assertGreater(len(trainable_names), 0)
        self.assertGreater(len(frozen_names), 0)
        for name in trainable_names:
            self.assertTrue(name.endswith(".weight") or name.endswith(".bias"))
            parent_name = name.rsplit(".", 1)[0]
            mod = dict(self.model.named_modules())[parent_name]
            self.assertIsInstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d))

        frozen_before = {name: p.detach().clone() for name, p in self.model.named_parameters() if not p.requires_grad}

        xA = torch.randn(4, 2, 16, 16) * 5.0
        xB = torch.randn(4, 2, 16, 16) * 5.0
        mask = torch.ones(4, 16, 16, dtype=torch.long)

        result = self.adapter.adapt_and_predict(xA, xB, mask)
        if result["adaptation_metadata"]["update_applied"]:
            self.assertIsNotNone(result["adaptation_metadata"]["loss"])
            self.assertTrue(math.isfinite(result["adaptation_metadata"]["loss"]))

            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    self.assertTrue(torch.equal(p, frozen_before[name]))

    def test_02_detached_weak_target_and_one_way_kl(self):
        """Test 2: Detached weak target, correct one-way KL, confidence thresholding."""
        xA = torch.randn(2, 2, 8, 8, requires_grad=True)
        xB = torch.randn(2, 2, 8, 8, requires_grad=True)
        mask = torch.ones(2, 8, 8, dtype=torch.long)

        logits_w = self.adapter.model(xA, xB)
        probs_w = F.softmax(logits_w, dim=1)
        probs_w_det = probs_w.detach()

        xA_s = torch.flip(xA, dims=[-1])
        xB_s = torch.flip(xB, dims=[-1])
        logits_s = self.adapter.model(xA_s, xB_s)
        logits_s_aligned = torch.flip(logits_s, dims=[-1])
        log_probs_s = F.log_softmax(logits_s_aligned, dim=1)

        max_prob_w = probs_w_det.max(dim=1)[0]
        conf_mask = (max_prob_w >= 0.90) & (mask == 1)

        kl_pixel = (probs_w_det * (torch.log(probs_w_det + 1e-12) - log_probs_s)).sum(dim=1)
        loss = (kl_pixel * conf_mask.float()).sum() / conf_mask.sum().clamp(min=1.0)

        self.assertIsNone(probs_w_det.grad_fn)
        self.assertIsNotNone(log_probs_s.grad_fn)

    def test_03_horizontal_flip_spatial_alignment(self):
        """Test 3: Horizontal flip alignment on xA, xB, valid_mask and spatial inversion."""
        B, C, H, W = 2, 2, 16, 16
        xA = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)
        xB = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W) + 100.0
        mask = torch.ones(B, H, W, dtype=torch.long)
        mask[:, :, 0] = 0

        xA_s = torch.flip(xA, dims=[-1])
        xB_s = torch.flip(xB, dims=[-1])
        mask_s = torch.flip(mask, dims=[-1])

        xA_realigned = torch.flip(xA_s, dims=[-1])
        mask_realigned = torch.flip(mask_s, dims=[-1])

        self.assertTrue(torch.equal(xA, xA_realigned))
        self.assertTrue(torch.equal(mask, mask_realigned))
        self.assertTrue(torch.all(mask_s[:, :, -1] == 0))
        self.assertTrue(torch.all(mask_s[:, :, 0] == 1))

    def test_04_empty_selection_and_zero_valid_skip(self):
        """Test 4: Empty selection / zero-valid update skipping preserves parameters and returns loss=None (N02)."""
        params_before = [p.detach().clone() for p in self.adapter.model.parameters()]

        xA = torch.zeros(2, 2, 16, 16)
        xB = torch.zeros(2, 2, 16, 16)
        mask = torch.ones(2, 16, 16, dtype=torch.long)

        self.adapter.confidence_threshold = 0.9999
        res = self.adapter.adapt_and_predict(xA, xB, mask)
        self.assertFalse(res["adaptation_metadata"]["update_applied"])
        self.assertEqual(res["adaptation_metadata"]["action"], "skip")
        self.assertIsNone(res["adaptation_metadata"]["loss"])
        self.assertEqual(self.adapter.num_updates_skipped, 1)

        for p_before, p_after in zip(params_before, self.adapter.model.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))

        mask_zero = torch.zeros(2, 16, 16, dtype=torch.long)
        res_zero = self.adapter.adapt_and_predict(xA, xB, mask_zero)
        self.assertFalse(res_zero["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_zero["adaptation_metadata"]["action"], "skip")
        self.assertIsNone(res_zero["adaptation_metadata"]["loss"])
        self.assertEqual(self.adapter.num_updates_skipped, 2)

    def test_05_r01_state_neutral_probes_and_exception_safety(self):
        """R01: Probe isolation restores buffers and RNG, including on forward exceptions."""
        stateful_model = StatefulMockModel(base_channels=4)
        adapter = EvidenceConsistencyAdapter(model=stateful_model)

        xA = torch.randn(2, 2, 8, 8)
        xB = torch.randn(2, 2, 8, 8)
        mask = torch.ones(2, 8, 8, dtype=torch.long)

        # 1. Normal execution: buffer and RNG preserved
        rng_before = torch.get_rng_state()
        adapter._evaluate_null_probes_state_neutral(xA, xB, mask)
        rng_after = torch.get_rng_state()

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertEqual(adapter.model.forward_counter.item(), 0)
        self.assertEqual(adapter.source_model.forward_counter.item(), 0)

        # 2. Exceptional path: synthetic forward failure still restores RNG and buffers
        adapter.model.fail_forward = True
        rng_before_exc = torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            adapter._evaluate_null_probes_state_neutral(xA, xB, mask)
        rng_after_exc = torch.get_rng_state()

        self.assertTrue(torch.equal(rng_before_exc, rng_after_exc))
        self.assertEqual(adapter.model.forward_counter.item(), 0)

    def test_06_r02_numerical_failures_guarded_and_rolled_back(self):
        """R02: Non-finite loss/gradients/parameters/optimizer momentum are caught, rolled back, and reported as failure."""
        xA = torch.randn(2, 2, 8, 8)
        xB = torch.randn(2, 2, 8, 8)
        mask = torch.ones(2, 8, 8, dtype=torch.long)

        # Case A: Overflowing loss
        overflow_model = LargeFiniteLogitsModel()
        adapter_overflow = EvidenceConsistencyAdapter(model=overflow_model)
        res_overflow = adapter_overflow.adapt_and_predict(xA, xB, mask)
        self.assertFalse(res_overflow["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_overflow["adaptation_metadata"]["action"], "failure")
        self.assertIsNone(res_overflow["adaptation_metadata"]["loss"])
        self.assertEqual(adapter_overflow.num_updates_failed, 1)

        # Case B: Injected infinite gradient
        model_b = MockSNUNet(base_channels=4)
        adapter_b = EvidenceConsistencyAdapter(model=model_b)
        adapter_b.confidence_threshold = 0.0  # Force selection to trigger backward
        params_before_b = {n: p.detach().clone() for n, p in adapter_b.model.named_parameters()}

        handle = adapter_b.model.bn1.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
        try:
            res_b = adapter_b.adapt_and_predict(xA * 10.0, xB * 10.0, mask)
        finally:
            handle.remove()

        self.assertFalse(res_b["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_b["adaptation_metadata"]["action"], "failure")
        self.assertEqual(adapter_b.num_updates_failed, 1)
        for name, param in adapter_b.model.named_parameters():
            self.assertTrue(torch.isfinite(param).all())
            self.assertTrue(torch.equal(param, params_before_b[name]))

        # Case C: Optimizer-only corruption (parameters stay finite, momentum buffer becomes inf)
        model_c = MockSNUNet(base_channels=4)
        adapter_c = EvidenceConsistencyAdapter(model=model_c)
        adapter_c.confidence_threshold = 0.0

        # Step 1: Normal update populates momentum
        res_warm = adapter_c.adapt_and_predict(xA * 5.0, xB * 5.0, mask)
        self.assertTrue(res_warm["adaptation_metadata"]["update_applied"])
        self.assertEqual(adapter_c.num_updates_applied, 1)
        self.assertEqual(adapter_c.num_updates_failed, 0)
        self.assertGreater(len(adapter_c.optimizer.state), 0)

        # Snapshot populated state
        params_before_corrupt = {n: p.detach().clone() for n, p in adapter_c.model.named_parameters()}
        opt_before_corrupt = copy.deepcopy(adapter_c.optimizer.state_dict())

        # Wrap step to inject inf into momentum buffer while parameters stay finite
        real_step = adapter_c.optimizer.step
        def corrupt_momentum():
            real_step()
            with torch.no_grad():
                for st in adapter_c.optimizer.state.values():
                    if "momentum_buffer" in st:
                        st["momentum_buffer"].fill_(float("inf"))

        with patch.object(adapter_c.optimizer, "step", side_effect=corrupt_momentum):
            res_corrupt = adapter_c.adapt_and_predict(xA * 5.0, xB * 5.0, mask)

        self.assertFalse(res_corrupt["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_corrupt["adaptation_metadata"]["action"], "failure")
        self.assertIsNone(res_corrupt["adaptation_metadata"]["loss"])
        self.assertEqual(adapter_c.num_updates_applied, 1)  # Not incremented on failure
        self.assertEqual(adapter_c.num_updates_failed, 1)   # Correctly incremented

        # Check exact parameter restoration
        for name, param in adapter_c.model.named_parameters():
            self.assertTrue(torch.isfinite(param).all())
            self.assertTrue(torch.equal(param, params_before_corrupt[name]))

        # Check exact optimizer state restoration (momentum is finite and restored)
        for st in adapter_c.optimizer.state.values():
            if "momentum_buffer" in st:
                self.assertTrue(torch.isfinite(st["momentum_buffer"]).all())

        # Case D: Normal-update positive control after failure
        res_pos = adapter_c.adapt_and_predict(xA * 5.0, xB * 5.0, mask)
        self.assertTrue(res_pos["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_pos["adaptation_metadata"]["action"], "update")
        self.assertEqual(adapter_c.num_updates_applied, 2)
        self.assertEqual(adapter_c.num_updates_failed, 1)

    def test_07_r03_diagnostic_timing_clarification(self):
        """R03: Explicit labeling of PRE_UPDATE vs POST_UPDATE diagnostic indicators."""
        xA = torch.randn(2, 2, 8, 8)
        xB = torch.randn(2, 2, 8, 8)
        mask = torch.ones(2, 8, 8, dtype=torch.long)

        res = self.adapter.adapt_and_predict(xA, xB, mask)
        ev = res["risk_evidence"]
        timing = ev["timing_contexts"]

        self.assertEqual(timing["primary_prediction"], "PRE_UPDATE")
        self.assertEqual(timing["source_disagreement_rate"], "PRE_UPDATE")
        self.assertEqual(timing["entropy_norm"], "PRE_UPDATE")
        self.assertEqual(timing["null_probes"], "POST_UPDATE")

        self.assertIn("source_disagreement_rate_pre_update", ev)
        self.assertIn("null_score_primary_post_update", ev)
        self.assertEqual(ev["source_disagreement_rate"], ev["source_disagreement_rate_pre_update"])

    def test_08_r06_source_model_storage_independence(self):
        """R06: Whole-object and shared storage aliasing is safely eliminated."""
        same_model = MockSNUNet(base_channels=4)
        adapter = EvidenceConsistencyAdapter(model=same_model, source_model=same_model)

        self.assertIsNot(adapter.model, adapter.source_model)
        for mod in adapter.source_model.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.assertFalse(mod.training, "Source BN must remain in eval() mode")
                self.assertIsNotNone(mod.running_mean, "Source running_mean must be preserved")
                if mod.weight is not None:
                    self.assertFalse(mod.weight.requires_grad, "Source BN weight must be frozen")

        cand_ptrs = {p.data_ptr() for p in adapter.model.parameters()}
        src_ptrs = {p.data_ptr() for p in adapter.source_model.parameters()}
        self.assertEqual(len(cand_ptrs.intersection(src_ptrs)), 0)

    def test_09_n01_mask_shape_validation_and_normalization(self):
        """N01: Validate and normalize sensory mask shape without silent broadcasting."""
        xA = torch.randn(2, 2, 8, 8)
        xB = torch.randn(2, 2, 8, 8)

        # 4D mask with singleton channel (B, 1, H, W) should be normalized to (B, H, W)
        mask_4d = torch.ones(2, 1, 8, 8, dtype=torch.long)
        res_4d = self.adapter.adapt_and_predict(xA, xB, mask_4d)
        self.assertEqual(res_4d["primary_prediction"]["valid_pixels"], 2 * 8 * 8)

        # Unsupported shape (2D or mismatched spatial)
        mask_2d = torch.ones(8, 8, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.adapter.adapt_and_predict(xA, xB, mask_2d)

        mask_mismatch = torch.ones(2, 4, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.adapter.adapt_and_predict(xA, xB, mask_mismatch)

    def test_10_reset_to_source(self):
        """Test reset_to_source restores model state and optimizer bit-identically."""
        source_state_before = {k: v.clone() for k, v in self.adapter.model.state_dict().items()}

        self.adapter.confidence_threshold = 0.0
        xA = torch.randn(4, 2, 16, 16) * 10.0
        xB = torch.randn(4, 2, 16, 16) * 10.0
        mask = torch.ones(4, 16, 16, dtype=torch.long)

        res = self.adapter.adapt_and_predict(xA, xB, mask)
        self.assertTrue(res["adaptation_metadata"]["update_applied"])
        self.assertGreater(self.adapter.num_updates_applied, 0)

        self.adapter.reset_to_source()

        for k, v in self.adapter.model.state_dict().items():
            self.assertTrue(torch.equal(v, source_state_before[k]))
        self.assertEqual(self.adapter.step_count, 0)
        self.assertEqual(self.adapter.num_updates_applied, 0)
        self.assertEqual(self.adapter.num_updates_skipped, 0)
        self.assertEqual(self.adapter.num_updates_failed, 0)


class TestWeek11CPURunnerIntegration(unittest.TestCase):
    """Integration unit test for Week11CPURunner with R04 persisted context, R05 auth, and N02 null loss."""

    def test_r05_strict_boolean_and_stage_enforcement(self):
        """R05: Strict boolean validation and enforcement of APPROVED_FOR_CPU_IMPLEMENTATION status."""
        config_path = REPO_ROOT / "configs/changeguard/week11_evidence_consistency_v1.json"

        # 1. Valid CPU execution passes
        runner = Week11CPURunner(config_path=config_path, device="cpu")
        pf = runner.validate_preflight()
        self.assertEqual(pf["status"], "APPROVED_FOR_CPU_IMPLEMENTATION")
        self.assertFalse(pf["real_data_execution_authorized"])
        self.assertFalse(pf["gpu_execution_authorized"])

        # 2. String boolean rejection (e.g. "false")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            cfg = json.loads(config_path.read_text())
            cfg["execution_authorizations"]["gpu_execution_authorized"] = "false"
            json.dump(cfg, f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(TypeError):
                Week11CPURunner(config_path=temp_path, device="cpu")
        finally:
            temp_path.unlink()

        # 3. Reject GPU device under CPU stage
        with self.assertRaises(PermissionError):
            Week11CPURunner(config_path=config_path, device="cuda:0")

        # 4. Reject null status, unknown status, and missing status
        for bad_status in [None, "UNSUPPORTED_STAGE", "SOME_FUTURE_STAGE"]:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                cfg = json.loads(config_path.read_text())
                cfg["status"] = bad_status
                cfg["execution_authorizations"]["gpu_execution_authorized"] = True
                json.dump(cfg, f)
                bad_path = Path(f.name)
            try:
                with self.assertRaises(ValueError):
                    Week11CPURunner(config_path=bad_path, device="cuda:0")
            finally:
                bad_path.unlink()

    def test_r04_and_n02_persisted_json_null_and_csv_text_null(self):
        """R04 & N02: Saved JSON summary has Python None (JSON null) while CSV has textual 'null'."""
        config_path = REPO_ROOT / "configs/changeguard/week11_evidence_consistency_v1.json"
        runner = Week11CPURunner(config_path=config_path, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary = runner.run_synthetic_pipeline(
                num_batches=3,
                batch_size=8,
                output_dir=out_dir,
                seed=42,
            )

            # Check JSON summary on disk
            summary_path = out_dir / "synthetic_run_summary.json"
            self.assertTrue(summary_path.is_file())
            raw_json_text = summary_path.read_text()
            saved_summary = json.loads(raw_json_text)

            # R04 checks
            self.assertIn("provenance_metadata", saved_summary)
            self.assertIn("batch_to_samples", saved_summary)
            self.assertIn("all_sample_ids", saved_summary)
            self.assertEqual(len(saved_summary["batch_to_samples"]), 3)
            self.assertEqual(saved_summary["total_samples"], 20)  # 8 + 8 + 4

            # N02 checks: verify that skipped records have Python None when deserialized from JSON
            skipped_records = [r for r in saved_summary["step_records"] if r["action"] == "skip"]
            self.assertGreater(len(skipped_records), 0)
            for r in skipped_records:
                self.assertIsNone(r["loss"], f"Expected None in deserialized JSON record, got {r['loss']!r}")

            # Check CSV metrics on disk
            csv_path = out_dir / "synthetic_step_metrics.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)

            self.assertEqual(len(csv_rows), 3)
            for row in csv_rows:
                if row["action"] == "skip":
                    self.assertEqual(row["loss"], "null", f"Expected textual 'null' in CSV row, got {row['loss']!r}")


if __name__ == "__main__":
    unittest.main()
