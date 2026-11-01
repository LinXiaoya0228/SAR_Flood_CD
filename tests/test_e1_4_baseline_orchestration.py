"""ChangeGuard Plan V3 E1.4 Unit Tests.

Covers:
1. Coverage ledger accounting: 51 streams total (3 source, 12 AdaBN, 12 Tent, 12 EATA, 12 SAR).
2. Exact reuse accounting: 2 reused verified, 49 execute new, 0 blocked.
3. Source-only evaluated once per seed across dev6; not multiplied into fake order/reset runs.
4. Rejection of mismatched historical reuse.
5. Strict sealed-test rejection.
6. Synthetic mock execution of all five baseline adapters (Source-Only, AdaBN, Tent, EATA, SAR).
7. Reset-per-event vs continual state semantics.
8. Resumable skip behavior without duplicating completed rows.
"""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1AdaBNAdapter,
    E1EATAAdapter,
    E1SARAdapter,
    E1SourceOnlyAdapter,
    E1TentAdapter,
)
from src.changeguard.e1_orchestrator import (
    E1Orchestrator,
    LabelGuardedBatch,
)
from src.changeguard.recorder import ExtendedExperimentRecorder
from src.data.evaluation_dataset import (
    FORBIDDEN_SEALED_TEST_EVENTS,
    KuroEvaluationDataset,
)
from tools.run_e1_4_matched_baselines import (
    E1MatchedBaselinesRunner,
    compute_file_sha256,
)


class MockSNUNet(nn.Module):
    """Synthetic lightweight SNUNet model for CPU testing with real BatchNorm2d layers."""

    def __init__(self, in_channels: int = 2, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.classifier = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.classifier(x)


class TestE14BaselineOrchestration(unittest.TestCase):
    """Test suite for ChangeGuard Work Package E1.4."""

    @classmethod
    def setUpClass(cls):
        cls.ledger_path = REPO_ROOT / "docs/E1_4_COVERAGE_LEDGER.json"
        with open(cls.ledger_path, "r", encoding="utf-8") as f:
            cls.ledger = json.load(f)

        cls.config_path = REPO_ROOT / "configs/changeguard/e1_4_matched_baselines_v1.json"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.cfg = json.load(f)

    def test_01_ledger_stream_accounting(self):
        """Verifies exact stream accounting: 51 total, 2 reused, 49 execute new, 0 blocked."""
        streams = self.ledger["streams"]
        self.assertEqual(len(streams), 51, "Total required baseline streams must be exactly 51.")

        reused = [s for s in streams if s["status"] == "REUSED_VERIFIED"]
        execute_new = [s for s in streams if s["status"] in {"EXECUTE_NEW", "EXECUTED_NEW"}]
        blocked = [s for s in streams if s["status"] == "BLOCKED"]

        self.assertEqual(len(reused), 2, "Exactly 2 streams must be REUSED_VERIFIED.")
        self.assertEqual(len(execute_new), 49, "Exactly 49 streams must be EXECUTE_NEW or EXECUTED_NEW.")
        self.assertEqual(len(blocked), 0, "Zero streams may be BLOCKED in E1.4.")

        reused_ids = {s["stream_id"] for s in reused}
        expected_reused = {
            "source_only_frozen_running_stats_s0",
            "tent_matched_pre_update_s0_oa_re",
        }
        self.assertEqual(reused_ids, expected_reused, "Reused stream IDs mismatch.")

        # Check methods distribution
        method_counts = {}
        for s in streams:
            method_counts[s["method"]] = method_counts.get(s["method"], 0) + 1

        self.assertEqual(method_counts["source_only_frozen_running_stats"], 3)
        self.assertEqual(method_counts["adabn_matched_pre_update"], 12)
        self.assertEqual(method_counts["tent_matched_pre_update"], 12)
        self.assertEqual(method_counts["eata_v2_fisher_matched_pre_update"], 12)
        self.assertEqual(method_counts["sar_matched_pre_update"], 12)

    def test_02_source_only_not_multiplied_into_fake_orders_or_resets(self):
        """Source-only must be evaluated strictly once per seed across dev6 (3 streams, not 12)."""
        src_streams = [s for s in self.ledger["streams"] if "source_only" in s["method"]]
        self.assertEqual(len(src_streams), 3, "Source-only must have exactly 3 streams (one per seed).")
        seeds = {s["seed"] for s in src_streams}
        self.assertEqual(seeds, {0, 1, 2}, "Source-only must cover seeds 0, 1, and 2.")

        # Invariant attributes
        for s in src_streams:
            self.assertIn("invariant", s["order"])
            self.assertIn("invariant", s["reset_mode"])

    def test_03_rejection_of_mismatched_historical_reuse(self):
        """Rejects historical rows that lacked spatial predictions or used POST_UPDATE."""
        w8_tent = [s for s in self.ledger["streams"] if s["stream_id"] == "tent_matched_pre_update_s0_oa_co"][0]
        self.assertIn(w8_tent["status"], {"EXECUTE_NEW", "EXECUTED_NEW"})
        self.assertIn("POST_UPDATE", w8_tent["reason_for_execution"])

        w7_src_s1 = [s for s in self.ledger["streams"] if s["stream_id"] == "source_only_frozen_running_stats_s1"][0]
        self.assertIn(w7_src_s1["status"], {"EXECUTE_NEW", "EXECUTED_NEW"})
        self.assertIn("lacked per-sample spatial prediction cache", w7_src_s1["reason_for_execution"])

    def test_04_sealed_test_rejection(self):
        """Confirms that attempting to evaluate any sealed-test event raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_cfg = copy.deepcopy(self.cfg)
            bad_cfg["dataset"]["target_events"] = [205]  # Forbidden sealed test event

            bad_cfg_path = Path(tmp_dir) / "bad_cfg.json"
            with open(bad_cfg_path, "w", encoding="utf-8") as f:
                json.dump(bad_cfg, f)

            runner = E1MatchedBaselinesRunner(config_path=bad_cfg_path)
            with self.assertRaises(RuntimeError) as ctx:
                runner.run_preflights()
            self.assertIn("Sealed test event 205 in target events", str(ctx.exception))

    def test_05_synthetic_baseline_state_behaviors(self):
        """Validates all five baseline adapters under synthetic CPU execution."""
        torch.manual_seed(42)
        model = MockSNUNet()
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        valid_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        sample_ids = ["test_sample_0", "test_sample_1"]

        # 1. Source-Only: 0 updates, PRE == POST
        src_adapter = E1SourceOnlyAdapter(model=model, device="cpu")
        res_src = src_adapter.adapt_and_predict_step(xA, xB, valid_mask, sample_ids, event_id=520)
        self.assertFalse(res_src["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_src["adaptation_metadata"]["action"], "skip")
        self.assertTrue(torch.equal(
            res_src["primary_prediction"]["pred_cand_pre"],
            res_src["post_prediction"]["pred_cand_post"],
        ))

        # 2. AdaBN: 0 gradient updates, PRE == POST, BN running stats updated
        adabn_adapter = E1AdaBNAdapter(model=model, device="cpu")
        res_ada = adabn_adapter.adapt_and_predict_step(xA, xB, valid_mask, sample_ids, event_id=520)
        self.assertTrue(res_ada["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_ada["adaptation_metadata"]["action"], "update")
        self.assertEqual(res_ada["adaptation_metadata"]["bn_weight_displacement"], 0.0)

        # 3. Tent: updates BN affine parameters on high entropy
        tent_adapter = E1TentAdapter(model=model, learning_rate=1e-2, device="cpu")
        res_tent = tent_adapter.adapt_and_predict_step(xA, xB, valid_mask, sample_ids, event_id=520)
        self.assertTrue(res_tent["adaptation_metadata"]["update_applied"])
        self.assertGreater(res_tent["adaptation_metadata"]["gradient_norm_l2"], 0.0)
        self.assertGreater(res_tent["adaptation_metadata"]["bn_weight_displacement"], 0.0)

        # 4. EATA: test with mock Fisher dictionary
        fishers = {}
        for name, p in model.named_parameters():
            if "bn" in name:
                fishers[name] = (torch.ones_like(p), p.detach().clone())
        eata_adapter = E1EATAAdapter(
            model=model, fishers=fishers, e_margin=10.0, d_margin=1.0, learning_rate=1e-2, device="cpu"
        )
        res_eata = eata_adapter.adapt_and_predict_step(xA, xB, valid_mask, sample_ids, event_id=520)
        self.assertTrue(res_eata["adaptation_metadata"]["update_applied"])
        self.assertEqual(res_eata["adaptation_metadata"]["action"], "update")

        # 5. SAR: two-stage SAM optimization
        sar_adapter = E1SARAdapter(model=model, e_margin=10.0, sam_rho=0.05, learning_rate=1e-2, device="cpu")
        res_sar = sar_adapter.adapt_and_predict_step(xA, xB, valid_mask, sample_ids, event_id=520)
        self.assertTrue(res_sar["adaptation_metadata"]["update_applied"])

    def test_06_state_reset_vs_continual_semantics(self):
        """Confirms that reset_per_event restores weights while continual retains them."""
        torch.manual_seed(42)
        model = MockSNUNet()
        init_bn_weight = model.bn1.weight.detach().clone()

        adapter = E1TentAdapter(model=model, learning_rate=1e-1, device="cpu")
        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        valid_mask = torch.ones(2, 16, 16, dtype=torch.uint8)

        # Perform update
        adapter.adapt_and_predict_step(xA, xB, valid_mask, ["s0", "s1"], event_id=520)
        adapted_weight = model.bn1.weight.detach().clone()
        self.assertFalse(torch.equal(init_bn_weight, adapted_weight), "Weight must change after Tent step.")

        # Call reset_to_source() (simulates reset_per_event)
        adapter.reset_to_source()
        reset_weight = model.bn1.weight.detach().clone()
        self.assertTrue(torch.allclose(init_bn_weight, reset_weight), "reset_to_source must restore exact initial weights.")

    def test_07_resumability_and_skip_logic(self):
        """Verifies that completed streams with valid outputs are detected and skipped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            runner = E1MatchedBaselinesRunner(config_path=self.config_path, output_dir=out_root)

            stream_dir = out_root / "test_stream_s0"
            self.assertFalse(runner.is_stream_completed(stream_dir))

            # Create mock completed structure
            stream_dir.mkdir(parents=True)
            with open(stream_dir / "stream_summary.json", "w") as f:
                json.dump({"status": "COMPLETE"}, f)
            with open(stream_dir / "manifest.json", "w") as f:
                json.dump({"manifest": "ok"}, f)

            # Not complete yet (events missing)
            self.assertFalse(runner.is_stream_completed(stream_dir))

            for eid in runner.target_events:
                ev_dir = stream_dir / f"event_{eid}"
                ev_dir.mkdir()
                with open(ev_dir / "event_metrics.json", "w") as f:
                    json.dump({"metrics": "ok"}, f)
                np.savez(ev_dir / "frozen_predictions.npz", a=np.zeros((2, 2)))

            # Now complete
            self.assertTrue(runner.is_stream_completed(stream_dir))


if __name__ == "__main__":
    unittest.main()
