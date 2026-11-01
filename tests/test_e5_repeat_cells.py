#!/usr/bin/env python3
"""Tests for ChangeGuard Work Package E5.3 repeat cells (Seed Repeat and Order Repeat).

Covers:
1. Correct seed/checkpoint binding across seeds 0 and 1.
2. Checkpoint-hash rejection fail-closed behavior.
3. Exact order-A and order-B event sequences and within-event sample ordering.
4. Run ID and output directory separation across cells.
5. Rejection of unsupported seed and stream order options.
6. Sealed-test event quarantine and rejection.
7. Prevention of output directory overwrite.
8. Persistence-before-label enforcement.
9. Equality of all non-seed / non-order scientific settings with canonical cell.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)
from tools.run_e5_risk_scoring import (
    CANONICAL_CHECKPOINTS,
    CANONICAL_STREAM_ORDERS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SEED1_SHA256,
    E5RiskScoringRunner,
    compute_file_sha256,
)


class TestE5RepeatCells(unittest.TestCase):
    """Rigorous test suite for orthogonal E5.3 repeat cells."""

    def setUp(self):
        self.canonical_cfg_path = REPO_ROOT / "configs/changeguard/e5_temporal_risk_v1.json"
        self.s0_ob_cfg_path = REPO_ROOT / "configs/changeguard/e5_temporal_risk_s0_ob_v1.json"
        self.s1_oa_cfg_path = REPO_ROOT / "configs/changeguard/e5_temporal_risk_s1_oa_v1.json"

        self.assertTrue(self.canonical_cfg_path.exists(), "Canonical config missing")
        self.assertTrue(self.s0_ob_cfg_path.exists(), "Order-B repeat config missing")
        self.assertTrue(self.s1_oa_cfg_path.exists(), "Seed-1 repeat config missing")

        with open(self.canonical_cfg_path, "r", encoding="utf-8") as f:
            self.canonical_cfg = json.load(f)
        with open(self.s0_ob_cfg_path, "r", encoding="utf-8") as f:
            self.s0_ob_cfg = json.load(f)
        with open(self.s1_oa_cfg_path, "r", encoding="utf-8") as f:
            self.s1_oa_cfg = json.load(f)

    def test_01_seed_and_checkpoint_binding(self):
        """Verifies correct checkpoint path and SHA-256 binding for seeds 0 and 1."""
        # Seed 0 binding
        runner_s0 = E5RiskScoringRunner(
            config_path=self.s0_ob_cfg_path,
            device="cpu",
            is_smoke=True,
            allow_dirty=True,
        )
        preflight_s0 = runner_s0.run_preflight_checks()
        self.assertEqual(preflight_s0["checkpoint_seed"], 0)
        self.assertEqual(preflight_s0["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        self.assertTrue(Path(preflight_s0["checkpoint_path"]).is_file())
        self.assertEqual(
            compute_file_sha256(Path(preflight_s0["checkpoint_path"])),
            EXPECTED_CHECKPOINT_SHA256,
        )

        # Seed 1 binding
        runner_s1 = E5RiskScoringRunner(
            config_path=self.s1_oa_cfg_path,
            device="cpu",
            is_smoke=True,
            allow_dirty=True,
        )
        preflight_s1 = runner_s1.run_preflight_checks()
        self.assertEqual(preflight_s1["checkpoint_seed"], 1)
        self.assertEqual(preflight_s1["checkpoint_sha256"], EXPECTED_CHECKPOINT_SEED1_SHA256)
        self.assertTrue(Path(preflight_s1["checkpoint_path"]).is_file())
        self.assertEqual(
            compute_file_sha256(Path(preflight_s1["checkpoint_path"])),
            EXPECTED_CHECKPOINT_SEED1_SHA256,
        )

        # Distinct checkpoints confirmed
        self.assertNotEqual(preflight_s0["checkpoint_sha256"], preflight_s1["checkpoint_sha256"])

    def test_02_checkpoint_hash_rejection(self):
        """Verifies that an altered or mismatched checkpoint hash causes preflight rejection."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_p = Path(tmp.name)
            tampered_cfg = copy.deepcopy(self.s1_oa_cfg)
            tampered_cfg["source_checkpoints"]["seeds"]["1"]["expected_sha256"] = "0" * 64
            json.dump(tampered_cfg, tmp)

        try:
            runner = E5RiskScoringRunner(
                config_path=tmp_p,
                device="cpu",
                is_smoke=True,
                allow_dirty=True,
            )
            with self.assertRaises(ValueError) as ctx:
                runner.run_preflight_checks()
            self.assertIn("Checkpoint SHA-256 mismatch", str(ctx.exception))
        finally:
            if tmp_p.exists():
                tmp_p.unlink()

    def test_03_exact_order_a_and_order_b_event_sequence(self):
        """Verifies exact event sequences and sample-level sorting for Order A vs Order B."""
        expected_order_a = [279, 437, 520, 559, 1111003, 1111008]
        expected_order_b = [1111008, 1111003, 559, 520, 437, 279]

        self.assertEqual(
            self.s0_ob_cfg["screening_specification"]["event_sequence"],
            expected_order_b,
        )
        self.assertEqual(
            self.s1_oa_cfg["screening_specification"]["event_sequence"],
            expected_order_a,
        )

        # Order A runner sequence
        runner_oa = E5RiskScoringRunner(
            config_path=self.s1_oa_cfg_path,
            device="cpu",
            is_smoke=False,
            allow_dirty=True,
        )
        self.assertEqual(runner_oa.target_events, expected_order_a)
        self.assertEqual(runner_oa.stream_order, "order_a")

        # Order B runner sequence
        runner_ob = E5RiskScoringRunner(
            config_path=self.s0_ob_cfg_path,
            device="cpu",
            is_smoke=False,
            allow_dirty=True,
        )
        self.assertEqual(runner_ob.target_events, expected_order_b)
        self.assertEqual(runner_ob.stream_order, "order_b")

        # Check within-event sorting in batches
        batches_oa = runner_oa.build_ordered_stream_batches()
        batches_ob = runner_ob.build_ordered_stream_batches()

        # Both should have exactly 179 batches and 2,826 samples
        total_samples_oa = sum(len(b.sample_ids) for _, b_list in batches_oa for b in b_list)
        total_batches_oa = sum(len(b_list) for _, b_list in batches_oa)
        total_samples_ob = sum(len(b.sample_ids) for _, b_list in batches_ob for b in b_list)
        total_batches_ob = sum(len(b_list) for _, b_list in batches_ob)

        self.assertEqual(total_samples_oa, 2826)
        self.assertEqual(total_batches_oa, 179)
        self.assertEqual(total_samples_ob, 2826)
        self.assertEqual(total_batches_ob, 179)

        # Verify within-event sample IDs are ascending for Order A and descending for Order B
        for eid, b_list in batches_oa:
            all_sids = [sid for b in b_list for sid in b.sample_ids]
            self.assertEqual(all_sids, sorted(all_sids))

        for eid, b_list in batches_ob:
            all_sids = [sid for b in b_list for sid in b.sample_ids]
            self.assertEqual(all_sids, sorted(all_sids, reverse=True))

    def test_04_run_id_and_output_path_separation(self):
        """Verifies strictly disjoint run IDs and output paths across canonical and repeat cells."""
        canon_run_id = self.canonical_cfg["screening_specification"]["run_id"]
        s0_ob_run_id = self.s0_ob_cfg["screening_specification"]["run_id"]
        s1_oa_run_id = self.s1_oa_cfg["screening_specification"]["run_id"]

        self.assertEqual(canon_run_id, "e5_temporal_risk_s0_oa_v1")
        self.assertEqual(s0_ob_run_id, "e5_temporal_risk_s0_ob_v1")
        self.assertEqual(s1_oa_run_id, "e5_temporal_risk_s1_oa_v1")
        self.assertEqual(len({canon_run_id, s0_ob_run_id, s1_oa_run_id}), 3)

        canon_out = self.canonical_cfg["screening_specification"]["output_dir"]
        s0_ob_out = self.s0_ob_cfg["screening_specification"]["output_dir"]
        s1_oa_out = self.s1_oa_cfg["screening_specification"]["output_dir"]

        self.assertEqual(canon_out, "outputs/e5_temporal_evidence/formal_s0_oa_v1/")
        self.assertEqual(s0_ob_out, "outputs/e5_temporal_evidence/formal_s0_ob_v1/")
        self.assertEqual(s1_oa_out, "outputs/e5_temporal_evidence/formal_s1_oa_v1/")
        self.assertEqual(len({canon_out, s0_ob_out, s1_oa_out}), 3)

    def test_05_rejection_of_unsupported_seed_or_order(self):
        """Verifies that invalid or unconfigured seeds/orders raise ValueError immediately."""
        # Unsupported seed
        with self.assertRaises(ValueError) as ctx_seed:
            _ = E5RiskScoringRunner(
                config_path=self.s0_ob_cfg_path,
                checkpoint_seed=99,
                device="cpu",
                is_smoke=True,
                allow_dirty=True,
            )
        self.assertIn("Unsupported checkpoint seed", str(ctx_seed.exception))

        # Unsupported order
        with self.assertRaises(ValueError) as ctx_order:
            _ = E5RiskScoringRunner(
                config_path=self.s0_ob_cfg_path,
                stream_order="order_invalid",
                device="cpu",
                is_smoke=True,
                allow_dirty=True,
            )
        self.assertIn("Unsupported stream_order", str(ctx_order.exception))

    def test_06_sealed_test_rejection_in_repeat_cells(self):
        """Verifies that injecting a sealed-test event into repeat configs triggers critical quarantine failure."""
        for forbidden in FORBIDDEN_SEALED_TEST_EVENTS:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                tmp_p = Path(tmp.name)
                tampered = copy.deepcopy(self.s0_ob_cfg)
                tampered["screening_specification"]["event_sequence"].append(forbidden)
                json.dump(tampered, tmp)

            try:
                with self.assertRaises(RuntimeError) as ctx:
                    _ = E5RiskScoringRunner(
                        config_path=tmp_p,
                        device="cpu",
                        is_smoke=False,
                        allow_dirty=True,
                    )
                self.assertIn("CRITICAL SECURITY VIOLATION: Sealed-test event", str(ctx.exception))
            finally:
                if tmp_p.exists():
                    tmp_p.unlink()

    def test_07_prevention_of_output_overwrite(self):
        """Verifies fail-closed execution if planned output directory exists and is non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "test_out"
            out_p.mkdir()
            (out_p / "existing_shard.npz").write_text("pre-existing data")

            runner = E5RiskScoringRunner(
                config_path=self.s0_ob_cfg_path,
                output_dir=out_p,
                device="cpu",
                is_smoke=False,
                allow_dirty=True,
            )
            with self.assertRaises(RuntimeError) as ctx:
                runner.execute_streaming_evaluation()
            self.assertIn("Fresh output directory violation", str(ctx.exception))

    def test_08_persistence_before_label_enforcement(self):
        """Verifies that evidence shard recording occurs before label interaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "test_smoke_out"
            runner = E5RiskScoringRunner(
                config_path=self.s0_ob_cfg_path,
                output_dir=out_p,
                device="cpu",
                is_smoke=True,
                limit_samples=16,
                allow_dirty=True,
            )
            # Execute one batch smoke
            summary = runner.execute_streaming_evaluation()
            self.assertEqual(summary["total_batches_processed"], 1)

            # Check that shard was written and indexed before completion
            shard_path = out_p / "event_520" / "evidence_shards" / "batch_000000.npz"
            self.assertTrue(shard_path.exists(), "Evidence shard was not persisted!")

            # Verify index has the entry with prelabel timing recorded
            index_path = out_p / "event_520" / "evidence_index.json"
            self.assertTrue(index_path.exists())
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            self.assertEqual(len(idx_data["shards"]), 1)
            shard_entry = idx_data["shards"][0]
            self.assertIn("timing_ms", shard_entry)
            self.assertIn("prelabel_evidence_elapsed_ms", shard_entry)
            self.assertGreater(shard_entry["timing_ms"], 0.0)

    def test_09_equality_of_non_factor_scientific_settings(self):
        """Verifies all non-seed/non-order scientific settings are identical across configs."""
        configs = [
            ("s0_ob", self.s0_ob_cfg),
            ("s1_oa", self.s1_oa_cfg),
        ]

        canon = self.canonical_cfg

        for name, cfg in configs:
            # Model specification identical
            self.assertEqual(cfg["model"], canon["model"], f"{name}: model spec mismatch")

            # Dataset specification identical
            self.assertEqual(cfg["dataset"], canon["dataset"], f"{name}: dataset spec mismatch")

            # Risk signals specification identical
            self.assertEqual(
                cfg["risk_signals_specification"],
                canon["risk_signals_specification"],
                f"{name}: risk signals spec mismatch",
            )

            # Evaluation criteria identical
            self.assertEqual(
                cfg["evaluation_criteria"],
                canon["evaluation_criteria"],
                f"{name}: evaluation criteria mismatch",
            )

            # Candidate adaptation parameters identical
            cand = cfg["contexts"]["candidate_e2_conf90"]
            cand_canon = canon["contexts"]["candidate_e2_conf90"]
            for key in [
                "confidence_threshold",
                "augmentation",
                "prediction_timing",
                "update_parameters",
                "optimizer",
                "learning_rate",
                "betas",
                "weight_decay",
                "reset_mode",
            ]:
                self.assertEqual(cand[key], cand_canon[key], f"{name}: candidate param {key} mismatch")

            # Batch size and bounded memory settings identical
            self.assertEqual(
                cfg["screening_specification"]["batch_size"],
                canon["screening_specification"]["batch_size"],
            )
            self.assertEqual(
                cfg["screening_specification"]["tail_batch_handling"],
                canon["screening_specification"]["tail_batch_handling"],
            )
            self.assertEqual(
                cfg["screening_specification"]["produce_convenience_aggregate"],
                canon["screening_specification"]["produce_convenience_aggregate"],
            )
            self.assertEqual(
                cfg["screening_specification"]["authoritative_shard_policy"],
                canon["screening_specification"]["authoritative_shard_policy"],
            )


if __name__ == "__main__":
    unittest.main()

