"""Unit tests for Week 3 source replication safeguards and provenance verification.

Verifies:
1. Seed fields in replication configs are exactly 1 or 2 as intended.
2. Seed-0 configs remain unchanged (matching frozen hashes).
3. Paired A/B configs differ only in approved channel and slope fields.
4. Seed replications differ only in seed and run metadata.
5. Protected events remain strictly excluded in configs, split, and locks.
6. All configuration and lock hashes resolve in validate_provenance_and_config.
7. Draft, mismatched, or tampered configurations fail fast and cannot launch.
"""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.run_manifest import compute_file_sha256
from tools.train_source_model import validate_provenance_and_config

REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_SEED0_MODEL_A_HASH = "528d0f2ca2a0b1e30873f0bac4cfefa5f5637714d17800c94a9a99bf5199dbc3"
FROZEN_SEED0_MODEL_B_HASH = "e99e980a9afc0f72b526ccff5484e4aadaabece12c5a542bd20f94be74ef6b56"
FROZEN_PROTOCOL_V2_HASH = "b9790bf84185cf55ec6947cfe6cb7e16df52ab2b7776e1bedbaf795c57050144"
FROZEN_SPLIT_HASH = "7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3"
FROZEN_SLOPE_STATS_HASH = "bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04"

WEEK3_PROTOCOL_PATH = REPO_ROOT / "docs/week3_source_replication_protocol.md"
WEEK3_LOCK_PATH = REPO_ROOT / "configs/source/source_training_week3_replication_lock.json"

CONFIG_PATHS = {
    "model_a_seed0": REPO_ROOT / "configs/source/snunet_vvvh_v2_fp32.json",
    "model_b_seed0": REPO_ROOT / "configs/source/snunet_vvvh_slope_v2_fp32.json",
    "model_a_seed1": REPO_ROOT / "configs/source/snunet_vvvh_seed1_v2_fp32.json",
    "model_b_seed1": REPO_ROOT / "configs/source/snunet_vvvh_slope_seed1_v2_fp32.json",
    "model_a_seed2": REPO_ROOT / "configs/source/snunet_vvvh_seed2_v2_fp32.json",
    "model_b_seed2": REPO_ROOT / "configs/source/snunet_vvvh_slope_seed2_v2_fp32.json",
}


class TestWeek3ReplicationSafeguards(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_seed_fields_match_intended_integers(self):
        """Verify seed fields in all replication configs are exactly 1 or 2 as intended."""
        with open(CONFIG_PATHS["model_a_seed1"], "r", encoding="utf-8") as f:
            cfg_a1 = json.load(f)
        with open(CONFIG_PATHS["model_b_seed1"], "r", encoding="utf-8") as f:
            cfg_b1 = json.load(f)
        with open(CONFIG_PATHS["model_a_seed2"], "r", encoding="utf-8") as f:
            cfg_a2 = json.load(f)
        with open(CONFIG_PATHS["model_b_seed2"], "r", encoding="utf-8") as f:
            cfg_b2 = json.load(f)

        self.assertEqual(cfg_a1["optimization_protocol"]["seeds"], [1])
        self.assertEqual(cfg_b1["optimization_protocol"]["seeds"], [1])
        self.assertEqual(cfg_a2["optimization_protocol"]["seeds"], [2])
        self.assertEqual(cfg_b2["optimization_protocol"]["seeds"], [2])

    def test_seed0_configs_remain_unchanged(self):
        """Verify seed-0 configs remain completely identical to frozen hashes."""
        actual_a0_hash = compute_file_sha256(CONFIG_PATHS["model_a_seed0"])
        actual_b0_hash = compute_file_sha256(CONFIG_PATHS["model_b_seed0"])

        self.assertEqual(actual_a0_hash, FROZEN_SEED0_MODEL_A_HASH)
        self.assertEqual(actual_b0_hash, FROZEN_SEED0_MODEL_B_HASH)

        with open(CONFIG_PATHS["model_a_seed0"], "r", encoding="utf-8") as f:
            cfg_a0 = json.load(f)
        with open(CONFIG_PATHS["model_b_seed0"], "r", encoding="utf-8") as f:
            cfg_b0 = json.load(f)

        self.assertEqual(cfg_a0["optimization_protocol"]["seeds"], [0])
        self.assertEqual(cfg_b0["optimization_protocol"]["seeds"], [0])

    def test_paired_ab_configs_differ_only_in_approved_channel_and_slope_fields(self):
        """Verify paired Model A and Model B configs differ only in channel and slope specifications."""
        for seed_idx, (key_a, key_b) in [(1, ("model_a_seed1", "model_b_seed1")), (2, ("model_a_seed2", "model_b_seed2"))]:
            with open(CONFIG_PATHS[key_a], "r", encoding="utf-8") as f:
                cfg_a = json.load(f)
            with open(CONFIG_PATHS[key_b], "r", encoding="utf-8") as f:
                cfg_b = json.load(f)

            # Invariant sections that must be 100% identical between A and B
            identical_sections = [
                "loss_specification",
                "label_specification",
                "data_split",
                "batch_size_policy",
                "model_selection_metrics",
                "final_retraining_rule",
            ]
            for sec in identical_sections:
                self.assertEqual(cfg_a[sec], cfg_b[sec], f"Section {sec} differed between A and B for seed {seed_idx}")

            # Optimization protocol must match exactly (same optimizer, lr, precision, seeds)
            self.assertEqual(cfg_a["optimization_protocol"], cfg_b["optimization_protocol"])

            # Architecture must differ only in in_channels_per_branch (2 vs 3)
            self.assertEqual(cfg_a["architecture"]["in_channels_per_branch"], 2)
            self.assertEqual(cfg_b["architecture"]["in_channels_per_branch"], 3)
            self.assertEqual(cfg_a["architecture"]["base_channel"], cfg_b["architecture"]["base_channel"])
            self.assertEqual(cfg_a["architecture"]["out_channels"], cfg_b["architecture"]["out_channels"])

            # Input channels
            self.assertEqual(cfg_a["input_specification"]["channels_per_branch"], ["vv", "vh"])
            self.assertEqual(cfg_b["input_specification"]["channels_per_branch"], ["vv", "vh", "slope"])
            self.assertNotIn("slope_policy", cfg_a["input_specification"])
            self.assertIn("slope_policy", cfg_b["input_specification"])

    def test_seed_replications_differ_only_in_seed_and_metadata(self):
        """Verify seed replications of the same model differ only in seeds and run metadata."""
        # Test Model A family: seed 0, seed 1, seed 2
        with open(CONFIG_PATHS["model_a_seed0"], "r", encoding="utf-8") as f:
            cfg_a0 = json.load(f)
        with open(CONFIG_PATHS["model_a_seed1"], "r", encoding="utf-8") as f:
            cfg_a1 = json.load(f)
        with open(CONFIG_PATHS["model_a_seed2"], "r", encoding="utf-8") as f:
            cfg_a2 = json.load(f)

        scientific_sections = [
            "architecture",
            "input_specification",
            "label_specification",
            "loss_specification",
            "batch_size_policy",
            "data_split",
            "model_selection_metrics",
        ]
        for sec in scientific_sections:
            self.assertEqual(cfg_a0[sec], cfg_a1[sec])
            self.assertEqual(cfg_a1[sec], cfg_a2[sec])

        # Optimization hyperparams match except seed
        for opt_key in ["optimizer", "learning_rate", "weight_decay", "betas", "precision", "mixed_precision", "autocast", "grad_scaler", "max_epochs", "early_stopping_patience"]:
            self.assertEqual(cfg_a0["optimization_protocol"][opt_key], cfg_a1["optimization_protocol"][opt_key])
            self.assertEqual(cfg_a1["optimization_protocol"][opt_key], cfg_a2["optimization_protocol"][opt_key])

        # Test Model B family: seed 0, seed 1, seed 2
        with open(CONFIG_PATHS["model_b_seed0"], "r", encoding="utf-8") as f:
            cfg_b0 = json.load(f)
        with open(CONFIG_PATHS["model_b_seed1"], "r", encoding="utf-8") as f:
            cfg_b1 = json.load(f)
        with open(CONFIG_PATHS["model_b_seed2"], "r", encoding="utf-8") as f:
            cfg_b2 = json.load(f)

        for sec in scientific_sections:
            self.assertEqual(cfg_b0[sec], cfg_b1[sec])
            self.assertEqual(cfg_b1[sec], cfg_b2[sec])

        for opt_key in ["optimizer", "learning_rate", "weight_decay", "betas", "precision", "mixed_precision", "autocast", "grad_scaler", "max_epochs", "early_stopping_patience"]:
            self.assertEqual(cfg_b0["optimization_protocol"][opt_key], cfg_b1["optimization_protocol"][opt_key])
            self.assertEqual(cfg_b1["optimization_protocol"][opt_key], cfg_b2["optimization_protocol"][opt_key])

    def test_protected_events_remain_excluded_across_all_configs_and_locks(self):
        """Verify quarantined TTA-val and sealed-test events are never referenced in training splits."""
        with open(WEEK3_LOCK_PATH, "r", encoding="utf-8") as f:
            lock_data = json.load(f)

        quarantined = set(lock_data["protected_events"]["quarantined_tta_validation"])
        sealed = set(lock_data["protected_events"]["sealed_test"])

        self.assertEqual(len(quarantined), 6)
        self.assertEqual(len(sealed), 10)
        self.assertTrue(quarantined.isdisjoint(sealed))

        split_path = REPO_ROOT / lock_data["frozen_split"]["path"]
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        train_events = set(split_data["source_inner_train_events"])
        val_events = set(split_data["source_inner_validation_events"])

        self.assertEqual(len(train_events), 22)
        self.assertEqual(len(val_events), 5)
        self.assertTrue(train_events.isdisjoint(val_events))
        self.assertTrue(train_events.isdisjoint(quarantined))
        self.assertTrue(train_events.isdisjoint(sealed))
        self.assertTrue(val_events.isdisjoint(quarantined))
        self.assertTrue(val_events.isdisjoint(sealed))

    def test_all_config_and_lock_hashes_resolve(self):
        """Verify validate_provenance_and_config resolves 100% for all replication and parent configs."""
        # 1. Test all 4 replication configs against Week 3 lock
        for key in ["model_a_seed1", "model_b_seed1", "model_a_seed2", "model_b_seed2"]:
            cfg_path = CONFIG_PATHS[key]
            res = validate_provenance_and_config(
                config_path=cfg_path,
                lock_path=WEEK3_LOCK_PATH,
                repo_root=REPO_ROOT,
            )
            self.assertEqual(res["status"], "VALIDATED")
            self.assertEqual(res["precision"], "fp32")
            self.assertEqual(res["split_hash"], FROZEN_SPLIT_HASH)
            if "model_b" in key:
                self.assertEqual(res["slope_stats_hash"], FROZEN_SLOPE_STATS_HASH)

        # 2. Test parent seed-0 configs against Week 2 lock
        res_a0 = validate_provenance_and_config(
            config_path=CONFIG_PATHS["model_a_seed0"],
            lock_path=REPO_ROOT / "configs/source/source_training_v2_lock.json",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(res_a0["status"], "VALIDATED")

        res_b0 = validate_provenance_and_config(
            config_path=CONFIG_PATHS["model_b_seed0"],
            lock_path=REPO_ROOT / "configs/source/source_training_v2_lock.json",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(res_b0["status"], "VALIDATED")

    def test_draft_or_mismatched_configs_cannot_launch(self):
        """Verify mismatched seed, tampered config, unapproved run ID, or protocol mismatch fail fast."""
        cfg_a1 = CONFIG_PATHS["model_a_seed1"]

        # 1. Seed mismatch between CLI and config
        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a1,
                seed=2,  # Config declared seed is 1
                lock_path=WEEK3_LOCK_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertIn("Seed mismatch", str(ctx.exception))

        # 2. Unapproved seed
        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a1,
                seed=999,
                lock_path=WEEK3_LOCK_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertIn("not in approved lock seeds", str(ctx.exception))

        # 3. Mismatched run ID
        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=cfg_a1,
                run_id="source_snunet_vvvh_seed99_v3_fp32",
                lock_path=WEEK3_LOCK_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertIn("not in approved lock run IDs", str(ctx.exception))

        # 4. Tampered config content
        with open(cfg_a1, "r", encoding="utf-8") as f:
            tampered_data = json.load(f)
        tampered_data["architecture"]["base_channel"] = 64
        tampered_path = self.temp_dir / "tampered_cfg.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_data, f)

        with self.assertRaises(RuntimeError) as ctx:
            validate_provenance_and_config(
                config_path=tampered_path,
                lock_path=WEEK3_LOCK_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertIn("config hash mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
