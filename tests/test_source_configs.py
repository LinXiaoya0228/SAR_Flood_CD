import hashlib
import json
import unittest
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from src.training.source_trainer import SourceTrainer


class TestSourceConfigs(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.config_a_draft_path = self.repo_root / "configs" / "source" / "snunet_vvvh_draft.json"
        self.config_b_draft_path = self.repo_root / "configs" / "source" / "snunet_vvvh_slope_draft.json"
        self.config_a_v1_path = self.repo_root / "configs" / "source" / "snunet_vvvh_v1.json"
        self.config_b_v1_path = self.repo_root / "configs" / "source" / "snunet_vvvh_slope_v1.json"
        self.split_v1_path = self.repo_root / "configs" / "splits" / "kuro_source_inner_split_v1.json"
        self.kuro_split_path = self.repo_root / "configs" / "kuro_event_split.json"

        with open(self.config_a_draft_path, "r") as f:
            self.cfg_a_draft = json.load(f)

        with open(self.config_b_draft_path, "r") as f:
            self.cfg_b_draft = json.load(f)

        with open(self.config_a_v1_path, "r") as f:
            self.cfg_a_v1 = json.load(f)

        with open(self.config_b_v1_path, "r") as f:
            self.cfg_b_v1 = json.load(f)

        with open(self.kuro_split_path, "r") as f:
            self.kuro_split = json.load(f)

    def test_json_parsing_and_schema_drafts(self):
        for cfg in [self.cfg_a_draft, self.cfg_b_draft]:
            self.assertEqual(cfg["$schema_version"], "1.0.0")
            self.assertEqual(cfg["status"], "AWAITING_USER_APPROVAL")
            self.assertEqual(cfg["task"], "change_detection")

    def test_json_parsing_and_schema_v1(self):
        for cfg in [self.cfg_a_v1, self.cfg_b_v1]:
            self.assertEqual(cfg["$schema_version"], "1.0.0")
            self.assertEqual(cfg["status"], "WEEK0_FROZEN_PENDING_WEEK1_SMOKE")
            self.assertEqual(cfg["approval_date"], "2026-10-06")
            self.assertEqual(cfg["task"], "change_detection")

    def test_channel_counts_and_ablation(self):
        # Model A must have exactly 2 per-time channels
        for cfg_a in [self.cfg_a_draft, self.cfg_a_v1]:
            self.assertEqual(cfg_a["architecture"]["in_channels_per_branch"], 2)
            self.assertEqual(cfg_a["input_specification"]["channels_per_branch"], ["vv", "vh"])

        # Model B must have exactly 3 per-time channels
        for cfg_b in [self.cfg_b_draft, self.cfg_b_v1]:
            self.assertEqual(cfg_b["architecture"]["in_channels_per_branch"], 3)
            self.assertEqual(cfg_b["input_specification"]["channels_per_branch"], ["vv", "vh", "slope"])

    def test_model_b_provenance_and_ablation_status(self):
        prov = self.cfg_b_v1["provenance_clarification"]
        self.assertEqual(prov["ablation_status"], "CONTROLLED_SCIENTIFIC_ABLATION")
        self.assertIn("UNKNOWN", prov["official_checkpoint_relationship"])
        self.assertIn("slope", self.cfg_b_v1["input_specification"]["channels_per_branch"])

    def test_output_classes_and_label_mapping(self):
        for cfg in [self.cfg_a_draft, self.cfg_b_draft, self.cfg_a_v1, self.cfg_b_v1]:
            self.assertEqual(cfg["architecture"]["out_channels"], 3)
            self.assertEqual(cfg["label_specification"]["num_classes"], 3)
            self.assertEqual(cfg["label_specification"]["ignore_index"], 3)

    def test_v1_split_reference_and_sample_policy(self):
        for cfg in [self.cfg_a_v1, self.cfg_b_v1]:
            self.assertEqual(cfg["data_split"]["split_config_path"], "configs/splits/kuro_source_inner_split_v1.json")
            self.assertEqual(cfg["data_split"]["source_inner_train_event_count"], 22)
            self.assertEqual(cfg["data_split"]["source_inner_val_event_count"], 5)
            self.assertIn("no_filtering", cfg["data_split"]["sample_policy"])

    def test_test_event_and_tta_isolation_all_configs(self):
        sealed_tests = set(self.kuro_split["effective_splits"]["test"])
        tta_vals = set(self.kuro_split["effective_splits"]["validation"])
        for cfg in [self.cfg_a_draft, self.cfg_b_draft, self.cfg_a_v1, self.cfg_b_v1]:
            cfg_str = json.dumps(cfg)
            for tid in sealed_tests:
                self.assertNotIn(f'"{tid}"', cfg_str)
            for vid in tta_vals:
                self.assertNotIn(f'"{vid}"', cfg_str)

    def test_loss_specification_exactness_v1(self):
        for cfg in [self.cfg_a_v1, self.cfg_b_v1]:
            loss = cfg["loss_specification"]
            self.assertEqual(loss["name"], "unweighted_ce_plus_soft_dice")
            self.assertEqual(loss["weight_ce"], 0.5)
            self.assertEqual(loss["weight_dice"], 0.5)
            self.assertEqual(loss["classes_in_dice"], [0, 1, 2])
            self.assertEqual(loss["ignore_index"], 3)
            self.assertEqual(loss["smooth_epsilon"], 1.0)
            self.assertEqual(loss["class_weighting"], "unweighted_1_1_1")

    def test_batch_size_and_divisibility_rules_v1(self):
        for cfg in [self.cfg_a_v1, self.cfg_b_v1]:
            bp = cfg["batch_size_policy"]
            self.assertEqual(bp["effective_batch_size"], 32)
            self.assertIn(bp["physical_microbatch_size"], [8, None])
            self.assertIn(bp["physical_microbatch_status"], ["CONFIRMED_LOCKED", "TO_BE_CONFIRMED_BY_WEEK1_MEMORY_SMOKE_TEST"])
            if bp["physical_microbatch_size"] is not None:
                self.assertEqual(bp["effective_batch_size"] % bp["physical_microbatch_size"], 0)
            for candidate in bp["candidate_microbatches_for_smoke_test"]:
                self.assertEqual(bp["effective_batch_size"] % candidate, 0, f"Microbatch {candidate} must divide effective 32")

    def test_execution_guards_block_formal_training_v1(self):
        # Model A: blocked because microbatch is unresolved
        self.assertFalse(self.cfg_a_v1["execution_guards"]["ready_for_formal_training"])
        self.assertTrue(self.cfg_a_v1["execution_guards"]["ready_for_week1_smoke_test"])
        self.assertIn("microbatch", self.cfg_a_v1["execution_guards"]["launch_block_reason"].lower())

        # Model B: blocked because microbatch is unresolved AND slope stats are absent
        self.assertFalse(self.cfg_b_v1["execution_guards"]["ready_for_formal_training"])
        self.assertTrue(self.cfg_b_v1["execution_guards"]["ready_for_week1_smoke_test"])
        reasons = " ".join(self.cfg_b_v1["execution_guards"]["launch_block_reasons"]).lower()
        self.assertIn("microbatch", reasons)
        self.assertIn("slope", reasons)

    def test_seed_propagation_and_retraining_rule_v1(self):
        for cfg in [self.cfg_a_v1, self.cfg_b_v1]:
            self.assertEqual(cfg["optimization_protocol"]["seeds"], [42, 43, 44])
            retrain = cfg["final_retraining_rule"]
            self.assertEqual(retrain["retraining_corpus"], "all_27_source_events")
            self.assertEqual(retrain["total_samples"], 21462)
            self.assertIn("median", retrain["retraining_epochs_formula"])

    def test_approved_optimizer_protocol_and_runtime_agreement(self):
        """Verify primary configs and runtime SourceTrainer instantiate the approved Adam protocol."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            for cfg_name, cfg in [("Model A", self.cfg_a_v1), ("Model B", self.cfg_b_v1)]:
                opt = cfg["optimization_protocol"]
                batch = cfg["batch_size_policy"]

                # 1. Config Field Assertions
                self.assertEqual(opt["optimizer"].lower(), "adam", f"{cfg_name} optimizer must be Adam")
                self.assertEqual(float(opt["learning_rate"]), 1e-3, f"{cfg_name} learning rate must be 1e-3")
                self.assertEqual(float(opt["weight_decay"]), 0.0, f"{cfg_name} weight decay must be 0.0")
                self.assertEqual(tuple(opt["betas"]), (0.9, 0.999), f"{cfg_name} betas must be (0.9, 0.999)")
                self.assertIsNone(opt.get("lr_scheduler"), f"{cfg_name} scheduler must be absent/None")
                self.assertTrue(opt["mixed_precision"], f"{cfg_name} mixed_precision must be enabled")
                self.assertEqual(batch["physical_microbatch_size"], 8)
                self.assertEqual(batch["gradient_accumulation_steps"], 4)
                self.assertEqual(batch["effective_batch_size"], 32)

                # 2. Runtime Instantiation Assertions
                dummy = DummyModel()
                trainer = SourceTrainer.from_config(
                    model=dummy,
                    config=cfg,
                    output_dir=temp_dir,
                    device="cpu",
                )

                # Must be Adam, NOT AdamW
                self.assertIsInstance(trainer.optimizer, torch.optim.Adam, f"{cfg_name} runtime optimizer must be Adam")
                self.assertNotIsInstance(trainer.optimizer, torch.optim.AdamW, f"{cfg_name} must NOT be AdamW")
                self.assertIsNone(trainer.lr_scheduler, f"{cfg_name} runtime scheduler must be None")

                # Verify optimizer parameter groups
                pg = trainer.optimizer.param_groups[0]
                self.assertEqual(pg["lr"], 1e-3)
                self.assertEqual(pg["weight_decay"], 0.0)
                self.assertEqual(pg["betas"], (0.9, 0.999))
                self.assertEqual(trainer.physical_microbatch_size, 8)
                self.assertEqual(trainer.accumulation_steps, 4)
                self.assertEqual(trainer.effective_batch_size, 32)

    def test_production_config_hashes_match_final_protocol_manifests(self):
        """Verify production config and split hashes match the final protocol artifacts."""
        def compute_sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        # 1. Production Config Hashes
        expected_hash_a = "9c9aefbefcc5800f870cc205207e0d4083252961a57ce402d2862984f581d1fc"
        expected_hash_b = "f75e38c7ea7ce14c636b17554774e3936aa5042340a1accef3f059e70a4b44ec"
        expected_split_hash = "7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3"
        expected_slope_stats_hash = "bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04"

        actual_hash_a = compute_sha256(self.config_a_v1_path)
        actual_hash_b = compute_sha256(self.config_b_v1_path)
        actual_split_hash = compute_sha256(self.split_v1_path)
        actual_slope_stats_hash = compute_sha256(self.repo_root / "artifacts" / "week1" / "source_train_slope_statistics.json")

        self.assertEqual(actual_hash_a, expected_hash_a, "Model A production config hash mismatch")
        self.assertEqual(actual_hash_b, expected_hash_b, "Model B production config hash mismatch")
        self.assertEqual(actual_split_hash, expected_split_hash, "Approved split file hash mismatch")
        self.assertEqual(actual_slope_stats_hash, expected_slope_stats_hash, "Slope statistics artifact hash mismatch")

        # 2. Agreement with final-protocol manifests in artifacts/week1/
        manifest_a_path = self.repo_root / "artifacts" / "week1" / "final_protocol_model_a_run_manifest.json"
        manifest_b_path = self.repo_root / "artifacts" / "week1" / "final_protocol_model_b_run_manifest.json"
        metrics_a_path = self.repo_root / "artifacts" / "week1" / "final_protocol_model_a_smoke_metrics.json"
        metrics_b_path = self.repo_root / "artifacts" / "week1" / "final_protocol_model_b_smoke_metrics.json"

        with open(manifest_a_path, "r") as f:
            man_a = json.load(f)
        with open(manifest_b_path, "r") as f:
            man_b = json.load(f)
        with open(metrics_a_path, "r") as f:
            met_a = json.load(f)
        with open(metrics_b_path, "r") as f:
            met_b = json.load(f)

        self.assertEqual(man_a["reproducibility"]["config_sha256"], actual_hash_a)
        self.assertEqual(man_b["reproducibility"]["config_sha256"], actual_hash_b)
        self.assertEqual(met_a["config_sha256"], actual_hash_a)
        self.assertEqual(met_b["config_sha256"], actual_hash_b)


if __name__ == "__main__":
    unittest.main()
