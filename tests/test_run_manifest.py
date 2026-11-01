"""
Unit tests for RunManifest and Research Integrity compliance helper.
"""

import json
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_manifest import RunManifest, DEFAULT_SEALED_TEST_EVENTS


class TestRunManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a dummy checkpoint file to test hashing
        self.dummy_ckpt = self.temp_path / "dummy_checkpoint.pt"
        self.dummy_ckpt.write_bytes(b"dummy_model_weights_bytes_12345")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_manifest_initialization_and_serialization(self):
        """Test valid initialization, JSON serialization, and required fields."""
        manifest = RunManifest(
            run_id="test_run_001",
            config_path="configs/source/snunet_vvvh_v1.json",
            checkpoint_path=self.dummy_ckpt,
            data_manifest="data/manifest.pkl",
            accessed_event_ids=[279, 437, 520],
            sealed_test_events=DEFAULT_SEALED_TEST_EVENTS,
            random_seed=42,
            labels_available_to_adaptation=False,
            labels_used_for_updates=False,
            is_partial_run=False,
            expected_sample_count=2642,
            processed_sample_count=2642,
            output_paths={"metrics": "outputs/metrics.csv"},
        )
        manifest.mark_completed("SUCCESS")

        out_json = self.temp_path / "run_manifest.json"
        saved_path = manifest.save(out_json)
        self.assertTrue(saved_path.is_file())

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check reproducibility fields
        rep = data["reproducibility"]
        self.assertEqual(rep["run_id"], "test_run_001")
        self.assertEqual(rep["random_seed"], 42)
        self.assertEqual(rep["completion_status"], "SUCCESS")
        self.assertEqual(rep["event_ids"], [279, 437, 520])
        self.assertIn("checkpoint_sha256", rep)
        self.assertNotEqual(rep["checkpoint_sha256"], "FILE_NOT_FOUND")

        # Check compliance fields
        comp = data["compliance"]
        self.assertEqual(comp["accessed_event_ids"], [279, 437, 520])
        self.assertFalse(comp["sealed_event_accessed"])
        self.assertFalse(comp["labels_used_for_updates"])
        self.assertFalse(comp["is_partial_run"])
        self.assertEqual(comp["expected_sample_count"], 2642)
        self.assertEqual(comp["processed_sample_count"], 2642)
        self.assertTrue(comp["sample_count_matched"])

    def test_rejects_sealed_test_event_overlap(self):
        """Test that attempting to access a sealed test event raises ValueError."""
        # Event 205 is in DEFAULT_SEALED_TEST_EVENTS
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="illegal_test_run",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=self.dummy_ckpt,
                data_manifest="data/manifest.pkl",
                accessed_event_ids=[279, 205],  # 205 is sealed!
                sealed_test_events=DEFAULT_SEALED_TEST_EVENTS,
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("205", str(ctx.exception))

    def test_rejects_labels_used_for_updates(self):
        """Test that declaring target labels were used for model updates raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="illegal_supervised_update_run",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=self.dummy_ckpt,
                data_manifest="data/manifest.pkl",
                accessed_event_ids=[279, 437],
                labels_used_for_updates=True,  # Prohibited!
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("target labels were used for model updates", str(ctx.exception))

    def test_records_partial_diagnostic_status(self):
        """Test that a partial diagnostic run is explicitly recorded as such."""
        manifest = RunManifest(
            run_id="diagnostic_run_002",
            config_path="configs/source/snunet_vvvh_v1.json",
            checkpoint_path=self.dummy_ckpt,
            data_manifest="data/manifest.pkl",
            accessed_event_ids=[279],
            is_partial_run=True,
            expected_sample_count=2381,
            processed_sample_count=100,
        )
        d = manifest.to_dict()
        comp = d["compliance"]
        self.assertTrue(comp["is_partial_run"])
        self.assertEqual(comp["expected_sample_count"], 2381)
        self.assertEqual(comp["processed_statistics_samples"] if "processed_statistics_samples" in comp else comp["processed_sample_count"], 100)
        self.assertIsNone(comp["sample_count_matched"])

    def test_rejects_quarantined_events_in_source_training(self):
        """Test that source_training run rejects quarantined TTA validation events."""
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="illegal_source_train_run",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=self.dummy_ckpt,
                data_manifest="data/manifest.pkl",
                accessed_event_ids=[118, 279],  # 279 is quarantined TTA-val!
                run_type="source_training",
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("279", str(ctx.exception))

    def test_rejects_test_manifest_in_source_training(self):
        """Test that source_training run rejects any test data manifest."""
        with self.assertRaises(ValueError) as ctx:
            RunManifest(
                run_id="illegal_source_manifest_run",
                config_path="configs/source/snunet_vvvh_v1.json",
                checkpoint_path=self.dummy_ckpt,
                data_manifest="data/KuroV2_grid_dict_test_0_100.gz",
                accessed_event_ids=[118],
                run_type="source_training",
            )
        self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))
        self.assertIn("Source training run declared test data manifest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
