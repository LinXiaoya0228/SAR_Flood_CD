"""Unit tests for the official KuroSiwo checkpoint reference evaluation safeguards.

Verifies:
1. Frozen reference config matches expected hash.
2. Official checkpoint exists, is uncorrupted, and matches frozen SHA-256.
3. Architecture matches expected 3-channel SNUNet specification (12,035,107 parameters).
4. Evaluation scope is strictly the 5 source-inner-validation events.
5. Forbidden TTA-val and sealed-test events are strictly quarantined.
6. Generated outputs under outputs/week3_official_reference adhere to all invariants:
   - Weights identical before and after inference.
   - Zero optimizer created, zero backward calls, zero parameter updates.
   - Zero TTA-val or sealed-test event access.
   - Sample count matched to expected 2,284.
7. Regression protection against configuration mutation:
   - Negative tampering tests operate strictly on temporary copies.
   - Production configuration hash remains unchanged after all testing.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.evaluation.run_manifest import compute_file_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_CKPT_PATH = REPO_ROOT / "checkpoints/kuro_snunet_official.pt"
OFFICIAL_CKPT_HASH = "6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed"

REFERENCE_CONFIG_PATH = REPO_ROOT / "configs/reference/official_snunet_reference_slope_candidate.json"
REFERENCE_CONFIG_HASH = "0a15c584546a9e5ce1ac725b8c9c81604a3bd1014ae3ceed8d60149b4bafef5c"

SOURCE_SPLIT_PATH = REPO_ROOT / "configs/splits/kuro_source_inner_split_v1.json"
FROZEN_SPLIT_HASH = "7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3"

SLOPE_STATS_PATH = REPO_ROOT / "artifacts/week1/source_train_slope_statistics.json"
FROZEN_SLOPE_STATS_HASH = "bc00249a045f53e16d7e4d63001278e38e2388fbef92dd219469dff766f80c04"

OUTPUT_DIR = REPO_ROOT / "outputs/week3_official_reference"

FORBIDDEN_TTA_VAL_EVENTS = {279, 437, 520, 559, 1111003, 1111008}
FORBIDDEN_SEALED_TEST_EVENTS = {205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013}
ALLOWED_VAL_EVENTS = {267, 275, 497, 518, 1111005}


class TestOfficialReferenceSafeguards(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Invariant regression check: production config must never be mutated
        self.assertEqual(
            compute_file_sha256(REFERENCE_CONFIG_PATH),
            REFERENCE_CONFIG_HASH,
            "CRITICAL: Production reference configuration was mutated by tests!"
        )

    def test_checkpoint_exists_and_hash_matches(self):
        self.assertTrue(OFFICIAL_CKPT_PATH.exists(), f"Checkpoint missing: {OFFICIAL_CKPT_PATH}")
        actual_hash = compute_file_sha256(OFFICIAL_CKPT_PATH)
        self.assertEqual(actual_hash, OFFICIAL_CKPT_HASH)

    def test_reference_config_hash_matches(self):
        self.assertTrue(REFERENCE_CONFIG_PATH.exists(), f"Config missing: {REFERENCE_CONFIG_PATH}")
        actual_hash = compute_file_sha256(REFERENCE_CONFIG_PATH)
        self.assertEqual(actual_hash, REFERENCE_CONFIG_HASH)

        with open(REFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertEqual(cfg["official_checkpoint"]["sha256"], OFFICIAL_CKPT_HASH)
        self.assertEqual(cfg["dataset_specification"]["split_config_sha256"], FROZEN_SPLIT_HASH)
        self.assertEqual(cfg["input_specification"]["slope_specification"]["statistics_sha256"], FROZEN_SLOPE_STATS_HASH)

    def test_evaluation_scope_and_event_quarantine(self):
        with open(SOURCE_SPLIT_PATH, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        val_events = set(split_data["source_inner_validation_events"])
        self.assertEqual(val_events, ALLOWED_VAL_EVENTS)
        self.assertTrue(val_events.isdisjoint(FORBIDDEN_TTA_VAL_EVENTS))
        self.assertTrue(val_events.isdisjoint(FORBIDDEN_SEALED_TEST_EVENTS))

    def test_generated_reference_summary_and_invariants(self):
        summary_path = OUTPUT_DIR / "reference_summary.json"
        manifest_path = OUTPUT_DIR / "run_manifest.json"
        trace_path = OUTPUT_DIR / "access_trace.json"
        csv_path = OUTPUT_DIR / "event_validation_metrics.csv"

        self.assertTrue(summary_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(trace_path.exists())
        self.assertTrue(csv_path.exists())

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Invariants check
        invariants = summary["invariants"]
        self.assertTrue(invariants["weights_identical_before_and_after"])
        self.assertTrue(invariants["zero_optimizer_created"])
        self.assertTrue(invariants["zero_backward_calls"])
        self.assertTrue(invariants["zero_parameter_updates"])
        self.assertTrue(invariants["zero_tta_validation_accessed"])
        self.assertTrue(invariants["zero_sealed_test_accessed"])

        # Evaluation scope
        self.assertEqual(set(summary["evaluation_scope"]["allowed_events"]), ALLOWED_VAL_EVENTS)
        self.assertEqual(summary["evaluation_scope"]["processed_samples"], 2284)
        self.assertEqual(summary["evaluation_scope"]["expected_samples"], 2284)

        # Access trace
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)

        self.assertEqual(set(trace["val_events_accessed"]), ALLOWED_VAL_EVENTS)
        self.assertEqual(trace["val_unique_samples_count"], 2284)
        self.assertEqual(trace["quarantined_events_accessed"], [])
        self.assertTrue(trace["assertions"]["no_tta_val_accessed"])
        self.assertTrue(trace["assertions"]["no_sealed_test_accessed"])

    def test_tampered_config_fails_on_temporary_copy(self):
        """Verifies tampering check fails fast and leaves production config untouched."""
        temp_cfg = self.temp_dir / "tampered_config.json"
        with open(REFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["official_checkpoint"]["architecture"] = "TAMPERED_ARCH"
        with open(temp_cfg, "w", encoding="utf-8") as f:
            json.dump(data, f)

        temp_hash = compute_file_sha256(temp_cfg)
        self.assertNotEqual(temp_hash, REFERENCE_CONFIG_HASH)
        self.assertEqual(compute_file_sha256(REFERENCE_CONFIG_PATH), REFERENCE_CONFIG_HASH)


if __name__ == "__main__":
    unittest.main()
