"""Unit tests for risk feature schema enforcement and zero-GT compliance."""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.schema import validate_risk_feature_dict, check_forbidden_field_names


class TestRiskSchema(unittest.TestCase):
    """Verifies that risk records cannot contain ground truth or GT-derived metrics."""

    def setUp(self):
        self.valid_record = {
            "run_id": "tent_lr1e-4_seed0",
            "event_id": 279,
            "seed": 0,
            "batch_idx": 1,
            "sample_ids": ["sample_001", "sample_002"],
            "protocol_hash": "1f1dbd2a1313d9f4317cb86ee615bd604f6011badb7e91372b4aedcb1d512c69",
            "checkpoint_hash": "6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed",
            "mean_prediction_entropy": 0.45,
            "mean_max_confidence": 0.92,
            "bn_affine_drift_l2": 0.012,
            "null_pre2_pre2_activation_rate": 0.003,
            "disappear_fraction_source_erased": 0.05,
            "predeclared_change_risk_score": 0.0265,
        }

    def test_valid_record_passes(self):
        """Clean label-free record must pass validation without error."""
        validate_risk_feature_dict(self.valid_record)

    def test_missing_identifier_raises(self):
        """Missing any required identifier must raise ValueError."""
        record = dict(self.valid_record)
        del record["protocol_hash"]
        with self.assertRaises(ValueError) as ctx:
            validate_risk_feature_dict(record)
        self.assertIn("SCHEMA VIOLATION", str(ctx.exception))
        self.assertIn("protocol_hash", str(ctx.exception))

    def test_forbidden_gt_fields_raise(self):
        """Presence of ground-truth or evaluation metric fields must raise ValueError."""
        forbidden_test_cases = [
            {"label": [0, 1, 2]},
            {"labels": [0, 1, 2]},
            {"gt_mask": [0, 1]},
            {"ground_truth_flood": 100},
            {"tp": 50},
            {"fp": 10},
            {"fn": 5},
            {"tn": 1000},
            {"f1": 0.85},
            {"batch_f1": 0.85},
            {"iou": 0.72},
            {"precision": 0.80},
            {"recall": 0.90},
            {"gt_flood_area": 500},
            {"is_harmful_update": True},
            {"beneficial_step": False},
        ]
        for bad_entry in forbidden_test_cases:
            with self.subTest(bad_entry=bad_entry):
                record = dict(self.valid_record)
                record.update(bad_entry)
                with self.assertRaises(ValueError) as ctx:
                    validate_risk_feature_dict(record)
                self.assertIn("RESEARCH INTEGRITY VIOLATION", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

