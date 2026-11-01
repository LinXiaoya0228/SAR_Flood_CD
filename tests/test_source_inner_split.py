"""Unit tests for the proposed and approved 22/5 source-inner event splits."""

import hashlib
import json
import unittest
from pathlib import Path


class TestSourceInnerSplit(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.split_proposed_path = self.repo_root / "configs" / "splits" / "kuro_source_inner_split_proposed.json"
        self.split_v1_path = self.repo_root / "configs" / "splits" / "kuro_source_inner_split_v1.json"
        self.kuro_split_path = self.repo_root / "configs" / "kuro_event_split.json"
        self.stats_csv_path = self.repo_root / "outputs" / "week0" / "source_event_statistics.csv"

        with open(self.split_proposed_path, "r") as f:
            self.proposed = json.load(f)

        with open(self.split_v1_path, "r") as f:
            self.split_v1 = json.load(f)

        with open(self.kuro_split_path, "r") as f:
            self.kuro_split = json.load(f)

    def test_proposed_file_structure_and_status(self):
        self.assertEqual(self.proposed["$schema_version"], "1.0.0")
        self.assertEqual(self.proposed["status"], "AWAITING_USER_APPROVAL")
        self.assertIn("selection_algorithm", self.proposed)
        self.assertIn("proposed_primary_split", self.proposed)

    def test_v1_file_structure_and_approved_status(self):
        self.assertEqual(self.split_v1["$schema_version"], "1.0.0")
        self.assertEqual(self.split_v1["status"], "APPROVED_FROZEN")
        self.assertEqual(self.split_v1["approval_date"], "2026-10-06")
        self.assertIn("source_inner_train_events", self.split_v1)
        self.assertIn("source_inner_validation_events", self.split_v1)
        self.assertIn("sample_counts", self.split_v1)
        self.assertIn("class_distribution_summary", self.split_v1)

    def test_v1_exact_approved_events(self):
        expected_val = [267, 275, 497, 518, 1111005]
        expected_train = [
            118, 130, 147, 174, 273, 324, 417, 421, 427,
            470, 492, 496, 498, 502, 554, 555, 567,
            1111004, 1111006, 1111009, 1111010, 1111011
        ]
        self.assertEqual(sorted(self.split_v1["source_inner_validation_events"]), sorted(expected_val))
        self.assertEqual(sorted(self.split_v1["source_inner_train_events"]), sorted(expected_train))

    def test_event_counts_and_disjointness_both_splits(self):
        for split_data, name in [
            (self.proposed["proposed_primary_split"], "Proposed"),
            (self.split_v1, "V1_Approved")
        ]:
            train_events = split_data["source_inner_train_events"]
            val_events = split_data["source_inner_validation_events"]

            self.assertEqual(len(train_events), 22, f"{name}: Must have 22 train events")
            self.assertEqual(len(val_events), 5, f"{name}: Must have 5 val events")
            self.assertEqual(len(set(train_events)), 22, f"{name}: Duplicate in train events")
            self.assertEqual(len(set(val_events)), 5, f"{name}: Duplicate in val events")

            set_train = set(train_events)
            set_val = set(val_events)
            self.assertTrue(set_train.isdisjoint(set_val), f"{name}: Inner train and inner val must be disjoint!")

    def test_union_equals_27_source_events_both_splits(self):
        expected_27 = set(self.kuro_split["effective_splits"]["train"])
        for split_data, name in [
            (self.proposed["proposed_primary_split"], "Proposed"),
            (self.split_v1, "V1_Approved")
        ]:
            combined = set(split_data["source_inner_train_events"]).union(set(split_data["source_inner_validation_events"]))
            self.assertEqual(combined, expected_27, f"{name}: Union of 22/5 split must equal configured 27 source events exactly!")

    def test_no_tta_validation_or_test_leakage_both_splits(self):
        tta_val = set(self.kuro_split["effective_splits"]["validation"])
        sealed_test = set(self.kuro_split["effective_splits"]["test"])

        for split_data, name in [
            (self.proposed["proposed_primary_split"], "Proposed"),
            (self.split_v1, "V1_Approved")
        ]:
            combined = set(split_data["source_inner_train_events"]).union(set(split_data["source_inner_validation_events"]))
            self.assertTrue(combined.isdisjoint(tta_val), f"{name}: Security violation: Source split overlaps with TTA validation events!")
            self.assertTrue(combined.isdisjoint(sealed_test), f"{name}: Security violation: Source split overlaps with sealed test events!")

    def test_v1_sample_counts_match_ground_truth(self):
        counts = self.split_v1["sample_counts"]
        self.assertEqual(counts["total_source_samples"], 21462)
        self.assertEqual(counts["source_inner_train_sample_count"], 19178)
        self.assertEqual(counts["source_inner_validation_sample_count"], 2284)
        self.assertEqual(counts["source_inner_train_sample_count"] + counts["source_inner_validation_sample_count"], 21462)

    def test_source_statistics_hash_integrity(self):
        self.assertTrue(self.stats_csv_path.exists(), "Source statistics CSV must exist!")
        with open(self.stats_csv_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        recorded_hash_prop = self.proposed["selection_algorithm"]["source_statistics_csv_sha256"]
        recorded_hash_v1 = self.split_v1["selection_algorithm"]["source_statistics_csv_sha256"]
        self.assertEqual(actual_hash, recorded_hash_prop)
        self.assertEqual(actual_hash, recorded_hash_v1)


if __name__ == "__main__":
    unittest.main()
