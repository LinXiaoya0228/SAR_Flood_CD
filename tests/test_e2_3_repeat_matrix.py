"""Focused Unit Tests for ChangeGuard Plan V3 Work Package E2.3 Repeat Matrix."""

import hashlib
import json
from pathlib import Path
import unittest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    EXPECTED_TTA_DEV_COUNTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)
from tools.run_e2_3_repeat_matrix import E23RepeatMatrixRunner, compute_file_sha256


class TestE23RepeatMatrix(unittest.TestCase):
    """Test suite for Work Package E2.3 Repeat Matrix."""

    def setUp(self):
        self.cfg_p = REPO_ROOT / "configs" / "changeguard" / "e2_3_repeat_matrix_v1.json"
        self.assertTrue(self.cfg_p.is_file(), f"Config file missing: {self.cfg_p}")
        with open(self.cfg_p, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def test_01_config_integrity(self):
        """Verifies that E2.3 config adheres to Plan V3 repeat specification."""
        self.assertEqual(self.cfg["work_package"], "E2.3")
        spec = self.cfg["screening_specification"]
        self.assertEqual(spec["checkpoint_seed"], 1)
        self.assertEqual(spec["stream_order"], "order_b")
        self.assertEqual(spec["reset_mode"], "reset_per_event")
        self.assertEqual(spec["batch_size"], 16)
        self.assertEqual(spec["within_event_ordering"], "deterministic_sorted_sample_id_descending")

        cells = spec["repeat_cells"]
        self.assertEqual(len(cells), 2)
        cell_ids = [c["cell_id"] for c in cells]
        self.assertEqual(cell_ids, ["E2_REP_CONF_90", "E2_REP_CONF_NONE"])

        self.assertEqual(cells[0]["threshold"], 0.9)
        self.assertEqual(cells[0]["role"], "RETAINED_RECIPE_REPEAT")
        self.assertEqual(cells[1]["threshold"], 0.0)
        self.assertEqual(cells[1]["role"], "DECISIVE_COMPONENT_CONTRAST")

    def test_02_seed1_checkpoint_cryptographic_hash(self):
        """Verifies that the Seed 1 checkpoint exists on disk and matches expected SHA-256."""
        seed_info = self.cfg["source_checkpoints"]["seeds"]["1"]
        ck_path = REPO_ROOT / seed_info["checkpoint_path"]
        expected_hash = seed_info["expected_sha256"]

        self.assertTrue(ck_path.is_file(), f"Seed 1 checkpoint missing: {ck_path}")
        actual_hash = compute_file_sha256(ck_path)
        self.assertEqual(actual_hash, expected_hash, f"Hash mismatch: {actual_hash} != {expected_hash}")

    def test_03_sealed_test_quarantine(self):
        """Verifies that no sealed test events are present in E2.3 target events."""
        event_seq = self.cfg["screening_specification"]["event_sequence"]
        forbidden = set(FORBIDDEN_SEALED_TEST_EVENTS)
        overlap = set(event_seq).intersection(forbidden)
        self.assertEqual(len(overlap), 0, f"Sealed test quarantine breached: {overlap}")
        self.assertEqual(set(event_seq), set(DEFAULT_TTA_DEV_EVENTS), "Target events must match Dev 6")

    def test_04_order_b_event_sequence(self):
        """Verifies that order_b has reverse chronological event order."""
        event_seq = self.cfg["screening_specification"]["event_sequence"]
        self.assertEqual(event_seq, [1111008, 1111003, 559, 520, 437, 279])

    def test_05_runner_smoke_initialization(self):
        """Verifies runner instantiates in smoke mode cleanly."""
        runner = E23RepeatMatrixRunner(
            config_path=self.cfg_p,
            device_str="cpu",
            smoke_mode=True,
        )
        self.assertTrue(runner.smoke_mode)
        self.assertEqual(runner.target_events, [520])
        self.assertEqual(runner.checkpoint_seed, 1)
        self.assertEqual(runner.stream_order, "order_b")


if __name__ == "__main__":
    unittest.main()

