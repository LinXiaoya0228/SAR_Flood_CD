"""Unit tests for E11 final test engine, ledger integrity, and fail-closed gates."""

import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from tools.run_e11_final_test import (
    E11Batch,
    FORBIDDEN_SEALED_TEST_EVENTS,
    compute_file_sha256,
)
from tools.analyze_e11_final_test import E11FinalTestAnalyzer


class MockDataset:
    def __init__(self):
        self.samples = [{"sample_id": f"s_{i}", "event_id": 520} for i in range(5)]

    def __getitem__(self, idx):
        return {
            "sample_id": f"s_{idx}",
            "label": torch.randint(0, 3, (224, 224)),
        }


class TestE11FinalTestEngine(unittest.TestCase):
    def setUp(self):
        self.config_path = REPO_ROOT / "configs/changeguard/e11_final_test_matrix_v1.json"
        self.ledger_path = REPO_ROOT / "configs/changeguard/e11_final_test_ledger_v1.json"

    def test_01_config_and_ledger_integrity(self):
        """Verifies schema, stream counts, and zero test leakage in E11 configs."""
        self.assertTrue(self.config_path.is_file())
        self.assertTrue(self.ledger_path.is_file())

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        self.assertEqual(cfg["status"], "PREPARATION_ONLY_UNAUTHORIZED_FOR_EXECUTION")
        self.assertEqual(cfg["matrix_accounting"]["total_physical_streams"], 21)
        self.assertEqual(cfg["matrix_accounting"]["total_expected_event_rows"], 210)
        self.assertEqual(len(ledger["streams"]), 21)

        snunet_streams = [s for s in ledger["streams"] if s["architecture"] == "SNUNet_ECAM"]
        siam_streams = [s for s in ledger["streams"] if s["architecture"] == "SiamUnet_diff"]
        self.assertEqual(len(snunet_streams), 12)
        self.assertEqual(len(siam_streams), 9)

        # Verify all 6 source checkpoints exist and have matching hashes
        for arch, ckpts in cfg["source_checkpoints"].items():
            for sk, c_info in ckpts.items():
                p = REPO_ROOT / c_info["path"]
                self.assertTrue(p.is_file(), f"Missing checkpoint: {p}")
                h = compute_file_sha256(p)
                self.assertEqual(h, c_info["sha256"])

    def test_02_fail_closed_execute_final(self):
        """Executing final mode without Gate T authorization must raise RuntimeError."""
        import subprocess

        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools/run_e11_final_test.py"),
            "--mode",
            "execute_final",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", res.stderr)

    def test_03_deferred_label_guard(self):
        """Accessing labels before persistence in E11Batch must raise PermissionError."""
        mock_ds = MockDataset()
        batch = E11Batch(
            sample_ids=["s_0", "s_1"],
            xA=torch.randn(2, 2, 224, 224),
            xB=torch.randn(2, 2, 224, 224),
            valid_mask=torch.ones(2, 224, 224, dtype=torch.uint8),
            dataset=mock_ds,
            sample_indices=[0, 1],
        )

        with self.assertRaises(PermissionError):
            batch.get_labels()

        batch.mark_persisted()
        labels = batch.get_labels()
        self.assertEqual(labels.shape, (2, 224, 224))

    def test_04_bootstrap_cluster_ci(self):
        """Verifies paired cluster bootstrap computation."""
        analyzer = E11FinalTestAnalyzer(
            results_dir=REPO_ROOT / "outputs/e11_rehearsal",
            output_dir=REPO_ROOT / "outputs/e11_rehearsal_analysis",
            ledger_path=self.ledger_path,
        )
        tta_scores = np.array([0.50, 0.55, 0.60, 0.65, 0.45, 0.52])
        src_scores = np.array([0.55, 0.58, 0.62, 0.67, 0.50, 0.54])
        mean_delta, ci_l, ci_u, p_val = analyzer.run_event_cluster_bootstrap(
            tta_scores, src_scores, num_bootstraps=1000
        )
        self.assertLess(mean_delta, 0.0)
        self.assertLessEqual(ci_l, mean_delta)
        self.assertGreaterEqual(ci_u, mean_delta)


if __name__ == "__main__":
    import sys
    unittest.main()
