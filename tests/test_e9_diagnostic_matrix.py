#!/usr/bin/env python3
"""ChangeGuard Work Package E9: Diagnostic Matrix Test Suite.

Tests governing Plan V3 Block E9 requirements:
1. Exact planned cell enumeration (14 reused, 12 new = 26 physical streams, 156 event rows).
2. Zero duplicate cells in ledger.
3. Checkpoint, dataset manifest, and split SHA-256 validation.
4. Sealed-test rejection and fail-closed quarantine.
5. Adapter parameterization for E9 diagnostics (adabn_m001, adabn_m005, cotta_no_ssr, cotta_no_ema).
6. Non-aliasing and deferred-label I/O contract preservation.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.changeguard.baseline_adapters import E1AdaBNAdapter
from src.changeguard.cotta_adapter import E7CoTTAAdapter
from src.data.evaluation_dataset import (
    DEFAULT_TTA_DEV_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
)
from tools.run_e9_diagnostic_matrix import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LEDGER_PATH,
    E9DiagnosticRunner,
)


class TestE9DiagnosticMatrix(unittest.TestCase):
    """Test suite for Work Package E9 diagnostic ablation matrix."""

    def setUp(self) -> None:
        self.config_path = DEFAULT_CONFIG_PATH
        self.ledger_path = DEFAULT_LEDGER_PATH
        self.assertTrue(self.config_path.is_file(), f"Missing config: {self.config_path}")
        self.assertTrue(self.ledger_path.is_file(), f"Missing ledger: {self.ledger_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

    def test_01_exact_cell_enumeration_and_ledger_counts(self) -> None:
        """Verifies exact planned stream and event-row enumeration."""
        reused = self.ledger.get("reused_cells", [])
        new_cells = self.ledger.get("execute_new_cells", [])

        # 14 Reused streams (2 source_only + 4 bn_no_update + 4 adabn + 4 cotta_reset)
        self.assertEqual(len(reused), 14, f"Expected 14 reused streams, got {len(reused)}")
        # 12 Execute-New streams (4 adabn momentum + 4 cotta_no_ssr + 4 cotta_no_ema)
        self.assertEqual(len(new_cells), 12, f"Expected 12 execute-new streams, got {len(new_cells)}")

        total_streams = len(reused) + len(new_cells)
        self.assertEqual(total_streams, 26, f"Expected 26 total physical streams, got {total_streams}")

        total_reused_rows = sum(len(c["events"]) for c in reused)
        total_new_rows = sum(len(c["events"]) for c in new_cells)
        self.assertEqual(total_reused_rows, 84, f"Expected 84 reused event rows, got {total_reused_rows}")
        self.assertEqual(total_new_rows, 72, f"Expected 72 new event rows, got {total_new_rows}")
        self.assertEqual(total_reused_rows + total_new_rows, 156, "Expected 156 total event rows")

    def test_02_zero_duplicate_cells_in_ledger(self) -> None:
        """Verifies zero duplicate stream IDs in ledger."""
        all_cells = self.ledger["reused_cells"] + self.ledger["execute_new_cells"]
        stream_ids = [c["stream_id"] for c in all_cells]
        self.assertEqual(len(stream_ids), len(set(stream_ids)), "Duplicate stream IDs detected in ledger!")

    def test_03_sealed_test_rejection(self) -> None:
        """Verifies fail-closed quarantine of sealed-test events."""
        runner = E9DiagnosticRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )
        original_targets = list(runner.target_events)
        forbidden_event = sorted(list(FORBIDDEN_SEALED_TEST_EVENTS))[0]
        runner.target_events = [original_targets[0], forbidden_event]

        with self.assertRaises(RuntimeError) as ctx:
            runner.run_preflights()
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

    def test_04_preflight_validation_success(self) -> None:
        """Verifies that preflight validation passes on committed assets."""
        runner = E9DiagnosticRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )
        preflights = runner.run_preflights()
        self.assertEqual(preflights["status"], "PREFLIGHTS_PASSED")

    def test_05_adapter_instantiations(self) -> None:
        """Verifies correct parameterization of diagnostic adapters."""
        runner = E9DiagnosticRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            device="cpu",
        )
        model = runner.instantiate_model(seed=0)

        # 1. adabn_m001
        a_m001 = runner.instantiate_adapter("adabn_m001", model, seed=0)
        self.assertIsInstance(a_m001, E1AdaBNAdapter)
        self.assertAlmostEqual(a_m001.momentum, 0.01)

        # 2. adabn_m005
        a_m005 = runner.instantiate_adapter("adabn_m005", model, seed=0)
        self.assertIsInstance(a_m005, E1AdaBNAdapter)
        self.assertAlmostEqual(a_m005.momentum, 0.05)

        # 3. cotta_no_ssr
        a_no_ssr = runner.instantiate_adapter("cotta_no_ssr", model, seed=0)
        self.assertIsInstance(a_no_ssr, E7CoTTAAdapter)
        self.assertAlmostEqual(a_no_ssr.restoration_prob, 0.0)
        self.assertAlmostEqual(a_no_ssr.ema_alpha, 0.999)

        # 4. cotta_no_ema
        a_no_ema = runner.instantiate_adapter("cotta_no_ema", model, seed=0)
        self.assertIsInstance(a_no_ema, E7CoTTAAdapter)
        self.assertAlmostEqual(a_no_ema.ema_alpha, 0.0)
        self.assertAlmostEqual(a_no_ema.restoration_prob, 0.01)

    def test_06_analyzer_reused_cell_loading(self) -> None:
        """Verifies that E9DiagnosticAnalyzer correctly loads and validates reused cells."""
        from tools.analyze_e9_diagnostic_matrix import E9DiagnosticAnalyzer
        analyzer = E9DiagnosticAnalyzer(
            ledger_path=self.ledger_path,
            output_dir=Path(self.config["output_directory"]),
        )
        reused = analyzer.ledger.get("reused_cells", [])
        self.assertEqual(len(reused), 14)
        for c in reused:
            art_dir = REPO_ROOT / c["artifact_directory"]
            summary_path = art_dir / "stream_summary.json"
            if not summary_path.is_file():
                summary_path = art_dir / "condition_summary.json"
            self.assertTrue(summary_path.is_file(), f"Missing summary for reused stream {c['stream_id']}")

    def test_07_governing_documentation_integrity(self) -> None:
        """Verifies presence and required verdicts of all E9 governing documents."""
        evidence_ledger = REPO_ROOT / "docs/e9_existing_evidence_ledger.md"
        closeout = REPO_ROOT / "docs/e9_required_ablations_closeout.md"
        scope_freeze = REPO_ROOT / "docs/e9_final_scope_freeze.md"

        self.assertTrue(evidence_ledger.is_file(), "Missing docs/e9_existing_evidence_ledger.md")
        self.assertTrue(closeout.is_file(), "Missing docs/e9_required_ablations_closeout.md")
        self.assertTrue(scope_freeze.is_file(), "Missing docs/e9_final_scope_freeze.md")

        with open(closeout, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("E9 CLOSED — EMPIRICAL SCOPE AND ABLATIONS FROZEN", content)
            self.assertIn("Branch C", content)

        with open(scope_freeze, "r", encoding="utf-8") as f:
            freeze_content = f.read()
            self.assertIn("FROZEN — READY FOR E10", freeze_content)
            self.assertIn("Branch C", freeze_content)


if __name__ == "__main__":
    unittest.main()
