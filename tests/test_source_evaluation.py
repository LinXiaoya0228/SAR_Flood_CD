"""Synthetic Unit Tests for Source Model Evaluation Engine.

Protocol: Week 7A Controlled Source-Only Baseline Evaluation
Constraints:
- Pure synthetic data (mock tensors / in-memory records).
- Zero GPU requirements (CPU only).
- Zero disk-raster loading from KuroSiwo.
- Verifies double-masking, confusion metrics, macro vs pooled math, quarantine enforcement,
  sample counting/duplicate checks, and model parameter immutability.
"""

import copy
import hashlib
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.evaluation_dataset import (
    KuroEvaluationDataset,
    FORBIDDEN_SEALED_TEST_EVENTS,
    DEFAULT_TTA_DEV_EVENTS,
)
from src.evaluation.metrics import (
    compute_confusion_counts,
    compute_flood_metrics,
    compute_macro_metrics,
    compute_pooled_metrics,
)
from src.evaluation.evaluator import (
    SourceModelEvaluator,
    compute_model_state_hash,
    aggregate_three_seed_metrics,
)


class MockModel(nn.Module):
    """Simple deterministic mock model for testing evaluation logic on CPU."""

    def __init__(self, out_classes: int = 3, pred_class: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(4, out_classes, kernel_size=1, bias=True)
        # Initialize weights such that pred_class always has the highest logit
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.bias[pred_class] = 10.0

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        return self.conv(x)


class MockMutatingModel(nn.Module):
    """Mock model that illegally mutates its parameters during forward pass."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        # Illegally mutate parameter in-place during inference
        self.weight.data += 1.0
        B, _, H, W = xA.shape
        logits = torch.zeros(B, 3, H, W)
        logits[:, 0, :, :] = 10.0
        return logits


class TestSourceEvaluation(unittest.TestCase):
    """Comprehensive test suite for Week 7A evaluation components."""

    def setUp(self):
        self.H, self.W = 16, 16

    def test_double_masking_rule(self):
        """Verifies valid = (valid_mask == 1) & (label != 3)."""
        # Create ground truth label: 0, 1, 2, 3
        # valid_mask: 0 and 1
        labels = np.array([
            [0, 1],
            [2, 3],
        ], dtype=np.int64)
        valid_masks = np.array([
            [1, 1],
            [0, 1],
        ], dtype=np.uint8)

        # Expected valid pixels:
        # (0, 0): label 0, vm 1 -> VALID
        # (0, 1): label 1, vm 1 -> VALID
        # (1, 0): label 2, vm 0 -> INVALID (vm=0)
        # (1, 1): label 3, vm 1 -> INVALID (label=3)
        expected_valid = np.array([
            [True, True],
            [False, False],
        ])

        actual_valid = (valid_masks == 1) & (labels != 3)
        np.testing.assert_array_equal(actual_valid, expected_valid)

    def test_known_confusion_and_metrics_math(self):
        """Tests exact precision, recall, F1, and IoU calculation against hand-verified counts."""
        # 4 pixels:
        # P0: GT=2, Pred=2 -> TP
        # P1: GT=0, Pred=2 -> FP
        # P2: GT=2, Pred=0 -> FN
        # P3: GT=0, Pred=0 -> TN
        preds = np.array([2, 2, 0, 0], dtype=np.int64)
        targets = np.array([2, 0, 2, 0], dtype=np.int64)
        valid = np.array([True, True, True, True], dtype=bool)

        counts = compute_confusion_counts(preds, targets, valid, flood_class=2)
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 1)
        self.assertEqual(counts["fn"], 1)
        self.assertEqual(counts["tn"], 1)

        m = compute_flood_metrics(counts)
        # Precision = 1 / (1 + 1) = 0.5
        # Recall    = 1 / (1 + 1) = 0.5
        # F1        = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        # IoU       = 1 / (1 + 1 + 1) = 1/3 = 0.3333333333
        self.assertAlmostEqual(m["precision"], 0.5, places=5)
        self.assertAlmostEqual(m["recall"], 0.5, places=5)
        self.assertAlmostEqual(m["f1"], 0.5, places=5)
        self.assertAlmostEqual(m["iou"], 1.0 / 3.0, places=5)

    def test_macro_vs_pooled_metrics_math(self):
        """Verifies unweighted macro average vs pooled confusion summation."""
        # Event A: 10 samples, high performance (F1 = 0.80)
        # Event B: 1 sample, low performance (F1 = 0.20)
        # Macro F1 should be (0.80 + 0.20) / 2 = 0.50
        # Pooled F1 should be weighted by pixel counts
        event_a_conf = {"tp": 80, "fp": 20, "fn": 20, "tn": 880}
        event_b_conf = {"tp": 2, "fp": 8, "fn": 8, "tn": 82}

        event_a_metrics = compute_flood_metrics(event_a_conf)
        event_b_metrics = compute_flood_metrics(event_b_conf)
        self.assertAlmostEqual(event_a_metrics["f1"], 0.80, places=5)
        self.assertAlmostEqual(event_b_metrics["f1"], 0.20, places=5)

        event_rows = [
            {"event_id": 100, **event_a_conf, **event_a_metrics},
            {"event_id": 200, **event_b_conf, **event_b_metrics},
        ]

        macro = compute_macro_metrics(event_rows)
        self.assertAlmostEqual(macro["macro_f1"], 0.50, places=5)

        pooled = compute_pooled_metrics(event_rows)
        # Pooled: TP = 82, FP = 28, FN = 28
        # Pooled F1 = (2 * 82) / (2 * 82 + 28 + 28) = 164 / 220 = 0.745454...
        expected_pooled_f1 = (2 * 82) / (2 * 82 + 28 + 28)
        self.assertAlmostEqual(pooled["pooled_f1"], expected_pooled_f1, places=5)
        self.assertNotEqual(macro["macro_f1"], pooled["pooled_f1"])

    def test_sealed_test_events_quarantine_enforcement(self):
        """Ensures loading or evaluating sealed test events raises immediate RuntimeError."""
        for sealed_id in FORBIDDEN_SEALED_TEST_EVENTS:
            with self.subTest(sealed_id=sealed_id):
                with self.assertRaises(RuntimeError) as ctx:
                    KuroEvaluationDataset(allowed_events=[sealed_id, 279])
                self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

    def test_mock_dataset_and_evaluator_execution(self):
        """Runs end-to-end evaluation with synthetic records on CPU."""
        records = [
            {
                "sample_id": "patch_279_001",
                "event_id": 279,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
            {
                "sample_id": "patch_279_002",
                "event_id": 279,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.zeros((self.H, self.W), dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
            {
                "sample_id": "patch_437_001",
                "event_id": 437,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
        ]

        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[279, 437])
        self.assertEqual(len(dataset), 3)

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = MockModel(out_classes=3, pred_class=2)  # Always predicts class 2 (Flood)
        evaluator = SourceModelEvaluator(model, device="cpu")

        res = evaluator.evaluate(loader, expected_counts={279: 2, 437: 1})
        self.assertTrue(res["sample_counts_verified"])
        self.assertTrue(res["model_immutability_verified"])
        self.assertTrue(res["quarantine_sealed_test_zero_leakage"])
        self.assertEqual(len(res["event_metrics"]), 2)

    def test_model_immutability_verification(self):
        """Verifies evaluator halts immediately if model weights change during inference."""
        records = [
            {
                "sample_id": "patch_279_001",
                "event_id": 279,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.zeros((self.H, self.W), dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            }
        ]
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[279])
        loader = DataLoader(dataset, batch_size=1)

        mutating_model = MockMutatingModel()
        evaluator = SourceModelEvaluator(mutating_model, device="cpu")

        with self.assertRaises(RuntimeError) as ctx:
            evaluator.evaluate(loader)
        self.assertIn("Model parameters or buffers changed during evaluation", str(ctx.exception))

    def test_three_seed_aggregation_math(self):
        """Verifies arithmetic mean and sample standard deviation (ddof=1) aggregation."""
        # Create 3 synthetic seed result dictionaries
        seed0_summary = {
            "seed": 0,
            "equal_event_macro": {"macro_f1": 0.60, "macro_iou": 0.50},
            "pooled_metrics": {"pooled_f1": 0.62},
            "event_metrics": [
                {"event_id": 279, "f1": 0.70, "iou": 0.60, "precision": 0.75, "recall": 0.65},
                {"event_id": 437, "f1": 0.50, "iou": 0.40, "precision": 0.55, "recall": 0.45},
            ],
        }
        seed1_summary = {
            "seed": 1,
            "equal_event_macro": {"macro_f1": 0.62, "macro_iou": 0.52},
            "pooled_metrics": {"pooled_f1": 0.64},
            "event_metrics": [
                {"event_id": 279, "f1": 0.72, "iou": 0.62, "precision": 0.77, "recall": 0.67},
                {"event_id": 437, "f1": 0.52, "iou": 0.42, "precision": 0.57, "recall": 0.47},
            ],
        }
        seed2_summary = {
            "seed": 2,
            "equal_event_macro": {"macro_f1": 0.58, "macro_iou": 0.48},
            "pooled_metrics": {"pooled_f1": 0.60},
            "event_metrics": [
                {"event_id": 279, "f1": 0.68, "iou": 0.58, "precision": 0.73, "recall": 0.63},
                {"event_id": 437, "f1": 0.48, "iou": 0.38, "precision": 0.53, "recall": 0.43},
            ],
        }

        agg = aggregate_three_seed_metrics([seed0_summary, seed1_summary, seed2_summary])

        # Macro F1 values: [0.60, 0.62, 0.58]
        # Mean = 0.60
        # Std (ddof=1) = sqrt(((0)^2 + (0.02)^2 + (-0.02)^2) / 2) = sqrt(0.0008 / 2) = sqrt(0.0004) = 0.02
        macro_f1_agg = agg["macro_summary"]["macro_f1"]
        self.assertAlmostEqual(macro_f1_agg["mean"], 0.60, places=5)
        self.assertAlmostEqual(macro_f1_agg["std"], 0.02, places=5)
        self.assertEqual(macro_f1_agg["formatted"], "0.6000 +/- 0.0200")

    def test_zero_denominators_and_invalid_only(self):
        """Verifies zero-denominator handling and invalid-only inputs."""
        # Case 1: Both GT and pred empty (all True Negatives)
        c1 = {"tp": 0, "fp": 0, "fn": 0, "tn": 100}
        m1 = compute_flood_metrics(c1)
        self.assertEqual(m1["precision"], 1.0)
        self.assertEqual(m1["recall"], 1.0)
        self.assertEqual(m1["f1"], 1.0)
        self.assertEqual(m1["iou"], 1.0)

        # Case 2: Zero predicted flood when GT flood exists (complete missed detection)
        c2 = {"tp": 0, "fp": 0, "fn": 10, "tn": 90}
        m2 = compute_flood_metrics(c2)
        self.assertEqual(m2["precision"], 0.0)
        self.assertEqual(m2["recall"], 0.0)
        self.assertEqual(m2["f1"], 0.0)
        self.assertEqual(m2["iou"], 0.0)

        # Case 3: Zero GT flood when predicted flood exists (all false alarms)
        c3 = {"tp": 0, "fp": 10, "fn": 0, "tn": 90}
        m3 = compute_flood_metrics(c3)
        self.assertEqual(m3["precision"], 0.0)
        self.assertEqual(m3["recall"], 0.0)
        self.assertEqual(m3["f1"], 0.0)
        self.assertEqual(m3["iou"], 0.0)

        # Case 4: Invalid-only input (0 valid pixels) -> technically unevaluable, not 1.0
        c4 = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        m4 = compute_flood_metrics(c4)
        self.assertIsNone(m4["precision"])
        self.assertIsNone(m4["recall"])
        self.assertIsNone(m4["f1"])
        self.assertIsNone(m4["iou"])

        # Case 5: Counts with explicit valid_pixels == 0 key
        c5 = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid_pixels": 0}
        m5 = compute_flood_metrics(c5)
        self.assertIsNone(m5["precision"])
        self.assertIsNone(m5["recall"])
        self.assertIsNone(m5["f1"])
        self.assertIsNone(m5["iou"])

    def test_collapse_audit_reporting(self):
        """Verifies whole-run and event-level collapse metrics in evaluator."""
        records = [
            {
                "sample_id": "patch_279_001",
                "event_id": 279,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
            {
                "sample_id": "patch_437_001",
                "event_id": 437,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
        ]
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[279, 437])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        bg_model = MockModel(out_classes=3, pred_class=0)
        evaluator = SourceModelEvaluator(bg_model, device="cpu")
        res = evaluator.evaluate(loader, expected_counts={279: 1, 437: 1})

        self.assertIn("collapse_audit", res)
        ca = res["collapse_audit"]
        self.assertTrue(ca["run_all_background"])
        self.assertTrue(ca["run_zero_flood_predicted"])
        self.assertEqual(set(ca["all_background_events"]), {279, 437})
        self.assertEqual(set(ca["zero_flood_on_flood_positive_events"]), {279, 437})

    def test_dataset_defense_in_depth_quarantine_rejection(self):
        """Verifies that attempting to read sealed-test rasters in __getitem__ raises RuntimeError."""
        dataset = KuroEvaluationDataset.from_mock_records([], allowed_events=[279])
        dataset.samples = [
            {"sample_id": "test_sealed", "event_id": 205, "relative_path": "205/01/sample"},
            {"sample_id": "test_unallowed", "event_id": 118, "relative_path": "118/01/sample"},
        ]
        with self.assertRaises(RuntimeError) as ctx:
            _ = dataset[0]
        self.assertIn("CRITICAL RESEARCH INTEGRITY VIOLATION", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            _ = dataset[1]
        self.assertIn("Access rejected", str(ctx.exception))

    def test_snunet_wrapper_loading_equivalence_synthetic(self):
        """Verifies strict parameter/buffer and output equivalence between direct load and load_from_checkpoint."""
        import tempfile
        from src.models.snunet_wrapper import SNUNetWrapper

        wrapper_orig = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
        state_dict = wrapper_orig.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
            torch.save({"model_state_dict": state_dict, "git_commit": "synthetic_test"}, tmp.name)

            # Direct load
            direct_wrapper = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
            ckpt = torch.load(tmp.name, map_location="cpu", weights_only=False)
            direct_wrapper.load_state_dict(ckpt["model_state_dict"], strict=True)
            direct_wrapper.eval()

            # load_from_checkpoint
            loaded_wrapper, meta = SNUNetWrapper.load_from_checkpoint(
                tmp.name, in_channels=2, out_ch=3, base_channel=32, device="cpu"
            )

            # Assert parameter equality
            for (n1, p1), (n2, p2) in zip(direct_wrapper.named_parameters(), loaded_wrapper.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(p1, p2))

            # Assert buffer equality
            for (n1, b1), (n2, b2) in zip(direct_wrapper.named_buffers(), loaded_wrapper.named_buffers()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(b1, b2))

            # Assert forward outputs equality
            xA = torch.randn(1, 2, 32, 32)
            xB = torch.randn(1, 2, 32, 32)
            with torch.no_grad():
                out1 = direct_wrapper(xA, xB)
                out2 = loaded_wrapper(xA, xB)
            self.assertTrue(torch.equal(out1, out2))

    def test_zero_valid_pixel_event_flagged_as_unevaluable(self):
        """Verifies zero-valid-pixel event is flagged as technically unevaluable and excluded from macro F1."""
        records = [
            # Event 279: Normal valid pixels, GT flood present
            {
                "sample_id": "patch_279_001",
                "event_id": 279,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
            # Event 437: Entirely invalid pixels (valid_mask == 0)
            {
                "sample_id": "patch_437_invalid",
                "event_id": 437,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.zeros((self.H, self.W), dtype=np.int64),
                "valid_mask": np.zeros((self.H, self.W), dtype=np.uint8),
            },
        ]
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[279, 437])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = MockModel(out_classes=3, pred_class=2)
        evaluator = SourceModelEvaluator(model, device="cpu")

        res = evaluator.evaluate(loader, expected_counts={279: 1, 437: 1})

        # All sample IDs must be tracked in sample_valid_pixel_counts
        self.assertIn("sample_valid_pixel_counts", res)
        self.assertEqual(res["sample_valid_pixel_counts"]["patch_279_001"], self.H * self.W)
        self.assertEqual(res["sample_valid_pixel_counts"]["patch_437_invalid"], 0)

        # Event metrics verification
        event_dict = {r["event_id"]: r for r in res["event_metrics"]}
        self.assertTrue(event_dict[279]["is_evaluable"])
        self.assertEqual(event_dict[279]["f1"], 1.0)

        row_437 = event_dict[437]
        self.assertFalse(row_437["is_evaluable"])
        self.assertEqual(row_437["evaluation_status"], "technically_unevaluable_zero_valid_pixels")
        self.assertIsNone(row_437["f1"])
        self.assertIsNone(row_437["precision"])
        self.assertIsNone(row_437["recall"])
        self.assertIsNone(row_437["iou"])
        self.assertEqual(row_437["valid_pixels"], 0)

        # Macro F1 must be equal to event 279's F1 alone (not distorted by 437)
        self.assertAlmostEqual(res["equal_event_macro"]["macro_f1"], 1.0)
        self.assertEqual(res["collapse_audit"]["unevaluable_events"], [437])

        # Technical validity must be marked as failed because an event had zero valid pixels
        self.assertFalse(res["technical_validity_passed"])
        self.assertFalse(res["technical_validity"]["passed"])
        self.assertEqual(res["evaluable_events_count"], 1)
        self.assertIn("unevaluable_zero_valid_pixel_events_present", res["technical_validity"]["failure_reasons"])
        self.assertEqual(res["equal_event_macro"]["macro_type"], "reduced_event_diagnostic_only")
        self.assertFalse(res["equal_event_macro"]["is_complete_six_event_macro"])

    def test_genuine_flood_free_event_perfect_negative(self):
        """Verifies genuine flood-free event with valid pixels receives F1=1.0 and is marked evaluable."""
        records = [
            # Event 520: Genuine flood-free scene (all valid pixels are background, pred is background)
            {
                "sample_id": "patch_520_dry",
                "event_id": 520,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.zeros((self.H, self.W), dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
        ]
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[520])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = MockModel(out_classes=3, pred_class=0)  # Predicts background (class 0)
        evaluator = SourceModelEvaluator(model, device="cpu")

        res = evaluator.evaluate(loader, expected_counts={520: 1})
        row_520 = res["event_metrics"][0]
        self.assertTrue(row_520["is_evaluable"])
        self.assertEqual(row_520["evaluation_status"], "evaluable_flood_free_perfect_negative")
        self.assertEqual(row_520["f1"], 1.0)
        self.assertEqual(row_520["precision"], 1.0)
        self.assertEqual(row_520["recall"], 1.0)
        self.assertEqual(row_520["iou"], 1.0)
        self.assertEqual(row_520["gt_flood_pixels"], 0)
        self.assertEqual(row_520["pred_flood_pixels"], 0)
        self.assertGreater(row_520["valid_pixels"], 0)

    def test_invalid_samples_not_silently_dropped(self):
        """Verifies invalid-only samples are tracked in sample_valid_pixel_counts and not silently dropped."""
        records = [
            {
                "sample_id": "patch_559_valid",
                "event_id": 559,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 2, dtype=np.int64),
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
            {
                "sample_id": "patch_559_invalid",
                "event_id": 559,
                "xA": np.zeros((2, self.H, self.W), dtype=np.float32),
                "xB": np.zeros((2, self.H, self.W), dtype=np.float32),
                "label": np.full((self.H, self.W), 3, dtype=np.int64),  # Class 3 is ignored
                "valid_mask": np.ones((self.H, self.W), dtype=np.uint8),
            },
        ]
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[559])
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = MockModel(out_classes=3, pred_class=2)
        evaluator = SourceModelEvaluator(model, device="cpu")

        res = evaluator.evaluate(loader, expected_counts={559: 2})
        self.assertEqual(res["total_evaluated_samples"], 2)
        self.assertTrue(res["sample_counts_verified"])
        self.assertIn("patch_559_valid", res["sample_valid_pixel_counts"])
        self.assertIn("patch_559_invalid", res["sample_valid_pixel_counts"])
        self.assertEqual(res["sample_valid_pixel_counts"]["patch_559_invalid"], 0)
        row = res["event_metrics"][0]
        self.assertEqual(row["sample_count"], 2)
        self.assertEqual(row["invalid_only_samples_count"], 1)


if __name__ == "__main__":
    unittest.main()
