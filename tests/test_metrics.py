"""Unit tests for binary flood change detection metrics."""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_confusion_counts, compute_flood_metrics, EventMetricAccumulator


def test_perfect_flood_prediction():
    """Case 1: Perfect flood prediction."""
    # 4 pixels: [0: no water, 1: perm water, 2: flood, 2: flood]
    labels = np.array([[0, 1], [2, 2]], dtype=np.int64)
    preds = np.array([[0, 1], [2, 2]], dtype=np.int64)
    valid_mask = np.ones((2, 2), dtype=np.uint8)

    counts = compute_confusion_counts(preds, labels, valid_mask)
    metrics = compute_flood_metrics(counts)

    assert counts["tp"] == 2
    assert counts["fp"] == 0
    assert counts["fn"] == 0
    assert counts["tn"] == 2
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0
    print("PASS: test_perfect_flood_prediction")


def test_all_background_prediction():
    """Case 2: Model predicts all pixels as background (0 or 1), failing to detect any flood."""
    labels = np.array([[0, 1], [2, 2]], dtype=np.int64)
    preds = np.array([[0, 1], [0, 1]], dtype=np.int64)  # No class 2
    valid_mask = np.ones((2, 2), dtype=np.uint8)

    counts = compute_confusion_counts(preds, labels, valid_mask)
    metrics = compute_flood_metrics(counts)

    assert counts["tp"] == 0
    assert counts["fp"] == 0
    assert counts["fn"] == 2
    assert counts["tn"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0
    print("PASS: test_all_background_prediction")


def test_all_flood_prediction():
    """Case 3: Model predicts all pixels as flood (class 2)."""
    labels = np.array([[0, 1], [2, 2]], dtype=np.int64)
    preds = np.array([[2, 2], [2, 2]], dtype=np.int64)
    valid_mask = np.ones((2, 2), dtype=np.uint8)

    counts = compute_confusion_counts(preds, labels, valid_mask)
    metrics = compute_flood_metrics(counts)

    assert counts["tp"] == 2
    assert counts["fp"] == 2  # class 0 and 1 falsely predicted as 2
    assert counts["fn"] == 0
    assert counts["tn"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 2 * (0.5 * 1.0) / (0.5 + 1.0)
    assert metrics["iou"] == 2 / 4
    print("PASS: test_all_flood_prediction")


def test_no_ground_truth_flood():
    """Case 4: No ground-truth flood exists in the scene."""
    labels = np.array([[0, 1], [0, 1]], dtype=np.int64)
    valid_mask = np.ones((2, 2), dtype=np.uint8)

    # Subcase 4a: Model correctly predicts no flood
    preds_correct = np.array([[0, 1], [0, 1]], dtype=np.int64)
    counts_a = compute_confusion_counts(preds_correct, labels, valid_mask)
    metrics_a = compute_flood_metrics(counts_a)
    assert counts_a["tp"] == 0
    assert counts_a["fp"] == 0
    assert counts_a["fn"] == 0
    assert counts_a["tn"] == 4
    assert metrics_a["precision"] == 1.0
    assert metrics_a["recall"] == 1.0
    assert metrics_a["f1"] == 1.0
    assert metrics_a["iou"] == 1.0

    # Subcase 4b: Model hallucinates flood
    preds_hallucinated = np.array([[2, 1], [0, 0]], dtype=np.int64)
    counts_b = compute_confusion_counts(preds_hallucinated, labels, valid_mask)
    metrics_b = compute_flood_metrics(counts_b)
    assert counts_b["tp"] == 0
    assert counts_b["fp"] == 1
    assert counts_b["fn"] == 0
    assert counts_b["tn"] == 3
    assert metrics_b["precision"] == 0.0
    assert metrics_b["recall"] == 0.0
    assert metrics_b["f1"] == 0.0
    assert metrics_b["iou"] == 0.0
    print("PASS: test_no_ground_truth_flood")


def test_class3_marked_valid_exclusion():
    """Case 5: Class-3 (Invalid) pixels are marked valid by valid_mask == 1, but must be ignored."""
    labels = np.array([[0, 3], [2, 3]], dtype=np.int64)
    # Notice valid_mask has 1 at the positions where label is 3!
    valid_mask = np.ones((2, 2), dtype=np.uint8)
    preds = np.array([[0, 2], [2, 0]], dtype=np.int64)

    counts = compute_confusion_counts(preds, labels, valid_mask)
    # Valid pixels should only be (0, 0) and (1, 0) -> total 2 valid pixels!
    assert counts["valid_pixels"] == 2
    assert counts["tp"] == 1  # (1, 0)
    assert counts["tn"] == 1  # (0, 0)
    assert counts["fp"] == 0  # (0, 1) was ignored because label is 3
    assert counts["fn"] == 0  # (1, 1) was ignored because label is 3
    print("PASS: test_class3_marked_valid_exclusion")


def test_permanent_water_exclusion_from_flood_target():
    """Case 6: Permanent water (class 1) is strictly a negative class for flood detection."""
    labels = np.array([[1, 1], [0, 2]], dtype=np.int64)
    valid_mask = np.ones((2, 2), dtype=np.uint8)
    
    # Model predicts class 1 as permanent water (correctly negative for flood)
    # and predicts class 2 as flood (correctly positive for flood)
    preds = np.array([[1, 1], [0, 2]], dtype=np.int64)
    counts = compute_confusion_counts(preds, labels, valid_mask)
    assert counts["gt_flood_pixels"] == 1
    assert counts["tp"] == 1
    assert counts["tn"] == 3  # Both class 1 pixels and class 0 pixel are non-flood negatives
    assert counts["fp"] == 0
    assert counts["fn"] == 0

    # Model incorrectly predicts permanent water as flood
    preds_confused = np.array([[2, 1], [0, 2]], dtype=np.int64)
    counts_confused = compute_confusion_counts(preds_confused, labels, valid_mask)
    assert counts_confused["fp"] == 1  # Permanent water falsely predicted as flood
    print("PASS: test_permanent_water_exclusion_from_flood_target")


if __name__ == "__main__":
    test_perfect_flood_prediction()
    test_all_background_prediction()
    test_all_flood_prediction()
    test_no_ground_truth_flood()
    test_class3_marked_valid_exclusion()
    test_permanent_water_exclusion_from_flood_target()
    print("\nALL 6 METRIC UNIT TESTS PASSED SUCCESSFULLY!")

