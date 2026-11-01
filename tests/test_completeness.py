"""Unit tests for evaluation completeness tracker."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.completeness import CompletenessTracker


def test_completeness_pass():
    """Verify clean full evaluation passes completeness check."""
    expected = {
        279: ["sample_279_a", "sample_279_b", "sample_279_c"],
        559: ["sample_559_x", "sample_559_y"],
    }
    tracker = CompletenessTracker(expected)
    tracker.update([279, 279], ["sample_279_a", "sample_279_b"], [1000, 2000])
    tracker.update([279, 559], ["sample_279_c", "sample_559_x"], [1500, 800])
    tracker.update([559], ["sample_559_y"], [900])

    report = tracker.validate_or_raise()
    assert report["status"] == "PASS"
    assert report["total_expected_samples"] == 5
    assert report["total_processed_samples"] == 5
    assert report["processed_batch_count"] == 3
    assert report["events"]["279"]["total_valid_pixels"] == 4500
    assert report["events"]["559"]["total_valid_pixels"] == 1700
    print("PASS: test_completeness_pass")


def test_completeness_missing_fails():
    """Verify missing sample raises RuntimeError."""
    expected = {
        279: ["sample_279_a", "sample_279_b", "sample_279_c"],
    }
    tracker = CompletenessTracker(expected)
    tracker.update([279, 279], ["sample_279_a", "sample_279_b"])

    try:
        tracker.validate_or_raise()
        assert False, "Should have raised RuntimeError on missing sample!"
    except RuntimeError as e:
        assert "Event 279 FAILED" in str(e)
        assert "count mismatch" in str(e)
    print("PASS: test_completeness_missing_fails")


def test_completeness_duplicate_fails():
    """Verify duplicate sample ID raises RuntimeError."""
    expected = {
        279: ["sample_279_a", "sample_279_b"],
    }
    tracker = CompletenessTracker(expected)
    tracker.update([279, 279], ["sample_279_a", "sample_279_a"])  # Duplicate!

    try:
        tracker.validate_or_raise()
        assert False, "Should have raised RuntimeError on duplicate sample ID!"
    except RuntimeError as e:
        assert "duplicate sample IDs detected" in str(e)
    print("PASS: test_completeness_duplicate_fails")


def test_completeness_allow_partial_diagnostic():
    """Verify allow_partial=True reports FAIL without raising unhandled exception."""
    expected = {
        279: ["sample_279_a", "sample_279_b", "sample_279_c"],
    }
    tracker = CompletenessTracker(expected, allow_partial=True)
    tracker.update([279], ["sample_279_a"])  # Only 1 of 3

    report = tracker.validate_or_raise()
    assert report["status"] == "FAIL"
    assert report["events"]["279"]["status"] == "FAIL"
    assert report["events"]["279"]["processed_sample_count"] == 1
    assert report["events"]["279"]["expected_sample_count"] == 3
    print("PASS: test_completeness_allow_partial_diagnostic")


if __name__ == "__main__":
    test_completeness_pass()
    test_completeness_missing_fails()
    test_completeness_duplicate_fails()
    test_completeness_allow_partial_diagnostic()
    print("\nALL COMPLETENESS UNIT TESTS PASSED SUCCESSFULLY!")
