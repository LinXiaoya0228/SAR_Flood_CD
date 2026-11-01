"""Unit tests for Week 14 offline class error diagnosis.

Verifies:
1. 3x3 confusion matrix sums strictly equal valid_pixels.
2. False positives decomposition: C[0,2] + C[1,2] == binary FP.
3. False negatives decomposition: C[2,0] + C[2,1] == binary FN.
4. Exclusion of sensory-invalid and label-3 pixels.
5. Strict rejection of forbidden sealed test events.
"""

from __future__ import annotations

import unittest
import numpy as np

from tools.run_week14_offline_diagnosis import (
    compute_3x3_confusion,
    DEV_EVENTS,
    FORBIDDEN_TEST_EVENTS,
)


class TestWeek14OfflineDiagnosis(unittest.TestCase):
    """Focused test suite for Week 14 offline error diagnosis."""

    def test_01_confusion_matrix_sum_equals_valid_pixels(self):
        """Sum over all 9 cells of the 3x3 matrix must strictly equal count of valid pixels."""
        H, W = 64, 64
        gt = np.random.randint(0, 4, size=(H, W))  # Includes 0, 1, 2, and 3
        pred = np.random.randint(0, 3, size=(H, W))  # 0, 1, 2
        mna = np.random.randint(0, 2, size=(H, W), dtype=np.uint8)  # 0 or 1

        # Double-masking
        valid_mask = (mna == 1) & (gt != 3)
        n_valid = int(np.sum(valid_mask))

        cm = compute_3x3_confusion(pred, gt, valid_mask)
        self.assertEqual(int(np.sum(cm)), n_valid)

    def test_02_error_decomposition_consistency(self):
        """Validates that 3x3 sub-cells decompose binary FP and FN exactly."""
        H, W = 32, 32
        gt = np.random.randint(0, 3, size=(H, W))
        pred = np.random.randint(0, 3, size=(H, W))
        valid_mask = np.ones((H, W), dtype=bool)

        cm = compute_3x3_confusion(pred, gt, valid_mask)

        # Binary flood definitions
        p_flood = (pred == 2)
        g_flood = (gt == 2)

        tp = int(np.sum(p_flood & g_flood & valid_mask))
        fp = int(np.sum(p_flood & ~g_flood & valid_mask))
        fn = int(np.sum(~p_flood & g_flood & valid_mask))
        tn = int(np.sum(~p_flood & ~g_flood & valid_mask))

        # Check equivalence
        self.assertEqual(int(cm[2, 2]), tp)
        self.assertEqual(int(cm[0, 2] + cm[1, 2]), fp)
        self.assertEqual(int(cm[2, 0] + cm[2, 1]), fn)
        self.assertEqual(int(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]), tn)
        self.assertEqual(tp + fp + fn + tn, int(np.sum(cm)))

    def test_03_invalid_and_label3_exclusion(self):
        """Pixels with mna=0 or gt=3 are strictly excluded from all confusion cells."""
        H, W = 16, 16
        gt = np.full((H, W), 3, dtype=np.int64)  # All label-3
        pred = np.zeros((H, W), dtype=np.int64)
        mna = np.ones((H, W), dtype=np.uint8)

        valid_mask = (mna == 1) & (gt != 3)
        self.assertEqual(int(np.sum(valid_mask)), 0)

        cm = compute_3x3_confusion(pred, gt, valid_mask)
        self.assertEqual(int(np.sum(cm)), 0)

    def test_04_quarantine_integrity(self):
        """Sealed test events must have zero overlap with development events."""
        overlap = set(DEV_EVENTS).intersection(FORBIDDEN_TEST_EVENTS)
        self.assertEqual(len(overlap), 0, f"Integrity breach: overlap {overlap}")


if __name__ == "__main__":
    unittest.main()

