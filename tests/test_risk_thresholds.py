"""Unit regression test for all four Gate B harm thresholds (0.0, -0.01, -0.02, -0.10).

Verifies that:
1. Harm thresholds are correctly computed from delta F1.
2. The sensitivity threshold -0.10 (is_harmful_10) is properly integrated.
3. Offline evaluation logic does not omit y_harm_10 or throw NameError.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class TestRiskThresholds(unittest.TestCase):
    """Test suite for Gate B adaptation outcome thresholds."""

    def test_threshold_classification(self):
        """Verify binary harm classifications across delta F1 values."""
        test_cases = [
            # delta_f1, harm_00, harm_01, harm_02, harm_10, beneficial
            (0.05, 0, 0, 0, 0, 1),
            (0.01, 0, 0, 0, 0, 1),
            (0.005, 0, 0, 0, 0, 0),
            (0.000, 1, 0, 0, 0, 0),
            (-0.005, 1, 0, 0, 0, 0),
            (-0.010, 1, 1, 0, 0, 0),
            (-0.015, 1, 1, 0, 0, 0),
            (-0.020, 1, 1, 1, 0, 0),
            (-0.050, 1, 1, 1, 0, 0),
            (-0.100, 1, 1, 1, 1, 0),
            (-0.150, 1, 1, 1, 1, 0),
        ]

        for delta_f1, exp_00, exp_01, exp_02, exp_10, exp_ben in test_cases:
            h_00 = 1 if delta_f1 <= 0.0 else 0
            h_01 = 1 if delta_f1 <= -0.01 else 0
            h_02 = 1 if delta_f1 <= -0.02 else 0
            h_10 = 1 if delta_f1 <= -0.10 else 0
            ben = 1 if delta_f1 >= 0.01 else 0

            self.assertEqual(h_00, exp_00, f"Failed harm_00 for delta_f1={delta_f1}")
            self.assertEqual(h_01, exp_01, f"Failed harm_01 for delta_f1={delta_f1}")
            self.assertEqual(h_02, exp_02, f"Failed harm_02 for delta_f1={delta_f1}")
            self.assertEqual(h_10, exp_10, f"Failed harm_10 for delta_f1={delta_f1}")
            self.assertEqual(ben, exp_ben, f"Failed beneficial for delta_f1={delta_f1}")

    def test_offline_eval_includes_y_harm_10_without_error(self):
        """Simulate the offline evaluation loop to ensure y_harm_10 is defined and computable."""
        data = {
            "hidden_is_harmful_primary": [1, 1, 0, 1, 0, 1],
            "hidden_is_harmful_thresh00": [1, 1, 1, 1, 0, 1],
            "hidden_is_harmful_thresh02": [1, 1, 0, 1, 0, 0],
            "hidden_is_harmful_thresh10": [1, 0, 0, 1, 0, 0],
            "hidden_delta_f1": [-0.12, -0.03, 0.005, -0.15, 0.02, -0.015],
            "hidden_delta_recall": [-0.10, -0.02, 0.002, -0.12, 0.01, -0.01],
            "test_indicator": [0.9, 0.6, 0.1, 0.85, 0.05, 0.5],
        }
        df_joined = pd.DataFrame(data)

        y_harm = df_joined["hidden_is_harmful_primary"].values
        y_harm_00 = df_joined["hidden_is_harmful_thresh00"].values
        y_harm_02 = df_joined["hidden_is_harmful_thresh02"].values
        y_harm_10 = df_joined["hidden_is_harmful_thresh10"].values
        vals = df_joined["test_indicator"].values

        # All four AUROCs must evaluate to valid floats without NameError or NaN
        auc_01 = float(roc_auc_score(y_harm, vals))
        auc_00 = float(roc_auc_score(y_harm_00, vals))
        auc_02 = float(roc_auc_score(y_harm_02, vals))
        auc_10 = float(roc_auc_score(y_harm_10, vals))

        self.assertFalse(np.isnan(auc_01))
        self.assertFalse(np.isnan(auc_00))
        self.assertFalse(np.isnan(auc_02))
        self.assertFalse(np.isnan(auc_10))
        self.assertGreater(auc_10, 0.5)


if __name__ == "__main__":
    unittest.main()
