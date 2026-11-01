"""Unit tests for generic and change-specific risk indicator functions."""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.indicators import compute_generic_indicators
from src.risk.probes import compute_spatial_concentration
from src.risk.scores import compute_predeclared_change_score


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)


class TestRiskIndicators(unittest.TestCase):
    """Verifies correctness and bounds of risk metrics and score formulas."""

    def test_spatial_concentration_bounded(self):
        """Spatial concentration must be 0 for empty and ~1 for dense solid blocks."""
        # Case 1: Empty mask
        empty_mask = torch.zeros((1, 10, 10), dtype=torch.bool)
        self.assertEqual(compute_spatial_concentration(empty_mask), 0.0)

        # Case 2: Solid 5x5 square
        solid_mask = torch.zeros((1, 10, 10), dtype=torch.bool)
        solid_mask[0, 2:7, 2:7] = True
        conc = compute_spatial_concentration(solid_mask)
        self.assertGreater(conc, 0.6)
        self.assertLessEqual(conc, 1.0)

    def test_generic_indicators_finite_and_bounded(self):
        """Generic indicators must return non-negative finite numbers."""
        B, C, H, W = 2, 3, 16, 16
        cand_logits = torch.randn(B, C, H, W)
        src_logits = torch.randn(B, C, H, W)
        cand_feat = torch.randn(B, 16, H, W)
        src_feat = torch.randn(B, 16, H, W)
        valid_mask = torch.ones(B, H, W, dtype=torch.uint8)

        model = DummyModel()
        source_bn_state = {
            "bn.weight": model.bn.weight.clone(),
            "bn.bias": model.bn.bias.clone(),
            "bn.running_mean": model.bn.running_mean.clone(),
            "bn.running_var": model.bn.running_var.clone(),
        }

        indicators = compute_generic_indicators(
            candidate_logits=cand_logits,
            source_logits=src_logits,
            candidate_features=cand_feat,
            source_features=src_feat,
            candidate_model=model,
            source_bn_state=source_bn_state,
            sar_valid_mask=valid_mask,
        )

        for k, v in indicators.items():
            self.assertTrue(torch.isfinite(torch.tensor(v)), f"Indicator {k} is non-finite: {v}")
            self.assertGreaterEqual(v, 0.0, f"Indicator {k} is negative: {v}")

    def test_predeclared_score_formula_bounds(self):
        """Composite score must be bounded in [0.0, 1.0] and respond correctly to inputs."""
        # Case 1: Zero risk
        zero_risk = {"null_pre2_pre2_area_increase": 0.0, "disappear_fraction_source_erased": 0.0}
        score_dict = compute_predeclared_change_score(zero_risk, total_valid_pixels=1000.0)
        self.assertEqual(score_dict["predeclared_change_risk_score"], 0.0)

        # Case 2: Maximum risk
        max_risk = {"null_pre2_pre2_area_increase": 2000.0, "disappear_fraction_source_erased": 1.0}
        score_dict = compute_predeclared_change_score(max_risk, total_valid_pixels=1000.0)
        self.assertAlmostEqual(score_dict["predeclared_change_risk_score"], 1.0, places=3)

        # Case 3: Intermediate risk
        inter_risk = {"null_pre2_pre2_area_increase": 500.0, "disappear_fraction_source_erased": 0.2}
        score_dict = compute_predeclared_change_score(inter_risk, total_valid_pixels=1000.0)
        self.assertGreater(score_dict["predeclared_change_risk_score"], 0.0)
        self.assertLess(score_dict["predeclared_change_risk_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

