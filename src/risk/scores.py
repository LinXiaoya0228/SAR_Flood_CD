"""Predeclared, frozen change-specific risk score formulas for Gate B.

This module defines the explicit, transparent mathematical combination of
null-response growth and change-disappearance evidence. It is frozen before
any offline ground-truth evaluation or label joining occurs.
"""

from __future__ import annotations

from typing import Dict


def compute_predeclared_change_score(
    risk_dict: Dict[str, float],
    total_valid_pixels: float,
    epsilon: float = 1.0,
) -> Dict[str, float]:
    """Computes the predeclared, frozen label-free change-specific risk score.

    Mathematical Formulation:
    -------------------------
    1. Null Activation Growth (Signal_null):
       Measures the rate of unprompted class-2 (flood) activation on the primary
       counterfactual null probe (pre2, pre2) relative to the source baseline:
         Signal_null = max(0.0, null_pre2_pre2_area_increase) / (total_valid_pixels + epsilon)
       Direction: Higher -> Greater risk of false-positive explosion / hallucination.
       Bounds: [0.0, 1.0].

    2. Flood Erasure Fraction (Signal_erase):
       Measures the fraction of source-predicted flood pixels erased by adaptation
       on the real observation pair (pre2, post):
         Signal_erase = disappear_fraction_source_erased
       Direction: Higher -> Greater risk of false-negative collapse / flood erasure.
       Bounds: [0.0, 1.0].

    3. Predeclared Composite Change Score (Score_composite):
       A balanced, explainable linear combination:
         Score_composite = 0.5 * Signal_null + 0.5 * Signal_erase
       Bounds: [0.0, 1.0].
       Direction: Higher -> Greater overall operational change-detection risk.

    Args:
        risk_dict: Dictionary containing raw indicators and probe metrics.
        total_valid_pixels: Count of valid observation pixels in the batch.
        epsilon: Numerical stability constant.

    Returns:
        Dictionary containing the individual signals and the composite score.
    """
    # 1. Null Response Growth
    null_area_inc = risk_dict.get("null_pre2_pre2_area_increase", 0.0)
    denom = max(total_valid_pixels, 1.0) + epsilon
    signal_null = max(0.0, float(null_area_inc)) / denom
    signal_null = min(max(signal_null, 0.0), 1.0)

    # 2. Flood Erasure Fraction
    signal_erase = float(risk_dict.get("disappear_fraction_source_erased", 0.0))
    signal_erase = min(max(signal_erase, 0.0), 1.0)

    # 3. Predeclared Composite Score
    score_composite = 0.5 * signal_null + 0.5 * signal_erase

    return {
        "predeclared_signal_null_growth": signal_null,
        "predeclared_signal_flood_erasure": signal_erase,
        "predeclared_change_risk_score": score_composite,
    }

