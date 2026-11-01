"""Risk-signal validation and probing package for ChangeGuard Gate B."""

from src.risk.schema import validate_risk_feature_dict, FORBIDDEN_GT_PATTERNS
from src.risk.indicators import compute_generic_indicators, IntermediateFeatureHook
from src.risk.probes import run_null_probes, run_disappearance_probes
from src.risk.scores import compute_predeclared_change_score
from src.risk.collector import RiskFeatureCollector

__all__ = [
    "validate_risk_feature_dict",
    "FORBIDDEN_GT_PATTERNS",
    "compute_generic_indicators",
    "IntermediateFeatureHook",
    "run_null_probes",
    "run_disappearance_probes",
    "compute_predeclared_change_score",
    "RiskFeatureCollector",
]

