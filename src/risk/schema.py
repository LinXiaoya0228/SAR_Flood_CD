"""Schema enforcement for label-free risk generation artifacts.

Ensures strict scientific integrity by mathematically verifying that zero
ground-truth labels or GT-derived metrics contaminate the risk-feature artifacts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

REQUIRED_IDENTIFIERS: Set[str] = {
    "run_id",
    "event_id",
    "seed",
    "batch_idx",
    "sample_ids",
    "protocol_hash",
    "checkpoint_hash",
}

# Forbidden keywords and regex patterns indicating ground truth leakage
FORBIDDEN_GT_PATTERNS: List[re.Pattern] = [
    re.compile(r"^label", re.IGNORECASE),
    re.compile(r"_label", re.IGNORECASE),
    re.compile(r"^labels$", re.IGNORECASE),
    re.compile(r"^gt_", re.IGNORECASE),
    re.compile(r"_gt$", re.IGNORECASE),
    re.compile(r"^ground_truth", re.IGNORECASE),
    re.compile(r"_ground_truth", re.IGNORECASE),
    re.compile(r"^tp$", re.IGNORECASE),
    re.compile(r"^fp$", re.IGNORECASE),
    re.compile(r"^fn$", re.IGNORECASE),
    re.compile(r"^tn$", re.IGNORECASE),
    re.compile(r"^f1$", re.IGNORECASE),
    re.compile(r"_f1$", re.IGNORECASE),
    re.compile(r"^iou$", re.IGNORECASE),
    re.compile(r"_iou$", re.IGNORECASE),
    re.compile(r"^precision$", re.IGNORECASE),
    re.compile(r"_precision$", re.IGNORECASE),
    re.compile(r"^recall$", re.IGNORECASE),
    re.compile(r"_recall$", re.IGNORECASE),
    re.compile(r"gt_flood", re.IGNORECASE),
    re.compile(r"harmful", re.IGNORECASE),
    re.compile(r"beneficial", re.IGNORECASE),
    re.compile(r"critical_recall", re.IGNORECASE),
    re.compile(r"collapse", re.IGNORECASE),
]


def check_forbidden_field_names(keys: Set[str]) -> List[str]:
    """Checks whether any key in the set matches a forbidden ground truth pattern."""
    violations: List[str] = []
    for k in sorted(keys):
        for pat in FORBIDDEN_GT_PATTERNS:
            if pat.search(k):
                violations.append(k)
                break
    return violations


def validate_risk_feature_dict(record: Dict[str, Any]) -> None:
    """Validates that a risk feature record conforms to the label-free schema.

    Raises:
        ValueError: If required identifiers are missing or forbidden GT fields exist.
    """
    keys = set(record.keys())
    
    # 1. Verify required identifiers exist
    missing_ids = REQUIRED_IDENTIFIERS - keys
    if missing_ids:
        raise ValueError(
            f"SCHEMA VIOLATION: Risk feature record missing required identifiers: {sorted(missing_ids)}"
        )

    # 2. Verify no forbidden GT-derived fields exist
    violations = check_forbidden_field_names(keys)
    if violations:
        raise ValueError(
            f"RESEARCH INTEGRITY VIOLATION: Risk feature record contains forbidden "
            f"ground-truth derived fields: {violations}. Test-time risk features must be strictly label-free!"
        )

