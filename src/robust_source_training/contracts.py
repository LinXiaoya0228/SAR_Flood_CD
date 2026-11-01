"""Strict event boundary and research integrity contract validation for RST."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class ContractViolationError(RuntimeError):
    """Raised when an RST event contract or partition constraint is violated."""
    pass


# Authoritative event sets defined in configs/robust_source_training/rst0_data_boundary_v1.json
AUTHORITATIVE_SOURCE_EVENTS = {
    118, 130, 147, 174, 267, 273, 275, 324, 417, 421, 427,
    470, 492, 496, 497, 498, 502, 518, 554, 555, 567,
    1111004, 1111005, 1111006, 1111009, 1111010, 1111011
}

FORBIDDEN_TARGET_EVENTS = {
    205, 277, 279, 321, 411, 437, 445, 520, 559, 561, 562,
    1111002, 1111003, 1111007, 1111008, 1111013
}


def validate_event_contracts(
    train_event_ids: Any,
    evaluation_event_ids: Any,
    fold_id: str,
    run_id: str,
    seed: int,
    is_smoke: bool = False,
    config_hash: Optional[str] = None,
    folds_config_path: Optional[Union[str, Path]] = None,
    custom_source_population: Optional[Set[int]] = None,
    custom_forbidden_population: Optional[Set[int]] = None,
    allow_empty_eval: bool = False,
) -> Dict[str, Any]:
    """Validates explicit event allowlists against research integrity contracts.

    Raises:
        ContractViolationError: on any validation failure (fails closed).

    Returns:
        Validated metadata dictionary with integer lists of event IDs.
    """
    source_pop = custom_source_population if custom_source_population is not None else AUTHORITATIVE_SOURCE_EVENTS
    forbidden_pop = custom_forbidden_population if custom_forbidden_population is not None else FORBIDDEN_TARGET_EVENTS

    # 1. Allowlist presence and non-emptiness
    if train_event_ids is None or not isinstance(train_event_ids, (list, tuple, set)):
        raise ContractViolationError("train_event_ids must be a non-empty sequence of integer event IDs.")
    if evaluation_event_ids is None or not isinstance(evaluation_event_ids, (list, tuple, set)):
        raise ContractViolationError("evaluation_event_ids must be a sequence of integer event IDs.")

    if len(train_event_ids) == 0:
        raise ContractViolationError("train_event_ids cannot be empty.")
    if len(evaluation_event_ids) == 0 and not allow_empty_eval:
        raise ContractViolationError("evaluation_event_ids cannot be empty.")

    # 2. Type consistency & duplicate checks
    clean_train: List[int] = []
    seen_train: Set[int] = set()
    for e in train_event_ids:
        if not isinstance(e, int) or isinstance(e, bool):
            raise ContractViolationError(f"Inconsistent or invalid type for train event ID: {e} ({type(e)})")
        if e in seen_train:
            raise ContractViolationError(f"Duplicate event ID in train_event_ids: {e}")
        seen_train.add(e)
        clean_train.append(e)

    clean_eval: List[int] = []
    seen_eval: Set[int] = set()
    for e in evaluation_event_ids:
        if not isinstance(e, int) or isinstance(e, bool):
            raise ContractViolationError(f"Inconsistent or invalid type for evaluation event ID: {e} ({type(e)})")
        if e in seen_eval:
            raise ContractViolationError(f"Duplicate event ID in evaluation_event_ids: {e}")
        seen_eval.add(e)
        clean_eval.append(e)

    # 3. Disjointness check
    overlap = seen_train.intersection(seen_eval)
    if len(overlap) > 0:
        raise ContractViolationError(f"Train and evaluation event allowlists overlap! Overlapping events: {sorted(list(overlap))}")

    # 4. Forbidden target events check
    forbidden_in_train = seen_train.intersection(forbidden_pop)
    if len(forbidden_in_train) > 0:
        raise ContractViolationError(f"CRITICAL INTEGRITY VIOLATION: Forbidden target events found in train_event_ids: {sorted(list(forbidden_in_train))}")

    forbidden_in_eval = seen_eval.intersection(forbidden_pop)
    if len(forbidden_in_eval) > 0:
        raise ContractViolationError(f"CRITICAL INTEGRITY VIOLATION: Forbidden target events found in evaluation_event_ids: {sorted(list(forbidden_in_eval))}")

    # 5. Unknown event ID check (must belong to authorized source population)
    unknown_train = seen_train.difference(source_pop)
    if len(unknown_train) > 0:
        raise ContractViolationError(f"Unknown or unauthorized events in train_event_ids: {sorted(list(unknown_train))}")

    unknown_eval = seen_eval.difference(source_pop)
    if len(unknown_eval) > 0:
        raise ContractViolationError(f"Unknown or unauthorized events in evaluation_event_ids: {sorted(list(unknown_eval))}")

    # 6. Fold role validation against rst1_event_folds_v1.json if full scientific run
    if not is_smoke and folds_config_path is not None:
        p = Path(folds_config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                fold_data = json.load(f)
            exp_folds = fold_data.get("experimental_folds", {})
            if fold_id in exp_folds:
                expected_train = set(exp_folds[fold_id]["training_event_ids"])
                expected_eval = set(exp_folds[fold_id]["evaluation_event_ids"])
                if seen_train != expected_train:
                    raise ContractViolationError(
                        f"Configured train_event_ids do not match frozen fold {fold_id}! Discrepancy: {seen_train ^ expected_train}"
                    )
                if seen_eval != expected_eval:
                    raise ContractViolationError(
                        f"Configured evaluation_event_ids do not match frozen fold {fold_id}! Discrepancy: {seen_eval ^ expected_eval}"
                    )

    # 7. Smoke vs Scientific policy check
    if not is_smoke and "smoke" in run_id.lower():
        raise ContractViolationError(f"Scientific run ID '{run_id}' contains 'smoke' but is_smoke=False.")

    return {
        "train_event_ids": clean_train,
        "evaluation_event_ids": clean_eval,
        "train_event_count": len(clean_train),
        "evaluation_event_count": len(clean_eval),
        "fold_id": fold_id,
        "run_id": run_id,
        "seed": seed,
        "is_smoke": is_smoke,
        "is_valid": True,
    }

