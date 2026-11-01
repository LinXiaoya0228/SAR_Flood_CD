"""Unit and contract tests for RST7 Confirmatory Offline Analysis."""

import copy
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
)

CONFIG_PATH = REPO_ROOT / "configs" / "robust_source_training" / "rst7_analysis_config_v1.json"


def _load_analysis_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_analysis_config_structure():
    """Verify rst7_analysis_config_v1.json exists, adheres to schema, and specifies valid runs."""
    cfg = _load_analysis_config()
    assert cfg["$schema_version"] == "1.0.0"
    assert "runs" in cfg
    assert "fold_a" in cfg["runs"]
    assert "fold_b" in cfg["runs"]

    runs_a = cfg["runs"]["fold_a"]
    runs_b = cfg["runs"]["fold_b"]

    assert len(runs_a["r0"]) == 3
    assert len(runs_a["r2"]) == 3
    assert len(runs_b["r0"]) == 3
    assert len(runs_b["r2"]) == 3

    assert len(runs_a["evaluation_events"]) == 9
    assert len(runs_b["evaluation_events"]) == 9


def test_2_zero_forbidden_target_events_in_analysis_inputs():
    """Verify that none of the 16 forbidden target events appear in analysis inputs."""
    cfg = _load_analysis_config()
    forbidden = set(cfg["forbidden_target_events"])
    assert forbidden == set(FORBIDDEN_TARGET_EVENTS)

    events_a = set(cfg["runs"]["fold_a"]["evaluation_events"])
    events_b = set(cfg["runs"]["fold_b"]["evaluation_events"])

    assert not (events_a & forbidden), f"Forbidden events in Fold A: {events_a & forbidden}"
    assert not (events_b & forbidden), f"Forbidden events in Fold B: {events_b & forbidden}"
    assert not (events_a & events_b), f"Overlap between Fold A and Fold B: {events_a & events_b}"

    all_eval = events_a | events_b
    assert len(all_eval) == 18
    assert all_eval.issubset(set(AUTHORITATIVE_SOURCE_EVENTS))


def test_3_all_12_runs_exist_and_contain_event_metrics():
    """Verify all 12 on-disk run directories exist, have completed status, and have valid event_metrics.json."""
    cfg = _load_analysis_config()
    total_event_rows = 0

    for fold_name in ["fold_a", "fold_b"]:
        fold_cfg = cfg["runs"][fold_name]
        expected_events = set(fold_cfg["evaluation_events"])

        for recipe in ["r0", "r2"]:
            for run_rel in fold_cfg[recipe]:
                run_dir = REPO_ROOT / run_rel
                assert run_dir.exists(), f"Missing directory: {run_dir}"

                # Terminal status
                tstat_p = run_dir / "terminal_status.json"
                assert tstat_p.exists(), f"Missing terminal status: {tstat_p}"
                with open(tstat_p, "r", encoding="utf-8") as f:
                    tstat = json.load(f)
                assert tstat["status"] == "COMPLETED_SCIENTIFIC_RUN"
                assert tstat["exit_code"] == 0

                # Event metrics
                em_p = run_dir / "event_metrics.json"
                assert em_p.exists(), f"Missing event metrics: {em_p}"
                with open(em_p, "r", encoding="utf-8") as f:
                    em = json.load(f)

                assert em["n_events_evaluated"] == 9
                assert len(em["per_event"]) == 9
                observed_events = {e["event_id"] for e in em["per_event"]}
                assert observed_events == expected_events, f"Event set mismatch in {run_rel}"
                total_event_rows += len(em["per_event"])

    assert total_event_rows == 108, f"Expected exactly 108 event rows, found {total_event_rows}"


def test_4_synthetic_paired_aggregation_mathematics():
    """Verify paired per-event delta, grand mean, worst-event, and SD calculation formulas."""
    # Mock data for 2 folds x 2 events each, 2 seeds
    # Event 1, 2 in Fold A; Event 3, 4 in Fold B
    mock_r0 = {
        1: [0.40, 0.42],  # mean = 0.41
        2: [0.60, 0.62],  # mean = 0.61
        3: [0.30, 0.32],  # mean = 0.31
        4: [0.70, 0.72],  # mean = 0.71
    }
    mock_r2 = {
        1: [0.45, 0.47],  # mean = 0.46 (delta = +0.05)
        2: [0.62, 0.64],  # mean = 0.63 (delta = +0.02)
        3: [0.32, 0.34],  # mean = 0.33 (delta = +0.02)
        4: [0.71, 0.73],  # mean = 0.72 (delta = +0.01)
    }

    deltas = {e: np.mean(mock_r2[e]) - np.mean(mock_r0[e]) for e in [1, 2, 3, 4]}
    assert abs(deltas[1] - 0.05) < 1e-6
    assert abs(deltas[2] - 0.02) < 1e-6
    assert abs(deltas[3] - 0.02) < 1e-6
    assert abs(deltas[4] - 0.01) < 1e-6

    grand_delta = np.mean(list(deltas.values()))
    assert abs(grand_delta - 0.025) < 1e-6

    # Test single-event driver test
    max_event = max(deltas, key=deltas.get)
    assert max_event == 1
    remaining_deltas = [deltas[e] for e in deltas if e != max_event]
    delta_without_max = np.mean(remaining_deltas)
    assert abs(delta_without_max - (0.05 / 3)) < 1e-6
    assert delta_without_max > 0.0  # Not driven by single event


def test_5_synthetic_decision_rule_logic():
    """Verify P1, P2, P3 decision evaluation logic with synthetic edge cases."""
    def evaluate_criteria(grand_delta, max_event_delta, n_events, p_drop, r_drop, dual_drop_count,
                          tail_delta, worst_delta, sd_r0, sd_r2, sd_a_drop, sd_b_drop):
        # Remaining delta excluding max event
        rem_delta = (grand_delta * n_events - max_event_delta) / (n_events - 1)
        gain_not_driven_by_one = rem_delta > 0.0
        no_pr_collapse = (p_drop > -0.050) and (r_drop > -0.050) and (dual_drop_count <= 4)

        # P1
        p1 = (grand_delta >= 0.010) and gain_not_driven_by_one and no_pr_collapse

        # P2
        p2 = ((tail_delta >= 0.020) or (worst_delta >= 0.020)) and (grand_delta >= -0.005)

        # P3
        sd_red = (sd_r0 - sd_r2) / sd_r0 if sd_r0 > 0 else 0.0
        p3 = (grand_delta >= -0.005) and (sd_red >= 0.20) and sd_a_drop and sd_b_drop

        return p1, p2, p3

    # Case A: Clear P1 pass
    p1, p2, p3 = evaluate_criteria(
        grand_delta=0.020, max_event_delta=0.050, n_events=18,
        p_drop=0.010, r_drop=0.015, dual_drop_count=0,
        tail_delta=0.010, worst_delta=0.005,
        sd_r0=0.02, sd_r2=0.02, sd_a_drop=False, sd_b_drop=False
    )
    assert p1 is True
    assert p2 is False
    assert p3 is False

    # Case B: P1 fails due to single-event driver
    # Grand delta = 0.010, but single event delta = 0.200
    p1, p2, p3 = evaluate_criteria(
        grand_delta=0.010, max_event_delta=0.200, n_events=18,
        p_drop=0.000, r_drop=0.000, dual_drop_count=0,
        tail_delta=0.000, worst_delta=0.000,
        sd_r0=0.02, sd_r2=0.02, sd_a_drop=False, sd_b_drop=False
    )
    assert p1 is False

    # Case C: P2 pass (tail improvement)
    p1, p2, p3 = evaluate_criteria(
        grand_delta=0.002, max_event_delta=0.010, n_events=18,
        p_drop=0.000, r_drop=0.000, dual_drop_count=0,
        tail_delta=0.025, worst_delta=0.010,
        sd_r0=0.02, sd_r2=0.02, sd_a_drop=False, sd_b_drop=False
    )
    assert p1 is False
    assert p2 is True

    # Case D: P3 pass (seed stability)
    p1, p2, p3 = evaluate_criteria(
        grand_delta=0.001, max_event_delta=0.005, n_events=18,
        p_drop=0.000, r_drop=0.000, dual_drop_count=0,
        tail_delta=0.000, worst_delta=0.000,
        sd_r0=0.040, sd_r2=0.025, sd_a_drop=True, sd_b_drop=True
    )
    assert p1 is False
    assert p2 is False
    assert p3 is True  # (0.040 - 0.025) / 0.040 = 0.375 >= 0.20

    # Case E: Inconclusive
    p1, p2, p3 = evaluate_criteria(
        grand_delta=0.003, max_event_delta=0.008, n_events=18,
        p_drop=0.000, r_drop=0.000, dual_drop_count=0,
        tail_delta=0.005, worst_delta=0.000,
        sd_r0=0.020, sd_r2=0.019, sd_a_drop=True, sd_b_drop=False
    )
    assert p1 is False
    assert p2 is False
    assert p3 is False


def test_6_bootstrap_determinism():
    """Verify that paired event-level bootstrap produces bit-identical results with fixed seed."""
    def run_mock_bootstrap(deltas, n_resamples=1000, seed=42):
        rng = np.random.RandomState(seed)
        n = len(deltas)
        means = []
        for _ in range(n_resamples):
            indices = rng.randint(0, n, size=n)
            means.append(np.mean(deltas[indices]))
        return np.percentile(means, [2.5, 97.5])

    sample_deltas = np.array([0.01, -0.02, 0.03, 0.00, 0.05, 0.02, -0.01, 0.04, 0.01] * 2)
    ci1 = run_mock_bootstrap(sample_deltas, seed=42)
    ci2 = run_mock_bootstrap(sample_deltas, seed=42)
    assert np.allclose(ci1, ci2)


if __name__ == "__main__":
    tests = [
        test_1_analysis_config_structure,
        test_2_zero_forbidden_target_events_in_analysis_inputs,
        test_3_all_12_runs_exist_and_contain_event_metrics,
        test_4_synthetic_paired_aggregation_mathematics,
        test_5_synthetic_decision_rule_logic,
        test_6_bootstrap_determinism,
    ]

    print(f"Running {len(tests)} RST7 analysis tests...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} RST7 analysis tests PASSED!")
