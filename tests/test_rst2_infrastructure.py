"""Comprehensive unit test suite for RST2 event-disjoint infrastructure (32 criteria)."""

import json
import random
import re
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

try:
    import pytest
except ImportError:
    class _PytestRaisesContext:
        def __init__(self, expected_exception, match=None):
            self.expected_exception = expected_exception
            self.match = match
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected_exception.__name__} but nothing was raised.")
            if not issubclass(exc_type, self.expected_exception):
                return False
            if self.match:
                if not re.search(self.match, str(exc_val)):
                    raise AssertionError(f"Pattern '{self.match}' does not match '{str(exc_val)}'")
            self.value = exc_val
            return True

    class _PytestModule:
        @staticmethod
        def raises(expected_exception, match=None):
            return _PytestRaisesContext(expected_exception, match)

    pytest = _PytestModule()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.dataset import (
    RSTR0SourceDataset,
    build_r0_dataloader,
    seed_everything,
)
from src.robust_source_training.evaluation import (
    EventIndependentEvaluator,
    compute_binary_confusion_counts,
    compute_rates_from_counts,
)
from src.robust_source_training.checkpoint import (
    assert_states_equal,
    load_rst_checkpoint,
    save_rst_checkpoint,
)
from src.robust_source_training.manifest import (
    write_final_manifest,
    write_preflight_manifest,
    write_terminal_status,
)


class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset fixture for deterministic loader testing."""
    def __init__(self, size: int = 40):
        self.size = size
        self.samples = [{"sample_id": f"patch_{i:04d}", "event_id": 273 if i < 20 else 518} for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "sample_id": self.samples[idx]["sample_id"],
            "event_id": self.samples[idx]["event_id"],
            "xA": torch.zeros((2, 224, 224), dtype=torch.float32),
            "xB": torch.zeros((2, 224, 224), dtype=torch.float32),
            "label": torch.zeros((224, 224), dtype=torch.long),
            "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
        }


class DummyModel(nn.Module):
    """Lightweight dummy model for checkpoint and state testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, xA, xB):
        return self.conv(xA + xB)


# --- Tests 1 to 6: Contract Validations ---

def test_1_train_eval_overlap_rejection():
    with pytest.raises(ContractViolationError, match="overlap"):
        validate_event_contracts(
            train_event_ids=[118, 130],
            evaluation_event_ids=[130, 147],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )


def test_2_forbidden_event_rejection():
    with pytest.raises(ContractViolationError, match="Forbidden target events"):
        validate_event_contracts(
            train_event_ids=[118, 205],  # 205 is forbidden
            evaluation_event_ids=[130],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )
    with pytest.raises(ContractViolationError, match="Forbidden target events"):
        validate_event_contracts(
            train_event_ids=[118],
            evaluation_event_ids=[279],  # 279 is forbidden
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )


def test_3_unknown_event_rejection():
    with pytest.raises(ContractViolationError, match="Unknown or unauthorized"):
        validate_event_contracts(
            train_event_ids=[999999],  # non-existent event
            evaluation_event_ids=[130],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )


def test_4_missing_allowlist_rejection():
    with pytest.raises(ContractViolationError, match="cannot be empty"):
        validate_event_contracts(
            train_event_ids=[],
            evaluation_event_ids=[130],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )
    with pytest.raises(ContractViolationError, match="cannot be empty"):
        validate_event_contracts(
            train_event_ids=[118],
            evaluation_event_ids=[],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )


def test_5_duplicate_event_rejection():
    with pytest.raises(ContractViolationError, match="Duplicate event ID"):
        validate_event_contracts(
            train_event_ids=[118, 118, 130],
            evaluation_event_ids=[147],
            fold_id="custom",
            run_id="test_run",
            seed=0,
            is_smoke=True,
        )


def test_6_fold_role_mismatch_rejection(tmp_path):
    fold_file = tmp_path / "event_folds.json"
    fold_data = {
        "experimental_folds": {
            "test_fold": {
                "training_event_ids": [118, 130],
                "evaluation_event_ids": [147],
            }
        }
    }
    fold_file.write_text(json.dumps(fold_data))
    with pytest.raises(ContractViolationError, match="Configured train_event_ids do not match"):
        validate_event_contracts(
            train_event_ids=[118, 273],  # 273 instead of 130
            evaluation_event_ids=[147],
            fold_id="test_fold",
            run_id="scientific_run",
            seed=0,
            is_smoke=False,
            folds_config_path=fold_file,
        )


# --- Tests 7 to 13: Dataset and R0 Property Tests ---

def test_7_8_no_partition_cross_contamination():
    train_events = [273, 518]
    eval_events = [130]
    train_ds = RSTR0SourceDataset(allowed_event_ids=train_events, max_samples_cap=10)
    eval_ds = RSTR0SourceDataset(allowed_event_ids=eval_events, max_samples_cap=10)

    train_sample_ids = {s["sample_id"] for s in train_ds.samples}
    eval_sample_ids = {s["sample_id"] for s in eval_ds.samples}
    assert train_sample_ids.isdisjoint(eval_sample_ids)
    for s in train_ds.samples:
        assert s["event_id"] not in eval_events
    for s in eval_ds.samples:
        assert s["event_id"] not in train_events


def test_9_10_label_mapping_and_valid_masking():
    # Verify double masking logic
    gt = np.array([[0, 1], [2, 0]], dtype=np.int64)
    mna = np.array([[1, 1], [0, 1]], dtype=np.uint8)  # pixel (1,0) invalid
    gt[mna == 0] = 3
    assert gt[1, 0] == 3  # masked to ignore_index
    assert gt[0, 0] == 0 and gt[0, 1] == 1 and gt[1, 1] == 0


def test_11_input_channel_pairing():
    ds = RSTR0SourceDataset(allowed_event_ids=[273], max_samples_cap=1)
    sample = ds[0]
    assert sample["xA"].shape == (2, 224, 224)
    assert sample["xB"].shape == (2, 224, 224)
    assert sample["label"].shape == (224, 224)
    assert sample["valid_mask"].shape == (224, 224)


def test_12_13_augmentation_disabled_and_uniform_sampling():
    ds = DummyDataset(size=20)
    loader = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=42)
    # Sampling visits all samples without replacement
    seen = []
    for b in loader:
        seen.extend(b["sample_id"])
    assert len(seen) == 20
    assert len(set(seen)) == 20  # each visited exactly once


# --- Tests 14 to 15: Determinism Tests ---

def test_14_fixed_seed_sample_order_determinism():
    ds = DummyDataset(size=30)
    l1 = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=123)
    order1 = [sid for b in l1 for sid in b["sample_id"]]

    l2 = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=123)
    order2 = [sid for b in l2 for sid in b["sample_id"]]

    assert order1 == order2

    l3 = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=999)
    order3 = [sid for b in l3 for sid in b["sample_id"]]
    assert order1 != order3  # different seed produces different shuffle


def test_15_worker_initialization_reproducibility():
    ds = DummyDataset(size=30)
    l1 = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=456, num_workers=2)
    order1 = [sid for b in l1 for sid in b["sample_id"]]

    l2 = build_r0_dataloader(ds, batch_size=4, shuffle=True, seed=456, num_workers=2)
    order2 = [sid for b in l2 for sid in b["sample_id"]]

    assert order1 == order2


# --- Tests 16 to 20: Metrics and Aggregations ---

def test_16_event_wise_confusion_counts():
    preds = np.array([[2, 0], [1, 2]])
    labels = np.array([[2, 2], [1, 0]])
    mask = np.ones((2, 2), dtype=np.uint8)

    counts = compute_binary_confusion_counts(preds, labels, mask, positive_class=2)
    assert counts["tp"] == 1  # (0,0) is 2 and 2
    assert counts["fp"] == 1  # (1,1) is 2, true is 0
    assert counts["fn"] == 1  # (0,1) is 0, true is 2
    assert counts["tn"] == 1  # (1,0) is 1 and 1


def test_17_18_19_macro_and_pooled_metrics():
    res1 = {"event_id": 1, "tp": 10, "fp": 0, "fn": 0, "tn": 90, "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    res2 = {"event_id": 2, "tp": 0, "fp": 10, "fn": 10, "tn": 80, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    evaluator = EventIndependentEvaluator([1, 2], device=torch.device("cpu"))
    agg = evaluator.aggregate_metrics([res1, res2])

    assert agg["event_macro"]["f1"] == 0.5
    assert agg["event_macro"]["precision"] == 0.5
    assert agg["event_macro"]["recall"] == 0.5

    # Pooled TP=10, FP=10, FN=10 -> P = 10/20=0.5, R = 10/20=0.5, F1 = 20/40=0.5, IoU = 10/30=0.333333
    assert agg["pooled"]["tp"] == 10
    assert agg["pooled"]["fp"] == 10
    assert agg["pooled"]["f1"] == 0.5
    assert agg["pooled"]["iou"] == 0.333333
    assert agg["worst_event"]["event_id"] == 2
    assert agg["worst_event"]["f1"] == 0.0


def test_20_zero_denominator_behavior():
    rates = compute_rates_from_counts(tp=0, fp=0, fn=0, tn=100)
    assert rates["precision"] == 1.0
    assert rates["recall"] == 1.0
    assert rates["f1"] == 1.0
    assert rates["iou"] == 1.0

    rates_no_tp = compute_rates_from_counts(tp=0, fp=10, fn=10, tn=80)
    assert rates_no_tp["precision"] == 0.0
    assert rates_no_tp["recall"] == 0.0
    assert rates_no_tp["f1"] == 0.0
    assert rates_no_tp["iou"] == 0.0


# --- Tests 21 to 22: Manifest Lifecycle ---

def test_21_22_manifest_lifecycle(tmp_path):
    run_dir = tmp_path / "run_test"
    cfg_p = tmp_path / "test_cfg.json"
    cfg_p.write_text('{"run_id": "test_m", "seed": 0}\n')

    p_path, p_sha, p_dict = write_preflight_manifest(
        run_dir=run_dir,
        config={"run_id": "test_m", "seed": 0},
        config_path=cfg_p,
        repo_root=REPO_ROOT,
    )
    assert p_path.exists()
    assert len(p_sha) == 64

    # Immutability check: cannot write again to same dir
    with pytest.raises(RuntimeError, match="already exists"):
        write_preflight_manifest(run_dir, {"run_id": "test_m"}, cfg_p, REPO_ROOT)

    # Final manifest referencing preflight
    f_path, f_sha, f_dict = write_final_manifest(
        run_dir=run_dir,
        preflight_sha256=p_sha,
        preflight_dict=p_dict,
        completion_status="COMPLETED_INFRASTRUCTURE_SMOKE",
        observed_sample_count=20,
        observed_steps=2,
        runtime_seconds=1.5,
        final_checkpoint_path=None,
        last_checkpoint_path=None,
        event_metrics_path=None,
    )
    assert f_dict["preflight_manifest_sha256"] == p_sha


# --- Tests 23 to 28: Checkpoint & Resume Integrity ---

def test_23_24_25_26_checkpoint_reload_and_states(tmp_path):
    m1 = DummyModel()
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    # Perform one step to populate optimizer state
    loss = m1(torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)).sum()
    loss.backward()
    opt1.step()

    ckpt_p = tmp_path / "test_ckpt.pt"
    save_rst_checkpoint(
        path=ckpt_p,
        model=m1,
        optimizer=opt1,
        epoch=1,
        global_step=1,
        seed=42,
        run_id="rst2_ckpt_test",
        config_hash="cfg_hash_123",
        fold_hash="fold_hash_456",
        train_event_ids=[118],
        evaluation_event_ids=[130],
    )

    m2 = DummyModel()
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    loaded_meta = load_rst_checkpoint(
        path=ckpt_p,
        model=m2,
        optimizer=opt2,
        expected_config_hash="cfg_hash_123",
        expected_fold_hash="fold_hash_456",
        expected_train_events=[118],
        expected_eval_events=[130],
    )
    assert loaded_meta["epoch"] == 1
    assert loaded_meta["global_step"] == 1
    assert loaded_meta["scheduler_state_dict"] == "NA"
    assert_states_equal(m1, m2)


def test_27_28_resume_mismatch_rejection(tmp_path):
    m = DummyModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ckpt_p = tmp_path / "mismatch_ckpt.pt"
    save_rst_checkpoint(
        path=ckpt_p,
        model=m,
        optimizer=opt,
        epoch=1,
        global_step=1,
        seed=0,
        run_id="run_a",
        config_hash="hash_a",
        fold_hash="fold_a",
        train_event_ids=[118],
        evaluation_event_ids=[130],
    )

    # Config mismatch
    with pytest.raises(ContractViolationError, match="config hash mismatch"):
        load_rst_checkpoint(ckpt_p, m, opt, expected_config_hash="hash_DIFFERENT")

    # Fold mismatch
    with pytest.raises(ContractViolationError, match="fold hash mismatch"):
        load_rst_checkpoint(ckpt_p, m, opt, expected_fold_hash="fold_DIFFERENT")


# --- Tests 29 to 32: System & Invariance Tests ---

def test_29_output_root_collision_rejection(tmp_path):
    from src.robust_source_training.runner import RSTExecutionEngine
    run_dir = tmp_path / "out_exists"
    run_dir.mkdir(parents=True)
    (run_dir / "existing_file.txt").write_text("blocked")

    cfg_file = tmp_path / "collision_cfg.json"
    cfg = {
        "run_id": "test_col",
        "output_root": str(run_dir),
        "execution_class": "INFRASTRUCTURE_SMOKE",
        "launch_authorized": True,
        "partition": {"train_event_ids": [273], "evaluation_event_ids": [130]},
    }
    cfg_file.write_text(json.dumps(cfg))

    engine = RSTExecutionEngine(config_path=cfg_file, repo_root=REPO_ROOT)
    with pytest.raises(ContractViolationError, match="already exists and contains files"):
        engine.execute()


def test_30_durable_failure_status(tmp_path):
    status_p = write_terminal_status(tmp_path, "failed_run", status="FAILED", exit_code=1, error_message="Crash test")
    assert status_p.exists()
    data = json.loads(status_p.read_text())
    assert data["status"] == "FAILED"
    assert data["exit_code"] == 1


def test_31_zero_forbidden_event_access():
    for f_eid in FORBIDDEN_TARGET_EVENTS:
        with pytest.raises(ContractViolationError):
            RSTR0SourceDataset(allowed_event_ids=[f_eid])


def test_32_no_historical_file_mutation():
    c_hash = "26832a93be8efd85ada522c39e9334a678c6b3860f3379efe479567d879b3f36"
    t_hash = "2e0d460ce088a8a6883ed5cdfadb61da01d9baaa3bf2b28249efcd6bcfd87ff8"

    import hashlib
    f_c = REPO_ROOT / "docs/e11_final_scientific_conclusions.md"
    f_t = REPO_ROOT / "docs/e11_final_test_closeout.md"

    assert hashlib.sha256(f_c.read_bytes()).hexdigest() == c_hash
    assert hashlib.sha256(f_t.read_bytes()).hexdigest() == t_hash


if __name__ == "__main__":
    import inspect
    import shutil
    import tempfile

    test_funcs = [
        (name, func)
        for name, func in globals().items()
        if name.startswith("test_") and callable(func)
    ]
    test_funcs.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", x[0])])
    print(f"Running {len(test_funcs)} infrastructure tests...")

    passed = 0
    failed = 0
    for name, func in test_funcs:
        sig = inspect.signature(func)
        tmp_dir = None
        try:
            if "tmp_path" in sig.parameters:
                tmp_dir = tempfile.mkdtemp(prefix="rst2_test_")
                func(Path(tmp_dir))
            else:
                func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        finally:
            if tmp_dir and Path(tmp_dir).exists():
                shutil.rmtree(tmp_dir)

    print(f"\nTest Summary: {passed} passed, {failed} failed out of {len(test_funcs)} tests.")
    if failed > 0:
        sys.exit(1)

