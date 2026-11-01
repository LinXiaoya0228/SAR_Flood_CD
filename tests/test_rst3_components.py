"""Comprehensive test suite for RST3 robust source training components (M1, M2, M3, and R0 Parity)."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

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
from src.robust_source_training.sampler import (
    EventBalancedSampler,
    compute_m1_seed,
)
from src.robust_source_training.loss import (
    RobustSourceLoss,
    compute_training_class_weights,
)
from src.robust_source_training.augmentation import (
    SynchronizedPairedSARAugmentation,
    compute_m3_sample_seed,
)
from src.robust_source_training.checkpoint import (
    assert_states_equal,
    load_rst_checkpoint,
    save_rst_checkpoint,
)
from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.losses import ControlledSourceLoss

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


# --- Fixtures ---

def _build_synthetic_metadata(num_events: int = 3, samples_per_event: List[int] = None) -> List[Dict[str, Any]]:
    if samples_per_event is None:
        samples_per_event = [10, 30, 60]  # E=3, total N=100
    meta = []
    events = [118, 130, 147][:num_events]
    for eid, count in zip(events, samples_per_event):
        for s in range(count):
            meta.append({
                "sample_id": f"event_{eid}_patch_{s:04d}",
                "event_id": eid,
                "relative_path": f"data/event_{eid}/patch_{s:04d}",
            })
    return meta


# --- Group 1: Component M1 (Event-Balanced Sampler) Tests ---

def test_m1_1_fixed_seed_sample_order_reproducibility():
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[5, 20, 50])
    s1 = EventBalancedSampler(meta, run_seed=42, epoch=1)
    order1 = list(s1)
    s2 = EventBalancedSampler(meta, run_seed=42, epoch=1)
    order2 = list(s2)
    assert order1 == order2, "Fixed seed and epoch must yield identical sample order."

    s3 = EventBalancedSampler(meta, run_seed=999, epoch=1)
    order3 = list(s3)
    assert order1 != order3, "Different seed must yield different order."

    s4 = EventBalancedSampler(meta, run_seed=42, epoch=2)
    order4 = list(s4)
    assert order1 != order4, "Different epoch must yield different order."


def test_m1_2_epoch_length_matches_r0():
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[8, 25, 47])
    total_samples = len(meta)  # 80
    sampler = EventBalancedSampler(meta, run_seed=123, epoch=0)
    order = list(sampler)
    assert len(order) == total_samples, f"Epoch length {len(order)} must equal total dataset size {total_samples}."
    assert len(sampler) == total_samples


def test_m1_3_event_representation_uniformity():
    # E=3 events, N=100 patches: 100 // 3 = 33 base quota, 1 remainder
    # Quotas: 34, 33, 33
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[5, 25, 70])
    sampler = EventBalancedSampler(meta, run_seed=777, epoch=0)
    order = list(sampler)

    realized = {}
    for idx in order:
        eid = meta[idx]["event_id"]
        realized[eid] = realized.get(eid, 0) + 1

    expected = sampler.expected_draws_per_event
    assert realized == expected
    assert max(realized.values()) - min(realized.values()) <= 1, "Draw counts must differ by at most 1."


def test_m1_4_small_event_recycling_safety():
    # Event 118 has only 2 samples, but receives 20 draws out of 60 total
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[2, 18, 40])
    sampler = EventBalancedSampler(meta, run_seed=101, epoch=0)
    summary = sampler.get_epoch_summary()

    # Event 118 received 20 draws using 2 unique samples
    assert summary["expected_draws_per_event"][118] == 20
    assert summary["realized_draws_per_event"][118] == 20
    assert summary["unique_samples_used_per_event"][118] == 2
    assert summary["repeat_count_per_event"][118] == 18


def test_m1_5_no_unknown_or_forbidden_event_access():
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[10, 20, 30])
    sampler = EventBalancedSampler(meta, run_seed=42, epoch=0)
    for idx in sampler:
        eid = meta[idx]["event_id"]
        assert eid in [118, 130, 147]
        assert eid not in FORBIDDEN_TARGET_EVENTS


def test_m1_6_resume_slice_reproducibility():
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[10, 20, 30])
    s_full = EventBalancedSampler(meta, run_seed=55, epoch=3)
    full_order = list(s_full)

    start_offset = 25
    s_resumed = EventBalancedSampler(meta, run_seed=55, epoch=3, start_index=start_offset)
    resumed_order = list(s_resumed)

    assert resumed_order == full_order[start_offset:]
    assert len(s_resumed) == len(full_order) - start_offset


def test_m1_7_logging_digest_and_counts():
    meta = _build_synthetic_metadata(num_events=3, samples_per_event=[10, 20, 30])
    sampler = EventBalancedSampler(meta, run_seed=42, epoch=0)
    summary = sampler.get_epoch_summary()

    assert summary["total_draws"] == 60
    assert "ordered_sample_id_digest" in summary
    assert len(summary["ordered_sample_id_digest"]) == 64
    assert summary["min_event_draws"] == 20
    assert summary["max_event_draws"] == 20


# --- Group 2: Component M2 (Flood-Aware Loss) Tests ---

def test_m2_1_frozen_formula_exact_reproduction():
    # Synthetic class pixel proportions
    # Suppose N_total = 1,000,000
    # c0: 800,000 (80%), c1: 180,000 (18%), c2: 20,000 (2%)
    p0, p1, p2 = 0.80, 0.18, 0.02
    eps = 1e-12
    raw0 = 1.0 / math.sqrt(p0)
    raw1 = 1.0 / math.sqrt(p1)
    raw2 = 1.0 / math.sqrt(p2)
    mean_raw = (raw0 + raw1 + raw2) / 3.0

    w0 = raw0 / mean_raw
    w1 = raw1 / mean_raw
    w2 = raw2 / mean_raw

    w0_c = min(max(w0, 0.5), 2.0)
    w1_c = min(max(w1, 0.5), 2.0)
    w2_c = min(max(w2, 0.5), 2.0)
    mean_clipped = (w0_c + w1_c + w2_c) / 3.0

    expected_final = [w0_c / mean_clipped, w1_c / mean_clipped, w2_c / mean_clipped]

    # Verify our formula logic
    p_arr = np.array([p0, p1, p2])
    raw = 1.0 / np.sqrt(np.maximum(p_arr, eps))
    norm_w = raw / np.mean(raw)
    clipped = np.clip(norm_w, 0.5, 2.0)
    final_w = clipped / np.mean(clipped)

    for i in range(3):
        assert abs(final_w[i] - expected_final[i]) < 1e-6
        assert 0.3 <= final_w[i] <= 3.0  # normalized around 1.0
    assert abs(np.mean(final_w) - 1.0) < 1e-6


def test_m2_exact_operation_order_and_recorded_smoke_weights():
    """Confirms exact operation order and bit-level reproduction of smoke weights."""
    c0, c1, c2 = 2573977, 292708, 110145
    total = c0 + c1 + c2
    counts = np.array([c0, c1, c2], dtype=np.float64)
    p = counts / float(total)

    # 1. raw_weight = 1 / sqrt(max(p, 1e-12))
    raw = 1.0 / np.sqrt(np.maximum(p, 1e-12))
    # 2. normalized_weight = raw / mean(raw)
    norm = raw / float(np.mean(raw))
    # 3. clipped_weight = clip(norm, 0.5, 2.0)
    clipped = np.clip(norm, 0.5, 2.0)
    # 4. final_weight = clipped / mean(clipped)
    final = clipped / float(np.mean(clipped))

    expected_smoke = [0.474823, 0.960079, 1.565099]
    for i in range(3):
        assert abs(round(float(final[i]), 6) - expected_smoke[i]) < 1e-6


def test_m2_2_loss_finite_gradients_and_weight_application():
    weights = [0.6, 0.9, 1.5]  # sum/3 = 1.0
    loss_fn = RobustSourceLoss(class_weights=weights)

    logits = torch.randn(4, 3, 32, 32, requires_grad=True)
    target = torch.randint(0, 3, (4, 32, 32), dtype=torch.long)
    target[0, :5, :5] = 3  # some ignore index pixels

    loss, comps = loss_fn(logits, target, return_components=True)
    assert torch.isfinite(loss)
    assert comps["loss_total"] > 0
    assert comps["loss_ce"] > 0
    assert comps["loss_dice"] > 0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_m2_3_forbidden_event_fails_closed():
    with pytest.raises(ContractViolationError, match="FATAL: Attempted to compute class weights"):
        compute_training_class_weights(
            train_samples_metadata=[],
            train_event_ids=[279],  # forbidden event
        )


def test_m2_4_unauthorized_event_fails_closed():
    with pytest.raises(ContractViolationError, match="Training events contain unauthorized IDs"):
        compute_training_class_weights(
            train_samples_metadata=[],
            train_event_ids=[99999],  # unknown event
        )


def test_m2_5_unweighted_r0_loss_bit_exact_parity():
    # RobustSourceLoss with class_weights=None must match ControlledSourceLoss exactly
    ctrl_loss = ControlledSourceLoss()
    robust_r0 = RobustSourceLoss(class_weights=None)

    torch.manual_seed(42)
    logits = torch.randn(2, 3, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16), dtype=torch.long)

    l1 = ctrl_loss(logits, target)
    l2 = robust_r0(logits, target)
    assert torch.equal(l1, l2), "Unweighted RobustSourceLoss must match ControlledSourceLoss bit-for-bit."


# --- Group 3: Component M3 (Synchronized Augmentation) Tests ---

def test_m3_1_synchronized_geometry_pre_post_label_mask():
    aug = SynchronizedPairedSARAugmentation()
    # Create distinct asymmetric patterns
    H, W = 16, 16
    pre = np.arange(2 * H * W, dtype=np.float32).reshape(2, H, W)
    post = pre * 1.5 + 2.0
    label = np.zeros((H, W), dtype=np.int64)
    label[2:5, 3:7] = 2  # rectangle patch
    mask = np.ones((H, W), dtype=np.uint8)
    mask[0:2, :] = 0  # invalid top rows

    # Force a non-trivial transform: hflip=True, vflip=True, rot_k=1
    params = {"hflip": True, "vflip": True, "rot_k": 1, "gain": 1.0, "sample_seed": 42}
    pre_aug = aug.apply_geometry(pre.copy(), params)
    post_aug = aug.apply_geometry(post.copy(), params)
    lbl_aug = aug.apply_geometry(label.copy(), params)
    mask_aug = aug.apply_geometry(mask.copy(), params)

    # Check shapes
    assert pre_aug.shape == (2, H, W)
    assert post_aug.shape == (2, H, W)
    assert lbl_aug.shape == (H, W)
    assert mask_aug.shape == (H, W)

    # Check relation between pre and post is preserved
    assert np.allclose(post_aug, pre_aug * 1.5 + 2.0)

    # Check label pixel count is strictly preserved
    assert (lbl_aug == 2).sum() == (label == 2).sum()
    assert (mask_aug == 0).sum() == (mask == 0).sum()


def test_m3_2_common_mode_radiometric_gain():
    aug = SynchronizedPairedSARAugmentation()
    pre = np.ones((2, 8, 8), dtype=np.float32)
    post = np.ones((2, 8, 8), dtype=np.float32) * 2.0
    label = np.ones((8, 8), dtype=np.int64)
    mask = np.ones((8, 8), dtype=np.uint8)

    pre_aug, post_aug, lbl_aug, mask_aug, params = aug(pre, post, label, mask, sample_seed=1234)
    gain = params["gain"]

    assert 0.95 <= gain <= 1.05
    # Both pre and post scaled by exact same gain
    assert np.allclose(pre_aug, pre * gain)
    assert np.allclose(post_aug, post * gain)
    # Both channels (VV and VH) scaled by exact same gain
    assert np.allclose(pre_aug[0], pre_aug[1])
    # Label and mask values completely unaffected by radiometric gain
    assert np.array_equal(lbl_aug, label)
    assert np.array_equal(mask_aug, mask)


def test_m3_conservative_uniform_gain_distribution():
    """Verifies that M3 radiometric gain follows Uniform(0.95, 1.05) conservative transform."""
    aug = SynchronizedPairedSARAugmentation(min_gain=0.95, max_gain=1.05)
    gains = []
    for s in range(1000):
        params = aug.sample_parameters(sample_seed=s)
        g = params["gain"]
        assert 0.95 <= g <= 1.05, f"Gain {g} outside approved [0.95, 1.05] interval!"
        gains.append(g)

    mean_gain = float(np.mean(gains))
    assert 0.99 <= mean_gain <= 1.01, f"Expected mean gain near 1.0, got {mean_gain}"


def test_m3_3_deterministic_sample_seed():
    aug = SynchronizedPairedSARAugmentation()
    p1 = aug.sample_parameters(sample_seed=98765)
    p2 = aug.sample_parameters(sample_seed=98765)
    assert p1 == p2, "Identical sample_seed must yield identical augmentation parameters."

    p3 = aug.sample_parameters(sample_seed=56789)
    assert p1 != p3, "Different sample_seed must yield different parameters."


def test_m3_4_label_values_and_ignore_preservation():
    aug = SynchronizedPairedSARAugmentation()
    raw_lbl = np.array([[0, 1], [2, 3]], dtype=np.int64)
    raw_mna = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    pre = np.ones((2, 2, 2), dtype=np.float32)
    post = np.ones((2, 2, 2), dtype=np.float32)

    for s in range(50):
        _, _, aug_l, aug_m, _ = aug(pre, post, raw_lbl, raw_mna, sample_seed=s)
        assert set(np.unique(aug_l)).issubset({0, 1, 2, 3})
        assert set(np.unique(aug_m)).issubset({0, 1})


# --- Group 4: R0 Parity Verification Tests ---

def test_r0_parity_synthetic_end_to_end():
    """Compares RST2 matched-R0 path against RST3 runner with M1/M2/M3 all disabled."""
    seed = 42
    seed_everything(seed)

    # 1. Models initialized with exact same seed
    torch.manual_seed(seed)
    model_rst2 = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)
    torch.manual_seed(seed)
    model_rst3 = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16)

    assert_states_equal(model_rst2, model_rst3)

    # 2. Losses: ControlledSourceLoss (RST2) vs RobustSourceLoss(class_weights=None) (RST3)
    loss_rst2 = ControlledSourceLoss()
    loss_rst3 = RobustSourceLoss(class_weights=None)

    # 3. Optimizers
    opt_rst2 = torch.optim.Adam(model_rst2.parameters(), lr=1e-3)
    opt_rst3 = torch.optim.Adam(model_rst3.parameters(), lr=1e-3)

    # 4. Forward + Backward + Step with synthetic inputs
    torch.manual_seed(seed + 10)
    xA = torch.randn(2, 2, 32, 32)
    xB = torch.randn(2, 2, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32), dtype=torch.long)

    # RST2 forward
    out2 = model_rst2(xA, xB)
    l2 = loss_rst2(out2, target)
    l2.backward()
    opt_rst2.step()

    # RST3 forward
    out3 = model_rst3(xA, xB)
    l3 = loss_rst3(out3, target)
    l3.backward()
    opt_rst3.step()

    # Assert bit-exact logits, loss, and updated parameters
    assert torch.equal(out2, out3), "Logits must be bit-exact equal."
    assert torch.equal(l2, l3), "Loss must be bit-exact equal."
    assert_states_equal(model_rst2, model_rst3)


def test_r0_parity_real_source_dataset_sample():
    """Verifies that RSTR0SourceDataset produces bit-identical samples with augmenter=None."""
    # Build dataset without augmenter (R0)
    ds_r0 = RSTR0SourceDataset(allowed_event_ids=[273], max_samples_cap=3, augmenter=None, is_training=False)
    # Build dataset with augmenter=None, is_training=True (R0 training)
    ds_r0_train = RSTR0SourceDataset(allowed_event_ids=[273], max_samples_cap=3, augmenter=None, is_training=True)

    for i in range(len(ds_r0)):
        s1 = ds_r0[i]
        s2 = ds_r0_train[i]
        assert s1["sample_id"] == s2["sample_id"]
        assert torch.equal(s1["xA"], s2["xA"])
        assert torch.equal(s1["xB"], s2["xB"])
        assert torch.equal(s1["label"], s2["label"])
        assert torch.equal(s1["valid_mask"], s2["valid_mask"])


# --- Group 5: Recipe Staircase & Configuration Validation ---

def test_staircase_recipe_contract_validation(tmp_path):
    from src.robust_source_training.runner import RSTExecutionEngine

    # Invalid recipe: M2 without M1
    bad_cfg = {
        "run_id": "test_invalid_recipe",
        "launch_authorized": True,
        "recipe_id": "R1",
        "components": {
            "m1_event_balance": False,  # Mismatch: R1 requires M1=True!
            "m2_flood_aware_loss": False,
            "m3_paired_augmentation": False,
        },
        "partition": {"train_event_ids": [273], "evaluation_event_ids": [130]},
    }
    cfg_p = tmp_path / "bad_recipe.json"
    cfg_p.write_text(json.dumps(bad_cfg))

    engine = RSTExecutionEngine(cfg_p, repo_root=REPO_ROOT)
    with pytest.raises(ContractViolationError, match="Recipe 'R1' switches mismatch"):
        engine.execute()


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
    print(f"Running {len(test_funcs)} RST3 component & parity tests...")

    passed = 0
    failed = 0
    for name, func in test_funcs:
        sig = inspect.signature(func)
        tmp_dir = None
        try:
            if "tmp_path" in sig.parameters:
                tmp_dir = tempfile.mkdtemp(prefix="rst3_test_")
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
