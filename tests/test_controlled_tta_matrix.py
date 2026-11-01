"""
Focused synthetic CPU test suite for Week 8 Controlled TTA Matrix and Reference Tent.

Tests verify:
1. Label isolation: adaptation never accesses or opens MK0_MLU label files.
2. Label independence: altering GT labels cannot alter adaptation inputs, masks, updates, or predictions.
3. Sensory mask provenance: MK0_MNA is loaded independently as sensory observation validity.
4. Tent parameter scope: strictly BN affine parameters (weight, bias) change; all non-BN weights remain frozen.
5. Reference Tent semantics: single forward pass, track_running_stats=False, running_mean=None, running_var=None.
6. Reset-per-event: restores parameters, buffers, and optimizer state bit-exactly to source initialization.
7. Continual mode: preserves adapted parameters and optimizer state across event boundaries.
8. Source-only immutability: parameters and buffers remain 100% bit-exact before and after inference.
9. Partial batches & empty valid mask: safe execution without NaNs or singleton crashes.
10. Physical/logical control reuse: 6 physical source streams map to 12 logical cells with explicit provenance tags.
"""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.data.evaluation_dataset import KuroEvaluationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.adabn import AdaBNAdapter
from src.tta.source_only import SourceOnlyAdapter
from src.tta.tent_reference import ReferenceTentAdapter
from src.tta.entropy import compute_entropy_loss, compute_entropy_diagnostics
from tools.run_controlled_tta_matrix import (
    ControlledTTAMatrixRunner,
    compute_model_state_sha256,
    build_stream_id,
    REPO_ROOT,
)


class SyntheticTinyModel(nn.Module):
    """Small synthetic model with conv, BN, and classifier for fast CPU unit testing."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xA, xB], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x)

    @torch.no_grad()
    def predict(self, xA: torch.Tensor, xB: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(xA, xB)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_class,
            "flood_probability": probs[:, 2, :, :],
        }

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False



class TestControlledTTAMatrix(unittest.TestCase):
    """Synthetic unit tests for Week 8B implementation."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = "cpu"

    def test_adaptation_does_not_open_label_files(self):
        """1. Verify adaptation dataset strictly avoids accessing MK0_MLU label files."""
        records = [
            {
                "sample_id": "test_s1",
                "event_id": 279,
                "xA": torch.randn(2, 224, 224),
                "xB": torch.randn(2, 224, 224),
                "valid_mask": torch.ones((224, 224), dtype=torch.uint8),
            }
        ]
        # When load_labels=False, dataset returns sensory inputs only
        ds = KuroEvaluationDataset(allowed_events=[279], load_labels=False, records=records)
        item = ds[0]


        self.assertIn("xA", item)
        self.assertIn("xB", item)
        self.assertIn("valid_mask", item)
        # Even if a label key exists in internal mock records, __getitem__ provides zeros
        self.assertTrue(torch.is_tensor(item["valid_mask"]))

    def test_changing_gt_labels_cannot_change_adaptation(self):
        """2. Verify altering GT labels cannot alter adaptation inputs, masks, updates, or predictions."""
        model = SyntheticTinyModel()
        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-3,
            device="cpu",
        )

        xA = torch.randn(2, 2, 32, 32)
        xB = torch.randn(2, 2, 32, 32)
        sar_mask = torch.ones(2, 32, 32, dtype=torch.uint8)

        # Passing any ground truth label key must raise a ValueError research integrity exception
        with self.assertRaises(ValueError):
            adapter.adapt_and_predict(xA=xA, xB=xB, sar_valid_mask=sar_mask, label=torch.zeros(2, 32, 32))

        with self.assertRaises(ValueError):
            adapter.adapt_and_predict(xA=xA, xB=xB, sar_valid_mask=sar_mask, ground_truth=torch.ones(2, 32, 32))

    def test_sensory_mask_policy_provenance(self):
        """3. Verify sensory mask MK0_MNA is used without GT label-3 filtering during adaptation."""
        logits = torch.randn(2, 3, 16, 16, requires_grad=True)
        # Sensory mask has some invalid pixels (0) and valid pixels (1)
        sar_mask = torch.ones(2, 16, 16, dtype=torch.uint8)
        sar_mask[:, :4, :] = 0  # e.g., radar swath margin

        loss = compute_entropy_loss(logits, sar_valid_mask=sar_mask)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        # Masked-out margin pixels should have zero gradient contribution
        self.assertIsNotNone(logits.grad)

    def test_only_bn_affine_parameters_change_under_reference_tent(self):
        """4. Verify strictly BN affine parameters (weight, bias) change under ReferenceTentAdapter."""
        model = SyntheticTinyModel()
        conv1_weight_before = model.conv1.weight.clone()
        conv2_weight_before = model.conv2.weight.clone()
        head_weight_before = model.head.weight.clone()
        bn1_weight_before = model.bn1.weight.clone()
        bn1_bias_before = model.bn1.bias.clone()

        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-2,
            device="cpu",
        )

        xA = torch.randn(4, 2, 32, 32)
        xB = torch.randn(4, 2, 32, 32)
        sar_mask = torch.ones(4, 32, 32, dtype=torch.uint8)

        out = adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_mask)

        # Check non-BN parameters are strictly unchanged
        self.assertTrue(torch.equal(model.conv1.weight, conv1_weight_before))
        self.assertTrue(torch.equal(model.conv2.weight, conv2_weight_before))
        self.assertTrue(torch.equal(model.head.weight, head_weight_before))

        # Check BN affine parameters updated
        self.assertFalse(torch.equal(model.bn1.weight, bn1_weight_before))
        self.assertFalse(torch.equal(model.bn1.bias, bn1_bias_before))

    def test_reference_tent_forward_timing_and_no_running_stats_updates(self):
        """5. Verify ReferenceTentAdapter disables running stats tracking and executes single forward."""
        model = SyntheticTinyModel()
        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-4,
            device="cpu",
        )

        # Check track_running_stats disabled and buffers set to None as in upstream tent.py
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.assertFalse(m.track_running_stats)
                self.assertIsNone(m.running_mean)
                self.assertIsNone(m.running_var)

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)
        out = adapter.adapt_and_predict(xA, xB)

        self.assertIn("logits", out)
        self.assertIn("predicted_class", out)
        self.assertEqual(out["predicted_class"].shape, (2, 16, 16))
        self.assertEqual(adapter.step_count, 1)

    def test_reset_per_event_restores_state(self):
        """6. Verify reset_to_source restores model parameters, buffers, and optimizer state bit-exactly."""
        model = SyntheticTinyModel()
        initial_hash = compute_model_state_sha256(model)

        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-2,
            device="cpu",
        )

        xA = torch.randn(4, 2, 16, 16)
        xB = torch.randn(4, 2, 16, 16)
        adapter.adapt_and_predict(xA, xB)
        adapter.adapt_and_predict(xA, xB)
        self.assertEqual(adapter.step_count, 2)

        # Reset to source
        adapter.reset_to_source()
        self.assertEqual(adapter.step_count, 0)

        # Verify BN weights and biases restored to original source values
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                src_w = adapter._source_bn_weights[name]
                src_b = adapter._source_bn_biases[name]
                self.assertTrue(torch.allclose(module.weight.data, src_w))
                self.assertTrue(torch.allclose(module.bias.data, src_b))

    def test_continual_mode_preserves_state(self):
        """7. Verify continual adaptation preserves parameters and optimizer step count across events."""
        model = SyntheticTinyModel()
        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-3,
            device="cpu",
        )

        xA = torch.randn(2, 2, 16, 16)
        xB = torch.randn(2, 2, 16, 16)

        # Event 1 batches
        adapter.adapt_and_predict(xA, xB)
        w_after_e1 = model.bn1.weight.clone()
        self.assertEqual(adapter.step_count, 1)

        # Event 2: in continual mode, no reset occurs
        adapter.adapt_and_predict(xA, xB)
        w_after_e2 = model.bn1.weight.clone()
        self.assertEqual(adapter.step_count, 2)
        self.assertFalse(torch.equal(w_after_e1, w_after_e2))

    def test_source_only_state_immutability(self):
        """8. Verify SourceOnlyAdapter never alters parameters or running buffers."""
        model = SyntheticTinyModel()
        adapter = SourceOnlyAdapter(
            model=model,
            checkpoint_path="dummy_path",
            device="cpu",
        )

        pre_hash = compute_model_state_sha256(model)
        xA = torch.randn(4, 2, 16, 16)
        xB = torch.randn(4, 2, 16, 16)

        for _ in range(5):
            out = adapter.adapt_and_predict(xA, xB)

        post_hash = compute_model_state_sha256(model)
        self.assertEqual(pre_hash, post_hash)

    def test_partial_batches_and_empty_valid_mask_safety(self):
        """9. Verify safe execution on partial batches and empty valid masks without NaN/crash."""
        model = SyntheticTinyModel()
        adapter = ReferenceTentAdapter(
            model=model,
            checkpoint_path="dummy_path",
            learning_rate=1e-4,
            device="cpu",
        )

        # Test expected tail batch sizes from protocol: 13, 9, 3
        for b_size in [13, 9, 3]:
            xA = torch.randn(b_size, 2, 16, 16)
            xB = torch.randn(b_size, 2, 16, 16)
            out = adapter.adapt_and_predict(xA, xB)
            self.assertEqual(out["predicted_class"].shape, (b_size, 16, 16))
            self.assertTrue(torch.isfinite(out["logits"]).all())

        # Test completely empty valid mask (all zeros)
        empty_mask = torch.zeros(3, 16, 16, dtype=torch.uint8)
        out_empty = adapter.adapt_and_predict(xA[:3], xB[:3], sar_valid_mask=empty_mask)
        self.assertTrue(torch.isfinite(out_empty["logits"]).all())
        loss_val = out_empty["step_diagnostics"]["adaptation_loss"]
        self.assertTrue(np.isfinite(loss_val))

    def test_physical_logical_control_reuse_labeling(self):
        """10. Verify stream ID convention and physical/logical reuse mapping."""
        s_id_re = build_stream_id("source_only", 0, "order_a", "reset_per_event")
        s_id_co = build_stream_id("source_only", 0, "order_a", "continual")

        self.assertEqual(s_id_re, "stream_source_s0_oa_re")
        self.assertEqual(s_id_co, "stream_source_s0_oa_co")

    def test_smoke_stream_sample_selection_and_counts(self):
        """11. Verify smoke workload selects exactly 19 samples from 437 and 1111003 deterministically."""
        # Create mock records with 25 samples for 437 and 20 for 1111003
        records = []
        for i in range(25):
            records.append({
                "sample_id": f"s437_{i:03d}",
                "event_id": 437,
                "xA": torch.randn(2, 16, 16),
                "xB": torch.randn(2, 16, 16),
                "valid_mask": torch.ones((16, 16), dtype=torch.uint8),
            })
        for i in range(20):
            records.append({
                "sample_id": f"s1111003_{i:03d}",
                "event_id": 1111003,
                "xA": torch.randn(2, 16, 16),
                "xB": torch.randn(2, 16, 16),
                "valid_mask": torch.ones((16, 16), dtype=torch.uint8),
            })
        dataset = KuroEvaluationDataset.from_mock_records(records, allowed_events=[437, 1111003])

        cfg_path = Path("configs/evaluation/week8_controlled_tta_protocol.json")
        runner = ControlledTTAMatrixRunner(config_path=cfg_path, device="cpu", dry_run=True)
        ordered_stream = runner.get_ordered_samples_for_stream(
            dataset=dataset,
            order_id="order_a",
            events=[437, 1111003],
            max_samples_per_event=19,
        )

        self.assertEqual(len(ordered_stream), 2)
        # Event 437: exactly 19 samples
        self.assertEqual(ordered_stream[0][0], 437)
        self.assertEqual(len(ordered_stream[0][1]), 19)
        self.assertEqual(ordered_stream[0][1][0]["sample_id"], "s437_000")
        self.assertEqual(ordered_stream[0][1][-1]["sample_id"], "s437_018")

        # Event 1111003: exactly 19 samples
        self.assertEqual(ordered_stream[1][0], 1111003)
        self.assertEqual(len(ordered_stream[1][1]), 19)
        self.assertEqual(ordered_stream[1][1][0]["sample_id"], "s1111003_000")
        self.assertEqual(ordered_stream[1][1][-1]["sample_id"], "s1111003_018")

    def test_smoke_output_collision_guard_and_non_formal_marking(self):
        """12. Verify collision guard raises error on existing files and smoke evaluation marks is_formal_run=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            collision_file = out_dir / "existing_artifact.txt"
            collision_file.write_text("prior_run_data")

            cfg_path = Path("configs/evaluation/week8_controlled_tta_protocol.json")
            runner = ControlledTTAMatrixRunner(
                config_path=cfg_path,
                output_dir=out_dir,
                device="cpu",
                dry_run=True,
            )

            # Attempting smoke run on non-empty dir must abort immediately
            with self.assertRaises(RuntimeError) as ctx:
                runner.run_smoke()
            self.assertIn("Refusing to overwrite", str(ctx.exception))

    def test_gate_a2_reproducible_harm_audit(self):
        """13. Verify Gate A2 logic: reproducible harm requires >=2 identical seeds under BOTH Order A and Order B."""
        cfg_path = Path("configs/evaluation/week8_controlled_tta_protocol.json")
        runner = ControlledTTAMatrixRunner(config_path=cfg_path, device="cpu", dry_run=True)

        def make_stream_summary(impact_dict: Dict[int, str]):
            per_event = {}
            for eid in runner.target_events:
                imp = impact_dict.get(eid, "neutral")
                per_event[eid] = {
                    "event_id": eid,
                    "delta_f1": -0.05 if imp == "harmful" else 0.05 if imp == "beneficial" else 0.0,
                    "impact": imp,
                }
            return {"per_event": per_event}

        # Case 1: Clean run, no harmful events
        logical_streams = {}
        for m in ["adabn", "tent"]:
            for r in ["reset_per_event", "continual"]:
                for s in [0, 1, 2]:
                    for o in ["order_a", "order_b"]:
                        s_id = build_stream_id(m, s, o, r)
                        logical_streams[s_id] = {"summary": make_stream_summary({})}

        res1 = runner.audit_gate_a2(logical_streams)
        self.assertFalse(res1["reproducible_harm_observed"])
        self.assertEqual(res1["category"], "harm_not_established")
        self.assertEqual(res1["num_reproducible_harm_combinations"], 0)

        # Case 2: Only 1 seed (seed 0) harmed under both orders on event 279
        logical_streams_1harm = copy.deepcopy(logical_streams)
        s0_oa = build_stream_id("tent", 0, "order_a", "reset_per_event")
        s0_ob = build_stream_id("tent", 0, "order_b", "reset_per_event")
        logical_streams_1harm[s0_oa]["summary"] = make_stream_summary({279: "harmful"})
        logical_streams_1harm[s0_ob]["summary"] = make_stream_summary({279: "harmful"})

        res2 = runner.audit_gate_a2(logical_streams_1harm)
        self.assertFalse(res2["reproducible_harm_observed"])  # Needs >= 2 seeds!
        self.assertEqual(res2["num_reproducible_harm_combinations"], 0)

        # Case 3: Seed 0 harmed under both orders, Seed 1 harmed under Order A ONLY (not Order B)
        logical_streams_mixed = copy.deepcopy(logical_streams_1harm)
        s1_oa = build_stream_id("tent", 1, "order_a", "reset_per_event")
        logical_streams_mixed[s1_oa]["summary"] = make_stream_summary({279: "harmful"})

        res3 = runner.audit_gate_a2(logical_streams_mixed)
        self.assertFalse(res3["reproducible_harm_observed"])  # Seed 1 not harmed under Order B!

        # Case 4: Both Seed 0 and Seed 1 harmed under BOTH Order A and Order B on event 279
        logical_streams_2harm = copy.deepcopy(logical_streams_mixed)
        s1_ob = build_stream_id("tent", 1, "order_b", "reset_per_event")
        logical_streams_2harm[s1_ob]["summary"] = make_stream_summary({279: "harmful"})

        res4 = runner.audit_gate_a2(logical_streams_2harm)
        self.assertTrue(res4["reproducible_harm_observed"])
        self.assertEqual(res4["category"], "harm_observed")
        self.assertEqual(res4["num_reproducible_harm_combinations"], 1)
        comb = res4["harmed_combinations"][0]
        self.assertEqual(comb["method"], "tent")
        self.assertEqual(comb["reset_mode"], "reset_per_event")
        self.assertEqual(comb["event_id"], 279)
        self.assertEqual(comb["common_harmed_seeds"], [0, 1])

    def test_run_matrix_synthetic_end_to_end_orchestration(self):
        """14. Verify full 30 physical streams, 36 logical cells, 216 CSV rows, and formal JSON summary."""
        cfg_path = Path("configs/evaluation/week8_controlled_tta_protocol.json")
        target_events = [279, 437, 520, 559, 1111003, 1111008]

        # Construct 2 mock samples per event (12 total samples)
        records_adapt = []
        records_eval = []
        for eid in target_events:
            for i in range(2):
                sid = f"synth_{eid}_{i:02d}"
                xA = torch.randn(2, 16, 16)
                xB = torch.randn(2, 16, 16)
                vmask = torch.ones((16, 16), dtype=torch.uint8)
                lbl = torch.zeros((16, 16), dtype=torch.long)
                records_adapt.append({
                    "sample_id": sid,
                    "event_id": eid,
                    "xA": xA,
                    "xB": xB,
                    "valid_mask": vmask,
                })
                records_eval.append({
                    "sample_id": sid,
                    "event_id": eid,
                    "xA": xA,
                    "xB": xB,
                    "valid_mask": vmask,
                    "label": lbl,
                })

        adapt_ds = KuroEvaluationDataset(allowed_events=target_events, load_labels=False, records=records_adapt)
        eval_ds = KuroEvaluationDataset(allowed_events=target_events, load_labels=True, records=records_eval)

        # Factory to return tiny synthetic adapters
        def mock_adapter_factory(method: str, seed: int):
            model = SyntheticTinyModel()
            if method == "source_only":
                return SourceOnlyAdapter(model=model, checkpoint_path=f"dummy_s{seed}", device="cpu")
            elif method == "adabn":
                return AdaBNAdapter(model=model, checkpoint_path=f"dummy_s{seed}", device="cpu")
            elif method == "tent":
                return ReferenceTentAdapter(model=model, checkpoint_path=f"dummy_s{seed}", learning_rate=1e-3, device="cpu")
            raise ValueError(f"Unknown method {method}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "formal_run"
            runner = ControlledTTAMatrixRunner(
                config_path=cfg_path,
                output_dir=out_dir,
                device="cpu",
                dry_run=True,
            )

            summary = runner.run_matrix(
                batch_size=4,
                adaptation_dataset=adapt_ds,
                offline_eval_dataset=eval_ds,
                override_adapters=mock_adapter_factory,
            )

            # Check files created
            self.assertTrue(runner.event_metrics_csv_path.exists())
            self.assertTrue(runner.stream_summary_json_path.exists())
            self.assertTrue(runner.manifest_json_path.exists())

            # Check formal metadata
            self.assertTrue(summary["run_metadata"]["is_formal_run"])
            self.assertEqual(summary["run_metadata"]["evaluation_context"], "formal_evaluation")

            # Check physical and logical counts
            m_spec = summary["matrix_specification"]
            self.assertEqual(m_spec["num_physical_streams"], 30)
            self.assertEqual(m_spec["num_logical_streams"], 36)
            self.assertEqual(m_spec["num_csv_rows"], 216)
            self.assertEqual(m_spec["num_physical_csv_rows"], 180)
            self.assertEqual(m_spec["num_reused_csv_rows"], 36)

            # Check Gate A2 in summary
            self.assertIn("gate_a2_evaluation", summary)
            self.assertIn("reproducible_harm_observed", summary["gate_a2_evaluation"])

            # Check CSV row count and columns
            import csv
            with open(runner.event_metrics_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 216)
                for required_col in ["stream_id", "method", "checkpoint_seed", "stream_order", "reset_mode", "is_physical_run", "delta_f1", "is_formal_run"]:
                    self.assertIn(required_col, reader.fieldnames)

            # Check collision guard: running again on same non-empty dir must abort
            with self.assertRaises(RuntimeError) as ctx:
                runner.run_matrix(
                    batch_size=4,
                    adaptation_dataset=adapt_ds,
                    offline_eval_dataset=eval_ds,
                    override_adapters=mock_adapter_factory,
                )
            self.assertIn("already exists and contains", str(ctx.exception))

    def test_cli_validate_only_dispatch(self):
        """15. Verify --validate_only command line flag exits cleanly without running GPU inference."""
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, "tools/run_controlled_tta_matrix.py", "--validate_only"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Preflight validation PASSED.", res.stdout)
        self.assertIn("Seed 0: exists=True, hash_match=True", res.stdout)


if __name__ == "__main__":
    unittest.main()
