"""Unit tests for E10 evaluation engine and deferred label guard."""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.siam_diff_wrapper import SiamDiffWrapper
from src.changeguard.baseline_adapters import (
    E1SourceOnlyAdapter,
    E1MatchedBNNoUpdateAdapter,
    E1TentAdapter,
)
from src.changeguard.synthetic_shift import MultiplicativeBackscatterShift
from tools.run_e10_evaluations import E10Batch, load_adapter


class MockDataset:
    def __init__(self):
        self.samples = [{"sample_id": f"s_{i}", "event_id": 520} for i in range(10)]

    def __getitem__(self, idx):
        return {
            "sample_id": f"s_{idx}",
            "label": torch.randint(0, 3, (224, 224)),
        }


class TestE10EvaluationEngine(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.model = SiamDiffWrapper(in_channels=2, out_channels=3, dropout_p=0.2).to(self.device)

    def test_01_adapter_instantiations(self):
        """Verify that all three evaluated methods initialize properly for SiamDiffWrapper."""
        import copy
        model_source = copy.deepcopy(self.model)
        ad_source = load_adapter("source_only", model_source, device=self.device)
        self.assertIsInstance(ad_source, E1SourceOnlyAdapter)
        ad_source.reset_to_source()

        model_bn = copy.deepcopy(self.model)
        ad_bn = load_adapter("bn_no_update", model_bn, device=self.device)
        self.assertIsInstance(ad_bn, E1MatchedBNNoUpdateAdapter)
        # Test reset_to_source on bn_no_update
        ad_bn.reset_to_source()

        model_tent = copy.deepcopy(self.model)
        ad_tent = load_adapter("tent", model_tent, device=self.device)
        self.assertIsInstance(ad_tent, E1TentAdapter)
        self.assertEqual(len(ad_tent.optimizer.param_groups[0]["params"]), 38)
        ad_tent.reset_to_source()

    def test_02_deferred_label_guard(self):
        """Accessing labels before persistence must raise PermissionError."""
        mock_ds = MockDataset()
        batch = E10Batch(
            sample_ids=["s_0", "s_1"],
            xA=torch.randn(2, 2, 224, 224),
            xB=torch.randn(2, 2, 224, 224),
            valid_mask=torch.ones(2, 224, 224, dtype=torch.uint8),
            dataset=mock_ds,
            sample_indices=[0, 1],
        )

        with self.assertRaises(PermissionError):
            batch.get_labels()

        # Mark persisted
        batch.mark_persisted()
        labels = batch.get_labels()
        self.assertEqual(labels.shape, (2, 224, 224))

    def test_03_synthetic_shift_in_adapter_step(self):
        """Applying synthetic shift prior to adapt_and_predict_step must run without error."""
        ad_tent = load_adapter("tent", self.model, device=self.device)
        shifter = MultiplicativeBackscatterShift(severity_factor=0.10)

        xA = torch.randn(2, 2, 224, 224)
        xB = torch.randn(2, 2, 224, 224)
        vm = torch.ones(2, 224, 224, dtype=torch.uint8)

        xA_shifted = shifter.transform_normalized_tensor(xA).to(self.device)
        xB_shifted = shifter.transform_normalized_tensor(xB).to(self.device)

        res = ad_tent.adapt_and_predict_step(
            xA=xA_shifted,
            xB=xB_shifted,
            valid_mask=vm.to(self.device),
            sample_ids=["s_0", "s_1"],
            event_id=520,
        )

        pred = res["primary_prediction"]["pred_cand_pre"]
        self.assertEqual(pred.shape, (2, 224, 224))
        self.assertTrue(torch.isfinite(res["primary_prediction"]["probs_cand_pre"]).all())


if __name__ == "__main__":
    unittest.main()

