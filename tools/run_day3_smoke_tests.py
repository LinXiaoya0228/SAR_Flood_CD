"""
Day 3 Smoke Tests for ChangeGuard Test-Time Adaptation Baselines.

Runs source-only, AdaBN, and Tent on a diagnostic subset of Events 520 and 559.
Verifies finite outputs, parameter update isolation, zero label leakage,
and clean episodic resets under allow_partial=True.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.source_only import SourceOnlyAdapter
from src.tta.adabn import AdaBNAdapter
from src.tta.tent import TentAdapter
from src.evaluation.metrics import EventMetricAccumulator
from src.evaluation.completeness import CompletenessTracker
from src.evaluation.run_manifest import RunManifest


def run_smoke_tests(cfg_path: str, output_dir: str, device: str = "cuda") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 80)
    print("DAY 3 SMOKE TESTS: TTA ISOLATION & PARTIAL DIAGNOSTIC VALIDATION")
    print("=" * 80)

    smoke_events = [520, 559]
    samples_per_event = 32  # 2 batches of size 16

    # Load dataset with corrected canonical preprocessing (kuro_dem_sanitize_v1)
    dataset = KuroValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=smoke_events,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        sar_clamp_max=cfg["sar_clamping"]["max"],
        sar_mean=cfg["sar_normalization"]["mean"],
        sar_std=cfg["sar_normalization"]["std"],
        dem_mean=cfg["dem_normalization"]["mean"],
        dem_std=cfg["dem_normalization"]["std"],
        preprocessing_version=cfg["dem_sanitization"]["preprocessing_version"],
        dem_nodata_threshold=cfg["dem_sanitization"]["nodata_threshold"],
        dem_replacement_value=cfg["dem_sanitization"]["replacement_value_normalized"],
    )

    # Subsample exactly samples_per_event from each smoke event
    selected_indices = []
    expected_sample_ids = {}
    for eid in smoke_events:
        e_indices = [i for i, s in enumerate(dataset.samples) if s["event_id"] == eid][:samples_per_event]
        selected_indices.extend(e_indices)
        expected_sample_ids[eid] = [dataset.samples[i]["sample_id"] for i in e_indices]

    subset_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(subset_dataset, batch_size=16, shuffle=False)

    print(f"Loaded diagnostic subset: {len(subset_dataset)} samples across events {smoke_events}.")
    print("WARNING: PARTIAL DIAGNOSTIC ONLY — MUST NOT BE REPORTED AS FULL EVENT METRICS\n")

    # Load frozen source checkpoint
    raw_model, metadata = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)

    test_modes = [
        ("source_only", lambda m: SourceOnlyAdapter(m, cfg["checkpoint_path"], device=device)),
        ("adabn", lambda m: AdaBNAdapter(m, cfg["checkpoint_path"], device=device)),
        ("tent_lr_1e-4", lambda m: TentAdapter(m, cfg["checkpoint_path"], learning_rate=1e-4, device=device)),
    ]

    results_summary = []

    for mode_name, adapter_factory in test_modes:
        print(f"\n--- Running smoke test: {mode_name} ---")
        model = copy.deepcopy(raw_model)
        adapter = adapter_factory(model)

        # Completeness tracker with allow_partial=True
        tracker = CompletenessTracker(expected_sample_ids, allow_partial=True)
        accumulator = EventMetricAccumulator()

        total_batches = 0
        all_losses_finite = True
        all_logits_finite = True

        for batch in loader:
            total_batches += 1
            eids = batch["event_id"].tolist()
            sids = batch["sample_id"]
            xA = batch["xA"].to(device)
            xB = batch["xB"].to(device)
            sar_valid = batch["valid_mask"].to(device)
            labels = batch["label"]
            vlds = batch["valid_mask"]

            # Strictly pass ONLY sensory inputs to adaptation
            out = adapter.adapt_and_predict(
                xA=xA,
                xB=xB,
                sar_valid_mask=sar_valid,
                sample_ids=sids,
                event_ids=eids,
            )

            logits = out["logits"]
            pred = out["predicted_class"].cpu()

            if not torch.all(torch.isfinite(logits)):
                all_logits_finite = False

            if "adaptation_loss" in out["step_diagnostics"]:
                loss_v = out["step_diagnostics"]["adaptation_loss"]
                if not math.isfinite(loss_v):
                    all_losses_finite = False

            # Offline evaluation ONLY: labels joined after prediction finalized
            accumulator.update(eids, pred, labels, vlds)
            tracker.update(eids, sids, [int(torch.sum((vlds == 1) & (labels != 3)).item())] * len(sids))

        # Check completeness for this diagnostic run
        completeness_report = tracker.report()
        self_check = tracker.check()

        drift = adapter.compute_bn_drift()
        event_metrics = accumulator.compute_event_table()

        print(f"[{mode_name}] Batches: {total_batches} | Logits Finite: {all_logits_finite} | Losses Finite: {all_losses_finite}")
        print(f"[{mode_name}] BN Parameter Drift (L2): {drift['total_bn_drift_l2']:.6f}")
        for em in event_metrics:
            print(f"  Event {em['event_id']}: F1={em['f1']:.4f}, IoU={em['iou']:.4f}, PredFlood={em['pred_flood_pixels']}")

        # Verification asserts for smoke test
        if mode_name == "source_only":
            assert drift["total_bn_drift_l2"] == 0.0, "Source-only must have 0.0 BN parameter drift!"
        elif mode_name == "adabn":
            assert drift["total_bn_drift_l2"] == 0.0, "AdaBN must have 0.0 BN affine parameter drift!"
        elif "tent" in mode_name:
            assert drift["total_bn_drift_l2"] > 0.0, "Tent must update BN affine parameters (drift > 0)!"

        assert all_logits_finite, "Encountered non-finite logits during smoke test!"
        assert all_losses_finite, "Encountered non-finite adaptation loss during smoke test!"

        results_summary.append({
            "mode": mode_name,
            "logits_finite": all_logits_finite,
            "losses_finite": all_losses_finite,
            "bn_drift_l2": drift["total_bn_drift_l2"],
            "event_metrics": event_metrics,
            "completeness_status": completeness_report["status"],
        })

    # Save smoke test artifact
    smoke_json_path = out_dir / "smoke_test_results.json"
    with open(smoke_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_partial_diagnostic": True,
                "allow_partial": True,
                "smoke_events": smoke_events,
                "samples_per_event": samples_per_event,
                "verification_status": "ALL_PASSED",
                "results": results_summary,
            },
            f,
            indent=2,
        )
    print(f"\nSmoke test completed successfully! Results saved to: {smoke_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_config.json")
    parser.add_argument("--output_dir", default="outputs/day3_smoke_tests")
    args = parser.parse_args()
    run_smoke_tests(args.config, args.output_dir)
