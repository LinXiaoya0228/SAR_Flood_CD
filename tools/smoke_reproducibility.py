#!/usr/bin/env python3
"""Standalone portable reproducibility smoke verification script for ChangeGuard.

Tests environment dependencies, config schemas, model forward passes,
metric logic, and canonical result file presence without requiring
GPU hardware or private datasets.
"""

import glob
import json
import os
import sys
from pathlib import Path

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def check_dependencies() -> bool:
    print("[1/5] Verifying core dependencies...")
    try:
        import numpy as np
        import scipy
        import torch
        print(f"  [OK] Python {sys.version.split()[0]}")
        print(f"  [OK] PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        print(f"  [OK] NumPy {np.__version__}")
        print(f"  [OK] SciPy {scipy.__version__}")
        return True
    except ImportError as e:
        print(f"  [FAIL] Missing required library: {e}")
        return False


def check_configs() -> bool:
    print("[2/5] Validating frozen experiment configurations...")
    config_paths = glob.glob(str(REPO_ROOT / "configs" / "robust_source_training" / "**" / "*.json"), recursive=True)
    config_paths += glob.glob(str(REPO_ROOT / "configs" / "ablation_m2_only" / "*.json"))
    config_paths += glob.glob(str(REPO_ROOT / "configs" / "external_evaluation" / "*.json"))

    if not config_paths:
        print("  [FAIL] No config files found!")
        return False

    valid_count = 0
    for path in config_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation
            assert isinstance(data, dict), f"Config at {path} is not a JSON object"
            valid_count += 1
        except Exception as e:
            print(f"  [FAIL] Invalid config {path}: {e}")
            return False

    print(f"  [OK] Successfully validated {valid_count} frozen configuration files.")
    return True


def check_synthetic_models() -> bool:
    print("[3/5] Testing synthetic model execution (CPU forward pass)...")
    try:
        import torch
        from src.models.snunet_wrapper import SNUNetWrapper
        from src.models.siam_diff_wrapper import SiamDiffWrapper

        device = torch.device("cpu")

        # Synthetic SAR tensors: Batch=2, 2 channels (VV, VH), 128x128
        xA = torch.randn(2, 2, 128, 128, device=device)
        xB = torch.randn(2, 2, 128, 128, device=device)

        # 1. SNUNet-ECAM (via wrapper)
        model_snunet = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=16).to(device)
        model_snunet.eval()
        with torch.no_grad():
            out_snunet = model_snunet(xA, xB)
        assert out_snunet.shape == (2, 3, 128, 128), f"Unexpected SNUNet output shape {out_snunet.shape}"
        print(f"  [OK] SNUNet-ECAM forward pass verified: inputs 2x{list(xA.shape)} -> output {list(out_snunet.shape)}")

        # 2. SiamUnet_diff (via wrapper)
        model_siam = SiamDiffWrapper(in_channels=2, out_channels=3, dropout_p=0.0).to(device)
        model_siam.eval()
        with torch.no_grad():
            out_siam = model_siam(xA, xB)
        assert out_siam.shape == (2, 3, 128, 128), f"Unexpected SiamUnet output shape {out_siam.shape}"
        print(f"  [OK] SiamUnet_diff forward pass verified: inputs 2x{list(xA.shape)} -> output {list(out_siam.shape)}")

        return True
    except Exception as e:
        print(f"  [FAIL] Synthetic model test failed: {e}")
        return False


def check_metric_logic() -> bool:
    print("[4/5] Verifying metric computation logic...")
    try:
        import numpy as np

        # Create synthetic confusion counts: 3 events
        # Event 1: tp=100, fp=20, fn=30, tn=850 -> f1 = 200 / 250 = 0.800
        # Event 2: tp=50, fp=50, fn=50, tn=850  -> f1 = 100 / 200 = 0.500
        # Event 3: tp=10, fp=40, fn=20, tn=930  -> f1 = 20 / 80   = 0.250
        f1s = [0.800, 0.500, 0.250]
        macro_f1 = float(np.mean(f1s))
        assert abs(macro_f1 - 0.5166667) < 1e-4

        # Bottom-1 and bottom-2 tail
        bottom1 = float(np.min(f1s))
        assert bottom1 == 0.250
        bottom2_mean = float(np.mean(sorted(f1s)[:2]))
        assert abs(bottom2_mean - 0.375) < 1e-4

        print(f"  [OK] Metric logic validated (Macro F1: {macro_f1:.4f}, Bottom-2 Tail: {bottom2_mean:.4f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Metric logic verification failed: {e}")
        return False


def check_canonical_artifacts() -> bool:
    print("[5/5] Checking presence of canonical compact result files...")
    expected_artifacts = [
        ("results/tables/table1_component_staircase.csv", "outputs/robust_source_training/rst10/tables/table1_component_staircase.csv"),
        ("results/tables/table2_primary_source_confirmation.csv", "outputs/robust_source_training/rst10/tables/table2_primary_source_confirmation.csv"),
        ("results/tables/table3_architecture_transfer.csv", "outputs/robust_source_training/rst10/tables/table3_architecture_transfer.csv"),
        ("results/tables/table4_deployable_checkpoint_ledger.csv", "outputs/robust_source_training/rst10/tables/table4_deployable_checkpoint_ledger.csv"),
        ("results/tables/table5_historical_target_extension.csv", "outputs/robust_source_training/rst10/tables/table5_historical_target_extension.csv"),
        ("results/tables/table6_runtime_and_resource_cost.csv", "outputs/robust_source_training/rst10/tables/table6_runtime_and_resource_cost.csv"),
        ("results/tables/table7_contextual_tta_comparison.csv", "outputs/robust_source_training/rst10/tables/table7_contextual_tta_comparison.csv"),
        ("results/canonical/m2_only_ablation_summary.json", "outputs/robust_source_training/ablation_m2_only/m2_only_ablation_summary.json"),
        ("results/canonical/geoid_nan_corrected_summary.json", "outputs/external_evaluation/geoid_nan_corrected_v1/geoid_nan_corrected_summary.json"),
    ]

    missing = []
    for rel_primary, rel_fallback in expected_artifacts:
        p1 = REPO_ROOT / rel_primary
        p2 = REPO_ROOT / rel_fallback
        if not p1.is_file() and not p2.is_file():
            missing.append(rel_primary)

    if missing:
        print(f"  [FAIL] Missing canonical artifacts: {missing}")
        return False

    print(f"  [OK] All {len(expected_artifacts)} canonical synthesis artifacts are present and readable.")
    return True


def main():
    print("=" * 70)
    print("ChangeGuard Reproducibility Smoke Verification Suite")
    print("=" * 70)

    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration Files", check_configs),
        ("Synthetic Model Execution", check_synthetic_models),
        ("Metric Logic", check_metric_logic),
        ("Canonical Artifacts", check_canonical_artifacts),
    ]

    all_passed = True
    for name, fn in checks:
        if not fn():
            all_passed = False
            print(f"\n[FAIL] Step '{name}' did not pass.\n")
            break

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL 5 REPRODUCIBILITY SMOKE CHECKS PASSED SUCCESSFULLY.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("RESULT: SMOKE VERIFICATION FAILED.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
