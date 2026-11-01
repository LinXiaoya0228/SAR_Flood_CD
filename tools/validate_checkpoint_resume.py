"""Validation script for checkpoint resume consistency (Run A vs Run B).

Verifies that:
- Run A: 10 continuous training steps without interruption.
- Run B: 5 steps -> save checkpoint -> reload from disk -> 5 steps.
- Bit-level / numerical equivalence between Run A and Run B parameter states and loss values.
"""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KURO_ROOT = REPO_ROOT.parent / "third_party/KuroSiwo"
if str(KURO_ROOT) not in sys.path:
    sys.path.insert(0, str(KURO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.training.source_trainer import SourceTrainer
from src.data.source_dataset import KuroSourceDataset


def run_resume_validation(device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    print(f"Starting Checkpoint Resume Validation on {device}...")
    torch.manual_seed(42)

    # Use actual KuroSourceDataset (split="train", first 64 samples)
    train_dataset = KuroSourceDataset(split="train", include_slope=False)
    # Take a small subset for quick exact reproduction
    indices = list(range(64))
    subset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=8, shuffle=False)

    # Initial model architecture
    torch.manual_seed(1234)
    model_init = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    initial_weights = copy.deepcopy(model_init.state_dict())

    # --- Run A: 10 continuous steps ---
    print("\n--- Executing Run A (10 continuous steps) ---")
    torch.manual_seed(1234)
    model_a = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    model_a.load_state_dict(copy.deepcopy(initial_weights))

    temp_dir_a = Path(tempfile.mkdtemp(prefix="run_a_"))
    trainer_a = SourceTrainer(
        model=model_a,
        output_dir=temp_dir_a,
        device=device,
        effective_batch_size=32,
        physical_microbatch_size=8,
        mixed_precision=(device != "cpu"),
        seed=1234,
    )
    metrics_a = trainer_a.train_steps(train_loader, num_optimizer_steps=10)

    # --- Run B: 5 steps -> save -> reload -> 5 steps ---
    print("\n--- Executing Run B (5 steps -> save -> reload -> 5 steps) ---")
    torch.manual_seed(1234)
    model_b = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    model_b.load_state_dict(copy.deepcopy(initial_weights))

    temp_dir_b = Path(tempfile.mkdtemp(prefix="run_b_"))
    trainer_b = SourceTrainer(
        model=model_b,
        output_dir=temp_dir_b,
        device=device,
        effective_batch_size=32,
        physical_microbatch_size=8,
        mixed_precision=(device != "cpu"),
        seed=1234,
    )
    metrics_b1 = trainer_b.train_steps(train_loader, num_optimizer_steps=5)
    ckpt_path = trainer_b.save_checkpoint("step5_checkpoint.pt")
    print(f"Saved intermediate Run B checkpoint at step 5: {ckpt_path}")

    # Create fresh model and trainer, reload
    torch.manual_seed(9999)  # intentionally different seed before loading
    model_b_resumed = SNUNetWrapper(in_channels=2, out_ch=3, base_channel=32)
    trainer_b_resumed = SourceTrainer(
        model=model_b_resumed,
        output_dir=temp_dir_b,
        device=device,
        effective_batch_size=32,
        physical_microbatch_size=8,
        mixed_precision=(device != "cpu"),
        seed=1234,
    )
    trainer_b_resumed.resume_from_checkpoint(ckpt_path)
    metrics_b2 = trainer_b_resumed.train_steps(train_loader, num_optimizer_steps=5)

    all_metrics_b = metrics_b1 + metrics_b2

    # Verification: Compare metrics step by step
    max_loss_diff = 0.0
    for i, (m_a, m_b) in enumerate(zip(metrics_a, all_metrics_b)):
        diff = abs(m_a["loss_total"] - m_b["loss_total"])
        max_loss_diff = max(max_loss_diff, diff)
        print(f"Step {i+1:02d} | Run A: {m_a['loss_total']:.6f} | Run B: {m_b['loss_total']:.6f} | Diff: {diff:.2e}")

    # Parameter comparison
    param_max_diff = 0.0
    for (name_a, p_a), (name_b, p_b) in zip(
        trainer_a.model.named_parameters(), trainer_b_resumed.model.named_parameters()
    ):
        diff = torch.max(torch.abs(p_a - p_b)).item()
        param_max_diff = max(param_max_diff, diff)

    print(f"\nFinal Parameter Max Difference: {param_max_diff:.2e}")
    print(f"Maximum Step Loss Difference: {max_loss_diff:.2e}")

    # Strict assertion: Under deterministic execution with same microbatch sequence,
    # loss difference must be within float32 tolerance (< 1e-4) and parameters aligned.
    assert max_loss_diff < 1e-4, f"Loss mismatch between Run A and Run B! {max_loss_diff}"
    assert param_max_diff < 1e-4, f"Parameter mismatch between Run A and Run B! {param_max_diff}"

    output_summary = {
        "device": device,
        "steps_total": 10,
        "interruption_step": 5,
        "max_step_loss_difference": float(max_loss_diff),
        "max_parameter_difference": float(param_max_diff),
        "run_a_final_loss": metrics_a[-1]["loss_total"],
        "run_b_final_loss": all_metrics_b[-1]["loss_total"],
        "status": "VALIDATED_PASS",
    }

    out_file = REPO_ROOT / "outputs/week1/checkpoint_resume_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\nCheckpoint Resume Validation PASSED! Saved report to {out_file}")
    return output_summary


if __name__ == "__main__":
    run_resume_validation()
