"""Memory and microbatch smoke test on GPU 0.

Evaluates physical microbatch sizes [2, 4, 8, 16] under AMP FP16
for controlled Model A and Model B. Records peak VRAM allocation,
peak VRAM reservation, execution throughput, and verifies accumulation divisibility (32 % b == 0).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also ensure KuroSiwo is on path for SNUNet
KURO_ROOT = REPO_ROOT.parent / "third_party/KuroSiwo"
if str(KURO_ROOT) not in sys.path:
    sys.path.insert(0, str(KURO_ROOT))

from src.models.snunet_wrapper import SNUNetWrapper
from src.evaluation.losses import ControlledSourceLoss


def benchmark_microbatch(
    in_channels: int,
    microbatch: int,
    effective_batch: int = 32,
    num_optimizer_steps: int = 3,
    device: str = "cuda:0",
) -> dict:
    """Benchmark a single microbatch configuration."""
    assert effective_batch % microbatch == 0, f"effective_batch ({effective_batch}) must be divisible by microbatch ({microbatch})"
    accum_steps = effective_batch // microbatch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Instantiate model and loss
    model = SNUNetWrapper(in_channels=in_channels, out_ch=3, base_channel=32).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    loss_fn = ControlledSourceLoss()

    # Synthetic batch inputs (224x224)
    xA = torch.randn(microbatch, in_channels, 224, 224, device=device)
    xB = torch.randn(microbatch, in_channels, 224, 224, device=device)
    target = torch.randint(0, 3, (microbatch, 224, 224), device=device)

    # Warmup
    for _ in range(2):
        with autocast():
            logits = model(xA, xB)
            loss = loss_fn(logits, target) / accum_steps
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # Timed run
    total_microbatches = num_optimizer_steps * accum_steps
    for step in range(num_optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for acc in range(accum_steps):
            with autocast():
                logits = model(xA, xB)
                loss = loss_fn(logits, target) / accum_steps
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    total_samples = total_microbatches * microbatch
    throughput = total_samples / elapsed

    del model, optimizer, scaler, loss_fn, xA, xB, target
    torch.cuda.empty_cache()

    return {
        "microbatch_size": microbatch,
        "gradient_accumulation_steps": accum_steps,
        "effective_batch_size": effective_batch,
        "peak_allocated_mib": round(allocated_mib, 2),
        "peak_reserved_mib": round(reserved_mib, 2),
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(throughput, 2),
        "status": "PASS",
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available! Must run on GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"Running Memory Smoke Test on: {device_name} (Total VRAM: {total_vram_mib:.0f} MiB)")

    candidate_microbatches = [2, 4, 8, 16]
    results = {
        "device": device_name,
        "total_vram_mib": round(total_vram_mib, 2),
        "effective_batch_size": 32,
        "model_a_results": [],
        "model_b_results": [],
    }

    print("\n--- Benchmarking Model A (2 channels: [VV, VH]) ---")
    for mb in candidate_microbatches:
        print(f"Testing Microbatch {mb} (Accumulation {32 // mb})...", end=" ", flush=True)
        res = benchmark_microbatch(in_channels=2, microbatch=mb)
        print(f"Allocated: {res['peak_allocated_mib']} MiB, Reserved: {res['peak_reserved_mib']} MiB, Throughput: {res['samples_per_second']} samples/s")
        results["model_a_results"].append(res)

    print("\n--- Benchmarking Model B (3 channels: [VV, VH, slope]) ---")
    for mb in candidate_microbatches:
        print(f"Testing Microbatch {mb} (Accumulation {32 // mb})...", end=" ", flush=True)
        res = benchmark_microbatch(in_channels=3, microbatch=mb)
        print(f"Allocated: {res['peak_allocated_mib']} MiB, Reserved: {res['peak_reserved_mib']} MiB, Throughput: {res['samples_per_second']} samples/s")
        results["model_b_results"].append(res)

    output_dir = REPO_ROOT / "outputs/week1"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "memory_smoke_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
