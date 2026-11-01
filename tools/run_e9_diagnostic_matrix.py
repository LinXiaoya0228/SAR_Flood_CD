#!/usr/bin/env python3
"""ChangeGuard Plan V3 E9: Diagnostic Ablation Matrix Runner.

Executes bounded diagnostics supporting Branch C empirical findings:
1. Normalization dynamics:
   - Evaluates BN running-statistics momentum sensitivity (m=0.01, m=0.05) with zero parameter updates.
   - Compares against frozen source statistics (m=0.0), default AdaBN (m=0.10), and pure batch stats (m=1.0).
2. CoTTA component diagnosis:
   - CoTTA without Stochastic Source Restoration (SSR disabled: restoration_prob=0.0).
   - CoTTA without Mean Teacher EMA (EMA disabled: ema_alpha=0.0, direct student self-training).
   - Bounded repeat design: Seeds 0 and 1; Orders A and B; reset-per-event mode across all 6 dev events.

Governance rules strictly enforced:
- Zero sealed-test event access (fail-closed check on [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]).
- Physical ground-truth reads deferred until atomic shard persistence and fsync.
- Strict non-overwriting policy.
- Bit-exact source checkpoint verification.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.changeguard.baseline_adapters import (
    BaseE1Adapter,
    E1AdaBNAdapter,
    E1MatchedBNNoUpdateAdapter,
    E1SourceOnlyAdapter,
)
from src.changeguard.cotta_adapter import E7CoTTAAdapter
from tools.run_e8_confirmation_matrix import (
    DEFAULT_TTA_DEV_EVENTS,
    FORBIDDEN_SEALED_TEST_EVENTS,
    DeferredBatchSpec,
    DeferredLabelLoader,
    E8ConfirmationRunner,
    atomic_write_json,
    compute_file_sha256,
    get_git_info,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/changeguard/e9_diagnostic_matrix_v1.json"
DEFAULT_LEDGER_PATH = REPO_ROOT / "configs/changeguard/e9_execution_ledger_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/e9_diagnostic"


class E9DiagnosticRunner(E8ConfirmationRunner):
    """Runner for Work Package E9 diagnostic ablation matrix."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        device: str = "cpu",
        allow_overwrite: bool = False,
    ):
        cfg_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        led_path = ledger_path if ledger_path is not None else DEFAULT_LEDGER_PATH
        out_path = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        super().__init__(
            config_path=cfg_path,
            ledger_path=led_path,
            output_dir=out_path,
            device=device,
            allow_overwrite=allow_overwrite,
        )

    def instantiate_adapter(
        self, method_name: str, model: torch.nn.Module, seed: int = 42
    ) -> BaseE1Adapter:
        """Instantiates adapter for E9 diagnostic methods."""
        m_cfg = self.cfg["evaluated_methods"].get(method_name)
        if m_cfg is None:
            return super().instantiate_adapter(method_name, model, seed=seed)

        adapter_class_name = m_cfg.get("adapter_class")

        if adapter_class_name == "E1AdaBNAdapter":
            momentum = m_cfg.get("momentum")
            return E1AdaBNAdapter(
                model=model,
                momentum=float(momentum) if momentum is not None else None,
                device=self.device_str,
            )

        if adapter_class_name == "E7CoTTAAdapter":
            mode = m_cfg.get("mode", "reset_per_event")
            return E7CoTTAAdapter(
                model=model,
                learning_rate=float(m_cfg.get("learning_rate", 1e-4)),
                ema_alpha=float(m_cfg.get("ema_alpha", 0.999)),
                restoration_prob=float(m_cfg.get("restoration_prob", 0.01)),
                confidence_threshold=float(m_cfg.get("confidence_threshold", 0.90)),
                mode=mode,
                seed=seed,
                device=self.device_str,
            )

        return super().instantiate_adapter(method_name, model, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChangeGuard Plan V3 E9 Diagnostic Matrix Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to diagnostic matrix JSON configuration.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Path to execution ledger JSON.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run fail-closed preflights and ledger validation, then exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Display exact planned cell enumeration, reuse vs execute-new statuses, and exit.",
    )
    parser.add_argument(
        "--stream_id",
        type=str,
        default=None,
        help="Execute single specific stream by ID.",
    )
    parser.add_argument(
        "--execute_new_only",
        action="store_true",
        help="Execute all pending execute-new streams from ledger sequentially.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional override path for output root directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2-batch smoke test on Event 520.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    args = parser.parse_args()

    runner = E9DiagnosticRunner(
        config_path=args.config,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.validate_only:
        runner.run_preflights()
        print("\nPreflight validation check complete: SUCCESS (--validate_only). Exiting.")
        sys.exit(0)

    if args.dry_run:
        runner.run_preflights()
        reused, new_cells = runner.planned_cells()
        print("\n" + "=" * 80)
        print("ChangeGuard E9 Diagnostic Matrix Dry-Run Audit")
        print("=" * 80)
        print(f"Total Physical Streams Planned: {len(reused) + len(new_cells)}")
        print(f"  - Reused Canonical Streams:   {len(reused)} ({len(reused) * 6} event rows)")
        print(f"  - Execute-New Diagnostics:   {len(new_cells)} ({len(new_cells) * 6} event rows)")
        print(f"Total Event-Result Rows:        {(len(reused) + len(new_cells)) * 6}")
        print("\nExecute-New Diagnostic Streams:")
        for i, c in enumerate(new_cells, 1):
            print(f"  {i:2d}. {c['stream_id']}: method={c['method']}, seed={c['seed']}, order={c['order']}, reset={c['reset_mode']}")
        print("=" * 80)
        print("Dry-run audit complete: SUCCESS (--dry_run). Exiting.")
        sys.exit(0)

    runner.run_preflights()

    if args.smoke:
        if args.stream_id is None:
            parser.error("--smoke requires --stream_id to specify which stream to smoke-test.")
        _, new_cells = runner.planned_cells()
        target_cell = next((c for c in new_cells if c["stream_id"] == args.stream_id), None)
        if target_cell is None:
            raise ValueError(f"Stream ID {args.stream_id} not found in execute-new cells.")
        print(f"Executing 2-batch smoke for {args.stream_id} on Event 520...")
        runner.execute_stream(
            stream_id=target_cell["stream_id"],
            method_name=target_cell["method"],
            seed=int(target_cell["seed"]),
            order_id=target_cell["order"],
            reset_mode=target_cell["reset_mode"],
            max_samples_per_event=32,
            event_subset=[520],
        )
        print("Smoke test complete: SUCCESS.")
        sys.exit(0)

    if args.stream_id is not None:
        _, new_cells = runner.planned_cells()
        target_cell = next((c for c in new_cells if c["stream_id"] == args.stream_id), None)
        if target_cell is None:
            raise ValueError(f"Stream ID {args.stream_id} not found in execute-new cells.")
        runner.execute_stream(
            stream_id=target_cell["stream_id"],
            method_name=target_cell["method"],
            seed=int(target_cell["seed"]),
            order_id=target_cell["order"],
            reset_mode=target_cell["reset_mode"],
        )
    elif args.execute_new_only:
        _, new_cells = runner.planned_cells()
        print(f"\n================================================================================")
        print(f"ChangeGuard E9: Launching Full Execution of {len(new_cells)} Diagnostic Streams")
        print(f"================================================================================")
        for i, cell in enumerate(new_cells, 1):
            print(f"\n[{i}/{len(new_cells)}] Processing {cell['stream_id']}...")
            runner.execute_stream(
                stream_id=cell["stream_id"],
                method_name=cell["method"],
                seed=int(cell["seed"]),
                order_id=cell["order"],
                reset_mode=cell["reset_mode"],
            )
        print("\n================================================================================")
        print("ALL E9 DIAGNOSTIC STREAMS COMPLETED SUCCESSFULLY.")
        print("================================================================================")
    else:
        parser.error("Choose --validate_only, --dry_run, --stream_id, or --execute_new_only.")


if __name__ == "__main__":
    main()

