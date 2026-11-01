#!/usr/bin/env python3
"""CLI runner for RST2 infrastructure validation, dry-run, execution, and resume."""

import argparse
import sys
from pathlib import Path

# Ensure ChangeGuard repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robust_source_training.runner import RSTExecutionEngine
from src.robust_source_training.manifest import write_terminal_status


def main():
    parser = argparse.ArgumentParser(description="RST2 Robust Source Training Infrastructure Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--validate_only", action="store_true", help="Perform static contract and config validation only")
    parser.add_argument("--dry_run", action="store_true", help="Inspect datasets and metadata on CPU without training")
    parser.add_argument("--execute", action="store_true", help="Launch authorized training and evaluation execution")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")

    args = parser.parse_args()

    # Require exactly one primary mode
    mode_count = sum([args.validate_only, args.dry_run, args.execute])
    if mode_count != 1:
        parser.error("Must specify exactly one of --validate_only, --dry_run, or --execute.")

    engine = RSTExecutionEngine(config_path=args.config, repo_root=REPO_ROOT)

    try:
        if args.validate_only:
            engine.validate_only()
            print("VALIDATION_ONLY: SUCCESS")
            sys.exit(0)
        elif args.dry_run:
            res = engine.dry_run()
            print(f"DRY_RUN: SUCCESS (Train samples: {res['train_sample_count']}, Eval events: {list(res['eval_sample_counts'].keys())})")
            sys.exit(0)
        elif args.execute:
            res = engine.execute(resume_checkpoint_path=args.resume)
            print(f"EXECUTION: SUCCESS (Status: {res['status']}, Steps: {res['optimizer_steps']}, Runtime: {res['runtime_seconds']:.2f}s)")
            sys.exit(0)
    except Exception as e:
        engine.logger.error(f"Execution failed with exception: {e}", exc_info=True)
        # Attempt to write terminal status if output_root exists
        try:
            write_terminal_status(
                run_dir=engine.output_root,
                run_id=engine.run_id,
                status="FAILED",
                exit_code=1,
                error_message=str(e),
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

