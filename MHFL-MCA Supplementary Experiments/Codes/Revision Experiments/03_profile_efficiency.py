"""Experiment 03 - isolated FLOPs, storage, latency, and GPU-memory profiling.

Purpose: profile an existing KAIST checkpoint without training or loading raw data.
Protocol: CPU-only static-graph worker plus a fresh GPU runtime worker.
Outputs: partial worker JSON files and the merged efficiency profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mhfl_review.profiling import (
    run_controller,
    run_flops_worker,
    run_runtime_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3-5/R4-5 deployment-oriented MHFL-MCA profiling.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--protocol", choices=("stage2", "stage3"), default="stage2")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--accept-kaist-spec", action="store_true")

    # Internal worker arguments. Normal users should invoke the controller only.
    parser.add_argument(
        "--worker",
        choices=("controller", "flops", "runtime"),
        default="controller",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--out-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--weights-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _required_worker_path(value: Path, option: str) -> Path:
    if value is None:
        raise RuntimeError("Internal worker invocation requires {0}.".format(option))
    return Path(value)


def main() -> None:
    args = parse_args()
    if args.worker == "controller":
        run_controller(args, Path(__file__).resolve())
        return

    out_dir = _required_worker_path(args.out_dir, "--out-dir")
    weights_path = _required_worker_path(args.weights_path, "--weights-path")
    manifest_path = _required_worker_path(args.manifest_path, "--manifest-path")
    if args.worker == "flops":
        run_flops_worker(
            out_dir=out_dir,
            weights_path=weights_path,
            manifest_path=manifest_path,
            protocol=args.protocol,
            mode=args.mode,
        )
    else:
        run_runtime_worker(
            out_dir=out_dir,
            weights_path=weights_path,
            manifest_path=manifest_path,
            protocol=args.protocol,
            mode=args.mode,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
