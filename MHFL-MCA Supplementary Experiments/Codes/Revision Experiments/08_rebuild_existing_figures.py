"""Experiment 08 - rebuild figures from existing source-data files only.

This utility is intentionally separated from training. It regenerates visual assets
from recorded CSV/JSON inputs and keeps the numerical source data unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mhfl_review.diagnostic_plots import plot_confusion_grid, plot_efficiency_two_panel, plot_performance_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild existing result figures from standardized source-data tables.")
    sub = parser.add_subparsers(dest="kind", required=True)
    performance = sub.add_parser("performance")
    performance.add_argument("--input", required=True, type=Path)
    performance.add_argument("--output", required=True, type=Path)
    performance.add_argument("--case-label", default="Case")
    confusion = sub.add_parser("confusion")
    confusion.add_argument("--input", required=True, type=Path)
    confusion.add_argument("--output", required=True, type=Path)
    confusion.add_argument("--counts", action="store_true", help="Show counts instead of row-normalized proportions.")
    efficiency = sub.add_parser("efficiency")
    efficiency.add_argument("--input", required=True, type=Path)
    efficiency.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if args.kind == "performance":
        plot_performance_grid(frame, args.input, args.output, args.case_label)
    elif args.kind == "confusion":
        plot_confusion_grid(frame, args.input, args.output, normalize=not args.counts)
    elif args.kind == "efficiency":
        plot_efficiency_two_panel(frame, args.input, args.output)
    print("Figure bundle written next to:", args.output)


if __name__ == "__main__":
    main()
