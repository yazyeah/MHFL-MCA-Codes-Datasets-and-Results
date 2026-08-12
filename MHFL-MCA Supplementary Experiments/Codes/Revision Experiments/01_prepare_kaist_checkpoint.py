"""Experiment 01 - prepare reproducible KAIST Stage-2/Stage-3 checkpoints.

Purpose: train or load the manuscript model under the frozen KAIST protocol.
Protocol: confirmed Optuna configuration, exact U-phase current, and xA vibration.
Outputs: checkpoint weights, training history, metrics, and a hash-bound manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mhfl_review import config
from mhfl_review.train import evaluate_with_noise, load_or_train_kaist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/load source-faithful KAIST Stage-2 or Stage-3 checkpoints.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--protocol", choices=("stage2", "stage3", "both"), default="both")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    parser.add_argument("--accept-kaist-spec", action="store_true")
    return parser.parse_args()


def run_protocol(args: argparse.Namespace, protocol: str) -> None:
    bundle = load_or_train_kaist(
        protocol=protocol,
        mode=args.mode,
        seed=args.seed,
        force=args.force,
        rebuild_cache=args.rebuild_cache,
        rebuild_split=args.rebuild_split,
        accept_kaist_spec=args.accept_kaist_spec,
        verbose=1,
    )
    rows = []
    for load in ("2Nm", "4Nm"):
        metrics, _, _, _ = evaluate_with_noise(
            bundle.model,
            bundle.splits[load],
            config.BASE_SNR_DB,
            config.BASE_SNR_DB,
            seed=args.seed + (2 if load == "2Nm" else 4),
        )
        rows.append({"protocol": protocol, "load": load, "snr_db": config.BASE_SNR_DB, **metrics})
        print("[{0}/{1}] {2}".format(protocol, load, metrics))

    out_dir = config.OUTPUT_ROOT / "01_checkpoint" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "checkpoint_baseline_metrics.csv", index=False)
    (out_dir / "model_spec.json").write_text(
        json.dumps(bundle.model_spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = []
    bundle.model.summary(print_fn=summary_rows.append)
    (out_dir / "model_summary.txt").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    (out_dir / "checkpoint_manifest.json").write_text(
        json.dumps(dict(bundle.checkpoint_manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Outputs:", out_dir)


def main() -> None:
    args = parse_args()
    protocols = ("stage2", "stage3") if args.protocol == "both" else (args.protocol,)
    for protocol in protocols:
        run_protocol(args, protocol)


if __name__ == "__main__":
    main()
