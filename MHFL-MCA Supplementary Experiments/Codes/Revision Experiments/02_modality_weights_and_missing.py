"""Experiment 02 - modality-weight and missing-modality evidence.

Purpose: quantify AMRM weights and controlled sensor-degradation behavior.
Protocol: reuse frozen Stage-3 checkpoints; do not tune or retrain inside analysis.
Outputs: per-run CSV source data, summaries, and publication figure bundles.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.data import SplitData, add_gaussian_noise, replace_modality
from mhfl_review.plotting import plot_missing_modality_stress, plot_modality_weight_evidence
from mhfl_review.specs import branch_semantics
from mhfl_review.train import load_or_train_kaist, metric_dict, predict_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R2-6/R4-4 weight analysis and R4-6 modality stress tests.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--protocol", choices=("stage2", "stage3"), default="stage3")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--accept-kaist-spec", action="store_true")
    return parser.parse_args()


def evaluate_inputs(model, weight_model, split: SplitData, x1: np.ndarray, x2: np.ndarray) -> Dict[str, float]:
    probabilities = predict_prob(model, x1, x2, batch_size=64)
    weights = np.asarray(weight_model.predict([x1, x2], batch_size=64, verbose=0))
    metrics = metric_dict(split.y_int, probabilities)
    metrics.update(
        {
            "weight_branch1_mean": float(np.mean(weights[:, 0])),
            "weight_branch2_mean": float(np.mean(weights[:, 1])),
            "weight_branch1_within_sd": float(np.std(weights[:, 0], ddof=0)),
            "weight_branch2_within_sd": float(np.std(weights[:, 1], ddof=0)),
        }
    )
    return metrics


def noisy_inputs(split: SplitData, snr_v: float, snr_c: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    x1 = add_gaussian_noise(split.x1, snr_v, np.random.RandomState(seed))
    x2 = add_gaussian_noise(split.x2, snr_c, np.random.RandomState(seed + 9973))
    return x1, x2


def summarize_weight_raw(raw: pd.DataFrame) -> pd.DataFrame:
    # First average independent noise realizations within each trained model,
    # then report between-model variability. This prevents noise draws from
    # being misrepresented as independent trained models.
    per_model = (
        raw.groupby(["load", "scenario", "snr_db", "model_seed"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            weight_vibration=("weight_vibration", "mean"),
            weight_current=("weight_current", "mean"),
        )
    )
    summary = (
        per_model.groupby(["load", "scenario", "snr_db"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", lambda values: float(np.std(values, ddof=0))),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", lambda values: float(np.std(values, ddof=0))),
            weight_vibration_mean=("weight_vibration", "mean"),
            weight_vibration_sd=("weight_vibration", lambda values: float(np.std(values, ddof=0))),
            weight_current_mean=("weight_current", "mean"),
            weight_current_sd=("weight_current", lambda values: float(np.std(values, ddof=0))),
            model_seeds=("model_seed", "nunique"),
        )
    )
    return summary


def summarize_missing(raw: pd.DataFrame) -> pd.DataFrame:
    per_model = (
        raw.groupby(["load", "condition", "model_seed"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            weight_vibration=("weight_vibration", "mean"),
            weight_current=("weight_current", "mean"),
        )
    )
    baseline = per_model[per_model["condition"] == "baseline_0dB"][
        ["load", "model_seed", "accuracy", "macro_f1"]
    ].rename(columns={"accuracy": "baseline_accuracy", "macro_f1": "baseline_macro_f1"})
    paired = per_model.merge(baseline, on=["load", "model_seed"], how="left")
    paired["accuracy_delta"] = paired["accuracy"] - paired["baseline_accuracy"]
    paired["macro_f1_delta"] = paired["macro_f1"] - paired["baseline_macro_f1"]
    return (
        paired.groupby(["load", "condition"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", lambda values: float(np.std(values, ddof=0))),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", lambda values: float(np.std(values, ddof=0))),
            weight_vibration_mean=("weight_vibration", "mean"),
            weight_current_mean=("weight_current", "mean"),
            accuracy_delta_mean=("accuracy_delta", "mean"),
            accuracy_delta_sd=("accuracy_delta", lambda values: float(np.std(values, ddof=0))),
            macro_f1_delta_mean=("macro_f1_delta", "mean"),
            macro_f1_delta_sd=("macro_f1_delta", lambda values: float(np.std(values, ddof=0))),
            model_seeds=("model_seed", "nunique"),
        )
    )


def main() -> None:
    args = parse_args()
    model_seeds = config.MODEL_SEEDS_FAST if args.mode == "fast" else config.MODEL_SEEDS_FULL
    noise_repeats = config.NOISE_REALIZATIONS_FAST if args.mode == "fast" else config.NOISE_REALIZATIONS_FULL
    out_dir = config.OUTPUT_ROOT / "02_weights_missing"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for model_seed in model_seeds:
        trained = load_or_train_kaist(
            protocol=args.protocol,
            mode=args.mode,
            seed=model_seed,
            force=args.force_train,
            rebuild_cache=args.rebuild_cache,
            accept_kaist_spec=args.accept_kaist_spec,
            verbose=1,
        )
        semantics = branch_semantics(trained.model_spec)
        if semantics != ("vibration_source", "other_source"):
            raise RuntimeError(
                "KAIST AMRM interpretation expects value-source order; found {0}.".format(semantics)
            )
        weight_model = trained.auxiliary["weights"]
        for load in ("2Nm", "4Nm"):
            split = trained.splits[load]
            scenarios = {
                "both_modalities_degraded": lambda snr: (snr, snr),
                "vibration_only_degraded": lambda snr: (snr, config.BASE_SNR_DB),
                "current_only_degraded": lambda snr: (config.BASE_SNR_DB, snr),
            }
            for scenario, snr_function in scenarios.items():
                for snr in config.SNR_GRID:
                    for noise_repeat in range(noise_repeats):
                        seed = int(model_seed + 100_000 * noise_repeat + abs(int(snr)) * 101 + (2 if load == "2Nm" else 4))
                        snr_v, snr_c = snr_function(float(snr))
                        x1, x2 = noisy_inputs(split, snr_v, snr_c, seed)
                        result = evaluate_inputs(trained.model, weight_model, split, x1, x2)
                        weight_rows.append(
                            {
                                "protocol": args.protocol,
                                "load": load,
                                "scenario": scenario,
                                "snr_db": float(snr),
                                "model_seed": int(model_seed),
                                "noise_repeat": noise_repeat + 1,
                                "weight_vibration": result.pop("weight_branch1_mean"),
                                "weight_current": result.pop("weight_branch2_mean"),
                                **result,
                            }
                        )

            conditions = {
                "baseline_0dB": ("noise", 0.0, 0.0),
                "vibration_missing": ("missing1", 0.0, 0.0),
                "current_missing": ("missing2", 0.0, 0.0),
                "vibration_severe": ("noise", config.SEVERE_SNR_DB, 0.0),
                "current_severe": ("noise", 0.0, config.SEVERE_SNR_DB),
                "both_severe": ("noise", config.SEVERE_SNR_DB, config.SEVERE_SNR_DB),
            }
            for condition, (kind, snr_v, snr_c) in conditions.items():
                repeats = 1 if "missing" in kind else noise_repeats
                for noise_repeat in range(repeats):
                    seed = int(model_seed + 200_000 * noise_repeat + (2 if load == "2Nm" else 4))
                    if kind == "missing1":
                        stress = replace_modality(split, 1)
                        x1, x2 = stress.x1, stress.x2
                    elif kind == "missing2":
                        stress = replace_modality(split, 2)
                        x1, x2 = stress.x1, stress.x2
                    else:
                        x1, x2 = noisy_inputs(split, float(snr_v), float(snr_c), seed)
                    result = evaluate_inputs(trained.model, weight_model, split, x1, x2)
                    missing_rows.append(
                        {
                            "protocol": args.protocol,
                            "load": load,
                            "condition": condition,
                            "model_seed": int(model_seed),
                            "noise_repeat": noise_repeat + 1,
                            "weight_vibration": result.pop("weight_branch1_mean"),
                            "weight_current": result.pop("weight_branch2_mean"),
                            **result,
                        }
                    )

    weight_raw = pd.DataFrame(weight_rows)
    weight_raw_path = out_dir / "modality_weight_raw.csv"
    weight_raw.to_csv(weight_raw_path, index=False)
    weight_summary = summarize_weight_raw(weight_raw)
    weight_summary_path = out_dir / "modality_weight_summary.csv"
    weight_summary.to_csv(weight_summary_path, index=False)

    missing_raw = pd.DataFrame(missing_rows)
    missing_raw_path = out_dir / "missing_modality_raw.csv"
    missing_raw.to_csv(missing_raw_path, index=False)
    missing_summary = summarize_missing(missing_raw)
    missing_summary_path = out_dir / "missing_modality_summary.csv"
    missing_summary.to_csv(missing_summary_path, index=False)

    for load in ("2Nm", "4Nm"):
        plot_modality_weight_evidence(
            weight_summary,
            load,
            weight_summary_path,
            out_dir / ("modality_weight_evidence_{0}".format(load)),
        )
        plot_missing_modality_stress(
            missing_summary,
            load,
            missing_summary_path,
            out_dir / ("missing_modality_stress_{0}".format(load)),
        )
    print("Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
