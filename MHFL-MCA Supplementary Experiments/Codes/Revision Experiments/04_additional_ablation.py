"""Experiment 04 - additional controlled KAIST ablations.

Purpose: evaluate encoder, attention-dimension, and gating controls requested in review.
Protocol: Stage-2 load shift, ten seeds, metric-wise drop-one-high/drop-one-low summaries.
Outputs: 150-row source evidence, 15-group summary, manifests, and audit gates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.provenance import sha256_file, sha256_json
from mhfl_review.stats import trimmed_mean_sd


VARIANT_NAMES = (
    "homogeneous_vibration",
    "homogeneous_other",
    "attention_dim_128",
    "direct_softmax",
    "equal_weights",
)
LEGACY_FULL_VARIANT = "full"
LEGACY_VARIANT_NAMES = (LEGACY_FULL_VARIANT,) + VARIANT_NAMES
CONDITIONS = (
    ("2Nm_0dB", "2Nm", 0.0, 0.0),
    ("4Nm_0dB", "4Nm", 0.0, 0.0),
    ("4Nm_-8dB", "4Nm", -8.0, -8.0),
)
CONDITION_NAMES = tuple(row[0] for row in CONDITIONS)
RAW_KEY = ("variant", "condition", "seed")
RAW_REQUIRED_COLUMNS = (
    "variant",
    "condition",
    "seed",
    "params_m",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
LEGACY_SEED_COUNT = 5
TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
HYBRID_REFERENCE_MODE = "hybrid_reference_ablation"
LEGACY_FULL_ARCHIVE_NAME = "legacy_supplementary_full_not_used_in_final_hybrid_table"
LEGACY_RAW_NAME = "additional_ablation_raw.csv"
VARIANT_RAW_NAME = "additional_ablation_variant_raw.csv"
VARIANT_SUMMARY_NAME = "additional_ablation_variant_summary.csv"
HYBRID_CANDIDATE_CSV_NAME = "manuscript_hybrid_ablation_candidate.csv"
HYBRID_CANDIDATE_TEX_NAME = "manuscript_hybrid_ablation_candidate.tex"
PROVENANCE_STATEMENT = (
    "Explicit channel selection was configured and fallback was prohibited by the full-mode "
    "execution policy; per-model fallback_used was not directly measured."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled R4-6 encoder/attention/gating ablations.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--accept-kaist-spec", action="store_true")
    return parser.parse_args()


def full_seed_sequence() -> List[int]:
    return [config.GLOBAL_SEED + index for index in range(config.ABLATION_REPEATS_FULL)]


def legacy_seed_sequence() -> List[int]:
    return full_seed_sequence()[:LEGACY_SEED_COUNT]


def missing_extension_seed_sequence() -> List[int]:
    return full_seed_sequence()[LEGACY_SEED_COUNT:]


def expected_raw_rows(seed_count: int) -> int:
    return len(VARIANT_NAMES) * len(CONDITION_NAMES) * int(seed_count)


def variant_specs(base: Any) -> Dict[str, Any]:
    return {
        "homogeneous_vibration": base.updated(
            encoder_mode="homogeneous_vibration", expected_params_m=None, spec_status="reviewer_ablation"
        ),
        "homogeneous_other": base.updated(
            encoder_mode="homogeneous_other", expected_params_m=None, spec_status="reviewer_ablation"
        ),
        "attention_dim_128": base.updated(
            attention_dim=128, expected_params_m=None, spec_status="reviewer_ablation"
        ),
        "direct_softmax": base.updated(
            gate_mode="direct_softmax", expected_params_m=None, spec_status="reviewer_ablation"
        ),
        "equal_weights": base.updated(
            gate_mode="equal", expected_params_m=None, spec_status="reviewer_ablation"
        ),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _sorted_raw(raw: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(VARIANT_NAMES)}
    condition_order = {name: index for index, name in enumerate(CONDITION_NAMES)}
    result = raw.copy()
    result["_variant_order"] = result["variant"].map(order)
    result["_condition_order"] = result["condition"].map(condition_order)
    result = result.sort_values(["seed", "_variant_order", "_condition_order"], kind="mergesort")
    return result.drop(columns=["_variant_order", "_condition_order"]).reset_index(drop=True)


def _sorted_legacy_raw(raw: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(LEGACY_VARIANT_NAMES)}
    condition_order = {name: index for index, name in enumerate(CONDITION_NAMES)}
    result = raw.copy()
    result["_variant_order"] = result["variant"].map(order)
    result["_condition_order"] = result["condition"].map(condition_order)
    result = result.sort_values(["seed", "_variant_order", "_condition_order"], kind="mergesort")
    return result.drop(columns=["_variant_order", "_condition_order"]).reset_index(drop=True)


def validate_raw_results(
    raw: pd.DataFrame,
    allowed_seeds: Sequence[int],
    require_complete: bool = False,
) -> Dict[str, Any]:
    missing_columns = [name for name in RAW_REQUIRED_COLUMNS if name not in raw.columns]
    if missing_columns:
        raise RuntimeError("Additional-ablation raw CSV is missing columns: {0}".format(", ".join(missing_columns)))
    if raw.duplicated(list(RAW_KEY)).any():
        raise RuntimeError("Additional-ablation raw CSV contains duplicate variant-condition-seed rows.")
    if raw[list(RAW_REQUIRED_COLUMNS)].isna().any().any():
        raise RuntimeError("Additional-ablation raw CSV contains NaN values.")

    numeric_columns = ("seed", "params_m", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    numeric = raw[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Additional-ablation raw CSV contains non-numeric or infinite values.")
    if not raw["variant"].isin(VARIANT_NAMES).all():
        raise RuntimeError("Additional-ablation raw CSV contains an unknown variant.")
    if not raw["condition"].isin(CONDITION_NAMES).all():
        raise RuntimeError("Additional-ablation raw CSV contains an unknown condition.")

    allowed = {int(seed) for seed in allowed_seeds}
    observed = {int(seed) for seed in numeric["seed"].tolist()}
    if not observed.issubset(allowed):
        raise RuntimeError("Additional-ablation raw CSV contains a seed outside the fixed target sequence.")
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if not raw[metric].between(0.0, 1.0).all():
            raise RuntimeError("Additional-ablation metric {0} is outside [0, 1].".format(metric))
    if not (raw["params_m"] > 0.0).all():
        raise RuntimeError("Additional-ablation params_m must be positive.")

    unit_sizes = raw.groupby(["variant", "seed"])["condition"].nunique()
    if not unit_sizes.empty and not (unit_sizes == len(CONDITION_NAMES)).all():
        raise RuntimeError("Every existing seed-variant unit must contain all three evaluation conditions.")
    if require_complete:
        target = {int(seed) for seed in allowed_seeds}
        if observed != target:
            raise RuntimeError("The fixed ten-seed set is incomplete; manuscript outputs are prohibited.")
        group_counts = raw.groupby(["variant", "condition"])["seed"].nunique()
        if len(group_counts) != len(VARIANT_NAMES) * len(CONDITION_NAMES):
            raise RuntimeError("The final raw CSV does not contain all variant-condition groups.")
        if not (group_counts == len(target)).all():
            raise RuntimeError("Every final variant-condition group must contain the identical ten seeds.")
        if len(raw) != expected_raw_rows(len(target)):
            raise RuntimeError("The final raw CSV must contain exactly {0} rows.".format(expected_raw_rows(len(target))))

    return {
        "rows": int(len(raw)),
        "variants": int(raw["variant"].nunique()),
        "conditions": int(raw["condition"].nunique()),
        "seeds": sorted(observed),
        "duplicate_rows": 0,
        "nan_values": 0,
        "infinite_values": 0,
        "failed_seeds": [],
    }


def validate_legacy_candidate(raw: pd.DataFrame) -> Dict[str, Any]:
    missing_columns = [name for name in RAW_REQUIRED_COLUMNS if name not in raw.columns]
    if missing_columns:
        raise RuntimeError("Legacy additional-ablation raw CSV is missing columns: {0}".format(", ".join(missing_columns)))
    legacy = raw[
        raw["variant"].isin(LEGACY_VARIANT_NAMES) & raw["seed"].isin(legacy_seed_sequence())
    ].copy()
    expected_rows = len(LEGACY_VARIANT_NAMES) * len(CONDITION_NAMES) * LEGACY_SEED_COUNT
    if len(legacy) != expected_rows:
        raise RuntimeError("The legacy candidate must contain exactly 6 variants x 3 conditions x 5 seeds = 90 rows.")
    if legacy.duplicated(list(RAW_KEY)).any():
        raise RuntimeError("The legacy candidate contains duplicate variant-condition-seed rows.")
    if set(legacy["variant"]) != set(LEGACY_VARIANT_NAMES):
        raise RuntimeError("The legacy candidate must contain Full and all five reviewer variants.")
    if set(legacy["condition"]) != set(CONDITION_NAMES):
        raise RuntimeError("The legacy candidate must contain all three evaluation conditions.")
    numeric_columns = ("seed", "params_m", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    numeric = legacy[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("The legacy candidate contains NaN, Inf, or non-numeric values.")
    groups = legacy.groupby(["variant", "condition"])["seed"].agg(lambda values: {int(v) for v in values})
    if len(groups) != len(LEGACY_VARIANT_NAMES) * len(CONDITION_NAMES) or not all(
        value == set(legacy_seed_sequence()) for value in groups
    ):
        raise RuntimeError("Every legacy variant-condition group must contain the same five seeds.")
    return {
        "rows": int(len(legacy)),
        "variant_rows": int((legacy["variant"] != LEGACY_FULL_VARIANT).sum()),
        "legacy_full_rows": int((legacy["variant"] == LEGACY_FULL_VARIANT).sum()),
        "seeds": legacy_seed_sequence(),
        "duplicate_rows": 0,
        "nan_values": 0,
        "infinite_values": 0,
    }


def summarize(
    raw: pd.DataFrame,
    expected_seed_count: int = 10,
    full_reference: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    target_seeds = [config.GLOBAL_SEED + index for index in range(int(expected_seed_count))]
    validate_raw_results(raw, target_seeds, require_complete=True)
    reference = (
        config.load_kaist_additional_ablation_full_reference()
        if full_reference is None
        else dict(full_reference)
    )
    reference_conditions = reference.get("conditions", {})
    rows: List[Dict[str, Any]] = []
    for (variant, condition), block in raw.groupby(["variant", "condition"], sort=True):
        ordered = block.sort_values("seed")
        accuracy_mean, accuracy_sd = trimmed_mean_sd(ordered["accuracy"].tolist())
        macro_f1_mean, macro_f1_sd = trimmed_mean_sd(ordered["macro_f1"].tolist())
        retained = int(expected_seed_count) - 2 if int(expected_seed_count) >= 3 else int(expected_seed_count)
        condition_reference = reference_conditions.get(condition)
        if isinstance(condition_reference, dict):
            comparison_type = "unpaired_aggregated_reference_difference"
            reference_mean_difference: Any = accuracy_mean - float(condition_reference["accuracy_mean"])
            macro_f1_reference_mean_difference: Any = macro_f1_mean - float(condition_reference["macro_f1_mean"])
        else:
            comparison_type = "no_stage2_full_reference_for_noise_condition"
            reference_mean_difference = "not_applicable"
            macro_f1_reference_mean_difference = "not_applicable"
        rows.append(
            {
                "variant": variant,
                "condition": condition,
                "accuracy_mean": accuracy_mean,
                "accuracy_sd": accuracy_sd,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_sd": macro_f1_sd,
                "reference_mean_difference": reference_mean_difference,
                "macro_f1_reference_mean_difference": macro_f1_reference_mean_difference,
                "comparison_type": comparison_type,
                "params_m": float(ordered["params_m"].iloc[0]),
                "seeds": int(expected_seed_count),
                "retained_after_trim": retained,
                "aggregation": TRIMMED_AGGREGATION,
                "accuracy_untrimmed_mean": float(np.mean(ordered["accuracy"].to_numpy(dtype=np.float64))),
                "accuracy_untrimmed_sd": float(np.std(ordered["accuracy"].to_numpy(dtype=np.float64), ddof=0)),
                "macro_f1_untrimmed_mean": float(np.mean(ordered["macro_f1"].to_numpy(dtype=np.float64))),
                "macro_f1_untrimmed_sd": float(np.std(ordered["macro_f1"].to_numpy(dtype=np.float64), ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "condition"]).reset_index(drop=True)


def validate_additional_ablation_post_gate(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    full_reference: Mapping[str, Any],
    model_inventory: Sequence[Mapping[str, Any]],
    archive: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_gate = validate_raw_results(raw, full_seed_sequence(), require_complete=True)
    expected_summary = summarize(
        raw,
        expected_seed_count=config.ABLATION_REPEATS_FULL,
        full_reference=full_reference,
    )
    required_summary_columns = (
        "variant",
        "condition",
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "reference_mean_difference",
        "macro_f1_reference_mean_difference",
        "comparison_type",
        "params_m",
        "seeds",
        "retained_after_trim",
        "aggregation",
    )
    missing = [name for name in required_summary_columns if name not in summary.columns]
    if missing:
        raise RuntimeError("Additional-ablation summary is missing columns: {0}".format(", ".join(missing)))
    if len(summary) != len(VARIANT_NAMES) * len(CONDITION_NAMES):
        raise RuntimeError("Additional-ablation summary must contain exactly 15 variant-condition rows.")
    if summary.duplicated(["variant", "condition"]).any():
        raise RuntimeError("Additional-ablation summary contains duplicate variant-condition rows.")
    if set(summary["variant"]) != set(VARIANT_NAMES) or set(summary["condition"]) != set(CONDITION_NAMES):
        raise RuntimeError("Additional-ablation summary variant or condition set is incomplete.")
    if not (summary["seeds"] == config.ABLATION_REPEATS_FULL).all():
        raise RuntimeError("Every additional-ablation summary group must report ten seeds.")
    if not (summary["retained_after_trim"] == config.ABLATION_REPEATS_FULL - 2).all():
        raise RuntimeError("Every additional-ablation summary group must retain eight values.")
    if not (summary["aggregation"] == TRIMMED_AGGREGATION).all():
        raise RuntimeError("Additional-ablation summary uses an invalid aggregation protocol.")
    numeric_columns = (
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "params_m",
    )
    numeric = summary[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Additional-ablation summary contains NaN, Inf, or non-numeric metrics.")
    forbidden = [
        name
        for name in summary.columns
        if "delta_to_full" in name or "p_value" in name or "significance" in name
    ]
    if forbidden:
        raise RuntimeError("Aggregated Full reference cannot support paired/statistical fields: {0}".format(forbidden))

    clean = summary[summary["condition"].isin(("2Nm_0dB", "4Nm_0dB"))]
    noise = summary[summary["condition"] == "4Nm_-8dB"]
    clean_difference_columns = ("reference_mean_difference", "macro_f1_reference_mean_difference")
    clean_differences = clean[list(clean_difference_columns)].apply(pd.to_numeric, errors="coerce")
    if len(clean) != 10 or clean_differences.isna().any().any() or not np.isfinite(
        clean_differences.to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Clean-load reference differences are incomplete or non-finite.")
    if not (clean["comparison_type"] == "unpaired_aggregated_reference_difference").all():
        raise RuntimeError("Clean-load comparisons must be labeled as unpaired aggregated-reference differences.")
    if len(noise) != 5 or not (
        noise["comparison_type"] == "no_stage2_full_reference_for_noise_condition"
    ).all():
        raise RuntimeError("The -8 dB rows must remain source data without a Full reference.")

    compare_columns = (
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "params_m",
        "seeds",
        "retained_after_trim",
    )
    actual_ordered = summary.sort_values(["variant", "condition"]).reset_index(drop=True)
    expected_ordered = expected_summary.sort_values(["variant", "condition"]).reset_index(drop=True)
    if actual_ordered[["variant", "condition"]].to_dict("records") != expected_ordered[
        ["variant", "condition"]
    ].to_dict("records"):
        raise RuntimeError("Additional-ablation summary ordering or group keys do not match raw results.")
    if not np.allclose(
        actual_ordered[list(compare_columns)].to_numpy(dtype=np.float64),
        expected_ordered[list(compare_columns)].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Additional-ablation summary does not reproduce independent trimmed statistics.")
    actual_clean = actual_ordered[actual_ordered["condition"].isin(("2Nm_0dB", "4Nm_0dB"))]
    expected_clean = expected_ordered[expected_ordered["condition"].isin(("2Nm_0dB", "4Nm_0dB"))]
    if not np.allclose(
        actual_clean[list(clean_difference_columns)].to_numpy(dtype=np.float64),
        expected_clean[list(clean_difference_columns)].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Additional-ablation reference mean differences do not match the aggregate reference.")

    expected_model_keys = {
        (variant, seed) for variant in VARIANT_NAMES for seed in full_seed_sequence()
    }
    model_keys = {(str(row["variant"]), int(row["seed"])) for row in model_inventory}
    if len(model_inventory) != 50 or model_keys != expected_model_keys:
        raise RuntimeError("Additional-ablation post-gate requires exactly 50 validated variant model runs.")
    archived_keys = {
        (str(row["variant"]), int(row["seed"]))
        for row in archive.get("reused_variant_model_files", [])
    }
    baseline_keys = {
        (variant, seed) for variant in VARIANT_NAMES for seed in legacy_seed_sequence()
    }
    extension_keys = {
        (variant, seed) for variant in VARIANT_NAMES for seed in missing_extension_seed_sequence()
    }
    if not baseline_keys.issubset(archived_keys):
        raise RuntimeError("The immutable first-five-seed 25-model baseline is not fully archived.")
    if not extension_keys.issubset(model_keys):
        raise RuntimeError("The fixed later-five-seed 25-model extension is incomplete.")
    if archive.get("status") != LEGACY_FULL_ARCHIVE_NAME:
        raise RuntimeError("Legacy Full archive status is invalid.")

    conditions = full_reference.get("conditions", {})
    if (
        full_reference.get("model") != "Full MHFL-MCA"
        or full_reference.get("source") != "main_manuscript_stage2_experiment"
        or full_reference.get("protocol") != "stage2_load_shift"
        or full_reference.get("source_load") != "0Nm"
        or full_reference.get("n_train_per_class") != 30
        or full_reference.get("runs") != 10
        or full_reference.get("aggregation") != TRIMMED_AGGREGATION
        or full_reference.get("reference_type") != "main_manuscript_aggregated_reference"
        or not isinstance(conditions, dict)
        or set(conditions) != {"2Nm_0dB", "4Nm_0dB"}
    ):
        raise RuntimeError("The Full manuscript reference metadata is incomplete or incompatible.")
    reference_values = [
        conditions[condition][field]
        for condition in ("2Nm_0dB", "4Nm_0dB")
        for field in ("accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd")
    ]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        for value in reference_values
    ):
        raise RuntimeError("The Full manuscript reference contains non-finite metric values.")
    if not np.isclose(
        float(full_reference.get("params_m", np.nan)),
        config.KAIST_ADDITIONAL_ABLATION_EXPECTED_PARAMS_M,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("The Full manuscript reference does not match the published 7.380 M value.")
    for condition, expected_metrics in config.KAIST_ADDITIONAL_ABLATION_EXPECTED_METRICS.items():
        for field, expected_value in expected_metrics.items():
            if not np.isclose(
                float(conditions[condition][field]),
                float(expected_value),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "The Full manuscript reference does not match the published clean-load value."
                )

    return {
        "status": "PASS",
        "raw_gate": raw_gate,
        "summary_rows": int(len(summary)),
        "summary_groups": int(len(summary)),
        "retained_after_trim": config.ABLATION_REPEATS_FULL - 2,
        "aggregation": TRIMMED_AGGREGATION,
        "model_runs": int(len(model_inventory)),
        "reused_first_five_model_runs": int(len(baseline_keys)),
        "planned_extension_model_slots": int(len(extension_keys)),
        "legacy_full_archive_status": archive.get("status"),
        "failed_seeds": [],
        "duplicates": 0,
        "nan_or_inf": 0,
        "paired_delta_to_full_generated": False,
        "stage3_minus8db_full_used": False,
    }


def write_hybrid_candidate_table(
    summary: pd.DataFrame,
    full_reference: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Path]:
    def formatted(mean: Any, sd: Any) -> str:
        return "{0:.4f} ± {1:.4f}".format(float(mean), float(sd))

    conditions = full_reference["conditions"]
    rows: List[Dict[str, Any]] = [
        {
            "Method": "Full MHFL-MCA reference",
            "Params (M)": "{0:.3f}".format(float(full_reference["params_m"])),
            "2 Nm Accuracy": formatted(conditions["2Nm_0dB"]["accuracy_mean"], conditions["2Nm_0dB"]["accuracy_sd"]),
            "2 Nm Macro-F1": formatted(conditions["2Nm_0dB"]["macro_f1_mean"], conditions["2Nm_0dB"]["macro_f1_sd"]),
            "4 Nm Accuracy": formatted(conditions["4Nm_0dB"]["accuracy_mean"], conditions["4Nm_0dB"]["accuracy_sd"]),
            "4 Nm Macro-F1": formatted(conditions["4Nm_0dB"]["macro_f1_mean"], conditions["4Nm_0dB"]["macro_f1_sd"]),
            "source": "main_manuscript_aggregated_reference",
            "comparison_type": "reference",
        }
    ]
    labels = {
        "homogeneous_vibration": "Homogeneous-vibration",
        "homogeneous_other": "Homogeneous-other",
        "attention_dim_128": "Attention (D=128)",
        "direct_softmax": "Direct softmax",
        "equal_weights": "Equal weights",
    }
    clean = summary[summary["condition"].isin(("2Nm_0dB", "4Nm_0dB"))]
    for variant in VARIANT_NAMES:
        block = clean[clean["variant"] == variant].set_index("condition")
        two = block.loc["2Nm_0dB"]
        four = block.loc["4Nm_0dB"]
        rows.append(
            {
                "Method": labels[variant],
                "Params (M)": "{0:.6f}".format(float(two["params_m"])),
                "2 Nm Accuracy": formatted(two["accuracy_mean"], two["accuracy_sd"]),
                "2 Nm Macro-F1": formatted(two["macro_f1_mean"], two["macro_f1_sd"]),
                "4 Nm Accuracy": formatted(four["accuracy_mean"], four["accuracy_sd"]),
                "4 Nm Macro-F1": formatted(four["macro_f1_mean"], four["macro_f1_sd"]),
                "source": "new_stage2_additional_ablation_variant",
                "comparison_type": "unpaired_aggregated_reference_difference",
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != 6:
        raise RuntimeError("Hybrid additional-ablation candidate table must contain exactly six rows.")
    csv_path = Path(out_dir) / HYBRID_CANDIDATE_CSV_NAME
    tex_path = Path(out_dir) / HYBRID_CANDIDATE_TEX_NAME
    _atomic_write_csv(table, csv_path)
    temporary = tex_path.with_name(tex_path.name + ".tmp")
    temporary.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")
    os.replace(str(temporary), str(tex_path))
    return {"csv": csv_path, "tex": tex_path}


def _file_record(path: Path) -> Dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError("Required provenance file is missing: {0}".format(target))
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(target),
    }


def _staged_archive_record(staged_path: Path, final_path: Path) -> Dict[str, Any]:
    record = _file_record(staged_path)
    record["path"] = str(Path(final_path).resolve())
    return record


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object: {0}".format(path))
    return payload


def _split_file(seed: int) -> Path:
    return config.SPLIT_ROOT / ("kaist_source_seed{0}.json".format(int(seed)))


def inspect_model_cache(
    out_dir: Path,
    variant: str,
    seed: int,
    expected_model_spec: Optional[Any] = None,
) -> Dict[str, Any]:
    run_dir = Path(out_dir) / "models" / variant / ("seed{0}".format(int(seed)))
    manifest_path = run_dir / "manifest.json"
    weights_path = run_dir / "model.weights.h5"
    history_path = run_dir / ("{0}_stage2.history.csv".format(variant))
    split_path = _split_file(seed)
    manifest = _read_json(manifest_path)
    split = _read_json(split_path)

    model_spec = manifest.get("model_spec")
    if not isinstance(model_spec, dict) or manifest.get("model_spec_fingerprint") != sha256_json(model_spec):
        raise RuntimeError("Cached model specification fingerprint mismatch: {0}".format(manifest_path))
    if expected_model_spec is not None and model_spec != expected_model_spec.to_dict():
        raise RuntimeError("Cached model specification does not match the requested variant: {0}".format(run_dir))

    training = manifest.get("training_spec", {})
    expected_training = {
        "protocol": "stage2",
        "epochs": config.STAGE2_EPOCHS,
        "batch_size": config.KAIST_MANUSCRIPT_BATCH_SIZE,
        "learning_rate": config.KAIST_MANUSCRIPT_LEARNING_RATE,
        "seed": int(seed),
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError("Cached model training_spec mismatch for {0}, seed {1}: {2}".format(variant, seed, key))
    protocol_signature = manifest.get("protocol_signature", {})
    if protocol_signature.get("variant") != variant:
        raise RuntimeError("Cached model variant signature mismatch: {0}".format(run_dir))
    if protocol_signature.get("evaluation_conditions") != list(CONDITION_NAMES):
        raise RuntimeError("Cached model evaluation-condition signature mismatch: {0}".format(run_dir))
    if protocol_signature.get("split_signature") != split.get("signature"):
        raise RuntimeError("Cached model split signature mismatch: {0}".format(run_dir))
    if split.get("max_train_per_class") != 30 or split.get("val_per_class") != 50:
        raise RuntimeError("Split sidecar does not prove N=30 and val_per_class=50: {0}".format(split_path))

    expected_payload = {
        "suite_version": manifest.get("suite_version"),
        "model_spec": manifest.get("model_spec"),
        "model_spec_fingerprint": manifest.get("model_spec_fingerprint"),
        "training_spec": training,
        "protocol_signature": protocol_signature,
    }
    if manifest.get("run_signature") != sha256_json(expected_payload):
        raise RuntimeError("Cached model run_signature does not match its signed payload: {0}".format(run_dir))
    weights_record = _file_record(weights_path)
    if manifest.get("weights_sha256") != weights_record["sha256"]:
        raise RuntimeError("Cached model weights hash mismatch: {0}".format(weights_path))
    if Path(str(manifest.get("weights_path", ""))).resolve() != weights_path.resolve():
        raise RuntimeError("Cached model weights path mismatch: {0}".format(manifest_path))
    if Path(str(manifest.get("history_path", ""))).resolve() != history_path.resolve():
        raise RuntimeError("Cached model history path mismatch: {0}".format(manifest_path))

    return {
        "seed": int(seed),
        "variant": variant,
        "model_manifest": _file_record(manifest_path),
        "weights": weights_record,
        "history": _file_record(history_path),
        "split_file": _file_record(split_path),
        "split_signature": str(split["signature"]),
        "max_train_per_class": int(split["max_train_per_class"]),
        "val_per_class": int(split["val_per_class"]),
        "git_commit": manifest.get("environment", {}).get("git_commit"),
    }


def collect_model_inventory(
    out_dir: Path,
    seeds: Sequence[int],
    expected_specs: Optional[Mapping[str, Any]] = None,
    variants: Sequence[str] = VARIANT_NAMES,
) -> List[Dict[str, Any]]:
    return [
        inspect_model_cache(
            out_dir,
            variant,
            seed,
            None if expected_specs is None else expected_specs[variant],
        )
        for seed in seeds
        for variant in variants
    ]


def _legacy_integrity_projection(inventory: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in inventory:
        projected.append(
            {
                "seed": int(row["seed"]),
                "variant": str(row["variant"]),
                "model_manifest": dict(row["model_manifest"]),
                "weights": dict(row["weights"]),
                "history": dict(row["history"]),
            }
        )
    return projected


def assert_legacy_inventory_unchanged(
    baseline: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> None:
    if list(baseline) != _legacy_integrity_projection(current):
        raise RuntimeError("A legacy manifest, weights file, history file, SHA-256, or mtime changed; extension aborted.")


def ensure_legacy_full_archive(
    out_dir: Path,
    legacy_raw: pd.DataFrame,
    expected_specs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    archive_dir = out_dir / LEGACY_FULL_ARCHIVE_NAME
    manifest_path = archive_dir / "archive_manifest.json"
    validate_legacy_candidate(legacy_raw)
    legacy_first_five = legacy_raw[
        legacy_raw["variant"].isin(LEGACY_VARIANT_NAMES)
        & legacy_raw["seed"].isin(legacy_seed_sequence())
    ].copy()
    legacy_first_five = _sorted_legacy_raw(legacy_first_five)
    full_raw = _sorted_legacy_raw(
        legacy_raw[
            (legacy_raw["variant"] == LEGACY_FULL_VARIANT)
            & legacy_raw["seed"].isin(full_seed_sequence())
        ].copy()
    )
    full_model_seeds = [
        seed
        for seed in full_seed_sequence()
        if (
            out_dir
            / "models"
            / LEGACY_FULL_VARIANT
            / ("seed{0}".format(seed))
            / "manifest.json"
        ).is_file()
    ]
    reused_variant_seeds = [
        seed
        for seed in full_seed_sequence()
        if all(
            (
                out_dir
                / "models"
                / variant
                / ("seed{0}".format(seed))
                / "manifest.json"
            ).is_file()
            for variant in VARIANT_NAMES
        )
    ]
    if not set(legacy_seed_sequence()).issubset(reused_variant_seeds):
        raise RuntimeError("The five fixed legacy seeds do not contain all five reusable variant models.")
    if manifest_path.is_file():
        archive = _read_json(manifest_path)
        if archive.get("legacy_full_seeds") != full_model_seeds:
            raise RuntimeError("The set of legacy Full checkpoints changed after archival; extension aborted.")
        archived_variant_seeds = archive.get("reused_variant_seeds", legacy_seed_sequence())
        current_variants = collect_model_inventory(
            out_dir, archived_variant_seeds, expected_specs, variants=VARIANT_NAMES
        )
        current_full = collect_model_inventory(
            out_dir, full_model_seeds, expected_specs, variants=(LEGACY_FULL_VARIANT,)
        )
        assert_legacy_inventory_unchanged(archive.get("reused_variant_model_files", []), current_variants)
        assert_legacy_inventory_unchanged(archive.get("legacy_full_model_files", []), current_full)
        current_source = _file_record(out_dir / LEGACY_RAW_NAME)
        archived_source = archive.get("source_legacy_raw", {})
        if current_source.get("sha256") != archived_source.get("sha256"):
            raise RuntimeError("The immutable legacy raw source changed after archival; extension aborted.")
        archived_first_five_path = archive_dir / "legacy_first_five_raw.csv"
        archived_first_five_record = archive.get("legacy_first_five_raw", {})
        if (
            not archived_first_five_path.is_file()
            or sha256_file(archived_first_five_path) != archived_first_five_record.get("sha256")
        ):
            raise RuntimeError("The archived first-five-seed raw evidence is missing or changed.")
        archived_first_five = pd.read_csv(archived_first_five_path)
        pd.testing.assert_frame_equal(
            _sorted_legacy_raw(archived_first_five),
            legacy_first_five,
            check_dtype=False,
            check_exact=True,
        )
        archived_raw = pd.read_csv(archive_dir / "legacy_full_raw.csv")
        pd.testing.assert_frame_equal(
            _sorted_legacy_raw(archived_raw),
            full_raw,
            check_dtype=False,
            check_exact=True,
        )
        return archive
    if archive_dir.exists():
        raise RuntimeError("Incomplete archive directory exists without archive_manifest.json: {0}".format(archive_dir))

    reused_inventory = collect_model_inventory(
        out_dir, reused_variant_seeds, expected_specs, variants=VARIANT_NAMES
    )
    full_inventory = collect_model_inventory(
        out_dir, full_model_seeds, expected_specs, variants=(LEGACY_FULL_VARIANT,)
    )
    temporary = out_dir / (LEGACY_FULL_ARCHIVE_NAME + ".tmp")
    if temporary.exists():
        raise RuntimeError("Stale temporary archive exists; inspect it before continuing: {0}".format(temporary))
    temporary.mkdir(parents=False, exist_ok=False)
    _atomic_write_csv(full_raw, temporary / "legacy_full_raw.csv")
    _atomic_write_csv(legacy_first_five, temporary / "legacy_first_five_raw.csv")
    old_summary = out_dir / "additional_ablation_summary.csv"
    archived_summary: Optional[Dict[str, Any]] = None
    if old_summary.is_file():
        summary = pd.read_csv(old_summary)
        if "variant" in summary.columns:
            full_summary = summary[summary["variant"] == LEGACY_FULL_VARIANT].copy()
            if not full_summary.empty:
                _atomic_write_csv(full_summary, temporary / "legacy_full_summary.csv")
                archived_summary = _staged_archive_record(
                    temporary / "legacy_full_summary.csv", archive_dir / "legacy_full_summary.csv"
                )
    reused_projection = _legacy_integrity_projection(reused_inventory)
    full_projection = _legacy_integrity_projection(full_inventory)
    payload = {
        "status": LEGACY_FULL_ARCHIVE_NAME,
        "purpose": "Preserve the old supplementary Full run without using it in the hybrid-reference final table.",
        "run_tag": config.RUN_TAG,
        "seed_count": LEGACY_SEED_COUNT,
        "seeds": legacy_seed_sequence(),
        "reused_variant_seeds": reused_variant_seeds,
        "baseline_legacy_variant_model_runs": LEGACY_SEED_COUNT * len(VARIANT_NAMES),
        "preexisting_extension_variant_model_runs": (
            len(reused_variant_seeds) - LEGACY_SEED_COUNT
        ) * len(VARIANT_NAMES),
        "legacy_full_seeds": full_model_seeds,
        "legacy_full_raw_rows": int(len(full_raw)),
        "legacy_full_raw": _staged_archive_record(
            temporary / "legacy_full_raw.csv", archive_dir / "legacy_full_raw.csv"
        ),
        "legacy_first_five_raw": _staged_archive_record(
            temporary / "legacy_first_five_raw.csv", archive_dir / "legacy_first_five_raw.csv"
        ),
        "legacy_full_summary": archived_summary,
        "source_legacy_raw": _file_record(out_dir / LEGACY_RAW_NAME),
        "reused_variant_model_files": reused_projection,
        "reused_variant_model_integrity_signature": sha256_json(reused_projection),
        "legacy_full_model_files": full_projection,
        "legacy_full_model_integrity_signature": sha256_json(full_projection),
        "legacy_full_checkpoint_policy": "left_in_place_read_only_not_deleted_not_overwritten",
        "used_in_final_table": False,
    }
    _atomic_write_json(temporary / "archive_manifest.json", payload)
    os.replace(str(temporary), str(archive_dir))
    return payload


def _evidence_context() -> Dict[str, Any]:
    vibration_path = config.PROVENANCE_ROOT / "vibration_manifest.json"
    preflight_path = config.PROVENANCE_ROOT / "preflight_report.json"
    channel_path = config.PROVENANCE_ROOT / "channel_manifest.json"
    batch_path = config.SUITE_ROOT / "windows" / "00_set_environment.bat"
    vibration = _read_json(vibration_path)
    preflight = _read_json(preflight_path)
    channel = _read_json(channel_path)
    batch_text = batch_path.read_text(encoding="utf-8")
    if vibration.get("configured_column") != 0 or not vibration.get("configured_column_available"):
        raise RuntimeError("Vibration provenance does not confirm zero-based column 0.")
    if not preflight.get("current_channel_explicit") or not preflight.get("vibration_channel_valid"):
        raise RuntimeError("Full-run preflight does not confirm explicit current and valid vibration selection.")
    exact_channel = "cDAQ9185-1F486B5Mod2/ai0"
    if not any(row.get("group") == "Log" and row.get("channel") == exact_channel for row in channel.get("channels", [])):
        raise RuntimeError("Channel manifest does not contain the confirmed Log/U-phase current channel.")
    if 'set "MHFL_CURRENT_CHANNEL_NAME={0}"'.format(exact_channel) not in batch_text:
        raise RuntimeError("Windows environment sidecar does not configure the exact current channel.")
    active_lines = [line.strip().lower() for line in batch_text.splitlines() if line.strip().lower().startswith("set ")]
    if any("mhfl_allow_current_fallback" in line for line in active_lines):
        raise RuntimeError("Windows environment sidecar enables prohibited current-channel fallback.")
    if config.ALLOW_CURRENT_CHANNEL_FALLBACK:
        raise RuntimeError("Full additional-ablation provenance cannot be generated with fallback enabled.")

    per_load = sorted(config.PROVENANCE_ROOT.rglob("channel_selection_*.json"))
    evidence_paths: List[Path]
    selection_mode = "explicit_exact_name"
    if per_load:
        for path in per_load:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
            if not rows or not all(
                row.get("group") == "Log"
                and row.get("channel") == exact_channel
                and row.get("selection_rule") == selection_mode
                and not row.get("fallback_used")
                for row in rows
            ):
                raise RuntimeError("Per-load channel selection evidence is inconsistent: {0}".format(path))
        evidence_paths = per_load
        evidence_source = "per_load_channel_selection"
    else:
        evidence_paths = [preflight_path, channel_path, vibration_path, batch_path]
        evidence_source = "joint_preflight_channel_vibration_environment_sidecars"
    return {
        "vibration_evidence": _file_record(vibration_path),
        "current_preflight_evidence": [_file_record(path) for path in evidence_paths],
        "current_evidence_source": evidence_source,
        "current_selection_mode": selection_mode,
    }


def write_model_provenance(out_dir: Path, seeds: Sequence[int]) -> Path:
    evidence = _evidence_context()
    inventory = collect_model_inventory(out_dir, seeds)
    rows: List[Dict[str, Any]] = []
    for record in inventory:
        rows.append(
            {
                **record,
                "vibration_column": 0,
                "vibration_physical_meaning": "bearing housing A x-direction",
                "vibration_evidence_path": evidence["vibration_evidence"]["path"],
                "vibration_evidence_sha256": evidence["vibration_evidence"]["sha256"],
                "current_channel": "cDAQ9185-1F486B5Mod2/ai0",
                "current_group": "Log",
                "current_selection_mode": evidence["current_selection_mode"],
                "fallback_policy": "prohibited_in_full_mode",
                "current_preflight_evidence": evidence["current_preflight_evidence"],
                "protocol": "stage2",
                "epochs": config.STAGE2_EPOCHS,
                "learning_rate": config.KAIST_MANUSCRIPT_LEARNING_RATE,
                "batch_size": config.KAIST_MANUSCRIPT_BATCH_SIZE,
                "run_tag": config.RUN_TAG,
            }
        )
    payload = {
        "confirmation_status": "joint_sidecar_provenance",
        "statement": PROVENANCE_STATEMENT,
        "sidecar_is_excluded_from_model_run_signature": True,
        "current_evidence_source": evidence["current_evidence_source"],
        "model_runs": len(rows),
        "records": rows,
    }
    path = Path(out_dir) / "additional_ablation_model_provenance.json"
    _atomic_write_json(path, payload)
    return path


def _completed_units(raw: pd.DataFrame) -> set:
    return {(str(row.variant), int(row.seed)) for row in raw[["variant", "seed"]].drop_duplicates().itertuples(index=False)}


def _available_model_seeds(out_dir: Path, target_seeds: Sequence[int]) -> List[int]:
    available: List[int] = []
    for seed in target_seeds:
        if all((Path(out_dir) / "models" / variant / ("seed{0}".format(seed)) / "manifest.json").is_file() for variant in VARIANT_NAMES):
            available.append(int(seed))
    return available


def _final_run_manifest(
    out_dir: Path,
    raw_path: Path,
    summary_path: Path,
    provenance_path: Path,
    archive: Mapping[str, Any],
) -> Path:
    raw = pd.read_csv(raw_path)
    gate = validate_raw_results(raw, full_seed_sequence(), require_complete=True)
    legacy_current = collect_model_inventory(
        out_dir,
        archive.get("reused_variant_seeds", legacy_seed_sequence()),
        variants=VARIANT_NAMES,
    )
    legacy_full_current = collect_model_inventory(
        out_dir, archive.get("legacy_full_seeds", legacy_seed_sequence()), variants=(LEGACY_FULL_VARIANT,)
    )
    assert_legacy_inventory_unchanged(archive["reused_variant_model_files"], legacy_current)
    assert_legacy_inventory_unchanged(archive["legacy_full_model_files"], legacy_full_current)
    all_inventory = collect_model_inventory(out_dir, full_seed_sequence())
    reference = config.load_kaist_additional_ablation_full_reference()
    summary = pd.read_csv(summary_path, keep_default_na=False)
    post_gate = validate_additional_ablation_post_gate(
        raw,
        summary,
        reference,
        all_inventory,
        archive,
    )
    candidate_paths = write_hybrid_candidate_table(summary, reference, out_dir)
    payload = {
        "status": "complete_hybrid_reference_candidate",
        "result_mode": HYBRID_REFERENCE_MODE,
        "run_tag": config.RUN_TAG,
        "protocol": "stage2",
        "target_seeds": full_seed_sequence(),
        "raw_gate": gate,
        "model_runs": len(all_inventory),
        "reused_first_five_model_runs": LEGACY_SEED_COUNT * len(VARIANT_NAMES),
        "reused_preexisting_extension_model_runs": len(legacy_current) - LEGACY_SEED_COUNT * len(VARIANT_NAMES),
        "planned_extension_model_slots": len(missing_extension_seed_sequence()) * len(VARIANT_NAMES),
        "new_model_runs_remaining_at_hybrid_start": len(all_inventory) - len(legacy_current),
        "target_extension_model_runs_semantics": "resume-safe missing five-variant seed slots; preexisting valid extension caches are reused",
        "full_model_training_performed": False,
        "aggregation": TRIMMED_AGGREGATION,
        "retained_after_trim": config.ABLATION_REPEATS_FULL - 2,
        "archive_manifest_path": str(
            (Path(out_dir) / LEGACY_FULL_ARCHIVE_NAME / "archive_manifest.json").resolve()
        ),
        "reused_variant_model_integrity_signature": archive[
            "reused_variant_model_integrity_signature"
        ],
        "legacy_full_model_integrity_signature": archive["legacy_full_model_integrity_signature"],
        "legacy_full_checkpoint_policy": "left_in_place_read_only_not_deleted_not_overwritten",
        "raw": _file_record(raw_path),
        "summary": _file_record(summary_path),
        "model_provenance": _file_record(provenance_path),
        "full_reference": _file_record(config.KAIST_ADDITIONAL_ABLATION_FULL_REFERENCE_PATH),
        "full_reference_metadata": {
            "source": reference["source"],
            "protocol": reference["protocol"],
            "n_train_per_class": reference["n_train_per_class"],
            "runs": reference["runs"],
        },
        "comparison_policy": {
            "clean_conditions_in_hybrid_table": ["2Nm_0dB", "4Nm_0dB"],
            "noise_condition_source_data_only": "4Nm_-8dB",
            "paired_delta_to_full_generated": False,
            "comparison_type": "unpaired_aggregated_reference_difference",
            "stage3_table12_full_used": False,
            "statistical_significance_claim_allowed": False,
        },
        "post_gate": post_gate,
        "hybrid_candidate_csv": _file_record(candidate_paths["csv"]),
        "hybrid_candidate_tex": _file_record(candidate_paths["tex"]),
    }
    path = Path(out_dir) / "additional_ablation_run_manifest.json"
    _atomic_write_json(path, payload)
    return path


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    config.enforce_run_safety(
        args.mode,
        allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
        accept_kaist_spec=args.accept_kaist_spec,
        require_kaist_spec=True,
    )
    kaist_learning_rate, kaist_batch_size = config.require_confirmed_kaist_training_config(args.mode)
    from mhfl_review.specs import manuscript_spec, parameter_count_m
    from mhfl_review.train import TrainingSpec, evaluate_with_noise, load_or_train_variant_cached, prepare_kaist_splits

    base_spec = manuscript_spec("kaist")
    variants = variant_specs(base_spec)
    repeats = config.ABLATION_REPEATS_FAST if args.mode == "fast" else config.ABLATION_REPEATS_FULL
    seeds = [config.GLOBAL_SEED + index for index in range(repeats)]
    epochs = config.FAST_EPOCHS if args.mode == "fast" else config.STAGE2_EPOCHS
    out_dir = config.OUTPUT_ROOT / "04_additional_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / VARIANT_RAW_NAME

    archive: Optional[Dict[str, Any]] = None
    if args.mode == "full":
        legacy_raw_path = out_dir / LEGACY_RAW_NAME
        if not legacy_raw_path.is_file():
            raise FileNotFoundError(
                "The immutable five-seed legacy raw source is required before extension: {0}".format(
                    legacy_raw_path
                )
            )
        legacy_raw = pd.read_csv(legacy_raw_path)
        validate_legacy_candidate(legacy_raw)
        archive_specs = dict(variants)
        archive_specs[LEGACY_FULL_VARIANT] = base_spec
        archive = ensure_legacy_full_archive(out_dir, legacy_raw, expected_specs=archive_specs)
        if raw_path.is_file():
            raw = pd.read_csv(raw_path)
        else:
            raw = legacy_raw[
                legacy_raw["variant"].isin(VARIANT_NAMES)
                & legacy_raw["seed"].isin(seeds)
            ][list(RAW_REQUIRED_COLUMNS)].copy()
            raw = _sorted_raw(raw)
            validate_raw_results(raw, seeds, require_complete=False)
            _atomic_write_csv(raw, raw_path)
        legacy_current = collect_model_inventory(
            out_dir,
            archive.get("reused_variant_seeds", legacy_seed_sequence()),
            expected_specs=variants,
            variants=VARIANT_NAMES,
        )
        assert_legacy_inventory_unchanged(archive["reused_variant_model_files"], legacy_current)
        write_model_provenance(out_dir, _available_model_seeds(out_dir, seeds))
    else:
        raw = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame(columns=RAW_REQUIRED_COLUMNS)
    validate_raw_results(raw, seeds, require_complete=False)

    completed = _completed_units(raw)
    if args.mode == "full":
        for variant_name, seed in sorted(completed, key=lambda row: (row[1], row[0])):
            inspect_model_cache(out_dir, variant_name, seed, expected_model_spec=variants[variant_name])
    for seed in seeds:
        missing_variants = [name for name in VARIANT_NAMES if (name, int(seed)) not in completed]
        if not missing_variants:
            print("[Ablation] seed={0} reused from existing raw results; no evaluation or training performed.".format(seed))
            continue
        splits, split_meta = prepare_kaist_splits(
            n_train=30,
            seed=seed,
            allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
        )
        for variant_name in missing_variants:
            spec = variants[variant_name]
            run_dir = out_dir / "models" / variant_name / ("seed{0}".format(seed))
            training_spec = TrainingSpec(
                protocol="stage2",
                epochs=epochs,
                batch_size=kaist_batch_size,
                learning_rate=kaist_learning_rate,
                seed=seed,
            )
            model, _, _, _ = load_or_train_variant_cached(
                spec,
                splits["train"],
                splits["val"],
                training_spec,
                run_dir,
                "{0}_stage2".format(variant_name),
                protocol_signature={
                    "variant": variant_name,
                    "split_signature": split_meta["plan_signature"],
                    "data_signatures": split_meta["data_signatures"],
                    "evaluation_conditions": list(CONDITION_NAMES),
                },
                verbose=0,
            )
            new_rows: List[Dict[str, Any]] = []
            for condition, load, snr1, snr2 in CONDITIONS:
                metrics, _, _, _ = evaluate_with_noise(
                    model,
                    splits[load],
                    snr1,
                    snr2,
                    seed=seed + abs(int(snr1)) * 100 + (2 if load == "2Nm" else 4),
                )
                new_rows.append(
                    {
                        "variant": variant_name,
                        "condition": condition,
                        "seed": seed,
                        "params_m": parameter_count_m(spec),
                        **metrics,
                    }
                )
                print(
                    "[Ablation] seed={0} variant={1} {2} acc={3:.4f}".format(
                        seed, variant_name, condition, metrics["accuracy"]
                    )
                )
            raw = _sorted_raw(pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True))
            validate_raw_results(raw, seeds, require_complete=False)
            _atomic_write_csv(raw, raw_path)
            completed.add((variant_name, int(seed)))
            if args.mode == "full":
                write_model_provenance(out_dir, _available_model_seeds(out_dir, seeds))

    validate_raw_results(raw, seeds, require_complete=True)
    summary = summarize(raw, expected_seed_count=repeats)
    summary_path = out_dir / VARIANT_SUMMARY_NAME
    _atomic_write_csv(summary, summary_path)
    if args.mode == "full":
        if archive is None:
            raise RuntimeError("Internal error: legacy Full archive was not loaded.")
        provenance_path = write_model_provenance(out_dir, seeds)
        manifest_path = _final_run_manifest(out_dir, raw_path, summary_path, provenance_path, archive)
        result = {
            "post_gate": _read_json(manifest_path)["post_gate"],
            "manifest": _file_record(manifest_path),
        }
    else:
        result = {"post_gate": {"status": "PASS", "mode": "fast", "final_outputs_authorized": False}}
    print("Outputs saved to:", out_dir)
    return result


def main() -> None:
    args = parse_args()
    out_dir = config.OUTPUT_ROOT / "04_additional_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "additional_ablation_post_gate.json"
    _atomic_write_json(
        gate_path,
        {
            "status": "RUNNING",
            "mode": args.mode,
            "run_tag": config.RUN_TAG,
            "final_outputs_authorized": False,
        },
    )
    try:
        # Keep the confirmed KAIST training configuration explicit at the controller boundary.
        config.require_confirmed_kaist_training_config(args.mode)
        result = run_experiment(args)
    except BaseException as exc:
        _atomic_write_json(
            gate_path,
            {
                "status": "FAIL",
                "mode": args.mode,
                "run_tag": config.RUN_TAG,
                "final_outputs_authorized": False,
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
            },
        )
        raise
    post_gate = result["post_gate"]
    _atomic_write_json(
        gate_path,
        {
            **post_gate,
            "status": "PASS",
            "mode": args.mode,
            "run_tag": config.RUN_TAG,
            "final_outputs_authorized": args.mode == "full",
            "manifest": result.get("manifest"),
        },
    )


if __name__ == "__main__":
    main()
