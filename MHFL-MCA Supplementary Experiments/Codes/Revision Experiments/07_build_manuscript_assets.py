"""Experiment 07 - fail-closed manuscript-asset builder.

Purpose: validate upstream experiment contracts before exporting tables and figures.
Protocol: recompute row/group/hash gates; never execute training in this script.
Outputs: manuscript candidate assets only when every mandatory gate passes.
"""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.provenance import sha256_file, write_json


TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
TARGET_SEEDS = tuple(config.GLOBAL_SEED + index for index in range(10))
ABLATION_SOURCE_VARIANTS = (
    "homogeneous_vibration",
    "homogeneous_other",
    "attention_dim_128",
    "direct_softmax",
    "equal_weights",
)
ABLATION_MAIN_VARIANTS = (
    "homogeneous_vibration",
    "homogeneous_other",
    "attention_dim_128",
    "direct_softmax",
)
ABLATION_EXCLUDED_MAIN_VARIANT = "equal_weights"
# Backward-compatible source-evidence alias used by existing audit helpers.
ABLATION_VARIANTS = ABLATION_SOURCE_VARIANTS
ABLATION_CONDITIONS = ("2Nm_0dB", "4Nm_0dB", "4Nm_-8dB")
ABLATION_TABLE_CONDITIONS = ("2Nm_0dB", "4Nm_0dB")
HYBRID_REFERENCE_MODE = "hybrid_reference_ablation"
ABLATION_VARIANT_RAW_NAME = "additional_ablation_variant_raw.csv"
ABLATION_VARIANT_SUMMARY_NAME = "additional_ablation_variant_summary.csv"
ABLATION_MAIN_RAW_NAME = "additional_ablation_variant_raw_main_scope.csv"
ABLATION_MAIN_SUMMARY_NAME = "additional_ablation_variant_summary_main_scope.csv"
ABLATION_MAIN_CANDIDATE_NAME = "manuscript_hybrid_ablation_candidate_main_scope.csv"
ABLATION_MAIN_CANDIDATE_TEX_NAME = "manuscript_hybrid_ablation_candidate_main_scope.tex"
ABLATION_MAIN_MANIFEST_NAME = "ablation_main_scope_without_equal_weights_manifest.json"
ABLATION_EQUAL_ARCHIVE_DIR_NAME = "archive_equal_weights_observed_control"
ABLATION_EQUAL_ARCHIVE_NAME = "equal_weights_exploratory_control.csv"
ABLATION_EQUAL_CANDIDATE_NAME = "equal_weights_candidate_row.csv"
ABLATION_MAIN_NOTE_NAME = "additional_ablation_main_scope_note.txt"
ABLATION_METHOD_LABELS = {
    "homogeneous_vibration": "Homogeneous-vibration",
    "homogeneous_other": "Homogeneous-other",
    "attention_dim_128": "Attention (D=128)",
    "direct_softmax": "Direct softmax",
    "equal_weights": "Equal weights",
}
ABLATION_MAIN_NOTE = (
    "The Full MHFL-MCA results are reused from the original Stage-2 main experiment at N=30 "
    "to avoid redundant retraining. The compact main-text table focuses on encoder "
    "homogeneity, attention dimension, and two-stage versus direct-softmax gating. "
    "All displayed ablation variants use trimmed mean and standard deviation over 10 "
    "independent runs after removing one highest and one lowest value. The exploratory "
    "equal-weight control was evaluated and retained under "
    "archive_equal_weights_observed_control for supplementary/provenance reporting; "
    "it was not deleted and is omitted only from the compact main display. "
    "The Stage-3 4Nm_-8dB Full result is not used in this table."
)
LOWSHOT_VARIANTS = ("full", "no_caim")
LOWSHOT_N_GRID = (1, 2, 3, 4, 5, 7, 10)
LOWSHOT_SEED_MAP = {
    n_train: tuple(100 * n_train + run_idx for run_idx in range(1, 11))
    for n_train in LOWSHOT_N_GRID
}
LOWSHOT_SUMMARY_METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "train_accuracy",
    "heldout_accuracy",
    "generalization_gap",
)
LOWSHOT_TABLE5_ANCHORS = {
    5: {
        "test_accuracy_mean": 0.9279,
        "test_accuracy_sd": 0.0288,
        "test_macro_precision_mean": 0.9361,
        "test_macro_precision_sd": 0.0237,
        "test_macro_f1_mean": 0.9266,
        "test_macro_f1_sd": 0.0299,
    },
    10: {
        "test_accuracy_mean": 0.9707,
        "test_accuracy_sd": 0.0175,
        "test_macro_precision_mean": 0.9723,
        "test_macro_precision_sd": 0.0160,
        "test_macro_f1_mean": 0.9704,
        "test_macro_f1_sd": 0.0178,
    },
}
TRADITIONAL_GROUP_COLUMNS = ("case", "load", "n_train", "snr_db", "method")
TRADITIONAL_METHODS = ("TF-SVM early fusion", "TF-SVM late fusion")
TRADITIONAL_CANDIDATE_COLUMNS = (
    "source_type",
    "reference_type",
    "source_table",
    "case",
    "load",
    "n_train",
    "model",
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "macro_precision_mean",
    "macro_precision_sd",
    "macro_recall_mean",
    "macro_recall_sd",
    "runs",
    "retained_after_trim",
    "aggregation",
    "benchmark_scope",
    "comparison_type",
)
DEEP_REFERENCE_MODELS = (
    "MRCFN",
    "CFFN",
    "CDTFAFN",
    "MSF-DFormer",
    "KDCNN-DF",
    "Full MHFL-MCA",
)
DEEP_REFERENCE_TABLES = {
    "UO_Table_5": ("UO", "held-out", "Table 5"),
    "KAIST_Table_9": ("KAIST", "2Nm", "Table 9"),
    "KAIST_Table_10": ("KAIST", "4Nm", "Table 10"),
}
DEEP_REFERENCE_N_GRID = (5, 10, 15, 20, 25, 30)
CURRENT_MANUSCRIPT_EXTRACTION_METHOD = "current_manuscript_table_source"
CURRENT_MANUSCRIPT_METRICS = ("accuracy", "macro_precision", "macro_f1")
CURRENT_MANUSCRIPT_TABLE_LABELS = {
    "UO_Table_5": "tab:case1_compare_main",
    "KAIST_Table_9": "tab:case2_quant_2nm",
    "KAIST_Table_10": "tab:case2_quant_4nm",
}
DEEP_REFERENCE_PATH = config.MAIN_MANUSCRIPT_DEEP_REFERENCE_PATH
EFFICIENCY_POSITIVE_FIELDS = (
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p25_ms",
    "latency_p75_ms",
    "latency_p95_ms",
    "throughput_samples_per_s",
    "gpu_allocator_current_mb",
    "gpu_allocator_peak_mb",
    "flops",
    "flops_g",
    "macs_estimated",
    "macs_g_estimated",
    "weights_size_mb",
    "savedmodel_size_mb",
)


def write_latex(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    text = df.to_latex(index=index, escape=False, float_format=lambda value: "{0:.4f}".format(value))
    path.write_text(text, encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError("{0} is missing required columns: {1}".format(source, missing))


def full_run_gate(source: Path) -> Dict[str, object]:
    if not source.is_file():
        return {"exists": False, "usable": False, "reason": "missing"}
    run_tag = config.RUN_TAG.lower()
    usable = not any(token in run_tag for token in ("fast", "smoke", "debug", "manual"))
    return {
        "exists": True,
        "usable": usable,
        "run_tag": config.RUN_TAG,
        "sha256": sha256_file(source),
        "reason": "full-run tag" if usable else "run tag appears to be smoke/debug/manual",
    }


def _new_gate(name: str) -> Dict[str, Any]:
    return {"name": name, "usable": False, "checks": [], "failure_reasons": [], "sources": {}}


def _check(
    gate: Dict[str, Any],
    name: str,
    passed: bool,
    expected: Any,
    actual: Any,
    reason: str,
) -> None:
    item = {"name": name, "passed": bool(passed), "expected": expected, "actual": actual}
    if not passed:
        item["reason"] = reason
        gate["failure_reasons"].append(reason)
    gate["checks"].append(item)


def _finish_gate(gate: Dict[str, Any]) -> Dict[str, Any]:
    gate["usable"] = bool(gate["checks"]) and all(item["passed"] for item in gate["checks"])
    gate["post_audit"] = {
        "status": "PASS" if gate["usable"] else "FAIL",
        "failed_checks": [item["name"] for item in gate["checks"] if not item["passed"]],
    }
    return gate


def _load_csv(gate: Dict[str, Any], name: str, path: Path) -> pd.DataFrame:
    exists = Path(path).is_file()
    _check(gate, name + "_exists", exists, True, exists, "Required source is missing: {0}".format(path))
    if not exists:
        return pd.DataFrame()
    gate["sources"][name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
    try:
        return pd.read_csv(path)
    except Exception as exc:
        _check(gate, name + "_readable", False, True, False, "Cannot read {0}: {1}".format(path, exc))
        return pd.DataFrame()


def _load_manifest(gate: Dict[str, Any], name: str, path: Path) -> Dict[str, Any]:
    exists = Path(path).is_file()
    _check(gate, name + "_exists", exists, True, exists, "Required manifest is missing: {0}".format(path))
    if not exists:
        return {}
    gate["sources"][name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
    try:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        _check(gate, name + "_readable", False, True, False, "Cannot read {0}: {1}".format(path, exc))
        return {}
    if not isinstance(payload, dict):
        _check(gate, name + "_object", False, "JSON object", type(payload).__name__, "Manifest is not an object.")
        return {}
    return payload


def _load_json_value(gate: Dict[str, Any], name: str, path: Path) -> Any:
    exists = Path(path).is_file()
    _check(gate, name + "_exists", exists, True, exists, "Required JSON source is missing: {0}".format(path))
    if not exists:
        return None
    gate["sources"][name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        _check(gate, name + "_readable", False, True, False, "Cannot read {0}: {1}".format(path, exc))
        return None


def _has_columns(gate: Dict[str, Any], frame: pd.DataFrame, columns: Sequence[str], label: str) -> bool:
    missing = sorted(set(columns).difference(frame.columns))
    _check(gate, label + "_columns", not missing, list(columns), sorted(frame.columns), "{0} is missing columns: {1}".format(label, missing))
    return not missing


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    if frame.empty or not set(columns).issubset(frame.columns):
        return False
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return not values.isna().any().any() and bool(np.isfinite(values.to_numpy(dtype=np.float64)).all())


def _manifest_hash_matches(manifest: Mapping[str, Any], key: str, path: Path) -> bool:
    record = manifest.get(key, {})
    return bool(Path(path).is_file() and isinstance(record, dict) and record.get("sha256") == sha256_file(path))


def _manifest_output_matches(outputs: Mapping[str, Any], key: str, path: Path) -> bool:
    record = outputs.get(key, {})
    if not isinstance(record, Mapping) or not Path(path).is_file():
        return False
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str):
        return False
    return (
        Path(recorded_path).resolve() == Path(path).resolve()
        and record.get("sha256") == sha256_file(path)
    )


def _lowshot_output_record_matches(
    outputs: Mapping[str, Any],
    key: str,
    path: Path,
    rows: Any = None,
) -> bool:
    """Validate the strict paper-aligned Experiment-05 output record."""

    record = outputs.get(key, {})
    if not isinstance(record, Mapping) or not Path(path).is_file():
        return False
    try:
        matches = (
            Path(str(record.get("path", ""))).resolve() == Path(path).resolve()
            and record.get("sha256") == sha256_file(path)
            and int(record.get("size_bytes", -1)) == int(Path(path).stat().st_size)
        )
        if rows is not None:
            matches = matches and int(record.get("rows", -1)) == int(rows)
        return bool(matches)
    except (OSError, TypeError, ValueError):
        return False


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _validate_lowshot_run_artifacts(
    manifest: Mapping[str, Any],
    raw: pd.DataFrame,
    output_dir: Path,
    dataset_content_signature: str,
) -> tuple[bool, Dict[str, Any]]:
    artifacts = manifest.get("run_artifacts", [])
    expected_identities = {
        (variant, n_train, run_idx, 100 * n_train + run_idx)
        for variant in LOWSHOT_VARIANTS
        for n_train in LOWSHOT_N_GRID
        for run_idx in range(1, 11)
    }
    details: Dict[str, Any] = {
        "manifest_count": manifest.get("run_artifact_count"),
        "records": len(artifacts) if isinstance(artifacts, list) else 0,
        "expected_identities": len(expected_identities),
        "validated_identities": 0,
        "validated_files": 0,
        "failures": [],
    }
    if (
        manifest.get("run_artifact_count") != 140
        or not isinstance(artifacts, list)
        or len(artifacts) != 140
    ):
        return False, details

    raw_lookup: Dict[Any, Any] = {}
    raw_identity_ok = set(("variant", "n_train", "run_idx", "seed")).issubset(raw.columns)
    if raw_identity_ok:
        try:
            for _, row in raw.iterrows():
                identity = (
                    str(row["variant"]),
                    int(row["n_train"]),
                    int(row["run_idx"]),
                    int(row["seed"]),
                )
                if identity in raw_lookup:
                    raw_identity_ok = False
                    break
                raw_lookup[identity] = row
        except (TypeError, ValueError):
            raw_identity_ok = False
    if not raw_identity_ok or set(raw_lookup) != expected_identities:
        details["failures"].append("raw_identity_contract")
        return False, details

    hparams = manifest.get("hparams", {}) if isinstance(manifest.get("hparams"), Mapping) else {}
    hparam_signature = _canonical_json_sha256(
        {key: value for key, value in hparams.items() if not str(key).startswith("source_")}
    )
    script_path = config.SUITE_ROOT / "05_lowshot_threshold.py"
    provenance_ready = (
        script_path.is_file()
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(hparams.get("source_evidence_sha256", ""))))
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(hparams.get("source_summary_sha256", ""))))
    )
    expected_provenance_signature = ""
    if provenance_ready:
        expected_provenance_signature = _canonical_json_sha256(
            {
                "script_sha256": sha256_file(script_path),
                "optuna_evidence_sha256": hparams["source_evidence_sha256"],
                "summary_evidence_sha256": hparams["source_summary_sha256"],
            }
        )

    identities = set()
    artifact_paths = set()
    root = Path(output_dir).resolve()
    numeric_raw_fields = (
        "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
        "train_accuracy", "heldout_accuracy", "generalization_gap", "train_time_s",
    )
    text_raw_fields = (
        "split_signature", "data_signature", "hyperparameter_signature", "source_type",
    )
    for index, artifact in enumerate(artifacts):
        failure_prefix = "record_{0:03d}".format(index)
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "variant", "n_train", "run_idx", "seed", "run_signature",
            "provenance_signature", "model_parameter_count", "metrics", "weights", "history",
        }:
            details["failures"].append(failure_prefix + ":schema")
            continue
        try:
            identity = (
                str(artifact["variant"]),
                int(artifact["n_train"]),
                int(artifact["run_idx"]),
                int(artifact["seed"]),
            )
            expected_dir = root / "models" / identity[0] / "N{0}_run{1:02d}_seed{2}".format(
                identity[1], identity[2], identity[3]
            )
            parameter_count = int(artifact["model_parameter_count"])
        except (KeyError, TypeError, ValueError):
            details["failures"].append(failure_prefix + ":identity")
            continue
        record_ok = (
            identity in expected_identities
            and identity not in identities
            and identity[3] == 100 * identity[1] + identity[2]
            and ((identity[0] == "full" and parameter_count == 5_111_759) or (identity[0] == "no_caim" and parameter_count > 0))
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("run_signature", ""))))
            and artifact.get("provenance_signature") == expected_provenance_signature
        )
        file_paths: Dict[str, Path] = {}
        for label, filename, rows in (
            ("metrics", "metrics.json", None),
            ("weights", "model.weights.h5", None),
            ("history", "history.csv", 80),
        ):
            file_record = artifact.get(label, {})
            path = Path(str(file_record.get("path", ""))) if isinstance(file_record, Mapping) else Path("")
            file_ok = (
                isinstance(file_record, Mapping)
                and set(file_record) == ({"path", "sha256", "size_bytes", "rows"} if rows is not None else {"path", "sha256", "size_bytes"})
                and path.is_absolute()
                and path.name == filename
                and path.parent.resolve() == expected_dir.resolve()
                and _path_is_within(path, root)
                and _lowshot_output_record_matches({label: file_record}, label, path, rows)
                and path.stat().st_size > 0
                and str(path.resolve()).lower() not in artifact_paths
            )
            record_ok = record_ok and file_ok
            if file_ok:
                file_paths[label] = path
                artifact_paths.add(str(path.resolve()).lower())
                details["validated_files"] += 1
        if set(file_paths) != {"metrics", "weights", "history"}:
            details["failures"].append(failure_prefix + ":files")
            continue

        try:
            metrics = json.loads(file_paths["metrics"].read_text(encoding="utf-8"))
            history = pd.read_csv(file_paths["history"])
        except Exception:
            details["failures"].append(failure_prefix + ":readable")
            continue
        if not isinstance(metrics, Mapping):
            details["failures"].append(failure_prefix + ":metrics_object")
            continue
        metrics_identity = (
            str(metrics.get("variant", "")),
            int(metrics.get("n_train", -1)),
            int(metrics.get("run_idx", -1)),
            int(metrics.get("seed", -1)),
        )
        raw_row = raw_lookup[identity]
        metrics_match_raw = all(_close(metrics.get(field), raw_row[field]) for field in numeric_raw_fields) and all(
            str(metrics.get(field, "")) == str(raw_row[field]) for field in text_raw_fields
        )
        expected_run_signature = _canonical_json_sha256(
            {
                "variant": identity[0],
                "n_train": identity[1],
                "run_idx": identity[2],
                "seed": identity[3],
                "split_signature": metrics.get("split_signature"),
                "data_signature": metrics.get("data_signature"),
                "hparam_signature": metrics.get("hyperparameter_signature"),
                "provenance_signature": metrics.get("provenance_signature"),
                "protocol": "uo_source_aligned_paired_deterministic_extension_v1",
            }
        )
        try:
            metrics_ok = (
                metrics_identity == identity
                and metrics_match_raw
                and metrics.get("data_signature") == dataset_content_signature
                and metrics.get("hyperparameter_signature") == hparam_signature
                and metrics.get("provenance_signature") == expected_provenance_signature
                and metrics.get("run_signature") == artifact.get("run_signature") == expected_run_signature
                and int(metrics.get("model_parameter_count", -1)) == parameter_count
                and Path(str(metrics.get("metrics_path", ""))).resolve() == file_paths["metrics"].resolve()
                and Path(str(metrics.get("weights_path", ""))).resolve() == file_paths["weights"].resolve()
                and Path(str(metrics.get("history_path", ""))).resolve() == file_paths["history"].resolve()
                and metrics.get("weights_sha256") == artifact["weights"].get("sha256")
                and metrics.get("history_sha256") == artifact["history"].get("sha256")
                and len(history) == 80
                and "epoch" in history.columns
                and list(pd.to_numeric(history["epoch"], errors="coerce")) == list(range(1, 81))
            )
        except (OSError, TypeError, ValueError):
            metrics_ok = False
        record_ok = record_ok and metrics_ok
        if record_ok:
            identities.add(identity)
        else:
            details["failures"].append(failure_prefix + ":content")

    details["validated_identities"] = len(identities)
    details["failures"] = details["failures"][:20]
    complete = (
        provenance_ready
        and identities == expected_identities
        and len(artifact_paths) == 420
        and details["validated_files"] == 420
        and not details["failures"]
    )
    return complete, details


def _archive_record_matches(
    record: Mapping[str, Any],
    source_path: Path,
    archive_path: Path,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    source_text = record.get("source")
    archive_text = record.get("archive")
    if not isinstance(source_text, str) or not isinstance(archive_text, str):
        return False
    if not source_path.is_file() or not archive_path.is_file():
        return False
    try:
        return (
            Path(source_text).resolve() == source_path.resolve()
            and Path(archive_text).resolve() == archive_path.resolve()
            and int(record.get("size_bytes", -1)) == int(source_path.stat().st_size)
            and record.get("source_sha256") == sha256_file(source_path)
            and record.get("archive_sha256") == sha256_file(archive_path)
            and record.get("source_sha256") == record.get("archive_sha256")
        )
    except (OSError, TypeError, ValueError):
        return False


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) >= 0.0
    )


def _trimmed_statistics(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 10 or not np.isfinite(array).all():
        raise ValueError("Final manuscript statistics require ten finite values.")
    ordered = np.sort(array)
    retained = ordered[1:-1]
    return (
        float(np.mean(retained)),
        float(np.std(retained, ddof=0)),
        float(np.mean(array)),
        float(np.std(array, ddof=0)),
    )


def _close(actual: Any, expected: float) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=1.0e-10, atol=1.0e-12))
    except (TypeError, ValueError):
        return False


def gate_efficiency(output_root: Path) -> Dict[str, Any]:
    gate = _new_gate("03_efficiency")
    profile_path = Path(output_root) / "03_efficiency" / "efficiency_profile.json"
    profile = _load_manifest(gate, "efficiency_profile", profile_path)
    _check(
        gate,
        "trainable_params",
        profile.get("trainable_params") == 7_380_173,
        7_380_173,
        profile.get("trainable_params"),
        "03 trainable parameter count does not match the confirmed KAIST model.",
    )
    _check(
        gate,
        "trainable_params_m",
        _close(profile.get("trainable_params_m"), 7.380173),
        7.380173,
        profile.get("trainable_params_m"),
        "03 trainable_params_m is inconsistent with 7,380,173 parameters.",
    )
    _check(
        gate,
        "flops_worker_exit_code",
        profile.get("flops_worker_exit_code") == 0,
        0,
        profile.get("flops_worker_exit_code"),
        "03 CPU FLOPs worker did not exit successfully.",
    )
    _check(
        gate,
        "runtime_worker_exit_code",
        profile.get("runtime_worker_exit_code") == 0,
        0,
        profile.get("runtime_worker_exit_code"),
        "03 GPU runtime worker did not exit successfully.",
    )
    for field in EFFICIENCY_POSITIVE_FIELDS:
        _check(
            gate,
            "positive_" + field,
            _positive_finite(profile.get(field)),
            "finite positive number",
            profile.get(field),
            "03 field {0} is missing, non-finite, zero, or negative.".format(field),
        )
    latency_order = all(_positive_finite(profile.get(name)) for name in (
        "latency_p25_ms", "latency_median_ms", "latency_p75_ms", "latency_p95_ms"
    )) and (
        float(profile["latency_p25_ms"])
        <= float(profile["latency_median_ms"])
        <= float(profile["latency_p75_ms"])
        <= float(profile["latency_p95_ms"])
    )
    _check(
        gate,
        "latency_percentile_order",
        latency_order,
        "p25 <= median <= p75 <= p95",
        None if not profile else [
            profile.get("latency_p25_ms"),
            profile.get("latency_median_ms"),
            profile.get("latency_p75_ms"),
            profile.get("latency_p95_ms"),
        ],
        "03 latency percentiles are internally inconsistent.",
    )
    memory_order = (
        _positive_finite(profile.get("gpu_allocator_current_mb"))
        and _positive_finite(profile.get("gpu_allocator_peak_mb"))
        and float(profile["gpu_allocator_current_mb"]) <= float(profile["gpu_allocator_peak_mb"])
    )
    _check(
        gate,
        "allocator_memory_order",
        memory_order,
        "current memory <= peak memory",
        [profile.get("gpu_allocator_current_mb"), profile.get("gpu_allocator_peak_mb")],
        "03 TensorFlow allocator current memory exceeds peak memory.",
    )
    architecture_ok = (
        profile.get("profiler_architecture") == "isolated_cpu_flops_gpu_runtime"
        and profile.get("device") == "gpu"
        and profile.get("protocol_checkpoint") == "stage2"
    )
    _check(
        gate,
        "isolated_gpu_profile",
        architecture_ok,
        "isolated CPU FLOPs + GPU runtime using Stage-2 checkpoint",
        {
            "architecture": profile.get("profiler_architecture"),
            "device": profile.get("device"),
            "protocol": profile.get("protocol_checkpoint"),
        },
        "03 is not the required isolated Stage-2 GPU efficiency profile.",
    )
    checkpoint_sha = profile.get("checkpoint_sha256")
    _check(
        gate,
        "checkpoint_sha256",
        isinstance(checkpoint_sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha) is not None,
        "64-character SHA-256",
        checkpoint_sha,
        "03 does not identify the profiled checkpoint by SHA-256.",
    )
    return _finish_gate(gate)


def _resolve_reference_source(reference_path: Path, source_path: Any, source_root: Any = None) -> Path:
    candidate = Path(str(source_path))
    if candidate.is_absolute():
        return candidate
    if source_root == "suite_provenance_main_manuscript_sources":
        return (config.MAIN_MANUSCRIPT_SOURCE_ROOT / candidate).resolve()
    if source_root == "workspace_root_two_levels_above_suite":
        return (config.SUITE_ROOT.parents[1] / candidate).resolve()
    beside_reference = (Path(reference_path).parent / candidate).resolve()
    if beside_reference.is_file():
        return beside_reference
    return (config.SUITE_ROOT / candidate).resolve()


def _read_current_manuscript_tex(source_tex_path: str) -> tuple[bytes, Path, Any]:
    if not isinstance(source_tex_path, str) or not source_tex_path or "\ufffd" in source_tex_path:
        raise RuntimeError("Current-manuscript source_tex_path is missing or invalid.")
    delimiter = "::" if "::" in source_tex_path else ("!" if ".zip!" in source_tex_path.lower() else None)
    if delimiter is None:
        path = Path(source_tex_path)
        if not path.is_absolute():
            path = config.SUITE_ROOT / path
        if not path.is_file():
            raise RuntimeError("Current-manuscript LaTeX source is missing: {0}".format(path))
        return path.read_bytes(), path.resolve(), None
    archive_text, member = source_tex_path.split(delimiter, 1)
    archive_path = Path(archive_text)
    if not archive_path.is_absolute():
        archive_path = config.SUITE_ROOT / archive_path
    if not archive_path.is_file() or not member:
        raise RuntimeError("Current-manuscript archive/member is missing.")
    try:
        with zipfile.ZipFile(str(archive_path), "r") as archive:
            payload = archive.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Cannot read current-manuscript LaTeX archive member: {0}".format(exc)) from exc
    return payload, archive_path.resolve(), member


def _strip_latex_emphasis(value: str) -> str:
    cleaned = str(value)
    pattern = re.compile(r"\\(?:textbf|underline)\{([^{}]*)\}")
    while pattern.search(cleaned):
        cleaned = pattern.sub(r"\1", cleaned)
    return cleaned


def _canonical_latex_model(row_text: str) -> Any:
    head = _strip_latex_emphasis(row_text.split("&", 1)[0]).strip()
    if "Proposed (MHFL-MCA)" in head:
        return "Full MHFL-MCA"
    return head if head in DEEP_REFERENCE_MODELS[:-1] else None


def _parse_current_manuscript_tables(tex_bytes: bytes) -> Dict[str, Dict[tuple[str, int], Dict[str, str]]]:
    try:
        tex = tex_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Current-manuscript LaTeX source is not valid UTF-8.") from exc
    parsed: Dict[str, Dict[tuple[str, int], Dict[str, str]]] = {}
    pair_pattern = re.compile(r"([01]\.\d{4})\s*\$\\pm\$\s*([01]\.\d{4})")
    for table_name, label in CURRENT_MANUSCRIPT_TABLE_LABELS.items():
        label_index = tex.find("\\label{{{0}}}".format(label))
        start = tex.rfind("\\begin{table", 0, label_index)
        end = tex.find("\\end{table}", label_index)
        if label_index < 0 or start < 0 or end < 0:
            raise RuntimeError("Cannot isolate current-manuscript table {0}.".format(label))
        region = tex[start : end + len("\\end{table}")]
        tabulars = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", region, flags=re.DOTALL)
        if len(tabulars) != 3:
            raise RuntimeError("Current-manuscript table must contain exactly three paired-N tabulars.")
        table_rows: Dict[tuple[str, int], Dict[str, str]] = {}
        for tabular in tabulars:
            n_values: List[int] = []
            for token in re.findall(r"(\d+)\s+training samples", tabular):
                n_train = int(token)
                if n_train not in n_values:
                    n_values.append(n_train)
            if len(n_values) != 2 or "\\midrule" not in tabular or "\\bottomrule" not in tabular:
                raise RuntimeError("Current-manuscript paired-N subtable structure is invalid.")
            body = tabular.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            observed_models = set()
            for row_text in re.split(r"\\\\", body):
                model = _canonical_latex_model(row_text)
                if model is None:
                    continue
                pairs = pair_pattern.findall(_strip_latex_emphasis(row_text))
                if len(pairs) != 6 or model in observed_models:
                    raise RuntimeError("Current-manuscript model row is duplicate or does not contain six metric pairs.")
                observed_models.add(model)
                for n_offset, n_train in enumerate(n_values):
                    values: Dict[str, str] = {}
                    for metric_offset, metric in enumerate(CURRENT_MANUSCRIPT_METRICS):
                        mean_text, sd_text = pairs[n_offset * 3 + metric_offset]
                        values[metric + "_mean"] = mean_text
                        values[metric + "_sd"] = sd_text
                    identity = (str(model), int(n_train))
                    if identity in table_rows:
                        raise RuntimeError("Current-manuscript table contains a duplicate model/N identity.")
                    table_rows[identity] = values
            if observed_models != set(DEEP_REFERENCE_MODELS):
                raise RuntimeError("Current-manuscript subtable does not contain the exact six-model set.")
        expected = {(model, n_train) for model in DEEP_REFERENCE_MODELS for n_train in DEEP_REFERENCE_N_GRID}
        if set(table_rows) != expected:
            raise RuntimeError("Current-manuscript table does not contain the exact six-model/six-N grid.")
        parsed[table_name] = table_rows
    return parsed


def _decimal_matches(reference_value: Any, latex_value: str) -> bool:
    try:
        return Decimal(str(reference_value)) == Decimal(str(latex_value))
    except (InvalidOperation, ValueError):
        return False


def gate_deep_reference(reference_path: Path = DEEP_REFERENCE_PATH) -> Dict[str, Any]:
    gate = _new_gate("06_main_manuscript_deep_reference")
    path = Path(reference_path)
    reference = _load_manifest(gate, "deep_reference", path)
    _check(
        gate,
        "schema_version",
        reference.get("schema_version") in (1, 2),
        "1 (legacy test fixture) or 2 (current manuscript source)",
        reference.get("schema_version"),
        "Deep-reference schema version is unsupported.",
    )
    _check(
        gate,
        "reference_type",
        reference.get("reference_type") == "main_manuscript_aggregated_reference",
        "main_manuscript_aggregated_reference",
        reference.get("reference_type"),
        "06 deep results must be explicitly identified as aggregated main-manuscript references.",
    )
    _check(gate, "reference_runs", reference.get("runs") == 10, 10, reference.get("runs"), "Deep reference must report the original ten runs.")
    _check(gate, "reference_aggregation", reference.get("aggregation") == TRIMMED_AGGREGATION, TRIMMED_AGGREGATION, reference.get("aggregation"), "Deep reference uses an incompatible aggregation.")
    scope = reference.get("benchmark_scope")
    scope_ok = isinstance(scope, str) and "no deep-model training" in scope and "no" in scope.lower() and "pair" in scope.lower()
    _check(
        gate,
        "reference_scope",
        scope_ok,
        "aggregated deep references; no deep-model training or per-seed pairing",
        scope,
        "Deep-reference scope does not prohibit retraining and per-seed statistical pairing.",
    )

    source_files = reference.get("source_files") if isinstance(reference.get("source_files"), list) else []
    source_records_ok = bool(source_files)
    source_failures: List[str] = []
    for index, record in enumerate(source_files):
        if not isinstance(record, dict):
            source_records_ok = False
            source_failures.append("source_files[{0}] is not an object".format(index))
            continue
        source_path = record.get("source_path")
        expected_sha = record.get("sha256")
        role = record.get("source_role")
        if not isinstance(source_path, str) or not source_path or not isinstance(role, str) or not role:
            source_records_ok = False
            source_failures.append("source_files[{0}] lacks path/role".format(index))
            continue
        resolved = _resolve_reference_source(path, source_path, reference.get("source_root_at_confirmation"))
        if not resolved.is_file():
            source_records_ok = False
            source_failures.append("missing source: {0}".format(resolved))
            continue
        if not isinstance(expected_sha, str) or expected_sha != sha256_file(resolved):
            source_records_ok = False
            source_failures.append("SHA-256 mismatch: {0}".format(resolved))
            continue
        gate["sources"]["deep_source_{0}".format(index)] = {
            "path": str(resolved),
            "sha256": expected_sha,
            "source_role": role,
        }
    _check(
        gate,
        "source_file_hashes",
        source_records_ok,
        "one or more existing source files with matching SHA-256",
        source_failures,
        "Deep-reference source provenance is missing or its SHA-256 no longer matches.",
    )

    tables = reference.get("tables") if isinstance(reference.get("tables"), dict) else {}
    _check(
        gate,
        "reference_tables",
        set(tables) == set(DEEP_REFERENCE_TABLES),
        sorted(DEEP_REFERENCE_TABLES),
        sorted(tables),
        "Deep reference must contain exactly UO Table 5 and KAIST Tables 9/10.",
    )
    table_failures: List[str] = []
    metric_fields = (
        "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
        "macro_precision_mean", "macro_precision_sd", "macro_recall_mean", "macro_recall_sd",
    )
    for table_name, (expected_case, expected_load, expected_source_table) in DEEP_REFERENCE_TABLES.items():
        table = tables.get(table_name)
        if not isinstance(table, dict):
            table_failures.append("{0}: missing object".format(table_name))
            continue
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        identities = {
            (str(row.get("model")), int(row.get("n_train")))
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("n_train"), int)
        }
        expected_identities = set((model, n_train) for model in DEEP_REFERENCE_MODELS for n_train in DEEP_REFERENCE_N_GRID)
        metadata_ok = (
            table.get("source_table") == expected_source_table
            and table.get("case") == expected_case
            and table.get("load") == expected_load
            and isinstance(table.get("protocol"), str)
            and bool(table.get("protocol"))
            and table.get("n_values") == list(DEEP_REFERENCE_N_GRID)
        )
        values_ok = len(rows) == 36 and identities == expected_identities
        for row in rows:
            if not isinstance(row, dict):
                values_ok = False
                continue
            for field in metric_fields:
                value = row.get(field)
                if field.endswith("_sd"):
                    values_ok = values_ok and _nonnegative_finite(value)
                else:
                    values_ok = values_ok and _nonnegative_finite(value) and float(value) <= 1.0
        if not metadata_ok or not values_ok:
            table_failures.append("{0}: metadata/36-row finite metric contract failed".format(table_name))
    _check(
        gate,
        "reference_table_contents",
        not table_failures,
        "3 tables x 6 N x 6 deep models with finite metrics",
        table_failures,
        "Deep-reference table contents are incomplete or incompatible.",
    )
    current_source_required = reference.get("confirmation_status") == "confirmed" or isinstance(
        reference.get("current_manuscript_source"), dict
    )
    current_failures: List[str] = []
    current_values_verified = 0
    if current_source_required:
        current_source = reference.get("current_manuscript_source")
        if not isinstance(current_source, dict):
            current_failures.append("current_manuscript_source is missing")
        else:
            source_tex_path = current_source.get("source_tex_path")
            source_sha256 = current_source.get("source_sha256")
            reference_version = current_source.get("reference_version")
            extraction_method = current_source.get("extraction_method")
            try:
                if extraction_method != CURRENT_MANUSCRIPT_EXTRACTION_METHOD:
                    raise RuntimeError("extraction_method is not current_manuscript_table_source")
                if not isinstance(reference_version, str) or not reference_version:
                    raise RuntimeError("reference_version is missing")
                if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
                    raise RuntimeError("source_sha256 is invalid")
                tex_bytes, tex_container, tex_member = _read_current_manuscript_tex(str(source_tex_path))
                if hashlib.sha256(tex_bytes).hexdigest() != source_sha256:
                    raise RuntimeError("LaTeX member SHA-256 mismatch")
                if tex_member is not None:
                    archive_sha256 = current_source.get("archive_sha256")
                    if not isinstance(archive_sha256, str) or sha256_file(tex_container).lower() != archive_sha256:
                        raise RuntimeError("LaTeX archive SHA-256 mismatch")
                latex_tables = _parse_current_manuscript_tables(tex_bytes)
                gate["sources"]["current_manuscript_tex"] = {
                    "path": str(tex_container),
                    "member": tex_member,
                    "source_sha256": source_sha256,
                    "archive_sha256": current_source.get("archive_sha256"),
                    "reference_version": reference_version,
                }
                for table_name, (_, _, source_table) in DEEP_REFERENCE_TABLES.items():
                    table = tables.get(table_name)
                    rows = table.get("rows", []) if isinstance(table, dict) else []
                    for row in rows:
                        if not isinstance(row, dict):
                            raise RuntimeError("reference row is not an object")
                        expected_provenance = {
                            "source_tex_path": source_tex_path,
                            "source_table": source_table,
                            "source_sha256": source_sha256,
                            "reference_version": reference_version,
                            "extraction_method": CURRENT_MANUSCRIPT_EXTRACTION_METHOD,
                        }
                        for field, expected in expected_provenance.items():
                            if row.get(field) != expected:
                                raise RuntimeError("row provenance mismatch: {0}".format(field))
                        identity = (str(row.get("model")), int(row.get("n_train")))
                        source_values = latex_tables[table_name].get(identity)
                        if source_values is None:
                            raise RuntimeError("row identity is absent from current LaTeX table")
                        for metric in CURRENT_MANUSCRIPT_METRICS:
                            for suffix in ("mean", "sd"):
                                field = metric + "_" + suffix
                                if not _decimal_matches(row.get(field), source_values[field]):
                                    raise RuntimeError("LaTeX metric mismatch: {0}".format(field))
                                current_values_verified += 1
                if current_values_verified != 648:
                    raise RuntimeError("expected 648 exact current-manuscript metric values")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                current_failures.append(str(exc))
    _check(
        gate,
        "current_manuscript_table_source_consistency",
        (not current_source_required) or not current_failures,
        "108 rows / 648 Accuracy-Precision-F1 mean/SD values exactly matching current LaTeX",
        {"values_verified": current_values_verified, "failures": current_failures},
        "Deep reference does not exactly reproduce the current Table 5/9/10 LaTeX source.",
    )
    forbidden_keys = {
        str(key)
        for table in tables.values()
        if isinstance(table, dict)
        for row in table.get("rows", [])
        if isinstance(row, dict)
        for key in row
        if str(key) in {"seed", "raw", "paired_delta", "p_value", "significance"}
    }
    _check(gate, "aggregated_only", not forbidden_keys, [], sorted(forbidden_keys), "Deep reference contains raw, paired, or significance fields.")
    return _finish_gate(gate)


def gate_full_reference(reference_path: Path) -> Dict[str, Any]:
    gate = _new_gate("04_full_reference")
    reference = _load_manifest(gate, "full_reference", Path(reference_path))
    _check(
        gate,
        "reference_source",
        reference.get("source") == "main_manuscript_stage2_experiment",
        "main_manuscript_stage2_experiment",
        reference.get("source"),
        "Full reference must come from the main manuscript Stage-2 experiment.",
    )
    _check(
        gate,
        "reference_type",
        reference.get("reference_type") == "main_manuscript_aggregated_reference",
        "main_manuscript_aggregated_reference",
        reference.get("reference_type"),
        "Full reference must be labeled as an aggregated main-manuscript reference.",
    )
    _check(gate, "reference_protocol", reference.get("protocol") == "stage2_load_shift", "stage2_load_shift", reference.get("protocol"), "Full reference protocol is not Stage-2 load shift.")
    _check(gate, "reference_n", reference.get("n_train_per_class") == 30, 30, reference.get("n_train_per_class"), "Full reference must use N=30.")
    _check(gate, "reference_runs", reference.get("runs") == 10, 10, reference.get("runs"), "Full reference must report ten runs.")
    _check(gate, "reference_aggregation", reference.get("aggregation") == TRIMMED_AGGREGATION, TRIMMED_AGGREGATION, reference.get("aggregation"), "Full reference aggregation is incompatible.")
    conditions = reference.get("conditions", {}) if isinstance(reference.get("conditions"), dict) else {}
    _check(
        gate,
        "reference_clean_conditions_only",
        set(conditions) == set(ABLATION_TABLE_CONDITIONS),
        list(ABLATION_TABLE_CONDITIONS),
        sorted(conditions),
        "Full reference must contain exactly 2Nm_0dB and 4Nm_0dB; Stage-3 -8 dB Full is prohibited.",
    )
    metric_fields = ("accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd")
    complete = all(
        isinstance(conditions.get(condition), dict)
        and all(field in conditions[condition] for field in metric_fields)
        for condition in ABLATION_TABLE_CONDITIONS
    )
    finite = complete and all(
        isinstance(conditions[condition][field], (int, float))
        and not isinstance(conditions[condition][field], bool)
        and np.isfinite(float(conditions[condition][field]))
        for condition in ABLATION_TABLE_CONDITIONS
        for field in metric_fields
    )
    _check(gate, "reference_values_complete", complete, True, complete, "Full reference metric fields are incomplete.")
    _check(gate, "reference_values_finite", finite, True, finite, "Full reference contains NaN, Inf, or non-numeric values.")
    expected_values = {
        "2Nm_0dB": {
            "accuracy_mean": 0.9999,
            "accuracy_sd": 0.0002,
            "macro_f1_mean": 0.9999,
            "macro_f1_sd": 0.0002,
        },
        "4Nm_0dB": {
            "accuracy_mean": 0.9925,
            "accuracy_sd": 0.0096,
            "macro_f1_mean": 0.9925,
            "macro_f1_sd": 0.0097,
        },
    }
    values_locked = finite and all(
        np.isclose(
            float(conditions[condition][field]),
            float(expected),
            rtol=0.0,
            atol=1.0e-12,
        )
        for condition, expected_metrics in expected_values.items()
        for field, expected in expected_metrics.items()
    )
    _check(gate, "reference_table_values", values_locked, expected_values, conditions, "Full reference values no longer match the frozen Table 9/Table 10 N=30 results.")
    params_m = reference.get("params_m")
    params_ok = (
        isinstance(params_m, (int, float))
        and not isinstance(params_m, bool)
        and np.isclose(float(params_m), 7.380, rtol=0.0, atol=1.0e-12)
    )
    _check(gate, "reference_params", params_ok, 7.380, params_m, "Full reference parameter count must remain 7.380 M.")
    return _finish_gate(gate)


def gate_additional_ablation(
    output_root: Path,
    reference_path: Path = config.KAIST_ADDITIONAL_ABLATION_FULL_REFERENCE_PATH,
) -> Dict[str, Any]:
    gate = _new_gate("04_additional_ablation")
    gate["contract"] = {
        "scope": "source_evidence",
        "variants": 5,
        "groups": 15,
        "raw_rows": 150,
        "seeds_per_group": 10,
    }
    base = Path(output_root) / "04_additional_ablation"
    raw_path = base / ABLATION_VARIANT_RAW_NAME
    summary_path = base / ABLATION_VARIANT_SUMMARY_NAME
    manifest_path = base / "additional_ablation_run_manifest.json"
    post_gate_path = base / "additional_ablation_post_gate.json"
    raw = _load_csv(gate, "raw", raw_path)
    summary = _load_csv(gate, "summary", summary_path)
    manifest = _load_manifest(gate, "run_manifest", manifest_path)
    post_gate = _load_manifest(gate, "post_gate", post_gate_path)

    raw_columns = ["variant", "condition", "seed", "accuracy", "macro_f1", "macro_precision", "macro_recall"]
    raw_ready = _has_columns(gate, raw, raw_columns, "raw")
    if raw_ready:
        duplicate_rows = int(raw.duplicated(["variant", "condition", "seed"], keep=False).sum())
        observed_seeds = sorted(int(value) for value in pd.to_numeric(raw["seed"]).unique())
        groups = raw.groupby(["variant", "condition"])["seed"].agg(lambda values: {int(v) for v in values})
        _check(gate, "raw_rows", len(raw) == 150, 150, int(len(raw)), "04 variant raw must contain exactly 150 rows.")
        _check(gate, "variants", set(raw["variant"]) == set(ABLATION_SOURCE_VARIANTS), list(ABLATION_SOURCE_VARIANTS), sorted(raw["variant"].unique()), "04 variant raw must contain exactly five reviewer variants and no Full training row.")
        _check(gate, "conditions", set(raw["condition"]) == set(ABLATION_CONDITIONS), list(ABLATION_CONDITIONS), sorted(raw["condition"].unique()), "04 must contain exactly three conditions.")
        _check(gate, "target_seeds", observed_seeds == list(TARGET_SEEDS), list(TARGET_SEEDS), observed_seeds, "04 must contain the fixed ten seeds.")
        _check(gate, "group_seed_sets", len(groups) == 15 and all(value == set(TARGET_SEEDS) for value in groups), "15 groups x identical 10 seeds", int(len(groups)), "Each 04 variant/condition group must contain all ten seeds.")
        _check(gate, "duplicates", duplicate_rows == 0, 0, duplicate_rows, "04 raw contains duplicate rows.")
        _check(gate, "finite_metrics", _finite(raw, ["seed", "accuracy", "macro_f1", "macro_precision", "macro_recall"]), True, False, "04 raw contains NaN, Inf, or non-numeric metrics.")

    summary_columns = [
        "variant", "condition", "seeds", "retained_after_trim", "aggregation",
        "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
        "reference_mean_difference", "macro_f1_reference_mean_difference", "comparison_type",
    ]
    summary_ready = _has_columns(gate, summary, summary_columns, "summary")
    if summary_ready:
        duplicate_summary = int(summary.duplicated(["variant", "condition"], keep=False).sum())
        _check(gate, "summary_groups", len(summary) == 15 and duplicate_summary == 0, 15, int(len(summary)), "04 summary must contain one row for each of 15 variant groups.")
        _check(gate, "summary_seed_count", bool((summary["seeds"] == 10).all()), 10, sorted(summary["seeds"].unique().tolist()), "04 summary must report ten seeds per group.")
        _check(gate, "retained_after_trim", bool((summary["retained_after_trim"] == 8).all()), 8, sorted(summary["retained_after_trim"].unique().tolist()), "04 summary must retain eight values.")
        _check(gate, "aggregation", bool((summary["aggregation"] == TRIMMED_AGGREGATION).all()), TRIMMED_AGGREGATION, sorted(summary["aggregation"].astype(str).unique().tolist()), "04 aggregation protocol is not manuscript-compatible.")
        _check(gate, "summary_finite", _finite(summary, ["accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"]), True, False, "04 summary contains NaN or Inf in reported metrics.")
        clean = summary[summary["condition"].isin(ABLATION_TABLE_CONDITIONS)]
        noise = summary[summary["condition"] == "4Nm_-8dB"]
        clean_differences_finite = _finite(clean, ["reference_mean_difference", "macro_f1_reference_mean_difference"])
        _check(gate, "unpaired_reference_differences", len(clean) == 10 and clean_differences_finite and bool((clean["comparison_type"] == "unpaired_aggregated_reference_difference").all()), "10 clean rows with finite unpaired aggregated-reference differences", int(len(clean)), "Clean-load comparisons are missing or incorrectly labeled.")
        noise_ok = len(noise) == 5 and bool((noise["comparison_type"] == "no_stage2_full_reference_for_noise_condition").all()) and bool((noise["reference_mean_difference"].astype(str) == "not_applicable").all())
        _check(gate, "noise_source_data_only", noise_ok, "five -8 dB rows with no Full reference", int(len(noise)), "The -8 dB source rows incorrectly use a Full reference.")
        forbidden = [name for name in summary.columns if "delta_to_full" in name or "p_value" in name or "significance" in name]
        _check(gate, "no_paired_or_significance_fields", not forbidden, [], forbidden, "Aggregated Full reference cannot support paired deltas, p-values, or significance claims.")

    _check(gate, "manifest_status", manifest.get("status") == "complete_hybrid_reference_candidate", "complete_hybrid_reference_candidate", manifest.get("status"), "04 hybrid run manifest is not complete.")
    _check(gate, "manifest_mode", manifest.get("result_mode") == HYBRID_REFERENCE_MODE, HYBRID_REFERENCE_MODE, manifest.get("result_mode"), "04 result mode is not hybrid reference ablation.")
    _check(gate, "no_full_training", manifest.get("full_model_training_performed") is False, False, manifest.get("full_model_training_performed"), "04 manifest does not prove that Full retraining was skipped.")
    _check(gate, "manifest_aggregation", manifest.get("aggregation") == TRIMMED_AGGREGATION, TRIMMED_AGGREGATION, manifest.get("aggregation"), "04 manifest aggregation is invalid.")
    manifest_post = manifest.get("post_gate", {}) if isinstance(manifest.get("post_gate"), dict) else {}
    manifest_post_ok = (
        manifest_post.get("status") == "PASS"
        and manifest_post.get("summary_rows") == 15
        and manifest_post.get("retained_after_trim") == 8
        and manifest_post.get("aggregation") == TRIMMED_AGGREGATION
        and manifest_post.get("model_runs") == 50
        and manifest_post.get("reused_first_five_model_runs") == 25
        and manifest_post.get("planned_extension_model_slots") == 25
        and manifest_post.get("failed_seeds") == []
        and manifest_post.get("paired_delta_to_full_generated") is False
        and manifest_post.get("stage3_minus8db_full_used") is False
    )
    _check(gate, "manifest_post_gate", manifest_post_ok, "PASS final 150-row/50-model hybrid post-gate", manifest_post.get("status"), "04 manifest does not contain a successful complete hybrid post-gate.")
    sidecar_ok = (
        post_gate.get("status") == "PASS"
        and post_gate.get("mode") == "full"
        and post_gate.get("final_outputs_authorized") is True
        and _manifest_hash_matches(post_gate, "manifest", manifest_path)
    )
    _check(gate, "internal_post_gate", sidecar_ok, "PASS/full/authorized with matching manifest hash", post_gate.get("status"), "04 did not finish its standalone final-results gate; stale success assets are prohibited.")
    raw_gate = manifest.get("raw_gate", {}) if isinstance(manifest.get("raw_gate"), dict) else {}
    failed = raw_gate.get("failed_seeds", None)
    _check(gate, "failed_seeds", failed == [], [], failed, "04 manifest reports failed or unknown seeds.")
    _check(gate, "raw_hash", _manifest_hash_matches(manifest, "raw", raw_path), True, False, "04 raw hash does not match its manifest.")
    _check(gate, "summary_hash", _manifest_hash_matches(manifest, "summary", summary_path), True, False, "04 summary hash does not match its manifest.")
    reference_gate = gate_full_reference(Path(reference_path))
    gate["full_reference_gate"] = reference_gate
    for item in reference_gate["checks"]:
        _check(
            gate,
            "full_" + str(item["name"]),
            bool(item["passed"]),
            item.get("expected"),
            item.get("actual"),
            item.get("reason", "Full reference gate failed."),
        )
    if Path(reference_path).is_file():
        gate["sources"]["full_reference"] = {
            "path": str(Path(reference_path).resolve()),
            "sha256": sha256_file(reference_path),
        }
    _check(gate, "reference_hash", _manifest_hash_matches(manifest, "full_reference", Path(reference_path)), True, False, "04 manifest Full-reference hash does not match the configured reference JSON.")
    return _finish_gate(gate)


def gate_ablation_main_display(output_root: Path) -> Dict[str, Any]:
    gate = _new_gate("04_additional_ablation_main_display")
    gate["contract"] = {
        "scope": "main_display",
        "variants": 4,
        "groups": 12,
        "raw_rows": 120,
        "table_rows": 5,
        "excluded_from_display": ABLATION_EXCLUDED_MAIN_VARIANT,
    }
    base = Path(output_root) / "04_additional_ablation"
    archive_dir = base / ABLATION_EQUAL_ARCHIVE_DIR_NAME

    source_raw_path = base / ABLATION_VARIANT_RAW_NAME
    source_summary_path = base / ABLATION_VARIANT_SUMMARY_NAME
    source_manifest_path = base / "additional_ablation_run_manifest.json"
    source_post_gate_path = base / "additional_ablation_post_gate.json"
    source_candidate_path = base / "manuscript_hybrid_ablation_candidate.csv"
    raw_path = base / ABLATION_MAIN_RAW_NAME
    summary_path = base / ABLATION_MAIN_SUMMARY_NAME
    candidate_path = base / ABLATION_MAIN_CANDIDATE_NAME
    candidate_tex_path = base / ABLATION_MAIN_CANDIDATE_TEX_NAME
    manifest_path = base / ABLATION_MAIN_MANIFEST_NAME
    equal_archive_path = archive_dir / ABLATION_EQUAL_ARCHIVE_NAME
    equal_candidate_path = archive_dir / ABLATION_EQUAL_CANDIDATE_NAME
    note_path = base / ABLATION_MAIN_NOTE_NAME

    source_raw = _load_csv(gate, "source_raw", source_raw_path)
    source_summary = _load_csv(gate, "source_summary", source_summary_path)
    raw = _load_csv(gate, "main_raw", raw_path)
    summary = _load_csv(gate, "main_summary", summary_path)
    candidate = _load_csv(gate, "main_candidate", candidate_path)
    equal_archive = _load_csv(gate, "equal_weights_archive", equal_archive_path)
    equal_candidate = _load_csv(gate, "equal_weights_candidate", equal_candidate_path)
    manifest = _load_manifest(gate, "main_scope_manifest", manifest_path)

    raw_columns = [
        "variant", "condition", "seed", "accuracy", "macro_f1",
        "macro_precision", "macro_recall",
    ]
    source_raw_ready = _has_columns(gate, source_raw, raw_columns, "source_raw")
    raw_ready = _has_columns(gate, raw, raw_columns, "main_raw")
    if raw_ready:
        groups = raw.groupby(["variant", "condition"])["seed"].agg(
            lambda values: {int(value) for value in values}
        )
        observed_seeds = sorted(int(value) for value in pd.to_numeric(raw["seed"]).unique())
        duplicate_rows = int(raw.duplicated(["variant", "condition", "seed"], keep=False).sum())
        _check(gate, "main_raw_rows", len(raw) == 120, 120, int(len(raw)), "Main-display raw must contain exactly 120 rows.")
        _check(gate, "main_variants", set(raw["variant"]) == set(ABLATION_MAIN_VARIANTS), list(ABLATION_MAIN_VARIANTS), sorted(raw["variant"].unique()), "Main-display raw must contain exactly four controls.")
        _check(gate, "main_conditions", set(raw["condition"]) == set(ABLATION_CONDITIONS), list(ABLATION_CONDITIONS), sorted(raw["condition"].unique()), "Main-display raw must contain exactly three conditions.")
        _check(gate, "main_target_seeds", observed_seeds == list(TARGET_SEEDS), list(TARGET_SEEDS), observed_seeds, "Main-display raw must retain all ten source seeds.")
        _check(gate, "main_group_seed_sets", len(groups) == 12 and all(values == set(TARGET_SEEDS) for values in groups), "12 groups x identical 10 seeds", int(len(groups)), "Each main-display group must contain all ten seeds.")
        _check(gate, "main_duplicates", duplicate_rows == 0, 0, duplicate_rows, "Main-display raw contains duplicate rows.")
        _check(gate, "main_finite_metrics", _finite(raw, ["seed", "accuracy", "macro_f1", "macro_precision", "macro_recall"]), True, False, "Main-display raw contains NaN or Inf.")
        _check(gate, "equal_weights_absent_from_main_raw", ABLATION_EXCLUDED_MAIN_VARIANT not in set(raw["variant"]), True, False, "equal_weights leaked into main-display raw.")

    raw_derivation_ok = False
    if source_raw_ready and raw_ready:
        expected_raw = source_raw[
            source_raw["variant"].isin(ABLATION_MAIN_VARIANTS)
        ].reset_index(drop=True)
        actual_raw = raw.reset_index(drop=True)
        raw_derivation_ok = list(actual_raw.columns) == list(expected_raw.columns)
        if raw_derivation_ok:
            try:
                pd.testing.assert_frame_equal(actual_raw, expected_raw, check_dtype=False)
            except AssertionError:
                raw_derivation_ok = False
    _check(gate, "main_raw_exact_derivation", raw_derivation_ok, "exact source raw filtered to four declared variants", raw_derivation_ok, "Main-display raw is not an exact non-destructive filter of source raw.")

    summary_columns = [
        "variant", "condition", "seeds", "retained_after_trim", "aggregation",
        "params_m", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
        "reference_mean_difference", "macro_f1_reference_mean_difference",
        "comparison_type",
    ]
    source_summary_ready = _has_columns(gate, source_summary, summary_columns, "source_summary")
    summary_ready = _has_columns(gate, summary, summary_columns, "main_summary")
    if summary_ready:
        duplicate_summary = int(summary.duplicated(["variant", "condition"], keep=False).sum())
        clean = summary[summary["condition"].isin(ABLATION_TABLE_CONDITIONS)]
        noise = summary[summary["condition"] == "4Nm_-8dB"]
        _check(gate, "main_summary_groups", len(summary) == 12 and duplicate_summary == 0, 12, int(len(summary)), "Main-display summary must contain exactly 12 unique groups.")
        _check(gate, "main_summary_variants", set(summary["variant"]) == set(ABLATION_MAIN_VARIANTS), list(ABLATION_MAIN_VARIANTS), sorted(summary["variant"].unique()), "Main-display summary contains an unexpected variant.")
        _check(gate, "main_summary_seed_count", bool((summary["seeds"] == 10).all()), 10, sorted(summary["seeds"].unique().tolist()), "Main-display summary must report ten seeds.")
        _check(gate, "main_retained_after_trim", bool((summary["retained_after_trim"] == 8).all()), 8, sorted(summary["retained_after_trim"].unique().tolist()), "Main-display summary must retain eight values.")
        _check(gate, "main_aggregation", bool((summary["aggregation"] == TRIMMED_AGGREGATION).all()), TRIMMED_AGGREGATION, sorted(summary["aggregation"].astype(str).unique().tolist()), "Main-display aggregation is incompatible.")
        _check(gate, "main_summary_finite", _finite(summary, ["params_m", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"]), True, False, "Main-display summary contains NaN or Inf.")
        _check(gate, "main_clean_comparisons", len(clean) == 8 and _finite(clean, ["reference_mean_difference", "macro_f1_reference_mean_difference"]) and bool((clean["comparison_type"] == "unpaired_aggregated_reference_difference").all()), "8 clean-load unpaired comparisons", int(len(clean)), "Main-display clean comparisons are invalid.")
        _check(gate, "main_noise_source_only", len(noise) == 4 and bool((noise["comparison_type"] == "no_stage2_full_reference_for_noise_condition").all()) and bool((noise["reference_mean_difference"].astype(str) == "not_applicable").all()), "4 noise source-data rows without Full reference", int(len(noise)), "Main-display noise rows incorrectly use a Full reference.")

    summary_derivation_ok = False
    if source_summary_ready and summary_ready:
        expected_summary = source_summary[
            source_summary["variant"].isin(ABLATION_MAIN_VARIANTS)
        ].reset_index(drop=True)
        actual_summary = summary.reset_index(drop=True)
        summary_derivation_ok = list(actual_summary.columns) == list(expected_summary.columns)
        if summary_derivation_ok:
            try:
                pd.testing.assert_frame_equal(actual_summary, expected_summary, check_dtype=False)
            except AssertionError:
                summary_derivation_ok = False
    _check(gate, "main_summary_exact_derivation", summary_derivation_ok, "exact source summary filtered to four declared variants", summary_derivation_ok, "Main-display summary is not an exact non-destructive filter of source summary.")

    expected_methods = {"Full MHFL-MCA reference"} | {
        ABLATION_METHOD_LABELS[variant] for variant in ABLATION_MAIN_VARIANTS
    }
    actual_methods = set(candidate["Method"].astype(str)) if "Method" in candidate.columns else set()
    candidate_ok = (
        len(candidate) == 5
        and actual_methods == expected_methods
        and not any("equal weight" in value.lower() for value in actual_methods)
        and not any("-8" in str(column) for column in candidate.columns)
    )
    _check(gate, "main_candidate_contract", candidate_ok, sorted(expected_methods), sorted(actual_methods), "Compact main table must contain Full plus four controls, with no equal-weight or -8 dB row.")
    candidate_tex_exists = candidate_tex_path.is_file()
    candidate_tex = candidate_tex_path.read_text(encoding="utf-8") if candidate_tex_exists else ""
    candidate_tex_lower = candidate_tex.lower()
    candidate_tex_ok = (
        candidate_tex_exists
        and "equal weight" not in candidate_tex_lower
        and all(method.lower() in candidate_tex_lower for method in expected_methods)
    )
    if candidate_tex_exists:
        gate["sources"]["main_candidate_tex"] = {
            "path": str(candidate_tex_path.resolve()),
            "sha256": sha256_file(candidate_tex_path),
        }
    _check(gate, "main_candidate_tex_contract", candidate_tex_ok, "Full plus four controls and no equal-weight row", candidate_tex.strip(), "Compact main LaTeX is missing, incomplete, or leaks equal_weights.")

    equal_ready = _has_columns(gate, equal_archive, ["variant", "record_type"], "equal_weights_archive")
    if equal_ready:
        raw_equal = equal_archive[equal_archive["record_type"] == "raw"]
        summary_equal = equal_archive[equal_archive["record_type"] == "summary"]
        archived_variants = set(equal_archive["variant"].dropna().astype(str))
        _check(gate, "equal_weights_rows_archived", len(raw_equal) == 30 and len(summary_equal) == 3 and archived_variants == {ABLATION_EXCLUDED_MAIN_VARIANT}, "30 raw + 3 summary equal_weights rows", {"raw": int(len(raw_equal)), "summary": int(len(summary_equal)), "variants": sorted(archived_variants)}, "The observed equal-weight evidence is not completely archived.")

    equal_candidate_methods = (
        equal_candidate["Method"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        if "Method" in equal_candidate.columns
        else pd.Series(dtype=str)
    )
    _check(gate, "equal_weights_candidate_archived", len(equal_candidate) == 1 and bool((equal_candidate_methods == "equal weights").all()), "one archived Equal weights candidate row", int(len(equal_candidate)), "The displayed equal-weight candidate row was not archived.")

    manifest_source_contract = manifest.get("source_evidence_contract", {}) if isinstance(manifest.get("source_evidence_contract"), Mapping) else {}
    manifest_main_contract = manifest.get("main_display_contract", {}) if isinstance(manifest.get("main_display_contract"), Mapping) else {}
    manifest_archive_contract = manifest.get("equal_weight_archive_contract", {}) if isinstance(manifest.get("equal_weight_archive_contract"), Mapping) else {}
    core_manifest_ok = (
        manifest.get("status") == "PASS"
        and manifest.get("operation") == "non_destructive_main_text_scope_filter"
        and manifest_source_contract.get("status") == "PASS"
        and manifest_source_contract.get("variants") == list(ABLATION_SOURCE_VARIANTS)
        and manifest_source_contract.get("conditions") == list(ABLATION_CONDITIONS)
        and manifest_source_contract.get("raw_rows") == 150
        and manifest_source_contract.get("summary_groups") == 15
        and manifest_source_contract.get("seeds_per_group") == 10
        and manifest_source_contract.get("retained_after_trim") == 8
        and manifest_source_contract.get("aggregation") == TRIMMED_AGGREGATION
        and manifest_main_contract.get("status") == "PASS"
        and manifest_main_contract.get("variants") == list(ABLATION_MAIN_VARIANTS)
        and manifest_main_contract.get("conditions") == list(ABLATION_CONDITIONS)
        and manifest_main_contract.get("raw_rows") == 120
        and manifest_main_contract.get("summary_groups") == 12
        and manifest_main_contract.get("seeds_per_group") == 10
        and manifest_main_contract.get("compact_table_rows") == 5
        and manifest_archive_contract == {"raw_rows": 30, "summary_rows": 3, "candidate_rows": 1}
        and manifest.get("main_scope_variants") == list(ABLATION_MAIN_VARIANTS)
        and manifest.get("excluded_from_main_display") == ABLATION_EXCLUDED_MAIN_VARIANT
        and manifest.get("equal_weight_result_deleted") is False
        and manifest.get("equal_weight_result_archived") is True
    )
    _check(gate, "main_scope_manifest_contract", core_manifest_ok, "PASS non-destructive 4-variant main scope", manifest.get("status"), "Main-scope manifest is incomplete or misstates equal-weight preservation.")
    claim_boundary = str(manifest.get("claim_boundary", "")).lower()
    _check(gate, "main_scope_claim_boundary", "equal-weight" in claim_boundary and "supplementary/provenance" in claim_boundary, "explicit evaluated-and-retained claim boundary", claim_boundary, "Main-scope manifest omits the equal-weight disclosure boundary.")

    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), Mapping) else {}
    output_targets = {
        "raw_main_scope": raw_path,
        "summary_main_scope": summary_path,
        "equal_weights_archive": equal_archive_path,
        "candidate_main_scope": candidate_path,
        "candidate_main_scope_tex": candidate_tex_path,
        "candidate_equal_weights": equal_candidate_path,
        "main_scope_note": note_path,
    }
    for key, path in output_targets.items():
        _check(gate, "main_manifest_{0}_hash".format(key), _manifest_output_matches(outputs, key, path), True, False, "Main-scope output path/hash mismatch: {0}".format(key))

    required_sources = (
        source_raw_path,
        source_summary_path,
        source_candidate_path,
        base / "manuscript_hybrid_ablation_candidate.tex",
        source_manifest_path,
        source_post_gate_path,
    )
    archive_records = manifest.get("archive_records", [])
    archive_records_ok = isinstance(archive_records, list)
    for source_path in required_sources:
        archive_path = archive_dir / source_path.name
        matches = [
            record for record in archive_records
            if isinstance(record, Mapping)
            and str(record.get("archive", "")).lower() == str(archive_path.resolve()).lower()
        ] if isinstance(archive_records, list) else []
        archive_records_ok = archive_records_ok and len(matches) == 1 and _archive_record_matches(
            matches[0], source_path, archive_path
        )
        if archive_path.is_file():
            gate["sources"]["archived_" + source_path.name] = {
                "path": str(archive_path.resolve()),
                "sha256": sha256_file(archive_path),
            }
    _check(gate, "source_archive_records", archive_records_ok, "six original 04 evidence files copied with matching source/archive SHA-256", archive_records_ok, "Original 04 evidence archive is missing, stale, or hash-inconsistent.")

    expected_source_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in required_sources
        if path.is_file()
    }
    source_hashes_ok = (
        len(expected_source_hashes) == len(required_sources)
        and manifest.get("source_hashes_before") == expected_source_hashes
        and manifest.get("source_hashes_after") == expected_source_hashes
        and manifest.get("original_source_files_modified") is False
    )
    _check(gate, "source_hashes_unchanged", source_hashes_ok, "identical before/after hashes for all six source files", {"before": manifest.get("source_hashes_before"), "after": manifest.get("source_hashes_after")}, "Main-scope derivation does not prove that all original 04 sources remained unchanged.")

    note_exists = note_path.is_file()
    note_text = note_path.read_text(encoding="utf-8") if note_exists else ""
    if note_exists:
        gate["sources"]["main_scope_note"] = {
            "path": str(note_path.resolve()),
            "sha256": sha256_file(note_path),
        }
    note_lower = note_text.lower()
    note_ok = (
        note_exists
        and "equal-weight" in note_lower
        and "control" in note_lower
        and "not been deleted" in note_lower
        and "supplementary/provenance" in note_lower
    )
    _check(gate, "supplementary_provenance_note", note_ok, "explicit evaluated, retained, not-deleted equal-weight disclosure", note_text.strip(), "Main-scope provenance note is missing or misleading.")
    return _finish_gate(gate)


def build_hybrid_ablation_table(
    summary: pd.DataFrame,
    reference: Mapping[str, Any],
) -> pd.DataFrame:
    required = {
        "variant", "condition", "params_m", "accuracy_mean", "accuracy_sd",
        "macro_f1_mean", "macro_f1_sd", "comparison_type",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError("Hybrid ablation summary is missing columns: {0}".format(missing))
    if "full" in set(summary["variant"].astype(str)):
        raise RuntimeError("Hybrid ablation summary must not contain a newly trained Full row.")
    if set(summary["variant"]) != set(ABLATION_MAIN_VARIANTS):
        raise RuntimeError("Hybrid ablation main-display summary must contain exactly the four displayed reviewer controls.")
    clean = summary[summary["condition"].isin(ABLATION_TABLE_CONDITIONS)].copy()
    if len(clean) != len(ABLATION_MAIN_VARIANTS) * len(ABLATION_TABLE_CONDITIONS):
        raise RuntimeError("Hybrid manuscript table requires exactly eight clean-load main-display rows.")
    if not (clean["comparison_type"] == "unpaired_aggregated_reference_difference").all():
        raise RuntimeError("Hybrid clean-load rows must use unpaired aggregated-reference comparisons.")
    conditions = reference.get("conditions", {})
    if not isinstance(conditions, dict) or set(conditions) != set(ABLATION_TABLE_CONDITIONS):
        raise RuntimeError("Hybrid Full reference must contain only the two Stage-2 clean-load conditions.")

    def formatted(mean: Any, sd: Any) -> str:
        return "{0:.4f} ± {1:.4f}".format(float(mean), float(sd))

    full_row = {
        "Method": "Full MHFL-MCA reference",
        "Params (M)": "{0:.3f}".format(float(reference["params_m"])),
        "2 Nm Accuracy": formatted(conditions["2Nm_0dB"]["accuracy_mean"], conditions["2Nm_0dB"]["accuracy_sd"]),
        "2 Nm Macro-F1": formatted(conditions["2Nm_0dB"]["macro_f1_mean"], conditions["2Nm_0dB"]["macro_f1_sd"]),
        "4 Nm Accuracy": formatted(conditions["4Nm_0dB"]["accuracy_mean"], conditions["4Nm_0dB"]["accuracy_sd"]),
        "4 Nm Macro-F1": formatted(conditions["4Nm_0dB"]["macro_f1_mean"], conditions["4Nm_0dB"]["macro_f1_sd"]),
        "Source": "main manuscript Table 9/Table 10, N=30, 10-run Stage-2",
        "Comparison type": "published_main_experiment_summary_reference",
    }
    rows: List[Dict[str, Any]] = [full_row]
    for variant in ABLATION_MAIN_VARIANTS:
        variant_rows = clean[clean["variant"] == variant].set_index("condition")
        if set(variant_rows.index) != set(ABLATION_TABLE_CONDITIONS):
            raise RuntimeError("Variant {0} is missing a clean-load result.".format(variant))
        two = variant_rows.loc["2Nm_0dB"]
        four = variant_rows.loc["4Nm_0dB"]
        rows.append(
            {
                "Method": ABLATION_METHOD_LABELS[variant],
                "Params (M)": "{0:.6f}".format(float(two["params_m"])),
                "2 Nm Accuracy": formatted(two["accuracy_mean"], two["accuracy_sd"]),
                "2 Nm Macro-F1": formatted(two["macro_f1_mean"], two["macro_f1_sd"]),
                "4 Nm Accuracy": formatted(four["accuracy_mean"], four["accuracy_sd"]),
                "4 Nm Macro-F1": formatted(four["macro_f1_mean"], four["macro_f1_sd"]),
                "Source": "new 10-run Stage-2 additional-ablation variant",
                "Comparison type": "unpaired_aggregated_reference_difference",
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != 5 or any("-8" in name for name in table.columns):
        raise RuntimeError("Hybrid manuscript table must contain Full plus four controls and no -8 dB Full comparison.")
    if any("equal weight" in value.lower() for value in table["Method"].astype(str)):
        raise RuntimeError("The compact main table must not display the archived equal-weight control.")
    if any("delta_to_full" in name or "p_value" in name for name in table.columns):
        raise RuntimeError("Hybrid manuscript table cannot contain paired deltas or p-values.")
    return table


def _lowshot_figure_paths(base: Path) -> Dict[str, Path]:
    figure_bundle = Path(base) / "lowshot_evidence_bundle"
    return {
        "png": figure_bundle / "lowshot_evidence.png",
        "pdf": figure_bundle / "lowshot_evidence.pdf",
        "svg": figure_bundle / "lowshot_evidence.svg",
        "tiff": figure_bundle / "lowshot_evidence.tiff",
        "contract": figure_bundle / "figure_contract.json",
    }


def gate_lowshot(output_root: Path) -> Dict[str, Any]:
    gate = _new_gate("05_lowshot_threshold")
    base = Path(output_root) / "05_lowshot_threshold"
    raw_path = base / "lowshot_raw.csv"
    summary_path = base / "lowshot_summary.csv"
    paired_path = base / "caim_paired_summary.csv"
    threshold_path = base / "operational_thresholds.json"
    anchor_path = base / "lowshot_anchor_gate.json"
    manifest_path = base / "lowshot_run_manifest.json"
    post_gate_path = base / "lowshot_post_gate.json"
    execution_state_path = base / "lowshot_execution_state.json"
    raw = _load_csv(gate, "raw", raw_path)
    summary = _load_csv(gate, "summary", summary_path)
    paired = _load_csv(gate, "paired_summary", paired_path)
    thresholds = _load_json_value(gate, "operational_thresholds", threshold_path)
    anchor = _load_manifest(gate, "anchor_gate", anchor_path)
    manifest = _load_manifest(gate, "run_manifest", manifest_path)
    post_gate = _load_manifest(gate, "post_gate", post_gate_path)
    execution_state = _load_manifest(gate, "execution_state", execution_state_path)

    raw_columns = [
        "variant", "n_train", "run_idx", "seed", "test_accuracy",
        "test_macro_precision", "test_macro_recall", "test_macro_f1",
        "train_accuracy", "heldout_accuracy", "generalization_gap",
        "split_signature", "data_signature", "hyperparameter_signature",
        "source_type", "train_time_s",
    ]
    raw_ready = _has_columns(gate, raw, raw_columns, "raw")
    if raw_ready:
        duplicate_rows = int(raw.duplicated(["variant", "n_train", "seed"], keep=False).sum())
        expected_group_seeds = {
            (variant, n_train): set(LOWSHOT_SEED_MAP[n_train])
            for variant in LOWSHOT_VARIANTS
            for n_train in LOWSHOT_N_GRID
        }
        numeric_n = pd.to_numeric(raw["n_train"], errors="coerce")
        numeric_run = pd.to_numeric(raw["run_idx"], errors="coerce")
        numeric_seed = pd.to_numeric(raw["seed"], errors="coerce")
        schedule_numeric_ok = bool(
            numeric_n.notna().all()
            and numeric_run.notna().all()
            and numeric_seed.notna().all()
            and (numeric_n == np.floor(numeric_n)).all()
            and (numeric_run == np.floor(numeric_run)).all()
            and (numeric_seed == np.floor(numeric_seed)).all()
        )
        observed_group_seeds: Dict[Any, Any] = {}
        pair_sizes = pd.Series(dtype=np.int64)
        split_pairs = pd.DataFrame()
        observed_n_grid: List[int] = []
        seed_formula_ok = False
        if schedule_numeric_ok:
            n_values = numeric_n.astype(int)
            run_values = numeric_run.astype(int)
            seed_values = numeric_seed.astype(int)
            indexed = raw.assign(_n_train=n_values, _run_idx=run_values, _seed=seed_values)
            groups = indexed.groupby(["variant", "_n_train"])["_seed"].agg(lambda values: set(int(v) for v in values))
            observed_group_seeds = {
                (str(variant), int(n_train)): seeds
                for (variant, n_train), seeds in groups.items()
            }
            pair_sizes = indexed.groupby(["_n_train", "_seed"]).size()
            split_pairs = indexed.groupby(["_n_train", "_seed"])[["split_signature", "data_signature"]].nunique()
            observed_n_grid = sorted(n_values.unique().tolist())
            seed_formula_ok = bool(
                run_values.between(1, 10).all()
                and (seed_values == 100 * n_values + run_values).all()
            )
        _check(gate, "raw_rows", len(raw) == 140, 140, int(len(raw)), "05 raw must contain exactly 140 rows.")
        _check(gate, "variants", set(raw["variant"]) == set(LOWSHOT_VARIANTS), list(LOWSHOT_VARIANTS), sorted(raw["variant"].unique()), "05 must contain full and no-CAIM variants.")
        _check(gate, "n_grid", schedule_numeric_ok and set(observed_n_grid) == set(LOWSHOT_N_GRID), list(LOWSHOT_N_GRID), observed_n_grid, "05 must contain the seven predefined N values.")
        _check(gate, "paper_seed_formula", seed_formula_ok, "seed = 100*N + run_idx, run_idx=1..10", seed_formula_ok, "05 does not use the original N-specific paper seed schedule.")
        _check(gate, "group_seed_sets", len(observed_group_seeds) == 14 and observed_group_seeds == expected_group_seeds, {str(key): sorted(value) for key, value in expected_group_seeds.items()}, {str(key): sorted(value) for key, value in observed_group_seeds.items()}, "Each 05 variant/N group must contain its exact ten N-specific paper seeds.")
        matched_splits_ok = (
            len(split_pairs) == 70
            and bool((pair_sizes == 2).all())
            and bool((split_pairs == 1).all().all())
        )
        _check(gate, "matched_splits", matched_splits_ok, "70 Full/no-CAIM N/seed pairs with identical split/data signatures", int(len(split_pairs)), "Full and no-CAIM are not exactly paired on all 70 paper-aligned splits.")
        _check(gate, "duplicates", duplicate_rows == 0, 0, duplicate_rows, "05 raw contains duplicate rows.")
        raw_numeric = ["n_train", "run_idx", "seed", "train_time_s"] + list(LOWSHOT_SUMMARY_METRICS)
        _check(gate, "finite_metrics", _finite(raw, raw_numeric), True, False, "05 raw contains NaN, Inf, or non-numeric metrics.")
        signature_columns = ["split_signature", "data_signature", "hyperparameter_signature", "source_type"]
        nonempty_signatures = all(raw[column].astype(str).str.strip().ne("").all() for column in signature_columns)
        source_type_ok = set(raw["source_type"].astype(str)) == {"new_paper_aligned_training"}
        _check(gate, "paper_aligned_source_signatures", nonempty_signatures and source_type_ok, "non-empty signatures and new_paper_aligned_training", sorted(raw["source_type"].astype(str).unique().tolist()), "05 raw is not identified as paper-aligned training evidence.")

    metric_names = LOWSHOT_SUMMARY_METRICS
    summary_columns = ["variant", "n_train", "seeds", "seeds_total", "retained_after_trim", "aggregation"] + [
        metric + suffix
        for metric in metric_names
        for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")
    ]
    summary_ready = _has_columns(gate, summary, summary_columns, "summary")
    if summary_ready:
        _check(gate, "summary_groups", len(summary) == 14 and not summary.duplicated(["variant", "n_train"]).any(), 14, int(len(summary)), "05 summary must contain exactly 14 unique groups.")
        summary_n = pd.to_numeric(summary["n_train"], errors="coerce")
        summary_n_ok = bool(summary_n.notna().all() and (summary_n == np.floor(summary_n)).all())
        observed_summary_n = sorted(summary_n.dropna().astype(int).unique().tolist()) if summary_n_ok else []
        summary_scope_ok = (
            set(summary["variant"].astype(str)) == set(LOWSHOT_VARIANTS)
            and summary_n_ok
            and set(observed_summary_n) == set(LOWSHOT_N_GRID)
        )
        _check(gate, "summary_scope", summary_scope_ok, "2 variants x 7 N", {"variants": sorted(summary["variant"].astype(str).unique().tolist()), "n_grid": observed_summary_n}, "05 summary does not cover the exact paper-aligned 2 x 7 scope.")
        summary_seed_ok = bool((pd.to_numeric(summary["seeds"], errors="coerce") == 10).all() and (pd.to_numeric(summary["seeds_total"], errors="coerce") == 10).all())
        _check(gate, "summary_seed_count", summary_seed_ok, 10, {"seeds": sorted(summary["seeds"].unique().tolist()), "seeds_total": sorted(summary["seeds_total"].unique().tolist())}, "05 summary must report ten seeds in both seed-count fields.")
        _check(gate, "retained_after_trim", bool((summary["retained_after_trim"] == 8).all()), 8, sorted(summary["retained_after_trim"].unique().tolist()), "05 summary must retain eight values.")
        _check(gate, "aggregation", bool((summary["aggregation"] == TRIMMED_AGGREGATION).all()), TRIMMED_AGGREGATION, sorted(summary["aggregation"].astype(str).unique().tolist()), "05 aggregation protocol is not manuscript-compatible.")
        numeric_summary = [metric + suffix for metric in metric_names for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")]
        _check(gate, "summary_finite", _finite(summary, numeric_summary), True, False, "05 summary contains NaN or Inf.")

    raw_metric_finite = raw_ready and _finite(raw, list(LOWSHOT_SUMMARY_METRICS))
    summary_metric_finite = summary_ready and _finite(summary, [metric + suffix for metric in metric_names for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")])
    summary_matches_raw = raw_metric_finite and summary_metric_finite and len(raw) == 140 and len(summary) == 14
    if summary_matches_raw:
        try:
            for (variant, n_train), block in raw.groupby(["variant", "n_train"], sort=True):
                rows = summary[(summary["variant"] == variant) & (summary["n_train"] == int(n_train))]
                if len(rows) != 1:
                    summary_matches_raw = False
                    break
                actual = rows.iloc[0]
                for metric in metric_names:
                    mean, sd, untrimmed_mean, untrimmed_sd = _trimmed_statistics(block[metric].to_numpy(dtype=np.float64))
                    if not all((
                        _close(actual[metric + "_mean"], mean),
                        _close(actual[metric + "_sd"], sd),
                        _close(actual[metric + "_untrimmed_mean"], untrimmed_mean),
                        _close(actual[metric + "_untrimmed_sd"], untrimmed_sd),
                    )):
                        summary_matches_raw = False
                        break
                if not summary_matches_raw:
                    break
        except (TypeError, ValueError):
            summary_matches_raw = False
    _check(gate, "summary_recomputed_from_raw", summary_matches_raw, True, summary_matches_raw, "05 summary does not reproduce independent metric-wise trimming of the raw ten-seed results.")

    paired_columns = [
        "n_train", "seeds", "seeds_total", "retained_after_trim", "aggregation",
        "paired_gain_mean", "paired_gain_sd", "paired_gain_untrimmed_mean", "paired_gain_untrimmed_sd",
        "caim_gain_mean", "caim_gain_sd", "caim_gain_untrimmed_mean", "caim_gain_untrimmed_sd",
    ]
    paired_ready = _has_columns(gate, paired, paired_columns, "paired_summary")
    if paired_ready:
        paired_n = pd.to_numeric(paired["n_train"], errors="coerce")
        paired_n_ok = bool(paired_n.notna().all() and (paired_n == np.floor(paired_n)).all())
        observed_paired_n = set(paired_n.dropna().astype(int).tolist()) if paired_n_ok else set()
        paired_ok = (
            len(paired) == 7
            and not paired.duplicated(["n_train"]).any()
            and paired_n_ok
            and observed_paired_n == set(LOWSHOT_N_GRID)
            and bool((pd.to_numeric(paired["seeds"], errors="coerce") == 10).all())
            and bool((pd.to_numeric(paired["seeds_total"], errors="coerce") == 10).all())
            and bool((paired["retained_after_trim"] == 8).all())
            and bool((paired["aggregation"] == TRIMMED_AGGREGATION).all())
        )
        _check(gate, "paired_summary_protocol", paired_ok, "7 rows, 10 seeds, 8 retained, trimmed aggregation", int(len(paired)), "05 paired CAIM summary is incomplete or uses the wrong aggregation.")
        paired_metric_columns = [
            prefix + suffix
            for prefix in ("paired_gain", "caim_gain")
            for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")
        ]
        _check(gate, "paired_summary_finite", _finite(paired, paired_metric_columns), True, False, "05 paired summary contains NaN or Inf.")
        aliases_match = all(
            np.allclose(
                pd.to_numeric(paired["paired_gain" + suffix]),
                pd.to_numeric(paired["caim_gain" + suffix]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")
        )
        _check(gate, "paired_gain_aliases", aliases_match, "canonical paired_gain_* equals caim_gain_* aliases", aliases_match, "05 backward-compatible CAIM aliases disagree with the canonical paired-gain fields.")

    paired_metric_columns = [
        prefix + suffix
        for prefix in ("paired_gain", "caim_gain")
        for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")
    ]
    paired_matches_raw = (
        raw_ready
        and paired_ready
        and _finite(raw, ["n_train", "seed", "test_accuracy"])
        and _finite(paired, ["n_train"] + paired_metric_columns)
        and len(raw) == 140
        and len(paired) == 7
    )
    if paired_matches_raw:
        try:
            pivot = raw.pivot(index=["n_train", "seed"], columns="variant", values="test_accuracy")
            if set(pivot.columns) != set(LOWSHOT_VARIANTS) or len(pivot) != 70:
                paired_matches_raw = False
            else:
                differences = (pivot["full"] - pivot["no_caim"]).rename("gain").reset_index()
                for n_train, block in differences.groupby("n_train", sort=True):
                    rows = paired[paired["n_train"] == int(n_train)]
                    if len(rows) != 1:
                        paired_matches_raw = False
                        break
                    mean, sd, untrimmed_mean, untrimmed_sd = _trimmed_statistics(block["gain"].to_numpy(dtype=np.float64))
                    actual = rows.iloc[0]
                    if not all((
                        _close(actual["paired_gain_mean"], mean),
                        _close(actual["paired_gain_sd"], sd),
                        _close(actual["paired_gain_untrimmed_mean"], untrimmed_mean),
                        _close(actual["paired_gain_untrimmed_sd"], untrimmed_sd),
                    )):
                        paired_matches_raw = False
                        break
        except (KeyError, TypeError, ValueError):
            paired_matches_raw = False
    _check(gate, "paired_gain_recomputed_from_seed_pairs", paired_matches_raw, True, paired_matches_raw, "05 CAIM gain was not computed full-minus-noCAIM by seed before trimming.")

    threshold_ok = isinstance(thresholds, list) and len(thresholds) == 2
    if threshold_ok:
        threshold_ok = {row.get("variant") for row in thresholds if isinstance(row, dict)} == set(LOWSHOT_VARIANTS)
        threshold_ok = threshold_ok and all(
            isinstance(row, dict)
            and "trimmed test accuracy" in str(row.get("criterion", ""))
            and "smallest evaluated n satisfying" in str(row.get("claim_limit", "")).lower()
            and "criterion" in str(row.get("claim_limit", "")).lower()
            and "not a universal" in str(row.get("claim_limit", "")).lower()
            and (row.get("first_empirical_n") is None or row.get("first_empirical_n") in LOWSHOT_N_GRID)
            for row in thresholds
        )
    _check(gate, "operational_threshold_scope", threshold_ok, "two protocol-specific trimmed operational thresholds", threshold_ok, "05 operational thresholds are missing or overstate a universal theoretical threshold.")

    anchor_summary_ok = summary_ready
    anchor_summary_details: Dict[str, Any] = {}
    if anchor_summary_ok:
        full_summary = summary[summary["variant"].astype(str) == "full"]
        for n_train, expected_fields in LOWSHOT_TABLE5_ANCHORS.items():
            rows = full_summary[pd.to_numeric(full_summary["n_train"], errors="coerce") == n_train]
            field_details: Dict[str, Any] = {}
            if len(rows) != 1:
                anchor_summary_ok = False
                anchor_summary_details[str(n_train)] = {"row_count": int(len(rows))}
                continue
            summary_row = rows.iloc[0]
            for field, expected in expected_fields.items():
                actual = summary_row.get(field)
                matches = "{0:.4f}".format(float(actual)) == "{0:.4f}".format(float(expected)) if _close(actual, actual) else False
                field_details[field] = {
                    "expected_4dp": "{0:.4f}".format(expected),
                    "actual_4dp": None if not _close(actual, actual) else "{0:.4f}".format(float(actual)),
                    "passed": bool(matches),
                }
                anchor_summary_ok = anchor_summary_ok and matches
            anchor_summary_details[str(n_train)] = field_details
    _check(gate, "table5_anchor_recomputed_from_summary", anchor_summary_ok, LOWSHOT_TABLE5_ANCHORS, anchor_summary_details, "05 Full summary does not reproduce the predeclared Table-5 N=5/N=10 anchors to four decimals.")

    expected_anchor_keys = {str(value) for value in LOWSHOT_TABLE5_ANCHORS}
    anchor_records = anchor.get("anchors", {}) if isinstance(anchor.get("anchors"), Mapping) else {}
    anchor_json_ok = (
        anchor.get("status") == "PASS"
        and anchor.get("required_for_manuscript") is True
        and anchor.get("final_assets_authorized") is True
        and set(anchor_records) == expected_anchor_keys
    )
    for n_train, expected_fields in LOWSHOT_TABLE5_ANCHORS.items():
        record = anchor_records.get(str(n_train), {}) if isinstance(anchor_records.get(str(n_train)), Mapping) else {}
        fields = record.get("fields", {}) if isinstance(record.get("fields"), Mapping) else {}
        anchor_json_ok = anchor_json_ok and record.get("passed") is True and set(fields) == set(expected_fields)
        for field, expected in expected_fields.items():
            item = fields.get(field, {}) if isinstance(fields.get(field), Mapping) else {}
            anchor_json_ok = anchor_json_ok and (
                item.get("passed") is True
                and item.get("expected_4dp") == "{0:.4f}".format(expected)
                and item.get("actual_4dp") == "{0:.4f}".format(expected)
            )
    _check(gate, "anchor_gate", anchor_json_ok, "PASS/authorized Table-5 N=5,N=10 anchor", anchor.get("status"), "05 anchor gate is absent, incomplete, unauthorized, or inconsistent with Table 5.")

    expected_seed_map_json = {str(key): list(value) for key, value in LOWSHOT_SEED_MAP.items()}
    expected_anchor_json = {str(key): value for key, value in LOWSHOT_TABLE5_ANCHORS.items()}
    raw_gate = manifest.get("raw_gate", {}) if isinstance(manifest.get("raw_gate"), Mapping) else {}
    raw_gate_ok = (
        raw_gate.get("status") == "PASS"
        and raw_gate.get("rows") == 140
        and raw_gate.get("n_grid") == list(LOWSHOT_N_GRID)
        and raw_gate.get("runs_per_n") == 10
        and raw_gate.get("seed_map") == expected_seed_map_json
        and raw_gate.get("duplicates") == 0
        and raw_gate.get("nan_or_inf") == 0
        and raw_gate.get("matched_split_pairs") == 70
    )
    hparams = manifest.get("hparams", {}) if isinstance(manifest.get("hparams"), Mapping) else {}
    optimizer = manifest.get("optimizer", {}) if isinstance(manifest.get("optimizer"), Mapping) else {}
    optimizer_ok = (
        str(optimizer.get("name", "")).lower() == "adamax"
        and _positive_finite(optimizer.get("learning_rate"))
        and isinstance(optimizer.get("batch_size"), int)
        and not isinstance(optimizer.get("batch_size"), bool)
        and int(optimizer.get("batch_size")) > 0
        and _close(optimizer.get("learning_rate"), hparams.get("lr"))
        and optimizer.get("batch_size") == hparams.get("batch_size")
    )
    manifest_protocol_ok = (
        manifest.get("status") == "PASS"
        and manifest.get("mode") == "full"
        and manifest.get("protocol") == "uo_source_aligned_paired_deterministic_extension_v1"
        and bool(str(manifest.get("protocol_alignment_scope", "")).strip())
        and bool(str(manifest.get("random_initialization_boundary", "")).strip())
        and manifest.get("seed_schedule") == "seed = 100*N + run_idx"
        and "held-out" in str(manifest.get("split_protocol", "")).lower()
        and "final evaluation" in str(manifest.get("split_protocol", "")).lower()
        and manifest.get("epochs") == 80
        and optimizer_ok
        and manifest.get("gradient_clipping") is None
        and manifest.get("early_stopping") is False
        and manifest.get("final_epoch_weights") is True
        and manifest.get("model_parameter_count") == 5_111_759
        and manifest.get("aggregation") == TRIMMED_AGGREGATION
        and manifest.get("paired_by_seed_before_trim") is True
        and manifest.get("failed_runs") == 0
        and manifest.get("failed_seeds") == []
        and raw_gate_ok
        and manifest.get("anchor_gate") == anchor
        and manifest.get("paper_anchors") == expected_anchor_json
    )
    _check(gate, "paper_aligned_manifest", manifest_protocol_ok, "source-aligned paired deterministic-extension full manifest with an explicit initializer boundary", manifest.get("protocol"), "05 run manifest does not prove the declared source-aligned protocol and deterministic-extension boundary.")
    execution_state_ok = (
        execution_state.get("status") == "PASS"
        and execution_state.get("mode") == "full"
        and execution_state.get("protocol") == "uo_source_aligned_paired_deterministic_extension_v1"
        and execution_state.get("completed_run_artifacts") == 140
        and execution_state.get("final_outputs_authorized") is True
    )
    _check(gate, "execution_state", execution_state_ok, "PASS/full protocol-matched state with 140 completed artifacts and final authorization", {key: execution_state.get(key) for key in ("status", "mode", "protocol", "completed_run_artifacts", "final_outputs_authorized")}, "05 execution state is missing, incomplete, or not authorized for final manuscript assets.")

    manifest_post = manifest.get("post_gate", {}) if isinstance(manifest.get("post_gate"), Mapping) else {}
    post_ok = (
        post_gate.get("status") == "PASS"
        and post_gate.get("mode") == "full"
        and post_gate.get("final_outputs_authorized") is True
        and post_gate.get("raw_rows") == 140
        and post_gate.get("summary_rows") == 14
        and post_gate.get("paired_gain_rows") == 7
        and post_gate.get("variants") == list(LOWSHOT_VARIANTS)
        and post_gate.get("n_grid") == list(LOWSHOT_N_GRID)
        and post_gate.get("runs_per_group") == 10
        and post_gate.get("seed_map") == expected_seed_map_json
        and post_gate.get("retained_after_trim") == 8
        and post_gate.get("aggregation") == TRIMMED_AGGREGATION
        and post_gate.get("duplicates") == 0
        and post_gate.get("nan_or_inf") == 0
        and post_gate.get("failed_runs") == 0
        and post_gate.get("failed_seeds") == []
        and post_gate.get("matched_split_pairs") == 70
        and post_gate.get("summary_recomputed_from_raw") is True
        and post_gate.get("paired_gain_aligned_by_seed_before_trim") is True
        and post_gate.get("operational_threshold_recomputed_from_trimmed_statistics") is True
        and post_gate.get("anchor_gate_pass") is True
        and manifest_post == post_gate
    )
    _check(gate, "internal_post_gate", post_ok, "authorized exact 140/14/7/70 paper-aligned post-gate", post_gate.get("status"), "05 did not finish its strict paper-aligned final-results gate.")

    figure_path_map = _lowshot_figure_paths(base)
    figure_contract = _load_manifest(gate, "figure_contract", figure_path_map["contract"])
    figure_paths = list(figure_path_map.values())
    figures_ok = all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)
    _check(gate, "figure_bundle", figures_ok, [path.name for path in figure_paths], [path.name for path in figure_paths if path.is_file() and path.stat().st_size > 0], "05 final PNG/PDF/SVG/TIFF/contract bundle is incomplete.")
    contract_sources = figure_contract.get("source_data", [])
    expected_contract_sources = [str(summary_path.resolve()), str(paired_path.resolve())]
    panel_labels = figure_contract.get("panel_y_labels", [])
    role_text = str(figure_contract.get("evaluation_set_role", "")).lower()
    claim_limits = figure_contract.get("claim_limits", [])
    contract_ok = (
        figure_contract.get("figure_id") == "extreme-lowshot-paper-aligned"
        and contract_sources == expected_contract_sources
        and figure_contract.get("replicate_unit") == "paper seed schedule 100*N+run_idx"
        and "trimmed mean" in str(figure_contract.get("center_statistic", "")).lower()
        and "population sd" in str(figure_contract.get("spread_definition", "")).lower()
        and "eight retained" in str(figure_contract.get("spread_definition", "")).lower()
        and panel_labels == ["Held-out accuracy", "Train–held-out gap", "Paired CAIM accuracy gain"]
        and "held-out" in role_text
        and ("no independent validation" in role_text or "not an independent validation" in role_text)
        and "test" in role_text
        and "train–validation gap" not in json.dumps(figure_contract, ensure_ascii=False).lower()
        and figure_contract.get("anchor_gate") == anchor
        and isinstance(claim_limits, list)
        and any("not a universal" in str(value).lower() for value in claim_limits)
    )
    _check(gate, "figure_contract", contract_ok, "paper-aligned Train–held-out bundle contract with no independent validation/test claim", figure_contract.get("figure_id"), "05 figure contract is stale, mislabeled, or overstates an independent validation/test set.")

    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), Mapping) else {}
    required_output_keys = {
        "raw", "summary", "paired_summary", "operational_thresholds", "anchor_gate",
        "post_gate", "hparams_config", "execution_state", "figure_png", "figure_pdf",
        "figure_svg", "figure_tiff", "figure_contract",
    }
    _check(gate, "manifest_output_keys", set(outputs) == required_output_keys, sorted(required_output_keys), sorted(outputs), "05 manifest output inventory is incomplete or contains a stale contract.")
    output_targets = {
        "raw": (raw_path, 140),
        "summary": (summary_path, 14),
        "paired_summary": (paired_path, 7),
        "operational_thresholds": (threshold_path, None),
        "anchor_gate": (anchor_path, None),
        "post_gate": (post_gate_path, None),
        "execution_state": (execution_state_path, None),
        "figure_png": (figure_path_map["png"], None),
        "figure_pdf": (figure_path_map["pdf"], None),
        "figure_svg": (figure_path_map["svg"], None),
        "figure_tiff": (figure_path_map["tiff"], None),
        "figure_contract": (figure_path_map["contract"], None),
    }
    for key, (path, rows) in output_targets.items():
        _check(gate, "manifest_{0}_hash".format(key), _lowshot_output_record_matches(outputs, key, path, rows), True, False, "05 output path/size/hash mismatch: {0}.".format(key))
    hparams_path = Path(str(hparams.get("source_path", "")))
    hparams_ok = (
        hparams_path.is_file()
        and hparams.get("source_sha256") == sha256_file(hparams_path)
        and _lowshot_output_record_matches(outputs, "hparams_config", hparams_path)
    )
    _check(gate, "manifest_hparams_hash", hparams_ok, "confirmed UO hparams path/size/SHA-256", str(hparams_path), "05 manifest does not bind the run to its confirmed UO hyperparameter artifact.")

    dataset_provenance = manifest.get("dataset_provenance", {}) if isinstance(manifest.get("dataset_provenance"), Mapping) else {}
    dataset_files = dataset_provenance.get("files", [])
    dataset_settings = dataset_provenance.get("settings", {})
    dataset_signature = dataset_provenance.get("content_signature")
    dataset_files_ok = (
        dataset_provenance.get("source_file_count") == 14
        and isinstance(dataset_files, list)
        and len(dataset_files) == 14
        and isinstance(dataset_settings, Mapping)
        and bool(dataset_settings)
        and isinstance(dataset_signature, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", dataset_signature))
    )
    resolved_dataset_paths: List[str] = []
    if isinstance(dataset_files, list):
        for index, record in enumerate(dataset_files):
            record_ok = isinstance(record, Mapping) and set(record) == {
                "path", "size_bytes", "modification_time_ns", "sha256"
            }
            source_path = Path(str(record.get("path", ""))) if isinstance(record, Mapping) else Path("")
            if record_ok:
                try:
                    stat = source_path.stat()
                    current_sha256 = sha256_file(source_path) if source_path.is_file() else ""
                    record_ok = (
                        source_path.is_absolute()
                        and source_path.is_file()
                        and source_path.suffix.lower() == ".mat"
                        and int(record.get("size_bytes", -1)) == int(stat.st_size)
                        and int(record.get("modification_time_ns", -1)) == int(stat.st_mtime_ns)
                        and record.get("sha256") == current_sha256
                    )
                    if record_ok:
                        resolved_dataset_paths.append(str(source_path.resolve()).lower())
                        gate["sources"]["uo_source_mat_{0:02d}".format(index + 1)] = {
                            "path": str(source_path.resolve()),
                            "sha256": current_sha256,
                        }
                except (OSError, TypeError, ValueError):
                    record_ok = False
            dataset_files_ok = dataset_files_ok and record_ok
    dataset_files_ok = dataset_files_ok and len(set(resolved_dataset_paths)) == 14
    recomputed_dataset_signature = ""
    if isinstance(dataset_files, list) and isinstance(dataset_settings, Mapping):
        recomputed_dataset_signature = _canonical_json_sha256(
            {"files": dataset_files, "settings": dict(dataset_settings)}
        )
    dataset_provenance_ok = dataset_files_ok and dataset_signature == recomputed_dataset_signature
    _check(gate, "dataset_provenance", dataset_provenance_ok, "14 current UO MAT files with matching size/mtime/SHA-256 and recomputable content signature", {"source_file_count": dataset_provenance.get("source_file_count"), "content_signature": recomputed_dataset_signature}, "05 dataset provenance is incomplete, stale, or not reproducible from the current 14 source MAT files and settings.")
    raw_data_signatures = (
        sorted(set(raw["data_signature"].astype(str)))
        if raw_ready and "data_signature" in raw.columns
        else []
    )
    raw_dataset_signature_ok = (
        dataset_provenance_ok
        and len(raw_data_signatures) == 1
        and raw_data_signatures[0] == dataset_signature
    )
    _check(gate, "raw_dataset_content_signature", raw_dataset_signature_ok, [dataset_signature] if dataset_signature else [], raw_data_signatures, "Every 05 raw row must bind to the one verified UO dataset content signature.")
    run_artifacts_ok, run_artifact_details = _validate_lowshot_run_artifacts(
        manifest,
        raw,
        base,
        str(dataset_signature or ""),
    )
    _check(gate, "run_artifacts", run_artifacts_ok, "140 unique identities and 420 in-scope, hash-bound metrics/weights/80-row histories", run_artifact_details, "05 run-level evidence is missing, outside the run directory, hash-inconsistent, identity-inconsistent, or incomplete.")

    for index, path in enumerate(figure_paths):
        if path.is_file():
            gate["sources"]["figure_{0}".format(index)] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return _finish_gate(gate)


def _expected_live_manuscript_candidate(summary: pd.DataFrame, reference: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    tables = reference.get("tables", {}) if isinstance(reference.get("tables"), dict) else {}
    for table_name, (case, load, source_table) in DEEP_REFERENCE_TABLES.items():
        table = tables.get(table_name, {}) if isinstance(tables.get(table_name), dict) else {}
        for source_row in table.get("rows", []):
            row: Dict[str, Any] = {
                "source_type": "main_manuscript_deep_reference",
                "reference_type": "main_manuscript_aggregated_reference",
                "source_table": source_table,
                "case": case,
                "load": load,
                "n_train": int(source_row["n_train"]),
                "model": str(source_row["model"]),
                "runs": 10,
                "retained_after_trim": 8,
                "aggregation": TRIMMED_AGGREGATION,
                "benchmark_scope": "main-manuscript deep-model aggregated references; no deep-model training or per-seed pairing",
                "comparison_type": "descriptive_aggregated_reference_only",
            }
            for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
                row[metric + "_mean"] = source_row[metric + "_mean"]
                row[metric + "_sd"] = source_row[metric + "_sd"]
            rows.append(row)
    clean = summary[summary["case"].isin(("UO", "KAIST"))].copy()
    for _, source_row in clean.iterrows():
        row = {
            "source_type": "new_traditional_baseline",
            "reference_type": "new_per_seed_traditional_baseline",
            "source_table": "reviewer_suite_06_clean",
            "case": str(source_row["case"]),
            "load": str(source_row["load"]),
            "n_train": int(source_row["n_train"]),
            "model": str(source_row["method"]),
            "runs": int(source_row["seeds_total"]),
            "retained_after_trim": int(source_row["retained_after_trim"]),
            "aggregation": str(source_row["aggregation"]),
            "benchmark_scope": "interpretable time/frequency-feature SVM reference baselines",
            "comparison_type": "descriptive_cross_method_benchmark_only",
        }
        for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
            row[metric + "_mean"] = source_row[metric + "_mean"]
            row[metric + "_sd"] = source_row[metric + "_sd"]
        rows.append(row)
    return pd.DataFrame(rows, columns=TRADITIONAL_CANDIDATE_COLUMNS)


def _live_candidate_mismatches(
    candidate: pd.DataFrame,
    summary: pd.DataFrame,
    reference: Mapping[str, Any],
) -> List[str]:
    try:
        expected = _expected_live_manuscript_candidate(summary, reference)
    except (KeyError, TypeError, ValueError) as exc:
        return ["cannot construct live expected candidate: {0}".format(exc)]
    identity = ["source_type", "source_table", "case", "load", "n_train", "model"]
    if len(expected) != 144 or len(candidate) != 144:
        return ["candidate row count does not equal live 144-row contract"]
    if expected.duplicated(identity, keep=False).any() or candidate.duplicated(identity, keep=False).any():
        return ["candidate/reference identity is duplicated"]
    expected_rows = {tuple(row[column] for column in identity): row for _, row in expected.iterrows()}
    actual_rows = {tuple(row[column] for column in identity): row for _, row in candidate.iterrows()}
    if set(expected_rows) != set(actual_rows):
        return ["candidate identities differ from current reference plus frozen clean summary"]
    numeric_fields = {
        "n_train", "runs", "retained_after_trim",
        "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
        "macro_precision_mean", "macro_precision_sd", "macro_recall_mean", "macro_recall_sd",
    }
    current_latex_fields = {
        "accuracy_mean", "accuracy_sd", "macro_precision_mean", "macro_precision_sd",
        "macro_f1_mean", "macro_f1_sd",
    }
    mismatches: List[str] = []
    for key in sorted(expected_rows, key=lambda value: tuple(str(item) for item in value)):
        expected_row = expected_rows[key]
        actual_row = actual_rows[key]
        for field in TRADITIONAL_CANDIDATE_COLUMNS:
            if field in numeric_fields:
                if key[0] == "main_manuscript_deep_reference" and field in current_latex_fields:
                    matches = _decimal_matches(actual_row[field], str(expected_row[field]))
                elif field in {"n_train", "runs", "retained_after_trim"}:
                    matches = int(actual_row[field]) == int(expected_row[field])
                else:
                    matches = bool(
                        np.isclose(
                            float(actual_row[field]),
                            float(expected_row[field]),
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                    )
            else:
                matches = str(actual_row[field]) == str(expected_row[field])
            if not matches:
                mismatches.append("{0} field {1}".format(key, field))
                if len(mismatches) >= 20:
                    return mismatches
    return mismatches


def gate_traditional(
    output_root: Path,
    reference_path: Path = DEEP_REFERENCE_PATH,
) -> Dict[str, Any]:
    gate = _new_gate("06_traditional_baselines")
    base = Path(output_root) / "06_traditional_baselines"
    raw_path = base / "traditional_baselines_raw.csv"
    summary_path = base / "traditional_baselines_summary.csv"
    candidate_path = base / "manuscript_candidate_rows.csv"
    scope_path = base / "benchmark_scope.txt"
    manifest_path = base / "traditional_baselines_run_manifest.json"
    tuning_path = base / "traditional_baseline_tuning_audit.json"
    raw = _load_csv(gate, "raw", raw_path)
    summary = _load_csv(gate, "summary", summary_path)
    candidate = _load_csv(gate, "manuscript_candidate", candidate_path)
    manifest = _load_manifest(gate, "run_manifest", manifest_path)
    tuning = _load_manifest(gate, "tuning_audit", tuning_path)

    raw_columns = list(TRADITIONAL_GROUP_COLUMNS) + ["seed", "accuracy", "macro_f1", "macro_precision", "macro_recall"]
    raw_ready = _has_columns(gate, raw, raw_columns, "raw")
    if raw_ready:
        duplicate_rows = int(raw.duplicated(list(TRADITIONAL_GROUP_COLUMNS) + ["seed"], keep=False).sum())
        observed_seeds = sorted(int(value) for value in pd.to_numeric(raw["seed"]).unique())
        groups = raw.groupby(list(TRADITIONAL_GROUP_COLUMNS), dropna=False)["seed"].agg(lambda values: {int(v) for v in values})
        components = {name: int((raw["case"] == name).sum()) for name in ("UO", "KAIST", "KAIST-noise")}
        non_uo_snr = pd.to_numeric(raw.loc[raw["case"] != "UO", "snr_db"], errors="coerce")
        snr_ok = raw.loc[raw["case"] == "UO", "snr_db"].isna().all() and not non_uo_snr.isna().any() and bool(np.isfinite(non_uo_snr.to_numpy(dtype=np.float64)).all())
        _check(gate, "raw_rows", len(raw) == 480, 480, int(len(raw)), "06 raw must contain exactly 480 rows.")
        _check(gate, "component_rows", components == {"UO": 120, "KAIST": 240, "KAIST-noise": 120}, {"UO": 120, "KAIST": 240, "KAIST-noise": 120}, components, "06 component row counts are incomplete.")
        _check(gate, "groups", len(groups) == 48, 48, int(len(groups)), "06 must contain exactly 48 result groups.")
        _check(gate, "target_seeds", observed_seeds == list(TARGET_SEEDS), list(TARGET_SEEDS), observed_seeds, "06 must contain the fixed ten seeds.")
        _check(gate, "group_seed_sets", all(value == set(TARGET_SEEDS) for value in groups), "identical 10 seeds per group", False, "Every 06 group must contain all ten seeds.")
        _check(gate, "traditional_methods", set(raw["method"]) == set(TRADITIONAL_METHODS), list(TRADITIONAL_METHODS), sorted(raw["method"].unique()), "06 raw must contain only the two declared TF-SVM methods.")
        _check(gate, "duplicates", duplicate_rows == 0, 0, duplicate_rows, "06 raw contains duplicate rows.")
        _check(gate, "finite_metrics", _finite(raw, ["seed", "n_train", "accuracy", "macro_f1", "macro_precision", "macro_recall"]), True, False, "06 raw contains NaN, Inf, or non-numeric metrics.")
        _check(gate, "snr_encoding", snr_ok, "UO structural NA only; finite KAIST SNR", False, "06 contains invalid SNR values.")

    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary_columns = list(TRADITIONAL_GROUP_COLUMNS) + ["seeds_total", "retained_after_trim", "aggregation"] + [
        metric + suffix
        for metric in metric_names
        for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")
    ]
    summary_ready = _has_columns(gate, summary, summary_columns, "summary")
    if summary_ready:
        _check(gate, "summary_groups", len(summary) == 48 and not summary.duplicated(list(TRADITIONAL_GROUP_COLUMNS)).any(), 48, int(len(summary)), "06 summary must contain exactly 48 unique groups.")
        _check(gate, "summary_seed_count", bool((summary["seeds_total"] == 10).all()), 10, sorted(summary["seeds_total"].unique().tolist()), "06 summary must report ten seeds.")
        _check(gate, "retained_after_trim", bool((summary["retained_after_trim"] == 8).all()), 8, sorted(summary["retained_after_trim"].unique().tolist()), "06 summary must retain eight values.")
        _check(gate, "aggregation", bool((summary["aggregation"] == TRIMMED_AGGREGATION).all()), TRIMMED_AGGREGATION, sorted(summary["aggregation"].astype(str).unique().tolist()), "06 aggregation protocol is not manuscript-compatible.")
        numeric_summary = [metric + suffix for metric in metric_names for suffix in ("_mean", "_sd", "_untrimmed_mean", "_untrimmed_sd")]
        _check(gate, "summary_finite", _finite(summary, numeric_summary), True, False, "06 summary contains NaN or Inf.")

    summary_matches_raw = raw_ready and summary_ready and len(raw) == 480 and len(summary) == 48
    if summary_matches_raw:
        for keys, block in raw.groupby(list(TRADITIONAL_GROUP_COLUMNS), dropna=False, sort=False):
            mask = pd.Series(True, index=summary.index)
            for column, value in zip(TRADITIONAL_GROUP_COLUMNS, keys):
                mask &= summary[column].isna() if pd.isna(value) else summary[column].eq(value)
            rows = summary[mask]
            if len(rows) != 1:
                summary_matches_raw = False
                break
            actual = rows.iloc[0]
            for metric in metric_names:
                mean, sd, untrimmed_mean, untrimmed_sd = _trimmed_statistics(block[metric].to_numpy(dtype=np.float64))
                if not all((
                    _close(actual[metric + "_mean"], mean),
                    _close(actual[metric + "_sd"], sd),
                    _close(actual[metric + "_untrimmed_mean"], untrimmed_mean),
                    _close(actual[metric + "_untrimmed_sd"], untrimmed_sd),
                )):
                    summary_matches_raw = False
                    break
            if not summary_matches_raw:
                break
    _check(gate, "summary_recomputed_independently", summary_matches_raw, True, summary_matches_raw, "06 summary does not reproduce independent metric-wise trimming of the raw ten-seed results.")

    candidate_ready = _has_columns(gate, candidate, TRADITIONAL_CANDIDATE_COLUMNS, "manuscript_candidate")
    if candidate_ready:
        candidate_duplicates = int(candidate.duplicated(["source_type", "case", "load", "n_train", "model"], keep=False).sum())
        source_counts = candidate["source_type"].value_counts().to_dict()
        clean_scope = (
            set(candidate["case"]) == {"UO", "KAIST"}
            and set(candidate.loc[candidate["case"] == "UO", "load"]) == {"held-out"}
            and set(candidate.loc[candidate["case"] == "KAIST", "load"]) == {"2Nm", "4Nm"}
        )
        deep = candidate[candidate["source_type"] == "main_manuscript_deep_reference"]
        traditional = candidate[candidate["source_type"] == "new_traditional_baseline"]
        deep_identity_ok = (
            len(deep) == 108
            and set(deep["model"]) == set(DEEP_REFERENCE_MODELS)
            and set(pd.to_numeric(deep["n_train"]).astype(int)) == set(DEEP_REFERENCE_N_GRID)
            and bool((deep["comparison_type"] == "descriptive_aggregated_reference_only").all())
        )
        traditional_identity_ok = (
            len(traditional) == 36
            and set(traditional["model"]) == set(TRADITIONAL_METHODS)
            and set(pd.to_numeric(traditional["n_train"]).astype(int)) == set(DEEP_REFERENCE_N_GRID)
            and bool((traditional["comparison_type"] == "descriptive_cross_method_benchmark_only").all())
        )
        metric_columns = [metric + suffix for metric in metric_names for suffix in ("_mean", "_sd")]
        _check(gate, "candidate_rows", len(candidate) == 144, 144, int(len(candidate)), "06 hybrid manuscript candidate must contain 108 deep-reference and 36 SVM rows.")
        _check(gate, "candidate_source_counts", source_counts == {"main_manuscript_deep_reference": 108, "new_traditional_baseline": 36}, {"main_manuscript_deep_reference": 108, "new_traditional_baseline": 36}, source_counts, "06 candidate source counts are invalid.")
        _check(gate, "candidate_clean_scope", clean_scope, "UO clean + KAIST 2Nm/4Nm clean only", sorted(set(candidate["case"])), "06 candidate improperly includes KAIST-noise or an unknown load.")
        _check(gate, "candidate_deep_identities", deep_identity_ok, "6 models x 6 N x 3 manuscript tables", int(len(deep)), "06 deep-reference candidate rows are incomplete or mislabeled.")
        _check(gate, "candidate_svm_identities", traditional_identity_ok, "2 SVM methods x 6 N x UO/2Nm/4Nm", int(len(traditional)), "06 new SVM candidate rows are incomplete or mislabeled.")
        _check(gate, "candidate_protocol", bool((candidate["runs"] == 10).all()) and bool((candidate["retained_after_trim"] == 8).all()) and bool((candidate["aggregation"] == TRIMMED_AGGREGATION).all()), "10 runs, 8 retained, trimmed aggregation", None, "06 candidate uses an incompatible run count or aggregation.")
        _check(gate, "candidate_duplicates", candidate_duplicates == 0, 0, candidate_duplicates, "06 hybrid candidate contains duplicate identities.")
        _check(gate, "candidate_finite", _finite(candidate, ["n_train", "runs", "retained_after_trim"] + metric_columns), True, False, "06 candidate contains NaN, Inf, or non-numeric metrics.")
        forbidden = [name for name in candidate.columns if name in {"seed", "raw", "paired_delta", "p_value", "significance"} or "delta_to" in name]
        _check(gate, "candidate_descriptive_only", not forbidden, [], forbidden, "06 aggregate deep references cannot support paired deltas, p-values, or significance claims.")

    _check(gate, "manifest_status", manifest.get("status") == "PASS" and manifest.get("mode") == "full", "PASS/full", "{0}/{1}".format(manifest.get("status"), manifest.get("mode")), "06 run manifest is not a successful full run.")
    _check(gate, "manifest_aggregation", manifest.get("aggregation") == TRIMMED_AGGREGATION and manifest.get("metrics_trimmed_independently") is True, "independent trimmed aggregation", manifest.get("aggregation"), "06 manifest does not confirm independent metric trimming.")
    failed = manifest.get("failed_seeds", None)
    _check(gate, "failed_seeds", failed == [], [], failed, "06 manifest reports failed or unknown seeds.")
    failed_groups = manifest.get("failed_groups", None)
    _check(gate, "failed_groups", failed_groups == [], [], failed_groups, "06 manifest reports failed or unknown result groups.")
    channel = manifest.get("channel_gate", {}) if isinstance(manifest.get("channel_gate"), dict) else {}
    channel_ok = channel.get("current_channel") == "cDAQ9185-1F486B5Mod2/ai0" and channel.get("vibration_column") == 0 and channel.get("current_fallback") is False
    _check(gate, "kaist_channels", channel_ok, "U-phase exact channel, xA column 0, fallback false", channel, "06 full run does not prove the confirmed KAIST channels.")
    scope_exists = scope_path.is_file()
    scope_text = scope_path.read_text(encoding="utf-8") if scope_exists else ""
    scope_lower = scope_text.lower()
    scope_ok = (
        scope_exists
        and "time/frequency-feature svm reference baselines" in scope_lower
        and ("not exact" in scope_lower or "do not reproduce" in scope_lower)
        and "frozen" in scope_lower
    )
    _check(gate, "benchmark_scope", scope_ok, "handcrafted TF-SVM reference; frozen tuning; not exact CSC/GJO-OMP reproduction", scope_text.strip(), "06 benchmark scope is missing, overstated, or omits the frozen protocol.")
    _check(gate, "no_deep_model_training", manifest.get("deep_model_training_performed") is False, False, manifest.get("deep_model_training_performed"), "06 does not prove that deep models were not trained.")

    evaluation_seeds = [int(value) for value in tuning.get("evaluation_seeds", [])] if isinstance(tuning.get("evaluation_seeds"), list) else []
    tuning_seed = tuning.get("tuning_seed")
    tuning_n = tuning.get("tuning_n")
    tuning_protocol_ok = (
        tuning_seed == config.GLOBAL_SEED - 1
        and tuning_seed == 20260805
        and tuning_seed not in set(TARGET_SEEDS)
        and tuning_n == 15
        and evaluation_seeds == list(TARGET_SEEDS)
        and tuning.get("test_or_target_data_used_for_selection") is False
        and tuning.get("frozen_across_all_reported_N_and_seeds") is True
        and "regardless of whether" in str(tuning.get("result_acceptance_rule", ""))
    )
    _check(gate, "frozen_tuning_protocol", tuning_protocol_ok, {"tuning_seed": 20260805, "tuning_n": 15, "evaluation_seeds": list(TARGET_SEEDS), "frozen": True, "test_or_target_used": False}, {"tuning_seed": tuning_seed, "tuning_n": tuning_n, "evaluation_seeds": evaluation_seeds}, "06 tuning audit does not implement the predeclared frozen fairness protocol.")

    cases = tuning.get("cases", {}) if isinstance(tuning.get("cases"), Mapping) else {}
    model_names = {"early", "mod1", "mod2"}
    selected_parameters_ok = set(cases) == {"UO", "KAIST"}
    selected_parameters: Dict[str, Any] = {}
    for case_name in ("UO", "KAIST"):
        case_payload = cases.get(case_name, {}) if isinstance(cases.get(case_name), Mapping) else {}
        models = case_payload.get("models", {}) if isinstance(case_payload.get("models"), Mapping) else {}
        selected_parameters_ok = selected_parameters_ok and set(models) == model_names
        selected_parameters[case_name] = {}
        for model_name in sorted(model_names):
            model_payload = models.get(model_name, {}) if isinstance(models.get(model_name), Mapping) else {}
            selected = model_payload.get("selected", {}) if isinstance(model_payload.get("selected"), Mapping) else {}
            try:
                c_value = float(selected.get("C"))
            except (TypeError, ValueError):
                c_value = float("nan")
            gamma = str(selected.get("gamma", ""))
            parameter_ok = np.isfinite(c_value) and c_value > 0.0 and gamma in {"scale", "auto"}
            selected_parameters_ok = selected_parameters_ok and bool(parameter_ok)
            selected_parameters[case_name][model_name] = {"C": c_value, "gamma": gamma}
    _check(gate, "frozen_tuning_parameters", selected_parameters_ok, "valid C/gamma for UO and KAIST early/mod1/mod2", selected_parameters, "06 tuning audit is missing a valid frozen SVM parameter profile.")

    uo_case = cases.get("UO", {}) if isinstance(cases.get("UO"), Mapping) else {}
    tuning_split = uo_case.get("split_audit", {}) if isinstance(uo_case.get("split_audit"), Mapping) else {}
    tuning_overlap_ok = tuning_split.get("exact_sample_overlap") == 0
    _check(gate, "uo_tuning_exact_sample_overlap", tuning_overlap_ok, 0, tuning_split.get("exact_sample_overlap"), "UO tuning split has exact sample overlap or lacks an audit.")

    evaluation_audits = tuning.get("uo_evaluation_split_audits", [])
    evaluation_audits_ready = isinstance(evaluation_audits, list) and len(evaluation_audits) == 60
    expected_audit_ids = {(seed, n_train) for seed in TARGET_SEEDS for n_train in DEEP_REFERENCE_N_GRID}
    actual_audit_ids = set()
    evaluation_overlap_ok = evaluation_audits_ready
    shared_claim_ok = True
    shared_recording_count = 0
    if isinstance(evaluation_audits, list):
        for item in evaluation_audits:
            if not isinstance(item, Mapping):
                evaluation_overlap_ok = False
                shared_claim_ok = False
                continue
            try:
                actual_audit_ids.add((int(item.get("seed")), int(item.get("n_train"))))
            except (TypeError, ValueError):
                evaluation_overlap_ok = False
            evaluation_overlap_ok = evaluation_overlap_ok and item.get("exact_sample_overlap") == 0
            if item.get("shared_recordings"):
                shared_recording_count += 1
                claim = str(item.get("claim_limit", "")).lower()
                shared_claim_ok = shared_claim_ok and (
                    "segment-disjoint" in claim
                    and "does not establish recording-disjoint generalization" in claim
                )
    evaluation_overlap_ok = evaluation_overlap_ok and actual_audit_ids == expected_audit_ids
    _check(gate, "uo_evaluation_split_audits", evaluation_overlap_ok, "60 unique seed/N audits with zero exact sample overlap", {"rows": len(evaluation_audits) if isinstance(evaluation_audits, list) else None, "identities": len(actual_audit_ids)}, "UO evaluation split audits are incomplete, duplicated, or contain exact overlap.")

    tuning_shared = tuning_split.get("shared_recordings", [])
    if tuning_shared:
        tuning_claim = str(tuning_split.get("claim_limit", "")).lower()
        shared_claim_ok = shared_claim_ok and (
            "segment-disjoint" in tuning_claim
            and "does not establish recording-disjoint generalization" in tuning_claim
        )
    _check(gate, "uo_shared_recording_claim_boundary", shared_claim_ok, "shared recordings described only as segment-disjoint, not recording-disjoint", {"evaluation_audits_with_shared_recordings": shared_recording_count, "tuning_shared_recordings": len(tuning_shared) if isinstance(tuning_shared, list) else None}, "UO shared-recording evidence is missing the required segment-disjoint claim boundary.")

    manifest_protocol = manifest.get("svm_hyperparameter_protocol", {}) if isinstance(manifest.get("svm_hyperparameter_protocol"), Mapping) else {}
    manifest_protocol_ok = (
        "single predeclared development tuning" in str(manifest_protocol.get("selection", ""))
        and manifest_protocol.get("tuning_n") == tuning_n == 15
        and manifest_protocol.get("tuning_seed") == tuning_seed == 20260805
        and manifest_protocol.get("evaluation_seeds") == list(TARGET_SEEDS)
        and manifest_protocol.get("frozen_across_all_reported_points") is True
        and manifest_protocol.get("test_or_target_data_used_for_selection") is False
    )
    _check(gate, "manifest_frozen_tuning_protocol", manifest_protocol_ok, "manifest mirrors the complete frozen tuning audit", manifest_protocol, "06 manifest does not mirror the frozen tuning protocol.")
    _check(gate, "tuning_audit_hash", _manifest_output_matches(manifest, "tuning_audit", tuning_path), True, False, "06 tuning-audit path/hash does not match its manifest.")
    candidate_scope = manifest.get("candidate_scope", {}) if isinstance(manifest.get("candidate_scope"), dict) else {}
    candidate_scope_ok = (
        candidate_scope.get("included_cases") == ["UO", "KAIST"]
        and candidate_scope.get("excluded_cases") == ["KAIST-noise"]
        and candidate_scope.get("expected_rows") == 144
        and candidate_scope.get("deep_reference_rows") == 108
        and candidate_scope.get("traditional_rows") == 36
        and candidate_scope.get("n_values") == list(DEEP_REFERENCE_N_GRID)
    )
    _check(gate, "manifest_candidate_scope", candidate_scope_ok, "clean-only 144-row hybrid candidate", candidate_scope, "06 manifest candidate scope is incomplete or includes noise.")
    post_gate = manifest.get("post_gate", {}) if isinstance(manifest.get("post_gate"), dict) else {}
    post_gate_ok = (
        post_gate.get("status") == "PASS"
        and post_gate.get("raw_rows") == 480
        and post_gate.get("summary_rows") == 48
        and post_gate.get("candidate_rows") == 144
        and post_gate.get("failed_groups") == []
        and post_gate.get("deep_model_training_performed") is False
    )
    _check(gate, "manifest_post_gate", post_gate_ok, "PASS with 480/48/144 rows and no failures/deep training", post_gate.get("status"), "06 internal post-gate did not pass its complete result contract.")
    manifest_reference_gate = manifest.get("deep_reference_gate", {}) if isinstance(manifest.get("deep_reference_gate"), dict) else {}
    _check(gate, "manifest_deep_reference_gate", manifest_reference_gate.get("status") == "PASS" and manifest_reference_gate.get("rows") == 108, "historical PASS/108 rows (superseded by live source gate for schema 2)", manifest_reference_gate, "06 manifest does not contain a successful historical deep-reference gate.")
    _check(gate, "raw_hash", _manifest_hash_matches(manifest, "raw", raw_path), True, False, "06 raw hash does not match its manifest.")
    _check(gate, "summary_hash", _manifest_hash_matches(manifest, "summary", summary_path), True, False, "06 summary hash does not match its manifest.")
    reference_gate = gate_deep_reference(Path(reference_path))
    gate["deep_reference_gate"] = reference_gate
    for item in reference_gate["checks"]:
        _check(
            gate,
            "deep_" + str(item["name"]),
            bool(item["passed"]),
            item.get("expected"),
            item.get("actual"),
            item.get("reason", "Deep-reference gate failed."),
        )
    for source_name, record in reference_gate.get("sources", {}).items():
        if isinstance(record, Mapping):
            gate["sources"]["deep_" + str(source_name)] = dict(record)
    if Path(reference_path).is_file():
        gate["sources"]["deep_reference"] = {
            "path": str(Path(reference_path).resolve()),
            "sha256": sha256_file(reference_path),
        }
    try:
        reference_payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        reference_payload = {}
    live_reference = reference_payload.get("schema_version") == 2
    if live_reference:
        live_mismatches = _live_candidate_mismatches(candidate, summary, reference_payload)
        live_candidate_ok = reference_gate.get("usable") is True and not live_mismatches
        _check(
            gate,
            "candidate_live_source_consistency",
            live_candidate_ok,
            "exact 108 current-LaTeX rows + exact 36 frozen-summary clean TF-SVM rows",
            {"mismatches": live_mismatches, "candidate_sha256": sha256_file(candidate_path) if candidate_path.is_file() else None},
            "06 candidate is not an exact live rebuild from the current manuscript reference and frozen summary.",
        )
        _check(
            gate,
            "candidate_hash",
            live_candidate_ok,
            "live derivation gate supersedes stale pre-refresh manifest candidate hash",
            manifest.get("manuscript_candidate", {}).get("sha256") if isinstance(manifest.get("manuscript_candidate"), dict) else None,
            "Live candidate derivation gate failed.",
        )
        _check(
            gate,
            "deep_reference_hash",
            reference_gate.get("usable") is True,
            "live current-LaTeX source/hash/value gate supersedes stale pre-refresh manifest reference hash",
            sha256_file(reference_path) if Path(reference_path).is_file() else None,
            "Live current-manuscript reference gate failed.",
        )
    else:
        _check(gate, "candidate_hash", _manifest_hash_matches(manifest, "manuscript_candidate", candidate_path), True, False, "06 candidate hash does not match its manifest.")
        _check(gate, "deep_reference_hash", _manifest_hash_matches(manifest, "deep_reference", Path(reference_path)), True, False, "06 manifest deep-reference hash does not match the configured JSON.")
    return _finish_gate(gate)


def _remove_stale_asset(out_dir: Path, stem: str) -> None:
    for suffix in (".csv", ".tex", ".txt", ".json", ".md", ".png", ".pdf"):
        path = Path(out_dir) / (stem + suffix)
        if path.is_file():
            path.unlink()


def _source_data_index(gates: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for gate_name, gate in gates.items():
        if not isinstance(gate, Mapping):
            continue
        sources = gate.get("sources", {})
        if not isinstance(sources, Mapping):
            continue
        for source_name, record in sources.items():
            if isinstance(record, Mapping):
                rows.append(
                    {
                        "gate": str(gate_name),
                        "source_name": str(source_name),
                        "absolute_path": str(record.get("path", "")),
                        "sha256": str(record.get("sha256", "")),
                    }
                )
    return pd.DataFrame(rows, columns=["gate", "source_name", "absolute_path", "sha256"])


def main() -> None:
    out_dir = config.OUTPUT_ROOT / "07_manuscript_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (
        "efficiency_table",
        "additional_ablation_table",
        "additional_ablation_table_note",
        "lowshot_sensitivity_table",
        "lowshot_paired_caim_table",
        "lowshot_sensitivity_figure",
        "lowshot_sensitivity_figure_contract",
        "traditional_baseline_table",
        "traditional_baseline_uo_table",
        "traditional_baseline_kaist_2nm_table",
        "traditional_baseline_kaist_4nm_table",
        "traditional_baseline_table_note",
        "source_data_index",
        "response_skeletons",
    ):
        _remove_stale_asset(out_dir, stem)

    response_sections: List[str] = [
        "# Reviewer-experiment response skeletons",
        "",
        "Numerical claims are omitted unless the corresponding source table passes its data-level manuscript gate.",
        "Insert page/line numbers only after final layout stabilization.",
        "",
    ]
    gates: Dict[str, Dict[str, Any]] = {}

    gates["efficiency"] = gate_efficiency(config.OUTPUT_ROOT)

    weights = config.OUTPUT_ROOT / "02_weights_missing" / "modality_weight_summary.csv"
    gates["weights"] = full_run_gate(weights)

    missing = config.OUTPUT_ROOT / "02_weights_missing" / "missing_modality_summary.csv"
    gates["missing"] = full_run_gate(missing)

    gates["ablation"] = gate_additional_ablation(config.OUTPUT_ROOT)
    gates["ablation_main_display"] = gate_ablation_main_display(config.OUTPUT_ROOT)
    gates["lowshot"] = gate_lowshot(config.OUTPUT_ROOT)
    gates["traditional"] = gate_traditional(config.OUTPUT_ROOT)
    mandatory = ("efficiency", "ablation", "ablation_main_display", "lowshot", "traditional")
    failed = [name for name in mandatory if not gates[name]["usable"]]
    gates["overall"] = {
        "usable": not failed,
        "mandatory_gates": list(mandatory),
        "failed_gates": failed,
        "status": "PASS" if not failed else "FAIL",
    }
    write_json(out_dir / "asset_gates.json", gates)
    if failed:
        raise RuntimeError("Manuscript asset gates failed: {0}. See asset_gates.json.".format(", ".join(failed)))

    try:
        efficiency_path = config.OUTPUT_ROOT / "03_efficiency" / "efficiency_profile.json"
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        efficiency_table = pd.DataFrame(
            [
                {
                    "Params": int(efficiency["trainable_params"]),
                    "Params (M)": float(efficiency["trainable_params_m"]),
                    "FLOPs (G)": float(efficiency["flops_g"]),
                    "Estimated MACs (G)": float(efficiency["macs_g_estimated"]),
                    "Mean latency (ms)": float(efficiency["latency_mean_ms"]),
                    "Median latency (ms)": float(efficiency["latency_median_ms"]),
                    "P95 latency (ms)": float(efficiency["latency_p95_ms"]),
                    "Throughput (samples/s)": float(efficiency["throughput_samples_per_s"]),
                    "Current TF GPU memory (MB)": float(efficiency["gpu_allocator_current_mb"]),
                    "Peak TF GPU memory (MB)": float(efficiency["gpu_allocator_peak_mb"]),
                    "Weights size (MB)": float(efficiency["weights_size_mb"]),
                    "SavedModel size (MB)": float(efficiency["savedmodel_size_mb"]),
                }
            ]
        )
        efficiency_table.to_csv(out_dir / "efficiency_table.csv", index=False)
        write_latex(efficiency_table, out_dir / "efficiency_table.tex")
        response_sections += ["## R3-5 / R4-5", "Deployment-oriented efficiency measurements were added from the isolated CPU-FLOPs/GPU-runtime profile.", ""]

        if gates["weights"]["usable"]:
            frame = pd.read_csv(weights)
            columns = ["scenario", "snr_db", "accuracy_mean", "weight_vibration_mean", "weight_current_mean"]
            require_columns(frame, ["load"] + columns, weights)
            selected = frame[
                (frame["load"] == "4Nm")
                & frame["scenario"].isin(["both_modalities_degraded", "vibration_only_degraded", "current_only_degraded"])
                & frame["snr_db"].isin([0, -4, -8, -10])
            ][columns].copy()
            selected.to_csv(out_dir / "modality_weight_table.csv", index=False)
            write_latex(selected, out_dir / "modality_weight_table.tex")

        if gates["missing"]["usable"]:
            frame = pd.read_csv(missing)
            columns = ["load", "condition", "accuracy_mean", "macro_f1_mean", "weight_vibration_mean", "weight_current_mean"]
            require_columns(frame, columns, missing)
            selected = frame[columns].copy()
            selected.to_csv(out_dir / "missing_modality_table.csv", index=False)
            write_latex(selected, out_dir / "missing_modality_table.tex")

        source = config.OUTPUT_ROOT / "04_additional_ablation" / ABLATION_MAIN_SUMMARY_NAME
        frame = pd.read_csv(source, keep_default_na=False)
        reference = config.load_kaist_additional_ablation_full_reference()
        table = build_hybrid_ablation_table(frame, reference)
        table.to_csv(out_dir / "additional_ablation_table.csv", index=False)
        write_latex(table, out_dir / "additional_ablation_table.tex")
        (out_dir / "additional_ablation_table_note.txt").write_text(
            ABLATION_MAIN_NOTE + "\n",
            encoding="utf-8",
        )
        response_sections += ["## R4-6", ABLATION_MAIN_NOTE, ""]

        source = config.OUTPUT_ROOT / "05_lowshot_threshold" / "lowshot_summary.csv"
        frame = pd.read_csv(source)
        columns = [
            "variant", "n_train", "test_accuracy_mean", "test_accuracy_sd",
            "generalization_gap_mean", "generalization_gap_sd", "seeds_total",
            "retained_after_trim", "aggregation",
        ]
        selected = frame[columns].copy()
        selected.to_csv(out_dir / "lowshot_sensitivity_table.csv", index=False)
        write_latex(selected, out_dir / "lowshot_sensitivity_table.tex")
        paired = pd.read_csv(config.OUTPUT_ROOT / "05_lowshot_threshold" / "caim_paired_summary.csv")
        paired.to_csv(out_dir / "lowshot_paired_caim_table.csv", index=False)
        write_latex(paired, out_dir / "lowshot_paired_caim_table.tex")
        lowshot_figures = _lowshot_figure_paths(config.OUTPUT_ROOT / "05_lowshot_threshold")
        shutil.copyfile(lowshot_figures["png"], out_dir / "lowshot_sensitivity_figure.png")
        shutil.copyfile(lowshot_figures["pdf"], out_dir / "lowshot_sensitivity_figure.pdf")
        shutil.copyfile(lowshot_figures["svg"], out_dir / "lowshot_sensitivity_figure.svg")
        shutil.copyfile(lowshot_figures["tiff"], out_dir / "lowshot_sensitivity_figure.tiff")
        shutil.copyfile(lowshot_figures["contract"], out_dir / "lowshot_sensitivity_figure_contract.json")
        response_sections += [
            "## R3-2",
            "The reported point is the smallest evaluated N satisfying the predefined stability criterion, not a universal theoretical threshold.",
            "The train–held-out gap uses the same all-remaining held-out set for the training curve and final evaluation; no independent validation/test partition is claimed.",
            "",
        ]

        source = config.OUTPUT_ROOT / "06_traditional_baselines" / "manuscript_candidate_rows.csv"
        frame = pd.read_csv(source)
        selected = frame[list(TRADITIONAL_CANDIDATE_COLUMNS)].copy()
        selected.to_csv(out_dir / "traditional_baseline_table.csv", index=False)
        write_latex(selected, out_dir / "traditional_baseline_table.tex")
        for case, load, stem in (
            ("UO", "held-out", "traditional_baseline_uo_table"),
            ("KAIST", "2Nm", "traditional_baseline_kaist_2nm_table"),
            ("KAIST", "4Nm", "traditional_baseline_kaist_4nm_table"),
        ):
            table = selected[(selected["case"] == case) & (selected["load"] == load)].copy()
            table.to_csv(out_dir / (stem + ".csv"), index=False)
            write_latex(table, out_dir / (stem + ".tex"))
        traditional_note = (
            "The deep-model rows reuse aggregated 10-run results from the main manuscript and are descriptive only. "
            "The two newly evaluated methods are interpretable time/frequency-feature SVM references. No deep model "
            "was trained by Experiment 06, and no paired or significance claim is made across aggregate references."
        )
        (out_dir / "traditional_baseline_table_note.txt").write_text(traditional_note + "\n", encoding="utf-8")
        response_sections += [
            "## R1-8 / R3-4",
            traditional_note,
            "",
        ]

        source_index = _source_data_index(gates)
        source_index.to_csv(out_dir / "source_data_index.csv", index=False)
        write_json(out_dir / "source_data_index.json", source_index.to_dict(orient="records"))
        (out_dir / "response_skeletons.md").write_text("\n".join(response_sections) + "\n", encoding="utf-8")
    except BaseException as exc:
        gates["asset_generation"] = {
            "usable": False,
            "status": "FAIL",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }
        gates["overall"]["usable"] = False
        gates["overall"]["status"] = "FAIL"
        gates["overall"]["failed_gates"] = ["asset_generation"]
        write_json(out_dir / "asset_gates.json", gates)
        raise
    gates["asset_generation"] = {"usable": True, "status": "PASS"}
    write_json(out_dir / "asset_gates.json", gates)
    print("Manuscript assets saved to:", out_dir)


if __name__ == "__main__":
    main()
