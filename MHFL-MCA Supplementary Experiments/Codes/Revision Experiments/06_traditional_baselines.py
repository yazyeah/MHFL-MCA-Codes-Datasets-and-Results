"""Experiment 06 - frozen-protocol handcrafted-feature TF-SVM baselines.

Purpose: provide early/late-fusion reference baselines without deep-model training.
Protocol: select C/gamma once at N=15 with seed 20260805, then freeze them across
all reported N values and ten evaluation seeds.
Outputs: 480-row raw data, 48-group summary, tuning audit, and protocol validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.provenance import sha256_file, write_json
from mhfl_review.stats import mean_sd, trimmed_mean_sd


METHOD_NAMES = ("TF-SVM early fusion", "TF-SVM late fusion")
FULL_N_VALUES = (5, 10, 15, 20, 25, 30)
FAST_N_VALUES = (5, 15, 30)
KAIST_LOADS = ("2Nm", "4Nm")
KAIST_NOISE_SNRS = (0.0, -4.0, -8.0)
TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
FAST_AGGREGATION = "mean_sd_untrimmed_fast"
EXACT_CURRENT_CHANNEL = "cDAQ9185-1F486B5Mod2/ai0"
METRIC_COLUMNS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
GROUP_COLUMNS = ("case", "load", "n_train", "snr_db", "method")
RAW_REQUIRED_COLUMNS = GROUP_COLUMNS + ("seed",) + METRIC_COLUMNS

# Fairness correction: SVM hyperparameters are selected once on a dedicated
# development split at N=15 and then frozen for every evaluation N/seed.
# This mirrors the manuscript's Stage-1 -> Stage-2 separation and avoids
# re-tuning the conventional baseline separately at every reported point.
SVM_TUNING_N = 15
SVM_TUNING_SEED_OFFSET = -1
SVM_PARAM_GRID = (
    {"C": 0.1, "gamma": "scale"},
    {"C": 0.1, "gamma": "auto"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 1.0, "gamma": "auto"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "auto"},
    {"C": 100.0, "gamma": "scale"},
    {"C": 100.0, "gamma": "auto"},
)
DEEP_REFERENCE_PATH = config.MAIN_MANUSCRIPT_DEEP_REFERENCE_PATH
DEEP_MODEL_NAMES = ("MRCFN", "CFFN", "CDTFAFN", "MSF-DFormer", "KDCNN-DF", "Full MHFL-MCA")
DEEP_REFERENCE_TABLES = {
    "UO_Table_5": {"source_table": "Table 5", "case": "UO", "load": "held-out", "protocol": "uo_fixed_holdout_fewshot"},
    "KAIST_Table_9": {"source_table": "Table 9", "case": "KAIST", "load": "2Nm", "protocol": "stage2_load_shift"},
    "KAIST_Table_10": {"source_table": "Table 10", "case": "KAIST", "load": "4Nm", "protocol": "stage2_load_shift"},
}
DEEP_REFERENCE_TYPE = "main_manuscript_aggregated_reference"
CURRENT_MANUSCRIPT_EXTRACTION_METHOD = "current_manuscript_table_source"
CURRENT_MANUSCRIPT_METRICS = ("accuracy", "macro_precision", "macro_f1")
CURRENT_MANUSCRIPT_TABLE_LABELS = {
    "UO_Table_5": "tab:case1_compare_main",
    "KAIST_Table_9": "tab:case2_quant_2nm",
    "KAIST_Table_10": "tab:case2_quant_4nm",
}
TRADITIONAL_REFERENCE_TYPE = "new_per_seed_traditional_baseline"
BENCHMARK_SCOPE = "interpretable time/frequency-feature SVM reference baselines"
DEEP_BENCHMARK_SCOPE = "main-manuscript deep-model aggregated references; no deep-model training or per-seed pairing"
FULL_CANDIDATE_COLUMNS = (
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
FULL_DEEP_REFERENCE_ROWS = len(DEEP_REFERENCE_TABLES) * len(FULL_N_VALUES) * len(DEEP_MODEL_NAMES)
FULL_TRADITIONAL_CANDIDATE_ROWS = (
    len(FULL_N_VALUES) * len(METHOD_NAMES)
    + len(FULL_N_VALUES) * len(KAIST_LOADS) * len(METHOD_NAMES)
)
FULL_MANUSCRIPT_CANDIDATE_ROWS = FULL_DEEP_REFERENCE_ROWS + FULL_TRADITIONAL_CANDIDATE_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3-4/R1-8 interpretable non-neural multimodal SVM benchmarks.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--skip-uo", action="store_true")
    parser.add_argument("--skip-kaist", action="store_true")
    parser.add_argument(
        "--tuning-seed",
        type=int,
        default=None,
        help="Dedicated development seed; defaults to GLOBAL_SEED-1 and must not be an evaluation seed in full mode.",
    )
    parser.add_argument(
        "--tuning-n",
        type=int,
        default=SVM_TUNING_N,
        help="Single development sample size used to choose SVM hyperparameters; full mode requires N=15.",
    )
    return parser.parse_args()


def make_svc(seed: int, c_value: float, gamma_value: str) -> Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(c_value),
                    kernel="rbf",
                    gamma=gamma_value,
                    probability=True,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _model_classes(model: object) -> np.ndarray:
    """Return the fitted class order exposed by a Pipeline/SVC."""
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("svm"), "classes_", None)
    if classes is None:
        raise RuntimeError("A fitted SVM does not expose classes_.")
    return np.asarray(classes, dtype=np.int64).reshape(-1)


def _aligned_predict_proba(
    model: object,
    features: np.ndarray,
    class_order: np.ndarray,
) -> np.ndarray:
    """Align probability columns before cross-modality averaging."""
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    model_classes = _model_classes(model)
    target = np.asarray(class_order, dtype=np.int64).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != len(model_classes):
        raise RuntimeError("SVM probability output has an invalid shape.")
    positions = {int(label): index for index, label in enumerate(target)}
    aligned = np.zeros((raw.shape[0], len(target)), dtype=np.float64)
    for source_index, label in enumerate(model_classes):
        if int(label) not in positions:
            raise RuntimeError("SVM predicted a class outside the frozen class order.")
        aligned[:, positions[int(label)]] = raw[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if not np.isfinite(aligned).all() or np.any(row_sums <= 0.0):
        raise RuntimeError("Aligned SVM probabilities contain invalid values.")
    return aligned / row_sums


def predict_models(
    models: Mapping[str, object],
    split: SplitData,
    sampling_rate: float,
) -> Dict[str, np.ndarray]:
    from mhfl_review.features import extract_multimodal_features

    f1, f2, fused = extract_multimodal_features(split.x1, split.x2, sampling_rate)
    class_order = np.asarray(models.get("class_order"), dtype=np.int64).reshape(-1)
    if class_order.size == 0:
        raise RuntimeError("Frozen SVM models are missing their class order.")
    early = _aligned_predict_proba(models["early"], fused, class_order)
    first = _aligned_predict_proba(models["mod1"], f1, class_order)
    second = _aligned_predict_proba(models["mod2"], f2, class_order)
    late = 0.5 * (first + second)
    late /= late.sum(axis=1, keepdims=True)
    return {"TF-SVM early fusion": early, "TF-SVM late fusion": late}


def _recording_ids(sample_ids: np.ndarray) -> set:
    """Return source-recording identifiers from ``relative/path:start`` sample IDs."""
    return {str(value).rsplit(":", 1)[0] for value in np.asarray(sample_ids).astype(str).tolist()}


def _split_audit(split_train: SplitData, split_test: SplitData) -> Dict[str, object]:
    train_ids = set(np.asarray(split_train.sample_ids).astype(str).tolist())
    test_ids = set(np.asarray(split_test.sample_ids).astype(str).tolist())
    exact_overlap = sorted(train_ids.intersection(test_ids))
    if exact_overlap:
        raise RuntimeError("Traditional baseline split contains exact train/test sample overlap.")
    train_recordings = _recording_ids(split_train.sample_ids)
    test_recordings = _recording_ids(split_test.sample_ids)
    return {
        "train_samples": int(len(split_train)),
        "test_samples": int(len(split_test)),
        "exact_sample_overlap": 0,
        "train_recordings": sorted(train_recordings),
        "test_recordings": sorted(test_recordings),
        "shared_recordings": sorted(train_recordings.intersection(test_recordings)),
        "claim_limit": (
            "The UO paper protocol is segment-disjoint but may share source recordings between train and test; "
            "it does not establish recording-disjoint generalization."
        ),
    }


def _feature_audit(features: np.ndarray, name: str) -> Dict[str, object]:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise RuntimeError("Invalid feature matrix for {0}: {1}.".format(name, array.shape))
    if not np.isfinite(array).all():
        raise RuntimeError("Feature matrix {0} contains NaN or Inf.".format(name))
    return {
        "name": name,
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "zero_variance_columns": int(np.sum(np.std(array, axis=0) <= 1.0e-12)),
        "finite": True,
    }


def _normalize_selected_params(best_params: Mapping[str, object]) -> Dict[str, object]:
    return {
        "C": float(best_params.get("C", best_params.get("svm__C"))),
        "gamma": str(best_params.get("gamma", best_params.get("svm__gamma"))),
    }


def select_with_training_cv(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Select one RBF-SVM setting using shuffled stratified CV on development training data only."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        raise RuntimeError("SVM development labels are empty.")
    folds = min(int(config.UO_SVM_CV_FOLDS), int(nonzero.min()))
    if folds < 2:
        raise RuntimeError("At least two samples per class are required for SVM development CV.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    search = GridSearchCV(
        make_svc(seed, 1.0, "scale"),
        param_grid={"svm__C": [0.1, 1.0, 10.0, 100.0], "svm__gamma": ["scale", "auto"]},
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(features, y)
    selected = _normalize_selected_params(search.best_params_)
    audit = {
        "selection_source": "development-training-only shuffled stratified CV",
        "folds": int(folds),
        "best_macro_f1_cv": float(search.best_score_),
        "selected": selected,
    }
    return selected, audit


def select_with_source_validation(
    train_features: np.ndarray,
    train_y: np.ndarray,
    val_features: np.ndarray,
    val_y: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Select one setting on a dedicated source-domain validation split; target loads are untouched."""
    from mhfl_review.train import metric_dict

    trials: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for candidate in SVM_PARAM_GRID:
        model = make_svc(seed, float(candidate["C"]), str(candidate["gamma"]))
        model.fit(train_features, train_y)
        probability = model.predict_proba(val_features)
        score = float(metric_dict(val_y, probability)["macro_f1"])
        row = {"C": float(candidate["C"]), "gamma": str(candidate["gamma"]), "macro_f1": score}
        trials.append(row)
        if best is None or score > float(best["macro_f1"]) + 1.0e-15:
            best = row
    if best is None:
        raise RuntimeError("SVM source-validation selection failed.")
    selected = {"C": float(best["C"]), "gamma": str(best["gamma"])}
    return selected, {
        "selection_source": "dedicated source-domain validation; target loads never used",
        "selected": selected,
        "best_macro_f1_validation": float(best["macro_f1"]),
        "trials": trials,
    }


def fit_fixed_models(
    train: SplitData,
    sampling_rate: float,
    seed: int,
    selected: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Fit the three classifiers with preselected, frozen hyperparameters."""
    from mhfl_review.features import extract_multimodal_features

    f1, f2, fused = extract_multimodal_features(train.x1, train.x2, sampling_rate)
    matrices = {"early": fused, "mod1": f1, "mod2": f2}
    class_order = np.sort(np.unique(np.asarray(train.y_int, dtype=np.int64)))
    if class_order.size < 2:
        raise RuntimeError("Frozen SVM training requires at least two classes.")
    models: Dict[str, object] = {"class_order": class_order}
    for offset, key in enumerate(("early", "mod1", "mod2")):
        params = selected[key]
        model = make_svc(seed + offset, float(params["C"]), str(params["gamma"]))
        model.fit(matrices[key], train.y_int)
        models[key] = model
    return models


def select_frozen_hyperparameters(
    mode: str,
    tuning_seed: int,
    tuning_n: int,
    rebuild_cache: bool,
    rebuild_split: bool,
    skip_uo: bool,
    skip_kaist: bool,
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, object]]:
    """Select once on a predeclared development split, then freeze across all reported points."""
    from mhfl_review.features import extract_multimodal_features
    from mhfl_review.train import prepare_kaist_splits, prepare_uo_paper_splits

    selected_all: Dict[str, Dict[str, Dict[str, object]]] = {}
    audit: Dict[str, object] = {
        "protocol": "single predeclared development tuning, frozen across all evaluation N and seeds",
        "tuning_seed": int(tuning_seed),
        "tuning_n": int(tuning_n),
        "evaluation_seeds": seed_sequence(mode),
        "test_or_target_data_used_for_selection": False,
        "frozen_across_all_reported_N_and_seeds": True,
        "result_acceptance_rule": (
            "All complete finite outputs are retained regardless of whether a baseline is above or below MHFL-MCA."
        ),
        "cases": {},
    }
    if not skip_uo:
        splits, meta = prepare_uo_paper_splits(
            int(tuning_n), int(tuning_seed), rebuild_cache=rebuild_cache, rebuild_split=rebuild_split
        )
        f1, f2, fused = extract_multimodal_features(
            splits["train"].x1, splits["train"].x2, config.UO_SAMPLING_RATE
        )
        matrices = {"early": fused, "mod1": f1, "mod2": f2}
        selected_case: Dict[str, Dict[str, object]] = {}
        audits: Dict[str, object] = {}
        for offset, key in enumerate(("early", "mod1", "mod2")):
            selected_case[key], audits[key] = select_with_training_cv(
                matrices[key], splits["train"].y_int, int(tuning_seed) + offset
            )
            audits[key]["feature_audit"] = _feature_audit(matrices[key], "UO development " + key)
        selected_all["UO"] = selected_case
        audit["cases"]["UO"] = {
            "split_signature": meta.get("plan_signature"),
            "split_audit": _split_audit(splits["train"], splits["test"]),
            "models": audits,
        }
    if not skip_kaist:
        splits, meta = prepare_kaist_splits(
            int(tuning_n), int(tuning_seed), rebuild_cache=rebuild_cache, rebuild_split=rebuild_split,
            allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
        )
        train = noisy_split(splits["train"], 0.0, int(tuning_seed) + 101)
        val = noisy_split(splits["val"], 0.0, int(tuning_seed) + 211)
        tr1, tr2, trf = extract_multimodal_features(train.x1, train.x2, config.KAIST_SAMPLING_RATE)
        va1, va2, vaf = extract_multimodal_features(val.x1, val.x2, config.KAIST_SAMPLING_RATE)
        tr_m = {"early": trf, "mod1": tr1, "mod2": tr2}
        va_m = {"early": vaf, "mod1": va1, "mod2": va2}
        selected_case = {}
        audits = {}
        for offset, key in enumerate(("early", "mod1", "mod2")):
            selected_case[key], audits[key] = select_with_source_validation(
                tr_m[key], train.y_int, va_m[key], val.y_int, int(tuning_seed) + offset
            )
            audits[key]["train_feature_audit"] = _feature_audit(tr_m[key], "KAIST development train " + key)
            audits[key]["val_feature_audit"] = _feature_audit(va_m[key], "KAIST development val " + key)
        selected_all["KAIST"] = selected_case
        audit["cases"]["KAIST"] = {
            "split_signature": meta.get("plan_signature"),
            "data_signatures": meta.get("data_signatures"),
            "models": audits,
        }
    return selected_all, audit


def noisy_split(split: SplitData, snr_db: float, seed: int) -> SplitData:
    from mhfl_review.data import SplitData, add_gaussian_noise

    return SplitData(
        add_gaussian_noise(split.x1, snr_db, np.random.RandomState(seed)),
        add_gaussian_noise(split.x2, snr_db, np.random.RandomState(seed + 9973)),
        split.y_onehot,
        split.y_int,
        split.sample_ids,
    )


def append_metrics(
    rows: List[Dict[str, object]],
    case: str,
    load: str,
    n_train: int,
    seed: int,
    snr_db: Optional[float],
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> None:
    from mhfl_review.train import metric_dict

    for method, probability in predictions.items():
        rows.append(
            {
                "case": case,
                "load": load,
                "n_train": int(n_train),
                "seed": int(seed),
                "snr_db": snr_db,
                "method": method,
                **metric_dict(truth, probability),
            }
        )


def seed_sequence(mode: str) -> List[int]:
    repeats = config.TRADITIONAL_REPEATS_FAST if mode == "fast" else config.TRADITIONAL_REPEATS_FULL
    return [config.GLOBAL_SEED + index for index in range(int(repeats))]


def n_values_for_mode(mode: str) -> Tuple[int, ...]:
    return FAST_N_VALUES if mode == "fast" else FULL_N_VALUES


def expected_raw_counts(mode: str) -> Dict[str, int]:
    repeats = len(seed_sequence(mode))
    n_count = len(n_values_for_mode(mode))
    uo = repeats * n_count * len(METHOD_NAMES)
    kaist = repeats * n_count * len(KAIST_LOADS) * len(METHOD_NAMES)
    kaist_noise = repeats * len(KAIST_LOADS) * len(KAIST_NOISE_SNRS) * len(METHOD_NAMES)
    return {"UO": uo, "KAIST": kaist, "KAIST-noise": kaist_noise, "total": uo + kaist + kaist_noise}


def expected_group_identities(mode: str) -> set:
    n_values = n_values_for_mode(mode)
    groups = {
        ("UO", "held-out", int(n_train), None, method)
        for n_train in n_values
        for method in METHOD_NAMES
    }
    groups.update(
        {
            ("KAIST", load, int(n_train), 0.0, method)
            for n_train in n_values
            for load in KAIST_LOADS
            for method in METHOD_NAMES
        }
    )
    groups.update(
        {
            ("KAIST-noise", load, int(max(n_values)), float(snr_db), method)
            for load in KAIST_LOADS
            for snr_db in KAIST_NOISE_SNRS
            for method in METHOD_NAMES
        }
    )
    return groups


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant is prohibited: {0}".format(value))


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Deep reference field {0} must be numeric.".format(field))
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError("Deep reference field {0} must be finite.".format(field))
    return number


def _read_current_manuscript_tex(source_tex_path: str) -> Tuple[bytes, Path, Optional[str]]:
    """Read a plain .tex file or a ``zip::member`` source without extracting it."""
    if not isinstance(source_tex_path, str) or not source_tex_path or "\ufffd" in source_tex_path:
        raise RuntimeError("Current-manuscript source_tex_path is missing or contains invalid Unicode.")
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
        raise RuntimeError("Current-manuscript archive/member source is missing: {0}".format(source_tex_path))
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


def _canonical_latex_model(row_text: str) -> Optional[str]:
    head = _strip_latex_emphasis(row_text.split("&", 1)[0]).strip()
    if "Proposed (MHFL-MCA)" in head:
        return "Full MHFL-MCA"
    for model in DEEP_MODEL_NAMES[:-1]:
        if head == model:
            return model
    return None


def _parse_current_manuscript_tables(tex_bytes: bytes) -> Dict[str, Dict[Tuple[str, int], Dict[str, str]]]:
    try:
        tex = tex_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Current-manuscript LaTeX source is not valid UTF-8.") from exc
    parsed: Dict[str, Dict[Tuple[str, int], Dict[str, str]]] = {}
    pair_pattern = re.compile(r"([01]\.\d{4})\s*\$\\pm\$\s*([01]\.\d{4})")
    for table_name, label in CURRENT_MANUSCRIPT_TABLE_LABELS.items():
        marker = "\\label{{{0}}}".format(label)
        label_index = tex.find(marker)
        if label_index < 0:
            raise RuntimeError("Current-manuscript LaTeX table label is missing: {0}".format(label))
        start = tex.rfind("\\begin{table", 0, label_index)
        end = tex.find("\\end{table}", label_index)
        if start < 0 or end < 0:
            raise RuntimeError("Cannot isolate current-manuscript table {0}.".format(label))
        region = tex[start : end + len("\\end{table}")]
        tabulars = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", region, flags=re.DOTALL)
        if len(tabulars) != 3:
            raise RuntimeError("Current-manuscript table {0} must contain exactly three paired-N tabulars.".format(label))
        table_rows: Dict[Tuple[str, int], Dict[str, str]] = {}
        for tabular in tabulars:
            n_values: List[int] = []
            for token in re.findall(r"(\d+)\s+training samples", tabular):
                value = int(token)
                if value not in n_values:
                    n_values.append(value)
            if len(n_values) != 2:
                raise RuntimeError("Every current-manuscript subtable must declare exactly two N values.")
            if "\\midrule" not in tabular or "\\bottomrule" not in tabular:
                raise RuntimeError("Current-manuscript subtable row boundaries are missing.")
            body = tabular.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            observed_models = set()
            for row_text in re.split(r"\\\\", body):
                model = _canonical_latex_model(row_text)
                if model is None:
                    continue
                cleaned = _strip_latex_emphasis(row_text)
                pairs = pair_pattern.findall(cleaned)
                if len(pairs) != 6:
                    raise RuntimeError(
                        "Current-manuscript row {0}/{1} must contain exactly six four-decimal mean/SD pairs.".format(
                            table_name, model
                        )
                    )
                if model in observed_models:
                    raise RuntimeError("Current-manuscript subtable contains a duplicate model row.")
                observed_models.add(model)
                for n_offset, n_train in enumerate(n_values):
                    base = n_offset * 3
                    values: Dict[str, str] = {}
                    for metric_offset, metric in enumerate(CURRENT_MANUSCRIPT_METRICS):
                        mean_text, sd_text = pairs[base + metric_offset]
                        values[metric + "_mean"] = mean_text
                        values[metric + "_sd"] = sd_text
                    identity = (model, int(n_train))
                    if identity in table_rows:
                        raise RuntimeError("Current-manuscript table contains a duplicate model/N identity.")
                    table_rows[identity] = values
            if observed_models != set(DEEP_MODEL_NAMES):
                raise RuntimeError("Current-manuscript subtable does not contain the exact six-model set.")
        expected = {(model, n_train) for model in DEEP_MODEL_NAMES for n_train in FULL_N_VALUES}
        if set(table_rows) != expected:
            raise RuntimeError("Current-manuscript table does not contain the exact six-model/six-N grid.")
        parsed[table_name] = table_rows
    return parsed


def _decimal_matches(reference_value: object, latex_value: str) -> bool:
    try:
        return Decimal(str(reference_value)) == Decimal(str(latex_value))
    except (InvalidOperation, ValueError):
        return False


def load_deep_reference(path: Path = DEEP_REFERENCE_PATH) -> Mapping[str, object]:
    reference_path = Path(path)
    if not reference_path.is_file():
        raise RuntimeError("Main-manuscript deep reference JSON is missing: {0}".format(reference_path))
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Main-manuscript deep reference JSON is invalid: {0}".format(exc)) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Main-manuscript deep reference JSON must contain an object.")
    validate_deep_reference(payload)
    return payload


def validate_deep_reference(reference: Mapping[str, object]) -> Dict[str, object]:
    expected_top = {
        "schema_version": 2,
        "confirmation_status": "confirmed",
        "reference_type": DEEP_REFERENCE_TYPE,
        "runs": 10,
        "aggregation": TRIMMED_AGGREGATION,
        "retained_after_trim": 8,
        "metrics_aggregated_independently": True,
        "per_seed_values_available": False,
        "paired_comparison_permitted": False,
        "statistical_significance_claim_permitted": False,
        "benchmark_scope": DEEP_BENCHMARK_SCOPE,
    }
    for field, expected in expected_top.items():
        if reference.get(field) != expected:
            raise RuntimeError("Deep reference field {0} must equal {1!r}.".format(field, expected))
    source_root_token = reference.get("source_root_at_confirmation")
    allowed_source_roots = {
        "workspace_root_two_levels_above_suite",
        "suite_provenance_main_manuscript_sources",
    }
    if source_root_token not in allowed_source_roots:
        raise RuntimeError("Deep reference source_root_at_confirmation is unsupported.")

    current_source = reference.get("current_manuscript_source")
    if not isinstance(current_source, Mapping):
        raise RuntimeError("Deep reference must record the current manuscript LaTeX source.")
    source_tex_path = current_source.get("source_tex_path")
    source_sha256 = current_source.get("source_sha256")
    reference_version = current_source.get("reference_version")
    extraction_method = current_source.get("extraction_method")
    if extraction_method != CURRENT_MANUSCRIPT_EXTRACTION_METHOD:
        raise RuntimeError("Deep reference must use the current-manuscript table extraction method.")
    if not isinstance(reference_version, str) or not reference_version:
        raise RuntimeError("Deep reference must record a non-empty current manuscript reference_version.")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise RuntimeError("Current-manuscript source SHA-256 must be lowercase hexadecimal.")
    tex_bytes, tex_container_path, tex_member = _read_current_manuscript_tex(str(source_tex_path))
    actual_tex_sha256 = hashlib.sha256(tex_bytes).hexdigest()
    if actual_tex_sha256 != source_sha256:
        raise RuntimeError("Current-manuscript LaTeX source SHA-256 mismatch.")
    if tex_member is not None:
        archive_sha256 = current_source.get("archive_sha256")
        if not isinstance(archive_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None:
            raise RuntimeError("Current-manuscript archive SHA-256 must be lowercase hexadecimal.")
        if sha256_file(tex_container_path).lower() != archive_sha256:
            raise RuntimeError("Current-manuscript archive SHA-256 mismatch.")
    latex_tables = _parse_current_manuscript_tables(tex_bytes)

    source_files = reference.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("Deep reference must record source_files provenance.")
    source_records: Dict[str, Mapping[str, object]] = {}
    verified_sources = 0
    source_frames: Dict[str, pd.DataFrame] = {}
    if source_root_token == "suite_provenance_main_manuscript_sources":
        workspace_root = config.MAIN_MANUSCRIPT_SOURCE_ROOT.resolve()
    else:
        workspace_root = config.SUITE_ROOT.parents[1].resolve()
    for record in source_files:
        if not isinstance(record, Mapping):
            raise RuntimeError("Every deep-reference source record must be an object.")
        source_id = record.get("id")
        source_path = record.get("source_path")
        digest = record.get("sha256")
        if not isinstance(source_id, str) or not source_id or source_id in source_records:
            raise RuntimeError("Deep-reference source ids must be non-empty and unique.")
        if not isinstance(source_path, str) or not source_path or Path(source_path).is_absolute():
            raise RuntimeError("Deep-reference source paths must be portable workspace-relative paths.")
        if record.get("workspace_relative_path") != source_path:
            raise RuntimeError("Deep-reference workspace-relative source paths are inconsistent.")
        if "\ufffd" in source_path or "\ufffd" in str(record.get("workspace_relative_path", "")):
            raise RuntimeError("Deep-reference source paths contain Unicode replacement characters.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("Deep-reference source SHA-256 values must be lowercase hexadecimal.")
        source_records[source_id] = record
        local_path = (workspace_root / Path(source_path)).resolve()
        try:
            local_path.relative_to(workspace_root)
        except ValueError as exc:
            raise RuntimeError("Deep-reference source escapes the recorded workspace root.") from exc
        if not local_path.is_file():
            raise RuntimeError("Deep-reference source file is missing: {0}".format(local_path))
        if sha256_file(local_path).lower() != digest:
            raise RuntimeError("Deep-reference source hash mismatch: {0}".format(local_path))
        expected_size = record.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or local_path.stat().st_size != expected_size:
            raise RuntimeError("Deep-reference source size mismatch: {0}".format(local_path))
        try:
            source_frames[source_id] = pd.read_csv(local_path, encoding="utf-8-sig")
        except Exception as exc:
            raise RuntimeError("Cannot read deep-reference source CSV: {0}".format(local_path)) from exc
        verified_sources += 1

    tables = reference.get("tables")
    if not isinstance(tables, Mapping) or set(tables) != set(DEEP_REFERENCE_TABLES):
        raise RuntimeError("Deep reference must contain exactly Tables 5, 9, and 10.")
    seen_rows = set()
    referenced_source_ids = set()
    metric_fields = tuple(field for metric in METRIC_COLUMNS for field in (metric + "_mean", metric + "_sd"))
    current_metric_fields = tuple(
        field for metric in CURRENT_MANUSCRIPT_METRICS for field in (metric + "_mean", metric + "_sd")
    )
    total_rows = 0
    source_values_verified = 0
    current_manuscript_values_verified = 0
    legacy_recall_values_verified = 0
    for table_name, table_contract in DEEP_REFERENCE_TABLES.items():
        table = tables.get(table_name)
        if not isinstance(table, Mapping):
            raise RuntimeError("Deep reference table {0} must be an object.".format(table_name))
        for field, expected in table_contract.items():
            if table.get(field) != expected:
                raise RuntimeError("Deep reference {0}.{1} must equal {2!r}.".format(table_name, field, expected))
        if tuple(table.get("n_values", ())) != FULL_N_VALUES:
            raise RuntimeError("Deep reference {0} must use the final six-N grid.".format(table_name))
        if tuple(table.get("models", ())) != DEEP_MODEL_NAMES:
            raise RuntimeError("Deep reference {0} must contain the six manuscript deep models.".format(table_name))
        rows = table.get("rows")
        if not isinstance(rows, list) or len(rows) != len(FULL_N_VALUES) * len(DEEP_MODEL_NAMES):
            raise RuntimeError("Deep reference {0} must contain exactly 36 rows.".format(table_name))
        table_pairs = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("Deep-reference metric rows must be objects.")
            model = row.get("model")
            n_train = row.get("n_train")
            if model not in DEEP_MODEL_NAMES or isinstance(n_train, bool) or n_train not in FULL_N_VALUES:
                raise RuntimeError("Deep-reference model/N identity is invalid.")
            pair = (str(model), int(n_train))
            if pair in table_pairs:
                raise RuntimeError("Deep-reference table contains duplicate model/N rows.")
            table_pairs.add(pair)
            source_id = row.get("source_file_id")
            if source_id not in source_records:
                raise RuntimeError("Deep-reference row points to an unknown source file.")
            referenced_source_ids.add(str(source_id))
            expected_provenance = {
                "source_tex_path": source_tex_path,
                "source_table": table_contract["source_table"],
                "source_sha256": source_sha256,
                "reference_version": reference_version,
                "extraction_method": CURRENT_MANUSCRIPT_EXTRACTION_METHOD,
            }
            for field, expected in expected_provenance.items():
                if row.get(field) != expected:
                    raise RuntimeError(
                        "Deep-reference row {0}/{1}/{2} has inconsistent {3}.".format(
                            table_name, model, n_train, field
                        )
                    )
            for field in metric_fields:
                value = _finite_number(row.get(field), "{0}.{1}".format(table_name, field))
                if field.endswith("_mean") and not 0.0 <= value <= 1.0:
                    raise RuntimeError("Deep-reference metric means must lie in [0, 1].")
                if field.endswith("_sd") and value < 0.0:
                    raise RuntimeError("Deep-reference metric SD values must be non-negative.")
            latex_row = latex_tables[table_name].get((str(model), int(n_train)))
            if latex_row is None:
                raise RuntimeError("Current-manuscript LaTeX table is missing a deep-reference identity.")
            for field in current_metric_fields:
                if not _decimal_matches(row.get(field), latex_row[field]):
                    raise RuntimeError(
                        "Deep-reference {0} does not match the current LaTeX table source exactly.".format(field)
                    )
                current_manuscript_values_verified += 1
                source_values_verified += 1
            source_frame = source_frames[str(source_id)]
            n_key = next(
                (candidate for candidate in ("Samples", "n_train", "N") if candidate in source_frame.columns),
                None,
            )
            if n_key is None:
                raise RuntimeError("Deep-reference source CSV has no recognized N column.")
            source_n = pd.to_numeric(source_frame[n_key], errors="coerce")
            matches = source_frame[source_n == int(n_train)]
            if len(matches) != 1:
                raise RuntimeError("Deep-reference source CSV must contain exactly one row for each referenced N.")
            source_row = matches.iloc[0]
            if table_contract["case"] == "UO":
                recall_columns = ("Recall_Mean", "Recall_Std")
            elif "n_train" in source_frame.columns:
                load = table_contract["load"]
                recall_columns = ("Recall_{0}_mean".format(load), "Recall_{0}_std".format(load))
            else:
                load_number = "2" if table_contract["load"] == "2Nm" else "4"
                recall_columns = ("Rec{0}_Mean".format(load_number), "Rec{0}_Std".format(load_number))
            if any(column not in source_frame.columns for column in recall_columns):
                raise RuntimeError("Legacy recall source CSV columns are incomplete.")
            for suffix, column in (("mean", recall_columns[0]), ("sd", recall_columns[1])):
                source_value = float(source_row[column])
                reference_value = float(row["macro_recall_" + suffix])
                if not np.isclose(reference_value, source_value, rtol=0.0, atol=1.0e-15):
                    raise RuntimeError("Deep-reference recall does not match its legacy compatibility source CSV.")
                legacy_recall_values_verified += 1
                source_values_verified += 1
            identity = (table_contract["case"], table_contract["load"], int(n_train), str(model))
            if identity in seen_rows:
                raise RuntimeError("Deep reference contains duplicate table identities.")
            seen_rows.add(identity)
            total_rows += 1
        expected_pairs = {(model, n_train) for model in DEEP_MODEL_NAMES for n_train in FULL_N_VALUES}
        if table_pairs != expected_pairs:
            raise RuntimeError("Deep-reference table model/N coverage is incomplete.")

    if total_rows != FULL_DEEP_REFERENCE_ROWS:
        raise RuntimeError("Deep reference must contain exactly {0} metric rows.".format(FULL_DEEP_REFERENCE_ROWS))
    if referenced_source_ids != set(source_records):
        raise RuntimeError("Every deep-reference source record must be used by at least one metric row.")
    if source_values_verified != FULL_DEEP_REFERENCE_ROWS * len(metric_fields):
        raise RuntimeError("Deep-reference source-value verification count is incomplete.")
    if current_manuscript_values_verified != FULL_DEEP_REFERENCE_ROWS * len(current_metric_fields):
        raise RuntimeError("Current-manuscript LaTeX value verification count is incomplete.")
    if legacy_recall_values_verified != FULL_DEEP_REFERENCE_ROWS * 2:
        raise RuntimeError("Legacy recall compatibility verification count is incomplete.")
    return {
        "status": "PASS",
        "reference_type": DEEP_REFERENCE_TYPE,
        "tables": sorted(DEEP_REFERENCE_TABLES),
        "rows": total_rows,
        "models": list(DEEP_MODEL_NAMES),
        "n_values": list(FULL_N_VALUES),
        "runs": 10,
        "retained_after_trim": 8,
        "aggregation": TRIMMED_AGGREGATION,
        "source_records": len(source_records),
        "source_records_referenced": len(referenced_source_ids),
        "source_files_present_and_hash_verified": verified_sources,
        "source_metric_values_verified": source_values_verified,
        "current_manuscript_source_tex_path": str(source_tex_path),
        "current_manuscript_source_sha256": source_sha256,
        "current_manuscript_archive_or_file": str(tex_container_path),
        "current_manuscript_archive_member": tex_member,
        "current_manuscript_metric_values_verified": current_manuscript_values_verified,
        "legacy_recall_values_verified": legacy_recall_values_verified,
        "extraction_method": CURRENT_MANUSCRIPT_EXTRACTION_METHOD,
        "reference_version": reference_version,
        "paired_comparison_permitted": False,
        "statistical_significance_claim_permitted": False,
    }


def validate_full_kaist_channel_config(
    current_channel: Optional[str] = None,
    vibration_column: Optional[int] = None,
    fallback_enabled: Optional[bool] = None,
) -> Dict[str, object]:
    channel = config.CURRENT_CHANNEL_NAME if current_channel is None else str(current_channel).strip()
    column = config.KAIST_VIB_COLUMN if vibration_column is None else int(vibration_column)
    fallback = config.ALLOW_CURRENT_CHANNEL_FALLBACK if fallback_enabled is None else bool(fallback_enabled)
    if channel != EXACT_CURRENT_CHANNEL:
        raise RuntimeError("Full traditional baselines require the exact confirmed U-phase current channel.")
    if column != 0:
        raise RuntimeError("Full traditional baselines require zero-based KAIST vibration column 0 (xA).")
    if fallback:
        raise RuntimeError("Full traditional baselines prohibit current-channel fallback.")
    return {
        "current_group": "Log",
        "current_channel": channel,
        "current_physical_meaning": "U-phase motor current",
        "vibration_column": column,
        "vibration_physical_meaning": "bearing housing A x-direction vibration",
        "current_fallback": False,
    }


def validate_raw_results(
    raw: pd.DataFrame,
    mode: str,
    expected_seeds: Sequence[int],
    require_complete: bool = True,
) -> Dict[str, Any]:
    missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(raw.columns))
    if missing:
        raise RuntimeError("Traditional-baseline raw results are missing columns: {0}".format(", ".join(missing)))
    if raw.empty:
        raise RuntimeError("Traditional-baseline raw results are empty.")
    duplicate_count = int(raw.duplicated(list(GROUP_COLUMNS) + ["seed"], keep=False).sum())
    if duplicate_count:
        raise RuntimeError("Traditional-baseline raw results contain duplicate group/seed rows.")
    nonnullable = [column for column in RAW_REQUIRED_COLUMNS if column != "snr_db"]
    if raw[nonnullable].isna().any().any():
        raise RuntimeError("Traditional-baseline raw results contain NaN values in required fields.")
    numeric_columns = ["n_train", "seed"] + list(METRIC_COLUMNS)
    numeric = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Traditional-baseline raw results contain non-numeric or infinite values.")
    for metric in METRIC_COLUMNS:
        if not numeric[metric].between(0.0, 1.0).all():
            raise RuntimeError("Traditional-baseline metric {0} is outside [0, 1].".format(metric))
    uo_snr = raw.loc[raw["case"] == "UO", "snr_db"]
    non_uo_snr = pd.to_numeric(raw.loc[raw["case"] != "UO", "snr_db"], errors="coerce")
    if not uo_snr.isna().all():
        raise RuntimeError("UO rows must mark SNR as not applicable.")
    if non_uo_snr.isna().any() or not np.isfinite(non_uo_snr.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("KAIST SNR values must be finite.")
    if not raw["method"].isin(METHOD_NAMES).all():
        raise RuntimeError("Traditional-baseline raw results contain an unknown method.")

    target_seeds = {int(seed) for seed in expected_seeds}
    observed_seeds = {int(seed) for seed in numeric["seed"].tolist()}
    if not observed_seeds.issubset(target_seeds):
        raise RuntimeError("Traditional-baseline raw results contain a seed outside the fixed sequence.")

    component_counts = {name: int((raw["case"] == name).sum()) for name in ("UO", "KAIST", "KAIST-noise")}
    if require_complete:
        expected = expected_raw_counts(mode)
        if observed_seeds != target_seeds:
            raise RuntimeError("The fixed traditional-baseline seed set is incomplete.")
        if len(raw) != expected["total"] or any(component_counts[key] != expected[key] for key in component_counts):
            raise RuntimeError("Traditional-baseline raw row counts do not match the complete protocol.")
        grouped = raw.groupby(list(GROUP_COLUMNS), dropna=False)["seed"].agg(lambda values: {int(v) for v in values})
        expected_group_count = expected["total"] // len(target_seeds)
        if len(grouped) != expected_group_count or not all(value == target_seeds for value in grouped):
            raise RuntimeError("Every traditional-baseline group must contain the identical fixed seed set.")
        observed_groups = {
            (
                str(row.case),
                str(row.load),
                int(row.n_train),
                None if pd.isna(row.snr_db) else float(row.snr_db),
                str(row.method),
            )
            for row in raw[list(GROUP_COLUMNS)].drop_duplicates().itertuples(index=False)
        }
        if observed_groups != expected_group_identities(mode):
            raise RuntimeError("Traditional-baseline groups do not match the exact N/load/SNR/method protocol.")

    return {
        "rows": int(len(raw)),
        "component_rows": component_counts,
        "groups": int(raw.groupby(list(GROUP_COLUMNS), dropna=False).ngroups),
        "seeds": sorted(observed_seeds),
        "duplicate_rows": 0,
        "nan_metric_values": 0,
        "infinite_values": 0,
        "failed_seeds": [],
        "uo_snr_structural_na_only": True,
    }


def summarize(raw: pd.DataFrame, expected_seed_count: int = 10, use_trim: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, block in raw.groupby(list(GROUP_COLUMNS), dropna=False, sort=False):
        if block["seed"].nunique() != int(expected_seed_count):
            raise RuntimeError("Traditional-baseline summary requires the expected seeds in every group.")
        row = dict(zip(GROUP_COLUMNS, keys))
        row.update(
            {
                "seeds_total": int(expected_seed_count),
                "seeds_used": int(expected_seed_count) - 2 if use_trim else int(expected_seed_count),
                "retained_after_trim": int(expected_seed_count) - 2 if use_trim else int(expected_seed_count),
                "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
            }
        )
        for metric in METRIC_COLUMNS:
            values = block[metric].to_numpy(dtype=np.float64)
            untrimmed_mean, untrimmed_sd = mean_sd(values)
            center, spread = trimmed_mean_sd(values) if use_trim else (untrimmed_mean, untrimmed_sd)
            row[metric + "_mean"] = center
            row[metric + "_sd"] = spread
            row[metric + "_untrimmed_mean"] = untrimmed_mean
            row[metric + "_untrimmed_sd"] = untrimmed_sd
        rows.append(row)
    return pd.DataFrame(rows)


def deep_reference_candidate_rows(reference: Mapping[str, object]) -> pd.DataFrame:
    validate_deep_reference(reference)
    rows: List[Dict[str, object]] = []
    tables = reference["tables"]
    for table_name, contract in DEEP_REFERENCE_TABLES.items():
        table = tables[table_name]
        for source_row in table["rows"]:
            row: Dict[str, object] = {
                "source_type": "main_manuscript_deep_reference",
                "reference_type": DEEP_REFERENCE_TYPE,
                "source_table": contract["source_table"],
                "case": contract["case"],
                "load": contract["load"],
                "n_train": int(source_row["n_train"]),
                "model": str(source_row["model"]),
                "runs": 10,
                "retained_after_trim": 8,
                "aggregation": TRIMMED_AGGREGATION,
                "benchmark_scope": DEEP_BENCHMARK_SCOPE,
                "comparison_type": "descriptive_aggregated_reference_only",
            }
            for metric in METRIC_COLUMNS:
                row[metric + "_mean"] = float(source_row[metric + "_mean"])
                row[metric + "_sd"] = float(source_row[metric + "_sd"])
            rows.append(row)
    return pd.DataFrame(rows, columns=FULL_CANDIDATE_COLUMNS)


def build_full_manuscript_candidate(
    summary: pd.DataFrame,
    deep_reference: Mapping[str, object],
) -> pd.DataFrame:
    deep_rows = deep_reference_candidate_rows(deep_reference)
    clean = summary[summary["case"].isin(("UO", "KAIST"))].copy()
    if len(clean) != FULL_TRADITIONAL_CANDIDATE_ROWS:
        raise RuntimeError("Full traditional clean summary must contain exactly 36 rows.")
    traditional_rows: List[Dict[str, object]] = []
    for _, source_row in clean.iterrows():
        case = str(source_row["case"])
        row: Dict[str, object] = {
            "source_type": "new_traditional_baseline",
            "reference_type": TRADITIONAL_REFERENCE_TYPE,
            "source_table": "reviewer_suite_06_clean",
            "case": case,
            "load": str(source_row["load"]),
            "n_train": int(source_row["n_train"]),
            "model": str(source_row["method"]),
            "runs": int(source_row["seeds_total"]),
            "retained_after_trim": int(source_row["retained_after_trim"]),
            "aggregation": str(source_row["aggregation"]),
            "benchmark_scope": BENCHMARK_SCOPE,
            "comparison_type": "descriptive_cross_method_benchmark_only",
        }
        for metric in METRIC_COLUMNS:
            row[metric + "_mean"] = float(source_row[metric + "_mean"])
            row[metric + "_sd"] = float(source_row[metric + "_sd"])
        traditional_rows.append(row)
    combined = pd.concat(
        [deep_rows, pd.DataFrame(traditional_rows, columns=FULL_CANDIDATE_COLUMNS)],
        ignore_index=True,
    )
    return combined.loc[:, FULL_CANDIDATE_COLUMNS]


def refresh_manuscript_candidate_from_existing_summary(
    summary_path: Path,
    candidate_path: Path,
    reference_path: Path = DEEP_REFERENCE_PATH,
) -> Dict[str, object]:
    """Refresh the derived hybrid candidate without running SVMs or reading experiment raw data."""
    frozen_summary_path = Path(summary_path)
    output_path = Path(candidate_path)
    if not frozen_summary_path.is_file():
        raise RuntimeError("Completed traditional_baselines_summary.csv is missing: {0}".format(frozen_summary_path))
    summary_sha256_before = sha256_file(frozen_summary_path)
    summary_mtime_before = frozen_summary_path.stat().st_mtime_ns
    summary = pd.read_csv(frozen_summary_path)
    required_columns = set(GROUP_COLUMNS).union(
        {"seeds_total", "retained_after_trim", "aggregation"},
        {field for metric in METRIC_COLUMNS for field in (metric + "_mean", metric + "_sd")},
    )
    if not required_columns.issubset(summary.columns):
        raise RuntimeError("Completed traditional summary is missing candidate-generation fields.")
    if len(summary) != 48 or summary.duplicated(list(GROUP_COLUMNS), keep=False).any():
        raise RuntimeError("Completed traditional summary must contain exactly 48 unique protocol groups.")
    actual_groups = set()
    for row in summary.itertuples(index=False):
        snr_value = getattr(row, "snr_db")
        actual_groups.add(
            (
                str(getattr(row, "case")),
                str(getattr(row, "load")),
                int(getattr(row, "n_train")),
                None if pd.isna(snr_value) else float(snr_value),
                str(getattr(row, "method")),
            )
        )
    if actual_groups != expected_group_identities("full"):
        raise RuntimeError("Completed traditional summary does not match the frozen 48-group full protocol.")
    if not (pd.to_numeric(summary["seeds_total"], errors="coerce") == 10).all():
        raise RuntimeError("Completed traditional summary must report ten seeds per group.")
    if not (pd.to_numeric(summary["retained_after_trim"], errors="coerce") == 8).all():
        raise RuntimeError("Completed traditional summary must retain eight values per group.")
    if not (summary["aggregation"].astype(str) == TRIMMED_AGGREGATION).all():
        raise RuntimeError("Completed traditional summary aggregation is not the frozen full protocol.")
    numeric_fields = [field for metric in METRIC_COLUMNS for field in (metric + "_mean", metric + "_sd")]
    numeric = summary[numeric_fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Completed traditional summary contains NaN or Inf in candidate metrics.")

    deep_reference = load_deep_reference(Path(reference_path))
    candidate = build_full_manuscript_candidate(summary, deep_reference)
    if len(candidate) != FULL_MANUSCRIPT_CANDIDATE_ROWS:
        raise RuntimeError("Hybrid manuscript candidate must contain exactly 144 rows.")
    source_counts = candidate["source_type"].value_counts().to_dict()
    if source_counts != {"main_manuscript_deep_reference": 108, "new_traditional_baseline": 36}:
        raise RuntimeError("Hybrid manuscript candidate must contain 108 deep and 36 traditional rows.")
    scope_counts = {
        "UO": int(((candidate["case"] == "UO") & (candidate["load"] == "held-out")).sum()),
        "KAIST_2Nm": int(((candidate["case"] == "KAIST") & (candidate["load"] == "2Nm")).sum()),
        "KAIST_4Nm": int(((candidate["case"] == "KAIST") & (candidate["load"] == "4Nm")).sum()),
    }
    if scope_counts != {"UO": 48, "KAIST_2Nm": 48, "KAIST_4Nm": 48}:
        raise RuntimeError("Hybrid manuscript candidate must contain 48 rows for each manuscript case/load table.")
    if set(candidate["case"]) != {"UO", "KAIST"}:
        raise RuntimeError("Hybrid manuscript candidate must not contain KAIST-noise rows.")
    identity_columns = ["source_type", "source_table", "case", "load", "n_train", "model"]
    if candidate.duplicated(identity_columns, keep=False).any():
        raise RuntimeError("Hybrid manuscript candidate contains duplicate identities.")
    candidate_numeric = candidate[["n_train", "runs", "retained_after_trim"] + numeric_fields].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(candidate_numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Hybrid manuscript candidate contains NaN or Inf.")
    if sha256_file(frozen_summary_path) != summary_sha256_before or frozen_summary_path.stat().st_mtime_ns != summary_mtime_before:
        raise RuntimeError("Completed traditional summary changed during candidate refresh; refusing to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_path, index=False)
    if sha256_file(frozen_summary_path) != summary_sha256_before or frozen_summary_path.stat().st_mtime_ns != summary_mtime_before:
        raise RuntimeError("Completed traditional summary changed while writing the derived candidate.")
    return {
        "status": "PASS",
        "summary_path": str(frozen_summary_path.resolve()),
        "summary_sha256_before": summary_sha256_before,
        "summary_sha256_after": sha256_file(frozen_summary_path),
        "summary_rows": int(len(summary)),
        "traditional_clean_rows": 36,
        "deep_reference_rows": 108,
        "candidate_rows": int(len(candidate)),
        "candidate_sha256": sha256_file(output_path),
        "scope_counts": scope_counts,
        "duplicates": 0,
        "nan_or_inf": 0,
        "svm_or_model_execution": False,
    }


def full_candidate_scope() -> Dict[str, object]:
    return {
        "included_cases": ["UO", "KAIST"],
        "excluded_cases": ["KAIST-noise"],
        "n_values": list(FULL_N_VALUES),
        "expected_rows": FULL_MANUSCRIPT_CANDIDATE_ROWS,
        "deep_reference_rows": FULL_DEEP_REFERENCE_ROWS,
        "traditional_rows": FULL_TRADITIONAL_CANDIDATE_ROWS,
        "deep_models": list(DEEP_MODEL_NAMES),
        "traditional_methods": list(METHOD_NAMES),
        "noise_source_data_retained_outside_candidate": True,
    }


def validate_full_post_gate(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    channel_gate: Mapping[str, object],
    deep_reference: Mapping[str, object],
) -> Dict[str, object]:
    raw_gate = validate_raw_results(raw, "full", seed_sequence("full"), require_complete=True)
    expected_summary = summarize(raw, expected_seed_count=10, use_trim=True)
    if set(summary.columns) != set(expected_summary.columns):
        raise RuntimeError("Traditional summary columns do not match the frozen full protocol.")
    summary_sort = list(GROUP_COLUMNS)
    actual_summary = summary.sort_values(summary_sort, na_position="first").reset_index(drop=True)
    actual_summary = actual_summary[list(expected_summary.columns)]
    expected_summary = expected_summary.sort_values(summary_sort, na_position="first").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual_summary,
            expected_summary,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise RuntimeError("Traditional summary does not reproduce metric-wise trimming from raw results.") from exc
    channel_report = validate_full_kaist_channel_config(
        str(channel_gate.get("current_channel", "")),
        int(channel_gate.get("vibration_column", -1)),
        bool(channel_gate.get("current_fallback", True)),
    )
    deep_gate = validate_deep_reference(deep_reference)

    required_summary_columns = set(GROUP_COLUMNS).union(
        {"seeds_total", "retained_after_trim", "aggregation"},
        {field for metric in METRIC_COLUMNS for field in (metric + "_mean", metric + "_sd")},
    )
    if not required_summary_columns.issubset(summary.columns):
        raise RuntimeError("Traditional summary is missing final post-gate columns.")
    if len(summary) != 48 or summary.duplicated(list(GROUP_COLUMNS), keep=False).any():
        raise RuntimeError("Traditional summary must contain 48 unique protocol groups.")
    if not (summary["seeds_total"] == 10).all() or not (summary["retained_after_trim"] == 8).all():
        raise RuntimeError("Every traditional summary group must retain 8 of 10 seeds.")
    if not (summary["aggregation"] == TRIMMED_AGGREGATION).all():
        raise RuntimeError("Traditional summary aggregation is not the final trimmed protocol.")
    summary_metric_columns = [field for metric in METRIC_COLUMNS for field in (metric + "_mean", metric + "_sd")]
    summary_metrics = summary[summary_metric_columns].apply(pd.to_numeric, errors="coerce")
    if summary_metrics.isna().any().any() or not np.isfinite(summary_metrics.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Traditional summary contains NaN/Inf metric values.")

    if tuple(candidate.columns) != FULL_CANDIDATE_COLUMNS:
        raise RuntimeError("Hybrid traditional candidate columns do not match the frozen contract.")
    if len(candidate) != FULL_MANUSCRIPT_CANDIDATE_ROWS:
        raise RuntimeError("Hybrid traditional candidate must contain exactly 144 rows.")
    if candidate.duplicated(["case", "load", "n_train", "model"], keep=False).any():
        raise RuntimeError("Hybrid traditional candidate contains duplicate model/N rows.")
    if set(candidate["case"]) != {"UO", "KAIST"} or "KAIST-noise" in set(candidate["case"]):
        raise RuntimeError("Hybrid manuscript candidate must be clean-only and exclude KAIST noise rows.")
    expected_sources = {
        "main_manuscript_deep_reference": FULL_DEEP_REFERENCE_ROWS,
        "new_traditional_baseline": FULL_TRADITIONAL_CANDIDATE_ROWS,
    }
    source_counts = candidate["source_type"].value_counts().to_dict()
    if source_counts != expected_sources:
        raise RuntimeError("Hybrid candidate deep/traditional row counts are invalid.")
    candidate_metrics = candidate[summary_metric_columns].apply(pd.to_numeric, errors="coerce")
    if candidate_metrics.isna().any().any() or not np.isfinite(candidate_metrics.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Hybrid candidate contains NaN/Inf metric values.")
    if not (candidate["runs"] == 10).all() or not (candidate["retained_after_trim"] == 8).all():
        raise RuntimeError("Hybrid candidate must record 10 runs and 8 retained values.")
    if not (candidate["aggregation"] == TRIMMED_AGGREGATION).all():
        raise RuntimeError("Hybrid candidate aggregation metadata is invalid.")
    traditional_candidate = candidate[candidate["source_type"] == "new_traditional_baseline"]
    if not (traditional_candidate["benchmark_scope"] == BENCHMARK_SCOPE).all():
        raise RuntimeError("Traditional candidate benchmark_scope is invalid.")
    deep_candidate = candidate[candidate["source_type"] == "main_manuscript_deep_reference"]
    if not (deep_candidate["reference_type"] == DEEP_REFERENCE_TYPE).all():
        raise RuntimeError("Deep candidate reference_type is invalid.")
    if not (deep_candidate["comparison_type"] == "descriptive_aggregated_reference_only").all():
        raise RuntimeError("Aggregated deep references cannot claim paired comparisons.")
    forbidden_tokens = ("paired_delta", "p_value", "pvalue", "significance")
    if any(any(token in column.lower() for token in forbidden_tokens) for column in candidate.columns):
        raise RuntimeError("Hybrid candidate contains a prohibited paired/significance field.")
    expected_models = set(DEEP_MODEL_NAMES).union(METHOD_NAMES)
    for (case, load, n_train), block in candidate.groupby(["case", "load", "n_train"], sort=False):
        if int(n_train) not in FULL_N_VALUES or set(block["model"]) != expected_models:
            raise RuntimeError("Hybrid candidate model coverage is incomplete for a clean protocol group.")
        if case == "UO" and load != "held-out":
            raise RuntimeError("UO hybrid candidate load label is invalid.")
        if case == "KAIST" and load not in KAIST_LOADS:
            raise RuntimeError("KAIST hybrid candidate load label is invalid.")
    expected_candidate = build_full_manuscript_candidate(expected_summary, deep_reference)
    candidate_sort = ["case", "load", "n_train", "model", "source_type"]
    actual_candidate = candidate.sort_values(candidate_sort).reset_index(drop=True)
    expected_candidate = expected_candidate.sort_values(candidate_sort).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual_candidate,
            expected_candidate,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise RuntimeError("Hybrid traditional candidate does not reproduce its validated sources.") from exc

    return {
        "status": "PASS",
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "candidate_rows": int(len(candidate)),
        "methods": list(METHOD_NAMES),
        "seeds_per_group": 10,
        "retained_after_trim": 8,
        "duplicates": 0,
        "nan_or_inf": 0,
        "failed_groups": [],
        "metrics_trimmed_independently": True,
        "summary_recomputed_from_raw": True,
        "candidate_recomputed_from_validated_sources": True,
        "channel_gate": channel_report,
        "deep_reference_gate": deep_gate,
        "candidate_scope": full_candidate_scope(),
        "deep_model_training_performed": False,
        "raw_gate": raw_gate,
    }


def write_candidate_if_gate_passed(
    candidate: pd.DataFrame,
    candidate_path: Path,
    post_gate: Mapping[str, object],
) -> None:
    if post_gate.get("status") != "PASS":
        raise RuntimeError("Refusing to write manuscript candidate rows because the post-gate did not pass.")
    candidate.to_csv(candidate_path, index=False)


def _file_record(path: Path) -> Dict[str, object]:
    target = Path(path).resolve()
    return {"path": str(target), "size_bytes": int(target.stat().st_size), "sha256": sha256_file(target)}


def write_run_manifest(
    out_dir: Path,
    mode: str,
    raw_gate: Mapping[str, Any],
    channel_gate: Optional[Mapping[str, object]],
    raw_path: Path,
    summary_path: Path,
    candidate_path: Path,
    scope_path: Path,
    deep_reference_path: Optional[Path] = None,
    deep_reference_gate: Optional[Mapping[str, object]] = None,
    post_gate: Optional[Mapping[str, object]] = None,
) -> Path:
    seeds = seed_sequence(mode)
    use_trim = mode == "full"
    payload = {
        "status": "PASS",
        "mode": mode,
        "run_tag": config.RUN_TAG,
        "target_seeds": seeds,
        "expected_raw_rows": expected_raw_counts(mode),
        "raw_gate": dict(raw_gate),
        "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
        "retained_after_trim": len(seeds) - 2 if use_trim else len(seeds),
        "metrics_trimmed_independently": True,
        "failed_seeds": [],
        "failed_groups": [],
        "channel_gate": None if channel_gate is None else dict(channel_gate),
        "benchmark_scope": BENCHMARK_SCOPE,
        "deep_model_training_performed": False,
        "candidate_scope": full_candidate_scope() if mode == "full" else {"mode": "fast", "final_candidate": False},
        "deep_reference": None if deep_reference_path is None else _file_record(deep_reference_path),
        "deep_reference_gate": None if deep_reference_gate is None else dict(deep_reference_gate),
        "post_gate": None if post_gate is None else dict(post_gate),
        "raw": _file_record(raw_path),
        "summary": _file_record(summary_path),
        "manuscript_candidate": _file_record(candidate_path) if mode == "full" else None,
        "fast_summary_preview": _file_record(candidate_path) if mode == "fast" else None,
        "benchmark_scope_file": _file_record(scope_path),
    }
    path = Path(out_dir) / "traditional_baselines_run_manifest.json"
    write_json(path, payload)
    return path


def update_full_figure_contract(figure_paths: Mapping[str, Path]) -> None:
    contract_path = Path(figure_paths["contract"])
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["center_statistic"] = "metric-wise trimmed mean after removing one highest and one lowest result"
    payload["spread_definition"] = "population SD (ddof=0) across eight independently retained values per metric"
    payload["notes"] = [
        "Accuracy, Macro-F1, Macro-Precision, and Macro-Recall are trimmed independently over 10 fixed seeds."
    ]
    write_json(contract_path, payload)


def run_uo(
    rows: List[Dict[str, object]],
    mode: str,
    rebuild_cache: bool,
    rebuild_split: bool,
    selected: Mapping[str, Mapping[str, object]],
    split_audits: List[Dict[str, object]],
) -> None:
    from mhfl_review.train import prepare_uo_paper_splits

    n_values = n_values_for_mode(mode)
    repeats = config.TRADITIONAL_REPEATS_FAST if mode == "fast" else config.TRADITIONAL_REPEATS_FULL
    for repeat in range(repeats):
        seed = config.GLOBAL_SEED + repeat
        for n_train in n_values:
            splits, meta = prepare_uo_paper_splits(
                n_train, seed, rebuild_cache=rebuild_cache, rebuild_split=rebuild_split
            )
            split_audit = _split_audit(splits["train"], splits["test"])
            split_audit.update({"case": "UO", "seed": int(seed), "n_train": int(n_train),
                                "split_signature": meta.get("plan_signature")})
            split_audits.append(split_audit)
            models = fit_fixed_models(splits["train"], config.UO_SAMPLING_RATE, seed, selected)
            predictions = predict_models(models, splits["test"], config.UO_SAMPLING_RATE)
            append_metrics(rows, "UO", "held-out", n_train, seed, None, splits["test"].y_int, predictions)
            print("[Traditional/UO frozen] seed={0} N={1}".format(seed, n_train))


def run_kaist(
    rows: List[Dict[str, object]],
    mode: str,
    rebuild_cache: bool,
    rebuild_split: bool,
    selected: Mapping[str, Mapping[str, object]],
) -> None:
    from mhfl_review.train import prepare_kaist_splits

    n_values = n_values_for_mode(mode)
    repeats = config.TRADITIONAL_REPEATS_FAST if mode == "fast" else config.TRADITIONAL_REPEATS_FULL
    for repeat in range(repeats):
        seed = config.GLOBAL_SEED + repeat
        for n_train in n_values:
            splits, _ = prepare_kaist_splits(
                n_train, seed, rebuild_cache=rebuild_cache, rebuild_split=rebuild_split,
                allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
            )
            train = noisy_split(splits["train"], 0.0, seed + 101)
            models = fit_fixed_models(train, config.KAIST_SAMPLING_RATE, seed, selected)
            for load in ("2Nm", "4Nm"):
                test = noisy_split(splits[load], 0.0, seed + (2 if load == "2Nm" else 4))
                predictions = predict_models(models, test, config.KAIST_SAMPLING_RATE)
                append_metrics(rows, "KAIST", load, n_train, seed, 0.0, test.y_int, predictions)
            if n_train == max(n_values):
                for load in ("2Nm", "4Nm"):
                    for snr_db in (0.0, -4.0, -8.0):
                        test = noisy_split(
                            splits[load], snr_db,
                            seed + abs(int(snr_db)) * 100 + (2 if load == "2Nm" else 4),
                        )
                        predictions = predict_models(models, test, config.KAIST_SAMPLING_RATE)
                        append_metrics(rows, "KAIST-noise", load, n_train, seed, snr_db, test.y_int, predictions)
            print("[Traditional/KAIST frozen] seed={0} N={1}".format(seed, n_train))


def main() -> None:
    args = parse_args()
    from mhfl_review.plotting import plot_traditional_baseline_evidence

    tuning_seed = int(config.GLOBAL_SEED + SVM_TUNING_SEED_OFFSET if args.tuning_seed is None else args.tuning_seed)
    tuning_n = int(args.tuning_n)
    if args.mode == "full":
        if tuning_n != SVM_TUNING_N:
            raise RuntimeError("Full mode requires the predeclared SVM tuning point N=15.")
        if tuning_seed in set(seed_sequence("full")):
            raise RuntimeError("The dedicated SVM tuning seed must be outside the ten evaluation seeds.")

    out_dir = config.OUTPUT_ROOT / "06_traditional_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    manuscript_candidate_path = out_dir / "manuscript_candidate_rows.csv"
    candidate_path = (
        manuscript_candidate_path
        if args.mode == "full"
        else out_dir / "fast_summary_preview.csv"
    )
    if args.mode == "full" and manuscript_candidate_path.exists():
        manuscript_candidate_path.unlink()
    try:
        config.enforce_run_safety(
            args.mode,
            allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
            require_kaist_spec=False,
        )
        channel_gate: Optional[Dict[str, object]] = None
        deep_reference: Optional[Mapping[str, object]] = None
        deep_reference_gate: Optional[Dict[str, object]] = None
        if args.mode == "full":
            if args.skip_uo or args.skip_kaist:
                raise RuntimeError("Full traditional-baseline runs prohibit --skip-uo and --skip-kaist.")
            channel_gate = validate_full_kaist_channel_config()
            deep_reference = load_deep_reference(DEEP_REFERENCE_PATH)
            deep_reference_gate = validate_deep_reference(deep_reference)

        selected_params, tuning_audit = select_frozen_hyperparameters(
            args.mode, tuning_seed, tuning_n, args.rebuild_cache, args.rebuild_split,
            args.skip_uo, args.skip_kaist,
        )
        tuning_path = out_dir / "traditional_baseline_tuning_audit.json"
        write_json(tuning_path, tuning_audit)

        rows: List[Dict[str, object]] = []
        split_audits: List[Dict[str, object]] = []
        if not args.skip_uo:
            run_uo(
                rows, args.mode, args.rebuild_cache, args.rebuild_split,
                selected_params["UO"], split_audits,
            )
        if not args.skip_kaist:
            run_kaist(
                rows, args.mode, args.rebuild_cache, args.rebuild_split, selected_params["KAIST"]
            )
        tuning_audit["uo_evaluation_split_audits"] = split_audits
        write_json(tuning_path, tuning_audit)
        raw = pd.DataFrame(rows)
        seeds = seed_sequence(args.mode)
        raw_gate = validate_raw_results(raw, args.mode, seeds, require_complete=args.mode == "full")
        raw_path = out_dir / "traditional_baselines_raw.csv"
        raw.to_csv(raw_path, index=False)
        summary = summarize(raw, expected_seed_count=len(seeds), use_trim=args.mode == "full")
        summary_path = out_dir / "traditional_baselines_summary.csv"
        summary.to_csv(summary_path, index=False)

        post_gate: Optional[Dict[str, object]] = None
        if args.mode == "full":
            if channel_gate is None or deep_reference is None:
                raise RuntimeError("Full traditional post-gate inputs were not loaded.")
            candidate = build_full_manuscript_candidate(summary, deep_reference)
            post_gate = validate_full_post_gate(raw, summary, candidate, channel_gate, deep_reference)
        else:
            candidate = summary

        figure_paths = plot_traditional_baseline_evidence(
            summary,
            summary_path,
            out_dir / "traditional_baseline_evidence",
        )
        if args.mode == "full":
            update_full_figure_contract(figure_paths)
        if args.mode == "full":
            if post_gate is None:
                raise RuntimeError("Full traditional post-gate result is missing.")
            write_candidate_if_gate_passed(candidate, candidate_path, post_gate)
        else:
            candidate.to_csv(candidate_path, index=False)
        scope_path = out_dir / "benchmark_scope.txt"
        scope_path.write_text(
            BENCHMARK_SCOPE
            + ". SVM hyperparameters are selected once at the predeclared N=15 development point "
            + "using a dedicated non-evaluation seed and are frozen for all reported N/seeds. "
            + "These are non-neural multimodal references, not exact reproductions of CSC/GJO-OMP; "
            + "the deep-model rows are frozen main-manuscript aggregated references and are not retrained by 06.\n",
            encoding="utf-8",
        )
        manifest_path = write_run_manifest(
            out_dir,
            args.mode,
            raw_gate,
            channel_gate,
            raw_path,
            summary_path,
            candidate_path,
            scope_path,
            deep_reference_path=DEEP_REFERENCE_PATH if args.mode == "full" else None,
            deep_reference_gate=deep_reference_gate,
            post_gate=post_gate,
        )
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_payload["svm_hyperparameter_protocol"] = {
            "selection": "single predeclared development tuning",
            "tuning_n": tuning_n,
            "tuning_seed": tuning_seed,
            "evaluation_seeds": seed_sequence(args.mode),
            "frozen_across_all_reported_points": True,
            "test_or_target_data_used_for_selection": False,
        }
        manifest_payload["tuning_audit"] = _file_record(tuning_path)
        write_json(manifest_path, manifest_payload)
        print("Outputs saved to:", out_dir)
    except Exception:
        if args.mode == "full" and manuscript_candidate_path.exists():
            manuscript_candidate_path.unlink()
        raise


if __name__ == "__main__":
    main()
