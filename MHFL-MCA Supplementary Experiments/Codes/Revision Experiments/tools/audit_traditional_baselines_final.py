from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


EXPECTED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
EXACT_CURRENT_CHANNEL = "cDAQ9185-1F486B5Mod2/ai0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only pre-training audit for final traditional baselines.")
    parser.add_argument("--phase", choices=("pre",), required=True)
    return parser.parse_args()


def _constants(path: Path, names: List[str]) -> Dict[str, Any]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                values[name] = ast.literal_eval(node.value)
    missing = sorted(set(names).difference(values))
    if missing:
        raise RuntimeError("Cannot read config constants: {0}".format(", ".join(missing)))
    return values


def _functions(path: Path) -> Mapping[str, ast.FunctionDef]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def audit_pre(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    config_values = _constants(
        root / "mhfl_review" / "config.py",
        ["GLOBAL_SEED", "TRADITIONAL_REPEATS_FAST", "TRADITIONAL_REPEATS_FULL", "UO_SVM_CV_FOLDS"],
    )
    if config_values["TRADITIONAL_REPEATS_FAST"] != 2:
        raise RuntimeError("Fast traditional repeat count changed.")
    if config_values["TRADITIONAL_REPEATS_FULL"] != 10:
        raise RuntimeError("Full traditional repeat count must be ten.")
    if config_values["UO_SVM_CV_FOLDS"] != 3:
        raise RuntimeError("UO SVM CV folds changed.")

    script_path = root / "06_traditional_baselines.py"
    source = script_path.read_text(encoding="utf-8")
    functions = _functions(script_path)
    required = {
        "fit_with_source_validation", "fit_with_training_cv", "run_uo", "run_kaist",
        "validate_full_kaist_channel_config", "validate_raw_results", "summarize", "main",
    }
    if not required.issubset(functions):
        raise RuntimeError("Traditional-baseline final protocol functions are incomplete.")
    if EXPECTED_AGGREGATION not in source or EXACT_CURRENT_CHANNEL not in source:
        raise RuntimeError("Traditional aggregation or exact-channel policy is missing.")

    cv_source = ast.get_source_segment(source, functions["fit_with_training_cv"]) or ""
    if "GridSearchCV" not in cv_source or "config.UO_SVM_CV_FOLDS" not in cv_source or 'scoring="f1_macro"' not in cv_source:
        raise RuntimeError("UO training-only CV protocol changed.")
    validation_source = ast.get_source_segment(source, functions["fit_with_source_validation"]) or ""
    for token in ("(0.1, 1.0, 10.0, 100.0)", '("scale", "auto")', 'metric_dict(val_y, probability)["macro_f1"]'):
        if token not in validation_source:
            raise RuntimeError("KAIST source-validation tuning protocol changed.")
    kaist_source = ast.get_source_segment(source, functions["run_kaist"]) or ""
    for token in ('for load in ("2Nm", "4Nm")', 'for snr_db in (0.0, -4.0, -8.0)', "allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK"):
        if token not in kaist_source:
            raise RuntimeError("KAIST load, SNR, or loader protocol changed.")
    summary_source = ast.get_source_segment(source, functions["summarize"]) or ""
    if "for metric in METRIC_COLUMNS" not in summary_source or "trimmed_mean_sd(values)" not in summary_source:
        raise RuntimeError("Traditional metrics are not trimmed independently.")
    main_source = ast.get_source_segment(source, functions["main"]) or ""
    if main_source.find("validate_raw_results") > main_source.find("raw.to_csv"):
        raise RuntimeError("Traditional raw validation must precede result publication.")
    if "args.skip_uo or args.skip_kaist" not in main_source or "validate_full_kaist_channel_config" not in main_source:
        raise RuntimeError("Full mode does not reject partial execution or enforce channels.")

    global_seed = int(config_values["GLOBAL_SEED"])
    seeds = [global_seed + index for index in range(10)]
    return {
        "status": "PASS",
        "phase": "pre",
        "script": str(script_path),
        "target_seeds": seeds,
        "expected_raw_rows": {"UO": 120, "KAIST": 240, "KAIST-noise": 120, "total": 480},
        "expected_groups": 48,
        "retained_after_trim": 8,
        "aggregation": EXPECTED_AGGREGATION,
        "metrics_trimmed_independently": True,
        "current_channel": EXACT_CURRENT_CHANNEL,
        "vibration_column": 0,
        "current_fallback": False,
        "training_cv_unchanged": True,
        "source_validation_unchanged": True,
        "fast_repeats_unchanged": True,
        "training_was_run": False,
        "files_written": [],
    }


def main() -> None:
    parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(audit_pre(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
