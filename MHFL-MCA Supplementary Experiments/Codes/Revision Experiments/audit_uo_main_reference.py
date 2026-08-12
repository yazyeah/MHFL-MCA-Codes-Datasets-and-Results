"""Read-only UO Optuna historical-evidence audit.

Purpose: recover the exact seven-parameter winner and study metadata from retained
artifacts. It does not run Optuna, train a model, or infer parameters from table values.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

REQUIRED_KEYS = (
    "dropout_vib",
    "dropout_aco",
    "atten_dim",
    "n_layers_vib",
    "n_layers_aco",
    "lr",
    "batch_size",
)

PAPER_ANCHORS = {
    5: {
        "accuracy_mean": 0.9279,
        "accuracy_sd": 0.0288,
        "macro_precision_mean": 0.9361,
        "macro_precision_sd": 0.0237,
        "macro_f1_mean": 0.9266,
        "macro_f1_sd": 0.0299,
    },
    10: {
        "accuracy_mean": 0.9707,
        "accuracy_sd": 0.0175,
        "macro_precision_mean": 0.9723,
        "macro_precision_sd": 0.0160,
        "macro_f1_mean": 0.9704,
        "macro_f1_sd": 0.0178,
    },
}

TEXT_SUFFIXES = {".txt", ".log", ".json", ".md", ".py", ".csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the original UO Optuna configuration and verify the manuscript "
            "Table-5 anchors before rerunning Experiment 05."
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(__file__).resolve().parent / "provenance" / "original_optuna",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional explicit path to the original Final_Summary_Stats.csv.",
    )
    parser.add_argument(
        "--source-program",
        type=Path,
        default=None,
        help="Optional explicit path to the original MHCNN_Oputuna_Experiment.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/uo_optuna_confirmed.json"),
    )
    return parser.parse_args()


def iter_text_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 20 * 1024 * 1024:
            files.append(path)
    return files


def normalize_params(value: Mapping[str, object]) -> Optional[Dict[str, object]]:
    aliases = {
        "dropout_vib": ("dropout_vib", "vib_dropout"),
        "dropout_aco": ("dropout_aco", "dropout_acoustic", "other_dropout"),
        "atten_dim": ("atten_dim", "attention_dim"),
        "n_layers_vib": ("n_layers_vib", "vib_layers"),
        "n_layers_aco": ("n_layers_aco", "n_layers_acoustic", "other_layers"),
        "lr": ("lr", "learning_rate"),
        "batch_size": ("batch_size",),
    }
    result: Dict[str, object] = {}
    for target, names in aliases.items():
        found = None
        for name in names:
            if name in value:
                found = value[name]
                break
        if found is None:
            return None
        result[target] = found
    try:
        return {
            "dropout_vib": float(result["dropout_vib"]),
            "dropout_aco": float(result["dropout_aco"]),
            "atten_dim": int(result["atten_dim"]),
            "n_layers_vib": int(result["n_layers_vib"]),
            "n_layers_aco": int(result["n_layers_aco"]),
            "lr": float(result["lr"]),
            "batch_size": int(result["batch_size"]),
        }
    except (TypeError, ValueError):
        return None


def parse_dict_candidates(text: str) -> Iterable[Mapping[str, object]]:
    patterns = (
        r"Best params\s*:\s*(\{[^\n\r]+\})",
        r"best_params\s*[=:]\s*(\{[^\n\r]+\})",
        r"study\.best_params\s*[=:]\s*(\{[^\n\r]+\})",
        r"(?:找到的)?最佳参数\s*:\s*(\{[^\n\r]+\})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(1)
            try:
                payload = ast.literal_eval(raw)
            except Exception:
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
            if isinstance(payload, Mapping):
                yield payload

    # Also inspect complete JSON objects when a retained JSON file stores the parameters directly.
    try:
        payload = json.loads(text)
    except Exception:
        return
    if isinstance(payload, Mapping):
        yield payload
        for value in payload.values():
            if isinstance(value, Mapping):
                yield value


def recover_params(root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    unique: Dict[str, Dict[str, object]] = {}
    evidence: List[Dict[str, object]] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for payload in parse_dict_candidates(text):
            params = normalize_params(payload)
            if params is None:
                continue
            key = json.dumps(params, sort_keys=True, separators=(",", ":"))
            unique[key] = params
            evidence.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "modification_time": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "params": params,
                }
            )
    deduplicated_evidence: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in evidence:
        key = (str(row["path"]), json.dumps(row["params"], sort_keys=True, separators=(",", ":")))
        deduplicated_evidence[key] = row
    return list(unique.values()), list(deduplicated_evidence.values())


def parse_optuna_metadata(path: Path, expected_params: Mapping[str, object]) -> Dict[str, object]:
    """Extract retained study metadata without treating trial candidates as winners."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    study_match = re.search(r"study created in memory with name:\s*([^\s]+)", text, flags=re.IGNORECASE)
    trials = [int(value) for value in re.findall(r"\bTrial\s+(\d+)\s+(?:finished|pruned)\b", text)]
    completed_trials = len(re.findall(r"\bTrial\s+\d+\s+finished\b", text))
    pruned_trials = len(re.findall(r"\bTrial\s+\d+\s+pruned\b", text))
    best_rows = list(
        re.finditer(
            r"Trial\s+(\d+)\s+finished with value:\s*([0-9.eE+-]+)\s+and parameters:\s*(\{[^\n\r]+?\})\.\s+Best is trial\s+(\d+)\s+with value:\s*([0-9.eE+-]+)",
            text,
        )
    )
    matching_rows = []
    for match in best_rows:
        try:
            params = normalize_params(ast.literal_eval(match.group(3)))
        except Exception:
            params = None
        if params == dict(expected_params):
            matching_rows.append(match)
    if not matching_rows:
        raise RuntimeError("The explicit winning UO parameter set is not tied to a retained Optuna trial row.")
    winner = matching_rows[-1]
    best_trial_number = int(winner.group(1))
    best_value = float(winner.group(2))
    if int(winner.group(4)) != best_trial_number or float(winner.group(5).rstrip(".")) != best_value:
        raise RuntimeError("The retained UO log has an inconsistent best-trial declaration.")
    return {
        "study_name": None if study_match is None else study_match.group(1),
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "number_of_trials": 0 if not trials else max(trials) + 1,
        "completed_trials": completed_trials,
        "pruned_trials": pruned_trials,
        "storage": "in-memory (no persistent Optuna database declared in the retained run log)",
        "sampler": "Optuna default sampler (not explicitly printed)",
        "pruner": "MedianPruner",
    }


def locate_summary(root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    candidates = list(root.rglob("Final_Summary_Stats.csv")) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    return None


def locate_source_program(root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit.resolve() if explicit.is_file() else None
    candidates = (
        root / "MHCNN_Oputuna_Experiment.py",
        root.parent / "机械故障诊断实验" / "MHCNN_Oputuna_Experiment.py",
        root.parent / "MHCNN_Oputuna_Experiment.py",
    )
    existing = {path.resolve() for path in candidates if path.is_file()}
    return next(iter(existing)) if len(existing) == 1 else None


def verify_source_program(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="strict")
    checks = {
        "dropout_vib_search": 'suggest_float("dropout_vib", 0.1, 0.5)' in text,
        "dropout_aco_search": 'suggest_float("dropout_aco", 0.1, 0.5)' in text,
        "attention_search": 'suggest_categorical("atten_dim", [128, 256])' in text,
        "vibration_depth_search": 'suggest_int("n_layers_vib", 3, 5)' in text,
        "acoustic_depth_search": 'suggest_int("n_layers_aco", 3, 5)' in text,
        "learning_rate_search": 'suggest_float("lr", 1e-4, 1e-2, log=True)' in text,
        "batch_size_search": 'suggest_categorical("batch_size", [16, 32, 64])' in text,
        "thirty_trials": bool(
            re.search(r"SEARCH_TRIALS\s*=\s*30", text)
            and re.search(r"study\.optimize\(objective,\s*n_trials=SEARCH_TRIALS\)", text)
        ),
        "median_pruner": "optuna.pruners.MedianPruner()" in text,
        "paper_seed_schedule": bool(
            re.search(r"seed\s*=\s*num_samples\s*\*\s*100\s*\+\s*run_idx", text)
        ),
        "adamax": "tf.keras.optimizers.Adamax(learning_rate=best_lr)" in text,
        "eighty_epochs": bool(re.search(r"model\.fit\([^\n]*", text)) and "epochs=80" in text,
        "heldout_as_validation": "validation_data=([x_te_v, x_te_a], y_te)" in text,
        "uniform_cross_attention_initializer": text.count("initializer='uniform'") >= 3,
        "metricwise_drop_one_high_low": "trimmed_values = values[1:-1]" in text,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "The original UO source program failed protocol checks: {0}.".format(", ".join(failed))
        )
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
        "modification_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "protocol_checks": checks,
        "random_initialization_note": (
            "The retained source resets NumPy for data splitting but does not reset Python/TensorFlow "
            "model-initialization RNG state before each Stage-2 run. The exact historical initializer "
            "stream cannot be reconstructed without its process RNG state or checkpoints."
        ),
    }


def verify_summary(path: Path) -> Dict[str, object]:
    frame = pd.read_csv(path)
    n_column = next((name for name in ("Samples", "n_train", "N") if name in frame.columns), None)
    if n_column is None:
        raise RuntimeError("The original UO summary has no recognized N column.")
    metric_map = {
        "accuracy_mean": ("Acc_Mean", "accuracy_mean"),
        "accuracy_sd": ("Acc_Std", "accuracy_sd"),
        "macro_precision_mean": ("Prec_Mean", "macro_precision_mean"),
        "macro_precision_sd": ("Prec_Std", "macro_precision_sd"),
        "macro_f1_mean": ("F1_Mean", "macro_f1_mean"),
        "macro_f1_sd": ("F1_Std", "macro_f1_sd"),
    }
    resolved: Dict[str, str] = {}
    for target, names in metric_map.items():
        column = next((name for name in names if name in frame.columns), None)
        if column is None:
            raise RuntimeError("The original UO summary is missing column for {0}.".format(target))
        resolved[target] = column

    checks: Dict[str, object] = {}
    all_passed = True
    for n_value, expected in PAPER_ANCHORS.items():
        rows = frame[pd.to_numeric(frame[n_column], errors="coerce") == int(n_value)]
        if len(rows) != 1:
            raise RuntimeError("Expected exactly one original UO summary row for N={0}.".format(n_value))
        row = rows.iloc[0]
        actual = {key: float(row[column]) for key, column in resolved.items()}
        passed = all(round(actual[key], 4) == round(float(expected[key]), 4) for key in expected)
        all_passed = all_passed and passed
        checks[str(n_value)] = {"expected": expected, "actual": actual, "passed_to_4dp": passed}
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "modification_time": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "anchor_checks": checks,
        "anchors_passed": all_passed,
    }


def main() -> None:
    args = parse_args()
    root = args.search_root.resolve()
    parameter_sets, evidence = recover_params(root)
    summary_path = locate_summary(root, args.summary_csv)
    source_program_path = locate_source_program(root, args.source_program)

    report: Dict[str, object] = {
        "search_root": str(root),
        "parameter_candidate_count": len(parameter_sets),
        "parameter_candidates": parameter_sets,
        "parameter_evidence": evidence,
        "paper_anchors": PAPER_ANCHORS,
    }

    if source_program_path is None:
        report["status"] = "BLOCKED"
        report["reason"] = "A unique original MHCNN_Oputuna_Experiment.py was not found."
    elif summary_path is None:
        report["status"] = "BLOCKED"
        report["reason"] = "A unique original Final_Summary_Stats.csv was not found."
    else:
        report["source_program_evidence"] = verify_source_program(source_program_path)
        report["summary_evidence"] = verify_summary(summary_path)
        if not report["summary_evidence"]["anchors_passed"]:  # type: ignore[index]
            report["status"] = "BLOCKED"
            report["reason"] = "The located summary does not reproduce the current Table-5 anchors."
        elif len(parameter_sets) != 1:
            report["status"] = "BLOCKED"
            report["reason"] = (
                "Exactly one explicit seven-parameter Optuna configuration is required; "
                "found {0}.".format(len(parameter_sets))
            )
        else:
            report.update(parameter_sets[0])
            winning_evidence = [row for row in evidence if row.get("params") == parameter_sets[0]]
            if len(winning_evidence) != 1:
                report["status"] = "BLOCKED"
                report["reason"] = "Exactly one retained artifact must explicitly print the winning UO parameter set."
            else:
                evidence_path = Path(str(winning_evidence[0]["path"]))
                report.update(parse_optuna_metadata(evidence_path, parameter_sets[0]))
                report["evidence_path"] = str(evidence_path.resolve())
                report["evidence_sha256"] = str(winning_evidence[0]["sha256"])
                report["audit_generated_at"] = datetime.now(timezone.utc).isoformat()
                report["confirmation_status"] = "confirmed_from_original_uo_artifact"
                report["evidence_level"] = "B_original_run_log_explicit_best_params"
                report["status"] = "CONFIRMED"
                report["source_note"] = (
                    "Recovered from the original UO Optuna console log's explicit final best-parameter line; "
                    "cross-checked against its retained winning trial and the original Final_Summary_Stats.csv/Table-5 anchors."
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if report.get("status") == "CONFIRMED":
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    else:
        blocked_path = args.output.with_name(args.output.stem + ".blocked_audit.json")
        blocked_path.write_text(serialized, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nWritten to:", (args.output if report.get("status") == "CONFIRMED" else blocked_path).resolve())
    if report.get("status") != "CONFIRMED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
