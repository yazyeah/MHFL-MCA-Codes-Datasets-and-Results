"""Experiment 00 - environment and data-channel preflight.

Purpose: validate paths, dependencies, confirmed model specifications, and the
exact KAIST vibration/current channels before an experiment is started.
Inputs: environment variables documented in ``windows/00_set_environment.example.bat``.
Outputs: a provenance preflight report; this entry point never trains a model.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

from mhfl_review import config
from mhfl_review.data import inspect_kaist_vibration_matrix, inspect_tdms_channels
from mhfl_review.provenance import environment_manifest, write_json
from mhfl_review.specs import candidate_architectures, manuscript_spec, parameter_count_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight paths, runtime, data channels, and manuscript model specs.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on every unresolved full-run requirement.")
    return parser.parse_args()


def check_path(path: Path, label: str, report: Dict[str, Any]) -> bool:
    ok = path.exists()
    print("[{0}] {1}: {2}".format("OK" if ok else "MISSING", label, path))
    report.setdefault("paths", {})[label] = {"path": str(path), "exists": bool(ok)}
    return ok


def first_tdms(directory: Path) -> Path:
    candidates = sorted(list(directory.glob("*.tdms")) + list(directory.glob("*.TDMS")))
    if not candidates:
        raise FileNotFoundError("No TDMS file found under {0}.".format(directory))
    return candidates[0]


def first_mat(directory: Path) -> Path:
    candidates = sorted(list(directory.glob("*.mat")) + list(directory.glob("*.MAT")))
    if not candidates:
        raise FileNotFoundError("No MAT file found under {0}.".format(directory))
    return candidates[0]


def main() -> None:
    args = parse_args()
    config.ensure_runtime_dirs()
    report: Dict[str, Any] = {"suite_version": config.SUITE_VERSION, "python": sys.version}
    print("=" * 88)
    print("MHFL-MCA reviewer suite preflight")
    print("=" * 88)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print(config.describe_paths())
    print("-" * 88)

    kaist_training_ok = True
    kaist_training_error = None
    try:
        kaist_learning_rate, kaist_batch_size = config.require_confirmed_kaist_training_config("full")
    except RuntimeError as exc:
        kaist_training_ok = False
        kaist_training_error = str(exc)
        kaist_learning_rate = config.KAIST_MANUSCRIPT_LEARNING_RATE
        kaist_batch_size = config.KAIST_MANUSCRIPT_BATCH_SIZE
        print("[FAIL] KAIST confirmed training configuration:", exc)
    print("[KAIST CONFIG] confirmed learning rate:", kaist_learning_rate)
    print("[KAIST CONFIG] confirmed batch size:", kaist_batch_size)
    print("[KAIST CONFIG] evidence status:", config.KAIST_OPTUNA_CONFIRMATION_STATUS)
    report["kaist_training_config"] = {
        "path": str(config.KAIST_OPTUNA_CONFIRMED_PATH),
        "confirmation_status": config.KAIST_OPTUNA_CONFIRMATION_STATUS,
        "learning_rate": kaist_learning_rate,
        "batch_size": kaist_batch_size,
        "loaded": kaist_training_ok,
        "error": kaist_training_error,
    }
    print("-" * 88)

    ok = sys.version_info >= (3, 9)
    if not ok:
        print("[FAIL] Python 3.9 or newer is required.")
    ok &= check_path(config.PROJECT_ROOT, "Project root", report)
    ok &= check_path(config.UO_DATA_ROOT, "UO MATLAB data root", report)
    ok &= check_path(config.KAIST_VIB_DIR, "KAIST vibration directory", report)
    ok &= check_path(config.KAIST_CURRENT_DIR, "KAIST current directory", report)

    dependencies: Dict[str, Any] = {}
    for package_name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "nptdms"):
        try:
            module = __import__(package_name)
            dependencies[package_name] = getattr(module, "__version__", "installed")
            print("[OK] {0}: {1}".format(package_name, dependencies[package_name]))
        except Exception as exc:
            dependencies[package_name] = None
            ok = False
            print("[MISSING] {0}: {1}".format(package_name, exc))
    try:
        import tensorflow as tf

        dependencies["tensorflow"] = tf.__version__
        dependencies["gpus"] = [str(item) for item in tf.config.list_physical_devices("GPU")]
        print("[OK] TensorFlow:", tf.__version__)
        print("[INFO] GPUs:", dependencies["gpus"])
        if str(tf.__version__).split(".")[:2] != ["2", "7"]:
            print("[WARN] Paper code reports TensorFlow 2.7.0; use that environment for final timings.")
    except Exception as exc:
        dependencies["tensorflow"] = None
        dependencies["gpus"] = []
        ok = False
        print("[MISSING] tensorflow:", exc)
    report["dependencies"] = dependencies

    specs = {}
    for case in ("uo", "kaist"):
        spec = manuscript_spec(case)
        specs[case] = {"spec": spec.to_dict(), "analytical_params_m": parameter_count_m(spec)}
        print("[SPEC] {0}: {1:.6f} M | {2}".format(case.upper(), parameter_count_m(spec), spec.spec_status))
    specs["kaist_candidates_near_7_380M"] = candidate_architectures(7.380, tolerance_m=0.02)
    report["model_specs"] = specs

    channel_ok = True
    if config.KAIST_CURRENT_DIR.is_dir():
        try:
            tdms_path = first_tdms(config.KAIST_CURRENT_DIR)
            channels = inspect_tdms_channels(tdms_path)
            write_json(config.PROVENANCE_ROOT / "channel_manifest.json", {"file": str(tdms_path), "channels": channels})
            eligible = [row for row in channels if row["eligible"]]
            print("[INFO] TDMS channel manifest:", config.PROVENANCE_ROOT / "channel_manifest.json")
            for row in eligible:
                print("       {0}/{1} | len={2} | var={3:.6g}".format(row["group"], row["channel"], row["length"], row["variance"]))
            if not config.CURRENT_CHANNEL_NAME and not config.CURRENT_CHANNEL_REGEX:
                print(
                    "[ACTION] Confirm the U-phase channel. Then set MHFL_CURRENT_CHANNEL_NAME or "
                    "MHFL_CURRENT_CHANNEL_REGEX. Full runs intentionally reject max-variance fallback."
                )
                channel_ok = False
        except Exception as exc:
            print("[CHANNEL ERROR]", exc)
            channel_ok = False
    report["current_channel_explicit"] = bool(config.CURRENT_CHANNEL_NAME or config.CURRENT_CHANNEL_REGEX)

    vibration_ok = True
    if config.KAIST_VIB_DIR.is_dir():
        try:
            vibration_path = first_mat(config.KAIST_VIB_DIR)
            vibration_manifest = inspect_kaist_vibration_matrix(vibration_path)
            write_json(config.PROVENANCE_ROOT / "vibration_manifest.json", vibration_manifest)
            print("[INFO] KAIST vibration manifest:", config.PROVENANCE_ROOT / "vibration_manifest.json")
            print("       file={0} shape={1} configured_column={2}".format(
                vibration_manifest.get("file"), vibration_manifest.get("shape"), config.KAIST_VIB_COLUMN
            ))
            if not vibration_manifest.get("configured_column_available", False):
                print("[ACTION] Set MHFL_KAIST_VIB_COLUMN to the verified housing-A x-direction channel.")
                vibration_ok = False
        except Exception as exc:
            print("[VIBRATION CHANNEL ERROR]", exc)
            vibration_ok = False
    report["vibration_channel_valid"] = vibration_ok
    report["environment"] = environment_manifest(config.PROJECT_ROOT)
    write_json(config.PROVENANCE_ROOT / "preflight_report.json", report)

    print("-" * 88)
    print("Preflight report:", config.PROVENANCE_ROOT / "preflight_report.json")
    print("KAIST spec acceptance:", "accepted" if config.ACCEPT_KAIST_SPEC else "not yet accepted")
    if args.strict and (
        not ok
        or not channel_ok
        or not vibration_ok
        or not config.ACCEPT_KAIST_SPEC
        or not kaist_training_ok
    ):
        raise SystemExit("Strict preflight failed. Resolve the actions above before a full run.")
    if not ok:
        raise SystemExit("Environment check failed. Correct missing paths/dependencies before experiments.")
    print("Basic environment checks passed. Full-run gates may still require channel/spec confirmation.")


if __name__ == "__main__":
    main()
