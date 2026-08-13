#!/usr/bin/env python3
"""Validate the compact GitHub publication package without running training.

The validator checks syntax, JSON readability, source-data row contracts,
identity uniqueness, declared source mappings, prohibited binary extensions,
portable metadata paths, and immutable source hashes. With
``--write-report`` it creates the release manifest, checksum list, and a JSON
validation report at the package root.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CONTROL_FILES = {
    "PACKAGE_VALIDATION.json",
    "PUBLICATION_MANIFEST.csv",
    "SHA256SUMS.txt",
}
FORBIDDEN_SUFFIXES = {
    ".h5", ".keras", ".mat", ".tdms", ".tif", ".tiff", ".zip",
    ".npz", ".pb", ".pkl", ".pickle", ".joblib", ".sqlite", ".sqlite3",
}
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def category(relative_path: str) -> str:
    if relative_path.startswith("Codes/Revision Experiments/provenance/original_optuna/"):
        return "immutable_historical_evidence"
    if relative_path.startswith("Codes/"):
        return "code_and_configuration"
    if "/05_Extreme_Lowshot_CAIM/controlled_extension/" in relative_path:
        return "transparent_controlled_extension"
    if "/05_Extreme_Lowshot_CAIM/paper_protocol_aware_bundle/" in relative_path:
        return "paper_hybrid_derivative"
    if "/07_Manuscript_Assets/" in relative_path:
        return "status_only"
    if relative_path.startswith("Results/"):
        return "result_and_source_data"
    if relative_path.startswith("Provenance/"):
        return "publication_provenance"
    return "documentation"


class Validation:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def assert_unique(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> bool:
    identities = [tuple(row.get(key, "") for key in keys) for row in rows]
    return len(identities) == len(set(identities))


def numeric_finite(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> bool:
    try:
        return all(
            math.isfinite(float(row[column]))
            for row in rows
            for column in columns
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_python_and_json(root: Path, result: Validation) -> None:
    python_files = sorted(root.rglob("*.py"))
    syntax_errors = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    result.check("python_ast", not syntax_errors, {"files": len(python_files), "errors": syntax_errors})

    json_files = sorted(root.rglob("*.json"))
    json_errors = []
    for path in json_files:
        if path.name == "PACKAGE_VALIDATION.json":
            continue
        try:
            read_json(path)
        except (ValueError, UnicodeDecodeError) as exc:
            json_errors.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    result.check("json_parse", not json_errors, {"files": len(json_files), "errors": json_errors})


def validate_paths_and_payload(root: Path, result: Validation) -> None:
    forbidden = []
    large = []
    absolute_hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative)
        if path.stat().st_size > 10 * 1024 * 1024:
            large.append({"path": relative, "size_bytes": path.stat().st_size})
        if path.suffix.lower() not in {".csv", ".json", ".md", ".tex", ".txt", ".bat"}:
            continue
        if "provenance/original_optuna/" in relative or "source_snapshots/" in relative:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ABSOLUTE_WINDOWS_PATH.search(text) or ("C:" + "/Users/") in text:
            absolute_hits.append(relative)
    result.check("forbidden_extensions", not forbidden, forbidden)
    result.check("files_over_10_mb", not large, large)
    result.check("portable_metadata_paths", not absolute_hits, absolute_hits)


def validate_result_contracts(root: Path, result: Validation) -> None:
    base = root / "Results" / "Revision Experiments" / "full_20260807_r1"
    contracts: Tuple[Tuple[str, Path, int], ...] = (
        ("02_modality_weight_raw", base / "02_Modality_Weights_and_Missing" / "modality_weight_raw.csv", 540),
        ("02_modality_weight_summary", base / "02_Modality_Weights_and_Missing" / "modality_weight_summary.csv", 36),
        ("02_missing_raw", base / "02_Modality_Weights_and_Missing" / "missing_modality_raw.csv", 132),
        ("02_missing_summary", base / "02_Modality_Weights_and_Missing" / "missing_modality_summary.csv", 12),
        ("04_raw", base / "04_Additional_Controlled_Ablation" / "additional_ablation_variant_raw.csv", 150),
        ("04_summary", base / "04_Additional_Controlled_Ablation" / "additional_ablation_variant_summary.csv", 15),
        ("04_paper_table", base / "04_Additional_Controlled_Ablation" / "paper_current_table_15" / "table_15_additional_ablation.csv", 4),
        ("05_raw", base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_raw.csv", 140),
        ("05_summary", base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_summary.csv", 14),
        ("05_paired", base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "caim_paired_summary.csv", 7),
        ("06_raw", base / "06_Traditional_Baselines" / "traditional_baselines_raw.csv", 480),
        ("06_summary", base / "06_Traditional_Baselines" / "traditional_baselines_summary.csv", 48),
        ("06_candidate", base / "06_Traditional_Baselines" / "manuscript_candidate_rows.csv", 144),
    )
    counts = {}
    rows_by_name = {}
    for name, path, expected in contracts:
        rows = read_csv(path)
        rows_by_name[name] = rows
        counts[name] = len(rows)
        result.check(name + "_row_count", len(rows) == expected, {"expected": expected, "actual": len(rows)})

    result.check("04_raw_unique", assert_unique(rows_by_name["04_raw"], ("variant", "condition", "seed")), "variant/condition/seed")
    result.check("05_raw_unique", assert_unique(rows_by_name["05_raw"], ("variant", "n_train", "seed")), "variant/N/seed")
    result.check("06_raw_unique", assert_unique(rows_by_name["06_raw"], ("case", "load", "n_train", "snr_db", "method", "seed")), "case/load/N/SNR/method/seed")

    result.check("04_metrics_finite", numeric_finite(rows_by_name["04_raw"], ("accuracy", "macro_f1", "macro_precision", "macro_recall")), "four metrics")
    result.check("05_metrics_finite", numeric_finite(rows_by_name["05_raw"], ("test_accuracy", "test_macro_f1", "test_macro_precision", "test_macro_recall", "train_accuracy", "heldout_accuracy", "generalization_gap")), "seven metrics")
    result.check("06_metrics_finite", numeric_finite(rows_by_name["06_raw"], ("accuracy", "macro_f1", "macro_precision", "macro_recall")), "four metrics")

    post04 = read_json(base / "04_Additional_Controlled_Ablation" / "additional_ablation_post_gate.json")
    anchor05 = read_json(base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_anchor_gate.json")
    post05 = read_json(base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_post_gate.json")
    manifest05 = read_json(base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_run_manifest.json")
    protocol06 = read_json(base / "06_Traditional_Baselines" / "traditional_baseline_protocol_validation.json")
    result.check("04_gate_pass", post04.get("status") == "PASS", post04.get("status"))
    result.check(
        "05_protocol_boundary_recorded",
        anchor05.get("status") == "FAIL"
        and post05.get("status") == "FAIL"
        and post05.get("final_outputs_authorized") is False
        and manifest05.get("status") == "BLOCKED",
        "Historical-replay status is retained separately from the controlled extension.",
    )
    result.check("06_protocol_pass", protocol06.get("status") == "PASS", protocol06.get("status"))

    table15 = rows_by_name["04_paper_table"]
    table15_by_variant = {row["source_variant"]: row for row in table15}
    expected_table15 = {
        "reused_stage2_full": (7.380, 0.9999, 0.0002, 0.9999, 0.0002, 0.9925, 0.0096, 0.9925, 0.0097),
        "homogeneous_other": (8.784, 0.9445, 0.0418, 0.9405, 0.0477, 0.9261, 0.0481, 0.9246, 0.0511),
        "attention_dim_128": (4.601, 0.9997, 0.0007, 0.9997, 0.0007, 0.9841, 0.0188, 0.9838, 0.0194),
        "direct_softmax": (7.380, 0.9941, 0.0087, 0.9941, 0.0088, 0.9752, 0.0331, 0.9746, 0.0342),
    }
    table15_columns = (
        "params_m", "2Nm_accuracy_mean", "2Nm_accuracy_sd", "2Nm_macro_f1_mean",
        "2Nm_macro_f1_sd", "4Nm_accuracy_mean", "4Nm_accuracy_sd",
        "4Nm_macro_f1_mean", "4Nm_macro_f1_sd",
    )
    table15_values_ok = set(table15_by_variant) == set(expected_table15)
    if table15_values_ok:
        for variant, expected in expected_table15.items():
            actual = tuple(float(table15_by_variant[variant][column]) for column in table15_columns)
            if actual != expected:
                table15_values_ok = False
                break
    result.check("04_paper_table_values", table15_values_ok, "Full plus three current-paper controls")

    hybrid_path = base / "05_Extreme_Lowshot_CAIM" / "paper_protocol_aware_bundle" / "lowshot_sensitivity_protocol_aware_revised_data.csv"
    hybrid = {int(row["n_train"]): row for row in read_csv(hybrid_path)}
    anchors_ok = (
        set(hybrid) == {1, 2, 3, 4, 5, 7, 10}
        and float(hybrid[5]["sensitivity_full_mean"]) == 0.9279
        and float(hybrid[5]["sensitivity_full_sd"]) == 0.0288
        and hybrid[5]["sensitivity_full_source"] == "main_manuscript_table_5"
        and float(hybrid[10]["sensitivity_full_mean"]) == 0.9707
        and float(hybrid[10]["sensitivity_full_sd"]) == 0.0175
        and hybrid[10]["sensitivity_full_source"] == "main_manuscript_table_5"
        and all(
            hybrid[n]["sensitivity_full_source"] == "controlled_sensitivity_extension"
            for n in (1, 2, 3, 4, 7)
        )
    )
    result.check("05_hybrid_anchor_mapping", anchors_ok, "N=5/10 Table 5; other Full points controlled extension")

    hybrid_manifest_path = base / "05_Extreme_Lowshot_CAIM" / "paper_protocol_aware_bundle" / "hybrid_derivation_manifest.json"
    hybrid_manifest = read_json(hybrid_manifest_path)
    hybrid_hashes_ok = (
        hybrid_manifest.get("display_data_sha256") == sha256_file(hybrid_path)
        and hybrid_manifest.get("controlled_summary_sha256")
        == sha256_file(base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "lowshot_summary.csv")
        and hybrid_manifest.get("controlled_paired_summary_sha256")
        == sha256_file(base / "05_Extreme_Lowshot_CAIM" / "controlled_extension" / "caim_paired_summary.csv")
        and hybrid_manifest.get("figure_pdf_sha256")
        == sha256_file(base / "05_Extreme_Lowshot_CAIM" / "paper_protocol_aware_bundle" / "lowshot_sensitivity_protocol_aware_revised.pdf")
        and hybrid_manifest.get("figure_svg_sha256")
        == sha256_file(base / "05_Extreme_Lowshot_CAIM" / "paper_protocol_aware_bundle" / "lowshot_sensitivity_protocol_aware_revised.svg")
    )
    result.check("05_hybrid_hash_binding", hybrid_hashes_ok, "source summaries, display data, PDF, and SVG")

    thresholds = read_json(
        base
        / "05_Extreme_Lowshot_CAIM"
        / "controlled_extension"
        / "operational_thresholds.json"
    )
    threshold_map = {
        str(item.get("variant")): item for item in thresholds
        if isinstance(item, Mapping)
    }
    threshold_ok = (
        set(threshold_map) == {"full", "no_caim"}
        and threshold_map["full"].get("first_empirical_n") == 3
        and threshold_map["no_caim"].get("first_empirical_n") == 5
        and all(
            item.get("criterion")
            == "trimmed test accuracy >= 0.80 and trimmed seed SD <= 0.10"
            and "not a universal theoretical threshold" in str(item.get("claim_limit", ""))
            for item in threshold_map.values()
        )
    )
    result.check(
        "05_operational_threshold_contract",
        threshold_ok,
        "Full N=3 and no-CAIM N=5 under the predefined protocol-specific criterion",
    )


def validate_reference(root: Path, result: Validation) -> None:
    code = root / "Codes" / "Revision Experiments"
    reference = read_json(code / "configs" / "main_manuscript_deep_reference.json")
    row_count = sum(len(table.get("rows", [])) for table in reference.get("tables", {}).values())
    result.check("deep_reference_rows", row_count == 108, row_count)
    tex = code / "provenance" / "main_manuscript_sources" / "MHFL-MCA_20260810.tex"
    expected = "cd6bc52feb997a7cdc3b56d9e951ed212f15019c25a88ab21edd3b3db4ad83ce"
    actual = sha256_file(tex)
    result.check("recorded_tex_sha256", actual == expected, {"expected": expected, "actual": actual})
    result.check(
        "deep_reference_scope_disclosed",
        reference.get("current_manuscript_source", {}).get("public_release_note")
        == "Portable byte-identical LaTeX snapshot of the declared 2026-08-10 table source.",
        reference.get("current_manuscript_source", {}).get("public_release_note"),
    )

    verification = read_json(
        root / "Provenance" / "audits" / "ARTIFACT_VALUE_VERIFICATION.json"
    )
    totals = verification.get("totals", {})
    checks = verification.get("checks", [])
    result.check(
        "artifact_value_verification",
        verification.get("scope_status") == "VERIFIED_FOR_DECLARED_SCOPE"
        and verification.get("training_was_run") is False
        and len(checks) == 8
        and all(item.get("status") == "PASS" for item in checks)
        and totals.get("checks") == 8
        and totals.get("failed_checks") == 0
        and totals.get("table_rows_checked") == 135
        and totals.get("numeric_fields_checked") == 808
        and totals.get("numeric_mismatches") == 0,
        {
            "scope_status": verification.get("scope_status"),
            "training_was_run": verification.get("training_was_run"),
            "totals": totals,
        },
    )


def payload_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in CONTROL_FILES:
            continue
        yield path


def write_release_indexes(root: Path) -> None:
    records = []
    for path in payload_files(root):
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "category": category(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = root / "PUBLICATION_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "category", "size_bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    checksums = root / "SHA256SUMS.txt"
    with checksums.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "".join(
                "{0}  {1}\n".format(item["sha256"], item["path"])
                for item in records
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.package_root.resolve()
    result = Validation()
    validate_python_and_json(root, result)
    validate_paths_and_payload(root, result)
    validate_result_contracts(root, result)
    validate_reference(root, result)

    if args.write_report:
        write_release_indexes(root)
        report = {
            "status": "PASS" if result.passed else "FAIL",
            "validation_scope": "publication_package_only_no_training",
            "training_was_run": False,
            "checks": result.checks,
        }
        report_path = root / "PACKAGE_VALIDATION.json"
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    failed = [item["name"] for item in result.checks if not item["passed"]]
    print("Publication package validation: {0}".format("PASS" if not failed else "FAIL"))
    print("Checks: {0}; failed: {1}".format(len(result.checks), len(failed)))
    if failed:
        print("Failed checks: {0}".format(", ".join(failed)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
