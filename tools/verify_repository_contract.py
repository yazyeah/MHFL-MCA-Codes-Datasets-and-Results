#!/usr/bin/env python3
"""Verify the public repository contract without training or raw-data reads."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "MHFL-MCA Supplementary Experiments"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_files() -> List[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(LFS_HEADER)) == LFS_HEADER


class Checks:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.passed = 0

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(message)


def syntax_checks(checks: Checks) -> None:
    python_files = sorted(ROOT.rglob("*.py"))
    json_files = sorted(ROOT.rglob("*.json"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            checks.require(False, "Python parse failed: {0}: {1}".format(path, exc))
    for path in json_files:
        try:
            read_json(path)
        except (ValueError, UnicodeDecodeError) as exc:
            checks.require(False, "JSON parse failed: {0}: {1}".format(path, exc))
    checks.require(not checks.failures, "One or more syntax checks failed.")


def verify_publication_manifest(checks: Checks, require_resolved: bool) -> None:
    manifest_path = SUPPLEMENT / "PUBLICATION_MANIFEST.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    control_names = {
        "PACKAGE_VALIDATION.json",
        "PUBLICATION_MANIFEST.csv",
        "SHA256SUMS.txt",
    }
    expected_count = sum(
        1 for path in SUPPLEMENT.rglob("*")
        if path.is_file() and path.name not in control_names
    )
    checks.require(
        len(rows) == expected_count,
        "Publication manifest has {0} rows; expected {1}.".format(len(rows), expected_count),
    )
    unresolved = []
    mismatches = []
    for row in rows:
        path = SUPPLEMENT / Path(row["path"])
        if not path.is_file():
            mismatches.append(row["path"] + " (missing)")
            continue
        if is_lfs_pointer(path):
            unresolved.append(row["path"])
            continue
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            mismatches.append(row["path"])
    checks.require(not mismatches, "Publication payload mismatch: {0}".format(mismatches[:5]))
    if require_resolved:
        checks.require(not unresolved, "Supplementary LFS payload is unresolved: {0}".format(unresolved[:5]))


def full_checks(checks: Checks, require_resolved: bool) -> None:
    required = (
        "LICENSE",
        "LICENSE_SCOPE.md",
        "DATA.md",
        "DATA_LICENSES.md",
        "THIRD_PARTY_NOTICES.md",
        "ENVIRONMENT.md",
        "HARDWARE.md",
        "MODEL_ZOO.md",
        "REPRODUCIBILITY.md",
        "MHFL-MCA Supplementary Experiments/Provenance/audits/ARTIFACT_VALUE_VERIFICATION.md",
        "MHFL-MCA Supplementary Experiments/Provenance/audits/ARTIFACT_VALUE_VERIFICATION.json",
        "environment.yml",
        "requirements-lock.txt",
        ".github/repository-metadata.json",
    )
    for relative in required:
        checks.require((ROOT / relative).is_file(), "Required repository file is missing: " + relative)

    metadata = read_json(ROOT / ".github/repository-metadata.json")
    checks.require(bool(metadata.get("description")), "GitHub description contract is empty.")
    checks.require(len(metadata.get("topics", [])) >= 8, "At least eight repository topics are required.")
    checks.require(
        "social_preview" not in metadata,
        "Repository metadata must not define a social preview.",
    )
    checks.require(
        not (ROOT / "assets" / "social-preview.png").exists(),
        "Social preview asset must not be committed.",
    )

    license_path = ROOT / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8")
    checks.require(
        metadata.get("software_license_status") == "selected",
        "Metadata must mark the software license as selected.",
    )
    checks.require(
        metadata.get("software_license_spdx_id") == "MIT"
        and metadata.get("software_license") == "MIT",
        "Repository metadata must identify the MIT license.",
    )
    for phrase in (
        "MIT License",
        "Permission is hereby granted, free of charge",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    ):
        checks.require(phrase in license_text, "LICENSE is missing MIT text: " + phrase)

    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    for pin in (
        "tensorflow==2.7.0",
        "keras==2.7.0",
        "numpy==1.21.0",
        "scipy==1.8.1",
        "pandas==1.4.4",
        "scikit-learn==1.1.3",
        "nptdms==1.10.0",
    ):
        checks.require(pin in lock, "Missing recorded environment pin: " + pin)

    data_license = (ROOT / "DATA_LICENSES.md").read_text(encoding="utf-8")
    checks.require("UNCONFIRMED" in data_license, "UO exact-version limitation is missing.")
    checks.require("10.17632/ztmf3m7h5x.6" in data_license, "KAIST v6 DOI is missing.")
    checks.require("CC BY 4.0" in data_license, "Dataset CC BY 4.0 attribution is missing.")

    verification = read_json(
        SUPPLEMENT / "Provenance" / "audits" / "ARTIFACT_VALUE_VERIFICATION.json"
    )
    checks.require(
        verification.get("scope_status") == "VERIFIED_FOR_DECLARED_SCOPE",
        "Artifact verification scope is not authorized.",
    )
    totals = verification.get("totals", {})
    checks.require(
        totals.get("failed_checks") == 0 and totals.get("numeric_mismatches") == 0,
        "Artifact verification reports failed checks or numeric mismatches.",
    )

    package_report = read_json(SUPPLEMENT / "PACKAGE_VALIDATION.json")
    checks.require(package_report.get("status") == "PASS", "Publication package validation report is not PASS.")
    checks.require(package_report.get("training_was_run") is False, "Package validation must not run training.")

    tracked = tracked_files()
    dataset_files = [path for path in tracked if path.startswith("Datasets/")]
    checks.require(len(dataset_files) == 210, "Expected 210 tracked dataset files, found {0}.".format(len(dataset_files)))
    forbidden_weights = [
        path for path in tracked
        if path.lower().endswith(
            (".weights.h5", ".h5", ".keras", ".ckpt", ".onnx", ".pb", ".pt", ".pth")
        )
    ]
    checks.require(not forbidden_weights, "Model weights must not be committed: {0}".format(forbidden_weights[:5]))

    readme = (ROOT / "Readme.md").read_text(encoding="utf-8")
    checks.require("For each sample $i$, MHFL-MCA takes" in readme, "README still uses the legacy model name in the problem setup.")
    checks.require("The paper evaluates MHFL-MCA on two public datasets." in readme, "README dataset scope uses the wrong model name.")
    checks.require("respository" not in readme, "README contains the 'respository' typo.")

    verify_publication_manifest(checks, require_resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax-only", action="store_true")
    parser.add_argument("--require-supplementary-lfs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = Checks()
    syntax_checks(checks)
    if not args.syntax_only:
        full_checks(checks, args.require_supplementary_lfs)
    if checks.failures:
        print("Repository contract: FAIL")
        for failure in checks.failures:
            print("- " + failure)
        raise SystemExit(1)
    print("Repository contract: PASS ({0} checks)".format(checks.passed))


if __name__ == "__main__":
    main()
