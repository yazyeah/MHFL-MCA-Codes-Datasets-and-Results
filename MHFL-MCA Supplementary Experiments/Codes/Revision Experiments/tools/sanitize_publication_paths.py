#!/usr/bin/env python3
"""Sanitize machine-specific paths in a publication-copy directory.

Purpose
-------
Replace local Windows roots in copied JSON, CSV, Markdown, and text metadata
with documented environment placeholders. Numeric results, hashes of source
artifacts, and the byte-identical historical Optuna evidence are not changed.

Usage
-----
python tools/sanitize_publication_paths.py --package-root <upload-package>

The script is idempotent. It intentionally skips ``original_optuna`` and
``source_snapshots`` because those folders are immutable historical evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Tuple


TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
SKIP_PARTS = {"original_optuna", "source_snapshots"}


def _replacement_pairs() -> Tuple[Tuple[str, str], ...]:
    """Return user-supplied local-root replacements, longest paths first."""
    variables = (
        ("MHFL_ORIGINAL_SUITE_ROOT", "${MHFL_SUITE_ROOT}"),
        ("MHFL_ORIGINAL_UO_DIR", "${MHFL_UO_DIR}"),
        ("MHFL_ORIGINAL_KAIST_VIB_DIR", "${MHFL_KAIST_VIB_DIR}"),
        ("MHFL_ORIGINAL_KAIST_CURRENT_DIR", "${MHFL_KAIST_CURRENT_DIR}"),
        ("MHFL_ORIGINAL_MANUSCRIPT_SOURCE_DIR", "${MHFL_MANUSCRIPT_SOURCE_DIR}"),
        ("MHFL_ORIGINAL_PROJECT_ROOT", "${MHFL_PROJECT_ROOT}"),
        ("MHFL_ORIGINAL_DATA_ROOT", "${MHFL_DATA_ROOT}"),
        ("MHFL_ORIGINAL_TEMP_ROOT", "${MHFL_TEMP_ROOT}"),
    )
    pairs = tuple(
        (os.environ[name], replacement)
        for name, replacement in variables
        if os.environ.get(name)
    )
    expanded = []
    for source, replacement in pairs:
        expanded.append((source, replacement))
        expanded.append((source.replace("\\", "/"), replacement))
    return tuple(expanded)


def sanitize_text(value: str) -> str:
    """Replace known local roots and normalize placeholder paths."""
    updated = value
    for source, replacement in _replacement_pairs():
        updated = updated.replace(source, replacement)
    if "${MHFL_" in updated:
        updated = updated.replace("\\", "/")
    return updated


def sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize JSON string values without changing numbers."""
    if isinstance(value, dict):
        sanitized = {key: sanitize_json_value(item) for key, item in value.items()}
        if "source_tex_path" in sanitized:
            sanitized["source_tex_path"] = (
                "provenance/main_manuscript_sources/MHFL-MCA_20260810.tex"
            )
        return sanitized
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def iter_targets(package_root: Path) -> Iterable[Path]:
    """Yield publication metadata while excluding immutable evidence."""
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if SKIP_PARTS.intersection(path.parts):
            continue
        yield path


def sanitize_file(path: Path) -> bool:
    """Sanitize one file and return whether its content changed."""
    original = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(original)
        updated = json.dumps(
            sanitize_json_value(payload),
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    else:
        updated = sanitize_text(original)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    changed = [path for path in iter_targets(package_root) if sanitize_file(path)]
    print("Sanitized publication metadata files: {0}".format(len(changed)))


if __name__ == "__main__":
    main()
