from __future__ import annotations

"""Build a non-destructive main-text view of Experiment 04.

The frozen Experiment 04 evidence remains a five-variant experiment.  This
utility validates that evidence, archives byte-identical copies with hashes,
and derives a four-control main-display view.  It never trains a model and it
never rewrites the original raw, summary, candidate, manifest, post-gate, or
checkpoint files.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EQUAL_VARIANT = "equal_weights"
MAIN_VARIANTS = (
    "homogeneous_vibration",
    "homogeneous_other",
    "attention_dim_128",
    "direct_softmax",
)
SOURCE_VARIANTS = MAIN_VARIANTS + (EQUAL_VARIANT,)
CONDITIONS = ("2Nm_0dB", "4Nm_0dB", "4Nm_-8dB")
TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
SOURCE_METHODS = (
    "Full MHFL-MCA reference",
    "Homogeneous-vibration",
    "Homogeneous-other",
    "Attention (D=128)",
    "Direct softmax",
    "Equal weights",
)
MAIN_METHODS = SOURCE_METHODS[:-1]

RAW_NAME = "additional_ablation_variant_raw.csv"
SUMMARY_NAME = "additional_ablation_variant_summary.csv"
CANDIDATE_NAME = "manuscript_hybrid_ablation_candidate.csv"
CANDIDATE_TEX_NAME = "manuscript_hybrid_ablation_candidate.tex"
RUN_MANIFEST_NAME = "additional_ablation_run_manifest.json"
POST_GATE_NAME = "additional_ablation_post_gate.json"

MAIN_RAW_NAME = "additional_ablation_variant_raw_main_scope.csv"
MAIN_SUMMARY_NAME = "additional_ablation_variant_summary_main_scope.csv"
MAIN_CANDIDATE_NAME = "manuscript_hybrid_ablation_candidate_main_scope.csv"
MAIN_CANDIDATE_TEX_NAME = "manuscript_hybrid_ablation_candidate_main_scope.tex"
MAIN_NOTE_NAME = "additional_ablation_main_scope_note.txt"
MAIN_MANIFEST_NAME = "ablation_main_scope_without_equal_weights_manifest.json"

ARCHIVE_DIR_NAME = "archive_equal_weights_observed_control"
EQUAL_ARCHIVE_NAME = "equal_weights_exploratory_control.csv"
EQUAL_CANDIDATE_NAME = "equal_weights_candidate_row.csv"

RAW_REQUIRED_COLUMNS = {
    "variant",
    "condition",
    "seed",
    "params_m",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
}
RAW_NUMERIC_COLUMNS = (
    "seed",
    "params_m",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
SUMMARY_REQUIRED_COLUMNS = {
    "variant",
    "condition",
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "params_m",
    "seeds",
    "retained_after_trim",
    "aggregation",
}
SUMMARY_NUMERIC_COLUMNS = (
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "params_m",
    "seeds",
    "retained_after_trim",
    "accuracy_untrimmed_mean",
    "accuracy_untrimmed_sd",
    "macro_f1_untrimmed_mean",
    "macro_f1_untrimmed_sd",
)

MAIN_NOTE = (
    "Main-text scope: homogeneous-vibration, homogeneous-other, attention "
    "D=128, and direct-softmax controls.\n"
    "The equal-weight exploratory control was evaluated and has not been "
    "deleted; its complete observed evidence is retained under "
    "archive_equal_weights_observed_control for supplementary/provenance "
    "reporting.\n"
    "The compact main table therefore omits only the display row; it must "
    "not be stated that equal weighting was not evaluated.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the four-control Experiment 04 main display while "
            "preserving and hashing all five-variant source evidence."
        )
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path(os.environ.get("MHFL_SUITE_ROOT", Path.cwd())),
    )
    parser.add_argument(
        "--run-tag",
        default=os.environ.get("MHFL_RUN_TAG", "full_20260807_r1"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_global_seed(suite_root: Path) -> int:
    """Read GLOBAL_SEED without importing the training package."""

    config_path = suite_root / "mhfl_review" / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError("Configuration file not found: {0}".format(config_path))
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "GLOBAL_SEED" for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("GLOBAL_SEED must be a literal integer.") from exc
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("GLOBAL_SEED must be a literal integer.")
        return int(value)
    raise RuntimeError("GLOBAL_SEED was not found in {0}.".format(config_path))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError("Required frozen Experiment 04 file is missing: {0}".format(path))
    return path


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError("{0} is missing columns: {1}".format(label, ", ".join(missing)))


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    present = [column for column in columns if column in frame.columns]
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("{0} contains a NaN, Inf, or non-numeric metric.".format(label))


def _expected_seeds(global_seed: int) -> set[int]:
    return {global_seed + offset for offset in range(10)}


def validate_source_contract(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    global_seed: int,
) -> None:
    _require_columns(raw, RAW_REQUIRED_COLUMNS, "source raw")
    _require_columns(summary, SUMMARY_REQUIRED_COLUMNS, "source summary")
    _require_columns(candidate, {"Method"}, "source candidate")
    _require_finite(raw, RAW_NUMERIC_COLUMNS, "source raw")
    _require_finite(summary, SUMMARY_NUMERIC_COLUMNS, "source summary")

    seeds = _expected_seeds(global_seed)
    raw_groups = raw.groupby(["variant", "condition"], sort=False)["seed"].agg(
        lambda values: {int(value) for value in values}
    )
    if len(raw) != 150:
        raise RuntimeError("Source raw must contain exactly 150 rows; got {0}.".format(len(raw)))
    if set(raw["variant"].astype(str)) != set(SOURCE_VARIANTS):
        raise RuntimeError("Source raw must contain exactly the five frozen variants.")
    if set(raw["condition"].astype(str)) != set(CONDITIONS):
        raise RuntimeError("Source raw must contain exactly the three frozen conditions.")
    if len(raw_groups) != 15 or any(group_seeds != seeds for group_seeds in raw_groups):
        raise RuntimeError("Source raw must contain 15 groups with the same ten expected seeds.")
    if raw.duplicated(["variant", "condition", "seed"]).any():
        raise RuntimeError("Source raw contains duplicate variant/condition/seed rows.")

    if len(summary) != 15:
        raise RuntimeError("Source summary must contain exactly 15 rows; got {0}.".format(len(summary)))
    if set(summary["variant"].astype(str)) != set(SOURCE_VARIANTS):
        raise RuntimeError("Source summary must contain exactly the five frozen variants.")
    if set(summary["condition"].astype(str)) != set(CONDITIONS):
        raise RuntimeError("Source summary must contain exactly the three frozen conditions.")
    if summary.duplicated(["variant", "condition"]).any():
        raise RuntimeError("Source summary contains duplicate variant/condition rows.")
    if not (pd.to_numeric(summary["seeds"], errors="coerce") == 10).all():
        raise RuntimeError("Every source summary group must report seeds=10.")
    if not (pd.to_numeric(summary["retained_after_trim"], errors="coerce") == 8).all():
        raise RuntimeError("Every source summary group must retain eight values after trimming.")
    if not (summary["aggregation"].astype(str) == TRIMMED_AGGREGATION).all():
        raise RuntimeError("Source summary aggregation does not match the frozen protocol.")

    methods = candidate["Method"].astype(str)
    if len(candidate) != 6 or methods.duplicated().any() or set(methods) != set(SOURCE_METHODS):
        raise RuntimeError("Source candidate must contain Full reference plus exactly five controls.")
    _require_finite(candidate, ("Params (M)",), "source candidate")


def validate_main_contract(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    global_seed: int,
) -> None:
    seeds = _expected_seeds(global_seed)
    groups = raw.groupby(["variant", "condition"], sort=False)["seed"].agg(
        lambda values: {int(value) for value in values}
    )
    if (
        len(raw) != 120
        or set(raw["variant"].astype(str)) != set(MAIN_VARIANTS)
        or set(raw["condition"].astype(str)) != set(CONDITIONS)
        or len(groups) != 12
        or any(group_seeds != seeds for group_seeds in groups)
        or raw.duplicated(["variant", "condition", "seed"]).any()
    ):
        raise RuntimeError("Main-display raw must satisfy the exact 4/12/120/10-seed contract.")
    if (
        len(summary) != 12
        or set(summary["variant"].astype(str)) != set(MAIN_VARIANTS)
        or set(summary["condition"].astype(str)) != set(CONDITIONS)
        or summary.duplicated(["variant", "condition"]).any()
    ):
        raise RuntimeError("Main-display summary must contain exactly 12 groups for four controls.")
    methods = candidate["Method"].astype(str)
    if len(candidate) != 5 or methods.duplicated().any() or tuple(methods) != MAIN_METHODS:
        raise RuntimeError("Compact main candidate must contain Full reference plus four controls in source order.")


def copy_with_record(path: Path, archive_dir: Path) -> Dict[str, object]:
    """Copy once; on rerun, accept only a byte-identical archived file."""

    target = archive_dir / path.name
    source_sha256 = sha256(path)
    if target.exists():
        if not target.is_file():
            raise RuntimeError("Archive target is not a file: {0}".format(target))
        archive_sha256 = sha256(target)
        if archive_sha256 != source_sha256:
            raise RuntimeError(
                "Existing archive differs from the frozen source; refusing to overwrite: {0}".format(target)
            )
    else:
        shutil.copy2(str(path), str(target))
        archive_sha256 = sha256(target)
        if archive_sha256 != source_sha256:
            raise RuntimeError("Archived copy SHA-256 mismatch: {0}".format(target))
    return {
        "source": str(path.resolve()),
        "archive": str(target.resolve()),
        "size_bytes": int(path.stat().st_size),
        "source_sha256": source_sha256,
        "archive_sha256": archive_sha256,
    }


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    columns = [str(column) for column in frame.columns]
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(column) for column in columns) + r" \\")
    lines.append(r"\midrule")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_record(path: Path, rows: Optional[int] = None) -> Dict[str, object]:
    record: Dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _source_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in paths}


def _assert_sources_unchanged(paths: Sequence[Path], before: Mapping[str, str]) -> None:
    after = _source_hashes(paths)
    if dict(before) != after:
        changed = [path for path, digest in before.items() if after.get(path) != digest]
        raise RuntimeError("A frozen Experiment 04 source changed: {0}".format(", ".join(changed)))


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    ablation_dir = suite_root / "outputs" / str(args.run_tag) / "04_additional_ablation"
    if not ablation_dir.is_dir():
        raise FileNotFoundError("Ablation output directory not found: {0}".format(ablation_dir))

    raw_path = _require_file(ablation_dir / RAW_NAME)
    summary_path = _require_file(ablation_dir / SUMMARY_NAME)
    candidate_path = _require_file(ablation_dir / CANDIDATE_NAME)
    candidate_tex_path = _require_file(ablation_dir / CANDIDATE_TEX_NAME)
    run_manifest_path = _require_file(ablation_dir / RUN_MANIFEST_NAME)
    post_gate_path = _require_file(ablation_dir / POST_GATE_NAME)
    source_paths = (
        raw_path,
        summary_path,
        candidate_path,
        candidate_tex_path,
        run_manifest_path,
        post_gate_path,
    )
    source_hashes_before = _source_hashes(source_paths)

    global_seed = _read_global_seed(suite_root)
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    candidate = pd.read_csv(candidate_path)
    validate_source_contract(raw, summary, candidate, global_seed)

    raw_main = raw[raw["variant"].astype(str).isin(MAIN_VARIANTS)].copy()
    summary_main = summary[summary["variant"].astype(str).isin(MAIN_VARIANTS)].copy()
    candidate_main = candidate[candidate["Method"].astype(str).isin(MAIN_METHODS)].copy()
    raw_equal = raw[raw["variant"].astype(str) == EQUAL_VARIANT].copy()
    summary_equal = summary[summary["variant"].astype(str) == EQUAL_VARIANT].copy()
    candidate_equal = candidate[candidate["Method"].astype(str) == "Equal weights"].copy()

    validate_main_contract(raw_main, summary_main, candidate_main, global_seed)
    if len(raw_equal) != 30 or len(summary_equal) != 3 or len(candidate_equal) != 1:
        raise RuntimeError("Equal-weight archive must contain exactly 30 raw, 3 summary, and 1 candidate row.")

    archive_dir = ablation_dir / ARCHIVE_DIR_NAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_records = [copy_with_record(path, archive_dir) for path in source_paths]

    raw_main_path = ablation_dir / MAIN_RAW_NAME
    summary_main_path = ablation_dir / MAIN_SUMMARY_NAME
    candidate_main_path = ablation_dir / MAIN_CANDIDATE_NAME
    candidate_main_tex_path = ablation_dir / MAIN_CANDIDATE_TEX_NAME
    equal_path = archive_dir / EQUAL_ARCHIVE_NAME
    candidate_equal_path = archive_dir / EQUAL_CANDIDATE_NAME
    note_path = ablation_dir / MAIN_NOTE_NAME
    manifest_path = ablation_dir / MAIN_MANIFEST_NAME

    raw_main.to_csv(raw_main_path, index=False)
    summary_main.to_csv(summary_main_path, index=False)
    candidate_main.to_csv(candidate_main_path, index=False)
    write_latex(candidate_main, candidate_main_tex_path)
    equal_evidence = pd.concat(
        [
            raw_equal.assign(record_type="raw"),
            summary_equal.assign(record_type="summary"),
        ],
        ignore_index=True,
        sort=False,
    )
    equal_evidence.to_csv(equal_path, index=False)
    candidate_equal.to_csv(candidate_equal_path, index=False)
    note_path.write_text(MAIN_NOTE, encoding="utf-8")

    source_hashes_after = _source_hashes(source_paths)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A frozen Experiment 04 source changed while deriving the main display.")

    outputs = {
        "raw_main_scope": _output_record(raw_main_path, len(raw_main)),
        "summary_main_scope": _output_record(summary_main_path, len(summary_main)),
        "candidate_main_scope": _output_record(candidate_main_path, len(candidate_main)),
        "candidate_main_scope_tex": _output_record(candidate_main_tex_path),
        "equal_weights_archive": _output_record(equal_path, len(equal_evidence)),
        "candidate_equal_weights": _output_record(candidate_equal_path, len(candidate_equal)),
        "main_scope_note": _output_record(note_path),
    }
    manifest = {
        "status": "PASS",
        "operation": "non_destructive_main_text_scope_filter",
        "run_tag": str(args.run_tag),
        "global_seed": global_seed,
        "expected_seeds": sorted(_expected_seeds(global_seed)),
        "source_evidence_contract": {
            "status": "PASS",
            "variants": list(SOURCE_VARIANTS),
            "conditions": list(CONDITIONS),
            "raw_rows": 150,
            "summary_groups": 15,
            "seeds_per_group": 10,
            "retained_after_trim": 8,
            "aggregation": TRIMMED_AGGREGATION,
        },
        "main_display_contract": {
            "status": "PASS",
            "variants": list(MAIN_VARIANTS),
            "conditions": list(CONDITIONS),
            "raw_rows": 120,
            "summary_groups": 12,
            "seeds_per_group": 10,
            "compact_table_rows": 5,
        },
        "main_scope_variants": list(MAIN_VARIANTS),
        "excluded_from_main_display": EQUAL_VARIANT,
        "equal_weight_result_deleted": False,
        "equal_weight_result_archived": True,
        "original_source_files_modified": False,
        "equal_weight_archive_contract": {
            "raw_rows": 30,
            "summary_rows": 3,
            "candidate_rows": 1,
        },
        "scope_change_timing": "after_initial_experiment; disclosed as a display-only scope change",
        "claim_boundary": (
            "The compact main table focuses on encoder heterogeneity, attention dimension, and learned-gate "
            "formulation. The observed equal-weight control remains available in supplementary/provenance "
            "materials and must not be described as unevaluated."
        ),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "archive_records": archive_records,
        "outputs": outputs,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _assert_sources_unchanged(source_paths, source_hashes_before)
    print("Source evidence contract: PASS (5 variants / 15 groups / 150 rows / 10 seeds)")
    print("Main display contract: PASS (4 variants / 12 groups / 120 rows / 5 table rows)")
    print("Equal-weight evidence retained:", archive_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
