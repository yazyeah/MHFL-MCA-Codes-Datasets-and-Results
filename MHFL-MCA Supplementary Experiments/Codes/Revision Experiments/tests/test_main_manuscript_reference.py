import hashlib
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


SUITE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = SUITE_ROOT / "configs" / "main_manuscript_deep_reference.json"
CURRENT_TEX_SHA256 = (
    "cd6bc52feb997a7cdc3b56d9e951ed212f15019c25a88ab21edd3b3db4ad83ce"
)
CURRENT_REFERENCE_VERSION = "current_submission_2026-08-10"
EXPECTED_MODELS = (
    "MRCFN",
    "CFFN",
    "CDTFAFN",
    "MSF-DFormer",
    "KDCNN-DF",
    "Full MHFL-MCA",
)
EXPECTED_N = (5, 10, 15, 20, 25, 30)
TABLES = {
    "UO_Table_5": ("case1_compare_main", "Table 5", "UO", "held-out"),
    "KAIST_Table_9": ("case2_quant_2nm", "Table 9", "KAIST", "2Nm"),
    "KAIST_Table_10": ("case2_quant_4nm", "Table 10", "KAIST", "4Nm"),
}
AUTHORITATIVE_FIELDS = (
    "accuracy_mean",
    "accuracy_sd",
    "macro_precision_mean",
    "macro_precision_sd",
    "macro_f1_mean",
    "macro_f1_sd",
)
PROVENANCE_FIELDS = (
    "source_tex_path",
    "source_table",
    "source_sha256",
    "reference_version",
    "extraction_method",
)
_LATEX_MODELS = EXPECTED_MODELS[:-1] + ("Proposed (MHFL-MCA)",)
_VALUE_PATTERN = re.compile(r"(0\.\d+)\s*\$\\pm\$\s*(0\.\d+)")


def _load_reference():
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SUITE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_tex(reference):
    source = reference["current_manuscript_source"]
    tex_path = Path(source["source_tex_path"])
    if not tex_path.is_absolute():
        tex_path = SUITE_ROOT / tex_path
    tex_bytes = tex_path.read_bytes()
    return source, tex_bytes, tex_bytes.decode("utf-8")


def _table_block(tex, label):
    marker = r"\label{tab:" + label + "}"
    marker_pos = tex.index(marker)
    start = tex.rfind(r"\begin{table}", 0, marker_pos)
    end = tex.index(r"\end{table}", marker_pos) + len(r"\end{table}")
    return tex[start:end]


def _is_data_row(line):
    return "&" in line and any(
        line.startswith(model)
        or (model.startswith("Proposed") and "Proposed (MHFL-MCA)" in line)
        for model in _LATEX_MODELS
    )


def _extract_latex_rows(tex, label):
    logical_rows = []
    pending = ""
    for raw_line in _table_block(tex, label).splitlines():
        line = raw_line.strip()
        if pending:
            pending += " " + line
            if r"\\" in line:
                logical_rows.append(pending)
                pending = ""
        elif _is_data_row(line):
            pending = line
            if r"\\" in line:
                logical_rows.append(pending)
                pending = ""
    assert pending == ""

    n_pairs = ((5, 10), (15, 20), (25, 30))
    occurrences = defaultdict(int)
    extracted = {}
    for line in logical_rows:
        if "Proposed (MHFL-MCA)" in line:
            model = "Full MHFL-MCA"
        else:
            model = next(name for name in EXPECTED_MODELS[:-1] if line.startswith(name))
        occurrence = occurrences[model]
        occurrences[model] += 1
        cells = [(float(mean), float(sd)) for mean, sd in _VALUE_PATTERN.findall(line)]
        assert len(cells) == 6
        for pair_index, n_train in enumerate(n_pairs[occurrence]):
            offset = pair_index * 3
            extracted[(model, n_train)] = {
                "accuracy_mean": cells[offset][0],
                "accuracy_sd": cells[offset][1],
                "macro_precision_mean": cells[offset + 1][0],
                "macro_precision_sd": cells[offset + 1][1],
                "macro_f1_mean": cells[offset + 2][0],
                "macro_f1_sd": cells[offset + 2][1],
            }
    assert len(extracted) == 36
    assert occurrences == {model: 3 for model in EXPECTED_MODELS}
    return extracted


def _row(reference, table_name, model, n_train):
    return next(
        row
        for row in reference["tables"][table_name]["rows"]
        if row["model"] == model and row["n_train"] == n_train
    )


def test_reference_has_exact_108_row_cross_product_and_provenance():
    reference = _load_reference()
    source, tex_bytes, _ = _current_tex(reference)
    source_sha256 = hashlib.sha256(tex_bytes).hexdigest()

    assert reference["schema_version"] == 2
    assert reference["confirmation_status"] == "confirmed"
    assert reference["reference_version"] == source["reference_version"]
    assert source["source_sha256"] == source_sha256
    assert source["source_sha256"] == CURRENT_TEX_SHA256
    assert source["reference_version"] == CURRENT_REFERENCE_VERSION
    assert source["extraction_method"] == "current_manuscript_table_source"

    all_keys = []
    for table_name, (_, source_table, expected_case, expected_load) in TABLES.items():
        table = reference["tables"][table_name]
        assert table["source_table"] == source_table
        assert table["case"] == expected_case
        assert table["load"] == expected_load
        assert len(table["rows"]) == 36
        assert {row["model"] for row in table["rows"]} == set(EXPECTED_MODELS)
        assert {row["n_train"] for row in table["rows"]} == set(EXPECTED_N)
        for row in table["rows"]:
            key = (table_name, row["model"], row["n_train"])
            all_keys.append(key)
            assert all(field in row for field in PROVENANCE_FIELDS)
            assert row["source_tex_path"] == source["source_tex_path"]
            assert row["source_table"] == source_table
            assert row["source_sha256"] == source_sha256
            assert row["reference_version"] == source["reference_version"]
            assert row["extraction_method"] == "current_manuscript_table_source"
            assert all(math.isfinite(float(row[field])) for field in AUTHORITATIVE_FIELDS)

    assert len(all_keys) == 108
    assert len(set(all_keys)) == 108


def test_reference_matches_current_latex_tables_row_by_row():
    reference = _load_reference()
    _, _, tex = _current_tex(reference)
    compared_values = 0
    for table_name, (label, _, _, _) in TABLES.items():
        expected = _extract_latex_rows(tex, label)
        actual_rows = reference["tables"][table_name]["rows"]
        actual = {(row["model"], row["n_train"]): row for row in actual_rows}
        assert set(actual) == set(expected)
        for key, expected_metrics in expected.items():
            for field, expected_value in expected_metrics.items():
                assert actual[key][field] == expected_value
                compared_values += 1
    assert compared_values == 108 * 6


def test_known_reference_samples_match_current_submission():
    reference = _load_reference()

    for n_train, mean, sd in (
        (15, 0.8310, 0.0331),
        (20, 0.8620, 0.0448),
        (25, 0.9047, 0.0446),
        (30, 0.9577, 0.0226),
    ):
        row = _row(reference, "UO_Table_5", "MRCFN", n_train)
        assert row["accuracy_mean"] == mean
        assert row["accuracy_sd"] == sd

    row = _row(reference, "KAIST_Table_9", "MRCFN", 5)
    assert (row["accuracy_mean"], row["accuracy_sd"]) == (0.5929, 0.0446)

    row = _row(reference, "KAIST_Table_10", "MRCFN", 5)
    assert (row["accuracy_mean"], row["accuracy_sd"]) == (0.5384, 0.0568)

    row = _row(reference, "KAIST_Table_10", "Full MHFL-MCA", 30)
    assert (row["accuracy_mean"], row["accuracy_sd"]) == (0.9925, 0.0096)
    assert (row["macro_f1_mean"], row["macro_f1_sd"]) == (0.9925, 0.0097)


def test_legacy_macro_recall_is_not_claimed_as_latex_evidence():
    reference = _load_reference()
    assert "do not report macro-Recall" in reference["macro_recall_compatibility_policy"]
    for table in reference["tables"].values():
        for row in table["rows"]:
            assert row["macro_recall_evidence_scope"] == (
                "legacy_schema_compatibility_only_not_reported_in_current_manuscript_table"
            )
    for source in reference["source_files"]:
        assert source["evidence_scope"].startswith("legacy_macro_recall")


def test_reference_gate_and_candidate_refresh_use_only_current_tex_plus_frozen_summary(tmp_path, monkeypatch):
    traditional = _load_script("main_reference_traditional_test", "06_traditional_baselines.py")
    assets = _load_script("main_reference_assets_test", "07_build_manuscript_assets.py")
    report = traditional.validate_deep_reference(traditional.load_deep_reference())
    assert report["rows"] == 108
    assert report["current_manuscript_metric_values_verified"] == 648
    assert report["legacy_recall_values_verified"] == 216

    summary_rows = []
    for case, load, n_train, snr_db, method in traditional.expected_group_identities("full"):
        row = {
            "case": case,
            "load": load,
            "n_train": n_train,
            "snr_db": snr_db,
            "method": method,
            "seeds_total": 10,
            "retained_after_trim": 8,
            "aggregation": traditional.TRIMMED_AGGREGATION,
        }
        for metric in traditional.METRIC_COLUMNS:
            row[metric + "_mean"] = 0.75
            row[metric + "_sd"] = 0.01
        summary_rows.append(row)
    summary_path = tmp_path / "traditional_baselines_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    summary_sha_before = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    summary_mtime_before = summary_path.stat().st_mtime_ns

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate-only refresh must not execute SVM/model experiment code")

    monkeypatch.setattr(traditional, "run_uo", forbidden)
    monkeypatch.setattr(traditional, "run_kaist", forbidden)
    monkeypatch.setattr(traditional, "make_svc", forbidden)
    candidate_path = tmp_path / "manuscript_candidate_rows.csv"
    refresh = traditional.refresh_manuscript_candidate_from_existing_summary(
        summary_path,
        candidate_path,
        REFERENCE_PATH,
    )
    assert refresh["status"] == "PASS"
    assert refresh["candidate_rows"] == 144
    assert refresh["scope_counts"] == {"UO": 48, "KAIST_2Nm": 48, "KAIST_4Nm": 48}
    assert refresh["svm_or_model_execution"] is False
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == summary_sha_before
    assert summary_path.stat().st_mtime_ns == summary_mtime_before

    candidate = pd.read_csv(candidate_path)
    summary = pd.read_csv(summary_path)
    reference = _load_reference()
    assert assets._live_candidate_mismatches(candidate, summary, reference) == []
    candidate.loc[candidate["source_type"] == "main_manuscript_deep_reference", "accuracy_mean"] += 0.0001
    assert assets._live_candidate_mismatches(candidate, summary, reference)
