from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhfl_review import config
from mhfl_review.stats import trimmed_mean_sd


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


traditional = _load_module("traditional_final_test_module", config.SUITE_ROOT / "06_traditional_baselines.py")


def _metrics(seed_index: int):
    return {
        "accuracy": 0.50 + 0.03 * seed_index,
        "macro_f1": [0.90, 0.10, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54][seed_index],
        "macro_precision": [0.55, 0.57, 0.99, 0.20, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69][seed_index],
        "macro_recall": [0.35, 0.37, 0.39, 0.95, 0.15, 0.41, 0.43, 0.45, 0.47, 0.49][seed_index],
    }


def synthetic_traditional(seed_count: int = 10) -> pd.DataFrame:
    rows = []
    methods = traditional.METHOD_NAMES
    for seed_index in range(seed_count):
        seed = config.GLOBAL_SEED + seed_index
        values = _metrics(seed_index)
        for n_train in traditional.FULL_N_VALUES:
            for method in methods:
                rows.append({"case": "UO", "load": "held-out", "n_train": n_train, "seed": seed, "snr_db": None, "method": method, **values})
                for load in traditional.KAIST_LOADS:
                    rows.append({"case": "KAIST", "load": load, "n_train": n_train, "seed": seed, "snr_db": 0.0, "method": method, **values})
        for method in methods:
            for load in traditional.KAIST_LOADS:
                for snr_db in traditional.KAIST_NOISE_SNRS:
                    rows.append({"case": "KAIST-noise", "load": load, "n_train": 30, "seed": seed, "snr_db": snr_db, "method": method, **values})
    return pd.DataFrame(rows)


def test_full_traditional_protocol_has_ten_seeds_and_480_rows():
    raw = synthetic_traditional()
    report = traditional.validate_raw_results(raw, "full", traditional.seed_sequence("full"))
    assert config.TRADITIONAL_REPEATS_FULL == 10
    assert config.TRADITIONAL_REPEATS_FAST == 2
    assert traditional.expected_raw_counts("full") == {"UO": 120, "KAIST": 240, "KAIST-noise": 120, "total": 480}
    assert len(raw) == 480
    assert report["groups"] == 48
    assert report["component_rows"] == {"UO": 120, "KAIST": 240, "KAIST-noise": 120}


def test_traditional_metrics_are_trimmed_independently():
    raw = synthetic_traditional()
    summary = traditional.summarize(raw, expected_seed_count=10, use_trim=True)
    block = raw[(raw["case"] == "UO") & (raw["n_train"] == 5) & (raw["method"] == traditional.METHOD_NAMES[0])]
    row = summary[(summary["case"] == "UO") & (summary["n_train"] == 5) & (summary["method"] == traditional.METHOD_NAMES[0])].iloc[0]
    for metric in traditional.METRIC_COLUMNS:
        expected_mean, expected_sd = trimmed_mean_sd(block[metric])
        assert row[metric + "_mean"] == pytest.approx(expected_mean)
        assert row[metric + "_sd"] == pytest.approx(expected_sd)
        assert metric + "_untrimmed_mean" in summary.columns
        assert metric + "_untrimmed_sd" in summary.columns
    accuracy_selected = block.sort_values("accuracy").iloc[1:-1]
    assert row["macro_f1_mean"] != pytest.approx(float(accuracy_selected["macro_f1"].mean()))
    assert row["retained_after_trim"] == 8
    assert row["aggregation"] == traditional.TRIMMED_AGGREGATION


def test_deep_reference_is_strict_complete_and_unicode_safe():
    reference = traditional.load_deep_reference()
    report = traditional.validate_deep_reference(reference)
    serialized = json.dumps(reference, ensure_ascii=False)
    assert "\ufffd" not in serialized
    assert report["status"] == "PASS"
    assert report["rows"] == 108
    assert report["models"] == list(traditional.DEEP_MODEL_NAMES)
    assert report["n_values"] == list(traditional.FULL_N_VALUES)
    assert report["paired_comparison_permitted"] is False
    assert report["statistical_significance_claim_permitted"] is False
    assert report["source_files_present_and_hash_verified"] == 14
    assert report["source_metric_values_verified"] == 864
    assert all(len(record["sha256"]) == 64 for record in reference["source_files"])
    assert all(not Path(record["source_path"]).is_absolute() for record in reference["source_files"])
    cffn_n5 = next(
        row for row in reference["tables"]["KAIST_Table_9"]["rows"]
        if row["model"] == "CFFN" and row["n_train"] == 5
    )
    kdcnn_n5 = next(
        row for row in reference["tables"]["KAIST_Table_10"]["rows"]
        if row["model"] == "KDCNN-DF" and row["n_train"] == 5
    )
    assert cffn_n5["source_file_id"] == "kaist_cffn_n5_patch"
    assert kdcnn_n5["source_file_id"] == "kaist_kdcnn_df_n5_patch"


def test_deep_reference_rejects_protocol_mismatch_and_replacement_character():
    reference = traditional.load_deep_reference()
    invalid = copy.deepcopy(reference)
    invalid["tables"]["KAIST_Table_9"]["protocol"] = "stage3_noise"
    with pytest.raises(RuntimeError):
        traditional.validate_deep_reference(invalid)
    invalid = copy.deepcopy(reference)
    invalid["source_files"][0]["source_path"] += "\ufffd"
    with pytest.raises(RuntimeError):
        traditional.validate_deep_reference(invalid)


def test_deep_reference_rejects_missing_source_and_tampered_metric():
    reference = traditional.load_deep_reference()
    invalid = copy.deepcopy(reference)
    invalid["source_files"][0]["source_path"] = "missing/source.csv"
    invalid["source_files"][0]["workspace_relative_path"] = "missing/source.csv"
    with pytest.raises(RuntimeError, match="source file is missing"):
        traditional.validate_deep_reference(invalid)
    invalid = copy.deepcopy(reference)
    invalid["tables"]["UO_Table_5"]["rows"][0]["accuracy_mean"] -= 0.01
    with pytest.raises(RuntimeError, match="does not match"):
        traditional.validate_deep_reference(invalid)


def test_hybrid_candidate_is_144_clean_rows_without_paired_claims():
    raw = synthetic_traditional()
    summary = traditional.summarize(raw, expected_seed_count=10, use_trim=True)
    reference = traditional.load_deep_reference()
    candidate = traditional.build_full_manuscript_candidate(summary, reference)
    assert tuple(candidate.columns) == traditional.FULL_CANDIDATE_COLUMNS
    assert len(candidate) == 144
    assert set(candidate["case"]) == {"UO", "KAIST"}
    assert "KAIST-noise" not in set(candidate["case"])
    assert candidate["source_type"].value_counts().to_dict() == {
        "main_manuscript_deep_reference": 108,
        "new_traditional_baseline": 36,
    }
    assert not any("paired" in column.lower() for column in candidate.columns)
    deep = candidate[candidate["source_type"] == "main_manuscript_deep_reference"]
    assert (deep["comparison_type"] == "descriptive_aggregated_reference_only").all()


def test_full_post_gate_validates_reference_scope_channels_and_candidate():
    raw = synthetic_traditional()
    summary = traditional.summarize(raw, expected_seed_count=10, use_trim=True)
    reference = traditional.load_deep_reference()
    candidate = traditional.build_full_manuscript_candidate(summary, reference)
    channel = traditional.validate_full_kaist_channel_config(traditional.EXACT_CURRENT_CHANNEL, 0, False)
    report = traditional.validate_full_post_gate(raw, summary, candidate, channel, reference)
    assert report["status"] == "PASS"
    assert report["raw_rows"] == 480
    assert report["candidate_rows"] == 144
    assert report["failed_groups"] == []
    assert report["deep_model_training_performed"] is False
    assert report["candidate_scope"] == traditional.full_candidate_scope()
    assert report["summary_recomputed_from_raw"] is True
    assert report["candidate_recomputed_from_validated_sources"] is True


def test_full_post_gate_rejects_tampered_summary_and_candidate():
    raw = synthetic_traditional()
    summary = traditional.summarize(raw, expected_seed_count=10, use_trim=True)
    reference = traditional.load_deep_reference()
    candidate = traditional.build_full_manuscript_candidate(summary, reference)
    channel = traditional.validate_full_kaist_channel_config(traditional.EXACT_CURRENT_CHANNEL, 0, False)
    tampered_summary = summary.copy()
    tampered_summary.loc[0, "accuracy_mean"] = 0.123456
    with pytest.raises(RuntimeError, match="does not reproduce"):
        traditional.validate_full_post_gate(raw, tampered_summary, candidate, channel, reference)
    tampered_candidate = candidate.copy()
    tampered_candidate.loc[0, "accuracy_mean"] -= 0.01
    with pytest.raises(RuntimeError, match="does not reproduce"):
        traditional.validate_full_post_gate(raw, summary, tampered_candidate, channel, reference)


def test_failed_post_gate_cannot_write_candidate(monkeypatch):
    writes = []
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: writes.append((args, kwargs)))
    candidate_path = Path("must_not_be_written.csv")
    candidate = pd.DataFrame(columns=traditional.FULL_CANDIDATE_COLUMNS)
    with pytest.raises(RuntimeError):
        traditional.write_candidate_if_gate_passed(candidate, candidate_path, {"status": "FAIL"})
    assert writes == []


def test_06_does_not_train_or_require_deep_models():
    source = (config.SUITE_ROOT / "06_traditional_baselines.py").read_text(encoding="utf-8")
    assert "load_or_train_kaist" not in source
    assert "build_models(" not in source
    assert "model.load_weights" not in source
    assert "deep_model_training_performed\": False" in source
    assert "run_uo(" in source
    assert "run_kaist(" in source


def test_full_channel_gate_requires_u_phase_xa_and_no_fallback():
    report = traditional.validate_full_kaist_channel_config(
        traditional.EXACT_CURRENT_CHANNEL,
        0,
        False,
    )
    assert report["current_fallback"] is False
    with pytest.raises(RuntimeError):
        traditional.validate_full_kaist_channel_config("", 0, False)
    with pytest.raises(RuntimeError):
        traditional.validate_full_kaist_channel_config(traditional.EXACT_CURRENT_CHANNEL, 1, False)
    with pytest.raises(RuntimeError):
        traditional.validate_full_kaist_channel_config(traditional.EXACT_CURRENT_CHANNEL, 0, True)


def test_traditional_gate_rejects_missing_duplicate_and_nan():
    raw = synthetic_traditional()
    with pytest.raises(RuntimeError):
        traditional.validate_raw_results(raw.iloc[:-1], "full", traditional.seed_sequence("full"))
    with pytest.raises(RuntimeError):
        traditional.validate_raw_results(pd.concat([raw, raw.iloc[[0]]], ignore_index=True), "full", traditional.seed_sequence("full"))
    invalid = raw.copy()
    invalid.loc[0, "accuracy"] = float("nan")
    with pytest.raises(RuntimeError):
        traditional.validate_raw_results(invalid, "full", traditional.seed_sequence("full"))
    wrong_snr = raw.copy()
    wrong_snr.loc[(wrong_snr["case"] == "KAIST-noise") & (wrong_snr["snr_db"] == -8.0), "snr_db"] = -12.0
    with pytest.raises(RuntimeError, match="exact N/load/SNR/method protocol"):
        traditional.validate_raw_results(wrong_snr, "full", traditional.seed_sequence("full"))


def test_late_fusion_probability_columns_are_class_aligned():
    class Model:
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, features):
            assert features.shape == (1, 1)
            return np.array([[0.2, 0.5, 0.3]])

    aligned = traditional._aligned_predict_proba(
        Model(),
        np.zeros((1, 1)),
        np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(aligned, np.array([[0.5, 0.3, 0.2]]))


def test_uo_split_audit_reports_shared_recordings_without_overclaim():
    class Split:
        def __init__(self, sample_ids):
            self.sample_ids = np.asarray(sample_ids)

        def __len__(self):
            return len(self.sample_ids)

    report = traditional._split_audit(
        Split(["recording-a:0", "recording-b:0"]),
        Split(["recording-a:2048", "recording-c:0"]),
    )
    assert report["exact_sample_overlap"] == 0
    assert report["shared_recordings"] == ["recording-a"]
    assert "segment-disjoint" in report["claim_limit"]
    assert "does not establish recording-disjoint" in report["claim_limit"]

    with pytest.raises(RuntimeError, match="exact train/test sample overlap"):
        traditional._split_audit(
            Split(["recording-a:0"]),
            Split(["recording-a:0"]),
        )


def test_fast_repeat_count_and_training_protocol_are_unchanged():
    source = (config.SUITE_ROOT / "06_traditional_baselines.py").read_text(encoding="utf-8")
    assert traditional.seed_sequence("fast") == [config.GLOBAL_SEED, config.GLOBAL_SEED + 1]
    assert traditional.SVM_TUNING_N == 15
    assert traditional.SVM_TUNING_SEED_OFFSET == -1
    assert "GridSearchCV" in source
    assert "StratifiedKFold" in source
    assert "shuffle=True" in source
    assert "select_with_source_validation" in source
    assert "select_frozen_hyperparameters" in source
    assert "fit_fixed_models" in source
    assert 'for snr_db in (0.0, -4.0, -8.0)' in source
    assert 'else out_dir / "fast_summary_preview.csv"' in source
    assert '"manuscript_candidate": _file_record(candidate_path) if mode == "full" else None' in source
