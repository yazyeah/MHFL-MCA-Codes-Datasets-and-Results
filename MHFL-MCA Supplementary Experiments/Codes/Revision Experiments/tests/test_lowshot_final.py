from __future__ import annotations

import importlib.util
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhfl_review import config
from mhfl_review.data import DatasetBundle, PairedClassData
from mhfl_review.stats import mean_sd, trimmed_mean_sd


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lowshot = _load_module("lowshot_final_test_module", config.SUITE_ROOT / "05_lowshot_threshold.py")
lowshot_audit = _load_module(
    "lowshot_final_audit_test_module",
    config.SUITE_ROOT / "tools" / "audit_lowshot_final.py",
)


def synthetic_lowshot(
    n_grid=lowshot.N_GRID_FULL,
    runs_per_n: int = lowshot.RUNS_PER_N_FULL,
) -> pd.DataFrame:
    rows = []
    metric_offsets = {
        "test_accuracy": np.array([0.31, -0.23, 0.13, -0.09, 0.07, -0.05, 0.04, -0.03, 0.02, 0.00]),
        "test_macro_precision": np.array([-0.24, 0.32, 0.11, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01, 0.00]),
        "test_macro_recall": np.array([0.10, -0.07, 0.30, -0.25, 0.06, -0.04, 0.03, -0.02, 0.01, 0.00]),
        "test_macro_f1": np.array([0.09, -0.06, 0.05, 0.31, -0.26, -0.04, 0.03, -0.02, 0.01, 0.00]),
    }
    gains = np.array([0.30, -0.20, 0.10, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01, 0.00])
    for n_train in n_grid:
        for run_idx in range(1, int(runs_per_n) + 1):
            seed = lowshot.paper_seed(int(n_train), run_idx)
            split_signature = "split-N{0}-seed{1}".format(n_train, seed)
            data_signature = "uo-source-data"
            hparam_signature = "confirmed-uo-optuna"
            index = run_idx - 1
            for variant in lowshot.VARIANT_NAMES:
                base = 0.60 + 0.01 * int(n_train)
                values = {
                    metric: float(np.clip(base + offsets[index], 0.01, 0.99))
                    for metric, offsets in metric_offsets.items()
                }
                if variant == "no_caim":
                    values["test_accuracy"] = float(np.clip(values["test_accuracy"] - gains[index], 0.01, 0.99))
                train_accuracy = float(min(0.995, values["test_accuracy"] + 0.12))
                heldout_accuracy = float(values["test_accuracy"])
                rows.append(
                    {
                        "variant": variant,
                        "n_train": int(n_train),
                        "run_idx": run_idx,
                        "seed": seed,
                        **values,
                        "train_accuracy": train_accuracy,
                        "heldout_accuracy": heldout_accuracy,
                        "generalization_gap": train_accuracy - heldout_accuracy,
                        "split_signature": split_signature,
                        "data_signature": data_signature,
                        "hyperparameter_signature": hparam_signature,
                        "source_type": "new_paper_aligned_training",
                        "train_time_s": float(run_idx),
                    }
                )
    return pd.DataFrame(rows, columns=lowshot.RAW_COLUMNS)


def _synthetic_uo_bundle() -> DatasetBundle:
    classes = []
    for label in range(2):
        values = np.arange(400 * 4, dtype=np.float32).reshape(400, 4) + label * 10_000
        classes.append(
            PairedClassData(
                label_id=label,
                label_name="class-{0}".format(label),
                x1=values,
                x2=values + 50_000,
                sample_ids=np.asarray(["c{0}-s{1:03d}".format(label, index) for index in range(400)]),
            )
        )
    return DatasetBundle(
        name="synthetic-uo",
        classes=tuple(classes),
        metadata={"signature": {"signature": "synthetic-uo-signature"}},
    )


def test_full_contract_uses_n_specific_ten_run_seed_map_and_140_rows():
    raw = synthetic_lowshot()
    report = lowshot.validate_raw(raw, lowshot.N_GRID_FULL, lowshot.RUNS_PER_N_FULL)

    assert lowshot.N_GRID_FULL == (1, 2, 3, 4, 5, 7, 10)
    assert lowshot.N_GRID_FAST == (5, 10)
    assert lowshot.RUNS_PER_N_FULL == 10
    assert config.LOWSHOT_REPEATS_FULL == 10
    assert config.LOWSHOT_REPEATS_FAST == 2
    assert config.LOWSHOT_N_GRID == lowshot.N_GRID_FULL
    assert lowshot.expected_seed_map(lowshot.N_GRID_FULL, 10) == {
        n_train: list(range(100 * n_train + 1, 100 * n_train + 11))
        for n_train in lowshot.N_GRID_FULL
    }
    assert report["status"] == "PASS"
    assert report["rows"] == 140
    assert report["matched_split_pairs"] == 70


def test_source_exact_split_uses_first_n_and_all_remaining_as_heldout():
    bundle = _synthetic_uo_bundle()
    n_train = 5
    seed = lowshot.paper_seed(n_train, 1)
    splits, metadata = lowshot.prepare_source_exact_split(bundle, n_train, seed)

    expected_order = np.arange(400, dtype=np.int64)
    np.random.RandomState(seed).shuffle(expected_order)
    expected_train_ids = {
        "c{0}-s{1:03d}".format(label, index)
        for label in range(2)
        for index in expected_order[:n_train]
    }
    assert len(splits["train"]) == 2 * n_train
    assert len(splits["heldout"]) == 2 * (400 - n_train)
    assert set(splits["train"].sample_ids) == expected_train_ids
    assert set(splits["train"].sample_ids).isdisjoint(set(splits["heldout"].sample_ids))
    assert set(splits["train"].sample_ids) | set(splits["heldout"].sample_ids) == {
        "c{0}-s{1:03d}".format(label, index)
        for label in range(2)
        for index in range(400)
    }
    assert metadata == {
        "protocol": "uploaded_uo_first_N_train_all_remainder_heldout",
        "split_signature": metadata["split_signature"],
        "data_signature": "synthetic-uo-signature",
        "train_size": 10,
        "heldout_size": 790,
    }


def test_source_exact_split_requires_400_paired_segments_per_class():
    bundle = _synthetic_uo_bundle()
    bad = bundle.classes[0]
    truncated = PairedClassData(
        label_id=bad.label_id,
        label_name=bad.label_name,
        x1=bad.x1[:-1],
        x2=bad.x2[:-1],
        sample_ids=bad.sample_ids[:-1],
    )
    invalid = DatasetBundle(bundle.name, (truncated,) + bundle.classes[1:], bundle.metadata)
    with pytest.raises(RuntimeError, match="exactly 400 paired segments"):
        lowshot.prepare_source_exact_split(invalid, 5, 501)


def test_paired_model_seed_sets_python_tensorflow_without_resetting_numpy():
    observed = []
    fake_tf = SimpleNamespace(random=SimpleNamespace(set_seed=lambda value: observed.append(value)))
    np.random.seed(314159)
    state_before = np.random.get_state()
    lowshot.set_paired_model_seed(fake_tf, 501)
    state_after = np.random.get_state()

    assert observed == [501]
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_summary_trims_every_metric_independently_and_retains_untrimmed_audit_fields():
    raw = synthetic_lowshot()
    summary = lowshot.summarize(raw, runs_per_n=10, use_trim=True)
    block = raw[(raw["variant"] == "full") & (raw["n_train"] == 1)]
    row = summary[(summary["variant"] == "full") & (summary["n_train"] == 1)].iloc[0]

    for metric in lowshot.SUMMARY_METRICS:
        expected_mean, expected_sd = trimmed_mean_sd(block[metric])
        untrimmed_mean, untrimmed_sd = mean_sd(block[metric])
        assert row[metric + "_mean"] == pytest.approx(expected_mean)
        assert row[metric + "_sd"] == pytest.approx(expected_sd)
        assert row[metric + "_untrimmed_mean"] == pytest.approx(untrimmed_mean)
        assert row[metric + "_untrimmed_sd"] == pytest.approx(untrimmed_sd)
    assert row["seeds"] == 10
    assert row["seeds_total"] == 10
    assert row["retained_after_trim"] == 8
    assert row["aggregation"] == lowshot.TRIMMED_AGGREGATION


def test_caim_gain_is_aligned_by_n_and_seed_before_trimming():
    raw = synthetic_lowshot()
    paired = lowshot.paired_caim(raw, runs_per_n=10, use_trim=True)
    full = raw[(raw["variant"] == "full") & (raw["n_train"] == 1)][["seed", "test_accuracy"]]
    no_caim = raw[(raw["variant"] == "no_caim") & (raw["n_train"] == 1)][["seed", "test_accuracy"]]
    aligned = full.merge(no_caim, on="seed", suffixes=("_full", "_no_caim"), validate="one_to_one")
    differences = aligned["test_accuracy_full"] - aligned["test_accuracy_no_caim"]
    expected_mean, expected_sd = trimmed_mean_sd(differences)
    row = paired[paired["n_train"] == 1].iloc[0]

    assert row["paired_gain_mean"] == pytest.approx(expected_mean)
    assert row["paired_gain_sd"] == pytest.approx(expected_sd)
    assert row["caim_gain_mean"] == pytest.approx(expected_mean)
    assert row["caim_gain_sd"] == pytest.approx(expected_sd)
    assert row["seeds_total"] == 10
    assert row["retained_after_trim"] == 8


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nan", "wrong_seed", "wrong_split", "wrong_source"])
def test_raw_validator_rejects_incomplete_or_non_paper_aligned_evidence(failure: str):
    raw = synthetic_lowshot()
    if failure == "missing":
        raw = raw.iloc[:-1].copy()
    elif failure == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    elif failure == "nan":
        raw.loc[0, "test_accuracy"] = np.nan
    elif failure == "wrong_seed":
        raw.loc[0, "seed"] = 20260806
    elif failure == "wrong_split":
        mask = (raw["variant"] == "no_caim") & (raw["n_train"] == 1) & (raw["run_idx"] == 1)
        raw.loc[mask, "split_signature"] = "different-split"
    elif failure == "wrong_source":
        raw.loc[0, "source_type"] = "legacy_result"

    with pytest.raises(RuntimeError):
        lowshot.validate_raw(raw, lowshot.N_GRID_FULL, 10)


def test_post_gate_recomputes_140_14_7_and_70_pair_contract():
    raw = synthetic_lowshot()
    raw_gate = lowshot.validate_raw(raw, lowshot.N_GRID_FULL, 10)
    summary = lowshot.summarize(raw, runs_per_n=10, use_trim=True)
    paired = lowshot.paired_caim(raw, runs_per_n=10, use_trim=True)
    thresholds = lowshot.build_operational_thresholds(summary, use_trim=True)
    anchors = {"status": "PASS", "final_assets_authorized": True}
    report = lowshot.build_post_gate(
        raw=raw,
        summary=summary,
        paired=paired,
        thresholds=thresholds,
        anchors=anchors,
        mode="full",
        n_grid=lowshot.N_GRID_FULL,
        runs_per_n=10,
        use_trim=True,
        raw_gate=raw_gate,
        figure_bundle_complete=True,
    )

    assert report["status"] == "PASS"
    assert report["final_outputs_authorized"] is True
    assert report["raw_rows"] == 140
    assert report["summary_rows"] == 14
    assert report["paired_gain_rows"] == 7
    assert report["matched_split_pairs"] == 70
    assert report["seed_map"] == lowshot.expected_seed_map(lowshot.N_GRID_FULL, 10)
    assert report["retained_after_trim"] == 8
    assert report["aggregation"] == lowshot.TRIMMED_AGGREGATION
    assert report["summary_recomputed_from_raw"] is True
    assert report["paired_gain_aligned_by_seed_before_trim"] is True

    tampered = summary.copy()
    tampered.loc[0, "test_accuracy_mean"] += 0.01
    failed = lowshot.build_post_gate(
        raw, tampered, paired, thresholds, anchors, "full", lowshot.N_GRID_FULL,
        10, True, raw_gate, True,
    )
    assert failed["status"] == "FAIL"
    assert failed["final_outputs_authorized"] is False
    assert failed["summary_recomputed_from_raw"] is False


def test_anchor_gate_requires_exact_four_decimal_table5_match():
    summary = lowshot.summarize(synthetic_lowshot(), runs_per_n=10, use_trim=True)
    for n_train, fields in lowshot.PAPER_ANCHORS.items():
        mask = (summary["variant"] == "full") & (summary["n_train"] == n_train)
        for field, value in fields.items():
            summary.loc[mask, field] = value
    report = lowshot.anchor_gate(summary, require=True)
    assert report["status"] == "PASS"
    assert report["final_assets_authorized"] is True

    summary.loc[(summary["variant"] == "full") & (summary["n_train"] == 5), "test_accuracy_mean"] += 0.0001
    failed = lowshot.anchor_gate(summary, require=True)
    assert failed["status"] == "FAIL"
    assert failed["final_assets_authorized"] is False


def test_fast_mode_remains_two_runs_per_n_without_trim_or_final_authorization():
    raw = synthetic_lowshot(lowshot.N_GRID_FAST, runs_per_n=2)
    raw_gate = lowshot.validate_raw(raw, lowshot.N_GRID_FAST, 2)
    summary = lowshot.summarize(raw, runs_per_n=2, use_trim=False)
    paired = lowshot.paired_caim(raw, runs_per_n=2, use_trim=False)
    thresholds = lowshot.build_operational_thresholds(summary, use_trim=False)
    report = lowshot.build_post_gate(
        raw, summary, paired, thresholds,
        {"status": "NOT_REQUIRED_FAST", "final_assets_authorized": False},
        "fast", lowshot.N_GRID_FAST, 2, False, raw_gate, False,
    )

    assert lowshot.expected_seed_map(lowshot.N_GRID_FAST, 2) == {5: [501, 502], 10: [1001, 1002]}
    assert len(raw) == 8
    assert (summary["seeds_total"] == 2).all()
    assert (summary["retained_after_trim"] == 2).all()
    assert (summary["aggregation"] == lowshot.FAST_AGGREGATION).all()
    assert all("untrimmed fast-smoke accuracy" in row["criterion"] for row in thresholds)
    assert report["status"] == "PASS"
    assert report["final_outputs_authorized"] is False


def test_full_mode_rejects_smoke_or_manual_run_tags_before_tensorflow(monkeypatch):
    monkeypatch.setattr(
        lowshot,
        "parse_args",
        lambda: SimpleNamespace(mode="full", run_tag="smoke"),
    )
    monkeypatch.setattr(
        lowshot,
        "configure_tensorflow",
        lambda: pytest.fail("TensorFlow must not initialize for a rejected full run tag"),
    )
    with pytest.raises(RuntimeError, match="explicit non-smoke run tag"):
        lowshot.main()


def test_run_artifact_record_binds_small_metrics_weights_and_history_files(tmp_path: Path):
    output_dir = tmp_path / "05_lowshot_threshold"
    run_dir = output_dir / "models" / "full" / "N1_run01_seed101"
    run_dir.mkdir(parents=True)
    weights_path = run_dir / "model.weights.h5"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    weights_path.write_bytes(b"unit-test-weights")
    history_path.write_text("epoch,loss\n" + "\n".join("{0},0.1".format(i) for i in range(1, 81)), encoding="utf-8")
    row = {
        "variant": "full",
        "n_train": 1,
        "run_idx": 1,
        "seed": 101,
        "run_signature": "run-signature",
        "provenance_signature": "provenance-signature",
        "model_parameter_count": lowshot.EXPECTED_FULL_PARAMETER_COUNT,
        "metrics_path": str(metrics_path),
        "weights_path": str(weights_path),
        "history_path": str(history_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
    }
    metrics_path.write_text(json.dumps(row), encoding="utf-8")

    record = lowshot.run_artifact_record(row, output_dir)
    assert (record["variant"], record["n_train"], record["run_idx"], record["seed"]) == ("full", 1, 1, 101)
    assert record["model_parameter_count"] == lowshot.EXPECTED_FULL_PARAMETER_COUNT
    assert record["metrics"]["sha256"] == hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert record["weights"]["sha256"] == row["weights_sha256"]
    assert record["history"]["sha256"] == row["history_sha256"]
    assert record["history"]["rows"] == 80

    weights_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="weights SHA-256 changed"):
        lowshot.run_artifact_record(row, output_dir)


def test_static_pre_audit_proves_source_exact_80_epoch_adamax_contract_without_training():
    report = lowshot_audit.audit_pre(config.SUITE_ROOT)
    assert report["status"] == "PASS"
    assert report["protocol"] == "uo_source_aligned_paired_deterministic_extension_v1"
    assert report["seed_schedule"] == "seed = 100*N + run_idx"
    assert report["target_seed_map"]["5"] == list(range(501, 511))
    assert report["expected_model_slots"] == 140
    assert report["expected_raw_rows"] == 140
    assert report["expected_summary_rows"] == 14
    assert report["expected_paired_rows"] == 7
    assert report["matched_split_pairs"] == 70
    assert report["training"]["epochs"] == 80
    assert report["training"]["optimizer"] == "Adamax"
    assert report["training"]["gradient_clipping"] is None
    assert report["training"]["paired_model_rngs"] == ["Python", "TensorFlow"]
    assert report["training"]["numpy_reseeded_after_split"] is False
    assert report["split"]["evaluation_role"] == "all_remaining_segments_heldout"
    assert report["confirmed_uo_config"]["status"] == "CONFIRMED"
    assert report["execution_evidence"]["expected_run_artifact_count"] == 140
    assert report["execution_evidence"]["source_mat_content_hash_count"] == 14
    assert report["execution_evidence"]["final_state_bound_in_manifest_outputs"] is True
    assert report["training_was_run"] is False
    assert report["files_written"] == []
