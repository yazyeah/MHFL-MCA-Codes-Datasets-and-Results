from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mhfl_review import config
from mhfl_review.stats import trimmed_mean_sd


SUITE_ROOT = config.SUITE_ROOT
SCRIPT_PATH = SUITE_ROOT / "04_additional_ablation.py"
AUDIT_PATH = SUITE_ROOT / "tools" / "audit_additional_ablation_extension.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation = _load_module("additional_ablation_test_module", SCRIPT_PATH)
extension_audit = _load_module("additional_ablation_audit_test_module", AUDIT_PATH)


def _reference():
    return {
        "model": "Full MHFL-MCA",
        "source": "main_manuscript_stage2_experiment",
        "protocol": "stage2_load_shift",
        "source_load": "0Nm",
        "n_train_per_class": 30,
        "runs": 10,
        "aggregation": ablation.TRIMMED_AGGREGATION,
        "reference_type": "main_manuscript_aggregated_reference",
        "params_m": 7.380,
        "conditions": {
            "2Nm_0dB": {
                "accuracy_mean": 0.9999,
                "accuracy_sd": 0.0002,
                "macro_f1_mean": 0.9999,
                "macro_f1_sd": 0.0002,
            },
            "4Nm_0dB": {
                "accuracy_mean": 0.9925,
                "accuracy_sd": 0.0096,
                "macro_f1_mean": 0.9925,
                "macro_f1_sd": 0.0097,
            },
        },
    }


def _synthetic_raw(seed_count: int = 10) -> pd.DataFrame:
    seeds = [config.GLOBAL_SEED + index for index in range(seed_count)]
    accuracy_offsets = [0.30, -0.20, 0.08, -0.07, 0.06, -0.05, 0.04, -0.03, 0.02, 0.01]
    f1_offsets = [-0.25, 0.28, 0.07, -0.06, 0.05, -0.04, 0.03, -0.02, 0.01, 0.00]
    rows = []
    for variant_index, variant in enumerate(ablation.VARIANT_NAMES):
        for condition_index, condition in enumerate(ablation.CONDITION_NAMES):
            center = 0.55 + 0.02 * variant_index - 0.03 * condition_index
            for seed_index, seed in enumerate(seeds):
                accuracy = center + accuracy_offsets[seed_index]
                macro_f1 = center - 0.01 + f1_offsets[seed_index]
                rows.append(
                    {
                        "variant": variant,
                        "condition": condition,
                        "seed": seed,
                        "params_m": 6.5 + 0.01 * variant_index,
                        "accuracy": accuracy,
                        "macro_precision": macro_f1,
                        "macro_recall": macro_f1,
                        "macro_f1": macro_f1,
                    }
                )
    return pd.DataFrame(rows)


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == function_name)
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_full_repeat_count_is_ten_and_fast_repeat_count_is_unchanged():
    assert config.ABLATION_REPEATS_FULL == 10
    assert config.ABLATION_REPEATS_FAST == 1


def test_fixed_full_existing_and_missing_seed_sets():
    expected = [config.GLOBAL_SEED + index for index in range(10)]
    assert ablation.full_seed_sequence() == expected
    assert ablation.legacy_seed_sequence() == expected[:5]
    assert ablation.missing_extension_seed_sequence() == expected[5:]


def test_04_has_five_variants_and_does_not_train_full():
    assert len(ablation.VARIANT_NAMES) == 5
    assert "full" not in ablation.VARIANT_NAMES
    source = _function_source(SCRIPT_PATH, "run_experiment")
    assert "for variant_name in missing_variants" in source
    assert "VARIANT_NAMES" in source
    assert "LEGACY_FULL_VARIANT" not in source[source.index("for seed in seeds:") :]


def test_ten_seed_variant_raw_target_is_150_and_every_group_has_ten_seeds():
    raw = _synthetic_raw()
    report = ablation.validate_raw_results(raw, ablation.full_seed_sequence(), require_complete=True)
    assert ablation.expected_raw_rows(10) == 150
    assert report["rows"] == 150
    assert len(raw.groupby(["variant", "condition"])) == 15
    assert (raw.groupby(["variant", "condition"])["seed"].nunique() == 10).all()


def test_accuracy_and_macro_f1_are_trimmed_independently_without_paired_delta():
    raw = _synthetic_raw()
    summary = ablation.summarize(raw, full_reference=_reference())
    row = summary[
        (summary["variant"] == "homogeneous_vibration") & (summary["condition"] == "2Nm_0dB")
    ].iloc[0]
    block = raw[
        (raw["variant"] == "homogeneous_vibration") & (raw["condition"] == "2Nm_0dB")
    ]
    assert row["accuracy_mean"] == pytest.approx(trimmed_mean_sd(block["accuracy"])[0])
    assert row["macro_f1_mean"] == pytest.approx(trimmed_mean_sd(block["macro_f1"])[0])
    assert row["reference_mean_difference"] == pytest.approx(row["accuracy_mean"] - 0.9999)
    assert row["comparison_type"] == "unpaired_aggregated_reference_difference"
    assert not any("delta_to_full" in column for column in summary.columns)
    assert not any("p_value" in column or "significance" in column for column in summary.columns)
    assert (summary["seeds"] == 10).all()
    assert (summary["retained_after_trim"] == 8).all()


def test_minus8db_is_source_data_only_and_has_no_full_reference():
    summary = ablation.summarize(_synthetic_raw(), full_reference=_reference())
    noise = summary[summary["condition"] == "4Nm_-8dB"]
    assert len(noise) == 5
    assert (noise["comparison_type"] == "no_stage2_full_reference_for_noise_condition").all()
    assert (noise["reference_mean_difference"] == "not_applicable").all()


def test_missing_seed_blocks_summary():
    incomplete = _synthetic_raw()
    incomplete = incomplete[incomplete["seed"] != config.GLOBAL_SEED]
    with pytest.raises(RuntimeError, match="incomplete"):
        ablation.summarize(incomplete, full_reference=_reference())


def test_exactly_25_legacy_variant_models_are_reused_and_25_slots_extended():
    assert len(ablation.legacy_seed_sequence()) * len(ablation.VARIANT_NAMES) == 25
    assert len(ablation.missing_extension_seed_sequence()) * len(ablation.VARIANT_NAMES) == 25


def test_legacy_full_raw_and_checkpoint_paths_are_never_deleted_or_overwritten():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ablation.LEGACY_FULL_ARCHIVE_NAME == "legacy_supplementary_full_not_used_in_final_hybrid_table"
    assert ablation.LEGACY_RAW_NAME != ablation.VARIANT_RAW_NAME
    assert ".unlink(" not in source
    assert "rmtree(" not in source
    assert "load_or_train_variant_cached" in source
    experiment = _function_source(SCRIPT_PATH, "run_experiment")
    assert "run_dir = out_dir / \"models\" / variant_name" in experiment
    assert "variant_name" in experiment
    assert "LEGACY_RAW_NAME" in experiment


def test_sidecar_fields_are_not_added_to_training_protocol_signature():
    source = _function_source(SCRIPT_PATH, "run_experiment")
    protocol_node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.keyword) and node.arg == "protocol_signature"
    )
    keys = {key.value for key in protocol_node.value.keys if isinstance(key, ast.Constant)}
    assert keys == {"variant", "split_signature", "data_signatures", "evaluation_conditions"}


def test_script_has_no_destructive_or_rebuild_flags():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for token in ('"--force"', '"--rebuild-cache"', '"--rebuild-split"', "rebuild_cache=", "rebuild_split="):
        assert token not in source


def test_legacy_integrity_gate_detects_hash_or_mtime_change():
    baseline = [
        {
            "seed": 20260806,
            "variant": "full",
            "model_manifest": {"path": "manifest", "sha256": "a", "mtime_ns": 1},
            "weights": {"path": "weights", "sha256": "b", "mtime_ns": 2},
            "history": {"path": "history", "sha256": "c", "mtime_ns": 3},
        }
    ]
    current = json.loads(json.dumps(baseline))
    ablation.assert_legacy_inventory_unchanged(baseline, current)
    current[0]["weights"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="legacy"):
        ablation.assert_legacy_inventory_unchanged(baseline, current)


def test_read_only_audit_rejects_changed_checkpoint_hash(tmp_path):
    weights = tmp_path / "model.weights.h5"
    weights.write_bytes(b"current")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"weights_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Weights hash differs"):
        extension_audit.audit_weight_record(manifest, weights)


def test_extension_audit_declares_read_only_pre_phase_and_five_variants():
    source = AUDIT_PATH.read_text(encoding="utf-8")
    assert 'choices=("pre",)' in source
    assert "write_text(" not in source
    assert "write_bytes(" not in source
    assert 'open("w"' not in source
    assert len(extension_audit.VARIANTS) == 5
    assert extension_audit.LEGACY_FULL_VARIANT not in extension_audit.VARIANTS
    assert "training_was_run" in source
    assert "files_written" in source


def test_04_internal_post_gate_requires_150_rows_50_models_and_writes_six_rows(tmp_path):
    raw = _synthetic_raw()
    reference = _reference()
    summary = ablation.summarize(raw, full_reference=reference)
    inventory = [
        {"variant": variant, "seed": seed}
        for variant in ablation.VARIANT_NAMES
        for seed in ablation.full_seed_sequence()
    ]
    archive = {
        "status": ablation.LEGACY_FULL_ARCHIVE_NAME,
        "reused_variant_model_files": [
            {"variant": variant, "seed": seed}
            for variant in ablation.VARIANT_NAMES
            for seed in ablation.legacy_seed_sequence()
        ],
    }
    gate = ablation.validate_additional_ablation_post_gate(
        raw, summary, reference, inventory, archive
    )
    assert gate["status"] == "PASS"
    assert gate["raw_gate"]["rows"] == 150
    assert gate["model_runs"] == 50
    assert gate["reused_first_five_model_runs"] == 25
    assert gate["planned_extension_model_slots"] == 25
    paths = ablation.write_hybrid_candidate_table(summary, reference, tmp_path)
    table = pd.read_csv(paths["csv"])
    assert len(table) == 6
    assert table.iloc[0]["Method"] == "Full MHFL-MCA reference"
    assert not any("-8" in column for column in table.columns)


def test_04_post_gate_failure_does_not_create_candidate_table(tmp_path):
    raw = _synthetic_raw()
    incomplete = raw.iloc[:-1].copy()
    summary = ablation.summarize(raw, full_reference=_reference())
    inventory = [
        {"variant": variant, "seed": seed}
        for variant in ablation.VARIANT_NAMES
        for seed in ablation.full_seed_sequence()
    ]
    archive = {
        "status": ablation.LEGACY_FULL_ARCHIVE_NAME,
        "reused_variant_model_files": [
            {"variant": variant, "seed": seed}
            for variant in ablation.VARIANT_NAMES
            for seed in ablation.legacy_seed_sequence()
        ],
    }
    with pytest.raises(RuntimeError):
        ablation.validate_additional_ablation_post_gate(
            incomplete, summary, _reference(), inventory, archive
        )
    assert not (tmp_path / ablation.HYBRID_CANDIDATE_CSV_NAME).exists()


def test_04_rejects_tampered_published_full_reference_values(tmp_path):
    raw = _synthetic_raw()
    tampered = json.loads(json.dumps(_reference()))
    tampered["conditions"]["2Nm_0dB"]["accuracy_mean"] = 0.95
    summary = ablation.summarize(raw, full_reference=tampered)
    inventory = [
        {"variant": variant, "seed": seed}
        for variant in ablation.VARIANT_NAMES
        for seed in ablation.full_seed_sequence()
    ]
    archive = {
        "status": ablation.LEGACY_FULL_ARCHIVE_NAME,
        "reused_variant_model_files": [
            {"variant": variant, "seed": seed}
            for variant in ablation.VARIANT_NAMES
            for seed in ablation.legacy_seed_sequence()
        ],
    }
    with pytest.raises(RuntimeError, match="published clean-load value"):
        ablation.validate_additional_ablation_post_gate(
            raw, summary, tampered, inventory, archive
        )
    reference_path = tmp_path / "tampered_reference.json"
    reference_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="published clean-load value"):
        config.load_kaist_additional_ablation_full_reference(reference_path)


def test_04_main_records_fail_gate_and_propagates_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation.config, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(ablation.config, "RUN_TAG", "unit_test")
    monkeypatch.setattr(ablation, "parse_args", lambda: SimpleNamespace(mode="full"))

    def fail_run(_args):
        raise RuntimeError("synthetic 04 failure")

    monkeypatch.setattr(ablation, "run_experiment", fail_run)
    with pytest.raises(RuntimeError, match="synthetic 04 failure"):
        ablation.main()
    gate = json.loads(
        (tmp_path / "04_additional_ablation" / "additional_ablation_post_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "FAIL"
    assert gate["final_outputs_authorized"] is False
