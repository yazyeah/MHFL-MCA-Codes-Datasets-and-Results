from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhfl_review import config
from mhfl_review.provenance import sha256_file


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = _load_module("manuscript_gate_test_module", config.SUITE_ROOT / "07_build_manuscript_assets.py")


def _reference_payload():
    return {
        "model": "Full MHFL-MCA",
        "source": "main_manuscript_stage2_experiment",
        "reference_type": "main_manuscript_aggregated_reference",
        "protocol": "stage2_load_shift",
        "n_train_per_class": 30,
        "runs": 10,
        "aggregation": assets.TRIMMED_AGGREGATION,
        "params_m": 7.38,
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


def _write_ablation_bundle(root: Path, reference_path: Path, seed_count: int = 10) -> None:
    out = root / "04_additional_ablation"
    out.mkdir(parents=True)
    rows = []
    for variant_index, variant in enumerate(assets.ABLATION_VARIANTS):
        for condition_index, condition in enumerate(assets.ABLATION_CONDITIONS):
            for seed_index in range(seed_count):
                seed = config.GLOBAL_SEED + seed_index
                value = 0.9 - 0.01 * variant_index - 0.02 * condition_index + 0.0001 * seed_index
                rows.append(
                    {
                        "variant": variant,
                        "condition": condition,
                        "seed": seed,
                        "params_m": 6.5 + variant_index * 0.01,
                        "accuracy": value,
                        "macro_f1": value - 0.001,
                        "macro_precision": value - 0.001,
                        "macro_recall": value - 0.001,
                    }
                )
    raw = pd.DataFrame(rows)
    raw_path = out / assets.ABLATION_VARIANT_RAW_NAME
    raw.to_csv(raw_path, index=False)
    retained = seed_count - 2 if seed_count >= 3 else seed_count
    summary_rows = []
    for variant_index, variant in enumerate(assets.ABLATION_VARIANTS):
        for condition_index, condition in enumerate(assets.ABLATION_CONDITIONS):
            is_clean = condition in assets.ABLATION_TABLE_CONDITIONS
            summary_rows.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "params_m": 6.5 + variant_index * 0.01,
                    "accuracy_mean": 0.9 - 0.01 * variant_index - 0.02 * condition_index,
                    "accuracy_sd": 0.001,
                    "macro_f1_mean": 0.899 - 0.01 * variant_index - 0.02 * condition_index,
                    "macro_f1_sd": 0.001,
                    "seeds": seed_count,
                    "retained_after_trim": retained,
                    "aggregation": assets.TRIMMED_AGGREGATION,
                    "reference_mean_difference": -0.01 if is_clean else "not_applicable",
                    "macro_f1_reference_mean_difference": -0.011 if is_clean else "not_applicable",
                    "comparison_type": (
                        "unpaired_aggregated_reference_difference"
                        if is_clean
                        else "no_stage2_full_reference_for_noise_condition"
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = out / assets.ABLATION_VARIANT_SUMMARY_NAME
    summary.to_csv(summary_path, index=False)
    manifest = {
        "status": "complete_hybrid_reference_candidate",
        "result_mode": assets.HYBRID_REFERENCE_MODE,
        "full_model_training_performed": False,
        "aggregation": assets.TRIMMED_AGGREGATION,
        "raw_gate": {"failed_seeds": []},
        "post_gate": {
            "status": "PASS",
            "summary_rows": 15,
            "retained_after_trim": 8,
            "aggregation": assets.TRIMMED_AGGREGATION,
            "model_runs": 50,
            "reused_first_five_model_runs": 25,
            "planned_extension_model_slots": 25,
            "failed_seeds": [],
            "paired_delta_to_full_generated": False,
            "stage3_minus8db_full_used": False,
        },
        "raw": {"sha256": sha256_file(raw_path)},
        "summary": {"sha256": sha256_file(summary_path)},
        "full_reference": {"sha256": sha256_file(reference_path)},
    }
    manifest_path = out / "additional_ablation_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    post_gate = {
        **manifest["post_gate"],
        "status": "PASS",
        "mode": "full",
        "final_outputs_authorized": True,
        "manifest": {"sha256": sha256_file(manifest_path)},
    }
    (out / "additional_ablation_post_gate.json").write_text(json.dumps(post_gate), encoding="utf-8")


def _write_ablation_main_scope(root: Path) -> None:
    out = root / "04_additional_ablation"
    archive = out / assets.ABLATION_EQUAL_ARCHIVE_DIR_NAME
    archive.mkdir(parents=True)

    source_raw_path = out / assets.ABLATION_VARIANT_RAW_NAME
    source_summary_path = out / assets.ABLATION_VARIANT_SUMMARY_NAME
    source_raw = pd.read_csv(source_raw_path)
    source_summary = pd.read_csv(source_summary_path, keep_default_na=False)

    raw_main = source_raw[source_raw["variant"].isin(assets.ABLATION_MAIN_VARIANTS)].copy()
    summary_main = source_summary[
        source_summary["variant"].isin(assets.ABLATION_MAIN_VARIANTS)
    ].copy()
    raw_main_path = out / assets.ABLATION_MAIN_RAW_NAME
    summary_main_path = out / assets.ABLATION_MAIN_SUMMARY_NAME
    raw_main.to_csv(raw_main_path, index=False)
    summary_main.to_csv(summary_main_path, index=False)

    source_candidate = pd.DataFrame(
        {
            "Method": ["Full MHFL-MCA reference"]
            + [assets.ABLATION_METHOD_LABELS[value] for value in assets.ABLATION_SOURCE_VARIANTS]
        }
    )
    source_candidate_path = out / "manuscript_hybrid_ablation_candidate.csv"
    source_candidate.to_csv(source_candidate_path, index=False)
    source_candidate_tex_path = out / "manuscript_hybrid_ablation_candidate.tex"
    source_candidate_tex_path.write_text("source candidate latex\n", encoding="utf-8")
    candidate_main = source_candidate[source_candidate["Method"] != "Equal weights"].copy()
    candidate_equal = source_candidate[source_candidate["Method"] == "Equal weights"].copy()
    candidate_main_path = out / assets.ABLATION_MAIN_CANDIDATE_NAME
    candidate_main_tex_path = out / assets.ABLATION_MAIN_CANDIDATE_TEX_NAME
    candidate_equal_path = archive / assets.ABLATION_EQUAL_CANDIDATE_NAME
    candidate_main.to_csv(candidate_main_path, index=False)
    candidate_main_tex_path.write_text(
        "\n".join(candidate_main["Method"].astype(str)) + "\n",
        encoding="utf-8",
    )
    candidate_equal.to_csv(candidate_equal_path, index=False)

    equal_path = archive / assets.ABLATION_EQUAL_ARCHIVE_NAME
    pd.concat(
        [
            source_raw[source_raw["variant"] == "equal_weights"].assign(record_type="raw"),
            source_summary[source_summary["variant"] == "equal_weights"].assign(record_type="summary"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(equal_path, index=False)

    archive_records = []
    for source in (
        source_raw_path,
        source_summary_path,
        source_candidate_path,
        source_candidate_tex_path,
        out / "additional_ablation_run_manifest.json",
        out / "additional_ablation_post_gate.json",
    ):
        archived = archive / source.name
        archived.write_bytes(source.read_bytes())
        digest = sha256_file(source)
        archive_records.append(
            {
                "source": str(source.resolve()),
                "archive": str(archived.resolve()),
                "size_bytes": source.stat().st_size,
                "source_sha256": digest,
                "archive_sha256": digest,
            }
        )

    def output_record(path: Path):
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    note_path = out / assets.ABLATION_MAIN_NOTE_NAME
    note_path.write_text(
        "The previously evaluated equal-weight control has not been deleted; "
        "it is retained for supplementary/provenance reporting.\n",
        encoding="utf-8",
    )
    source_paths = (
        source_raw_path,
        source_summary_path,
        source_candidate_path,
        source_candidate_tex_path,
        out / "additional_ablation_run_manifest.json",
        out / "additional_ablation_post_gate.json",
    )
    source_hashes = {str(path.resolve()): sha256_file(path) for path in source_paths}
    manifest = {
        "status": "PASS",
        "operation": "non_destructive_main_text_scope_filter",
        "source_evidence_contract": {
            "status": "PASS",
            "variants": list(assets.ABLATION_SOURCE_VARIANTS),
            "conditions": list(assets.ABLATION_CONDITIONS),
            "raw_rows": 150,
            "summary_groups": 15,
            "seeds_per_group": 10,
            "retained_after_trim": 8,
            "aggregation": assets.TRIMMED_AGGREGATION,
        },
        "main_display_contract": {
            "status": "PASS",
            "variants": list(assets.ABLATION_MAIN_VARIANTS),
            "conditions": list(assets.ABLATION_CONDITIONS),
            "raw_rows": 120,
            "summary_groups": 12,
            "seeds_per_group": 10,
            "compact_table_rows": 5,
        },
        "main_scope_variants": list(assets.ABLATION_MAIN_VARIANTS),
        "excluded_from_main_display": "equal_weights",
        "equal_weight_result_deleted": False,
        "equal_weight_result_archived": True,
        "original_source_files_modified": False,
        "equal_weight_archive_contract": {
            "raw_rows": 30,
            "summary_rows": 3,
            "candidate_rows": 1,
        },
        "claim_boundary": (
            "The equal-weight control remains available in "
            "supplementary/provenance materials."
        ),
        "source_hashes_before": source_hashes,
        "source_hashes_after": source_hashes,
        "archive_records": archive_records,
        "outputs": {
            "raw_main_scope": output_record(raw_main_path),
            "summary_main_scope": output_record(summary_main_path),
            "equal_weights_archive": output_record(equal_path),
            "candidate_main_scope": output_record(candidate_main_path),
            "candidate_main_scope_tex": output_record(candidate_main_tex_path),
            "candidate_equal_weights": output_record(candidate_equal_path),
            "main_scope_note": output_record(note_path),
        },
    }
    (out / assets.ABLATION_MAIN_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _refresh_lowshot_output_record(root: Path, key: str, path: Path, rows=None) -> dict:
    manifest_path = root / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    manifest["outputs"][key] = record
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _anchor_values(mean: float, sd: float) -> np.ndarray:
    retained = mean + sd * np.asarray(
        [-np.sqrt(7.0)] + [1.0 / np.sqrt(7.0)] * 7,
        dtype=np.float64,
    )
    return np.concatenate(([retained.min() - 0.01], retained, [retained.max() + 0.01]))


def _write_lowshot_bundle(root: Path, seed_count: int, aggregation: str) -> None:
    out = root / "05_lowshot_threshold"
    out.mkdir(parents=True)
    hparams_path = root / "uo_optuna_confirmed.json"
    hparams_payload = {
        "status": "CONFIRMED",
        "confirmation_status": "confirmed_from_original_uo_artifact",
        "dropout_vib": 0.1,
        "dropout_aco": 0.2,
        "atten_dim": 256,
        "n_layers_vib": 4,
        "n_layers_aco": 5,
        "lr": 0.001,
        "batch_size": 16,
    }
    hparams_path.write_text(json.dumps(hparams_payload), encoding="utf-8")
    dataset_dir = root / "uo_source_mat"
    dataset_dir.mkdir()
    dataset_files = []
    for index in range(14):
        source_path = dataset_dir / "uo_source_{0:02d}.mat".format(index + 1)
        source_path.write_bytes(("uo-source-{0}".format(index + 1)).encode("ascii"))
        stat = source_path.stat()
        dataset_files.append(
            {
                "path": str(source_path.resolve()),
                "size_bytes": stat.st_size,
                "modification_time_ns": stat.st_mtime_ns,
                "sha256": sha256_file(source_path),
            }
        )
    dataset_settings = {
        "segment_length": 2048,
        "vibration_channel": "vibration",
        "acoustic_channel": "acoustic",
    }
    dataset_content_payload = {"files": dataset_files, "settings": dataset_settings}
    dataset_content_signature = hashlib.sha256(
        json.dumps(
            dataset_content_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    confirmed_hparams = {
        key: hparams_payload[key]
        for key in (
            "dropout_vib", "dropout_aco", "atten_dim", "n_layers_vib",
            "n_layers_aco", "lr", "batch_size",
        )
    }
    confirmed_hparams.update(
        {
            "source_path": str(hparams_path.resolve()),
            "source_sha256": sha256_file(hparams_path),
            "source_evidence_path": str((root / "optuna_evidence.log").resolve()),
            "source_evidence_sha256": "a" * 64,
            "source_summary_path": str((root / "summary_evidence.csv").resolve()),
            "source_summary_sha256": "b" * 64,
            "source_program_path": str((root / "source_program.py").resolve()),
            "source_program_sha256": "c" * 64,
        }
    )
    hparam_signature = assets._canonical_json_sha256(
        {key: value for key, value in confirmed_hparams.items() if not key.startswith("source_")}
    )
    provenance_signature = assets._canonical_json_sha256(
        {
            "script_sha256": sha256_file(config.SUITE_ROOT / "05_lowshot_threshold.py"),
            "optuna_evidence_sha256": confirmed_hparams["source_evidence_sha256"],
            "summary_evidence_sha256": confirmed_hparams["source_summary_sha256"],
        }
    )

    anchor_arrays = {
        (n_train, field): _anchor_values(expected[field + "_mean"], expected[field + "_sd"])
        for n_train, expected in assets.LOWSHOT_TABLE5_ANCHORS.items()
        for field in ("test_accuracy", "test_macro_precision", "test_macro_f1")
    }
    rows = []
    for n_train in assets.LOWSHOT_N_GRID:
        for run_idx in range(1, seed_count + 1):
            seed = 100 * n_train + run_idx
            for variant in assets.LOWSHOT_VARIANTS:
                base = 0.62 + 0.02 * assets.LOWSHOT_N_GRID.index(n_train) + 0.001 * run_idx
                test_accuracy = base + (0.02 if variant == "full" else 0.0)
                test_precision = test_accuracy + 0.004
                test_f1 = test_accuracy - 0.002
                if variant == "full" and n_train in assets.LOWSHOT_TABLE5_ANCHORS and seed_count == 10:
                    test_accuracy = float(anchor_arrays[(n_train, "test_accuracy")][run_idx - 1])
                    test_precision = float(anchor_arrays[(n_train, "test_macro_precision")][run_idx - 1])
                    test_f1 = float(anchor_arrays[(n_train, "test_macro_f1")][run_idx - 1])
                heldout_accuracy = test_accuracy
                train_accuracy = min(1.0, heldout_accuracy + 0.04)
                rows.append(
                    {
                        "variant": variant,
                        "n_train": n_train,
                        "run_idx": run_idx,
                        "seed": seed,
                        "test_accuracy": test_accuracy,
                        "test_macro_precision": test_precision,
                        "test_macro_recall": test_f1 + 0.001,
                        "test_macro_f1": test_f1,
                        "train_accuracy": train_accuracy,
                        "heldout_accuracy": heldout_accuracy,
                        "generalization_gap": train_accuracy - heldout_accuracy,
                        "split_signature": "split-{0}-{1}".format(n_train, seed),
                        "data_signature": dataset_content_signature,
                        "hyperparameter_signature": hparam_signature,
                        "source_type": "new_paper_aligned_training",
                        "train_time_s": 1.0 + 0.01 * run_idx,
                    }
                )
    raw = pd.DataFrame(rows)
    raw_path = out / "lowshot_raw.csv"
    raw.to_csv(raw_path, index=False)

    def output_record(path: Path, rows_count=None):
        record = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if rows_count is not None:
            record["rows"] = rows_count
        return record

    run_artifacts = []
    for row in rows:
        variant = str(row["variant"])
        n_train = int(row["n_train"])
        run_idx = int(row["run_idx"])
        seed = int(row["seed"])
        run_dir = out / "models" / variant / "N{0}_run{1:02d}_seed{2}".format(n_train, run_idx, seed)
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        weights_path = run_dir / "model.weights.h5"
        history_path = run_dir / "history.csv"
        weights_path.write_bytes(("weights-{0}-{1}-{2}".format(variant, n_train, seed)).encode("ascii"))
        pd.DataFrame(
            {
                "loss": np.linspace(1.0, 0.1, 80),
                "accuracy": np.linspace(0.1, 0.9, 80),
                "val_loss": np.linspace(1.1, 0.2, 80),
                "val_accuracy": np.linspace(0.1, 0.85, 80),
                "epoch": np.arange(1, 81),
            }
        ).to_csv(history_path, index=False)
        run_signature = assets._canonical_json_sha256(
            {
                "variant": variant,
                "n_train": n_train,
                "run_idx": run_idx,
                "seed": seed,
                "split_signature": row["split_signature"],
                "data_signature": row["data_signature"],
                "hparam_signature": hparam_signature,
                "provenance_signature": provenance_signature,
                "protocol": "uo_source_aligned_paired_deterministic_extension_v1",
            }
        )
        parameter_count = 5_111_759 if variant == "full" else 4_950_001
        metrics = {
            **row,
            "run_signature": run_signature,
            "weights_path": str(weights_path.resolve()),
            "history_path": str(history_path.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "model_parameter_count": parameter_count,
            "provenance_signature": provenance_signature,
            "weights_sha256": sha256_file(weights_path),
            "history_sha256": sha256_file(history_path),
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        run_artifacts.append(
            {
                "variant": variant,
                "n_train": n_train,
                "run_idx": run_idx,
                "seed": seed,
                "run_signature": run_signature,
                "provenance_signature": provenance_signature,
                "model_parameter_count": parameter_count,
                "metrics": output_record(metrics_path),
                "weights": output_record(weights_path),
                "history": output_record(history_path, 80),
            }
        )

    def statistics(values):
        array = np.sort(np.asarray(values, dtype=np.float64))
        selected = array[1:-1] if array.size >= 3 else array
        return float(selected.mean()), float(selected.std(ddof=0)), float(array.mean()), float(array.std(ddof=0))

    retained = seed_count - 2 if seed_count >= 3 else seed_count
    summary_rows = []
    for (variant, n_train), block in raw.groupby(["variant", "n_train"]):
        row = {
            "variant": variant,
            "n_train": int(n_train),
            "seeds": seed_count,
            "seeds_total": seed_count,
            "retained_after_trim": retained,
            "aggregation": aggregation,
        }
        for metric in assets.LOWSHOT_SUMMARY_METRICS:
            mean, sd, untrimmed_mean, untrimmed_sd = statistics(block[metric])
            row[metric + "_mean"] = mean
            row[metric + "_sd"] = sd
            row[metric + "_untrimmed_mean"] = untrimmed_mean
            row[metric + "_untrimmed_sd"] = untrimmed_sd
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "lowshot_summary.csv"
    summary.to_csv(summary_path, index=False)

    pivot = raw.pivot(index=["n_train", "seed"], columns="variant", values="test_accuracy")
    gains = (pivot["full"] - pivot["no_caim"]).rename("gain").reset_index()
    paired_rows = []
    for n_train, block in gains.groupby("n_train"):
        mean, sd, untrimmed_mean, untrimmed_sd = statistics(block["gain"])
        paired_rows.append(
            {
                "n_train": int(n_train),
                "paired_gain_mean": mean,
                "paired_gain_sd": sd,
                "paired_gain_untrimmed_mean": untrimmed_mean,
                "paired_gain_untrimmed_sd": untrimmed_sd,
                "caim_gain_mean": mean,
                "caim_gain_sd": sd,
                "caim_gain_untrimmed_mean": untrimmed_mean,
                "caim_gain_untrimmed_sd": untrimmed_sd,
                "seeds": seed_count,
                "seeds_total": seed_count,
                "retained_after_trim": retained,
                "aggregation": aggregation,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired_path = out / "caim_paired_summary.csv"
    paired.to_csv(paired_path, index=False)

    thresholds = [
        {
            "variant": variant,
            "criterion": "trimmed test accuracy >= 0.80 and trimmed seed SD <= 0.10",
            "first_empirical_n": 5,
            "claim_limit": "Smallest evaluated N satisfying the criterion; not a universal theoretical threshold.",
        }
        for variant in assets.LOWSHOT_VARIANTS
    ]
    threshold_path = out / "operational_thresholds.json"
    threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")

    anchor_records = {}
    for n_train, expected in assets.LOWSHOT_TABLE5_ANCHORS.items():
        fields = {
            field: {
                "expected": value,
                "actual": value,
                "expected_4dp": "{0:.4f}".format(value),
                "actual_4dp": "{0:.4f}".format(value),
                "passed": True,
            }
            for field, value in expected.items()
        }
        anchor_records[str(n_train)] = {"passed": True, "fields": fields}
    anchor = {
        "status": "PASS",
        "required_for_manuscript": True,
        "comparison_precision": "exact after four-decimal manuscript rounding",
        "anchors": anchor_records,
        "final_assets_authorized": True,
    }
    anchor_path = out / "lowshot_anchor_gate.json"
    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")

    expected_seed_map = {str(key): list(value[:seed_count]) for key, value in assets.LOWSHOT_SEED_MAP.items()}
    post_gate = {
        "status": "PASS",
        "mode": "full",
        "final_outputs_authorized": True,
        "raw_rows": seed_count * 14,
        "summary_rows": 14,
        "paired_gain_rows": 7,
        "variants": list(assets.LOWSHOT_VARIANTS),
        "n_grid": list(assets.LOWSHOT_N_GRID),
        "runs_per_group": seed_count,
        "seed_map": expected_seed_map,
        "retained_after_trim": retained,
        "aggregation": aggregation,
        "duplicates": 0,
        "nan_or_inf": 0,
        "failed_runs": 0,
        "failed_seeds": [],
        "matched_split_pairs": seed_count * 7,
        "summary_recomputed_from_raw": True,
        "paired_gain_aligned_by_seed_before_trim": True,
        "operational_threshold_recomputed_from_trimmed_statistics": True,
        "anchor_gate_pass": True,
    }
    post_path = out / "lowshot_post_gate.json"
    post_path.write_text(json.dumps(post_gate), encoding="utf-8")
    execution_state = {
        "status": "PASS",
        "mode": "full",
        "run_tag": "test_full",
        "protocol": "uo_source_aligned_paired_deterministic_extension_v1",
        "completed_run_artifacts": len(run_artifacts),
        "final_outputs_authorized": True,
        "finished_unix_time": 1.0,
    }
    execution_state_path = out / "lowshot_execution_state.json"
    execution_state_path.write_text(json.dumps(execution_state), encoding="utf-8")

    figure_bundle = out / "lowshot_evidence_bundle"
    figure_bundle.mkdir()
    figure_paths = {
        "figure_png": figure_bundle / "lowshot_evidence.png",
        "figure_pdf": figure_bundle / "lowshot_evidence.pdf",
        "figure_svg": figure_bundle / "lowshot_evidence.svg",
        "figure_tiff": figure_bundle / "lowshot_evidence.tiff",
    }
    for path in figure_paths.values():
        path.write_bytes(b"paper-aligned-test-figure")
    contract = {
        "figure_id": "extreme-lowshot-paper-aligned",
        "source_data": [str(summary_path.resolve()), str(paired_path.resolve())],
        "replicate_unit": "paper seed schedule 100*N+run_idx",
        "center_statistic": "trimmed mean after removing one highest and one lowest result",
        "spread_definition": "population SD across the eight retained runs",
        "panel_y_labels": ["Held-out accuracy", "Train–held-out gap", "Paired CAIM accuracy gain"],
        "evaluation_set_role": "Held-out data are reused for curves and final evaluation; no independent validation/test partition is claimed.",
        "anchor_gate": anchor,
        "claim_limits": ["Empirical protocol-specific sensitivity point, not a universal theorem"],
    }
    contract_path = figure_bundle / "figure_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    def output_record(path: Path, rows_count=None):
        record = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if rows_count is not None:
            record["rows"] = rows_count
        return record

    outputs = {
        "raw": output_record(raw_path, len(raw)),
        "summary": output_record(summary_path, len(summary)),
        "paired_summary": output_record(paired_path, len(paired)),
        "operational_thresholds": output_record(threshold_path),
        "anchor_gate": output_record(anchor_path),
        "post_gate": output_record(post_path),
        "hparams_config": output_record(hparams_path),
        "execution_state": output_record(execution_state_path),
        "figure_contract": output_record(contract_path),
    }
    outputs.update({key: output_record(path) for key, path in figure_paths.items()})
    manifest = {
        "status": "PASS",
        "mode": "full",
        "protocol": "uo_source_aligned_paired_deterministic_extension_v1",
        "protocol_alignment_scope": "Source-aligned split, topology, hparams, optimizer, epochs, held-out evaluation, and final weights.",
        "random_initialization_boundary": "Historical initializer state is unavailable; paired deterministic initialization is used for this reviewer extension.",
        "split_protocol": "first N train / all remaining held-out; same held-out used for validation curves and final evaluation",
        "seed_schedule": "seed = 100*N + run_idx",
        "epochs": 80,
        "optimizer": {
            "name": "Adamax",
            "learning_rate": hparams_payload["lr"],
            "batch_size": hparams_payload["batch_size"],
        },
        "gradient_clipping": None,
        "early_stopping": False,
        "final_epoch_weights": True,
        "hparams": confirmed_hparams,
        "model_parameter_count": 5_111_759,
        "dataset_provenance": {
            "source_file_count": 14,
            "content_signature": dataset_content_signature,
            "settings": dataset_settings,
            "files": dataset_files,
        },
        "raw_gate": {
            "status": "PASS",
            "rows": len(raw),
            "n_grid": list(assets.LOWSHOT_N_GRID),
            "runs_per_n": seed_count,
            "seed_map": expected_seed_map,
            "duplicates": 0,
            "nan_or_inf": 0,
            "matched_split_pairs": seed_count * 7,
        },
        "anchor_gate": anchor,
        "paper_anchors": {str(key): value for key, value in assets.LOWSHOT_TABLE5_ANCHORS.items()},
        "aggregation": aggregation,
        "paired_by_seed_before_trim": True,
        "failed_runs": 0,
        "failed_seeds": [],
        "run_artifact_count": len(run_artifacts),
        "run_artifacts": run_artifacts,
        "post_gate": post_gate,
        "outputs": outputs,
    }
    (out / "lowshot_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_efficiency_bundle(root: Path) -> Path:
    out = root / "03_efficiency"
    out.mkdir(parents=True)
    payload = {
        "model": "MHFL-MCA",
        "protocol_checkpoint": "stage2",
        "device": "gpu",
        "latency_mean_ms": 4.4,
        "latency_median_ms": 4.5,
        "latency_p25_ms": 3.8,
        "latency_p75_ms": 4.9,
        "latency_p95_ms": 5.1,
        "throughput_samples_per_s": 226.0,
        "gpu_allocator_current_mb": 28.0,
        "gpu_allocator_peak_mb": 41.0,
        "trainable_params": 7_380_173,
        "trainable_params_m": 7.380173,
        "flops": 243_171_883,
        "flops_g": 0.243171883,
        "macs_estimated": 121_585_941,
        "macs_g_estimated": 0.1215859415,
        "weights_size_mb": 28.2,
        "savedmodel_size_mb": 28.7,
        "profiler_architecture": "isolated_cpu_flops_gpu_runtime",
        "checkpoint_sha256": "a" * 64,
        "flops_worker_exit_code": 0,
        "runtime_worker_exit_code": 0,
    }
    path = out / "efficiency_profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_deep_reference(root: Path) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "main_manuscript_tables.tex"
    source.write_text("Table 5; Table 9; Table 10", encoding="utf-8")
    tables = {}
    for table_name, (case, load, source_table) in assets.DEEP_REFERENCE_TABLES.items():
        rows = []
        for model_index, model in enumerate(assets.DEEP_REFERENCE_MODELS):
            for n_train in assets.DEEP_REFERENCE_N_GRID:
                value = 0.70 + 0.01 * model_index + 0.001 * n_train
                rows.append(
                    {
                        "model": model,
                        "n_train": n_train,
                        "accuracy_mean": value,
                        "accuracy_sd": 0.01,
                        "macro_f1_mean": value - 0.001,
                        "macro_f1_sd": 0.011,
                        "macro_precision_mean": value - 0.002,
                        "macro_precision_sd": 0.012,
                        "macro_recall_mean": value - 0.003,
                        "macro_recall_sd": 0.013,
                    }
                )
        tables[table_name] = {
            "source_table": source_table,
            "case": case,
            "load": load,
            "protocol": "few_shot" if case == "UO" else "stage2_load_shift",
            "n_values": list(assets.DEEP_REFERENCE_N_GRID),
            "rows": rows,
        }
    payload = {
        "schema_version": 1,
        "reference_type": "main_manuscript_aggregated_reference",
        "runs": 10,
        "aggregation": assets.TRIMMED_AGGREGATION,
        "benchmark_scope": "main-manuscript deep-model aggregated references; no deep-model training or per-seed pairing",
        "source_files": [
            {
                "source_path": str(source),
                "sha256": sha256_file(source),
                "source_role": "original manuscript table source",
            }
        ],
        "tables": tables,
    }
    path = root / "main_manuscript_deep_reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _write_traditional_bundle(root: Path, reference_path: Path, reference: dict) -> None:
    out = root / "06_traditional_baselines"
    out.mkdir(parents=True)
    rows = []

    def append(case, load, n_train, seed, snr_db, method, method_index):
        seed_index = seed - config.GLOBAL_SEED
        base = 0.60 + 0.01 * method_index + 0.002 * seed_index + 0.001 * int(n_train)
        rows.append(
            {
                "case": case,
                "load": load,
                "n_train": n_train,
                "snr_db": snr_db,
                "method": method,
                "seed": seed,
                "accuracy": base,
                "macro_f1": base - 0.001,
                "macro_precision": base - 0.002,
                "macro_recall": base - 0.003,
            }
        )

    for n_train in assets.DEEP_REFERENCE_N_GRID:
        for seed in assets.TARGET_SEEDS:
            for method_index, method in enumerate(assets.TRADITIONAL_METHODS):
                append("UO", "held-out", n_train, seed, np.nan, method, method_index)
                for load in ("2Nm", "4Nm"):
                    append("KAIST", load, n_train, seed, 0.0, method, method_index)
    for seed in assets.TARGET_SEEDS:
        for method_index, method in enumerate(assets.TRADITIONAL_METHODS):
            for load in ("2Nm", "4Nm"):
                for snr_db in (0.0, -4.0, -8.0):
                    append("KAIST-noise", load, 30, seed, snr_db, method, method_index)
    raw = pd.DataFrame(rows)
    raw_path = out / "traditional_baselines_raw.csv"
    raw.to_csv(raw_path, index=False)

    metric_names = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    summary_rows = []
    for keys, block in raw.groupby(list(assets.TRADITIONAL_GROUP_COLUMNS), dropna=False, sort=False):
        row = dict(zip(assets.TRADITIONAL_GROUP_COLUMNS, keys))
        row.update({"seeds_total": 10, "retained_after_trim": 8, "aggregation": assets.TRIMMED_AGGREGATION})
        for metric in metric_names:
            mean, sd, untrimmed_mean, untrimmed_sd = assets._trimmed_statistics(block[metric])
            row[metric + "_mean"] = mean
            row[metric + "_sd"] = sd
            row[metric + "_untrimmed_mean"] = untrimmed_mean
            row[metric + "_untrimmed_sd"] = untrimmed_sd
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "traditional_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)

    candidate_rows = []
    scope = (
        "interpretable time/frequency-feature SVM reference baselines; "
        "hyperparameters frozen after one development tuning; "
        "not exact CSC/GJO-OMP reproductions"
    )
    for table in reference["tables"].values():
        for row in table["rows"]:
            candidate_rows.append(
                {
                    "source_type": "main_manuscript_deep_reference",
                    "reference_type": "main_manuscript_aggregated_reference",
                    "source_table": table["source_table"],
                    "case": table["case"],
                    "load": table["load"],
                    "n_train": row["n_train"],
                    "model": row["model"],
                    **{name: row[name] for name in (
                        "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
                        "macro_precision_mean", "macro_precision_sd", "macro_recall_mean", "macro_recall_sd",
                    )},
                    "runs": 10,
                    "retained_after_trim": 8,
                    "aggregation": assets.TRIMMED_AGGREGATION,
                    "benchmark_scope": scope,
                    "comparison_type": "descriptive_aggregated_reference_only",
                }
            )
    clean = summary[summary["case"].isin(["UO", "KAIST"])]
    for _, row in clean.iterrows():
        candidate_rows.append(
            {
                "source_type": "new_traditional_baseline",
                "reference_type": "new_ten_seed_trimmed_result",
                "source_table": "Experiment 06",
                "case": row["case"],
                "load": row["load"],
                "n_train": int(row["n_train"]),
                "model": row["method"],
                **{name: row[name] for name in (
                    "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
                    "macro_precision_mean", "macro_precision_sd", "macro_recall_mean", "macro_recall_sd",
                )},
                "runs": 10,
                "retained_after_trim": 8,
                "aggregation": assets.TRIMMED_AGGREGATION,
                "benchmark_scope": scope,
                "comparison_type": "descriptive_cross_method_benchmark_only",
            }
        )
    candidate = pd.DataFrame(candidate_rows, columns=assets.TRADITIONAL_CANDIDATE_COLUMNS)
    candidate_path = out / "manuscript_candidate_rows.csv"
    candidate.to_csv(candidate_path, index=False)
    scope_path = out / "benchmark_scope.txt"
    scope_path.write_text(scope, encoding="utf-8")
    model_audits = {
        name: {"selected": {"C": 1.0 + index, "gamma": "scale"}}
        for index, name in enumerate(("early", "mod1", "mod2"))
    }
    split_claim = (
        "The UO paper protocol is segment-disjoint but may share source recordings "
        "between train and test; it does not establish recording-disjoint generalization."
    )
    tuning = {
        "protocol": "single predeclared development tuning, frozen across all evaluation N and seeds",
        "tuning_seed": 20260805,
        "tuning_n": 15,
        "evaluation_seeds": list(assets.TARGET_SEEDS),
        "test_or_target_data_used_for_selection": False,
        "frozen_across_all_reported_N_and_seeds": True,
        "result_acceptance_rule": (
            "All complete finite outputs are retained regardless of whether a baseline "
            "is above or below MHFL-MCA."
        ),
        "cases": {
            "UO": {
                "split_audit": {
                    "exact_sample_overlap": 0,
                    "shared_recordings": ["recording-a"],
                    "claim_limit": split_claim,
                },
                "models": model_audits,
            },
            "KAIST": {"models": model_audits},
        },
        "uo_evaluation_split_audits": [
            {
                "case": "UO",
                "seed": seed,
                "n_train": n_train,
                "exact_sample_overlap": 0,
                "shared_recordings": ["recording-a"],
                "claim_limit": split_claim,
            }
            for seed in assets.TARGET_SEEDS
            for n_train in assets.DEEP_REFERENCE_N_GRID
        ],
    }
    tuning_path = out / "traditional_baseline_tuning_audit.json"
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    manifest = {
        "status": "PASS",
        "mode": "full",
        "aggregation": assets.TRIMMED_AGGREGATION,
        "metrics_trimmed_independently": True,
        "failed_seeds": [],
        "failed_groups": [],
        "channel_gate": {
            "current_channel": "cDAQ9185-1F486B5Mod2/ai0",
            "vibration_column": 0,
            "current_fallback": False,
        },
        "deep_model_training_performed": False,
        "candidate_scope": {
            "included_cases": ["UO", "KAIST"],
            "excluded_cases": ["KAIST-noise"],
            "expected_rows": 144,
            "deep_reference_rows": 108,
            "traditional_rows": 36,
            "n_values": list(assets.DEEP_REFERENCE_N_GRID),
        },
        "post_gate": {
            "status": "PASS",
            "raw_rows": 480,
            "summary_rows": 48,
            "candidate_rows": 144,
            "failed_groups": [],
            "deep_model_training_performed": False,
        },
        "deep_reference_gate": {"status": "PASS", "rows": 108},
        "raw": {"sha256": sha256_file(raw_path)},
        "summary": {"sha256": sha256_file(summary_path)},
        "manuscript_candidate": {"sha256": sha256_file(candidate_path)},
        "deep_reference": {"sha256": sha256_file(reference_path)},
        "svm_hyperparameter_protocol": {
            "selection": "single predeclared development tuning",
            "tuning_n": 15,
            "tuning_seed": 20260805,
            "evaluation_seeds": list(assets.TARGET_SEEDS),
            "frozen_across_all_reported_points": True,
            "test_or_target_data_used_for_selection": False,
        },
        "tuning_audit": {
            "path": str(tuning_path.resolve()),
            "sha256": sha256_file(tuning_path),
        },
    }
    (out / "traditional_baselines_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_07_efficiency_gate_reads_json_and_validates_all_deployment_metrics(tmp_path: Path):
    path = _write_efficiency_bundle(tmp_path)
    gate = assets.gate_efficiency(tmp_path)
    assert gate["usable"] is True
    checked = {item["name"] for item in gate["checks"]}
    assert {"positive_" + name for name in assets.EFFICIENCY_POSITIVE_FIELDS}.issubset(checked)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["latency_mean_ms"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    failed = assets.gate_efficiency(tmp_path)
    assert failed["usable"] is False
    assert "positive_latency_mean_ms" in failed["post_audit"]["failed_checks"]


def test_07_efficiency_gate_rejects_parameter_or_worker_mismatch(tmp_path: Path):
    path = _write_efficiency_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trainable_params"] = 7_380_172
    payload["runtime_worker_exit_code"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    gate = assets.gate_efficiency(tmp_path)
    assert gate["usable"] is False
    assert {"trainable_params", "runtime_worker_exit_code"}.issubset(gate["post_audit"]["failed_checks"])


def test_07_accepts_valid_lowshot_and_recomputes_seed_paired_gain(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    figure_paths = assets._lowshot_figure_paths(tmp_path / "05_lowshot_threshold")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is True
    assert figure_paths["contract"].name == "figure_contract.json"
    assert all(path.parent.name == "lowshot_evidence_bundle" for path in figure_paths.values())
    passed = {item["name"] for item in gate["checks"] if item["passed"]}
    assert "summary_recomputed_from_raw" in passed
    assert "paired_gain_recomputed_from_seed_pairs" in passed
    assert "internal_post_gate" in passed


def test_07_requires_the_canonical_lowshot_figure_bundle_layout(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    base = tmp_path / "05_lowshot_threshold"
    bundle = base / "lowshot_evidence_bundle"
    for source in bundle.iterdir():
        if source.is_file():
            (base / source.name).write_bytes(source.read_bytes())
    (base / "lowshot_evidence.contract.json").write_bytes((bundle / "figure_contract.json").read_bytes())
    (bundle / "figure_contract.json").unlink()

    gate = assets.gate_lowshot(tmp_path)

    assert gate["usable"] is False
    assert "figure_bundle" in gate["post_audit"]["failed_checks"]


def test_07_rejects_paired_gain_not_recomputed_from_seed_pairs(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    paired_path = tmp_path / "05_lowshot_threshold" / "caim_paired_summary.csv"
    paired = pd.read_csv(paired_path)
    paired.loc[0, "paired_gain_mean"] += 0.1
    paired.loc[0, "caim_gain_mean"] += 0.1
    paired.to_csv(paired_path, index=False)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["paired_summary"].update(
        {
            "sha256": sha256_file(paired_path),
            "size_bytes": paired_path.stat().st_size,
            "rows": len(paired),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "paired_gain_recomputed_from_seed_pairs" in gate["post_audit"]["failed_checks"]


def test_07_lowshot_uses_n_specific_paper_seeds_without_changing_04_06_seeds(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is True
    assert assets.LOWSHOT_SEED_MAP[1] == tuple(range(101, 111))
    assert assets.LOWSHOT_SEED_MAP[10] == tuple(range(1001, 1011))
    assert len(set().union(*(set(values) for values in assets.LOWSHOT_SEED_MAP.values()))) == 70
    assert assets.TARGET_SEEDS == tuple(config.GLOBAL_SEED + index for index in range(10))


def test_07_rejects_wrong_lowshot_seed_formula_or_unpaired_split(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    base = tmp_path / "05_lowshot_threshold"
    raw_path = base / "lowshot_raw.csv"
    raw = pd.read_csv(raw_path)
    pair = (raw["n_train"] == 1) & (raw["seed"] == 101)
    raw.loc[pair, "seed"] = 199
    raw.to_csv(raw_path, index=False)
    _refresh_lowshot_output_record(tmp_path, "raw", raw_path, len(raw))
    seed_gate = assets.gate_lowshot(tmp_path)
    assert seed_gate["usable"] is False
    assert "paper_seed_formula" in seed_gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "split_case", 10, assets.TRIMMED_AGGREGATION)
    split_base = tmp_path / "split_case" / "05_lowshot_threshold"
    split_raw_path = split_base / "lowshot_raw.csv"
    split_raw = pd.read_csv(split_raw_path)
    target = (split_raw["n_train"] == 2) & (split_raw["seed"] == 201) & (split_raw["variant"] == "no_caim")
    split_raw.loc[target, "split_signature"] = "mismatched-split"
    split_raw.to_csv(split_raw_path, index=False)
    _refresh_lowshot_output_record(tmp_path / "split_case", "raw", split_raw_path, len(split_raw))
    split_gate = assets.gate_lowshot(tmp_path / "split_case")
    assert split_gate["usable"] is False
    assert "matched_splits" in split_gate["post_audit"]["failed_checks"]


def test_07_rejects_anchor_claim_not_backed_by_table5_summary(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    base = tmp_path / "05_lowshot_threshold"
    summary_path = base / "lowshot_summary.csv"
    summary = pd.read_csv(summary_path)
    row = (summary["variant"] == "full") & (summary["n_train"] == 5)
    summary.loc[row, "test_accuracy_mean"] += 0.001
    summary.to_csv(summary_path, index=False)
    _refresh_lowshot_output_record(tmp_path, "summary", summary_path, len(summary))
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "table5_anchor_recomputed_from_summary" in gate["post_audit"]["failed_checks"]


def test_07_requires_authorized_anchor_and_matching_manifest_copy(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    anchor_path = tmp_path / "05_lowshot_threshold" / "lowshot_anchor_gate.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["final_assets_authorized"] = False
    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
    _refresh_lowshot_output_record(tmp_path, "anchor_gate", anchor_path)
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "anchor_gate" in gate["post_audit"]["failed_checks"]
    assert "paper_aligned_manifest" in gate["post_audit"]["failed_checks"]


def test_07_rejects_validation_wording_or_incomplete_lowshot_bundle(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    base = tmp_path / "05_lowshot_threshold"
    contract_path = base / "lowshot_evidence_bundle" / "figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["panel_y_labels"][1] = "Train–validation gap"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    _refresh_lowshot_output_record(tmp_path, "figure_contract", contract_path)
    wording_gate = assets.gate_lowshot(tmp_path)
    assert wording_gate["usable"] is False
    assert "figure_contract" in wording_gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "missing_svg", 10, assets.TRIMMED_AGGREGATION)
    svg_path = tmp_path / "missing_svg" / "05_lowshot_threshold" / "lowshot_evidence_bundle" / "lowshot_evidence.svg"
    svg_path.unlink()
    bundle_gate = assets.gate_lowshot(tmp_path / "missing_svg")
    assert bundle_gate["usable"] is False
    assert "figure_bundle" in bundle_gate["post_audit"]["failed_checks"]


@pytest.mark.parametrize(
    ("filename", "failed_check"),
    [
        ("lowshot_raw.csv", "raw_rows"),
        ("lowshot_summary.csv", "summary_groups"),
        ("caim_paired_summary.csv", "paired_summary_protocol"),
    ],
)
def test_07_enforces_exact_lowshot_140_14_7_contract(tmp_path: Path, filename: str, failed_check: str):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    base = tmp_path / "05_lowshot_threshold"
    path = base / filename
    frame = pd.read_csv(path).iloc[:-1].copy()
    frame.to_csv(path, index=False)
    key = {
        "lowshot_raw.csv": "raw",
        "lowshot_summary.csv": "summary",
        "caim_paired_summary.csv": "paired_summary",
    }[filename]
    _refresh_lowshot_output_record(tmp_path, key, path, len(frame))
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert failed_check in gate["post_audit"]["failed_checks"]


def test_07_rejects_lowshot_raw_gate_or_output_hash_mismatch(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_gate"]["matched_split_pairs"] = 69
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    raw_gate = assets.gate_lowshot(tmp_path)
    assert raw_gate["usable"] is False
    assert "paper_aligned_manifest" in raw_gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "hash_case", 10, assets.TRIMMED_AGGREGATION)
    threshold_path = tmp_path / "hash_case" / "05_lowshot_threshold" / "operational_thresholds.json"
    threshold_path.write_text(threshold_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    hash_gate = assets.gate_lowshot(tmp_path / "hash_case")
    assert hash_gate["usable"] is False
    assert "manifest_operational_thresholds_hash" in hash_gate["post_audit"]["failed_checks"]


def test_07_rejects_lowshot_optimizer_or_initializer_boundary_mismatch(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optimizer"]["learning_rate"] *= 2.0
    manifest["random_initialization_boundary"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "paper_aligned_manifest" in gate["post_audit"]["failed_checks"]


def test_07_rejects_lowshot_nan_without_crashing(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    raw_path = tmp_path / "05_lowshot_threshold" / "lowshot_raw.csv"
    raw = pd.read_csv(raw_path)
    raw.loc[0, "test_accuracy"] = np.nan
    raw.to_csv(raw_path, index=False)
    _refresh_lowshot_output_record(tmp_path, "raw", raw_path, len(raw))
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "finite_metrics" in gate["post_audit"]["failed_checks"]
    assert "summary_recomputed_from_raw" in gate["post_audit"]["failed_checks"]


def test_07_verifies_all_14_lowshot_dataset_source_files(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is True
    passed = {item["name"] for item in gate["checks"] if item["passed"]}
    assert "dataset_provenance" in passed
    assert "raw_dataset_content_signature" in passed
    assert len([name for name in gate["sources"] if name.startswith("uo_source_mat_")]) == 14


def test_07_rejects_changed_lowshot_dataset_file_metadata_or_content(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(manifest["dataset_provenance"]["files"][0]["path"])
    original = source_path.read_bytes()
    source_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "dataset_provenance" in gate["post_audit"]["failed_checks"]


def test_07_rejects_unrecomputable_or_unbound_lowshot_dataset_signature(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_provenance"]["content_signature"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    signature_gate = assets.gate_lowshot(tmp_path)
    assert signature_gate["usable"] is False
    assert "dataset_provenance" in signature_gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "raw_binding", 10, assets.TRIMMED_AGGREGATION)
    raw_path = tmp_path / "raw_binding" / "05_lowshot_threshold" / "lowshot_raw.csv"
    raw = pd.read_csv(raw_path)
    raw.loc[0, "data_signature"] = "f" * 64
    raw.to_csv(raw_path, index=False)
    _refresh_lowshot_output_record(tmp_path / "raw_binding", "raw", raw_path, len(raw))
    raw_gate = assets.gate_lowshot(tmp_path / "raw_binding")
    assert raw_gate["usable"] is False
    assert "raw_dataset_content_signature" in raw_gate["post_audit"]["failed_checks"]


def test_07_rejects_incomplete_lowshot_dataset_inventory(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_provenance"]["source_file_count"] = 13
    manifest["dataset_provenance"]["files"] = manifest["dataset_provenance"]["files"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "dataset_provenance" in gate["post_audit"]["failed_checks"]


def test_07_requires_completed_hash_bound_lowshot_execution_state(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    state_path = tmp_path / "05_lowshot_threshold" / "lowshot_execution_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed_run_artifacts"] = 139
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _refresh_lowshot_output_record(tmp_path, "execution_state", state_path)
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "execution_state" in gate["post_audit"]["failed_checks"]


def test_07_rejects_missing_or_changed_lowshot_run_artifact(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = Path(manifest["run_artifacts"][0]["weights"]["path"])
    weights_path.write_bytes(weights_path.read_bytes() + b"tampered")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert "run_artifacts" in gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "missing_record", 10, assets.TRIMMED_AGGREGATION)
    missing_manifest_path = tmp_path / "missing_record" / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    missing_manifest = json.loads(missing_manifest_path.read_text(encoding="utf-8"))
    missing_manifest["run_artifacts"] = missing_manifest["run_artifacts"][:-1]
    missing_manifest["run_artifact_count"] = 139
    missing_manifest_path.write_text(json.dumps(missing_manifest), encoding="utf-8")
    missing_gate = assets.gate_lowshot(tmp_path / "missing_record")
    assert missing_gate["usable"] is False
    assert "run_artifacts" in missing_gate["post_audit"]["failed_checks"]


def test_07_rejects_metrics_identity_or_non_80_row_history(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, assets.TRIMMED_AGGREGATION)
    manifest_path = tmp_path / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["run_artifacts"][0]
    metrics_path = Path(artifact["metrics"]["path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["seed"] = 999
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    artifact["metrics"].update(
        {"sha256": sha256_file(metrics_path), "size_bytes": metrics_path.stat().st_size}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    identity_gate = assets.gate_lowshot(tmp_path)
    assert identity_gate["usable"] is False
    assert "run_artifacts" in identity_gate["post_audit"]["failed_checks"]

    _write_lowshot_bundle(tmp_path / "history_case", 10, assets.TRIMMED_AGGREGATION)
    history_manifest_path = tmp_path / "history_case" / "05_lowshot_threshold" / "lowshot_run_manifest.json"
    history_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    history_artifact = history_manifest["run_artifacts"][0]
    history_path = Path(history_artifact["history"]["path"])
    history = pd.read_csv(history_path).iloc[:-1]
    history.to_csv(history_path, index=False)
    history_artifact["history"].update(
        {"sha256": sha256_file(history_path), "size_bytes": history_path.stat().st_size, "rows": 80}
    )
    history_metrics_path = Path(history_artifact["metrics"]["path"])
    history_metrics = json.loads(history_metrics_path.read_text(encoding="utf-8"))
    history_metrics["history_sha256"] = sha256_file(history_path)
    history_metrics_path.write_text(json.dumps(history_metrics), encoding="utf-8")
    history_artifact["metrics"].update(
        {"sha256": sha256_file(history_metrics_path), "size_bytes": history_metrics_path.stat().st_size}
    )
    history_manifest_path.write_text(json.dumps(history_manifest), encoding="utf-8")
    history_gate = assets.gate_lowshot(tmp_path / "history_case")
    assert history_gate["usable"] is False
    assert "run_artifacts" in history_gate["post_audit"]["failed_checks"]


def test_07_accepts_hybrid_deep_reference_plus_two_svm_methods_without_deep_raw(tmp_path: Path):
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is True
    assert not (tmp_path / "06_traditional_baselines" / "deep_model_raw.csv").exists()
    candidate = pd.read_csv(tmp_path / "06_traditional_baselines" / "manuscript_candidate_rows.csv")
    assert len(candidate) == 144
    assert candidate["source_type"].value_counts().to_dict() == {
        "main_manuscript_deep_reference": 108,
        "new_traditional_baseline": 36,
    }
    passed = set(gate["post_audit"]["failed_checks"])
    assert passed == set()


def test_07_rejects_evaluation_seed_or_target_data_in_svm_tuning(tmp_path: Path):
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    out = tmp_path / "06_traditional_baselines"
    tuning_path = out / "traditional_baseline_tuning_audit.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    tuning["tuning_seed"] = config.GLOBAL_SEED
    tuning["test_or_target_data_used_for_selection"] = True
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    manifest_path = out / "traditional_baselines_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tuning_audit"]["sha256"] = sha256_file(tuning_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is False
    assert "frozen_tuning_protocol" in gate["post_audit"]["failed_checks"]


def test_07_rejects_uo_overlap_or_recording_disjoint_overclaim(tmp_path: Path):
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    out = tmp_path / "06_traditional_baselines"
    tuning_path = out / "traditional_baseline_tuning_audit.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    tuning["uo_evaluation_split_audits"][0]["exact_sample_overlap"] = 1
    tuning["uo_evaluation_split_audits"][1]["claim_limit"] = (
        "The split establishes recording-disjoint generalization."
    )
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    manifest_path = out / "traditional_baselines_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tuning_audit"]["sha256"] = sha256_file(tuning_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is False
    failed = set(gate["post_audit"]["failed_checks"])
    assert "uo_evaluation_split_audits" in failed
    assert "uo_shared_recording_claim_boundary" in failed


def test_07_rejects_tuning_audit_hash_or_manifest_protocol_mismatch(tmp_path: Path):
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    out = tmp_path / "06_traditional_baselines"
    tuning_path = out / "traditional_baseline_tuning_audit.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    tuning["audit_note"] = "tampered after manifest"
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    manifest_path = out / "traditional_baselines_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["svm_hyperparameter_protocol"]["frozen_across_all_reported_points"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is False
    failed = set(gate["post_audit"]["failed_checks"])
    assert "manifest_frozen_tuning_protocol" in failed
    assert "tuning_audit_hash" in failed


def test_07_rejects_noise_or_paired_claim_in_hybrid_traditional_candidate(tmp_path: Path):
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    candidate_path = tmp_path / "06_traditional_baselines" / "manuscript_candidate_rows.csv"
    candidate = pd.read_csv(candidate_path)
    candidate.loc[0, "case"] = "KAIST-noise"
    candidate["p_value"] = 0.05
    candidate.to_csv(candidate_path, index=False)
    manifest_path = tmp_path / "06_traditional_baselines" / "traditional_baselines_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manuscript_candidate"]["sha256"] = sha256_file(candidate_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is False
    failed = set(gate["post_audit"]["failed_checks"])
    assert "candidate_clean_scope" in failed
    assert "candidate_descriptive_only" in failed


def test_07_rejects_missing_deep_reference_or_deep_training_claim(tmp_path: Path):
    missing = assets.gate_deep_reference(tmp_path / "missing.json")
    assert missing["usable"] is False
    reference_path, reference = _write_deep_reference(tmp_path)
    _write_traditional_bundle(tmp_path, reference_path, reference)
    manifest_path = tmp_path / "06_traditional_baselines" / "traditional_baselines_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deep_model_training_performed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_traditional(tmp_path, reference_path)
    assert gate["usable"] is False
    assert "no_deep_model_training" in gate["post_audit"]["failed_checks"]


def test_07_resolves_confirmed_sources_from_workspace_root_token(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    suite = workspace / "outer" / "suite"
    config_dir = suite / "configs"
    config_dir.mkdir(parents=True)
    source = workspace / "paper_sources" / "table_source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("audited source", encoding="utf-8")
    _, payload = _write_deep_reference(tmp_path / "template")
    payload["source_root_at_confirmation"] = "workspace_root_two_levels_above_suite"
    payload["source_files"] = [
        {
            "source_path": "paper_sources/table_source.csv",
            "sha256": sha256_file(source),
            "source_role": "original manuscript table source",
        }
    ]
    reference_path = config_dir / "main_manuscript_deep_reference.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(assets.config, "SUITE_ROOT", suite)
    gate = assets.gate_deep_reference(reference_path)
    assert gate["usable"] is True
    assert gate["sources"]["deep_source_0"]["path"] == str(source.resolve())


def test_07_does_not_hide_latex_export_failure(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame({"value": [1.0]})

    def fail(*args, **kwargs):
        raise RuntimeError("latex unavailable")

    monkeypatch.setattr(frame, "to_latex", fail)
    with pytest.raises(RuntimeError, match="latex unavailable"):
        assets.write_latex(frame, tmp_path / "invalid.tex")
    assert not (tmp_path / "invalid.tex").exists()


def test_07_writes_asset_gates_before_nonzero_exit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(assets.config, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="Manuscript asset gates failed"):
        assets.main()
    gate_path = tmp_path / "07_manuscript_assets" / "asset_gates.json"
    assert gate_path.is_file()
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    assert payload["overall"]["status"] == "FAIL"
    assert {
        "efficiency", "ablation", "ablation_main_display", "lowshot", "traditional"
    }.issubset(payload["overall"]["failed_gates"])


def test_07_separates_ablation_source_evidence_and_main_display_contracts(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload()), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path)
    source_gate = assets.gate_additional_ablation(tmp_path, reference_path)
    assert source_gate["usable"] is True
    source_raw = pd.read_csv(
        tmp_path / "04_additional_ablation" / assets.ABLATION_VARIANT_RAW_NAME
    )
    source_summary = pd.read_csv(
        tmp_path / "04_additional_ablation" / assets.ABLATION_VARIANT_SUMMARY_NAME,
        keep_default_na=False,
    )
    assert len(source_raw) == 150
    assert len(source_summary) == 15
    assert set(source_raw["variant"]) == set(assets.ABLATION_SOURCE_VARIANTS)
    with pytest.raises(RuntimeError, match="four displayed reviewer controls"):
        assets.build_hybrid_ablation_table(source_summary, _reference_payload())

    _write_ablation_main_scope(tmp_path)
    display_gate = assets.gate_ablation_main_display(tmp_path)
    assert display_gate["usable"] is True
    main_raw = pd.read_csv(
        tmp_path / "04_additional_ablation" / assets.ABLATION_MAIN_RAW_NAME
    )
    main_summary = pd.read_csv(
        tmp_path / "04_additional_ablation" / assets.ABLATION_MAIN_SUMMARY_NAME,
        keep_default_na=False,
    )
    assert len(main_raw) == 120
    assert len(main_summary) == 12
    assert set(main_raw["variant"]) == set(assets.ABLATION_MAIN_VARIANTS)
    table = assets.build_hybrid_ablation_table(main_summary, _reference_payload())
    assert len(table) == 5
    assert table.iloc[0]["Method"] == "Full MHFL-MCA reference"
    assert "Equal weights" not in set(table["Method"])
    assert not any("-8" in column for column in table.columns)
    assert not any("delta_to_full" in column or "p_value" in column for column in table.columns)
    note = assets.ABLATION_MAIN_NOTE.lower()
    assert "equal-weight control was evaluated and retained" in note
    assert "not deleted" in note
    assert "supplementary/provenance" in note


def test_07_rejects_missing_equal_weight_archive(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload()), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path)
    _write_ablation_main_scope(tmp_path)
    (
        tmp_path
        / "04_additional_ablation"
        / assets.ABLATION_EQUAL_ARCHIVE_DIR_NAME
        / assets.ABLATION_EQUAL_ARCHIVE_NAME
    ).unlink()
    gate = assets.gate_ablation_main_display(tmp_path)
    assert gate["usable"] is False
    assert "equal_weights_archive_exists" in gate["post_audit"]["failed_checks"]


def test_07_rejects_tampered_source_evidence_archive(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload()), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path)
    _write_ablation_main_scope(tmp_path)
    archived_raw = (
        tmp_path
        / "04_additional_ablation"
        / assets.ABLATION_EQUAL_ARCHIVE_DIR_NAME
        / assets.ABLATION_VARIANT_RAW_NAME
    )
    archived_raw.write_bytes(archived_raw.read_bytes() + b"\n")
    gate = assets.gate_ablation_main_display(tmp_path)
    assert gate["usable"] is False
    assert "source_archive_records" in gate["post_audit"]["failed_checks"]


def test_07_rejects_stale_ablation_assets_when_latest_post_gate_failed(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload()), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path)
    post_path = tmp_path / "04_additional_ablation" / "additional_ablation_post_gate.json"
    post_gate = json.loads(post_path.read_text(encoding="utf-8"))
    post_gate.update({"status": "FAIL", "final_outputs_authorized": False})
    post_path.write_text(json.dumps(post_gate), encoding="utf-8")
    gate = assets.gate_additional_ablation(tmp_path, reference_path)
    assert gate["usable"] is False
    assert "internal_post_gate" in gate["post_audit"]["failed_checks"]


def test_07_rejects_stage3_minus8db_full_reference(tmp_path: Path):
    payload = _reference_payload()
    payload["conditions"]["4Nm_-8dB"] = dict(payload["conditions"]["4Nm_0dB"])
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path)
    gate = assets.gate_additional_ablation(tmp_path, reference_path)
    assert gate["usable"] is False
    assert "full_reference_clean_conditions_only" in gate["post_audit"]["failed_checks"]


def test_07_rejects_missing_or_wrong_protocol_full_reference(tmp_path: Path):
    missing = assets.gate_full_reference(tmp_path / "missing.json")
    assert missing["usable"] is False
    payload = _reference_payload()
    payload["protocol"] = "stage3_robustness"
    reference_path = tmp_path / "wrong_protocol.json"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    wrong = assets.gate_full_reference(reference_path)
    assert wrong["usable"] is False
    assert "reference_protocol" in wrong["post_audit"]["failed_checks"]


def test_07_rejects_five_seed_ablation_and_ordinary_mean(tmp_path: Path):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload()), encoding="utf-8")
    _write_ablation_bundle(tmp_path, reference_path, seed_count=5)
    summary_path = tmp_path / "04_additional_ablation" / assets.ABLATION_VARIANT_SUMMARY_NAME
    summary = pd.read_csv(summary_path)
    summary["aggregation"] = "mean_sd"
    summary.to_csv(summary_path, index=False)
    manifest_path = tmp_path / "04_additional_ablation" / "additional_ablation_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["sha256"] = sha256_file(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = assets.gate_additional_ablation(tmp_path, reference_path)
    assert gate["usable"] is False
    failed = set(gate["post_audit"]["failed_checks"])
    assert {"raw_rows", "target_seeds", "summary_seed_count", "aggregation"}.issubset(failed)


def test_07_rejects_five_seed_lowshot_results(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 5, assets.TRIMMED_AGGREGATION)
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    assert gate["post_audit"]["status"] == "FAIL"
    failed = set(gate["post_audit"]["failed_checks"])
    assert "raw_rows" in failed
    assert "group_seed_sets" in failed
    assert "summary_seed_count" in failed


def test_07_rejects_ordinary_mean_even_with_ten_seeds(tmp_path: Path):
    _write_lowshot_bundle(tmp_path, 10, "mean_sd")
    gate = assets.gate_lowshot(tmp_path)
    assert gate["usable"] is False
    failed = set(gate["post_audit"]["failed_checks"])
    assert "aggregation" in failed
    assert "paired_summary_protocol" in failed
    assert "paper_aligned_manifest" in failed


def test_07_has_mandatory_data_level_gates_and_nonzero_failure_path():
    source = (config.SUITE_ROOT / "07_build_manuscript_assets.py").read_text(encoding="utf-8")
    assert "gate_efficiency(config.OUTPUT_ROOT)" in source
    assert "gate_additional_ablation(config.OUTPUT_ROOT)" in source
    assert "gate_ablation_main_display(config.OUTPUT_ROOT)" in source
    assert "gate_lowshot(config.OUTPUT_ROOT)" in source
    assert "gate_traditional(config.OUTPUT_ROOT)" in source
    assert 'mandatory = ("efficiency", "ablation", "ablation_main_display", "lowshot", "traditional")' in source
    assert 'raise RuntimeError("Manuscript asset gates failed:' in source
    assert "asset_gates.json" in source
