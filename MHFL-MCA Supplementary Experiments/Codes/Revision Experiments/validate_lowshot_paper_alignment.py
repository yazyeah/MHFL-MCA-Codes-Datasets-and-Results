"""Validate the controlled low-shot result contract and paper-anchor disclosures.

This utility is read-only with respect to training artifacts and exits non-zero when
the 140/14/7-row contracts, hashes, or Table-5 provenance do not agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


N_GRID = (1, 2, 3, 4, 5, 7, 10)
VARIANTS = ("full", "no_caim")
AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
PROTOCOL = "uo_source_aligned_paired_deterministic_extension_v1"
METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "train_accuracy",
    "heldout_accuracy",
    "generalization_gap",
)
PAPER_ANCHORS = {
    5: {
        "test_accuracy_mean": 0.9279,
        "test_accuracy_sd": 0.0288,
        "test_macro_precision_mean": 0.9361,
        "test_macro_precision_sd": 0.0237,
        "test_macro_f1_mean": 0.9266,
        "test_macro_f1_sd": 0.0299,
    },
    10: {
        "test_accuracy_mean": 0.9707,
        "test_accuracy_sd": 0.0175,
        "test_macro_precision_mean": 0.9723,
        "test_macro_precision_sd": 0.0160,
        "test_macro_f1_mean": 0.9704,
        "test_macro_f1_sd": 0.0178,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper-aligned Experiment-05 outputs without training.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trimmed(values: Iterable[float]) -> Tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    ordered = np.sort(array)
    retained = ordered[1:-1]
    return (
        float(np.mean(retained)),
        float(np.std(retained, ddof=0)),
        float(np.mean(array)),
        float(np.std(array, ddof=0)),
    )


def close(first: object, second: object) -> bool:
    try:
        return bool(np.isclose(float(first), float(second), rtol=1.0e-12, atol=1.0e-12))
    except (TypeError, ValueError):
        return False


def output_hash_matches(manifest: Mapping[str, object], key: str, path: Path) -> bool:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        return False
    row = outputs.get(key, {})
    return (
        isinstance(row, Mapping)
        and Path(str(row.get("path", ""))).resolve() == path.resolve()
        and str(row.get("sha256", "")).lower() == sha256_file(path).lower()
        and int(row.get("size_bytes", -1)) == int(path.stat().st_size)
    )


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    paths = {
        "raw": root / "lowshot_raw.csv",
        "summary": root / "lowshot_summary.csv",
        "paired_summary": root / "caim_paired_summary.csv",
        "anchor_gate": root / "lowshot_anchor_gate.json",
        "operational_thresholds": root / "operational_thresholds.json",
        "post_gate": root / "lowshot_post_gate.json",
        "execution_state": root / "lowshot_execution_state.json",
        "manifest": root / "lowshot_run_manifest.json",
        "figure_png": root / "lowshot_evidence_bundle" / "lowshot_evidence.png",
        "figure_pdf": root / "lowshot_evidence_bundle" / "lowshot_evidence.pdf",
        "figure_svg": root / "lowshot_evidence_bundle" / "lowshot_evidence.svg",
        "figure_tiff": root / "lowshot_evidence_bundle" / "lowshot_evidence.tiff",
        "figure_contract": root / "lowshot_evidence_bundle" / "figure_contract.json",
    }
    missing = [key for key, path in paths.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError("Missing or empty final output files: {0}".format(", ".join(missing)))

    raw = pd.read_csv(paths["raw"])
    summary = pd.read_csv(paths["summary"])
    paired = pd.read_csv(paths["paired_summary"])
    anchor = json.loads(paths["anchor_gate"].read_text(encoding="utf-8"))
    thresholds = json.loads(paths["operational_thresholds"].read_text(encoding="utf-8"))
    post_gate = json.loads(paths["post_gate"].read_text(encoding="utf-8"))
    execution_state = json.loads(paths["execution_state"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    figure_contract = json.loads(paths["figure_contract"].read_text(encoding="utf-8"))

    required_raw = {
        "variant", "n_train", "run_idx", "seed", "test_accuracy", "test_macro_precision",
        "test_macro_recall", "test_macro_f1", "train_accuracy", "heldout_accuracy",
        "generalization_gap", "split_signature", "data_signature", "hyperparameter_signature",
        "source_type", "train_time_s",
    }
    finite_columns = list(METRICS) + ["n_train", "run_idx", "seed", "train_time_s"]
    numeric = raw[finite_columns].apply(pd.to_numeric, errors="coerce") if required_raw.issubset(raw.columns) else pd.DataFrame()
    expected_seed_map = {n_value: set(range(100 * n_value + 1, 100 * n_value + 11)) for n_value in N_GRID}
    group_seed_ok = required_raw.issubset(raw.columns)
    if group_seed_ok:
        observed_groups = {
            (str(variant), int(n_train))
            for variant, n_train in raw[["variant", "n_train"]].itertuples(index=False, name=None)
        }
        expected_groups = {(variant, n_train) for variant in VARIANTS for n_train in N_GRID}
        group_seed_ok = observed_groups == expected_groups
        for (variant, n_train), block in raw.groupby(["variant", "n_train"]):
            if (
                str(variant) not in VARIANTS
                or int(n_train) not in expected_seed_map
                or set(int(value) for value in block["seed"]) != expected_seed_map[int(n_train)]
            ):
                group_seed_ok = False
                break
            if not all(int(seed) == 100 * int(n_train) + int(run_idx) for seed, run_idx in zip(block["seed"], block["run_idx"])):
                group_seed_ok = False
                break
    pair_signatures = raw.groupby(["n_train", "seed"])[
        ["split_signature", "data_signature", "hyperparameter_signature"]
    ].nunique() if required_raw.issubset(raw.columns) else pd.DataFrame()

    summary_recomputed = len(summary) == 14
    if summary_recomputed:
        for (variant, n_train), block in raw.groupby(["variant", "n_train"], sort=True):
            selected = summary[(summary["variant"] == variant) & (summary["n_train"] == int(n_train))]
            if len(selected) != 1:
                summary_recomputed = False
                break
            row = selected.iloc[0]
            if row.get("aggregation") != AGGREGATION or int(row.get("seeds_total", -1)) != 10 or int(row.get("retained_after_trim", -1)) != 8:
                summary_recomputed = False
                break
            for metric in METRICS:
                mean, sd, untrimmed_mean, untrimmed_sd = trimmed(block[metric])
                expected = {
                    metric + "_mean": mean,
                    metric + "_sd": sd,
                    metric + "_untrimmed_mean": untrimmed_mean,
                    metric + "_untrimmed_sd": untrimmed_sd,
                }
                if not all(close(row.get(key), value) for key, value in expected.items()):
                    summary_recomputed = False
                    break
            if not summary_recomputed:
                break

    paired_recomputed = len(paired) == 7
    if paired_recomputed:
        pivot = raw.pivot(index=["n_train", "seed"], columns="variant", values="test_accuracy")
        paired_recomputed = set(pivot.columns) == set(VARIANTS) and len(pivot) == 70
        if paired_recomputed:
            gains = (pivot["full"] - pivot["no_caim"]).rename("gain").reset_index()
            for n_train, block in gains.groupby("n_train", sort=True):
                selected = paired[paired["n_train"] == int(n_train)]
                if len(selected) != 1:
                    paired_recomputed = False
                    break
                row = selected.iloc[0]
                mean, sd, untrimmed_mean, untrimmed_sd = trimmed(block["gain"])
                values = {
                    "paired_gain_mean": mean,
                    "paired_gain_sd": sd,
                    "paired_gain_untrimmed_mean": untrimmed_mean,
                    "paired_gain_untrimmed_sd": untrimmed_sd,
                    "caim_gain_mean": mean,
                    "caim_gain_sd": sd,
                    "caim_gain_untrimmed_mean": untrimmed_mean,
                    "caim_gain_untrimmed_sd": untrimmed_sd,
                }
                if not all(close(row.get(key), value) for key, value in values.items()):
                    paired_recomputed = False
                    break

    anchors_recomputed = summary_recomputed
    if anchors_recomputed:
        full = summary[summary["variant"] == "full"].set_index("n_train")
        for n_train, expected in PAPER_ANCHORS.items():
            if n_train not in full.index:
                anchors_recomputed = False
                break
            row = full.loc[n_train]
            if not all("{0:.4f}".format(float(row[key])) == "{0:.4f}".format(value) for key, value in expected.items()):
                anchors_recomputed = False
                break

    output_hashes_ok = all(
        output_hash_matches(manifest, key, paths[key])
        for key in (
            "raw", "summary", "paired_summary", "anchor_gate", "operational_thresholds",
            "post_gate", "execution_state", "figure_png", "figure_pdf", "figure_svg", "figure_tiff", "figure_contract",
        )
    )
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), Mapping) else {}
    hparams_record = outputs.get("hparams_config", {}) if isinstance(outputs, Mapping) else {}
    hparams_path = Path(str(hparams_record.get("path", ""))) if isinstance(hparams_record, Mapping) else Path()
    hparams_ok = False
    hparams: Dict[str, object] = {}
    if hparams_path.is_file() and isinstance(hparams_record, Mapping):
        hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
        hparams_ok = (
            str(hparams_record.get("sha256", "")).lower() == sha256_file(hparams_path).lower()
            and int(hparams_record.get("size_bytes", -1)) == int(hparams_path.stat().st_size)
            and hparams.get("status") == "CONFIRMED"
            and hparams.get("confirmation_status") == "confirmed_from_original_uo_artifact"
            and hparams.get("summary_evidence", {}).get("anchors_passed") is True
        )
    optimizer = manifest.get("optimizer", {}) if isinstance(manifest.get("optimizer"), Mapping) else {}
    optimizer_ok = bool(
        hparams_ok
        and str(optimizer.get("name", "")).lower() == "adamax"
        and close(optimizer.get("learning_rate"), hparams.get("lr"))
        and int(optimizer.get("batch_size", -1)) == int(hparams.get("batch_size", -2))
    )
    dataset_provenance = (
        manifest.get("dataset_provenance", {})
        if isinstance(manifest.get("dataset_provenance"), Mapping)
        else {}
    )
    dataset_files = dataset_provenance.get("files", [])
    dataset_files_ok = isinstance(dataset_files, list) and len(dataset_files) == 14
    if dataset_files_ok:
        for item in dataset_files:
            if not isinstance(item, Mapping):
                dataset_files_ok = False
                break
            source_path = Path(str(item.get("path", "")))
            if (
                not source_path.is_file()
                or int(item.get("size_bytes", -1)) != int(source_path.stat().st_size)
                or int(item.get("modification_time_ns", -1)) != int(source_path.stat().st_mtime_ns)
                or str(item.get("sha256", "")).lower() != sha256_file(source_path).lower()
            ):
                dataset_files_ok = False
                break
    dataset_payload = {"files": dataset_files, "settings": dataset_provenance.get("settings", {})}
    dataset_content_signature = str(dataset_provenance.get("content_signature", ""))
    dataset_provenance_ok = bool(
        dataset_files_ok
        and dataset_provenance.get("source_file_count") == 14
        and dataset_content_signature == sha256_json(dataset_payload)
        and required_raw.issubset(raw.columns)
        and set(raw["data_signature"].astype(str)) == {dataset_content_signature}
    )
    run_artifacts = manifest.get("run_artifacts", [])
    expected_artifact_ids = {
        (variant, n_train, run_idx, 100 * n_train + run_idx)
        for variant in VARIANTS
        for n_train in N_GRID
        for run_idx in range(1, 11)
    }
    observed_artifact_ids = set()
    run_artifacts_ok = isinstance(run_artifacts, list) and len(run_artifacts) == 140
    if run_artifacts_ok:
        for item in run_artifacts:
            if not isinstance(item, Mapping):
                run_artifacts_ok = False
                break
            identity = (
                str(item.get("variant")),
                int(item.get("n_train", -1)),
                int(item.get("run_idx", -1)),
                int(item.get("seed", -1)),
            )
            observed_artifact_ids.add(identity)
            records = {key: item.get(key, {}) for key in ("metrics", "weights", "history")}
            resolved: Dict[str, Path] = {}
            for key, record in records.items():
                if not isinstance(record, Mapping):
                    run_artifacts_ok = False
                    break
                artifact_path = Path(str(record.get("path", ""))).resolve()
                try:
                    artifact_path.relative_to(root)
                except ValueError:
                    run_artifacts_ok = False
                    break
                if (
                    not artifact_path.is_file()
                    or int(record.get("size_bytes", -1)) != int(artifact_path.stat().st_size)
                    or str(record.get("sha256", "")).lower() != sha256_file(artifact_path).lower()
                ):
                    run_artifacts_ok = False
                    break
                resolved[key] = artifact_path
            if not run_artifacts_ok:
                break
            metrics_payload = json.loads(resolved["metrics"].read_text(encoding="utf-8"))
            metrics_identity = (
                str(metrics_payload.get("variant")),
                int(metrics_payload.get("n_train", -1)),
                int(metrics_payload.get("run_idx", -1)),
                int(metrics_payload.get("seed", -1)),
            )
            parameter_count = int(item.get("model_parameter_count", -1))
            if (
                metrics_identity != identity
                or metrics_payload.get("run_signature") != item.get("run_signature")
                or metrics_payload.get("provenance_signature") != item.get("provenance_signature")
                or Path(str(metrics_payload.get("weights_path", ""))).resolve() != resolved["weights"]
                or Path(str(metrics_payload.get("history_path", ""))).resolve() != resolved["history"]
                or str(metrics_payload.get("weights_sha256", "")).lower() != str(records["weights"].get("sha256", "")).lower()
                or str(metrics_payload.get("history_sha256", "")).lower() != str(records["history"].get("sha256", "")).lower()
                or len(pd.read_csv(resolved["history"])) != 80
                or (identity[0] == "full" and parameter_count != 5_111_759)
                or (identity[0] == "no_caim" and parameter_count <= 0)
            ):
                run_artifacts_ok = False
                break
    run_artifacts_ok = bool(run_artifacts_ok and observed_artifact_ids == expected_artifact_ids)
    execution_state_ok = (
        execution_state.get("status") == "PASS"
        and execution_state.get("mode") == "full"
        and execution_state.get("protocol") == PROTOCOL
        and execution_state.get("completed_run_artifacts") == 140
        and execution_state.get("final_outputs_authorized") is True
    )
    labels = figure_contract.get("panel_y_labels", [])
    figure_contract_ok = (
        labels == ["Held-out accuracy", "Train–held-out gap", "Paired CAIM accuracy gain"]
        and "no independent validation/test split" in str(figure_contract.get("evaluation_set_role", ""))
        and figure_contract.get("anchor_gate") == anchor
    )

    checks: Dict[str, bool] = {
        "raw_rows": len(raw) == 140,
        "summary_rows": len(summary) == 14,
        "paired_rows": len(paired) == 7,
        "variants": set(raw.get("variant", [])) == set(VARIANTS),
        "n_grid": set(pd.to_numeric(raw.get("n_train", pd.Series(dtype=float))).astype(int)) == set(N_GRID),
        "n_specific_seed_schedule": group_seed_ok,
        "no_duplicates": required_raw.issubset(raw.columns) and not raw.duplicated(["variant", "n_train", "seed"]).any(),
        "finite": not numeric.empty and not numeric.isna().any().any() and np.isfinite(numeric.to_numpy(dtype=np.float64)).all(),
        "metric_ranges": (
            not numeric.empty
            and all(bool(((numeric[name] >= 0.0) & (numeric[name] <= 1.0)).all()) for name in METRICS[:-1])
            and bool(((numeric["generalization_gap"] >= -1.0) & (numeric["generalization_gap"] <= 1.0)).all())
            and bool((numeric["train_time_s"] >= 0.0).all())
        ),
        "source_type": required_raw.issubset(raw.columns) and set(raw["source_type"].astype(str)) == {"new_paper_aligned_training"},
        "matched_pairs": len(pair_signatures) == 70 and bool((pair_signatures == 1).all().all()),
        "summary_recomputed_from_raw": summary_recomputed,
        "paired_gain_recomputed_by_seed_before_trim": paired_recomputed,
        "anchor_json_pass": anchor.get("status") == "PASS" and anchor.get("required_for_manuscript") is True and anchor.get("final_assets_authorized") is True,
        "anchors_recomputed_from_summary": anchors_recomputed,
        "post_gate_pass": post_gate.get("status") == "PASS" and post_gate.get("mode") == "full" and post_gate.get("final_outputs_authorized") is True,
        "manifest_pass": (
            manifest.get("status") == "PASS"
            and manifest.get("mode") == "full"
            and manifest.get("protocol") == PROTOCOL
            and manifest.get("gradient_clipping") is None
            and manifest.get("early_stopping") is False
            and manifest.get("final_epoch_weights") is True
            and int(manifest.get("epochs", -1)) == 80
            and "not retained" in str(manifest.get("random_initialization_boundary", "")).lower()
            and manifest.get("failed_runs") == 0
            and manifest.get("failed_seeds") == []
        ),
        "confirmed_hparams_hash": hparams_ok,
        "optimizer_exact": optimizer_ok,
        "dataset_content_provenance": dataset_provenance_ok,
        "run_artifact_inventory": run_artifacts_ok,
        "execution_state": execution_state_ok,
        "manifest_output_hashes": output_hashes_ok,
        "threshold_scope": isinstance(thresholds, list) and len(thresholds) == 2 and all("not a universal" in str(row.get("claim_limit", "")).lower() for row in thresholds),
        "figure_contract_heldout_semantics": figure_contract_ok,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    report = {"status": "PASS" if not failed else "FAIL", "checks": checks, "failed": failed}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
