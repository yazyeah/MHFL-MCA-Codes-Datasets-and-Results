from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


EXPECTED_N_GRID = (1, 2, 3, 4, 5, 7, 10)
EXPECTED_VARIANTS = ("full", "no_caim")
EXPECTED_RUNS_PER_N = 10
EXPECTED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
EXPECTED_PARAMETER_COUNT = 5_111_759
EXPECTED_PROTOCOL_ID = "uo_source_aligned_paired_deterministic_extension_v1"
EXPECTED_PARAMS: Dict[str, object] = {
    "dropout_vib": 0.4223065128432895,
    "dropout_aco": 0.3497078380248016,
    "atten_dim": 256,
    "n_layers_vib": 4,
    "n_layers_aco": 5,
    "lr": 0.0013865546776879656,
    "batch_size": 16,
}
EXPECTED_METADATA: Dict[str, object] = {
    "study_name": "no-name-0b7f7870-670c-450d-a7d9-781d2c4ed78a",
    "best_trial_number": 22,
    "best_value": 0.994434118270874,
    "number_of_trials": 30,
    "completed_trials": 12,
    "pruned_trials": 18,
    "storage": "in-memory (no persistent Optuna database declared in the retained run log)",
    "sampler": "Optuna default sampler (not explicitly printed)",
    "pruner": "MedianPruner",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only pre-training audit for the paper-aligned Experiment 05 protocol."
    )
    parser.add_argument("--phase", choices=("pre",), required=True)
    parser.add_argument("--root", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(path: Path) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))


def _literal_constants(tree: ast.Module, names: Iterable[str]) -> Dict[str, Any]:
    requested = set(names)
    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in requested:
                try:
                    values[name] = ast.literal_eval(node.value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Protocol constant {0} is not a literal.".format(name)) from exc
    missing = sorted(requested.difference(values))
    if missing:
        raise RuntimeError("Cannot read low-shot protocol constants: {0}.".format(", ".join(missing)))
    return values


def _functions(tree: ast.Module) -> Mapping[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def _require_tokens(label: str, source: str, tokens: Sequence[str]) -> None:
    missing = [token for token in tokens if token not in source]
    if missing:
        raise RuntimeError("{0} is missing required protocol tokens: {1}.".format(label, ", ".join(missing)))


def _reject_tokens(label: str, source: str, tokens: Sequence[str]) -> None:
    found = [token for token in tokens if token in source]
    if found:
        raise RuntimeError("{0} contains prohibited protocol tokens: {1}.".format(label, ", ".join(found)))


def _confirmed_uo_config(root: Path) -> Dict[str, Any]:
    config_path = root / "configs" / "uo_optuna_confirmed.json"
    if not config_path.is_file():
        raise RuntimeError("Confirmed UO Optuna configuration is missing: {0}.".format(config_path))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "CONFIRMED":
        raise RuntimeError("UO Optuna evidence status is not CONFIRMED.")
    if payload.get("confirmation_status") != "confirmed_from_original_uo_artifact":
        raise RuntimeError("UO Optuna confirmation did not come from an original artifact.")
    if payload.get("parameter_candidate_count") != 1:
        raise RuntimeError("Exactly one UO Optuna parameter candidate is required.")
    candidates = payload.get("parameter_candidates")
    if not isinstance(candidates, list) or candidates != [EXPECTED_PARAMS]:
        raise RuntimeError("The unique UO Optuna candidate does not match the confirmed seven parameters.")
    if any(payload.get(key) != value for key, value in EXPECTED_PARAMS.items()):
        raise RuntimeError("Top-level UO Optuna parameters differ from the confirmed original values.")
    if any(payload.get(key) != value for key, value in EXPECTED_METADATA.items()):
        raise RuntimeError("UO Optuna study metadata differs from the retained original run.")

    parameter_evidence = payload.get("parameter_evidence")
    if not isinstance(parameter_evidence, list) or len(parameter_evidence) != 1:
        raise RuntimeError("Exactly one explicit winning-parameter artifact is required.")
    if parameter_evidence[0].get("params") != EXPECTED_PARAMS:
        raise RuntimeError("The winning-parameter evidence does not contain the confirmed seven parameters.")

    evidence_path = Path(str(payload.get("evidence_path", "")))
    evidence_sha256 = str(payload.get("evidence_sha256", "")).lower()
    if not evidence_path.is_file() or _sha256_file(evidence_path).lower() != evidence_sha256:
        raise RuntimeError("Original UO Optuna evidence is missing or its SHA-256 has changed.")
    evidence_record = parameter_evidence[0]
    if (
        Path(str(evidence_record.get("path", ""))).resolve() != evidence_path.resolve()
        or str(evidence_record.get("sha256", "")).lower() != evidence_sha256
    ):
        raise RuntimeError("The parameter-evidence record disagrees with the selected winning artifact.")

    summary = payload.get("summary_evidence")
    if not isinstance(summary, Mapping) or summary.get("anchors_passed") is not True:
        raise RuntimeError("The original UO summary did not pass the Table-5 anchor audit.")
    summary_path = Path(str(summary.get("path", "")))
    summary_sha256 = str(summary.get("sha256", "")).lower()
    if not summary_path.is_file() or _sha256_file(summary_path).lower() != summary_sha256:
        raise RuntimeError("Original UO summary evidence is missing or its SHA-256 has changed.")

    source = payload.get("source_program_evidence")
    if not isinstance(source, Mapping):
        raise RuntimeError("Original UO source-program evidence is missing.")
    source_path = Path(str(source.get("path", "")))
    source_sha256 = str(source.get("sha256", "")).lower()
    checks = source.get("protocol_checks")
    if not source_path.is_file() or _sha256_file(source_path).lower() != source_sha256:
        raise RuntimeError("Original UO source program is missing or its SHA-256 has changed.")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise RuntimeError("Original UO source-program protocol checks are incomplete.")

    return {
        "status": payload["status"],
        "confirmation_status": payload["confirmation_status"],
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "parameters": dict(EXPECTED_PARAMS),
        "metadata": dict(EXPECTED_METADATA),
        "parameter_evidence_path": str(evidence_path.resolve()),
        "parameter_evidence_sha256": evidence_sha256,
        "summary_evidence_path": str(summary_path.resolve()),
        "summary_evidence_sha256": summary_sha256,
        "source_program_path": str(source_path.resolve()),
        "source_program_sha256": source_sha256,
    }


def audit_pre(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    script_path = root / "05_lowshot_threshold.py"
    source = script_path.read_text(encoding="utf-8")
    tree = _tree(script_path)
    constants = _literal_constants(
        tree,
        (
            "VARIANT_NAMES",
            "N_GRID_FULL",
            "N_GRID_FAST",
            "RUNS_PER_N_FULL",
            "TRIMMED_AGGREGATION",
            "EXPECTED_FULL_PARAMETER_COUNT",
            "TRAINING_PROTOCOL_ID",
            "RAW_COLUMNS",
            "SUMMARY_METRICS",
        ),
    )
    if tuple(constants["VARIANT_NAMES"]) != EXPECTED_VARIANTS:
        raise RuntimeError("Experiment 05 must compare Full and no-CAIM only.")
    if tuple(constants["N_GRID_FULL"]) != EXPECTED_N_GRID:
        raise RuntimeError("Full Experiment 05 N grid is not the seven-value paper protocol.")
    if tuple(constants["N_GRID_FAST"]) != (5, 10):
        raise RuntimeError("Fast Experiment 05 N grid changed.")
    if constants["RUNS_PER_N_FULL"] != EXPECTED_RUNS_PER_N:
        raise RuntimeError("Full Experiment 05 must use ten runs per N.")
    if constants["TRIMMED_AGGREGATION"] != EXPECTED_AGGREGATION:
        raise RuntimeError("Experiment 05 aggregation label changed.")
    if constants["EXPECTED_FULL_PARAMETER_COUNT"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("The source-exact UO model parameter-count contract changed.")
    if constants["TRAINING_PROTOCOL_ID"] != EXPECTED_PROTOCOL_ID:
        raise RuntimeError("The paired deterministic reviewer-extension protocol ID changed.")
    required_metrics = {
        "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
        "train_accuracy", "heldout_accuracy", "generalization_gap",
    }
    if set(constants["SUMMARY_METRICS"]) != required_metrics:
        raise RuntimeError("Experiment 05 summary metrics do not match the paper-aligned held-out contract.")
    if "validation_accuracy" in constants["RAW_COLUMNS"] or "heldout_accuracy" not in constants["RAW_COLUMNS"]:
        raise RuntimeError("Experiment 05 raw schema must identify the reused evaluation set as held-out.")

    functions = _functions(tree)
    required_functions = {
        "paper_seed",
        "set_paired_model_seed",
        "expected_seed_map",
        "load_confirmed_hparams",
        "bind_dataset_content_hashes",
        "prepare_source_exact_split",
        "build_source_exact_model",
        "train_one",
        "summarize",
        "paired_caim",
        "anchor_gate",
        "build_operational_thresholds",
        "validate_raw",
        "build_post_gate",
        "run_artifact_record",
        "main",
    }
    missing_functions = sorted(required_functions.difference(functions))
    if missing_functions:
        raise RuntimeError("Paper-aligned Experiment 05 functions are missing: {0}.".format(", ".join(missing_functions)))

    paper_seed_source = _source_segment(source, functions["paper_seed"])
    _require_tokens("paper_seed", paper_seed_source, ("int(n_train) * 100 + int(run_idx)",))

    paired_seed_source = _source_segment(source, functions["set_paired_model_seed"])
    _require_tokens(
        "set_paired_model_seed",
        paired_seed_source,
        ("random.seed(value)", "tf.random.set_seed(value)"),
    )
    _reject_tokens(
        "set_paired_model_seed",
        paired_seed_source,
        ("np.random.seed", "np.random.set_state", "np.random.default_rng"),
    )

    config_loader_source = _source_segment(source, functions["load_confirmed_hparams"])
    _require_tokens(
        "load_confirmed_hparams",
        config_loader_source,
        (
            'payload.get("status") != "CONFIRMED"',
            'payload.get("confirmation_status") != "confirmed_from_original_uo_artifact"',
            'sha256_file(evidence_path).lower() != evidence_sha256',
            'summary_evidence.get("anchors_passed") is not True',
            'sha256_file(summary_path).lower() != summary_sha256',
            'sha256_file(source_program_path).lower() != source_program_sha256',
            "not all(value is True for value in checks.values())",
            "must be an exact JSON integer",
            "must be a finite JSON number",
            "np.isfinite",
        ),
    )

    dataset_binding_source = _source_segment(source, functions["bind_dataset_content_hashes"])
    _require_tokens(
        "bind_dataset_content_hashes",
        dataset_binding_source,
        (
            "len(source_rows) != 14",
            "sha256_file(path)",
            '"content_signature"',
            '"source_file_count"',
            '"files": inventory',
        ),
    )

    split_source = _source_segment(source, functions["prepare_source_exact_split"])
    _require_tokens(
        "prepare_source_exact_split",
        split_source,
        (
            "available != 400",
            "np.random.seed(seed_value)",
            "train_idx = order[:n_value]",
            "heldout_idx = order[n_value:]",
            '"train": train',
            '"heldout": heldout',
            "intersection(heldout.sample_ids.tolist())",
            '"uploaded_uo_first_N_train_all_remainder_heldout"',
        ),
    )
    _reject_tokens("prepare_source_exact_split", split_source, ('"val"', "validation_idx", "test_idx"))

    training_source = _source_segment(source, functions["train_one"])
    _require_tokens(
        "train_one",
        training_source,
        (
            "seed = paper_seed(n_train, run_idx)",
            "prepare_source_exact_split(bundle, n_train, seed)",
            "set_paired_model_seed(tf, seed)",
            "build_source_exact_model(tf, hparams",
            "tf.keras.optimizers.Adamax",
            'learning_rate=float(hparams["lr"])',
            "epochs=80",
            'batch_size=int(hparams["batch_size"])',
            'splits["heldout"].x1',
            "shuffle=True",
            "model.save_weights",
            '"new_paper_aligned_training"',
            '"protocol": TRAINING_PROTOCOL_ID',
        ),
    )
    _reject_tokens(
        "train_one",
        training_source,
        (
            "clipnorm", "clipvalue", "EarlyStopping", "early_stopping",
            "restore_best_weights", "np.random.seed", "np.random.set_state",
        ),
    )
    if training_source.find("prepare_source_exact_split") > training_source.find("set_paired_model_seed"):
        raise RuntimeError(
            "The historical NumPy split/shuffle stream must be established before paired model seeding."
        )
    if training_source.find("set_paired_model_seed") > training_source.find("build_source_exact_model"):
        raise RuntimeError("Paired Python/TensorFlow seeding must occur before model construction.")

    summary_source = _source_segment(source, functions["summarize"])
    _require_tokens(
        "summarize",
        summary_source,
        ("for metric in SUMMARY_METRICS", "_metric_statistics"),
    )
    paired_source = _source_segment(source, functions["paired_caim"])
    _require_tokens(
        "paired_caim",
        paired_source,
        ('on=["n_train", "seed"]', "validate=\"one_to_one\"", "full_accuracy", "no_caim_accuracy", "caim_gain"),
    )

    validator_source = _source_segment(source, functions["validate_raw"])
    _require_tokens(
        "validate_raw",
        validator_source,
        (
            "tuple(raw.columns) != RAW_COLUMNS",
            'raw.duplicated(["variant", "n_train", "seed"])',
            "np.isfinite",
            "expected_seed_map(n_grid, runs_per_n)",
            '["split_signature", "data_signature", "hyperparameter_signature"]',
            'set(raw["source_type"].astype(str))',
            '"new_paper_aligned_training"',
        ),
    )
    post_gate_source = _source_segment(source, functions["build_post_gate"])
    _require_tokens(
        "build_post_gate",
        post_gate_source,
        (
            "expected_summary = summarize",
            "expected_paired = paired_caim",
            "expected_thresholds = build_operational_thresholds",
            '"summary_recomputed_from_raw"',
            '"paired_gain_aligned_by_seed_before_trim"',
            '"operational_threshold_recomputed_from_trimmed_statistics"',
            '"final_outputs_authorized"',
        ),
    )

    artifact_source = _source_segment(source, functions["run_artifact_record"])
    _require_tokens(
        "run_artifact_record",
        artifact_source,
        (
            '"metrics": Path',
            '"weights": Path',
            '"history": Path',
            "relative_to(root)",
            'sha256_file(paths["weights"])',
            'sha256_file(paths["history"])',
            "metrics_identity != identity",
            'metrics.get("run_signature")',
            '"model_parameter_count"',
            '"history": output_record(paths["history"], 80)',
        ),
    )

    main_source = _source_segment(source, functions["main"])
    _require_tokens(
        "main",
        main_source,
        (
            "hparams = load_confirmed_hparams(args.uo_config)",
            'run_tag.lower() in {"", "manual", "fast", "smoke", "debug"}',
            '"Full Experiment 05 requires an explicit non-smoke run tag."',
            "n_grid = N_GRID_FAST if args.mode == \"fast\" else N_GRID_FULL",
            "runs_per_n = 2 if args.mode == \"fast\" else RUNS_PER_N_FULL",
            "bundle = load_uo_dataset",
            "bundle, dataset_provenance = bind_dataset_content_hashes(bundle)",
            '"status": "RUNNING"',
            'execution_state_path = out_dir / "lowshot_execution_state.json"',
            "row = train_one(",
            "run_artifacts.append(run_artifact_record(row, out_dir))",
            "raw_gate = validate_raw(raw, n_grid, runs_per_n)",
            "summary = summarize(raw, runs_per_n, use_trim)",
            "paired = paired_caim(raw, runs_per_n, use_trim)",
            "post_gate = build_post_gate(",
            '"protocol": TRAINING_PROTOCOL_ID',
            '"random_initialization_boundary"',
            '"seed_schedule": "seed = 100*N + run_idx"',
            '"epochs": 80',
            '"name": "Adamax"',
            '"gradient_clipping": None',
            '"final_epoch_weights": True',
            '"execution_state": output_record(execution_state_path)',
            '"run_artifact_count": len(run_artifacts)',
            '"run_artifacts": run_artifacts',
            '"dataset_provenance": dataset_provenance',
            '"declared Keras batch-shuffle stream',
            '"state solely to preserve stacked split ordering',
        ),
    )
    if main_source.find("raw_gate = validate_raw") > main_source.find("raw.to_csv"):
        raise RuntimeError("Experiment 05 must validate raw evidence before publishing CSV outputs.")
    _reject_tokens("main", main_source, ("load_or_train_", "prepare_uo_splits", "TrainingSpec("))

    confirmed = _confirmed_uo_config(root)
    seed_map = {
        str(n_train): [100 * n_train + run_idx for run_idx in range(1, EXPECTED_RUNS_PER_N + 1)]
        for n_train in EXPECTED_N_GRID
    }
    return {
        "status": "PASS",
        "phase": "pre",
        "script": str(script_path.resolve()),
        "script_sha256": _sha256_file(script_path),
        "protocol": EXPECTED_PROTOCOL_ID,
        "variants": list(EXPECTED_VARIANTS),
        "n_grid": list(EXPECTED_N_GRID),
        "runs_per_n": EXPECTED_RUNS_PER_N,
        "seed_schedule": "seed = 100*N + run_idx",
        "target_seed_map": seed_map,
        "expected_model_slots": len(EXPECTED_VARIANTS) * len(EXPECTED_N_GRID) * EXPECTED_RUNS_PER_N,
        "expected_raw_rows": 140,
        "expected_summary_rows": 14,
        "expected_paired_rows": 7,
        "matched_split_pairs": 70,
        "retained_after_trim": 8,
        "aggregation": EXPECTED_AGGREGATION,
        "split": {
            "protocol": "uploaded_uo_first_N_train_all_remainder_heldout",
            "train_selection": "first_N_after_per_class_seeded_shuffle",
            "evaluation_role": "all_remaining_segments_heldout",
            "expected_segments_per_class": 400,
            "independent_validation_set": False,
            "matched_between_variants": True,
        },
        "training": {
            "epochs": 80,
            "optimizer": "Adamax",
            "learning_rate_source": "confirmed UO Optuna lr",
            "batch_size_source": "confirmed UO Optuna batch_size",
            "gradient_clipping": None,
            "early_stopping": False,
            "final_epoch_weights": True,
            "model_parameter_count": EXPECTED_PARAMETER_COUNT,
            "paired_model_rngs": ["Python", "TensorFlow"],
            "numpy_reseeded_after_split": False,
        },
        "validator": {
            "exact_raw_schema": True,
            "n_specific_seed_sets": True,
            "duplicates_rejected": True,
            "nan_or_inf_rejected": True,
            "matched_split_data_hparams": True,
            "source_type_required": "new_paper_aligned_training",
        },
        "post_gate": {
            "summary_recomputed_from_raw": True,
            "paired_gain_aligned_by_seed_before_trim": True,
            "operational_threshold_recomputed_from_trimmed_statistics": True,
            "anchor_required_for_full_assets": True,
        },
        "execution_evidence": {
            "full_run_tag_rejects_manual_fast_smoke_debug": True,
            "running_state_written_before_dataset_loading": True,
            "final_state_bound_in_manifest_outputs": True,
            "expected_run_artifact_count": 140,
            "per_run_metrics_weights_history_inventory": True,
            "source_mat_content_hash_count": 14,
        },
        "confirmed_uo_config": confirmed,
        "training_was_run": False,
        "files_written": [],
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1] if args.root is None else args.root
    print(json.dumps(audit_pre(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
