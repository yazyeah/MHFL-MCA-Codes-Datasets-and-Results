from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


VARIANTS = (
    "homogeneous_vibration",
    "homogeneous_other",
    "attention_dim_128",
    "direct_softmax",
    "equal_weights",
)
LEGACY_FULL_VARIANT = "full"
LEGACY_VARIANTS = (LEGACY_FULL_VARIANT,) + VARIANTS
CONDITIONS = ("2Nm_0dB", "4Nm_0dB", "4Nm_-8dB")
TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
EXPECTED_LEARNING_RATE = 0.0004156294449523281
EXPECTED_BATCH_SIZE = 16
EXPECTED_EPOCHS = 80
LEGACY_SEED_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for the ten-seed R4-6 extension.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--phase", choices=("pre",), required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_constants(path: Path) -> Dict[str, Any]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
    values: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in {"GLOBAL_SEED", "ABLATION_REPEATS_FAST", "ABLATION_REPEATS_FULL", "STAGE2_EPOCHS"}:
            values[name] = ast.literal_eval(node.value)
    missing = sorted(
        {"GLOBAL_SEED", "ABLATION_REPEATS_FAST", "ABLATION_REPEATS_FULL", "STAGE2_EPOCHS"} - set(values)
    )
    if missing:
        raise RuntimeError("Cannot read required config constants: {0}".format(", ".join(missing)))
    return values


def read_json_object(path: Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("Required audit input is missing: {0}".format(target))
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object: {0}".format(target))
    return payload


def file_record(path: Path) -> Dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError("Required audit input is missing: {0}".format(target))
    stat = target.stat()
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "mtime_ns": int(stat.st_mtime_ns),
        "size_bytes": int(stat.st_size),
    }


def audit_weight_record(manifest_path: Path, weights_path: Path) -> Dict[str, Any]:
    manifest = read_json_object(manifest_path)
    record = file_record(weights_path)
    if manifest.get("weights_sha256") != record["sha256"]:
        raise RuntimeError("Weights hash differs from the immutable model manifest: {0}".format(weights_path))
    return record


def _audit_raw(raw_path: Path, legacy_seeds: Sequence[int]) -> Dict[str, Any]:
    with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_seed_strings = {str(seed) for seed in legacy_seeds}
    legacy_rows = [
        row for row in rows
        if row.get("variant") in LEGACY_VARIANTS and row.get("seed") in expected_seed_strings
    ]
    if len(legacy_rows) != len(LEGACY_VARIANTS) * len(CONDITIONS) * len(legacy_seeds):
        raise RuntimeError("Pre-extension raw CSV must contain the complete 90-row six-variant legacy block.")
    keys = [(row.get("variant"), row.get("condition"), row.get("seed")) for row in legacy_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Pre-extension raw CSV contains duplicate rows.")
    for row in legacy_rows:
        if row.get("variant") not in LEGACY_VARIANTS or row.get("condition") not in CONDITIONS:
            raise RuntimeError("Pre-extension raw CSV contains an unexpected variant or condition.")
        if row.get("seed") not in expected_seed_strings:
            raise RuntimeError("Pre-extension raw CSV contains an unexpected seed.")
        for column in ("params_m", "accuracy", "macro_precision", "macro_recall", "macro_f1"):
            value = float(row[column])
            if not math.isfinite(value):
                raise RuntimeError("Pre-extension raw CSV contains NaN or Inf.")
    for variant in LEGACY_VARIANTS:
        for condition in CONDITIONS:
            observed = {row["seed"] for row in legacy_rows if row["variant"] == variant and row["condition"] == condition}
            if observed != expected_seed_strings:
                raise RuntimeError("Every legacy variant-condition group must contain the same five seeds.")
    return {
        "source_rows": len(rows),
        "legacy_rows": len(legacy_rows),
        "reusable_variant_rows": len([row for row in legacy_rows if row["variant"] in VARIANTS]),
        "legacy_full_rows": len([row for row in legacy_rows if row["variant"] == LEGACY_FULL_VARIANT]),
        "variants": len({row["variant"] for row in legacy_rows}),
        "conditions": len({row["condition"] for row in legacy_rows}),
        "seeds": sorted(int(value) for value in expected_seed_strings),
        "duplicate_rows": 0,
        "nan_or_inf": 0,
    }


def _audit_model(
    out_dir: Path,
    split_dir: Path,
    variant: str,
    seed: int,
) -> Dict[str, Any]:
    run_dir = out_dir / "models" / variant / ("seed{0}".format(seed))
    manifest_path = run_dir / "manifest.json"
    weights_path = run_dir / "model.weights.h5"
    history_path = run_dir / ("{0}_stage2.history.csv".format(variant))
    split_path = split_dir / ("kaist_source_seed{0}.json".format(seed))
    manifest = read_json_object(manifest_path)
    split = read_json_object(split_path)

    model_spec = manifest.get("model_spec")
    if not isinstance(model_spec, dict) or manifest.get("model_spec_fingerprint") != sha256_json(model_spec):
        raise RuntimeError("Legacy model specification fingerprint mismatch: {0}".format(manifest_path))

    training = manifest.get("training_spec", {})
    expected_training = {
        "protocol": "stage2",
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "seed": seed,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError("Training metadata mismatch for {0}, seed {1}: {2}".format(variant, seed, key))
    protocol = manifest.get("protocol_signature", {})
    if set(protocol) != {"variant", "split_signature", "data_signatures", "evaluation_conditions"}:
        raise RuntimeError("Legacy protocol_signature was changed or contains sidecar fields: {0}".format(manifest_path))
    if protocol.get("variant") != variant or protocol.get("evaluation_conditions") != list(CONDITIONS):
        raise RuntimeError("Legacy protocol metadata mismatch: {0}".format(manifest_path))
    if protocol.get("split_signature") != split.get("signature"):
        raise RuntimeError("Legacy split signature mismatch: {0}".format(manifest_path))
    if split.get("max_train_per_class") != 30 or split.get("val_per_class") != 50:
        raise RuntimeError("Split sidecar does not prove N=30/val=50: {0}".format(split_path))

    signed = {
        "suite_version": manifest.get("suite_version"),
        "model_spec": manifest.get("model_spec"),
        "model_spec_fingerprint": manifest.get("model_spec_fingerprint"),
        "training_spec": training,
        "protocol_signature": protocol,
    }
    if manifest.get("run_signature") != sha256_json(signed):
        raise RuntimeError("Legacy run_signature mismatch: {0}".format(manifest_path))
    weights_record = audit_weight_record(manifest_path, weights_path)
    return {
        "seed": seed,
        "variant": variant,
        "manifest": file_record(manifest_path),
        "weights": weights_record,
        "history": file_record(history_path),
        "split": file_record(split_path),
        "split_signature": split["signature"],
    }


def _audit_channel_sidecars(root: Path, run_tag: str) -> Dict[str, Any]:
    provenance = root / "provenance" / run_tag
    vibration_path = provenance / "vibration_manifest.json"
    preflight_path = provenance / "preflight_report.json"
    channel_path = provenance / "channel_manifest.json"
    batch_path = root / "windows" / "00_set_environment.bat"
    vibration = read_json_object(vibration_path)
    preflight = read_json_object(preflight_path)
    channels = read_json_object(channel_path)
    batch = batch_path.read_text(encoding="utf-8")
    exact_channel = "cDAQ9185-1F486B5Mod2/ai0"
    if vibration.get("configured_column") != 0 or not vibration.get("configured_column_available"):
        raise RuntimeError("Vibration sidecar does not confirm xA/column 0.")
    if not preflight.get("current_channel_explicit") or not preflight.get("vibration_channel_valid"):
        raise RuntimeError("Preflight does not confirm the fixed channel selections.")
    if not any(row.get("group") == "Log" and row.get("channel") == exact_channel for row in channels.get("channels", [])):
        raise RuntimeError("Channel sidecar does not contain the exact U-phase channel.")
    active = [line.strip().lower() for line in batch.splitlines() if line.strip().lower().startswith("set ")]
    if 'set "mhfl_current_channel_name={0}"'.format(exact_channel.lower()) not in active:
        raise RuntimeError("Windows sidecar does not configure the exact U-phase channel.")
    if any("mhfl_allow_current_fallback" in line for line in active):
        raise RuntimeError("Windows sidecar enables prohibited current fallback.")
    return {
        "vibration_column": 0,
        "vibration_physical_meaning": "bearing housing A x-direction",
        "current_group": "Log",
        "current_channel": exact_channel,
        "current_selection_mode": "explicit_exact_name",
        "fallback_policy": "prohibited_in_full_mode",
        "evidence": [file_record(path) for path in (vibration_path, preflight_path, channel_path, batch_path)],
    }


def _audit_full_reference(path: Path) -> Dict[str, Any]:
    payload = read_json_object(path)
    expected = {
        "source": "main_manuscript_stage2_experiment",
        "protocol": "stage2_load_shift",
        "n_train_per_class": 30,
        "runs": 10,
        "aggregation": TRIMMED_AGGREGATION,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError("Full-reference field mismatch: {0}".format(field))
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict) or set(conditions) != {"2Nm_0dB", "4Nm_0dB"}:
        raise RuntimeError("Full reference must contain only the two Stage-2 clean-load conditions.")
    for condition in ("2Nm_0dB", "4Nm_0dB"):
        for field in ("accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"):
            value = conditions[condition].get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError("Full-reference metric is invalid: {0}.{1}".format(condition, field))
    return {
        "file": file_record(path),
        "source": payload["source"],
        "protocol": payload["protocol"],
        "n_train_per_class": payload["n_train_per_class"],
        "runs": payload["runs"],
        "aggregation": payload["aggregation"],
        "conditions": sorted(conditions),
        "stage3_minus8db_full_present": False,
    }


def audit_pre(root: Path, run_tag: str) -> Dict[str, Any]:
    root = Path(root).resolve()
    constants = config_constants(root / "mhfl_review" / "config.py")
    if constants["ABLATION_REPEATS_FAST"] != 1:
        raise RuntimeError("Fast ablation repeat count changed unexpectedly.")
    if constants["ABLATION_REPEATS_FULL"] != 10:
        raise RuntimeError("Full ablation repeat count must be 10.")
    if constants["STAGE2_EPOCHS"] != EXPECTED_EPOCHS:
        raise RuntimeError("Stage-2 epoch count changed unexpectedly.")
    global_seed = int(constants["GLOBAL_SEED"])
    target_seeds = [global_seed + index for index in range(10)]
    legacy_seeds = target_seeds[:LEGACY_SEED_COUNT]
    missing_seeds = target_seeds[LEGACY_SEED_COUNT:]
    out_dir = root / "outputs" / run_tag / "04_additional_ablation"
    raw_report = _audit_raw(out_dir / "additional_ablation_raw.csv", legacy_seeds)
    records = [
        _audit_model(out_dir, root / "splits" / run_tag, variant, seed)
        for seed in legacy_seeds
        for variant in VARIANTS
    ]
    legacy_full_seeds = [
        seed
        for seed in target_seeds
        if (
            out_dir / "models" / LEGACY_FULL_VARIANT / ("seed{0}".format(seed)) / "manifest.json"
        ).is_file()
    ]
    legacy_full_records = [
        _audit_model(out_dir, root / "splits" / run_tag, LEGACY_FULL_VARIANT, seed)
        for seed in legacy_full_seeds
    ]
    preexisting_extension_records: List[Dict[str, Any]] = []
    for seed in missing_seeds:
        for variant in VARIANTS:
            if (out_dir / "models" / variant / ("seed{0}".format(seed)) / "manifest.json").exists():
                preexisting_extension_records.append(
                    _audit_model(out_dir, root / "splits" / run_tag, variant, seed)
                )
    sidecars = _audit_channel_sidecars(root, run_tag)
    reference = _audit_full_reference(root / "configs" / "kaist_additional_ablation_full_reference.json")
    integrity_projection = [
        {
            "seed": row["seed"],
            "variant": row["variant"],
            "manifest": row["manifest"],
            "weights": row["weights"],
            "history": row["history"],
        }
        for row in records
    ]
    legacy_full_projection = [
        {
            "seed": row["seed"],
            "variant": row["variant"],
            "manifest": row["manifest"],
            "weights": row["weights"],
            "history": row["history"],
        }
        for row in legacy_full_records
    ]
    target_extension_slots = len(missing_seeds) * len(VARIANTS)
    return {
        "status": "PASS",
        "phase": "pre",
        "run_tag": run_tag,
        "target_seeds": target_seeds,
        "legacy_seeds": legacy_seeds,
        "missing_extension_seeds": missing_seeds,
        "raw": raw_report,
        "reused_variant_model_runs": len(records),
        "legacy_model_files_verified": len(records) * 3,
        "legacy_hashes_match_manifests": True,
        "legacy_hash_mtime_baseline_signature": sha256_json(integrity_projection),
        "legacy_mtime_archive_lock": "will be persisted before formal extension training",
        "legacy_full_model_runs_preserved": len(legacy_full_records),
        "legacy_full_seeds_preserved": legacy_full_seeds,
        "legacy_full_hash_mtime_baseline_signature": sha256_json(legacy_full_projection),
        "legacy_full_policy": "legacy_supplementary_full_not_used_in_final_table; no delete or overwrite",
        "full_reference": reference,
        "channel_sidecars": sidecars,
        "target_extension_model_runs": target_extension_slots,
        "preexisting_valid_extension_model_runs": len(preexisting_extension_records),
        "currently_missing_extension_model_runs": target_extension_slots - len(preexisting_extension_records),
        "full_model_training_planned": False,
        "training_was_run": False,
        "files_written": [],
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = audit_pre(root, args.run_tag)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
