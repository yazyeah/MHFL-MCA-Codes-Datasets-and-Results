from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mhfl_review import config


EXPECTED_PARAMS = {
    "dropout_vib": 0.4223065128432895,
    "dropout_aco": 0.3497078380248016,
    "atten_dim": 256,
    "n_layers_vib": 4,
    "n_layers_aco": 5,
    "lr": 0.0013865546776879656,
    "batch_size": 16,
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


uo_audit = _load_module(
    "uo_optuna_confirmation_audit_test_module",
    config.SUITE_ROOT / "audit_uo_main_reference.py",
)
lowshot = _load_module(
    "uo_optuna_confirmation_lowshot_test_module",
    config.SUITE_ROOT / "05_lowshot_threshold.py",
)


def test_chinese_best_parameter_line_is_recognized_exactly():
    text = "找到的最佳参数: {0}\n".format(repr(EXPECTED_PARAMS))
    candidates = [uo_audit.normalize_params(value) for value in uo_audit.parse_dict_candidates(text)]
    assert candidates == [EXPECTED_PARAMS]


def test_retained_log_metadata_is_tied_to_exact_winning_trial(tmp_path: Path):
    log_path = tmp_path / "best_parameters.txt"
    lines = ["A new study created in memory with name: no-name-0b7f7870-670c-450d-a7d9-781d2c4ed78a"]
    lines.extend("Trial {0} pruned.".format(index) for index in range(30) if index != 22)
    lines.append(
        "Trial 22 finished with value: 0.994434118270874 and parameters: {0}. "
        "Best is trial 22 with value: 0.994434118270874.".format(repr(EXPECTED_PARAMS))
    )
    log_path.write_text("\n".join(lines), encoding="utf-8")

    metadata = uo_audit.parse_optuna_metadata(log_path, EXPECTED_PARAMS)
    assert metadata == {
        "study_name": "no-name-0b7f7870-670c-450d-a7d9-781d2c4ed78a",
        "best_trial_number": 22,
            "best_value": 0.994434118270874,
            "number_of_trials": 30,
            "completed_trials": 1,
            "pruned_trials": 29,
            "storage": "in-memory (no persistent Optuna database declared in the retained run log)",
            "sampler": "Optuna default sampler (not explicitly printed)",
        "pruner": "MedianPruner",
    }


def test_confirmed_json_contains_exact_parameters_metadata_and_live_evidence_hashes():
    path = config.SUITE_ROOT / "configs" / "uo_optuna_confirmed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "CONFIRMED"
    assert payload["confirmation_status"] == "confirmed_from_original_uo_artifact"
    assert payload["parameter_candidate_count"] == 1
    assert payload["parameter_candidates"] == [EXPECTED_PARAMS]
    assert {key: payload[key] for key in EXPECTED_PARAMS} == EXPECTED_PARAMS
    assert payload["study_name"] == "no-name-0b7f7870-670c-450d-a7d9-781d2c4ed78a"
    assert payload["best_trial_number"] == 22
    assert payload["best_value"] == pytest.approx(0.994434118270874, rel=0.0, abs=0.0)
    assert payload["number_of_trials"] == 30
    assert payload["completed_trials"] == 12
    assert payload["pruned_trials"] == 18
    assert payload["storage"] == "in-memory (no persistent Optuna database declared in the retained run log)"
    assert payload["pruner"] == "MedianPruner"
    assert payload["summary_evidence"]["anchors_passed"] is True

    for path_key, hash_key, container in (
        ("evidence_path", "evidence_sha256", payload),
        ("path", "sha256", payload["summary_evidence"]),
        ("path", "sha256", payload["source_program_evidence"]),
    ):
        evidence_path = Path(str(container[path_key]))
        if not evidence_path.is_absolute():
            evidence_path = config.SUITE_ROOT / evidence_path
        assert evidence_path.is_file()
        assert _sha256(evidence_path) == str(container[hash_key]).lower()


def _temporary_confirmation(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    evidence_path = tmp_path / "best_parameters.txt"
    summary_path = tmp_path / "Final_Summary_Stats.csv"
    source_path = tmp_path / "MHCNN_Oputuna_Experiment.py"
    config_path = tmp_path / "uo_optuna_confirmed.json"
    evidence_path.write_text("找到的最佳参数: {0}\n".format(repr(EXPECTED_PARAMS)), encoding="utf-8")
    summary_path.write_text("Samples,Acc_Mean\n5,0.9279\n10,0.9707\n", encoding="utf-8")
    source_path.write_text("# retained source evidence\n", encoding="utf-8")
    payload = {
        **EXPECTED_PARAMS,
        "status": "CONFIRMED",
        "confirmation_status": "confirmed_from_original_uo_artifact",
        "parameter_candidate_count": 1,
        "parameter_candidates": [EXPECTED_PARAMS],
        "best_trial_number": 22,
        "best_value": 0.994434118270874,
        "number_of_trials": 30,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "summary_evidence": {
            "path": str(summary_path.resolve()),
            "sha256": _sha256(summary_path),
            "anchors_passed": True,
        },
        "source_program_evidence": {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
            "protocol_checks": {"source_exact": True},
        },
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path, evidence_path, summary_path, source_path


def test_loader_accepts_exact_confirmation_and_rejects_primary_evidence_sha_drift(tmp_path: Path):
    config_path, evidence_path, _summary_path, _source_path = _temporary_confirmation(tmp_path)
    loaded = lowshot.load_confirmed_hparams(config_path)
    assert {key: loaded[key] for key in EXPECTED_PARAMS} == EXPECTED_PARAMS
    assert loaded["source_evidence_sha256"] == _sha256(evidence_path)

    evidence_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Optuna evidence is missing or its SHA-256 has changed"):
        lowshot.load_confirmed_hparams(config_path)


def test_loader_rejects_summary_evidence_sha_drift(tmp_path: Path):
    config_path, _evidence_path, summary_path, _source_path = _temporary_confirmation(tmp_path)
    summary_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary evidence is missing or its SHA-256 has changed"):
        lowshot.load_confirmed_hparams(config_path)


def test_loader_rejects_source_program_sha_drift(tmp_path: Path):
    config_path, _evidence_path, _summary_path, source_path = _temporary_confirmation(tmp_path)
    source_path.write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-program evidence is missing, changed, or incomplete"):
        lowshot.load_confirmed_hparams(config_path)


def test_loader_rejects_unconfirmed_or_incomplete_configuration(tmp_path: Path):
    config_path, _evidence_path, _summary_path, _source_path = _temporary_confirmation(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["status"] = "BLOCKED"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="original UO Optuna artifact"):
        lowshot.load_confirmed_hparams(config_path)

    payload["status"] = "CONFIRMED"
    del payload["batch_size"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing: batch_size"):
        lowshot.load_confirmed_hparams(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 16.0, "exact JSON integer"),
        ("atten_dim", True, "exact JSON integer"),
        ("lr", float("nan"), "finite JSON number"),
    ],
)
def test_loader_rejects_non_exact_or_non_finite_json_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    config_path, _evidence_path, _summary_path, _source_path = _temporary_confirmation(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = value
    config_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        lowshot.load_confirmed_hparams(config_path)
