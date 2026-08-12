import ast

import pytest

from mhfl_review import config


EXPECTED_LEARNING_RATE = 0.0004156294449523281
EXPECTED_BATCH_SIZE = 16
EXPECTED_OPTUNA_PARAMETERS = {
    "dropout_vib",
    "dropout_cur",
    "atten_dim",
    "n_layers_vib",
    "n_layers_cur",
    "lr",
    "batch_size",
}


def _function_source(path, function_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_confirmed_kaist_json_loads_exact_training_values():
    payload = config.load_confirmed_kaist_optuna_config()
    assert EXPECTED_OPTUNA_PARAMETERS.issubset(payload)
    assert payload["confirmation_status"] == "confirmed"
    assert payload["lr"] == EXPECTED_LEARNING_RATE
    assert payload["batch_size"] == EXPECTED_BATCH_SIZE
    assert config.KAIST_MANUSCRIPT_LEARNING_RATE == EXPECTED_LEARNING_RATE
    assert config.KAIST_MANUSCRIPT_BATCH_SIZE == EXPECTED_BATCH_SIZE


def test_uo_training_configuration_is_unchanged():
    assert config.DEFAULT_LEARNING_RATE == 1.0e-3
    assert config.DEFAULT_BATCH_SIZE == 16
    assert config.UO_MANUSCRIPT_LEARNING_RATE == 0.00139


def test_full_kaist_uses_confirmed_values_without_default_fallback():
    learning_rate, batch_size = config.require_confirmed_kaist_training_config("full")
    assert learning_rate == EXPECTED_LEARNING_RATE
    assert batch_size == EXPECTED_BATCH_SIZE
    assert learning_rate != config.DEFAULT_LEARNING_RATE

    train_source = _function_source(
        config.SUITE_ROOT / "mhfl_review" / "train.py", "load_or_train_kaist"
    )
    assert "require_confirmed_kaist_training_config" in train_source
    assert "DEFAULT_LEARNING_RATE" not in train_source
    assert "DEFAULT_BATCH_SIZE" not in train_source

    ablation_source = _function_source(
        config.SUITE_ROOT / "04_additional_ablation.py", "main"
    )
    assert "require_confirmed_kaist_training_config" in ablation_source
    assert "DEFAULT_LEARNING_RATE" not in ablation_source
    assert "DEFAULT_BATCH_SIZE" not in ablation_source


def test_full_kaist_rejects_unloaded_confirmation(monkeypatch):
    monkeypatch.setattr(config, "KAIST_OPTUNA_CONFIG_ERROR", "test configuration error")
    monkeypatch.setattr(config, "KAIST_MANUSCRIPT_LEARNING_RATE", None)
    monkeypatch.setattr(config, "KAIST_MANUSCRIPT_BATCH_SIZE", None)
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        config.require_confirmed_kaist_training_config("full")
