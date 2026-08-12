import ast
import os
import subprocess
import sys

import pytest

from mhfl_review import config


CONFIRMED_CURRENT_CHANNEL = "cDAQ9185-1F486B5Mod2/ai0"


def _config_value_in_clean_process(variable_name, value=None):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop(variable_name, None)
    if value is not None:
        environment[variable_name] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mhfl_review import config; print(config.KAIST_VIB_COLUMN)",
        ],
        cwd=str(config.SUITE_ROOT),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


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


def test_kaist_vibration_column_defaults_to_confirmed_x_direction():
    assert _config_value_in_clean_process("MHFL_KAIST_VIB_COLUMN") == 0


def test_kaist_vibration_column_environment_override():
    assert _config_value_in_clean_process("MHFL_KAIST_VIB_COLUMN", "3") == 3


def test_full_mode_prohibits_current_channel_fallback(monkeypatch):
    monkeypatch.setattr(config, "RUN_TAG", "full_channel_test")
    with pytest.raises(RuntimeError, match="fallback is prohibited"):
        config.enforce_run_safety(
            "full",
            allow_current_fallback=True,
            accept_kaist_spec=True,
        )


def test_data_loader_uses_exact_current_channel_name_match():
    function_source = _function_source(
        config.SUITE_ROOT / "mhfl_review" / "data.py",
        "_read_tdms_current",
    )
    assert "item[1] == config.CURRENT_CHANNEL_NAME" in function_source
    assert 'rule = "explicit_exact_name"' in function_source
    assert config.CURRENT_CHANNEL_REGEX == ""
    assert config.ALLOW_CURRENT_CHANNEL_FALLBACK is False


def test_confirmed_windows_environment_uses_exact_channel_without_fallback():
    script = (config.SUITE_ROOT / "windows" / "00_set_environment.bat").read_text(
        encoding="utf-8"
    )
    assert 'set "MHFL_CURRENT_CHANNEL_NAME={0}"'.format(
        CONFIRMED_CURRENT_CHANNEL
    ) in script
    assert 'set "MHFL_KAIST_VIB_COLUMN=0"' in script
    assert 'set "MHFL_ACCEPT_KAIST_SPEC=1"' in script
    assert "MHFL_ALLOW_CURRENT_FALLBACK" not in script
    assert "MHFL_CURRENT_CHANNEL_REGEX" not in script


def test_uo_data_defaults_are_unchanged():
    assert config.SEGMENT_LENGTH == 2048
    assert config.UO_SAMPLING_RATE == 42_000.0
    assert config.UO_VIB_COLUMN == 0
    assert config.UO_ACOUSTIC_COLUMN == 1
