from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mhfl_review import config
from mhfl_review import profiling


SUITE_ROOT = config.SUITE_ROOT
SCRIPT_PATH = SUITE_ROOT / "03_profile_efficiency.py"
PROFILING_PATH = SUITE_ROOT / "mhfl_review" / "profiling.py"


def _top_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def _function_source(path: Path, function_name: str) -> str:
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


def _checkpoint_payload(tmp_path: Path):
    return {
        "weights_path": str(tmp_path / "checkpoint.weights.h5"),
        "manifest_path": str(tmp_path / "checkpoint.manifest.json"),
        "sha256": "a" * 64,
    }


def _write_valid_partial(path: Path, kind: str, checkpoint_sha256: str) -> None:
    if kind == "flops_storage":
        metrics = {name: 1 for name in profiling.FLOPS_METRIC_KEYS}
        metrics["flop_profiler_note"] = "test"
    else:
        metrics = {name: 1 for name in profiling.RUNTIME_METRIC_KEYS}
        metrics["device"] = "gpu"
        metrics["device_name"] = "/GPU:0"
    payload = {
        "kind": kind,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics": metrics,
    }
    if kind == "runtime":
        payload["environment"] = {"tensorflow": "test"}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_controller_has_no_top_level_tensorflow_train_or_model_import():
    script_imports = _top_level_imports(SCRIPT_PATH)
    profiling_imports = _top_level_imports(PROFILING_PATH)
    forbidden = {"tensorflow", "mhfl_review.train", "mhfl_review.model"}
    assert forbidden.isdisjoint(script_imports)
    assert forbidden.isdisjoint(profiling_imports)

    before = set(__import__("sys").modules)
    spec = importlib.util.spec_from_file_location("efficiency_controller_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported_by_controller = set(__import__("sys").modules) - before
    assert "tensorflow" not in imported_by_controller
    assert "mhfl_review.train" not in imported_by_controller
    assert "mhfl_review.model" not in imported_by_controller


def test_flops_worker_environment_hides_gpu():
    environment = profiling.worker_environment(
        "flops",
        {"CUDA_VISIBLE_DEVICES": "0"},
        seed=20260806,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert environment["PYTHONHASHSEED"] == "20260806"


def test_runtime_worker_environment_is_explicit_and_growth_enabled():
    environment = profiling.worker_environment("runtime", {}, seed=20260806)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert environment["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert environment["PYTHONHASHSEED"] == "20260806"


def test_set_worker_random_seed_sets_python_numpy_and_tensorflow_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(
        profiling.random,
        "seed",
        lambda value: calls.append(("python", value)),
    )
    monkeypatch.setattr(
        profiling.numpy.random,
        "seed",
        lambda value: calls.append(("numpy", value)),
    )
    fake_tf = SimpleNamespace(
        random=SimpleNamespace(
            set_seed=lambda value: calls.append(("tensorflow", value))
        )
    )

    profiling._set_worker_random_seed(fake_tf, 20260807)

    assert calls == [
        ("python", 20260807),
        ("numpy", 20260807),
        ("tensorflow", 20260807),
    ]


def test_checkpoint_manifest_seed_is_set_before_model_build(monkeypatch, tmp_path):
    events = []
    manifest = {"training_spec": {"seed": 314159}}

    def validate_checkpoint(weights_path, manifest_path):
        events.append(("manifest", manifest_path))
        return manifest, "a" * 64

    def set_seed(tf, seed):
        events.append(("seed", seed))

    class FakeModel:
        def load_weights(self, path):
            events.append(("load_weights", path))

    fake_model = FakeModel()
    spec_token = object()

    fake_model_module = ModuleType("mhfl_review.model")

    def build_models(spec):
        events.append(("build_models", spec))
        return fake_model, {"unused": object()}

    def assert_expected_parameter_count(model, spec):
        events.append(("assert_expected_parameter_count", model, spec))

    fake_model_module.build_models = build_models
    fake_model_module.assert_expected_parameter_count = assert_expected_parameter_count

    fake_tf = SimpleNamespace(
        keras=SimpleNamespace(
            backend=SimpleNamespace(
                clear_session=lambda: events.append(("clear_session",))
            )
        )
    )

    monkeypatch.setattr(
        profiling,
        "_validate_worker_checkpoint",
        validate_checkpoint,
    )
    monkeypatch.setattr(profiling, "_set_worker_random_seed", set_seed)
    monkeypatch.setattr(
        profiling,
        "manuscript_spec",
        lambda case: spec_token,
    )
    monkeypatch.setitem(sys.modules, "mhfl_review.model", fake_model_module)

    model, checkpoint_sha256 = profiling._build_checkpoint_model(
        fake_tf,
        tmp_path / "checkpoint.weights.h5",
        tmp_path / "checkpoint.manifest.json",
    )

    assert model is fake_model
    assert checkpoint_sha256 == "a" * 64
    assert [event[0] for event in events] == [
        "manifest",
        "clear_session",
        "seed",
        "build_models",
        "assert_expected_parameter_count",
        "load_weights",
    ]
    assert events[2] == ("seed", 314159)


def test_worker_seed_falls_back_to_global_seed_when_manifest_seed_missing(
    monkeypatch,
):
    monkeypatch.setattr(config, "GLOBAL_SEED", 271828)
    assert profiling._worker_seed_from_manifest({"training_spec": {}}) == 271828
    assert profiling._worker_seed_from_manifest({}) == 271828


def test_worker_environments_are_independent_and_do_not_mutate_parent():
    parent = {
        "CUDA_VISIBLE_DEVICES": "parent",
        "TF_DETERMINISTIC_OPS": "1",
    }

    flops_environment = profiling.worker_environment(
        "flops",
        parent,
        seed=20260806,
    )
    runtime_environment = profiling.worker_environment(
        "runtime",
        parent,
        seed=20260806,
    )

    assert flops_environment is not runtime_environment
    assert flops_environment is not parent
    assert runtime_environment is not parent
    assert parent["CUDA_VISIBLE_DEVICES"] == "parent"
    assert "PYTHONHASHSEED" not in parent
    assert flops_environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert runtime_environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert flops_environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert flops_environment["TF_DETERMINISTIC_OPS"] == "1"
    assert runtime_environment["TF_DETERMINISTIC_OPS"] == "1"


def test_controller_uses_fresh_parent_environment_copy_for_each_worker():
    source = _function_source(PROFILING_PATH, "run_controller")
    assert source.count("os.environ.copy()") == 2


def test_full_checkpoint_missing_refuses_automatic_training(monkeypatch, tmp_path):
    missing_weights = tmp_path / "missing.weights.h5"
    missing_manifest = tmp_path / "missing.manifest.json"
    monkeypatch.setattr(
        config,
        "checkpoint_paths",
        lambda protocol, seed, mode: (missing_weights, missing_manifest),
    )
    with pytest.raises(FileNotFoundError, match="01_prepare_kaist_checkpoint.py --mode full"):
        profiling.resolve_checkpoint("stage2", "full")
    assert "load_or_train_kaist" not in PROFILING_PATH.read_text(encoding="utf-8")


def test_final_output_requires_both_valid_partials(tmp_path):
    checkpoint = _checkpoint_payload(tmp_path)
    _write_valid_partial(
        tmp_path / profiling.FLOPS_PARTIAL_NAME,
        "flops_storage",
        checkpoint["sha256"],
    )
    with pytest.raises(FileNotFoundError):
        profiling.finalize_profile_outputs(tmp_path, checkpoint, "stage2", 0, 0)
    assert not (tmp_path / "efficiency_profile.csv").exists()

    _write_valid_partial(
        tmp_path / profiling.RUNTIME_PARTIAL_NAME,
        "runtime",
        checkpoint["sha256"],
    )
    row = profiling.finalize_profile_outputs(tmp_path, checkpoint, "stage2", 0, 0)
    assert (tmp_path / "efficiency_profile.csv").is_file()
    assert row["profiler_architecture"] == "isolated_cpu_flops_gpu_runtime"


def test_gpu_failure_cannot_fall_back_to_cpu():
    with pytest.raises(RuntimeError, match="CPU runtime fallback is prohibited"):
        profiling.validate_runtime_request("cpu", 100, 1000)
    source = _function_source(PROFILING_PATH, "run_controller")
    assert "CPU fallback is prohibited" in source
    assert "select_device" not in source


def test_runtime_command_preserves_requested_warmup_and_repeats(tmp_path):
    command = profiling.build_worker_command(
        SCRIPT_PATH,
        "runtime",
        "stage2",
        "full",
        tmp_path,
        _checkpoint_payload(tmp_path),
        warmup=100,
        repeats=1000,
    )
    assert command[command.index("--warmup") + 1] == "100"
    assert command[command.index("--repeats") + 1] == "1000"


def test_gpu_worker_does_not_swallow_memory_growth_failure():
    source = _function_source(PROFILING_PATH, "_configure_gpu_before_model_creation")
    tree = ast.parse(source)
    assert "except RuntimeError as exc:" in source
    assert "raise RuntimeError(" in source
    assert "except Exception" not in source
    assert not any(isinstance(node, ast.Pass) for node in ast.walk(tree))


def test_workers_build_checkpoint_only_without_training_or_data_loading():
    source = PROFILING_PATH.read_text(encoding="utf-8")
    assert "from .model import assert_expected_parameter_count, build_models" in source
    assert "model.load_weights" in source
    assert "load_or_train_kaist" not in source
    assert "prepare_kaist_splits" not in source
    assert "load_kaist_dataset" not in source
    assert "model.compile" not in source
    assert "tf.keras.optimizers" not in source
    assert "optimizer =" not in source.lower()
    assert "save_weights" not in source
    assert ".fit(" not in source
    assert 'TF_DETERMINISTIC_OPS\"] = \"0\"' not in source
