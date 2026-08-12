from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy

from . import config
from .provenance import sha256_file, write_json
from .specs import manuscript_spec, spec_fingerprint


PROFILER_ARCHITECTURE = "isolated_cpu_flops_gpu_runtime"
FLOPS_PARTIAL_NAME = "flops_storage_partial.json"
RUNTIME_PARTIAL_NAME = "runtime_partial.json"
FINAL_OUTPUT_NAMES = (
    "efficiency_profile.csv",
    "efficiency_profile.json",
    "environment_manifest.json",
    "reporting_notes.txt",
)

FLOPS_METRIC_KEYS = (
    "trainable_params",
    "trainable_params_m",
    "flops",
    "flops_g",
    "macs_estimated",
    "macs_g_estimated",
    "weights_size_mb",
    "savedmodel_size_mb",
    "flop_profiler_note",
)
RUNTIME_METRIC_KEYS = (
    "device",
    "device_name",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p25_ms",
    "latency_p75_ms",
    "latency_p95_ms",
    "throughput_samples_per_s",
    "warmup_runs",
    "timed_runs",
    "gpu_allocator_current_mb",
    "gpu_allocator_peak_mb",
)


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot read valid JSON from {0}: {1}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object in {0}.".format(path))
    return value


def _checkpoint_missing_message(protocol: str, mode: str) -> str:
    return (
        "Efficiency profiling never trains or rebuilds a checkpoint. Required checkpoint files are missing. "
        "Run this first: python 01_prepare_kaist_checkpoint.py --mode {0} --protocol {1} "
        "--accept-kaist-spec"
    ).format(mode, protocol)


def resolve_checkpoint(protocol: str, mode: str) -> Dict[str, Any]:
    weights_path, manifest_path = config.checkpoint_paths(protocol, config.GLOBAL_SEED, mode)
    weights_path = weights_path.resolve()
    manifest_path = manifest_path.resolve()
    if not weights_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(_checkpoint_missing_message(protocol, mode))

    manifest = _read_json_object(manifest_path)
    expected_spec_fingerprint = spec_fingerprint(manuscript_spec("kaist"))
    if manifest.get("model_spec_fingerprint") != expected_spec_fingerprint:
        raise RuntimeError(
            "Checkpoint model specification does not match configs/manuscript_kaist.json: {0}".format(
                manifest_path
            )
        )

    training_spec = manifest.get("training_spec")
    if not isinstance(training_spec, dict):
        raise RuntimeError("Checkpoint manifest has no valid training_spec: {0}".format(manifest_path))
    if training_spec.get("protocol") != protocol:
        raise RuntimeError("Checkpoint protocol does not match the profiling request: {0}".format(manifest_path))
    if training_spec.get("seed") != config.GLOBAL_SEED:
        raise RuntimeError("Checkpoint seed does not match GLOBAL_SEED: {0}".format(manifest_path))

    expected_sha256 = manifest.get("weights_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise RuntimeError("Checkpoint manifest has no weights_sha256: {0}".format(manifest_path))
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint SHA-256 mismatch: {0}".format(weights_path))

    return {
        "weights_path": str(weights_path),
        "manifest_path": str(manifest_path),
        "sha256": actual_sha256,
        "seed": int(training_spec["seed"]),
    }


def worker_environment(
    worker: str,
    base: Mapping[str, str] = None,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    environment = dict(os.environ.copy() if base is None else base)
    worker_seed = config.GLOBAL_SEED if seed is None else int(seed)
    environment["PYTHONHASHSEED"] = str(worker_seed)
    if worker == "flops":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    elif worker == "runtime":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    else:
        raise ValueError("worker must be 'flops' or 'runtime'.")
    return environment


def validate_runtime_request(device: str, warmup: int, repeats: int) -> None:
    if device == "cpu":
        raise RuntimeError(
            "This isolated profiler requires GPU runtime profiling; CPU runtime fallback is prohibited. "
            "Use --device gpu (or --device auto, which still requires a visible GPU)."
        )
    if int(warmup) < 1:
        raise ValueError("--warmup must be a positive integer.")
    if int(repeats) < 1:
        raise ValueError("--repeats must be a positive integer.")


def build_worker_command(
    script_path: Path,
    worker: str,
    protocol: str,
    mode: str,
    out_dir: Path,
    checkpoint: Mapping[str, Any],
    warmup: int,
    repeats: int,
) -> Sequence[str]:
    command = [
        sys.executable,
        str(Path(script_path).resolve()),
        "--worker",
        worker,
        "--protocol",
        protocol,
        "--mode",
        mode,
        "--out-dir",
        str(Path(out_dir).resolve()),
        "--weights-path",
        str(checkpoint["weights_path"]),
        "--manifest-path",
        str(checkpoint["manifest_path"]),
    ]
    if worker == "runtime":
        command.extend(["--warmup", str(int(warmup)), "--repeats", str(int(repeats))])
    return command


def _run_worker(command: Sequence[str], environment: Mapping[str, str]) -> int:
    completed = subprocess.run(list(command), env=dict(environment), check=False)
    return int(completed.returncode)


def _remove_stale_outputs(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (FLOPS_PARTIAL_NAME, RUNTIME_PARTIAL_NAME) + FINAL_OUTPUT_NAMES:
        path = out_dir / name
        if path.is_file():
            path.unlink()
    saved_model_path = out_dir / "saved_model"
    if saved_model_path.is_dir():
        shutil.rmtree(str(saved_model_path))


def _load_partial(path: Path, kind: str, checkpoint_sha256: str) -> Dict[str, Any]:
    payload = _read_json_object(path)
    if payload.get("kind") != kind:
        raise RuntimeError("Unexpected partial kind in {0}.".format(path))
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("Partial/checkpoint SHA-256 mismatch in {0}.".format(path))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Partial has no metrics object: {0}".format(path))
    required = FLOPS_METRIC_KEYS if kind == "flops_storage" else RUNTIME_METRIC_KEYS
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError("Partial {0} is missing metrics: {1}".format(path, ", ".join(missing)))
    return payload


def finalize_profile_outputs(
    out_dir: Path,
    checkpoint: Mapping[str, Any],
    protocol: str,
    flops_worker_exit_code: int,
    runtime_worker_exit_code: int,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    if int(flops_worker_exit_code) != 0 or int(runtime_worker_exit_code) != 0:
        raise RuntimeError("Final efficiency results require two successful workers.")

    flops_payload = _load_partial(
        out_dir / FLOPS_PARTIAL_NAME, "flops_storage", str(checkpoint["sha256"])
    )
    runtime_payload = _load_partial(
        out_dir / RUNTIME_PARTIAL_NAME, "runtime", str(checkpoint["sha256"])
    )
    environment = runtime_payload.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError("Runtime partial has no environment manifest.")

    row: Dict[str, Any] = {
        "model": "MHFL-MCA",
        "protocol_checkpoint": protocol,
        **dict(runtime_payload["metrics"]),
        **dict(flops_payload["metrics"]),
        "profiler_architecture": PROFILER_ARCHITECTURE,
        "checkpoint_path": str(checkpoint["weights_path"]),
        "checkpoint_manifest_path": str(checkpoint["manifest_path"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "tf_gpu_allocator": "cuda_malloc_async",
        "tf_force_gpu_allow_growth": "true",
        "flops_worker_exit_code": int(flops_worker_exit_code),
        "runtime_worker_exit_code": int(runtime_worker_exit_code),
    }

    csv_path = out_dir / "efficiency_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    write_json(out_dir / "efficiency_profile.json", row)
    environment["profiler_architecture"] = PROFILER_ARCHITECTURE
    write_json(out_dir / "environment_manifest.json", environment)
    (out_dir / "reporting_notes.txt").write_text(
        "FLOPs are TensorFlow graph float operations for batch size 1; MACs are estimated as FLOPs/2. "
        "Latency excludes data loading and uses synchronized batch-1 GPU forward passes after warm-up. "
        "TensorFlow GPU memory is allocator current/peak memory measured after warm-up, not total board VRAM. "
        "FLOPs/storage and GPU runtime were measured in separate fresh Python interpreters. Cross-model "
        "deployment plots are valid only when every baseline is re-profiled with this same script, hardware, "
        "and software environment.\n",
        encoding="utf-8",
    )
    return row


def run_controller(args: Any, script_path: Path) -> Dict[str, Any]:
    validate_runtime_request(args.device, args.warmup, args.repeats)
    config.enforce_run_safety(
        args.mode,
        allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
        accept_kaist_spec=args.accept_kaist_spec,
        require_kaist_spec=True,
    )
    checkpoint = resolve_checkpoint(args.protocol, args.mode)
    out_dir = config.OUTPUT_ROOT / "03_efficiency"
    _remove_stale_outputs(out_dir)

    flops_command = build_worker_command(
        script_path,
        "flops",
        args.protocol,
        args.mode,
        out_dir,
        checkpoint,
        args.warmup,
        args.repeats,
    )
    flops_environment = worker_environment(
        "flops",
        os.environ.copy(),
        checkpoint["seed"],
    )
    flops_exit_code = _run_worker(flops_command, flops_environment)
    if flops_exit_code != 0:
        raise RuntimeError("CPU FLOPs/storage worker failed with exit code {0}.".format(flops_exit_code))

    runtime_command = build_worker_command(
        script_path,
        "runtime",
        args.protocol,
        args.mode,
        out_dir,
        checkpoint,
        args.warmup,
        args.repeats,
    )
    runtime_environment = worker_environment(
        "runtime",
        os.environ.copy(),
        checkpoint["seed"],
    )
    runtime_exit_code = _run_worker(runtime_command, runtime_environment)
    if runtime_exit_code != 0:
        raise RuntimeError(
            "GPU runtime worker failed with exit code {0}; CPU fallback is prohibited.".format(
                runtime_exit_code
            )
        )

    row = finalize_profile_outputs(
        out_dir,
        checkpoint,
        args.protocol,
        flops_exit_code,
        runtime_exit_code,
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("Outputs saved to:", out_dir)
    return row


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def _validate_worker_checkpoint(weights_path: Path, manifest_path: Path) -> Tuple[Dict[str, Any], str]:
    weights_path = Path(weights_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not weights_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Worker checkpoint files are missing.")
    manifest = _read_json_object(manifest_path)
    actual_sha256 = sha256_file(weights_path)
    expected_sha256 = manifest.get("weights_sha256")
    if not isinstance(expected_sha256, str) or actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError("Worker checkpoint SHA-256 mismatch: {0}".format(weights_path))
    if manifest.get("model_spec_fingerprint") != spec_fingerprint(manuscript_spec("kaist")):
        raise RuntimeError("Worker checkpoint model specification mismatch: {0}".format(manifest_path))
    return manifest, actual_sha256


def _worker_seed_from_manifest(manifest: Mapping[str, Any]) -> int:
    training_spec = manifest.get("training_spec")
    if isinstance(training_spec, dict) and "seed" in training_spec:
        return int(training_spec["seed"])
    return int(config.GLOBAL_SEED)


def _set_worker_random_seed(tf: Any, seed: int) -> None:
    value = int(seed)
    random.seed(value)
    numpy.random.seed(value)
    tf.random.set_seed(value)


def _build_checkpoint_model(tf: Any, weights_path: Path, manifest_path: Path) -> Tuple[Any, str]:
    manifest, checkpoint_sha256 = _validate_worker_checkpoint(
        weights_path,
        manifest_path,
    )
    worker_seed = _worker_seed_from_manifest(manifest)
    tf.keras.backend.clear_session()
    _set_worker_random_seed(tf, worker_seed)

    from .model import assert_expected_parameter_count, build_models

    spec = manuscript_spec("kaist")
    model, auxiliary = build_models(spec)
    del auxiliary
    assert_expected_parameter_count(model, spec)
    model.load_weights(str(Path(weights_path).resolve()))
    return model, checkpoint_sha256


def _configure_cpu_before_model_creation() -> Any:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("FLOPs worker requires CUDA_VISIBLE_DEVICES=-1 before TensorFlow import.")
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("FLOPs worker must not see a GPU.")
    return tf


def _configure_gpu_before_model_creation() -> Any:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Runtime worker requires CUDA_VISIBLE_DEVICES=0.")
    if os.environ.get("TF_GPU_ALLOCATOR") != "cuda_malloc_async":
        raise RuntimeError("Runtime worker requires TF_GPU_ALLOCATOR=cuda_malloc_async.")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("Runtime worker requires TF_FORCE_GPU_ALLOW_GROWTH=true.")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU runtime profiling requires a visible GPU. CPU fallback is prohibited.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be configured before TensorFlow initializes the device."
            ) from exc
    return tf


def _count_graph_flops(tf: Any, model: Any) -> int:
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function
    def forward(x1: Any, x2: Any) -> Any:
        return model([x1, x2], training=False)

    concrete = forward.get_concrete_function(
        tf.TensorSpec([1, config.SEGMENT_LENGTH, 1], tf.float32),
        tf.TensorSpec([1, config.SEGMENT_LENGTH, 1], tf.float32),
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def(add_shapes=True)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(graph=graph, cmd="op", options=options)
    if profile is None:
        raise RuntimeError("TensorFlow graph profiler returned no FLOPs result.")
    return int(profile.total_float_ops)


def run_flops_worker(
    out_dir: Path,
    weights_path: Path,
    manifest_path: Path,
    protocol: str,
    mode: str,
) -> Dict[str, Any]:
    tf = _configure_cpu_before_model_creation()
    model, checkpoint_sha256 = _build_checkpoint_model(tf, weights_path, manifest_path)
    flops = _count_graph_flops(tf, model)

    saved_model_path = Path(out_dir) / "saved_model"
    if saved_model_path.exists():
        shutil.rmtree(str(saved_model_path))
    model.save(str(saved_model_path), include_optimizer=False)
    import numpy as np

    trainable_params = int(
        sum(int(np.prod(variable.shape.as_list())) for variable in model.trainable_weights)
    )
    metrics: Dict[str, Any] = {
        "trainable_params": trainable_params,
        "trainable_params_m": float(trainable_params) / 1e6,
        "flops": flops,
        "flops_g": float(flops) / 1e9,
        "macs_estimated": int(flops // 2),
        "macs_g_estimated": float(flops) / 2e9,
        "weights_size_mb": Path(weights_path).stat().st_size / (1024.0 ** 2),
        "savedmodel_size_mb": _directory_size(saved_model_path) / (1024.0 ** 2),
        "flop_profiler_note": (
            "TensorFlow graph float operations; batch size = 1; MACs are estimated as FLOPs / 2."
        ),
    }
    payload = {
        "kind": "flops_storage",
        "protocol": protocol,
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics": metrics,
    }
    partial_path = Path(out_dir) / FLOPS_PARTIAL_NAME
    write_json(partial_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("FLOPs/storage partial saved to:", partial_path)
    return payload


def run_runtime_worker(
    out_dir: Path,
    weights_path: Path,
    manifest_path: Path,
    protocol: str,
    mode: str,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    validate_runtime_request("gpu", warmup, repeats)
    tf = _configure_gpu_before_model_creation()
    model, checkpoint_sha256 = _build_checkpoint_model(tf, weights_path, manifest_path)
    import numpy as np

    with tf.device("/GPU:0"):
        x1 = tf.zeros([1, config.SEGMENT_LENGTH, 1], dtype=tf.float32)
        x2 = tf.zeros([1, config.SEGMENT_LENGTH, 1], dtype=tf.float32)

        @tf.function
        def infer(first: Any, second: Any) -> Any:
            return model([first, second], training=False)

        for _ in range(int(warmup)):
            infer(x1, x2).numpy()

        tf.config.experimental.reset_memory_stats("GPU:0")
        elapsed_ms = []
        for _ in range(int(repeats)):
            started_ns = time.perf_counter_ns()
            infer(x1, x2).numpy()
            elapsed_ms.append((time.perf_counter_ns() - started_ns) / 1e6)

    values = np.asarray(elapsed_ms, dtype=np.float64)
    memory = tf.config.experimental.get_memory_info("GPU:0")
    metrics: Dict[str, Any] = {
        "device": "gpu",
        "device_name": "/GPU:0",
        "latency_mean_ms": float(np.mean(values)),
        "latency_median_ms": float(np.median(values)),
        "latency_p25_ms": float(np.quantile(values, 0.25)),
        "latency_p75_ms": float(np.quantile(values, 0.75)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "throughput_samples_per_s": float(1000.0 / np.mean(values)),
        "warmup_runs": int(warmup),
        "timed_runs": int(repeats),
        "gpu_allocator_current_mb": float(memory["current"]) / (1024.0 ** 2),
        "gpu_allocator_peak_mb": float(memory["peak"]) / (1024.0 ** 2),
    }
    from .provenance import environment_manifest

    payload = {
        "kind": "runtime",
        "protocol": protocol,
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics": metrics,
        "environment": environment_manifest(config.PROJECT_ROOT),
    }
    partial_path = Path(out_dir) / RUNTIME_PARTIAL_NAME
    write_json(partial_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Runtime partial saved to:", partial_path)
    return payload
