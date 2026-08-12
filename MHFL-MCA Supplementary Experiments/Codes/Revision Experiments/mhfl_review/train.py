from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config
from .data import (
    DatasetBundle,
    SplitData,
    add_gaussian_noise,
    apply_split_plan,
    build_external_test,
    create_uo_paper_split,
    get_or_create_split_plan,
    load_kaist_load,
    load_uo_dataset,
)
from .provenance import (
    checkpoint_signature,
    environment_manifest,
    sha256_file,
    sha256_json,
    validate_checkpoint_manifest,
    write_json,
)
from .specs import ModelSpec, manuscript_spec, spec_fingerprint

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for training. Activate the manuscript environment "
        "(Python 3.9.x, TensorFlow 2.7.x) before running these scripts."
    ) from exc

from .model import assert_expected_parameter_count, build_models


@dataclass(frozen=True)
class TrainingSpec:
    protocol: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    gradient_clip_norm: float = config.GRADIENT_CLIP_NORM
    base_snr_db: float = config.BASE_SNR_DB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.protocol not in {"standard", "stage2", "stage3"}:
            raise ValueError("Unsupported protocol: {0}".format(self.protocol))
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid epochs, batch size, or learning rate.")


@dataclass(frozen=True)
class TrainedModelBundle:
    model: tf.keras.Model
    auxiliary: Mapping[str, tf.keras.Model]
    model_spec: ModelSpec
    training_spec: TrainingSpec
    splits: Mapping[str, SplitData]
    checkpoint_manifest: Mapping[str, Any]


def configure_tensorflow() -> None:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)


def clear_session() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.argmax(np.asarray(y_prob), axis=1)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision_score(truth, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def predict_prob(model: tf.keras.Model, split_or_x1: Any, x2: Optional[np.ndarray] = None, batch_size: int = 64) -> np.ndarray:
    if isinstance(split_or_x1, SplitData):
        x1_values = split_or_x1.x1
        x2_values = split_or_x1.x2
    else:
        if x2 is None:
            raise ValueError("x2 is required when the first argument is not SplitData.")
        x1_values = np.asarray(split_or_x1)
        x2_values = np.asarray(x2)
    return np.asarray(model.predict([x1_values, x2_values], batch_size=int(batch_size), verbose=0))


def evaluate_split(model: tf.keras.Model, split: SplitData, batch_size: int = 64) -> Dict[str, float]:
    return metric_dict(split.y_int, predict_prob(model, split, batch_size=batch_size))


def evaluate_with_noise(
    model: tf.keras.Model,
    split: SplitData,
    snr_x1: float,
    snr_x2: float,
    seed: int,
    batch_size: int = 64,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    rng1 = np.random.RandomState(int(seed))
    rng2 = np.random.RandomState(int(seed) + 9973)
    x1_noisy = add_gaussian_noise(split.x1, snr_x1, rng1)
    x2_noisy = add_gaussian_noise(split.x2, snr_x2, rng2)
    probability = predict_prob(model, x1_noisy, x2_noisy, batch_size=batch_size)
    return metric_dict(split.y_int, probability), probability, x1_noisy, x2_noisy


def trimmed_mean_std(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) >= 3:
        array = np.sort(array)[1:-1]
    return float(np.mean(array)), float(np.std(array, ddof=0))


def _tf_add_noise_snr(x: tf.Tensor, snr_db: tf.Tensor) -> tf.Tensor:
    values = tf.cast(x, tf.float32)
    signal_power = tf.reduce_mean(tf.square(values), axis=[1, 2], keepdims=True) + 1e-12
    snr_linear = tf.pow(10.0, tf.cast(snr_db, tf.float32) / 10.0)
    noise_power = signal_power / snr_linear
    return values + tf.random.normal(tf.shape(values), dtype=tf.float32) * tf.sqrt(noise_power)


def _sample_snr(batch_size: tf.Tensor, low_db: float, high_db: float, bias_power: float) -> tf.Tensor:
    uniform = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
    biased = tf.pow(uniform, float(bias_power))
    return float(low_db) + (float(high_db) - float(low_db)) * biased


def _corrupt_one_modality(x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    batch = tf.shape(x1)[0]
    trigger = tf.random.uniform([batch, 1, 1]) < float(config.MODALITY_CORRUPT_PROB)
    choose_first = tf.random.uniform([batch, 1, 1]) < 0.5
    severe = tf.ones([batch, 1, 1], tf.float32) * float(config.MODALITY_EXTRA_SNR_DB)
    x1_bad = _tf_add_noise_snr(x1, severe)
    x2_bad = _tf_add_noise_snr(x2, severe)
    return (
        tf.where(tf.logical_and(trigger, choose_first), x1_bad, x1),
        tf.where(tf.logical_and(trigger, tf.logical_not(choose_first)), x2_bad, x2),
    )


def _kl_stop_gradient(teacher: tf.Tensor, student: tf.Tensor) -> tf.Tensor:
    epsilon = 1e-7
    teacher_safe = tf.stop_gradient(tf.clip_by_value(teacher, epsilon, 1.0 - epsilon))
    student_safe = tf.clip_by_value(student, epsilon, 1.0 - epsilon)
    return tf.reduce_mean(tf.reduce_sum(teacher_safe * tf.math.log(teacher_safe / student_safe), axis=1))


def _fixed_validation_view(split: SplitData, snr_db: float, seed: int) -> SplitData:
    return SplitData(
        add_gaussian_noise(split.x1, snr_db, np.random.RandomState(seed + 101)),
        add_gaussian_noise(split.x2, snr_db, np.random.RandomState(seed + 211)),
        split.y_onehot,
        split.y_int,
        split.sample_ids,
    )


def train_standard(
    model: tf.keras.Model,
    train_split: SplitData,
    val_split: SplitData,
    training_spec: TrainingSpec,
    output_dir: Path,
    run_name: str,
    verbose: int = 0,
) -> pd.DataFrame:
    training_spec.validate()
    seed_everything(training_spec.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=float(training_spec.learning_rate),
        clipnorm=float(training_spec.gradient_clip_norm),
    )
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        [train_split.x1, train_split.x2],
        train_split.y_onehot,
        validation_data=([val_split.x1, val_split.x2], val_split.y_onehot),
        epochs=int(training_spec.epochs),
        batch_size=int(training_spec.batch_size),
        shuffle=True,
        verbose=int(verbose),
    )
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.to_csv(output_dir / (run_name + ".history.csv"), index=False)
    return frame


def train_noise_protocol(
    model: tf.keras.Model,
    train_split: SplitData,
    val_split: SplitData,
    training_spec: TrainingSpec,
    output_dir: Path,
    run_name: str,
    verbose: int = 1,
) -> pd.DataFrame:
    """Train source-faithful Stage-2 or Stage-3 KAIST objectives."""
    training_spec.validate()
    if training_spec.protocol not in {"stage2", "stage3"}:
        raise ValueError("train_noise_protocol requires stage2 or stage3.")
    seed_everything(training_spec.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = tf.data.Dataset.from_tensor_slices(
        ((train_split.x1, train_split.x2), train_split.y_onehot)
    )
    dataset = dataset.shuffle(len(train_split), seed=training_spec.seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(int(training_spec.batch_size), drop_remainder=False)
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=float(training_spec.learning_rate),
        clipnorm=float(training_spec.gradient_clip_norm),
    )
    validation = _fixed_validation_view(val_split, training_spec.base_snr_db, training_spec.seed + 100_000)
    rows = []

    for epoch in range(1, int(training_spec.epochs) + 1):
        if training_spec.protocol == "stage2":
            lambda_value = float(config.CONSISTENCY_MAX_LAMBDA) * min(
                1.0, epoch / float(config.CONSISTENCY_WARMUP_EPOCHS)
            )
        else:
            lambda_value = float(config.STAGE3_MAX_LAMBDA) * min(
                1.0, epoch / float(config.STAGE3_WARMUP_EPOCHS)
            )
        epoch_values: Dict[str, List[float]] = {
            "loss": [],
            "base_ce": [],
            "noisy_ce": [],
            "hard_ce": [],
            "kl": [],
            "hard_kl": [],
        }
        for (batch_x1, batch_x2), batch_y in dataset:
            batch_n = tf.shape(batch_x1)[0]
            base_snr = tf.ones([batch_n, 1, 1], tf.float32) * float(training_spec.base_snr_db)
            if training_spec.protocol == "stage2":
                random_snr = _sample_snr(
                    batch_n,
                    config.CONSISTENCY_SNR_LOW_DB,
                    config.CONSISTENCY_SNR_HIGH_DB,
                    config.CONSISTENCY_SNR_BIAS_P,
                )
            else:
                random_snr = _sample_snr(
                    batch_n,
                    config.STAGE3_NOISY_SNR_LOW_DB,
                    config.STAGE3_NOISY_SNR_HIGH_DB,
                    config.STAGE3_SNR_BIAS_P,
                )
            base_x1 = _tf_add_noise_snr(batch_x1, base_snr)
            base_x2 = _tf_add_noise_snr(batch_x2, base_snr)
            noisy_x1 = _tf_add_noise_snr(batch_x1, random_snr)
            noisy_x2 = _tf_add_noise_snr(batch_x2, random_snr)
            noisy_x1, noisy_x2 = _corrupt_one_modality(noisy_x1, noisy_x2)

            hard_x1 = None
            hard_x2 = None
            if training_spec.protocol == "stage3":
                hard_snr = tf.ones([batch_n, 1, 1], tf.float32) * float(config.STAGE3_HARD_SNR_DB)
                hard_x1 = _tf_add_noise_snr(batch_x1, hard_snr)
                hard_x2 = _tf_add_noise_snr(batch_x2, hard_snr)
                hard_x1, hard_x2 = _corrupt_one_modality(hard_x1, hard_x2)

            with tf.GradientTape() as tape:
                p_base = model([base_x1, base_x2], training=True)
                p_noisy = model([noisy_x1, noisy_x2], training=True)
                base_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch_y, p_base))
                noisy_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch_y, p_noisy))
                kl_value = _kl_stop_gradient(p_base, p_noisy)
                regularization = tf.add_n(model.losses) if model.losses else tf.constant(0.0, tf.float32)
                if training_spec.protocol == "stage2":
                    loss = base_ce + tf.cast(lambda_value, tf.float32) * kl_value + regularization
                    hard_ce = tf.constant(0.0, tf.float32)
                    hard_kl = tf.constant(0.0, tf.float32)
                else:
                    assert hard_x1 is not None and hard_x2 is not None
                    p_hard = model([hard_x1, hard_x2], training=True)
                    hard_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch_y, p_hard))
                    hard_kl = _kl_stop_gradient(p_base, p_hard)
                    loss = (
                        base_ce
                        + float(config.STAGE3_NOISY_CE_WEIGHT) * noisy_ce
                        + tf.cast(lambda_value, tf.float32) * kl_value
                        + float(config.STAGE3_HARD_CE_WEIGHT) * hard_ce
                        + tf.cast(lambda_value * config.STAGE3_HARD_KL_WEIGHT, tf.float32) * hard_kl
                        + regularization
                    )
            gradients = tape.gradient(loss, model.trainable_variables)
            pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None]
            optimizer.apply_gradients(pairs)
            for key, value in (
                ("loss", loss),
                ("base_ce", base_ce),
                ("noisy_ce", noisy_ce),
                ("hard_ce", hard_ce),
                ("kl", kl_value),
                ("hard_kl", hard_kl),
            ):
                epoch_values[key].append(float(value.numpy()))

        validation_probability = predict_prob(model, validation)
        validation_metrics = metric_dict(validation.y_int, validation_probability)
        row = {key: float(np.mean(value)) for key, value in epoch_values.items()}
        row.update(
            {
                "epoch": epoch,
                "lambda": lambda_value,
                "val_accuracy": validation_metrics["accuracy"],
                "val_macro_f1": validation_metrics["macro_f1"],
            }
        )
        rows.append(row)
        if verbose:
            print(
                "[{0}] epoch={1:03d} loss={2:.4f} base_ce={3:.4f} kl={4:.4f} val_acc={5:.4f}".format(
                    run_name,
                    epoch,
                    row["loss"],
                    row["base_ce"],
                    row["kl"],
                    row["val_accuracy"],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / (run_name + ".history.csv"), index=False)
    return frame


def prepare_kaist_splits(
    n_train: int,
    seed: int,
    rebuild_cache: bool = False,
    rebuild_split: bool = False,
    allow_current_fallback: Optional[bool] = None,
) -> Tuple[Dict[str, SplitData], Dict[str, Any]]:
    source_bundle = load_kaist_load("0Nm", rebuild_cache, allow_current_fallback)
    load2_bundle = load_kaist_load("2Nm", rebuild_cache, allow_current_fallback)
    load4_bundle = load_kaist_load("4Nm", rebuild_cache, allow_current_fallback)
    split_path = config.SPLIT_ROOT / ("kaist_source_seed{0}.json".format(int(seed)))
    plan = get_or_create_split_plan(
        source_bundle,
        split_path,
        seed=int(seed),
        max_train_per_class=30,
        val_per_class=config.VAL_PER_CLASS_KAIST,
        test_per_class=None,
        rebuild=rebuild_split,
    )
    source = apply_split_plan(source_bundle, plan, n_train_per_class=int(n_train))
    splits = {
        "train": source["train"],
        "val": source["val"],
        "source_test": source["test"],
        "2Nm": build_external_test(load2_bundle, seed=seed + 2),
        "4Nm": build_external_test(load4_bundle, seed=seed + 4),
    }
    metadata = {
        "plan": plan,
        "plan_signature": plan["signature"],
        "data_signatures": {
            "0Nm": source_bundle.metadata["signature"]["signature"],
            "2Nm": load2_bundle.metadata["signature"]["signature"],
            "4Nm": load4_bundle.metadata["signature"]["signature"],
        },
    }
    return splits, metadata


def prepare_uo_splits(
    n_train: int,
    seed: int,
    rebuild_cache: bool = False,
    rebuild_split: bool = False,
) -> Tuple[Dict[str, SplitData], Dict[str, Any]]:
    bundle = load_uo_dataset(rebuild_cache=rebuild_cache)
    split_path = config.SPLIT_ROOT / ("uo_nested_seed{0}.json".format(int(seed)))
    plan = get_or_create_split_plan(
        bundle,
        split_path,
        seed=int(seed),
        max_train_per_class=max(config.UO_MAX_TRAIN_PER_CLASS, max(config.LOWSHOT_N_GRID)),
        val_per_class=config.VAL_PER_CLASS_UO,
        test_per_class=config.TEST_PER_CLASS_UO,
        rebuild=rebuild_split,
    )
    splits = apply_split_plan(bundle, plan, n_train_per_class=int(n_train))
    metadata = {
        "plan": plan,
        "plan_signature": plan["signature"],
        "data_signature": bundle.metadata["signature"]["signature"],
    }
    return splits, metadata


def prepare_uo_paper_splits(
    n_train: int,
    seed: int,
    rebuild_cache: bool = False,
    rebuild_split: bool = False,
) -> Tuple[Dict[str, SplitData], Dict[str, Any]]:
    bundle = load_uo_dataset(rebuild_cache=rebuild_cache)
    # The helper stores a per-N plan. Delete/rebuild only when explicitly requested.
    split_path = config.SPLIT_ROOT / ("uo_paper_seed{0}_N{1}.json".format(int(seed), int(n_train)))
    if rebuild_split and split_path.exists():
        split_path.unlink()
    splits = create_uo_paper_split(bundle, seed=int(seed), n_train_per_class=int(n_train))
    plan = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = {
        "plan": plan,
        "plan_signature": plan["signature"],
        "data_signature": bundle.metadata["signature"]["signature"],
        "protocol": "uploaded_uo_first_N_train_remainder_test",
    }
    return splits, metadata


def _model_summary_text(model: tf.keras.Model) -> str:
    rows: List[str] = []
    model.summary(print_fn=rows.append)
    return "\n".join(rows) + "\n"


def _write_checkpoint_manifest(
    weights_path: Path,
    manifest_path: Path,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    split_signature: str,
    data_signature: str,
    history_path: Path,
) -> Dict[str, Any]:
    signature = checkpoint_signature(model_spec, training_spec, split_signature, data_signature)
    manifest = {
        "suite_version": config.SUITE_VERSION,
        "checkpoint_signature": signature,
        "model_spec": model_spec.to_dict(),
        "model_spec_fingerprint": spec_fingerprint(model_spec),
        "training_spec": training_spec.to_dict(),
        "split_signature": split_signature,
        "data_signature": data_signature,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "history_path": str(history_path),
        "created_unix": time.time(),
        "environment": environment_manifest(config.PROJECT_ROOT),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_or_train_variant_cached(
    model_spec: ModelSpec,
    train_split: SplitData,
    val_split: SplitData,
    training_spec: TrainingSpec,
    output_dir: Path,
    run_name: str,
    protocol_signature: Mapping[str, Any],
    force: bool = False,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, Mapping[str, tf.keras.Model], pd.DataFrame, Mapping[str, Any]]:
    """Resume a reviewer ablation only when model/training/split signatures match exactly."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.weights.h5"
    history_path = output_dir / (run_name + ".history.csv")
    manifest_path = output_dir / "manifest.json"
    expected_payload = {
        "suite_version": config.SUITE_VERSION,
        "model_spec": model_spec.to_dict(),
        "model_spec_fingerprint": spec_fingerprint(model_spec),
        "training_spec": training_spec.to_dict(),
        "protocol_signature": dict(protocol_signature),
    }
    expected_signature = sha256_json(expected_payload)

    clear_session()
    seed_everything(training_spec.seed)
    model, auxiliary = build_models(model_spec)
    assert_expected_parameter_count(model, model_spec)
    if weights_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") != expected_signature:
            raise RuntimeError(
                "Cached reviewer variant is incompatible with the requested spec/protocol: {0}. "
                "Use --force only after verifying the intended change.".format(output_dir)
            )
        expected_hash = manifest.get("weights_sha256")
        if expected_hash and sha256_file(weights_path) != expected_hash:
            raise RuntimeError("Cached reviewer-variant weights hash mismatch: {0}".format(weights_path))
        model.load_weights(str(weights_path))
        history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
        return model, auxiliary, history, manifest

    if training_spec.protocol == "standard":
        history = train_standard(model, train_split, val_split, training_spec, output_dir, run_name, verbose)
    else:
        history = train_noise_protocol(model, train_split, val_split, training_spec, output_dir, run_name, verbose)
    model.save_weights(str(weights_path))
    history.to_csv(history_path, index=False)
    manifest = dict(expected_payload)
    manifest.update(
        {
            "run_signature": expected_signature,
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "history_path": str(history_path),
            "created_unix": time.time(),
            "environment": environment_manifest(config.PROJECT_ROOT),
        }
    )
    write_json(manifest_path, manifest)
    return model, auxiliary, history, manifest


def load_or_train_kaist(
    protocol: str,
    mode: str,
    seed: int,
    force: bool = False,
    rebuild_cache: bool = False,
    rebuild_split: bool = False,
    accept_kaist_spec: bool = False,
    model_spec: Optional[ModelSpec] = None,
    verbose: int = 1,
) -> TrainedModelBundle:
    config.enforce_run_safety(
        mode,
        allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
        accept_kaist_spec=accept_kaist_spec,
        require_kaist_spec=True,
    )
    kaist_learning_rate, kaist_batch_size = config.require_confirmed_kaist_training_config(mode)
    configure_tensorflow()
    spec = manuscript_spec("kaist") if model_spec is None else model_spec
    epochs = config.FAST_EPOCHS if mode == "fast" else (config.STAGE2_EPOCHS if protocol == "stage2" else config.STAGE3_EPOCHS)
    training_spec = TrainingSpec(
        protocol=protocol,
        epochs=epochs,
        batch_size=kaist_batch_size,
        learning_rate=kaist_learning_rate,
        seed=int(seed),
    )
    splits, split_meta = prepare_kaist_splits(
        n_train=30,
        seed=int(seed),
        rebuild_cache=rebuild_cache,
        rebuild_split=rebuild_split,
        allow_current_fallback=config.ALLOW_CURRENT_CHANNEL_FALLBACK,
    )
    data_signature = sha256_json(split_meta["data_signatures"])
    split_signature = str(split_meta["plan_signature"])
    weights_path, manifest_path = config.checkpoint_paths(protocol, seed, mode)
    expected_signature = checkpoint_signature(spec, training_spec, split_signature, data_signature)

    clear_session()
    seed_everything(seed)
    model, auxiliary = build_models(spec)
    assert_expected_parameter_count(model, spec)

    manifest: Dict[str, Any]
    if weights_path.is_file() and manifest_path.is_file() and not force:
        manifest = validate_checkpoint_manifest(manifest_path, expected_signature, weights_path)
        model.load_weights(str(weights_path))
    else:
        run_dir = config.CHECKPOINT_ROOT / ("train_{0}_{1}_seed{2}".format(protocol, mode, int(seed)))
        history = train_noise_protocol(
            model,
            splits["train"],
            splits["val"],
            training_spec,
            run_dir,
            run_name="kaist_{0}".format(protocol),
            verbose=verbose,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(weights_path))
        history_path = run_dir / ("kaist_{0}.history.csv".format(protocol))
        manifest = _write_checkpoint_manifest(
            weights_path,
            manifest_path,
            spec,
            training_spec,
            split_signature,
            data_signature,
            history_path,
        )
        (run_dir / "model_summary.txt").write_text(_model_summary_text(model), encoding="utf-8")
        history.to_csv(history_path, index=False)
    return TrainedModelBundle(model, auxiliary, spec, training_spec, splits, manifest)


def train_variant(
    model_spec: ModelSpec,
    train_split: SplitData,
    val_split: SplitData,
    training_spec: TrainingSpec,
    output_dir: Path,
    run_name: str,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, Mapping[str, tf.keras.Model], pd.DataFrame]:
    clear_session()
    seed_everything(training_spec.seed)
    model, auxiliary = build_models(model_spec)
    if training_spec.protocol == "standard":
        history = train_standard(model, train_split, val_split, training_spec, output_dir, run_name, verbose)
    else:
        history = train_noise_protocol(model, train_split, val_split, training_spec, output_dir, run_name, verbose)
    return model, auxiliary, history
