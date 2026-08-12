"""Experiment 05 - controlled extreme low-shot CAIM sensitivity on UO.

Purpose: compare Full MHFL-MCA with a matched no-CAIM control for N=1..10.
Protocol: paired source-aligned splits, ten runs per N, independent metric trimming.
Outputs: raw/summary/paired CSV files and explicit Table-5 anchor/post gates.
"""

from __future__ import annotations

import os

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.data import DatasetBundle, SplitData, load_uo_dataset
from mhfl_review.provenance import sha256_file, sha256_json, write_json
from mhfl_review.stats import mean_sd, trimmed_mean_sd

VARIANT_NAMES = ("full", "no_caim")
N_GRID_FULL = (1, 2, 3, 4, 5, 7, 10)
N_GRID_FAST = (5, 10)
RUNS_PER_N_FULL = 10
TRIMMED_AGGREGATION = "trimmed_mean_sd_drop_one_high_one_low"
FAST_AGGREGATION = "mean_sd_untrimmed_fast"

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

EXPECTED_FULL_PARAMETER_COUNT = 5_111_759
TRAINING_PROTOCOL_ID = "uo_source_aligned_paired_deterministic_extension_v1"

RAW_COLUMNS = (
    "variant",
    "n_train",
    "run_idx",
    "seed",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "train_accuracy",
    "heldout_accuracy",
    "generalization_gap",
    "split_signature",
    "data_signature",
    "hyperparameter_signature",
    "source_type",
    "train_time_s",
)

SUMMARY_METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "train_accuracy",
    "heldout_accuracy",
    "generalization_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-aligned R3-2 UO sensitivity study. The final manuscript assets are "
            "written only when Full MHFL-MCA reproduces Table 5 at N=5 and N=10 to four decimals."
        )
    )
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--uo-config",
        type=Path,
        default=Path("configs/uo_optuna_confirmed.json"),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional output tag. When omitted, MHFL_RUN_TAG/config.RUN_TAG is used.",
    )
    return parser.parse_args()


def configure_tensorflow() -> Any:
    # The original UO executable did not enable deterministic ops. Do this only
    # for the standalone worker, not while the module is imported by pytest.
    os.environ["TF_DETERMINISTIC_OPS"] = "0"
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Paper-aligned UO training requires a visible GPU; CPU fallback is not permitted.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be configured before TensorFlow initializes the device."
            ) from exc
    return tf


def set_paired_model_seed(tf: Any, seed: int) -> None:
    """Seed model RNGs without disturbing the source-exact NumPy split/shuffle stream."""
    value = int(seed)
    random.seed(value)
    tf.random.set_seed(value)


def paper_seed(n_train: int, run_idx: int) -> int:
    return int(n_train) * 100 + int(run_idx)


def expected_seed_map(n_grid: Sequence[int], runs_per_n: int) -> Dict[int, List[int]]:
    return {
        int(n_value): [paper_seed(int(n_value), run_idx) for run_idx in range(1, int(runs_per_n) + 1)]
        for n_value in n_grid
    }


def bind_dataset_content_hashes(bundle: DatasetBundle) -> Tuple[DatasetBundle, Dict[str, object]]:
    metadata = dict(bundle.metadata)
    signature = metadata.get("signature", {})
    if not isinstance(signature, Mapping):
        raise RuntimeError("UO dataset metadata has no source-file manifest.")
    source_rows = signature.get("files", [])
    if not isinstance(source_rows, list) or len(source_rows) != 14:
        raise RuntimeError("The source-aligned UO protocol requires exactly 14 source MAT files.")
    inventory: List[Dict[str, object]] = []
    for source in source_rows:
        if not isinstance(source, Mapping):
            raise RuntimeError("UO source-file manifest contains an invalid record.")
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            raise RuntimeError("UO source MAT file is missing: {0}".format(path))
        stat = path.stat()
        if int(source.get("size", -1)) != int(stat.st_size) or int(source.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            raise RuntimeError("UO source MAT metadata changed after dataset loading: {0}".format(path))
        inventory.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "modification_time_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    content_payload = {
        "files": inventory,
        "settings": signature.get("settings", {}),
    }
    content_signature = sha256_json(content_payload)
    metadata["content_signature"] = content_signature
    bound = DatasetBundle(name=bundle.name, classes=bundle.classes, metadata=metadata)
    return bound, {
        "source_file_count": len(inventory),
        "content_signature": content_signature,
        "settings": signature.get("settings", {}),
        "files": inventory,
    }


def load_confirmed_hparams(path: Path) -> Dict[str, object]:
    target = Path(path)
    if not target.is_file():
        raise RuntimeError(
            "Confirmed UO Optuna configuration is missing: {0}. Run audit_uo_main_reference.py first.".format(target)
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("status") != "CONFIRMED" or payload.get("confirmation_status") != "confirmed_from_original_uo_artifact":
        raise RuntimeError(
            "Full paper-aligned runs require an explicit original UO Optuna artifact; rounded manuscript values are insufficient."
        )
    required = (
        "dropout_vib",
        "dropout_aco",
        "atten_dim",
        "n_layers_vib",
        "n_layers_aco",
        "lr",
        "batch_size",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError("Confirmed UO configuration is missing: {0}.".format(", ".join(missing)))
    for key in ("atten_dim", "n_layers_vib", "n_layers_aco", "batch_size"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise RuntimeError("Confirmed UO field {0} must be an exact JSON integer.".format(key))
    for key in ("dropout_vib", "dropout_aco", "lr"):
        if (
            isinstance(payload[key], bool)
            or not isinstance(payload[key], (int, float))
            or not np.isfinite(float(payload[key]))
        ):
            raise RuntimeError("Confirmed UO field {0} must be a finite JSON number.".format(key))
    result: Dict[str, object] = {
        "dropout_vib": float(payload["dropout_vib"]),
        "dropout_aco": float(payload["dropout_aco"]),
        "atten_dim": int(payload["atten_dim"]),
        "n_layers_vib": int(payload["n_layers_vib"]),
        "n_layers_aco": int(payload["n_layers_aco"]),
        "lr": float(payload["lr"]),
        "batch_size": int(payload["batch_size"]),
        "source_path": str(target.resolve()),
        "source_sha256": sha256_file(target),
    }
    expected_params = {key: result[key] for key in required}
    if payload.get("parameter_candidate_count") != 1:
        raise RuntimeError("Confirmed UO configuration must contain exactly one historical winner.")
    candidates = payload.get("parameter_candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], Mapping)
    ):
        raise RuntimeError("Confirmed UO configuration has an invalid winning-parameter audit trail.")
    for key in ("atten_dim", "n_layers_vib", "n_layers_aco", "batch_size"):
        if isinstance(candidates[0].get(key), bool) or not isinstance(candidates[0].get(key), int):
            raise RuntimeError("Audited UO winner contains a non-integer field: {0}.".format(key))
    for key in ("dropout_vib", "dropout_aco", "lr"):
        value = candidates[0].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError("Audited UO winner contains a non-finite numeric field: {0}.".format(key))
    normalized_candidate = {
        "dropout_vib": float(candidates[0].get("dropout_vib")),
        "dropout_aco": float(candidates[0].get("dropout_aco")),
        "atten_dim": int(candidates[0].get("atten_dim")),
        "n_layers_vib": int(candidates[0].get("n_layers_vib")),
        "n_layers_aco": int(candidates[0].get("n_layers_aco")),
        "lr": float(candidates[0].get("lr")),
        "batch_size": int(candidates[0].get("batch_size")),
    }
    if normalized_candidate != expected_params:
        raise RuntimeError("Top-level UO parameters do not match the uniquely audited historical winner.")
    if (
        int(payload.get("best_trial_number", -1)) != 22
        or not np.isfinite(float(payload.get("best_value", float("nan"))))
        or float(payload.get("best_value", float("nan"))) != 0.994434118270874
        or int(payload.get("number_of_trials", -1)) != 30
    ):
        raise RuntimeError("Confirmed UO study metadata does not match the retained original run.")
    if not (0.0 < float(result["dropout_vib"]) < 1.0 and 0.0 < float(result["dropout_aco"]) < 1.0):
        raise RuntimeError("Recovered UO dropout values must lie strictly between zero and one.")
    if not np.isfinite(float(result["lr"])) or float(result["lr"]) <= 0.0 or int(result["batch_size"]) <= 0:
        raise RuntimeError("Recovered UO learning rate and batch size must be positive.")
    if result["atten_dim"] != 256 or result["n_layers_vib"] != 4 or result["n_layers_aco"] != 5:
        raise RuntimeError("Recovered UO architecture does not match the current manuscript Table 3.")
    evidence_path = Path(str(payload.get("evidence_path", "")))
    if not evidence_path.is_absolute():
        evidence_path = config.SUITE_ROOT / evidence_path
    evidence_sha256 = str(payload.get("evidence_sha256", "")).lower()
    if not evidence_path.is_file() or sha256_file(evidence_path).lower() != evidence_sha256:
        raise RuntimeError("Confirmed UO Optuna evidence is missing or its SHA-256 has changed.")
    summary_evidence = payload.get("summary_evidence", {})
    if not isinstance(summary_evidence, Mapping) or summary_evidence.get("anchors_passed") is not True:
        raise RuntimeError("Confirmed UO configuration does not contain a successful Table-5 summary audit.")
    summary_path = Path(str(summary_evidence.get("path", "")))
    if not summary_path.is_absolute():
        summary_path = config.SUITE_ROOT / summary_path
    summary_sha256 = str(summary_evidence.get("sha256", "")).lower()
    if not summary_path.is_file() or sha256_file(summary_path).lower() != summary_sha256:
        raise RuntimeError("Original UO summary evidence is missing or its SHA-256 has changed.")
    result["source_evidence_path"] = str(evidence_path.resolve())
    result["source_evidence_sha256"] = evidence_sha256
    result["source_summary_path"] = str(summary_path.resolve())
    result["source_summary_sha256"] = summary_sha256
    source_program = payload.get("source_program_evidence", {})
    if not isinstance(source_program, Mapping):
        raise RuntimeError("Confirmed UO configuration has no original source-program evidence.")
    source_program_path = Path(str(source_program.get("path", "")))
    if not source_program_path.is_absolute():
        source_program_path = config.SUITE_ROOT / source_program_path
    source_program_sha256 = str(source_program.get("sha256", "")).lower()
    checks = source_program.get("protocol_checks", {})
    if (
        not source_program_path.is_file()
        or sha256_file(source_program_path).lower() != source_program_sha256
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise RuntimeError("Original UO source-program evidence is missing, changed, or incomplete.")
    result["source_program_path"] = str(source_program_path.resolve())
    result["source_program_sha256"] = source_program_sha256
    return result


def _stack_split_global_rng(
    x1_parts: Sequence[np.ndarray],
    x2_parts: Sequence[np.ndarray],
    y_parts: Sequence[np.ndarray],
    id_parts: Sequence[np.ndarray],
    num_classes: int,
) -> SplitData:
    """Stack and shuffle with NumPy's global RNG, as in MHCNN_UO.py."""
    x1 = np.vstack(x1_parts).astype(np.float32)
    x2 = np.vstack(x2_parts).astype(np.float32)
    y_int = np.concatenate(y_parts).astype(np.int64)
    sample_ids = np.concatenate(id_parts).astype(str)
    order = np.arange(len(y_int), dtype=np.int64)
    np.random.shuffle(order)
    y_int = y_int[order]
    return SplitData(
        x1=x1[order][..., None],
        x2=x2[order][..., None],
        y_onehot=np.eye(num_classes, dtype=np.float32)[y_int],
        y_int=y_int,
        sample_ids=sample_ids[order],
    )


def prepare_source_exact_split(bundle: DatasetBundle, n_train: int, seed: int) -> Tuple[Dict[str, SplitData], Dict[str, object]]:
    """Reproduce MHCNN_UO.py get_split_data exactly, including RNG order.

    The original executable resets NumPy to the same seed inside every class,
    takes the first N rows after the per-class shuffle, uses all remaining rows
    as the held-out set, and then shuffles the stacked train and held-out arrays.
    """
    n_value = int(n_train)
    seed_value = int(seed)
    train_x1: List[np.ndarray] = []
    train_x2: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    train_ids: List[np.ndarray] = []
    test_x1: List[np.ndarray] = []
    test_x2: List[np.ndarray] = []
    test_y: List[np.ndarray] = []
    test_ids: List[np.ndarray] = []
    class_indices: Dict[str, Dict[str, List[int]]] = {}

    for class_data in bundle.classes:
        available = len(class_data.x1)
        if available != 400 or len(class_data.x2) != 400 or len(class_data.sample_ids) != 400:
            raise RuntimeError(
                "The source-exact UO protocol requires exactly 400 paired segments per class; "
                "class {0} has {1}.".format(class_data.label_id, available)
            )
        if n_value < 1 or n_value >= available:
            raise ValueError("N must lie in [1, available-1] for the paper UO protocol.")
        # The uploaded source resets the global NumPy RNG to the same seed
        # inside every class before shuffling the paired rows.
        np.random.seed(seed_value)
        order = np.arange(available, dtype=np.int64)
        np.random.shuffle(order)
        train_idx = order[:n_value]
        heldout_idx = order[n_value:]
        train_x1.append(class_data.x1[train_idx])
        train_x2.append(class_data.x2[train_idx])
        train_y.append(np.full(len(train_idx), class_data.label_id, dtype=np.int64))
        train_ids.append(class_data.sample_ids[train_idx])
        test_x1.append(class_data.x1[heldout_idx])
        test_x2.append(class_data.x2[heldout_idx])
        test_y.append(np.full(len(heldout_idx), class_data.label_id, dtype=np.int64))
        test_ids.append(class_data.sample_ids[heldout_idx])
        class_indices[str(class_data.label_id)] = {
            "train": train_idx.tolist(),
            "heldout": heldout_idx.tolist(),
        }

    if not bundle.classes:
        raise RuntimeError("UO dataset has no classes.")
    # After the final per-class shuffle, the uploaded source continues with
    # the same global RNG state to shuffle the stacked train and held-out rows.
    train = _stack_split_global_rng(train_x1, train_x2, train_y, train_ids, bundle.num_classes)
    heldout = _stack_split_global_rng(test_x1, test_x2, test_y, test_ids, bundle.num_classes)
    overlap = set(train.sample_ids.tolist()).intersection(heldout.sample_ids.tolist())
    if overlap:
        raise RuntimeError("Paper-aligned UO split contains overlapping paired segments.")
    data_signature = str(
        dict(bundle.metadata).get("content_signature")
        or dict(bundle.metadata).get("signature", {}).get("signature", "")
    )
    split_payload = {
        "protocol": "uploaded_uo_first_N_train_all_remainder_heldout",
        "seed": seed_value,
        "n_train_per_class": n_value,
        "class_indices": class_indices,
        "data_signature": data_signature,
    }
    return {
        "train": train,
        "heldout": heldout,
    }, {
        "protocol": split_payload["protocol"],
        "split_signature": sha256_json(split_payload),
        "data_signature": data_signature,
        "train_size": len(train),
        "heldout_size": len(heldout),
    }


def build_source_exact_model(tf: Any, hparams: Mapping[str, object], use_caim: bool) -> Any:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Activation,
        AveragePooling1D,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        Input,
        Lambda,
        LeakyReLU,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model

    class SourceCrossAttention(tf.keras.layers.Layer):
        def __init__(self, output_dim: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.output_dim = int(output_dim)

        def build(self, input_shape: Any) -> None:
            self.W_q = self.add_weight(
                name="W_q",
                shape=(input_shape[0][-1], self.output_dim),
                initializer="uniform",
                trainable=True,
            )
            self.W_k = self.add_weight(
                name="W_k",
                shape=(input_shape[1][-1], self.output_dim),
                initializer="uniform",
                trainable=True,
            )
            self.W_v = self.add_weight(
                name="W_v",
                shape=(input_shape[1][-1], self.output_dim),
                initializer="uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Any, **kwargs: Any) -> Any:
            query, key = inputs
            query_proj = K.dot(query, self.W_q)
            key_proj = K.dot(key, self.W_k)
            value_proj = K.dot(key, self.W_v)
            scores = K.batch_dot(query_proj, key_proj, axes=[2, 2])
            scores = K.softmax(scores, axis=-1)
            return K.batch_dot(scores, value_proj)

    def weighted_features(inputs: Any) -> Any:
        features, weights = inputs
        return features * weights

    dropout_vib = float(hparams["dropout_vib"])
    dropout_aco = float(hparams["dropout_aco"])
    atten_dim = int(hparams["atten_dim"])
    n_layers_vib = int(hparams["n_layers_vib"])
    n_layers_aco = int(hparams["n_layers_aco"])

    input_v = Input(shape=(config.SEGMENT_LENGTH, 1), name="input_vibration")
    x = input_v
    filters = [32, 64, 128, 256, 256]
    for index in range(n_layers_vib):
        output_filters = filters[index] if index < len(filters) else 256
        stride = 2 if index < 2 else 1
        x = Conv1D(output_filters, 16, strides=stride, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        if index == n_layers_vib - 1:
            x = Dropout(dropout_vib)(x)
        x = AveragePooling1D(2, strides=2, padding="same")(x)
    features_v = x

    input_a = Input(shape=(config.SEGMENT_LENGTH, 1), name="input_acoustic")
    y = input_a
    for index in range(n_layers_aco):
        output_filters = filters[index] if index < len(filters) else 256
        stride = 2 if index < 2 else 1
        y = Conv1D(output_filters, 8, strides=stride, padding="same")(y)
        y = Activation(tf.nn.gelu)(y)
        if index == n_layers_aco - 1:
            y = Dropout(dropout_aco)(y)
        y = MaxPooling1D(2, strides=2, padding="same")(y)
    features_a = y

    cross_att_va = SourceCrossAttention(output_dim=atten_dim, name="cross_att_v_from_a")
    full_v = cross_att_va([features_v, features_a])
    cross_att_av = SourceCrossAttention(output_dim=atten_dim, name="cross_att_a_from_v")
    full_a = cross_att_av([features_a, features_v])

    if use_caim:
        branch_a, branch_v = full_a, full_v
    else:
        # The two CAIM layers are still instantiated before the bypass so the
        # downstream random-initialization sequence remains matched to Full.
        branch_a, branch_v = features_a, features_v

    flat_a = Flatten(name="flat_a")(branch_a)
    flat_v = Flatten(name="flat_v")(branch_v)
    w_logits = Concatenate(name="w_concat")(
        [
            Dense(1, activation="sigmoid", name="w_sig_a")(flat_a),
            Dense(1, activation="sigmoid", name="w_sig_v")(flat_v),
        ]
    )
    w_final = Dense(2, activation="softmax", name="w_softmax")(w_logits)
    fused = Concatenate(name="fused_concat")(
        [
            Lambda(weighted_features, name="scale_a")([flat_a, w_final[:, 0:1]]),
            Lambda(weighted_features, name="scale_v")([flat_v, w_final[:, 1:2]]),
        ]
    )
    fused = Dense(256, name="fcb_dense")(fused)
    fused = LeakyReLU(alpha=0.2, name="fcb_lrelu")(fused)
    output = Dense(7, activation="softmax", name="cls_head")(fused)
    return Model(inputs=[input_v, input_a], outputs=output, name="MHCNN_UO" if use_caim else "MHCNN_UO_no_CAIM")


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.argmax(np.asarray(probabilities), axis=1)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }


def train_one(
    tf: Any,
    bundle: DatasetBundle,
    hparams: Mapping[str, object],
    n_train: int,
    run_idx: int,
    variant: str,
    output_dir: Path,
    force: bool,
) -> Dict[str, object]:
    seed = paper_seed(n_train, run_idx)
    run_dir = output_dir / "models" / variant / ("N{0}_run{1:02d}_seed{2}".format(n_train, run_idx, seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    row_path = run_dir / "metrics.json"
    weights_path = run_dir / "model.weights.h5"
    history_path = run_dir / "history.csv"
    hparam_signature = sha256_json({key: hparams[key] for key in hparams if not key.startswith("source_")})
    provenance_signature = sha256_json(
        {
            "script_sha256": sha256_file(Path(__file__)),
            "optuna_evidence_sha256": hparams["source_evidence_sha256"],
            "summary_evidence_sha256": hparams["source_summary_sha256"],
        }
    )

    # Reproduce the historical NumPy split order first. The historical process
    # did not retain enough state to recover its TensorFlow initializer stream.
    splits, split_meta = prepare_source_exact_split(bundle, n_train, seed)
    cache_signature = sha256_json(
        {
            "variant": variant,
            "n_train": int(n_train),
            "run_idx": int(run_idx),
            "seed": int(seed),
            "split_signature": split_meta["split_signature"],
            "data_signature": split_meta["data_signature"],
            "hparam_signature": hparam_signature,
            "provenance_signature": provenance_signature,
            "protocol": TRAINING_PROTOCOL_ID,
        }
    )
    if row_path.is_file() and not force:
        cached = json.loads(row_path.read_text(encoding="utf-8"))
        if cached.get("run_signature") != cache_signature:
            raise RuntimeError("Cached paper-aligned low-shot run has an incompatible signature: {0}".format(run_dir))
        if (
            not weights_path.is_file()
            or not history_path.is_file()
            or str(cached.get("weights_sha256", "")).lower() != sha256_file(weights_path).lower()
            or str(cached.get("history_sha256", "")).lower() != sha256_file(history_path).lower()
        ):
            raise RuntimeError("Cached paper-aligned run artifacts are missing or have changed: {0}".format(run_dir))
        return cached

    tf.keras.backend.clear_session()
    gc.collect()
    # Controlled reviewer-extension addition: reset Python/TensorFlow model
    # RNGs after the source-aligned split and before model construction. NumPy
    # is deliberately not reset here so the stacked train/held-out ordering is
    # exactly the ordering produced by the historical split routine. Keras 2.7
    # uses TensorFlow RNG for array batch shuffling, so both model initialization
    # and epoch shuffling belong to the new paired TF seed protocol; neither is
    # represented as recovery of the unavailable historical TF stream.
    set_paired_model_seed(tf, seed)
    model = build_source_exact_model(tf, hparams, use_caim=(variant == "full"))
    if variant == "full" and int(model.count_params()) != EXPECTED_FULL_PARAMETER_COUNT:
        raise RuntimeError("Full UO model parameter count differs from Table 3: {0}".format(model.count_params()))
    optimizer = tf.keras.optimizers.Adamax(learning_rate=float(hparams["lr"]))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    start = time.perf_counter()
    history = model.fit(
        [splits["train"].x1, splits["train"].x2],
        splits["train"].y_onehot,
        epochs=80,
        batch_size=int(hparams["batch_size"]),
        validation_data=([splits["heldout"].x1, splits["heldout"].x2], splits["heldout"].y_onehot),
        shuffle=True,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    probabilities = np.asarray(
        model.predict([splits["heldout"].x1, splits["heldout"].x2], verbose=0)
    )
    test_metrics = metric_dict(splits["heldout"].y_int, probabilities)
    train_prob = np.asarray(model.predict([splits["train"].x1, splits["train"].x2], verbose=0))
    train_metrics = metric_dict(splits["train"].y_int, train_prob)
    heldout_accuracy = float(history.history["val_accuracy"][-1])
    row: Dict[str, object] = {
        "variant": variant,
        "n_train": int(n_train),
        "run_idx": int(run_idx),
        "seed": int(seed),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_accuracy": train_metrics["accuracy"],
        "heldout_accuracy": heldout_accuracy,
        "generalization_gap": float(train_metrics["accuracy"] - heldout_accuracy),
        "split_signature": split_meta["split_signature"],
        "data_signature": split_meta["data_signature"],
        "hyperparameter_signature": hparam_signature,
        "source_type": "new_paper_aligned_training",
        "train_time_s": float(elapsed),
        "run_signature": cache_signature,
        "weights_path": str(weights_path.resolve()),
        "history_path": str(history_path.resolve()),
        "metrics_path": str(row_path.resolve()),
        "model_parameter_count": int(model.count_params()),
        "provenance_signature": provenance_signature,
    }
    model.save_weights(str(weights_path))
    pd.DataFrame(history.history).assign(epoch=np.arange(1, 81)).to_csv(history_path, index=False)
    row["weights_sha256"] = sha256_file(weights_path)
    row["history_sha256"] = sha256_file(history_path)
    row_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _metric_statistics(values: Sequence[float], use_trim: bool) -> Dict[str, float]:
    untrimmed_mean, untrimmed_sd = mean_sd(values)
    mean_value, sd_value = trimmed_mean_sd(values) if use_trim else (untrimmed_mean, untrimmed_sd)
    return {
        "mean": float(mean_value),
        "sd": float(sd_value),
        "untrimmed_mean": float(untrimmed_mean),
        "untrimmed_sd": float(untrimmed_sd),
    }


def summarize(raw: pd.DataFrame, runs_per_n: int, use_trim: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (variant, n_train), block in raw.groupby(["variant", "n_train"], sort=True):
        if block["seed"].nunique() != int(runs_per_n):
            raise RuntimeError("Every low-shot variant/N group must contain the expected run count.")
        row: Dict[str, object] = {
            "variant": str(variant),
            "n_train": int(n_train),
            "seeds": int(runs_per_n),
            "seeds_total": int(runs_per_n),
            "retained_after_trim": int(runs_per_n) - 2 if use_trim else int(runs_per_n),
            "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
        }
        for metric in SUMMARY_METRICS:
            stats = _metric_statistics(block[metric].to_numpy(dtype=np.float64), use_trim)
            for key, value in stats.items():
                row[metric + "_" + key] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "n_train"]).reset_index(drop=True)


def paired_caim(raw: pd.DataFrame, runs_per_n: int, use_trim: bool) -> pd.DataFrame:
    full = raw[raw["variant"] == "full"][["n_train", "seed", "test_accuracy", "split_signature"]].rename(
        columns={"test_accuracy": "full_accuracy", "split_signature": "full_split_signature"}
    )
    variant = raw[raw["variant"] == "no_caim"][["n_train", "seed", "test_accuracy", "split_signature"]].rename(
        columns={"test_accuracy": "no_caim_accuracy", "split_signature": "no_caim_split_signature"}
    )
    paired = full.merge(variant, on=["n_train", "seed"], validate="one_to_one")
    if not (paired["full_split_signature"] == paired["no_caim_split_signature"]).all():
        raise RuntimeError("Paired CAIM rows do not share the same exact split.")
    paired["caim_gain"] = paired["full_accuracy"] - paired["no_caim_accuracy"]
    rows: List[Dict[str, object]] = []
    for n_train, block in paired.groupby("n_train", sort=True):
        if len(block) != int(runs_per_n):
            raise RuntimeError("Paired CAIM summary has an incomplete run set.")
        stats = _metric_statistics(block["caim_gain"].to_numpy(dtype=np.float64), use_trim)
        rows.append(
            {
                "n_train": int(n_train),
                "paired_gain_mean": stats["mean"],
                "paired_gain_sd": stats["sd"],
                "paired_gain_untrimmed_mean": stats["untrimmed_mean"],
                "paired_gain_untrimmed_sd": stats["untrimmed_sd"],
                "caim_gain_mean": stats["mean"],
                "caim_gain_sd": stats["sd"],
                "caim_gain_untrimmed_mean": stats["untrimmed_mean"],
                "caim_gain_untrimmed_sd": stats["untrimmed_sd"],
                "seeds": int(runs_per_n),
                "seeds_total": int(runs_per_n),
                "retained_after_trim": int(runs_per_n) - 2 if use_trim else int(runs_per_n),
                "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
            }
        )
    return pd.DataFrame(rows)


def anchor_gate(summary: pd.DataFrame, require: bool) -> Dict[str, object]:
    checks: Dict[str, object] = {}
    all_passed = True
    full = summary[summary["variant"] == "full"].set_index("n_train")
    for n_value, expected in PAPER_ANCHORS.items():
        if n_value not in full.index:
            checks[str(n_value)] = {"passed": False, "reason": "anchor N not evaluated"}
            all_passed = False
            continue
        row = full.loc[n_value]
        actual = {key: float(row[key]) for key in expected}
        per_field = {
            key: {
                "expected": float(expected[key]),
                "actual": actual[key],
                "expected_4dp": "{0:.4f}".format(float(expected[key])),
                "actual_4dp": "{0:.4f}".format(actual[key]),
                "passed": "{0:.4f}".format(actual[key]) == "{0:.4f}".format(float(expected[key])),
            }
            for key in expected
        }
        passed = all(item["passed"] for item in per_field.values())
        all_passed = all_passed and passed
        checks[str(n_value)] = {"passed": passed, "fields": per_field}
    return {
        "status": ("PASS" if all_passed else "FAIL") if require else "NOT_REQUIRED_FAST",
        "required_for_manuscript": bool(require),
        "comparison_precision": "exact after four-decimal manuscript rounding",
        "anchors": checks,
        "observed_anchor_match": bool(all_passed),
        "final_assets_authorized": bool(require and all_passed),
    }


def build_operational_thresholds(summary: pd.DataFrame, use_trim: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in VARIANT_NAMES:
        block = summary[summary["variant"] == variant]
        stable = block[
            (block["test_accuracy_mean"] >= 0.80) & (block["test_accuracy_sd"] <= 0.10)
        ].sort_values("n_train")
        rows.append(
            {
                "variant": variant,
                "criterion": (
                    "trimmed test accuracy >= 0.80 and trimmed seed SD <= 0.10"
                    if use_trim
                    else "untrimmed fast-smoke accuracy >= 0.80 and untrimmed seed SD <= 0.10"
                ),
                "first_empirical_n": None if stable.empty else int(stable.iloc[0]["n_train"]),
                "claim_limit": (
                    "Smallest evaluated N satisfying the predefined stability criterion; "
                    "a protocol-specific operational stability point, not a universal theoretical threshold."
                ),
            }
        )
    return rows


def plot_outputs(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    full = summary[summary["variant"] == "full"].sort_values("n_train")
    no_caim = summary[summary["variant"] == "no_caim"].sort_values("n_train")
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 63 / 25.4))

    for frame, label, marker in ((full, "Full MHFL-MCA", "o"), (no_caim, "Without CAIM", "s")):
        axes[0].errorbar(
            frame["n_train"], frame["test_accuracy_mean"], yerr=frame["test_accuracy_sd"],
            marker=marker, linewidth=1.4, markersize=4.0, capsize=2.5, label=label,
        )
        axes[1].errorbar(
            frame["n_train"], frame["generalization_gap_mean"], yerr=frame["generalization_gap_sd"],
            marker=marker, linewidth=1.4, markersize=4.0, capsize=2.5, label=label,
        )
    axes[0].set_title("a   Extreme low-shot performance", loc="left", fontweight="bold")
    axes[0].set_xlabel("Training samples per class")
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_ylim(0.0, 1.02)

    axes[1].set_title("b   Overfitting indicator", loc="left", fontweight="bold")
    axes[1].set_xlabel("Training samples per class")
    axes[1].set_ylabel("Train–held-out gap")
    axes[1].axhline(0.0, linewidth=0.8)

    axes[2].errorbar(
        paired["n_train"], paired["paired_gain_mean"], yerr=paired["paired_gain_sd"],
        marker="o", linewidth=1.4, markersize=4.0, capsize=2.5,
    )
    axes[2].set_title("c   CAIM sensitivity", loc="left", fontweight="bold")
    axes[2].set_xlabel("Training samples per class")
    axes[2].set_ylabel("Paired CAIM accuracy gain")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))

    bundle = output_dir / "lowshot_evidence_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": bundle / "lowshot_evidence.pdf",
        "png": bundle / "lowshot_evidence.png",
        "svg": bundle / "lowshot_evidence.svg",
        "tiff": bundle / "lowshot_evidence.tiff",
    }
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["tiff"], dpi=600)
    plt.close(fig)
    return {key: str(value.resolve()) for key, value in paths.items()}


def validate_raw(raw: pd.DataFrame, n_grid: Sequence[int], runs_per_n: int) -> Dict[str, object]:
    if tuple(raw.columns) != RAW_COLUMNS:
        raise RuntimeError("Paper-aligned low-shot raw columns do not match the contract.")
    if raw.duplicated(["variant", "n_train", "seed"]).any():
        raise RuntimeError("Duplicate low-shot variant/N/seed rows detected.")
    numeric_columns = [
        "n_train", "run_idx", "seed", "test_accuracy", "test_macro_precision",
        "test_macro_recall", "test_macro_f1", "train_accuracy", "heldout_accuracy",
        "generalization_gap", "train_time_s",
    ]
    numeric = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Low-shot raw output contains NaN or Inf.")
    expected_rows = len(VARIANT_NAMES) * len(tuple(n_grid)) * int(runs_per_n)
    if len(raw) != expected_rows:
        raise RuntimeError("Expected {0} low-shot rows, found {1}.".format(expected_rows, len(raw)))
    bounded_metrics = [
        "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
        "train_accuracy", "heldout_accuracy",
    ]
    if not all(((numeric[column] >= 0.0) & (numeric[column] <= 1.0)).all() for column in bounded_metrics):
        raise RuntimeError("Low-shot classification metrics must lie in [0, 1].")
    if not ((numeric["generalization_gap"] >= -1.0) & (numeric["generalization_gap"] <= 1.0)).all():
        raise RuntimeError("Low-shot train-heldout gaps must lie in [-1, 1].")
    if not (numeric["train_time_s"] >= 0.0).all():
        raise RuntimeError("Low-shot training times cannot be negative.")
    seeds = expected_seed_map(n_grid, runs_per_n)
    for n_value in n_grid:
        for variant in VARIANT_NAMES:
            observed = set(
                int(value) for value in raw[(raw["n_train"] == n_value) & (raw["variant"] == variant)]["seed"]
            )
            if observed != set(seeds[int(n_value)]):
                raise RuntimeError("Paper seed schedule is incomplete for {0}/N={1}.".format(variant, n_value))
    matched = raw.groupby(["n_train", "seed"])[
        ["split_signature", "data_signature", "hyperparameter_signature"]
    ].nunique()
    pair_sizes = raw.groupby(["n_train", "seed"])["variant"].nunique()
    if not (matched == 1).all().all() or not (pair_sizes == len(VARIANT_NAMES)).all():
        raise RuntimeError("Full and no-CAIM do not share identical data, hyperparameters, and exact-paper splits.")
    if set(raw["source_type"].astype(str)) != {"new_paper_aligned_training"}:
        raise RuntimeError("Low-shot raw rows contain an unexpected source type.")
    return {
        "status": "PASS",
        "rows": int(len(raw)),
        "n_grid": [int(value) for value in n_grid],
        "runs_per_n": int(runs_per_n),
        "seed_map": seeds,
        "duplicates": 0,
        "nan_or_inf": 0,
        "matched_split_pairs": int(len(matched)),
    }


def _frames_match(expected: pd.DataFrame, actual: pd.DataFrame, sort_columns: Sequence[str]) -> bool:
    try:
        left = expected.sort_values(list(sort_columns)).reset_index(drop=True)
        right = actual.sort_values(list(sort_columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        return True
    except (AssertionError, KeyError, ValueError):
        return False


def build_post_gate(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    thresholds: Sequence[Mapping[str, object]],
    anchors: Mapping[str, object],
    mode: str,
    n_grid: Sequence[int],
    runs_per_n: int,
    use_trim: bool,
    raw_gate: Mapping[str, object],
    figure_bundle_complete: bool,
) -> Dict[str, object]:
    expected_summary = summarize(raw, runs_per_n, use_trim)
    expected_paired = paired_caim(raw, runs_per_n, use_trim)
    expected_thresholds = build_operational_thresholds(expected_summary, use_trim)
    summary_ok = _frames_match(expected_summary, summary, ("variant", "n_train"))
    paired_ok = _frames_match(expected_paired, paired, ("n_train",))
    thresholds_ok = list(thresholds) == expected_thresholds
    anchor_ok = anchors.get("status") == "PASS" and anchors.get("final_assets_authorized") is True
    full_mode = str(mode) == "full"
    internal_ok = summary_ok and paired_ok and thresholds_ok and raw_gate.get("status") == "PASS"
    final_authorized = bool(full_mode and internal_ok and anchor_ok and figure_bundle_complete)
    return {
        "status": "PASS" if internal_ok and (not full_mode or final_authorized) else "FAIL",
        "mode": str(mode),
        "final_outputs_authorized": final_authorized,
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "paired_gain_rows": int(len(paired)),
        "variants": list(VARIANT_NAMES),
        "n_grid": [int(value) for value in n_grid],
        "runs_per_group": int(runs_per_n),
        "seed_map": expected_seed_map(n_grid, runs_per_n),
        "retained_after_trim": int(runs_per_n) - 2 if use_trim else int(runs_per_n),
        "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
        "duplicates": int(raw.duplicated(["variant", "n_train", "seed"]).sum()),
        "nan_or_inf": 0,
        "failed_runs": 0,
        "failed_seeds": [],
        "matched_split_pairs": int(raw_gate.get("matched_split_pairs", 0)),
        "summary_recomputed_from_raw": summary_ok,
        "paired_gain_aligned_by_seed_before_trim": paired_ok,
        "operational_threshold_recomputed_from_trimmed_statistics": thresholds_ok,
        "anchor_gate_pass": bool(anchor_ok),
        "figure_bundle_complete": bool(figure_bundle_complete),
    }


def output_record(path: Path, rows: Optional[int] = None) -> Dict[str, object]:
    target = Path(path)
    record: Dict[str, object] = {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "size_bytes": int(target.stat().st_size),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def run_artifact_record(row: Mapping[str, object], output_dir: Path) -> Dict[str, object]:
    paths = {
        "metrics": Path(str(row.get("metrics_path", ""))),
        "weights": Path(str(row.get("weights_path", ""))),
        "history": Path(str(row.get("history_path", ""))),
    }
    root = Path(output_dir).resolve()
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("Run artifact is missing or empty ({0}): {1}".format(label, path))
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Run artifact is outside the Experiment-05 output directory: {0}".format(path)) from exc
    if sha256_file(paths["weights"]).lower() != str(row.get("weights_sha256", "")).lower():
        raise RuntimeError("Run weights SHA-256 changed before manifest construction: {0}".format(paths["weights"]))
    if sha256_file(paths["history"]).lower() != str(row.get("history_sha256", "")).lower():
        raise RuntimeError("Run history SHA-256 changed before manifest construction: {0}".format(paths["history"]))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    identity = (str(row["variant"]), int(row["n_train"]), int(row["run_idx"]), int(row["seed"]))
    metrics_identity = (
        str(metrics.get("variant")),
        int(metrics.get("n_train", -1)),
        int(metrics.get("run_idx", -1)),
        int(metrics.get("seed", -1)),
    )
    if metrics_identity != identity or metrics.get("run_signature") != row.get("run_signature"):
        raise RuntimeError("Run metrics identity/signature changed before manifest construction.")
    return {
        "variant": identity[0],
        "n_train": identity[1],
        "run_idx": identity[2],
        "seed": identity[3],
        "run_signature": str(row.get("run_signature", "")),
        "provenance_signature": str(row.get("provenance_signature", "")),
        "model_parameter_count": int(row.get("model_parameter_count", -1)),
        "metrics": output_record(paths["metrics"]),
        "weights": output_record(paths["weights"]),
        "history": output_record(paths["history"], 80),
    }


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag.strip() or config.RUN_TAG
    if args.mode == "full" and run_tag.lower() in {"", "manual", "fast", "smoke", "debug"}:
        raise RuntimeError("Full Experiment 05 requires an explicit non-smoke run tag.")
    tf = configure_tensorflow()
    runtime_environment = {
        "python": sys.version,
        "tensorflow": str(tf.__version__),
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
    }
    hparams = load_confirmed_hparams(args.uo_config)
    n_grid = N_GRID_FAST if args.mode == "fast" else N_GRID_FULL
    runs_per_n = 2 if args.mode == "fast" else RUNS_PER_N_FULL
    use_trim = args.mode == "full"
    require_anchor = args.mode == "full"

    output_root = config.SUITE_ROOT / "outputs" / run_tag
    out_dir = output_root / "05_lowshot_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    execution_state_path = out_dir / "lowshot_execution_state.json"
    write_json(
        execution_state_path,
        {
            "status": "RUNNING",
            "mode": args.mode,
            "run_tag": run_tag,
            "protocol": TRAINING_PROTOCOL_ID,
            "final_outputs_authorized": False,
            "started_unix_time": time.time(),
        },
    )
    bundle = load_uo_dataset(rebuild_cache=args.rebuild_cache)
    bundle, dataset_provenance = bind_dataset_content_hashes(bundle)

    rows: List[Dict[str, object]] = []
    run_artifacts: List[Dict[str, object]] = []
    for n_train in n_grid:
        for run_idx in range(1, runs_per_n + 1):
            for variant in VARIANT_NAMES:
                row = train_one(
                    tf=tf,
                    bundle=bundle,
                    hparams=hparams,
                    n_train=int(n_train),
                    run_idx=int(run_idx),
                    variant=variant,
                    output_dir=out_dir,
                    force=args.force,
                )
                rows.append({key: row[key] for key in RAW_COLUMNS})
                run_artifacts.append(run_artifact_record(row, out_dir))
                print(
                    "[Paper-aligned low-shot] N={0} run={1} seed={2} {3}: acc={4:.4f}, gap={5:.4f}".format(
                        n_train, run_idx, row["seed"], variant, row["test_accuracy"], row["generalization_gap"]
                    )
                )

    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_gate = validate_raw(raw, n_grid, runs_per_n)
    summary = summarize(raw, runs_per_n, use_trim)
    paired = paired_caim(raw, runs_per_n, use_trim)
    anchors = anchor_gate(summary, require=require_anchor)
    thresholds = build_operational_thresholds(summary, use_trim)

    # Always retain audit outputs. Manuscript assets are blocked when anchors fail.
    raw_path = out_dir / "lowshot_raw.csv"
    summary_path = out_dir / "lowshot_summary.csv"
    paired_path = out_dir / "caim_paired_summary.csv"
    anchor_path = out_dir / "lowshot_anchor_gate.json"
    threshold_path = out_dir / "operational_thresholds.json"
    post_gate_path = out_dir / "lowshot_post_gate.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    write_json(anchor_path, anchors)
    write_json(threshold_path, thresholds)

    figure_paths: Dict[str, str] = {}
    figure_contract_path = out_dir / "lowshot_evidence_bundle" / "figure_contract.json"
    if require_anchor and anchors["final_assets_authorized"]:
        figure_paths = plot_outputs(summary, paired, out_dir)
        figure_contract = {
            "figure_id": "extreme-lowshot-paper-aligned",
            "source_data": [str(summary_path.resolve()), str(paired_path.resolve())],
            "replicate_unit": "paper seed schedule 100*N+run_idx",
            "center_statistic": "trimmed mean after removing one highest and one lowest result",
            "spread_definition": "population SD across the eight retained runs",
            "panel_y_labels": [
                "Held-out accuracy",
                "Train–held-out gap",
                "Paired CAIM accuracy gain",
            ],
            "evaluation_set_role": (
                "All non-training segments form one held-out set used both as Keras validation_data "
                "for the training curve and for final diagnostic evaluation; no independent validation/test split."
            ),
            "anchor_gate": anchors,
            "claim_limits": [
                "Empirical protocol-specific sensitivity point, not a universal theorem",
                "Segment-disjoint held-out evaluation does not establish recording-disjoint generalization",
            ],
        }
        write_json(figure_contract_path, figure_contract)

    expected_figure_files = [
        out_dir / "lowshot_evidence_bundle" / "lowshot_evidence.png",
        out_dir / "lowshot_evidence_bundle" / "lowshot_evidence.pdf",
        out_dir / "lowshot_evidence_bundle" / "lowshot_evidence.svg",
        out_dir / "lowshot_evidence_bundle" / "lowshot_evidence.tiff",
        figure_contract_path,
    ]
    figure_bundle_complete = bool(
        require_anchor
        and anchors["final_assets_authorized"]
        and all(path.is_file() and path.stat().st_size > 0 for path in expected_figure_files)
    )
    post_gate = build_post_gate(
        raw=raw,
        summary=summary,
        paired=paired,
        thresholds=thresholds,
        anchors=anchors,
        mode=args.mode,
        n_grid=n_grid,
        runs_per_n=runs_per_n,
        use_trim=use_trim,
        raw_gate=raw_gate,
        figure_bundle_complete=figure_bundle_complete,
    )
    write_json(post_gate_path, post_gate)
    execution_state = {
        "status": "PASS" if post_gate["status"] == "PASS" else "BLOCKED",
        "mode": args.mode,
        "run_tag": run_tag,
        "protocol": TRAINING_PROTOCOL_ID,
        "completed_run_artifacts": len(run_artifacts),
        "final_outputs_authorized": bool(post_gate["final_outputs_authorized"]),
        "finished_unix_time": time.time(),
    }
    write_json(execution_state_path, execution_state)

    output_records: Dict[str, object] = {
        "raw": output_record(raw_path, len(raw)),
        "summary": output_record(summary_path, len(summary)),
        "paired_summary": output_record(paired_path, len(paired)),
        "operational_thresholds": output_record(threshold_path),
        "anchor_gate": output_record(anchor_path),
        "post_gate": output_record(post_gate_path),
        "hparams_config": output_record(args.uo_config),
        "execution_state": output_record(execution_state_path),
    }
    if figure_bundle_complete:
        for key, path in (
            ("figure_png", expected_figure_files[0]),
            ("figure_pdf", expected_figure_files[1]),
            ("figure_svg", expected_figure_files[2]),
            ("figure_tiff", expected_figure_files[3]),
            ("figure_contract", expected_figure_files[4]),
        ):
            output_records[key] = output_record(path)

    manifest: Dict[str, object] = {
        "status": "PASS" if post_gate["status"] == "PASS" else "BLOCKED",
        "mode": args.mode,
        "protocol": TRAINING_PROTOCOL_ID,
        "protocol_alignment_scope": (
            "Historical UO data partition, model topology, exact Optuna hyperparameters, Adamax, "
            "80 epochs, held-out evaluation, and final-epoch weights are source-aligned."
        ),
        "random_initialization_boundary": (
            "The historical Python/TensorFlow initializer stream was not retained. This reviewer "
            "extension resets Python and TensorFlow to seed=100*N+run_idx before each model "
            "construction so Full and no-CAIM use paired common-layer initialization and a "
            "declared Keras batch-shuffle stream. NumPy is left at the historical post-split "
            "state solely to preserve stacked split ordering. Table-5 "
            "aggregates remain an external hard anchor and are not assumed to be reproducible a priori."
        ),
        "split_protocol": "first N train / all remaining held-out; same held-out used for validation curves and final evaluation",
        "seed_schedule": "seed = 100*N + run_idx",
        "epochs": 80,
        "optimizer": {
            "name": "Adamax",
            "learning_rate": float(hparams["lr"]),
            "batch_size": int(hparams["batch_size"]),
        },
        "gradient_clipping": None,
        "early_stopping": False,
        "final_epoch_weights": True,
        "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
        "runtime_environment": runtime_environment,
        "hparams": hparams,
        "dataset_signature": dict(bundle.metadata).get("signature", {}),
        "dataset_provenance": dataset_provenance,
        "model_parameter_count": EXPECTED_FULL_PARAMETER_COUNT,
        "raw_gate": raw_gate,
        "anchor_gate": anchors,
        "paper_anchors": PAPER_ANCHORS,
        "aggregation": TRIMMED_AGGREGATION if use_trim else FAST_AGGREGATION,
        "paired_by_seed_before_trim": True,
        "failed_runs": 0,
        "failed_seeds": [],
        "run_artifact_count": len(run_artifacts),
        "run_artifacts": run_artifacts,
        "post_gate": post_gate,
        "outputs": output_records,
        "claim_boundary": "Protocol-specific empirical sensitivity; not a universal minimum-sample theorem.",
    }
    if figure_bundle_complete:
        manifest["figure_paths"] = figure_paths
    write_json(out_dir / "lowshot_run_manifest.json", manifest)

    print("\nOutputs saved to:", out_dir.resolve())
    print("Anchor status:", anchors["status"])
    if require_anchor and anchors["status"] != "PASS":
        raise RuntimeError(
            "Full MHFL-MCA did not reproduce Table 5 to four decimals. "
            "No manuscript figure/table was authorized. Inspect lowshot_anchor_gate.json; "
            "do not rerun selectively or alter results to force agreement."
        )
    if post_gate["status"] != "PASS":
        raise RuntimeError("Paper-aligned low-shot internal post-gate failed; inspect lowshot_post_gate.json.")


if __name__ == "__main__":
    main()
