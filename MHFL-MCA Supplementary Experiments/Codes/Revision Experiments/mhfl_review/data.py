from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

from . import config
from .provenance import sha256_json, write_json


@dataclass(frozen=True)
class PairedClassData:
    label_id: int
    label_name: str
    x1: np.ndarray
    x2: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    classes: Tuple[PairedClassData, ...]
    metadata: Mapping[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class SplitData:
    x1: np.ndarray
    x2: np.ndarray
    y_onehot: np.ndarray
    y_int: np.ndarray
    sample_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.y_int.shape[0])


@dataclass(frozen=True)
class CurrentChannelSelection:
    group: str
    channel: str
    length: int
    variance: float
    selection_rule: str
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "channel": self.channel,
            "length": int(self.length),
            "variance": float(self.variance),
            "selection_rule": self.selection_rule,
            "fallback_used": bool(self.fallback_used),
        }


def zscore_per_segment(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim not in {2, 3}:
        raise ValueError("Expected segmented data with shape (n, points) or (n, points, channels).")
    axes = tuple(range(1, values.ndim))
    mean = np.mean(values, axis=axes, keepdims=True)
    std = np.std(values, axis=axes, keepdims=True)
    return ((values - mean) / np.maximum(std, float(eps))).astype(np.float32)


def add_gaussian_noise(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    """Add independent sample-wise Gaussian noise using power-domain SNR."""
    values = np.asarray(x, dtype=np.float32)
    if np.isinf(float(snr_db)):
        return values.copy()
    if values.ndim not in {2, 3}:
        raise ValueError("Expected 2-D or 3-D segmented input, got {0}.".format(values.shape))
    axes = tuple(range(1, values.ndim))
    signal_power = np.mean(np.square(values), axis=axes, keepdims=True) + 1e-12
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    unit_noise = rng.normal(0.0, 1.0, size=values.shape).astype(np.float32)
    return (values + unit_noise * np.sqrt(noise_power).astype(np.float32)).astype(np.float32)


def empirical_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean_f = np.asarray(clean, dtype=np.float64)
    noise_f = np.asarray(noisy, dtype=np.float64) - clean_f
    p_signal = float(np.mean(np.square(clean_f))) + 1e-18
    p_noise = float(np.mean(np.square(noise_f))) + 1e-18
    return float(10.0 * np.log10(p_signal / p_noise))


def segment_nonoverlap(
    signal: np.ndarray,
    length: int,
    max_segments: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    count = len(values) // int(length)
    if max_segments is not None:
        count = min(count, int(max_segments))
    if count <= 0:
        raise ValueError("Signal length {0} is insufficient for {1}-point segmentation.".format(len(values), length))
    starts = np.arange(count, dtype=np.int64) * int(length)
    return values[: count * int(length)].reshape(count, int(length)).astype(np.float32), starts


def _stat_manifest(paths: Sequence[Path], settings: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for path in sorted([Path(item).resolve() for item in paths], key=lambda item: str(item).lower()):
        stat = path.stat()
        rows.append({"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    payload = {"files": rows, "settings": dict(settings)}
    payload["signature"] = sha256_json(payload)
    return payload


def _cache_save(path: Path, bundle: DatasetBundle, signature: Mapping[str, Any]) -> None:
    payload: Dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(dict(bundle.metadata), ensure_ascii=False)),
        "signature_json": np.asarray(json.dumps(dict(signature), ensure_ascii=False)),
        "name": np.asarray(bundle.name),
        "num_classes": np.asarray(bundle.num_classes, dtype=np.int64),
    }
    for class_data in bundle.classes:
        idx = class_data.label_id
        payload["x1_{0}".format(idx)] = np.asarray(class_data.x1, dtype=np.float32)
        payload["x2_{0}".format(idx)] = np.asarray(class_data.x2, dtype=np.float32)
        payload["ids_{0}".format(idx)] = np.asarray(class_data.sample_ids, dtype="U256")
        payload["label_{0}".format(idx)] = np.asarray(class_data.label_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def _cache_load(path: Path, expected_signature: Mapping[str, Any]) -> Optional[DatasetBundle]:
    if not path.is_file():
        return None
    with np.load(str(path), allow_pickle=False) as npz:
        found = json.loads(str(npz["signature_json"].item()))
        if found.get("signature") != expected_signature.get("signature"):
            return None
        num_classes = int(npz["num_classes"].item())
        classes = []
        for idx in range(num_classes):
            classes.append(
                PairedClassData(
                    label_id=idx,
                    label_name=str(npz["label_{0}".format(idx)].item()),
                    x1=npz["x1_{0}".format(idx)].astype(np.float32),
                    x2=npz["x2_{0}".format(idx)].astype(np.float32),
                    sample_ids=npz["ids_{0}".format(idx)].astype(str),
                )
            )
        metadata = json.loads(str(npz["metadata_json"].item()))
        return DatasetBundle(name=str(npz["name"].item()), classes=tuple(classes), metadata=metadata)


def _resolve_uo_file(root: Path, folder: str, stem: str) -> Path:
    folder_path = root / folder
    variants = [stem, stem.replace("_", "-")]
    for value in variants:
        for suffix in (".mat", ".MAT"):
            path = folder_path / (value + suffix)
            if path.is_file():
                return path
    target = re.sub(r"[^a-z0-9]", "", stem.lower())
    if folder_path.is_dir():
        for path in folder_path.iterdir():
            if path.suffix.lower() == ".mat" and re.sub(r"[^a-z0-9]", "", path.stem.lower()) == target:
                return path
    raise FileNotFoundError("Cannot locate UO file '{0}' under {1}.".format(stem, folder_path))


def _largest_numeric_matrix(mat: Mapping[str, object]) -> np.ndarray:
    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.size and np.issubdtype(array.dtype, np.number):
            candidates.append(np.squeeze(array))
    if not candidates:
        raise KeyError("No numeric MATLAB matrix was found.")
    candidates.sort(key=lambda array: array.size, reverse=True)
    result = np.asarray(candidates[0])
    if result.ndim == 1:
        raise ValueError("Largest numeric MATLAB variable is one-dimensional.")
    if result.ndim > 2:
        result = result.reshape(result.shape[0], -1)
    if result.shape[0] <= 10 and result.shape[1] > result.shape[0]:
        result = result.T
    return np.asarray(result, dtype=np.float32)


def _read_uo_record(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mat = loadmat(str(path))
    array = None
    for key in (path.stem, path.stem.replace("-", "_"), path.stem.replace("_", "-")):
        if key in mat:
            candidate = np.asarray(mat[key])
            if candidate.size and np.issubdtype(candidate.dtype, np.number):
                array = np.squeeze(candidate)
                break
    if array is None:
        array = _largest_numeric_matrix(mat)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] <= 10 and array.shape[1] > array.shape[0]:
        array = array.T
    required = max(config.UO_VIB_COLUMN, config.UO_ACOUSTIC_COLUMN) + 1
    if array.ndim != 2 or array.shape[1] < required:
        raise ValueError("UO file has insufficient columns: {0}, shape={1}.".format(path, array.shape))
    return (
        np.asarray(array[:, config.UO_VIB_COLUMN], dtype=np.float32),
        np.asarray(array[:, config.UO_ACOUSTIC_COLUMN], dtype=np.float32),
    )


def load_uo_dataset(rebuild_cache: bool = False) -> DatasetBundle:
    files = []
    for _, sources in config.UO_CLASS_SOURCES:
        for folder, stem in sources:
            files.append(_resolve_uo_file(config.UO_DATA_ROOT, folder, stem))
    settings = {
        "segment_length": config.SEGMENT_LENGTH,
        "vib_column": config.UO_VIB_COLUMN,
        "acoustic_column": config.UO_ACOUSTIC_COLUMN,
        "segments_per_file": 200,
        "normalization": "none",
    }
    signature = _stat_manifest(files, settings)
    cache_path = config.CACHE_ROOT / "uo_7class_2048_v3.npz"
    if not rebuild_cache:
        cached = _cache_load(cache_path, signature)
        if cached is not None:
            return cached

    classes = []
    file_cursor = 0
    for label_id, (label_name, sources) in enumerate(config.UO_CLASS_SOURCES):
        x1_parts = []
        x2_parts = []
        ids = []
        for folder, stem in sources:
            path = files[file_cursor]
            file_cursor += 1
            vibration, acoustic = _read_uo_record(path)
            vib_segments, starts = segment_nonoverlap(vibration, config.SEGMENT_LENGTH, 200)
            aco_segments, starts_a = segment_nonoverlap(acoustic, config.SEGMENT_LENGTH, 200)
            count = min(len(vib_segments), len(aco_segments))
            if not np.array_equal(starts[:count], starts_a[:count]):
                raise RuntimeError("UO paired segmentation starts differ for {0}.".format(path))
            x1_parts.append(vib_segments[:count])
            x2_parts.append(aco_segments[:count])
            relative = path.relative_to(config.UO_DATA_ROOT) if config.UO_DATA_ROOT in path.parents else path
            ids.extend(["{0}:{1}".format(relative.as_posix(), int(start)) for start in starts[:count]])
        classes.append(
            PairedClassData(
                label_id=label_id,
                label_name=label_name,
                x1=np.vstack(x1_parts).astype(np.float32),
                x2=np.vstack(x2_parts).astype(np.float32),
                sample_ids=np.asarray(ids, dtype="U256"),
            )
        )
    bundle = DatasetBundle("UO", tuple(classes), {"signature": signature, "settings": settings})
    _cache_save(cache_path, bundle, signature)
    return bundle


def _extract_kaist_values(mat: Mapping[str, object]) -> np.ndarray:
    if "Signal" in mat:
        try:
            signal = mat["Signal"]
            values = signal["y_values"][0, 0]["values"][0, 0]
            array = np.asarray(values)
            return array.reshape(-1, 1) if array.ndim == 1 else np.asarray(array)
        except Exception as exc:
            raise RuntimeError("Failed to parse Signal.y_values.values: {0}".format(exc))
    return _largest_numeric_matrix(mat)


def inspect_kaist_vibration_matrix(path: Path) -> Dict[str, Any]:
    values = _extract_kaist_values(loadmat(str(path)))
    array = np.asarray(values)
    if array.ndim != 2:
        array = array.reshape(-1, 1)
    return {
        "file": str(path),
        "shape": [int(value) for value in array.shape],
        "configured_column": int(config.KAIST_VIB_COLUMN),
        "configured_column_available": bool(array.shape[1] > int(config.KAIST_VIB_COLUMN)),
        "column_variances": [float(np.var(array[:, index])) for index in range(array.shape[1])],
    }


def _pick_kaist_vibration(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        array = array.reshape(-1, 1)
    column = int(config.KAIST_VIB_COLUMN)
    if array.shape[1] <= column:
        raise RuntimeError(
            "Configured KAIST vibration column {0} is unavailable for matrix shape {1}. "
            "Do not silently fall back to another sensor channel; inspect the file and set MHFL_KAIST_VIB_COLUMN."
            .format(column, array.shape)
        )
    return np.asarray(array[:, column], dtype=np.float32)


def _normalize_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _is_time_or_temperature(name: str) -> bool:
    token = _normalize_channel_name(name)
    return any(part in token for part in ("time", "timestamp", "timestmp", "temp", "temperature", "housing"))


def _is_phase_u(name: str) -> bool:
    token = _normalize_channel_name(name)
    return token == "u" or "uphase" in token or token.endswith("uphase")


def inspect_tdms_channels(path: Path) -> List[Dict[str, Any]]:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("nptdms is required: pip install nptdms") from exc
    tdms = TdmsFile.read(str(path))
    rows = []
    for group in tdms.groups():
        for channel in group.channels():
            try:
                values = np.asarray(channel[:], dtype=np.float32).squeeze()
            except Exception:
                continue
            if values.ndim == 2 and values.shape[1] >= 1:
                values = values[:, 0]
            if values.ndim != 1:
                continue
            rows.append(
                {
                    "group": group.name or "",
                    "channel": channel.name or "",
                    "length": int(values.size),
                    "variance": float(np.var(values)) if values.size else 0.0,
                    "eligible": bool(values.size >= config.SEGMENT_LENGTH * 2 and not _is_time_or_temperature(channel.name or "")),
                }
            )
    return rows


def _read_tdms_current(path: Path, allow_fallback: bool) -> Tuple[np.ndarray, CurrentChannelSelection]:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("nptdms is required: pip install nptdms") from exc
    tdms = TdmsFile.read(str(path))
    channels = []
    for group in tdms.groups():
        for channel in group.channels():
            try:
                values = np.asarray(channel[:], dtype=np.float32).squeeze()
            except Exception:
                continue
            if values.ndim == 2 and values.shape[1] >= 1:
                values = values[:, 0]
            if values.ndim != 1 or values.size < config.SEGMENT_LENGTH * 2:
                continue
            if _is_time_or_temperature(channel.name or ""):
                continue
            channels.append((group.name or "", channel.name or "", values.astype(np.float32)))
    if not channels:
        raise RuntimeError("No eligible current channel found in {0}.".format(path))

    selected = None
    rule = ""
    if config.CURRENT_CHANNEL_NAME:
        selected = next((item for item in channels if item[1] == config.CURRENT_CHANNEL_NAME), None)
        rule = "explicit_exact_name"
        if selected is None:
            raise RuntimeError(
                "Configured current channel '{0}' was not found in {1}.".format(config.CURRENT_CHANNEL_NAME, path)
            )
    elif config.CURRENT_CHANNEL_REGEX:
        pattern = re.compile(config.CURRENT_CHANNEL_REGEX)
        matches = [item for item in channels if pattern.search(item[1])]
        if len(matches) != 1:
            raise RuntimeError(
                "Current-channel regex must match exactly one eligible channel in {0}; matched {1}.".format(path, len(matches))
            )
        selected = matches[0]
        rule = "explicit_regex"
    else:
        phase_matches = [item for item in channels if _is_phase_u(item[1])]
        if len(phase_matches) == 1:
            selected = phase_matches[0]
            rule = "recognized_u_phase"
        elif allow_fallback:
            selected = max(channels, key=lambda item: float(np.var(item[2])))
            rule = "max_variance_fallback"
        else:
            names = ["{0}/{1}".format(group, name) for group, name, _ in channels]
            raise RuntimeError(
                "U-phase current channel could not be identified unambiguously in {0}. Set "
                "MHFL_CURRENT_CHANNEL_NAME or MHFL_CURRENT_CHANNEL_REGEX after running 00_check_environment.py. "
                "Eligible channels: {1}".format(path, names)
            )
    assert selected is not None
    group, channel, values = selected
    fallback = rule == "max_variance_fallback"
    return values, CurrentChannelSelection(group, channel, len(values), float(np.var(values)), rule, fallback)


def _resolve_kaist_file(directory: Path, load_tag: str, fault: str, suffixes: Iterable[str]) -> Path:
    suffix_list = list(suffixes)
    for suffix in suffix_list:
        path = directory / ("{0}_{1}{2}".format(load_tag, fault, suffix))
        if path.is_file():
            return path
    target = re.sub(r"[^a-z0-9]", "", "{0}_{1}".format(load_tag, fault).lower())
    if directory.is_dir():
        allowed = {suffix.lower() for suffix in suffix_list}
        for path in directory.iterdir():
            if path.suffix.lower() in allowed and re.sub(r"[^a-z0-9]", "", path.stem.lower()) == target:
                return path
    raise FileNotFoundError("Cannot find {0}_{1} under {2}.".format(load_tag, fault, directory))


def load_kaist_load(
    load_tag: str,
    rebuild_cache: bool = False,
    allow_current_fallback: Optional[bool] = None,
) -> DatasetBundle:
    if load_tag not in config.KAIST_LOADS:
        raise ValueError("Unknown KAIST load: {0}".format(load_tag))
    allow_fallback = config.ALLOW_CURRENT_CHANNEL_FALLBACK if allow_current_fallback is None else bool(allow_current_fallback)
    vib_files = []
    current_files = []
    for _, fault in config.KAIST_CLASSES:
        vib_files.append(_resolve_kaist_file(config.KAIST_VIB_DIR, load_tag, fault, (".mat", ".MAT")))
        current_files.append(_resolve_kaist_file(config.KAIST_CURRENT_DIR, load_tag, fault, (".tdms", ".TDMS")))
    settings = {
        "segment_length": config.SEGMENT_LENGTH,
        "vibration_column": config.KAIST_VIB_COLUMN,
        "current_channel_name": config.CURRENT_CHANNEL_NAME,
        "current_channel_regex": config.CURRENT_CHANNEL_REGEX,
        "allow_fallback": allow_fallback,
        "normalization": "per_segment_zscore_before_noise",
        "max_segments_per_class": 400,
        "load": load_tag,
    }
    signature = _stat_manifest(vib_files + current_files, settings)
    cache_path = config.CACHE_ROOT / ("kaist_{0}_5class_2048_v3.npz".format(load_tag))
    if not rebuild_cache:
        cached = _cache_load(cache_path, signature)
        if cached is not None:
            return cached

    classes = []
    channel_selections = []
    for label_id, ((label_name, _), vib_path, current_path) in enumerate(zip(config.KAIST_CLASSES, vib_files, current_files)):
        vibration = _pick_kaist_vibration(_extract_kaist_values(loadmat(str(vib_path))))
        current, selection = _read_tdms_current(current_path, allow_fallback=allow_fallback)
        vib_segments, vib_starts = segment_nonoverlap(vibration, config.SEGMENT_LENGTH, 400)
        current_segments, current_starts = segment_nonoverlap(current, config.SEGMENT_LENGTH, 400)
        count = min(len(vib_segments), len(current_segments))
        ids = np.asarray(
            ["{0}:{1}|{2}:{3}".format(vib_path.name, int(vib_starts[i]), current_path.name, int(current_starts[i])) for i in range(count)],
            dtype="U256",
        )
        classes.append(
            PairedClassData(
                label_id=label_id,
                label_name=label_name,
                x1=zscore_per_segment(vib_segments[:count]),
                x2=zscore_per_segment(current_segments[:count]),
                sample_ids=ids,
            )
        )
        channel_selections.append({"file": str(current_path), **selection.to_dict()})
    metadata = {"signature": signature, "settings": settings, "channel_selections": channel_selections}
    bundle = DatasetBundle("KAIST-{0}".format(load_tag), tuple(classes), metadata)
    _cache_save(cache_path, bundle, signature)
    write_json(config.PROVENANCE_ROOT / ("channel_selection_{0}.json".format(load_tag)), channel_selections)
    return bundle


def create_split_plan(
    bundle: DatasetBundle,
    seed: int,
    max_train_per_class: int,
    val_per_class: int,
    test_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    sample_id_manifest = {
        str(class_data.label_id): sha256_json(class_data.sample_ids.tolist()) for class_data in bundle.classes
    }
    bundle_signature = str(dict(bundle.metadata).get("signature", {}).get("signature", ""))
    plan: Dict[str, Any] = {
        "dataset": bundle.name,
        "bundle_signature": bundle_signature,
        "sample_id_manifest": sample_id_manifest,
        "seed": int(seed),
        "max_train_per_class": int(max_train_per_class),
        "val_per_class": int(val_per_class),
        "test_per_class": None if test_per_class is None else int(test_per_class),
        "classes": {},
    }
    for class_data in bundle.classes:
        available = len(class_data.x1)
        required = int(max_train_per_class) + int(val_per_class)
        if test_per_class is not None:
            required += int(test_per_class)
        if required > available:
            raise ValueError(
                "Class {0} has {1} samples but the split requires {2}.".format(class_data.label_name, available, required)
            )
        order = rng.permutation(available)
        train_pool = order[: int(max_train_per_class)]
        val_start = int(max_train_per_class)
        val = order[val_start : val_start + int(val_per_class)]
        test_start = val_start + int(val_per_class)
        if test_per_class is None:
            test = order[test_start:]
        else:
            test = order[test_start : test_start + int(test_per_class)]
        plan["classes"][str(class_data.label_id)] = {
            "label_name": class_data.label_name,
            "train_pool": train_pool.tolist(),
            "val": val.tolist(),
            "test": test.tolist(),
        }
    plan["signature"] = sha256_json(plan)
    return plan


def get_or_create_split_plan(
    bundle: DatasetBundle,
    path: Path,
    seed: int,
    max_train_per_class: int,
    val_per_class: int,
    test_per_class: Optional[int] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    path = Path(path)
    expected_bundle_signature = str(dict(bundle.metadata).get("signature", {}).get("signature", ""))
    expected_id_manifest = {
        str(class_data.label_id): sha256_json(class_data.sample_ids.tolist()) for class_data in bundle.classes
    }
    if path.is_file() and not rebuild:
        plan = json.loads(path.read_text(encoding="utf-8"))
        mismatches = []
        expected_values = {
            "dataset": bundle.name,
            "bundle_signature": expected_bundle_signature,
            "sample_id_manifest": expected_id_manifest,
            "seed": int(seed),
            "max_train_per_class": int(max_train_per_class),
            "val_per_class": int(val_per_class),
            "test_per_class": None if test_per_class is None else int(test_per_class),
        }
        for key, expected in expected_values.items():
            if plan.get(key) != expected:
                mismatches.append(key)
        if mismatches:
            raise RuntimeError(
                "Persistent split plan is incompatible with the requested dataset/protocol ({0}). "
                "Re-run with --rebuild-split after confirming the intended protocol.".format(", ".join(mismatches))
            )
        return plan
    plan = create_split_plan(bundle, seed, max_train_per_class, val_per_class, test_per_class)
    write_json(path, plan)
    return plan


def _pack(bundle: DatasetBundle, selections: Mapping[int, Sequence[int]]) -> SplitData:
    x1_rows = []
    x2_rows = []
    y_rows = []
    id_rows = []
    for class_data in bundle.classes:
        indices = np.asarray(selections[class_data.label_id], dtype=np.int64)
        x1_rows.append(class_data.x1[indices])
        x2_rows.append(class_data.x2[indices])
        y_rows.append(np.full(len(indices), class_data.label_id, dtype=np.int64))
        id_rows.append(class_data.sample_ids[indices])
    x1 = np.vstack(x1_rows).astype(np.float32)[..., None]
    x2 = np.vstack(x2_rows).astype(np.float32)[..., None]
    y_int = np.concatenate(y_rows).astype(np.int64)
    ids = np.concatenate(id_rows).astype(str)
    y_onehot = np.eye(bundle.num_classes, dtype=np.float32)[y_int]
    return SplitData(x1, x2, y_onehot, y_int, ids)


def apply_split_plan(bundle: DatasetBundle, plan: Mapping[str, Any], n_train_per_class: int) -> Dict[str, SplitData]:
    if int(n_train_per_class) > int(plan["max_train_per_class"]):
        raise ValueError("n_train_per_class exceeds the persistent train pool.")
    selections: Dict[str, Dict[int, Sequence[int]]] = {"train": {}, "val": {}, "test": {}}
    for class_data in bundle.classes:
        row = plan["classes"][str(class_data.label_id)]
        selections["train"][class_data.label_id] = row["train_pool"][: int(n_train_per_class)]
        selections["val"][class_data.label_id] = row["val"]
        selections["test"][class_data.label_id] = row["test"]
    result = {name: _pack(bundle, mapping) for name, mapping in selections.items()}
    assert_no_overlap(result["train"], result["val"])
    assert_no_overlap(result["train"], result["test"])
    assert_no_overlap(result["val"], result["test"])
    return result


def create_uo_paper_split(
    bundle: DatasetBundle,
    seed: int,
    n_train_per_class: int,
) -> Dict[str, SplitData]:
    """Mirror the uploaded UO evaluation: first N paired samples train, all remaining samples test.

    A per-N persistent permutation is used so the added traditional baseline can be
    compared under the same sample-count logic as the original UO scripts. No test
    observation is used for hyperparameter selection; the traditional baseline uses
    training-only cross-validation.
    """
    n_train = int(n_train_per_class)
    path = config.SPLIT_ROOT / ("uo_paper_seed{0}_N{1}.json".format(int(seed), n_train))
    plan = get_or_create_split_plan(
        bundle,
        path,
        seed=int(seed),
        max_train_per_class=n_train,
        val_per_class=0,
        test_per_class=None,
        rebuild=False,
    )
    result = apply_split_plan(bundle, plan, n_train_per_class=n_train)
    if len(result["val"]) != 0:
        raise RuntimeError("UO paper split unexpectedly contains validation samples.")
    return {"train": result["train"], "test": result["test"]}


def build_external_test(bundle: DatasetBundle, seed: int = 0) -> SplitData:
    selections = {class_data.label_id: np.arange(len(class_data.x1), dtype=np.int64) for class_data in bundle.classes}
    split = _pack(bundle, selections)
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(split))
    return SplitData(split.x1[order], split.x2[order], split.y_onehot[order], split.y_int[order], split.sample_ids[order])


def assert_no_overlap(first: SplitData, second: SplitData) -> None:
    overlap = set(first.sample_ids.tolist()).intersection(second.sample_ids.tolist())
    if overlap:
        raise RuntimeError("Detected {0} overlapping paired segments between splits.".format(len(overlap)))


def split_with_noise(split: SplitData, snr_x1: float, snr_x2: float, seed: int) -> SplitData:
    rng1 = np.random.RandomState(int(seed))
    rng2 = np.random.RandomState(int(seed) + 9973)
    return SplitData(
        add_gaussian_noise(split.x1, snr_x1, rng1),
        add_gaussian_noise(split.x2, snr_x2, rng2),
        split.y_onehot,
        split.y_int,
        split.sample_ids,
    )


def replace_modality(split: SplitData, modality: int, value: float = 0.0) -> SplitData:
    if modality not in {1, 2}:
        raise ValueError("modality must be 1 or 2.")
    x1 = np.full_like(split.x1, float(value)) if modality == 1 else split.x1.copy()
    x2 = np.full_like(split.x2, float(value)) if modality == 2 else split.x2.copy()
    return SplitData(x1, x2, split.y_onehot, split.y_int, split.sample_ids)
