from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


SUITE_VERSION = "3.2.0"
GLOBAL_SEED = 20260806


def _first_existing(candidates: Iterable[Path]) -> Path:
    values = list(candidates)
    if not values:
        raise ValueError("At least one path candidate is required.")
    for candidate in values:
        if candidate.exists():
            return candidate
    return values[0]


# -----------------------------------------------------------------------------
# Relocatable suite paths and user-overridable local data paths.
# -----------------------------------------------------------------------------
SUITE_ROOT = Path(os.environ.get("MHFL_SUITE_ROOT", str(Path(__file__).resolve().parents[1])))
PROJECT_ROOT = Path(os.environ.get("MHFL_PROJECT_ROOT", str(SUITE_ROOT)))
DATA_ROOT = Path(os.environ.get("MHFL_DATA_ROOT", str(SUITE_ROOT / "data")))
UO_DATA_ROOT = Path(
    os.environ.get(
        "MHFL_UO_DATA_ROOT",
        str(
            _first_existing(
                [
                    DATA_ROOT / "3_MatLab_Raw_Data",
                    DATA_ROOT / "3_MATLAB_Raw_Data",
                    DATA_ROOT / "3_Matlab_Raw_Data",
                ]
            )
        ),
    )
)
KAIST_VIB_DIR = Path(
    os.environ.get(
        "MHFL_KAIST_VIB_DIR",
        str(_first_existing([DATA_ROOT / "vibration" / "vibration", DATA_ROOT / "vibration"])),
    )
)
KAIST_CURRENT_DIR = Path(
    os.environ.get(
        "MHFL_KAIST_CURRENT_DIR",
        str(_first_existing([DATA_ROOT / "current", DATA_ROOT / "current" / "current"])),
    )
)

RUN_TAG = os.environ.get("MHFL_RUN_TAG", "manual").strip() or "manual"
OUTPUT_ROOT = SUITE_ROOT / "outputs" / RUN_TAG
CACHE_ROOT = SUITE_ROOT / "cache"
CHECKPOINT_ROOT = SUITE_ROOT / "checkpoints" / RUN_TAG
SPLIT_ROOT = SUITE_ROOT / "splits" / RUN_TAG
PROVENANCE_ROOT = SUITE_ROOT / "provenance" / RUN_TAG
TEMP_ROOT = Path(os.environ.get("MHFL_TEMP_ROOT", str(SUITE_ROOT / "tmp" / RUN_TAG)))
CONFIG_ROOT = SUITE_ROOT / "configs"
KAIST_OPTUNA_CONFIRMED_PATH = CONFIG_ROOT / "kaist_optuna_confirmed.json"
KAIST_ADDITIONAL_ABLATION_FULL_REFERENCE_PATH = (
    CONFIG_ROOT / "kaist_additional_ablation_full_reference.json"
)
MAIN_MANUSCRIPT_DEEP_REFERENCE_PATH = CONFIG_ROOT / "main_manuscript_deep_reference.json"
MAIN_MANUSCRIPT_SOURCE_ROOT = Path(
    os.environ.get(
        "MHFL_MAIN_MANUSCRIPT_SOURCE_ROOT",
        str(SUITE_ROOT / "provenance" / "main_manuscript_sources"),
    )
)
KAIST_ADDITIONAL_ABLATION_EXPECTED_PARAMS_M = 7.380
KAIST_ADDITIONAL_ABLATION_EXPECTED_METRICS = {
    "2Nm_0dB": {
        "accuracy_mean": 0.9999,
        "accuracy_sd": 0.0002,
        "macro_f1_mean": 0.9999,
        "macro_f1_sd": 0.0002,
    },
    "4Nm_0dB": {
        "accuracy_mean": 0.9925,
        "accuracy_sd": 0.0096,
        "macro_f1_mean": 0.9925,
        "macro_f1_sd": 0.0097,
    },
}
KAIST_OPTUNA_REQUIRED_PARAMETERS = (
    "dropout_vib",
    "dropout_cur",
    "atten_dim",
    "n_layers_vib",
    "n_layers_cur",
    "lr",
    "batch_size",
)


def _confirmed_finite_number(payload: Dict[str, Any], field: str) -> float:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Confirmed KAIST field '{0}' must be numeric.".format(field))
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Confirmed KAIST field '{0}' must be finite.".format(field))
    return number


def _confirmed_positive_integer(payload: Dict[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError("Confirmed KAIST field '{0}' must be a positive integer.".format(field))
    return int(value)


def load_confirmed_kaist_optuna_config(path: Optional[Path] = None) -> Dict[str, Any]:
    target = KAIST_OPTUNA_CONFIRMED_PATH if path is None else Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError("Confirmed KAIST Optuna configuration is missing: {0}".format(target)) from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Cannot read confirmed KAIST Optuna configuration: {0}".format(target)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Confirmed KAIST Optuna configuration is invalid JSON: {0}".format(target)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Confirmed KAIST Optuna configuration must contain a JSON object.")

    missing = [name for name in KAIST_OPTUNA_REQUIRED_PARAMETERS if name not in payload]
    if missing:
        raise RuntimeError(
            "Confirmed KAIST Optuna configuration is missing required parameters: {0}".format(
                ", ".join(missing)
            )
        )
    if payload.get("confirmation_status") != "confirmed":
        raise RuntimeError(
            "KAIST confirmation_status must be 'confirmed'; found {0!r}.".format(
                payload.get("confirmation_status")
            )
        )

    for field in ("dropout_vib", "dropout_cur"):
        value = _confirmed_finite_number(payload, field)
        if not 0.0 <= value < 1.0:
            raise RuntimeError("Confirmed KAIST field '{0}' must be in [0, 1).".format(field))
    for field in ("atten_dim", "n_layers_vib", "n_layers_cur"):
        _confirmed_positive_integer(payload, field)
    learning_rate = _confirmed_finite_number(payload, "lr")
    if learning_rate <= 0.0:
        raise RuntimeError("Confirmed KAIST field 'lr' must be positive.")
    _confirmed_positive_integer(payload, "batch_size")
    return dict(payload)


def load_kaist_additional_ablation_full_reference(path: Optional[Path] = None) -> Dict[str, Any]:
    target = KAIST_ADDITIONAL_ABLATION_FULL_REFERENCE_PATH if path is None else Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("KAIST additional-ablation Full reference is missing: {0}".format(target)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot read KAIST additional-ablation Full reference: {0}".format(target)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("KAIST additional-ablation Full reference must be a JSON object.")

    expected_metadata = {
        "model": "Full MHFL-MCA",
        "source": "main_manuscript_stage2_experiment",
        "protocol": "stage2_load_shift",
        "source_load": "0Nm",
        "n_train_per_class": 30,
        "runs": 10,
        "aggregation": "trimmed_mean_sd_drop_one_high_one_low",
        "reference_type": "main_manuscript_aggregated_reference",
    }
    for field, expected in expected_metadata.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                "KAIST Full reference field '{0}' must be {1!r}; found {2!r}.".format(
                    field, expected, payload.get(field)
                )
            )
    params_m = payload.get("params_m")
    if isinstance(params_m, bool) or not isinstance(params_m, (int, float)) or not math.isfinite(float(params_m)):
        raise RuntimeError("KAIST Full reference params_m must be finite.")
    if not math.isclose(
        float(params_m),
        KAIST_ADDITIONAL_ABLATION_EXPECTED_PARAMS_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("KAIST Full reference params_m does not match the published 7.380 M value.")

    conditions = payload.get("conditions")
    required_conditions = {"2Nm_0dB", "4Nm_0dB"}
    if not isinstance(conditions, dict) or set(conditions) != required_conditions:
        raise RuntimeError(
            "KAIST Full reference must contain exactly the two Stage-2 clean-load conditions: {0}.".format(
                sorted(required_conditions)
            )
        )
    metric_fields = ("accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd")
    for condition in sorted(required_conditions):
        values = conditions.get(condition)
        if not isinstance(values, dict):
            raise RuntimeError("KAIST Full reference condition {0} must be an object.".format(condition))
        for field in metric_fields:
            value = values.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError("KAIST Full reference {0}.{1} must be numeric.".format(condition, field))
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("KAIST Full reference {0}.{1} must be finite.".format(condition, field))
            if field.endswith("_sd"):
                valid = number >= 0.0
            else:
                valid = 0.0 <= number <= 1.0
            if not valid:
                raise RuntimeError("KAIST Full reference {0}.{1} is outside its valid range.".format(condition, field))
            expected = KAIST_ADDITIONAL_ABLATION_EXPECTED_METRICS[condition][field]
            if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(
                    "KAIST Full reference {0}.{1} does not match the published clean-load value.".format(
                        condition, field
                    )
                )
    return dict(payload)


def ensure_runtime_dirs() -> None:
    for path in (
        SUITE_ROOT,
        OUTPUT_ROOT,
        CACHE_ROOT,
        CHECKPOINT_ROOT,
        SPLIT_ROOT,
        PROVENANCE_ROOT,
        TEMP_ROOT,
        CONFIG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


# TensorFlow reads these variables at import/runtime. Executable scripts import
# config before TensorFlow-dependent modules.
os.environ.setdefault("TEMP", str(TEMP_ROOT))
os.environ.setdefault("TMP", str(TEMP_ROOT))
os.environ.setdefault("TMPDIR", str(TEMP_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# -----------------------------------------------------------------------------
# Dataset configuration.
# -----------------------------------------------------------------------------
SEGMENT_LENGTH = 2048
UO_SAMPLING_RATE = 42_000.0
KAIST_SAMPLING_RATE = 25_600.0
UO_VIB_COLUMN = int(os.environ.get("MHFL_UO_VIB_COLUMN", "0"))
UO_ACOUSTIC_COLUMN = int(os.environ.get("MHFL_UO_ACOUSTIC_COLUMN", "1"))
KAIST_VIB_COLUMN = int(
    os.environ.get("MHFL_KAIST_VIB_COLUMN", "0")
)
CURRENT_CHANNEL_NAME = os.environ.get("MHFL_CURRENT_CHANNEL_NAME", "").strip()
CURRENT_CHANNEL_REGEX = os.environ.get("MHFL_CURRENT_CHANNEL_REGEX", "").strip()
ALLOW_CURRENT_CHANNEL_FALLBACK = os.environ.get("MHFL_ALLOW_CURRENT_FALLBACK", "0") == "1"
ACCEPT_KAIST_SPEC = os.environ.get("MHFL_ACCEPT_KAIST_SPEC", "0") == "1"

UO_CLASS_SOURCES = [
    ("Healthy", [("1_Healthy", "H_1_0"), ("1_Healthy", "H_2_0")]),
    ("Developing fault (inner race)", [("2_Inner_Race_Faults", "I_1_1"), ("2_Inner_Race_Faults", "I_2_1")]),
    ("Faulty (inner race)", [("2_Inner_Race_Faults", "I_1_2"), ("2_Inner_Race_Faults", "I_2_2")]),
    ("Faulty (outer race)", [("3_Outer_Race_Faults", "O_6_2"), ("3_Outer_Race_Faults", "O_7_2")]),
    ("Faulty (ball)", [("4_Ball_Faults", "B_11_2"), ("4_Ball_Faults", "B_12_2")]),
    ("Developing fault (cage)", [("5_Cage_Faults", "C_16_1"), ("5_Cage_Faults", "C_17_1")]),
    ("Faulty (cage)", [("5_Cage_Faults", "C_16_2"), ("5_Cage_Faults", "C_17_2")]),
]
KAIST_CLASSES = [
    ("Normal", "Normal"),
    ("IF-1", "BPFI_03"),
    ("IF-2", "BPFI_10"),
    ("OF-1", "BPFO_03"),
    ("OF-2", "BPFO_10"),
]
KAIST_LOADS = ("0Nm", "2Nm", "4Nm")

# -----------------------------------------------------------------------------
# Training and reviewer-experiment protocols.
# -----------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1.0e-3
UO_MANUSCRIPT_LEARNING_RATE = 0.00139

try:
    KAIST_OPTUNA_CONFIRMED = load_confirmed_kaist_optuna_config()
    KAIST_OPTUNA_CONFIG_ERROR: Optional[str] = None
except RuntimeError as exc:
    # Keep UO/import-only workflows available, but every KAIST training entry
    # point must call require_confirmed_kaist_training_config before use.
    KAIST_OPTUNA_CONFIRMED = {}
    KAIST_OPTUNA_CONFIG_ERROR = str(exc)

KAIST_OPTUNA_CONFIRMATION_STATUS = (
    str(KAIST_OPTUNA_CONFIRMED.get("confirmation_status", "invalid"))
    if KAIST_OPTUNA_CONFIG_ERROR is None
    else "invalid"
)
KAIST_MANUSCRIPT_LEARNING_RATE: Optional[float] = (
    float(KAIST_OPTUNA_CONFIRMED["lr"]) if KAIST_OPTUNA_CONFIG_ERROR is None else None
)
KAIST_MANUSCRIPT_BATCH_SIZE: Optional[int] = (
    int(KAIST_OPTUNA_CONFIRMED["batch_size"]) if KAIST_OPTUNA_CONFIG_ERROR is None else None
)


def require_confirmed_kaist_training_config(mode: str) -> Tuple[float, int]:
    token = str(mode).lower().strip()
    if token not in {"fast", "full"}:
        raise ValueError("mode must be 'fast' or 'full'.")
    if (
        KAIST_OPTUNA_CONFIG_ERROR is not None
        or KAIST_OPTUNA_CONFIRMATION_STATUS != "confirmed"
        or KAIST_MANUSCRIPT_LEARNING_RATE is None
        or KAIST_MANUSCRIPT_BATCH_SIZE is None
    ):
        reason = KAIST_OPTUNA_CONFIG_ERROR or "confirmed KAIST training parameters were not loaded"
        raise RuntimeError(
            "Confirmed KAIST training configuration is required for {0} mode: {1}. "
            "Refusing to fall back to DEFAULT_LEARNING_RATE={2}.".format(
                token, reason, DEFAULT_LEARNING_RATE
            )
        )
    return float(KAIST_MANUSCRIPT_LEARNING_RATE), int(KAIST_MANUSCRIPT_BATCH_SIZE)


FAST_EPOCHS = 3
STAGE2_EPOCHS = 80
STAGE3_EPOCHS = 70
DEFAULT_PATIENCE = 10
VAL_PER_CLASS_KAIST = 50
VAL_PER_CLASS_UO = 50
TEST_PER_CLASS_UO = 200
UO_MAX_TRAIN_PER_CLASS = 30
UO_SVM_CV_FOLDS = 3

BASE_SNR_DB = 0.0
CONSISTENCY_SNR_LOW_DB = -10.0
CONSISTENCY_SNR_HIGH_DB = 0.0
CONSISTENCY_SNR_BIAS_P = 2.0
CONSISTENCY_MAX_LAMBDA = 0.12
CONSISTENCY_WARMUP_EPOCHS = 15
MODALITY_CORRUPT_PROB = 0.08
MODALITY_EXTRA_SNR_DB = -12.0
GRADIENT_CLIP_NORM = 1.0

# Stage-3 robustness training mirrored from the uploaded KAIST source.
STAGE3_NOISY_SNR_LOW_DB = -10.0
STAGE3_NOISY_SNR_HIGH_DB = 0.0
STAGE3_SNR_BIAS_P = 3.0
STAGE3_NOISY_CE_WEIGHT = 0.35
STAGE3_MAX_LAMBDA = 0.10
STAGE3_WARMUP_EPOCHS = 15
STAGE3_HARD_SNR_DB = -10.0
STAGE3_HARD_CE_WEIGHT = 0.25
STAGE3_HARD_KL_WEIGHT = 0.50

SNR_GRID = (0, -2, -4, -6, -8, -10)
SEVERE_SNR_DB = -12.0
MODEL_SEEDS_FAST = (GLOBAL_SEED,)
MODEL_SEEDS_FULL = (GLOBAL_SEED, GLOBAL_SEED + 1, GLOBAL_SEED + 2)
NOISE_REALIZATIONS_FAST = 2
NOISE_REALIZATIONS_FULL = 5
ABLATION_REPEATS_FAST = 1
ABLATION_REPEATS_FULL = 10
LOWSHOT_REPEATS_FAST = 2
LOWSHOT_REPEATS_FULL = 10
TRADITIONAL_REPEATS_FAST = 2
TRADITIONAL_REPEATS_FULL = 10
# Seven predefined values make the full CAIM-threshold study 7 x 2 x 10 = 140 model slots.
LOWSHOT_N_GRID = (1, 2, 3, 4, 5, 7, 10)

# -----------------------------------------------------------------------------
# Publication-figure defaults (Nature-figure compatible, Python backend).
# -----------------------------------------------------------------------------
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 183.0
FIGURE_FONT_PT = 7.5
PANEL_LABEL_PT = 8.5
MIN_PDF_GLYPH_PT = 5.0
RASTER_DPI = 600
FIGURE_SIZE_TOLERANCE_MM = 0.8


def checkpoint_paths(protocol: str, seed: int, mode: Optional[str] = None) -> Tuple[Path, Path]:
    token = str(protocol).lower().strip()
    if token not in {"stage2", "stage3"}:
        raise ValueError("protocol must be 'stage2' or 'stage3'.")
    mode_token = "" if mode is None else "_{0}".format(str(mode).lower().strip())
    stem = "kaist_{0}{1}_seed{2}".format(token, mode_token, int(seed))
    return CHECKPOINT_ROOT / (stem + ".weights.h5"), CHECKPOINT_ROOT / (stem + ".manifest.json")


def enforce_run_safety(
    mode: str,
    allow_current_fallback: bool = False,
    accept_kaist_spec: bool = False,
    require_kaist_spec: bool = True,
) -> None:
    token = str(mode).lower().strip()
    if token not in {"fast", "full"}:
        raise ValueError("mode must be 'fast' or 'full'.")
    if token == "full" and require_kaist_spec:
        require_confirmed_kaist_training_config(token)
    if token == "full" and bool(allow_current_fallback):
        raise RuntimeError(
            "Current-channel fallback is prohibited in full mode. Inspect channel_manifest.json and set "
            "MHFL_CURRENT_CHANNEL_NAME or MHFL_CURRENT_CHANNEL_REGEX explicitly."
        )
    if token == "full" and RUN_TAG.lower() in {"fast", "smoke", "debug", "manual"}:
        raise RuntimeError(
            "Use a dedicated full-run tag, e.g. set MHFL_RUN_TAG=full_20260806 before running full mode."
        )
    accepted = bool(accept_kaist_spec) or bool(ACCEPT_KAIST_SPEC)
    if token == "full" and require_kaist_spec and not accepted:
        raise RuntimeError(
            "The KAIST architecture must be explicitly confirmed before a full run. The manuscript reports "
            "7.380 M parameters, while the public Optuna script does not hard-code its selected architecture. "
            "Run 09_model_spec_audit.py, compare against the original Optuna/model-summary artifact, then pass "
            "--accept-kaist-spec or set MHFL_ACCEPT_KAIST_SPEC=1."
        )


def describe_paths() -> str:
    return "\n".join(
        [
            "PROJECT_ROOT       = {0}".format(PROJECT_ROOT),
            "SUITE_ROOT         = {0}".format(SUITE_ROOT),
            "RUN_TAG            = {0}".format(RUN_TAG),
            "DATA_ROOT          = {0}".format(DATA_ROOT),
            "UO_DATA_ROOT       = {0}".format(UO_DATA_ROOT),
            "KAIST_VIB_DIR      = {0}".format(KAIST_VIB_DIR),
            "KAIST_CURRENT_DIR  = {0}".format(KAIST_CURRENT_DIR),
            "OUTPUT_ROOT        = {0}".format(OUTPUT_ROOT),
            "CHECKPOINT_ROOT    = {0}".format(CHECKPOINT_ROOT),
        ]
    )


ensure_runtime_dirs()
