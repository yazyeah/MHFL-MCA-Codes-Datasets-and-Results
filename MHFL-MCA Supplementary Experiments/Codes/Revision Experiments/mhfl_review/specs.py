from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import config


@dataclass(frozen=True)
class ModelSpec:
    num_classes: int
    segment_length: int = config.SEGMENT_LENGTH
    vib_layers: int = 4
    other_layers: int = 5
    vib_dropout: float = 0.2
    other_dropout: float = 0.3
    attention_dim: int = 256
    fcb_units: int = 256
    encoder_mode: str = "heterogeneous"
    use_caim: bool = True
    gate_mode: str = "two_stage"
    attention_initializer: str = "uniform"
    cross_feature_order: str = "value_source"
    expected_params_m: Optional[float] = None
    spec_status: str = "unspecified"
    source_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def updated(self, **changes: Any) -> "ModelSpec":
        result = replace(self, **changes)
        result.validate()
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelSpec":
        allowed = {field.name for field in fields(cls)}
        result = cls(**{key: value for key, value in dict(values).items() if key in allowed})
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: Path) -> "ModelSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.encoder_mode not in {"heterogeneous", "homogeneous_vibration", "homogeneous_other"}:
            raise ValueError("Invalid encoder_mode: {0}".format(self.encoder_mode))
        if self.gate_mode not in {"two_stage", "direct_softmax", "equal"}:
            raise ValueError("Invalid gate_mode: {0}".format(self.gate_mode))
        if self.cross_feature_order not in {"value_source", "query_order"}:
            raise ValueError("Invalid cross_feature_order: {0}".format(self.cross_feature_order))
        if self.vib_layers < 1 or self.other_layers < 1:
            raise ValueError("Encoder depths must be positive.")
        if self.attention_dim < 1 or self.fcb_units < 1 or self.num_classes < 2:
            raise ValueError("attention_dim, fcb_units and num_classes must be positive.")
        if not 0.0 <= self.vib_dropout < 1.0 or not 0.0 <= self.other_dropout < 1.0:
            raise ValueError("Dropout values must lie in [0, 1).")


def manuscript_spec(case: str, **overrides: Any) -> ModelSpec:
    token = str(case).lower().strip()
    if token not in {"kaist", "uo"}:
        raise ValueError("case must be 'kaist' or 'uo'.")
    file_name = "manuscript_kaist.json" if token == "kaist" else "manuscript_uo.json"
    values = json.loads((config.CONFIG_ROOT / file_name).read_text(encoding="utf-8"))
    values.update(overrides)
    return ModelSpec.from_dict(values)


def _encoder_style(spec: ModelSpec, branch: str) -> Tuple[int, int, str, str, float]:
    if spec.encoder_mode == "heterogeneous":
        style = "vibration" if branch == "vibration" else "other"
    elif spec.encoder_mode == "homogeneous_vibration":
        style = "vibration"
    else:
        style = "other"
    if style == "vibration":
        return spec.vib_layers, 16, "leaky_relu", "average", spec.vib_dropout
    return spec.other_layers, 8, "gelu", "max", spec.other_dropout


def encoder_shape_and_params(segment_length: int, n_layers: int, kernel_size: int) -> Tuple[int, int, int]:
    filters = [32, 64, 128, 256, 256]
    temporal = int(segment_length)
    input_channels = 1
    params = 0
    output_channels = 1
    for index in range(int(n_layers)):
        output_channels = filters[index] if index < len(filters) else 256
        params += int(kernel_size) * input_channels * output_channels + output_channels
        if index < 2:
            temporal = int(math.ceil(temporal / 2.0))
        temporal = int(math.ceil(temporal / 2.0))
        input_channels = output_channels
    return temporal, output_channels, params


def analytical_parameter_count(spec: ModelSpec) -> int:
    spec.validate()
    v_layers, v_kernel, _, _, _ = _encoder_style(spec, "vibration")
    o_layers, o_kernel, _, _, _ = _encoder_style(spec, "other")
    tv, cv, pv = encoder_shape_and_params(spec.segment_length, v_layers, v_kernel)
    to, co, po = encoder_shape_and_params(spec.segment_length, o_layers, o_kernel)

    if spec.use_caim:
        cross_params = (
            cv * spec.attention_dim
            + co * spec.attention_dim
            + co * spec.attention_dim
            + co * spec.attention_dim
            + cv * spec.attention_dim
            + cv * spec.attention_dim
        )
        query_v_dim = tv * spec.attention_dim
        query_o_dim = to * spec.attention_dim
        if spec.cross_feature_order == "query_order":
            branch1_dim, branch2_dim = query_v_dim, query_o_dim
        else:
            branch1_dim, branch2_dim = query_o_dim, query_v_dim
    else:
        cross_params = 0
        branch1_dim, branch2_dim = tv * cv, to * co

    if spec.gate_mode == "two_stage":
        # Two modality-specific sigmoid scores plus a learned 2x2 softmax calibrator.
        gate_params = branch1_dim + 1 + branch2_dim + 1 + 2 * 2 + 2
    elif spec.gate_mode == "direct_softmax":
        # Fair reviewer ablation: remove sigmoid and 2x2 calibrator, but retain one
        # independent linear logit per complete modality representation.
        gate_params = branch1_dim + 1 + branch2_dim + 1
    else:
        gate_params = 0

    fused_dim = branch1_dim + branch2_dim
    classifier_params = fused_dim * spec.fcb_units + spec.fcb_units
    classifier_params += spec.fcb_units * spec.num_classes + spec.num_classes
    return int(pv + po + cross_params + gate_params + classifier_params)


def parameter_count_m(spec: ModelSpec) -> float:
    return analytical_parameter_count(spec) / 1_000_000.0


def spec_fingerprint(spec: ModelSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_architectures(
    target_params_m: float,
    num_classes: int = 5,
    tolerance_m: float = 0.02,
    attention_dims: Iterable[int] = (128, 256),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vib_layers in range(3, 6):
        for other_layers in range(3, 6):
            for attention_dim in attention_dims:
                spec = ModelSpec(
                    num_classes=num_classes,
                    vib_layers=vib_layers,
                    other_layers=other_layers,
                    attention_dim=int(attention_dim),
                    cross_feature_order="value_source",
                )
                params_m = parameter_count_m(spec)
                delta = abs(params_m - float(target_params_m))
                if delta <= float(tolerance_m):
                    rows.append(
                        {
                            "vib_layers": vib_layers,
                            "other_layers": other_layers,
                            "attention_dim": int(attention_dim),
                            "params_m": params_m,
                            "absolute_delta_m": delta,
                        }
                    )
    return sorted(rows, key=lambda row: (row["absolute_delta_m"], row["vib_layers"], row["other_layers"]))


def branch_semantics(spec: ModelSpec) -> Tuple[str, str]:
    if not spec.use_caim:
        return "vibration_encoder", "other_encoder"
    if spec.cross_feature_order == "value_source":
        return "vibration_source", "other_source"
    return "vibration_query", "other_query"
