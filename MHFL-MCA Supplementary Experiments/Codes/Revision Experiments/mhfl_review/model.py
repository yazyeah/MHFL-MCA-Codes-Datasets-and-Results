from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - exercised on the user's TF environment
    raise ImportError(
        "TensorFlow is required for MHFL-MCA model construction. Use the paper environment "
        "(Python 3.9.x and TensorFlow 2.7.x) before running training scripts."
    ) from exc

from .specs import ModelSpec, analytical_parameter_count, branch_semantics, manuscript_spec


@tf.keras.utils.register_keras_serializable(package="MHFL")
class CrossAttention(tf.keras.layers.Layer):
    """Unscaled bidirectional cross-attention used by the uploaded sources.

    The executable UO and KAIST scripts and the latest manuscript use
    softmax(QK^T), without the Transformer 1/sqrt(D) factor. This layer also
    exposes the row-stochastic attention matrix for reviewer visualizations.
    """

    def __init__(self, output_dim: int, kernel_initializer: str = "uniform", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.kernel_initializer = str(kernel_initializer)
        self.w_q: Optional[tf.Variable] = None
        self.w_k: Optional[tf.Variable] = None
        self.w_v: Optional[tf.Variable] = None

    def build(self, input_shape: Any) -> None:
        query_shape, key_value_shape = input_shape
        self.w_q = self.add_weight(
            name="W_q",
            shape=(int(query_shape[-1]), self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.w_k = self.add_weight(
            name="W_k",
            shape=(int(key_value_shape[-1]), self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.w_v = self.add_weight(
            name="W_v",
            shape=(int(key_value_shape[-1]), self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Any, **kwargs: Any) -> Any:
        query, key_value = inputs
        q = tf.linalg.matmul(query, self.w_q)
        k = tf.linalg.matmul(key_value, self.w_k)
        v = tf.linalg.matmul(key_value, self.w_v)
        scores = tf.linalg.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.linalg.matmul(weights, v), weights

    def get_config(self) -> Dict[str, Any]:
        result = super().get_config()
        result.update({"output_dim": self.output_dim, "kernel_initializer": self.kernel_initializer})
        return result


@tf.keras.utils.register_keras_serializable(package="MHFL")
class GELUActivation(tf.keras.layers.Layer):
    def call(self, inputs: Any, **kwargs: Any) -> Any:
        return tf.nn.gelu(inputs)


@tf.keras.utils.register_keras_serializable(package="MHFL")
class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs: Any, **kwargs: Any) -> Any:
        return tf.identity(inputs)


@tf.keras.utils.register_keras_serializable(package="MHFL")
class SelectColumn(tf.keras.layers.Layer):
    def __init__(self, index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs[:, self.index : self.index + 1]

    def get_config(self) -> Dict[str, Any]:
        result = super().get_config()
        result.update({"index": self.index})
        return result


@tf.keras.utils.register_keras_serializable(package="MHFL")
class EqualWeights(tf.keras.layers.Layer):
    def call(self, inputs: Any, **kwargs: Any) -> Any:
        batch_size = tf.shape(inputs)[0]
        return tf.ones((batch_size, 2), dtype=inputs.dtype) * tf.cast(0.5, inputs.dtype)


def _encoder_settings(spec: ModelSpec, branch: str) -> Dict[str, Any]:
    if spec.encoder_mode == "heterogeneous":
        style = "vibration" if branch == "vibration" else "other"
    elif spec.encoder_mode == "homogeneous_vibration":
        style = "vibration"
    else:
        style = "other"
    if style == "vibration":
        return {
            "n_layers": spec.vib_layers,
            "kernel_size": 16,
            "activation": "leaky_relu",
            "pooling": "average",
            "dropout": spec.vib_dropout,
        }
    return {
        "n_layers": spec.other_layers,
        "kernel_size": 8,
        "activation": "gelu",
        "pooling": "max",
        "dropout": spec.other_dropout,
    }


def _conv_encoder(
    input_tensor: tf.Tensor,
    prefix: str,
    n_layers: int,
    kernel_size: int,
    activation: str,
    pooling: str,
    dropout: float,
) -> tf.Tensor:
    x = input_tensor
    filters = [32, 64, 128, 256, 256]
    for layer_index in range(int(n_layers)):
        filters_out = filters[layer_index] if layer_index < len(filters) else 256
        stride = 2 if layer_index < 2 else 1
        x = tf.keras.layers.Conv1D(
            filters_out,
            int(kernel_size),
            strides=stride,
            padding="same",
            name="{0}_conv_{1}".format(prefix, layer_index + 1),
        )(x)
        if activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.2, name="{0}_act_{1}".format(prefix, layer_index + 1))(x)
        elif activation == "gelu":
            x = GELUActivation(name="{0}_act_{1}".format(prefix, layer_index + 1))(x)
        else:
            x = tf.keras.layers.Activation(activation, name="{0}_act_{1}".format(prefix, layer_index + 1))(x)
        if layer_index == int(n_layers) - 1:
            x = tf.keras.layers.Dropout(float(dropout), name="{0}_dropout".format(prefix))(x)
        pool_name = "{0}_pool_{1}".format(prefix, layer_index + 1)
        if pooling == "average":
            x = tf.keras.layers.AveragePooling1D(2, strides=2, padding="same", name=pool_name)(x)
        elif pooling == "max":
            x = tf.keras.layers.MaxPooling1D(2, strides=2, padding="same", name=pool_name)(x)
        else:
            raise ValueError("Unsupported pooling mode: {0}".format(pooling))
    return x


def build_models(spec: ModelSpec) -> Tuple[tf.keras.Model, Dict[str, tf.keras.Model]]:
    """Build classifier and interpretation submodels from a source-faithful spec.

    The UO source flattens directional outputs in query order, whereas the
    KAIST source flattens them in value-source order. This affects which AMRM
    weight should be interpreted as vibration- or second-modality evidence.
    """
    spec.validate()
    input_v = tf.keras.Input((spec.segment_length, 1), name="input_vibration")
    input_o = tf.keras.Input((spec.segment_length, 1), name="input_other")
    feat_v = _conv_encoder(input_v, prefix="vib", **_encoder_settings(spec, "vibration"))
    feat_o = _conv_encoder(input_o, prefix="other", **_encoder_settings(spec, "other"))

    attention_v_query_o = None
    attention_o_query_v = None
    if spec.use_caim:
        query_v_output, attention_v_query_o = CrossAttention(
            spec.attention_dim,
            kernel_initializer=spec.attention_initializer,
            name="caim_v_query_other_value",
        )([feat_v, feat_o])
        query_o_output, attention_o_query_v = CrossAttention(
            spec.attention_dim,
            kernel_initializer=spec.attention_initializer,
            name="caim_other_query_vibration_value",
        )([feat_o, feat_v])
        if spec.cross_feature_order == "query_order":
            branch1, branch2 = query_v_output, query_o_output
        else:
            branch1, branch2 = query_o_output, query_v_output
    else:
        branch1 = IdentityLayer(name="no_caim_branch1")(feat_v)
        branch2 = IdentityLayer(name="no_caim_branch2")(feat_o)

    flat1 = tf.keras.layers.Flatten(name="flat_branch1")(branch1)
    flat2 = tf.keras.layers.Flatten(name="flat_branch2")(branch2)

    score1 = None
    score2 = None
    if spec.gate_mode == "two_stage":
        score1 = tf.keras.layers.Dense(1, activation="sigmoid", name="score_branch1")(flat1)
        score2 = tf.keras.layers.Dense(1, activation="sigmoid", name="score_branch2")(flat2)
        score_pair = tf.keras.layers.Concatenate(name="score_pair")([score1, score2])
        modality_weights = tf.keras.layers.Dense(2, activation="softmax", name="modality_weights")(score_pair)
    elif spec.gate_mode == "direct_softmax":
        # Reviewer-requested fair comparison: remove the sigmoid stage and the
        # learned 2x2 calibration, retaining one independent linear logit per
        # complete modality representation before a parameter-free softmax.
        logit1 = tf.keras.layers.Dense(1, activation=None, name="direct_logit_branch1")(flat1)
        logit2 = tf.keras.layers.Dense(1, activation=None, name="direct_logit_branch2")(flat2)
        logit_pair = tf.keras.layers.Concatenate(name="direct_logit_pair")([logit1, logit2])
        modality_weights = tf.keras.layers.Softmax(name="modality_weights")(logit_pair)
    else:
        gate_reference = tf.keras.layers.Concatenate(name="equal_gate_reference")([flat1, flat2])
        modality_weights = EqualWeights(name="modality_weights")(gate_reference)

    weight1 = SelectColumn(0, name="weight_branch1")(modality_weights)
    weight2 = SelectColumn(1, name="weight_branch2")(modality_weights)
    scaled1 = tf.keras.layers.Multiply(name="scaled_branch1")([flat1, weight1])
    scaled2 = tf.keras.layers.Multiply(name="scaled_branch2")([flat2, weight2])
    fused_input = tf.keras.layers.Concatenate(name="fused_concat")([scaled1, scaled2])
    fused = tf.keras.layers.Dense(spec.fcb_units, name="fcb_dense")(fused_input)
    fused = tf.keras.layers.LeakyReLU(alpha=0.2, name="fcb_leaky_relu")(fused)
    posterior = tf.keras.layers.Dense(spec.num_classes, activation="softmax", name="class_posterior")(fused)

    classifier = tf.keras.Model([input_v, input_o], posterior, name="MHFL_MCA")
    auxiliary: Dict[str, tf.keras.Model] = {
        "weights": tf.keras.Model([input_v, input_o], modality_weights, name="MHFL_MCA_weights"),
        "features": tf.keras.Model([input_v, input_o], fused, name="MHFL_MCA_features"),
        "encoder_features": tf.keras.Model([input_v, input_o], [feat_v, feat_o], name="MHFL_MCA_encoder_features"),
        "cross_features": tf.keras.Model([input_v, input_o], [branch1, branch2], name="MHFL_MCA_cross_features"),
    }
    if score1 is not None and score2 is not None:
        auxiliary["scores"] = tf.keras.Model([input_v, input_o], [score1, score2], name="MHFL_MCA_scores")
    if attention_v_query_o is not None and attention_o_query_v is not None:
        auxiliary["attention"] = tf.keras.Model(
            [input_v, input_o],
            [attention_v_query_o, attention_o_query_v],
            name="MHFL_MCA_attention",
        )
    return classifier, auxiliary


def assert_model_count_matches_analytic(model: tf.keras.Model, spec: ModelSpec) -> None:
    actual = int(model.count_params())
    expected = int(analytical_parameter_count(spec))
    if actual != expected:
        raise RuntimeError(
            "TensorFlow model count {0} differs from analytical count {1}; check implementation/spec alignment.".format(
                actual, expected
            )
        )


def assert_expected_parameter_count(model: tf.keras.Model, spec: ModelSpec, tolerance_m: float = 0.02) -> None:
    assert_model_count_matches_analytic(model, spec)
    if spec.expected_params_m is None:
        return
    actual_m = float(model.count_params()) / 1_000_000.0
    if abs(actual_m - float(spec.expected_params_m)) > float(tolerance_m):
        raise RuntimeError(
            "Model parameter count {0:.6f} M differs from the declared manuscript value {1:.6f} M.".format(
                actual_m, float(spec.expected_params_m)
            )
        )


__all__ = [
    "ModelSpec",
    "CrossAttention",
    "build_models",
    "manuscript_spec",
    "branch_semantics",
    "assert_model_count_matches_analytic",
    "assert_expected_parameter_count",
]
