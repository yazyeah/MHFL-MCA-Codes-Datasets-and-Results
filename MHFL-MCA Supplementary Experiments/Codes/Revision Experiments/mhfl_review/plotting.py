from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .publication import (
    CATEGORICAL,
    PALETTE,
    FigureContract,
    add_panel_label,
    apply_publication_style,
    finalize_figure,
)


def _ordered_pivot(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    index_order: Sequence[str],
    column_order: Sequence[float],
) -> pd.DataFrame:
    pivot = frame.pivot(index=index, columns=columns, values=values)
    return pivot.reindex(index=list(index_order), columns=list(column_order))


def _heatmap(
    ax: object,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    value_format: str = ".2f",
) -> object:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, loc="left")
    normalizer = image.norm
    color_map = image.cmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                r, g, b, _ = color_map(normalizer(float(value)))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.48 else PALETTE["black"]
                ax.text(
                    j,
                    i,
                    format(float(value), value_format),
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=text_color,
                )
    ax.tick_params(length=0)
    return image


def plot_modality_weight_evidence(
    summary: pd.DataFrame,
    load: str,
    source_csv: Path,
    output_stem: Path,
) -> Dict[str, Path]:
    """Overview -> performance -> relationship evidence chain for R2-6/R4-4."""
    apply_publication_style()
    part = summary[summary["load"] == load].copy()
    scenarios = ["vibration_only_degraded", "current_only_degraded", "both_modalities_degraded"]
    display = ["Vibration degraded", "Current degraded", "Both degraded"]
    snrs = [0, -2, -4, -6, -8, -10]
    weights = _ordered_pivot(part, "scenario", "snr_db", "weight_vibration_mean", scenarios, snrs)
    accuracy = _ordered_pivot(part, "scenario", "snr_db", "accuracy_mean", scenarios, snrs)

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.25), gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]})
    image_a = _heatmap(
        axes[0],
        weights.values,
        display,
        [str(value) for value in snrs],
        "Vibration-source weight",
        "Blues",
        0.0,
        1.0,
    )
    image_b = _heatmap(
        axes[1],
        accuracy.values,
        display,
        [str(value) for value in snrs],
        "Diagnostic accuracy",
        "viridis",
        0.0,
        1.0,
    )
    axes[0].set_xlabel("SNR (dB)")
    axes[1].set_xlabel("SNR (dB)")
    fig.colorbar(image_a, ax=axes[0], fraction=0.045, pad=0.02)
    fig.colorbar(image_b, ax=axes[1], fraction=0.045, pad=0.02)

    relationship_rows = []
    for scenario, label, degraded_column, color, marker in (
        ("vibration_only_degraded", "Vibration degraded", "weight_vibration_mean", PALETTE["hero"], "o"),
        ("current_only_degraded", "Current degraded", "weight_current_mean", PALETTE["accent"], "s"),
    ):
        block = part[part["scenario"] == scenario].sort_values("snr_db", ascending=False)
        baseline = block[block["snr_db"] == 0].iloc[0]
        delta_weight = block[degraded_column].to_numpy() - float(baseline[degraded_column])
        delta_accuracy = block["accuracy_mean"].to_numpy() - float(baseline["accuracy_mean"])
        axes[2].plot(delta_weight, delta_accuracy, marker=marker, color=color, label=label)
        for snr, x_value, y_value in zip(block["snr_db"], delta_weight, delta_accuracy):
            axes[2].annotate(str(int(snr)), (x_value, y_value), xytext=(3, 2), textcoords="offset points", fontsize=5.5)
            relationship_rows.append(
                {
                    "load": load,
                    "scenario": scenario,
                    "snr_db": float(snr),
                    "delta_degraded_modality_weight": float(x_value),
                    "delta_accuracy": float(y_value),
                }
            )
    axes[2].axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.8)
    axes[2].axvline(0.0, color=PALETTE["neutral_light"], linewidth=0.8)
    axes[2].set_xlabel("Change in degraded-modality weight")
    axes[2].set_ylabel("Change in accuracy")
    axes[2].set_title("Weight–performance association", loc="left", fontweight="bold")
    axes[2].legend(loc="best")
    for label, ax in zip(("a", "b", "c"), axes):
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.87, wspace=0.62)

    relationship_csv = Path(output_stem).parent / (Path(output_stem).name + "_relationship.csv")
    pd.DataFrame(relationship_rows).to_csv(relationship_csv, index=False)
    contract = FigureContract(
        figure_id="modality-weight-evidence-{0}".format(load),
        core_conclusion=(
            "AMRM weights are evaluated as sample-wise reliability indicators across load and noise conditions; "
            "the figure reports association rather than claiming causal calibration."
        ),
        archetype="overview heatmap -> performance heatmap -> paired relationship",
        evidence_hierarchy=(
            "Panel a: learned vibration-source weight across controlled degradations",
            "Panel b: matched diagnostic accuracy",
            "Panel c: within-scenario changes relative to 0 dB",
        ),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=61.0,
        source_data=(str(source_csv), str(relationship_csv)),
        replicate_unit="independent model seeds, averaged over independent noise realizations",
        center_statistic="mean",
        spread_definition="between-model seed SD retained in source data; heatmaps display means",
        claim_limits=("Post-hoc interpretability analysis", "No causal claim from weight–accuracy association"),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv, relationship_csv])


def plot_missing_modality_stress(
    summary: pd.DataFrame,
    load: str,
    source_csv: Path,
    output_stem: Path,
) -> Dict[str, Path]:
    apply_publication_style()
    part = summary[summary["load"] == load].copy()
    order = [
        "vibration_missing",
        "current_missing",
        "vibration_severe",
        "current_severe",
        "both_severe",
    ]
    labels = ["Vibration missing", "Current missing", "Vibration −12 dB", "Current −12 dB", "Both −12 dB"]
    part = part.set_index("condition").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(3.50, 2.45))
    y = np.arange(len(part))
    ax.errorbar(
        part["accuracy_delta_mean"],
        y,
        xerr=part["accuracy_delta_sd"],
        fmt="o",
        color=PALETTE["hero"],
        ecolor=PALETTE["neutral_mid"],
        capsize=2.5,
    )
    ax.axvline(0.0, color=PALETTE["black"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change from paired 0 dB baseline")
    ax.set_title("Missing/degraded-modality stress test", loc="left", fontweight="bold")
    add_panel_label(ax, "a")
    fig.subplots_adjust(left=0.40, right=0.97, bottom=0.22, top=0.84)
    contract = FigureContract(
        figure_id="missing-modality-stress-{0}".format(load),
        core_conclusion="Quantify degradation under post-hoc modality removal or severe corruption.",
        archetype="paired-difference point-range plot",
        evidence_hierarchy=("Paired accuracy changes relative to the same model seed and baseline",),
        width_mm=config.SINGLE_COLUMN_MM,
        height_mm=64.0,
        source_data=(str(source_csv),),
        replicate_unit="independent trained model seeds",
        center_statistic="mean paired difference",
        spread_definition="one between-seed SD",
        claim_limits=("Stress test, not missing-modality-aware training",),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv])


def plot_ablation_evidence(
    summary: pd.DataFrame,
    source_csv: Path,
    output_stem: Path,
    raw_source_csv: Optional[Path] = None,
) -> Dict[str, Path]:
    apply_publication_style()
    variant_order = [
        "full",
        "homogeneous_vibration",
        "homogeneous_other",
        "attention_dim_128",
        "direct_softmax",
        "equal_weights",
    ]
    condition_order = ["2Nm_0dB", "4Nm_0dB", "4Nm_-8dB"]
    labels = ["Full", "Homogeneous-vib", "Homogeneous-other", "D=128", "Direct softmax", "Equal weights"]
    pivot = _ordered_pivot(summary, "variant", "condition", "accuracy_mean", variant_order, condition_order)
    delta = _ordered_pivot(summary, "variant", "condition", "delta_to_full_mean", variant_order[1:], condition_order)
    delta_sd = _ordered_pivot(summary, "variant", "condition", "delta_to_full_sd", variant_order[1:], condition_order)

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.55), gridspec_kw={"width_ratios": [1.10, 1.25]})
    image = _heatmap(
        axes[0],
        pivot.values,
        labels,
        ["2 Nm, 0 dB", "4 Nm, 0 dB", "4 Nm, −8 dB"],
        "Accuracy by controlled variant",
        "viridis",
        0.0,
        1.0,
    )
    fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.02)
    axes[0].tick_params(axis="x", rotation=20)
    for tick in axes[0].get_xticklabels():
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")

    y_base = np.arange(len(variant_order) - 1)
    offsets = [-0.20, 0.0, 0.20]
    for index, condition in enumerate(condition_order):
        axes[1].errorbar(
            delta[condition].values,
            y_base + offsets[index],
            xerr=delta_sd[condition].values,
            fmt="o",
            color=CATEGORICAL[index],
            label=["2 Nm, 0 dB", "4 Nm, 0 dB", "4 Nm, −8 dB"][index],
            capsize=2.0,
        )
    axes[1].axvline(0.0, color=PALETTE["black"], linewidth=0.8)
    axes[1].set_yticks(y_base)
    axes[1].set_yticklabels(labels[1:])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Accuracy difference versus full model")
    axes[1].set_title("Paired ablation effect", loc="left", fontweight="bold")
    axes[1].legend(loc="lower right")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.23, top=0.86, wspace=0.48)
    source_files = [Path(source_csv)]
    if raw_source_csv is not None:
        source_files.append(Path(raw_source_csv))
    contract = FigureContract(
        figure_id="additional-ablation",
        core_conclusion="Separate encoder heterogeneity, attention dimension, and modality-weighting design effects.",
        archetype="overview heatmap -> paired effect forest",
        evidence_hierarchy=("Panel a: absolute accuracy", "Panel b: paired change relative to the full model"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=69.0,
        source_data=tuple(str(path) for path in source_files),
        replicate_unit="identical persistent split per random seed",
        center_statistic="trimmed mean after removing one highest and one lowest result",
        spread_definition="population SD (ddof=0) across the eight retained model seeds",
        notes=(
            "Values are reported as the trimmed mean ± standard deviation over 10 independent training runs "
            "after removing the highest and lowest result.",
        ),
    )
    return finalize_figure(fig, output_stem, contract, source_files)


def plot_lowshot_evidence(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    summary_csv: Path,
    paired_csv: Path,
    output_stem: Path,
) -> Dict[str, Path]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35))
    for index, (variant, label) in enumerate((("full", "Full MHFL-MCA"), ("no_caim", "Without CAIM"))):
        block = summary[summary["variant"] == variant].sort_values("n_train")
        axes[0].errorbar(
            block["n_train"],
            block["test_accuracy_mean"],
            yerr=block["test_accuracy_sd"],
            marker="o" if index == 0 else "s",
            color=CATEGORICAL[index],
            capsize=2.0,
            label=label,
        )
        axes[1].errorbar(
            block["n_train"],
            block["generalization_gap_mean"],
            yerr=block["generalization_gap_sd"],
            marker="o" if index == 0 else "s",
            color=CATEGORICAL[index],
            capsize=2.0,
            label=label,
        )
    axes[0].set_xlabel("Training samples per class")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Extreme low-shot performance", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.8)
    axes[1].set_xlabel("Training samples per class")
    axes[1].set_ylabel("Train–validation gap")
    axes[1].set_title("Overfitting indicator", loc="left", fontweight="bold")

    paired = paired.sort_values("n_train")
    axes[2].errorbar(
        paired["n_train"],
        paired["caim_gain_mean"],
        yerr=paired["caim_gain_sd"],
        marker="o",
        color=PALETTE["positive"],
        capsize=2.0,
    )
    axes[2].axhline(0.0, color=PALETTE["black"], linewidth=0.8)
    axes[2].set_xlabel("Training samples per class")
    axes[2].set_ylabel("Paired CAIM accuracy gain")
    axes[2].set_title("CAIM sensitivity", loc="left")
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.98), ncol=2)
    for label, ax in zip(("a", "b", "c"), axes):
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.79, wspace=0.38)
    contract = FigureContract(
        figure_id="extreme-lowshot-threshold",
        core_conclusion="Characterize empirical instability and CAIM benefit below the original N=5 setting.",
        archetype="performance -> overfitting indicator -> paired component effect",
        evidence_hierarchy=("Test accuracy", "Train-validation gap", "Paired full-minus-noCAIM effect"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=63.0,
        source_data=(str(summary_csv), str(paired_csv)),
        replicate_unit="persistent disjoint split seeds",
        center_statistic="mean",
        spread_definition="one seed SD",
        claim_limits=("Empirical protocol-specific stability point, not a universal minimum-sample theorem",),
    )
    return finalize_figure(fig, output_stem, contract, [summary_csv, paired_csv])


def plot_traditional_baseline_evidence(
    summary: pd.DataFrame,
    source_csv: Path,
    output_stem: Path,
) -> Dict[str, Path]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.35))
    uo = summary[summary["case"] == "UO"]
    method_order = sorted(uo["method"].dropna().unique().tolist())
    method_color = {name: CATEGORICAL[index % len(CATEGORICAL)] for index, name in enumerate(method_order)}
    for index, method in enumerate(method_order):
        block = uo[uo["method"] == method]
        block = block.sort_values("n_train")
        axes[0].errorbar(
            block["n_train"], block["accuracy_mean"], yerr=block["accuracy_sd"],
            marker="o" if index == 0 else "s", capsize=2.0, color=method_color[method], label=method,
        )
    axes[0].set_xlabel("Training samples per class")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("UO limited-sample benchmark", loc="left", fontweight="bold")
    axes[0].legend(loc="best")

    kaist = summary[(summary["case"] == "KAIST") & (summary["snr_db"] == 0.0)]
    markers = {"2Nm": "o", "4Nm": "s"}
    for (method, load), block in kaist.groupby(["method", "load"], sort=True):
        block = block.sort_values("n_train")
        axes[1].errorbar(
            block["n_train"], block["accuracy_mean"], yerr=block["accuracy_sd"],
            marker=markers.get(str(load), "o"), capsize=2.0,
            color=method_color.get(method, PALETTE["neutral_dark"]),
            linestyle="-" if str(load) == "2Nm" else "--",
            label="{0}, {1}".format(method, load),
        )
    axes[1].set_xlabel("Training samples per class")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("KAIST cross-load benchmark", loc="left", fontweight="bold")
    axes[1].legend(loc="best", ncol=2)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.85, wspace=0.32)
    contract = FigureContract(
        figure_id="traditional-multimodal-baselines",
        core_conclusion="Add interpretable non-neural multimodal references without replacing the five deep baselines.",
        archetype="two-case benchmark comparison",
        evidence_hierarchy=("UO limited-sample", "KAIST cross-load"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=63.0,
        source_data=(str(source_csv),),
        replicate_unit="persistent split seeds",
        center_statistic="mean",
        spread_definition="one seed SD",
        claim_limits=("Not a reproduction of the reviewer-recommended optimized OMP/CSC algorithms",),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv])
