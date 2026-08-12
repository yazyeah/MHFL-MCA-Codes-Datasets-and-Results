from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config
from .publication import CATEGORICAL, PALETTE, FigureContract, add_panel_label, apply_publication_style, finalize_figure


METRIC_LABELS = {
    "accuracy": "Macro-accuracy",
    "precision": "Macro-precision",
    "recall": "Macro-recall",
    "f1": "Macro-F1",
}


def _method_style(methods: Sequence[str]) -> Dict[str, Dict[str, object]]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    styles: Dict[str, Dict[str, object]] = {}
    baseline_colors = [PALETTE["accent"], PALETTE["positive"], PALETTE["violet"], PALETTE["secondary"], PALETTE["negative"], PALETTE["neutral_mid"]]
    baseline_index = 0
    for index, method in enumerate(methods):
        is_hero = "MHFL" in str(method).upper() or "PROPOSED" in str(method).upper()
        color = PALETTE["hero"] if is_hero else baseline_colors[baseline_index % len(baseline_colors)]
        if not is_hero:
            baseline_index += 1
        styles[str(method)] = {
            "color": color,
            "marker": markers[index % len(markers)],
            "linewidth": 1.8 if "MHFL" in str(method).upper() or "PROPOSED" in str(method).upper() else 1.0,
            "zorder": 5 if "MHFL" in str(method).upper() or "PROPOSED" in str(method).upper() else 2,
        }
    return styles


def plot_performance_grid(frame: pd.DataFrame, source_csv: Path, output_stem: Path, case_label: str) -> Dict[str, Path]:
    required = {"method", "n_train", "metric", "mean", "sd"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Performance source data missing columns: {0}".format(sorted(missing)))
    apply_publication_style()
    metrics = [value for value in ("accuracy", "f1", "precision", "recall") if value in set(frame["metric"])]
    if len(metrics) != 4:
        raise ValueError("Performance grid requires accuracy, f1, precision and recall rows.")
    methods = frame["method"].drop_duplicates().astype(str).tolist()
    styles = _method_style(methods)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, metric, panel in zip(axes_flat, metrics, ("a", "b", "c", "d")):
        part = frame[frame["metric"] == metric]
        for method in methods:
            block = part[part["method"].astype(str) == method].sort_values("n_train")
            style = styles[method]
            ax.errorbar(
                block["n_train"], block["mean"], yerr=block["sd"], label=method,
                color=style["color"], marker=style["marker"], linewidth=style["linewidth"],
                zorder=style["zorder"], capsize=1.7, markerfacecolor="white", markeredgewidth=0.8,
            )
        ax.set_title(METRIC_LABELS[metric], loc="left")
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        add_panel_label(ax, panel)
    axes[1, 0].set_xlabel("Training samples per class")
    axes[1, 1].set_xlabel("Training samples per class")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles = [Line2D([0], [0], color=styles[m]["color"], marker=styles[m]["marker"], linewidth=styles[m]["linewidth"], markerfacecolor="white", label=m) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=min(4, len(methods)))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.84, wspace=0.12, hspace=0.30)
    contract = FigureContract(
        figure_id="performance-grid-{0}".format(case_label.lower().replace(" ", "-")),
        core_conclusion="Compare diagnostic performance across training sizes without repeated panel legends.",
        archetype="2x2 aligned repeated-measures performance grid",
        evidence_hierarchy=("Accuracy and F1", "Precision and recall"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=112.0,
        source_data=(str(source_csv),),
        replicate_unit="independent run/seed",
        center_statistic="reported mean or trimmed mean",
        spread_definition="reported standard deviation; definition must be stated in caption",
        notes=("Shared legend and aligned axes replace repeated legends in the original figure.",),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv])


def plot_confusion_grid(frame: pd.DataFrame, source_csv: Path, output_stem: Path, normalize: bool = True) -> Dict[str, Path]:
    required = {"method", "true_label", "pred_label", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Confusion source data missing columns: {0}".format(sorted(missing)))
    apply_publication_style()
    methods = frame["method"].drop_duplicates().astype(str).tolist()
    labels = sorted(set(frame["true_label"]).union(set(frame["pred_label"])))
    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    image = None
    for index, method in enumerate(methods):
        ax = axes.ravel()[index]
        block = frame[frame["method"].astype(str) == method]
        matrix = block.pivot_table(index="true_label", columns="pred_label", values="count", aggfunc="sum", fill_value=0).reindex(index=labels, columns=labels, fill_value=0).to_numpy(dtype=float)
        if normalize:
            denominator = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None, interpolation="nearest")
        ax.set_title(str(method))
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        if index % ncols == 0:
            ax.set_ylabel("True")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                color = "white" if value > 0.55 else PALETTE["black"]
                text = "{0:.2f}".format(value) if normalize else str(int(value))
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color=color)
        add_panel_label(ax, chr(ord("a") + index), x=-0.18)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.025)
        cbar.set_label("Row-normalized proportion" if normalize else "Count")
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.10, top=0.93, wspace=0.30, hspace=0.36)
    contract = FigureContract(
        figure_id="confusion-grid",
        core_conclusion="Compare class-level error structure on a common normalized scale.",
        archetype="small-multiple normalized confusion matrices",
        evidence_hierarchy=("Diagonal concentration", "Shared off-diagonal error patterns"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=118.0,
        source_data=(str(source_csv),),
        replicate_unit="one predeclared representative run or aggregated out-of-sample predictions",
        center_statistic="row-normalized counts",
        spread_definition="not applicable",
        claim_limits=("Representative-run figures must be labeled as such and use a predeclared selection rule.",),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv])


def plot_efficiency_two_panel(frame: pd.DataFrame, source_csv: Path, output_stem: Path) -> Dict[str, Path]:
    required = {"model", "params_m", "train_time_s", "accuracy_2nm", "accuracy_4nm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Efficiency source data missing columns: {0}".format(sorted(missing)))
    apply_publication_style()
    methods = frame["model"].astype(str).tolist()
    styles = _method_style(methods)
    fig, axes = plt.subplots(1, 2)
    max_params = max(float(frame["params_m"].max()), 1e-6)
    max_time = max(float(frame["train_time_s"].max()), 1e-6)
    axes[0].set_xlim(-0.03 * max_params, 1.18 * max_params)
    axes[1].set_xlim(-0.03 * max_time, 1.18 * max_time)
    for _, row in frame.iterrows():
        method = str(row["model"])
        st = styles[method]
        avg_acc = 0.5 * (float(row["accuracy_2nm"]) + float(row["accuracy_4nm"]))
        axes[0].scatter(row["params_m"], avg_acc, color=st["color"], marker=st["marker"], s=30, zorder=st["zorder"])
        axes[1].scatter(row["train_time_s"], avg_acc, color=st["color"], marker=st["marker"], s=30, zorder=st["zorder"])
        offset0 = (-3, 3) if float(row["params_m"]) > 0.86 * max_params else (3, 3)
        offset1 = (-3, 3) if float(row["train_time_s"]) > 0.86 * max_time else (3, 3)
        ha0 = "right" if offset0[0] < 0 else "left"
        ha1 = "right" if offset1[0] < 0 else "left"
        axes[0].annotate(method, (row["params_m"], avg_acc), xytext=offset0, textcoords="offset points", fontsize=5.8, ha=ha0)
        axes[1].annotate(method, (row["train_time_s"], avg_acc), xytext=offset1, textcoords="offset points", fontsize=5.8, ha=ha1)
    axes[0].set_xlabel("Trainable parameters (M)")
    axes[1].set_xlabel("Training time per run (s)")
    axes[0].set_ylabel("Mean target-load accuracy")
    axes[0].set_title("Model size trade-off", loc="left")
    axes[1].set_title("Training-cost trade-off", loc="left")
    for label, ax in zip(("a", "b"), axes):
        ax.grid(color="#E6E6E6", linewidth=0.5)
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.88, wspace=0.30)
    contract = FigureContract(
        figure_id="efficiency-tradeoff",
        core_conclusion="Summarize accuracy against parameter scale and training cost without redundant bubble encodings.",
        archetype="two aligned scatter trade-off panels",
        evidence_hierarchy=("Parameter efficiency", "Training-time efficiency"),
        width_mm=config.DOUBLE_COLUMN_MM,
        height_mm=67.0,
        source_data=(str(source_csv),),
        replicate_unit="same hardware/software protocol for every model",
        center_statistic="mean accuracy and mean training time",
        spread_definition="uncertainty should be supplied in the table/source data",
        claim_limits=("Cross-model latency or memory claims require profiling every baseline with the same profiler.",),
    )
    return finalize_figure(fig, output_stem, contract, [source_csv])
