#!/usr/bin/env python3
"""Build a protocol-aware extreme-low-shot figure and LaTeX table.

The displayed Full accuracy series is source-aware: N=5 and N=10 are taken
directly from the current manuscript Table 5, while N=1,2,3,4,7 are supplied
by the controlled sensitivity extension. Table 5 does not report train-
held-out gaps, a matched no-CAIM variant, or per-seed paired effects; panels
(b) and (c) therefore remain controlled-sensitivity-only quantities and are
never fabricated from aggregate Table 5 values.

Example
-------
python build_lowshot_protocol_aware_figure.py \
  --summary lowshot_summary.csv \
  --paired caim_paired_summary.csv \
  --output-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_GRID: Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 10)
WIDTH_MM = 183.0
HEIGHT_MM = 67.0
TABLE5_REFERENCE: Dict[int, Tuple[float, float]] = {
    5: (0.9279, 0.0288),
    10: (0.9707, 0.0175),
}


def build_display_full(full: pd.DataFrame) -> pd.DataFrame:
    """Return the single Full series shown in panel (a) and the paper table."""
    display = full.copy()
    for n_value, (mean_value, sd_value) in TABLE5_REFERENCE.items():
        if n_value not in display.index:
            raise ValueError(f"Table 5 anchor N={n_value} is outside the Full grid.")
        display.loc[n_value, "test_accuracy_mean"] = mean_value
        display.loc[n_value, "test_accuracy_sd"] = sd_value
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a protocol-aware low-shot sensitivity figure/table."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem", default="lowshot_sensitivity_protocol_aware",
        help="Output filename stem.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def select_variant(summary: pd.DataFrame, variant: str) -> pd.DataFrame:
    block = summary.loc[summary["variant"] == variant].copy()
    block["n_train"] = pd.to_numeric(block["n_train"], errors="raise").astype(int)
    block = block.set_index("n_train").reindex(N_GRID)
    if block.isna().any().any():
        missing_n = block.index[block.isna().any(axis=1)].tolist()
        raise ValueError(f"Incomplete {variant!r} summary at N={missing_n}")
    return block


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.0,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", width=0.8, length=3)
    axis.set_xticks(N_GRID)


def save_table(
    output_path: Path,
    display_full: pd.DataFrame,
    no_caim: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    paired_indexed = paired.set_index("n_train").reindex(N_GRID)
    gain_mean_col = "paired_gain_mean" if "paired_gain_mean" in paired_indexed else "caim_gain_mean"
    gain_sd_col = "paired_gain_sd" if "paired_gain_sd" in paired_indexed else "caim_gain_sd"

    lines = [
        r"\begin{table}[pos=htbp]",
        r"\centering",
        r"\caption{Protocol-aware extreme low-shot sensitivity on the UO dataset. In the displayed Full series, the $N=5$ and $N=10$ accuracy values are taken directly from Table~\ref{tab:case1_compare_main}; the remaining Full values and all without-CAIM and paired-gain values are from the controlled sensitivity extension. Sensitivity-study values are trimmed mean $\pm$ population standard deviation over 10 runs after independently removing the highest and lowest value.}",
        r"\label{tab:extreme_lowshot_caim}",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}cccc@{}}",
        r"\toprule",
        r"$N$ & \shortstack{Sensitivity\\Full MHFL-MCA} & \shortstack{Sensitivity\\without CAIM} & \shortstack{Paired CAIM\\gain} \\",
        r"\midrule",
    ]

    for n_value in N_GRID:
        full_text = f"{display_full.loc[n_value, 'test_accuracy_mean']:.4f}$\\pm${display_full.loc[n_value, 'test_accuracy_sd']:.4f}"
        no_text = f"{no_caim.loc[n_value, 'test_accuracy_mean']:.4f}$\\pm${no_caim.loc[n_value, 'test_accuracy_sd']:.4f}"
        gain_text = f"{paired_indexed.loc[n_value, gain_mean_col]:.4f}$\\pm${paired_indexed.loc[n_value, gain_sd_col]:.4f}"
        lines.append(f"{n_value} & {full_text} & {no_text} & {gain_text} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\vspace{1mm}",
            r"\raggedright\footnotesize Full accuracy at $N=5$ and $N=10$ is reproduced directly from Table~\ref{tab:case1_compare_main} within the single displayed Full series. Table~\ref{tab:case1_compare_main} contains neither train--held-out gaps nor matched no-CAIM per-seed results; those derived quantities remain from the controlled sensitivity extension and are not calculated from historical aggregate means.\par",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    paired = pd.read_csv(args.paired)
    require_columns(
        summary,
        (
            "variant",
            "n_train",
            "test_accuracy_mean",
            "test_accuracy_sd",
            "generalization_gap_mean",
            "generalization_gap_sd",
        ),
        "summary",
    )
    require_columns(paired, ("n_train",), "paired summary")

    full = select_variant(summary, "full")
    display_full = build_display_full(full)
    no_caim = select_variant(summary, "no_caim")
    paired["n_train"] = pd.to_numeric(paired["n_train"], errors="raise").astype(int)
    paired = paired.set_index("n_train").reindex(N_GRID).reset_index()
    if paired.isna().any().any():
        raise ValueError("Paired CAIM summary is incomplete.")

    gain_mean_col = "paired_gain_mean" if "paired_gain_mean" in paired else "caim_gain_mean"
    gain_sd_col = "paired_gain_sd" if "paired_gain_sd" in paired else "caim_gain_sd"
    if gain_mean_col not in paired or gain_sd_col not in paired:
        raise ValueError("Paired summary has no recognized gain mean/SD columns.")

    configure_fonts()
    width_in = WIDTH_MM / 25.4
    height_in = HEIGHT_MM / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.22, top=0.79, wspace=0.42)

    x = np.asarray(N_GRID, dtype=float)
    blue = "#2B6AA3"
    orange = "#D67A1F"
    green = "#2F8A4C"
    dark_gray = "#333333"

    # Panel (a): one source-aware Full series. N=5/10 come directly from Table 5.
    ax = axes[0]
    ax.errorbar(
        x,
        display_full["test_accuracy_mean"].to_numpy(float),
        yerr=display_full["test_accuracy_sd"].to_numpy(float),
        marker="o",
        markersize=4.6,
        linewidth=1.6,
        capsize=2.5,
        color=blue,
        label="Sensitivity Full",
    )
    ax.errorbar(
        x,
        no_caim["test_accuracy_mean"].to_numpy(float),
        yerr=no_caim["test_accuracy_sd"].to_numpy(float),
        marker="s",
        markersize=4.2,
        linewidth=1.6,
        capsize=2.5,
        color=orange,
        label="Without CAIM",
    )
    ax.set_ylabel("Held-out accuracy")
    ax.set_xlabel("Training samples per class")
    ax.set_title("Extreme low-shot performance", fontweight="bold", pad=5)
    ax.set_ylim(0.35, 1.02)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    style_axis(ax)

    # Panel (b): only the internally matched sensitivity protocol.
    ax = axes[1]
    ax.errorbar(
        x,
        full["generalization_gap_mean"].to_numpy(float),
        yerr=full["generalization_gap_sd"].to_numpy(float),
        marker="o",
        markersize=4.6,
        linewidth=1.6,
        capsize=2.5,
        color=blue,
    )
    ax.errorbar(
        x,
        no_caim["generalization_gap_mean"].to_numpy(float),
        yerr=no_caim["generalization_gap_sd"].to_numpy(float),
        marker="s",
        markersize=4.2,
        linewidth=1.6,
        capsize=2.5,
        color=orange,
    )
    ax.axhline(0.0, linewidth=0.8, color=dark_gray)
    ax.set_ylabel("Train-held-out gap")
    ax.set_xlabel("Training samples per class")
    ax.set_title("Overfitting indicator", fontweight="bold", pad=5)
    ax.set_ylim(-0.02, 0.63)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    style_axis(ax)

    # Panel (c): paired within-run effect; historical Table 5 has no matched no-CAIM data.
    ax = axes[2]
    ax.errorbar(
        x,
        paired[gain_mean_col].to_numpy(float),
        yerr=paired[gain_sd_col].to_numpy(float),
        marker="o",
        markersize=4.6,
        linewidth=1.6,
        capsize=2.5,
        color=green,
    )
    ax.axhline(0.0, linewidth=0.8, color=dark_gray)
    ax.set_ylabel("Paired CAIM accuracy gain")
    ax.set_xlabel("Training samples per class")
    ax.set_title("CAIM sensitivity", fontweight="bold", pad=5)
    upper = max(0.22, float(np.max(paired[gain_mean_col] + paired[gain_sd_col])) + 0.02)
    ax.set_ylim(-0.02, upper)
    style_axis(ax)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.18,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.4,
        handletextpad=0.45,
    )

    stem = args.output_dir / args.stem
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)

    merged_rows = []
    paired_indexed = paired.set_index("n_train")
    for n_value in N_GRID:
        merged_rows.append(
            {
                "n_train": n_value,
                "sensitivity_full_mean": display_full.loc[n_value, "test_accuracy_mean"],
                "sensitivity_full_sd": display_full.loc[n_value, "test_accuracy_sd"],
                "sensitivity_full_source": (
                    "main_manuscript_table_5"
                    if n_value in TABLE5_REFERENCE
                    else "controlled_sensitivity_extension"
                ),
                "sensitivity_no_caim_mean": no_caim.loc[n_value, "test_accuracy_mean"],
                "sensitivity_no_caim_sd": no_caim.loc[n_value, "test_accuracy_sd"],
                "sensitivity_no_caim_source": "controlled_sensitivity_extension",
                "full_generalization_gap_mean": full.loc[n_value, "generalization_gap_mean"],
                "full_generalization_gap_sd": full.loc[n_value, "generalization_gap_sd"],
                "gap_source": "controlled_sensitivity_extension",
                "paired_caim_gain_mean": paired_indexed.loc[n_value, gain_mean_col],
                "paired_caim_gain_sd": paired_indexed.loc[n_value, gain_sd_col],
                "paired_gain_source": "controlled_sensitivity_extension",
            }
        )
    pd.DataFrame(merged_rows).to_csv(stem.with_name(stem.name + "_data.csv"), index=False)
    save_table(stem.with_name(stem.name + "_table.tex"), display_full, no_caim, paired)

    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(f"Saved: {stem.with_suffix('.svg')}")
    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.tiff')}")
    print(f"Saved: {stem.with_name(stem.name + '_data.csv')}")
    print(f"Saved: {stem.with_name(stem.name + '_table.tex')}")


if __name__ == "__main__":
    main()
