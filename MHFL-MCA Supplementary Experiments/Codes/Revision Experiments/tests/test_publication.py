from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mhfl_review.publication import FigureContract, apply_publication_style, finalize_figure


def test_figure_bundle(tmp_path: Path):
    apply_publication_style()
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1, 2], "y": [2, 3]}).to_csv(source, index=False)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [2, 3], label="Data")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend()
    contract = FigureContract(
        figure_id="test",
        core_conclusion="test",
        archetype="line",
        evidence_hierarchy=("line",),
        width_mm=89.0,
        height_mm=55.0,
        source_data=(str(source),),
        replicate_unit="test",
        center_statistic="none",
        spread_definition="none",
    )
    result = finalize_figure(fig, tmp_path / "figure", contract, [source])
    for key in ("svg", "pdf", "tiff", "png", "qa", "contract", "source_index"):
        assert result[key].is_file()
