"""Experiment 10 - synthetic figure-generation smoke test.

Purpose: exercise plotting/export code with synthetic arrays only.
This script does not load the research datasets and is not a manuscript experiment.
"""

from __future__ import annotations

"""Synthetic-only smoke test for rendering/QA infrastructure.

Outputs from this script are marked DEMO and must never be used in the manuscript.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mhfl_review import config
from mhfl_review.diagnostic_plots import plot_confusion_grid, plot_efficiency_two_panel, plot_performance_grid


def main() -> None:
    out = config.OUTPUT_ROOT / "10_synthetic_figure_smoke_DEMO"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)

    methods = ["MRCFN", "CFFN", "CDTFAFN", "MSF-DFormer", "KDCNN-DF", "Proposed (MHFL-MCA)"]
    perf_rows = []
    for method_index, method in enumerate(methods):
        for n_train in (5, 10, 15, 20, 25, 30):
            for metric_index, metric in enumerate(("accuracy", "f1", "precision", "recall")):
                base = 0.52 + 0.07 * method_index + 0.012 * n_train - 0.015 * metric_index
                if "MHFL" in method:
                    base += 0.10
                perf_rows.append({"method": method, "n_train": n_train, "metric": metric, "mean": min(base, 0.995), "sd": 0.015 + 0.003 * rng.rand()})
    perf = pd.DataFrame(perf_rows)
    perf_path = out / "DEMO_performance.csv"; perf.to_csv(perf_path, index=False)
    plot_performance_grid(perf, perf_path, out / "DEMO_performance", "DEMO")

    cm_rows = []
    for method_index, method in enumerate(methods):
        matrix = np.eye(5) * (60 + method_index * 5) + rng.randint(0, 12, size=(5, 5))
        for true in range(5):
            for pred in range(5):
                cm_rows.append({"method": method, "true_label": true, "pred_label": pred, "count": int(matrix[true, pred])})
    cm = pd.DataFrame(cm_rows)
    cm_path = out / "DEMO_confusion.csv"; cm.to_csv(cm_path, index=False)
    plot_confusion_grid(cm, cm_path, out / "DEMO_confusion")

    efficiency = pd.DataFrame(
        {
            "model": methods,
            "params_m": [0.44, 1.00, 10.60, 1.35, 0.22, 7.38],
            "train_time_s": [88.4, 46.6, 355.4, 199.9, 5.1, 28.2],
            "accuracy_2nm": [0.80, 0.79, 0.88, 0.99, 0.86, 1.00],
            "accuracy_4nm": [0.73, 0.73, 0.81, 0.96, 0.79, 0.99],
        }
    )
    efficiency_path = out / "DEMO_efficiency.csv"; efficiency.to_csv(efficiency_path, index=False)
    plot_efficiency_two_panel(efficiency, efficiency_path, out / "DEMO_efficiency")
    (out / "DO_NOT_USE_IN_MANUSCRIPT.txt").write_text("Synthetic rendering smoke test only.\n", encoding="utf-8")
    print("Synthetic figure bundles:", out)


if __name__ == "__main__":
    main()
