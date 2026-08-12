"""Experiment 12 - manuscript metric-definition audit.

This retained diagnostic explains a historical accuracy-definition finding. See the
release README before treating it as current evidence; the final equation was revised.
"""

from __future__ import annotations

import json
from pathlib import Path


def macro_ovr_accuracy_from_overall(overall_accuracy: float, classes: int) -> float:
    return (classes - 2.0 + 2.0 * overall_accuracy) / classes


def main() -> None:
    examples = []
    for classes in (5, 7):
        for overall in (0.60, 0.90, 0.99):
            examples.append(
                {
                    "classes": classes,
                    "sklearn_overall_accuracy": overall,
                    "macro_one_vs_rest_accuracy_from_current_equation": macro_ovr_accuracy_from_overall(overall, classes),
                }
            )
    payload = {
        "status": "ACTION_REQUIRED",
        "finding": "The executable scripts use sklearn.metrics.accuracy_score, while the manuscript defines the average of one-vs-rest class accuracies. For K>2 these metrics are not equal.",
        "recommended_fix": "Define Accuracy as total correct predictions divided by total samples, or regenerate every reported accuracy value using the manuscript's one-vs-rest formula. The first option matches the source code and conventional multiclass accuracy.",
        "examples": examples,
    }
    out = Path(__file__).resolve().parent / "provenance" / "manuscript_metric_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
