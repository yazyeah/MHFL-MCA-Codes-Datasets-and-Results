"""Experiment 09 - analytical model-specification audit.

Purpose: verify UO/KAIST architecture fields and analytical parameter counts.
Output: a machine-readable audit; parameter-count agreement is not Optuna evidence.
"""

from __future__ import annotations

import json

from mhfl_review import config
from mhfl_review.provenance import write_json
from mhfl_review.specs import candidate_architectures, manuscript_spec, parameter_count_m, spec_fingerprint


def main() -> None:
    rows = []
    for case in ("uo", "kaist"):
        spec = manuscript_spec(case)
        row = {
            "case": case,
            "analytical_params": int(round(parameter_count_m(spec) * 1_000_000)),
            "analytical_params_m": parameter_count_m(spec),
            "expected_params_m": spec.expected_params_m,
            "absolute_delta_m": None if spec.expected_params_m is None else abs(parameter_count_m(spec) - spec.expected_params_m),
            "spec_status": spec.spec_status,
            "source_note": spec.source_note,
            "fingerprint": spec_fingerprint(spec),
            "spec": spec.to_dict(),
        }
        rows.append(row)
        print(
            "{0}: analytical={1:.6f} M, expected={2}, status={3}".format(
                case.upper(), row["analytical_params_m"], row["expected_params_m"], row["spec_status"]
            )
        )
    candidates = candidate_architectures(7.380, tolerance_m=0.02)
    print("\nKAIST candidates within 0.02 M of manuscript 7.380 M:")
    for row in candidates:
        print(json.dumps(row, ensure_ascii=False))
    payload = {
        "suite_version": config.SUITE_VERSION,
        "specs": rows,
        "kaist_candidates_near_7_380M": candidates,
        "acceptance_instruction": (
            "Do not set MHFL_ACCEPT_KAIST_SPEC=1 until the original Optuna best-trial JSON or model.summary output "
            "confirms the selected depths/dropouts/attention dimension. Parameter-count agreement alone is not proof."
        ),
    }
    out = config.PROVENANCE_ROOT / "model_spec_audit.json"
    write_json(out, payload)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
