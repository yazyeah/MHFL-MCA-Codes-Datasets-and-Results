"""Experiment 11 - source-to-implementation alignment audit.

Purpose: compare retained source snapshots with the executable manuscript model spec.
Output: explicit pass/action records; this script does not train a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from mhfl_review import config
from mhfl_review.specs import manuscript_spec, parameter_count_m


def contains(path: Path, token: str) -> bool:
    return path.is_file() and token in path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    uo_source = Path(__file__).resolve().parent / "source_snapshots" / "MHCNN_UO.py"
    kaist_source = Path(__file__).resolve().parent / "source_snapshots" / "MHCNN_KAIST.py"
    rows: List[Dict[str, object]] = []
    checks = [
        ("UO unscaled attention", uo_source, "K.softmax(attention_scores, axis=-1)"),
        ("KAIST unscaled attention", kaist_source, "K.softmax(attention_scores, axis=-1)"),
        ("UO query-order flattening", uo_source, "flat_v = Flatten(name=\"flat_v\")(feat_v_att)"),
        ("KAIST value-source flattening", kaist_source, "flat_v = Flatten()(v_att)"),
        ("KAIST per-segment z-score policy", kaist_source, "ENABLE_ZSCORE_PER_SEGMENT"),
    ]
    for name, path, token in checks:
        rows.append({"check": name, "source": str(path), "pass": contains(path, token), "token": token})
    rows.extend(
        [
            {"check": "UO analytical params", "pass": abs(parameter_count_m(manuscript_spec("uo")) - 5.111759) < 1e-6},
            {"check": "KAIST spec explicitly accepted", "pass": bool(config.ACCEPT_KAIST_SPEC), "note": "Required only for full KAIST runs."},
        ]
    )
    out = config.PROVENANCE_ROOT / "source_alignment_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Source alignment audit:", out)
    for row in rows:
        print("[{0}] {1}".format("PASS" if row.get("pass") else "ACTION", row["check"]))


if __name__ == "__main__":
    main()
