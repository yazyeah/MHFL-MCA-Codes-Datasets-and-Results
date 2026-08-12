"""Ordered, fail-fast command runner for the revision experiment suite.

Use explicit phases for audit, smoke, full experiments, figures, or assets. The runner
does not change protocol constants and stops immediately when a child command fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent


def run(command: List[str]) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordered, fail-fast reviewer-experiment pipeline.")
    parser.add_argument("phase", choices=("audit", "smoke", "full", "figures", "assets"))
    parser.add_argument("--accept-kaist-spec", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    accept = ["--accept-kaist-spec"] if args.accept_kaist_spec else []

    if args.phase == "audit":
        run([py, "tools/static_audit.py", "."])
        run([py, "09_model_spec_audit.py"])
        run([py, "11_source_alignment_audit.py"])
        run([py, "12_manuscript_metric_audit.py"])
        run([py, "-m", "pytest", "-q"])
    elif args.phase == "smoke":
        run([py, "00_check_environment.py"])
        run([py, "01_prepare_kaist_checkpoint.py", "--mode", "fast", "--protocol", "both"] + accept)
        run([py, "02_modality_weights_and_missing.py", "--mode", "fast"] + accept)
        run([py, "03_profile_efficiency.py", "--mode", "fast", "--device", "auto", "--warmup", "10", "--repeats", "50"] + accept)
        run([py, "04_additional_ablation.py", "--mode", "fast"] + accept)
        run([py, "05_lowshot_threshold.py", "--mode", "fast"])
        run([py, "06_traditional_baselines.py", "--mode", "fast"])
        run([py, "10_synthetic_figure_smoke.py"])
    elif args.phase == "full":
        run([py, "00_check_environment.py", "--strict"])
        run([py, "01_prepare_kaist_checkpoint.py", "--mode", "full", "--protocol", "both"] + accept)
        run([py, "02_modality_weights_and_missing.py", "--mode", "full"] + accept)
        run([py, "03_profile_efficiency.py", "--mode", "full", "--device", "gpu", "--warmup", "100", "--repeats", "1000"] + accept)
        run([py, "04_additional_ablation.py", "--mode", "full"] + accept)
        run([py, "05_lowshot_threshold.py", "--mode", "full"])
        run([py, "06_traditional_baselines.py", "--mode", "full"])
        run([py, "07_build_manuscript_assets.py"])
    elif args.phase == "figures":
        run([py, "10_synthetic_figure_smoke.py"])
    else:
        run([py, "07_build_manuscript_assets.py"])


if __name__ == "__main__":
    main()
