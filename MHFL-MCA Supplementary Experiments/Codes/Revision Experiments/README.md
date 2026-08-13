# MHFL-MCA supplementary experiments

This directory contains the supplementary experiment entry points in the same
experiment-oriented style as the public MHFL-MCA repository, while preserving
the tested modular implementation in `mhfl_review`.

Each top-level script has a module header describing its purpose, protocol,
inputs, outputs, and reproducibility boundary. Configuration constants are
centralized in `mhfl_review/config.py`; training and model logic are not
duplicated across scripts.

## Environment

The recorded environment used TensorFlow/Keras 2.7.0 on Windows. Install the
project requirements in an isolated environment, copy the example environment
file, and replace only its `YOUR_*` placeholders:

```bat
copy windows\00_set_environment.example.bat windows\00_set_environment.bat
call windows\00_set_environment.bat
python 00_check_environment.py --strict
```

Important dataset mappings:

- UO: vibration column 0 and acoustic column 1;
- KAIST vibration: `Signal.y_values.values`, zero-based column 0 (bearing
  housing A, x direction);
- KAIST current: TDMS group `Log`, exact U-phase channel
  `cDAQ9185-1F486B5Mod2/ai0`;
- current-channel fallback is disabled for full runs.

## Supported full reruns

After data configuration and preflight, the formal experiment commands are:

```bat
python 01_prepare_kaist_checkpoint.py --mode full --protocol both --accept-kaist-spec
python 02_modality_weights_and_missing.py --mode full --protocol stage3 --accept-kaist-spec
python 03_profile_efficiency.py --mode full --protocol stage2 --device gpu --warmup 100 --repeats 1000 --accept-kaist-spec
python 04_additional_ablation.py --mode full --accept-kaist-spec
python tools/prepare_ablation_main_without_equal_weights.py --suite-root . --run-tag %MHFL_RUN_TAG%
python 06_traditional_baselines.py --mode full
```

Do not run Optuna from these entry points. Full profiling requires an existing
Stage-2 checkpoint and uses isolated CPU and GPU subprocesses. Experiment 06
tunes TF-SVM hyperparameters once at N=15 with seed 20260805, then freezes
them across all N values and evaluation seeds.

Experiment 05 is a controlled sensitivity extension and must be interpreted
separately:

```bat
python 05_lowshot_threshold.py --mode full --run-tag %MHFL_RUN_TAG%
```

It writes the 140-row raw, 14-group summary, and seven-row paired-gain source
data. Because the original TensorFlow initializer and batch-shuffling state
was not retained, the extension is not represented as a byte-for-byte
stochastic replay. The protocol-aware plot can be reconstructed from the
published controlled-extension CSV files with:

```bat
python figures\build_lowshot_protocol_aware_figure.py ^
  --summary "..\..\Results\Revision Experiments\full_20260807_r1\05_Extreme_Lowshot_CAIM\controlled_extension\lowshot_summary.csv" ^
  --paired "..\..\Results\Revision Experiments\full_20260807_r1\05_Extreme_Lowshot_CAIM\controlled_extension\caim_paired_summary.csv" ^
  --output-dir "..\..\Results\Revision Experiments\full_20260807_r1\05_Extreme_Lowshot_CAIM\paper_protocol_aware_bundle" ^
  --stem lowshot_sensitivity_protocol_aware_reproduced
```

The `07_build_manuscript_assets.py` script is retained as an internal strict
gate for its recorded artifact contract. Public reproduction uses the
source-specific commands above rather than `run_pipeline.py full`.

## Result contracts

| Experiment | Full-run contract |
|---|---|
| 02 | weights raw 540 / summary 36; missing raw 132 / summary 12 |
| 03 | one profile; both worker exit codes 0; warm-up 100; repeats 1000 |
| 04 | 5 variants x 3 conditions x 10 seeds = 150 raw rows; 15 groups |
| 05 | 7 N x 2 variants x 10 runs = 140 raw rows; 14 groups; 7 paired rows; historical-anchor status reported separately |
| 06 | 480 raw rows; 48 groups; 144 clean manuscript-candidate rows |

The common full-run aggregation is
`trimmed_mean_sd_drop_one_high_one_low`. Accuracy, F1, precision, and recall
are trimmed independently. Paired CAIM effects are computed within seed before
trimming.

## Evidence boundaries

- `configs/kaist_optuna_confirmed.json` and
  `configs/uo_optuna_confirmed.json` bind the recovered configurations to raw
  historical evidence under `provenance/original_optuna`.
- The additional-ablation Full row is a reused aggregate from the original
  Stage-2 main experiment; it is not seed-paired with the new controls.
- The low-shot protocol-aware plot uses the declared main reference for Full
  accuracy at N=5 and N=10. Gaps and paired gains remain controlled-extension
  quantities.
- The deep-reference JSON is bound to its recorded, byte-identical LaTeX
  source snapshot.

## Tests and audits

No-cache validation commands:

```bat
set PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider
python tools\static_audit.py .
python 09_model_spec_audit.py
```

The code does not silently train inside the efficiency profiler, silently fall
back to a current channel, or modify the confirmed Optuna files.
