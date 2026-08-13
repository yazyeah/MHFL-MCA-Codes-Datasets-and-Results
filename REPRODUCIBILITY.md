# Reproducibility Guide

This guide distinguishes source-data verification from commands that perform
training. The supplementary package provides source data, configurations,
provenance records and explicit experiment contracts; it does not claim that
unrecorded historical random streams can be replayed byte-for-byte.

## 1. Start with a lightweight clone

The repository contains about 8.86 GiB of Git LFS objects. Clone without
automatically resolving all of them:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results.git
Set-Location .\MHFL-MCA-Codes-Datasets-and-Results
Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
```

Resolve only the data needed for a selected case. See [`DATA.md`](DATA.md) for
sizes, upstream licenses and integrity checks.

```powershell
# KAIST only (both modalities)
git lfs pull --include="Datasets/KAIST/**" --exclude=""

# UO MATLAB representation only
git lfs pull --include="Datasets/UO/3_MatLab_Raw_Data/**" --exclude=""

# Published supplementary figures only
git lfs pull --include="MHFL-MCA Supplementary Experiments/**" --exclude=""
```

## 2. Create the recorded environment

```powershell
conda env create -f environment.yml
conda activate mhfl-mca-revision
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pip check
```

The core version evidence and later validation-only pins are separated in
[`ENVIRONMENT.md`](ENVIRONMENT.md). Hardware evidence and resource limits are
in [`HARDWARE.md`](HARDWARE.md).

## 3. Validate the repository without training

From the repository root:

```powershell
python tools\verify_repository_contract.py
python ".\MHFL-MCA Supplementary Experiments\Codes\Revision Experiments\tools\validate_publication_package.py" `
  --package-root ".\MHFL-MCA Supplementary Experiments"
```

The package validator checks source-data row contracts, uniqueness, finite
metrics, manuscript-source hashes and explicit PASS/FAIL boundaries. It does
not load the raw datasets or run training.

## 4. Configure the revision suite

```powershell
Set-Location ".\MHFL-MCA Supplementary Experiments\Codes\Revision Experiments"

$runDate = Get-Date -Format 'yyyyMMdd'
$env:MHFL_RUN_TAG = "full_${runDate}_r1"
$env:MHFL_SUITE_ROOT = (Get-Location).Path
$env:MHFL_PROJECT_ROOT = (Resolve-Path '..\..\..').Path
$env:MHFL_UO_DATA_ROOT = (Resolve-Path '..\..\..\Datasets\UO\3_MatLab_Raw_Data').Path
$env:MHFL_KAIST_VIB_DIR = (Resolve-Path '..\..\..\Datasets\KAIST\vibration').Path
$env:MHFL_KAIST_CURRENT_DIR = (Resolve-Path '..\..\..\Datasets\KAIST\current').Path

$env:MHFL_TEMP_ROOT = 'D:\temp\mhfl-mca'
$env:TEMP = $env:MHFL_TEMP_ROOT
$env:TMP = $env:MHFL_TEMP_ROOT
$env:TMPDIR = $env:MHFL_TEMP_ROOT

$env:MHFL_CURRENT_CHANNEL_NAME = 'cDAQ9185-1F486B5Mod2/ai0'
$env:MHFL_KAIST_VIB_COLUMN = '0'
Remove-Item Env:MHFL_ALLOW_CURRENT_FALLBACK -ErrorAction SilentlyContinue
Remove-Item Env:MHFL_CURRENT_CHANNEL_REGEX -ErrorAction SilentlyContinue
```

Inspect the confirmed Optuna and channel provenance before explicitly
accepting the KAIST specification:

```powershell
$env:MHFL_ACCEPT_KAIST_SPEC = '1'
python 00_check_environment.py --strict
python 09_model_spec_audit.py
```

## 5. Full commands that perform training or model evaluation

These commands are intentionally separate so that each experiment contract is
invoked and audited independently.

```powershell
python 01_prepare_kaist_checkpoint.py --mode full --protocol both --accept-kaist-spec
python 02_modality_weights_and_missing.py --mode full --protocol stage3 --accept-kaist-spec
python 03_profile_efficiency.py --mode full --protocol stage2 --device gpu --warmup 100 --repeats 1000 --accept-kaist-spec
python 04_additional_ablation.py --mode full --accept-kaist-spec
python tools/prepare_ablation_main_without_equal_weights.py --suite-root . --run-tag $env:MHFL_RUN_TAG
python 06_traditional_baselines.py --mode full
```

Checkpoint distribution and verification are defined in
[`MODEL_ZOO.md`](MODEL_ZOO.md). No command silently downloads or substitutes a
checkpoint.

## 6. Experiment 05 boundary

```powershell
python 05_lowshot_threshold.py --mode full --run-tag $env:MHFL_RUN_TAG
```

The controlled extension produces 140 raw rows, 14 summary rows and seven
paired-gain rows. Because the original TensorFlow initializer and
batch-shuffling stream was not recorded, this extension is not presented as a
byte-for-byte stochastic replay.

The protocol-aware display uses a disclosed hybrid source mapping: Full accuracy at
N=5 and N=10 comes from Table 5; the other Full points, all no-CAIM values,
train-held-out gaps and paired gains come from the controlled extension. The
row-level provenance is retained in the published CSV and manifest.

## 7. Result contracts

| Experiment | Published contract |
| --- | --- |
| 02 | Modality weights: 540 raw / 36 summary; missing modality: 132 raw / 12 summary |
| 03 | One isolated CPU-FLOPs/GPU-runtime profile; 100 warm-up and 1000 timed runs |
| 04 | Five variants x three conditions x ten seeds = 150 raw rows and 15 summary groups |
| 05 | Seven N values x two variants x ten runs = 140 raw rows, 14 summaries and seven paired rows |
| 06 | 480 raw rows, 48 summary groups and 144 clean candidate rows |

The declared source-data values and byte identities are summarized in
[`ARTIFACT_VALUE_VERIFICATION.md`](MHFL-MCA%20Supplementary%20Experiments/Provenance/audits/ARTIFACT_VALUE_VERIFICATION.md).
