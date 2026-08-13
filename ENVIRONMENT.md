# Reproducibility Environment

This repository separates versions recorded by experiment artifacts from
versions observed later on a validation machine. Later observations do not
retroactively prove the original training environment.

## Evidence levels

- **Recorded experiment environment**: serialized by the 2026-08-07
  checkpoint and efficiency-profile manifests.
- **Repository pin**: declared by the repository but not serialized as a
  separate field in the historical manifest.
- **Current validation environment**: observed on 2026-08-13 while auditing
  this repository.
- **Not recorded**: must not be inferred.

## Recorded experiment environment

| Component | Recorded value |
| --- | --- |
| Python | 3.9.20, MSC v.1929, 64-bit AMD64 |
| TensorFlow | 2.7.0 |
| NumPy | 1.21.0 |
| pandas | 1.4.4 |
| SciPy | 1.8.1 |
| scikit-learn | 1.1.3 |
| Matplotlib | 3.5.3 |
| nptdms | 1.10.0 |
| Platform string | `Windows-10-10.0.26200-SP0` |
| Source commit recorded by the run | `da879af0acc2fcafda1023600123223b3925cb69` |

Keras 2.7.0 is a repository pin and was observed on the validation host; the
historical environment JSON did not serialize Keras separately. The Python
3.9 platform string labels build 26200 as Windows 10; the current Windows API
identifies that build as Windows 11. The historical string is retained
verbatim rather than rewritten.

Primary evidence:

- `MHFL-MCA Supplementary Experiments/Results/Revision Experiments/full_20260807_r1/03_Efficiency_Profile/environment_manifest.json`
  (SHA-256 `80f3f2db3b011b1967950c7ed43ff68a7e0a1e4da46713effe6ac923b3d910e0`);
- the four checkpoint manifests under
  `MHFL-MCA Supplementary Experiments/Results/Revision Experiments/full_20260807_r1/01_Checkpoint_Metadata/`.

## Validation-only package observations

The second block in [`requirements-lock.txt`](requirements-lock.txt) records
versions observed during the 2026-08-13 repository validation. The file pins
the experiment-facing packages and critical TensorFlow 2.7 compatibility
dependencies used by CI. It is a Windows/Python 3.9 validation constraint
set, not a claim that every historical transitive package was serialized by
the formal runs. In particular, the historical Optuna package version was not
recorded.

The audited workstation had user-site package contamination. A clean
environment should therefore set `PYTHONNOUSERSITE=1`; the lock file, not the
global or user site, is the installation contract.

## Installation

From an Anaconda or Miniconda PowerShell prompt:

```powershell
conda env create -f environment.yml
conda activate mhfl-mca-revision
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m pip check
```

The Conda file does not install the NVIDIA driver, CUDA driver runtime or
cuDNN. Those layers and their evidence limits are documented in
[`HARDWARE.md`](HARDWARE.md).

## Verification without training

```powershell
python tools\verify_repository_contract.py
python ".\MHFL-MCA Supplementary Experiments\Codes\Revision Experiments\tools\validate_publication_package.py" `
  --package-root ".\MHFL-MCA Supplementary Experiments"
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before running any full
experiment.
