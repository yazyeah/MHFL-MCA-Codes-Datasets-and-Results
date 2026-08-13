# Model Checkpoints

## Distribution status

Checkpoint manifests are versioned in this repository. The corresponding
HDF5 weight files are **not** distributed in Git and no public Release or
Zenodo download URL has been published. A manifest entry is not a download
link.

The byte sizes below were checked against the private local artifacts during
the repository-hardening audit. The public manifests provide protocol and
SHA-256 identity contracts, not downloadable weight payloads.

| Protocol | Seed | Epochs | Batch | Learning rate | Expected filename | Bytes | SHA-256 | Public status |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| KAIST Stage 2 | 20260806 | 80 | 16 | 0.0004156294449523281 | `kaist_stage2_full_seed20260806.weights.h5` | 29,597,464 | `e5a0fc8a3b8fc07a79300a65599bcc074dd0a56123b22b9153c68f58862e266e` | Manifest only |
| KAIST Stage 3 | 20260806 | 70 | 16 | 0.0004156294449523281 | `kaist_stage3_full_seed20260806.weights.h5` | 29,597,464 | `4e3a52c741623f36dddc31cdc82b1163206eba72c7f24611f5da687d32f41118` | Manifest only |
| KAIST Stage 3 | 20260807 | 70 | 16 | 0.0004156294449523281 | `kaist_stage3_full_seed20260807.weights.h5` | 29,597,464 | `9b1ce866e1c56f3bf55b252f6c23f41497c534c16a98f6fe7472194aea9eea60` | Manifest only |
| KAIST Stage 3 | 20260808 | 70 | 16 | 0.0004156294449523281 | `kaist_stage3_full_seed20260808.weights.h5` | 29,597,464 | `608ec0dcd1f3fc825cafc037d9395231996334f83931d928babf882c8cc0b2a6` | Manifest only |

All four manifests use model-spec fingerprint
`8294192c1b5ff1833b10232ab82adb6dac8dfcae2c7b21d816eddb11f07ef08d`
and data signature
`5f9da3db107a327dfd692a95e74f44267e2e0b82ab9e68e04456efc8fa4f93a1`.
Each uses base SNR 0 dB and gradient clipping norm 1.0.

Manifest directory:

`MHFL-MCA Supplementary Experiments/Results/Revision Experiments/full_20260807_r1/01_Checkpoint_Metadata/`

## Verify a separately supplied checkpoint

Place a separately supplied file under
`MHFL-MCA Supplementary Experiments/Codes/Revision Experiments/checkpoints/<run-tag>/`
with its exact filename. Verify it before loading:

```powershell
Get-FileHash -Algorithm SHA256 `
  '.\checkpoints\<run-tag>\kaist_stage2_full_seed20260806.weights.h5'
```

The digest must exactly match the corresponding manifest row above. If it
does not, do not load the file.

## Rebuild instead of download

From `Codes/Revision Experiments`, after the strict data/channel preflight:

```powershell
python 01_prepare_kaist_checkpoint.py --mode full --protocol both --accept-kaist-spec
```

This command performs training. It does not guarantee a historical checkpoint
will be reproduced byte-for-byte when unrecorded runtime and initializer state
differs. Use the manifests to identify the intended protocol and to validate
any later released artifact.

No UO, additional-ablation, optimizer-state, SavedModel or split-cache
checkpoint is distributed in this repository package.
