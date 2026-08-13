# Data Download, Layout and Integrity

The datasets in this repository are large Git LFS objects. Read
[DATA_LICENSES.md](DATA_LICENSES.md) before downloading or redistributing them.
The repository's software license does not override the dataset licenses.

## Included data inventory

The inventory audited on 2026-08-13 is:

```text
Datasets/
├── UO/
│   ├── 1_CSV_Raw_Data/       60 CSV files
│   ├── 2_Excel_Raw_Data/     60 XLSX files
│   └── 3_MatLab_Raw_Data/    60 MAT files
└── KAIST/
    ├── vibration/            15 MAT files
    └── current/              15 TDMS files
```

The UO tree provides multiple representations of the same 60 recordings. The
KAIST tree is a selected and reorganized subset of the upstream v6 dataset,
covering five health states at 0, 2 and 4 Nm. See
[DATA_LICENSES.md](DATA_LICENSES.md) for the precise attribution and version
qualification.

### Download-size planning

The audited Git LFS inventory on 2026-08-13 was:

| Scope | Files | Bytes | GiB |
| --- | ---: | ---: | ---: |
| All datasets | 210 | 4,227,597,643 | 3.937 |
| KAIST current TDMS | 15 | 1,143,793,466 | 1.065 |
| KAIST vibration MAT | 15 | 1,032,367,112 | 0.961 |
| UO CSV | 60 | 825,113,234 | 0.768 |
| UO Excel | 60 | 923,760,257 | 0.860 |
| UO MATLAB used by revision scripts | 60 | 302,563,574 | 0.282 |
| Entire repository LFS payload, including historical results | 22,946 | 9,511,070,035 | 8.858 |

These sizes describe the audited commit and may change in later releases.
Selective downloads avoid resolving the unrelated historical-result objects.

## Clone and resolve Git LFS data

Install [Git LFS](https://git-lfs.com/) before cloning. Suppress automatic LFS
checkout so cloning does not first download the complete historical payload:

```powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/yazyeah/MHFL-MCA-Codes-Datasets-and-Results.git
Set-Location .\MHFL-MCA-Codes-Datasets-and-Results
Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
git lfs pull --include="Datasets/**" --exclude=""
```

For an existing clone:

```powershell
git lfs install
git lfs fetch --include="Datasets/**" --exclude=""
git lfs pull --include="Datasets/**" --exclude=""
```

Do not run an experiment against unresolved pointer files. A Git LFS pointer
is a short text file beginning with
`version https://git-lfs.github.com/spec/v1`; it is not sensor data.

## Repository integrity checks

Run the following after `git lfs pull`:

```powershell
git lfs ls-files --long
git lfs fsck
git lfs status
```

`git lfs ls-files --long` reports the SHA-256 object ID recorded by the Git
commit. `git lfs fsck` verifies available local LFS objects against their
recorded IDs. `git lfs status` should not report unintended data changes.

To check that no pointer stubs remain under `Datasets/`:

```powershell
Get-ChildItem -LiteralPath .\Datasets -Recurse -File |
    Where-Object { $_.Length -le 1024 } |
    Where-Object {
        $header = [System.IO.File]::ReadAllBytes($_.FullName)
        $prefix = [System.Text.Encoding]::ASCII.GetString($header)
        $prefix.StartsWith('version https://git-lfs.github.com/spec/v1')
    } |
    Select-Object FullName, Length
```

Git LFS pointer stubs are under 1 KiB, so the size filter avoids opening large
resolved binary files. The command should produce no rows. For a resolved file, a direct content hash
can be recorded with, for example:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\Datasets\KAIST\vibration\0Nm_Normal.mat
```

These checks prove consistency with the checked-out Git commit. They do **not**
by themselves prove that a file matches a particular upstream Mendeley
version. Upstream-version verification requires comparison with authoritative
metadata from that exact release.

## Official-source fallback

If LFS transfer is unavailable, obtain data only from the official sources:

- UO dataset family:
  [https://data.mendeley.com/datasets/y2px5tg92h](https://data.mendeley.com/datasets/y2px5tg92h).
  The exact UO version represented by this repository remains **UNCONFIRMED**;
  do not silently mix releases. Record the selected version DOI and file
  hashes before use.
- KAIST v6:
  [https://data.mendeley.com/datasets/ztmf3m7h5x/6](https://data.mendeley.com/datasets/ztmf3m7h5x/6),
  DOI [10.17632/ztmf3m7h5x.6](https://doi.org/10.17632/ztmf3m7h5x.6).

Reconstruct the repository layout shown above without overwriting or mixing
files from another version. Do not commit replacement data until its source,
license, version and hashes have been documented.

## Experiment data roots

The original UO scripts expect their data-root placeholder to point directly
to:

```text
Datasets\UO\3_MatLab_Raw_Data
```

The supplementary revision suite uses environment variables. A local setup
can point them to the resolved repository paths:

```powershell
$env:MHFL_UO_DATA_ROOT = '<repository>\Datasets\UO\3_MatLab_Raw_Data'
$env:MHFL_KAIST_VIB_DIR = '<repository>\Datasets\KAIST\vibration'
$env:MHFL_KAIST_CURRENT_DIR = '<repository>\Datasets\KAIST\current'
$env:MHFL_CURRENT_CHANNEL_NAME = 'cDAQ9185-1F486B5Mod2/ai0'
$env:MHFL_KAIST_VIB_COLUMN = '0'
$env:MHFL_UO_VIB_COLUMN = '0'
$env:MHFL_UO_ACOUSTIC_COLUMN = '1'
```

For KAIST, the explicit channel configuration selects:

- TDMS group `Log`, channel `cDAQ9185-1F486B5Mod2/ai0`: U-phase motor
  current; and
- `Signal.y_values.values[:, 0]`: bearing housing A, x-direction vibration.

Do not enable a maximum-variance, first-numeric-channel or fuzzy-name fallback
as a substitute for these confirmed physical channels. The supplementary
suite's strict preflight should pass before a full experiment is started.

## Reporting data provenance

For a reproducible result, record at minimum:

1. repository commit and Git LFS object IDs;
2. upstream dataset DOI and exact version (or `UNCONFIRMED` for the current UO
   files);
3. local SHA-256 values for all consumed files;
4. selected variables, columns and TDMS group/channel names; and
5. split, seed and preprocessing manifests produced by the experiment.

Do not present a repository-level LFS integrity check as an upstream-version
confirmation.
