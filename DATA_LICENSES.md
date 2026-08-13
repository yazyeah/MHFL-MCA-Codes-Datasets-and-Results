# Dataset Licensing and Attribution

This document covers the third-party datasets stored under `Datasets/`. It was
last reviewed on **2026-08-13**.

> **License boundary:** the repository's MIT software license
> does **not** apply to `Datasets/**`. The upstream dataset licenses and
> attribution requirements continue to govern those files. Nothing in this
> repository relicenses third-party data.

## Summary

| Repository path | Upstream dataset | Upstream version represented | License | Local treatment |
| --- | --- | --- | --- | --- |
| `Datasets/UO/` | University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions (UORED-VAFCLS) | **UNCONFIRMED**. The local files have not been byte-matched to a specific Mendeley version. | The identified public versions are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). | 60 recordings are present in each of CSV, Excel and MATLAB representations and are stored with Git LFS. |
| `Datasets/KAIST/` | Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis | [Mendeley Data v6](https://doi.org/10.17632/ztmf3m7h5x.6) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | A selected and reorganized subset: 15 vibration MAT files and 15 TDMS files covering five health states at 0, 2 and 4 Nm. |

## University of Ottawa data

### Required attribution

- Dataset contributors listed by Mendeley Data: Mert Sehri and Patrick
  Dumond.
- Dataset family: [University of Ottawa Rolling-element Dataset – Vibration
  and Acoustic Faults under Constant Load and Speed conditions](https://data.mendeley.com/datasets/y2px5tg92h)
- Data article: Mert Sehri, Patrick Dumond and Michel Bouchard, “University of
  Ottawa constant load and speed rolling-element bearing vibration and
  acoustic fault signature datasets,” *Data in Brief*, 49 (2023), 109327.
  [https://doi.org/10.1016/j.dib.2023.109327](https://doi.org/10.1016/j.dib.2023.109327)
- Dataset DOI cited by that article: [Mendeley Data
  v2](https://doi.org/10.17632/y2px5tg92h.2)
- A later public release also exists as [Mendeley Data
  v5](https://doi.org/10.17632/y2px5tg92h.5).

### Version status

The repository contains the expected 60 recording names in three
representations, but it does not contain an upstream download receipt,
version manifest or upstream checksum list. File names alone cannot establish
whether these Git LFS objects came from v2, v5 or another public release.
Therefore the exact UO source version is **UNCONFIRMED**.

Before citing a specific version, a maintainer must compare the repository's
resolved file SHA-256 values (or Git LFS object IDs) with authoritative
metadata for that exact Mendeley release and record the comparison. Until
then, cite the dataset family and data article, retain this qualification, and
do not label the files as v2 or v5.

The identified v2 and v5 releases are published under CC BY 4.0. If a later
exact-version audit identifies a different release, its official license page
must be checked again before redistribution.

## KAIST data

### Required attribution

- Dataset: Wonho Jung, Seong-Hu Kim, SungHyun Yun, Jaewoong Bae and Yong-Hwa
  Park, “Vibration, Acoustic, Temperature, and Motor Current Dataset of
  Rotating Machine Under Varying Load Conditions for Fault Diagnosis,”
  [Mendeley Data v6](https://data.mendeley.com/datasets/ztmf3m7h5x/6),
  [https://doi.org/10.17632/ztmf3m7h5x.6](https://doi.org/10.17632/ztmf3m7h5x.6).
- Data article: Wonho Jung et al., “Vibration, acoustic, temperature, and motor
  current dataset of rotating machine under varying operating conditions for
  fault diagnosis,” *Data in Brief*, 48 (2023), 109049.
  [https://doi.org/10.1016/j.dib.2023.109049](https://doi.org/10.1016/j.dib.2023.109049)

The upstream v6 dataset is licensed under CC BY 4.0. The files included here
are not a representation of every upstream modality or operating condition.
They are a selected and reorganized subset containing:

- five health states: Normal, BPFI 0.3 mm, BPFI 1.0 mm, BPFO 0.3 mm and BPFO
  1.0 mm;
- three loads: 0, 2 and 4 Nm;
- MATLAB vibration files separated into `Datasets/KAIST/vibration/`; and
- TDMS acquisition files separated into `Datasets/KAIST/current/`.

The TDMS files contain more than one measured field; the `current` directory
name describes their use in this project, not the complete upstream TDMS
schema. For the reported experiments, the suite explicitly selects TDMS group
`Log`, channel `cDAQ9185-1F486B5Mod2/ai0` (U-phase motor current). From
`Signal.y_values.values` in the vibration MAT files, zero-based column `0` is
selected (bearing housing A, x direction). These are analysis choices made by
this project, not changes to the physical meaning assigned by the upstream
dataset authors.

The repository declares Mendeley Data v6 as the upstream source for this
KAIST subset. Git LFS integrity proves consistency with this Git commit; it is
not, by itself, an authoritative upstream byte-for-byte checksum comparison.

## Changes and derived processing

Relative to the upstream distributions, this repository may:

- select only the files and operating conditions listed above;
- reorganize files into modality-specific directories;
- select the named current channel and vibration column at load time; and
- segment, normalize or split signals at runtime as described by the
  experiment code and provenance records.

The raw files are not claimed as original work of this repository. Any
derived artifacts remain subject to applicable upstream data-license terms.
No upstream author or institution endorses this repository.

## CC BY 4.0 obligations

CC BY 4.0 permits sharing and adaptation, including commercial use, provided
users give appropriate credit, link to the license and indicate whether
changes were made. Users may not add legal or technological restrictions that
prevent others from exercising the licensed rights. See the official
[CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/) and linked legal
code for the controlling terms.

The datasets are provided without warranty by their respective upstream
providers. Consult [DATA.md](DATA.md) for download and integrity procedures and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the compact redistribution
notice.
