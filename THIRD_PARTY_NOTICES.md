# Third-Party Notices

This notice was reviewed on **2026-08-13**. It records the principal
third-party datasets distributed through Git LFS in this repository. The
respective upstream authors retain their rights.

## University of Ottawa rolling-element bearing data

**Material:** vibration and acoustic recordings under constant load and speed,
stored under `Datasets/UO/` in CSV, Excel and MATLAB representations.

**Dataset contributors:** Mert Sehri and Patrick Dumond.

**Article attribution:** Mert Sehri, Patrick Dumond and Michel Bouchard, “University of
Ottawa constant load and speed rolling-element bearing vibration and acoustic
fault signature datasets,” *Data in Brief*, 49 (2023), 109327,
[https://doi.org/10.1016/j.dib.2023.109327](https://doi.org/10.1016/j.dib.2023.109327).
Dataset family:
[https://data.mendeley.com/datasets/y2px5tg92h](https://data.mendeley.com/datasets/y2px5tg92h).

**License:** the identified public releases v2 and v5 are licensed under
[Creative Commons Attribution 4.0 International (CC BY
4.0)](https://creativecommons.org/licenses/by/4.0/).

**Version qualification:** the exact public release represented by the local
Git LFS objects has not been byte-matched to authoritative release metadata and
is therefore **UNCONFIRMED**. The data article cites
[v2](https://doi.org/10.17632/y2px5tg92h.2); a later
[v5](https://doi.org/10.17632/y2px5tg92h.5) also exists. This repository does
not currently claim either version for its local files.

**Repository changes:** the 60 recordings are organized into three format
directories. Experiment code selects vibration and acoustic columns and
performs segmentation, normalization and splitting at runtime.

## KAIST rotating-machine data

**Material:** a selected and reorganized subset stored under
`Datasets/KAIST/`, comprising 15 vibration MAT files and 15 TDMS acquisition
files for five health states at 0, 2 and 4 Nm.

**Attribution:** Wonho Jung, Seong-Hu Kim, SungHyun Yun, Jaewoong Bae and
Yong-Hwa Park, “Vibration, Acoustic, Temperature, and Motor Current Dataset of
Rotating Machine Under Varying Load Conditions for Fault Diagnosis,” Mendeley
Data v6,
[https://doi.org/10.17632/ztmf3m7h5x.6](https://doi.org/10.17632/ztmf3m7h5x.6).
Related data article: Wonho Jung et al., *Data in Brief*, 48 (2023), 109049,
[https://doi.org/10.1016/j.dib.2023.109049](https://doi.org/10.1016/j.dib.2023.109049).

**License:**
[Creative Commons Attribution 4.0 International (CC BY
4.0)](https://creativecommons.org/licenses/by/4.0/).

**Repository changes:** files are selected by health state and load and
reorganized into vibration and TDMS directories. The analysis selects TDMS
group `Log`, channel `cDAQ9185-1F486B5Mod2/ai0` (U-phase motor current), and
zero-based vibration column `0` from `Signal.y_values.values` (bearing housing
A, x direction). The upstream TDMS files contain additional fields; the local
directory name `current` does not imply that those fields were absent from the
source.

## License boundary and disclaimer

The repository's MIT software license does **not** cover or
relicense `Datasets/**`. Dataset users must preserve attribution, link to the
applicable license, identify modifications and comply with the upstream terms.
See [DATA_LICENSES.md](DATA_LICENSES.md) for the full attribution and version
status and [DATA.md](DATA.md) for download and integrity instructions.

Python and other software dependencies named by the project remain under the
licenses supplied by their respective authors and distributions. Dependency
declarations are not a grant of rights beyond those upstream licenses.

Third-party materials are provided without warranty, and inclusion does not
imply endorsement by the dataset authors, their institutions or the software
dependency authors.
