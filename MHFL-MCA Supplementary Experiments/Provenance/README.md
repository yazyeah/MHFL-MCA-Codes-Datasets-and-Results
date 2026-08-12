# Provenance policy

This folder records the audit artifacts used to assemble the public package.
All paths in publication metadata are repository-relative or `${MHFL_*}`
placeholders.

Byte-identical historical evidence is stored separately under
`Codes/Revision Experiments/provenance/original_optuna`. Those files are not
path-sanitized because their SHA-256 values are part of the evidence chain.
The current-main-manuscript reference is tied to the portable
`MHFL-MCA_20260810.tex` snapshot and 14 referenced summary CSV files.

`pre_sanitization_sha256.csv` records the hashes of copied artifacts before
machine-specific paths were replaced. `PUBLICATION_MANIFEST.csv` and
`SHA256SUMS.txt` at the package root record the distributed bytes.

The package does not claim that the recorded 2026-08-10 LaTeX snapshot is
identical to a later 2026-08-12 manuscript PDF. A fresh table-source audit is
required before making that claim.
