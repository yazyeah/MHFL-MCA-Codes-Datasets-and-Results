# MHFL-MCA Supplementary Experiments

This directory is a curated, portable publication copy for the revision
experiments associated with MHFL-MCA. It is deliberately isolated from the
working reviewer suite and can be reviewed or staged as one explicit Git
directory: `MHFL-MCA Supplementary Experiments`.

## What this package contains

- Runnable, modular experiment entry points under `Codes/Revision Experiments`.
- Confirmed UO and KAIST configurations and immutable historical Optuna
  evidence.
- Compact CSV/JSON/TEX/PDF/PNG/SVG results for Experiments 02--06.
- A paper-specific four-row Table 15 export and a provenance-explicit hybrid
  low-shot bundle.
- The current revised-manuscript workflow figure under `figures/`, bound to
  the source image by SHA-256.
- SHA-256 and publication manifests generated after validation.

## What this package excludes

- Raw UO or KAIST data.
- Model weights, SavedModels, caches, temporary files, and split caches.
- Patch directories, pre-fix scripts, smoke outputs, and stale Experiment 07
  assets.
- TIFF files and other large binary duplicates.

Checkpoint manifests and model-provenance records are retained, but their
referenced weight files are intentionally not distributed here. Large weights
should be released separately (for example, through GitHub Releases or a
research-data repository) and matched by SHA-256.

## Result status

| Scope | Public status |
|---|---|
| Experiment 02 | Source data and figures included; post-hoc analysis of three Stage-3 models. |
| Experiment 03 | Complete isolated CPU-FLOPs/GPU-runtime profile included. |
| Experiment 04 | Complete 5-variant source evidence included; paper Table 15 is a separate 3-control display. |
| Experiment 05 | Controlled extension included transparently with its failed Table-5 replay gate; the paper figure/table use an explicitly documented hybrid mapping. |
| Experiment 06 | Fixed-tuning TF-SVM experiment included; manuscript reference is bound to the recorded 2026-08-10 LaTeX snapshot. |
| Experiment 07 | Stale aggregate assets intentionally excluded; rebuild only after final manuscript gates are refreshed. |

This directory is therefore a **publication evidence package**, not a claim
that every final-manuscript gate currently passes.

## Reproduction guide

Start with [`Codes/Revision Experiments/README.md`](Codes/Revision%20Experiments/README.md)
for environment variables, exact channel mappings, formal commands, result
contracts, and the Experiment-05/07 evidence boundaries. The repository-level
README provides the concise end-to-end command sequence.

## Safe Git staging

From the repository root, stage only this directory:

```powershell
git status --short -- 'MHFL-MCA Supplementary Experiments'
git add -- 'MHFL-MCA Supplementary Experiments'
git diff --cached --check
git diff --cached --stat -- 'MHFL-MCA Supplementary Experiments'
git diff --cached --name-only -- 'MHFL-MCA Supplementary Experiments'
```

Do not use `git add .` or force-add the original `outputs/` tree. Before any
commit or push, verify that the configured remote is the intended public
repository.

## Validation

`PACKAGE_VALIDATION.json`, `PUBLICATION_MANIFEST.csv`, and `SHA256SUMS.txt`
are the machine-readable package checks. Paths in copied metadata use
`${MHFL_*}` placeholders; raw historical evidence is kept byte-identical and
is explicitly identified in the provenance README.
