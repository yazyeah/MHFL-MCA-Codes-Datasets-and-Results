## Supplementary experiments and source data

The `Codes/Revision Experiments` and `Results/Revision Experiments` folders
contain the modular supplementary experiments, source data, audit gates,
and figure-generation utilities used for the revised MHFL-MCA manuscript.

Key reproducibility contracts are:

- modality weights: 540 raw rows and 36 summary rows;
- missing modalities: 132 raw rows and 12 summary rows;
- additional KAIST ablations: 150 raw rows (5 variants x 3 conditions x 10 seeds);
- UO extreme low-shot extension: 140 raw rows (7 N values x 2 variants x 10 runs);
- TF-SVM baselines: 480 raw rows and 48 summary groups;
- all 10-run summaries use metric-wise trimming of one high and one low value,
  retaining eight values for the reported population standard deviation.

Raw datasets and large checkpoints are not redistributed. Configure their
locations through the environment template in
`Codes/Revision Experiments/windows/00_set_environment.example.bat`.

Please read `README_UPLOAD.md` and the per-experiment README files before
using manuscript-derived tables: some aggregates are reused references rather
than seed-matched comparisons, and the low-shot paper display has an explicit
hybrid source policy.
