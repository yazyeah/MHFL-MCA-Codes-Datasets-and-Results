# Experiment 02: modality weights and missing modalities

Published source-data contracts:

- `modality_weight_raw.csv`: 540 rows;
- `modality_weight_summary.csv`: 36 rows;
- `missing_modality_raw.csv`: 132 rows;
- `missing_modality_summary.csv`: 12 rows;
- two relationship files: 12 rows each.

This is a post-hoc analysis of three independently trained Stage-3 models.
Attention-derived modality weights are not calibrated sensor-health
probabilities. The models were not trained with a missing-modality objective,
so zeroed-modality results are stress tests rather than claims of optimized
missing-sensor robustness. Checkpoint manifests are supplied in the sibling
checkpoint-metadata folder; weight files are not distributed here.
