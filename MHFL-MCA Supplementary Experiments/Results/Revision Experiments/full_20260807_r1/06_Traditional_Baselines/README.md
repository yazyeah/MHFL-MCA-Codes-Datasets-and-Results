# Experiment 06: frozen-protocol TF-SVM baselines

The UO and KAIST TF-SVM baselines use one dedicated tuning point (N=15,
seed=20260805) that is disjoint from evaluation seeds 20260806--20260815.
Early fusion and both unimodal SVMs used for late fusion select C/gamma once;
the selected values are then frozen across all reported N and seeds.

- UO selection uses shuffled stratified CV within tuning-split training data;
  held-out samples are not used for selection.
- KAIST selection uses only the 0 Nm source train/validation split; 2 Nm and
  4 Nm target data are not used for selection.
- Late-fusion probability columns are aligned explicitly by class label.
- UO overlap evidence supports segment-disjoint, not recording-disjoint,
  wording when source recordings are shared.

Contracts: 480 raw rows, 48 summary groups, 10 seeds per group, eight retained
values per metric-wise trimmed summary. `manuscript_candidate_rows.csv`
combines 108 manuscript aggregates with 36 clean TF-SVM rows; it excludes
KAIST-noise rows.

The deep-reference portion is bound to the recorded 2026-08-10 LaTeX source.
It must be revalidated before claiming identity with a later final manuscript.
TF-SVM entries are hand-crafted time-frequency feature benchmarks, not exact
reproductions of unrelated CSC/GJO-OMP methods.
