# Experiment 05: extreme low-shot CAIM sensitivity

## Controlled extension

The controlled extension is complete at the data level: 140 raw rows, 14
summary rows, seven paired-gain rows, no duplicate identities, no non-finite
metrics, and 70 matched Full/no-CAIM split pairs. Its Full N=5 and N=10
aggregates do **not** reproduce both historical Table-5 values to four
decimals. The copied anchor and post-gate files therefore remain `FAIL`, and
the run must not be described as an exact stochastic replay.

## Paper protocol-aware bundle

The paper-facing display uses one Full series with explicit mixed provenance:

- Full accuracy at N=5 and N=10: current-manuscript Table 5;
- Full accuracy at N=1,2,3,4,7: controlled extension;
- without-CAIM accuracy, train-held-out gaps, and paired CAIM gains at every N:
  controlled extension.

Table 5 contains neither train-held-out gaps nor matched no-CAIM per-seed
results. Those quantities are never inferred by subtracting historical
aggregate means. See `paper_protocol_aware_bundle/hybrid_derivation_manifest.json`.

`operational_thresholds.json` is intentionally not distributed because its
stored operational points conflict with the current manuscript conclusion and
require manuscript-level resolution.
