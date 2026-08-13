# Experiment 05: extreme low-shot CAIM sensitivity

## Controlled extension

The controlled extension is complete at the data level: 140 raw rows, 14
summary rows, seven paired-gain rows, no duplicate identities, no non-finite
metrics, and 70 matched Full/no-CAIM split pairs. Because the historical
TensorFlow random stream was not recorded, the run is documented as a
controlled extension rather than an exact stochastic replay.

## Paper protocol-aware bundle

The paper-facing display uses one Full series with explicit mixed provenance:

- Full accuracy at N=5 and N=10: declared main reference values;
- Full accuracy at N=1,2,3,4,7: controlled extension;
- without-CAIM accuracy, train-held-out gaps, and paired CAIM gains at every N:
  controlled extension.

Table 5 contains neither train-held-out gaps nor matched no-CAIM per-seed
results. Those quantities are never inferred by subtracting historical
aggregate means. See `paper_protocol_aware_bundle/hybrid_derivation_manifest.json`.

The controlled extension's `operational_thresholds.json` is distributed with
this package. Under the predefined criterion (trimmed accuracy at least 0.80
and trimmed seed SD at most 0.10), the smallest evaluated operational points
are Full N=3 and no-CAIM N=5. They are protocol-specific empirical points,
not universal theoretical thresholds.
