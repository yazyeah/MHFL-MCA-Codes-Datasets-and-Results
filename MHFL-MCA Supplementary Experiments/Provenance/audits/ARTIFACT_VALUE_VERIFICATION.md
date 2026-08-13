# Artifact value verification

**Scope status:** `VERIFIED_FOR_DECLARED_SCOPE`

This audit verifies the numerical values, source mappings and selected byte
identities declared by the published repository artifacts. It is a
no-training verification: no model, SVM, Optuna study or data preprocessing
pipeline was executed.

The machine-readable companion is
[`ARTIFACT_VALUE_VERIFICATION.json`](ARTIFACT_VALUE_VERIFICATION.json).

## Verified contracts

| Artifact contract | Rows | Numeric fields | Mismatches | Status |
| --- | ---: | ---: | ---: | --- |
| Deep reference tables | 108 | 648 | 0 | PASS |
| Extreme low-shot display | 7 | 42 | 0 | PASS |
| Missing-modality display | 6 | 36 | 0 | PASS |
| Additional-ablation display | 4 | 36 | 0 | PASS |
| Efficiency display | 1 | 10 | 0 | PASS |
| Traditional-baseline display | 9 | 36 | 0 | PASS |

The protocol-aware low-shot PDF is byte-bound by SHA-256
`1a157cd5bdc213e61a2ab44d2f83830eb0b892b0404281a52315f3cb0aa5ad21`.

## Low-shot source mapping

- Full accuracy at N=5 and N=10 uses the declared main reference values.
- Other Full accuracy points use the controlled sensitivity extension.
- Without-CAIM accuracy, train-held-out gaps and paired CAIM gains use the
  controlled sensitivity extension.
- The two protocols are not pooled into a single estimator.

Under the predefined criterion (trimmed test accuracy at least 0.80 and
trimmed seed standard deviation at most 0.10), the smallest evaluated
empirical points are Full N=3 and no-CAIM N=5. These are protocol-specific
operational points, not universal theoretical thresholds.

## Scope limit

This verification is limited to the declared repository artifacts. It does
not claim byte-for-byte replay of historical random-number streams that were
not recorded.
