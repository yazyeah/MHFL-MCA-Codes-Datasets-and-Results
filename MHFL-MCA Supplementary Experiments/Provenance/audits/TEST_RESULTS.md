# Publication-package test results

Validation date: 2026-08-12 (Asia/Shanghai).

No training, Optuna study, smoke run, full experiment, or raw-dataset loading
was performed while validating this package.

## Passed

- 70 core protocol/configuration tests: PASS (`70 passed in 5.23s`).
- 28 model-spec, TF-SVM, and UO provenance tests: PASS (`28 passed in 4.12s`).
- 42 manuscript asset-gate tests: PASS (`42 passed in 272.17s`).
- Static audit: 53 Python files PASS, 0 WARN, 0 FAIL.
- Publication package validator: 35/35 checks PASS; the full results are
  recorded in `PACKAGE_VALIDATION.json`.

Total pytest assertions completed successfully: 140.

## Environment-blocked test

`tests/test_publication.py::test_figure_bundle` could not be executed in the
available local Python installations:

- base Python terminated inside the native plotting stack with Windows exit
  code `-1073740791`;
- the TensorFlow 2.7 environment could not import Pillow `_imaging` because a
  required DLL was unavailable.

This is an environment/native-library block, not a pytest assertion failure.
The distributed PDF/PNG/SVG bundles are still checked for presence, size, and
SHA-256 by the package validator. Re-run the one test in an environment with a
working Matplotlib/Pillow installation before claiming all 141 collected tests
passed.
