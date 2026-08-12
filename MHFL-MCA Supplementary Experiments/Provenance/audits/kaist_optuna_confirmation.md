# KAIST Optuna Configuration Confirmation

## Status

`confirmed`

The seven KAIST hyperparameters below were recovered from an original Optuna
run log. The confirmation does not rely on reverse engineering the manuscript's
reported 7.380 M parameter count.

## Primary evidence

- File: `${MHFL_PROJECT_ROOT}/KAIST/1. MHCNN_Optuna_KAIST_Advanced_4/log_MHCNN.txt`
- Relevant best-parameter record: line 15
- Corresponding model summary: lines 20-132
- File creation time: `2026-02-13T13:50:26.4322906+08:00`
- File last modification time: `2026-02-17T18:32:51.3579104+08:00`
- SHA-256: `B8809916148227949FF7389CBD1646CA4443A049AD39CC27626FFFD1D88CA555`

## Confirmed original Optuna parameters

```json
{
  "dropout_vib": 0.2905585371157774,
  "dropout_cur": 0.39293093464896633,
  "atten_dim": 256,
  "n_layers_vib": 5,
  "n_layers_cur": 3,
  "lr": 0.0004156294449523281,
  "batch_size": 16
}
```

## Optuna metadata

- Number of requested trials: `30`
- Pruner: `MedianPruner`
- Sampler: implicit single-objective Optuna default (`TPESampler`); the runtime
  Optuna version and sampler seed were not recorded
- Best value: not recorded in the retained artifact
- Best trial number: not recorded in the retained artifact
- Study name: not recorded; the source did not specify one
- Storage: in-memory; no persistent Optuna database was found

## ModelSpec mapping

- `n_layers_vib -> vib_layers = 5`
- `n_layers_cur -> other_layers = 3`
- `dropout_vib -> vib_dropout = 0.2905585371157774`
- `dropout_cur -> other_dropout = 0.39293093464896633`
- `atten_dim -> attention_dim = 256`

`lr` and `batch_size` are intentionally retained only in
`configs/kaist_optuna_confirmed.json`; they are not ModelSpec fields.

## Parameter-count cross-check

The corresponding historical `model.summary()` reports `7,380,173` parameters.

`round(7,380,173 / 1e6, 3) = 7.380`, matching the manuscript.

Parameter-count agreement is corroborating evidence only. It does not establish
the Dropout, learning-rate, or batch-size values independently.

## Integrity limitation

The retained log is currently named `log_MHCNN.txt`, while its contents refer to
the original name `log.txt`. Its last modification time is later than the run
completion time, and no Git-tracked copy or contemporaneous immutable hash was
found. The explicit best-parameter record and corresponding model summary are
therefore the primary retained historical evidence.
