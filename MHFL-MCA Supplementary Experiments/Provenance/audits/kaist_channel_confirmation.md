# KAIST channel confirmation

- Confirmation status: `confirmed`
- TDMS group: `Log`
- Exact current channel: `cDAQ9185-1F486B5Mod2/ai0`
- Current physical meaning: U-phase motor current
- Vibration variable path: `Signal.y_values.values`
- Zero-based vibration column: `0`
- Vibration physical meaning: bearing housing A x-direction vibration

## Evidence basis

- All 15 inspected KAIST TDMS files have the same group/channel structure.
- The official KAIST data description defines the signal order as two housing
  temperatures followed by U-phase, V-phase, and W-phase motor current. The
  TDMS metadata identifies `cDAQ9185-1F486B5Mod2/ai0` as the first current
  channel (`DAC~Channel~Type=Current`, `unit_string=A`) in that documented
  order.
- The official vibration order is time stamp, housing A x-direction, housing A
  y-direction, housing B x-direction, and housing B y-direction. In the MAT
  files the time axis is stored separately, while `Signal.y_values.values` is a
  four-column matrix; zero-based column `0` is therefore housing A x-direction.
- The conclusion uses the official channel order together with raw file
  metadata. It does not use maximum variance or first-numeric-channel
  selection.
- The original preflight manifest value `configured_column=1` established only
  that the column existed in the four-column matrix. It did not establish that
  the configured column had the required physical meaning.

## Audited artifacts

- `provenance/preflight_optuna_confirmed_20260806/channel_manifest.json`
- `provenance/preflight_optuna_confirmed_20260806/vibration_manifest.json`
- `provenance/preflight_optuna_confirmed_20260806/preflight_report.json`
- `${MHFL_KAIST_CURRENT_DIR}/0Nm_BPFI_03.tdms`
- `${MHFL_KAIST_VIB_DIR}/0Nm_BPFI_03.mat`
- Jung et al., *Vibration, acoustic, temperature, and motor current dataset of
  rotating machine under varying operating conditions for fault diagnosis*,
  Data in Brief 48 (2023), 109049, DOI: `10.1016/j.dib.2023.109049`.

Historical training scripts requested U-phase current but could fall back to a
maximum-variance channel when physical phase labels were absent from TDMS
channel names. That fallback behavior is not used as confirmation evidence and
must remain disabled for reviewer-suite full runs.
