[中文](README.md)

# Asteroid Radar Rotation-Period Inversion Module

This directory is now the third submodule in the full project. It only estimates
rotation-period candidates from the `echo.npz` file produced by `../echo/`.
Line-of-sight solving lives in `../observation/`, and echo simulation lives in
`../echo/`.

Core files:

```text
inversion/
├── configs/
│   └── inversion.json
├── scripts/
│   └── estimate_period.py
├── src/
│   ├── dataset.py
│   ├── signal.py
│   └── inversion.py
└── tests/
    └── test_inversion.py
```

Legacy `ephemeris.py`, `pointing.py`, and `ellipsoid.py` files remain for
transition and historical tests. New end-to-end runs should use the top-level
`pipeline.py`.

## Input

The module reads the stable `echo.npz` exchange format:

```text
elapsed_s
iq
clean_iq
valid
coherence_id
tx_los_icrs
rx_los_icrs
tx_range_m
rx_range_m
scatter_elapsed_s
emit_elapsed_s
metadata_json
```

New echo datasets no longer contain the old `translation_coefficients_hz` field.
If it is absent, inversion runs directly on the saved complex echo. If the field
is provided in config or metadata, the legacy translation-Doppler compensation is
still applied.

## Run

```powershell
conda activate pytorch
python scripts\estimate_period.py `
  --echo ..\outputs\pipeline\echo\echo.npz `
  --config configs\inversion.json `
  --output ..\outputs\pipeline\inversion
```

For the full chain:

```powershell
conda activate pytorch
python pipeline.py --config configs\pipeline_example.json
```
