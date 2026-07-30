[中文](README.md)

# Asteroid Rotation Period from Radar Echoes

This project simulates the complex baseband continuous-wave radar echo of a rotating asteroid, compensates translational Doppler, extracts time-frequency features, and estimates the asteroid rotation period.

The current version implements a validated ellipsoid triangle-mesh baseline. A real asteroid mesh will be introduced only after the physical and signal-processing chain is verified. [EXPERIMENTS.md](EXPERIMENTS.md) is the authoritative experiment specification.

## Implemented

- Strict JSON configuration validated by JSON Schema.
- Triaxial ellipsoid triangle meshes with a nonuniform scattering spot.
- ICRS spin poles and high-precision Astropy time handling.
- Far-field monostatic or bistatic complex CW echoes.
- Polynomial translational Doppler and phase compensation.
- STFT micro-Doppler dynamic spectra.
- Echo-power, spectral-centroid, and RMS-bandwidth features.
- Lomb-Scargle candidates evaluated on actual timestamps.
- Explicit half-period and double-period alias candidates.
- An acquisition schedule that supports arbitrary nonuniform epochs.

## Environment

Run all Python commands in the Conda `pytorch` environment:

```powershell
conda activate pytorch
```

The baseline uses NumPy, SciPy, Astropy, jsonschema, Matplotlib, and tqdm. It does not require trimesh, h5py, or pytest.

## Tests

```powershell
$env:PYTHONPATH="$PWD\src"
conda run -n pytorch python -m unittest discover -s tests -v
```

Eight tests currently cover strict configuration, outward mesh normals, scattering spots, nonuniform timestamps, analytic rotational Doppler, Lomb-Scargle recovery, and a short end-to-end CW experiment.

## Baseline

```powershell
conda run -n pytorch python scripts\run_baseline.py `
  --config configs\baseline_cw.json `
  --output outputs\baseline_cw
```

All adjustable parameters are stored in [configs/baseline_cw.json](configs/baseline_cw.json) and validated by [schemas/experiment.schema.json](schemas/experiment.schema.json).

The first six-hour run used a 7200 s ground-truth period, 10 dB SNR, 528 faces, and 345,600 complex samples. It completed in approximately 26.4 s. The RMS-bandwidth feature produced a coarse estimate of 7154.92 s, or 0.626% relative error.

This is a smoke-level baseline result, not the planned Monte Carlo validation. Physical local refinement, confidence intervals, JPL Horizons ephemerides, real asteroid meshes, and pulsed echoes remain future work.

