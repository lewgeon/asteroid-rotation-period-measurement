[中文](README.md)

# Asteroid Rotation Period from Radar Echoes

This project simulates the complex baseband continuous-wave radar echo of a rotating asteroid, compensates translational Doppler, extracts time-frequency features, and estimates the asteroid rotation period.

The current version implements a validated ellipsoid triangle-mesh baseline. A real asteroid mesh will be introduced only after the physical and signal-processing chain is verified. [EXPERIMENTS.md](EXPERIMENTS.md) is the authoritative experiment specification, and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) records implementation history and validation results.

## Implemented

- Strict JSON configuration validated by JSON Schema.
- PyTorch3D triangle meshes, icosphere ellipsoids, and nonuniform scattering spots.
- OBJ/PLY mesh loading.
- A CPU float64 reference path and a CUDA float32 acceleration path.
- ICRS spin poles and high-precision Astropy time handling.
- Far-field monostatic or bistatic complex CW echoes.
- Polynomial translational Doppler and phase compensation.
- STFT micro-Doppler dynamic spectra.
- Echo-power, spectral-centroid, and RMS-bandwidth features.
- Lomb-Scargle candidates evaluated on actual timestamps.
- Explicit half-period and double-period alias candidates.
- An acquisition schedule that supports arbitrary nonuniform epochs.
- A standalone JPL Horizons three-event light-time and transmit/receive pointing solver.

## Environment

Run all Python commands in the Conda `pytorch` environment:

```powershell
conda activate pytorch
```

The baseline uses PyTorch 2.3.1, PyTorch3D 0.7.9, NumPy, SciPy, Astropy, Astroquery, jsonschema, Matplotlib, and tqdm. It does not require trimesh, h5py, or pytest.

## Tests

```powershell
$env:PYTHONPATH="$PWD\src"
conda run -n pytorch python -m unittest discover -s tests -v
```

Sixteen tests cover strict configuration, outward mesh normals, OBJ loading, scattering spots, nonuniform timestamps, analytic rotational Doppler, CPU/CUDA echo parity, Lomb-Scargle recovery, a short end-to-end CW experiment, analytic light time, and receive-time inversion.

## Light-time-corrected pointing

Edit [configs/light_time_pointing.json](configs/light_time_pointing.json) to set the target, UTC time, time role, and station coordinates, then run:

```powershell
conda run -n pytorch python scripts\solve_light_time_pointing.py `
  --config configs\light_time_pointing.json
```

The solver treats transmission, target reflection, and reception as distinct
events. It queries geometric Solar-System-barycentric target states from JPL
Horizons and solves the uplink and downlink light-time equations iteratively.
Its output contains UTC/TDB event times, ranges, light times, ICRS line-of-sight
vectors, RA/Dec, azimuth/elevation, and convergence residuals. The tool is
currently independent of the echo simulator. Operational antenna use requires
current IERS Earth-orientation data plus hardware, atmosphere, relativistic
light-time, and mount-model corrections.

## Baseline

```powershell
conda run -n pytorch python scripts\run_baseline.py `
  --config configs\baseline_cw.json `
  --output outputs\baseline_cw
```

All adjustable parameters are stored in [configs/baseline_cw.json](configs/baseline_cw.json) and validated by [schemas/experiment.schema.json](schemas/experiment.schema.json).

The PyTorch3D/CUDA six-hour run used a 7200 s ground-truth period, 10 dB SNR, 1280 faces, and 345,600 complex samples. It completed in approximately 5.42 s and produced an RMS-bandwidth estimate of 7148.71 s, or 0.712% relative error. The previous NumPy run used 528 faces and took 26.44 s, so the new path is about 4.88 times faster while processing 2.42 times as many faces.

This is a smoke-level baseline result, not the planned Monte Carlo validation. Physical local refinement, confidence intervals, JPL Horizons ephemerides, real asteroid meshes, and pulsed echoes remain future work.
