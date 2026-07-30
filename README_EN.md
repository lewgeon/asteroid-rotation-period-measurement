[中文](README.md)

# Asteroid Radar Rotation Research

This project studies continuous-wave radar echoes from rotating asteroids and
rotation-period estimation from those echoes.

The code follows the research workflow:

```text
pointing  ──> observing geometry
                     ↓
simulation ──> EchoDataset ──> inversion
```

- `pointing`: ephemerides, two-way light time, transmit/receive directions;
- `simulation`: meshes, rigid rotation, scattering, and CW echoes;
- `inversion`: Doppler compensation, time-frequency features, period candidates;
- `data`: the `EchoDataset` passed between modules.

See [docs/architecture.md](docs/architecture.md) for the structure,
[EXPERIMENTS.md](EXPERIMENTS.md) for the experiment plan, and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for implementation history.

## Environment

```powershell
conda activate pytorch
$env:PYTHONPATH="$PWD\src"
```

## Run the stages

Generate a CW echo:

```powershell
conda run -n pytorch python scripts\simulation\run_cw.py
```

Estimate period candidates from the saved echo:

```powershell
conda run -n pytorch python scripts\inversion\estimate_period.py
```

Solve light-time-corrected pointing independently:

```powershell
conda run -n pytorch python scripts\pointing\solve.py
```

Run the 13 scientific-behaviour tests:

```powershell
conda run -n pytorch python -m unittest discover -s tests -v
```

The refactored CUDA simulation produces 345,600 complex samples from 1,280
facets in approximately 3.44 s. The RMS-bandwidth period estimate remains
7148.71 s for a 7200 s truth value (0.712% relative error).

The code deliberately favors readable research logic over product-level input
validation. Checks are retained only where failure could silently invalidate a
scientific result, such as Doppler aliasing, zero echo power, or failure of the
light-time iteration.
