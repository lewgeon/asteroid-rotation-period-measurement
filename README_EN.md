[中文](README.md)

# Asteroid Radar Rotation Research

The project studies CW radar echo simulation, rotation-period inversion, and
light-time-corrected radar pointing.

The source tree is intentionally flat:

```text
src/asteroid_radar/
├── ellipsoid.py
├── mesh.py
├── motion.py
├── echo.py
├── dataset.py
├── signal.py
├── inversion.py
├── ephemeris.py
└── pointing.py
```

## Environment

```powershell
conda activate pytorch
$env:PYTHONPATH="$PWD\src"
```

Generate the baseline OBJ model:

```powershell
conda run -n pytorch python scripts\make_ellipsoid.py `
  --axes 70 50 40 --subdivisions 3 `
  --output models\ellipsoid.obj
```

Generate an echo, estimate its period, and solve pointing:

```powershell
conda run -n pytorch python scripts\simulate_echo.py
conda run -n pytorch python scripts\estimate_period.py
conda run -n pytorch python scripts\solve_pointing.py
```

The echo configuration contains only a shape path:

```json
"model_path": "models/ellipsoid.obj"
```

Replacing it with a prepared real-asteroid OBJ/PLY requires no simulation-code
changes. Meshes are assumed to use metres, a body-fixed frame, an origin at the
rotation centre, outward face winding, and nondegenerate triangles.

Run the 12 scientific tests:

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -v
```

The 1,280-facet baseline produces 345,600 complex samples on CUDA in about
2.13 s. Its RMS-bandwidth period estimate remains 7148.71 s for a 7200 s truth.

See [docs/architecture.md](docs/architecture.md),
[EXPERIMENTS.md](EXPERIMENTS.md), and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for structure, experiment design, and
history.
