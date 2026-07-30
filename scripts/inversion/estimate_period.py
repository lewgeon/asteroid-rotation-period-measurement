"""Estimate rotation-period candidates from a saved echo dataset."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asteroid_radar.data import load_echo
from asteroid_radar.inversion import estimate_rotation


parser = argparse.ArgumentParser()
parser.add_argument("--echo", default="outputs/simulation/cw_ellipsoid/echo.npz")
parser.add_argument("--config", default="configs/inversion/lomb_scargle.json")
parser.add_argument("--output", default="outputs/inversion/lomb_scargle")
args = parser.parse_args()

echo = load_echo(args.echo)
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
result = estimate_rotation(echo, config)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

truth = echo.metadata.get("truth_period_s")
summary = {}
for name, estimate in result.periods.items():
    summary[name] = {
        "best_period_s": estimate.best_period_s,
        "relative_error": (
            abs(estimate.best_period_s - truth) / truth if truth else None
        ),
        "candidates": [
            {
                "period_s": candidate.period_s,
                "score": candidate.score,
                "source": candidate.source,
            }
            for candidate in estimate.candidates
        ],
    }

(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

power = result.dynamic_spectrum.power
power_db = 10 * np.log10(power / power.max() + np.finfo(float).tiny)
fig, ax = plt.subplots(figsize=(10, 5))
image = ax.pcolormesh(
    result.dynamic_spectrum.times_s / 3600,
    result.dynamic_spectrum.frequencies_hz,
    power_db,
    shading="auto",
    cmap="magma",
    vmin=-45,
    vmax=0,
)
ax.set(xlabel="Elapsed time (h)", ylabel="Doppler frequency (Hz)")
fig.colorbar(image, ax=ax, label="Relative power (dB)")
fig.tight_layout()
fig.savefig(output / "dynamic_spectrum.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for name, estimate in result.periods.items():
    order = np.argsort(estimate.grid_periods_s)
    ax.plot(
        estimate.grid_periods_s[order] / 3600,
        estimate.grid_scores[order],
        label=name,
    )
if truth:
    ax.axvline(truth / 3600, color="black", linestyle="--", label="truth")
ax.set(xlabel="Candidate period (h)", ylabel="Lomb-Scargle score")
ax.legend()
fig.tight_layout()
fig.savefig(output / "periodogram.png", dpi=180)
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
