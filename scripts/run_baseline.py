"""Run the configured CW baseline and write reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asteroid_rotation.config import load_experiment_config
from asteroid_rotation.experiment import run_continuous_wave_experiment


def _save_dynamic_spectrum(result, output_dir: Path) -> None:
    spectrum = result.dynamic_spectrum
    decibels = 10.0 * np.log10(
        spectrum.power / np.maximum(spectrum.power.max(), np.finfo(float).tiny)
        + np.finfo(float).tiny
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.pcolormesh(
        spectrum.times_s / 3600.0,
        spectrum.frequencies_hz,
        decibels,
        shading="auto",
        cmap="magma",
        vmin=-45,
        vmax=0,
    )
    axis.set_xlabel("Elapsed time (h)")
    axis.set_ylabel("Compensated Doppler frequency (Hz)")
    axis.set_title("Simulated asteroid micro-Doppler dynamic spectrum")
    figure.colorbar(image, ax=axis, label="Relative power (dB)")
    figure.tight_layout()
    figure.savefig(output_dir / "dynamic_spectrum.png", dpi=180)
    plt.close(figure)


def _save_period_diagnostics(result, truth_period_s: float, output_dir: Path) -> None:
    feature_map = {
        "total_power": result.features.total_power,
        "rms_bandwidth": result.features.rms_bandwidth_hz,
        "centroid": result.features.centroid_hz,
    }
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    for name, values in feature_map.items():
        if name not in result.period_estimates:
            continue
        axes[0].plot(
            result.features.times_s / 3600.0,
            (values - np.mean(values)) / max(np.std(values), np.finfo(float).eps),
            label=name,
            alpha=0.8,
        )
        estimate = result.period_estimates[name]
        order = np.argsort(estimate.grid_periods_s)
        axes[1].plot(
            estimate.grid_periods_s[order] / 3600.0,
            estimate.grid_scores[order],
            label=name,
        )
    axes[0].set_xlabel("Elapsed time (h)")
    axes[0].set_ylabel("Standardized feature")
    axes[0].legend()
    axes[1].axvline(
        truth_period_s / 3600.0,
        color="black",
        linestyle="--",
        label="truth",
    )
    axes[1].set_xlabel("Candidate period (h)")
    axes[1].set_ylabel("Lomb-Scargle score")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "period_diagnostics.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "baseline_cw.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "baseline_cw",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "config.snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump(config.data, stream, ensure_ascii=False, indent=2)

    progress = tqdm(desc="CW echo chunks", unit="chunk")
    last_completed = 0

    def update_progress(completed: int, total: int) -> None:
        nonlocal last_completed
        if progress.total is None:
            progress.total = total
        progress.update(completed - last_completed)
        last_completed = completed

    start = time.perf_counter()
    result = run_continuous_wave_experiment(config, progress_callback=update_progress)
    elapsed = time.perf_counter() - start
    progress.close()

    np.savez(
        args.output / "echo_and_features.npz",
        elapsed_s=result.echo.elapsed_s,
        iq=result.echo.iq,
        clean_iq=result.echo.clean_iq,
        compensated_iq=result.compensated_iq,
        stft_times_s=result.dynamic_spectrum.times_s,
        stft_frequencies_hz=result.dynamic_spectrum.frequencies_hz,
        stft_power=result.dynamic_spectrum.power.astype(np.float32),
        total_power=result.features.total_power,
        centroid_hz=result.features.centroid_hz,
        rms_bandwidth_hz=result.features.rms_bandwidth_hz,
    )
    truth_period_s = float(config.data["target"]["rotation_period_s"])
    summary = {
        "experiment": config.data["experiment"]["name"],
        "truth_period_s": truth_period_s,
        "runtime_s": elapsed,
        "face_count": int(len(result.mesh.faces)),
        "sample_count": int(len(result.echo.iq)),
        "sample_rate_hz": result.echo.sample_rate_hz,
        "max_rotation_doppler_bound_hz": result.echo.max_rotation_doppler_bound_hz,
        "period_estimates": {
            name: {
                "best_period_s": estimate.best_period_s,
                "relative_error": abs(estimate.best_period_s - truth_period_s)
                / truth_period_s,
                "candidates": [
                    {
                        "period_s": candidate.period_s,
                        "score": candidate.score,
                        "source": candidate.source,
                    }
                    for candidate in estimate.candidates
                ],
            }
            for name, estimate in result.period_estimates.items()
        },
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    _save_dynamic_spectrum(result, args.output)
    _save_period_diagnostics(result, truth_period_s, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

