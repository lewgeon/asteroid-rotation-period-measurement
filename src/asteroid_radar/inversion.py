"""Coarse rotation-period inversion from a saved EchoDataset."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .signal import compensate, spectral_features, stft


@dataclass
class PeriodCandidate:
    period_s: float
    score: float
    source: str


@dataclass
class PeriodEstimate:
    best_period_s: float
    candidates: tuple
    grid_periods_s: np.ndarray
    grid_scores: np.ndarray


@dataclass
class InversionResult:
    compensated_iq: np.ndarray
    dynamic_spectrum: object
    features: object
    periods: dict


def lomb_scargle(times, values, min_period, max_period, grid_size):
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    order = np.argsort(times)
    times, values = times[order], values[order]
    values = (values - values.mean()) / values.std()
    frequencies = np.linspace(1 / max_period, 1 / min_period, grid_size)
    scores = signal.lombscargle(
        times - times[0], values, 2 * np.pi * frequencies, normalize=True
    )
    periods = 1 / frequencies
    peaks, _ = signal.find_peaks(scores)
    peaks = peaks[np.argsort(scores[peaks])[::-1]][:5]
    candidates = tuple(
        PeriodCandidate(float(periods[i]), float(scores[i]), "lomb_scargle")
        for i in peaks
    )
    return PeriodEstimate(candidates[0].period_s, candidates, periods, scores)


def add_harmonics(estimate, min_period, max_period):
    candidates = list(estimate.candidates)
    for candidate in estimate.candidates:
        for scale, name in [(0.5, "half"), (2.0, "double")]:
            period = candidate.period_s * scale
            if min_period <= period <= max_period:
                candidates.append(
                    PeriodCandidate(period, candidate.score, f"{candidate.source}:{name}")
                )
    estimate.candidates = tuple(candidates)
    return estimate


def estimate_rotation(echo, config):
    compensated = compensate(
        echo.iq, echo.elapsed_s, echo.metadata["translation_coefficients_hz"]
    )
    dynamic = stft(
        compensated,
        echo.metadata["sample_rate_hz"],
        config["stft_window_samples"],
        config["stft_overlap_fraction"],
    )
    features = spectral_features(dynamic)
    periods = {}
    for name, values in {
        "total_power": features.total_power,
        "rms_bandwidth": features.rms_bandwidth_hz,
        "centroid": features.centroid_hz,
    }.items():
        estimate = lomb_scargle(
            features.times_s,
            values,
            config["period_min_s"],
            config["period_max_s"],
            config["period_grid_size"],
        )
        periods[name] = add_harmonics(
            estimate, config["period_min_s"], config["period_max_s"]
        )
    return InversionResult(compensated, dynamic, features, periods)
