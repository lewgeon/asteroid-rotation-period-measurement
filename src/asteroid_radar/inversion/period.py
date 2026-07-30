"""Lomb-Scargle period candidates and explicit harmonic aliases."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


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


def lomb_scargle(times, values, min_period, max_period, grid_size=10_000):
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
