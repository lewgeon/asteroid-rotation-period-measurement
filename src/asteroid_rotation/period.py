"""Classical, explicit rotation-period estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PeriodCandidate:
    period_s: float
    score: float
    source: str


@dataclass(frozen=True)
class PeriodEstimate:
    best_period_s: float
    candidates: tuple[PeriodCandidate, ...]
    grid_periods_s: np.ndarray
    grid_scores: np.ndarray


def _validate_series(times_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or samples.ndim != 1 or times.shape != samples.shape:
        raise ValueError("times_s and values must be matching one-dimensional arrays")
    finite = np.isfinite(times) & np.isfinite(samples)
    times = times[finite]
    samples = samples[finite]
    if len(times) < 8:
        raise ValueError("At least eight finite samples are required")
    order = np.argsort(times)
    times = times[order]
    samples = samples[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be unique")
    centered = samples - np.mean(samples)
    scale = np.std(centered)
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("The feature series has no measurable variation")
    return times - times[0], centered / scale


def estimate_period_lomb_scargle(
    times_s: np.ndarray,
    values: np.ndarray,
    min_period_s: float,
    max_period_s: float,
    grid_size: int = 10_000,
    candidate_count: int = 5,
) -> PeriodEstimate:
    """Estimate period candidates on actual, possibly nonuniform timestamps."""

    if not 0 < min_period_s < max_period_s:
        raise ValueError("Expected 0 < min_period_s < max_period_s")
    if grid_size < 100 or candidate_count < 1:
        raise ValueError("Period grid/candidate count is too small")
    times, normalized = _validate_series(times_s, values)

    cyclic_frequencies = np.linspace(
        1.0 / max_period_s,
        1.0 / min_period_s,
        grid_size,
    )
    angular_frequencies = 2.0 * np.pi * cyclic_frequencies
    scores = signal.lombscargle(
        times,
        normalized,
        angular_frequencies,
        precenter=False,
        normalize=True,
    )
    periods = 1.0 / cyclic_frequencies

    peaks, _ = signal.find_peaks(scores)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(scores))])
    ranked = peaks[np.argsort(scores[peaks])[::-1]]

    selected = []
    minimum_separation = (max_period_s - min_period_s) / max(grid_size, 1) * 5.0
    for peak in ranked:
        period = float(periods[peak])
        if all(abs(period - item.period_s) > minimum_separation for item in selected):
            selected.append(
                PeriodCandidate(
                    period_s=period,
                    score=float(scores[peak]),
                    source="lomb_scargle",
                )
            )
        if len(selected) >= candidate_count:
            break

    return PeriodEstimate(
        best_period_s=selected[0].period_s,
        candidates=tuple(selected),
        grid_periods_s=periods,
        grid_scores=scores,
    )


def harmonic_aliases(
    candidates: tuple[PeriodCandidate, ...],
    min_period_s: float,
    max_period_s: float,
) -> tuple[PeriodCandidate, ...]:
    """Add explicit half/double aliases for review without hiding ambiguity."""

    augmented = list(candidates)
    seen = [candidate.period_s for candidate in candidates]
    for candidate in candidates:
        for multiplier, label in ((0.5, "half_alias"), (2.0, "double_alias")):
            period = candidate.period_s * multiplier
            tolerance = max(1e-9, period * 1e-6)
            if (
                min_period_s <= period <= max_period_s
                and all(abs(period - item) > tolerance for item in seen)
            ):
                augmented.append(
                    PeriodCandidate(
                        period_s=period,
                        score=candidate.score,
                        source=f"{candidate.source}:{label}",
                    )
                )
                seen.append(period)
    return tuple(augmented)

