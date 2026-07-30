"""Doppler compensation and time-frequency feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal

from .echo import integrated_polynomial_phase_rad


@dataclass(frozen=True)
class DynamicSpectrum:
    times_s: np.ndarray
    frequencies_hz: np.ndarray
    complex_spectrum: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class SpectralFeatures:
    times_s: np.ndarray
    total_power: np.ndarray
    centroid_hz: np.ndarray
    rms_bandwidth_hz: np.ndarray
    lower_frequency_hz: np.ndarray
    upper_frequency_hz: np.ndarray


def compensate_translation_doppler(
    iq: np.ndarray,
    elapsed_s: np.ndarray,
    polynomial_coefficients_hz: Iterable[float],
) -> np.ndarray:
    """Remove a known or estimated translational Doppler polynomial."""

    samples = np.asarray(iq, dtype=np.complex128)
    times = np.asarray(elapsed_s, dtype=np.float64)
    if samples.shape != times.shape:
        raise ValueError("iq and elapsed_s must have identical shapes")
    phase = integrated_polynomial_phase_rad(polynomial_coefficients_hz, times)
    return samples * np.exp(-1j * phase)


def compute_dynamic_spectrum(
    iq: np.ndarray,
    sample_rate_hz: float,
    window_samples: int,
    overlap_fraction: float,
) -> DynamicSpectrum:
    """Compute a centered two-sided STFT."""

    samples = np.asarray(iq, dtype=np.complex128)
    if samples.ndim != 1:
        raise ValueError("iq must be one-dimensional")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not 16 <= window_samples <= len(samples):
        raise ValueError("window_samples must be between 16 and the sample count")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")

    overlap_samples = int(round(window_samples * overlap_fraction))
    frequencies, times, spectrum = signal.stft(
        samples,
        fs=sample_rate_hz,
        window="hann",
        nperseg=window_samples,
        noverlap=overlap_samples,
        nfft=window_samples,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    frequencies = np.fft.fftshift(frequencies)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    return DynamicSpectrum(
        times_s=times,
        frequencies_hz=frequencies,
        complex_spectrum=spectrum,
        power=np.abs(spectrum) ** 2,
    )


def extract_spectral_features(
    dynamic_spectrum: DynamicSpectrum,
    energy_fraction: float = 0.90,
) -> SpectralFeatures:
    """Extract robust one-dimensional features from every STFT column."""

    if not 0 < energy_fraction < 1:
        raise ValueError("energy_fraction must be in (0, 1)")
    power = np.asarray(dynamic_spectrum.power, dtype=np.float64)
    frequencies = dynamic_spectrum.frequencies_hz
    total = power.sum(axis=0)
    safe_total = np.maximum(total, np.finfo(np.float64).tiny)
    centroid = (power * frequencies[:, None]).sum(axis=0) / safe_total
    variance = (
        power * (frequencies[:, None] - centroid[None, :]) ** 2
    ).sum(axis=0) / safe_total

    cumulative = np.cumsum(power, axis=0)
    lower_fraction = 0.5 * (1.0 - energy_fraction)
    upper_fraction = 1.0 - lower_fraction
    lower_indices = np.argmax(cumulative >= lower_fraction * safe_total[None, :], axis=0)
    upper_indices = np.argmax(cumulative >= upper_fraction * safe_total[None, :], axis=0)

    return SpectralFeatures(
        times_s=dynamic_spectrum.times_s.copy(),
        total_power=total,
        centroid_hz=centroid,
        rms_bandwidth_hz=np.sqrt(np.maximum(variance, 0.0)),
        lower_frequency_hz=frequencies[lower_indices],
        upper_frequency_hz=frequencies[upper_indices],
    )

