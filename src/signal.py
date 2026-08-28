"""Doppler compensation, STFT, and spectral features."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DynamicSpectrum:
    times_s: np.ndarray
    frequencies_hz: np.ndarray
    spectrum: np.ndarray

    @property
    def power(self):
        return np.abs(self.spectrum) ** 2


@dataclass
class SpectralFeatures:
    times_s: np.ndarray
    total_power: np.ndarray
    centroid_hz: np.ndarray
    rms_bandwidth_hz: np.ndarray


def compensate(iq, times_s, coefficients_hz):
    coefficients_hz = np.asarray(coefficients_hz)
    integral = np.r_[0.0, coefficients_hz / np.arange(1, len(coefficients_hz) + 1)]
    phase = 2 * np.pi * np.polynomial.polynomial.polyval(times_s, integral)
    return iq * np.exp(-1j * phase)


def stft(iq, sample_rate_hz, window_samples, overlap_fraction):
    frequencies, times, spectrum = signal.stft(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=window_samples,
        noverlap=round(window_samples * overlap_fraction),
        nfft=window_samples,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    return DynamicSpectrum(
        times, np.fft.fftshift(frequencies), np.fft.fftshift(spectrum, axes=0)
    )


def spectral_features(dynamic):
    power = dynamic.power
    frequencies = dynamic.frequencies_hz[:, None]
    total = power.sum(0)
    centroid = (power * frequencies).sum(0) / total
    variance = (power * (frequencies - centroid) ** 2).sum(0) / total
    return SpectralFeatures(
        dynamic.times_s, total, centroid, np.sqrt(np.maximum(variance, 0))
    )
