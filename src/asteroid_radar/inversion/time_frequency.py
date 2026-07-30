"""Doppler compensation, STFT, and simple spectral features."""

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


def compensate(iq, times_s, translation_coefficients_hz):
    coefficients = np.asarray(translation_coefficients_hz)
    integral = np.r_[0.0, coefficients / np.arange(1, len(coefficients) + 1)]
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
        times,
        np.fft.fftshift(frequencies),
        np.fft.fftshift(spectrum, axes=0),
    )


def spectral_features(dynamic_spectrum):
    power = dynamic_spectrum.power
    frequencies = dynamic_spectrum.frequencies_hz[:, None]
    total = power.sum(0)
    centroid = (power * frequencies).sum(0) / total
    variance = (power * (frequencies - centroid) ** 2).sum(0) / total
    return SpectralFeatures(
        dynamic_spectrum.times_s,
        total,
        centroid,
        np.sqrt(np.maximum(variance, 0)),
    )
