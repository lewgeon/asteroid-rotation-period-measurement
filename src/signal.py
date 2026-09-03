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


@dataclass
class RangeProfiles:
    times_s: np.ndarray
    delays_s: np.ndarray
    compressed_iq: np.ndarray

    @property
    def power(self):
        return np.abs(self.compressed_iq) ** 2


@dataclass
class RangeFeatures:
    times_s: np.ndarray
    total_power: np.ndarray
    centroid_delay_s: np.ndarray
    rms_width_s: np.ndarray


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


def matched_filter_chirp(iq, fast_time_s, pulse_width_s, bandwidth_hz):
    """Range-compress every pulse independently along the fast-time axis."""

    iq = np.asarray(iq)
    fast_time_s = np.asarray(fast_time_s, dtype=float)
    if iq.ndim != 2 or iq.shape[1] != len(fast_time_s):
        raise ValueError("chirp 回波必须是 [pulse, fast_time] 二维数组")
    reference = np.zeros(len(fast_time_s), dtype=complex)
    inside = np.abs(fast_time_s) < 0.5 * float(pulse_width_s)
    slope = float(bandwidth_hz) / float(pulse_width_s)
    reference[inside] = np.exp(1j * np.pi * slope * fast_time_s[inside] ** 2)
    compressed = signal.fftconvolve(
        iq, np.conj(reference[::-1])[None, :], mode="same", axes=-1
    )
    return compressed


def range_features(times_s, fast_time_s, compressed_iq):
    power = np.abs(compressed_iq) ** 2
    total = power.sum(axis=1)
    safe_total = np.maximum(total, np.finfo(float).tiny)
    centroid = (power * fast_time_s[None, :]).sum(axis=1) / safe_total
    variance = (
        power * (fast_time_s[None, :] - centroid[:, None]) ** 2
    ).sum(axis=1) / safe_total
    return RangeFeatures(
        np.asarray(times_s, dtype=float), total, centroid, np.sqrt(np.maximum(variance, 0))
    )
