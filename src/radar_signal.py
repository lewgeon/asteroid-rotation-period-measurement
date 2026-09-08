"""Doppler compensation, STFT, and spectral features."""

from __future__ import annotations

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


@dataclass
class RangeDopplerCube:
    times_s: np.ndarray
    delays_s: np.ndarray
    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    coherence_id: np.ndarray
    cpi_start_pulse: np.ndarray
    cpi_end_pulse: np.ndarray
    effective_sample_count: np.ndarray | None = None
    frame_frequencies_hz: np.ndarray | None = None

    @property
    def power(self):
        return np.abs(self.spectrum) ** 2


@dataclass
class RangeDopplerFeatures:
    times_s: np.ndarray
    total_power: np.ndarray
    doppler_centroid_hz: np.ndarray
    rms_bandwidth_hz: np.ndarray
    range_centroid_s: np.ndarray
    rms_range_width_s: np.ndarray
    coherence_id: np.ndarray
    effective_sample_count: np.ndarray


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


def matched_filter_chirp(
    iq,
    fast_time_s,
    pulse_width_s,
    bandwidth_hz,
    *,
    baseband_convention="centered",
):
    """Range-compress every pulse independently along the fast-time axis."""

    iq = np.asarray(iq)
    fast_time_s = np.asarray(fast_time_s, dtype=float)
    if iq.ndim != 2 or iq.shape[1] != len(fast_time_s):
        raise ValueError("chirp 回波必须是 [pulse, fast_time] 二维数组")
    if len(fast_time_s) < 2:
        raise ValueError("快时间轴至少需要两个样本")
    sample_interval_s = float(np.median(np.diff(fast_time_s)))
    reference_count = chirp_reference_sample_count(pulse_width_s, sample_interval_s)
    local_time = np.arange(reference_count, dtype=float) * sample_interval_s
    slope = float(bandwidth_hz) / float(pulse_width_s)
    if baseband_convention == "zero_to_bandwidth":
        phase_time = local_time
    elif baseband_convention == "centered":
        phase_time = local_time - 0.5 * float(pulse_width_s)
    else:
        raise ValueError(f"不支持的 chirp 基带约定：{baseband_convention}")
    reference = np.exp(1j * np.pi * slope * phase_time**2)
    # A full correlation peaks at ``leading_edge_index + reference_count - 1``.
    # Slice away that known correlation lag while preserving the input shape,
    # so callers that historically expected an array remain compatible and its
    # sample index now directly denotes the echo leading edge.
    full = signal.fftconvolve(
        iq, np.conj(reference[::-1])[None, :], mode="full", axes=-1
    )
    lag = matched_filter_lag_samples(reference_count)
    return full[:, lag : lag + iq.shape[1]]


def chirp_reference_sample_count(pulse_width_s, sample_interval_s):
    """Number of samples used by the leading-edge LFM reference."""

    return max(2, int(np.ceil(float(pulse_width_s) / float(sample_interval_s))))


def matched_filter_lag_samples(reference_count):
    """Full-correlation lag removed from ``matched_filter_chirp`` output."""

    return int(reference_count) - 1


def matched_filter_group_delay_s(pulse_width_s, sample_interval_s):
    """Physical full-correlation lag removed by the leading-edge output slice."""

    count = chirp_reference_sample_count(pulse_width_s, sample_interval_s)
    return matched_filter_lag_samples(count) * float(sample_interval_s)


def centroid_delay_axes(
    fast_time_s,
    pulse_count,
    *,
    row_fast_time_offset_s=None,
    centroid_fractional_offset_s=None,
    output_lag_s=0.0,
):
    """Build per-row differential-delay axes relative to the target centroid.

    ``row_fast_time_offset_s`` is authoritative when present: reception
    planning defines it as ADC sample time minus centroid receive time, so it
    already contains the fractional ADC offset.  ``centroid_fractional_offset``
    is only used to reconstruct that correction for older datasets without row
    offsets, avoiding the former double application.
    """

    fast = np.asarray(fast_time_s, dtype=float)
    count = int(pulse_count)
    offsets = np.zeros(count, dtype=float)
    have_offsets = False
    if row_fast_time_offset_s is not None:
        candidate = np.asarray(row_fast_time_offset_s, dtype=float)
        if candidate.shape == (count,) and np.all(np.isfinite(candidate)):
            offsets = candidate
            have_offsets = True
        elif candidate.size and np.any(np.isfinite(candidate)):
            raise ValueError("row_fast_time_offset_s 必须是每脉冲一个有限值")
    if not have_offsets and centroid_fractional_offset_s is not None:
        fractional = np.asarray(centroid_fractional_offset_s, dtype=float)
        if fractional.shape == (count,) and np.all(np.isfinite(fractional)):
            offsets = -fractional
        elif fractional.size and np.any(np.isfinite(fractional)):
            raise ValueError("centroid_fractional_offset_s 必须是每脉冲一个有限值")
    return fast[None, :] + offsets[:, None] - float(output_lag_s)


def align_profiles_to_common_delay(
    compressed_iq,
    delay_axes_s,
    *,
    valid=None,
):
    """Interpolate row-specific centroid-delay axes onto one common axis."""

    iq = np.asarray(compressed_iq)
    axes = np.asarray(delay_axes_s, dtype=float)
    if iq.ndim != 2 or axes.shape != iq.shape:
        raise ValueError("距离像和逐行时延轴必须具有相同二维形状")
    step = float(np.median(np.diff(axes, axis=1)))
    common_start = float(np.median(axes[:, 0]))
    common = common_start + np.arange(iq.shape[1], dtype=float) * step
    if valid is None:
        masks = np.ones(iq.shape, dtype=bool)
    else:
        masks = np.asarray(valid, dtype=bool)
        if masks.ndim == 1:
            masks = np.broadcast_to(masks[:, None], iq.shape)
        if masks.shape != iq.shape:
            raise ValueError("valid 必须是逐脉冲或与 IQ 同形状的掩码")
    aligned = np.zeros_like(iq)
    for row in range(iq.shape[0]):
        mask = masks[row]
        if np.count_nonzero(mask) < 2:
            continue
        source_axis = axes[row, mask]
        source = iq[row, mask]
        aligned[row] = (
            np.interp(common, source_axis, source.real, left=0.0, right=0.0)
            + 1j * np.interp(common, source_axis, source.imag, left=0.0, right=0.0)
        )
    return common, aligned


def range_features(
    times_s,
    fast_time_s,
    compressed_iq,
    row_fast_time_offset_s=None,
    *,
    delay_axes_s=None,
):
    power = np.abs(compressed_iq) ** 2
    total = power.sum(axis=1)
    safe_total = np.maximum(total, np.finfo(float).tiny)
    if delay_axes_s is None:
        delay_axis = centroid_delay_axes(
            fast_time_s,
            len(compressed_iq),
            row_fast_time_offset_s=row_fast_time_offset_s,
        )
    else:
        delay_axis = np.asarray(delay_axes_s, dtype=float)
        if delay_axis.ndim == 1:
            delay_axis = np.broadcast_to(delay_axis[None, :], power.shape)
        if delay_axis.shape != power.shape:
            raise ValueError("delay_axes_s 必须是一维共同轴或与距离像同形状")
    centroid = (power * delay_axis).sum(axis=1) / safe_total
    variance = (
        power * (delay_axis - centroid[:, None]) ** 2
    ).sum(axis=1) / safe_total
    return RangeFeatures(
        np.asarray(times_s, dtype=float), total, centroid, np.sqrt(np.maximum(variance, 0))
    )


def range_doppler_cube(
    compressed_iq,
    receive_times_s,
    coherence_id,
    fast_time_s,
    *,
    cpi_pulses,
    cpi_hop_pulses,
    window="hann",
    valid=None,
    run_id=None,
    effective_pulse_weight=None,
):
    """Form sliding-CPI range-Doppler frames without crossing coherence groups."""

    iq = np.asarray(compressed_iq)
    times = np.asarray(receive_times_s, dtype=float)
    groups = np.asarray(coherence_id, dtype=int)
    fast_time = np.asarray(fast_time_s, dtype=float)
    cpi_pulses = int(cpi_pulses)
    cpi_hop_pulses = int(cpi_hop_pulses)
    if iq.ndim != 2 or len(times) != iq.shape[0] or len(groups) != iq.shape[0]:
        raise ValueError("CPI 输入的脉冲、时标和相干分组长度必须一致")
    if cpi_pulses < 2 or cpi_hop_pulses < 1:
        raise ValueError("cpi_pulses 至少为 2，cpi_hop_pulses 至少为 1")

    frame_times = []
    spectra = []
    frame_groups = []
    starts = []
    ends = []
    effective_counts = []
    frequency_axis = None
    frame_frequency_axes = []
    taper = signal.get_window(window, cpi_pulses, fftbins=True).astype(float)
    if valid is None:
        pulse_valid = np.ones(iq.shape[0], dtype=bool)
    else:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.ndim == 2:
            if valid_array.shape != iq.shape:
                raise ValueError("二维 valid 必须与 IQ 同形状")
            pulse_valid = np.any(valid_array, axis=1)
        elif valid_array.shape == (iq.shape[0],):
            pulse_valid = valid_array
        else:
            raise ValueError("valid 必须是逐脉冲或与 IQ 同形状的掩码")
    runs = np.zeros(iq.shape[0], dtype=int) if run_id is None else np.asarray(run_id, dtype=int)
    if runs.shape != groups.shape:
        raise ValueError("run_id 必须与脉冲数一致")
    weights = (
        np.ones(iq.shape[0], dtype=float)
        if effective_pulse_weight is None
        else np.asarray(effective_pulse_weight, dtype=float)
    )
    if weights.shape != groups.shape:
        raise ValueError("effective_pulse_weight 必须与脉冲数一致")
    # Pairing run and coherence IDs forbids a CPI from crossing either seam.
    keys = np.column_stack((runs, groups))
    for run, group in np.unique(keys, axis=0):
        indices = np.flatnonzero(
            (runs == run) & (groups == group) & pulse_valid
        )
        # A coherence group must be contiguous; silently joining separated
        # blocks would reintroduce the forbidden gap-filling behaviour.
        split_points = np.flatnonzero(np.diff(indices) != 1) + 1
        for block in np.split(indices, split_points):
            if len(block) < cpi_pulses:
                continue
            for local_start in range(0, len(block) - cpi_pulses + 1, cpi_hop_pulses):
                selection = block[local_start : local_start + cpi_pulses]
                local_times = times[selection]
                prt = float(np.median(np.diff(local_times)))
                if not np.allclose(np.diff(local_times), prt, rtol=1e-3, atol=1e-9):
                    # Resample only the tiny propagation-induced arrival jitter
                    # onto a local CPI grid; campaign gaps are never resampled.
                    target_times = local_times[0] + np.arange(cpi_pulses) * prt
                    real = np.vstack([
                        np.interp(target_times, local_times, iq[selection, column].real)
                        for column in range(iq.shape[1])
                    ]).T
                    imag = np.vstack([
                        np.interp(target_times, local_times, iq[selection, column].imag)
                        for column in range(iq.shape[1])
                    ]).T
                    cpi = real + 1j * imag
                else:
                    cpi = iq[selection]
                spectrum = np.fft.fftshift(
                    np.fft.fft(cpi * taper[:, None], axis=0), axes=0
                ).T
                local_frequency_axis = np.fft.fftshift(np.fft.fftfreq(cpi_pulses, d=prt))
                if frequency_axis is None:
                    frequency_axis = local_frequency_axis
                frame_frequency_axes.append(local_frequency_axis)
                spectra.append(spectrum)
                frame_times.append(float(np.mean(local_times)))
                frame_groups.append(int(group))
                starts.append(int(selection[0]))
                ends.append(int(selection[-1] + 1))
                # Overlapping CPIs are correlated.  Allocate each selected
                # pulse at most one unit of information across all windows
                # using the nominal overlap multiplicity.
                overlap_multiplicity = max(
                    1, int(np.ceil(cpi_pulses / cpi_hop_pulses))
                )
                effective_counts.append(
                    float(np.sum(weights[selection]) / overlap_multiplicity)
                )
    if not spectra:
        raise ValueError("没有任何相干分组包含足够脉冲形成一个 CPI")
    return RangeDopplerCube(
        times_s=np.asarray(frame_times),
        delays_s=fast_time,
        frequencies_hz=np.asarray(frequency_axis),
        spectrum=np.asarray(spectra),
        coherence_id=np.asarray(frame_groups, dtype=int),
        cpi_start_pulse=np.asarray(starts, dtype=int),
        cpi_end_pulse=np.asarray(ends, dtype=int),
        effective_sample_count=np.asarray(effective_counts, dtype=float),
        frame_frequencies_hz=np.asarray(frame_frequency_axes),
    )


def range_doppler_features(cube: RangeDopplerCube) -> RangeDopplerFeatures:
    power = cube.power
    total = power.sum(axis=(1, 2))
    safe = np.maximum(total, np.finfo(float).tiny)
    doppler_marginal = power.sum(axis=1)
    range_marginal = power.sum(axis=2)
    fd = (cube.frequencies_hz[None, :] if cube.frame_frequencies_hz is None
          else cube.frame_frequencies_hz)
    delay = cube.delays_s[None, :]
    doppler_centroid = (doppler_marginal * fd).sum(axis=1) / safe
    doppler_variance = (
        doppler_marginal * (fd - doppler_centroid[:, None]) ** 2
    ).sum(axis=1) / safe
    range_centroid = (range_marginal * delay).sum(axis=1) / safe
    range_variance = (
        range_marginal * (delay - range_centroid[:, None]) ** 2
    ).sum(axis=1) / safe
    effective = (
        cube.effective_sample_count
        if cube.effective_sample_count is not None
        else cube.cpi_end_pulse - cube.cpi_start_pulse
    )
    return RangeDopplerFeatures(
        times_s=cube.times_s,
        total_power=total,
        doppler_centroid_hz=doppler_centroid,
        rms_bandwidth_hz=np.sqrt(np.maximum(doppler_variance, 0.0)),
        range_centroid_s=range_centroid,
        rms_range_width_s=np.sqrt(np.maximum(range_variance, 0.0)),
        coherence_id=cube.coherence_id,
        effective_sample_count=effective,
    )
