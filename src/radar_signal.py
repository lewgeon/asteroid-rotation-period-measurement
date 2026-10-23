"""Doppler compensation, STFT, and spectral features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
try:
    from . import compute_backend as backend
except ImportError:
    import compute_backend as backend


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
    energy_width_90_s: np.ndarray


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
    energy_range_power: np.ndarray | None = None
    energy_delays_s: np.ndarray | None = None
    intrinsic_doppler_centroid_hz: np.ndarray | None = None
    intrinsic_rms_bandwidth_hz: np.ndarray | None = None
    doppler_fit_quality: np.ndarray | None = None
    doppler_window_rms_hz: np.ndarray | None = None
    doppler_nyquist_edge_fraction: np.ndarray | None = None
    doppler_nyquist_edge_half_width_hz: np.ndarray | None = None

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
    coherence_id: np.ndarray
    effective_sample_count: np.ndarray
    energy_range_width_90_s: np.ndarray
    doppler_window_rms_hz: np.ndarray | None = None
    doppler_nyquist_edge_fraction: np.ndarray | None = None
    doppler_nyquist_edge_half_width_hz: np.ndarray | None = None


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


def sampling_window_amplitude(times_s, frequencies_hz, workspace_budget_bytes=16 * 1024 * 1024):
    """Sampling-window diagnostic, without a campaign-size complex matrix.

    Frequency columns are independent. If even one column exceeds the budget,
    accumulate time chunks as well; this only affects the diagnostic, not fits.
    The budget covers one complex matrix, not total process memory.
    """
    times = np.asarray(times_s, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if times.ndim != 1 or frequencies.ndim != 1:
        raise ValueError("采样窗时刻和频率必须是一维数组")
    budget = int(workspace_budget_bytes)
    if budget < 16:
        raise ValueError("采样窗工作区预算至少为 16 字节")
    cells = budget // 16
    rows = min(max(1, len(times)), cells)
    columns = max(1, cells // rows)
    result = np.empty(len(frequencies), dtype=float)
    for start in range(0, len(frequencies), columns):
        local = frequencies[start:start + columns]
        total = np.zeros(len(local), dtype=np.complex128)
        for row in range(0, len(times), rows):
            phase = -2j * np.pi * times[row:row + rows, None] * local[None, :]
            np.exp(phase, out=phase)
            total += phase.sum(axis=0)
        result[start:start + columns] = abs(total) / max(1, len(times))
    return result


def matched_filter_chirp(
    iq,
    fast_time_s,
    pulse_width_s,
    bandwidth_hz,
    *,
    baseband_convention="centered",
    full_output=False,
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
    full = (backend.matched_filter(iq,reference) if backend.is_cuda() else signal.fftconvolve(
        iq, np.conj(reference[::-1])[None, :], mode="full", axes=-1))
    if full_output:
        return full
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
    preserve_support=False,
    common_grid=None,
    method="linear",
):
    """Interpolate row-specific centroid-delay axes onto one common axis."""

    iq = np.asarray(compressed_iq)
    axes = np.asarray(delay_axes_s, dtype=float)
    if iq.ndim != 2 or axes.shape != iq.shape:
        raise ValueError("距离像和逐行时延轴必须具有相同二维形状")
    step = float(np.median(np.diff(axes, axis=1))) if common_grid is None else float(common_grid[1])
    common_start = float(np.median(axes[:, 0]))
    count = iq.shape[1]
    if preserve_support and common_grid is None:
        # Cover the union of full-correlation row axes rather than trimming
        # physical tails at the median row boundaries.
        common_start = float(np.min(axes[:, 0])) - step
        count = int(np.ceil((np.max(axes[:, -1]) + step - common_start) / step)) + 1
    common = (common_start + np.arange(count, dtype=float) * step
              if common_grid is None else np.asarray(common_grid[0], dtype=float))
    count = len(common)
    if valid is None:
        masks = np.ones(iq.shape, dtype=bool)
    else:
        masks = np.asarray(valid, dtype=bool)
        if masks.ndim == 1:
            masks = np.broadcast_to(masks[:, None], iq.shape)
        if masks.shape != iq.shape:
            raise ValueError("valid 必须是逐脉冲或与 IQ 同形状的掩码")
    aligned = np.zeros((iq.shape[0], count), dtype=iq.dtype)
    if method not in {"linear", "fourier"}:
        raise ValueError("距离轴重建method必须为linear或fourier")
    if method=='fourier' and backend.is_cuda():
        return common,backend.align(iq,axes,masks,common,step)
    for row in range(iq.shape[0]):
        mask = masks[row]
        if np.count_nonzero(mask) < 2:
            continue
        if method == "fourier":
            from scipy import fft
            intervals=np.diff(axes[row])
            if not np.allclose(intervals,step,rtol=1e-6,atol=1e-15):
                raise ValueError("频域距离重建需要均匀快时间采样")
            if not np.allclose(np.diff(common),step,rtol=1e-6,atol=1e-15):
                raise ValueError("频域距离重建需要相同采样间隔的公共轴")
            shift=(common[0]-axes[row,0])/step
            # Zero extension prevents a fractional shift wrapping the record.
            # 128 guard samples are diagnostic-tested, not a phase-accuracy
            # guarantee for arbitrary truncated/non-bandlimited waveforms.
            guard=128+int(np.ceil(abs(shift)))
            length=fft.next_fast_len(max(iq.shape[1],count)+2*guard)
            padded=np.zeros(length,dtype=np.complex128)
            padded[guard:guard+iq.shape[1]]=np.where(mask,iq[row],0.)
            shifted=fft.ifft(fft.fft(padded)*np.exp(2j*np.pi*fft.fftfreq(length)*shift))
            aligned[row]=shifted[guard:guard+count]
            continue
        source_axis = axes[row, mask]
        source = iq[row, mask]
        if preserve_support:
            # Explicit zero extension avoids cutting an endpoint to zero
            # immediately after its sample when row grids have different phase.
            source_axis = np.r_[source_axis[0] - step, source_axis, source_axis[-1] + step]
            source = np.r_[0j, source, 0j]
        aligned[row] = (
            np.interp(common, source_axis, source.real, left=0.0, right=0.0)
            + 1j * np.interp(common, source_axis, source.imag, left=0.0, right=0.0)
        )
    return common, aligned


def power_energy_width_90_s(power, delay_axes_s):
    """Discrete equal-tail 5%-95% power width, using the entire supplied row.

    Quantiles are the first sample whose cumulative power reaches the target
    fraction. No gate, threshold, background subtraction or peak selection.
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 2:
        raise ValueError("power 必须为二维 [行, 距离样点]")
    axes = np.asarray(delay_axes_s, dtype=float)
    if axes.ndim == 1 and axes.shape == (power.shape[1],):
        axes = np.broadcast_to(axes[None, :], power.shape)
    if axes.shape != power.shape:
        raise ValueError("delay_axes_s 必须是一维共同轴或与功率同形状")
    if not np.all(np.isfinite(axes)) or np.any(np.diff(axes, axis=1) <= 0):
        raise ValueError("距离轴必须有限且严格递增")
    if np.any(power < 0):
        raise ValueError("功率不能为负数")
    widths = np.full(power.shape[0], np.nan)
    for row, values in enumerate(power):
        if not len(values) or not np.all(np.isfinite(values)):
            continue
        peak = values.max()
        if peak <= 0:
            continue
        cumulative = np.cumsum(values / peak)
        lo = np.searchsorted(cumulative, .05 * cumulative[-1], side="left")
        hi = np.searchsorted(cumulative, .95 * cumulative[-1], side="left")
        widths[row] = axes[row, hi] - axes[row, lo]
    return widths


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
    return RangeFeatures(
        np.asarray(times_s, dtype=float), total, centroid,
        power_energy_width_90_s(power, delay_axis),
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
    energy_iq=None,
    energy_delay_axis_s=None,
    doppler_estimator="spectral_moments",
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
    energy_samples = None if energy_iq is None else np.asarray(energy_iq)
    if energy_samples is not None:
        energy_axis = np.asarray(energy_delay_axis_s, dtype=float)
        if (energy_samples.ndim != 2 or energy_samples.shape[0] != iq.shape[0]
                or energy_axis.shape != (energy_samples.shape[1],)):
            raise ValueError("完整脉压能量输入必须匹配脉冲数及其距离轴")
    elif energy_delay_axis_s is not None:
        raise ValueError("energy_delay_axis_s 必须与 energy_iq 同时提供")

    frame_times = []
    spectra = []
    frame_groups = []
    starts = []
    ends = []
    effective_counts = []
    frequency_axis = None
    frame_frequency_axes = []
    energy_marginals = []
    window_widths = []
    nyquist_edge_fractions = []
    nyquist_edge_half_widths = []
    intrinsic_centroids=[];intrinsic_widths=[];fit_quality=[]
    if doppler_estimator not in {"spectral_moments","narrowband_lag"}:
        raise ValueError('不支持的doppler_estimator')
    if doppler_estimator == "narrowband_lag" and cpi_pulses < 12:
        raise ValueError('低频有效多普勒估计至少需要12个相干脉冲；请增大CPI或选择谱功率矩')
    taper = signal.get_window(window, cpi_pulses, fftbins=True).astype(float)
    window_power = np.abs(np.fft.fftshift(np.fft.fft(taper))) ** 2
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
                spectrum = backend.slow_fft(cpi,taper) if backend.is_cuda() else np.fft.fftshift(
                    np.fft.fft(cpi * taper[:, None], axis=0), axes=0
                ).T
                if doppler_estimator=="narrowband_lag":
                    mu,width,quality=narrowband_doppler_moments(cpi,prt)
                    accepted=(quality['min_coherence']>=.9 and quality['log_fit_rms']<=5e-4
                        and quality['phase_fit_rms']<=.005 and quality['variance_before_clipping']>=-1e-6
                        and 2*np.pi*width*quality['max_lag_s']<=.45)
                    intrinsic_centroids.append(mu if accepted else np.nan)
                    intrinsic_widths.append(width if accepted else np.nan)
                    fit_quality.append([quality['min_coherence'],quality['log_fit_rms'],quality['phase_fit_rms'],float(accepted)])
                if energy_samples is not None:
                    energy_cpi = energy_samples[selection]
                    if not np.allclose(np.diff(local_times), prt, rtol=1e-3, atol=1e-9):
                        energy_cpi = np.vstack([
                            np.interp(target_times, local_times, energy_cpi[:, column].real)
                            + 1j * np.interp(target_times, local_times, energy_cpi[:, column].imag)
                            for column in range(energy_samples.shape[1])
                        ]).T
                    # Parseval gives exactly the Doppler-integrated power,
                    # avoiding a second full-range FFT and spectrum cube.
                    energy_marginals.append(cpi_pulses * np.sum(
                        np.abs(energy_cpi * taper[:, None]) ** 2, axis=0
                    ))
                local_frequency_axis = np.fft.fftshift(np.fft.fftfreq(cpi_pulses, d=prt))
                # Diagnostic only: this is the RMS of the window's own power
                # response, not a correction to the measured spectrum. Coherent
                # mixtures do not in general allow variance subtraction.
                window_total = float(window_power.sum())
                window_mean = float(window_power @ local_frequency_axis / window_total)
                window_widths.append(float(np.sqrt(
                    window_power @ (local_frequency_axis - window_mean) ** 2 / window_total)))
                # Report measured power within two DFT spacings of the Nyquist
                # seam. This exposes wrap sensitivity without unwrapping,
                # rejecting data or claiming that out-of-band aliasing is known.
                edge_half_width = 2.0 / (cpi_pulses * prt)
                edge = np.abs(local_frequency_axis) >= 0.5 / prt - edge_half_width
                doppler_power = np.sum(np.abs(spectrum) ** 2, axis=0)
                doppler_total = float(doppler_power.sum())
                nyquist_edge_fractions.append(
                    float(doppler_power[edge].sum() / doppler_total) if doppler_total > 0 else np.nan)
                nyquist_edge_half_widths.append(edge_half_width)
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
        energy_range_power=(np.asarray(energy_marginals) if energy_samples is not None else None),
        energy_delays_s=(energy_axis if energy_samples is not None else None),
        intrinsic_doppler_centroid_hz=(np.asarray(intrinsic_centroids) if intrinsic_centroids else None),
        intrinsic_rms_bandwidth_hz=(np.asarray(intrinsic_widths) if intrinsic_widths else None),
        doppler_fit_quality=(np.asarray(fit_quality) if fit_quality else None),
        doppler_window_rms_hz=np.asarray(window_widths),
        doppler_nyquist_edge_fraction=np.asarray(nyquist_edge_fractions),
        doppler_nyquist_edge_half_width_hz=np.asarray(nyquist_edge_half_widths),
    )


def range_doppler_features(cube: RangeDopplerCube) -> RangeDopplerFeatures:
    power = cube.power
    doppler_total = power.sum(axis=(1, 2))
    doppler_safe = np.maximum(doppler_total, np.finfo(float).tiny)
    doppler_marginal = power.sum(axis=1)
    # All three range/power statistics share the complete matched-filter
    # support when provided. The saved RD image may use a cropped distance
    # view; its power is still the input to the two Doppler statistics.
    range_marginal = (power.sum(axis=2) if cube.energy_range_power is None
                      else np.asarray(cube.energy_range_power))
    range_axis = cube.delays_s if cube.energy_delays_s is None else cube.energy_delays_s
    total = range_marginal.sum(axis=1)
    safe = np.maximum(total, np.finfo(float).tiny)
    fd = (cube.frequencies_hz[None, :] if cube.frame_frequencies_hz is None
          else cube.frame_frequencies_hz)
    delay = range_axis[None, :]
    doppler_centroid = (doppler_marginal * fd).sum(axis=1) / doppler_safe
    doppler_variance = (
        doppler_marginal * (fd - doppler_centroid[:, None]) ** 2
    ).sum(axis=1) / doppler_safe
    measured_width=np.sqrt(np.maximum(doppler_variance,0.0))
    if cube.intrinsic_doppler_centroid_hz is not None:
        doppler_centroid=cube.intrinsic_doppler_centroid_hz
        measured_width=cube.intrinsic_rms_bandwidth_hz
    range_centroid = (range_marginal * delay).sum(axis=1) / safe
    effective = (
        cube.effective_sample_count
        if cube.effective_sample_count is not None
        else cube.cpi_end_pulse - cube.cpi_start_pulse
    )
    return RangeDopplerFeatures(
        times_s=cube.times_s,total_power=total,doppler_centroid_hz=doppler_centroid,
        rms_bandwidth_hz=measured_width,range_centroid_s=range_centroid,
        coherence_id=cube.coherence_id,effective_sample_count=effective,
        energy_range_width_90_s=power_energy_width_90_s(range_marginal, range_axis),
        doppler_window_rms_hz=(np.full(len(total), np.nan) if cube.doppler_window_rms_hz is None
                              else cube.doppler_window_rms_hz),
        doppler_nyquist_edge_fraction=(np.full(len(total), np.nan) if cube.doppler_nyquist_edge_fraction is None
                                      else cube.doppler_nyquist_edge_fraction),
        doppler_nyquist_edge_half_width_hz=(np.full(len(total), np.nan) if cube.doppler_nyquist_edge_half_width_hz is None
                                          else cube.doppler_nyquist_edge_half_width_hz))


def narrowband_doppler_moments(iq, sample_interval_s):
    """Research estimator: local log-correlation curvature, without Hann bias.

    Returns validity diagnostics; it does not claim arbitrary-spectrum RMS.
    The Gaussian/narrow-spread approximation must be externally validated.
    """
    from scipy import fft
    values=np.asarray(iq,dtype=np.complex128)
    n=len(values);dt=float(sample_interval_s)
    if values.ndim!=2 or not np.isfinite(dt) or dt<=0 or not np.all(np.isfinite(values)):
        raise ValueError('滞后估计需要有限二维IQ及正采样间隔')
    maximum=min(n//4,max(3,int(.2/dt)))
    if maximum<3:raise ValueError('低频有效多普勒估计至少需要12个相干脉冲')
    scale=float(np.max(abs(values)))
    if scale>0:values=values/scale
    transformed=fft.fft(values,n=fft.next_fast_len(2*n-1),axis=0)
    correlation=fft.ifft(np.sum(abs(transformed)**2,axis=1))
    energies=np.sum(abs(values)**2,axis=1);prefix=np.r_[0.,np.cumsum(energies)]
    lags=np.arange(1,maximum+1)
    denominator=np.sqrt(prefix[n-lags]*(prefix[n]-prefix[lags]))
    rho=correlation[lags]/np.maximum(denominator,np.finfo(float).tiny)
    elapsed=lags*dt;weight=n-lags
    phase=np.unwrap(np.angle(rho))
    design=np.column_stack((np.ones(len(lags)),elapsed))
    coefficients=np.linalg.lstsq(design*np.sqrt(weight[:,None]),phase*np.sqrt(weight),rcond=None)[0]
    centroid=float(coefficients[1]/(2*np.pi))
    magnitude=np.log(np.maximum(abs(rho),np.finfo(float).tiny))
    design=np.column_stack((np.ones(len(lags)),elapsed**2))
    beta=np.linalg.lstsq(design*np.sqrt(weight[:,None]),magnitude*np.sqrt(weight),rcond=None)[0]
    variance=float(-beta[1]/(2*np.pi**2))
    return centroid,float(np.sqrt(max(variance,0))),dict(min_coherence=float(np.min(abs(rho))),
        log_fit_rms=float(np.sqrt(np.mean((design@beta-magnitude)**2))),
        phase_fit_rms=float(np.sqrt(np.mean((coefficients[0]+coefficients[1]*elapsed-phase)**2))),
        variance_before_clipping=variance,max_lag_s=float(elapsed[-1]))
