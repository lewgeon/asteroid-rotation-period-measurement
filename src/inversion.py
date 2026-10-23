"""Coarse rotation-period inversion from a saved EchoDataset."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import optimize, signal
try:
    from .compute_backend import configured,resolve_device,is_cuda,period_grid_sse
except ImportError:
    from compute_backend import configured,resolve_device,is_cuda,period_grid_sse

try:
    from .period_enhancements import effective_look_phase, method_options, comparison_options
except ImportError:
    enhancement_spec = importlib.util.spec_from_file_location(
        "rotation_period_enhancements", Path(__file__).with_name("period_enhancements.py"))
    enhancement_module = importlib.util.module_from_spec(enhancement_spec)
    enhancement_spec.loader.exec_module(enhancement_module)
    effective_look_phase = enhancement_module.effective_look_phase
    method_options = enhancement_module.method_options
    comparison_options = enhancement_module.comparison_options

try:
    from .radar_signal import (
        RangeFeatures, RangeProfiles, align_profiles_to_common_delay, centroid_delay_axes,
        chirp_reference_sample_count, compensate, matched_filter_chirp, range_doppler_cube,
        range_doppler_features, range_features, spectral_features, stft,
        matched_filter_group_delay_s, power_energy_width_90_s,
    )
except ImportError:  # 支持 PYTHONPATH=src 时按单文件模块运行脚本。
    signal_path = Path(__file__).with_name("radar_signal.py")
    spec = importlib.util.spec_from_file_location("asteroid_radar_signal", signal_path)
    local_signal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_signal)
    compensate = local_signal.compensate
    spectral_features = local_signal.spectral_features
    stft = local_signal.stft
    RangeProfiles = local_signal.RangeProfiles
    RangeFeatures = local_signal.RangeFeatures
    matched_filter_chirp = local_signal.matched_filter_chirp
    chirp_reference_sample_count = local_signal.chirp_reference_sample_count
    range_features = local_signal.range_features
    range_doppler_cube = local_signal.range_doppler_cube
    range_doppler_features = local_signal.range_doppler_features
    centroid_delay_axes = local_signal.centroid_delay_axes
    align_profiles_to_common_delay = local_signal.align_profiles_to_common_delay
    matched_filter_group_delay_s = local_signal.matched_filter_group_delay_s
    power_energy_width_90_s = local_signal.power_energy_width_90_s


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
    false_alarm_probability: float = 1.0
    significant: bool = False
    significance_calibration: str = "single_harmonic_lomb_approximation"
    search_diagnostics: dict = field(default_factory=dict)


class PeriodSearchUnavailable(ValueError):
    """An expected data limitation, distinct from invalid search configuration."""

    def __init__(self, reason, message, diagnostics=None):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})


@dataclass
class InversionResult:
    compensated_iq: np.ndarray
    dynamic_spectrum: object
    features: object
    periods: dict
    time_axis_role: str = "receive_elapsed_s"
    notices: tuple = ()
    search_diagnostics: dict = field(default_factory=dict)
    method_comparisons: dict = field(default_factory=dict)
    uncertainty_report: dict = field(default_factory=dict)
    selection_report: dict = field(default_factory=dict)


def lomb_scargle(times, values, min_period, max_period, grid_size):
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    order = np.argsort(times)
    times, values = times[order], values[order]
    std = values.std()
    if len(values) < 3 or std == 0:
        raise ValueError("周期估计需要至少 3 个非恒定有效特征样本")
    values = (values - values.mean()) / std
    frequencies = np.linspace(1 / max_period, 1 / min_period, grid_size)
    scores = signal.lombscargle(
        times - times[0], values, 2 * np.pi * frequencies, normalize=True
    )
    periods = 1 / frequencies
    peaks, _ = signal.find_peaks(scores)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(scores))])
    peaks = peaks[np.argsort(scores[peaks])[::-1]][:5]
    candidates = tuple(
        PeriodCandidate(float(periods[i]), float(scores[i]), "lomb_scargle")
        for i in peaks
    )
    false_alarm_probability, significant = _peak_significance(
        times, scores[peaks[0]], frequencies
    )
    return PeriodEstimate(
        candidates[0].period_s,
        candidates,
        periods,
        scores,
        false_alarm_probability,
        significant,
    )


def multi_harmonic_lomb_scargle(
    times,
    values,
    min_period,
    max_period,
    grid_size,
    *,
    harmonics=1,
    groups=None,
    weights=None,
    phase_offset=None,
    refine_period=None,
    compute_device=None,
):
    """Search with a shared intercept; explicit groups are a research option.

    Project chirp callers deliberately omit groups to preserve between-run
    modulation. Zero/nonfinite weights or nonfinite observations are excluded.
    """

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("周期搜索 times 与 values 必须是同形状一维数组")
    groups = np.zeros(times.size, dtype=int) if groups is None else np.asarray(groups, dtype=int)
    weights = np.ones(times.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    legacy_refinement = refine_period is None
    if refine_period is not None and not isinstance(refine_period, (bool, np.bool_)):
        raise ValueError("refine_period 必须是布尔值")
    refine_period = True if legacy_refinement else refine_period
    offset = np.zeros(times.size) if phase_offset is None else np.asarray(phase_offset, dtype=float)
    if offset.shape != times.shape or not np.all(np.isfinite(offset)):
        raise ValueError("phase_offset 必须是与 times 同形状的有限一维数组")
    if groups.shape != times.shape or weights.shape != times.shape:
        raise ValueError("周期搜索 groups 与 weights 必须与 times 同形状")
    def positive_integer(value, name, minimum):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} 必须是至少为 {minimum} 的整数")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} 必须是至少为 {minimum} 的整数") from None
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
            raise ValueError(f"{name} 必须是至少为 {minimum} 的整数")
        return int(numeric)
    harmonics = positive_integer(harmonics, "harmonics", 1)
    grid_size = positive_integer(grid_size, "period_grid_size", 2)
    min_period, max_period = float(min_period), float(max_period)
    if not (np.isfinite(min_period) and np.isfinite(max_period)
            and 0 < min_period < max_period):
        raise ValueError("周期范围必须满足 0 < period_min_s < period_max_s，且为有限数")
    input_count = times.size
    finite = np.isfinite(times) & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    times, values, groups, weights = times[finite], values[finite], groups[finite], weights[finite]
    offset = offset[finite]
    order = np.argsort(times)
    times, values, groups, weights = times[order], values[order], groups[order], weights[order]
    offset = offset[order]
    unique_groups = np.unique(groups)
    parameter_count = 2 * harmonics + len(unique_groups)
    diagnostics = {
        "input_sample_count": int(input_count),
        "valid_sample_count": int(len(values)),
        "excluded_sample_count": int(input_count - len(values)),
        "parameter_count": int(parameter_count),
        "baseline_count": int(len(unique_groups)),
        "harmonics": harmonics,
        "refine_period": bool(refine_period),
        "phase_compensation": phase_offset is not None,
        "candidate_policy": "legacy_interior_peaks" if legacy_refinement and phase_offset is None else "peaks_and_boundaries",
        "unique_time_count": int(len(np.unique(times))),
    }
    search_execution = (resolve_device({'compute_device':compute_device}) if compute_device is not None
                        else dict(requested='cuda:0' if is_cuda() else 'cpu', actual='cuda:0' if is_cuda() else 'cpu', reason=None))
    diagnostics['period_compute'] = search_execution
    # A saturated design matrix can reproduce every sample and turns the
    # periodogram into an all-ones curve.  Preserve at least three residual
    # degrees of freedom so a candidate period must explain unseen variation.
    if len(values) < max(5, parameter_count + 3):
        raise PeriodSearchUnavailable(
            "insufficient_samples", "周期估计的有效特征样本不足；"
            f"当前 {len(values)} 点、{parameter_count} 个拟合参数，至少需要 3 个残差自由度",
            diagnostics,
        )
    if np.max(values) == np.min(values):
        raise PeriodSearchUnavailable("constant_feature", "有效特征样本为常量", diagnostics)
    if diagnostics["unique_time_count"] <= 2 * harmonics + 1:
        raise PeriodSearchUnavailable(
            "degenerate_time_axis", "不同有效时刻不足，无法区分周期模型与饱和拟合", diagnostics
        )
    centred_t = times - times[0]
    if not np.all(np.isfinite(centred_t)):
        raise ValueError("周期搜索时间跨度超出浮点可表示范围")
    if all(np.max(values[groups == group]) == np.min(values[groups == group])
           for group in unique_groups):
        raise PeriodSearchUnavailable(
            "no_baseline_variation", "特征在基线扣除后无可用于周期搜索的变化", diagnostics
        )
    group_columns = np.column_stack([(groups == item).astype(float) for item in unique_groups])
    # Uniform rescaling does not change regression improvement. Normalize before
    # squaring to avoid overflow/underflow for power and tiny delay features.
    weights = weights / np.max(weights)
    values = values / np.max(np.abs(values))
    values = values - np.average(values, weights=weights)
    root_w = np.sqrt(weights / np.mean(weights))

    def residual(period_s):
        omega = 2.0 * np.pi / float(period_s)
        phase = omega * centred_t + offset
        columns = [group_columns]
        for harmonic in range(1, harmonics + 1):
            harmonic_phase = (harmonic * omega * centred_t if phase_offset is None
                              else harmonic * phase)
            columns.extend(
                [
                    np.cos(harmonic_phase)[:, None],
                    np.sin(harmonic_phase)[:, None],
                ]
            )
        design = np.column_stack(columns)
        coefficient, _, rank, _ = np.linalg.lstsq(
            design * root_w[:, None], values * root_w, rcond=None
        )
        if rank < parameter_count:
            return np.inf
        error = values - design @ coefficient
        return float(np.sum(weights * error**2))

    baseline_coefficient, *_ = np.linalg.lstsq(
        group_columns * root_w[:, None], values * root_w, rcond=None
    )
    baseline_error = values - group_columns @ baseline_coefficient
    baseline_sse = float(np.sum(weights * baseline_error**2))
    if baseline_sse <= 0.0:
        raise PeriodSearchUnavailable(
            "no_baseline_variation", "特征在基线扣除后无可用于周期搜索的变化", diagnostics
        )
    frequencies = np.linspace(1.0 / max_period, 1.0 / min_period, int(grid_size))
    periods = 1.0 / frequencies
    if search_execution['actual'] == 'cuda:0':
        sse, guard, compute_info = period_grid_sse(centred_t, values, weights, group_columns,
            periods, harmonics, offset if phase_offset is not None else None)
        for index in guard:
            sse[index] = residual(periods[index])
        scores = 1.-sse/baseline_sse
        # Flat/near-tied neighbours are sensitive to floating point peak tests.
        finite_pairs = np.isfinite(scores[:-1]) & np.isfinite(scores[1:])
        differences = np.full(len(scores)-1, np.inf)
        np.subtract(scores[1:], scores[:-1], out=differences, where=finite_pairs)
        close = np.flatnonzero(finite_pairs & (abs(differences) <= 1e-10))
        ambiguous = np.unique(np.r_[close, close+1])
        for index in ambiguous:
            scores[index] = 1.-residual(periods[index])/baseline_sse
        guarded = np.union1d(guard, ambiguous)
        search_execution.update(compute_info, cpu_guard_grid_count=int(len(guarded)))
    else:
        scores = np.asarray([1.0 - residual(period) / baseline_sse for period in periods])
        search_execution.update(kernel='numpy_lstsq_SVD_reference', cpu_guard_grid_count=0)
    finite_scores = np.isfinite(scores)
    diagnostics["rank_deficient_grid_count"] = int(np.count_nonzero(~finite_scores))
    if not np.any(finite_scores):
        raise PeriodSearchUnavailable(
            "rank_deficient_design", "整个搜索网格的周期设计矩阵均退化", diagnostics
        )
    peak_scores = np.where(finite_scores, scores, -np.inf)
    scores[~finite_scores] = np.nan
    peaks, _ = signal.find_peaks(peak_scores)
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(peak_scores))])
    if phase_offset is not None or not legacy_refinement:
        # Enhanced/explicit grid searches include boundary maxima. Legacy
        # calls retain their historical peak selection and refinement.
        peaks = np.unique(np.r_[peaks, 0, len(periods) - 1])
        peaks = peaks[np.isfinite(peak_scores[peaks])]
    if search_execution['actual'] == 'cuda:0':
        # Compare all local maxima with the same reference objective before
        # truncating to five; distant near-tied peaks must not swap at the cut.
        for index in peaks:
            scores[index] = 1.-residual(periods[index])/baseline_sse
        search_execution['cpu_peak_check_count'] = int(len(peaks))
    peaks = peaks[np.argsort(scores[peaks])[::-1]][:5]
    candidates = []
    for index in peaks:
        lower, upper = sorted(
            (periods[min(index + 1, len(periods) - 1)], periods[max(index - 1, 0)])
        )
        period = float(periods[index])
        if refine_period:
            refined = optimize.minimize_scalar(residual, bounds=(lower, upper), method="bounded")
            if refined.success and np.isfinite(refined.fun):
                if (legacy_refinement and phase_offset is None) or refined.fun <= residual(period):
                    period = float(refined.x)
        score = float(1.0 - residual(period) / baseline_sse)
        candidates.append(
            PeriodCandidate(period, score, f"multi_harmonic_ls:H={harmonics}")
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    # This score is a weighted multi-parameter regression improvement, not the
    # normalized single-harmonic Lomb power assumed by _peak_significance.
    # A calibrated block bootstrap would be required for a meaningful FAP.
    false_alarm_probability, significant = np.nan, False
    return PeriodEstimate(
        candidates[0].period_s,
        tuple(candidates),
        periods,
        scores,
        false_alarm_probability,
        significant,
        "unavailable_multi_harmonic_regression",
        diagnostics,
    )


def _peak_significance(times, peak_score, frequencies):
    """白噪声零假设下最高峰的假警报概率（false-alarm probability）。

    注意：该指标只回答“这个峰是不是噪声碰出来的”，**不**回答“这个峰对应的
    周期是否正确”。半周期混叠（P/2）同样是真实的周期性信号，其 FAP 同样可以
    是 0，所以不能拿 FAP 当作“周期测对了”的证据。

    ``scipy.signal.lombscargle(..., normalize=True)`` 配合本函数入口处
    “去均值并除标准差”的数据，在单频点上的功率近似服从尺度为 ``2/N`` 的
    指数分布，因此单频点 p 值为 ``exp(-P * N / 2)``；搜索区间内独立频率
    个数约为 ``(f_max - f_min) * T``。
    """

    n = len(times)
    if n < 3:
        return 1.0, False
    duration = float(times[-1] - times[0])
    independent = max(1, int(round((frequencies[-1] - frequencies[0]) * duration)))
    # FAP = 1 - (1 - p)^M。用 log 空间计算，避免 p 极小时 (1 - p) 被
    # 舍入成 1 而得到假性的 0。
    log_single = -float(peak_score) * n / 2.0
    p = float(np.exp(log_single)) if log_single > -745.0 else 0.0
    false_alarm_probability = -float(np.expm1(independent * np.log1p(-p)))
    # 采用 5% 的常规显著性水平；短观测（如仅覆盖 3 个周期）即使干净信号也
    # 只能达到约 1% 的假警报概率，阈值过严会把正常结果误判为不可靠。
    return min(1.0, max(0.0, false_alarm_probability)), false_alarm_probability < 0.05


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


def translation_coefficients_hz(echo, config):
    """Return optional legacy translation-Doppler coefficients.

    New echo-module datasets already include the full centroid path phase and do
    not write ``translation_coefficients_hz``. In that case inversion should run
    directly on the saved complex echo instead of failing on the removed field.
    """

    if "translation_coefficients_hz" in config:
        return config["translation_coefficients_hz"]
    return echo.metadata.get("translation_coefficients_hz")


def _compensate_raw_centroid_iq(iq, echo, valid_samples):
    """Apply ideal centroid delay and carrier compensation to raw chirp IQ.

    The output fast-time coordinate removes the locally linear centroid delay.
    A row-wise resampling handles the corresponding chirp time-scale change;
    the complex phase then removes the common carrier path at that source time.
    """

    samples = np.asarray(iq)
    pulse_count, fast_count = samples.shape
    fast_time = np.asarray(echo.fast_time_s, dtype=float)
    offsets = np.asarray(
        getattr(echo, "row_fast_time_offset_s", np.zeros(pulse_count)),
        dtype=float,
    )
    if offsets.shape != (pulse_count,) or not np.all(np.isfinite(offsets)):
        offsets = np.zeros(pulse_count, dtype=float)
    rates = np.asarray(
        getattr(echo, "common_path_rate_m_s", np.zeros(pulse_count)),
        dtype=float,
    )
    if rates.shape != (pulse_count,) or not np.all(np.isfinite(rates)):
        rates = np.zeros(pulse_count, dtype=float)
    common_path = np.asarray(echo.tx_range_m, dtype=float) + np.asarray(
        echo.rx_range_m, dtype=float
    )
    if common_path.shape != (pulse_count,):
        raise ValueError("质心运动补偿需要每脉冲 tx_range_m 和 rx_range_m")
    carrier_hz = float(echo.metadata["carrier_frequency_hz"])
    c = 299_792_458.0
    compensated = np.zeros_like(samples)
    for row in range(pulse_count):
        source_axis = offsets[row] + fast_time
        scale = 1.0 - rates[row] / c
        if scale <= 0.0:
            raise ValueError("局部质心路径率产生了非物理的时间映射")
        source_query = source_axis / scale
        real = np.interp(
            source_query, source_axis, samples[row].real, left=0.0, right=0.0
        )
        imag = np.interp(
            source_query, source_axis, samples[row].imag, left=0.0, right=0.0
        )
        path_at_query = common_path[row] + rates[row] * source_query
        cycles = np.remainder(carrier_hz * path_at_query / c, 1.0)
        compensated[row] = (real + 1j * imag) * np.exp(2j * np.pi * cycles)
    return np.where(valid_samples, compensated, 0.0)


@configured()
def estimate_rotation(echo, config, progress_callback: Callable[[str, int, str], None] | None = None):
    try:
        from .dataset import echo_layout, normalize_inversion_policy
    except ImportError:
        from dataset import echo_layout, normalize_inversion_policy

    layout = echo_layout(echo)
    config = normalize_inversion_policy(config, layout)
    if layout == "chirp":
        return _estimate_chirp_rotation(echo, config, progress_callback)
    if progress_callback:
        progress_callback("inversion", 0, "准备反演输入")
    coefficients = translation_coefficients_hz(echo, config)
    if coefficients is None:
        compensated = np.asarray(echo.iq)
    else:
        compensated = compensate(echo.iq, echo.elapsed_s, coefficients)
    if progress_callback:
        progress_callback("inversion", 15, "计算动态频谱")
    if "stft_window_samples" not in config or "stft_overlap_fraction" not in config:
        raise ValueError("CW 反演需要 stft_window_samples 与 stft_overlap_fraction")
    dynamic = stft(
        compensated,
        echo.metadata["sample_rate_hz"],
        config["stft_window_samples"],
        config["stft_overlap_fraction"],
    )
    if progress_callback:
        progress_callback("inversion", 35, "提取频谱特征")
    features = spectral_features(dynamic)
    periods = {}
    feature_items = tuple({
        "total_power": features.total_power,
        "rms_bandwidth": features.rms_bandwidth_hz,
        "centroid": features.centroid_hz,
    }.items())
    for index, (name, values) in enumerate(feature_items, start=1):
        if progress_callback:
            percent = 35 + int(round(55 * (index - 1) / len(feature_items)))
            progress_callback("inversion", percent, f"周期网格搜索：{name}")
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
    if progress_callback:
        progress_callback("inversion", 92, "整理周期候选结果")
    return InversionResult(compensated, dynamic, features, periods)


@configured()
def prepare_chirp_pulse_profiles(echo, config, *, pulse_features_only=False):
    """Compress each chirp pulse and place it on the period-search time axis.

    Aligned cropped rows serve the CPI attempt and display. All per-pulse
    fallback moments use the full native correlation and its row-specific axis.
    """

    raw = np.asarray(echo.iq)
    valid = np.asarray(echo.valid, dtype=bool)
    if valid.ndim == 1:
        valid_samples = np.broadcast_to(valid[:, None], raw.shape)
    elif valid.shape == raw.shape:
        valid_samples = valid
    else:
        raise ValueError("chirp valid 必须是逐脉冲或与二维 IQ 同形状")
    output_reference = str(echo.metadata.get("echo_output_reference", "raw_baseband"))
    if output_reference not in {"raw_baseband", "centroid_compensated"}:
        raise ValueError(f"不支持的 echo_output_reference：{output_reference}")
    # Derive compensation from the echo artifact; config override is expert-only.
    processing = str(config.get("motion_compensation", "auto")).lower()
    already_compensated = output_reference == "centroid_compensated" or bool(
        echo.metadata.get("centroid_common_terms_removed", False)
    )
    if processing == "auto":
        processing = "none" if already_compensated else "centroid_geometry"
    elif processing == "centroid_geometry" and already_compensated:
        # Legacy configs often restated compensation even when echo already removed common terms.
        processing = "none"
    if processing not in {"centroid_geometry", "none"}:
        raise ValueError(f"不支持的 motion_compensation：{processing}")
    compensated_iq = np.where(valid_samples, raw, 0.0)
    if processing == "centroid_geometry" and output_reference == "raw_baseband":
        compensated_iq = _compensate_raw_centroid_iq(
            compensated_iq, echo, valid_samples
        )
    full_compressed = matched_filter_chirp(
        compensated_iq,
        echo.fast_time_s,
        echo.metadata["pulse_width_s"],
        echo.metadata["pulse_bandwidth_hz"],
        baseband_convention=echo.metadata.get("baseband_convention", "centered"),
        full_output=True,
    )
    step = float(np.median(np.diff(echo.fast_time_s)))
    lag_s = matched_filter_group_delay_s(echo.metadata["pulse_width_s"], step)
    lag = int(round(lag_s / step))
    compressed = full_compressed[:, lag:lag + raw.shape[1]]
    row_offsets = getattr(echo, "row_fast_time_offset_s", None)
    fractional = getattr(echo, "centroid_fractional_offset_s", None)
    common_delay = aligned = None
    if not pulse_features_only:
        delay_axes = centroid_delay_axes(
            echo.fast_time_s, len(compressed),
            row_fast_time_offset_s=row_offsets, centroid_fractional_offset_s=fractional,
        )
        common_delay, aligned = align_profiles_to_common_delay(
            compressed, delay_axes, valid=valid_samples,method="fourier"
        )
    full_fast_time = (float(echo.fast_time_s[0])
                      + (np.arange(full_compressed.shape[1]) - lag) * step)
    full_delay_axes = centroid_delay_axes(
        full_fast_time, len(full_compressed),
        row_fast_time_offset_s=row_offsets, centroid_fractional_offset_s=fractional,
    )
    full_energy_width = None
    if not pulse_features_only:
        full_energy_width = power_energy_width_90_s(
            np.abs(full_compressed) ** 2, full_delay_axes
        )
    requested_time_role = str(config.get("period_time_role", "scatter_centroid"))
    # Accept GUI legacy aliases.
    if requested_time_role in {"receive", "receive_elapsed_s"}:
        requested_time_role = "receive_centroid"
    if requested_time_role in {"scatter_elapsed_s"}:
        requested_time_role = "scatter_centroid"
    if requested_time_role == "emit":
        raise ValueError(
            "period_time_role=emit 尚未作为周期搜索时间轴实现；请使用 scatter_centroid 或 receive_centroid"
        )
    if requested_time_role == "scatter_centroid":
        processing_times = np.asarray(
            getattr(echo, "scatter_elapsed_s", echo.elapsed_s), dtype=float
        )
        if processing_times.shape != (len(compressed),) or not np.all(
            np.isfinite(processing_times)
        ):
            raise ValueError(
                "请求 scatter_centroid 时间轴，但 echo 缺少有效的 scatter_elapsed_s"
            )
        time_axis_role = "scatter_elapsed_s"
    elif requested_time_role == "receive_centroid":
        processing_times = np.asarray(echo.elapsed_s, dtype=float)
        time_axis_role = "receive_elapsed_s"
    else:
        raise ValueError(
            "period_time_role 必须是 scatter_centroid 或 receive_centroid"
        )
    run_id = np.asarray(getattr(echo, "run_id", np.array([])), dtype=int)
    if run_id.shape != (len(compressed),):
        run_id = np.zeros(len(compressed), dtype=int)
    row_weight = np.mean(valid_samples, axis=1)
    return {
        "aligned": aligned,
        "processing_times": processing_times,
        "common_delay": common_delay,
        "time_axis_role": time_axis_role,
        "pulse_valid": np.any(valid_samples, axis=1),
        "row_weight": row_weight,
        "run_id": run_id,
        "prf_hz": float(echo.metadata.get("prf_hz", 0.0) or 0.0),
        "profiles": RangeProfiles(processing_times, common_delay, aligned),
        "coherence_id": np.asarray(echo.coherence_id),
        "full_compressed": full_compressed,
        "full_delay_axes": full_delay_axes,
        "full_energy_width_90_s": full_energy_width,
    }


def _native_pulse_range_features(prepared):
    """Compute all pulse moments on the full native matched correlation."""
    features = range_features(
        prepared["processing_times"],
        prepared["full_delay_axes"],
        prepared["full_compressed"],
        delay_axes_s=prepared["full_delay_axes"],
    )
    if prepared["full_energy_width_90_s"] is not None:
        features.energy_width_90_s = prepared["full_energy_width_90_s"]
    return features


def chirp_pulse_range_features(echo, config):
    """Per-pulse native power, centroid and full-correlation energy width."""

    prepared = prepare_chirp_pulse_profiles(echo, config)
    features = _native_pulse_range_features(prepared)
    return features, prepared["time_axis_role"], prepared["run_id"]


# 与脉冲行轴对齐、需要在分块时同步切片的字段。``fast_time_s`` 属于快时间轴，
# 不在其中；只按“第 0 维等于脉冲数”判断会误切恰好等长的快时间轴。
_PULSE_ROW_FIELDS = (
    "iq",
    "clean_iq",
    "valid",
    "row_fast_time_offset_s",
    "centroid_fractional_offset_s",
    "scatter_elapsed_s",
    "elapsed_s",
    "run_id",
    "coherence_id",
    "common_path_rate_m_s",
    "tx_range_m",
    "rx_range_m",
)


def _echo_pulse_chunk(echo, start, stop, pulse_count):
    """``echo`` 的浅拷贝，脉冲行轴限制为 ``[start, stop)``，其余字段共享。

    逐行字段必须是 ndarray 且第一维等于脉冲数：静默跳过未切片的字段会让同一份
    回波因分块大小不同而给出不同结果，甚至把同一行重复计入（``list`` 输入下
    每块都会复制整行）。宁可在这里拒绝。
    """

    chunk = copy.copy(echo)
    for name in _PULSE_ROW_FIELDS:
        value = getattr(echo, name, None)
        if value is None:
            continue
        if not isinstance(value, np.ndarray) or value.ndim < 1:
            raise ValueError(
                f"分块逐脉冲特征需要 {name} 是 ndarray，收到 {type(value).__name__}"
            )
        if value.shape[0] != pulse_count:
            raise ValueError(
                f"分块逐脉冲特征需要 {name} 的第一维等于脉冲数 {pulse_count}，"
                f"收到 {value.shape}"
            )
        setattr(chunk, name, value[start:stop])
    return chunk


def _full_correlation_sample_count(echo) -> int:
    """全相关行长度（= 快时间样点数 + 匹配滤波参考长度 - 1）。"""

    fast_time_s = np.asarray(echo.fast_time_s, dtype=float)
    if fast_time_s.ndim != 1 or len(fast_time_s) < 2:
        raise ValueError("chirp 回波需要至少两个快时间样点")
    step = float(np.median(np.diff(fast_time_s)))
    reference_count = chirp_reference_sample_count(
        float(echo.metadata["pulse_width_s"]), step
    )
    return int(len(fast_time_s)) + int(reference_count) - 1


def bounded_pulse_chunk_size(echo, workspace_budget_bytes) -> int:
    """让全相关复数工作区不超过预算的最大脉冲分块。

    这是内存上界估计，不是数值参数：它只决定循环分几次，不参与任何计算。
    """

    budget = int(workspace_budget_bytes)
    if budget <= 0:
        raise ValueError("workspace_budget_bytes 必须是正整数")
    per_pulse = _full_correlation_sample_count(echo) * np.dtype(np.complex128).itemsize
    return max(1, budget // per_pulse)


def _pulse_feature_source(echo, pulse_count):
    """Validate/fix whole-input fallback decisions before any row selection."""
    centroid_delay_axes(
        np.array([0.0]), pulse_count,
        row_fast_time_offset_s=getattr(echo, "row_fast_time_offset_s", None),
        centroid_fractional_offset_s=getattr(echo, "centroid_fractional_offset_s", None),
    )
    source = copy.copy(echo)
    rates = getattr(echo, "common_path_rate_m_s", None)
    if rates is not None and isinstance(rates, np.ndarray) and rates.shape == (pulse_count,):
        if not np.all(np.isfinite(rates)):
            source.common_path_rate_m_s = np.zeros(pulse_count, dtype=float)
    return source


def streamed_chirp_pulse_features(echo, config, pulse_chunk_size, *, progress_callback=None):
    """按脉冲分块计算逐脉冲特征，数值与 ``chirp_pulse_range_features`` 相同。

    ``range_features`` 的每个观测量都是沿快时间轴对单行做的归约（总能量、
    质心、90% 能量宽度），匹配滤波也只在每行内部做卷积，行间没有耦合。
    因此按行分块只改变峰值内存，不改变任何一行上的运算顺序与结果。
    """

    iq = np.asarray(echo.iq)
    if iq.ndim != 2:
        raise ValueError("分块逐脉冲特征只适用于二维 [pulse, fast_time] 回波")
    pulse_count = int(iq.shape[0])
    if pulse_count == 0:
        raise ValueError("回波没有脉冲，无法计算逐脉冲特征")
    chunk_size = int(pulse_chunk_size)
    if chunk_size < 1:
        raise ValueError("pulse_chunk_size 至少为 1")
    # Offset validation and the legacy all-rows compensation fallback must be
    # decided before slicing. Otherwise a mixed finite/NaN array changes meaning
    # depending on which rows happen to land in the same chunk.
    source = _pulse_feature_source(echo, pulse_count)
    parts = []
    run_ids = []
    time_axis_role = None
    for start in range(0, pulse_count, chunk_size):
        chunk = _echo_pulse_chunk(
            source, start, min(start + chunk_size, pulse_count), pulse_count
        )
        prepared = prepare_chirp_pulse_profiles(chunk, config, pulse_features_only=True)
        parts.append(_native_pulse_range_features(prepared))
        run_ids.append(np.asarray(prepared["run_id"], dtype=int))
        time_axis_role = prepared["time_axis_role"]
        # Release this chunk's 2-D arrays before constructing the next chunk.
        del prepared
        if progress_callback:
            progress_callback(min(start + chunk_size, pulse_count), pulse_count)
    features = RangeFeatures(
        np.concatenate([np.asarray(part.times_s, dtype=float) for part in parts]),
        np.concatenate([np.asarray(part.total_power, dtype=float) for part in parts]),
        np.concatenate([np.asarray(part.centroid_delay_s, dtype=float) for part in parts]),
        np.concatenate([np.asarray(part.energy_width_90_s, dtype=float) for part in parts]),
    )
    return features, time_axis_role, np.concatenate(run_ids)


def _longest_coherent_block(run_id, coherence_id, pulse_valid) -> int:
    """Longest contiguous pulse run that a CPI is allowed to use."""

    runs = np.asarray(run_id, dtype=int)
    groups = np.asarray(coherence_id, dtype=int)
    valid = np.asarray(pulse_valid, dtype=bool)
    if runs.shape != groups.shape or valid.shape != groups.shape:
        return 0
    longest = 0
    keys = np.column_stack((runs, groups))
    for run, group in np.unique(keys, axis=0):
        indices = np.flatnonzero((runs == run) & (groups == group) & valid)
        if len(indices) == 0:
            continue
        split_points = np.flatnonzero(np.diff(indices) != 1) + 1
        for block in np.split(indices, split_points):
            longest = max(longest, len(block))
    return int(longest)


def _global_delay_grid(fast, offsets, *, preserve_support=False):
    """Exact old global median, using a bounded histogram of float intervals.

    Unusual axes with many distinct intervals use a temporary disk-backed
    selection array; no approximate median or per-CPI axis is substituted.
    """
    from collections import Counter
    import tempfile
    fast, offsets = np.asarray(fast), np.asarray(offsets)
    chunk = max(1, (8 * 1024 * 1024) // (8 * len(fast)))
    counts = Counter()
    for start in range(0, len(offsets), chunk):
        axes = fast[None, :] + offsets[start:start + chunk, None]
        intervals = np.diff(axes, axis=1)
        if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
            raise ValueError("距离轴必须有限且严格递增")
        values, numbers = np.unique(intervals, return_counts=True)
        counts.update({float(v): int(n) for v, n in zip(values, numbers)})
        if len(counts) > 4096:
            scratch = Path(__file__).resolve().parents[2] / "tmp"
            scratch.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="cpi_axis_", dir=scratch) as directory:
                size = len(offsets) * (len(fast) - 1)
                mmap = np.memmap(Path(directory) / "intervals.bin", mode="w+", dtype=float, shape=(size,))
                try:
                    position = 0
                    for row in range(0, len(offsets), chunk):
                        part = np.diff(fast[None, :] + offsets[row:row + chunk, None], axis=1).ravel()
                        if not np.all(np.isfinite(part)) or np.any(part <= 0):
                            raise ValueError("距离轴必须有限且严格递增")
                        mmap[position:position + part.size] = part
                        position += part.size
                    step = float(np.median(mmap, overwrite_input=True))
                finally:
                    mmap._mmap.close()
            break
    else:
        ordered = sorted(counts)
        total = sum(counts.values())
        selected, cumulative = [], 0
        targets = ((total - 1) // 2, total // 2)
        for value in ordered:
            previous, cumulative = cumulative, cumulative + counts[value]
            selected.extend(value for target in targets if previous <= target < cumulative)
        step = float(np.mean(selected))
    first = fast[0] + offsets
    count = len(fast)
    begin = float(np.median(first))
    if preserve_support:
        begin = float(np.min(first)) - step
        count = int(np.ceil((np.max(fast[-1] + offsets) + step - begin) / step)) + 1
    return begin + np.arange(count, dtype=float) * step, step


def _streamed_cpi_inputs(echo, config, cpi_pulses, hop, pulse_valid, row_weight, progress_callback):
    """Reduce each CPI immediately; retain only bounded diagnostic spectra."""
    source = _pulse_feature_source(echo, len(echo.iq))
    runs = np.asarray(getattr(echo, 'run_id', np.array([])), dtype=int)
    if runs.shape != (len(echo.iq),):
        runs = np.zeros(len(echo.iq), dtype=int)
    source.run_id = runs
    offsets = centroid_delay_axes(np.array([0.]), len(echo.iq),
        row_fast_time_offset_s=getattr(source, 'row_fast_time_offset_s', None),
        centroid_fractional_offset_s=getattr(source, 'centroid_fractional_offset_s', None))[:, 0]
    fast = np.asarray(echo.fast_time_s)
    common_grid = _global_delay_grid(fast, offsets)
    step = float(np.median(np.diff(fast)))
    reference = chirp_reference_sample_count(echo.metadata['pulse_width_s'], step)
    if cpi_pulses*(len(fast)+reference-1)*16*6>512*1024*1024:
        raise ValueError('单CPI工作区估算超过512 MiB，请缩短CPI；该预算不是进程峰值')
    full_fast = float(fast[0]) + (np.arange(len(fast) + reference - 1) - (reference - 1)) * step
    energy_grid = _global_delay_grid(full_fast, offsets, preserve_support=True)
    groups = np.asarray(echo.coherence_id, dtype=int)
    windows = []
    for run, group in np.unique(np.column_stack((runs, groups)), axis=0):
        indices = np.flatnonzero((runs == run) & (groups == group) & pulse_valid)
        for block in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
            for local in range(0, len(block) - cpi_pulses + 1, hop):
                begin = int(block[local])
                windows.append((begin, begin + cpi_pulses))
    shown = set(np.unique(np.linspace(0, len(windows) - 1, min(128, len(windows)), dtype=int)))
    echo._cpi_windows=np.asarray(windows,dtype=int)
    columns = np.unique(np.linspace(0, len(fast) - 1,
        min(len(fast), 1024, max(1, (128 * 1024 * 1024) // (16 * cpi_pulses * len(shown)))), dtype=int))
    results, previews = [], []
    role = None
    for number, (begin, end) in enumerate(windows):
        chunk = _echo_pulse_chunk(source, begin, end, len(echo.iq))
        native = prepare_chirp_pulse_profiles(chunk, config, pulse_features_only=True)
        role = native['time_axis_role']
        lag = reference - 1
        cropped = native['full_compressed'][:, lag:lag + len(fast)]
        axes = centroid_delay_axes(fast, len(chunk.iq),
            row_fast_time_offset_s=getattr(chunk, 'row_fast_time_offset_s', None),
            centroid_fractional_offset_s=getattr(chunk, 'centroid_fractional_offset_s', None))
        valid = np.asarray(chunk.valid, dtype=bool)
        _, aligned = align_profiles_to_common_delay(cropped, axes, valid=valid, common_grid=common_grid,method="fourier")
        _, energy = align_profiles_to_common_delay(native['full_compressed'], native['full_delay_axes'],
                                                  preserve_support=True, common_grid=energy_grid,method="fourier")
        cube = range_doppler_cube(aligned, native['processing_times'], chunk.coherence_id,
            common_grid[0], cpi_pulses=cpi_pulses, cpi_hop_pulses=hop, valid=pulse_valid[begin:end],
            run_id=runs[begin:end], effective_pulse_weight=row_weight[begin:end],
            energy_iq=energy, energy_delay_axis_s=energy_grid[0],doppler_estimator=config.get('doppler_estimator','spectral_moments'))
        results.append(range_doppler_features(cube))
        if getattr(echo,'_processing_store',None):
            try:
                from .processing_store import save_cpi
            except ImportError:
                from processing_store import save_cpi
            save_cpi(echo,config,cube,number,begin,end)
        cube.cpi_start_pulse += begin
        cube.cpi_end_pulse += begin
        if number in shown:
            cube.spectrum = cube.spectrum[:, columns, :].copy()
            cube.delays_s = common_grid[0][columns]
            cube.energy_range_power = None
            cube.energy_delays_s = None
            previews.append(cube)
        del native, cropped, axes, aligned, energy, cube
        if progress_callback:
            progress_callback('inversion', int(30 * (number + 1) / len(windows)),
                              f'逐 CPI 提取特征 {number + 1}/{len(windows)}')
    features = type(results[0])(**{name: np.concatenate([getattr(item, name) for item in results])
                                 for name in results[0].__dataclass_fields__})
    dynamic = previews[0]
    for name in ('times_s', 'spectrum', 'coherence_id', 'cpi_start_pulse', 'cpi_end_pulse',
                 'effective_sample_count', 'frame_frequencies_hz'):
        setattr(dynamic, name, np.concatenate([getattr(item, name) for item in previews]))
    for name in ('intrinsic_doppler_centroid_hz','intrinsic_rms_bandwidth_hz','doppler_fit_quality',
                 'doppler_window_rms_hz', 'doppler_nyquist_edge_fraction',
                 'doppler_nyquist_edge_half_width_hz'):
        if getattr(dynamic,name) is not None:
            setattr(dynamic,name,np.concatenate([getattr(item,name) for item in previews]))
    return features, dynamic, role


def _compute_chirp_observations(echo, config, progress_callback=None):
    """Invert chirp runs through pulse compression and sliding local CPIs."""

    if progress_callback:
        progress_callback("inversion", 0, "逐脉冲匹配滤波")
    prf_hz = float(echo.metadata.get("prf_hz", 0.0) or 0.0)

    def duration_to_pulses(key, minimum):
        duration = float(config[key])
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{key} 必须是正的有限秒数")
        if prf_hz <= 0.0:
            raise ValueError(f"{key} 需要 echo 元数据中的正 prf_hz")
        return max(minimum, int(np.ceil(duration * prf_hz)))

    cpi_pulses = duration_to_pulses("cpi_duration_s", 2)
    if config.get("doppler_estimator", "spectral_moments") == "narrowband_lag" and cpi_pulses < 12:
        raise ValueError("低频有效多普勒估计至少需要12个相干脉冲；请增大CPI或选择谱功率矩")
    default_hop_duration = float(config["cpi_duration_s"]) / 4.0
    if config.get("cpi_hop_duration_s") is not None:
        cpi_hop = duration_to_pulses("cpi_hop_duration_s", 1)
    else:
        cpi_hop = max(1, int(np.ceil(default_hop_duration * prf_hz)))
    valid = np.asarray(echo.valid, dtype=bool)
    pulse_count = len(echo.iq)
    if valid.ndim == 1 and valid.shape == (pulse_count,):
        pulse_valid, row_weight = valid, valid.astype(float)
    elif valid.shape == echo.iq.shape:
        pulse_valid, row_weight = np.any(valid, axis=1), np.mean(valid, axis=1)
    else:
        raise ValueError("chirp valid 必须是逐脉冲或与二维 IQ 同形状")
    run_id = np.asarray(getattr(echo, "run_id", np.array([])), dtype=int)
    if run_id.shape != (pulse_count,):
        run_id = np.zeros(pulse_count, dtype=int)
    longest = _longest_coherent_block(run_id, echo.coherence_id, pulse_valid)
    fallback_features = None
    cpi_features = None
    if longest < cpi_pulses:
        chunk_size = bounded_pulse_chunk_size(echo, 32 * 1024 * 1024)
        def pulse_progress(done, total):
            if progress_callback:
                progress_callback("inversion", int(30 * done / total),
                                  f"分块提取逐脉冲特征 {done}/{total}")
        fallback_features, time_axis_role, run_id = streamed_chirp_pulse_features(
            echo, config, chunk_size, progress_callback=pulse_progress)
        processing_times = fallback_features.times_s
        # A diagnostic image is not the scientific observation sequence. Select
        # a bounded number of rows only for this image; all pulses above remain
        # in the period search and exported period_observables.npz.
        selection = np.unique(np.linspace(0, pulse_count - 1,
                                          min(pulse_count, chunk_size, 128), dtype=int))
        source = _pulse_feature_source(echo, pulse_count)
        sample = _echo_pulse_chunk(source, 0, pulse_count, pulse_count)
        for name in _PULSE_ROW_FIELDS:
            value = getattr(sample, name, None)
            if value is not None:
                setattr(sample, name, value[selection])
        display = prepare_chirp_pulse_profiles(sample, config)
        columns = np.unique(np.linspace(0, display["aligned"].shape[1] - 1,
                                       min(1024, display["aligned"].shape[1]), dtype=int))
        profiles = RangeProfiles(processing_times[selection], display["common_delay"][columns],
                                 display["aligned"][:, columns].copy())
        aligned, common_delay = profiles.compressed_iq, profiles.delays_s
        del display
        prepared = {"pulse_valid": pulse_valid, "row_weight": row_weight,
                    "coherence_id": np.asarray(echo.coherence_id)}
    else:
        cpi_features, dynamic, time_axis_role = _streamed_cpi_inputs(
            echo, config, cpi_pulses, cpi_hop, pulse_valid, row_weight, progress_callback)
        processing_times = np.asarray(getattr(echo, 'scatter_elapsed_s', echo.elapsed_s) if time_axis_role == 'scatter_elapsed_s'
                                      else echo.elapsed_s)
        aligned, common_delay, profiles = None, dynamic.delays_s, dynamic
        prepared = {"pulse_valid": pulse_valid, "row_weight": row_weight,
                    "coherence_id": np.asarray(echo.coherence_id)}
    if progress_callback:
        progress_callback("inversion", 30, "形成滑动 CPI 距离—多普勒数据")
    notices: list[str] = []
    # Only allocate common full-range rows if a coherent block can form a CPI.
    energy_inputs = {}
    if longest >= cpi_pulses and cpi_features is None:
        energy_delay, energy_aligned = align_profiles_to_common_delay(
            prepared["full_compressed"], prepared["full_delay_axes"], preserve_support=True,method="fourier"
        )
        energy_inputs = {"energy_iq": energy_aligned, "energy_delay_axis_s": energy_delay}
    try:
        if fallback_features is not None:
            raise ValueError("没有任何相干分组包含足够脉冲形成一个 CPI")
        if cpi_features is None:
            dynamic = range_doppler_cube(
            aligned,
            processing_times,
            prepared["coherence_id"],
            common_delay,
            cpi_pulses=cpi_pulses,
            cpi_hop_pulses=cpi_hop,
            valid=prepared["pulse_valid"],
            run_id=run_id,
            effective_pulse_weight=prepared["row_weight"],
            doppler_estimator=config.get('doppler_estimator','spectral_moments'),
            **energy_inputs,
        )
        features = cpi_features if cpi_features is not None else range_doppler_features(dynamic)
        feature_items = (
            ("total_power", features.total_power),
            ("doppler_centroid", features.doppler_centroid_hz),
            ("doppler_bandwidth", features.rms_bandwidth_hz),
            ("range_centroid", features.range_centroid_s),
            ("range_energy_width", features.energy_range_width_90_s),
        )
        feature_times = features.times_s
        processing_path = "cpi"
        weights = np.maximum(features.effective_sample_count, np.finfo(float).eps)
    except ValueError as exc:
        # Only this exact data-shortage condition may use the documented
        # per-pulse fallback.  Configuration and shape errors must propagate.
        if str(exc) != "没有任何相干分组包含足够脉冲形成一个 CPI":
            raise
        notices.append(
            f"CPI 时长 {float(config['cpi_duration_s']):g} s 按脉冲重复频率 {prf_hz:g} Hz "
            f"需要 {cpi_pulses} 个脉冲，每个相干分组最多只有 {longest} 个。"
            "多普勒分析无法进行，本次改用逐脉冲的总能量、距离质心和 90% 能量宽度。"
        )
        dynamic = profiles
        features = fallback_features if fallback_features is not None else _native_pulse_range_features(prepared)
        feature_items = (
            ("total_power", features.total_power),
            ("range_centroid", features.centroid_delay_s),
            ("range_energy_width", features.energy_width_90_s),
        )
        feature_times = features.times_s
        processing_path = "pulse_fallback"
        weights = prepared["row_weight"]
    if progress_callback:
        progress_callback("inversion", 35, "提取带绝对脉冲时标的周期特征")
    return dict(aligned=aligned, dynamic=dynamic, features=features, feature_items=feature_items, feature_times=feature_times, processing_times=processing_times, processing_path=processing_path, weights=weights, time_axis_role=time_axis_role, notices=notices, run_id=run_id, pulse_count=pulse_count, profiles=profiles,cpi_windows=(echo._cpi_windows if processing_path=='cpi' else np.empty((0,2),int)))


@configured()
def prepare_chirp_observations(echo, config, progress_callback=None):
    """One scientific preprocessing stage, shared by preview and period search."""
    store=getattr(echo,'_processing_store',None)
    if store:
        try:
            from .processing_store import load_stage,save_stage,stage_lock
        except ImportError:
            from processing_store import load_stage,save_stage,stage_lock
        with stage_lock(store):
            stage=load_stage(echo,config)
            if stage is not None:
                if progress_callback:progress_callback('inversion',35,'复用已计算的反演中间结果')
                return stage
            stage=_compute_chirp_observations(echo,config,progress_callback)
            saved=dict(stage);saved.pop('feature_items')
            save_stage(echo,config,saved)
            return stage
    stage=_compute_chirp_observations(echo,config,progress_callback)
    # feature_items is reconstructed from the persisted feature fields below.
    return stage


def _estimate_chirp_rotation(echo, config, progress_callback=None):
    stage=prepare_chirp_observations(echo,config,progress_callback)
    aligned=stage['aligned']
    dynamic=stage['dynamic']
    features=stage['features']
    feature_times=stage['feature_times']
    processing_times=stage['processing_times']
    processing_path=stage['processing_path']
    weights=stage['weights']
    time_axis_role=stage['time_axis_role']
    notices=stage['notices']
    run_id=stage['run_id']
    pulse_count=stage['pulse_count']
    profiles=stage['profiles']
    if processing_path=='cpi':
        feature_items=(('total_power',features.total_power),('doppler_centroid',features.doppler_centroid_hz),
            ('doppler_bandwidth',features.rms_bandwidth_hz),('range_centroid',features.range_centroid_s),
            ('range_energy_width',features.energy_range_width_90_s))
    else:
        feature_items=(('total_power',features.total_power),('range_centroid',features.centroid_delay_s),
            ('range_energy_width',features.energy_width_90_s))
    selection = method_options(config)
    phase_offset = None
    if selection["los_phase_compensation"] or config.get("compare_methods", False):
        phase_offset = effective_look_phase(echo, feature_times, config)
    periods = {}
    search_diagnostics = {
        "compute_execution":resolve_device(config),
        "range_alignment":"fourier_zero_guard128",
        "doppler_estimator":config.get('doppler_estimator','spectral_moments'),
        "processing_path": processing_path,
        "baseline_model": "shared_constant",
        "aggregation": "none",
        "time_axis_role": time_axis_role,
        "input_pulse_count": int(len(processing_times)),
        "run_count": int(len(np.unique(run_id))),
        "observation_count": int(len(feature_times)),
        "weight_source": ("effective_sample_count" if processing_path == "cpi"
                          else "valid_sample_fraction"),
        "features": {},
        "method_options": selection,
        "dynamic_preview": ({"sampled": len(profiles.times_s) != pulse_count
                                        or len(profiles.delays_s) != echo.iq.shape[1],
                             "pulse_count": int(len(profiles.times_s)),
                             "fast_sample_count": int(len(profiles.delays_s)),
                             "role": "diagnostic_only"}
                            if processing_path == "pulse_fallback" else
                            {"sampled": len(dynamic.times_s) != len(feature_times)
                                         or len(dynamic.delays_s) != echo.iq.shape[1],
                             "cpi_count": int(len(dynamic.times_s)),
                             "fast_sample_count": int(len(dynamic.delays_s)),
                             "role": "diagnostic_only"}),
        "phase_geometry": ({"axis_source": "inversion_config", "spin_axis_ra_deg": config["spin_axis_ra_deg"],
                            "spin_axis_dec_deg": config["spin_axis_dec_deg"],
                            "period_convention": config.get("phase_period_convention", "half_rotation"),
                            "model": "effective_bistatic_azimuth", "cpi_mapping": "linear_scatter_time_interpolation"}
                           if phase_offset is not None else None),
    }
    path_label = "CPI" if processing_path == "cpi" else "逐脉冲"
    if processing_path == 'cpi':
        def finite_range(values):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return [float(np.min(values)), float(np.median(values)), float(np.max(values))] if len(values) else None
        search_diagnostics['doppler_observability'] = {
            'summary_order': 'minimum_median_maximum',
            'hann_window_rms_hz': finite_range(features.doppler_window_rms_hz),
            'nyquist_edge_power_fraction': finite_range(features.doppler_nyquist_edge_fraction),
            'nyquist_edge_half_width_hz': finite_range(features.doppler_nyquist_edge_half_width_hz),
            'definition': 'window_response_and_boundary_diagnostics_not_intrinsic_width_correction',
        }
        search_diagnostics['range_statistics_support'] = 'full_correlation'
    search_diagnostics['overlap_contract'] = {
        'physical_echo_overlap_pulses': int(np.count_nonzero(getattr(echo, 'signal_echo_overlap', []))),
        'saved_window_overlap_pulses': int(np.count_nonzero(getattr(echo, 'window_overlap', []))),
        'interpretation': 'shared_ADC_samples_are_not_independent_observations',
        'uncertainty_unit': 'whole_Run',
        'overlapping_echo_deconvolution': 'not_implemented',
    }
    if search_diagnostics['overlap_contract']['physical_echo_overlap_pulses']:
        notices.append('存在物理回波重叠；当前脉压未分离叠加脉冲，特征周期可能受相互干扰影响。')
    if processing_path == 'cpi' and config.get('doppler_estimator') == 'narrowband_lag':
        accepted=np.isfinite(features.doppler_centroid_hz)&np.isfinite(features.rms_bandwidth_hz)
        search_diagnostics['doppler_fit_quality']=dict(accepted=int(accepted.sum()),total=len(accepted),
            rejected=int((~accepted).sum()),width_definition='low_frequency_effective_width',
            limitation='Quality checks do not exclude weak high-frequency tails')
        if not np.all(accepted):
            notices.append(f"低频有效多普勒估计：{int((~accepted).sum())}/{len(accepted)}个CPI未通过相干拟合检验，多普勒特征不参与搜索")
    if progress_callback:
        progress_callback("inversion", 35,
                          f"{path_label}周期搜索：共同截距，保留 {len(feature_times)} 个原始特征时刻，不按 Run 聚合")
    for index, (name, values) in enumerate(feature_items, start=1):
        if progress_callback:
            progress_callback("inversion", 35 + int(50 * (index - 1) / len(feature_items)), f"不均匀周期搜索：{name}")
        try:
            estimate = multi_harmonic_lomb_scargle(
                feature_times,
                values,
                config["period_min_s"],
                config["period_max_s"],
                config["period_grid_size"],
                harmonics=selection["harmonics"],
                weights=weights,
                phase_offset=phase_offset if selection["los_phase_compensation"] else None,
                refine_period=config.get("refine_period"),
            )
        except PeriodSearchUnavailable as exc:
            search_diagnostics["features"][name] = {
                **exc.diagnostics, "status": "skipped", "reason": exc.reason,
                "message": str(exc),
            }
            notices.append(f"跳过特征 {name}：{exc}")
            continue
        search_diagnostics["features"][name] = {**estimate.search_diagnostics, "status": "searched"}
        periods[name] = add_harmonics(estimate, config["period_min_s"], config["period_max_s"])
    if not periods:
        reasons = "；".join(f"{name}：{item['message']}"
                           for name, item in search_diagnostics["features"].items())
        raise PeriodSearchUnavailable(
            "no_estimable_features", f"没有可估计周期的特征。{reasons}", search_diagnostics
        )
    comparisons = {}
    if config.get("compare_methods", False):
        for number, (label, options) in enumerate(comparison_options(config), 1):
            if progress_callback:
                progress_callback("inversion", 85 + number, f"方法比较 {number}/8：{label}")
            records = {}
            for name, values in feature_items:
                try:
                    estimate = multi_harmonic_lomb_scargle(
                        feature_times, values, config["period_min_s"], config["period_max_s"],
                        config["period_grid_size"], harmonics=options["harmonics"], weights=weights,
                        phase_offset=phase_offset if options["los_phase_compensation"] else None,
                        refine_period=options["refine_period"])
                    records[name] = {"status": "searched", "best_period_s": estimate.best_period_s,
                                     "score": estimate.candidates[0].score,
                                     "search_diagnostics": estimate.search_diagnostics}
                except PeriodSearchUnavailable as exc:
                    records[name] = {"status": "skipped", "reason": exc.reason, "message": str(exc)}
            comparisons[label] = {"options": options, "features": records}
    if progress_callback:
        progress_callback("inversion", 94, "整理周期候选结果")
    try:
        from .period_uncertainty import assess_period_uncertainty
    except ImportError:
        from period_uncertainty import assess_period_uncertainty
    observation_runs=(run_id[stage['cpi_windows'][:,0]] if processing_path=='cpi' else run_id)
    selection_report = {}
    if config.get('period_selection', 'predictive_joint') == 'predictive_joint':
        try:
            from .joint_period import select_predictive_period
        except ImportError:
            from joint_period import select_predictive_period
        if progress_callback:
            progress_callback('inversion', 94, '共同周期比较：整Run留出预测与稳健融合')
        selection_report = select_predictive_period(
            feature_times, dict(feature_items), observation_runs, periods,
            weights=weights, harmonics=selection['harmonics'],
            phase_offset=phase_offset if selection['los_phase_compensation'] else None,
            min_period=config['period_min_s'], max_period=config['period_max_s'],
            refine_period=selection['refine_period'])
        search_diagnostics['period_selection'] = selection_report['policy']
        if selection_report.get('status') not in (None, 'ok', 'candidate') or selection_report.get('diagnostics', {}).get('validation_mode') == 'shared_fit_fallback':
            notices.append('共同周期预测诊断：' + selection_report.get('message', selection_report['status']))
    else:
        search_diagnostics['period_selection'] = 'legacy_consensus'
    uncertainty=assess_period_uncertainty(feature_times,dict(feature_items),observation_runs,periods,config,
        weights=weights,phase_offset=phase_offset,
        progress=(lambda done,total:progress_callback('inversion',94+int(done/total),f'按Run重采样周期误差 {done}/{total}')) if progress_callback else None)
    return InversionResult(
        aligned, dynamic, features, periods, time_axis_role, tuple(notices), search_diagnostics, comparisons,uncertainty,selection_report
    )
