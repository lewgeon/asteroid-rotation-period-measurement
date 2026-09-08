"""Coarse rotation-period inversion from a saved EchoDataset."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import optimize, signal

try:
    from .radar_signal import (
        RangeProfiles, align_profiles_to_common_delay, centroid_delay_axes,
        compensate, matched_filter_chirp, range_doppler_cube,
        range_doppler_features, range_features, spectral_features, stft,
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
    matched_filter_chirp = local_signal.matched_filter_chirp
    range_features = local_signal.range_features
    range_doppler_cube = local_signal.range_doppler_cube
    range_doppler_features = local_signal.range_doppler_features
    centroid_delay_axes = local_signal.centroid_delay_axes
    align_profiles_to_common_delay = local_signal.align_profiles_to_common_delay


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


@dataclass
class InversionResult:
    compensated_iq: np.ndarray
    dynamic_spectrum: object
    features: object
    periods: dict
    time_axis_role: str = "receive_elapsed_s"


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
):
    """Generalized multi-harmonic search with an independent baseline per run."""

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    groups = np.zeros(len(times), dtype=int) if groups is None else np.asarray(groups, dtype=int)
    weights = np.ones(len(times), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    times, values, groups, weights = times[finite], values[finite], groups[finite], weights[finite]
    order = np.argsort(times)
    times, values, groups, weights = times[order], values[order], groups[order], weights[order]
    harmonics = int(harmonics)
    unique_groups = np.unique(groups)
    if harmonics < 1:
        raise ValueError("harmonics 必须为正整数")
    parameter_count = 2 * harmonics + len(unique_groups)
    # A saturated design matrix can reproduce every sample and turns the
    # periodogram into an all-ones curve.  Preserve at least three residual
    # degrees of freedom so a candidate period must explain unseen variation.
    if len(values) < max(5, parameter_count + 3) or values.std() == 0.0:
        raise ValueError(
            "多谐波周期估计的有效非恒定特征样本不足；"
            f"当前 {len(values)} 点、{parameter_count} 个拟合参数，至少需要 3 个残差自由度"
        )
    centred_t = times - times[0]
    group_columns = np.column_stack([(groups == item).astype(float) for item in unique_groups])
    root_w = np.sqrt(weights / np.mean(weights))

    def residual(period_s):
        omega = 2.0 * np.pi / float(period_s)
        columns = [group_columns]
        for harmonic in range(1, harmonics + 1):
            columns.extend(
                [
                    np.cos(harmonic * omega * centred_t)[:, None],
                    np.sin(harmonic * omega * centred_t)[:, None],
                ]
            )
        design = np.column_stack(columns)
        coefficient, *_ = np.linalg.lstsq(
            design * root_w[:, None], values * root_w, rcond=None
        )
        error = values - design @ coefficient
        return float(np.sum(weights * error**2))

    baseline_coefficient, *_ = np.linalg.lstsq(
        group_columns * root_w[:, None], values * root_w, rcond=None
    )
    baseline_error = values - group_columns @ baseline_coefficient
    baseline_sse = float(np.sum(weights * baseline_error**2))
    frequencies = np.linspace(1.0 / max_period, 1.0 / min_period, int(grid_size))
    periods = 1.0 / frequencies
    scores = np.asarray([1.0 - residual(period) / baseline_sse for period in periods])
    peaks, _ = signal.find_peaks(scores)
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(scores))])
    peaks = peaks[np.argsort(scores[peaks])[::-1]][:5]
    candidates = []
    for index in peaks:
        lower, upper = sorted(
            (periods[min(index + 1, len(periods) - 1)], periods[max(index - 1, 0)])
        )
        refined = optimize.minimize_scalar(residual, bounds=(lower, upper), method="bounded")
        period = float(refined.x if refined.success else periods[index])
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


def estimate_rotation(echo, config, progress_callback: Callable[[str, int, str], None] | None = None):
    if echo.metadata.get("data_layout") in {"pulse_fast_time", "pulse_adc_windows"} or np.asarray(echo.iq).ndim == 2:
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


def _estimate_chirp_rotation(echo, config, progress_callback=None):
    """Invert chirp runs through pulse compression and sliding local CPIs."""

    if progress_callback:
        progress_callback("inversion", 0, "逐脉冲匹配滤波")
    raw = np.asarray(echo.iq)
    valid = np.asarray(echo.valid, dtype=bool)
    if valid.ndim == 1:
        valid_samples = np.broadcast_to(valid[:, None], raw.shape)
    elif valid.shape == raw.shape:
        valid_samples = valid
    else:
        raise ValueError("chirp valid 必须是逐脉冲或与二维 IQ 同形状")
    processing = str(config.get("motion_compensation", "centroid_geometry"))
    output_reference = str(echo.metadata.get("echo_output_reference", "raw_baseband"))
    if processing not in {"centroid_geometry", "none"}:
        raise ValueError(f"不支持的 motion_compensation：{processing}")
    if output_reference not in {"raw_baseband", "centroid_compensated"}:
        raise ValueError(f"不支持的 echo_output_reference：{output_reference}")
    compensated_iq = np.where(valid_samples, raw, 0.0)
    if processing == "centroid_geometry" and output_reference == "raw_baseband":
        compensated_iq = _compensate_raw_centroid_iq(
            compensated_iq, echo, valid_samples
        )
    compressed = matched_filter_chirp(
        compensated_iq,
        echo.fast_time_s,
        echo.metadata["pulse_width_s"],
        echo.metadata["pulse_bandwidth_hz"],
        baseband_convention=echo.metadata.get("baseband_convention", "centered"),
    )
    row_offsets = getattr(echo, "row_fast_time_offset_s", None)
    fractional = getattr(echo, "centroid_fractional_offset_s", None)
    delay_axes = centroid_delay_axes(
        echo.fast_time_s,
        len(compressed),
        row_fast_time_offset_s=row_offsets,
        centroid_fractional_offset_s=fractional,
    )
    common_delay, aligned = align_profiles_to_common_delay(
        compressed, delay_axes, valid=valid_samples
    )
    requested_time_role = str(config.get("period_time_role", "scatter_centroid"))
    if requested_time_role == "scatter_centroid":
        processing_times = np.asarray(
            getattr(echo, "scatter_elapsed_s", echo.elapsed_s), dtype=float
        )
        if processing_times.shape != (len(compressed),) or not np.all(
            np.isfinite(processing_times)
        ):
            processing_times = np.asarray(echo.elapsed_s, dtype=float)
            time_axis_role = "receive_elapsed_s_fallback"
        else:
            time_axis_role = "scatter_elapsed_s"
    elif requested_time_role == "receive_centroid":
        processing_times = np.asarray(echo.elapsed_s, dtype=float)
        time_axis_role = "receive_elapsed_s"
    else:
        raise ValueError(
            "period_time_role 必须是 scatter_centroid 或 receive_centroid"
        )
    profiles = RangeProfiles(processing_times, common_delay, aligned)
    if progress_callback:
        progress_callback("inversion", 20, "形成滑动 CPI 距离—多普勒数据")
    cpi_pulses = int(config.get("cpi_pulses", 128))
    cpi_hop = int(config.get("cpi_hop_pulses", max(1, cpi_pulses // 4)))
    run_id = np.asarray(getattr(echo, "run_id", np.array([])), dtype=int)
    if run_id.shape != (len(compressed),):
        run_id = np.zeros(len(compressed), dtype=int)
    try:
        dynamic = range_doppler_cube(
            aligned,
            processing_times,
            echo.coherence_id,
            common_delay,
            cpi_pulses=cpi_pulses,
            cpi_hop_pulses=cpi_hop,
            valid=np.any(valid_samples, axis=1),
            run_id=run_id,
            effective_pulse_weight=np.mean(valid_samples, axis=1),
        )
        features = range_doppler_features(dynamic)
        feature_items = (
            ("total_power", features.total_power),
            ("doppler_centroid", features.doppler_centroid_hz),
            ("doppler_bandwidth", features.rms_bandwidth_hz),
            ("range_centroid", features.range_centroid_s),
            ("range_width", features.rms_range_width_s),
        )
        feature_times = features.times_s
        groups = features.coherence_id
        weights = np.maximum(features.effective_sample_count, np.finfo(float).eps)
    except ValueError:
        # Very short smoke-test runs cannot form a CPI.  Retain a documented
        # per-pulse fallback instead of crossing a coherence boundary.
        dynamic = profiles
        features = range_features(
            processing_times,
            common_delay,
            aligned,
            delay_axes_s=common_delay,
        )
        feature_items = (
            ("total_power", features.total_power),
            ("range_centroid", features.centroid_delay_s),
            ("range_width", features.rms_width_s),
        )
        feature_times = features.times_s
        groups = run_id
        weights = np.mean(valid_samples, axis=1)
    if progress_callback:
        progress_callback("inversion", 35, "提取带绝对脉冲时标的周期特征")
    periods = {}
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
                harmonics=int(config.get("harmonics", 1)),
                groups=groups,
                weights=weights,
            )
        except ValueError:
            continue
        periods[name] = add_harmonics(estimate, config["period_min_s"], config["period_max_s"])
    if not periods:
        raise ValueError("匹配滤波后的距离像特征均为常量，无法进行周期反演")
    if progress_callback:
        progress_callback("inversion", 92, "整理周期候选结果")
    return InversionResult(aligned, dynamic, features, periods, time_axis_role)
