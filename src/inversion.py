"""Coarse rotation-period inversion from a saved EchoDataset."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import signal

try:
    from .signal import RangeProfiles, compensate, matched_filter_chirp, range_features, spectral_features, stft
except ImportError:  # 支持 PYTHONPATH=src 时按单文件模块运行脚本。
    signal_path = Path(__file__).with_name("signal.py")
    spec = importlib.util.spec_from_file_location("asteroid_radar_signal", signal_path)
    local_signal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_signal)
    compensate = local_signal.compensate
    spectral_features = local_signal.spectral_features
    stft = local_signal.stft
    RangeProfiles = local_signal.RangeProfiles
    matched_filter_chirp = local_signal.matched_filter_chirp
    range_features = local_signal.range_features


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


@dataclass
class InversionResult:
    compensated_iq: np.ndarray
    dynamic_spectrum: object
    features: object
    periods: dict


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


def estimate_rotation(echo, config, progress_callback: Callable[[str, int, str], None] | None = None):
    if echo.metadata.get("data_layout") == "pulse_fast_time" or np.asarray(echo.iq).ndim == 2:
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
    """Invert a campaign of separated chirp acquisitions from range features."""

    if progress_callback:
        progress_callback("inversion", 0, "逐脉冲匹配滤波")
    compressed = matched_filter_chirp(
        echo.iq,
        echo.fast_time_s,
        echo.metadata["pulse_width_s"],
        echo.metadata["pulse_bandwidth_hz"],
    )
    profiles = RangeProfiles(echo.elapsed_s, echo.fast_time_s, compressed)
    if progress_callback:
        progress_callback("inversion", 30, "提取距离像特征")
    features = range_features(echo.elapsed_s, echo.fast_time_s, compressed)
    feature_items = (
        ("total_power", features.total_power),
        ("range_centroid", features.centroid_delay_s),
        ("range_width", features.rms_width_s),
    )
    periods = {}
    for index, (name, values) in enumerate(feature_items, start=1):
        if progress_callback:
            progress_callback("inversion", 30 + int(55 * (index - 1) / len(feature_items)), f"不均匀周期搜索：{name}")
        try:
            estimate = lomb_scargle(
                features.times_s, values, config["period_min_s"],
                config["period_max_s"], config["period_grid_size"],
            )
        except ValueError:
            continue
        periods[name] = add_harmonics(estimate, config["period_min_s"], config["period_max_s"])
    if not periods:
        raise ValueError("匹配滤波后的距离像特征均为常量，无法进行周期反演")
    if progress_callback:
        progress_callback("inversion", 92, "整理周期候选结果")
    return InversionResult(compressed, profiles, features, periods)
