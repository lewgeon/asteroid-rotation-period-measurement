"""Coarse rotation-period inversion from a saved EchoDataset."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import signal

try:
    from .signal import compensate, spectral_features, stft
except ImportError:  # 支持 PYTHONPATH=src 时按单文件模块运行脚本。
    signal_path = Path(__file__).with_name("signal.py")
    spec = importlib.util.spec_from_file_location("asteroid_radar_signal", signal_path)
    local_signal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_signal)
    compensate = local_signal.compensate
    spectral_features = local_signal.spectral_features
    stft = local_signal.stft


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
    return PeriodEstimate(candidates[0].period_s, candidates, periods, scores)


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
