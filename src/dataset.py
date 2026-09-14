"""Read echo datasets produced by the echo simulation module."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class EchoDataset:
    elapsed_s: np.ndarray
    iq: np.ndarray
    clean_iq: np.ndarray
    valid: np.ndarray
    coherence_id: np.ndarray
    tx_los_icrs: np.ndarray
    rx_los_icrs: np.ndarray
    tx_range_m: np.ndarray
    rx_range_m: np.ndarray
    scatter_elapsed_s: np.ndarray
    emit_elapsed_s: np.ndarray
    metadata: Dict[str, Any]
    acquisition_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fast_time_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    run_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    track_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    row_start_sample: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    row_fast_time_offset_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    centroid_fractional_offset_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rx_adc_start_elapsed_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rx_adc_stop_elapsed_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    signal_echo_overlap: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    window_overlap: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    common_path_rate_m_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _metadata_from_npz(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    value = data["metadata_json"]
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def _default_vectors(length: int) -> np.ndarray:
    return np.zeros((length, 3), dtype=float)


def _infer_sample_rate_hz(elapsed_s: np.ndarray):
    if len(elapsed_s) < 2:
        return None
    steps = np.diff(elapsed_s)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        return None
    return float(1.0 / steps[0])


def echo_layout(echo: EchoDataset) -> str:
    """Return 'cw' or 'chirp' from validated dataset shape/metadata."""

    iq = np.asarray(echo.iq)
    layout = str(echo.metadata.get("data_layout", ""))
    if layout in {"pulse_fast_time", "pulse_adc_windows"} or iq.ndim == 2:
        return "chirp"
    if iq.ndim == 1:
        return "cw"
    raise ValueError(f"无法识别回波布局：iq.ndim={iq.ndim}, data_layout={layout!r}")


def normalize_inversion_policy(config: Dict[str, Any], layout: str) -> Dict[str, Any]:
    """Validate and copy the minimal period-search policy for one echo layout."""

    config = dict(config or {})
    # output_path is CLI/IO metadata; strip before policy-key validation.
    config.pop("output_path", None)
    deprecated = {
        "cross_run_phase_coherent", "cpi_pulses", "cpi_hop_pulses",
    } & set(config)
    if deprecated:
        raise ValueError(f"已废弃的 inversion 字段：{', '.join(sorted(deprecated))}")
    role = str(config.get("period_time_role", "scatter_centroid"))
    if role not in {"scatter_centroid", "receive_centroid"}:
        raise ValueError("period_time_role 必须是 scatter_centroid 或 receive_centroid")
    shared = {"period_min_s", "period_max_s", "period_grid_size"}
    cw_only = {
        "stft_window_samples",
        "stft_overlap_fraction",
        "translation_coefficients_hz",
    }
    chirp_only = {
        "harmonics",
        "period_time_role",
        "motion_compensation",
        "cpi_duration_s",
        "cpi_hop_duration_s",
    }
    allowed = set(shared)
    if layout == "cw":
        allowed |= cw_only
    elif layout == "chirp":
        allowed |= chirp_only
        config.setdefault("motion_compensation", "auto")
        if "cpi_duration_s" not in config:
            raise ValueError("chirp inversion 必须提供 cpi_duration_s")
    else:
        raise ValueError(f"未知 inversion layout：{layout}")
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            "当前回波布局不接受 inversion 字段："
            + ", ".join(f"inversion.{key}" for key in unknown)
        )
    return config


def _validate_echo_axes(elapsed_s, iq, fast_time_s, metadata) -> None:
    iq = np.asarray(iq)
    layout = str(metadata.get("data_layout", ""))
    if iq.ndim > 0 and iq.shape[0] == 0:
        raise ValueError("echo IQ 不能为空")
    if iq.ndim == 1:
        if layout in {"pulse_fast_time", "pulse_adc_windows"}:
            raise ValueError(f"一维 IQ 与 chirp data_layout={layout!r} 不一致")
        if len(elapsed_s) != iq.shape[0]:
            raise ValueError("CW 回波 elapsed_s 长度必须等于 iq 长度")
        return
    if iq.ndim != 2:
        raise ValueError(f"不支持的 iq 维数：{iq.ndim}")
    if len(elapsed_s) != iq.shape[0]:
        raise ValueError("chirp 回波 elapsed_s 长度必须等于脉冲数（iq 第 0 维）")
    if layout and layout not in {"pulse_fast_time", "pulse_adc_windows"}:
        raise ValueError(f"二维 IQ 的 data_layout 无效：{layout!r}")
    if len(fast_time_s) != iq.shape[1]:
        raise ValueError("fast_time_s 长度必须等于 iq 第 1 维")


def _validate_echo_dataset(echo: EchoDataset) -> None:
    """Validate all arrays that share the echo row axis."""

    iq = np.asarray(echo.iq)
    sample_count = iq.shape[0]
    if np.asarray(echo.clean_iq).shape != iq.shape:
        raise ValueError("clean_iq 必须与 iq 同形状")
    valid_shape = np.asarray(echo.valid).shape
    if valid_shape not in {(sample_count,), iq.shape}:
        raise ValueError("valid 必须是逐行掩码或与 iq 同形状")
    for name in (
        "coherence_id",
        "acquisition_id",
        "run_id",
        "track_id",
        "row_start_sample",
        "row_fast_time_offset_s",
        "centroid_fractional_offset_s",
        "rx_adc_start_elapsed_s",
        "rx_adc_stop_elapsed_s",
        "signal_echo_overlap",
        "window_overlap",
        "common_path_rate_m_s",
        "tx_range_m",
        "rx_range_m",
        "scatter_elapsed_s",
        "emit_elapsed_s",
    ):
        if np.asarray(getattr(echo, name)).shape != (sample_count,):
            raise ValueError(f"{name} 长度必须等于 iq 第 0 维")
    for name in ("tx_los_icrs", "rx_los_icrs"):
        if np.asarray(getattr(echo, name)).shape != (sample_count, 3):
            raise ValueError(f"{name} 形状必须是 (样本数, 3)")


def load_echo(path) -> EchoDataset:
    """Load the stable ``echo.npz`` exchange format for inversion."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = ["elapsed_s", "iq"]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"回波文件缺少字段：{missing}")

        elapsed_s = np.asarray(data["elapsed_s"], dtype=float)
        if np.any(np.diff(elapsed_s) <= 0):
            raise ValueError("elapsed_s 必须严格递增")

        sample_count = len(elapsed_s)
        metadata = _metadata_from_npz(data)
        iq = np.asarray(data["iq"])
        fast_time_s = (
            np.asarray(data["fast_time_s"], dtype=float)
            if "fast_time_s" in data
            else np.array([], dtype=float)
        )
        _validate_echo_axes(elapsed_s, iq, fast_time_s, metadata)
        if "sample_rate_hz" not in metadata and iq.ndim == 1:
            sample_rate = _infer_sample_rate_hz(elapsed_s)
            if sample_rate is not None:
                metadata["sample_rate_hz"] = sample_rate

        echo = EchoDataset(
            elapsed_s=elapsed_s,
            iq=iq,
            clean_iq=np.asarray(data["clean_iq"]) if "clean_iq" in data else iq,
            valid=np.asarray(data["valid"], dtype=bool)
            if "valid" in data
            else np.ones(sample_count, dtype=bool),
            coherence_id=np.asarray(data["coherence_id"], dtype=int)
            if "coherence_id" in data
            else np.zeros(sample_count, dtype=int),
            acquisition_id=np.asarray(data["acquisition_id"], dtype=int)
            if "acquisition_id" in data
            else np.zeros(sample_count, dtype=int),
            fast_time_s=fast_time_s,
            run_id=np.asarray(data["run_id"], dtype=int)
            if "run_id" in data
            else (
                np.asarray(data["acquisition_id"], dtype=int)
                if "acquisition_id" in data
                else np.zeros(sample_count, dtype=int)
            ),
            track_id=np.asarray(data["track_id"], dtype=int)
            if "track_id" in data
            else np.zeros(sample_count, dtype=int),
            row_start_sample=np.asarray(data["row_start_sample"], dtype=np.int64)
            if "row_start_sample" in data
            else np.full(sample_count, -1, dtype=np.int64),
            row_fast_time_offset_s=np.asarray(data["row_fast_time_offset_s"], dtype=float)
            if "row_fast_time_offset_s" in data
            else np.full(sample_count, np.nan),
            centroid_fractional_offset_s=np.asarray(data["centroid_fractional_offset_s"], dtype=float)
            if "centroid_fractional_offset_s" in data
            else np.full(sample_count, np.nan),
            rx_adc_start_elapsed_s=np.asarray(data["rx_adc_start_elapsed_s"], dtype=float)
            if "rx_adc_start_elapsed_s" in data
            else np.full(sample_count, np.nan),
            rx_adc_stop_elapsed_s=np.asarray(data["rx_adc_stop_elapsed_s"], dtype=float)
            if "rx_adc_stop_elapsed_s" in data
            else np.full(sample_count, np.nan),
            signal_echo_overlap=np.asarray(data["signal_echo_overlap"], dtype=bool)
            if "signal_echo_overlap" in data
            else np.zeros(sample_count, dtype=bool),
            window_overlap=np.asarray(data["window_overlap"], dtype=bool)
            if "window_overlap" in data
            else np.zeros(sample_count, dtype=bool),
            common_path_rate_m_s=np.asarray(data["common_path_rate_m_s"], dtype=float)
            if "common_path_rate_m_s" in data
            else np.full(sample_count, np.nan),
            tx_los_icrs=np.asarray(data["tx_los_icrs"], dtype=float)
            if "tx_los_icrs" in data
            else _default_vectors(sample_count),
            rx_los_icrs=np.asarray(data["rx_los_icrs"], dtype=float)
            if "rx_los_icrs" in data
            else _default_vectors(sample_count),
            tx_range_m=np.asarray(data["tx_range_m"], dtype=float)
            if "tx_range_m" in data
            else np.zeros(sample_count, dtype=float),
            rx_range_m=np.asarray(data["rx_range_m"], dtype=float)
            if "rx_range_m" in data
            else np.zeros(sample_count, dtype=float),
            scatter_elapsed_s=np.asarray(data["scatter_elapsed_s"], dtype=float)
            if "scatter_elapsed_s" in data
            else elapsed_s,
            emit_elapsed_s=np.asarray(data["emit_elapsed_s"], dtype=float)
            if "emit_elapsed_s" in data
            else elapsed_s,
            metadata=metadata,
        )
        _validate_echo_dataset(echo)
        return echo
