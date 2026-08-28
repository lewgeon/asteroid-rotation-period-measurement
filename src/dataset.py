"""Read echo datasets produced by the echo simulation module."""

from dataclasses import dataclass
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
        if "sample_rate_hz" not in metadata:
            sample_rate = _infer_sample_rate_hz(elapsed_s)
            if sample_rate is not None:
                metadata["sample_rate_hz"] = sample_rate

        return EchoDataset(
            elapsed_s=elapsed_s,
            iq=np.asarray(data["iq"]),
            clean_iq=np.asarray(data["clean_iq"]) if "clean_iq" in data else np.asarray(data["iq"]),
            valid=np.asarray(data["valid"], dtype=bool)
            if "valid" in data
            else np.ones(sample_count, dtype=bool),
            coherence_id=np.asarray(data["coherence_id"], dtype=int)
            if "coherence_id" in data
            else np.zeros(sample_count, dtype=int),
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
