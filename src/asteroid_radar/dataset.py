"""Data exchanged between echo simulation and parameter inversion."""

from dataclasses import dataclass
import json

import numpy as np


@dataclass
class ObservationSchedule:
    reference_time_utc: str
    elapsed_s: np.ndarray
    coherence_id: np.ndarray
    valid: np.ndarray

    @classmethod
    def continuous_wave(cls, start_utc, duration_s, sample_rate_hz):
        elapsed = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
        return cls(
            start_utc,
            elapsed,
            np.zeros(len(elapsed), dtype=int),
            np.ones(len(elapsed), dtype=bool),
        )

    @classmethod
    def from_utc(cls, times_utc, coherence_id=None):
        times = np.asarray(times_utc, dtype="datetime64[ns]")
        elapsed = (times - times[0]) / np.timedelta64(1, "s")
        coherence_id = (
            np.arange(len(times)) if coherence_id is None else np.asarray(coherence_id)
        )
        return cls(times_utc[0], elapsed, coherence_id, np.ones(len(times), dtype=bool))


@dataclass
class EchoDataset:
    elapsed_s: np.ndarray
    iq: np.ndarray
    clean_iq: np.ndarray
    valid: np.ndarray
    coherence_id: np.ndarray
    translation_doppler_hz: np.ndarray
    metadata: dict


def save_echo(path, echo):
    np.savez(
        path,
        elapsed_s=echo.elapsed_s,
        iq=echo.iq,
        clean_iq=echo.clean_iq,
        valid=echo.valid,
        coherence_id=echo.coherence_id,
        translation_doppler_hz=echo.translation_doppler_hz,
        metadata_json=json.dumps(echo.metadata),
    )


def load_echo(path):
    data = np.load(path, allow_pickle=False)
    return EchoDataset(
        data["elapsed_s"],
        data["iq"],
        data["clean_iq"],
        data["valid"],
        data["coherence_id"],
        data["translation_doppler_hz"],
        json.loads(str(data["metadata_json"])),
    )
