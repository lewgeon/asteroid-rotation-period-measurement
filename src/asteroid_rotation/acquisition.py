"""Observation schedules with explicit high-precision timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers


# Reproducible experiments must not trigger an implicit network request.
# A frozen IERS table can be supplied explicitly when Earth-orientation accuracy
# becomes relevant to a real observation.
iers.conf.auto_download = False
iers.conf.auto_max_age = None


@dataclass(frozen=True)
class AcquisitionSchedule:
    """Explicit observation epochs and elapsed seconds from a local reference."""

    epochs: Time
    elapsed_s: np.ndarray
    coherence_id: np.ndarray
    valid_mask: np.ndarray

    @classmethod
    def continuous_wave(
        cls,
        start_time_utc: str,
        duration_s: float,
        sample_rate_hz: float,
    ) -> "AcquisitionSchedule":
        if duration_s <= 0 or sample_rate_hz <= 0:
            raise ValueError("Duration and sample rate must be positive")
        sample_count = int(np.floor(duration_s * sample_rate_hz))
        if sample_count < 2:
            raise ValueError("Observation must contain at least two samples")
        elapsed_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
        start = Time(start_time_utc, scale="utc", format="isot")
        epochs = start + elapsed_s * u.s
        return cls(
            epochs=epochs,
            elapsed_s=elapsed_s,
            coherence_id=np.zeros(sample_count, dtype=np.int32),
            valid_mask=np.ones(sample_count, dtype=bool),
        )

    @classmethod
    def from_epoch_strings(
        cls,
        epoch_times_utc: list[str],
        coherence_id: np.ndarray | None = None,
    ) -> "AcquisitionSchedule":
        """Create an arbitrary, potentially nonuniform pulse schedule."""

        epochs = Time(epoch_times_utc, scale="utc", format="isot")
        if len(epochs) < 2:
            raise ValueError("At least two epochs are required")
        elapsed_s = (epochs - epochs[0]).to_value("sec")
        if np.any(np.diff(elapsed_s) <= 0):
            raise ValueError("Epoch times must be strictly increasing")
        if coherence_id is None:
            coherence = np.arange(len(epochs), dtype=np.int32)
        else:
            coherence = np.asarray(coherence_id, dtype=np.int32)
            if coherence.shape != (len(epochs),):
                raise ValueError("coherence_id must match the epoch count")
        return cls(
            epochs=epochs,
            elapsed_s=np.asarray(elapsed_s, dtype=np.float64),
            coherence_id=coherence,
            valid_mask=np.ones(len(epochs), dtype=bool),
        )
