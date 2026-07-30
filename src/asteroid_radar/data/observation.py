"""Observation times shared by continuous-wave and future pulse experiments."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.time import Time


@dataclass
class ObservationSchedule:
    reference_time: Time
    elapsed_s: np.ndarray
    coherence_id: np.ndarray
    valid: np.ndarray

    @classmethod
    def continuous_wave(cls, start_utc, duration_s, sample_rate_hz):
        elapsed = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
        return cls(
            Time(start_utc, scale="utc"),
            elapsed,
            np.zeros(len(elapsed), dtype=int),
            np.ones(len(elapsed), dtype=bool),
        )

    @classmethod
    def from_utc(cls, times_utc, coherence_id=None):
        times = Time(times_utc, scale="utc")
        elapsed = (times - times[0]).to_value(u.s)
        coherence = (
            np.arange(len(times))
            if coherence_id is None
            else np.asarray(coherence_id)
        )
        return cls(times[0], elapsed, coherence, np.ones(len(times), dtype=bool))

    @property
    def epochs(self):
        return self.reference_time + self.elapsed_s * u.s
