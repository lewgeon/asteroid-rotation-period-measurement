"""Results returned by the pointing module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.time import Time


@dataclass(frozen=True)
class CartesianState:
    """Barycentric ICRS Cartesian state in km and km/s."""

    position_km: np.ndarray
    velocity_km_s: np.ndarray

@dataclass(frozen=True)
class LineOfSight:
    """A station-to-reflection-point pointing direction."""

    unit_icrs: np.ndarray
    right_ascension_deg: float
    declination_deg: float
    azimuth_deg: float
    elevation_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_vector_icrs": self.unit_icrs.tolist(),
            "right_ascension_deg": self.right_ascension_deg,
            "declination_deg": self.declination_deg,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


@dataclass(frozen=True)
class LightTimeSolution:
    """Three-event solution for transmission, reflection, and reception."""

    target: str
    time_role: str
    transmit_time: Time
    bounce_time: Time
    receive_time: Time
    uplink_light_time_s: float
    downlink_light_time_s: float
    uplink_range_km: float
    downlink_range_km: float
    transmit_los: LineOfSight
    receive_los: LineOfSight
    bounce_iterations: int
    endpoint_iterations: int
    bounce_residual_s: float
    endpoint_residual_s: float

    @staticmethod
    def _time_dict(value: Time) -> dict[str, str]:
        return {
            "utc": value.utc.isot,
            "tdb": value.tdb.isot,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "input_time_role": self.time_role,
            "events": {
                "transmit": self._time_dict(self.transmit_time),
                "bounce": self._time_dict(self.bounce_time),
                "receive": self._time_dict(self.receive_time),
            },
            "light_time_s": {
                "uplink": self.uplink_light_time_s,
                "downlink": self.downlink_light_time_s,
                "round_trip": self.uplink_light_time_s
                + self.downlink_light_time_s,
            },
            "range_km": {
                "uplink": self.uplink_range_km,
                "downlink": self.downlink_range_km,
            },
            "transmit_line_of_sight": self.transmit_los.to_dict(),
            "receive_line_of_sight": self.receive_los.to_dict(),
            "convergence": {
                "bounce_iterations": self.bounce_iterations,
                "endpoint_iterations": self.endpoint_iterations,
                "bounce_residual_s": self.bounce_residual_s,
                "endpoint_residual_s": self.endpoint_residual_s,
            },
            "conventions": {
                "vector_frame": "ICRS axes, Solar-System-barycentric origin",
                "line_of_sight_direction": (
                    "from station at its event time toward target at bounce time"
                ),
                "receive_wave_propagation_direction": (
                    "opposite to receive_line_of_sight.unit_vector_icrs"
                ),
                "azimuth": "north through east",
                "elevation": "vacuum AltAz (pressure=0)",
            },
        }
