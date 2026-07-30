"""Three-event light-time equations and line-of-sight conversion."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time, TimeDelta

from .ephemeris import TargetStateProvider, TerrestrialStation
from .models import LightTimeSolution, LineOfSight


C_KM_S = c.to_value(u.km / u.s)


def _offset(epoch: Time, seconds: float) -> Time:
    return epoch + TimeDelta(seconds, format="sec")


def _seconds_between(left: Time, right: Time) -> float:
    return float((left.tdb - right.tdb).to_value(u.s))


def _unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    distance = float(np.linalg.norm(vector))
    return vector / distance, distance


def _line_of_sight(
    station: TerrestrialStation,
    station_epoch: Time,
    vector_icrs_km: np.ndarray,
) -> LineOfSight:
    direction, _ = _unit(vector_icrs_km)
    ra = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    icrs = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    altaz = icrs.transform_to(
        AltAz(
            obstime=station_epoch,
            location=station.location,
            pressure=0 * u.hPa,
        )
    )
    return LineOfSight(
        unit_icrs=direction,
        right_ascension_deg=ra,
        declination_deg=dec,
        azimuth_deg=float(altaz.az.to_value(u.deg)),
        elevation_deg=float(altaz.alt.to_value(u.deg)),
    )


class LightTimePointingSolver:
    """Solve monostatic or bistatic radar light time and pointing."""

    def __init__(
        self,
        target_provider: TargetStateProvider,
        transmit_station: TerrestrialStation,
        receive_station: TerrestrialStation | None = None,
        tolerance_s: float = 1e-7,
        max_iterations: int = 20,
    ) -> None:
        self.target_provider = target_provider
        self.transmit_station = transmit_station
        self.receive_station = receive_station or transmit_station
        self.tolerance_s = tolerance_s
        self.max_iterations = max_iterations

    def solve(
        self,
        target: str,
        observation_time: Time,
        time_role: str = "receive",
    ) -> LightTimeSolution:
        method = {
            "transmit": self._solve_from_transmit,
            "receive": self._solve_from_receive,
        }[time_role]
        return method(target, observation_time)

    def _iterate(
        self,
        initial: Time,
        update,
        label: str,
    ) -> tuple[Time, int, float]:
        estimate = initial
        residual = np.inf
        for iteration in range(1, self.max_iterations + 1):
            updated = update(estimate)
            residual = abs(_seconds_between(updated, estimate))
            estimate = updated
            if residual <= self.tolerance_s:
                return estimate, iteration, residual
        raise RuntimeError(
            f"{label} light-time iteration did not converge after "
            f"{self.max_iterations} iterations; residual={residual:.3e} s"
        )

    def _solve_from_transmit(
        self, target: str, transmit_time: Time
    ) -> LightTimeSolution:
        transmitter = self.transmit_station.state(transmit_time).position_km
        initial_target = self.target_provider.state(
            target, transmit_time
        ).position_km
        bounce_initial = _offset(
            transmit_time, np.linalg.norm(initial_target - transmitter) / C_KM_S
        )

        def update_bounce(epoch: Time) -> Time:
            target_position = self.target_provider.state(
                target, epoch
            ).position_km
            return _offset(
                transmit_time,
                np.linalg.norm(target_position - transmitter) / C_KM_S,
            )

        bounce_time, bounce_iterations, bounce_residual = self._iterate(
            bounce_initial, update_bounce, "uplink"
        )
        target_at_bounce = self.target_provider.state(
            target, bounce_time
        ).position_km
        receive_initial = _offset(
            bounce_time,
            np.linalg.norm(
                self.receive_station.state(bounce_time).position_km
                - target_at_bounce
            )
            / C_KM_S,
        )

        def update_receive(epoch: Time) -> Time:
            receiver = self.receive_station.state(epoch).position_km
            return _offset(
                bounce_time,
                np.linalg.norm(receiver - target_at_bounce) / C_KM_S,
            )

        receive_time, endpoint_iterations, endpoint_residual = self._iterate(
            receive_initial, update_receive, "downlink"
        )
        return self._build_solution(
            target,
            "transmit",
            transmit_time,
            bounce_time,
            receive_time,
            bounce_iterations,
            endpoint_iterations,
            bounce_residual,
            endpoint_residual,
        )

    def _solve_from_receive(
        self, target: str, receive_time: Time
    ) -> LightTimeSolution:
        receiver = self.receive_station.state(receive_time).position_km
        initial_target = self.target_provider.state(
            target, receive_time
        ).position_km
        bounce_initial = _offset(
            receive_time, -np.linalg.norm(initial_target - receiver) / C_KM_S
        )

        def update_bounce(epoch: Time) -> Time:
            target_position = self.target_provider.state(
                target, epoch
            ).position_km
            return _offset(
                receive_time,
                -np.linalg.norm(receiver - target_position) / C_KM_S,
            )

        bounce_time, bounce_iterations, bounce_residual = self._iterate(
            bounce_initial, update_bounce, "downlink"
        )
        target_at_bounce = self.target_provider.state(
            target, bounce_time
        ).position_km
        transmit_initial = _offset(
            bounce_time,
            -np.linalg.norm(
                target_at_bounce
                - self.transmit_station.state(bounce_time).position_km
            )
            / C_KM_S,
        )

        def update_transmit(epoch: Time) -> Time:
            transmitter = self.transmit_station.state(epoch).position_km
            return _offset(
                bounce_time,
                -np.linalg.norm(target_at_bounce - transmitter) / C_KM_S,
            )

        transmit_time, endpoint_iterations, endpoint_residual = self._iterate(
            transmit_initial, update_transmit, "uplink"
        )
        return self._build_solution(
            target,
            "receive",
            transmit_time,
            bounce_time,
            receive_time,
            bounce_iterations,
            endpoint_iterations,
            bounce_residual,
            endpoint_residual,
        )

    def _build_solution(
        self,
        target: str,
        time_role: str,
        transmit_time: Time,
        bounce_time: Time,
        receive_time: Time,
        bounce_iterations: int,
        endpoint_iterations: int,
        bounce_residual_s: float,
        endpoint_residual_s: float,
    ) -> LightTimeSolution:
        transmitter = self.transmit_station.state(transmit_time).position_km
        receiver = self.receive_station.state(receive_time).position_km
        target_at_bounce = self.target_provider.state(
            target, bounce_time
        ).position_km
        uplink_vector = target_at_bounce - transmitter
        receive_pointing_vector = target_at_bounce - receiver
        transmit_los = _line_of_sight(
            self.transmit_station, transmit_time, uplink_vector
        )
        receive_los = _line_of_sight(
            self.receive_station, receive_time, receive_pointing_vector
        )
        uplink_range = float(np.linalg.norm(uplink_vector))
        downlink_range = float(np.linalg.norm(receive_pointing_vector))
        return LightTimeSolution(
            target=target,
            time_role=time_role,
            transmit_time=transmit_time,
            bounce_time=bounce_time,
            receive_time=receive_time,
            uplink_light_time_s=_seconds_between(bounce_time, transmit_time),
            downlink_light_time_s=_seconds_between(receive_time, bounce_time),
            uplink_range_km=uplink_range,
            downlink_range_km=downlink_range,
            transmit_los=transmit_los,
            receive_los=receive_los,
            bounce_iterations=bounce_iterations,
            endpoint_iterations=endpoint_iterations,
            bounce_residual_s=bounce_residual_s,
            endpoint_residual_s=endpoint_residual_s,
        )
