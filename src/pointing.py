"""Three-event light time and transmit/receive line of sight."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import TimeDelta
from astropy.utils import iers


iers.conf.auto_download = False
iers.conf.auto_max_age = None


C_KM_S = c.to_value(u.km / u.s)


@dataclass(frozen=True)
class CartesianState:
    position_km: np.ndarray
    velocity_km_s: np.ndarray


@dataclass(frozen=True)
class LineOfSight:
    unit_icrs: np.ndarray
    right_ascension_deg: float
    declination_deg: float
    azimuth_deg: float
    elevation_deg: float

    def to_dict(self):
        return {
            "unit_vector_icrs": self.unit_icrs.tolist(),
            "right_ascension_deg": self.right_ascension_deg,
            "declination_deg": self.declination_deg,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


@dataclass(frozen=True)
class LightTimeSolution:
    target: str
    time_role: str
    transmit_time: object
    bounce_time: object
    receive_time: object
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

    def to_dict(self):
        def event(time):
            return {"utc": time.utc.isot, "tdb": time.tdb.isot}

        return {
            "target": self.target,
            "input_time_role": self.time_role,
            "events": {
                "transmit": event(self.transmit_time),
                "bounce": event(self.bounce_time),
                "receive": event(self.receive_time),
            },
            "light_time_s": {
                "uplink": self.uplink_light_time_s,
                "downlink": self.downlink_light_time_s,
                "round_trip": self.uplink_light_time_s + self.downlink_light_time_s,
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
        }


def offset(epoch, seconds):
    return epoch + TimeDelta(seconds, format="sec")


def seconds_between(left, right):
    return float((left.tdb - right.tdb).to_value(u.s))


def line_of_sight(station, epoch, vector):
    direction = vector / np.linalg.norm(vector)
    ra = np.degrees(np.arctan2(direction[1], direction[0])) % 360
    dec = np.degrees(np.arcsin(direction[2]))
    altaz = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(
        AltAz(obstime=epoch, location=station.location, pressure=0 * u.hPa)
    )
    return LineOfSight(
        direction,
        float(ra),
        float(dec),
        float(altaz.az.to_value(u.deg)),
        float(altaz.alt.to_value(u.deg)),
    )


class LightTimePointingSolver:
    def __init__(
        self,
        target_states,
        transmit_station,
        receive_station=None,
        tolerance_s=1e-7,
        max_iterations=20,
    ):
        self.target_states = target_states
        self.tx_station = transmit_station
        self.rx_station = receive_station or transmit_station
        self.tolerance_s = tolerance_s
        self.max_iterations = max_iterations

    def solve(self, target, observation_time, time_role="receive"):
        method = {
            "transmit": self._from_transmit,
            "receive": self._from_receive,
        }[time_role]
        return method(target, observation_time)

    def iterate(self, initial, update):
        estimate = initial
        for iteration in range(1, self.max_iterations + 1):
            updated = update(estimate)
            residual = abs(seconds_between(updated, estimate))
            estimate = updated
            if residual <= self.tolerance_s:
                return estimate, iteration, residual
        raise RuntimeError(f"光行时迭代未收敛，残差为 {residual:.3e} s")

    def _from_transmit(self, target, tx_time):
        tx_position = self.tx_station.state(tx_time).position_km
        target_now = self.target_states.state(target, tx_time).position_km
        bounce_guess = offset(
            tx_time, np.linalg.norm(target_now - tx_position) / C_KM_S
        )

        def update_bounce(time):
            target_position = self.target_states.state(target, time).position_km
            return offset(
                tx_time, np.linalg.norm(target_position - tx_position) / C_KM_S
            )

        bounce_time, bounce_iterations, bounce_residual = self.iterate(
            bounce_guess, update_bounce
        )
        target_position = self.target_states.state(target, bounce_time).position_km
        receive_guess = offset(
            bounce_time,
            np.linalg.norm(
                self.rx_station.state(bounce_time).position_km - target_position
            )
            / C_KM_S,
        )

        def update_receive(time):
            return offset(
                bounce_time,
                np.linalg.norm(
                    self.rx_station.state(time).position_km - target_position
                )
                / C_KM_S,
            )

        rx_time, endpoint_iterations, endpoint_residual = self.iterate(
            receive_guess, update_receive
        )
        return self.build_solution(
            target,
            "transmit",
            tx_time,
            bounce_time,
            rx_time,
            bounce_iterations,
            endpoint_iterations,
            bounce_residual,
            endpoint_residual,
        )

    def _from_receive(self, target, rx_time):
        rx_position = self.rx_station.state(rx_time).position_km
        target_now = self.target_states.state(target, rx_time).position_km
        bounce_guess = offset(
            rx_time, -np.linalg.norm(target_now - rx_position) / C_KM_S
        )

        def update_bounce(time):
            target_position = self.target_states.state(target, time).position_km
            return offset(
                rx_time, -np.linalg.norm(rx_position - target_position) / C_KM_S
            )

        bounce_time, bounce_iterations, bounce_residual = self.iterate(
            bounce_guess, update_bounce
        )
        target_position = self.target_states.state(target, bounce_time).position_km
        transmit_guess = offset(
            bounce_time,
            -np.linalg.norm(
                target_position - self.tx_station.state(bounce_time).position_km
            )
            / C_KM_S,
        )

        def update_transmit(time):
            return offset(
                bounce_time,
                -np.linalg.norm(
                    target_position - self.tx_station.state(time).position_km
                )
                / C_KM_S,
            )

        tx_time, endpoint_iterations, endpoint_residual = self.iterate(
            transmit_guess, update_transmit
        )
        return self.build_solution(
            target,
            "receive",
            tx_time,
            bounce_time,
            rx_time,
            bounce_iterations,
            endpoint_iterations,
            bounce_residual,
            endpoint_residual,
        )

    def build_solution(
        self,
        target,
        role,
        tx_time,
        bounce_time,
        rx_time,
        bounce_iterations,
        endpoint_iterations,
        bounce_residual,
        endpoint_residual,
    ):
        tx_position = self.tx_station.state(tx_time).position_km
        rx_position = self.rx_station.state(rx_time).position_km
        target_position = self.target_states.state(target, bounce_time).position_km
        uplink = target_position - tx_position
        downlink_pointing = target_position - rx_position
        return LightTimeSolution(
            target,
            role,
            tx_time,
            bounce_time,
            rx_time,
            seconds_between(bounce_time, tx_time),
            seconds_between(rx_time, bounce_time),
            float(np.linalg.norm(uplink)),
            float(np.linalg.norm(downlink_pointing)),
            line_of_sight(self.tx_station, tx_time, uplink),
            line_of_sight(self.rx_station, rx_time, downlink_pointing),
            bounce_iterations,
            endpoint_iterations,
            bounce_residual,
            endpoint_residual,
        )
