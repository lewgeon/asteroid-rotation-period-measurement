"""Physics tests for the standalone three-event pointing solver."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from asteroid_pointing.config import load_pointing_config
from asteroid_pointing.models import CartesianState
from asteroid_pointing.solver import C_KM_S, LightTimePointingSolver


class LinearTarget:
    def __init__(
        self,
        reference: Time,
        position_km: np.ndarray,
        velocity_km_s: np.ndarray,
    ) -> None:
        self.reference = reference
        self.position = np.asarray(position_km, dtype=np.float64)
        self.velocity = np.asarray(velocity_km_s, dtype=np.float64)

    def state(self, target: str, epoch: Time) -> CartesianState:
        del target
        elapsed = float((epoch.tdb - self.reference.tdb).to_value("sec"))
        return CartesianState(
            self.position + self.velocity * elapsed,
            self.velocity,
        )


class FixedStation:
    def __init__(self, position_km: np.ndarray) -> None:
        self.name = "fixed"
        self.position = np.asarray(position_km, dtype=np.float64)
        self.location = EarthLocation.from_geodetic(0.0, 0.0, 0.0)

    def state(self, epoch: Time) -> CartesianState:
        del epoch
        return CartesianState(self.position, np.zeros(3))


class LightTimePointingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.epoch = Time("2020-01-01T00:00:00", scale="utc")
        self.station = FixedStation(np.zeros(3))

    def test_stationary_target_has_equal_two_way_light_times(self) -> None:
        distance_km = 3.0e7
        provider = LinearTarget(
            self.epoch,
            np.array([distance_km, 0.0, 0.0]),
            np.zeros(3),
        )
        solver = LightTimePointingSolver(provider, self.station)
        result = solver.solve("test", self.epoch, time_role="transmit")
        expected = distance_km / C_KM_S
        self.assertAlmostEqual(result.uplink_light_time_s, expected, places=7)
        self.assertAlmostEqual(result.downlink_light_time_s, expected, places=7)
        np.testing.assert_allclose(
            result.transmit_los.unit_icrs, [1.0, 0.0, 0.0], atol=1e-13
        )
        np.testing.assert_allclose(
            result.receive_los.unit_icrs, [1.0, 0.0, 0.0], atol=1e-13
        )

    def test_moving_target_matches_one_dimensional_uplink_solution(self) -> None:
        distance_km = 6.0e7
        speed_km_s = 20.0
        provider = LinearTarget(
            self.epoch,
            np.array([distance_km, 0.0, 0.0]),
            np.array([speed_km_s, 0.0, 0.0]),
        )
        solver = LightTimePointingSolver(provider, self.station)
        result = solver.solve("test", self.epoch, time_role="transmit")
        expected_uplink = distance_km / (C_KM_S - speed_km_s)
        bounce_range = distance_km + speed_km_s * expected_uplink
        expected_downlink = bounce_range / C_KM_S
        self.assertAlmostEqual(
            result.uplink_light_time_s, expected_uplink, places=6
        )
        self.assertAlmostEqual(
            result.downlink_light_time_s, expected_downlink, places=6
        )

    def test_receive_time_role_recovers_transmit_solution(self) -> None:
        provider = LinearTarget(
            self.epoch,
            np.array([4.0e7, 2.0e6, 1.0e6]),
            np.array([-10.0, 4.0, 1.0]),
        )
        solver = LightTimePointingSolver(provider, self.station)
        forward = solver.solve("test", self.epoch, time_role="transmit")
        backward = solver.solve(
            "test", forward.receive_time, time_role="receive"
        )
        difference = abs(
            (backward.transmit_time.tdb - self.epoch.tdb).to_value("sec")
        )
        self.assertLess(difference, 2e-7)
        self.assertLess(backward.bounce_residual_s, 1e-7)
        self.assertLess(backward.endpoint_residual_s, 1e-7)

    def test_invalid_time_role_is_rejected(self) -> None:
        provider = LinearTarget(self.epoch, np.ones(3), np.zeros(3))
        solver = LightTimePointingSolver(provider, self.station)
        with self.assertRaises(ValueError):
            solver.solve("test", self.epoch, time_role="bounce")

    def test_pointing_config_rejects_unknown_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        data = json.loads(
            (root / "configs" / "light_time_pointing.json").read_text(
                encoding="utf-8"
            )
        )
        data["unexpected"] = True
        data["$schema"] = str(
            (root / "schemas" / "light_time_pointing.schema.json").resolve()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pointing_config(path)


if __name__ == "__main__":
    unittest.main()
