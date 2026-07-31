import unittest

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from asteroid_radar.pointing import CartesianState, C_KM_S, LightTimePointingSolver


class LinearTarget:
    def __init__(self, reference, position, velocity):
        self.reference = reference
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

    def state(self, target, epoch):
        elapsed = (epoch.tdb - self.reference.tdb).to_value("sec")
        return CartesianState(
            self.position + self.velocity * elapsed, self.velocity
        )


class FixedStation:
    name = "fixed"
    location = EarthLocation.from_geodetic(0, 0, 0)

    def state(self, epoch):
        return CartesianState(np.zeros(3), np.zeros(3))


class PointingTests(unittest.TestCase):
    def setUp(self):
        self.epoch = Time("2020-01-01", scale="utc")
        self.station = FixedStation()

    def test_stationary_target(self):
        distance = 3e7
        target = LinearTarget(self.epoch, [distance, 0, 0], [0, 0, 0])
        result = LightTimePointingSolver(target, self.station).solve(
            "target", self.epoch, "transmit"
        )
        self.assertAlmostEqual(result.uplink_light_time_s, distance / C_KM_S, 7)
        self.assertAlmostEqual(result.downlink_light_time_s, distance / C_KM_S, 7)

    def test_moving_target(self):
        distance, speed = 6e7, 20.0
        target = LinearTarget(self.epoch, [distance, 0, 0], [speed, 0, 0])
        result = LightTimePointingSolver(target, self.station).solve(
            "target", self.epoch, "transmit"
        )
        self.assertAlmostEqual(
            result.uplink_light_time_s, distance / (C_KM_S - speed), 6
        )

    def test_receive_time_recovers_transmit_time(self):
        target = LinearTarget(self.epoch, [4e7, 2e6, 1e6], [-10, 4, 1])
        solver = LightTimePointingSolver(target, self.station)
        forward = solver.solve("target", self.epoch, "transmit")
        backward = solver.solve("target", forward.receive_time, "receive")
        error = abs((backward.transmit_time.tdb - self.epoch.tdb).to_value("sec"))
        self.assertLess(error, 2e-7)
