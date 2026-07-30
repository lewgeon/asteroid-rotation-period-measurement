import unittest

import numpy as np

from asteroid_rotation.acquisition import AcquisitionSchedule
from asteroid_rotation.echo import RadarGeometry, rotational_doppler_hz


class DynamicsAndEchoTests(unittest.TestCase):
    def test_nonuniform_schedule_preserves_actual_times(self):
        schedule = AcquisitionSchedule.from_epoch_strings(
            [
                "2026-01-01T00:00:00.000",
                "2026-01-01T00:00:01.250",
                "2026-01-01T00:00:04.000",
            ]
        )
        np.testing.assert_allclose(schedule.elapsed_s, [0.0, 1.25, 4.0], atol=1e-9)

    def test_rotational_doppler_matches_analytic_case(self):
        positions = np.array([[1.0, 0.0, 0.0]])
        angular_velocity = np.array([0.0, 0.0, 2.0])
        geometry = RadarGeometry(
            tx_line_of_sight_icrs=np.array([0.0, 1.0, 0.0]),
            rx_line_of_sight_icrs=np.array([0.0, 1.0, 0.0]),
        )
        doppler = rotational_doppler_hz(
            positions,
            angular_velocity,
            geometry,
            wavelength_m=0.5,
        )
        np.testing.assert_allclose(doppler, [8.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

