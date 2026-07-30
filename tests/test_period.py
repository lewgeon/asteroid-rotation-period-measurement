import unittest

import numpy as np

from asteroid_rotation.period import estimate_period_lomb_scargle


class PeriodTests(unittest.TestCase):
    def test_lomb_scargle_recovers_nonuniform_period(self):
        rng = np.random.default_rng(7)
        times = np.sort(rng.uniform(0.0, 250.0, 400))
        truth = 23.0
        values = np.sin(2.0 * np.pi * times / truth)
        values += 0.05 * rng.standard_normal(len(times))
        estimate = estimate_period_lomb_scargle(
            times,
            values,
            min_period_s=10.0,
            max_period_s=40.0,
            grid_size=5000,
        )
        self.assertLess(abs(estimate.best_period_s - truth) / truth, 0.01)


if __name__ == "__main__":
    unittest.main()

