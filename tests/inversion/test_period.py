import unittest

import numpy as np

from asteroid_radar.inversion.period import lomb_scargle


class PeriodTests(unittest.TestCase):
    def test_nonuniform_period_recovery(self):
        rng = np.random.default_rng(7)
        times = np.sort(rng.uniform(0, 250, 400))
        truth = 23.0
        values = np.sin(2 * np.pi * times / truth)
        values += 0.05 * rng.standard_normal(len(times))
        estimate = lomb_scargle(times, values, 10, 40, 5000)
        self.assertLess(abs(estimate.best_period_s - truth) / truth, 0.01)
