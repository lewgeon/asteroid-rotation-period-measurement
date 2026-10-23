import unittest
from fractions import Fraction
import numpy as np
from scipy.special import ndtr
from inversion.src.gaussian_period import gaussian_simultaneous_budget, gaussian_outer_period_set


class GaussianPeriodTests(unittest.TestCase):
    def test_simultaneous_budget_independent_tail_check_and_outward_product(self):
        sigma = np.array([.001, .002, 0.])
        for alpha in (.2, .0027, 1e-100, np.nextafter(0., 1.)):
            budget = gaussian_simultaneous_budget(sigma, alpha)
            self.assertLessEqual(budget['certified_union_failure_upper_bound'], alpha)
            self.assertLessEqual(2*len(sigma)*ndtr(-budget['k_sigma']), alpha)
            for value, bound in zip(sigma, budget['pointwise_absolute_bounds']):
                self.assertGreaterEqual(Fraction(bound), Fraction(float(value))*budget['k_sigma'])
        self.assertGreater(gaussian_simultaneous_budget(sigma, .0027)['k_sigma'],
                           gaussian_simultaneous_budget(sigma, .2)['k_sigma'])
        # Perfectly correlated/duplicated rows cannot reduce pointwise uncertainty.
        doubled = gaussian_simultaneous_budget(np.tile(sigma, 2))
        self.assertGreaterEqual(doubled['k_sigma'], gaussian_simultaneous_budget(sigma)['k_sigma'])

    def test_declared_gaussian_event_propagates_to_global_cover_and_keeps_alias(self):
        t = np.linspace(0, 90, 24)
        y = .2 + np.cos(2*np.pi*t/7.3) + .3*np.sin(2*np.pi*t/7.3)
        noise = np.random.default_rng(11).normal(0, .001, len(t))
        budget = gaussian_simultaneous_budget(np.full(len(t), .001))
        self.assertTrue(np.all(abs(noise) <= budget['pointwise_absolute_bounds']))
        cover = gaussian_outer_period_set(t, y+noise, 5., 10., [1., 2., 2.],
            {'synthesis_and_model': 1e-12}, np.full(len(t), .001),
            calibration_provenance='控制：生成时已知sigma=.001，外部声明模型与舍入界；非真实雷达',
            resolution_s=.04, max_boxes=3000, estimate_s=7.3)
        for period in (7.3, 1/(23/90-1/7.3)):
            self.assertTrue(any(a <= period <= b for a, b in cover['outer_intervals_s']))
        self.assertFalse(any(a <= 6. <= b for a, b in cover['outer_intervals_s']))
        self.assertGreater(cover['conditional_error_bound_s'], 1.)
        self.assertLessEqual(cover['confidence_lower_bound'], 1-.0027)
        self.assertIsNone(cover['final_3sigma_interval_s'])
        self.assertFalse(cover['physical_model_error_validated'])

    def test_bad_inputs_are_rejected(self):
        for sigma in ([], [-1.], [float('inf')], [[1.]]):
            with self.assertRaises(ValueError): gaussian_simultaneous_budget(sigma)
        for alpha in (True, 0., 1., float('nan')):
            with self.assertRaises(ValueError): gaussian_simultaneous_budget([1.], alpha)
        with self.assertRaises(ValueError): gaussian_simultaneous_budget([np.finfo(float).max])
        common = ([0., 1.], [0., 0.], 5., 10., [1., 2., 2.])
        for components, sigma, provenance in (({}, [1., 1.], 'x'), ({'b': 0.}, [1.], 'x'),
            ({'b': 0.}, [1., 1.], ''), ({'gaussian_simultaneous': 0.}, [1., 1.], 'x')):
            with self.assertRaises(ValueError):
                gaussian_outer_period_set(*common, components, sigma, calibration_provenance=provenance)
