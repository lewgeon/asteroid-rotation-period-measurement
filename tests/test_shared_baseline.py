"""Shared-baseline routing and distinguishable search failures."""
import unittest
from unittest.mock import patch

import numpy as np

from inversion.src import inversion as core
from inversion.tests import test_chirp_contracts as fixtures

chirp_echo = fixtures.chirp_echo
CONFIG = fixtures.ChirpInversionTests.config


class SharedBaselineTests(unittest.TestCase):
    def test_single_pulse_runs_keep_every_time_and_use_one_intercept(self):
        echo = chirp_echo("centroid_compensated")
        echo.run_id = np.arange(80)
        echo.coherence_id = echo.run_id.copy()
        config = {**CONFIG, "cpi_duration_s": 20.0}
        native = core.chirp_pulse_range_features(echo, config)[0]
        # The old grouped model really fails on this nonconstant input.
        self.assertGreater(np.ptp(native.total_power), 1.0)
        with self.assertRaises(ValueError):
            core.multi_harmonic_lomb_scargle(
                native.times_s, native.total_power, 1., 3., 80, groups=echo.run_id
            )
        with patch.object(core, "multi_harmonic_lomb_scargle",
                          wraps=core.multi_harmonic_lomb_scargle) as search:
            result = core.estimate_rotation(echo, config)
        for call in search.call_args_list:
            self.assertIsNone(call.kwargs.get("groups"))
            np.testing.assert_array_equal(call.args[0], echo.scatter_elapsed_s)
        self.assertAlmostEqual(result.periods["total_power"].best_period_s, 2., delta=.04)
        diag = result.search_diagnostics
        self.assertEqual(diag["processing_path"], "pulse_fallback")
        self.assertEqual(diag["baseline_model"], "shared_constant")
        self.assertEqual(diag["aggregation"], "none")
        self.assertEqual(diag["observation_count"], 80)
        self.assertEqual(diag["features"]["total_power"]["parameter_count"], 3)

    def test_cpi_with_one_frame_per_run_also_uses_shared_baseline(self):
        echo = chirp_echo("centroid_compensated")
        echo.run_id = np.repeat(np.arange(10), 8)
        echo.coherence_id = echo.run_id.copy()
        config = {**CONFIG,
                  "cpi_duration_s": .8, "cpi_hop_duration_s": .8}
        with patch.object(core, "multi_harmonic_lomb_scargle",
                          wraps=core.multi_harmonic_lomb_scargle) as search:
            result = core.estimate_rotation(echo, config)
        self.assertEqual(result.search_diagnostics["processing_path"], "cpi")
        self.assertEqual(result.search_diagnostics["observation_count"], 10)
        for call in search.call_args_list:
            self.assertIsNone(call.kwargs.get("groups"))
            self.assertEqual(len(call.args[0]), 10)

    def test_common_intercept_grid_matches_independent_weighted_fit(self):
        t = np.array([0., .2, 2., 5., 7., 11., 16., 19., 25., 31.])
        y = 4 + np.cos(2*np.pi*t/7.3) + .1*np.sin(t)
        w = np.linspace(.2, 1., len(t))
        result = core.multi_harmonic_lomb_scargle(t, y, 5., 10., 40, weights=w)
        baseline = np.sum(w*(y-np.average(y, weights=w))**2)
        expected = []
        for period in result.grid_periods_s:
            x = np.column_stack([np.ones(len(t)), np.cos(2*np.pi*t/period),
                                 np.sin(2*np.pi*t/period)])
            beta = np.linalg.lstsq(x*np.sqrt(w[:, None]), y*np.sqrt(w), rcond=None)[0]
            expected.append(1-np.sum(w*(y-x@beta)**2)/baseline)
        np.testing.assert_allclose(result.grid_scores, expected, atol=1e-12)

    def test_failures_are_distinct_and_invalid_configuration_propagates(self):
        cases = [(np.arange(10.), np.ones(10), "constant_feature"),
                 (np.arange(5.), np.arange(5.), "insufficient_samples"),
                 (np.zeros(10), np.arange(10.), "degenerate_time_axis")]
        for t, y, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(core.PeriodSearchUnavailable) as caught:
                    core.multi_harmonic_lomb_scargle(t, y, 1., 3., 20)
                self.assertEqual(caught.exception.reason, reason)
        echo = chirp_echo("centroid_compensated")
        config = {**CONFIG, "harmonics": 1.5}
        with self.assertRaisesRegex(ValueError, "harmonics"):
            core.estimate_rotation(echo, config)

    def test_partial_skip_and_all_failures_keep_diagnostics(self):
        echo = chirp_echo("centroid_compensated")
        config = {**CONFIG, "cpi_duration_s": 20.}
        original = core._native_pulse_range_features
        def features(prepared):
            result = original(prepared)
            result.centroid_delay_s[:] = 1.
            result.energy_width_90_s[:] = np.nan
            return result
        with patch.object(core, "_native_pulse_range_features", side_effect=features):
            result = core.estimate_rotation(echo, config)
        skipped = result.search_diagnostics["features"]
        self.assertEqual(skipped["range_centroid"]["reason"], "constant_feature")
        self.assertEqual(skipped["range_energy_width"]["reason"], "insufficient_samples")
        echo.iq[:] = 0
        with self.assertRaises(core.PeriodSearchUnavailable) as caught:
            core.estimate_rotation(echo, config)
        self.assertIn("total_power", str(caught.exception))
        self.assertIn("range_energy_width", str(caught.exception))
        self.assertEqual(caught.exception.reason, "no_estimable_features")

    def test_unexpected_value_error_is_not_swallowed(self):
        with patch.object(core, "multi_harmonic_lomb_scargle",
                          side_effect=ValueError("bad shape sentinel")):
            with self.assertRaisesRegex(ValueError, "bad shape sentinel"):
                core.estimate_rotation(chirp_echo("centroid_compensated"),
                                       CONFIG)

    def test_short_multi_pulse_runs_preserve_between_run_modulation(self):
        rng = np.random.default_rng(19)
        starts = np.sort(rng.uniform(0., 1800., 60))
        t = (starts[:, None] + np.arange(5)*.1).ravel()
        y = np.cos(2*np.pi*t/300.) + .12*rng.normal(size=len(t))
        groups = np.repeat(np.arange(60), 5)
        common = core.multi_harmonic_lomb_scargle(t, y, 200., 500., 240)
        grouped = core.multi_harmonic_lomb_scargle(t, y, 200., 500., 240, groups=groups)
        self.assertLess(abs(common.best_period_s/300.-1), .01)
        # Guard the fixture: it must expose actual absorption, not two equivalent fits.
        self.assertGreater(common.grid_scores.max() - grouped.grid_scores.max(), .7)

    def test_excluded_samples_and_unsorted_times_have_correct_counts(self):
        t = np.arange(20.)[::-1]
        y = np.cos(2*np.pi*t/7.)
        w = np.ones(20)
        w[0] = 0
        y[1] = np.nan
        result = core.multi_harmonic_lomb_scargle(t, y, 5., 10., 40, weights=w)
        self.assertEqual(result.search_diagnostics["valid_sample_count"], 18)
        self.assertEqual(result.search_diagnostics["excluded_sample_count"], 2)
        self.assertAlmostEqual(result.best_period_s, 7., delta=.01)

    def test_bad_shapes_ranges_grid_and_harmonics_raise_configuration_errors(self):
        t, y = np.arange(10.), np.arange(10.)
        cases = [({"harmonics": 0}, 1., 3., 20),
                 ({"harmonics": True}, 1., 3., 20),
                 ({}, 0., 3., 20), ({}, 3., 1., 20), ({}, 1., np.inf, 20),
                 ({}, 1., 3., 1), ({}, 1., 3., 2.5),
                 ({"weights": np.ones(9)}, 1., 3., 20),
                 ({"groups": np.zeros(9)}, 1., 3., 20)]
        for kwargs, lower, upper, grid in cases:
            with self.subTest(kwargs=kwargs, lower=lower, upper=upper, grid=grid):
                with self.assertRaises(ValueError) as caught:
                    core.multi_harmonic_lomb_scargle(t, y, lower, upper, grid, **kwargs)
                self.assertNotIsInstance(caught.exception, core.PeriodSearchUnavailable)

    def test_six_samples_are_the_common_h1_minimum(self):
        t = np.array([0., .1, 1., 2., 4., 7.])
        result = core.multi_harmonic_lomb_scargle(t, np.cos(2*np.pi*t/3.), 2., 4., 30)
        self.assertEqual(result.search_diagnostics["parameter_count"], 3)
        self.assertAlmostEqual(result.best_period_s, 3., delta=.01)

    def test_regression_is_invariant_to_extreme_feature_and_weight_scales(self):
        t = np.linspace(0., 50., 40)
        y = 2. + np.cos(2*np.pi*t/7.3)
        w = np.linspace(.2, 1., len(t))
        reference = core.multi_harmonic_lomb_scargle(t, y, 5., 10., 80, weights=w)
        for scale in (1e-170, 1e-100, 1e-12, 1e160):
            with self.subTest(scale=scale):
                got = core.multi_harmonic_lomb_scargle(t, y*scale, 5., 10., 80, weights=w*1e200)
                np.testing.assert_allclose(got.grid_scores, reference.grid_scores, atol=1e-12)
                self.assertAlmostEqual(got.best_period_s, reference.best_period_s, places=7)

    def test_repeated_times_do_not_create_identifiable_period_information(self):
        t = np.tile([0., 1., 2.], 10)
        with self.assertRaises(core.PeriodSearchUnavailable) as caught:
            core.multi_harmonic_lomb_scargle(t, np.sin(t), 5., 10., 40)
        self.assertEqual(caught.exception.reason, "degenerate_time_axis")

    def test_rank_deficient_grid_is_reported_instead_of_arbitrary_period(self):
        # Harmonic columns repeat at each deliberately integer-cycle time.
        t = np.arange(12.)
        with self.assertRaises(core.PeriodSearchUnavailable) as caught:
            core.multi_harmonic_lomb_scargle(t, np.cos(t), .5, 1., 2)
        self.assertEqual(caught.exception.reason, "rank_deficient_design")

    def test_constant_group_offsets_have_no_baseline_variation(self):
        groups = np.repeat(np.arange(4), 5)
        with self.assertRaises(core.PeriodSearchUnavailable) as caught:
            core.multi_harmonic_lomb_scargle(np.arange(20.), .1*(groups+1),
                                           5., 10., 40, groups=groups)
        self.assertEqual(caught.exception.reason, "no_baseline_variation")


if __name__ == "__main__":
    unittest.main()
