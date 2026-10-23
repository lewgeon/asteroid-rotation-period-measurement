"""Joint objective controls: truth generates fixtures, never selection inputs."""
import unittest
import numpy as np
from inversion.src.inversion import multi_harmonic_lomb_scargle
from inversion.src.joint_period import select_predictive_period, JointPeriodUnavailable


def fixture():
    rng = np.random.default_rng(1631)
    starts = np.sort(rng.uniform(0, 240, 40))
    t = np.repeat(starts, 3) + np.tile([0, .1, .2], len(starts))
    runs = np.repeat(np.arange(len(starts)), 3)
    good = {f'good{k}': np.cos(2*np.pi*t/20 + k*.7) + .1*rng.normal(size=len(t)) for k in range(3)}
    bad = {f'bad{k}': np.cos(2*np.pi*t/13.7 + k*.4) + .1*rng.normal(size=len(t)) for k in range(2)}
    features = {**good, **bad}
    estimates = {key: multi_harmonic_lomb_scargle(t, val, 10, 26, 220, refine_period=True) for key, val in features.items()}
    return t, runs, features, estimates


class PredictiveJointTests(unittest.TestCase):
    def test_all_missing_run_does_not_enable_cross_validation(self):
        t = np.arange(30, dtype=float) * 1.7
        runs = np.repeat(np.arange(10), 3)
        values = np.cos(2*np.pi*t/20)
        values[runs == 9] = np.nan
        periods = {'x': {'grid_periods_s': np.linspace(16., 24., 40)}}
        result = select_predictive_period(t, {'x': values}, runs, periods)
        valid = runs != 9
        reference = select_predictive_period(t[valid], {'x': values[valid]}, runs[valid], periods)
        self.assertEqual(result['diagnostics']['run_count'], 9)
        self.assertEqual(result['diagnostics']['validation_mode'], 'shared_fit_fallback')
        self.assertAlmostEqual(result['period_s'], reference['period_s'], places=8)

    def test_majority_predictive_channels_reject_conflicting_minority(self):
        t, runs, features, estimates = fixture()
        result = select_predictive_period(t, features, runs, estimates)
        self.assertLess(abs(result['period_s'] - 20), .15)
        self.assertEqual(result['diagnostics']['validation_mode'], 'whole_run_cross_validation')
        self.assertEqual(result['diagnostics']['fold_count'], 5)
        self.assertEqual(result['diagnostics']['unique_channel_count'], 5)
        # Guard the fixture: minority channels prefer a genuinely different period.
        self.assertLess(abs(estimates['bad0'].best_period_s - 13.7), .15)
        self.assertGreater(result['score'], .9)
        self.assertTrue(all(len(set(fold['train_runs']) & set(fold['test_runs'])) == 0
                            for fold in result['diagnostics']['folds']))

    def test_affine_copies_do_not_increase_channel_votes(self):
        t, runs, features, estimates = fixture()
        baseline = select_predictive_period(t, features, runs, estimates)
        for k in range(7):
            features[f'copy{k}'] = (k+2)*features['bad0']+k
            estimates[f'copy{k}'] = estimates['bad0']
        result = select_predictive_period(t, features, runs, estimates)
        self.assertAlmostEqual(result['period_s'], baseline['period_s'], places=8)
        self.assertEqual(result['diagnostics']['unique_channel_count'], 5)
        self.assertEqual(len(result['diagnostics']['duplicate_channels']), 7)

    def test_distinct_finite_masks_are_not_claimed_to_be_deduplicated(self):
        t, runs, features, estimates = fixture()
        features['masked_copy'] = features['bad0'].copy()
        features['masked_copy'][0] = np.nan
        estimates['masked_copy'] = estimates['bad0']
        result = select_predictive_period(t, features, runs, estimates)
        self.assertNotIn('masked_copy', result['diagnostics']['duplicate_channels'])
        self.assertEqual(result['diagnostics']['unique_channel_count'], 6)
        self.assertEqual(result['diagnostics']['duplicate_policy'], 'same_finite_mask_affine_within_1e-10')

    def test_constant_and_exact_linear_trend_have_no_periodic_evidence(self):
        t, runs, features, estimates = fixture()
        features.update(constant=np.ones(len(t)), trend=2+t*.1)
        estimates.update(constant=estimates['good0'], trend=estimates['good0'])
        result = select_predictive_period(t, features, runs, estimates)
        excluded = result['diagnostics']['excluded_channels']
        self.assertEqual(excluded['constant'], 'constant_feature')
        self.assertEqual(excluded['trend'], 'linear_trend_without_periodic_modulation')
        with self.assertRaises(ValueError):
            select_predictive_period(t, {'constant': np.ones(len(t))}, runs, estimates)

    def test_insufficient_runs_is_explicit_shared_fit_fallback(self):
        t, runs, features, estimates = fixture()
        result = select_predictive_period(t, features, runs*0, estimates)
        self.assertEqual(result['diagnostics']['validation_mode'], 'shared_fit_fallback')
        self.assertEqual(result['diagnostics']['fold_count'], 0)
        self.assertIn('insufficient_runs_for_cross_validation', result['warnings'])
        self.assertEqual(result['diagnostics']['score_definition'],
                         'median_channels(1-shared_fit_SSE/shared_mean_baseline_SSE)')

    def test_phase_correction_weights_missing_samples_and_grid_only(self):
        t, runs, features, estimates = fixture()
        offset = -.025*t+.1*np.cos(t/50)
        y = np.cos(2*np.pi*t/20+offset)+.2*np.sin(4*np.pi*t/20+2*offset)
        y[::17] = np.nan
        w = np.ones(len(t)); w[::19] = 0
        estimates['x'] = estimates['good0']
        result = select_predictive_period(t, {'x': y}, runs, estimates,
            weights=w, harmonics=2, phase_offset=offset, refine_period=False)
        corrected = select_predictive_period(t, {'x': y}, runs, estimates,
            weights=w, harmonics=2, phase_offset=offset, refine_period=True)
        uncorrected = select_predictive_period(t, {'x': y}, runs, estimates,
            weights=w, harmonics=2, refine_period=True)
        self.assertIn(result['period_s'], estimates['good0'].grid_periods_s)
        self.assertLess(abs(corrected['period_s']-20), 1e-4)
        self.assertGreater(abs(uncorrected['period_s']-20), 1)

    def test_invalid_configuration_and_weight_contract(self):
        t, runs, features, estimates = fixture()
        for kw in ({'harmonics': True}, {'harmonics': 0}, {'weights': np.ones(2)},
                   {'min_period': 30, 'max_period': 20}, {'phase_offset': np.full(len(t), np.nan)}):
            with self.assertRaises(ValueError):
                select_predictive_period(t, features, runs, estimates, **kw)

    def test_exact_harmonic_alias_is_reported_as_ambiguous(self):
        t, runs, _, _ = fixture()
        y = np.cos(2*np.pi*t/20)
        e = multi_harmonic_lomb_scargle(t, y, 16, 44, 500, harmonics=2, refine_period=True)
        result = select_predictive_period(t, {'signal': y}, runs, {'signal': e}, harmonics=2)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIn('distant_period_has_indistinguishable_score', result['warnings'])
        self.assertTrue(any(abs(c['period_s']-20) < .01 for c in result['candidates']))
        self.assertTrue(any(abs(c['period_s']-40) < .01 for c in result['candidates']))

    def test_boundary_is_not_reported_as_unqualified_candidate(self):
        t, runs, _, _ = fixture()
        y = np.cos(2*np.pi*t/20)
        e = multi_harmonic_lomb_scargle(t, y, 20, 26, 160, refine_period=True)
        result = select_predictive_period(t, {'signal': y}, runs, {'signal': e})
        self.assertEqual(result['period_s'], 20.)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIn('selected_period_at_search_boundary', result['warnings'])

    def test_json_summary_reconstructs_full_grid_without_assuming_candidate_coverage(self):
        t, runs, _, _ = fixture()
        y = np.cos(2*np.pi*t/20)
        # No candidate at the signal period: full search must still recover it.
        summary = {'signal': {'best_period_s': 18., 'candidates': [
            {'period_s': 18., 'score': .1, 'source': 'measured'},
            {'period_s': 22., 'score': .1, 'source': 'measured'}]}}
        result = select_predictive_period(t, {'signal': y}, runs, summary)
        self.assertLess(abs(result['period_s']-20.), 1e-4)
        self.assertEqual(result['diagnostics']['grid_source'], 'reconstructed_4000_point_frequency_grid')

    def test_nonfinite_times_and_nonpositive_weights_are_excluded(self):
        t, runs, features, estimates = fixture()
        weights = np.ones(len(t)); weights[::19] = -1
        corrupted = t.copy(); corrupted[::17] = np.nan
        result = select_predictive_period(corrupted, features, runs, estimates, weights=weights)
        valid = np.isfinite(corrupted) & (weights > 0)
        reference = select_predictive_period(t[valid], {k:v[valid] for k,v in features.items()},
                                            runs[valid], estimates, weights=weights[valid])
        self.assertAlmostEqual(result['period_s'], reference['period_s'], places=8)

    def test_all_rank_degenerate_candidates_use_data_unavailable_exception(self):
        t = np.arange(36.)
        features = {'x': np.cos(2*np.pi*t/7)}
        periods = {'x': {'grid_periods_s': [1., 2.], 'candidates': []}}
        with self.assertRaises(JointPeriodUnavailable):
            select_predictive_period(t, features, np.repeat(np.arange(12), 3), periods)

    def test_extreme_feature_units_and_weight_scale_do_not_change_selection(self):
        t, runs, features, estimates = fixture()
        reference = select_predictive_period(t, features, runs, estimates)
        scales = [1e-150, 1e150, 1e-90, 1e90, 1e-30]
        scaled = {name: value*scale for (name, value), scale in zip(features.items(), scales)}
        result = select_predictive_period(t, scaled, runs, estimates, weights=np.full(len(t), 1e150))
        self.assertAlmostEqual(result['period_s'], reference['period_s'], places=7)

    def test_nonpositive_predictive_skill_is_explicitly_weak(self):
        # Fold-dependent sign changes force the fitted coefficients to predict
        # the held-out fold in the wrong direction, a deterministic red signal.
        starts = np.arange(15)*7.1
        t = np.repeat(starts, 3)+np.tile([0, .1, .2], 15)
        runs = np.repeat(np.arange(15), 3)
        amplitude = np.array([4., -1., -1., -1., -1.])
        y = amplitude[runs % 5]*np.cos(2*np.pi*t/4)
        result = select_predictive_period(t, {'x': y}, runs,
            {'x': {'grid_periods_s': [4., 4.0001]}}, refine_period=False)
        self.assertLessEqual(result['score'], 0.)
        self.assertEqual(result['status'], 'weak')
        self.assertIn('no_positive_median_predictive_skill', result['warnings'])


if __name__ == '__main__':
    unittest.main()
