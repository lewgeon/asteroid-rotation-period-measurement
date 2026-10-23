from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from inversion.src.period_uncertainty import local_run_covariance,assess_period_uncertainty,write_uncertainty_report
from inversion.src.period_consensus import _cross_feature_consensus
from inversion.src.inversion import multi_harmonic_lomb_scargle
from inversion.src.dataset import normalize_inversion_policy


class PeriodUncertaintyTests(unittest.TestCase):
    def test_extreme_feature_scale_preserves_local_uncertainty(self):
        t,y,g=self.fixture()
        expected=local_run_covariance(t,y,g,7.3)['sigma_s']
        for scale in (1e-200,1e200):
            self.assertAlmostEqual(local_run_covariance(t,y*scale,g,7.3)['sigma_s'],expected,places=12)

    def fixture(self):
        rng=np.random.default_rng(123)
        times=np.sort(rng.uniform(0,100,40));runs=np.arange(40)
        values=np.cos(2*np.pi*times/7.3)+rng.normal(0,.1,len(times))
        return times,values,runs

    def test_cluster_covariance_against_independent_score_loop(self):
        t,y,g=self.fixture();period=7.3
        report=local_run_covariance(t,y,g,period)
        span=np.ptp(t);q=(t-t.min())/span;nu=span/period
        y=(y-y.mean())/y.std();phase=2*np.pi*nu*q
        x=np.column_stack((np.ones(len(t)),np.cos(phase),np.sin(phase)))
        beta=np.linalg.lstsq(x,y,rcond=None)[0];r=y-x@beta
        j=np.column_stack((x,2*np.pi*q*(-beta[1]*np.sin(phase)+beta[2]*np.cos(phase))))
        bread=np.linalg.inv(j.T@j);meat=np.zeros((4,4))
        for k in range(40):meat+=np.outer(j[k]*r[k],j[k]*r[k])
        cov=40/39*39/36*bread@meat@bread
        expected=np.sqrt(cov[-1,-1])*span/nu**2
        self.assertAlmostEqual(report['sigma_s'],expected,places=12)
        self.assertEqual(report['status'],'conditional_unvalidated')
        # Duplicating perfectly correlated points within a Run must not buy sqrt(2).
        repeated=local_run_covariance(np.repeat(t,2),np.repeat(y,2),np.repeat(g,2),period)
        self.assertGreater(repeated['sigma_s']/report['sigma_s'],.9)
        self.assertLess(repeated['sigma_s']/report['sigma_s'],1.1)
        amplified=local_run_covariance(t,np.cos(phase)+.6*r,g,period)
        self.assertNotEqual(amplified['sigma_s'],report['sigma_s'])

    def test_whole_run_bootstrap_keeps_correlated_rows_and_reruns_full_search(self):
        from inversion.src import period_uncertainty as implementation
        t,y,runs=self.fixture();t=np.repeat(t,3)+np.tile([0,.001,.002],40)
        y=np.repeat(y,3);runs=np.repeat(runs,3)
        cfg=dict(period_min_s=6.,period_max_s=9.,period_grid_size=80,refine_period=True,
            uncertainty_method='run_bootstrap',uncertainty_replicates=30,uncertainty_seed=12)
        estimate=multi_harmonic_lomb_scargle(t,y,6,9,80,refine_period=True)
        calls=[];original=implementation.multi_harmonic_lomb_scargle
        def checked(*args,**kwargs):
            weights=kwargs['weights'];calls.append(weights.copy())
            for run in range(40):self.assertEqual(len(np.unique(weights[runs==run])),1)
            self.assertEqual(args[2:5],(6.,9.,80))
            return original(*args,**kwargs)
        with patch.object(implementation,'multi_harmonic_lomb_scargle',side_effect=checked):
            report=assess_period_uncertainty(t,{'x':y},runs,{'x':estimate},cfg)
        self.assertEqual(len(calls),30);self.assertTrue(any(np.any(w==0) for w in calls))
        self.assertIsNone(report['final_3sigma_interval_s']);self.assertFalse(report['coverage_calibrated'])
        self.assertEqual(len(report['bootstrap_consensus_samples_s']),30)
        self.assertGreater(np.ptp(report['bootstrap_consensus_samples_s']),0)

    def test_consensus_interface_preserves_median_and_distinct_support(self):
        data={'a':dict(candidates=[dict(period_s=10.,score=.8,source='x')]),
              'b':dict(candidates=[dict(period_s=10.1,score=.5,source='x')])}
        result=_cross_feature_consensus(data)
        self.assertEqual(result[0],10.05);self.assertEqual(result[1],2)

    def test_bootstrap_does_not_mix_cv_and_small_run_fallback(self):
        t = np.repeat(np.linspace(0, 90, 10), 4) + np.tile([0., .1, .2, .3], 10)
        runs = np.repeat(np.arange(10), 4)
        y = np.cos(2*np.pi*t/20) + .02*np.sin(t)
        cfg = dict(period_min_s=18., period_max_s=22., period_grid_size=32,
                   refine_period=True, uncertainty_method='run_bootstrap',
                   uncertainty_replicates=30, uncertainty_seed=12)
        estimate = multi_harmonic_lomb_scargle(t, y, 18, 22, 32, refine_period=True)
        report = assess_period_uncertainty(t, {'x': y}, runs, {'x': estimate}, cfg)
        self.assertEqual(report['baseline_validation_mode'], 'whole_run_cross_validation')
        self.assertEqual(report['bootstrap_mode_switch_failure_count'], 30)
        self.assertEqual(report['status'], 'unavailable')
        self.assertIsNone(report['conditional_same_mode_95_interval_s'])
        self.assertEqual(report['bootstrap_failure_fraction'], 1.)
        from inversion.tests.scratch import scratch_directory
        with scratch_directory('cv_mode_switch') as folder:
            write_uncertainty_report(folder, report)
            content = (folder/'period_uncertainty.html').read_text(encoding='utf-8')
            self.assertIn('全部重采样不可用于', content)
            self.assertNotIn('未启用误差评估', content)

    def test_zero_weight_or_missing_runs_cannot_satisfy_bootstrap_population(self):
        t = np.arange(40, dtype=float); runs = np.repeat(np.arange(10), 4)
        y = np.cos(2*np.pi*t/7.3)
        cfg = dict(period_min_s=6., period_max_s=9., period_grid_size=32,
                   uncertainty_method='run_bootstrap', uncertainty_replicates=30)
        estimate = multi_harmonic_lomb_scargle(t, y, 6, 9, 32)
        for missing in (False, True):
            values = y.copy(); weights = np.ones(len(t))
            if missing:values[runs == 9] = np.nan
            else:weights[runs == 9] = 0
            report = assess_period_uncertainty(t, {'x': values}, runs, {'x': estimate}, cfg, weights)
            self.assertEqual(report['status'], 'unavailable')
            self.assertEqual(report['run_count'], 9)

    def test_policy_and_insufficient_groups(self):
        base=dict(cpi_duration_s=1,period_min_s=1,period_max_s=3,period_grid_size=40)
        for key,value in [('compute_device','auto'),('uncertainty_replicates',29),('uncertainty_seed',True),('uncertainty_method','3sigma')]:
            with self.assertRaises(ValueError):normalize_inversion_policy({**base,key:value},'chirp')
        t,y,g=self.fixture()
        estimate=multi_harmonic_lomb_scargle(t,y,6,9,80)
        report=assess_period_uncertainty(t,{'x':y},g%3,{'x':estimate},{**base,'uncertainty_method':'run_bootstrap'})
        self.assertEqual(report['status'],'unavailable')

    def test_report_displays_unstable_consensus_without_claiming_final_interval(self):
        import json
        from inversion.tests.scratch import scratch_directory
        report=dict(enabled=True,status='experimental_coverage_unvalidated',replicates=100,run_count=100,
            baseline_consensus_period_s=5748.,within_5_percent_of_baseline_fraction=.58,
            bootstrap_failure_fraction=0.,conditional_same_mode_95_interval_s=None,final_3sigma_interval_s=None)
        with scratch_directory('uncertainty_html') as folder:
            write_uncertainty_report(folder,report)
            html=(folder/'period_uncertainty.html').read_text(encoding='utf-8')
            self.assertIn('58%',html);self.assertIn('未通过单峰条件',html)
            self.assertIn('严格总误差上界：尚未建立',html)
            self.assertIn('period_uncertainty.json',html)
            self.assertIsNone(json.loads((folder/'period_uncertainty.json').read_text(encoding='utf-8'))['final_3sigma_interval_s'])
            self.assertIsNone(json.loads((folder/'period_uncertainty.json').read_text(encoding='utf-8'))['deterministic_error_bound_s'])
