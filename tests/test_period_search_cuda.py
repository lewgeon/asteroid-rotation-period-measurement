"""Nondegenerate CPU/CUDA period-grid and full-search contracts."""
import unittest
from unittest.mock import patch
import numpy as np
from inversion.src import compute_backend as backend
from inversion.src.inversion import multi_harmonic_lomb_scargle


def fixture():
    t = np.sort(np.random.default_rng(615).uniform(0., 93., 137))
    offset = -.037*t+.1*np.sin(t/9)
    y = 2+np.cos(2*np.pi*t/7.31+offset)+.37*np.sin(4*np.pi*t/7.31+2*offset)
    return t, y, np.linspace(.3, 1.7, len(t)), offset


class PeriodSearchDeviceTests(unittest.TestCase):
    def test_explicit_no_cuda_fallback(self):
        import torch
        t, y, w, offset = fixture()
        cpu = multi_harmonic_lomb_scargle(t, y, 6, 9, 120)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = multi_harmonic_lomb_scargle(t, y, 6, 9, 120, compute_device='cuda:0')
        np.testing.assert_array_equal(result.grid_scores, cpu.grid_scores)
        self.assertEqual(result.search_diagnostics['period_compute']['actual'], 'cpu')
        self.assertIsNotNone(result.search_diagnostics['period_compute']['reason'])


class CUDAPeriodGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available(): raise unittest.SkipTest('CUDA unavailable')

    def test_distinct_harmonic_phase_refinement_branches_against_cpu_reference(self):
        t, y, w, offset = fixture()
        outcomes = []
        for h in (1, 2):
            for phase in (None, offset):
                for refine in (False, True):
                    kw = dict(harmonics=h, weights=w, phase_offset=phase, refine_period=refine)
                    cpu = multi_harmonic_lomb_scargle(t, y, 6, 9, 300, **kw)
                    gpu = multi_harmonic_lomb_scargle(t, y, 6, 9, 300, compute_device='cuda:0', **kw)
                    np.testing.assert_allclose(gpu.grid_scores, cpu.grid_scores, rtol=0, atol=2e-11)
                    self.assertEqual(len(gpu.candidates), len(cpu.candidates))
                    np.testing.assert_allclose([c.period_s for c in gpu.candidates],
                                               [c.period_s for c in cpu.candidates], rtol=0, atol=1e-10)
                    self.assertEqual(gpu.search_diagnostics['period_compute']['actual'], 'cuda:0')
                    outcomes.append(cpu.best_period_s)
        self.assertEqual(len(np.unique(np.round(outcomes, 7))), 8)

    def test_independent_orthogonal_fourier_golden_and_boundaries(self):
        # Over exactly four seconds, the .5 Hz and 1 Hz modes are orthogonal.
        # Their powers are 1/2 and 1/8, so explained fractions are .8 and .2.
        t=np.arange(16)/4
        y=np.cos(np.pi*t)+.5*np.sin(2*np.pi*t)
        # H2 at a 1s fundamental includes a Nyquist sine column: rank deficient.
        expected={1:np.array([0.,.8,0.,.2]),2:np.array([.8,1.,0.,np.nan])}
        for h in (1,2):
            e=multi_harmonic_lomb_scargle(t,y,1.,4.,4,harmonics=h,
                refine_period=False,compute_device='cuda:0')
            np.testing.assert_allclose(e.grid_scores,expected[h],atol=2e-14,rtol=0)
        # A true maximum at the search boundary remains a boundary candidate.
        e=multi_harmonic_lomb_scargle(t,np.cos(np.pi*t),2.,4.,100,
            refine_period=True,compute_device='cuda:0')
        self.assertEqual(e.best_period_s,2.)

    def test_extreme_scales_weights_and_ill_conditioned_fallback(self):
        t, y, w, offset = fixture()
        # Long periods make the intercept and cosine columns almost identical.
        for lo, hi in ((6., 9.), (1e8, 2e8)):
            for scale in (1e-200, 1., 1e200):
                local = y*scale
                weights = w.copy(); weights[::5] = 0
                a = multi_harmonic_lomb_scargle(t, local, lo, hi, 80, weights=weights, refine_period=False)
                b = multi_harmonic_lomb_scargle(t, local, lo, hi, 80, weights=weights,
                    refine_period=False, compute_device='cuda:0')
                np.testing.assert_allclose(a.grid_scores, b.grid_scores, atol=2e-11, rtol=0, equal_nan=True)
                if lo > 1e6:
                    self.assertGreater(b.search_diagnostics['period_compute']['cpu_guard_grid_count'], 0)

    def test_grouped_research_and_context_routing(self):
        t, y, w, offset = fixture()
        groups = np.arange(len(t))%3
        y = y+groups*.7
        cpu = multi_harmonic_lomb_scargle(t, y, 6, 9, 150, weights=w, groups=groups)
        with backend.execution({'compute_device': 'cuda:0'}):
            gpu = multi_harmonic_lomb_scargle(t, y, 6, 9, 150, weights=w, groups=groups)
        np.testing.assert_allclose(cpu.grid_scores, gpu.grid_scores, atol=2e-11, rtol=0)
        self.assertEqual(cpu.best_period_s, gpu.best_period_s)

    def test_large_phase_offset_and_harmonics_take_reference_guard(self):
        t, y, w, offset = fixture()
        for h, off in ((1, np.full(len(t), 1e16)), (4, offset+3e5)):
            kw=dict(harmonics=h, phase_offset=off, weights=w, refine_period=False)
            cpu=multi_harmonic_lomb_scargle(t,y,6,9,60,**kw)
            gpu=multi_harmonic_lomb_scargle(t,y,6,9,60,compute_device='cuda:0',**kw)
            np.testing.assert_array_equal(cpu.grid_scores,gpu.grid_scores)
            self.assertEqual(gpu.search_diagnostics['period_compute']['cpu_guard_grid_count'],60)

    def test_near_equal_aliases_extreme_weights_and_parameter_guard(self):
        t=np.arange(64.)
        y=np.cos(2*np.pi*.27*t)+.01*np.sin(t*.13)
        for weights in (np.ones(len(t)),np.geomspace(1e-20,1,len(t))):
            kw=dict(weights=weights,refine_period=False)
            cpu=multi_harmonic_lomb_scargle(t,y,.8,5.,1001,**kw)
            gpu=multi_harmonic_lomb_scargle(t,y,.8,5.,1001,compute_device='cuda:0',**kw)
            np.testing.assert_allclose(cpu.grid_scores,gpu.grid_scores,atol=2e-11,rtol=0)
            self.assertEqual([c.period_s for c in cpu.candidates],[c.period_s for c in gpu.candidates])
        t,y,w,offset=fixture()
        groups=np.arange(len(t))%65
        a=multi_harmonic_lomb_scargle(t,y,6,9,12,groups=groups,refine_period=False)
        b=multi_harmonic_lomb_scargle(t,y,6,9,12,groups=groups,refine_period=False,compute_device='cuda:0')
        np.testing.assert_array_equal(a.grid_scores,b.grid_scores)
        self.assertEqual(b.search_diagnostics['period_compute']['cpu_guard_grid_count'],12)

    def test_bootstrap_explicitly_routes_every_replicate_to_requested_device(self):
        from inversion.src import period_uncertainty as uncertainty
        t,y,w,offset=fixture()
        estimate=multi_harmonic_lomb_scargle(t,y,6,9,30)
        cfg=dict(period_min_s=6,period_max_s=9,period_grid_size=30,
            uncertainty_method='run_bootstrap',uncertainty_replicates=30,compute_device='cuda:0')
        with patch.object(uncertainty,'multi_harmonic_lomb_scargle',return_value=estimate) as search:
            uncertainty.assess_period_uncertainty(t,{'x':y},np.arange(len(t)),{'x':estimate},cfg,weights=w)
        self.assertEqual(len(search.call_args_list),30)
        self.assertTrue(all(c.kwargs['compute_device']=='cuda:0' for c in search.call_args_list))
