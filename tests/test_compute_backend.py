import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from inversion.src import compute_backend as backend
from inversion.src import radar_signal as radar
from inversion.src.inversion import estimate_rotation,prepare_chirp_observations
from inversion.src.processing_store import stage_key,cpi_folder
from inversion.tests import test_chirp_contracts as fixtures
chirp_echo=fixtures.chirp_echo


class BackendContractTests(unittest.TestCase):
    def test_default_and_no_cuda_fallback_and_context_restore(self):
        self.assertEqual(backend.resolve_device({})['actual'],'cpu')
        import torch
        with patch.object(torch.cuda,'is_available',return_value=False):
            state=backend.resolve_device({'compute_device':'cuda:0'})
            self.assertEqual(state['actual'],'cpu');self.assertIsNotNone(state['reason'])
        with self.assertRaises(ValueError):backend.resolve_device({'compute_device':'auto'})
        self.assertFalse(backend.is_cuda())

    def test_device_and_error_settings_cache_contract(self):
        from pathlib import Path
        echo=SimpleNamespace(_source_identity=dict(path='echo',size=1,mtime=2),_processing_store=Path('cache'))
        cpu=stage_key(echo,{})
        self.assertEqual(cpu,stage_key(echo,{'compute_device':'cpu','uncertainty_method':'run_bootstrap'}))
        gpu=stage_key(echo,{'compute_device':'cuda:0'})
        self.assertNotEqual(cpu,gpu)
        self.assertNotIn(':',cpi_folder(echo,{'compute_device':'cuda:0'}).name)


class CUDAAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA unavailable')

    def test_compression_analytic_impulse_and_dtype_bandwidth_matrix(self):
        rng=np.random.default_rng(91)
        for convention in ('centered','zero_to_bandwidth'):
            for dtype in (np.complex64,np.complex128):
                for ratio in (.1,.75,.95):
                    fast=np.arange(128)/1000
                    iq=(rng.normal(size=(5,128))+1j*rng.normal(size=(5,128))).astype(dtype)
                    cpu=radar.matched_filter_chirp(iq,fast,.032,ratio*1000,baseband_convention=convention,full_output=True)
                    with backend.execution({'compute_device':'cuda:0'}):
                        gpu=radar.matched_filter_chirp(iq,fast,.032,ratio*1000,baseband_convention=convention,full_output=True)
                    tolerance=5e-6 if dtype==np.complex64 else 1e-10
                    self.assertLess(np.max(abs(gpu-cpu))/np.max(abs(cpu)),tolerance)
        iq=np.zeros((1,128),complex);iq[0,40]=1
        with backend.execution({'compute_device':'cuda:0'}):
            actual=radar.matched_filter_chirp(iq,fast,.032,750,full_output=True)[0]
        n=radar.chirp_reference_sample_count(.032,.001);t=np.arange(n)*.001
        reference=np.exp(1j*np.pi*750/.032*(t-.016)**2)
        golden=np.zeros(128+n-1,complex);golden[40:40+n]=np.conj(reference[::-1])
        np.testing.assert_allclose(actual,golden,atol=1e-12)

    def test_fourier_mask_shift_support_and_slow_fft_known_tone(self):
        rng=np.random.default_rng(94);fast=np.arange(80)*.001
        iq=rng.normal(size=(8,80))+1j*rng.normal(size=(8,80))
        axes=fast[None,:]+np.linspace(-.0015,.0025,8)[:,None]
        mask=np.ones(iq.shape,bool);mask[1]=False;mask[2,20:31]=False
        for count in (80,91):
            grid=(fast[0]-.003+np.arange(count)*.001,.001)
            _,cpu=radar.align_profiles_to_common_delay(iq,axes,valid=mask,common_grid=grid,method='fourier')
            with backend.execution({'compute_device':'cuda:0'}):
                _,gpu=radar.align_profiles_to_common_delay(iq,axes,valid=mask,common_grid=grid,method='fourier')
            np.testing.assert_allclose(gpu,cpu,rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(gpu[1],0)
        signal=np.exp(2j*np.pi*3*np.arange(32)/32)[:,None]*np.array([[1,2,3]])
        result=backend.slow_fft(signal,np.ones(32))
        self.assertEqual(np.argmax(abs(result[0])),19)
        np.testing.assert_allclose(abs(result[:,19]),32*np.arange(1,4),atol=1e-11)
        self.assertLess(np.max(abs(np.delete(result,19,axis=1))),1e-11)

    def test_real_entry_matrix_cpi_fallback_lag_and_compensation(self):
        for reference in ('centroid_compensated','raw_baseband'):
            echo=chirp_echo(reference)
            # A varying delay is required: the old fixture's centroid is constant,
            # so tiny roundoff selects arbitrary periods in that channel.
            for row in range(len(echo.iq)):
                echo.iq[row]=np.roll(echo.iq[row],int(np.rint(3*np.sin(2*np.pi*row/19))))
            for duration,method in ((1.6,'spectral_moments'),(1.6,'narrowband_lag'),(20.,'spectral_moments')):
                config={**fixtures.ChirpInversionTests.config,'cpi_duration_s':duration,'doppler_estimator':method}
                cpu=estimate_rotation(copy.deepcopy(echo),config)
                gpu=estimate_rotation(copy.deepcopy(echo),{**config,'compute_device':'cuda:0'})
                self.assertEqual(gpu.search_diagnostics['compute_execution']['actual'],'cuda:0')
                self.assertEqual(cpu.periods.keys(),gpu.periods.keys())
                for field in vars(cpu.features):
                    if isinstance(getattr(cpu.features,field),np.ndarray):
                        a=getattr(cpu.features,field);b=getattr(gpu.features,field)
                        np.testing.assert_allclose(a,b,rtol=2e-7,atol=1e-12,equal_nan=True)
                for name in cpu.periods:
                    self.assertLess(abs(cpu.periods[name].best_period_s-gpu.periods[name].best_period_s),1e-4,(reference,duration,method,name))
        # Guard the guard: raw/reference compensation does change this fixture.
        echo=chirp_echo('raw_baseband')
        a=prepare_chirp_observations(echo,{**fixtures.ChirpInversionTests.config,'motion_compensation':'none'})
        b=prepare_chirp_observations(echo,fixtures.ChirpInversionTests.config)
        self.assertGreater(np.max(abs(a['features'].doppler_centroid_hz-b['features'].doppler_centroid_hz)),1e-7)
