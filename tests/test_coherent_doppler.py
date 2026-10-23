import unittest
from types import SimpleNamespace
import numpy as np
from inversion.src.radar_signal import (align_profiles_to_common_delay,narrowband_doppler_moments,
    range_doppler_cube,range_doppler_features)
from inversion.src.inversion import prepare_chirp_pulse_profiles


class CoherentDopplerTests(unittest.TestCase):
    def test_short_cpi_rejected_and_amplitude_scale_invariant(self):
        times=np.arange(100)*.01
        signal=np.exp(2j*np.pi*times[:,None]*np.array([-.1,.1]))
        expected=narrowband_doppler_moments(signal,.01)[:2]
        for scale in (1e-150,1e150):
            np.testing.assert_allclose(narrowband_doppler_moments(signal*scale,.01)[:2],expected,atol=1e-10)
        with self.assertRaisesRegex(ValueError,'至少需要12'):
            range_doppler_cube(signal,times,np.zeros(100,int),np.arange(2),cpi_pulses=11,
                cpi_hop_pulses=11,doppler_estimator='narrowband_lag')
        from inversion.tests.test_chirp_contracts import chirp_echo,ChirpInversionTests
        from inversion.src.inversion import estimate_rotation
        with self.assertRaisesRegex(ValueError,'至少需要12'):
            estimate_rotation(chirp_echo('centroid_compensated'),dict(ChirpInversionTests.config,
                cpi_duration_s=1.1,doppler_estimator='narrowband_lag'))

    def test_weak_high_frequency_tail_is_outside_low_frequency_width_contract(self):
        times=np.arange(100)*.01
        signal=np.exp(2j*np.pi*times[:,None]*np.array([0.,20.]))*np.sqrt([.9999,.0001])
        cube=range_doppler_cube(signal,times,np.zeros(100,int),np.arange(2),cpi_pulses=100,
            cpi_hop_pulses=100,doppler_estimator='narrowband_lag')
        self.assertEqual(cube.doppler_fit_quality[0,3],1.)
        self.assertLess(range_doppler_features(cube).rms_bandwidth_hz[0],.01)
        self.assertGreater(np.sqrt(.9999*.0001)*20,.19)
    def test_known_gaussian_pulse_fractional_shift_preserves_amplitude_and_phase(self):
        # Exact analytic band-limited-to-machine-precision Gaussian, away from edges.
        axis=np.arange(1024,dtype=float);offsets=np.array([-.45,.13,.37])
        axes=axis[None,:]+offsets[:,None]
        for carrier in (0.,.15,.35):
            def signal(x):return np.exp(-((x-512)/25)**2)*np.exp(2j*np.pi*carrier*x)
            source=signal(axes);expected=signal(axis)
            _,linear=align_profiles_to_common_delay(source,axes,common_grid=(axis,1),method='linear')
            _,actual=align_profiles_to_common_delay(source,axes,common_grid=(axis,1),method='fourier')
            np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=1e-10,rtol=1e-10)
            self.assertGreater(np.max(abs(linear-expected)),1e-5)

    def test_static_chirp_jitter_no_longer_creates_large_doppler_width(self):
        fs=200e6;fast=np.arange(10403)/fs
        offsets=-1.01e-6+np.mod(np.arange(100)*.61803398875,1)/fs
        times=np.arange(100)/100;u=fast[None,:]+offsets[:,None]+300e-9
        for ratio in (.1,.5,.75,.95):
            iq=np.where((u>=0)&(u<50e-6),np.exp(1j*np.pi*(ratio*fs/50e-6)*(u-25e-6)**2),0)
            echo=SimpleNamespace(iq=iq,valid=np.ones(iq.shape,bool),fast_time_s=fast,row_fast_time_offset_s=offsets,
                elapsed_s=times,scatter_elapsed_s=times,run_id=np.zeros(100,int),coherence_id=np.zeros(100,int),
                metadata=dict(pulse_width_s=50e-6,pulse_bandwidth_hz=ratio*fs,echo_output_reference='centroid_compensated',prf_hz=100))
            prepared=prepare_chirp_pulse_profiles(echo,{})
            feature=range_doppler_features(range_doppler_cube(prepared['aligned'],times,echo.coherence_id,
                prepared['common_delay'],cpi_pulses=100,cpi_hop_pulses=100))
            self.assertLess(abs(feature.doppler_centroid_hz[0]),.002)
            bound={.1:.002,.5:.006,.75:.015,.95:.07}[ratio]
            self.assertLess(abs(feature.rms_bandwidth_hz[0]-1/np.sqrt(3)),bound)
            lag=prepared['full_compressed'].shape[1]-len(fast)
            _,old=align_profiles_to_common_delay(prepared['full_compressed'][:,lag:lag+len(fast)],
                prepared['full_delay_axes'][:,lag:lag+len(fast)],common_grid=(prepared['common_delay'],1/fs),method='linear')
            original=range_doppler_features(range_doppler_cube(old,times,echo.coherence_id,prepared['common_delay'],cpi_pulses=100,cpi_hop_pulses=100))
            self.assertGreater(original.rms_bandwidth_hz[0],feature.rms_bandwidth_hz[0])

    def test_intrinsic_estimator_known_single_and_two_frequency_lines(self):
        times=np.arange(100)*.01
        for mean in (-2.3,0.,.2):
            for width in (0.,.03,.1,.2):
                signal=np.exp(2j*np.pi*times[:,None]*(mean+np.array([-width,width]))[None,:])
                mu,sigma,quality=narrowband_doppler_moments(signal,.01)
                self.assertAlmostEqual(mu,mean,places=10)
                self.assertLess(abs(sigma-width),max(1e-6,.01*width))
                self.assertGreater(quality['min_coherence'],.9)
                if width>0:self.assertGreater(sigma,0)

    def test_narrowband_guard_rejects_broad_distribution_without_silent_fallback(self):
        times=np.arange(100)*.01
        signal=np.exp(2j*np.pi*times[:,None]*np.array([-2.,2.]))
        cube=range_doppler_cube(signal,times,np.zeros(100,int),np.arange(2),cpi_pulses=100,
            cpi_hop_pulses=100,doppler_estimator='narrowband_lag')
        feature=range_doppler_features(cube)
        self.assertTrue(np.isnan(feature.rms_bandwidth_hz[0]))
        self.assertEqual(cube.doppler_fit_quality[0,3],0.)
        self.assertGreater(feature.total_power[0],0.)

    def test_scalar_chirp_has_hann_width_but_zero_intrinsic_spread(self):
        times=np.arange(100)*.01;signal=np.exp(2j*np.pi*.2*times)[:,None]
        regular=range_doppler_features(range_doppler_cube(signal,times,np.zeros(100,int),np.arange(1),cpi_pulses=100,cpi_hop_pulses=100))
        intrinsic=range_doppler_features(range_doppler_cube(signal,times,np.zeros(100,int),np.arange(1),cpi_pulses=100,
            cpi_hop_pulses=100,doppler_estimator='narrowband_lag'))
        self.assertGreater(regular.rms_bandwidth_hz[0],.57)
        self.assertLess(intrinsic.rms_bandwidth_hz[0],1e-6)
        self.assertAlmostEqual(intrinsic.doppler_centroid_hz[0],.2,places=10)
