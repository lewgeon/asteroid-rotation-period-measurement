"""Full-correlation energy width, coordinates and CPI Parseval evidence."""

import unittest

import numpy as np
from scipy import signal

from inversion.src.inversion import chirp_pulse_range_features, prepare_chirp_pulse_profiles, multi_harmonic_lomb_scargle
from inversion.src.radar_signal import (
    align_profiles_to_common_delay, matched_filter_chirp,
    power_energy_width_90_s, range_doppler_cube, range_doppler_features,
)
from inversion.tests import test_chirp_contracts


class EnergyWidthTests(unittest.TestCase):
    def test_constant_small_width_does_not_divide_by_zero_in_period_search(self):
        with self.assertRaises(ValueError):
            multi_harmonic_lomb_scargle(np.arange(80.), np.full(80, .0006), 1, 3, 8)

    def test_discrete_quantiles_multiple_peaks_and_power_scale(self):
        p = np.array([[1., 4., 10., 4., 1.]]) * np.array([1., 1e-200, 1e200])[:, None]
        axes = np.array([[0., 1., 2., 3., 4.], [10., 12., 14., 16., 18.], [-5., -4., -3., -2., -1.]])
        np.testing.assert_allclose(power_energy_width_90_s(p, axes), [3., 6., 3.])
        self.assertEqual(power_energy_width_90_s([[1, 0, 0, 0, 1]], np.arange(5.))[0], 4.)
        self.assertEqual(power_energy_width_90_s([[0, 1, 0]], [0, 1, 2])[0], 0.)

    def test_zero_nonfinite_empty_and_invalid_inputs(self):
        p = [[0, 0], [1, np.nan], [1, np.inf]]
        self.assertTrue(np.all(np.isnan(power_energy_width_90_s(p, [0, 1]))))
        self.assertTrue(np.isnan(power_energy_width_90_s(np.empty((1, 0)), [])[0]))
        for p, axis in (([[1, -1]], [0, 1]), ([[1, 1]], [1, 0]),
                        ([[1, 1]], [0, np.inf]), ([[1, 1]], [0]), ([1, 1], [0, 1])):
            with self.subTest(p=p, axis=axis), self.assertRaises(ValueError):
                power_energy_width_90_s(p, axis)

    def test_full_correlation_matches_independent_golden_and_retains_left_tail(self):
        iq = np.array([[1., 1., 1., 1., 0.]])
        full = matched_filter_chirp(iq, np.arange(5.), 4., 0., full_output=True)
        golden = np.array([[1., 2., 3., 4., 3., 2., 1., 0.]])
        np.testing.assert_allclose(full, golden, atol=1e-14)
        cropped = matched_filter_chirp(iq, np.arange(5.), 4., 0.)
        np.testing.assert_allclose(cropped, golden[:, 3:], atol=1e-14)
        full_axis = np.arange(8.) - 3
        self.assertEqual(full_axis[np.argmax(abs(full[0]))], 0)
        self.assertEqual(power_energy_width_90_s(abs(full)**2, full_axis)[0], 4.)
        self.assertEqual(power_energy_width_90_s(abs(cropped)**2, np.arange(5.))[0], 2.)

    def test_raw_full_profile_in_production_pulse_features(self):
        echo = test_chirp_contracts.chirp_echo("centroid_compensated")
        echo.iq = np.roll(echo.iq, -10, axis=1)
        config = {"motion_compensation": "auto"}
        prepared = prepare_chirp_pulse_profiles(echo, config)
        features, _, _ = chirp_pulse_range_features(echo, config)
        fs, width, bandwidth = 10000., 1e-3, 100.
        u = np.arange(10) / fs
        reference = np.exp(1j*np.pi*bandwidth/width*(u-width/2)**2)
        golden = np.correlate(echo.iq[0], reference, mode="full")
        p = abs(golden)**2
        cumulative = p.cumsum()
        expected = (np.searchsorted(cumulative, .95*cumulative[-1])
                    - np.searchsorted(cumulative, .05*cumulative[-1])) / fs
        self.assertAlmostEqual(features.energy_width_90_s[0], expected)
        np.testing.assert_allclose(prepared["full_compressed"][0], golden, atol=1e-12)
        cropped_width = power_energy_width_90_s(abs(prepared["aligned"])**2, prepared["common_delay"])[0]
        self.assertGreater(abs(expected - cropped_width), 1e-5)

    def test_tail_energy_is_included_without_gate(self):
        main = np.array([1., 4., 10., 4., 1.])
        short = power_energy_width_90_s(main[None, :], np.arange(5.))[0]
        p = np.r_[main, np.full(100, .1)]
        long = power_energy_width_90_s(p[None, :], np.arange(105.))[0]
        self.assertGreater(long, short + 50)

    def test_full_production_coordinates_mask_and_both_baseband_conventions(self):
        outputs = []
        for convention in ("centered", "zero_to_bandwidth"):
            with self.subTest(convention=convention):
                echo = test_chirp_contracts.chirp_echo("centroid_compensated")
                echo.metadata["baseband_convention"] = convention
                u = np.arange(10) / 10000.
                phase_time = u - .0005 if convention == "centered" else u
                reference = np.exp(1j*np.pi*100/.001*phase_time**2)
                echo.iq = np.zeros_like(echo.iq)
                echo.iq[:, 10:20] = reference
                echo.valid[:, 10:12] = False
                echo.row_fast_time_offset_s = np.linspace(-2.25e-5, 1.75e-5, len(echo.iq))
                prepared = prepare_chirp_pulse_profiles(echo, {"motion_compensation": "auto"})
                golden = np.correlate(np.where(echo.valid[0], echo.iq[0], 0), reference, mode="full")
                unmasked = np.correlate(echo.iq[0], reference, mode="full")
                self.assertGreater(np.max(abs(golden - unmasked)), .1)
                np.testing.assert_allclose(prepared["full_compressed"][0], golden, atol=1e-12)
                np.testing.assert_allclose(prepared["full_delay_axes"][:, 0], -.0009 + echo.row_fast_time_offset_s)
                self.assertTrue(np.all(np.isfinite(prepared["full_energy_width_90_s"])))
                outputs.append(golden)
        self.assertGreater(np.max(abs(outputs[0] - outputs[1])), .01)

    def test_union_axis_covers_full_rows(self):
        axes = np.array([[-3., -2., -1., 0.], [-2.5, -1.5, -.5, .5]])
        axis, aligned = align_profiles_to_common_delay(np.ones((2, 4), dtype=complex), axes, preserve_support=True)
        self.assertLessEqual(axis[0], axes.min())
        self.assertGreaterEqual(axis[-1], axes.max())
        expected = np.interp(axis, [-3.5, -2.5, -1.5, -.5, .5, 1.5], [0, 1, 1, 1, 1, 0])
        np.testing.assert_allclose(aligned[1], expected)
        self.assertGreater(abs(aligned[1, np.flatnonzero(axis == -3)[0]]), 0)
        self.assertGreater(abs(aligned[1, np.flatnonzero(axis == 1)[0]]), 0)

    def test_cpi_full_power_parseval_with_and_without_time_resampling(self):
        for jitter in (False, True):
            with self.subTest(jitter=jitter):
                times = np.arange(4)*.1 + (np.array([0., .001, -.001, 0.]) if jitter else 0)
                full_iq = (1 + np.arange(4)[:, None]) * np.exp(.3j*np.arange(4)[:, None]) * np.array([.2, 1, 3, 2, .1])[None, :]
                cube = range_doppler_cube(full_iq[:, 2:], times, np.zeros(4, int), np.arange(3.),
                    cpi_pulses=4, cpi_hop_pulses=1, energy_iq=full_iq, energy_delay_axis_s=np.arange(5.))
                if jitter:
                    target = times[0] + np.arange(4)*np.median(np.diff(times))
                    full_iq = np.stack([np.interp(target, times, full_iq[:, k].real)
                                       + 1j*np.interp(target, times, full_iq[:, k].imag) for k in range(5)], axis=1)
                golden = abs(np.fft.fft(full_iq*signal.get_window("hann", 4)[:, None], axis=0))**2
                np.testing.assert_allclose(cube.energy_range_power[0], golden.sum(axis=0), rtol=1e-14)
                features = range_doppler_features(cube)
                self.assertEqual(features.energy_range_width_90_s[0], power_energy_width_90_s(golden.sum(axis=0)[None, :], np.arange(5.))[0])


if __name__ == "__main__":
    unittest.main()
