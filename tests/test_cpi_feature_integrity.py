"""Independent full-support CPI moments and diagnostic-only Doppler checks."""

import unittest

import numpy as np
from scipy import signal

from inversion.src.radar_signal import range_doppler_cube, range_doppler_features


class CpiFeatureIntegrityTests(unittest.TestCase):
    def test_full_support_moments_match_direct_dft_in_every_branch(self):
        for n in (9, 10):
            for window in ("hann", "boxcar"):
                for jitter in (False, True):
                    with self.subTest(n=n, window=window, jitter=jitter):
                        times = np.arange(n) * .01
                        if jitter:
                            times[1:-1] += .0002 * np.sin(np.arange(1, n - 1))
                        axis = np.array([-3., -1., 0., 2., 4.])
                        samples = ((1 + np.arange(n)[:, None] / n)
                                   * np.exp(.27j * np.arange(n)[:, None]
                                            * np.arange(1, 6)[None, :])
                                   * np.array([2., .7, 1.2, .6, .3]))
                        cube = range_doppler_cube(
                            samples[:, 2:], times, np.zeros(n, int), axis[2:],
                            cpi_pulses=n, cpi_hop_pulses=n, window=window,
                            energy_iq=samples, energy_delay_axis_s=axis)
                        sampled = samples
                        if jitter:
                            grid = times[0] + np.arange(n) * np.median(np.diff(times))
                            sampled = np.stack([
                                np.interp(grid, times, samples[:, k].real)
                                + 1j * np.interp(grid, times, samples[:, k].imag)
                                for k in range(len(axis))], axis=1)
                            self.assertGreater(np.max(abs(sampled - samples)), .001)
                        taper = signal.get_window(window, n)
                        # Explicit DFT is independent of both production FFT and Parseval.
                        kernel = np.exp(-2j * np.pi * np.arange(n)[:, None]
                                        * np.arange(n)[None, :] / n)
                        transformed = kernel @ (sampled * taper[:, None])
                        marginal = np.sum(abs(transformed) ** 2, axis=0)
                        full_total = marginal.sum()
                        full_centroid = float(marginal @ axis / full_total)
                        cropped_total = marginal[2:].sum()
                        cropped_centroid = float(marginal[2:] @ axis[2:] / cropped_total)
                        # Every cell must expose the old cropped-moment bug.
                        self.assertGreater(full_total - cropped_total, full_total * .5)
                        self.assertGreater(abs(full_centroid - cropped_centroid), 1.)
                        features = range_doppler_features(cube)
                        self.assertAlmostEqual(features.total_power[0] / full_total, 1., places=13)
                        self.assertAlmostEqual(features.range_centroid_s[0], full_centroid, places=13)
                        # The two Doppler features must still use the original RD spectrum.
                        power = abs(cube.spectrum[0]) ** 2
                        frequency = cube.frame_frequencies_hz[0]
                        expected_mu = np.sum(power * frequency[None, :]) / power.sum()
                        self.assertAlmostEqual(features.doppler_centroid_hz[0], expected_mu, places=13)

    def test_without_full_support_preserves_spectrum_moments(self):
        n = 16
        times = np.arange(n) * .01
        iq = np.exp(2j * np.pi * times[:, None] * [-4., 7.]) * [1., 2.]
        cube = range_doppler_cube(iq, times, np.zeros(n, int), [-2., 3.],
                                 cpi_pulses=n, cpi_hop_pulses=n)
        features = range_doppler_features(cube)
        power = abs(cube.spectrum) ** 2
        self.assertAlmostEqual(features.total_power[0], power.sum(), places=10)
        self.assertAlmostEqual(features.range_centroid_s[0], 2., places=13)

    def test_window_width_and_edge_mass_are_diagnostics_without_correction(self):
        n = 100
        times = np.arange(n) / 100.
        edge_values = []
        for frequency in (0., 49.6, -49.6):
            iq = np.exp(2j * np.pi * frequency * times)[:, None]
            cube = range_doppler_cube(iq, times, np.zeros(n, int), [0.],
                                     cpi_pulses=n, cpi_hop_pulses=n)
            features = range_doppler_features(cube)
            self.assertAlmostEqual(features.doppler_window_rms_hz[0], 1 / np.sqrt(3), places=12)
            self.assertAlmostEqual(features.doppler_nyquist_edge_half_width_hz[0], 2., places=12)
            edge_values.append(features.doppler_nyquist_edge_fraction[0])
            if frequency == 0:
                self.assertAlmostEqual(features.rms_bandwidth_hz[0], 1 / np.sqrt(3), places=12)
            else:
                # Keep the measured linear spectrum moment; the diagnostic exposes wrap risk.
                self.assertGreater(features.rms_bandwidth_hz[0], 10.)
                power = abs(cube.spectrum[0, 0]) ** 2
                fd = cube.frame_frequencies_hz[0]
                mean = float(power @ fd / power.sum())
                width = np.sqrt(power @ (fd - mean) ** 2 / power.sum())
                self.assertAlmostEqual(features.rms_bandwidth_hz[0], width, places=12)
        self.assertLess(edge_values[0], 1e-25)
        self.assertGreater(min(edge_values[1:]), .99)

    def test_diagnostics_follow_each_cpis_actual_spacing_and_window(self):
        n = 10
        times = np.r_[np.arange(n) * .01, 2 + np.arange(n) * .02]
        groups = np.repeat([0, 1], n)
        iq = np.ones((2 * n, 1), complex)
        for window in ("hann", "boxcar"):
            cube = range_doppler_cube(iq, times, groups, [0.], cpi_pulses=n,
                                     cpi_hop_pulses=n, window=window)
            feature = range_doppler_features(cube)
            expected = np.array([10., 5.]) / np.sqrt(3) if window == "hann" else np.zeros(2)
            np.testing.assert_allclose(feature.doppler_window_rms_hz, expected, atol=1e-12)
            np.testing.assert_allclose(feature.doppler_nyquist_edge_half_width_hz, [20., 10.], atol=1e-12)

    def test_zero_signal_has_unknown_edge_fraction(self):
        n = 12
        cube = range_doppler_cube(np.zeros((n, 2), complex), np.arange(n) * .01,
            np.zeros(n, int), [0., 1.], cpi_pulses=n, cpi_hop_pulses=n)
        feature = range_doppler_features(cube)
        self.assertTrue(np.isnan(feature.doppler_nyquist_edge_fraction[0]))
        self.assertTrue(np.isfinite(feature.doppler_window_rms_hz[0]))


if __name__ == "__main__":
    unittest.main()
