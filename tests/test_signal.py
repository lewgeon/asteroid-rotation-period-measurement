import unittest

import numpy as np

from inversion.src.inversion import multi_harmonic_lomb_scargle
from inversion.src.radar_signal import (
    align_profiles_to_common_delay,
    centroid_delay_axes,
    matched_filter_chirp,
    matched_filter_group_delay_s,
    range_doppler_cube,
)


class LeadingEdgeCompressionTests(unittest.TestCase):
    def test_output_index_reports_leading_edge(self):
        fs = 100_000.0
        width = 1e-3
        bandwidth = 10_000.0
        fast_time = np.arange(1000) / fs
        local = fast_time - 2e-3
        chirp = np.where(
            (local >= 0.0) & (local < width),
            np.exp(1j * np.pi * bandwidth / width * (local - width / 2) ** 2),
            0.0,
        )[None, :]
        compressed = matched_filter_chirp(
            chirp, fast_time, width, bandwidth
        )
        self.assertEqual(compressed.shape, chirp.shape)
        self.assertAlmostEqual(
            fast_time[np.argmax(np.abs(compressed[0]))], 2e-3, delta=1 / fs
        )
        self.assertAlmostEqual(
            matched_filter_group_delay_s(width, 1 / fs),
            (int(np.ceil(width * fs)) - 1) / fs,
        )

    def test_row_offset_is_not_fractionally_applied_twice(self):
        fast_time = np.arange(8) * 1e-6
        axes = centroid_delay_axes(
            fast_time,
            2,
            row_fast_time_offset_s=np.array([-2.25e-6, -1.75e-6]),
            centroid_fractional_offset_s=np.array([0.25e-6, -0.25e-6]),
        )
        np.testing.assert_allclose(
            axes[:, 0], np.array([-2.25e-6, -1.75e-6])
        )
        profile = np.zeros((2, 8), dtype=complex)
        profile[0, 3] = 1.0
        profile[1, 2] = 1.0
        common, aligned = align_profiles_to_common_delay(profile, axes)
        peaks = common[np.argmax(np.abs(aligned), axis=1)]
        self.assertLessEqual(abs(peaks[0] - peaks[1]), 1e-6 + 1e-15)


class CpiContractTests(unittest.TestCase):
    def test_cpi_does_not_bridge_invalid_pulse_or_run(self):
        iq = np.ones((10, 4), dtype=complex)
        times = np.arange(10, dtype=float)
        coherence = np.zeros(10, dtype=int)
        runs = np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)]
        valid = np.ones((10, 4), dtype=bool)
        valid[2] = False
        cube = range_doppler_cube(
            iq,
            times,
            coherence,
            np.arange(4, dtype=float),
            cpi_pulses=3,
            cpi_hop_pulses=1,
            valid=valid,
            run_id=runs,
        )
        selections = zip(cube.cpi_start_pulse, cube.cpi_end_pulse)
        for start, end in selections:
            self.assertTrue(np.all(valid[start:end].any(axis=1)))
            self.assertEqual(runs[start], runs[end - 1])
        self.assertTrue(np.all(cube.effective_sample_count <= 1.0))

    def test_regression_fap_is_explicitly_uncalibrated(self):
        times = np.arange(40, dtype=float)
        values = np.sin(2 * np.pi * times / 7.0)
        estimate = multi_harmonic_lomb_scargle(
            times, values, 5.0, 10.0, 100, harmonics=2
        )
        self.assertTrue(np.isnan(estimate.false_alarm_probability))
        self.assertFalse(estimate.significant)
        self.assertEqual(
            estimate.significance_calibration,
            "unavailable_multi_harmonic_regression",
        )


if __name__ == "__main__":
    unittest.main()
