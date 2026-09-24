import json
import unittest

import numpy as np

from inversion.src.dataset import EchoDataset, load_echo
from inversion.tests.scratch import scratch_directory
from inversion.src.inversion import (
    estimate_rotation,
    lomb_scargle,
    matched_filter_chirp,
    multi_harmonic_lomb_scargle,
)
from inversion.src.radar_signal import range_doppler_cube, range_doppler_features


class InversionTests(unittest.TestCase):
    def test_multi_harmonic_search_handles_run_baselines(self):
        times = np.r_[np.linspace(0.0, 40.0, 80), np.linspace(100.0, 140.0, 80)]
        groups = np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)]
        values = np.sin(2 * np.pi * times / 12.0) + 0.4 * np.cos(4 * np.pi * times / 12.0)
        values += np.where(groups == 0, -3.0, 5.0)
        result = multi_harmonic_lomb_scargle(
            times, values, 8.0, 20.0, 1200, harmonics=2, groups=groups
        )
        self.assertAlmostEqual(result.best_period_s, 12.0, delta=0.08)

    def test_range_doppler_cpi_never_crosses_groups(self):
        samples = 8
        times = np.r_[np.arange(8) * 0.1, 10.0 + np.arange(8) * 0.1]
        groups = np.r_[np.zeros(8, dtype=int), np.ones(8, dtype=int)]
        iq = np.exp(2j * np.pi * 1.25 * times)[:, None] * np.ones((1, samples))
        cube = range_doppler_cube(
            iq, times, groups, np.arange(samples) * 1e-6,
            cpi_pulses=8, cpi_hop_pulses=4,
        )
        self.assertEqual(cube.spectrum.shape[0], 2)
        np.testing.assert_array_equal(cube.coherence_id, [0, 1])
        self.assertEqual(len(range_doppler_features(cube).times_s), 2)

    def test_matched_filter_compresses_each_pulse_independently(self):
        sample_rate = 2.0e6
        fast_time = -20e-6 + np.arange(80) / sample_rate
        width, bandwidth = 20e-6, 0.5e6
        slope = bandwidth / width
        delay = 3e-6
        local = fast_time - delay
        pulse = np.where(
            np.abs(local) < width / 2,
            np.exp(1j * np.pi * slope * local**2),
            0.0,
        )
        compressed = matched_filter_chirp(np.vstack([pulse, pulse]), fast_time, width, bandwidth)
        peak_time = fast_time[np.argmax(np.abs(compressed[0]))]
        leading_edge = delay - width / 2
        self.assertLess(abs(peak_time - leading_edge), 2 / sample_rate)
    def test_nonuniform_period_recovery(self):
        rng = np.random.default_rng(7)
        times = np.sort(rng.uniform(0, 250, 400))
        truth = 23.0
        values = np.sin(2 * np.pi * times / truth)
        values += 0.05 * rng.standard_normal(len(times))
        estimate = lomb_scargle(times, values, 10, 40, 5000)
        self.assertLess(abs(estimate.best_period_s - truth) / truth, 0.01)

    def test_load_echo_reads_new_exchange_format(self):
        elapsed = np.arange(64, dtype=float) / 16.0
        metadata = {"sample_rate_hz": 16.0, "truth_period_s": 20.0}
        with scratch_directory("load_echo_new_format") as directory:
            path = directory / "echo.npz"
            np.savez(
                path,
                elapsed_s=elapsed,
                iq=np.exp(1j * elapsed),
                clean_iq=np.exp(1j * elapsed),
                valid=np.ones(len(elapsed), dtype=bool),
                coherence_id=np.zeros(len(elapsed), dtype=int),
                tx_los_icrs=np.zeros((len(elapsed), 3)),
                rx_los_icrs=np.zeros((len(elapsed), 3)),
                tx_range_m=np.zeros(len(elapsed)),
                rx_range_m=np.zeros(len(elapsed)),
                scatter_elapsed_s=elapsed,
                emit_elapsed_s=elapsed,
                metadata_json=json.dumps(metadata),
            )
            echo = load_echo(path)
        np.testing.assert_allclose(echo.elapsed_s, elapsed)
        self.assertEqual(echo.metadata["truth_period_s"], 20.0)

    def test_load_echo_rejects_mismatched_row_arrays(self):
        elapsed = np.arange(4, dtype=float)
        with scratch_directory("load_echo_bad_rows") as directory:
            path = directory / "bad-echo.npz"
            np.savez(
                path,
                elapsed_s=elapsed,
                iq=np.ones(4, dtype=complex),
                clean_iq=np.ones(3, dtype=complex),
            )
            with self.assertRaisesRegex(ValueError, "clean_iq"):
                load_echo(path)

    def test_estimate_rotation_does_not_require_legacy_translation_coefficients(self):
        sample_rate = 64.0
        period_s = 20.0
        elapsed = np.arange(int(240 * sample_rate)) / sample_rate
        envelope = 1.0 + 0.35 * np.cos(2 * np.pi * elapsed / period_s)
        phase = 0.15 * np.sin(2 * np.pi * elapsed / 7.0)
        iq = envelope * np.exp(1j * phase)
        echo = EchoDataset(
            elapsed_s=elapsed,
            iq=iq,
            clean_iq=iq,
            valid=np.ones(len(elapsed), dtype=bool),
            coherence_id=np.zeros(len(elapsed), dtype=int),
            tx_los_icrs=np.zeros((len(elapsed), 3)),
            rx_los_icrs=np.zeros((len(elapsed), 3)),
            tx_range_m=np.zeros(len(elapsed)),
            rx_range_m=np.zeros(len(elapsed)),
            scatter_elapsed_s=elapsed,
            emit_elapsed_s=elapsed,
            metadata={"sample_rate_hz": sample_rate, "truth_period_s": period_s},
        )
        result = estimate_rotation(
            echo,
            {
                "stft_window_samples": 256,
                "stft_overlap_fraction": 0.75,
                "period_min_s": 10.0,
                "period_max_s": 30.0,
                "period_grid_size": 3000,
            },
        )
        candidates = result.periods["total_power"].candidates
        errors = [abs(candidate.period_s - period_s) / period_s for candidate in candidates]
        self.assertLess(min(errors), 0.05)
