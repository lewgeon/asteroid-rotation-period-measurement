import unittest
from types import SimpleNamespace

import numpy as np

from inversion.src.inversion import estimate_rotation


def chirp_echo(output_reference):
    pulse_count = 80
    prf_hz = 10.0
    fs_hz = 10_000.0
    width_s = 1e-3
    bandwidth_hz = 100.0
    carrier_hz = 1e9
    scatter_times = np.arange(pulse_count) / prf_hz
    fast_time = np.arange(40) / fs_hz
    local = fast_time - 1e-3
    waveform = np.where(
        (local >= 0.0) & (local < width_s),
        np.exp(
            1j * np.pi * bandwidth_hz / width_s
            * (local - width_s / 2) ** 2
        ),
        0.0,
    )
    amplitude = 1.5 + 0.5 * np.sin(2 * np.pi * scatter_times / 2.0)
    common_delay = 20.0 + 2e-9 * np.arange(pulse_count)
    common_phase = np.exp(
        -2j * np.pi * np.remainder(carrier_hz * common_delay, 1.0)
    )
    iq = amplitude[:, None] * waveform[None, :]
    if output_reference == "raw_baseband":
        iq = iq * common_phase[:, None]
    return SimpleNamespace(
        elapsed_s=scatter_times + common_delay,
        scatter_elapsed_s=scatter_times,
        iq=iq,
        valid=np.ones(iq.shape, dtype=bool),
        coherence_id=np.zeros(pulse_count, dtype=int),
        run_id=np.zeros(pulse_count, dtype=int),
        fast_time_s=fast_time,
        row_fast_time_offset_s=np.zeros(pulse_count),
        centroid_fractional_offset_s=np.zeros(pulse_count),
        tx_range_m=0.5 * 299_792_458.0 * common_delay,
        rx_range_m=0.5 * 299_792_458.0 * common_delay,
        metadata={
            "data_layout": "pulse_adc_windows",
            "echo_output_reference": output_reference,
            "carrier_frequency_hz": carrier_hz,
            "pulse_width_s": width_s,
            "pulse_bandwidth_hz": bandwidth_hz,
            "fast_sample_rate_hz": fs_hz,
            "prf_hz": prf_hz,
            "baseband_convention": "centered",
        },
    )


class ChirpInversionTests(unittest.TestCase):
    config = {
        "cpi_duration_s": 1.6,
        "cpi_hop_duration_s": 0.4,
        "period_min_s": 1.0,
        "period_max_s": 3.0,
        "period_grid_size": 80,
        "harmonics": 1,
        "motion_compensation": "centroid_geometry",
    }

    def test_cpi_inversion_never_compresses_the_whole_campaign(self):
        from unittest.mock import patch
        from inversion.src import inversion as implementation
        echo = chirp_echo('centroid_compensated')
        config = {**self.config, 'cpi_duration_s': .8}
        original = implementation.prepare_chirp_pulse_profiles
        sizes = []
        def guarded(chunk, *args, **kwargs):
            sizes.append(len(chunk.iq))
            if len(chunk.iq) > 16:
                raise MemoryError('campaign-sized FFT sentinel')
            return original(chunk, *args, **kwargs)
        with patch.object(implementation, 'prepare_chirp_pulse_profiles', side_effect=guarded):
            result = estimate_rotation(echo, config)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 16)
        self.assertEqual(result.search_diagnostics['processing_path'], 'cpi')

    def test_streamed_cpi_matches_original_full_cube(self):
        from inversion.src import inversion as core
        from inversion.src.radar_signal import align_profiles_to_common_delay, range_doppler_cube, range_doppler_features
        for reference in ('centroid_compensated', 'raw_baseband'):
            for clock in ('scatter_centroid', 'receive_centroid'):
                echo = chirp_echo(reference)
                if clock == 'receive_centroid':
                    echo.elapsed_s += np.where(np.arange(80) % 2, .002, -.002)
                    differences = np.diff(echo.elapsed_s[:8])
                    self.assertFalse(np.allclose(differences, np.median(differences), rtol=1e-3, atol=1e-9))
                echo.row_fast_time_offset_s = np.linspace(-4.5e-5, 4.5e-5, 80)
                echo.common_path_rate_m_s = np.linspace(-15000., 15000., 80)
                echo.valid[35] = False
                echo.run_id = np.repeat([0, 1], 40)
                echo.coherence_id = echo.run_id.copy()
                config = {**self.config, 'cpi_duration_s': .8, 'period_time_role': clock}
                original = core.prepare_chirp_pulse_profiles(echo, config)
                energy_axis, energy = align_profiles_to_common_delay(
                    original['full_compressed'], original['full_delay_axes'], preserve_support=True,method='fourier')
                cube = range_doppler_cube(original['aligned'], original['processing_times'], echo.coherence_id,
                    original['common_delay'], cpi_pulses=8, cpi_hop_pulses=4,
                    valid=original['pulse_valid'], run_id=echo.run_id,
                    effective_pulse_weight=original['row_weight'], energy_iq=energy,
                    energy_delay_axis_s=energy_axis)
                expected = range_doppler_features(cube)
                self.assertGreater(np.ptp(expected.total_power), 1.)
                actual, shown, role = core._streamed_cpi_inputs(echo, config, 8, 4,
                    original['pulse_valid'], original['row_weight'], None)
                self.assertEqual(role, original['time_axis_role'])
                for name in expected.__dataclass_fields__:
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                np.testing.assert_array_equal(shown.spectrum, cube.spectrum)
                np.testing.assert_array_equal(shown.cpi_start_pulse, cube.cpi_start_pulse)
                np.testing.assert_array_equal(shown.cpi_end_pulse, cube.cpi_end_pulse)

    def test_cpi_keeps_legacy_missing_run_fallback(self):
        echo = chirp_echo('centroid_compensated')
        expected = estimate_rotation(echo, self.config)
        del echo.run_id
        actual = estimate_rotation(echo, self.config)
        for name in expected.features.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(actual.features, name), getattr(expected.features, name))
        echo = chirp_echo('centroid_compensated')
        echo.scatter_elapsed_s = echo.elapsed_s.copy()
        expected = estimate_rotation(echo, self.config)
        del echo.scatter_elapsed_s
        actual = estimate_rotation(echo, self.config)
        for name in expected.features.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(actual.features, name), getattr(expected.features, name))

    def test_global_grid_exact_median_and_disk_fallback(self):
        from inversion.src.inversion import _global_delay_grid
        from inversion.src.radar_signal import align_profiles_to_common_delay
        rng = np.random.default_rng(128)
        for fast in (np.arange(40) * .1, np.cumsum(rng.uniform(.1, .2, 5000))):
            offsets = np.array([-.031, 0., .023, .04])
            axes = fast[None, :] + offsets[:, None]
            values = np.zeros_like(axes, dtype=complex)
            for support in (False, True):
                expected, _ = align_profiles_to_common_delay(values, axes, preserve_support=support)
                actual, step = _global_delay_grid(fast, offsets, preserve_support=support)
                self.assertEqual(step, np.median(np.diff(axes, axis=1)))
                np.testing.assert_array_equal(actual, expected)

    def test_routing_scatter_clock_and_raw_motion_compensation(self):
        result = estimate_rotation(chirp_echo("raw_baseband"), self.config)
        self.assertEqual(result.time_axis_role, "scatter_elapsed_s")
        cube = result.dynamic_spectrum
        integrated = cube.power.sum(axis=(0, 1))
        peak_hz = cube.frequencies_hz[np.argmax(integrated)]
        self.assertLess(abs(peak_hz), 0.1)

    def test_cpi_longer_than_the_coherence_group_is_reported(self):
        config = dict(self.config)
        config["cpi_duration_s"] = 20.0
        config["period_grid_size"] = 8
        result = estimate_rotation(chirp_echo("centroid_compensated"), config)
        self.assertTrue(result.notices)
        self.assertIn("逐脉冲", result.notices[0])
        self.assertIn("20", result.notices[0])

    def test_centroid_compensated_input_is_not_compensated_twice(self):
        result = estimate_rotation(
            chirp_echo("centroid_compensated"), self.config
        )
        cube = result.dynamic_spectrum
        integrated = cube.power.sum(axis=(0, 1))
        peak_hz = cube.frequencies_hz[np.argmax(integrated)]
        self.assertLess(abs(peak_hz), 0.1)

    def test_physical_cpi_durations_control_window_size(self):
        config = {
            **self.config,
            "cpi_duration_s": 0.8,
            "cpi_hop_duration_s": 0.4,
        }
        result = estimate_rotation(chirp_echo("centroid_compensated"), config)
        cube = result.dynamic_spectrum
        np.testing.assert_array_equal(
            cube.cpi_end_pulse - cube.cpi_start_pulse,
            np.full(len(cube.cpi_start_pulse), 8),
        )


if __name__ == "__main__":
    unittest.main()
