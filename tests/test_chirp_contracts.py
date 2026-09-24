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

    def test_routing_scatter_clock_and_raw_motion_compensation(self):
        result = estimate_rotation(chirp_echo("raw_baseband"), self.config)
        self.assertEqual(result.time_axis_role, "scatter_elapsed_s")
        cube = result.dynamic_spectrum
        integrated = cube.power.sum(axis=(0, 1))
        peak_hz = cube.frequencies_hz[np.argmax(integrated)]
        self.assertLess(abs(peak_hz), 0.1)

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
