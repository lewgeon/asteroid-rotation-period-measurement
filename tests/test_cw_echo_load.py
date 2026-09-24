"""CW echo.npz 与 chirp echo.npz 走不同的行轴契约。

CW（一维 iq）没有脉冲行轴，因此不带行轴字段；加载器按 iq 维数区分布局，
缺键时用自己的兜底值。chirp（二维 iq）必须带全，长度必须等于脉冲数。
"""

import json
import unittest

import numpy as np

from inversion.src.dataset import load_echo
from inversion.tests.scratch import scratch_directory


def _cw_payload(n=8):
    """真实 CW 文件的样子：只有逐样本字段，没有行轴字段。"""

    times = np.arange(n, dtype=float)
    return dict(
        elapsed_s=times,
        iq=np.ones(n, dtype=np.complex64),
        clean_iq=np.ones(n, dtype=np.complex64),
        valid=np.ones(n, dtype=bool),
        fast_time_s=np.array([], dtype=float),
        tx_los_icrs=np.zeros((n, 3)),
        rx_los_icrs=np.zeros((n, 3)),
        tx_range_m=np.ones(n),
        rx_range_m=np.ones(n),
        scatter_elapsed_s=times,
        emit_elapsed_s=times,
        metadata_json=json.dumps(
            {
                "data_layout": "continuous_time",
                "waveform_type": "continuous_wave",
                "sample_rate_hz": 1.0,
                "carrier_frequency_hz": 1.0e9,
            }
        ),
    )


def _chirp_payload(n_pulse=3, n_fast=4):
    return dict(
        elapsed_s=np.arange(n_pulse, dtype=float),
        iq=np.ones((n_pulse, n_fast), dtype=np.complex64),
        clean_iq=np.ones((n_pulse, n_fast), dtype=np.complex64),
        valid=np.ones((n_pulse, n_fast), dtype=bool),
        coherence_id=np.zeros(n_pulse, dtype=int),
        acquisition_id=np.zeros(n_pulse, dtype=int),
        fast_time_s=np.arange(n_fast, dtype=float),
        run_id=np.zeros(n_pulse, dtype=int),
        track_id=np.zeros(n_pulse, dtype=int),
        row_start_sample=np.arange(n_pulse, dtype=np.int64),
        row_fast_time_offset_s=np.zeros(n_pulse),
        centroid_fractional_offset_s=np.zeros(n_pulse),
        rx_adc_start_elapsed_s=np.zeros(n_pulse),
        rx_adc_stop_elapsed_s=np.ones(n_pulse),
        signal_echo_overlap=np.zeros(n_pulse, dtype=bool),
        window_overlap=np.zeros(n_pulse, dtype=bool),
        common_path_rate_m_s=np.zeros(n_pulse),
        tx_los_icrs=np.zeros((n_pulse, 3)),
        rx_los_icrs=np.zeros((n_pulse, 3)),
        tx_range_m=np.ones(n_pulse),
        rx_range_m=np.ones(n_pulse),
        scatter_elapsed_s=np.arange(n_pulse, dtype=float),
        emit_elapsed_s=np.arange(n_pulse, dtype=float),
        metadata_json=json.dumps(
            {
                "data_layout": "pulse_adc_windows",
                "waveform_type": "chirp_pulse_train",
                "fast_sample_rate_hz": 10.0,
                "prf_hz": 10.0,
                "carrier_frequency_hz": 1.0e9,
            }
        ),
    )


class CwEchoLoadTests(unittest.TestCase):
    def test_one_dimensional_echo_without_row_axis_keys_loads(self):
        n = 8
        with scratch_directory("cw_load") as directory:
            path = directory / "echo.npz"
            np.savez(path, **_cw_payload(n))
            loaded = load_echo(path)

        self.assertEqual(loaded.iq.ndim, 1)
        # 缺键 -> 加载器的兜底值，形状与样本数一致
        self.assertEqual(loaded.run_id.shape, (n,))
        self.assertEqual(loaded.row_start_sample.shape, (n,))
        np.testing.assert_array_equal(loaded.row_start_sample, np.full(n, -1))
        self.assertTrue(np.all(np.isnan(loaded.common_path_rate_m_s)))

    def test_cw_rejects_a_row_axis_field_of_wrong_length(self):
        """CW 不要求行轴字段，但文件若提供了就必须长度正确。"""

        payload = _cw_payload(8)
        payload["run_id"] = np.zeros(3, dtype=int)  # 长度错
        with scratch_directory("cw_bad_len") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "run_id"):
                load_echo(path)

    def test_chirp_still_requires_the_full_row_axis(self):
        """chirp 的行轴字段是真实信息，缺一个都要拒绝。"""

        payload = _chirp_payload()
        del payload["window_overlap"]
        with scratch_directory("chirp_missing") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "window_overlap"):
                load_echo(path)

    def test_chirp_row_axis_length_must_match_pulse_count(self):
        payload = _chirp_payload()
        payload["row_start_sample"] = np.arange(5, dtype=np.int64)  # 与脉冲数不符
        with scratch_directory("chirp_bad_len") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "row_start_sample"):
                load_echo(path)

    def test_chirp_metadata_fast_count_must_match_iq_width(self):
        payload = _chirp_payload(n_pulse=3, n_fast=4)
        metadata = json.loads(payload["metadata_json"])
        metadata["fast_sample_count"] = 5
        payload["metadata_json"] = json.dumps(metadata)
        with scratch_directory("chirp_bad_fast_count") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(
                ValueError, "metadata.fast_sample_count.*iq 第 1 维"
            ):
                load_echo(path)

    def test_chirp_metadata_fast_count_must_be_an_exact_integer(self):
        payload = _chirp_payload(n_pulse=3, n_fast=4)
        metadata = json.loads(payload["metadata_json"])
        metadata["fast_sample_count"] = 4.9
        payload["metadata_json"] = json.dumps(metadata)
        with scratch_directory("chirp_fractional_fast_count") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "fast_sample_count.*整数"):
                load_echo(path)

    def test_chirp_fast_time_axis_must_be_one_dimensional(self):
        payload = _chirp_payload(n_pulse=3, n_fast=4)
        payload["fast_time_s"] = np.zeros((4, 2), dtype=float)
        with scratch_directory("chirp_fast_time_2d") as directory:
            path = directory / "echo.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "fast_time_s.*一维"):
                load_echo(path)


if __name__ == "__main__":
    unittest.main()
