"""Full native pulse moments, independent correlation and routing checks."""
import unittest
from unittest.mock import patch
import numpy as np
from inversion.src.inversion import (
    bounded_pulse_chunk_size,
    chirp_pulse_range_features,
    prepare_chirp_pulse_profiles,
    streamed_chirp_pulse_features,
    estimate_rotation,
    PeriodSearchUnavailable,
)
from inversion.src.radar_signal import matched_filter_chirp
from inversion.tests import test_chirp_contracts
chirp_echo = test_chirp_contracts.chirp_echo


class NativePulseFeatureTests(unittest.TestCase):
    def fixture(self, convention, offset_kind, mask_kind):
        echo = chirp_echo('centroid_compensated')
        echo.metadata['baseband_convention'] = convention
        echo.metadata['pulse_bandwidth_hz'] = 3000.
        echo.iq[:] = 0
        # Asymmetric complex values, including both crop boundaries.
        echo.iq[:, [0, 10, 11, 12, 38, 39]] = [1, 2, -1j, -2, .5j, 1]
        offsets = {'equal': np.zeros(80),
                   'fractional': np.linspace(-4.5e-5, 4.5e-5, 80),
                   'integer': np.tile([-1e-4, 0., 1e-4, 0.], 20)}
        echo.row_fast_time_offset_s = offsets[offset_kind]
        if mask_kind == 'samples':
            echo.valid[:, 11] = False
            echo.valid[0, :] = False
        else:
            echo.valid = np.ones(80, dtype=bool)
            echo.valid[0] = False
        return echo

    def golden(self, echo):
        step = np.median(np.diff(echo.fast_time_s))
        count = max(2, int(np.ceil(echo.metadata['pulse_width_s']/step)))
        t = np.arange(count)*step
        if echo.metadata['baseband_convention'] == 'centered':
            t -= echo.metadata['pulse_width_s']/2
        reference = np.exp(1j*np.pi*echo.metadata['pulse_bandwidth_hz']/echo.metadata['pulse_width_s']*t**2)
        valid = echo.valid if echo.valid.ndim == 2 else echo.valid[:, None]
        raw = np.where(valid, echo.iq, 0)
        full = np.stack([np.correlate(row, reference, 'full') for row in raw])
        p = abs(full)**2
        total = p.sum(axis=1)
        axis = (echo.fast_time_s[0] + (np.arange(full.shape[1])-(count-1))*step)[None, :] + echo.row_fast_time_offset_s[:, None]
        centroid = (p*axis).sum(axis=1)/np.maximum(total, np.finfo(float).tiny)
        full_power = abs(full)**2
        widths=[]
        for row in full_power:
            cumulative=row.cumsum()
            widths.append(np.nan if cumulative[-1] == 0 else
                          (np.searchsorted(cumulative,.95*cumulative[-1])-np.searchsorted(cumulative,.05*cumulative[-1]))*step)
        return total, centroid, np.array(widths)

    def test_native_moments_and_full_width_matrix(self):
        for convention in ('centered','zero_to_bandwidth'):
            for offset in ('equal','fractional','integer'):
                for mask in ('samples','pulses'):
                    with self.subTest(convention=convention,offset=offset,mask=mask):
                        echo=self.fixture(convention,offset,mask)
                        expected=self.golden(echo)
                        prepared=prepare_chirp_pulse_profiles(echo,{'motion_compensation':'none'})
                        old_total=(abs(prepared['aligned'])**2).sum(axis=1)
                        # Prove every cell distinguishes old aligned/cropped data.
                        self.assertGreater(np.max(abs(old_total-expected[0])), .01)
                        lag=max(2,int(np.ceil(echo.metadata['pulse_width_s']/np.median(np.diff(echo.fast_time_s)))))-1
                        cropped_total=(abs(prepared['full_compressed'][:,lag:lag+echo.iq.shape[1]])**2).sum(axis=1)
                        self.assertGreater(np.max(abs(cropped_total-expected[0])), .01)
                        features,_,_=chirp_pulse_range_features(echo,{'motion_compensation':'none'})
                        np.testing.assert_allclose(features.total_power,expected[0],rtol=1e-12,atol=1e-11)
                        np.testing.assert_allclose(features.centroid_delay_s,expected[1],rtol=1e-12,atol=1e-15)
                        np.testing.assert_allclose(features.energy_width_90_s,expected[2],atol=1e-15,equal_nan=True)

    def test_streamed_chunks_reproduce_the_full_correlation_features(self):
        for offset, mask in (('fractional', 'samples'), ('integer', 'pulses')):
            for output in ('centroid_compensated', 'raw_baseband'):
                echo = self.fixture('centered', offset, mask)
                # 夹具本身每行同值，且 90% 能量宽度在整份夹具上只有一个取值；按行
                # 改幅度只改总能量、不改宽度形状。这里每第 3 行再加两个内侧样点，
                # 使宽度逐行不同（实测 13 个不同取值），否则宽度的分块不变性可以靠
                # "两边都是常数"蒙混过去。
                echo.iq = echo.iq * (1.0 + 0.5 * np.sin(np.arange(len(echo.iq))))[:, None]
                echo.iq[::3, 25] = 2.0
                echo.iq[::3, 30] = -2.0
                echo.metadata['echo_output_reference'] = output
                config = {'motion_compensation': 'auto'}
                direct, direct_role, direct_runs = chirp_pulse_range_features(echo, config)
                # 夹具必须非退化，否则"分块相等"可以靠两边都一样而蒙混过关。
                self.assertGreater(len(np.unique(direct.total_power)), 10)
                finite_width = direct.energy_width_90_s[np.isfinite(direct.energy_width_90_s)]
                self.assertGreater(len(np.unique(finite_width)), 5)
                self.assertTrue(np.any(~np.isfinite(direct.energy_width_90_s)))
                self.assertFalse(np.array_equal(direct.total_power, np.roll(direct.total_power, 1)))
                for chunk in (1, 3, 7, len(echo.iq), len(echo.iq) + 5):
                    with self.subTest(offset=offset, mask=mask, output=output, chunk=chunk):
                        streamed, role, runs = streamed_chirp_pulse_features(echo, config, chunk)
                        self.assertEqual(role, direct_role)
                        np.testing.assert_array_equal(runs, direct_runs)
                        np.testing.assert_array_equal(streamed.times_s, direct.times_s)
                        np.testing.assert_array_equal(streamed.total_power, direct.total_power)
                        np.testing.assert_array_equal(streamed.centroid_delay_s, direct.centroid_delay_s)
                        np.testing.assert_array_equal(
                            streamed.energy_width_90_s, direct.energy_width_90_s
                        )

    def test_streamed_chunks_follow_row_rates_and_the_receive_clock(self):
        for role in ('scatter_centroid', 'receive_centroid'):
            echo = self.fixture('centered', 'fractional', 'samples')
            echo.iq = echo.iq * (1.0 + 0.5 * np.sin(np.arange(len(echo.iq))))[:, None]
            echo.metadata['echo_output_reference'] = 'raw_baseband'
            # 每行不同的路径率、公共路径与偏移让运动补偿逐行不同：分块若不按行
            # 同步切这些字段，除第 0 块以外都会用错行的补偿量。
            echo.common_path_rate_m_s = np.linspace(-15000.0, 15000.0, len(echo.iq))
            config = {'motion_compensation': 'auto', 'period_time_role': role}
            direct, direct_role, direct_runs = chirp_pulse_range_features(echo, config)
            self.assertEqual(
                direct_role,
                'receive_elapsed_s' if role == 'receive_centroid' else 'scatter_elapsed_s',
            )
            self.assertGreater(len(np.unique(direct.total_power)), 10)
            self.assertFalse(np.array_equal(direct.times_s, echo.scatter_elapsed_s)
                             and role == 'receive_centroid')
            for chunk in (1, 5, 79, 80):
                with self.subTest(role=role, chunk=chunk):
                    streamed, streamed_role, runs = streamed_chirp_pulse_features(
                        echo, config, chunk
                    )
                    self.assertEqual(streamed_role, direct_role)
                    np.testing.assert_array_equal(runs, direct_runs)
                    np.testing.assert_array_equal(streamed.times_s, direct.times_s)
                    np.testing.assert_array_equal(streamed.total_power, direct.total_power)
                    np.testing.assert_array_equal(streamed.centroid_delay_s, direct.centroid_delay_s)
                    np.testing.assert_array_equal(
                        streamed.energy_width_90_s, direct.energy_width_90_s
                    )

    def test_pulse_chunk_budget_is_the_largest_fitting_chunk(self):
        echo = self.fixture('centered', 'equal', 'pulses')
        # 用真正的匹配滤波输出量出全相关行长，而不是照抄实现里的公式重算。
        probe = matched_filter_chirp(
            np.asarray(echo.iq)[:1], echo.fast_time_s,
            echo.metadata['pulse_width_s'], echo.metadata['pulse_bandwidth_hz'],
            baseband_convention=echo.metadata['baseband_convention'], full_output=True,
        )
        per_pulse = probe.shape[1] * np.dtype(np.complex128).itemsize
        for multiple in (1, 10, 37):
            chunk = bounded_pulse_chunk_size(echo, per_pulse * multiple)
            self.assertEqual(chunk, multiple)
            # 最大性：再放一个脉冲就超预算。
            self.assertLessEqual(chunk * per_pulse, per_pulse * multiple)
            self.assertGreater((chunk + 1) * per_pulse, per_pulse * multiple)
        self.assertEqual(bounded_pulse_chunk_size(echo, per_pulse * 10 - 1), 9)
        # 预算小于一个脉冲时有意钳到 1，此时明知超预算。
        self.assertEqual(bounded_pulse_chunk_size(echo, 1), 1)
        with self.assertRaises(ValueError):
            bounded_pulse_chunk_size(echo, 0)

    def test_streamed_features_reject_inputs_the_chunking_cannot_serve(self):
        echo = self.fixture('centered', 'equal', 'pulses')
        with self.assertRaises(ValueError):
            streamed_chirp_pulse_features(echo, {'motion_compensation': 'none'}, 0)
        flat = self.fixture('centered', 'equal', 'pulses')
        flat.iq = flat.iq[0]
        with self.assertRaises(ValueError):
            streamed_chirp_pulse_features(flat, {'motion_compensation': 'none'}, 4)
        empty = self.fixture('centered', 'equal', 'pulses')
        empty.iq = empty.iq[:0]
        with self.assertRaises(ValueError):
            streamed_chirp_pulse_features(empty, {'motion_compensation': 'none'}, 4)

    def test_streamed_features_reject_row_fields_that_cannot_be_sliced(self):
        # list 输入下若静默跳过切片，每块都会重复整行：80 脉冲会得到 960 个观测。
        listed = self.fixture('centered', 'equal', 'pulses')
        for name in ('iq', 'valid', 'scatter_elapsed_s', 'row_fast_time_offset_s'):
            setattr(listed, name, list(getattr(listed, name)))
        with self.assertRaisesRegex(ValueError, 'ndarray'):
            streamed_chirp_pulse_features(listed, {'motion_compensation': 'none'}, 7)
        # 第一维不等于脉冲数的逐行字段同样拒绝，避免"同一回波因块大小不同而
        # 分别给出不同数字或不同异常"。
        short = self.fixture('centered', 'equal', 'pulses')
        short.run_id = np.zeros(40, dtype=int)
        with self.assertRaisesRegex(ValueError, '第一维等于脉冲数'):
            streamed_chirp_pulse_features(short, {'motion_compensation': 'none'}, 40)

    def test_streamed_preserves_global_nonfinite_rate_fallback(self):
        echo = self.fixture('centered', 'fractional', 'samples')
        echo.metadata['echo_output_reference'] = 'raw_baseband'
        echo.common_path_rate_m_s = np.full(80, 15000.)
        finite, _, _ = chirp_pulse_range_features(echo, {'motion_compensation': 'auto'})
        echo.common_path_rate_m_s[-1] = np.nan
        expected, role, runs = chirp_pulse_range_features(echo, {'motion_compensation': 'auto'})
        # The fixture must distinguish the global fallback from finite-rate compensation.
        self.assertGreater(np.max(abs(finite.total_power - expected.total_power)), .01)
        for size in (1, 7, 79, 80, 85):
            actual, actual_role, actual_runs = streamed_chirp_pulse_features(
                echo, {'motion_compensation': 'auto'}, size)
            self.assertEqual(actual_role, role)
            np.testing.assert_array_equal(actual_runs, runs)
            for name in ('times_s', 'total_power', 'centroid_delay_s', 'energy_width_90_s'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        self.assertTrue(np.isnan(echo.common_path_rate_m_s[-1]))

    def test_streamed_preserves_global_offset_validation(self):
        for field in ('row_fast_time_offset_s', 'centroid_fractional_offset_s'):
            echo = self.fixture('centered', 'equal', 'pulses')
            if field == 'centroid_fractional_offset_s':
                echo.row_fast_time_offset_s[:] = np.nan
            getattr(echo, field)[-1] = np.nan
            with self.assertRaises(ValueError):
                chirp_pulse_range_features(echo, {'motion_compensation': 'none'})
            for size in (1, 7, 79, 80):
                with self.assertRaises(ValueError):
                    streamed_chirp_pulse_features(echo, {'motion_compensation': 'none'}, size)

    def test_streamed_does_not_build_unused_alignment_or_duplicate_width(self):
        echo = self.fixture('centered', 'fractional', 'samples')
        import inversion.src.inversion as implementation
        with patch.object(implementation, 'align_profiles_to_common_delay',
                          side_effect=AssertionError('unused alignment')):
            with patch.object(implementation, 'power_energy_width_90_s',
                              side_effect=AssertionError('duplicate width')):
                streamed_chirp_pulse_features(echo, {'motion_compensation': 'none'}, 7)

    def test_actual_fallback_search_receives_native_moments(self):
        echo=self.fixture('centered','fractional','samples')
        config={**test_chirp_contracts.ChirpInversionTests.config,'cpi_duration_s':20.,'motion_compensation':'none'}
        expected=self.golden(echo)
        inputs=[]
        def capture(times,values,*args,**kwargs):
            inputs.append(np.array(values))
            raise PeriodSearchUnavailable('test_capture', 'test capture only')
        with patch('inversion.src.inversion.multi_harmonic_lomb_scargle',side_effect=capture):
            with self.assertRaisesRegex(PeriodSearchUnavailable,'没有可估计周期的特征'):
                estimate_rotation(echo,config)
        self.assertEqual(len(inputs),3)
        for actual,wanted in zip(inputs,expected):
            np.testing.assert_allclose(actual,wanted,rtol=1e-12,atol=1e-11,equal_nan=True)

    def test_inversion_routes_short_coherent_runs_before_full_preparation(self):
        import inversion.src.inversion as implementation
        echo = chirp_echo('centroid_compensated')
        config = {**test_chirp_contracts.ChirpInversionTests.config, 'cpi_duration_s': 20.}
        expected, _, _ = chirp_pulse_range_features(echo, config)
        original = prepare_chirp_pulse_profiles
        row_counts = []
        def bounded_prepare(chunk, *args, **kwargs):
            row_counts.append(len(chunk.iq))
            if len(chunk.iq) > 16:
                raise MemoryError('whole-array allocation sentinel')
            return original(chunk, *args, **kwargs)
        with patch.object(implementation, 'prepare_chirp_pulse_profiles', side_effect=bounded_prepare), \
             patch.object(implementation, 'bounded_pulse_chunk_size', return_value=7):
            result = estimate_rotation(echo, config)
        self.assertTrue(row_counts)
        self.assertLessEqual(max(row_counts), 16)
        self.assertEqual(result.search_diagnostics['processing_path'], 'pulse_fallback')
        for name in ('times_s', 'total_power', 'centroid_delay_s', 'energy_width_90_s'):
            np.testing.assert_array_equal(getattr(result.features, name), getattr(expected, name))

    def test_fallback_search_matches_direct_features_and_keeps_all_rows(self):
        import inversion.src.inversion as implementation
        echo = chirp_echo('raw_baseband')
        echo.common_path_rate_m_s = np.linspace(-15000., 15000., 80)
        echo.run_id = np.repeat(np.arange(10), 8)
        echo.coherence_id = echo.run_id.copy()
        echo.valid[5] = False  # gap must not merge coherent blocks
        config = {**test_chirp_contracts.ChirpInversionTests.config, 'cpi_duration_s': 20.}
        direct = chirp_pulse_range_features(echo, config)
        with patch.object(implementation, 'streamed_chirp_pulse_features', return_value=direct):
            reference = estimate_rotation(echo, config)
        for size in (1, 7, 79, 80, 85):
            with patch.object(implementation, 'bounded_pulse_chunk_size', return_value=size):
                actual = estimate_rotation(echo, config)
            self.assertEqual(actual.search_diagnostics['observation_count'], 80)
            self.assertLessEqual(len(actual.dynamic_spectrum.times_s), min(128, size, 80))
            for name, expected in reference.periods.items():
                self.assertEqual(actual.periods[name].best_period_s, expected.best_period_s)
                np.testing.assert_array_equal(actual.periods[name].grid_scores, expected.grid_scores)
            self.assertEqual(actual.search_diagnostics['weight_source'], 'valid_sample_fraction')

    def test_minimum_template_and_raw_none(self):
        echo=self.fixture('centered','fractional','pulses')
        echo.metadata['pulse_width_s']=2e-5
        echo.metadata['echo_output_reference']='raw_baseband'
        wanted=self.golden(echo)
        actual,_,_=chirp_pulse_range_features(echo,{'motion_compensation':'none'})
        for got,expected in zip((actual.total_power,actual.centroid_delay_s,actual.energy_width_90_s),wanted):
            np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-11,equal_nan=True)

    def test_raw_auto_compensation_uses_full_native_statistics(self):
        echo=self.fixture('centered','fractional','samples')
        echo.metadata['echo_output_reference']='raw_baseband'
        echo.common_path_rate_m_s=np.linspace(-15000.,15000.,80)
        # Independent evaluation of the unchanged compensation convention.
        valid=echo.valid
        samples=np.where(valid,echo.iq,0)
        compensated=np.zeros_like(samples)
        c=299792458.
        for row in range(len(samples)):
            source=echo.fast_time_s+echo.row_fast_time_offset_s[row]
            query=source/(1-echo.common_path_rate_m_s[row]/c)
            z=np.interp(query,source,samples[row].real,left=0,right=0)+1j*np.interp(query,source,samples[row].imag,left=0,right=0)
            path=echo.tx_range_m[row]+echo.rx_range_m[row]+echo.common_path_rate_m_s[row]*query
            compensated[row]=z*np.exp(2j*np.pi*np.remainder(echo.metadata['carrier_frequency_hz']*path/c,1))
        actual,_,_=chirp_pulse_range_features(echo,{'motion_compensation':'auto'})
        echo.iq=np.where(valid,compensated,0)
        expected=self.golden(echo)
        for got,wanted in zip((actual.total_power,actual.centroid_delay_s,actual.energy_width_90_s),expected):
            np.testing.assert_allclose(got,wanted,rtol=1e-12,atol=1e-11,equal_nan=True)


if __name__=='__main__':
    unittest.main()
