"""Dispatch explicit precision strategies without changing legacy defaults."""
import numpy as np


def select_precision_period(times, features, runs, weights, echo, config, *,
                            processing_path, phase_offset=None):
    policy = config['period_selection']
    if processing_path != 'cpi':
        raise ValueError(f'{policy} 仅支持 Chirp CPI，不能使用逐脉冲回退')
    if config.get('period_time_role', 'scatter_centroid') != 'scatter_centroid':
        raise ValueError(f'{policy} 仅支持原散射特征时刻')
    if any(np.any(getattr(echo, name, [])) for name in ('window_overlap', 'signal_echo_overlap')):
        raise ValueError(f'{policy} 尚不支持共享ADC保存行或物理回波重叠')
    if policy == 'predictive_skill_energy':
        try:
            from .predictive_energy import select_predictive_energy_period
        except ImportError:
            from predictive_energy import select_predictive_energy_period
        return select_predictive_energy_period(times, features, runs, weights=weights,
            phase_offset=phase_offset if config.get('los_phase_compensation', False) else None,
            min_period=config['period_min_s'], max_period=config['period_max_s'],
            period_grid_size=config['period_grid_size'], refine_period=config.get('refine_period', True))
    if policy == 'spherical_response':
        try:
            from .spherical_response import select_spherical_response_period
        except ImportError:
            from spherical_response import select_spherical_response_period
        return select_spherical_response_period(times, features, runs,
            scatter_elapsed_s=echo.scatter_elapsed_s,
            tx_los_icrs=echo.tx_los_icrs, rx_los_icrs=echo.rx_los_icrs,
            spin_axis_ra_deg=config['spin_axis_ra_deg'], spin_axis_dec_deg=config['spin_axis_dec_deg'],
            period_min_s=config['period_min_s'], period_max_s=config['period_max_s'],
            period_grid_size=config['period_grid_size'], effective_sample_count=weights)
    raise ValueError(f'未知精度策略：{policy}')
