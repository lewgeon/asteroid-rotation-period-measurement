"""Conditional finite-sample period covers for externally calibrated noise.

Known zero-mean Gaussian marginals suffice; independence is not assumed.
This module never estimates noise or systematic bounds from fitted residuals.
"""
from fractions import Fraction
import math
import numpy as np
from .bounded_period import outer_period_set, _outward_float


def gaussian_simultaneous_budget(standard_deviations, alpha=.0027):
    """Conservative simultaneous |epsilon_i| <= k*sigma_i with risk <= alpha.

    An integer k is chosen using a Gaussian Chernoff bound and union bound.
    A rational positive Taylor sum certifies exp(k*k/2) >= 2*N/alpha;
    floating log/sqrt only propose k and cannot certify an unsafe budget.
    """
    sigma = np.asarray(standard_deviations, dtype=float)
    if sigma.ndim != 1 or not len(sigma) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError('外部已知标准差必须为非空一维非负有限数组')
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError('alpha必须为0与1之间的有限概率')
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha必须为0与1之间的有限概率')
    target = Fraction(2*len(sigma)) / Fraction(alpha)
    k = max(1, math.ceil(math.sqrt(2*(math.log(2*len(sigma))-math.log(alpha)))))
    while True:
        x = Fraction(k*k, 2)
        term = lower_exp = Fraction(1)
        for j in range(1, k*k+129):
            term *= x / j
            lower_exp += term
            if lower_exp >= target:
                break
        if lower_exp >= target:
            break
        k += 1
    bounds = []
    for value in sigma:
        exact = Fraction(float(value))*k
        if exact > Fraction(float(np.finfo(float).max)):
            raise ValueError('概率预算超过有限binary64范围，请缩放特征及标准差')
        bounds.append(_outward_float(exact, True))
    return dict(pointwise_absolute_bounds=bounds, k_sigma=k, alpha=alpha,
        observation_count=len(sigma),
        certified_union_failure_upper_bound=_outward_float(Fraction(2*len(sigma))/lower_exp, True),
        method='Gaussian_Chernoff_union_bound_integer_k_rational_exp_certificate',
        assumptions='外部已知零均值Gaussian边际及标准差；允许相关，不从残差估计标准差')


def gaussian_outer_period_set(times, values, period_min, period_max,
                              coefficient_caps, systematic_components,
                              standard_deviations, *, calibration_provenance,
                              alpha=.0027, **solver_options):
    """Research confidence outer cover, conditional on externally valid budgets.

    Under the declared harmonic family, coefficient bounds, systematic bounds,
    fixed domain and calibrated Gaussian marginals, truth lies in this cover
    with probability at least 1-alpha. It may be disconnected, empty or broad.
    This is not a Gaussian approximation for the final consensus estimator.
    """
    if not isinstance(calibration_provenance, str) or not calibration_provenance.strip():
        raise ValueError('必须说明外部噪声与系统误差预算的校准来源及适用范围')
    if not systematic_components or 'gaussian_simultaneous' in systematic_components:
        raise ValueError('显式声明系统误差分量（无系统误差时填0），保留gaussian_simultaneous名称')
    sigma = np.asarray(standard_deviations, float)
    if sigma.shape != np.asarray(values).shape:
        raise ValueError('外部标准差与观测必须逐项同形状')
    budget = gaussian_simultaneous_budget(sigma, alpha)
    cover = outer_period_set(times, values, period_min, period_max, coefficient_caps,
        {**systematic_components, 'gaussian_simultaneous': budget['pointwise_absolute_bounds']},
        **solver_options)
    cover.update(confidence_scope='conditional_on_external_calibration_and_declared_model',
        requested_failure_probability=budget['alpha'],
        confidence_lower_bound=_outward_float(Fraction(1)-Fraction(budget['alpha']), False),
        gaussian_budget=budget, calibration_provenance=calibration_provenance,
        final_3sigma_interval_s=None, coverage_calibrated=False,
        statistical_warning='模型内有限样本置信外包络；不是100%确定界，不认证实测校准；空集合是拒绝/失败事件，不能解释为零误差。')
    return cover
