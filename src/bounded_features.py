"""Deterministic centroid propagation at a declared processed-complex-data seam.

Bounds supplied here must already include upstream receiver, geometry, signal
processing, numerical and model errors. This module does not estimate them from
the same residual being assessed.
"""
from fractions import Fraction
from decimal import Decimal
import numpy as np
from .bounded_period import Interval,_LOW,_HIGH,_outward_float


def range_centroid_budget(samples,delays_s,error_components):
    """Bound change of power centroid for ||true_samples-samples||_2 <= eta.

    Both vectors must use the same known delay coordinates. An error in those
    coordinates is a separate additive centroid term, not included here.
    """
    z=np.asarray(samples,complex).ravel();axis=np.asarray(delays_s,float).ravel()
    if z.shape!=axis.shape or not len(z) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(axis)):
        raise ValueError('Finite complex samples and one delay coordinate per sample required')
    if not error_components:raise ValueError('Explicit propagated complex L2 budgets required')
    energy=Interval(0);eta=Interval(0)
    for value in z:
        energy=energy+Interval.rational(Fraction(float(value.real))**2+Fraction(float(value.imag))**2)
    for value in error_components.values():
        if not np.isfinite(value) or value<0:raise ValueError('Finite nonnegative L2 bounds required')
        eta=eta+Interval.rational(Fraction(float(value)))
    span=Interval.rational(Fraction(float(axis.max()))-Fraction(float(axis.min())))
    root_low=max(Decimal(0),_LOW.sqrt(max(energy.low,Decimal(0))).next_minus(context=_LOW))
    root_high=_HIGH.sqrt(max(energy.high,Decimal(0))).next_plus(context=_HIGH)
    if energy.high==0 or root_low<=eta.high:
        return dict(status='unavailable_zero_energy_possible',centroid_error_bound_s=None,
            reason='The declared perturbation allows zero true energy; a centroid need not exist',physical_model_error_validated=False)
    difference=Interval(root_low,root_high)-eta
    true_energy_floor=(difference*difference).low
    power_l1=(2*Interval(root_low,root_high)*eta+eta*eta).high
    upper=min(span.high,_HIGH.divide(_HIGH.multiply(span.high,power_l1),true_energy_floor))
    if upper>Decimal.from_float(np.finfo(float).max):
        return dict(status='bound_exceeds_float64',centroid_error_bound_s=None,centroid_error_bound_decimal_s=str(upper),
            physical_model_error_validated=False,reason='Finite directed Decimal bound exceeds finite binary64 range')
    return dict(status='conditional_processed_sample_bound',centroid_error_bound_s=_outward_float(Fraction(upper),True),
        error_components_l2=dict(error_components),perturbation_l2_upper=str(eta.high),
        true_energy_lower=str(true_energy_floor),power_change_l1_upper=str(power_l1),
        physical_model_error_validated=False,
        assumptions='True and measured complex vectors have identical known delay coordinates and support; supplied additive L2 bounds cover all complex-data discrepancy at this processing seam. Coordinate errors require an extra bound.')
