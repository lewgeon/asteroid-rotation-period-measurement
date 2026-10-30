"""Known-pole response regression; descriptive whole-Run CV, not shape inversion.

The deliberately narrow first release accepts exactly 100 Runs of five Chirp
CPIs and nearly monostatic geometry. Ridge uses unit CPI weights and RSS sums.
"""
from __future__ import annotations

from contextlib import nullcontext
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, lpmv

FEATURE_NAMES = ('total_power', 'doppler_centroid', 'doppler_bandwidth',
                 'range_centroid', 'range_energy_width')
DEGREES = (1, 2, 3, 4, 6)
ALPHAS = np.array([1e-8, 1e-5, 1e-3, .1, 10.], dtype=np.float64)


def _geometry(times, scatter_elapsed_s, tx_los_icrs, rx_los_icrs, ra, dec):
    """Interpolate vector components first, then normalize each LOS."""
    if ra is None or dec is None or not np.all(np.isfinite([ra, dec])):
        raise ValueError('spherical_response 需要显式已知极轴')
    if not (0 <= ra <= 360 and -90 <= dec <= 90):
        raise ValueError('极轴 RA/Dec 超出范围')
    epochs = np.asarray(scatter_elapsed_s, dtype=np.float64)
    tx, rx = (np.asarray(v, dtype=np.float64) for v in (tx_los_icrs, rx_los_icrs))
    if (epochs.ndim != 1 or len(epochs) < 2 or tx.shape != (len(epochs), 3)
            or rx.shape != tx.shape or not np.all(np.isfinite(epochs))
            or not np.all(np.isfinite(tx)) or not np.all(np.isfinite(rx))):
        raise ValueError('LOS 时刻及分量必须完整、有限且等长')
    if np.any(np.linalg.norm(tx, axis=1) == 0) or np.any(np.linalg.norm(rx, axis=1) == 0):
        raise ValueError('不支持零 LOS')
    order = np.argsort(epochs, kind='stable')
    epochs, tx, rx = epochs[order], tx[order], rx[order]
    repeats = np.diff(epochs) == 0
    if np.any(repeats):
        if np.any(tx[1:][repeats] != tx[:-1][repeats]) or np.any(rx[1:][repeats] != rx[:-1][repeats]):
            raise ValueError('相同散射时刻的 LOS 分量不一致')
        keep = np.r_[True, ~repeats]
        epochs, tx, rx = epochs[keep], tx[keep], rx[keep]
    if len(epochs) < 2 or np.any(times < epochs[0]) or np.any(times > epochs[-1]):
        raise ValueError('特征时刻超出 LOS 范围，禁止外推')
    vectors = []
    for values in (tx, rx):
        v = np.column_stack([np.interp(times, epochs, values[:, i]) for i in range(3)])
        norm = np.linalg.norm(v, axis=1)
        if np.any(norm == 0):
            raise ValueError('插值后出现零 LOS')
        vectors.append(v/norm[:, None])
    tx, rx = vectors
    look = tx+rx
    norms = np.linalg.norm(look, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('零合成 LOS')
    beta = np.rad2deg(np.arccos(np.clip(np.sum(tx*rx, axis=1), -1, 1)))
    if np.max(beta) >= .000155:
        raise ValueError('spherical_response 首版仅支持双站夹角 <0.000155°')
    ra, dec = np.deg2rad([ra, dec])
    axis = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    helper = np.array([1., 0., 0.]) if abs(axis[2]) > .9 else np.array([0., 0., 1.])
    e1 = np.cross(helper, axis)
    e1 /= np.linalg.norm(e1)
    basis = np.column_stack([e1, np.cross(axis, e1), axis])
    body = (look/norms[:, None])@basis
    if np.any(np.linalg.norm(body[:, :2], axis=1) <= 1e-12):
        raise ValueError('极轴横向投影范数过小，经度未定义')
    return np.clip(body[:, 2], -1, 1), -np.arctan2(body[:, 1], body[:, 0]), float(np.max(beta))


def _terms(z, degree):
    result = []
    for l in range(1, degree+1):
        for m in range(l+1):
            norm = np.sqrt((2*l+1)*np.exp(gammaln(l-m+1)-gammaln(l+m+1)))
            amplitude = norm*lpmv(m, l, z)/(l*(l+1))
            result.append((m, False, amplitude))
            if m:
                result.append((m, True, amplitude))
    return result


def _design(times, phase, terms, frequency):
    angle = 2*np.pi*frequency*times+phase
    x = np.column_stack([amp*(np.sin(m*angle) if sine else np.cos(m*angle))
                         for m, sine, amp in terms])
    return x-x.mean(axis=0)


class _ResponseScore:
    def __init__(self, times, y, phase, z, group_indices, degree):
        self.times = times-times.mean()
        self.y = (y-y.mean(axis=0))/y.std(axis=0)
        self.phase = phase
        self.terms = _terms(z, degree)
        self.indices = group_indices
        # This standard deviation belongs to the remaining training CPIs.
        self.train_std = np.array([self.y[np.setdiff1d(np.arange(len(y)), g)].std(axis=0)
                                   for g in group_indices])
        if np.any(self.train_std <= 0) or not np.all(np.isfinite(self.train_std)):
            raise ValueError('留出 Run 后训练特征为常数')

    def evaluate(self, frequency, details=False):
        x = _design(self.times, self.phase, self.terms, frequency)
        u, singular, _ = np.linalg.svd(x, full_matrices=False)
        filters = singular[None, :]**2/(singular[None, :]**2+ALPHAS[:, None])
        fits = np.einsum('ij,aj,jk->aik', u, filters, u.T@self.y)
        residuals = self.y[None, :, :]-fits
        ug = u[self.indices]
        hat = 1/len(x)+np.einsum('gik,ak,gjk->agij', ug, filters, ug)
        errors = np.linalg.solve(np.eye(5)[None, None, :, :]-hat,
                                 residuals[:, self.indices, :])
        errors /= self.train_std[None, :, None, :]
        losses = np.mean(errors**2, axis=(1, 2))
        selected = losses.argmin(axis=0)
        loss = float(np.mean(losses[selected, np.arange(5)]))
        if not np.isfinite(loss):
            raise ValueError('球面响应评分出现非有限值')
        if details:
            return dict(score=loss, per_feature_min_cv=losses[selected, np.arange(5)].tolist(),
                        alpha_by_feature=ALPHAS[selected].tolist(), singular_values=singular.tolist(),
                        effective_dof=(1+filters.sum(axis=1))[selected].tolist())
        return loss


def select_spherical_response_period(times, features, run_ids, *, scatter_elapsed_s,
        tx_los_icrs, rx_los_icrs, spin_axis_ra_deg, spin_axis_dec_deg,
        period_min_s, period_max_s, period_grid_size, waveform='chirp',
        effective_sample_count=None):
    """Search all fixed degrees and all strict local minima plus endpoints.

    Features are the five production-name arrays. Sample counts are checked
    when supplied, but never used as regression weights. CPU float64 SVD is
    the stable reference implementation, independent of CUDA availability.
    """
    t = np.asarray(times, dtype=np.float64)
    ids = np.asarray(run_ids)
    if waveform != 'chirp':
        raise ValueError('spherical_response 仅支持 Chirp CPI')
    if t.shape != (500,) or ids.shape != t.shape or not np.all(np.isfinite(t)) or np.ptp(t) <= 0:
        raise ValueError('spherical_response 首版需要 500 个完整有效 CPI 时刻和 Run 标识')
    if ids.dtype.kind in 'fc' and not np.all(np.isfinite(ids)):
        raise ValueError('Run 标识必须有限')
    groups, counts = np.unique(ids, return_counts=True)
    if len(groups) != 100 or np.any(counts != 5):
        raise ValueError('spherical_response 首版需要 100 个 Run，每个恰有 5 CPI')
    indices = np.array([np.flatnonzero(ids == group) for group in groups])
    if effective_sample_count is not None:
        weights = np.asarray(effective_sample_count, dtype=np.float64)
        if weights.shape != t.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('有效样本数必须完整、有限且为正')
    if set(features) != set(FEATURE_NAMES):
        raise ValueError('spherical_response 需要固定五项完整特征')
    arrays = [np.asarray(features[name], dtype=np.float64) for name in FEATURE_NAMES]
    if any(v.shape != t.shape or not np.all(np.isfinite(v)) for v in arrays):
        raise ValueError('五项特征不能缺样或包含非有限值')
    y = np.column_stack(arrays)
    scale = np.max(np.abs(y), axis=0)
    if np.any(scale == 0) or np.any(np.ptp(y, axis=0) == 0):
        raise ValueError('不支持常数特征')
    normalized = y/scale
    normalized -= normalized.mean(axis=0)
    std = normalized.std(axis=0)
    if np.any(std <= 0):
        raise ValueError('不支持近常数特征')
    normalized /= std
    for c in range(5):
        for previous in range(c):
            if min(np.max(abs(normalized[:, c]-normalized[:, previous])),
                   np.max(abs(normalized[:, c]+normalized[:, previous]))) <= 1e-10:
                raise ValueError('不支持仿射重复通道')
    if not (np.all(np.isfinite([period_min_s, period_max_s])) and 0 < period_min_s < period_max_s):
        raise ValueError('周期范围必须满足 0 < min < max')
    if (isinstance(period_grid_size, (bool, np.bool_)) or not np.isfinite(period_grid_size)
            or int(period_grid_size) != period_grid_size or period_grid_size < 3):
        raise ValueError('period_grid_size 必须为至少 3 的整数')
    z, phase, beta = _geometry(t, scatter_elapsed_s, tx_los_icrs, rx_los_icrs,
                               spin_axis_ra_deg, spin_axis_dec_deg)
    frequencies = np.linspace(1/period_max_s, 1/period_min_s, int(period_grid_size))
    candidates, summaries = [], []
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=1)
    except ImportError:
        context = nullcontext()
    with context:
        for degree in DEGREES:
            # Affine scale invariance also has to hold numerically: the public
            # preprocessing above avoids squaring extreme original magnitudes.
            scorer = _ResponseScore(t, normalized, phase, z, indices, degree)
            curve = np.array([scorer.evaluate(f) for f in frequencies])
            local = np.flatnonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] < curve[2:]))+1
            for j in np.r_[0, local, len(curve)-1]:
                lo, hi = frequencies[max(0, j-1)], frequencies[min(len(curve)-1, j+1)]
                opt = minimize_scalar(scorer.evaluate, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-14})
                loss, frequency = min((float(curve[j]), frequencies[j]), (float(opt.fun), opt.x))
                candidate = dict(period_s=float(1/frequency), degree=degree, score=loss,
                                 frequency_bracket_hz=[float(lo), float(hi)],
                                 refinement_success=bool(opt.success), grid_index=int(j))
                candidates.append(candidate)
            best = min((c for c in candidates if c['degree'] == degree),
                       key=lambda c: (c['score'], c['period_s']))
            summaries.append(dict(best, **{k: v for k, v in scorer.evaluate(1/best['period_s'], True).items()
                                           if k != 'score'}))
    candidates.sort(key=lambda c: (c['score'], c['degree'], c['period_s']))
    winner = next(s for s in summaries if s['degree'] == candidates[0]['degree'])
    diagnostics = dict(search_protocol='original_grid_all_minima_endpoints',
                       validation_mode='whole_run_leave_one_out', run_count=100, cpi_count=500,
                       degree=winner['degree'], alpha_by_feature=winner['alpha_by_feature'],
                       singular_values=winner['singular_values'], effective_dof=winner['effective_dof'],
                       degrees=list(DEGREES), alphas=ALPHAS.tolist(), degree_results=summaries,
                       axis_prior={'source': 'inversion_config', 'ra_deg': float(spin_axis_ra_deg),
                                   'dec_deg': float(spin_axis_dec_deg)},
                       max_bistatic_angle_deg=beta, grid_count=len(frequencies),
                       weights='unit_cpi', compute_execution='cpu_float64_svd',
                       score_direction='minimize',
                       score_definition='mean_channels(min_alpha(mean_cpi(train_std_normalized_run_deleted_residual_squared)))',
                       independent_probability_evidence=False,
                       frequency_sensitivity_role='diagnostic_only_no_uncertainty_interval')
    return dict(winner, policy='spherical_response', strategy='spherical_response', status='candidate',
                message='已知极轴球面响应整 Run 留出评分', warnings=[], support=5,
                relative_spread=None, evidence=[dict(feature=name, cv_loss=loss, alpha=alpha)
                    for name, loss, alpha in zip(FEATURE_NAMES, winner['per_feature_min_cv'],
                                                  winner['alpha_by_feature'])],
                candidates=candidates, diagnostics=diagnostics)
