"""Common-period predictive scoring without a shape or period truth prior.

The objective is a robust descriptive score, not a likelihood or probability.
Whole Runs stay together in folds. Affine copies with identical valid masks
cannot acquire extra votes; different masks and general correlation remain.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, signal

PREDICTIVE_POLICY = 'median_whole_run_predictive_skill_v1'


class JointPeriodUnavailable(ValueError):
    """Expected missing information, distinct from an invalid configuration."""


def _grid(periods, minimum, maximum):
    grids = []
    candidates = []
    for item in periods.values():
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        grid = np.asarray(get('grid_periods_s', []), dtype=float)
        if grid.ndim != 1:
            raise ValueError('共同周期搜索网格必须是一维数组')
        grids.extend(grid[np.isfinite(grid) & (grid > 0)].tolist())
        for candidate in get('candidates', []):
            value = candidate.get('period_s') if isinstance(candidate, dict) else candidate.period_s
            if np.isfinite(value) and value > 0:
                candidates.append(float(value))
    values = grids or candidates
    if not values and (minimum is None or maximum is None):
        raise ValueError('共同周期搜索需要原周期网格或显式的周期范围')
    minimum = float(min(values) if minimum is None else minimum)
    maximum = float(max(values) if maximum is None else maximum)
    if not (np.isfinite(minimum) and np.isfinite(maximum) and 0 < minimum < maximum):
        raise ValueError('共同周期范围必须满足 0 < min_period < max_period')
    if grids:
        values = np.asarray(grids)
        values = values[(values >= minimum) & (values <= maximum)]
        values = np.unique(np.r_[minimum, values, maximum])
        source = 'full_input_frequency_grid'
    else:
        # Summary-only callers have discarded their grid: reconstruct an
        # explicit full frequency grid, never mistake sparse candidates for it.
        values = 1/np.linspace(1/maximum, 1/minimum, 4000)
        values = np.sort(values)
        source = 'reconstructed_4000_point_frequency_grid'
    if len(values) < 2:
        raise ValueError('共同周期搜索至少需要两个网格点')
    return values, minimum, maximum, source


def select_predictive_period(times, features, runs, periods, *, weights=None,
                             harmonics=1, phase_offset=None, min_period=None,
                             max_period=None, refine_period=True):
    """Evaluate every channel at the same periods and select median CV skill.

    ``periods`` accepts PeriodEstimate objects or JSON summaries. At least ten
    positively weighted Runs enable five folds, assigned by chronological Run
    order; otherwise a clearly labelled shared-fit fallback is returned. The
    harmonic order is supplied by the caller and is never chosen using truth.
    Missing samples are excluded per channel. Affine-copy deduplication requires
    identical valid masks and normalized differences at most 1e-10; different
    missingness is not silently treated as evidence of equivalence. A candidate must be estimable in
    every training fold of every retained channel; the channel set is fixed
    before searching, so a difficult channel cannot disappear at one period.
    """
    t = np.asarray(times, dtype=float)
    groups = np.asarray(runs)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(len(t)) if phase_offset is None else np.asarray(phase_offset, dtype=float)
    if t.ndim != 1 or groups.shape != t.shape or w.shape != t.shape or offset.shape != t.shape:
        raise ValueError('共同周期的时刻、Run、权重和相位必须是一维等长数组')
    if not isinstance(refine_period, (bool, np.bool_)):
        raise ValueError('refine_period 必须是布尔值')
    if (isinstance(harmonics, (bool, np.bool_)) or not np.isfinite(harmonics)
            or int(harmonics) != harmonics or harmonics < 1):
        raise ValueError('harmonics 必须为正整数')
    if not np.all(np.isfinite(offset)):
        raise ValueError('共同周期相位补偿必须为有限数')
    harmonics = int(harmonics)
    p = 2*harmonics+1
    observable = np.zeros(len(t), dtype=bool)
    for name, values in features.items():
        if name not in periods:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != t.shape:
            raise ValueError(f'共同周期特征 {name} 与时刻形状不一致')
        observable |= np.isfinite(values)
    # A nominal Run with no available feature is not predictive evidence.
    # Match the active population used by whole-Run resampling.
    base = np.isfinite(t) & np.isfinite(w) & (w > 0) & observable
    if np.count_nonzero(base) < p+3:
        raise JointPeriodUnavailable('共同周期有效观测不足')
    if groups.dtype.kind in 'fc' and not np.all(np.isfinite(groups[base])):
        raise ValueError('有效观测的 Run 标识必须为有限数')
    origin = float(np.min(t[base]))
    t = t-origin
    if not np.all(np.isfinite(t[base])) or np.ptp(t[base]) <= 0:
        raise JointPeriodUnavailable('共同周期需要有限且非零的时间跨度')
    w = np.where(base, w/np.max(w[base]), 0)
    t = np.where(base, t, 0.)
    grid, minimum, maximum, grid_source = _grid(periods, min_period, max_period)
    unique_runs = sorted(np.unique(groups[base]).tolist(),
                         key=lambda key: float(np.median(t[base & (groups == key)])))
    cv = len(unique_runs) >= 10
    fold_count = 5 if cv else 0
    assignment = {key: index % 5 for index, key in enumerate(unique_runs)}
    labels = np.array([assignment.get(key, -1) for key in groups])
    folds = []
    if cv:
        for k in range(5):
            folds.append(dict(train_runs=[key for key in unique_runs if assignment[key] != k],
                              test_runs=[key for key in unique_runs if assignment[key] == k]))
    excluded, duplicates, channel_names, data, masks = {}, {}, [], [], []
    for name, values in features.items():
        if name not in periods:
            excluded[name] = 'not_available_in_individual_search'
            continue
        y = np.asarray(values, dtype=float)
        if y.shape != t.shape:
            raise ValueError(f'共同周期特征 {name} 与时刻形状不一致')
        valid = base & np.isfinite(y)
        if np.count_nonzero(valid) < p+3 or len(np.unique(t[valid])) <= p:
            excluded[name] = 'insufficient_samples'
            continue
        yv = y[valid]
        if np.max(yv) == np.min(yv):
            excluded[name] = 'constant_feature'
            continue
        scale = float(np.max(np.abs(yv)))
        normalized = yv/scale
        normalized -= np.average(normalized, weights=w[valid])
        variance = float(np.average(normalized**2, weights=w[valid]))
        if variance <= 0 or not np.isfinite(variance):
            excluded[name] = 'numerically_constant_feature'
            continue
        normalized /= np.sqrt(variance)
        trend = np.column_stack((np.ones(len(yv)), t[valid]/np.ptp(t[base])))
        root = np.sqrt(w[valid])
        coefficient = np.linalg.lstsq(trend*root[:, None], normalized*root, rcond=None)[0]
        if np.linalg.norm((normalized-trend@coefficient)*root) <= 1e-12*np.linalg.norm(normalized*root):
            excluded[name] = 'linear_trend_without_periodic_modulation'
            continue
        if cv and any(np.count_nonzero(valid & (labels != k)) < p+3
                      or len(np.unique(t[valid & (labels != k)])) <= p
                      or not np.any(valid & (labels == k)) for k in range(5)):
            excluded[name] = 'insufficient_whole_run_fold_samples'
            continue
        z = np.zeros(len(t))
        z[valid] = normalized
        duplicate = next((old for old, old_z, old_mask in zip(channel_names, data, masks)
                          if np.array_equal(valid, old_mask)
                          and min(np.max(abs(z-old_z)), np.max(abs(z+old_z))) <= 1e-10), None)
        if duplicate is not None:
            duplicates[name] = duplicate
            continue
        channel_names.append(name)
        data.append(z)
        masks.append(valid)
    if not data:
        raise JointPeriodUnavailable('共同周期没有可辨识的非恒定、非纯趋势特征')
    z = np.column_stack(data)
    # Channels with the same finite mask share one matrix solve per fold.
    mask_groups = {}
    for j, mask in enumerate(masks):
        mask_groups.setdefault(mask.tobytes(), []).append(j)
    blocks = []
    denominator = np.zeros(len(data))
    for columns in mask_groups.values():
        valid = masks[columns[0]]
        for fold in range(5) if cv else [None]:
            train = valid & (labels != fold) if cv else valid
            test = valid & (labels == fold) if cv else valid
            center = np.average(z[train][:, columns], axis=0, weights=w[train])
            denominator[columns] += np.sum(w[test, None]*(z[test][:, columns]-center)**2, axis=0)
            blocks.append((columns, train, test, np.sqrt(w[train]), z[train][:, columns], z[test][:, columns]))
    if np.any(denominator <= 0):
        raise JointPeriodUnavailable('共同周期留出基线无有效变化')

    def evaluate(period):
        phase = 2*np.pi*t/float(period)+offset
        design = np.column_stack([np.ones(len(t))]+[f(h*phase) for h in range(1, harmonics+1)
                                                    for f in (np.cos, np.sin)])
        numerator = np.zeros(len(data))
        for columns, train, test, root, train_y, test_y in blocks:
            beta, _, rank, _ = np.linalg.lstsq(design[train]*root[:, None], train_y*root[:, None], rcond=None)
            if rank < p:
                return -np.inf, np.full(len(data), -np.inf)
            numerator[columns] += np.sum(w[test, None]*(test_y-design[test]@beta)**2, axis=0)
        skill = 1-numerator/denominator
        return float(np.median(skill)), skill

    scores = np.array([evaluate(value)[0] for value in grid])
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise JointPeriodUnavailable('共同周期搜索在全部候选上退化')
    scored = np.where(finite, scores, -np.inf)
    peaks = np.unique(np.r_[signal.find_peaks(scored)[0], 0, len(grid)-1])
    peaks = peaks[np.isfinite(scored[peaks])]
    # Refining all local maxima avoids rank-order swaps at an arbitrary top-five cut.
    candidates = []
    for index in peaks:
        period = float(grid[index])
        score = float(scores[index])
        if refine_period:
            lower, upper = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
            if lower < upper:
                result = optimize.minimize_scalar(lambda value: -evaluate(value)[0],
                    bounds=(lower, upper), method='bounded')
                if result.success and np.isfinite(result.fun) and -result.fun > score:
                    period, score = float(result.x), float(-result.fun)
        candidates.append(dict(period_s=period, score=score, source=PREDICTIVE_POLICY))
    candidates.sort(key=lambda item: (-item['score'], item['period_s']))
    best = candidates[0]
    score, skills = evaluate(best['period_s'])
    evidence = [dict(feature=name, period_s=best['period_s'], score=float(value), rank=0,
                     role='heldout_predictive_skill' if cv else 'shared_fit_skill')
                for name, value in zip(channel_names, skills)]
    individual_periods = []
    for item in evidence:
        estimate = periods.get(item['feature'])
        value = (estimate.get('best_period_s') if isinstance(estimate, dict)
                 else getattr(estimate, 'best_period_s', None))
        if value is not None and np.isfinite(value) and value > 0:
            item['individual_best_period_s'] = float(value)
            individual_periods.append(float(value))
    individual_peak_deviation = (float(np.median(abs(np.asarray(individual_periods)/best['period_s']-1)))
                                if individual_periods else None)
    warnings = []
    if not cv:
        warnings.append('insufficient_runs_for_cross_validation')
    if score <= 0:
        warnings.append('no_positive_median_predictive_skill')
    boundary = bool(np.isclose(best['period_s'], minimum, rtol=1e-6, atol=0)
                    or np.isclose(best['period_s'], maximum, rtol=1e-6, atol=0))
    if boundary:
        warnings.append('selected_period_at_search_boundary')
    cycles = float(np.ptp(t[base])/best['period_s'])
    if cycles < 2:
        warnings.append('less_than_two_cycles_observed')
    grid_phase_step = float(2*np.pi*harmonics*np.ptp(t[base])*np.max(np.diff(np.sort(1/grid))))
    if grid_phase_step > np.pi/2:
        warnings.append('coarse_grid_may_miss_narrow_peaks')
    distant = [row for row in candidates[1:] if abs(row['period_s']/best['period_s']-1) > .05]
    margin = None if not distant else float(score-distant[0]['score'])
    # A numerical tie certifies ambiguity; small nonzero differences are simply
    # reported. No arbitrary score gap is interpreted as a probability cutoff.
    if margin is not None and margin <= 1e-8:
        warnings.append('distant_period_has_indistinguishable_score')
    status = ('weak' if score <= 0 else 'ambiguous' if boundary or cycles < 2
              or 'distant_period_has_indistinguishable_score' in warnings else 'candidate')
    message = ('整 Run 留出共同周期候选；预测分数不是概率或精度保证。' if cv else
               '有效 Run 少于 10 个，使用共同拟合分数；未进行整 Run 留出验证。')
    return dict(period_s=float(best['period_s']), score=score,
        support=int(np.count_nonzero(skills > 0)), relative_spread=None, evidence=evidence,
        policy=PREDICTIVE_POLICY, status=status, message=message, warnings=warnings,
        candidates=candidates[:10], diagnostics=dict(
            validation_mode='whole_run_cross_validation' if cv else 'shared_fit_fallback',
            fold_count=fold_count, folds=folds, run_count=len(unique_runs),
            unique_channel_count=len(channel_names), channel_names=channel_names,
            duplicate_channels=duplicates, excluded_channels=excluded,
            duplicate_policy='same_finite_mask_affine_within_1e-10',
            harmonics=harmonics, phase_compensation=phase_offset is not None,
            grid_source=grid_source, grid_size=len(grid), period_range_s=[minimum, maximum],
            maximum_harmonic_grid_phase_step_rad=grid_phase_step,
            compute_device='cpu', relative_spread_definition='not_defined_for_a_shared_period_fit',
            individual_peak_median_relative_deviation=individual_peak_deviation,
            rank_deficient_grid_count=int(np.count_nonzero(~finite)),
            refine_period=bool(refine_period), independent_probability_evidence=False,
            score_definition=('median_channels(1-sum_heldout_SSE/sum_training_mean_baseline_SSE)' if cv else
                              'median_channels(1-shared_fit_SSE/shared_mean_baseline_SSE)'),
            selected_boundary=boundary, observed_cycles=cycles,
            distant_candidate_score_margin=margin,
            selection_warning='留出折参与周期选择，分数不是最终独立测试；相关通道与相干散射共同偏差仍可能影响结果。'))
