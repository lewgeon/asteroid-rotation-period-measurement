"""Five-channel, whole-Run positive predictive skill energy (EV5 contract).

The score is descriptive; it is neither a probability nor an uncertainty bound.
All search and refinement fits use float64 weighted SVD, without normal equations.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

try:  # Package imports and the standalone inversion CLI share this module.
    from .joint_period import JointPeriodUnavailable
except ImportError:
    from joint_period import JointPeriodUnavailable

ENERGY_CHANNELS = ('total_power', 'doppler_centroid', 'doppler_bandwidth',
                   'range_centroid', 'range_energy_width')
ENERGY_HARMONICS = (1, 2, 4, 8)
ENERGY_POLICY = 'predictive_skill_energy'


class PredictiveEnergyScorer:
    """Strict input contract and reusable five-right-hand-side CPU scorer."""

    def __init__(self, times, features, runs, *, weights=None, phase_offset=None):
        self.t = np.asarray(times, dtype=np.float64)
        if self.t.ndim != 1:
            raise ValueError('预测增益融合的时刻必须为一维数组')
        groups = np.asarray(runs)
        self.w = (np.ones(len(self.t)) if weights is None
                  else np.asarray(weights, dtype=np.float64).copy())
        self.offset = (np.zeros(len(self.t)) if phase_offset is None
                       else np.asarray(phase_offset, dtype=np.float64))
        if (self.t.ndim != 1 or groups.shape != self.t.shape
                or self.w.shape != self.t.shape or self.offset.shape != self.t.shape):
            raise ValueError('预测增益融合的时刻、Run、权重和相位必须一维等长')
        if (not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.w))
                or not np.all(np.isfinite(self.offset)) or np.any(self.w <= 0)):
            raise JointPeriodUnavailable('预测增益融合拒绝缺样、非有限数或非正权重')
        if groups.dtype.kind in 'fc' and not np.all(np.isfinite(groups)):
            raise JointPeriodUnavailable('预测增益融合的 Run 标识必须有限')
        if not len(self.t) or np.ptp(self.t) <= 0:
            raise JointPeriodUnavailable('预测增益融合需要非零观测时间跨度')
        if set(features) != set(ENERGY_CHANNELS):
            raise JointPeriodUnavailable('预测增益融合需要固定的五项完整 CPI 特征')
        columns = [np.asarray(features[name], dtype=np.float64) for name in ENERGY_CHANNELS]
        if any(col.shape != self.t.shape for col in columns):
            raise ValueError('预测增益融合特征必须与时刻一维等长')
        self.y = np.column_stack(columns)
        if not np.all(np.isfinite(self.y)):
            raise JointPeriodUnavailable('预测增益融合拒绝缺样或非有限特征')
        keys = sorted(np.unique(groups).tolist(), key=lambda key: float(np.mean(self.t[groups == key])))
        if len(keys) < 10:
            raise JointPeriodUnavailable('预测增益融合至少需要 10 个有效 Run')
        assignment = {key: i % 5 for i, key in enumerate(keys)}
        labels = np.array([assignment[key] for key in groups])
        for key in keys:
            mask = groups == key
            # Scaling before summation prevents overflow of finite positive weights.
            local = self.w[mask] / np.max(self.w[mask])
            self.w[mask] = local / np.sum(local)
        if np.any(self.w <= 0) or not np.all(np.isfinite(self.w)):
            raise JointPeriodUnavailable('Run 内权重动态范围超出 float64 可表示范围')
        self.root = np.sqrt(self.w)
        self.run_count = len(keys)
        self.folds = [dict(train_runs=[key for key in keys if assignment[key] != k],
                           test_runs=[key for key in keys if assignment[key] == k]) for k in range(5)]
        scale = np.max(np.abs(self.y), axis=0)
        if np.any(scale == 0) or np.any(np.ptp(self.y, axis=0) == 0):
            raise JointPeriodUnavailable('预测增益融合不支持常数或近常数通道')
        self.y = self.y / scale
        self.y -= np.average(self.y, weights=self.w, axis=0)
        variance = np.average(self.y ** 2, weights=self.w, axis=0)
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise JointPeriodUnavailable('预测增益融合不支持数值近常数通道')
        self.y /= np.sqrt(variance)
        # Preserve the existing common selector's pure-trend and affine-copy rules.
        trend = np.column_stack((np.ones(len(self.t)),
                                 (self.t - self.t.min()) / np.ptp(self.t)))
        beta = np.linalg.lstsq(trend*self.root[:, None], self.y*self.root[:, None], rcond=None)[0]
        residual = (self.y-trend@beta)*self.root[:, None]
        if np.any(np.linalg.norm(residual, axis=0) <= 1e-12*np.linalg.norm(self.y*self.root[:, None], axis=0)):
            raise JointPeriodUnavailable('预测增益融合不支持纯线性趋势通道')
        for j in range(5):
            for k in range(j):
                if min(np.max(abs(self.y[:, j]-self.y[:, k])),
                       np.max(abs(self.y[:, j]+self.y[:, k]))) <= 1e-10:
                    raise JointPeriodUnavailable('预测增益融合不支持仿射重复通道')
        self.blocks = []
        self.denominator = np.zeros(5)
        for k in range(5):
            train, test = labels != k, labels == k
            if train.sum() < 17 or len(np.unique(self.t[train])) < 17:
                raise JointPeriodUnavailable('预测增益融合训练折不足以估计 H8')
            center = np.average(self.y[train], weights=self.w[train], axis=0)
            self.denominator += np.sum(self.w[test, None]*(self.y[test]-center)**2, axis=0)
            self.blocks.append((train, test, self.y[train]*self.root[train, None]))
        if np.any(self.denominator <= 0) or not np.all(np.isfinite(self.denominator)):
            raise JointPeriodUnavailable('预测增益融合留出常数基线退化')

    def channel_errors(self, periods, harmonics):
        """Return E/D for a bounded chunk of periods; fail on any singular fold."""
        if (isinstance(harmonics,(bool,np.bool_)) or harmonics not in ENERGY_HARMONICS
                or int(harmonics) != harmonics):
            raise ValueError('预测增益融合的阶数必须属于 1、2、4、8')
        harmonics = int(harmonics)
        periods = np.atleast_1d(np.asarray(periods, dtype=np.float64))
        if periods.ndim != 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
            raise ValueError('候选周期必须为有限正数')
        phase = 2*np.pi*self.t[None, :]/periods[:, None]+self.offset[None, :]
        design = np.stack([np.ones_like(phase)] + [fun(h*phase)
                          for h in range(1, harmonics+1) for fun in (np.cos, np.sin)], axis=2)
        numerator = np.zeros((len(periods), 5))
        for train, test, rhs in self.blocks:
            a = design[:, train]*self.root[None, train, None]
            u, singular, vh = np.linalg.svd(a, full_matrices=False)
            tolerance = max(a.shape[1:])*np.finfo(np.float64).eps*singular[:, 0]
            if np.any(singular[:, -1] <= tolerance) or not np.all(np.isfinite(singular)):
                raise JointPeriodUnavailable('预测增益融合遇到秩亏训练折；未跳过困难候选')
            beta = vh.transpose(0, 2, 1) @ ((u.transpose(0, 2, 1) @ rhs)/singular[:, :, None])
            residual = self.y[None, test, :]-design[:, test]@beta
            numerator += np.sum(self.w[None, test, None]*residual**2, axis=1)
        errors = numerator/self.denominator
        if not np.all(np.isfinite(errors)):
            raise JointPeriodUnavailable('预测增益融合评分非有限')
        return errors

    def evaluate(self, period, harmonics, details=False):
        errors = self.channel_errors([period], harmonics)[0]
        return errors if details else float(np.sum(np.maximum(1-errors, 0)**2))


def select_predictive_energy_period(times, features, runs, *, weights=None,
                                    phase_offset=None, min_period, max_period,
                                    period_grid_size, refine_period=True):
    """Fixed dense8/top3/endpoints search; ``refine_period`` does not disable it.

    The legacy flag only controls individual-feature searches at the caller.
    This strategy always performs exactly 42 golden-section iterations.
    """
    if not (np.isfinite(min_period) and np.isfinite(max_period) and 0 < min_period < max_period):
        raise ValueError('周期范围必须满足 0 < min_period < max_period')
    if (isinstance(period_grid_size, (bool, np.bool_)) or not np.isfinite(period_grid_size)
            or int(period_grid_size) != period_grid_size or period_grid_size < 2):
        raise ValueError('period_grid_size 必须为至少 2 的整数')
    scorer = PredictiveEnergyScorer(times, features, runs, weights=weights, phase_offset=phase_offset)
    grid = np.sort(1/np.linspace(1/max_period, 1/min_period, 8*(int(period_grid_size)-1)+1))
    candidates = []
    for h in ENERGY_HARMONICS:
        scores = np.empty(len(grid))
        for first in range(0, len(grid), 128):
            errors = scorer.channel_errors(grid[first:first+128], h)
            scores[first:first+len(errors)] = np.sum(np.maximum(1-errors, 0)**2, axis=1)
        peaks = signal.find_peaks(scores)[0]
        peaks = sorted(peaks, key=lambda i: (-scores[i], grid[i]))[:3]
        for index in np.unique(np.r_[peaks, 0, len(grid)-1]).astype(int):
            period, score = float(grid[index]), float(scores[index])
            lower, upper = float(grid[max(0, index-1)]), float(grid[min(len(grid)-1, index+1)])
            lo, hi = lower, upper
            ratio = (np.sqrt(5)-1)/2
            c, d = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
            fc, fd = scorer.evaluate(c, h), scorer.evaluate(d, h)
            for _ in range(42):
                if fc > fd:
                    hi, d, fd = d, c, fc
                    c = hi-ratio*(hi-lo)
                    fc = scorer.evaluate(c, h)
                else:
                    lo, c, fc = c, d, fd
                    d = lo+ratio*(hi-lo)
                    fd = scorer.evaluate(d, h)
            refined_period, refined_score = (c, fc) if fc > fd else (d, fd)
            improved = refined_score > score
            if improved:
                period, score = float(refined_period), float(refined_score)
            errors = scorer.evaluate(period, h, details=True)
            skills = 1-errors
            candidates.append(dict(period_s=period, score=score, harmonics=h, h=h,
                grid_index=int(index), grid_period_s=float(grid[index]),
                bracket_period_s=[lower, upper], refinement_iterations=42,
                refinement_status='improved' if improved else 'grid_retained',
                channel_error_ratios=errors.tolist(), channel_skills=skills.tolist(),
                channel_positive_skill_squared=(np.maximum(skills, 0)**2).tolist(),
                source=ENERGY_POLICY))
    candidates.sort(key=lambda row: (-row['score'], row['harmonics'], row['period_s']))
    best = candidates[0]
    if best['score'] <= 0:
        raise JointPeriodUnavailable('预测增益融合全域无正预测增益')
    boundary = bool(best['period_s'] == min_period or best['period_s'] == max_period)
    warnings = ['selected_period_at_search_boundary'] if boundary else []
    return dict(period_s=best['period_s'], score=best['score'],
        support=int(np.count_nonzero(np.asarray(best['channel_skills']) > 0)),
        relative_spread=None, policy=ENERGY_POLICY, status='ambiguous' if boundary else 'candidate',
        message='整 Run 留出正预测增益平方和；分数不是概率或精度保证。', warnings=warnings,
        evidence=[dict(feature=name, period_s=best['period_s'], score=value,
                       role='heldout_predictive_skill', rank=0)
                  for name, value in zip(ENERGY_CHANNELS, best['channel_skills'])],
        candidates=candidates, diagnostics=dict(search_protocol='dense8_top3_endpoints',
            validation_mode='whole_run_cross_validation', fold_count=5, folds=scorer.folds,
            run_count=scorer.run_count, unique_channel_count=5, channel_names=list(ENERGY_CHANNELS),
            harmonics=best['harmonics'], harmonic_orders=list(ENERGY_HARMONICS),
            channel_error_ratios=best['channel_error_ratios'], channel_skills=best['channel_skills'],
            channel_positive_skill_squared=best['channel_positive_skill_squared'],
            score_definition='sum_channels(max(1-heldout_SSE/training_mean_baseline_SSE,0)^2)',
            phase_compensation=phase_offset is not None, grid_size=len(grid),
            grid_source='dense8_full_uniform_frequency_grid', period_range_s=[min_period, max_period],
            refinement_iterations=42, refine_period=True, legacy_refine_period_requested=bool(refine_period),
            numerical_solver='float64_weighted_svd', compute_device='cpu',
            selected_boundary=boundary, independent_probability_evidence=False,
            selection_warning='留出折参与周期与阶数选择；同场景回测不能提供总误差保证。'))


class PredictiveEnergyEnsemble:
    """Shared-design QR evaluator for paired cached-input engineering regression.

    This optional accelerator uses the same normalized data and objective as
    ``PredictiveEnergyScorer``. CUDA is optional; explicit ``device='cpu'`` is
    the fallback. It does not change the ordinary production CPU SVD selector.
    """

    def __init__(self, times, values, runs, *, weights=None, phase_offset=None,
                 device='cpu', batch_size=32):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != 5 or not len(values):
            raise ValueError('批量输入必须为 [realization,CPI,5]')
        self.scorers = [PredictiveEnergyScorer(times,dict(zip(ENERGY_CHANNELS,y.T)),runs,
                        weights=weights,phase_offset=phase_offset) for y in values]
        self.base = self.scorers[0]
        self.y = self.tensor(np.stack([scorer.y for scorer in self.scorers]))
        self.den = self.tensor(np.stack([scorer.denominator for scorer in self.scorers]))
        self.t, self.offset = self.tensor(self.base.t), self.tensor(self.base.offset)
        self.w, self.root = self.tensor(self.base.w), self.tensor(self.base.root)
        self.blocks = [(torch.as_tensor(train,device=self.device),torch.as_tensor(test,device=self.device))
                       for train,test,_ in self.base.blocks]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('batch_size 必须为正整数')

    def tensor(self, value):
        return self.torch.as_tensor(value,dtype=self.torch.float64,device=self.device)

    def _design(self, periods, h):
        torch = self.torch
        phase = 2*np.pi*self.t[None,:]/self.tensor(periods)[:,None]+self.offset[None,:]
        return torch.stack([torch.ones_like(phase)]+[fun(j*phase)
                           for j in range(1,h+1) for fun in (torch.cos,torch.sin)],dim=2)

    def _qr(self, design, train):
        torch = self.torch
        a = design[:,train]*self.root[None,train,None]
        q,r = torch.linalg.qr(a,mode='reduced')
        singular = torch.linalg.svdvals(r)
        tol = max(a.shape[1:])*torch.finfo(a.dtype).eps*singular[:,0]
        if torch.any(singular[:,-1] <= tol) or not torch.isfinite(singular).all():
            raise JointPeriodUnavailable('批量预测增益融合遇到秩亏训练折')
        return q,r

    def grid_errors(self, periods, h):
        """Return all [period,realization,channel] E/D values, bounded memory."""
        torch = self.torch
        if h not in ENERGY_HARMONICS:
            raise ValueError('harmonics 必须为固定阶数之一')
        periods = np.asarray(periods,dtype=np.float64)
        out = np.empty((len(periods),len(self.scorers),5))
        for first in range(0,len(periods),self.batch_size):
            local = periods[first:first+self.batch_size]
            design = self._design(local,h)
            numerator = torch.zeros((len(local),len(self.scorers),5),dtype=torch.float64,device=self.device)
            for train,test in self.blocks:
                q,r = self._qr(design,train)
                rhs = (self.y[:,train,:]*self.root[None,train,None]).permute(1,0,2).reshape(int(train.sum()),-1)
                beta = torch.linalg.solve_triangular(r,q.transpose(1,2)@rhs,upper=True)
                prediction = (design[:,test]@beta).reshape(len(local),int(test.sum()),len(self.scorers),5).permute(0,2,1,3)
                residual = self.y[None,:,test,:]-prediction
                numerator += torch.sum(self.w[None,None,test,None]*residual**2,dim=2)
            errors = numerator/self.den[None]
            if not torch.isfinite(errors).all():
                raise JointPeriodUnavailable('批量预测增益融合评分非有限')
            out[first:first+len(local)] = errors.cpu().numpy()
        return out

    def paired_scores(self, periods, trials, h):
        """Score one candidate per realization index for golden refinement."""
        torch = self.torch
        periods, trials = np.asarray(periods,float),np.asarray(trials,int)
        out = np.empty(len(periods))
        for first in range(0,len(periods),self.batch_size):
            local = periods[first:first+self.batch_size]
            trial = torch.as_tensor(trials[first:first+len(local)],device=self.device)
            design, y = self._design(local,h), self.y[trial]
            numerator = torch.zeros((len(local),5),dtype=torch.float64,device=self.device)
            for train,test in self.blocks:
                q,r = self._qr(design,train)
                rhs = y[:,train,:]*self.root[None,train,None]
                beta = torch.linalg.solve_triangular(r,q.transpose(1,2)@rhs,upper=True)
                residual = y[:,test,:]-design[:,test]@beta
                numerator += torch.sum(self.w[None,test,None]*residual**2,dim=1)
            errors = numerator/self.den[trial]
            out[first:first+len(local)] = torch.sum(torch.clamp(1-errors,min=0)**2,dim=1).cpu().numpy()
        return out

    def search(self, *, min_period, max_period, period_grid_size, progress=None):
        """Return one winner per input under dense8/top3/endpoints/42 protocol."""
        if not (np.isfinite(min_period) and np.isfinite(max_period) and 0 < min_period < max_period):
            raise ValueError('周期范围必须满足 0 < min_period < max_period')
        if isinstance(period_grid_size,bool) or int(period_grid_size) != period_grid_size or period_grid_size < 2:
            raise ValueError('period_grid_size 必须为至少 2 的整数')
        grid = np.sort(1/np.linspace(1/max_period,1/min_period,8*(int(period_grid_size)-1)+1))
        winners = [None]*len(self.scorers)
        for h in ENERGY_HARMONICS:
            errors = self.grid_errors(grid,h)
            scores = np.sum(np.maximum(1-errors,0)**2,axis=2)
            indices, trials = [], []
            for trial in range(len(self.scorers)):
                peaks = signal.find_peaks(scores[:,trial])[0]
                peaks = sorted(peaks,key=lambda i:(-scores[i,trial],grid[i]))[:3]
                local = np.unique(np.r_[peaks,0,len(grid)-1]).astype(int)
                indices.extend(local.tolist()); trials.extend([trial]*len(local))
            indices,trials = np.asarray(indices),np.asarray(trials)
            lo,hi = grid[np.maximum(indices-1,0)].copy(),grid[np.minimum(indices+1,len(grid)-1)].copy()
            ratio = (np.sqrt(5)-1)/2
            c,d = hi-ratio*(hi-lo),lo+ratio*(hi-lo)
            fc,fd = self.paired_scores(c,trials,h),self.paired_scores(d,trials,h)
            for _ in range(42):
                take = fc > fd
                hi[take],lo[~take] = d[take],c[~take]
                d[take],fd[take] = c[take],fc[take]
                c[~take],fc[~take] = d[~take],fd[~take]
                c[take] = hi[take]-ratio*(hi[take]-lo[take])
                d[~take] = lo[~take]+ratio*(hi[~take]-lo[~take])
                fresh = self.paired_scores(np.where(take,c,d),trials,h)
                fc[take],fd[~take] = fresh[take],fresh[~take]
            pp,ss = np.where(fc>fd,c,d),np.maximum(fc,fd)
            old = scores[indices,trials]
            retain = old >= ss
            pp[retain],ss[retain] = grid[indices[retain]],old[retain]
            for trial in range(len(self.scorers)):
                candidates = np.flatnonzero(trials==trial)
                index = min(candidates,key=lambda i:(-ss[i],pp[i]))
                row = dict(period_s=float(pp[index]),score=float(ss[index]),h=h,
                           search_protocol='dense8_top3_endpoints',status='candidate' if ss[index]>0 else 'no_positive_gain')
                if winners[trial] is None or (-row['score'],h,row['period_s']) < (-winners[trial]['score'],winners[trial]['h'],winners[trial]['period_s']):
                    winners[trial] = row
            if progress is not None:
                progress(h,winners)
        return winners
