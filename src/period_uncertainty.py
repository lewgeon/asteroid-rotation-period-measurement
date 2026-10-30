"""Conditional uncertainty diagnostics, never a calibrated total-error claim."""
import json
from pathlib import Path
import numpy as np
try:
    from .inversion import multi_harmonic_lomb_scargle,PeriodSearchUnavailable,add_harmonics
    from .period_consensus import _cross_feature_consensus,CONSENSUS_POLICY
    from .period_enhancements import method_options
except ImportError:
    import inversion as core
    if not hasattr(core,'multi_harmonic_lomb_scargle'):
        import importlib.util,sys
        spec=importlib.util.spec_from_file_location('uncertainty_core',Path(__file__).with_name('inversion.py'))
        core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    multi_harmonic_lomb_scargle=core.multi_harmonic_lomb_scargle
    PeriodSearchUnavailable=core.PeriodSearchUnavailable
    add_harmonics=core.add_harmonics
    from period_consensus import _cross_feature_consensus,CONSENSUS_POLICY
    from period_enhancements import method_options


def candidate_summary(estimate):
    return dict(best_period_s=float(estimate.best_period_s),candidates=[dict(period_s=float(c.period_s),score=float(c.score),source=c.source) for c in estimate.candidates])


def local_run_covariance(times,values,runs,period,harmonics=1,weights=None,phase_offset=None):
    try:
        return _local_run_covariance(times,values,runs,period,harmonics,weights,phase_offset)
    except (np.linalg.LinAlgError,FloatingPointError,OverflowError):
        return dict(status='unavailable',reason='局部协方差数值不可识别，不能给出区间')


def _local_run_covariance(times,values,runs,period,harmonics=1,weights=None,phase_offset=None):
    t=np.asarray(times,float);y=np.asarray(values,float);runs=np.asarray(runs)
    w=np.ones(len(t)) if weights is None else np.asarray(weights,float)
    off=np.zeros(len(t)) if phase_offset is None else np.asarray(phase_offset,float)
    finite=np.isfinite(t)&np.isfinite(y)&np.isfinite(w)&(w>0)&np.isfinite(off)
    t,y,runs,w,off=[v[finite] for v in (t,y,runs,w,off)]
    groups=np.unique(runs);p=2*harmonics+2
    if len(groups)<10 or len(t)<=p+2 or np.ptp(t)<=0 or np.ptp(y)<=0:
        return dict(status='unavailable',reason='至少10个Run及足够非恒定观测才能计算局部Run聚类协方差')
    span=np.ptp(t);q=(t-t.min())/span;nu=span/period
    y=y/np.max(np.abs(y));scale=np.std(y)
    if scale==0 or not np.isfinite(scale):
        return dict(status='unavailable',reason='特征的有效变化量无法分辨')
    y=(y-y.mean())/scale;w=w/np.max(w);w=w/w.mean()
    phase=2*np.pi*nu*q+off
    columns=[np.ones(len(t))]
    for h in range(1,harmonics+1):columns.extend((np.cos(h*phase),np.sin(h*phase)))
    x=np.column_stack(columns);coef=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    residual=y-x@coef;derivative=np.zeros(len(t))
    for h in range(1,harmonics+1):
        derivative+=2*np.pi*h*q*(-coef[2*h-1]*np.sin(h*phase)+coef[2*h]*np.cos(h*phase))
    j=np.column_stack((x,derivative));weighted=j*np.sqrt(w[:,None])
    if np.linalg.matrix_rank(weighted)<p or np.linalg.cond(weighted)>1e10:
        return dict(status='unavailable',reason='周期及波形参数局部不可区分')
    bread=np.linalg.inv(weighted.T@weighted)
    scores=np.stack([np.sum(j[runs==g]*(w[runs==g]*residual[runs==g])[:,None],axis=0) for g in groups])
    correction=len(groups)/(len(groups)-1)*(len(t)-1)/(len(t)-p)
    covariance=correction*bread@(scores.T@scores)@bread
    sigma=float(np.sqrt(max(0,covariance[-1,-1]))*span/nu**2)
    if not np.isfinite(sigma):
        return dict(status='unavailable',reason='局部协方差非有限，不能给出区间')
    return dict(status='conditional_unvalidated',period_s=float(period),sigma_s=sigma,
        conditional_3sigma_interval_s=[float(period-3*sigma),float(period+3*sigma)],
        run_count=len(groups),valid_observations=len(t),covariance_method='CR1_run_cluster_sandwich_local_linearization',
        assumptions='正确周期峰和波形模型；Run间独立、足量、局部近似；系统偏差未计入；未验证99.73%覆盖')


def assess_period_uncertainty(times,features,runs,periods,config,weights=None,phase_offset=None,progress=None):
    if config.get('period_selection') in ('predictive_skill_energy', 'spherical_response'):
        return dict(enabled=config.get('uncertainty_method', 'none') != 'none',
            status='unavailable' if config.get('uncertainty_method', 'none') != 'none' else 'not_evaluated',
            selection_policy=config['period_selection'], coverage_calibrated=False,
            run_count=int(len(np.unique(np.asarray(runs)))), replicates=None,
            within_5_percent_of_baseline_fraction=None, bootstrap_failure_fraction=None,
            execution_status='not_implemented',
            final_3sigma_interval_s=None,
            reason='新精度策略尚未接入每次完整重选阶数的重采样；不使用旧投票或固定阶数区间')
    if config.get('uncertainty_method','none')=='none':return dict(enabled=False,status='not_evaluated')
    t=np.asarray(times,float);runs=np.asarray(runs)
    w=np.ones(len(t)) if weights is None else np.asarray(weights,float)
    if t.ndim!=1 or runs.shape!=t.shape or w.shape!=t.shape or not np.all(np.isfinite(t)):
        raise ValueError('误差评估时刻、Run和权重必须是一维等长有限时标')
    observable = np.zeros(len(t), dtype=bool)
    for name in periods:
        values = np.asarray(features[name], dtype=float)
        if values.shape != t.shape:
            raise ValueError('误差评估特征与时刻必须等长')
        observable |= np.isfinite(values)
    active = np.isfinite(w) & (w > 0) & observable
    groups,inverse=np.unique(runs[active],return_inverse=True)
    w = np.where(active, w, 0.)
    if len(groups)<10:return dict(enabled=True,status='unavailable',run_count=len(groups),
        reason='按Run重采样至少需要10个具有有效正权观测的独立Run，不能把零权行或重叠CPI当独立重复')
    options=method_options(config);h=options['harmonics'];offset=phase_offset if options['los_phase_compensation'] else None
    summary={name:candidate_summary(estimate) for name,estimate in periods.items()}
    predictive = config.get('period_selection', 'predictive_joint') == 'predictive_joint'
    if predictive:
        try:from .joint_period import select_predictive_period, JointPeriodUnavailable
        except ImportError:from joint_period import select_predictive_period, JointPeriodUnavailable
        def select(current, selected_weight):
            return select_predictive_period(t, features, runs, current, weights=selected_weight,
                harmonics=h, phase_offset=offset, min_period=config['period_min_s'],
                max_period=config['period_max_s'], refine_period=options['refine_period'])
        selected = select(periods, w)
        center = selected['period_s']
        policy = selected['policy']
    else:
        center=_cross_feature_consensus(summary)[0]
        policy = CONSENSUS_POLICY
    local={name:local_run_covariance(t,features[name],runs,periods[name].best_period_s,h,w,offset) for name in periods}
    n=int(config.get('uncertainty_replicates',200));seed=int(config.get('uncertainty_seed',20261015))
    if n<30 or seed<0:raise ValueError('误差评估至少30次重采样且种子非负')
    rng=np.random.default_rng(seed);samples=[];feature_samples={name:[] for name in periods};failures=0
    mode_counts = {}; mode_switch_failures = 0
    baseline_mode = selected['diagnostics']['validation_mode'] if predictive else 'legacy_consensus'
    selection_errors = (PeriodSearchUnavailable, JointPeriodUnavailable) if predictive else (PeriodSearchUnavailable,)
    for repeat in range(n):
        multiplicity=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
        selected_weight=np.zeros(len(t));selected_weight[active]=w[active]*multiplicity[inverse]
        current={};current_periods={}
        for name in periods:
            try:
                result=multi_harmonic_lomb_scargle(t,features[name],config['period_min_s'],config['period_max_s'],
                    config['period_grid_size'],harmonics=h,weights=selected_weight,phase_offset=offset,refine_period=config.get('refine_period'),
                    compute_device=config.get('compute_device'))
                result=add_harmonics(result,config['period_min_s'],config['period_max_s'])
                current[name]=candidate_summary(result);current_periods[name]=result
                feature_samples[name].append(float(result.best_period_s))
            except PeriodSearchUnavailable:feature_samples[name].append(None)
        if current:
            try:
                if predictive:
                    resampled = select(current_periods, selected_weight)
                    mode = resampled['diagnostics']['validation_mode']
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                    if mode != baseline_mode:
                        # Small bootstrap populations often contain fewer than
                        # ten distinct Runs. Do not mix fitting and CV estimators.
                        mode_switch_failures += 1; failures += 1; samples.append(None)
                    else:
                        samples.append(float(resampled['period_s']))
                else:
                    samples.append(float(_cross_feature_consensus(current)[0]))
            except selection_errors:
                samples.append(None);failures+=1
        else:samples.append(None);failures+=1
        if progress:progress(repeat+1,n)
    finite=np.array([v for v in samples if v is not None])
    fraction=float(np.mean(abs(finite/center-1)<=.05)) if len(finite) else None
    quantiles=np.quantile(finite,[.025,.5,.975]).tolist() if len(finite) else None
    return dict(enabled=True,status='experimental_coverage_unvalidated' if len(finite) else 'unavailable',method='whole_Run_pairs_bootstrap_full_search_and_current_consensus',
        consensus_policy=policy,
        reason=('全部重采样不可用于基线估计规则；有效Run不足导致验证模式切换或周期不可辨识。'
                if not len(finite) else None),
        baseline_validation_mode=baseline_mode, bootstrap_validation_mode_counts=mode_counts,
        bootstrap_mode_switch_failure_count=mode_switch_failures,
        bootstrap_mode_switch_policy='reject_replicates_that_change_validation_mode',
        scope='保存特征条件下的Run采样敏感性；未重算IQ噪声、几何及前端，非最终总误差保证',
        seed=seed,replicates=n,run_count=len(groups),baseline_consensus_period_s=float(center),
        local_features=local,bootstrap_consensus_samples_s=samples,bootstrap_feature_samples_s=feature_samples,
        bootstrap_failure_fraction=failures/n,within_5_percent_of_baseline_fraction=fraction,
        consensus_empirical_quantiles_025_050_975_s=quantiles,
        conditional_same_mode_95_interval_s=(quantiles[::2] if quantiles and fraction>=.95 and failures==0 else None),
        coverage_calibrated=False,final_3sigma_interval_s=None,
        warning='95%经验分位不是已验证覆盖区间；可能跨别名峰。Run非交换、跨Run相关、偏差或模型错误时须另行验证；未输出最终3σ。')


def write_uncertainty_report(output,report):
    import html
    precision = report.get('selection_policy') in ('predictive_skill_energy', 'spherical_response')
    execution_description = ('当前精度策略尚未执行局部协方差或完整重选阶数的Run重采样，不提供相应区间。'
        if precision else '已实现：局部Run聚类协方差、完整Run重采样与受限控制实验。尚未完成：变化几何、未知形状、共享ADC噪声及系统修正的完整总误差校准。')
    resampling_description = ('本次没有执行Run重采样，未使用旧投票或固定阶数模型替代当前选择器。'
        if precision else '重采样在原始周期搜索范围内重新搜索并执行原共识规则；保留同一Run内各CPI的相关关系。分位数反映采样敏感性，不等于经过实测覆盖率验证的置信区间。')
    report={**report,'deterministic_error_bound_s':None,
        'deterministic_error_bound_status':'not_established_unknown_total_model_error_and_global_alias_separation'}
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    (folder/'period_uncertainty.json').write_bytes(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    labels=dict(total_power='总能量',doppler_centroid='多普勒质心',doppler_bandwidth='多普勒RMS带宽',range_centroid='距离质心',range_energy_width='距离90%能量宽度')
    def number(value):return '未计算' if value is None else f'{value:.8g}'
    def percent(key):
        value=report.get(key)
        return '未计算' if value is None else f'{100*value:.8g}%'
    rows=[]
    for name,value in report.get('local_features',{}).items():
        interval=value.get('conditional_3sigma_interval_s')
        rows.append('<tr><td>'+html.escape(labels.get(name,name))+'</td><td>'+number(value.get('period_s'))+'</td><td>'+number(value.get('sigma_s'))+'</td><td>'+(' 至 '.join(number(v) for v in interval) if interval else html.escape(value.get('reason','不可用')))+'</td></tr>')
    quantiles=report.get('consensus_empirical_quantiles_025_050_975_s')
    paragraph=('重采样共识周期的2.5%、50%、97.5%分位数（秒）：'+ '、'.join(number(v) for v in quantiles)) if quantiles else (report.get('reason') or ('误差评估已启用，但没有可用重采样分位数。' if report.get('enabled') else '未启用误差评估'))
    content=('<!doctype html><meta charset="utf-8"><title>周期误差诊断</title><style>body{font:16px Microsoft YaHei;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.7;color:#273244}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}h1{font-size:25px}</style>'
        '<h1>周期误差诊断</h1><p>这些是有前提的误差估计，尚未校准覆盖率，不能作为最终周期的3σ保证。局部区间假设选对周期峰、波形模型正确且不同Run独立；不包含系统偏差。</p>'
        '<h2>严格总误差上界：尚未建立</h2><p>缺少经过验证的几何／波形模型总误差范围，也没有排除全部远处别名峰，不能从拟合残差或局部σ直接生成严格上界。高斯随机噪声本身也没有有限的100%幅度界。3σ在适当正态和校准条件下表示约99.73%覆盖，不是每次一定覆盖真值。</p>'
        '<p>'+execution_description+'</p>'
        '<p>完整Run重采样次数：'+number(report.get('replicates'))+'；Run数量：'+number(report.get('run_count'))+'；基线共识周期：'+number(report.get('baseline_consensus_period_s'))+' s。</p><p>'+html.escape(paragraph)+'</p>'
        '<p>'+resampling_description+'</p>'
        '<p>共识周期落在基线±5%内的比例：'+percent('within_5_percent_of_baseline_fraction')+'；搜索失败比例：'+percent('bootstrap_failure_fraction')+'。'+('未通过单峰条件，不提供单峰条件95%区间。' if report.get('replicates') and report.get('conditional_same_mode_95_interval_s') is None else '')+'</p>'
        '<p>因验证模式切换而拒绝的重采样：'+number(report.get('bootstrap_mode_switch_failure_count'))+'次；不混合整Run留出和共同拟合的周期分布。</p>'
        '<table><tr><th>特征</th><th>周期(s)</th><th>局部σ(s)</th><th>条件性±3σ范围(s)</th></tr>'+''.join(rows)+'</table>'
        '<details><summary>局部σ从哪里来</summary><p>只拟合周期的教学情形中，gᵢ表示周期改变1秒造成的第i项模型变化。若观测误差独立且标准差为σᵧ，最小二乘的一阶传播给出 σ<sub>T</sub>²≈σ<sub>y</sub>²／Σgᵢ²。它由观测误差及模型灵敏度计算，不要求预先知道周期真值。旧策略的局部协方差实现同时拟合幅度、相位和截距，并按完整Run累积残差得分估计参数协方差，因此不能直接套用独立样本的简式；新精度策略本次未计算这项诊断。未知系统偏差不会自动被这个方差公式消除。</p></details>'
        '<details><summary>完整记录与逐次重采样结果</summary><pre>'+html.escape(text)+'</pre></details><p><a href="period_uncertainty.json">下载JSON记录</a></p>')
    (folder/'period_uncertainty.html').write_bytes(content.encode('utf-8'))
