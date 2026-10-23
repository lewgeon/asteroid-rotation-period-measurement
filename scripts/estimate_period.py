"""Estimate rotation-period candidates from a saved echo dataset."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import load_echo
from processing_store import attach_store
from inversion import estimate_rotation
from radar_signal import sampling_window_amplitude
from method_report import write_method_comparison
from result_views import file_identity, publish_saved_features, save_processing_view,write_html_atomic
from inspection_store import write_inspections
from compute_backend import resolve_device
from period_uncertainty import write_uncertainty_report


parser = argparse.ArgumentParser()
parser.add_argument("--echo", default="outputs/echo/echo.npz")
parser.add_argument("--config", default="configs/inversion.json")
parser.add_argument('--export-full-cpi-power',action='store_true',
    help='显式保存各CPI双精度科学功率矩阵（默认仅保存完整网格显示产物，磁盘占用更小）')
parser.add_argument(
    "--output",
    default=None,
    help="输出目录。覆盖 config.output_path；二者至少提供一个。",
)
args = parser.parse_args()


def report_progress(stage: str, percent: int, message: str) -> None:
    print(
        "__PROGRESS__ "
        + json.dumps(
            {"stage": stage, "percent": int(percent), "message": message},
            ensure_ascii=True,
        ),
        flush=True,
    )


def _fmt(value, unit: str = "", digits=None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.{digits}g}" if digits is not None else f"{value:g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def report_summary(title: str, items) -> None:
    """紧凑结果摘要；GUI 渲染成可读行，完整内容仍写入 summary.json。"""

    print(
        "__SUMMARY__ "
        + json.dumps(
            {
                "stage": "inversion",
                "title": title,
                "items": [[str(label), str(value)] for label, value in items],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


report_progress("inversion", 0, "读取回波数据")
loaded_echo_identity = file_identity(args.echo)
echo = attach_store(load_echo(args.echo,mapped_directory=Path(args.echo).parent/'processing_cache'/'mapped',load_clean=False),args.echo)
if file_identity(args.echo) != loaded_echo_identity:
    raise ValueError('读取回波期间文件发生变化，请重新运行')
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
if args.export_full_cpi_power:config['export_full_cpi_power']=True
output = args.output or config.get("output_path")
config.pop("output_path", None)
if not output:
    raise SystemExit("必须通过 --output 或 config.output_path 指定输出目录")
output=Path(output);output.mkdir(parents=True,exist_ok=True)
(output/'inversion_in_progress.json').write_text(json.dumps(dict(config=config,echo_identity=loaded_echo_identity)),encoding='utf-8')
execution=resolve_device(config)
report_summary('计算设备',[('请求',execution['requested']),('实际',execution['actual']),('加速范围','Chirp脉压／距离对齐／慢时间FFT／周期网格搜索；局部细化及病态网格CPU')])
if execution['reason']:
    print('__WARNING__ '+json.dumps(dict(stage='inversion',message=execution['reason'],dialog=False)),flush=True)
result = estimate_rotation(echo, config, progress_callback=report_progress)
if file_identity(args.echo)!=loaded_echo_identity:
    raise ValueError('反演计算期间回波改变，禁止将旧内存结果认证为新来源')
for notice in getattr(result, "notices", ()):
    print(
        "__WARNING__ "
        + json.dumps(
            {"stage": "inversion", "message": notice, "dialog": True},
            ensure_ascii=True,
        ),
        flush=True,
    )
output = Path(output)
output.mkdir(parents=True, exist_ok=True)


def _save_period_observables(path):
    features = result.features
    feature_names = []
    columns = []
    for name in (
        "total_power", "doppler_centroid_hz", "centroid_hz", "rms_bandwidth_hz",
        "range_centroid_s", "centroid_delay_s",
        "energy_width_90_s", "energy_range_width_90_s",
    ):
        if hasattr(features, name):
            feature_names.append(name)
            columns.append(np.asarray(getattr(features, name), dtype=float))
    times = np.asarray(features.times_s, dtype=float)
    values = np.column_stack(columns)
    source_times = (
        echo.scatter_elapsed_s
        if result.time_axis_role == "scatter_elapsed_s"
        else echo.elapsed_s
    )
    nearest = np.clip(np.searchsorted(source_times, times), 0, len(source_times) - 1)
    previous = np.maximum(nearest - 1, 0)
    use_previous = np.abs(source_times[previous] - times) < np.abs(source_times[nearest] - times)
    nearest[use_previous] = previous[use_previous]
    run_ids = echo.run_id[nearest] if len(echo.run_id) == len(echo.elapsed_s) else echo.coherence_id[nearest]
    coherence = (
        np.asarray(features.coherence_id, dtype=int)
        if hasattr(features, "coherence_id") else echo.coherence_id[nearest]
    )
    effective = (
        np.asarray(features.effective_sample_count, dtype=float)
        if hasattr(features, "effective_sample_count") else np.ones(len(times), dtype=float)
    )
    quality_flags = np.zeros(len(times), dtype=np.uint32)
    signal_overlap = getattr(echo, "signal_echo_overlap", np.array([]))
    if len(signal_overlap) == len(echo.elapsed_s):
        quality_flags[np.asarray(signal_overlap, dtype=bool)[nearest]] |= np.uint32(1)
    window_overlap = getattr(echo, "window_overlap", np.array([]))
    if len(window_overlap) == len(echo.elapsed_s):
        quality_flags[np.asarray(window_overlap, dtype=bool)[nearest]] |= np.uint32(2)
    uncertainty = np.empty_like(values)
    for column in range(values.shape[1]):
        differences = np.diff(values[:, column])
        differences = differences[np.isfinite(differences)]
        scale = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2.0) if len(differences) else 0.0
        uncertainty[:, column] = max(float(scale), np.finfo(float).eps)
    np.savez_compressed(
        path,
        **{name: np.asarray(getattr(features, name), dtype=float) for name in (
            'doppler_window_rms_hz', 'doppler_nyquist_edge_fraction',
            'doppler_nyquist_edge_half_width_hz') if getattr(features, name, None) is not None},
        reference_time_utc=np.array(echo.metadata.get("reference_time_utc", "")),
        time_elapsed_s=times,
        value=values,
        uncertainty=uncertainty,
        feature_names=np.asarray(feature_names),
        run_id=run_ids,
        coherence_id=coherence,
        effective_sample_count=effective,
        quality_flags=quality_flags,
        metadata_json=np.array(json.dumps({
            "time_role": "feature_center",
            "time_axis_role": result.time_axis_role,
            'inversion_config': config,
            'echo_identity': loaded_echo_identity,
            "search_diagnostics": result.search_diagnostics,
            "feature_algorithm_version": 4,
            "cpi_range_statistics_support": "full_correlation" if hasattr(features, 'energy_range_width_90_s') else None,
            **({
                "energy_width_definition": "central_90_percent_full_correlation_power",
                "energy_width_quantiles": [0.05, 0.95],
                "energy_width_quantile_rule": "first_sample_cumulative_power_at_least_fraction",
                "energy_width_gate": "none",
                "energy_width_zero_power": "NaN",
                "energy_width_processing": ("native_full_correlation" if hasattr(features, "energy_width_90_s")
                                            else "full_correlation_aligned_cpi_windowed_power"),
            } if hasattr(features, "energy_width_90_s") or hasattr(features, "energy_range_width_90_s") else {}),
            "quality_flag_bits": {
                "1": "physical_echo_overlap",
                "2": "saved_window_overlap",
            },
        })),
    )


_save_period_observables(output / "period_observables.npz")
if go is not None:
    publish_saved_features(output / 'period_observables.npz', config, output / 'features.html', args.echo)
    save_processing_view(result, output)

truth = echo.metadata.get("truth_period_s")
period_min = config["period_min_s"]
period_max = config["period_max_s"]


def _alias_fields(best_period_s):
    half = best_period_s / 2.0
    double = best_period_s * 2.0
    return {
        "half_period_s": half if period_min <= half <= period_max else None,
        "double_period_s": double if period_min <= double <= period_max else None,
    }


summary = {
    name: {
        "best_period_s": estimate.best_period_s,
        "relative_error": abs(estimate.best_period_s - truth) / truth if truth else None,
        "false_alarm_probability": (
            float(estimate.false_alarm_probability)
            if np.isfinite(estimate.false_alarm_probability)
            else None
        ),
        "significant": estimate.significant,
        "significance_status": ("not_evaluated" if not np.isfinite(estimate.false_alarm_probability)
                                else "significant" if estimate.significant else "not_significant"),
        "significance_calibration": estimate.significance_calibration,
        "search_diagnostics": estimate.search_diagnostics,
        **_alias_fields(estimate.best_period_s),
        "candidates": [
            {
                "period_s": candidate.period_s,
                "score": candidate.score,
                "source": candidate.source,
            }
            for candidate in estimate.candidates
        ],
    }
    for name, estimate in result.periods.items()
}


from period_consensus import _cross_feature_consensus, CONSENSUS_POLICY


selection_report = result.selection_report
if selection_report:
    consensus_period = selection_report['period_s']
    consensus_support = selection_report['support']
    consensus_spread = selection_report['relative_spread']
    consensus_evidence = selection_report['evidence']
    selection_policy = selection_report['policy']
else:
    consensus_period, consensus_support, consensus_spread, consensus_evidence = _cross_feature_consensus(summary)
    selection_policy = CONSENSUS_POLICY
contributing = [summary[item["feature"]] for item in consensus_evidence]
calibrated_faps = [] if selection_report else [
    item["false_alarm_probability"] for item in contributing
    if item["false_alarm_probability"] is not None
]
consensus_fap = max(calibrated_faps) if calibrated_faps else None
consensus_significant = consensus_support >= 2 and sum(
    bool(item["significant"]) for item in contributing
) >= 2
aliases = _alias_fields(consensus_period)
best_summary = {
    "period_uncertainty_status":result.uncertainty_report.get('status','not_evaluated'),
    "final_3sigma_interval_s":result.uncertainty_report.get('final_3sigma_interval_s'),
    "recommended_feature": "predictive_joint" if selection_report else "cross_feature_consensus",
    "best_period_s": consensus_period,
    "best_period_h": consensus_period / 3600,
    "relative_error": abs(consensus_period - truth) / truth if truth else None,
    "false_alarm_probability": consensus_fap,
    "significant": consensus_significant,
    "significance_status": ("not_evaluated" if not calibrated_faps
                            else "significant" if consensus_significant else "not_significant"),
    "significance_calibration": (
        "conservative_cross_feature"
        if calibrated_faps else "unavailable_multi_harmonic_regression"
    ),
    "time_axis_role": result.time_axis_role,
    "search_diagnostics": result.search_diagnostics,
    "consensus_feature_support": consensus_support,
    "consensus_policy": selection_policy,
    "period_selection_status": selection_report.get('status', 'legacy_consensus'),
    "period_selection": selection_report,
    "period_interpretation": "observed_modulation_period",
    "rotation_period_identifiability": "symmetry_and_harmonic_ambiguity_not_resolved",
    "relative_error_to_half_truth": abs(consensus_period-truth/2)/(truth/2) if truth else None,
    "truth_comparison_scope": "simulation_diagnostic_only_full_and_half_truth_not_used_in_selection",
    "consensus_relative_spread": consensus_spread,
    "consensus_evidence": consensus_evidence,
    "half_period_s": aliases["half_period_s"],
    "half_period_h": aliases["half_period_s"] / 3600 if aliases["half_period_s"] else None,
    "double_period_s": aliases["double_period_s"],
    "double_period_h": aliases["double_period_s"] / 3600 if aliases["double_period_s"] else None,
    "truth_period_s": truth,
}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
(output / "best_summary.json").write_text(
    json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
if selection_report:
    from selection_report import write_selection_report
    write_selection_report(output, selection_report)
write_uncertainty_report(output,result.uncertainty_report or dict(enabled=False,status='not_evaluated'))
write_method_comparison(output, result.method_comparisons)

# Sampling-window diagnostic. Peaks here identify aliases caused by campaign
# gaps independently of the measured feature values.
feature_times = np.asarray(result.features.times_s, dtype=float)
window_frequency_hz = np.linspace(1 / period_max, 1 / period_min, config["period_grid_size"])
window_amplitude = sampling_window_amplitude(feature_times, window_frequency_hz)
np.savez_compressed(
    output / "window_function.npz",
    frequency_hz=window_frequency_hz,
    period_s=1.0 / window_frequency_hz,
    amplitude=window_amplitude,
)

if not best_summary["significant"]:
    if best_summary["false_alarm_probability"] is not None:
        warning_message = (
            f"周期峰值不显著（保守假警报概率 "
            f"{best_summary['false_alarm_probability']:.3f}），该估计不应视为可靠结果。"
        )
    else:
        warning_message = (
            "多谐波回归分数尚无校准假警报概率；结果已按 significant=false 输出，"
            "不能把回归峰值解释为统计显著性。"
        )
    print(
        "__WARNING__ "
        + json.dumps(
            {
                "stage": "inversion",
                "message": warning_message,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

if best_summary["double_period_s"] is not None:
    print(
        "__WARNING__ "
        + json.dumps(
            {
                "stage": "inversion",
                "message": (
                    f"注意半周期混叠：共识周期为 {best_summary['best_period_s']/3600:.3f} h，"
                    f"其二倍周期 {best_summary['double_period_s']/3600:.3f} h 也在搜索范围内。"
                    "对称目标的回波功率包络以 P/2 为周期，真实自转周期可能是 2P；"
                    "请结合微多普勒特征或形状先验判别。"
                ),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

inspection_folder=write_inspections(echo,args.echo,config,report_progress) if go is not None else None
report_progress("inversion", 95, "绘制动态频谱")
power = result.dynamic_spectrum.power
if power.ndim == 3:
    # Range-Doppler-time cube: collapse Doppler only for the static preview.
    power = power.sum(axis=2)
power_db = 10 * np.log10(power / power.max() + np.finfo(float).tiny)
fig, ax = plt.subplots(figsize=(10, 5))
if hasattr(result.dynamic_spectrum, "delays_s"):
    image = ax.pcolormesh(
        result.dynamic_spectrum.times_s / 3600,
        result.dynamic_spectrum.delays_s * 299792458.0,
        power_db.T, shading="auto", cmap="magma", vmin=-45, vmax=0,
    )
    ax.set(xlabel="Elapsed time (h)", ylabel="Relative bistatic path (m)")
    preview_info = result.search_diagnostics.get("dynamic_preview")
    if preview_info and preview_info["sampled"]:
        ax.set_title("Sampled diagnostic range-time image (all pulses used in period search)")
else:
    image = ax.pcolormesh(
        result.dynamic_spectrum.times_s / 3600,
        result.dynamic_spectrum.frequencies_hz,
        power_db, shading="auto", cmap="magma", vmin=-45, vmax=0,
    )
    ax.set(xlabel="Elapsed time (h)", ylabel="Doppler frequency (Hz)")
fig.colorbar(image, ax=ax, label="Relative power (dB)")
fig.tight_layout()
fig.savefig(output / "dynamic_spectrum.png", dpi=240)
plt.close(fig)

report_progress("inversion", 98, "绘制周期图")
fig, ax = plt.subplots(figsize=(9, 5))
for name, estimate in result.periods.items():
    order = np.argsort(estimate.grid_periods_s)
    ax.plot(
        estimate.grid_periods_s[order] / 3600,
        estimate.grid_scores[order],
        label=name,
    )
if truth:
    ax.axvline(truth / 3600, color="black", linestyle="--", label="truth")
ax.set(xlabel="Candidate period (h)", ylabel="Period-model score")
ax.legend()
fig.tight_layout()
fig.savefig(output / "periodogram.png", dpi=240)
plt.close(fig)

if go is not None:
    plot = go.Figure()
    for name, estimate in result.periods.items():
        order = np.argsort(estimate.grid_periods_s)
        plot.add_trace(
            go.Scatter(
                x=estimate.grid_periods_s[order],
                y=estimate.grid_scores[order],
                mode="lines",
                name=name,
            )
        )
    if truth:
        plot.add_vline(x=truth, line_dash="dash", line_color="black", annotation_text="truth")
    for name, estimate in result.periods.items():
        peaks = [item for item in estimate.candidates[:5] if not item.source.endswith((':half', ':double'))]
        plot.add_trace(go.Scatter(x=[item.period_s for item in peaks], y=[item.score for item in peaks],
            mode='markers', name=name + ' 候选峰', marker=dict(size=9, symbol='diamond'),
            hovertemplate='候选周期 %{x:.8g} s<br>分数 %{y:.6g}<extra>%{fullData.name}</extra>'))
    plot.add_vline(x=best_summary['best_period_s'], line_color='red',
                   annotation_text='最终共识选择')
    plot.update_layout(
        title="周期搜索：曲线、候选峰与最终共识选择",
        xaxis_title="Candidate period (s)",
        yaxis_title="周期模型分数",
        margin={"l": 60, "r": 25, "t": 55, "b": 50},
        height=620,
    )
    write_html_atomic(plot,output/'periodogram.html')

completion=dict(observables=file_identity(output/'period_observables.npz'),
    selection_report_identity=file_identity(output/'period_selection.json') if selection_report else None,
    selection_html_identity=file_identity(output/'period_selection.html') if selection_report else None,
    echo_identity=loaded_echo_identity,config=config,
    inspection_folder=str(inspection_folder) if inspection_folder is not None else None)
if file_identity(args.echo)!=loaded_echo_identity:raise ValueError('结果导出期间回波改变，不能认证完成')
pending=output/'inversion_complete.pending.json'
pending.write_text(json.dumps(completion,ensure_ascii=False),encoding='utf-8')
pending.replace(output/'inversion_complete.json')
(output/'inversion_in_progress.json').unlink(missing_ok=True)
report_progress("inversion", 100, "周期反演完成")
report_summary(
    "周期反演结果",
    [
        ("推荐特征", best_summary["recommended_feature"]),
        (
            "推荐周期",
            f"{_fmt(best_summary['best_period_s'], 's', digits=8)}"
            f"（{_fmt(best_summary['best_period_h'], 'h', digits=5)}）",
        ),
        ("真值周期", _fmt(truth, "s")),
        (
            "对完整自转真值的误差",
            "—" if best_summary["relative_error"] is None else f"{best_summary['relative_error'] * 100:.2f}%",
        ),
        ("对半周期真值的误差（仅仿真诊断）", "—" if best_summary['relative_error_to_half_truth'] is None
         else f"{best_summary['relative_error_to_half_truth'] * 100:.5f}%"),
        ("周期含义", "观测调制周期；形体对称及谐波歧义仍需判定"),
        ("周期融合", ("整Run留出预测" if selection_report.get('diagnostics', {}).get('validation_mode') == 'whole_run_cross_validation'
                     else "共同拟合（Run不足，未留出验证）") if selection_report else "旧版候选投票"),
        ("显著性", {"not_evaluated": "未评估", "significant": "显著",
                    "not_significant": "不显著"}[best_summary["significance_status"]]),
        ("假警报概率", _fmt(best_summary["false_alarm_probability"], digits=4)),
        ("显著性校准", best_summary["significance_calibration"]),
        ("正预测增益通道" if selection_report else "共识特征支撑", f"{best_summary['consensus_feature_support']} 个"),
        ("共识相对离散度", _fmt(best_summary["consensus_relative_spread"], digits=4)),
        ("周期时间轴", best_summary["time_axis_role"]),
        ("误差评估",{'not_evaluated':'未启用','unavailable':'观测不足或不可识别','experimental_coverage_unvalidated':'已完成条件性评估，覆盖率未校准'}.get(best_summary['period_uncertainty_status'],'未启用')),
        ("半周期别称", _fmt(best_summary["half_period_s"], "s", digits=8)),
        ("二倍周期别称", _fmt(best_summary["double_period_s"], "s", digits=8)),
    ],
)
