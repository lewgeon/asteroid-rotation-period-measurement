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
from inversion import estimate_rotation


parser = argparse.ArgumentParser()
parser.add_argument("--echo", default="outputs/echo/echo.npz")
parser.add_argument("--config", default="configs/inversion.json")
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
echo = load_echo(args.echo)
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
output = args.output or config.get("output_path")
config.pop("output_path", None)
if not output:
    raise SystemExit("必须通过 --output 或 config.output_path 指定输出目录")
result = estimate_rotation(echo, config, progress_callback=report_progress)
output = Path(output)
output.mkdir(parents=True, exist_ok=True)


def _save_period_observables(path):
    features = result.features
    feature_names = []
    columns = []
    for name in (
        "total_power", "doppler_centroid_hz", "rms_bandwidth_hz",
        "range_centroid_s", "rms_range_width_s", "centroid_delay_s", "rms_width_s",
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
        scale = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2.0) if len(differences) else 0.0
        uncertainty[:, column] = max(float(scale), np.finfo(float).eps)
    np.savez_compressed(
        path,
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
            "quality_flag_bits": {
                "1": "physical_echo_overlap",
                "2": "saved_window_overlap",
            },
        })),
    )


_save_period_observables(output / "period_observables.npz")

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
        "significance_calibration": estimate.significance_calibration,
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


def _cross_feature_consensus(feature_summary, relative_tolerance=0.05):
    """Select a period supported by distinct observables, not one tallest peak.

    Multi-harmonic fits naturally admit integer-multiple fundamentals.  Clusters
    are therefore ranked first by independent feature support and then by their
    relative spread.  Half/double aliases appended for reporting are excluded
    from the vote so they cannot count twice for the same measured peak.
    """

    entries = []
    for feature, item in feature_summary.items():
        for rank, candidate in enumerate(item["candidates"][:5]):
            if candidate["source"].endswith((":half", ":double")):
                continue
            entries.append(
                {
                    "feature": feature,
                    "period_s": float(candidate["period_s"]),
                    "score": float(candidate["score"]),
                    "rank": rank,
                }
            )
    clusters = []
    for entry in sorted(entries, key=lambda item: item["period_s"]):
        compatible = [
            cluster for cluster in clusters
            if abs(entry["period_s"] / cluster["center_s"] - 1.0) <= relative_tolerance
        ]
        if compatible:
            cluster = min(
                compatible,
                key=lambda item: abs(np.log(entry["period_s"] / item["center_s"])),
            )
            cluster["entries"].append(entry)
            cluster["center_s"] = float(np.median([x["period_s"] for x in cluster["entries"]]))
        else:
            clusters.append({"center_s": entry["period_s"], "entries": [entry]})
    ranked = []
    for cluster in clusters:
        by_feature = {}
        for entry in cluster["entries"]:
            current = by_feature.get(entry["feature"])
            if current is None or entry["rank"] < current["rank"]:
                by_feature[entry["feature"]] = entry
        evidence = list(by_feature.values())
        center = float(np.median([item["period_s"] for item in evidence]))
        spread = float(
            np.median([abs(item["period_s"] / center - 1.0) for item in evidence])
        )
        ranked.append((len(evidence), spread, center, evidence))
    if not ranked:
        raise ValueError("没有可用于跨特征共识的周期候选")
    support, spread, period_s, evidence = sorted(
        ranked, key=lambda item: (-item[0], item[1], item[2])
    )[0]
    return period_s, support, spread, evidence


consensus_period, consensus_support, consensus_spread, consensus_evidence = (
    _cross_feature_consensus(summary)
)
contributing = [summary[item["feature"]] for item in consensus_evidence]
calibrated_faps = [
    item["false_alarm_probability"] for item in contributing
    if item["false_alarm_probability"] is not None
]
consensus_fap = max(calibrated_faps) if calibrated_faps else None
consensus_significant = consensus_support >= 2 and sum(
    bool(item["significant"]) for item in contributing
) >= 2
aliases = _alias_fields(consensus_period)
best_summary = {
    "recommended_feature": "cross_feature_consensus",
    "best_period_s": consensus_period,
    "best_period_h": consensus_period / 3600,
    "relative_error": abs(consensus_period - truth) / truth if truth else None,
    "false_alarm_probability": consensus_fap,
    "significant": consensus_significant,
    "significance_calibration": (
        "conservative_cross_feature"
        if calibrated_faps else "unavailable_multi_harmonic_regression"
    ),
    "time_axis_role": result.time_axis_role,
    "consensus_feature_support": consensus_support,
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

# Sampling-window diagnostic. Peaks here identify aliases caused by campaign
# gaps independently of the measured feature values.
feature_times = np.asarray(result.features.times_s, dtype=float)
window_frequency_hz = np.linspace(1 / period_max, 1 / period_min, config["period_grid_size"])
window_amplitude = np.abs(
    np.exp(-2j * np.pi * feature_times[:, None] * window_frequency_hz[None, :]).sum(axis=0)
) / max(1, len(feature_times))
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
ax.set(xlabel="Candidate period (h)", ylabel="Lomb-Scargle score")
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
    plot.update_layout(
        title="Rotation-period search",
        xaxis_title="Candidate period (s)",
        yaxis_title="Lomb-Scargle score",
        margin={"l": 60, "r": 25, "t": 55, "b": 50},
        height=620,
    )
    plot.write_html(output / "periodogram.html", include_plotlyjs=True)

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
            "相对误差",
            "—" if best_summary["relative_error"] is None else f"{best_summary['relative_error'] * 100:.2f}%",
        ),
        ("显著性", "显著" if best_summary["significant"] else "不显著"),
        ("假警报概率", _fmt(best_summary["false_alarm_probability"], digits=4)),
        ("显著性校准", best_summary["significance_calibration"]),
        ("共识特征支撑", f"{best_summary['consensus_feature_support']} 个"),
        ("共识相对离散度", _fmt(best_summary["consensus_relative_spread"], digits=4)),
        ("周期时间轴", best_summary["time_axis_role"]),
        ("半周期别称", _fmt(best_summary["half_period_s"], "s", digits=8)),
        ("二倍周期别称", _fmt(best_summary["double_period_s"], "s", digits=8)),
    ],
)
