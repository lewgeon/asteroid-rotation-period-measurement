"""Persist the auditable common-period decision without a confidence claim."""
import html
import json
from pathlib import Path


def write_selection_report(output, report):
    folder = Path(output)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (folder / 'period_selection.json').write_text(text, encoding='utf-8')
    page = ('<!doctype html><meta charset="utf-8"><title>共同周期预测诊断</title>'
            '<style>body{font:16px Microsoft YaHei;max-width:1050px;margin:36px auto;'
            'padding:0 24px;color:#263244;line-height:1.7}pre{white-space:pre-wrap;'
            'background:#f5f7fa;padding:20px}h1{font-size:26px}</style>'
            '<h1>共同周期预测诊断</h1>'
            f'<p>调制周期候选：<strong>{report["period_s"]:.9g} 秒</strong>。'
            f'状态：{html.escape(str(report.get("status", "未评估")))}。</p>'
            '<p>整Run留出时，同一Run的全部CPI一起进入训练或验证；在同一候选周期上拟合各特征，'
            '比较相对常量基线的预测改善。不同特征的改善取中位数，仅相同有效样点上的仿射重复通道去重。'
            '缺样位置不同或仅近似相关的通道仍可能重复影响结果，不能当独立证据。'
            'Run数量不足时报告拟合回退；详细记录说明实际采用的路径。</p>'
            '<p>预测评分用于选模型，不是概率或精度保证。特征共享同一IQ，仍可能共同受系统偏差影响。'
            '此结果是观测调制周期；对称形体的真实自转周期可能是其两倍，不能自动解除这种歧义。'
            '与旧投票和其他特征的差异应一起检查。</p>'
            '<p><a href="period_selection.json">下载完整JSON</a> · '
            '<a href="period_uncertainty.html">查看误差诊断</a></p>'
            '<details open><summary>候选、逐特征预测与选择证据</summary><pre>'
            + html.escape(text) + '</pre></details>')
    (folder / 'period_selection.html').write_text(page, encoding='utf-8')
