"""Persist the auditable common-period decision without a confidence claim."""
import html
import json
from pathlib import Path


def write_selection_report(output, report):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (folder / 'period_selection.json').write_text(text, encoding='utf-8')
    policy = report.get('policy')
    description = {
        'predictive_skill_energy': '五项完整特征采用整Run五折留出，Run内有效权重归一化；以各通道正预测增益的平方和J同时选择周期和H1/2/4/8。使用8倍均匀频率网格、前三局部峰及端点、42轮周期域黄金分割。J越大越好；H自动选择，独立特征的单值H不控制本模型。',
        'spherical_response': '以反演配置显式提供的极轴为先验，拟合体固观测方向的球面响应；整Run留出损失同时选择周期、L和各通道正则化系数。损失越小越好；仅支持100Run×5CPI和合同内的近单站几何。此模型不输出三维形状。',
    }.get(policy, '整Run留出时，同一Run的全部CPI一起进入训练或验证；在同一候选周期上拟合各特征，比较相对常量基线的预测改善。不同特征的改善取中位数，仅相同有效样点上的仿射重复通道去重。缺样位置不同或仅近似相关的通道仍可能重复影响结果，不能当独立证据。Run数量不足时报告拟合回退；详细记录说明实际采用的路径。')
    page = ('<!doctype html><meta charset="utf-8"><title>共同周期预测诊断</title>'
            '<style>body{font:16px Microsoft YaHei;max-width:1050px;margin:36px auto;'
            'padding:0 24px;color:#263244;line-height:1.7}pre{white-space:pre-wrap;'
            'background:#f5f7fa;padding:20px}h1{font-size:26px}</style>'
            '<h1>共同周期预测诊断</h1>'
            f'<p>调制周期候选：<strong>{report["period_s"]:.9g} 秒</strong>。'
            f'状态：{html.escape(str(report.get("status", "未评估")))}。</p>'
            f'<p>实际策略：{html.escape(str(policy))}。</p><p>{html.escape(description)}</p>'
            '<p>预测评分用于选模型，不是概率或精度保证。特征共享同一IQ，仍可能共同受系统偏差影响。'
            '此结果是观测调制周期；对称形体的真实自转周期可能是其两倍，不能自动解除这种歧义。'
            '与旧投票和其他特征的差异应一起检查。</p>'
            '<p><a href="period_selection.json">下载完整JSON</a> · '
            '<a href="period_uncertainty.html">查看误差诊断</a></p>'
            '<details open><summary>候选、逐特征预测与选择证据</summary><pre>'
            + html.escape(text) + '</pre></details>')
    (folder / 'period_selection.html').write_text(page, encoding='utf-8')
