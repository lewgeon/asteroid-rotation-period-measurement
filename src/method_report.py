"""Human and machine readable optional-method comparison reports."""
import html
import json
from pathlib import Path


def write_method_comparison(output, methods):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": bool(methods),
               "comparison_scope": "same_features_weights_times_and_search_range",
               "precision_claim": "fit_score_is_not_period_accuracy", "methods": methods}
    (output/"method_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for method in methods.values():
        options = method["options"]
        for name, record in method["features"].items():
            cells = ["开启" if options["los_phase_compensation"] else "关闭",
                     str(options["harmonics"]), "开启" if options["refine_period"] else "关闭",
                     name, f'{record["best_period_s"]:.12g}' if record["status"] == "searched" else "—",
                     f'{record["score"]:.8g}' if record["status"] == "searched" else "—",
                     "已搜索" if record["status"] == "searched" else record.get("message", "跳过")]
            rows.append("<tr>"+"".join("<td>"+html.escape(c)+"</td>" for c in cells)+"</tr>")
    text = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>反演方法比较</title>
<style>body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#25354a}table{border-collapse:collapse}td,th{padding:8px 12px;border:1px solid #c8d3df;text-align:left}th{background:#edf3fa}</style>
<h1>反演方法比较</h1><p>同一份回波提取的特征、时刻、权重和搜索范围。分数描述拟合程度，不代表周期精度；比较不使用形状或初相位真值。</p>"""
    if methods:
        text += "<table><tr><th>视线相位补偿</th><th>谐波数</th><th>局部细化</th><th>特征</th><th>周期（秒）</th><th>拟合分数</th><th>状态</th></tr>"+"".join(rows)+"</table>"
    else:
        text += "<p>本次未开启方法比较。当前报告已更新，不包含上次运行的比较结果。</p>"
    (output/"method_comparison.html").write_text(text+"</html>", encoding="utf-8")
