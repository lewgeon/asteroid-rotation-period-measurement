# 小行星雷达自转周期反演模块

chirp 数据采用 `[pulse, fast_time]` 输入，沿快时间逐脉冲匹配滤波。CPI 不足时，完整原生相关输出产生总功率、距离质心和中央 90% 能量宽度；CPI 可用时还提取多普勒特征。两条周期搜索路径均使用共同截距，保留原始特征时刻，不按 Run 聚合。Run 与相干组标签限制 CPI 构造，不再成为回归自由截距；不同相干采集不拼接复相位。此模型假定跨 Run 标量参考和标定一致，未知 Run 偏置需独立校准。实际处理路径、有效／排除样本数和特征跳过原因写入搜索诊断。未校准显著性显示“未评估”；CW 一维输入沿用 STFT—频谱特征流程。

[English](README_EN.md)

CPI的总功率、距离质心及90%能量宽度均用完整相关支撑，FFT多普勒谱仍保留Hann窗影响；窗RMS和Nyquist接缝仅作诊断，不自动校正宽度或删通道。标量缓存v4使旧特征重算，兼容的旧CPI显示谱可复用。融合报告输出为`period_selection.json/html`，另保留`period_uncertainty.json/html`条件误差诊断。

Chirp默认采用`period_selection="predictive_joint"`：在完整周期网格的同一候选周期上拟合各可用通道，以整Run五折留出预测改善的通道中位数选择周期；有效Run少于10个时明确退回共同拟合评分。独立单特征结果仍保留，`legacy_consensus`保留旧投票规则供对照。预测分数不是概率，相关通道的共同偏差与跨场景精度仍待校准。算法、去重边界及180组配对控制见[共同周期专题](https://github.com/lewgeon/radar-asteroid-rotation-period-pipeline/blob/chirp-pulse/docs/4.3/vs_4.2/changes/PERIOD_INVERSION_BASELINE.md#predictive-joint-20261016)。100Run／500CPI实际CLI在H1、无视线相位补偿、1800—14400秒搜索下得到5399.833177秒，相对5400秒重复周期偏差−0.166823秒；这只是一份输入的结果，不是稳定亚秒保证。

Chirp反演可显式配置`compute_device: "cpu"`（默认）或`"cuda:0"`，CUDA加速匹配滤波、Fourier距离对齐、慢时间FFT及周期网格搜索（含整Run重采样）；特征归约、局部细化和病态网格仍用CPU，无CUDA时明确记录回退。`uncertainty_method: "run_bootstrap"`可启用至少10个有效Run的整Run重采样诊断，默认200次，可由`uncertainty_replicates`和`uncertainty_seed`控制；默认`"none"`不增加重采样。结果保存为`period_uncertainty.json/html`，局部区间及经验分位尚未校准为最终3σ保证。操作见[GUI手册](../docs/4.3/GUI_USER_MANUAL.md#compute-and-uncertainty)。

本目录现在是总项目中的第三个子模块，只负责从上游 `echo/` 模块保存的
`echo.npz` 估计自转周期。视线向量解算已迁移到 `../observation/`，回波仿真已
迁移到 `../echo/`；本模块不再导入形状、回波或观测解算代码。

当前核心文件：

```text
inversion/
├── configs/
│   └── inversion.json
├── scripts/
│   └── estimate_period.py
├── src/
│   ├── dataset.py      读取 echo.npz 数据交换格式
│   ├── signal.py       多普勒补偿、STFT和谱特征
│   ├── radar_signal.py chirp 匹配滤波与距离—多普勒特征
│   └── inversion.py    周期估计
└── tests/
    ├── test_inversion.py
    ├── test_chirp_contracts.py
    └── test_00_pointing.py
```

`src/pointing.py` 仍由 `tests/test_00_pointing.py` 覆盖。`src/ephemeris.py` 与
`src/ellipsoid.py` 不再参与现行测试或流水线；`tests/test_mesh.py`、`tests/test_echo.py`
已跳过，相关覆盖在 `echo/tests`。新的完整链路请从项目顶层运行 `pipeline.py`。

## 环境

```powershell
conda activate pytorch
```

## 输入

反演入口只读取上游 `echo.npz`（不再导入观测/回波源码）。主要数组：

<table>
<thead>
<tr><th colspan="2">字段</th><th>物理意义</th><th>说明</th></tr>
</thead>
<tbody>
<tr>
  <td><code>elapsed_s</code></td>
  <td>接收时间轴</td>
  <td>单位 s</td>
</tr>
<tr>
  <td><code>iq</code></td>
  <td>含噪复基带回波</td>
  <td>CW 一维或 Chirp 二维</td>
</tr>
<tr>
  <td><code>clean_iq</code></td>
  <td>无噪声复基带回波</td>
  <td>形状同 <code>iq</code></td>
</tr>
<tr>
  <td><code>valid</code></td>
  <td>有效掩码</td>
  <td>bool</td>
</tr>
<tr>
  <td><code>coherence_id</code></td>
  <td>相干分组</td>
  <td>整数，对齐慢时间</td>
</tr>
<tr>
  <td><code>tx_los_icrs</code> 等几何列</td>
  <td>透传观测几何</td>
  <td>供补偿与特征用</td>
</tr>
<tr>
  <td><code>metadata_json</code></td>
  <td>布局、PRF、是否已质心补偿等</td>
  <td>JSON 文本</td>
</tr>
</tbody>
</table>

新版不再包含 `translation_coefficients_hz`；若元数据仍提供该字段，可走旧版平动多普勒补偿兼容路径。

## 单独运行

```powershell
conda activate pytorch
python scripts\estimate_period.py `
  --echo ..\outputs\pipeline\echo\echo.npz `
  --config configs\inversion.json `
  --output ..\outputs\pipeline\inversion
```

## 在完整链路中运行

推荐从项目顶层运行：

```powershell
conda activate pytorch
python pipeline.py --config configs\chirp_point_target_test.json
```

顶层流水线会先运行 `observation/solve_observation_info.py`，再运行
`echo/simulate_echo.py`，最后调用本模块的 `scripts/estimate_period.py`。

## 配置

[configs/inversion.json](configs/inversion.json) 与流水线 `inversion` 段控制特征提取和周期搜索。  
**字段**表头跨两格（与 observation / echo 一致）：无子键时合并；有子键时左父右子并与说明同行对齐。CW / Chirp 专有键拆表，主表用占位标明。

### 主表（共用）

<table>
<thead>
<tr><th colspan="2">字段</th><th>物理意义</th><th>说明</th></tr>
</thead>
<tbody>
<tr>
  <td colspan="2"><code>output_path</code></td>
  <td>反演输出目录</td>
  <td>与 observation / echo 一致：JSON 可写；命令行 <code>--output</code> 优先覆盖。二者至少提供一个。流水线 / GUI 会注入该键</td>
</tr>
<tr>
  <td colspan="2"><code>period_min_s</code></td>
  <td>周期搜索下限</td>
  <td>正有限数，s；应低于预期真值</td>
</tr>
<tr>
  <td colspan="2"><code>period_max_s</code></td>
  <td>周期搜索上限</td>
  <td>正有限数，s；应高于预期真值</td>
</tr>
<tr>
  <td colspan="2"><code>period_grid_size</code></td>
  <td>搜索网格点数</td>
  <td>正整数；越大越细、越慢</td>
</tr>
<tr>
  <td colspan="4" style="background:#f6f8fa"><em>占位：</em>时频 / CPI 专有参数由回波布局决定 → CW 见 <a href="#inv-cw">CW 专有</a>；Chirp 见 <a href="#inv-chirp">Chirp 专有</a>。不要混写另一布局的键。</td>
</tr>
</tbody>
</table>

### <span id="inv-cw">CW 专有</span>（一维 `iq`）

<table>
<thead>
<tr><th colspan="2">字段</th><th>物理意义</th><th>说明</th></tr>
</thead>
<tbody>
<tr>
  <td><code>stft_window_samples</code></td>
  <td>STFT 窗长</td>
  <td>正整数（样本）；短冒烟可减小</td>
</tr>
<tr>
  <td><code>stft_overlap_fraction</code></td>
  <td>STFT 重叠比例</td>
  <td>0–1，如 <code>0.75</code></td>
</tr>
</tbody>
</table>

### <span id="inv-chirp">Chirp 专有</span>（二维 `iq`）

<table>
<thead>
<tr><th colspan="2">字段</th><th>物理意义</th><th>说明</th></tr>
</thead>
<tbody>
<tr>
  <td><code>cpi_duration_s</code></td>
  <td>相干处理间隔（物理时长）</td>
  <td>正有限数，s；入口按 PRF 换成脉冲数；必填</td>
</tr>
<tr>
  <td><code>cpi_hop_duration_s</code></td>
  <td>CPI 滑动步长（物理时长）</td>
  <td>正有限数，s；可省略（默认约 CPI/4）</td>
</tr>
<tr>
  <td rowspan="2"><code>period_time_role</code></td>
  <td rowspan="2">周期搜索使用的时间角色</td>
  <td><code>scatter_centroid</code> — 用散射质心时刻</td>
</tr>
<tr>
  <td><code>receive_centroid</code> — 用接收质心时刻</td>
</tr>
<tr>
  <td><code>motion_compensation</code></td>
  <td>是否做质心运动补偿</td>
  <td>默认 <code>auto</code>（由回波是否已补偿派生）</td>
</tr>
<tr>
  <td><code>harmonics</code></td>
  <td>多谐波 Lomb–Scargle 阶数</td>
  <td>正整数；默认 1</td>
</tr>
</tbody>
</table>

已废弃并会报错：`cpi_pulses`、`cpi_hop_pulses`、`cross_run_phase_coherent`。  
短时冒烟应缩小搜索区间与窗长；正式实验应保证观测时长覆盖自转周期。
