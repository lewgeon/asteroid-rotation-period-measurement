[English](README_EN.md)

# 小行星雷达回波自转周期测量

本项目通过物理模型生成旋转小行星的连续波雷达复基带回波，经平动多普勒补偿、时频分析和周期特征提取，估计小行星自转周期。

当前版本完成了椭球三角网格基线，目标是先验证完整物理与信号链路，再接入真实小行星三维网格。正式实验设计和验收标准以 [EXPERIMENTS.md](EXPERIMENTS.md) 为准，历次实现与验证记录见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 当前功能

- JSON + JSON Schema 严格实验配置；
- PyTorch3D三角网格、均匀icosphere椭球和非均匀散射斑块；
- OBJ/PLY真实网格读取；
- CPU `float64`参考路径与CUDA `float32`加速路径；
- ICRS自转极轴与Astropy高精度时间；
- 单基地/双基地远场连续波复回波；
- 整体平动多普勒多项式与相位补偿；
- STFT微多普勒动态谱；
- 回波功率、频谱质心和RMS带宽特征；
- 基于真实时间戳的Lomb–Scargle周期候选；
- 显式列出半周期和双周期混叠候选；
- 支持任意非均匀观测时刻的采集数据结构。
- 独立的JPL Horizons三事件光行时与发射/接收指向解算器。

## 环境

所有Python命令均在Conda的`pytorch`环境中运行：

```powershell
conda activate pytorch
```

当前基线依赖PyTorch 2.3.1、PyTorch3D 0.7.9、NumPy、SciPy、Astropy、Astroquery、jsonschema、Matplotlib和tqdm，不要求安装trimesh、h5py或pytest。

## 运行测试

```powershell
$env:PYTHONPATH="$PWD\src"
conda run -n pytorch python -m unittest discover -s tests -v
```

当前共有16项测试，覆盖：

- JSON未知字段拒绝；
- 椭球网格面法向；
- OBJ网格读取；
- CPU `float64`与CUDA `float32`回波一致性；
- 散射斑块；
- 非均匀时间戳；
- 解析旋转多普勒；
- 非均匀采样周期恢复；
- 短周期端到端连续波实验。
- 静止与匀速目标的解析光行时；
- 由接收时刻反解发射时刻；
- 光行时配置的严格校验。

## 光行时修正与雷达指向

对于一次雷达观测，必须区分发射时刻 \(t_\mathrm{tx}\)、目标反射时刻
\(t_\mathrm{b}\) 和接收时刻 \(t_\mathrm{rx}\)。若输入的是发射时刻，程序求解

\[
t_\mathrm{b}-t_\mathrm{tx}
=\frac{\left\|\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{tx}(t_\mathrm{tx})\right\|}{c},
\]

\[
t_\mathrm{rx}-t_\mathrm{b}
=\frac{\left\|\mathbf r_\mathrm{rx}(t_\mathrm{rx})
-\mathbf r_\mathrm{ast}(t_\mathrm{b})\right\|}{c}.
\]

输入接收时刻时则反向求解同一组方程。发射天线指向小行星在反射时刻的位置，接收天线也从接收站在接收时刻的位置指向同一个反射事件：

\[
\widehat{\mathbf u}_\mathrm{tx}
=\frac{\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{tx}(t_\mathrm{tx})}
{\left\|\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{tx}(t_\mathrm{tx})\right\|},
\quad
\widehat{\mathbf u}_\mathrm{rx}
=\frac{\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{rx}(t_\mathrm{rx})}
{\left\|\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{rx}(t_\mathrm{rx})\right\|}.
\]

编辑[光行时JSON配置](configs/light_time_pointing.json)中的目标、观测时间、时间角色和测站经纬高，然后运行：

```powershell
conda run -n pytorch python scripts\solve_light_time_pointing.py `
  --config configs\light_time_pointing.json
```

也可在命令行覆盖目标与时间：

```powershell
conda run -n pytorch python scripts\solve_light_time_pointing.py `
  --config configs\light_time_pointing.json `
  --target 433 `
  --time 2026-07-30T12:00:00 `
  --time-role receive
```

输出包含三个事件的UTC/TDB时刻、两段光行时与距离、ICRS单位视线、赤经赤纬、方位俯仰和迭代残差。当前工具独立于回波仿真主链路；其地球状态默认使用Astropy内置星历，小行星几何状态来自JPL Horizons。用于实际天线控制前，必须加载覆盖观测日期的最新IERS地球定向参数，并进一步纳入设备时延、对流层/电离层、相对论光时和指向模型。

## 运行基线实验

```powershell
conda run -n pytorch python scripts\run_baseline.py `
  --config configs\baseline_cw.json `
  --output outputs\baseline_cw
```

所有参数均从[基线JSON配置](configs/baseline_cw.json)读取，不需要修改代码。配置由[JSON Schema](schemas/experiment.schema.json)校验。

输出包括：

- `config.snapshot.json`：本次运行的配置快照；
- `echo_and_features.npz`：I/Q、补偿后信号、动态谱和特征；
- `summary.json`：真值、候选周期、误差和运行时间；
- `dynamic_spectrum.png`：微多普勒动态谱；
- `period_diagnostics.png`：特征曲线和周期图。

## 首次基线结果

基线条件：

- 自转周期真值：7200 s；
- 观测时长：6 h，即3个周期；
- SNR：10 dB；
- 载频：7.15 GHz；
- PyTorch3D icosphere三级细分，共1280个三角面元；
- 345,600个复基带采样点。

CUDA运行时间约5.42 s。原NumPy基线使用528个面元、运行约26.44 s；新版本在面元数增至2.42倍的同时加速约4.88倍。粗周期结果：

| 特征 | 估计周期 | 相对误差 |
|---|---:|---:|
| 回波总功率 | 7100.59 s | 1.381% |
| RMS微多普勒带宽 | 7148.71 s | 0.712% |
| 频谱质心 | 7081.03 s | 1.652% |

这些结果只证明首轮链路可运行，尚未加入物理模型局部精细反演，也不能代替正式蒙特卡洛实验。

## 重要限制

- 当前基线仍使用解析椭球，虽然已经支持OBJ/PLY网格读取；
- 当前仅实现连续波，脉冲回波只完成了非均匀时刻数据结构；
- 回波仿真主链路仍使用配置中的视线和多普勒多项式；JPL Horizons光行时解算器目前作为独立工具，尚未整合进回波生成器；
- 当前周期算法是项目补充的可复现基线，不是原论文公开的周期算法；
- 暂未生成置信区间，也未完成SNR和观测时长参数扫描。
