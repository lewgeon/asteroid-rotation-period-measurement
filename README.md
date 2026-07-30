[English](README_EN.md)

# 小行星雷达回波自转周期测量

本项目通过物理模型生成旋转小行星的连续波雷达复基带回波，经平动多普勒补偿、时频分析和周期特征提取，估计小行星自转周期。

当前版本完成了椭球三角网格基线，目标是先验证完整物理与信号链路，再接入真实小行星三维网格。正式实验设计和验收标准以 [EXPERIMENTS.md](EXPERIMENTS.md) 为准。

## 当前功能

- JSON + JSON Schema 严格实验配置；
- 三轴椭球三角网格和非均匀散射斑块；
- ICRS自转极轴与Astropy高精度时间；
- 单基地/双基地远场连续波复回波；
- 整体平动多普勒多项式与相位补偿；
- STFT微多普勒动态谱；
- 回波功率、频谱质心和RMS带宽特征；
- 基于真实时间戳的Lomb–Scargle周期候选；
- 显式列出半周期和双周期混叠候选；
- 支持任意非均匀观测时刻的采集数据结构。

## 环境

所有Python命令均在Conda的`pytorch`环境中运行：

```powershell
conda activate pytorch
```

当前基线依赖NumPy、SciPy、Astropy、jsonschema、Matplotlib和tqdm，不要求安装trimesh、h5py或pytest。

## 运行测试

```powershell
$env:PYTHONPATH="$PWD\src"
conda run -n pytorch python -m unittest discover -s tests -v
```

当前共有8项测试，覆盖：

- JSON未知字段拒绝；
- 椭球网格面法向；
- 散射斑块；
- 非均匀时间戳；
- 解析旋转多普勒；
- 非均匀采样周期恢复；
- 短周期端到端连续波实验。

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
- 528个三角面元；
- 345,600个复基带采样点。

运行时间约26.4 s。粗周期结果：

| 特征 | 估计周期 | 相对误差 |
|---|---:|---:|
| 回波总功率 | 7104.27 s | 1.330% |
| RMS微多普勒带宽 | 7154.92 s | 0.626% |
| 频谱质心 | 7064.01 s | 1.889% |

这些结果只证明首轮链路可运行，尚未加入物理模型局部精细反演，也不能代替正式蒙特卡洛实验。

## 重要限制

- 当前实体模型仅为解析生成的椭球三角网格；
- 当前仅实现连续波，脉冲回波只完成了非均匀时刻数据结构；
- 当前星历仍使用配置中的视线和多普勒多项式，尚未接入JPL Horizons；
- 当前周期算法是项目补充的可复现基线，不是原论文公开的周期算法；
- 暂未生成置信区间，也未完成SNR和观测时长参数扫描。

