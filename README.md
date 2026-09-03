# 小行星雷达自转周期反演模块

chirp 数据采用 `[pulse, fast_time]` 输入时，本模块先沿快时间逐脉冲匹配滤波，
再提取总功率、距离质心和距离展宽。上述特征保留每个脉冲的真实历元，并直接用于
Lomb–Scargle 不均匀采样周期搜索；不同相干采集之间不拼接复相位。CW 一维输入仍沿用
原有 STFT—频谱特征流程。

[English](README_EN.md)

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
│   └── inversion.py    周期估计
└── tests/
    └── test_inversion.py
```

旧版保留文件 `ephemeris.py`、`pointing.py` 和 `ellipsoid.py` 仅用于历史测试和过渡，
新的完整链路请从项目顶层运行 `pipeline.py`。

## 环境

```powershell
conda activate pytorch
```

## 输入

反演入口只读取 `echo.npz`，其主要字段来自 `../echo/` 模块：

```text
elapsed_s
iq
clean_iq
valid
coherence_id
tx_los_icrs
rx_los_icrs
tx_range_m
rx_range_m
scatter_elapsed_s
emit_elapsed_s
metadata_json
```

新版 `echo.npz` 不再包含旧字段 `translation_coefficients_hz`。如果该字段不存在，
反演会直接使用保存的复回波；如果配置或元数据中提供该字段，则仍会执行旧版平动
多普勒补偿以保持兼容。

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
python pipeline.py --config configs\pipeline_example.json
```

顶层流水线会先运行 `observation/solve_observation_info.py`，再运行
`echo/simulate_echo.py`，最后调用本模块的 `scripts/estimate_period.py`。

## 配置

[configs/inversion.json](configs/inversion.json) 控制 STFT 和周期搜索范围：

```json
{
  "stft_window_samples": 4096,
  "stft_overlap_fraction": 0.75,
  "period_min_s": 1800.0,
  "period_max_s": 14400.0,
  "period_grid_size": 20000
}
```

短时冒烟测试应减小 `stft_window_samples` 和周期搜索范围；正式实验应保证观测时长
足以覆盖目标自转周期，并根据预期周期设置搜索区间。
