# 实验设计与验收

本文件定义当前 schema v4 重构后的最小验收口径。历史实验记录见 `../docs/archive/legacy/EXPERIMENTS.md` 与 `../docs/archive/legacy/EXPERIMENT_LOG.md`。

## 契约实验

1. v3 配置经 `pipeline.migrate_to_v4` 后只保存 v4 字段；重复且冲突的物理量必须硬失败。
2. Chirp 的 ADC 时序、脉宽和路径支撑只由 ObservationPlan 提供给 echo。
3. Chirp CPI 使用秒数配置，并按回波元数据中的 PRF 转为整数脉冲数。
4. 点目标与 mesh 配置互斥裁剪，点目标幅度只使用 `amplitude_scale`。
5. 一维/二维 IQ 与所有行轴字段必须通过严格形状校验。

## 验收命令

```powershell
conda activate pytorch
python -m pytest -q tests
python -m pytest -q inversion\tests
Push-Location observation; python -m pytest -q tests; Pop-Location
Push-Location echo; python -m pytest -q tests; Pop-Location
```

快速 smoke 只能证明链路可运行，不能冒充论文级复现。论文级实验还需要固定数据、参数、随机种子、硬件与误差指标，并记录训练或计算耗时。
