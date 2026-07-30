# 项目结构

## 模块与数据流

```text
src/asteroid_radar/
├── data/         模块间共享的数据
├── pointing/     星历、光行时和视线
├── simulation/   回波前向模型
└── inversion/    自转参数反演
```

依赖方向是单向的：

```text
pointing ──> PointingSolution
                   │
                   ▼
simulation ──> EchoDataset ──> inversion
```

当前仿真可以直接读取固定视线。后续整合真实星历时，只把
`PointingSolution`中的视线转换为仿真配置，不让仿真代码直接调用Horizons。

## data

公共接口是：

```python
EchoDataset
save_echo(path, echo)
load_echo(path)
```

`EchoDataset`包含复回波、时间、有效样本、相干段、平动多普勒和少量实验
元数据。它是仿真与反演之间唯一的seam。

## pointing

公共结果是`LightTimeSolution`。模块内部负责：

- JPL Horizons几何状态；
- 地面站太阳系质心状态；
- 发射、反射、接收时刻；
- ICRS和本地方位俯仰。

核心算法位于`pointing/solver.py`。

## simulation

公共接口只有：

```python
echo = simulate(config)
```

内部文件按公式含义划分：

- `geometry.py`：三角网格；
- `motion.py`：自转与坐标基；
- `cw.py`：连续波面元相干叠加；
- `experiment.py`：从配置构造一次仿真。

该模块不导入`inversion`。

## inversion

公共接口只有：

```python
result = estimate_rotation(echo, config)
```

内部目前分为：

- `time_frequency.py`：平动多普勒补偿、STFT和谱特征；
- `period.py`：Lomb–Scargle及谐波候选；
- `estimate.py`：组织一次反演。

物理模型局部反演以后在本模块内部增加，不改变`EchoDataset`。

## 配置和脚本

配置、脚本和测试都按同样的研究阶段排列：

```text
configs/{pointing,simulation,inversion}/
scripts/{pointing,simulation,inversion}/
tests/{pointing,simulation,inversion,integration}/
```

配置是普通JSON。研究者直接查看配置和算法文件即可，不需要先理解Schema、
依赖注入框架或配置对象。
