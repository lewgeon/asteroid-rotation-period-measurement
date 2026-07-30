[English](README_EN.md)

# 小行星雷达自转周期研究

本项目研究两个问题：

1. 旋转小行星产生怎样的连续波雷达复回波；
2. 如何从回波中估计小行星自转周期。

代码按研究阶段分为三个独立模块：

```text
pointing  ──> 观测几何
                  ↓
simulation ──> EchoDataset ──> inversion
```

- `pointing`：星历、双程光行时、发射与接收视线；
- `simulation`：目标网格、刚体旋转、散射和连续波回波；
- `inversion`：多普勒补偿、时频特征和周期候选；
- `data`：模块之间传递的 `EchoDataset`。

详细依赖关系见[项目结构说明](docs/architecture.md)。实验方案见
[EXPERIMENTS.md](EXPERIMENTS.md)，历次修改见
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 环境

所有命令在Conda的`pytorch`环境中执行：

```powershell
conda activate pytorch
$env:PYTHONPATH="$PWD\src"
```

## 1. 生成连续波回波

配置：[configs/simulation/cw_ellipsoid.json](configs/simulation/cw_ellipsoid.json)

```powershell
conda run -n pytorch python scripts\simulation\run_cw.py
```

输出：

```text
outputs/simulation/cw_ellipsoid/
├── echo.npz
└── summary.json
```

`echo.npz`是仿真与反演之间唯一的数据接口。反演模块不导入仿真模块的
网格、运动或散射实现。

## 2. 估计自转周期

配置：[configs/inversion/lomb_scargle.json](configs/inversion/lomb_scargle.json)

```powershell
conda run -n pytorch python scripts\inversion\estimate_period.py
```

输出周期候选、动态谱和周期图。目前这里只实现经典粗周期估计，物理模型
局部反演尚未实现。

## 3. 计算光行时与视线

配置：[configs/pointing/eros.json](configs/pointing/eros.json)

```powershell
conda run -n pytorch python scripts\pointing\solve.py
```

该模块独立求解发射、反射、接收三个事件，输出两段光行时以及发射、接收
视线。它目前没有强制接入回波仿真，因此解析实验仍可直接使用固定视线。

## 测试

```powershell
conda run -n pytorch python -m unittest discover -s tests -v
```

当前13项测试只检查关键科学行为：

- 面法向和散射斑块；
- 单散射点解析旋转多普勒；
- CPU/CUDA回波一致性；
- 非均匀观测时刻；
- 仿真文件能被反演模块直接读取；
- Lomb–Scargle周期恢复；
- 三事件光行时解析解与前后向一致性。

## 当前基线

椭球连续波基线包含1280个三角面元和345600个复采样。重构后CUDA回波生成
约3.44 s；RMS带宽周期估计仍为7148.71 s，对7200 s真值的相对误差为
0.712%。重构前后周期结果一致。

## 科研代码原则

- 优先让公式、数据流和实验步骤直接可见；
- 配置使用普通JSON，不使用大型配置框架或严格Schema；
- 不为理论上不会出现的调用方式增加重复类型和形状检查；
- 只保留会影响科研结论的检查，例如多普勒混叠、零回波和光行时不收敛；
- 仿真和反演通过数据文件连接，允许单独替换或验证任一阶段；
- 快速验证不能替代正式蒙特卡洛实验。
