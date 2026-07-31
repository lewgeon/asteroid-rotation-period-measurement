[English](README_EN.md)

# 小行星雷达自转周期研究

本项目包含三项研究任务：

1. 生成小行星雷达复回波；
2. 从回波估计自转周期；
3. 计算考虑双程光行时的发射与接收视线。

代码采用扁平结构，文件名直接对应算法：

```text
src/asteroid_radar/
├── ellipsoid.py    椭球网格生成
├── mesh.py         OBJ/PLY读取
├── motion.py       刚体自转
├── echo.py         连续波回波
├── dataset.py      回波数据保存与读取
├── signal.py       多普勒补偿和STFT
├── inversion.py    周期估计
├── ephemeris.py    Horizons和地面站状态
└── pointing.py     双程光行时与视线
```

更详细的调用关系见[结构说明](docs/architecture.md)，实验设计见
[EXPERIMENTS.md](EXPERIMENTS.md)，修改记录见
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 环境

```powershell
conda activate pytorch
$env:PYTHONPATH="$PWD\src"
```

## 第一步：准备形状模型

回波仿真器只接受一个已经准备好的OBJ或PLY模型，不包含任何椭球生成逻辑。

生成基线椭球：

```powershell
conda run -n pytorch python scripts\make_ellipsoid.py `
  --axes 70 50 40 `
  --subdivisions 3 `
  --output models\ellipsoid.obj
```

生成器输出的顶点单位为米、模型中心位于原点。当前基线模型见
[models/ellipsoid.obj](models/ellipsoid.obj)。

真实小行星实验只需把模型准备为相同坐标约定，然后修改
[configs/echo.json](configs/echo.json)：

```json
"model_path": "models/real_asteroid.obj"
```

回波代码不需要修改。

## 第二步：生成回波

```powershell
conda run -n pytorch python scripts\simulate_echo.py
```

输出：

```text
outputs/echo/
├── echo.npz
└── summary.json
```

## 第三步：估计周期

```powershell
conda run -n pytorch python scripts\estimate_period.py
```

反演只读取`echo.npz`，不导入网格生成、形状或回波实现。

## 独立计算视线

```powershell
conda run -n pytorch python scripts\solve_pointing.py
```

配置见[configs/pointing.json](configs/pointing.json)。

## 测试

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -v
```

当前12项测试覆盖网格生成和读取、解析旋转多普勒、回波文件接口、周期恢复、
CPU/CUDA一致性以及三事件光行时。

## 当前结果

- 基线模型：1280个三角面元；
- 复采样数：345600；
- CUDA回波生成：约2.13 s；
- 自转周期真值：7200 s；
- RMS带宽估计：7148.71 s，相对误差0.712%；
- OBJ改造前后最佳周期结果一致。

## 网格约定

仿真器假定输入模型已经满足：

- 顶点单位为米；
- 模型位于体固坐标系；
- 旋转中心位于原点；
- 三角面顶点顺序使法向朝外；
- 不包含退化三角形。

这些是模型准备阶段的约定，不在每次仿真时重复检查。
