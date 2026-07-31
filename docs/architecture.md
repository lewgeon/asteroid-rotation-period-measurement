# 项目结构

## 文件布局

```text
rotation-period-measurement/
├── models/
│   └── ellipsoid.obj
├── configs/
│   ├── echo.json
│   ├── inversion.json
│   └── pointing.json
├── scripts/
│   ├── make_ellipsoid.py
│   ├── simulate_echo.py
│   ├── estimate_period.py
│   └── solve_pointing.py
├── src/asteroid_radar/
│   ├── ellipsoid.py
│   ├── mesh.py
│   ├── motion.py
│   ├── echo.py
│   ├── dataset.py
│   ├── signal.py
│   ├── inversion.py
│   ├── ephemeris.py
│   └── pointing.py
└── tests/
    ├── test_mesh.py
    ├── test_echo.py
    ├── test_inversion.py
    ├── test_00_pointing.py
    └── test_z_backend.py
```

## 形状与回波的seam

```text
椭球参数 ──> make_ellipsoid.py ──> ellipsoid.obj
真实模型 ──> 模型准备 ───────────> asteroid.obj
                                      │
                                      ▼
                             configs/echo.json
                                      │
                                      ▼
                                  echo.py
```

`echo.py`只读取`model_path`，不知道模型是椭球还是真实小行星。OBJ/PLY是形状
准备与回波仿真之间的seam。

## 仿真与反演的seam

```text
echo.py ──> EchoDataset/echo.npz ──> inversion.py
```

`EchoDataset`定义在`dataset.py`。反演不导入`mesh.py`、`motion.py`或
`echo.py`。

## 视线模块

`ephemeris.py`负责返回目标和测站的太阳系质心状态，`pointing.py`只负责三
事件光行时方程和视线。视线目前独立运行；以后接入回波时，把计算结果写入
`echo.json`中的`tx_los_icrs`和`rx_los_icrs`即可。

## 为什么不拆成多个项目

三个研究任务共享时间、坐标、数据和实验日志，仍属于同一科学链路。当前用
平坦文件保持可见性；只有模块需要独立发布或由不同团队维护时才拆仓库。
