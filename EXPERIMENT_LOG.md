# 实验与修改日志

> 本文件于 2026-07-30 汇总建立。2026-07-29 至 2026-07-30 的早期条目根据对话记录、配置快照、输出文件和文件系统时间重建；当时未保存精确到秒的操作时间，因此只记录可确认的日期或时间区间，不虚构具体时分。后续条目应在修改完成时直接追加。

## 2026-07-29：文献分析与实验方案

### 文献实验分析

- 阅读 Calvés 等（2025）的空间碎片连续波雷达实验。
- 明确论文使用连续波回波及微多普勒动态谱开展目标表征。
- 明确论文展示了由回波变化得到自转周期的结果，但没有公开足以逐行复现的最终周期估计算法细节。
- 因此本项目把“Lomb–Scargle/自相关/谐波一致性粗候选 + 物理前向模型局部反演”定义为项目补充方案，而不声称它是论文原算法。
- 机器学习不作为核心验收算法；以后只有在具备覆盖形状、姿态、噪声和观测几何的代表性训练集，并且能在分布外数据上校准不确定性时，才考虑作为辅助候选生成器或质量控制器。

### 计划 v0.1 → v0.2

- 实验对象由空间碎片改为小行星。
- 实验目标改为：基于雷达复回波精确测量小行星自转周期。
- 将系统拆为小行星回波仿真、回波信号处理、自转参数反演三部分。
- 统一选择三角多面体作为实体表面表示；点云仅作为待重建输入或离散散射中心。
- 确定递进路线：解析散射点/球体 → 椭球网格 → 真实小行星网格。
- 当前采用连续波；数据结构预留不均匀、多时刻、跨越一个以上周期的脉冲序列。
- 所有可调参数迁移到 JSON，并用 JSON Schema 禁止未知字段。
- 采用 Astropy 管理时间、单位、坐标与测站；真实小天体状态计划从 JPL Horizons 获取。
- 检查 `D:\Downloads\NeuS_radar`：借鉴其 Astropy 时间、目标参数和视线组织方式，不复制其神经重建主流程；未发现可直接复用的连续波复回波生成器。
- 形成 [EXPERIMENTS.md](EXPERIMENTS.md) v0.2，规定 E0～E9、验收门槛和复现要求。

## 2026-07-29 17:28–18:19：首版 NumPy 连续波基线

### 新增工程与配置

- 初始化 Python 项目结构、`.gitignore`、Conda 环境文件和中英文 README。
- 新增严格的实验 JSON 配置及 Draft 2020-12 JSON Schema。
- 配置包含目标、雷达、观测、噪声、处理和输出参数；实验参数不再写死在代码中。

### 新增算法模块

- `acquisition.py`：Astropy UTC 时间和显式实际采样时刻；支持未来非均匀脉冲序列。
- `geometry.py`：椭球三角面元与 OBJ 网格入口。
- `dynamics.py`：固定自转轴刚体旋转。
- `echo.py`：连续波复基带面元相干叠加、平动多普勒和噪声。
- `processing.py`：平动多普勒相位补偿、STFT、功率、频谱质心和 RMS 带宽。
- `period.py`：基于实际时间戳的 Lomb–Scargle 周期候选，并显式报告半周期/双周期别名。
- `experiment.py` 与 `scripts/run_baseline.py`：配置驱动的端到端实验及结果输出。

### 验证结果

- 8 项自动测试通过。
- 初始基线：528 个面元，345600 个复采样，真实周期 7200 s，观测 6 h，SNR 10 dB。
- NumPy 运行约 26.44 s。
- RMS 带宽粗估计 7154.92 s，相对误差 0.626%。
- 该结果只属于链路冒烟验证，没有完成正式蒙特卡洛、置信区间或局部物理反演。

## 2026-07-30 11:46–11:53：PyTorch3D 几何与 CUDA 回波迁移

### 修改原因

- 原 `geometry.py` 自行实现了部分网格生成和读取逻辑。
- 用户要求优先使用 PyTorch3D；经审查，不存在必须保留全部自写几何代码的理由。
- 仍保留少量自写面法向和面积计算，因为 PyTorch3D 对应算子会把 `float64` 输入下转为 `float32`，无法满足 CPU 双精度参考路径。

### 修改内容

- `SurfaceMesh` 改为封装 PyTorch3D `Meshes`。
- 椭球由 PyTorch3D `ico_sphere` 生成后按三轴缩放。
- OBJ/PLY 由 PyTorch3D 读取。
- 姿态旋转、可见性、相位和面元相干叠加迁移到 Torch。
- 新增 `compute.backend/device/dtype` 配置。
- 新增 CPU `float64` 参考路径、CUDA `float32` 加速路径和后端一致性测试。
- 测试数从 8 项增加到 11 项。

### 验证结果

- PyTorch3D/CUDA 基线：1280 个面元，345600 个复采样。
- 运行时间 5.416 s；相对原 NumPy 基线约加速 4.88 倍，同时面元数增至 2.42 倍。
- 真实周期 7200 s。
- 回波总功率估计 7100.59 s，误差 1.381%。
- RMS 带宽估计 7148.71 s，误差 0.712%。
- 频谱质心估计 7081.03 s，误差 1.652%。
- 结果保存在 `outputs/baseline_cw_pytorch3d/`，该目录按 `.gitignore` 不纳入版本库。

## 2026-07-30 11:54–12:04：三事件光行时与雷达指向

### 问题定义

远距离雷达不能把同一时刻的测站位置和目标位置直接相减。一次回波对应三个事件：

1. 发射站在 \(t_\mathrm{tx}\) 发射；
2. 小行星在 \(t_\mathrm{b}\) 反射；
3. 接收站在 \(t_\mathrm{rx}\) 接收。

程序求解：

\[
t_\mathrm{b}-t_\mathrm{tx}
=\frac{\|\mathbf r_\mathrm{ast}(t_\mathrm{b})
-\mathbf r_\mathrm{tx}(t_\mathrm{tx})\|}{c},
\]

\[
t_\mathrm{rx}-t_\mathrm{b}
=\frac{\|\mathbf r_\mathrm{rx}(t_\mathrm{rx})
-\mathbf r_\mathrm{ast}(t_\mathrm{b})\|}{c}.
\]

发射和接收天线都指向同一个反射事件，但站点位置和本地时刻不同，因此两条视线分别计算。

### 新增内容

- 新增独立包 `src/asteroid_pointing/`，尚不依赖 `asteroid_rotation` 主链路。
- `ephemeris.py`：
  - JPL Horizons 目标几何状态查询；
  - 查询中心为太阳系质心 `@0`；
  - 历元按 TDB 儒略日提交；
  - 使用地球赤道参考平面与 `geometric` 状态，避免重复应用 Horizons 单程光时修正；
  - Astropy 地球质心状态加地面站 GCRS 位置/速度，形成测站质心状态；
  - 项目内可写缓存目录，避免用户主目录权限问题。
- `solver.py`：
  - 支持输入发射时刻向前求解；
  - 支持输入接收时刻向后求解，默认采用该模式；
  - 支持单基地和收发不同址的双基地配置；
  - 输出 ICRS 单位视线、赤经赤纬和真空方位俯仰。
- 新增独立 JSON 配置、JSON Schema 和命令行脚本。
- `pyproject.toml` 与 `environment.yml` 新增 `astroquery`。
- 新增 5 项测试，总数增加到 16 项。

### 离线测试

- 静止目标：上、下行光时均与 \(R/c\) 解析解一致。
- 一维匀速目标：上行光时与 \(R/(c-v)\) 解析解一致。
- 用前向结果的接收时刻执行反解，恢复原发射时刻，误差小于 \(2\times10^{-7}\) s。
- 严格配置与非法时间角色测试通过。
- 完整测试命令：

```powershell
$env:PYTHONPATH="$PWD\src"
conda run -n pytorch python -m unittest discover -s tests -v
```

- 结果：16 项全部通过。

### JPL Horizons 在线验证

- 验证时间：2026-07-30 12:04（Asia/Shanghai）。
- 目标：433 Eros。
- 输入：2026-07-30T12:00:00 UTC，解释为接收时刻。
- 示例测站：东经 116°、北纬 40°、高 100 m，单基地。
- 解算结果：
  - 发射：2026-07-30T11:35:15.973 UTC；
  - 反射：2026-07-30T11:47:37.918 UTC；
  - 接收：2026-07-30T12:00:00.000 UTC；
  - 上行光时：741.9445567 s；
  - 下行光时：742.0819657 s；
  - 双程光时：1484.0265224 s；
  - 上行距离：222429382.4 km；
  - 下行距离：222470576.6 km；
  - 两条 ICRS 指向相差约 13.69 角秒；
  - 两段固定点迭代均在 3 次内收敛。
- 完整机器可读结果保存在 `outputs/light_time_pointing/solution.json`。

### 已知精度边界

- 当前采用直线传播的几何光时，没有加入太阳引力 Shapiro 时延、行星摄动导致的光路弯曲、对流层、电离层、天线/收发机群时延。
- 示例运行使用 Astropy 随环境附带的 IERS 数据；该表没有覆盖示例日期，程序明确产生降级精度警告。ICRS 几何解仍可用于软件验证，但方位俯仰不能直接用于高精度天线控制。
- 正式观测必须安装覆盖观测日期的 IERS-A/B 数据，并记录数据版本；还应使用测站实测坐标、设备时延和天线指向模型。
- 当前地球质心状态使用 Astropy `builtin` 星历。需要更高精度时，应将配置改为本地 JPL SPK 星历并验证其覆盖期。
- Horizons 查询依赖网络。正式可复现实验应把所用几何状态或 Horizons 原始响应固化为本地数据。

## 2026-07-30 14:50–15:46：按科研流程重构与代码精简

### 重构原因

- 原 `asteroid_rotation` 包把目标几何、回波生成、信号处理和周期估计平铺在
  同一目录，`experiment.py`又把仿真与反演连成一个整体，不利于单独理解和
  验证各研究阶段。
- 两套严格JSON Schema、配置对象和大量输入类型/形状检查遮挡了公式主体，
  不符合当前快速验证科研想法的目标。
- 视线解算虽然已经单独成包，但配置、脚本和测试仍与主项目平铺，项目入口
  不够直观。

### 新结构

统一包名改为`asteroid_radar`，按研究任务划分：

```text
src/asteroid_radar/
├── data/
├── pointing/
├── simulation/
└── inversion/
```

配置、脚本和测试采用相同分组：

```text
configs/{pointing,simulation,inversion}/
scripts/{pointing,simulation,inversion}/
tests/{pointing,simulation,inversion,integration}/
```

新增`docs/architecture.md`以及三个模块的阅读说明。

### 模块接口

- 回波仿真：`simulate(config) -> EchoDataset`
- 自转反演：`estimate_rotation(echo, config) -> InversionResult`
- 仿真与反演之间只通过`EchoDataset`/`echo.npz`连接。
- 光行时模块继续输出`LightTimeSolution`，不强制依赖仿真模块。

### 精简内容

- 删除`jsonschema`依赖以及`schemas/`中的两份Schema。
- 删除`asteroid_rotation.config`和`asteroid_pointing.config`配置加载器。
- 脚本直接使用`json.loads`读取普通配置字典。
- 删除重复的数值类型、数组维数、字段存在性、正数范围和未知字段校验。
- 删除只测试配置非法输入的测试。
- 当前核心源码共950行；显式科研异常只剩3处：
  1. 多普勒超过Nyquist；
  2. 当前照射几何产生零回波；
  3. 光行时迭代不收敛。
- 清理旧包目录及生成的`__pycache__`。

### 保留的科学行为

- PyTorch3D三角网格、OBJ/PLY读取、CPU双精度和CUDA单精度路径；
- 面元相干连续波回波；
- 非均匀时间数据结构；
- 平动多普勒补偿、STFT和谱特征；
- Lomb–Scargle周期候选及半/双周期候选；
- 三事件双程光行时和两条视线。

### 回归测试

- 13项面向科学行为的测试全部通过。
- 测试覆盖仿真输出文件被反演读取，确保两个模块在文件seam处真实连接。
- CUDA/CPU回波相对RMS误差仍小于 \(2\times10^{-4}\)。
- 解析旋转多普勒、非均匀周期和三事件光行时测试保持通过。

### 基线对照

重构后独立运行回波仿真：

```powershell
conda run -n pytorch python scripts\simulation\run_cw.py
```

- 1280个面元；
- 345600个复采样；
- CUDA回波生成3.442 s；
- 重构前为5.416 s。

随后独立运行反演：

```powershell
conda run -n pytorch python scripts\inversion\estimate_period.py
```

周期结果与重构前一致：

| 特征 | 重构前 | 重构后 |
|---|---:|---:|
| 回波总功率 | 7100.5868 s | 7100.5868 s |
| RMS带宽 | 7148.7055 s | 7148.7055 s |
| 频谱质心 | 7081.0327 s | 7081.0327 s |

新结果保存在`outputs/refactor/simulation/`和
`outputs/refactor/inversion/`。

### 光行时回归

新入口：

```powershell
conda run -n pytorch python scripts\pointing\solve.py
```

433 Eros的发射、反射、接收时刻、双程光时、距离和两条视线与重构前完全
一致。结果保存在`outputs/refactor/pointing/eros.json`。IERS覆盖期不足的
方位俯仰警告仍然保留并记录，不做静默隐藏。

## 日志维护规则

以后每次修改至少记录：

1. 日期、时区和可确认的时间；
2. 修改原因与范围；
3. 涉及的配置、代码和数据；
4. 运行命令、环境与随机种子；
5. 测试和实验结果；
6. 已知失败、警告与精度边界；
7. 输出文件位置。
