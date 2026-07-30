# 回波仿真模块

入口：

```python
from asteroid_radar.simulation import simulate

echo = simulate(config)
```

建议阅读顺序：

1. `configs/simulation/cw_ellipsoid.json`
2. `simulation/experiment.py`
3. `simulation/cw.py`
4. `simulation/motion.py`
5. `simulation/geometry.py`

当前模型采用三角面元、余弦幂散射和远场连续波相干叠加。输出只包含回波
及仿真真值，不执行周期估计。
