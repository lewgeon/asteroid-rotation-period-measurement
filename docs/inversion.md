# 自转参数反演模块

入口：

```python
from asteroid_radar.inversion import estimate_rotation

result = estimate_rotation(echo, config)
```

建议阅读顺序：

1. `configs/inversion/lomb_scargle.json`
2. `inversion/estimate.py`
3. `inversion/time_frequency.py`
4. `inversion/period.py`

当前实现属于粗周期基线：平动多普勒补偿、STFT、功率/质心/RMS带宽特征和
Lomb–Scargle候选。物理模型局部反演尚未实现。
