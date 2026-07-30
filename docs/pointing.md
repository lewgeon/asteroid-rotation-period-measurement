# 视线与光行时模块

该模块把雷达观测表示为发射、目标反射和接收三个事件。

入口类：

```python
LightTimePointingSolver
```

建议阅读顺序：

1. `configs/pointing/eros.json`
2. `pointing/solver.py`
3. `pointing/ephemeris.py`
4. `pointing/models.py`

`solver.py`只关心“给定时刻返回目标/测站位置”；在线Horizons查询和地面站
状态位于`ephemeris.py`，测试使用线性目标替代在线星历。
