# Rotation-Period Inversion Module

[中文](README.md)

This module reads `echo.npz` and estimates rotation-period candidates. One-dimensional CW data uses STFT features; two-dimensional chirp data uses matched filtering and sliding range–Doppler CPIs without crossing run or coherence boundaries.

Schema v4 expresses chirp CPI intent only as `cpi_duration_s` and `cpi_hop_duration_s`; count-based configuration is rejected. Motion compensation defaults to `auto` and is derived from echo metadata to prevent double compensation.

Run tests from the repository root:

```powershell
conda activate pytorch
python -m pytest -q inversion\tests
```
