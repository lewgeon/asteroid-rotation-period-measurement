# Rotation-Period Inversion Module

[中文](README.md)

CPI power, range centroid and 90% energy width all use full correlation support. FFT Doppler moments retain Hann-window effects; window RMS and Nyquist-seam diagnostics do not correct widths or remove channels. Feature-cache v4 recomputes old scalars and can reuse compatible CPI display spectra. Selection reports are saved as `period_selection.json/html`, separately from conditional uncertainty diagnostics.

Chirp now defaults to `period_selection="predictive_joint"`: fit every available channel at the same periods over the full grid and select the median channel skill from five whole-Run folds. Fewer than 10 effective Runs explicitly use shared-fit scoring. Independent channel results remain available; `legacy_consensus` retains candidate voting for comparison. Predictive scores are not probabilities, and correlated channel biases and cross-scene accuracy remain uncalibrated. See the [joint-period contract and 180 paired controls](https://github.com/lewgeon/radar-asteroid-rotation-period-pipeline/blob/chirp-pulse/docs/4.3/vs_4.2/changes/PERIOD_INVERSION_BASELINE.md#predictive-joint-20261016). The 100-Run/500-CPI CLI job returned 5399.833177 s against a 5400 s repetition period (−0.166823 s), with H1, no LOS phase compensation and a 1800–14400 s range; this is one input, not a stable subsecond guarantee.

Chirp inversion supports `compute_device: "cpu"` (default) or `"cuda:0"`. CUDA accelerates matched filtering, Fourier range alignment, slow-time FFT and the period grid, including whole-Run resampling; feature reduction, local refinement and ill-conditioned grid points stay on CPU. Unavailable CUDA explicitly falls back to CPU. Optional `uncertainty_method: "run_bootstrap"` requires at least 10 effective Runs and saves conditional diagnostics in `period_uncertainty.json/html`; `uncertainty_replicates` defaults to 200 and `uncertainty_seed` to 20261015. The default method is `"none"`. Local intervals and empirical bootstrap quantiles are not a calibrated final-period 3-sigma guarantee. See the [GUI manual](../docs/4.3/GUI_USER_MANUAL.md#compute-and-uncertainty).

This module reads `echo.npz` and estimates rotation-period candidates. One-dimensional CW data uses STFT features; two-dimensional chirp data uses matched filtering and sliding range–Doppler CPIs without crossing run or coherence boundaries.

Schema v4 expresses chirp CPI intent only as `cpi_duration_s` and `cpi_hop_duration_s`; count-based configuration is rejected. Motion compensation defaults to `auto` and is derived from echo metadata to prevent double compensation.

Chirp pulse fallback and CPI searches use one shared intercept and retain all original feature times without run aggregation. Pulse features use the full native matched correlation. Run and coherence labels restrict CPI construction, not regression baselines. This assumes consistent scalar calibration across runs; unknown run biases require separate calibration. Search diagnostics and skipped-feature reasons are exported. Uncalibrated significance is reported as `not_evaluated`; the legacy `significant=false` field is retained for compatibility. CW numerics are unchanged.

Run tests from the repository root:

```powershell
conda activate pytorch
python -m pytest -q inversion\tests
```
