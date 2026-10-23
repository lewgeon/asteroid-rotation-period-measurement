"""Optional feature-period geometry correction, independent of shape truth."""
import numpy as np


def effective_look_phase(echo, feature_times, config):
    """Interpolate effective LOS longitude on scatter event epochs.

    This is an approximate azimuth correction, not a full bistatic model.
    The user supplies the pole in the inversion configuration.
    """
    ra, dec = np.deg2rad([config["spin_axis_ra_deg"], config["spin_axis_dec_deg"]])
    axis = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    helper = np.array([1., 0., 0.]) if abs(axis[2]) > .9 else np.array([0., 0., 1.])
    x = np.cross(helper, axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    times = np.asarray(echo.scatter_elapsed_s, dtype=float)
    tx, rx = (np.asarray(v, dtype=float) for v in (echo.tx_los_icrs, echo.rx_los_icrs))
    if times.ndim != 1 or tx.shape != (len(times), 3) or rx.shape != tx.shape:
        raise ValueError("视线相位补偿需要逐脉冲散射时刻和收发视线")
    features = np.asarray(feature_times, dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite([tx, rx])) or not np.all(np.isfinite(features)):
        raise ValueError("视线相位补偿的时刻和视线必须为有限数")
    for v in (tx, rx):
        if np.any(np.linalg.norm(v, axis=1) < 1e-12):
            raise ValueError("视线相位补偿不能使用缺失或零长度视线")
    look = tx/np.linalg.norm(tx, axis=1)[:, None] + rx/np.linalg.norm(rx, axis=1)[:, None]
    px, py = look@x, look@y
    if np.any(np.hypot(px, py) < 1e-8):
        raise ValueError("有效视线接近自转极点或收发方向相反，经度未定义")
    order = np.argsort(times, kind="stable")
    times, angle = times[order], np.arctan2(py[order], px[order])
    if np.any(np.diff(times) == 0):
        # Duplicate physical epochs must represent the same direction.
        repeats = np.diff(times) == 0
        if np.any(np.abs(np.angle(np.exp(1j*np.diff(angle)[repeats]))) > 1e-8):
            raise ValueError("相同散射时刻的视线方向不一致")
        keep = np.r_[True, np.diff(times) != 0]
        times, angle = times[keep], angle[keep]
    angle = np.unwrap(angle)
    if len(times) == 0 or np.any(features < times[0]) or np.any(features > times[-1]):
        raise ValueError("特征时刻超出视线几何范围，禁止外推")
    longitude = np.interp(features, times, angle)
    multiplier = 2 if config.get("phase_period_convention", "half_rotation") == "half_rotation" else 1
    return -multiplier*(longitude-angle[0])


def method_options(config):
    """Return the effective selection, preserving legacy omitted-field behavior."""
    h = config.get("harmonics", 1)
    enabled = config.get("multi_harmonic_enabled", True)
    return {"los_phase_compensation": config.get("los_phase_compensation", False),
            "harmonics": config.get("harmonics", 2 if "multi_harmonic_enabled" in config else 1) if enabled else 1,
            "refine_period": config.get("refine_period", True)}


def comparison_options(config):
    h = config.get("harmonics", 2)
    return [(f"phase{int(p)}_H{n}_refine{int(r)}",
             {"los_phase_compensation": p, "harmonics": n, "refine_period": r})
            for p in (False, True) for n in (1, h) for r in (False, True)]
