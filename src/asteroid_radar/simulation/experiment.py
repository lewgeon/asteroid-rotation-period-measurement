"""Build one CW simulation from a plain research configuration dictionary."""

import numpy as np

from asteroid_radar.data import ObservationSchedule
from .cw import simulate_cw
from .geometry import SurfaceMesh
from .motion import Spin, icrs_direction


def simulate(config, progress=None):
    """Run only the echo simulation stage."""

    target = config["target"]
    compute = config["compute"]
    mesh = SurfaceMesh.ellipsoid(
        target["semi_axes_m"],
        target["mesh_subdivisions"],
        device=compute["device"],
        dtype=compute["dtype"],
    )
    spot = target["scattering_spot"]
    mesh = mesh.add_scattering_spot(
        spot["direction_body"], spot["radius_deg"], spot["strength"]
    )
    pole = target["spin_pole_icrs_deg"]
    spin = Spin(
        target["rotation_period_s"],
        icrs_direction(pole[0], pole[1]),
        np.deg2rad(target["initial_phase_deg"]),
    )
    observation = config["observation"]
    radar = config["radar"]
    schedule = ObservationSchedule.continuous_wave(
        observation["start_utc"],
        observation["duration_s"],
        radar["sample_rate_hz"],
    )
    echo = simulate_cw(
        mesh,
        spin,
        schedule,
        radar,
        config["translation_doppler_hz"],
        config["snr_db"],
        np.random.default_rng(config["seed"]),
        scattering_power=tuple(target["scattering_power"]),
        progress=progress,
    )
    echo.metadata["truth_period_s"] = target["rotation_period_s"]
    echo.metadata["seed"] = config["seed"]
    return echo
