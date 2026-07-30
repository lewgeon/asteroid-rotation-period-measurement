"""End-to-end continuous-wave experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .acquisition import AcquisitionSchedule
from .config import ExperimentConfig
from .dynamics import SpinState, icrs_unit_vector
from .echo import EchoResult, RadarGeometry, simulate_continuous_wave_echo
from .geometry import SurfaceMesh
from .period import PeriodEstimate, estimate_period_lomb_scargle, harmonic_aliases
from .processing import (
    DynamicSpectrum,
    SpectralFeatures,
    compensate_translation_doppler,
    compute_dynamic_spectrum,
    extract_spectral_features,
)


@dataclass(frozen=True)
class ContinuousWaveExperimentResult:
    mesh: SurfaceMesh
    echo: EchoResult
    compensated_iq: np.ndarray
    dynamic_spectrum: DynamicSpectrum
    features: SpectralFeatures
    period_estimates: Dict[str, PeriodEstimate]


def build_mesh(config: ExperimentConfig) -> SurfaceMesh:
    target = config.section("target")
    mesh = SurfaceMesh.ellipsoid(
        target["semi_axes_m"],
        target["mesh_latitude_segments"],  # 这个是啥呀
        target["mesh_longitude_segments"],
    )
    scattering = target["scattering"]
    return mesh.with_scattering_spot(
        scattering["spot_direction_body"],
        scattering["spot_angular_radius_deg"],
        scattering["spot_strength"],
    )


def build_spin(config: ExperimentConfig) -> SpinState:
    target = config.section("target")
    pole = target["spin_pole"]
    return SpinState(
        period_s=float(target["rotation_period_s"]),
        axis_icrs=icrs_unit_vector(pole["lon_deg"], pole["lat_deg"]),
        initial_phase_rad=np.deg2rad(target["initial_phase_deg"]),
    )


def build_radar_geometry(config: ExperimentConfig) -> RadarGeometry:
    radar = config.section("radar")
    tx = np.asarray(radar["tx_line_of_sight_icrs"], dtype=np.float64)
    rx = np.asarray(radar["rx_line_of_sight_icrs"], dtype=np.float64)
    if radar["geometry"] == "monostatic":
        rx = tx
    return RadarGeometry(tx, rx)


def run_continuous_wave_experiment(
    config: ExperimentConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ContinuousWaveExperimentResult:
    """Run the validated baseline from mesh construction through period candidates."""

    data = config.data
    radar = data["radar"]
    observation = data["observation"]
    processing = data["processing"]
    scattering = data["target"]["scattering"]
    translation_coefficients = data["translation_doppler"]["polynomial_hz"]

    mesh = build_mesh(config)
    spin = build_spin(config)
    geometry = build_radar_geometry(config)
    schedule = AcquisitionSchedule.continuous_wave(
        observation["start_time_utc"],
        observation["duration_s"],
        radar["baseband_sample_rate_hz"],
    )
    rng = np.random.default_rng(data["experiment"]["seed"])
    echo = simulate_continuous_wave_echo(
        mesh=mesh,
        spin=spin,
        geometry=geometry,
        schedule=schedule,
        carrier_frequency_hz=radar["carrier_frequency_hz"],
        sample_rate_hz=radar["baseband_sample_rate_hz"],
        translation_coefficients_hz=translation_coefficients,
        snr_db=data["noise"]["snr_db"],
        rng=rng,
        cosine_power_tx=scattering["cosine_power_tx"],
        cosine_power_rx=scattering["cosine_power_rx"],
        progress_callback=progress_callback,
    )
    compensated = compensate_translation_doppler(
        echo.iq,
        echo.elapsed_s,
        translation_coefficients,
    )
    dynamic_spectrum = compute_dynamic_spectrum(
        compensated,
        echo.sample_rate_hz,
        processing["stft_window_samples"],
        processing["stft_overlap_fraction"],
    )
    features = extract_spectral_features(dynamic_spectrum)

    estimates: Dict[str, PeriodEstimate] = {}
    feature_values = {
        "total_power": features.total_power,
        "rms_bandwidth": features.rms_bandwidth_hz,
        "centroid": features.centroid_hz,
    }
    for name, values in feature_values.items():
        try:
            estimate = estimate_period_lomb_scargle(
                features.times_s,
                values,
                processing["period_min_s"],
                processing["period_max_s"],
                processing["period_grid_size"],
            )
        except ValueError:
            continue
        estimates[name] = PeriodEstimate(
            best_period_s=estimate.best_period_s,
            candidates=harmonic_aliases(
                estimate.candidates,
                processing["period_min_s"],
                processing["period_max_s"],
            ),
            grid_periods_s=estimate.grid_periods_s,
            grid_scores=estimate.grid_scores,
        )

    if not estimates:
        raise RuntimeError("No period feature contained measurable variation")

    return ContinuousWaveExperimentResult(
        mesh=mesh,
        echo=echo,
        compensated_iq=compensated,
        dynamic_spectrum=dynamic_spectrum,
        features=features,
        period_estimates=estimates,
    )

