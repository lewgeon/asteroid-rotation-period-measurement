"""The public inversion workflow."""

from dataclasses import dataclass

from .period import add_harmonics, lomb_scargle
from .time_frequency import compensate, spectral_features, stft


@dataclass
class InversionResult:
    compensated_iq: object
    dynamic_spectrum: object
    features: object
    periods: dict


def estimate_rotation(echo, config):
    """Estimate coarse rotation-period candidates from one EchoDataset."""

    compensated = compensate(
        echo.iq,
        echo.elapsed_s,
        echo.metadata["translation_coefficients_hz"],
    )
    dynamic = stft(
        compensated,
        echo.metadata["sample_rate_hz"],
        config["stft_window_samples"],
        config["stft_overlap_fraction"],
    )
    features = spectral_features(dynamic)

    periods = {}
    for name, values in {
        "total_power": features.total_power,
        "rms_bandwidth": features.rms_bandwidth_hz,
        "centroid": features.centroid_hz,
    }.items():
        estimate = lomb_scargle(
            features.times_s,
            values,
            config["period_min_s"],
            config["period_max_s"],
            config["period_grid_size"],
        )
        periods[name] = add_harmonics(
            estimate, config["period_min_s"], config["period_max_s"]
        )
    return InversionResult(compensated, dynamic, features, periods)
