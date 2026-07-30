"""Physics-based continuous-wave echo generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from .acquisition import AcquisitionSchedule
from .dynamics import SpinState, normalize_vector
from .geometry import SurfaceMesh


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class RadarGeometry:
    """Far-field transmitter and receiver lines of sight, station to target."""

    tx_line_of_sight_icrs: np.ndarray
    rx_line_of_sight_icrs: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tx_line_of_sight_icrs",
            normalize_vector(self.tx_line_of_sight_icrs),
        )
        object.__setattr__(
            self,
            "rx_line_of_sight_icrs",
            normalize_vector(self.rx_line_of_sight_icrs),
        )

    @property
    def bistatic_projection_vector(self) -> np.ndarray:
        return self.tx_line_of_sight_icrs + self.rx_line_of_sight_icrs

    def torch_vectors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(
                self.tx_line_of_sight_icrs, device=device, dtype=dtype
            ),
            torch.as_tensor(
                self.rx_line_of_sight_icrs, device=device, dtype=dtype
            ),
        )


@dataclass(frozen=True)
class EchoResult:
    elapsed_s: np.ndarray
    iq: np.ndarray
    clean_iq: np.ndarray
    translation_doppler_hz: np.ndarray
    wavelength_m: float
    sample_rate_hz: float
    snr_db: float | None
    max_rotation_doppler_bound_hz: float


def evaluate_polynomial(coefficients: Iterable[float], elapsed_s: np.ndarray) -> np.ndarray:
    """Evaluate ascending polynomial coefficients c0 + c1*t + ..."""

    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    if coefficients_array.ndim != 1 or len(coefficients_array) == 0:
        raise ValueError("Polynomial coefficients must be a non-empty vector")
    return np.polynomial.polynomial.polyval(elapsed_s, coefficients_array)


def integrated_polynomial_phase_rad(
    coefficients_hz: Iterable[float], elapsed_s: np.ndarray
) -> np.ndarray:
    """Integrate a frequency polynomial and return phase in radians."""

    coefficients = np.asarray(coefficients_hz, dtype=np.float64)
    integral_coefficients = np.zeros(len(coefficients) + 1, dtype=np.float64)
    integral_coefficients[1:] = coefficients / np.arange(1, len(coefficients) + 1)
    cycles = np.polynomial.polynomial.polyval(elapsed_s, integral_coefficients)
    return 2.0 * np.pi * cycles


def rotational_doppler_hz(
    positions_icrs_m: torch.Tensor,
    angular_velocity_icrs_rad_s: torch.Tensor,
    geometry: RadarGeometry,
    wavelength_m: float,
) -> torch.Tensor:
    """Instantaneous rotational Doppler under the positive-phase convention."""

    if positions_icrs_m.ndim != 2 or positions_icrs_m.shape[1] != 3:
        raise ValueError("positions_icrs_m must have shape (N, 3)")
    angular_velocity = angular_velocity_icrs_rad_s.to(
        device=positions_icrs_m.device, dtype=positions_icrs_m.dtype
    )
    velocity = torch.linalg.cross(
        angular_velocity.expand_as(positions_icrs_m),
        positions_icrs_m,
        dim=1,
    )
    projection = torch.as_tensor(
        geometry.bistatic_projection_vector,
        device=positions_icrs_m.device,
        dtype=positions_icrs_m.dtype,
    )
    return velocity @ projection / wavelength_m


def maximum_rotation_doppler_bound_hz(
    mesh: SurfaceMesh,
    spin: SpinState,
    geometry: RadarGeometry,
    wavelength_m: float,
) -> float:
    return float(
        np.linalg.norm(geometry.bistatic_projection_vector)
        * spin.angular_rate_rad_s
        * mesh.characteristic_radius_m
        / wavelength_m
    )


def validate_sampling(
    mesh: SurfaceMesh,
    spin: SpinState,
    geometry: RadarGeometry,
    wavelength_m: float,
    sample_rate_hz: float,
    translation_coefficients_hz: Iterable[float],
    duration_s: float,
) -> float:
    """Reject configurations that can alias the modeled raw baseband echo."""

    rotation_bound = maximum_rotation_doppler_bound_hz(
        mesh, spin, geometry, wavelength_m
    )  # 旋转引起的最大多普勒频移
    probe_times = np.linspace(0.0, duration_s, 2049)
    translation = evaluate_polynomial(translation_coefficients_hz, probe_times)
    modeled_bound = float(np.max(np.abs(translation)) + rotation_bound)
    nyquist = 0.5 * sample_rate_hz
    if modeled_bound >= 0.95 * nyquist:
        raise ValueError(
            "Baseband sample rate is too low: modeled frequency bound "
            f"{modeled_bound:.3f} Hz approaches/exceeds Nyquist {nyquist:.3f} Hz"
        )
    return rotation_bound


def simulate_continuous_wave_echo(
    mesh: SurfaceMesh,
    spin: SpinState,
    geometry: RadarGeometry,
    schedule: AcquisitionSchedule,
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    translation_coefficients_hz: Iterable[float],
    snr_db: float | None,
    rng: np.random.Generator,
    cosine_power_tx: float = 1.0,
    cosine_power_rx: float = 1.0,
    chunk_size: int = 2048,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EchoResult:
    """Simulate a coherent far-field CW echo from a rotating triangle mesh."""

    if carrier_frequency_hz <= 0 or sample_rate_hz <= 0:
        raise ValueError("Carrier frequency and sample rate must be positive")
    if cosine_power_tx < 0 or cosine_power_rx < 0:
        raise ValueError("Cosine scattering powers must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    wavelength_m = SPEED_OF_LIGHT_M_S / carrier_frequency_hz
    duration_s = float(schedule.elapsed_s[-1])  # 实现细节不重要，这些都好改，先看回波是如何生成的
    rotation_bound = validate_sampling(
        mesh,
        spin,
        geometry,
        wavelength_m,
        sample_rate_hz,
        translation_coefficients_hz,
        duration_s,
    )

    elapsed_s = schedule.elapsed_s
    translation_frequency = evaluate_polynomial(
        translation_coefficients_hz, elapsed_s
    )
    translation_phase = integrated_polynomial_phase_rad(
        translation_coefficients_hz, elapsed_s
    )
    device = mesh.device
    dtype = mesh.dtype
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    clean_tensor = torch.zeros(
        len(elapsed_s), device=device, dtype=complex_dtype
    )
    tx_line_of_sight, rx_line_of_sight = geometry.torch_vectors(device, dtype)
    projection = tx_line_of_sight + rx_line_of_sight
    direction_to_tx = -tx_line_of_sight
    direction_to_rx = -rx_line_of_sight
    face_scale = mesh.face_areas * mesh.scattering
    normalization = torch.clamp(
        mesh.face_areas.sum(), min=torch.finfo(dtype).eps
    )
    translation_phase_tensor = torch.as_tensor(
        translation_phase, device=device, dtype=dtype
    )

    total_chunks = (len(elapsed_s) + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, len(elapsed_s), chunk_size)):
            stop = min(start + chunk_size, len(elapsed_s))
            elapsed_chunk = torch.as_tensor(
                elapsed_s[start:stop], device=device, dtype=dtype
            )
            phases = spin.phases(elapsed_chunk)
            positions = spin.rotate_body_vectors(mesh.face_centroids, phases)
            normals = spin.rotate_body_vectors(mesh.face_normals, phases)

            illumination = torch.clamp(normals @ direction_to_tx, min=0.0)
            reception = torch.clamp(normals @ direction_to_rx, min=0.0)
            amplitude = (
                face_scale[None, :]
                * illumination**cosine_power_tx
                * reception**cosine_power_rx
            )
            relative_path_phase = (
                2.0 * np.pi * (positions @ projection) / wavelength_m
            )
            facet_echo = amplitude.to(complex_dtype) * torch.exp(
                1j * relative_path_phase
            )
            clean_tensor[start:stop] = (
                facet_echo.sum(dim=1)
                / normalization
                * torch.exp(1j * translation_phase_tensor[start:stop])
            )
            if progress_callback is not None:
                progress_callback(chunk_index + 1, total_chunks)

    clean = clean_tensor.detach().cpu().numpy()
    clean = np.where(schedule.valid_mask, clean, 0.0)
    if snr_db is None:
        observed = clean.copy()
    else:
        signal_power = float(np.mean(np.abs(clean[schedule.valid_mask]) ** 2))
        if signal_power <= 0:
            raise ValueError("The configured geometry produced zero visible echo power")
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(0.5 * noise_power)
        noise = sigma * (
            rng.standard_normal(len(clean)) + 1j * rng.standard_normal(len(clean))
        )
        observed = clean + noise
        observed = np.where(schedule.valid_mask, observed, 0.0)

    return EchoResult(
        elapsed_s=elapsed_s.copy(),
        iq=observed,
        clean_iq=clean,
        translation_doppler_hz=translation_frequency,
        wavelength_m=wavelength_m,
        sample_rate_hz=sample_rate_hz,
        snr_db=snr_db,
        max_rotation_doppler_bound_hz=rotation_bound,
    )
