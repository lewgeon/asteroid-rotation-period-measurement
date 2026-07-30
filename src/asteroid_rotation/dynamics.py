"""Spin-state and coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite 3-vector")
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("Vector cannot be zero")
    return value / norm


def icrs_unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Return an ICRS Cartesian unit vector from longitude/latitude."""

    coordinate = SkyCoord(ra=lon_deg * u.deg, dec=lat_deg * u.deg, frame="icrs")
    cartesian = coordinate.cartesian
    return normalize_vector([cartesian.x.value, cartesian.y.value, cartesian.z.value])


def body_to_icrs_basis(spin_axis_icrs: Iterable[float]) -> np.ndarray:
    """Create a right-handed basis whose body z-axis is the ICRS spin axis."""

    z_axis = normalize_vector(spin_axis_icrs)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(z_axis @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    x_axis = normalize_vector(np.cross(helper, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


@dataclass(frozen=True)
class SpinState:
    period_s: float
    axis_icrs: np.ndarray
    initial_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("Spin period must be positive")
        object.__setattr__(self, "axis_icrs", normalize_vector(self.axis_icrs))
        if not np.isfinite(self.initial_phase_rad):
            raise ValueError("Initial phase must be finite")

    @property
    def angular_rate_rad_s(self) -> float:
        return 2.0 * np.pi / self.period_s

    @property
    def angular_velocity_icrs(self) -> np.ndarray:
        return self.angular_rate_rad_s * self.axis_icrs

    def phases(
        self, elapsed_s: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(elapsed_s, torch.Tensor):
            return self.initial_phase_rad + self.angular_rate_rad_s * elapsed_s
        elapsed = np.asarray(elapsed_s, dtype=np.float64)
        return self.initial_phase_rad + self.angular_rate_rad_s * elapsed

    def rotate_body_vectors(
        self,
        vectors: torch.Tensor,
        phases_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate Torch body-frame vectors; output shape is (T, N, 3)."""

        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("vectors must have shape (N, 3)")
        if not vectors.is_floating_point():
            raise ValueError("vectors must use a floating Torch dtype")
        phases = phases_rad.to(device=vectors.device, dtype=vectors.dtype).reshape(-1)
        cosine = torch.cos(phases)[:, None]
        sine = torch.sin(phases)[:, None]
        x = cosine * vectors[None, :, 0] - sine * vectors[None, :, 1]
        y = sine * vectors[None, :, 0] + cosine * vectors[None, :, 1]
        z = vectors[None, :, 2].expand_as(x)
        rotated_body = torch.stack((x, y, z), dim=-1)
        basis = torch.as_tensor(
            body_to_icrs_basis(self.axis_icrs),
            device=vectors.device,
            dtype=vectors.dtype,
        )
        return rotated_body @ basis.T
