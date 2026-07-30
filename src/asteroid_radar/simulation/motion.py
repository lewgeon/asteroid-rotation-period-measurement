"""Rigid rotation of the asteroid mesh."""

from dataclasses import dataclass

import numpy as np
import torch


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def icrs_direction(ra_deg, dec_deg):
    ra, dec = np.deg2rad([ra_deg, dec_deg])
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def body_basis(axis):
    z = unit(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(z @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(helper, z))
    return np.column_stack([x, np.cross(z, x), z])


@dataclass
class Spin:
    period_s: float
    axis_icrs: np.ndarray
    initial_phase_rad: float = 0.0

    @property
    def rate(self):
        return 2 * np.pi / self.period_s

    @property
    def angular_velocity(self):
        return self.rate * unit(self.axis_icrs)

    def rotate(self, vectors, elapsed_s):
        phase = self.initial_phase_rad + self.rate * elapsed_s
        phase = phase.to(device=vectors.device, dtype=vectors.dtype)
        c, s = torch.cos(phase)[:, None], torch.sin(phase)[:, None]
        x = c * vectors[None, :, 0] - s * vectors[None, :, 1]
        y = s * vectors[None, :, 0] + c * vectors[None, :, 1]
        z = vectors[None, :, 2].expand_as(x)
        basis = torch.tensor(
            body_basis(self.axis_icrs), device=vectors.device, dtype=vectors.dtype
        )
        return torch.stack([x, y, z], dim=-1) @ basis.T
