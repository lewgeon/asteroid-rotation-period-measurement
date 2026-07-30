"""Triangle-mesh geometry used by the physical echo model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


def _as_float64_array(value: Iterable[float], shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"Expected trailing shape {shape_tail}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Geometry contains non-finite values")
    return array


@dataclass(frozen=True)
class SurfaceMesh:
    """A centered triangle surface with per-face scattering coefficients."""

    vertices: np.ndarray
    faces: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    scattering: np.ndarray

    @classmethod
    def from_vertices_faces(
        cls,
        vertices: Iterable[Iterable[float]],
        faces: Iterable[Iterable[int]],
        scattering: Iterable[float] | None = None,
        *,
        center: bool = True,
    ) -> "SurfaceMesh":
        vertex_array = _as_float64_array(vertices, (3,))
        face_array = np.asarray(faces, dtype=np.int64)
        if face_array.ndim != 2 or face_array.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {face_array.shape}")
        if face_array.size == 0:
            raise ValueError("Mesh must contain at least one face")
        if face_array.min() < 0 or face_array.max() >= len(vertex_array):
            raise ValueError("Face index is outside the vertex array")

        if center:
            vertex_array = vertex_array - vertex_array.mean(axis=0, keepdims=True)

        face_vertices = vertex_array[face_array]
        edge_a = face_vertices[:, 1] - face_vertices[:, 0]
        edge_b = face_vertices[:, 2] - face_vertices[:, 0]
        cross = np.cross(edge_a, edge_b)
        double_area = np.linalg.norm(cross, axis=1)
        if np.any(double_area <= np.finfo(np.float64).eps):
            raise ValueError("Mesh contains degenerate triangular faces")

        centroids = face_vertices.mean(axis=1)
        normals = cross / double_area[:, None]

        inward = np.einsum("ij,ij->i", normals, centroids) < 0
        if np.any(inward):
            face_array = face_array.copy()
            face_array[inward] = face_array[inward][:, [0, 2, 1]]
            normals = normals.copy()
            normals[inward] *= -1.0

        areas = 0.5 * double_area
        if scattering is None:
            scattering_array = np.ones(len(face_array), dtype=np.float64)
        else:
            scattering_array = np.asarray(scattering, dtype=np.float64)
            if scattering_array.shape != (len(face_array),):
                raise ValueError("Scattering must contain one value per face")
            if np.any(scattering_array < 0) or not np.all(np.isfinite(scattering_array)):
                raise ValueError("Scattering coefficients must be finite and non-negative")

        return cls(
            vertices=np.ascontiguousarray(vertex_array),
            faces=np.ascontiguousarray(face_array),
            face_centroids=np.ascontiguousarray(centroids),
            face_normals=np.ascontiguousarray(normals),
            face_areas=np.ascontiguousarray(areas),
            scattering=np.ascontiguousarray(scattering_array),
        )

    @classmethod
    def ellipsoid(
        cls,
        semi_axes_m: Iterable[float],
        latitude_segments: int,
        longitude_segments: int,
    ) -> "SurfaceMesh":
        """Generate an outward-oriented UV triangle mesh of an ellipsoid."""

        axes = np.asarray(semi_axes_m, dtype=np.float64)
        if axes.shape != (3,) or np.any(axes <= 0):
            raise ValueError("semi_axes_m must contain three positive values")
        if latitude_segments < 4 or longitude_segments < 8:
            raise ValueError("Ellipsoid mesh resolution is too low")

        vertices = [[0.0, 0.0, axes[2]]]
        for latitude_index in range(1, latitude_segments):
            theta = np.pi * latitude_index / latitude_segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for longitude_index in range(longitude_segments):
                phi = 2.0 * np.pi * longitude_index / longitude_segments
                vertices.append(
                    [
                        axes[0] * sin_theta * np.cos(phi),
                        axes[1] * sin_theta * np.sin(phi),
                        axes[2] * cos_theta,
                    ]
                )
        south_index = len(vertices)
        vertices.append([0.0, 0.0, -axes[2]])

        faces = []
        first_ring = 1
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            faces.append([0, first_ring + longitude_index, first_ring + next_longitude])

        for latitude_index in range(latitude_segments - 2):
            ring_a = 1 + latitude_index * longitude_segments
            ring_b = ring_a + longitude_segments
            for longitude_index in range(longitude_segments):
                next_longitude = (longitude_index + 1) % longitude_segments
                faces.append(
                    [
                        ring_a + longitude_index,
                        ring_b + longitude_index,
                        ring_b + next_longitude,
                    ]
                )
                faces.append(
                    [
                        ring_a + longitude_index,
                        ring_b + next_longitude,
                        ring_a + next_longitude,
                    ]
                )

        last_ring = 1 + (latitude_segments - 2) * longitude_segments
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            faces.append(
                [south_index, last_ring + next_longitude, last_ring + longitude_index]
            )

        return cls.from_vertices_faces(vertices, faces, center=False)

    def with_scattering_spot(
        self,
        direction_body: Iterable[float],
        angular_radius_deg: float,
        strength: float,
    ) -> "SurfaceMesh":
        """Increase face scattering inside a body-frame angular cap."""

        direction = np.asarray(direction_body, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Spot direction must be a finite 3-vector")
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Spot direction cannot be zero")
        if not 0 < angular_radius_deg <= 180:
            raise ValueError("Spot angular radius must be in (0, 180]")
        if strength < 1:
            raise ValueError("Spot strength must be at least one")

        direction = direction / norm
        radial = self.face_centroids / np.linalg.norm(
            self.face_centroids, axis=1, keepdims=True
        )
        threshold = np.cos(np.deg2rad(angular_radius_deg))
        selected = radial @ direction >= threshold
        coefficients = self.scattering.copy()
        coefficients[selected] *= strength
        return replace(self, scattering=coefficients)

    @property
    def characteristic_radius_m(self) -> float:
        return float(np.linalg.norm(self.vertices, axis=1).max())

