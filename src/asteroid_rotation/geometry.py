"""PyTorch3D-backed triangle geometry for the physical echo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


def resolve_device(value: str | torch.device) -> torch.device:
    """Resolve an explicit or automatic Torch device."""

    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def resolve_float_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        dtype = value
    else:
        mapping = {"float32": torch.float32, "float64": torch.float64}
        try:
            dtype = mapping[value]
        except KeyError as error:
            raise ValueError(f"Unsupported floating dtype: {value}") from error
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Only float32 and float64 are supported")
    return dtype


def _tensor(
    value: Iterable[float] | np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Geometry contains non-finite values")
    return result


@dataclass(frozen=True)
class SurfaceMesh:
    """One centered triangle mesh plus project-specific per-face scattering."""

    pytorch3d: Meshes
    vertices: torch.Tensor
    faces: torch.Tensor
    face_centroids: torch.Tensor
    face_normals: torch.Tensor
    face_areas: torch.Tensor
    scattering: torch.Tensor

    @classmethod
    def from_vertices_faces(
        cls,
        vertices: Iterable[Iterable[float]] | np.ndarray | torch.Tensor,
        faces: Iterable[Iterable[int]] | np.ndarray | torch.Tensor,
        scattering: Iterable[float] | np.ndarray | torch.Tensor | None = None,
        *,
        center: bool = True,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = torch.float64,
        orient_outward_from_origin: bool = True,
    ) -> "SurfaceMesh":
        target_device = resolve_device(device)
        target_dtype = resolve_float_dtype(dtype)
        vertex_tensor = _tensor(
            vertices, device=target_device, dtype=target_dtype
        )
        face_tensor = torch.as_tensor(faces, device=target_device, dtype=torch.int64)
        if vertex_tensor.ndim != 2 or vertex_tensor.shape[1] != 3:
            raise ValueError(f"Vertices must have shape (N, 3), got {vertex_tensor.shape}")
        if face_tensor.ndim != 2 or face_tensor.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {face_tensor.shape}")
        if face_tensor.numel() == 0:
            raise ValueError("Mesh must contain at least one face")
        if int(face_tensor.min()) < 0 or int(face_tensor.max()) >= len(vertex_tensor):
            raise ValueError("Face index is outside the vertex array")

        if center:
            vertex_tensor = vertex_tensor - vertex_tensor.mean(dim=0, keepdim=True)

        face_vertices = vertex_tensor[face_tensor]
        edge_a = face_vertices[:, 1] - face_vertices[:, 0]
        edge_b = face_vertices[:, 2] - face_vertices[:, 0]
        cross = torch.linalg.cross(edge_a, edge_b, dim=1)
        double_area = torch.linalg.vector_norm(cross, dim=1)
        epsilon = torch.finfo(target_dtype).eps
        if bool((double_area <= epsilon).any()):
            raise ValueError("Mesh contains degenerate triangular faces")

        centroids = face_vertices.mean(dim=1)
        normals = cross / double_area[:, None]
        if orient_outward_from_origin:
            inward = torch.sum(normals * centroids, dim=1) < 0
            if bool(inward.any()):
                face_tensor = face_tensor.clone()
                face_tensor[inward] = face_tensor[inward][:, [0, 2, 1]]
                normals = normals.clone()
                normals[inward] *= -1.0

        areas = 0.5 * double_area
        if scattering is None:
            scattering_tensor = torch.ones(
                len(face_tensor), device=target_device, dtype=target_dtype
            )
        else:
            scattering_tensor = _tensor(
                scattering, device=target_device, dtype=target_dtype
            )
            if scattering_tensor.shape != (len(face_tensor),):
                raise ValueError("Scattering must contain one value per face")
            if bool((scattering_tensor < 0).any()):
                raise ValueError("Scattering coefficients must be non-negative")

        mesh = Meshes(verts=[vertex_tensor], faces=[face_tensor])
        return cls(
            pytorch3d=mesh,
            vertices=vertex_tensor.contiguous(),
            faces=face_tensor.contiguous(),
            face_centroids=centroids.contiguous(),
            face_normals=normals.contiguous(),
            face_areas=areas.contiguous(),
            scattering=scattering_tensor.contiguous(),
        )

    @classmethod
    def ellipsoid(
        cls,
        semi_axes_m: Iterable[float],
        subdivision_level: int,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = torch.float64,
    ) -> "SurfaceMesh":
        """Generate a near-uniform icosphere and scale it into an ellipsoid."""

        target_device = resolve_device(device)
        target_dtype = resolve_float_dtype(dtype)
        axes = _tensor(semi_axes_m, device=target_device, dtype=target_dtype)
        if axes.shape != (3,) or bool((axes <= 0).any()):
            raise ValueError("semi_axes_m must contain three positive values")
        if not 0 <= subdivision_level <= 7:
            raise ValueError("subdivision_level must be between 0 and 7")

        sphere = ico_sphere(subdivision_level, device=target_device)
        vertices = sphere.verts_packed().to(dtype=target_dtype) * axes
        faces = sphere.faces_packed()
        return cls.from_vertices_faces(
            vertices,
            faces,
            center=False,
            device=target_device,
            dtype=target_dtype,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        scale_m_per_model_unit: float = 1.0,
        center: bool = True,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = torch.float64,
        orient_outward_from_origin: bool = True,
    ) -> "SurfaceMesh":
        """Load an OBJ or PLY triangle mesh through PyTorch3D."""

        if not np.isfinite(scale_m_per_model_unit) or scale_m_per_model_unit <= 0:
            raise ValueError("scale_m_per_model_unit must be positive")
        target_device = resolve_device(device)
        target_dtype = resolve_float_dtype(dtype)
        suffix = str(path).lower().rsplit(".", 1)[-1]
        if suffix == "obj":
            vertices, face_data, _ = load_obj(
                path, load_textures=False, device=target_device
            )
            faces = face_data.verts_idx
        elif suffix == "ply":
            vertices, faces = load_ply(path)
            vertices = vertices.to(target_device)
            faces = faces.to(target_device)
        else:
            raise ValueError("Only OBJ and PLY mesh files are supported")
        vertices = (
            vertices.to(device=target_device, dtype=target_dtype)
            * scale_m_per_model_unit
        )
        return cls.from_vertices_faces(
            vertices,
            faces,
            center=center,
            device=target_device,
            dtype=target_dtype,
            orient_outward_from_origin=orient_outward_from_origin,
        )

    def with_scattering_spot(
        self,
        direction_body: Iterable[float],
        angular_radius_deg: float,
        strength: float,
    ) -> "SurfaceMesh":
        """Increase face scattering inside a body-frame angular cap."""

        direction = _tensor(
            direction_body, device=self.device, dtype=self.dtype
        )
        if direction.shape != (3,):
            raise ValueError("Spot direction must be a finite 3-vector")
        norm = torch.linalg.vector_norm(direction)
        if float(norm) == 0:
            raise ValueError("Spot direction cannot be zero")
        if not 0 < angular_radius_deg <= 180:
            raise ValueError("Spot angular radius must be in (0, 180]")
        if strength < 1:
            raise ValueError("Spot strength must be at least one")

        direction = direction / norm
        radial = self.face_centroids / torch.linalg.vector_norm(
            self.face_centroids, dim=1, keepdim=True
        )
        threshold = np.cos(np.deg2rad(angular_radius_deg))
        selected = radial @ direction >= threshold
        coefficients = self.scattering.clone()
        coefficients[selected] *= strength
        return SurfaceMesh(
            pytorch3d=self.pytorch3d,
            vertices=self.vertices,
            faces=self.faces,
            face_centroids=self.face_centroids,
            face_normals=self.face_normals,
            face_areas=self.face_areas,
            scattering=coefficients,
        )

    def to(
        self,
        device: str | torch.device,
        dtype: str | torch.dtype | None = None,
    ) -> "SurfaceMesh":
        """Move the complete mesh and scattering state without changing its interface."""

        target_device = resolve_device(device)
        target_dtype = self.dtype if dtype is None else resolve_float_dtype(dtype)
        return SurfaceMesh.from_vertices_faces(
            self.vertices.to(device=target_device, dtype=target_dtype),
            self.faces.to(device=target_device),
            self.scattering.to(device=target_device, dtype=target_dtype),
            center=False,
            device=target_device,
            dtype=target_dtype,
            orient_outward_from_origin=False,
        )

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vertices.dtype

    @property
    def characteristic_radius_m(self) -> float:
        return float(torch.linalg.vector_norm(self.vertices, dim=1).max().item())

    def numpy_geometry(self) -> dict[str, np.ndarray]:
        """Copy geometry to NumPy only at an explicit I/O or plotting seam."""

        return {
            "vertices": self.vertices.detach().cpu().numpy(),
            "faces": self.faces.detach().cpu().numpy(),
            "face_centroids": self.face_centroids.detach().cpu().numpy(),
            "face_normals": self.face_normals.detach().cpu().numpy(),
            "face_areas": self.face_areas.detach().cpu().numpy(),
            "scattering": self.scattering.detach().cpu().numpy(),
        }
