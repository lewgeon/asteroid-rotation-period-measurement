"""Load a prepared OBJ/PLY mesh and compute quantities used by the echo model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_obj, load_ply


def torch_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def torch_dtype(name):
    return {"float32": torch.float32, "float64": torch.float64}[name]


@dataclass
class SurfaceMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    centroids: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    scattering: torch.Tensor

    @classmethod
    def from_tensors(cls, vertices, faces, scattering=None, device="cpu", dtype="float64"):
        device, dtype = torch_device(device), torch_dtype(dtype)
        vertices = torch.as_tensor(vertices, device=device, dtype=dtype)
        faces = torch.as_tensor(faces, device=device, dtype=torch.long)
        triangles = vertices[faces]
        cross = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        )
        lengths = torch.linalg.vector_norm(cross, dim=1)
        if scattering is None:
            scattering = torch.ones(len(faces), device=device, dtype=dtype)
        else:
            scattering = torch.as_tensor(scattering, device=device, dtype=dtype)
        return cls(
            vertices,
            faces,
            triangles.mean(1),
            cross / lengths[:, None],
            lengths / 2,
            scattering,
        )

    def add_scattering_spot(self, direction, radius_deg, strength):
        direction = torch.tensor(direction, device=self.device, dtype=self.dtype)
        direction /= torch.linalg.vector_norm(direction)
        radial = self.centroids / torch.linalg.vector_norm(
            self.centroids, dim=1, keepdim=True
        )
        scattering = self.scattering.clone()
        scattering[radial @ direction >= np.cos(np.deg2rad(radius_deg))] *= strength
        return SurfaceMesh(
            self.vertices,
            self.faces,
            self.centroids,
            self.normals,
            self.areas,
            scattering,
        )

    def to(self, device, dtype):
        return SurfaceMesh.from_tensors(
            self.vertices,
            self.faces,
            self.scattering,
            device=device,
            dtype=dtype,
        )

    @property
    def device(self):
        return self.vertices.device

    @property
    def dtype(self):
        return self.vertices.dtype

    @property
    def radius_m(self):
        return torch.linalg.vector_norm(self.vertices, dim=1).max().item()


def load_mesh(path, device="cpu", dtype="float64"):
    """Load a mesh already expressed in body-fixed metres."""

    if Path(path).suffix.lower() == ".obj":
        vertices, faces, _ = load_obj(path, load_textures=False)
        faces = faces.verts_idx
    else:
        vertices, faces = load_ply(path)
    return SurfaceMesh.from_tensors(vertices, faces, device=device, dtype=dtype)
