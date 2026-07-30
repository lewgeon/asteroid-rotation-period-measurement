"""Triangle meshes used by the echo model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


def torch_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def torch_dtype(name):
    return {"float32": torch.float32, "float64": torch.float64}[name]


@dataclass
class SurfaceMesh:
    """A triangle mesh with precomputed face quantities."""

    vertices: torch.Tensor
    faces: torch.Tensor
    centroids: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    scattering: torch.Tensor

    @classmethod
    def from_vertices_faces(
        cls, vertices, faces, scattering=None, center=True, device="cpu",
        dtype="float64"
    ):
        device, dtype = torch_device(device), torch_dtype(dtype)
        vertices = torch.as_tensor(vertices, device=device, dtype=dtype)
        faces = torch.as_tensor(faces, device=device, dtype=torch.long)
        if center:
            vertices = vertices - vertices.mean(0)

        triangles = vertices[faces]
        cross = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        )
        lengths = torch.linalg.vector_norm(cross, dim=1)
        centroids = triangles.mean(1)
        normals = cross / lengths[:, None]

        inward = (normals * centroids).sum(1) < 0
        faces = faces.clone()
        faces[inward] = faces[inward][:, [0, 2, 1]]
        normals[inward] *= -1

        if scattering is None:
            scattering = torch.ones(len(faces), device=device, dtype=dtype)
        else:
            scattering = torch.as_tensor(scattering, device=device, dtype=dtype)
        return cls(vertices, faces, centroids, normals, lengths / 2, scattering)

    @classmethod
    def ellipsoid(cls, axes_m, subdivisions, device="cpu", dtype="float64"):
        device, dtype = torch_device(device), torch_dtype(dtype)
        sphere = ico_sphere(subdivisions, device=device)
        vertices = sphere.verts_packed().to(dtype) * torch.tensor(
            axes_m, device=device, dtype=dtype
        )
        return cls.from_vertices_faces(
            vertices, sphere.faces_packed(), center=False,
            device=str(device), dtype=str(dtype).split(".")[-1]
        )

    @classmethod
    def from_file(
        cls, path, scale_m=1.0, center=True, device="cpu", dtype="float64"
    ):
        device = torch_device(device)
        if Path(path).suffix.lower() == ".obj":
            vertices, faces, _ = load_obj(path, load_textures=False, device=device)
            faces = faces.verts_idx
        else:
            vertices, faces = load_ply(path)
        return cls.from_vertices_faces(
            vertices * scale_m, faces, center=center, device=str(device), dtype=dtype
        )

    def add_scattering_spot(self, direction, radius_deg, strength):
        direction = torch.tensor(direction, device=self.device, dtype=self.dtype)
        direction /= torch.linalg.vector_norm(direction)
        radial = self.centroids / torch.linalg.vector_norm(
            self.centroids, dim=1, keepdim=True
        )
        selected = radial @ direction >= np.cos(np.deg2rad(radius_deg))
        scattering = self.scattering.clone()
        scattering[selected] *= strength
        return SurfaceMesh(
            self.vertices, self.faces, self.centroids, self.normals,
            self.areas, scattering
        )

    def to(self, device, dtype):
        return SurfaceMesh.from_vertices_faces(
            self.vertices.to(torch_device(device), torch_dtype(dtype)),
            self.faces.to(torch_device(device)),
            self.scattering.to(torch_device(device), torch_dtype(dtype)),
            center=False, device=device, dtype=dtype
        )

    @property
    def device(self):
        return self.vertices.device

    @property
    def dtype(self):
        return self.vertices.dtype

    @property
    def pytorch3d(self):
        return Meshes(verts=[self.vertices], faces=[self.faces])

    @property
    def radius_m(self):
        return torch.linalg.vector_norm(self.vertices, dim=1).max().item()
