"""Generate an ellipsoid mesh for controlled simulation experiments."""

from pathlib import Path

import torch
from pytorch3d.io import save_obj, save_ply
from pytorch3d.utils import ico_sphere


def ellipsoid_mesh(semi_axes_m, subdivisions):
    sphere = ico_sphere(subdivisions, device="cpu")
    axes = torch.tensor(semi_axes_m, dtype=torch.float32)
    return sphere.verts_packed() * axes, sphere.faces_packed()


def save_ellipsoid(path, semi_axes_m, subdivisions):
    """Write a body-centred ellipsoid in metres as OBJ or PLY."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = ellipsoid_mesh(semi_axes_m, subdivisions)
    if path.suffix.lower() == ".obj":
        save_obj(path, vertices, faces, decimal_places=10)
    else:
        save_ply(path, vertices, faces, ascii=True, decimal_places=10)
