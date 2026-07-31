import tempfile
import unittest
from pathlib import Path

import torch

from asteroid_radar.ellipsoid import save_ellipsoid
from asteroid_radar.mesh import load_mesh


class MeshTests(unittest.TestCase):
    def test_generated_ellipsoid_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ellipsoid.obj"
            save_ellipsoid(path, [3.0, 2.0, 1.0], 2)
            mesh = load_mesh(path)
        self.assertEqual(len(mesh.faces), 320)
        self.assertTrue(bool(torch.all((mesh.centroids * mesh.normals).sum(1) > 0)))

    def test_scattering_spot_changes_part_of_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ellipsoid.ply"
            save_ellipsoid(path, [3.0, 2.0, 1.0], 2)
            mesh = load_mesh(path)
        spotted = mesh.add_scattering_spot([1, 0, 0], 30, 4)
        changed = spotted.scattering > mesh.scattering
        self.assertGreater(changed.sum(), 0)
        self.assertLess(changed.sum(), len(mesh.faces))

    def test_float64_mesh_remains_float64(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ellipsoid.obj"
            save_ellipsoid(path, [3.0, 2.0, 1.0], 1)
            mesh = load_mesh(path, dtype="float64")
        self.assertEqual(mesh.normals.dtype, torch.float64)
        self.assertEqual(mesh.areas.dtype, torch.float64)
