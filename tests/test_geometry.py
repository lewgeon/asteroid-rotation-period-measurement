import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from asteroid_rotation.geometry import SurfaceMesh


class GeometryTests(unittest.TestCase):
    def test_ellipsoid_faces_point_outward(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 2)
        orientation = torch.sum(mesh.face_centroids * mesh.face_normals, dim=1)
        self.assertTrue(bool(torch.all(orientation > 0)))
        self.assertTrue(bool(torch.all(mesh.face_areas > 0)))
        self.assertEqual(len(mesh.pytorch3d), 1)

    def test_scattering_spot_changes_a_subset(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 2)
        spotted = mesh.with_scattering_spot([1.0, 0.0, 0.0], 30.0, 4.0)
        changed = spotted.scattering > mesh.scattering
        self.assertGreater(int(changed.sum()), 0)
        self.assertLess(int(changed.sum()), len(mesh.faces))

    def test_float64_reference_geometry_is_not_downcast(self):
        mesh = SurfaceMesh.ellipsoid(
            [3.0, 2.0, 1.0], 1, device="cpu", dtype="float64"
        )
        self.assertEqual(mesh.vertices.dtype, torch.float64)
        self.assertEqual(mesh.face_normals.dtype, torch.float64)
        self.assertEqual(mesh.face_areas.dtype, torch.float64)

    def test_obj_is_loaded_through_pytorch3d(self):
        obj_text = "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 3 2",
                "f 1 2 4",
                "f 1 4 3",
                "f 2 3 4",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tetrahedron.obj"
            path.write_text(obj_text, encoding="ascii")
            mesh = SurfaceMesh.from_file(
                str(path),
                scale_m_per_model_unit=2.0,
                device="cpu",
                dtype="float64",
            )
        self.assertEqual(len(mesh.faces), 4)
        self.assertEqual(mesh.vertices.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
