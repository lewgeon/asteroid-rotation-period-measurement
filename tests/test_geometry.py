import unittest

import numpy as np

from asteroid_rotation.geometry import SurfaceMesh


class GeometryTests(unittest.TestCase):
    def test_ellipsoid_faces_point_outward(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 8, 16)
        orientation = np.einsum(
            "ij,ij->i", mesh.face_centroids, mesh.face_normals
        )
        self.assertTrue(np.all(orientation > 0))
        self.assertTrue(np.all(mesh.face_areas > 0))

    def test_scattering_spot_changes_a_subset(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 8, 16)
        spotted = mesh.with_scattering_spot([1.0, 0.0, 0.0], 30.0, 4.0)
        changed = spotted.scattering > mesh.scattering
        self.assertGreater(changed.sum(), 0)
        self.assertLess(changed.sum(), len(mesh.faces))


if __name__ == "__main__":
    unittest.main()

