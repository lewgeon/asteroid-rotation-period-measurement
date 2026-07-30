import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asteroid_radar.data import ObservationSchedule
from asteroid_radar.simulation.cw import rotational_doppler, simulate_cw
from asteroid_radar.simulation.geometry import SurfaceMesh
from asteroid_radar.simulation.motion import Spin


class ForwardModelTests(unittest.TestCase):
    def test_ellipsoid_normals_point_outward(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 2)
        self.assertTrue(bool(torch.all((mesh.centroids * mesh.normals).sum(1) > 0)))

    def test_scattering_spot_changes_part_of_the_surface(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 2)
        spotted = mesh.add_scattering_spot([1, 0, 0], 30, 4)
        changed = spotted.scattering > mesh.scattering
        self.assertGreater(changed.sum(), 0)
        self.assertLess(changed.sum(), len(mesh.faces))

    def test_float64_geometry_remains_float64(self):
        mesh = SurfaceMesh.ellipsoid([3.0, 2.0, 1.0], 1, dtype="float64")
        self.assertEqual(mesh.normals.dtype, torch.float64)
        self.assertEqual(mesh.areas.dtype, torch.float64)

    def test_obj_loading(self):
        text = "\n".join([
            "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 1",
            "f 1 3 2", "f 1 2 4", "f 1 4 3", "f 2 3 4",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.obj"
            path.write_text(text, encoding="ascii")
            mesh = SurfaceMesh.from_file(path, scale_m=2.0)
        self.assertEqual(len(mesh.faces), 4)

    def test_nonuniform_times_are_preserved(self):
        schedule = ObservationSchedule.from_utc([
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:01.250",
            "2020-01-01T00:00:04",
        ])
        np.testing.assert_allclose(schedule.elapsed_s, [0, 1.25, 4])

    def test_rotational_doppler_matches_the_analytic_case(self):
        positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        omega = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        doppler = rotational_doppler(
            positions, omega, [0, 1, 0], [0, 1, 0], wavelength=0.5
        )
        torch.testing.assert_close(
            doppler, torch.tensor([8.0], dtype=torch.float64)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_and_cuda_echoes_agree(self):
        cpu_mesh = SurfaceMesh.ellipsoid(
            [1.0, 0.72, 0.55], 1, dtype="float64"
        ).add_scattering_spot([1, 0.2, 0.1], 45, 5)
        cuda_mesh = cpu_mesh.to("cuda:0", "float32")
        spin = Spin(20.0, np.array([0.2, -0.3, 0.9327379053]), 0.37)
        schedule = ObservationSchedule.continuous_wave(
            "2020-01-01", 10.0, 64.0
        )
        radar = {
            "carrier_frequency_hz": 7.15e9,
            "sample_rate_hz": 64.0,
            "tx_los_icrs": [0.92, 0.18, 0.35],
            "rx_los_icrs": [0.88, -0.12, 0.46],
        }
        common = (spin, schedule, radar, [0.5, 0.001], None)
        cpu = simulate_cw(
            cpu_mesh, *common, np.random.default_rng(1), chunk_size=256
        )
        cuda = simulate_cw(
            cuda_mesh, *common, np.random.default_rng(1), chunk_size=256
        )
        error = np.linalg.norm(cpu.clean_iq - cuda.clean_iq)
        error /= np.linalg.norm(cpu.clean_iq)
        self.assertLess(error, 2e-4)
