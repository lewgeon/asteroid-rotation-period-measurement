import unittest

import numpy as np
import torch

from asteroid_rotation.acquisition import AcquisitionSchedule
from asteroid_rotation.dynamics import SpinState
from asteroid_rotation.echo import RadarGeometry, simulate_continuous_wave_echo
from asteroid_rotation.geometry import SurfaceMesh


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class BackendParityTests(unittest.TestCase):
    def test_cpu_float64_and_cuda_float32_echoes_agree(self):
        cpu_mesh = SurfaceMesh.ellipsoid(
            [1.0, 0.72, 0.55], 1, device="cpu", dtype="float64"
        ).with_scattering_spot([1.0, 0.2, 0.1], 45.0, 5.0)
        cuda_mesh = cpu_mesh.to("cuda:0", "float32")
        spin = SpinState(
            period_s=20.0,
            axis_icrs=np.array([0.2, -0.3, 0.9327379053]),
            initial_phase_rad=0.37,
        )
        geometry = RadarGeometry(
            np.array([0.92, 0.18, 0.35]),
            np.array([0.88, -0.12, 0.46]),
        )
        schedule = AcquisitionSchedule.continuous_wave(
            "2026-01-01T00:00:00.000", duration_s=10.0, sample_rate_hz=64.0
        )
        common = dict(
            spin=spin,
            geometry=geometry,
            schedule=schedule,
            carrier_frequency_hz=7.15e9,
            sample_rate_hz=64.0,
            translation_coefficients_hz=[0.5, 0.001],
            snr_db=None,
            cosine_power_tx=1.0,
            cosine_power_rx=1.0,
            chunk_size=256,
        )
        cpu = simulate_continuous_wave_echo(
            mesh=cpu_mesh, rng=np.random.default_rng(1), **common
        )
        cuda = simulate_continuous_wave_echo(
            mesh=cuda_mesh, rng=np.random.default_rng(1), **common
        )
        relative_rms = np.linalg.norm(cpu.clean_iq - cuda.clean_iq) / np.linalg.norm(
            cpu.clean_iq
        )
        self.assertLess(relative_rms, 2e-4)


if __name__ == "__main__":
    unittest.main()
