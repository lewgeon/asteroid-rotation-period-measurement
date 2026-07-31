import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asteroid_radar.dataset import ObservationSchedule, load_echo, save_echo
from asteroid_radar.echo import rotational_doppler, simulate_echo
from asteroid_radar.ellipsoid import save_ellipsoid


ROOT = Path(__file__).resolve().parents[1]


class EchoTests(unittest.TestCase):
    def test_nonuniform_times_are_preserved(self):
        schedule = ObservationSchedule.from_utc([
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:01.250",
            "2020-01-01T00:00:04",
        ])
        np.testing.assert_allclose(schedule.elapsed_s, [0, 1.25, 4])

    def test_rotational_doppler_matches_analytic_case(self):
        positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        omega = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        doppler = rotational_doppler(
            positions, omega, [0, 1, 0], [0, 1, 0], wavelength=0.5
        )
        torch.testing.assert_close(
            doppler, torch.tensor([8.0], dtype=torch.float64)
        )

    def test_saved_echo_is_the_inversion_input(self):
        config = json.loads((ROOT / "configs/echo.json").read_text())
        config = copy.deepcopy(config)
        config["compute"] = {"device": "cpu", "dtype": "float64"}
        config["observation"]["duration_s"] = 2.0
        config["radar"]["sample_rate_hz"] = 64.0
        config["snr_db"] = None
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.obj"
            save_ellipsoid(model, [1.0, 0.72, 0.55], 1)
            config["model_path"] = str(model)
            echo = simulate_echo(config)
            path = Path(directory) / "echo.npz"
            save_echo(path, echo)
            loaded = load_echo(path)
        np.testing.assert_array_equal(loaded.iq, echo.iq)
        self.assertEqual(loaded.metadata, echo.metadata)
