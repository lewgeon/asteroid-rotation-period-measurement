import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asteroid_radar.echo import simulate_echo
from asteroid_radar.ellipsoid import save_ellipsoid


ROOT = Path(__file__).resolve().parents[1]


class BackendTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_and_cuda_echoes_agree(self):
        config = json.loads((ROOT / "configs/echo.json").read_text())
        config["observation"]["duration_s"] = 0.5
        config["radar"]["sample_rate_hz"] = 16.0
        config["snr_db"] = None
        cpu_config = copy.deepcopy(config)
        cpu_config["compute"] = {"device": "cpu", "dtype": "float64"}
        cuda_config = copy.deepcopy(config)
        cuda_config["compute"] = {"device": "cuda:0", "dtype": "float32"}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.obj"
            save_ellipsoid(model, [1.0, 0.72, 0.55], 1)
            cpu_config["model_path"] = str(model)
            cuda_config["model_path"] = str(model)
            cpu = simulate_echo(cpu_config)
            cuda = simulate_echo(cuda_config)
        error = np.linalg.norm(cpu.clean_iq - cuda.clean_iq)
        error /= np.linalg.norm(cpu.clean_iq)
        self.assertLess(error, 2e-4)
