import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from asteroid_radar.echo import simulate_echo
from asteroid_radar.ellipsoid import save_ellipsoid
from asteroid_radar.inversion import estimate_rotation, lomb_scargle


ROOT = Path(__file__).resolve().parents[1]


class InversionTests(unittest.TestCase):
    def test_nonuniform_period_recovery(self):
        rng = np.random.default_rng(7)
        times = np.sort(rng.uniform(0, 250, 400))
        truth = 23.0
        values = np.sin(2 * np.pi * times / truth)
        values += 0.05 * rng.standard_normal(len(times))
        estimate = lomb_scargle(times, values, 10, 40, 5000)
        self.assertLess(abs(estimate.best_period_s - truth) / truth, 0.01)

    def test_short_echo_recovers_rotation_candidate(self):
        echo_config = json.loads((ROOT / "configs/echo.json").read_text())
        inversion_config = json.loads((ROOT / "configs/inversion.json").read_text())
        echo_config = copy.deepcopy(echo_config)
        echo_config["seed"] = 11
        echo_config["compute"] = {"device": "cpu", "dtype": "float64"}
        echo_config["target"]["rotation_period_s"] = 20.0
        echo_config["scattering_spot"]["radius_deg"] = 45.0
        echo_config["scattering_spot"]["strength"] = 10.0
        echo_config["radar"]["sample_rate_hz"] = 64.0
        echo_config["observation"]["duration_s"] = 120.0
        echo_config["translation_doppler_hz"] = [1.0, 0.001]
        echo_config["snr_db"] = 20.0
        inversion_config.update({
            "stft_window_samples": 256,
            "period_min_s": 8.0,
            "period_max_s": 40.0,
            "period_grid_size": 4000,
        })
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.obj"
            save_ellipsoid(model, [1.0, 0.72, 0.55], 1)
            echo_config["model_path"] = str(model)
            echo = simulate_echo(echo_config)
        result = estimate_rotation(echo, inversion_config)
        candidates = result.periods["total_power"].candidates
        errors = [abs(item.period_s - 20.0) / 20.0 for item in candidates]
        self.assertLess(min(errors), 0.05)
