import copy
import json
import unittest
from pathlib import Path

from asteroid_radar.inversion import estimate_rotation
from asteroid_radar.simulation import simulate


ROOT = Path(__file__).resolve().parents[2]


class SimulationToInversionTests(unittest.TestCase):
    def test_short_experiment_recovers_rotation_candidate(self):
        simulation = json.loads(
            (ROOT / "configs/simulation/cw_ellipsoid.json").read_text()
        )
        inversion = json.loads(
            (ROOT / "configs/inversion/lomb_scargle.json").read_text()
        )
        simulation = copy.deepcopy(simulation)
        simulation["seed"] = 11
        simulation["compute"] = {"device": "cpu", "dtype": "float64"}
        simulation["target"]["semi_axes_m"] = [1.0, 0.72, 0.55]
        simulation["target"]["mesh_subdivisions"] = 1
        simulation["target"]["rotation_period_s"] = 20.0
        simulation["target"]["scattering_spot"]["radius_deg"] = 45.0
        simulation["target"]["scattering_spot"]["strength"] = 10.0
        simulation["radar"]["sample_rate_hz"] = 64.0
        simulation["observation"]["duration_s"] = 120.0
        simulation["translation_doppler_hz"] = [1.0, 0.001]
        simulation["snr_db"] = 20.0
        inversion.update({
            "stft_window_samples": 256,
            "period_min_s": 8.0,
            "period_max_s": 40.0,
            "period_grid_size": 4000,
        })

        echo = simulate(simulation)
        result = estimate_rotation(echo, inversion)
        candidates = result.periods["total_power"].candidates
        errors = [abs(item.period_s - 20.0) / 20.0 for item in candidates]
        self.assertLess(min(errors), 0.05)
