import copy
import unittest
from pathlib import Path

from asteroid_rotation.config import ExperimentConfig, load_experiment_config
from asteroid_rotation.experiment import run_continuous_wave_experiment


ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    def test_short_cw_experiment_produces_period_candidates(self):
        baseline = load_experiment_config(ROOT / "configs" / "baseline_cw.json")
        data = copy.deepcopy(baseline.data)
        data["experiment"]["seed"] = 11
        data["compute"]["device"] = "cpu"
        data["compute"]["dtype"] = "float64"
        data["target"]["semi_axes_m"] = [1.0, 0.72, 0.55]
        data["target"]["mesh_subdivision_level"] = 1
        data["target"]["rotation_period_s"] = 20.0
        data["target"]["scattering"]["spot_angular_radius_deg"] = 45.0
        data["target"]["scattering"]["spot_strength"] = 10.0
        data["radar"]["baseband_sample_rate_hz"] = 64.0
        data["observation"]["duration_s"] = 120.0
        data["translation_doppler"]["polynomial_hz"] = [1.0, 0.001]
        data["noise"]["snr_db"] = 20.0
        data["processing"]["stft_window_samples"] = 256
        data["processing"]["period_min_s"] = 8.0
        data["processing"]["period_max_s"] = 40.0
        data["processing"]["period_grid_size"] = 4000
        config = ExperimentConfig(data, baseline.path, baseline.schema_path)

        result = run_continuous_wave_experiment(config)
        self.assertGreater(len(result.mesh.faces), 0)
        self.assertEqual(result.echo.iq.shape, result.echo.elapsed_s.shape)
        self.assertIn("total_power", result.period_estimates)

        candidates = result.period_estimates["total_power"].candidates
        relative_errors = [
            abs(candidate.period_s - 20.0) / 20.0 for candidate in candidates
        ]
        self.assertLess(min(relative_errors), 0.05)


if __name__ == "__main__":
    unittest.main()
