import copy
import json
import tempfile
import unittest
from pathlib import Path

from asteroid_rotation.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_baseline_config_is_valid(self):
        config = load_experiment_config(ROOT / "configs" / "baseline_cw.json")
        self.assertEqual(config.data["radar"]["waveform"]["type"], "continuous_wave")

    def test_unknown_parameter_is_rejected(self):
        baseline = load_experiment_config(ROOT / "configs" / "baseline_cw.json")
        altered = copy.deepcopy(baseline.data)
        altered["radar"]["unexpected"] = 123
        altered["$schema"] = str(baseline.schema_path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()

