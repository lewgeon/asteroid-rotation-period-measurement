import tempfile
import unittest
from pathlib import Path

import numpy as np

from asteroid_radar.data import EchoDataset, load_echo, save_echo


class EchoDataTests(unittest.TestCase):
    def test_saved_echo_is_the_inversion_input(self):
        echo = EchoDataset(
            elapsed_s=np.arange(4.0),
            iq=np.arange(4) + 1j,
            clean_iq=np.arange(4) + 1j,
            valid=np.ones(4, dtype=bool),
            coherence_id=np.zeros(4, dtype=int),
            translation_doppler_hz=np.ones(4),
            metadata={"sample_rate_hz": 1.0, "truth_period_s": 2.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "echo.npz"
            save_echo(path, echo)
            loaded = load_echo(path)
        np.testing.assert_array_equal(loaded.iq, echo.iq)
        self.assertEqual(loaded.metadata, echo.metadata)
