"""Execute the real CLI and inspect its exported observable semantics."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from inversion.tests.scratch import scratch_directory
from inversion.tests import test_chirp_contracts


class RangeFwhmCliTests(unittest.TestCase):
    def test_chirp_cpi_fallback_and_cw_exports(self):
        root = Path(__file__).resolve().parents[2]
        for mode in ("cpi", "lag", "fallback", "cw", "gpu_error"):
            with self.subTest(mode=mode), scratch_directory("range_fwhm_cli") as scratch:
                if mode == "cw":
                    t = np.arange(2048) / 64.
                    values = {"elapsed_s": t, "iq": 1 + .3 * np.cos(2*np.pi*t/2),
                              "metadata_json": json.dumps({"sample_rate_hz": 64.})}
                    config = {"stft_window_samples": 64, "stft_overlap_fraction": .5,
                              "period_min_s": 1., "period_max_s": 3., "period_grid_size": 40}
                else:
                    e = test_chirp_contracts.chirp_echo("centroid_compensated")
                    values = vars(e).copy()
                    values["metadata_json"] = json.dumps(values.pop("metadata"))
                    n = len(e.elapsed_s)
                    for key in ("acquisition_id", "track_id", "row_start_sample",
                                "rx_adc_start_elapsed_s", "rx_adc_stop_elapsed_s",
                                "signal_echo_overlap", "window_overlap", "common_path_rate_m_s"):
                        values[key] = np.zeros(n)
                    # One zero row must produce an unestimable width, not zero.
                    values["iq"] = e.iq.copy()
                    values["iq"][-1] = 0
                    values["clean_iq"] = values["iq"].copy()
                    config = dict(test_chirp_contracts.ChirpInversionTests.config)
                    if mode=='gpu_error':
                        config.update(compute_device='cuda:0',uncertainty_method='run_bootstrap',uncertainty_replicates=30)
                    if mode == 'lag':config['doppler_estimator']='narrowband_lag'
                    if mode == "fallback":
                        config["cpi_duration_s"] = 20.
                echo_path = scratch / "echo.npz"
                np.savez(echo_path, **values)
                config_path = scratch / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(root / "inversion/scripts/estimate_period.py"),
                     "--echo", str(echo_path), "--config", str(config_path),
                     "--output", str(scratch / "output")],
                    cwd=root, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                completion=json.loads((scratch/'output/inversion_complete.json').read_text(encoding='utf-8'))
                self.assertFalse((scratch/'output/inversion_in_progress.json').exists())
                if mode=='cw':self.assertIsNone(completion['inspection_folder'])
                else:
                    inspection=Path(completion['inspection_folder'])
                    self.assertTrue((inspection/'manifest.json').is_file())
                    with np.load(inspection/'pulse_observables.npz') as inspection_data:
                        self.assertEqual(inspection_data['value'].shape,(len(e.iq),3))
                best = json.loads((scratch / "output/best_summary.json").read_text(encoding="utf-8"))
                uncertainty=json.loads((scratch/'output/period_uncertainty.json').read_text(encoding='utf-8'))
                self.assertEqual(uncertainty['enabled'],mode=='gpu_error')
                self.assertIsNone(best['final_3sigma_interval_s'])
                if mode=='gpu_error':
                    self.assertEqual(uncertainty['status'],'unavailable')
                    self.assertEqual(best['search_diagnostics']['compute_execution']['requested'],'cuda:0')
                    self.assertTrue((scratch/'output/period_uncertainty.html').is_file())
                summary = json.loads((scratch / "output/summary.json").read_text(encoding="utf-8"))
                for name in ('features.html', 'processing.html', 'processing_preview.npz', 'periodogram.html'):
                    self.assertTrue((scratch / 'output' / name).is_file(), name)
                from rotation_gui.window.inversion_feature_preview import cached_feature_html
                self.assertTrue(cached_feature_html(scratch / 'output/period_observables.npz', config,
                    scratch / 'output/features.html', saved=True, echo_npz=echo_path))
                with np.load(scratch / 'output/processing_preview.npz') as preview:
                    self.assertEqual('rd_power' in preview.files, mode in ('cpi','lag','gpu_error'))
                with np.load(scratch / "output/period_observables.npz") as data:
                    names = list(data["feature_names"])
                    self.assertFalse(any("fwhm" in name for name in names))
                    self.assertFalse(set(names) & {"rms_width_s", "rms_range_width_s", "range_width"})
                    metadata = json.loads(str(data["metadata_json"]))
                    self.assertEqual(metadata['inversion_config'], config)
                    self.assertEqual(metadata['echo_identity']['size_bytes'], echo_path.stat().st_size)
                    self.assertNotIn("range_width_definition", metadata)
                    self.assertTrue(np.all(np.isfinite(data["uncertainty"])))
                    if mode == "cw":
                        self.assertNotIn("range_width_definition", metadata)
                        self.assertIn("rms_bandwidth_hz", names)
                        self.assertIn('centroid_hz', names)
                        self.assertFalse(any("fwhm" in name for name in names))
                    else:
                        diag = best["search_diagnostics"]
                        self.assertEqual(diag["processing_path"], "pulse_fallback" if mode == "fallback" else "cpi")
                        self.assertEqual(diag["baseline_model"], "shared_constant")
                        self.assertEqual(diag["aggregation"], "none")
                        self.assertEqual(best["significance_status"], "not_evaluated")
                        for item in summary.values():
                            self.assertEqual(item["significance_status"], "not_evaluated")
                        self.assertEqual(metadata["search_diagnostics"], diag)
                        for item in summary.values():
                            self.assertEqual(item["search_diagnostics"]["parameter_count"], 3)
                        energy_key = "energy_width_90_s" if mode == "fallback" else "energy_range_width_90_s"
                        self.assertIn(energy_key, names)
                        self.assertEqual(metadata["energy_width_gate"], "none")
                        self.assertEqual(metadata["energy_width_quantiles"], [.05, .95])
                        widths = data["value"][:, names.index(energy_key)]
                        self.assertTrue(np.any(np.isfinite(widths) & (widths > 0)))
                        if mode == "fallback":
                            self.assertTrue(np.isnan(widths[-1]))
                            self.assertTrue(np.isnan(data["value"][-1, names.index(energy_key)]))


if __name__ == "__main__":
    unittest.main()
