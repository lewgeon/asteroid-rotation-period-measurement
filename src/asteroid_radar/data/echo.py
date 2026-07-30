"""The file-level seam between echo simulation and parameter inversion."""

from dataclasses import dataclass
import json

import numpy as np


@dataclass
class EchoDataset:
    """Complex radar samples plus the metadata needed by later experiments."""

    elapsed_s: np.ndarray
    iq: np.ndarray
    clean_iq: np.ndarray
    valid: np.ndarray
    coherence_id: np.ndarray
    translation_doppler_hz: np.ndarray
    metadata: dict


def save_echo(path, echo):
    np.savez(
        path,
        elapsed_s=echo.elapsed_s,
        iq=echo.iq,
        clean_iq=echo.clean_iq,
        valid=echo.valid,
        coherence_id=echo.coherence_id,
        translation_doppler_hz=echo.translation_doppler_hz,
        metadata_json=json.dumps(echo.metadata),
    )


def load_echo(path):
    data = np.load(path, allow_pickle=False)
    return EchoDataset(
        elapsed_s=data["elapsed_s"],
        iq=data["iq"],
        clean_iq=data["clean_iq"],
        valid=data["valid"],
        coherence_id=data["coherence_id"],
        translation_doppler_hz=data["translation_doppler_hz"],
        metadata=json.loads(str(data["metadata_json"])),
    )
