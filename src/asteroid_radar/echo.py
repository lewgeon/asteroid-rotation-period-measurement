"""Continuous-wave echo simulation from a prepared triangle mesh."""

import numpy as np
import torch

from .dataset import EchoDataset, ObservationSchedule
from .mesh import load_mesh
from .motion import Spin, icrs_direction, unit


C = 299_792_458.0


def polynomial(coefficients, times):
    return np.polynomial.polynomial.polyval(times, coefficients)


def integrated_phase(coefficients_hz, times):
    coefficients_hz = np.asarray(coefficients_hz)
    integral = np.r_[0.0, coefficients_hz / np.arange(1, len(coefficients_hz) + 1)]
    return 2 * np.pi * polynomial(integral, times)


def rotational_doppler(positions, angular_velocity, tx_los, rx_los, wavelength):
    velocity = torch.linalg.cross(
        angular_velocity.expand_as(positions), positions, dim=1
    )
    projection = torch.tensor(
        unit(tx_los) + unit(rx_los), device=positions.device, dtype=positions.dtype
    )
    return velocity @ projection / wavelength


def simulate_echo(config):
    """Load ``model_path`` and generate one coherent CW echo dataset."""

    compute, target, radar = config["compute"], config["target"], config["radar"]
    mesh = load_mesh(
        config["model_path"],
        device=compute["device"],
        dtype=compute["dtype"],
    )
    spot = config["scattering_spot"]
    mesh = mesh.add_scattering_spot(
        spot["direction_body"], spot["radius_deg"], spot["strength"]
    )
    pole = target["spin_pole_icrs_deg"]
    spin = Spin(
        target["rotation_period_s"],
        icrs_direction(pole[0], pole[1]),
        np.deg2rad(target["initial_phase_deg"]),
    )
    observation = config["observation"]
    schedule = ObservationSchedule.continuous_wave(
        observation["start_utc"], observation["duration_s"], radar["sample_rate_hz"]
    )

    times = schedule.elapsed_s
    wavelength = C / radar["carrier_frequency_hz"]
    tx, rx = unit(radar["tx_los_icrs"]), unit(radar["rx_los_icrs"])
    projection = tx + rx
    rotation_bound = np.linalg.norm(projection) * spin.rate * mesh.radius_m / wavelength
    translation = polynomial(config["translation_doppler_hz"], times)
    modeled_bound = np.max(np.abs(translation)) + rotation_bound
    if modeled_bound >= 0.5 * radar["sample_rate_hz"]:
        raise ValueError(
            f"采样率会造成多普勒混叠：{modeled_bound:.2f} Hz >= Nyquist"
        )

    device, dtype = mesh.device, mesh.dtype
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    clean = torch.zeros(len(times), device=device, dtype=complex_dtype)
    tx_t = torch.tensor(tx, device=device, dtype=dtype)
    rx_t = torch.tensor(rx, device=device, dtype=dtype)
    projection_t = tx_t + rx_t
    face_scale = mesh.areas * mesh.scattering / mesh.areas.sum()
    translation_phase = torch.tensor(
        integrated_phase(config["translation_doppler_hz"], times),
        device=device,
        dtype=dtype,
    )
    p_tx, p_rx = config["scattering_power"]
    chunk_size = config.get("chunk_size", 2048)

    with torch.no_grad():
        for start in range(0, len(times), chunk_size):
            stop = min(start + chunk_size, len(times))
            t = torch.tensor(times[start:stop], device=device, dtype=dtype)
            positions = spin.rotate(mesh.centroids, t)
            normals = spin.rotate(mesh.normals, t)
            illumination = torch.clamp(normals @ -tx_t, min=0)
            reception = torch.clamp(normals @ -rx_t, min=0)
            amplitude = face_scale[None] * illumination**p_tx * reception**p_rx
            phase = 2 * np.pi * (positions @ projection_t) / wavelength
            clean[start:stop] = (
                amplitude.to(complex_dtype) * torch.exp(1j * phase)
            ).sum(1) * torch.exp(1j * translation_phase[start:stop])

    clean = clean.cpu().numpy()
    iq = clean.copy()
    snr_db = config["snr_db"]
    if snr_db is not None:
        power = np.mean(np.abs(clean) ** 2)
        if power == 0:
            raise ValueError("当前视线没有照亮任何面元")
        sigma = np.sqrt(power / 10 ** (snr_db / 10) / 2)
        rng = np.random.default_rng(config["seed"])
        iq += sigma * (
            rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
        )

    return EchoDataset(
        times,
        iq,
        clean,
        schedule.valid,
        schedule.coherence_id,
        translation,
        {
            "model_path": config["model_path"],
            "reference_time_utc": schedule.reference_time_utc,
            "sample_rate_hz": radar["sample_rate_hz"],
            "carrier_frequency_hz": radar["carrier_frequency_hz"],
            "wavelength_m": wavelength,
            "translation_coefficients_hz": config["translation_doppler_hz"],
            "tx_los_icrs": tx.tolist(),
            "rx_los_icrs": rx.tolist(),
            "snr_db": snr_db,
            "max_rotation_doppler_hz": rotation_bound,
            "face_count": len(mesh.faces),
            "compute_device": str(mesh.device),
            "compute_dtype": str(mesh.dtype).split(".")[-1],
            "truth_period_s": target["rotation_period_s"],
            "seed": config["seed"],
        },
    )
