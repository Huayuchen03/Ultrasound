from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class BeamformConfig:
    f_number: Optional[float] = 1.5  # None -> full aperture
    apodization: str = "hanning"     # "hanning" or "box"
    dynamic_range_db: float = 60.0


def _apodization_weights(n: int, kind: str) -> np.ndarray:
    if kind.lower() == "hanning":
        return np.hanning(n).astype(np.float32)
    if kind.lower() == "box":
        return np.ones((n,), dtype=np.float32)
    raise ValueError(f"Unknown apodization: {kind}")


def das_beamform(
    rf: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    c0_m_s: float,
    cfg: BeamformConfig,
) -> np.ndarray:
    '''
    Delay-and-Sum (DAS) receive beamforming for a monostatic pulse-echo model.

    Args:
        rf: (N, T) RF channels
        t_s: (T,) time axis
        element_x_m: (N,) element x positions
        grid_x_m: (Nx,) lateral positions
        grid_z_m: (Nz,) depth positions
        c0_m_s: speed of sound
        cfg: beamforming config

    Returns:
        bf_rf: (Nz, Nx) beamformed RF (before envelope)
    '''
    N, T = rf.shape
    Nx = grid_x_m.shape[0]
    Nz = grid_z_m.shape[0]

    bf = np.zeros((Nz, Nx), dtype=np.float32)

    # Precompute full-aperture weights (later sub-select if dynamic aperture)
    w_full = _apodization_weights(N, cfg.apodization)

    # For each pixel, compute 2-way delay and sample each channel
    for ix, x in enumerate(grid_x_m):
        for iz, z in enumerate(grid_z_m):
            # dynamic receive aperture via F-number
            if cfg.f_number is None:
                active = np.arange(N)
            else:
                half_width = z / (2.0 * float(cfg.f_number) + 1e-12)
                active = np.where(np.abs(element_x_m - x) <= half_width)[0]
                if active.size < 4:  # ensure minimal aperture
                    active = np.arange(N)

            w = w_full[active]
            if w.sum() > 0:
                w = w / (w.sum() + 1e-12)

            # compute delays and sample
            dx = x - element_x_m[active]
            dist = np.sqrt(dx * dx + z * z)
            tau = (2.0 * dist) / float(c0_m_s)

            # Linear interpolation of rf at time tau
            # np.interp expects increasing x-axis
            vals = np.array([np.interp(tau_k, t_s, rf[ch]) for tau_k, ch in zip(tau, active)], dtype=np.float32)
            bf[iz, ix] = float(np.sum(w * vals))

    return bf


def envelope_and_log_compress(
    bf_rf: np.ndarray,
    dynamic_range_db: float = 60.0,
) -> np.ndarray:
    '''
    Envelope detection via Hilbert transform + log compression.
    Input expected as (Nz, Nx).
    Returns log-compressed image in dB with max at 0 dB.
    '''
    analytic = hilbert(bf_rf, axis=0)
    env = np.abs(analytic).astype(np.float32)
    env = env / (np.max(env) + 1e-12)
    bmode = 20.0 * np.log10(env + 1e-12)
    bmode = np.maximum(bmode, -dynamic_range_db)
    return bmode.astype(np.float32)
