from __future__ import annotations

from typing import Tuple

import numpy as np


def delayed_snapshots_full_aperture(
    rf: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    c0_m_s: float,
) -> np.ndarray:
    '''
    Return delayed samples per pixel across all channels (full aperture).

    Output shape: (Nz, Nx, N)
    '''
    N, _ = rf.shape
    Nx = grid_x_m.shape[0]
    Nz = grid_z_m.shape[0]
    snaps = np.zeros((Nz, Nx, N), dtype=np.float32)

    for ix, x in enumerate(grid_x_m):
        dx = x - element_x_m  # (N,)
        for iz, z in enumerate(grid_z_m):
            dist = np.sqrt(dx * dx + z * z)  # (N,)
            tau = (2.0 * dist) / float(c0_m_s)
            vals = np.array([np.interp(tau_k, t_s, rf[ch]) for ch, tau_k in enumerate(tau)], dtype=np.float32)
            snaps[iz, ix, :] = vals
    return snaps


def das_from_snapshots(
    snaps: np.ndarray,
    apodization: str = "hanning",
) -> np.ndarray:
    '''
    Compute DAS output from snapshots.

    snaps: (Nz, Nx, N)
    returns: (Nz, Nx)
    '''
    Nz, Nx, N = snaps.shape
    if apodization.lower() == "hanning":
        w = np.hanning(N).astype(np.float32)
    elif apodization.lower() == "box":
        w = np.ones((N,), dtype=np.float32)
    else:
        raise ValueError(f"Unknown apodization: {apodization}")

    w = w / (np.sum(w) + 1e-12)
    y = np.tensordot(snaps, w, axes=([-1], [0]))  # (Nz, Nx)
    return y.astype(np.float32)
