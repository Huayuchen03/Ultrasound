from __future__ import annotations

"""Plane-wave (PW) receive beamforming utilities.

This project primarily uses a simplified *monostatic* two-way delay model for
simulation and classical baselines (see :mod:`simulate_rf` and :mod:`beamform`).

For *real public datasets* recorded using coherent plane-wave compounding, the
delay model differs: transmit is a plane wave at angle \theta and receive is
element-wise propagation back to the array.

For a pixel (x, z) and receive element at x_e, the total time-of-flight is

    t_total(\theta) = t_tx(\theta) + t_rx

    t_tx(\theta) = (x sin\theta + z cos\theta) / c
    t_rx          = sqrt((x-x_e)^2 + z^2) / c

This module implements a minimal plane-wave DAS and CF-weighted DAS, intended
for course-project style experiments on open datasets.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlaneWaveBeamformConfig:
    """Configuration for plane-wave beamforming."""

    apodization: str = "hanning"  # "hanning" or "box"
    f_number: Optional[float] = 1.5  # dynamic receive aperture (None disables)
    compound: str = "mean"  # "sum" or "mean"
    eps: float = 1e-12


def _apodization_weights(n: int, kind: str) -> np.ndarray:
    kind = kind.lower().strip()
    if kind in {"hann", "hanning"}:
        w = np.hanning(n).astype(np.float32)
    elif kind in {"box", "rect", "rectangular", "uniform"}:
        w = np.ones((n,), dtype=np.float32)
    else:
        raise ValueError(f"Unknown apodization: {kind}")

    w = w / (np.sum(w) + 1e-12)
    return w


def plane_wave_das_cf(
    rf_wnt: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    c0_m_s: float,
    angles_rad: np.ndarray,
    cfg: PlaneWaveBeamformConfig = PlaneWaveBeamformConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Beamform plane-wave data using DAS and CF-weighted DAS.

    Parameters
    ----------
    rf_wnt:
        RF data as a float array of shape (W, N, T), where
        W = number of transmit plane waves (angles),
        N = number of receive elements,
        T = number of time samples.
    t_s:
        Time axis in seconds, shape (T,).
    element_x_m:
        Lateral element positions (meters), shape (N,).
    grid_x_m, grid_z_m:
        Lateral and depth axes of the reconstruction grid.
    c0_m_s:
        Assumed speed of sound.
    angles_rad:
        Plane-wave angles in radians, shape (W,).
    cfg:
        Beamforming configuration.

    Returns
    -------
    (das_rf, cf_rf):
        Two RF images of shape (Nz, Nx).

    Notes
    -----
    * This is a reference implementation intended for small grids. It uses
      Python loops and np.interp per channel.
    * For best quality on real datasets, compounding multiple angles is
      recommended, but can be slow in pure Python.
    """

    rf_wnt = np.asarray(rf_wnt)
    t_s = np.asarray(t_s).astype(np.float64)
    element_x_m = np.asarray(element_x_m).astype(np.float64)
    grid_x_m = np.asarray(grid_x_m).astype(np.float64)
    grid_z_m = np.asarray(grid_z_m).astype(np.float64)
    angles_rad = np.asarray(angles_rad).astype(np.float64)

    if rf_wnt.ndim != 3:
        raise ValueError(f"rf_wnt must have shape (W,N,T). Got {rf_wnt.shape}")
    W, N, T = rf_wnt.shape
    if t_s.shape[0] != T:
        raise ValueError(f"t_s length must match T={T}. Got {t_s.shape}")
    if element_x_m.shape[0] != N:
        raise ValueError(f"element_x_m length must match N={N}. Got {element_x_m.shape}")
    if angles_rad.shape[0] != W:
        raise ValueError(f"angles_rad length must match W={W}. Got {angles_rad.shape}")

    Nx = int(grid_x_m.shape[0])
    Nz = int(grid_z_m.shape[0])

    # Full-aperture apodization; sub-aperture uses a cropped and renormalized version.
    w_full = _apodization_weights(N, cfg.apodization)

    das = np.zeros((Nz, Nx), dtype=np.float32)
    cf = np.zeros((Nz, Nx), dtype=np.float32)

    c0 = float(c0_m_s)

    # Main loops
    for ix, x in enumerate(grid_x_m):
        dx_full = x - element_x_m  # (N,)
        for iz, z in enumerate(grid_z_m):
            # Dynamic aperture selection
            if cfg.f_number is None:
                active = np.arange(N, dtype=np.int32)
            else:
                half_width = float(z) / (2.0 * float(cfg.f_number) + 1e-12)
                active = np.where(np.abs(dx_full) <= half_width)[0].astype(np.int32)
                if active.size < 8:
                    active = np.arange(N, dtype=np.int32)

            w = w_full[active]
            w = w / (np.sum(w) + 1e-12)
            dx = x - element_x_m[active]
            t_rx = np.sqrt(dx * dx + z * z) / c0  # (Na,)

            y_das_sum = 0.0
            y_cf_sum = 0.0
            for iw, theta in enumerate(angles_rad):
                # Plane-wave transmit time to pixel
                t_tx = (x * np.sin(theta) + z * np.cos(theta)) / c0
                tau = t_tx + t_rx

                # Sample each active element at its delay
                # (np.interp is 1D; loop over channels)
                vals = np.array(
                    [np.interp(float(tau_k), t_s, rf_wnt[iw, int(ch)]) for tau_k, ch in zip(tau, active)],
                    dtype=np.float32,
                )

                y = float(np.sum(w * vals))
                y_das_sum += y
                cf_k = float(np.abs(np.sum(vals)) / (np.sum(np.abs(vals)) + cfg.eps))
                y_cf_sum += cf_k * y

            if cfg.compound.lower() == "mean":
                y_das_sum /= float(W)
                y_cf_sum /= float(W)
            elif cfg.compound.lower() != "sum":
                raise ValueError(f"Unknown compounding mode: {cfg.compound}")

            das[iz, ix] = np.float32(y_das_sum)
            cf[iz, ix] = np.float32(y_cf_sum)

    return das, cf
