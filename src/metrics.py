from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class CystROI:
    center_x_m: float
    center_z_m: float
    radius_m: float
    bg_inner_radius_m: float
    bg_outer_radius_m: float


def fwhm_lateral(
    bmode_db: np.ndarray,
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    target_x_m: float,
    target_z_m: float,
) -> float:
    '''
    Estimate lateral FWHM (-6 dB width) at the depth nearest target_z_m,
    around target_x_m.
    Returns width in meters.
    '''
    iz = int(np.argmin(np.abs(grid_z_m - target_z_m)))
    profile = bmode_db[iz, :]  # (Nx,)
    # normalize to peak at 0 dB
    profile = profile - np.max(profile)

    # find closest peak near target_x_m
    ix0 = int(np.argmin(np.abs(grid_x_m - target_x_m)))

    # search local region for peak
    win = 10
    left = max(0, ix0 - win)
    right = min(len(grid_x_m), ix0 + win + 1)
    ix_peak = left + int(np.argmax(profile[left:right]))

    # find -6 dB crossing points around peak
    thr = -6.0
    # left crossing
    ix_l = ix_peak
    while ix_l > 0 and profile[ix_l] > thr:
        ix_l -= 1
    # right crossing
    ix_r = ix_peak
    while ix_r < len(grid_x_m) - 1 and profile[ix_r] > thr:
        ix_r += 1

    width = float(grid_x_m[ix_r] - grid_x_m[ix_l])
    return max(width, 0.0)


def contrast_metrics(
    env: np.ndarray,
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    roi: CystROI,
) -> Tuple[float, float]:
    '''
    Compute CR and CNR from an envelope (linear scale).

    CR (dB): 20 log10(mu_bg / (mu_cyst + eps))
    CNR: |mu_bg - mu_cyst| / sqrt(sigma_bg^2 + sigma_cyst^2)

    env: (Nz, Nx) envelope (not dB)
    '''
    X, Z = np.meshgrid(grid_x_m, grid_z_m)
    dx = X - roi.center_x_m
    dz = Z - roi.center_z_m
    r = np.sqrt(dx * dx + dz * dz)

    cyst_mask = r <= roi.radius_m
    bg_mask = (r >= roi.bg_inner_radius_m) & (r <= roi.bg_outer_radius_m)

    cyst = env[cyst_mask]
    bg = env[bg_mask]

    eps = 1e-12
    mu_c = float(np.mean(cyst))
    mu_b = float(np.mean(bg))
    sig_c = float(np.std(cyst))
    sig_b = float(np.std(bg))

    cr_db = 20.0 * np.log10((mu_b + eps) / (mu_c + eps))
    cnr = abs(mu_b - mu_c) / (np.sqrt(sig_b * sig_b + sig_c * sig_c) + eps)
    return float(cr_db), float(cnr)
