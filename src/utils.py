from __future__ import annotations

from typing import Tuple

import numpy as np


def make_grid(x_min_m: float, x_max_m: float, nx: int, z_min_m: float, z_max_m: float, nz: int) -> Tuple[np.ndarray, np.ndarray]:
    grid_x = np.linspace(x_min_m, x_max_m, nx, dtype=np.float32)
    grid_z = np.linspace(z_min_m, z_max_m, nz, dtype=np.float32)
    return grid_x, grid_z


def bmode_to_env(bmode_db: np.ndarray) -> np.ndarray:
    '''
    Convert a normalized b-mode dB image (max=0 dB) back to linear envelope
    up to an unknown scale: env ~ 10^(bmode/20).
    '''
    return (10.0 ** (bmode_db / 20.0)).astype(np.float32)
