from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Cyst:
    center_x_m: float
    center_z_m: float
    radius_m: float


def generate_speckle_scatterers(
    num_scatterers: int,
    x_range_m: Tuple[float, float],
    z_range_m: Tuple[float, float],
    cyst: Optional[Cyst] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Generate random point scatterers (speckle) with Rayleigh amplitudes.

    Args:
        num_scatterers: number of scatterers.
        x_range_m: (xmin, xmax).
        z_range_m: (zmin, zmax).
        cyst: optional cyst region where scatterers are suppressed.
        seed: RNG seed.

    Returns:
        (xs, zs, amps)
    '''
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_range_m[0], x_range_m[1], size=num_scatterers)
    zs = rng.uniform(z_range_m[0], z_range_m[1], size=num_scatterers)

    # Rayleigh-like amplitude (speckle)
    amps = rng.rayleigh(scale=1.0, size=num_scatterers).astype(np.float32)
    amps *= rng.choice([-1.0, 1.0], size=num_scatterers).astype(np.float32)  # allow sign

    if cyst is not None:
        dx = xs - cyst.center_x_m
        dz = zs - cyst.center_z_m
        inside = (dx * dx + dz * dz) <= (cyst.radius_m * cyst.radius_m)
        amps[inside] = 0.0

    return xs.astype(np.float32), zs.astype(np.float32), amps.astype(np.float32)


def add_point_targets(
    xs: np.ndarray,
    zs: np.ndarray,
    amps: np.ndarray,
    targets: Tuple[Tuple[float, float, float], ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Add bright point targets.

    targets: list of (x_m, z_m, amplitude)
    '''
    tx = np.array([t[0] for t in targets], dtype=np.float32)
    tz = np.array([t[1] for t in targets], dtype=np.float32)
    ta = np.array([t[2] for t in targets], dtype=np.float32)

    xs2 = np.concatenate([xs, tx], axis=0)
    zs2 = np.concatenate([zs, tz], axis=0)
    amps2 = np.concatenate([amps, ta], axis=0)
    return xs2, zs2, amps2
