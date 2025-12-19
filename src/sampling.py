from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def sample_snapshot(
    rf: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    x_m: float,
    z_m: float,
    c0_m_s: float,
    active: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Sample delayed channel values for a single pixel (x,z) using linear interpolation.

    Returns:
        snapshot: (L,) delayed samples
        active_idx: (L,) channel indices used
    '''
    N, _ = rf.shape
    if active is None:
        active = np.arange(N)

    dx = x_m - element_x_m[active]
    dist = np.sqrt(dx * dx + z_m * z_m)
    tau = (2.0 * dist) / float(c0_m_s)

    snap = np.array([np.interp(tau_k, t_s, rf[ch]) for tau_k, ch in zip(tau, active)], dtype=np.float32)
    return snap, active
