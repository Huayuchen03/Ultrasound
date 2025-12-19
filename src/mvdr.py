from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MVDRConfig:
    diag_load: float = 1e-2       # diagonal loading fraction
    time_win_samples: int = 4     # +/- window in samples to estimate covariance
    subarray_len: Optional[int] = None  # if set, use centered subarray of this length


def _sample_snapshot(
    rf: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    x_m: float,
    z_m: float,
    c0_m_s: float,
    time_offset_s: float = 0.0,
    active: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Sample delayed channel values for a single pixel (x,z).
    Returns:
        snapshot: (L,) delayed samples
        active_idx: (L,) indices into channels used
    '''
    N, _ = rf.shape
    if active is None:
        active = np.arange(N)

    dx = x_m - element_x_m[active]
    dist = np.sqrt(dx * dx + z_m * z_m)
    tau = (2.0 * dist) / float(c0_m_s) + float(time_offset_s)

    snap = np.array([np.interp(tau_k, t_s, rf[ch]) for tau_k, ch in zip(tau, active)], dtype=np.float32)
    return snap, active


def mvdr_pixel(
    rf: np.ndarray,
    t_s: np.ndarray,
    element_x_m: np.ndarray,
    x_m: float,
    z_m: float,
    c0_m_s: float,
    cfg: MVDRConfig,
) -> float:
    '''
    MVDR/Capon beamforming for one pixel using a crude covariance estimate from
    a small window of time offsets around the focal delay.

    This is a simplified implementation intended for a course project.
    '''
    N, _ = rf.shape

    # choose active subarray
    if cfg.subarray_len is None or cfg.subarray_len >= N:
        active = np.arange(N)
    else:
        # center subarray around the closest element to x_m
        center = int(np.argmin(np.abs(element_x_m - x_m)))
        L = int(cfg.subarray_len)
        start = max(0, center - L // 2)
        end = min(N, start + L)
        start = max(0, end - L)
        active = np.arange(start, end)

    # central snapshot
    x0, active = _sample_snapshot(rf, t_s, element_x_m, x_m, z_m, c0_m_s, 0.0, active=active)
    L = x0.shape[0]

    # covariance estimate using time-shifted snapshots
    M = int(cfg.time_win_samples)
    snaps = []
    dt = float(t_s[1] - t_s[0])
    for m in range(-M, M + 1):
        xm, _ = _sample_snapshot(rf, t_s, element_x_m, x_m, z_m, c0_m_s, time_offset_s=m * dt, active=active)
        snaps.append(xm.astype(np.float32))

    X = np.stack(snaps, axis=0)  # (K, L)
    R = (X.T @ X) / float(X.shape[0])  # (L, L) real covariance

    # diagonal loading
    tr = float(np.trace(R))
    R = R + (cfg.diag_load * tr / float(L) + 1e-12) * np.eye(L, dtype=np.float32)

    a = np.ones((L,), dtype=np.float32)  # steering vector after delay alignment

    # Solve R^{-1} a
    try:
        Rinva = np.linalg.solve(R, a)
    except np.linalg.LinAlgError:
        # fallback
        Rinva = np.linalg.lstsq(R, a, rcond=None)[0]

    denom = float(a.T @ Rinva) + 1e-12
    w = Rinva / denom  # (L,)

    y = float(w.T @ x0)
    return y
