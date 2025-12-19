from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from .pulse import gaussian_modulated_sine


def simulate_rf_channels(
    element_x_m: np.ndarray,
    xs_m: np.ndarray,
    zs_m: np.ndarray,
    amps: np.ndarray,
    c0_m_s: float,
    fs_hz: float,
    f0_hz: float,
    pulse_cycles: int = 2,
    t_max_s: float = 90e-6,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Simulate RF receive channels for a simple 2D pulse-echo model:
        s_e(t) = Σ_i a_i * p(t - 2*dist(e,i)/c0) + noise

    Args:
        element_x_m: shape (N,)
        xs_m, zs_m, amps: scatterers (S,)
        c0_m_s: speed of sound
        fs_hz: sampling rate
        f0_hz: center frequency
        pulse_cycles: pulse length
        t_max_s: recording duration
        noise_std: additive noise std relative to pulse peak
        seed: RNG seed

    Returns:
        rf: shape (N, T)
        t:  shape (T,)
    '''
    rng = np.random.default_rng(seed)
    n_el = int(element_x_m.shape[0])
    t = np.arange(int(t_max_s * fs_hz), dtype=np.float32) / np.float32(fs_hz)
    T = t.shape[0]

    pulse = gaussian_modulated_sine(fs_hz=fs_hz, f0_hz=f0_hz, cycles=pulse_cycles)
    Lp = pulse.shape[0]

    rf = np.zeros((n_el, T), dtype=np.float32)

    # Vectorized distance computation per element in a loop (scatterers can be large).
    for e in range(n_el):
        dx = xs_m - element_x_m[e]
        dist = np.sqrt(dx * dx + zs_m * zs_m)  # meters
        tau = (2.0 * dist) / np.float32(c0_m_s)  # seconds

        # Add each scatterer contribution by pulse insertion
        idx_float = tau * np.float32(fs_hz)
        idx0 = np.floor(idx_float).astype(np.int32)

        for i in range(xs_m.shape[0]):
            k = idx0[i]
            if k < 0 or k >= T:
                continue
            # insert pulse with boundary checks
            start = k - Lp // 2
            end = start + Lp
            ps = 0
            pe = Lp
            if start < 0:
                ps = -start
                start = 0
            if end > T:
                pe = Lp - (end - T)
                end = T
            if pe > ps:
                rf[e, start:end] += amps[i] * pulse[ps:pe]

    # Add white Gaussian noise
    rf += (noise_std * rng.standard_normal(size=rf.shape)).astype(np.float32)
    return rf, t
