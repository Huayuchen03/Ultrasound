from __future__ import annotations

import numpy as np


def gaussian_modulated_sine(
    fs_hz: float,
    f0_hz: float,
    cycles: int = 2,
    frac_bw: float = 0.6,
) -> np.ndarray:
    '''
    Create a simple bandpass ultrasound pulse: sin(2πf0 t) * exp(-t^2 / (2σ^2)).

    Args:
        fs_hz: Sampling frequency.
        f0_hz: Center frequency.
        cycles: Approximate number of cycles in the pulse.
        frac_bw: Fractional bandwidth (roughly controls σ). Higher -> shorter pulse.

    Returns:
        1D numpy array pulse normalized to unit peak amplitude.
    '''
    # pulse duration in seconds (approx cycles / f0)
    dur_s = cycles / float(f0_hz)
    # time support: cover ~ +/- dur_s
    t = np.arange(-int(dur_s * fs_hz), int(dur_s * fs_hz) + 1) / fs_hz

    # Gaussian std derived heuristically from fractional bandwidth
    # Smaller σ => wider bandwidth. This is not exact; intended for simulation.
    sigma = dur_s / max(frac_bw * 6.0, 1e-6)

    env = np.exp(-(t**2) / (2.0 * sigma**2 + 1e-12))
    pulse = np.sin(2.0 * np.pi * f0_hz * t) * env
    pulse = pulse / (np.max(np.abs(pulse)) + 1e-12)
    return pulse.astype(np.float32)
