"""Simple analytic pulse utilities used by simulation stubs."""
from __future__ import annotations

import math
from typing import List


def gaussian_modulated_sine(
    num_samples: int = 256,
    center_freq_hz: float = 5e6,
    fs_hz: float = 40e6,
    bandwidth: float = 0.6,
) -> List[float]:
    """Generate a tiny analytic pulse for demos without NumPy."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if fs_hz <= 0 or center_freq_hz <= 0:
        raise ValueError("Frequencies must be positive")

    dt = 1.0 / fs_hz
    sigma = bandwidth / center_freq_hz
    samples = []
    for n in range(num_samples):
        t = (n - num_samples / 2) * dt
        envelope = math.exp(-(t ** 2) / (2 * sigma ** 2))
        carrier = math.sin(2 * math.pi * center_freq_hz * t)
        samples.append(envelope * carrier)
    return samples


__all__ = ["gaussian_modulated_sine"]
