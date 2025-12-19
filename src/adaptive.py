from __future__ import annotations

import numpy as np


def coherence_factor(snapshot: np.ndarray, eps: float = 1e-12) -> float:
    '''
    Classic coherence factor (CF):
        CF = |Σ x_i| / (Σ |x_i| + eps)
    Range ~ [0, 1]. Higher indicates coherent summation.

    snapshot: shape (N,)
    '''
    num = np.abs(np.sum(snapshot))
    den = np.sum(np.abs(snapshot)) + eps
    return float(num / den)


def apply_cf_weighting(
    snapshots: np.ndarray,
    das_values: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    '''
    Apply CF to DAS outputs.

    Args:
        snapshots: (P, N) delayed channel samples for each pixel
        das_values: (P,) DAS pixel values (weighted sum)
    Returns:
        cf_weighted: (P,)
    '''
    num = np.abs(np.sum(snapshots, axis=1))
    den = np.sum(np.abs(snapshots), axis=1) + eps
    cf = num / den
    return (cf * das_values).astype(np.float32)
