"""Minimal dataset builder with import-robust parameter loading.

This script keeps the original README interface but avoids relative-import
failures when run directly (``python src/build_dataset.py ...``) by falling
back to an absolute import path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "numpy is required for building datasets; install it with `pip install numpy`."
    ) from exc


def _import_params():
    """Import parameter dataclasses even if executed as a script.

    When ``__package__`` is unset (direct ``python src/build_dataset.py``),
    we temporarily add the repository root to ``sys.path`` so that the
    absolute ``src.params`` import succeeds.
    """

    try:
        from .params import ArrayParams, AcqParams, ImageGrid
        return ArrayParams, AcqParams, ImageGrid
    except ImportError:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.params import ArrayParams, AcqParams, ImageGrid  # type: ignore
        return ArrayParams, AcqParams, ImageGrid


ArrayParams, AcqParams, ImageGrid = _import_params()

DEFAULT_ARRAY = ArrayParams(num_elements=64, pitch_m=0.0003)
DEFAULT_ACQ = AcqParams(fs_hz=40e6)
DEFAULT_GRID = ImageGrid(
    nx=128,
    nz=256,
    x_min_m=-0.02,
    x_max_m=0.02,
    z_min_m=0.005,
    z_max_m=0.06,
)


def _dummy_teacher_weights(num_samples: int, array: ArrayParams, rng: np.random.Generator) -> np.ndarray:
    """Create simple teacher weights that emphasize the center aperture."""

    taper = np.hanning(array.num_elements)
    weights = rng.normal(loc=taper, scale=0.05, size=(num_samples, array.num_elements))
    weights = np.clip(weights, 0.0, None)
    # Normalize per sample
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    return (weights / weights_sum).astype(np.float32)


def _dummy_snapshots(num_samples: int, array: ArrayParams, rng: np.random.Generator) -> np.ndarray:
    """Return synthetic channel snapshots shaped (num_samples, num_elements)."""

    return rng.standard_normal(size=(num_samples, array.num_elements)).astype(np.float32)


def _dummy_metadata(array: ArrayParams, acq: AcqParams, grid: ImageGrid) -> dict:
    """Small metadata block to help downstream scripts."""

    return {
        "array": array.__dict__,
        "acq": acq.__dict__,
        "grid": grid.__dict__,
    }


def build_dataset(
    out_npz: Path,
    num_samples: int,
    teacher: str,
    seed: int,
    array: ArrayParams = DEFAULT_ARRAY,
    acq: AcqParams = DEFAULT_ACQ,
    grid: ImageGrid = DEFAULT_GRID,
) -> None:
    """Write a lightweight synthetic dataset for demos and tests."""

    rng = np.random.default_rng(seed)
    snapshots = _dummy_snapshots(num_samples, array, rng)
    weights = _dummy_teacher_weights(num_samples, array, rng)
    metadata = _dummy_metadata(array, acq, grid)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, snapshots=snapshots, teacher=weights, metadata=metadata)
    print(f"Saved {num_samples} samples to {out_npz} with teacher='{teacher}'.")


def parse_args(argv: Tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_npz", required=True, type=Path, help="Destination .npz path")
    parser.add_argument(
        "--teacher",
        choices=["cf", "mvdr"],
        default="cf",
        help="Teacher label type (placeholder for demo dataset)",
    )
    parser.add_argument("--num_samples", type=int, default=1024, help="Number of synthetic snapshots")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: Tuple[str, ...] | None = None) -> None:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    build_dataset(
        out_npz=args.out_npz,
        num_samples=args.num_samples,
        teacher=args.teacher,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
