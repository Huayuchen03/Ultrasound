"""Baseline beamformer CLI with robust imports.

This lightweight stub keeps the module executable via either
``python -m src.run_baselines`` or ``python src/run_baselines.py`` without
hitting relative-import errors. Replace the placeholder logic with the actual
beamforming pipeline once dependencies (NumPy, TensorFlow, etc.) are
available in the environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional


def _ensure_repo_on_path() -> None:
    """Make sure ``src`` can be imported when run as a script.

    When executed as ``python src/run_baselines.py``, ``__package__`` is empty
    and relative imports fail. By adding the repository root to ``sys.path``
    we can rely on absolute ``src.*`` imports in both script and module
    execution modes.
    """

    if __package__:
        return

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.params import AcqParams, ArrayParams, ImageGrid


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline DAS/CF beamformers (placeholder implementation). "
            "Use -m to avoid relative-import issues in real executions."
        )
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Output directory for figures and metrics (placeholder).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Minimal smoke-test objects to prove imports succeeded.
    array = ArrayParams(num_elements=2, pitch_m=0.0003)
    acq = AcqParams(fs_hz=40e6)
    grid = ImageGrid(
        nx=2,
        nz=2,
        x_min_m=-0.001,
        x_max_m=0.001,
        z_min_m=0.005,
        z_max_m=0.010,
    )

    print("Baseline runner stub. Replace with real beamforming implementation.")
    print(f"out_dir: {args.out_dir} | seed: {args.seed}")
    print(f"Array: {array}")
    print(f"Acquisition: {acq}")
    print(f"Image grid: {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
