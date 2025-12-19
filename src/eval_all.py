"""Evaluate DAS/CF/NN pipelines with import-safe execution.

This stub focuses on preventing ``ImportError: attempted relative import with no
known parent package`` when the file is run directly. Fill in the actual
metrics/plotting logic once project dependencies are installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional


def _ensure_repo_on_path() -> None:
    """Allow absolute ``src`` imports when executed as a script."""

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
            "Evaluate DAS/CF/NN beamformers (placeholder). Use module execution "
            "to avoid relative-import issues."
        )
    )
    parser.add_argument(
        "--model_dir",
        default="models/cf_teacher",
        help="Directory containing trained model checkpoints (placeholder).",
    )
    parser.add_argument(
        "--out_dir",
        default="results_nn",
        help="Directory to write evaluation outputs (placeholder).",
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

    print("Evaluation stub. Replace with full pipeline when ready.")
    print(f"model_dir: {args.model_dir} | out_dir: {args.out_dir} | seed: {args.seed}")
    print(f"Array: {array}")
    print(f"Acquisition: {acq}")
    print(f"Image grid: {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
