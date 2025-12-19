"""Sampling helper stub to avoid relative-import failures during demos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Optional


def _ensure_repo_on_path() -> None:
    if __package__:
        return
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()


def sample_scan_angles(num_angles: int, seed: int) -> list[float]:
    """Return deterministic pseudo-random steering angles (radians)."""

    if num_angles <= 0:
        raise ValueError("num_angles must be positive")

    rng = (seed * 1103515245 + 12345) % 2**31
    return [((rng >> i) % 41 - 20) * 0.001 for i in range(num_angles)]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_angles", type=int, default=5, help="Number of scan angles to sample")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic sampling")
    args = parser.parse_args(list(argv) if argv is not None else None)

    angles = sample_scan_angles(args.num_angles, args.seed)
    out_path = Path("data/demo_scan_angles.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"angles_rad": angles}, indent=2))
    print(f"Saved {len(angles)} angles to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
