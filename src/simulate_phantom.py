"""Placeholder phantom generation script with import-safe execution."""
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


def generate_phantom(num_scatterers: int, seed: int) -> dict:
    rng = (seed * 1664525 + 1013904223) % 2**32
    scatterers = [
        {
            "x_m": ((rng >> i) % 2000 - 1000) / 1e5,
            "z_m": ((rng >> (i + 5)) % 2000) / 1e4,
            "amplitude": 1.0,
        }
        for i in range(max(num_scatterers, 1))
    ]
    return {"num_scatterers": num_scatterers, "seed": seed, "scatterers": scatterers}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_scatterers", type=int, default=3, help="Number of synthetic scatterers")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic scatterers")
    args = parser.parse_args(list(argv) if argv is not None else None)

    phantom = generate_phantom(args.num_scatterers, args.seed)
    out_path = Path("data/demo_phantom.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(phantom, indent=2))
    print(f"Saved placeholder phantom to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
