"""Placeholder RF channel simulator that avoids relative-import errors."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional


def _ensure_repo_on_path() -> None:
    """Allow running as ``python src/simulate_rf.py`` from the repo root."""

    if __package__:
        return

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.pulse import gaussian_modulated_sine


def synthesize_rf(num_samples: int, center_freq_hz: float, fs_hz: float) -> Path:
    """Write a tiny placeholder RF file to disk."""

    pulse = gaussian_modulated_sine(num_samples=num_samples, center_freq_hz=center_freq_hz, fs_hz=fs_hz)
    out_path = Path("data/demo_rf.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{v:.6e}" for v in pulse))
    return out_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, default=128, help="Number of RF samples to synthesize")
    parser.add_argument("--center_freq_hz", type=float, default=5e6, help="Center frequency for the analytic pulse")
    parser.add_argument("--fs_hz", type=float, default=40e6, help="Sampling frequency")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = synthesize_rf(args.num_samples, args.center_freq_hz, args.fs_hz)
    print(f"Wrote placeholder RF samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
