"""Small demo that exercises the parameter containers.

Running this file previously failed with a relative-import error when launched
outside of a package. It now uses an absolute import (``from src.params``) so it
can be executed either as ``python -m src.run_params_demo`` or
``python src/run_params_demo.py`` from the repository root.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from src.params import AcqParams, ArrayParams, ImageGrid


def build_example_config() -> Dict[str, Any]:
    """Return a representative acquisition + grid configuration."""

    array = ArrayParams(num_elements=64, pitch_m=0.0003)
    acq = AcqParams(fs_hz=40e6, center_freq_hz=5e6)
    grid = ImageGrid(
        nx=128,
        nz=256,
        x_min_m=-0.02,
        x_max_m=0.02,
        z_min_m=0.005,
        z_max_m=0.06,
    )
    return {
        "array": asdict(array),
        "acquisition": asdict(acq),
        "grid": {
            **asdict(grid),
            "dx": grid.dx,
            "dz": grid.dz,
        },
    }


def save_config(config: Dict[str, Any], out_path: Path) -> None:
    """Serialize a configuration dictionary to JSON."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    config = build_example_config()
    save_path = Path("results/params_demo.json")
    save_config(config, save_path)
    print(f"Saved example configuration to {save_path}")
