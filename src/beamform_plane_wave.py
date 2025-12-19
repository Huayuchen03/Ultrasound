"""Lightweight plane-wave beamforming placeholders for real-data demos."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Tuple


def _ensure_repo_on_path() -> None:
    """Allow execution as ``python src/beamform_plane_wave.py``."""

    if __package__:
        return

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.params import AcqParams, ArrayParams, ImageGrid


@dataclass
class PlaneWaveBeamformConfig:
    """Minimal configuration for placeholder DAS/CF plane-wave processing."""

    nx: int = 128
    nz: int = 256
    x_min_m: float = -0.02
    x_max_m: float = 0.02
    z_min_m: float = 0.005
    z_max_m: float = 0.06
    max_waves: int = 11


@dataclass
class PlaneWaveBeamformResult:
    """Simple container for demo outputs."""

    file_path: str
    array: ArrayParams
    acquisition: AcqParams
    grid: ImageGrid
    config: PlaneWaveBeamformConfig
    das_summary: str
    cf_summary: str

    def to_json(self) -> str:
        payload = {
            "file_path": self.file_path,
            "array": asdict(self.array),
            "acquisition": asdict(self.acquisition),
            "grid": asdict(self.grid),
            "config": asdict(self.config),
            "das_summary": self.das_summary,
            "cf_summary": self.cf_summary,
        }
        return json.dumps(payload, indent=2)


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
DEFAULT_CONFIG = PlaneWaveBeamformConfig()


def plane_wave_das_cf(
    file_path: Path,
    array: ArrayParams = DEFAULT_ARRAY,
    acquisition: AcqParams = DEFAULT_ACQ,
    grid: ImageGrid = DEFAULT_GRID,
    config: PlaneWaveBeamformConfig = DEFAULT_CONFIG,
) -> PlaneWaveBeamformResult:
    """Return placeholder summaries for DAS/CF processing."""

    file_path = Path(file_path)
    das_summary = (
        "DAS placeholder output. Provide an HDF5/UFF file and real beamforming "
        "code to replace this stub."
    )
    cf_summary = (
        "CF placeholder output. Implement adaptive weighting once dependencies "
        "(NumPy/SciPy/TensorFlow) are available."
    )
    return PlaneWaveBeamformResult(
        file_path=str(file_path),
        array=array,
        acquisition=acquisition,
        grid=grid,
        config=config,
        das_summary=das_summary,
        cf_summary=cf_summary,
    )


__all__: Tuple[str, ...] = (
    "PlaneWaveBeamformConfig",
    "PlaneWaveBeamformResult",
    "plane_wave_das_cf",
)
