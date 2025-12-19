"""Real-data beamforming stub that remains import-safe when run as a script."""
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

from src.beamform_plane_wave import (
    PlaneWaveBeamformConfig,
    PlaneWaveBeamformResult,
    plane_wave_das_cf,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file_path", type=Path, default=Path("data/example.uff"), help="Path to a UFF/HDF5 dataset")
    parser.add_argument("--out_dir", type=Path, default=Path("results_real"), help="Output directory for figures/metrics")
    parser.add_argument("--nx", type=int, default=128, help="Lateral pixels")
    parser.add_argument("--nz", type=int, default=256, help="Axial pixels")
    parser.add_argument("--x_min_m", type=float, default=-0.02, help="Minimum lateral coordinate")
    parser.add_argument("--x_max_m", type=float, default=0.02, help="Maximum lateral coordinate")
    parser.add_argument("--z_min_m", type=float, default=0.005, help="Minimum axial coordinate")
    parser.add_argument("--z_max_m", type=float, default=0.06, help="Maximum axial coordinate")
    parser.add_argument("--max_waves", type=int, default=11, help="Maximum plane-wave count")
    parser.add_argument("--dump_hdf5", action="store_true", help="Dump metadata to JSON for inspection")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    config = PlaneWaveBeamformConfig(
        nx=args.nx,
        nz=args.nz,
        x_min_m=args.x_min_m,
        x_max_m=args.x_max_m,
        z_min_m=args.z_min_m,
        z_max_m=args.z_max_m,
        max_waves=args.max_waves,
    )
    result: PlaneWaveBeamformResult = plane_wave_das_cf(
        file_path=args.file_path,
        config=config,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "real_data_meta.json"
    meta_path.write_text(result.to_json())
    print(f"Wrote placeholder beamforming summaries to {meta_path}")

    if args.dump_hdf5:
        dump_path = args.out_dir / "real_data_dump.json"
        dump_payload = {"file_path": str(args.file_path), "note": "No HDF5 parser bundled in stub."}
        dump_path.write_text(json.dumps(dump_payload, indent=2))
        print(f"Saved dataset inspection dump to {dump_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
