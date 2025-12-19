from __future__ import annotations

"""Run a *real-data* beamforming demo on an HDF5/UFF acquisition.

This script is meant to satisfy the common rubric requirement:
    "use at least one publicly available real dataset in addition to simulation"

The repository does **not** bundle any real RF datasets. Instead, you download
one public dataset (e.g., an USTB/PICMUS UFF file) and point this script to it.

Example (PICMUS / USTB UFF):

    python -m src.run_real_uff \
        --file_path data/real/PICMUS_carotid_long.uff \
        --out_dir results_real \
        --nx 128 --nz 256 \
        --x_min_m -0.020 --x_max_m 0.020 \
        --z_min_m 0.005  --z_max_m 0.060 \
        --max_waves 11

If the loader cannot infer probe geometry or sampling frequency, pass
--pitch_m and/or --fs_hz explicitly.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .beamform_plane_wave import PlaneWaveBeamformConfig, plane_wave_das_cf
from .real_data_io import load_hdf5_channel_data_guess, summarize_hdf5


def rf_to_bmode_and_env(rf_img: np.ndarray, dyn_db: float = 60.0):
    analytic = hilbert(rf_img, axis=0)
    env = np.abs(analytic).astype(np.float32)
    env = env / (np.max(env) + 1e-12)
    bmode = 20.0 * np.log10(env + 1e-12)
    bmode = np.maximum(bmode, -dyn_db)
    return bmode.astype(np.float32), env.astype(np.float32)


def save_bmode_png(bmode_db: np.ndarray, grid_x_m: np.ndarray, grid_z_m: np.ndarray, out_path: Path, title: str):
    plt.figure()
    plt.imshow(
        bmode_db,
        extent=[grid_x_m[0]*1e3, grid_x_m[-1]*1e3, grid_z_m[-1]*1e3, grid_z_m[0]*1e3],
        aspect="auto",
        cmap="gray",
        vmin=-60,
        vmax=0,
    )
    plt.xlabel("Lateral x (mm)")
    plt.ylabel("Depth z (mm)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_grid(x_min_m: float, x_max_m: float, nx: int, z_min_m: float, z_max_m: float, nz: int):
    grid_x = np.linspace(x_min_m, x_max_m, int(nx), dtype=np.float64)
    grid_z = np.linspace(z_min_m, z_max_m, int(nz), dtype=np.float64)
    return grid_x, grid_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to a public HDF5/UFF file.")
    parser.add_argument("--out_dir", type=str, default="results_real")
    parser.add_argument("--rf_dataset_path", type=str, default=None, help="Optional explicit HDF5 dataset path for RF.")
    parser.add_argument("--fs_hz", type=float, default=None, help="Override sampling frequency (Hz).")
    parser.add_argument("--pitch_m", type=float, default=None, help="Assume uniform pitch if geometry isn't found.")
    parser.add_argument("--c0", type=float, default=1540.0)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--x_min_m", type=float, default=-0.020)
    parser.add_argument("--x_max_m", type=float, default=0.020)
    parser.add_argument("--z_min_m", type=float, default=0.005)
    parser.add_argument("--z_max_m", type=float, default=0.060)
    parser.add_argument("--max_waves", type=int, default=11, help="Use at most this many plane-wave angles.")
    parser.add_argument("--max_elements", type=int, default=None, help="Optionally subsample receive elements for speed.")
    parser.add_argument("--apodization", type=str, default="hanning")
    parser.add_argument("--f_number", type=float, default=1.5)
    parser.add_argument("--compound", type=str, default="mean")
    parser.add_argument("--dump_hdf5", action="store_true", help="Print a summary of HDF5 datasets then exit.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_hdf5:
        summary = summarize_hdf5(args.file_path)
        (out_dir / "hdf5_summary.txt").write_text("\n".join(summary))
        print("\n".join(summary[:50]))
        print(f"Wrote full summary to: {out_dir/'hdf5_summary.txt'}")
        return

    loaded = load_hdf5_channel_data_guess(
        file_path=args.file_path,
        fs_hz=args.fs_hz,
        pitch_m=args.pitch_m,
        angles_rad=None,
        rf_dataset_path=args.rf_dataset_path,
    )

    rf_wnt = loaded.rf_wnt
    t_s = loaded.t_s
    element_x_m = loaded.element_x_m
    angles = loaded.angles_rad

    # Optional subsampling for speed
    if args.max_elements is not None and int(args.max_elements) < rf_wnt.shape[1]:
        m = int(args.max_elements)
        # Take a centered sub-aperture
        N = rf_wnt.shape[1]
        start = (N - m) // 2
        rf_wnt = rf_wnt[:, start:start+m, :]
        element_x_m = element_x_m[start:start+m]

    if int(args.max_waves) < rf_wnt.shape[0]:
        rf_wnt = rf_wnt[: int(args.max_waves), :, :]
        angles = angles[: int(args.max_waves)]

    grid_x, grid_z = make_grid(args.x_min_m, args.x_max_m, args.nx, args.z_min_m, args.z_max_m, args.nz)

    cfg = PlaneWaveBeamformConfig(
        apodization=args.apodization,
        f_number=float(args.f_number) if args.f_number is not None else None,
        compound=args.compound,
    )

    # --- Beamform ---
    t0 = time.perf_counter()
    das_rf, cf_rf = plane_wave_das_cf(
        rf_wnt=rf_wnt,
        t_s=t_s,
        element_x_m=element_x_m,
        grid_x_m=grid_x,
        grid_z_m=grid_z,
        c0_m_s=float(args.c0),
        angles_rad=angles,
        cfg=cfg,
    )
    bf_time = time.perf_counter() - t0

    das_bmode, _ = rf_to_bmode_and_env(das_rf)
    cf_bmode, _ = rf_to_bmode_and_env(cf_rf)

    save_bmode_png(das_bmode, grid_x, grid_z, out_dir / "real_bmode_das.png", f"Real-data DAS (t={bf_time:.2f}s)")
    save_bmode_png(cf_bmode, grid_x, grid_z, out_dir / "real_bmode_cf.png", f"Real-data CF (t={bf_time:.2f}s)")

    # Save a small metadata bundle
    (out_dir / "real_data_meta.json").write_text(json.dumps(loaded.meta, indent=2))

    df = pd.DataFrame(
        [
            {
                "method": "DAS_PW",
                "bf_time_s": bf_time,
                "W": int(rf_wnt.shape[0]),
                "N": int(rf_wnt.shape[1]),
                "T": int(rf_wnt.shape[2]),
                "nx": int(args.nx),
                "nz": int(args.nz),
            },
            {
                "method": "CF_PW",
                "bf_time_s": bf_time,
                "W": int(rf_wnt.shape[0]),
                "N": int(rf_wnt.shape[1]),
                "T": int(rf_wnt.shape[2]),
                "nx": int(args.nx),
                "nz": int(args.nz),
            },
        ]
    )
    df.to_csv(out_dir / "real_metrics.csv", index=False)
    print(df)

    print("\nMetadata (key points):")
    for k in ["rf_dataset_path", "rf_dataset_shape", "fs_hz", "element_x_path", "angles_source", "angles_path"]:
        if k in loaded.meta:
            print(f"  {k}: {loaded.meta[k]}")


if __name__ == "__main__":
    main()
