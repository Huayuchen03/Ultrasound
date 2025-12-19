from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from .params import ArrayParams, AcqParams, ImageGrid
from .simulate_phantom import Cyst, generate_speckle_scatterers, add_point_targets
from .simulate_rf import simulate_rf_channels
from .snapshots import delayed_snapshots_full_aperture, das_from_snapshots
from .adaptive import apply_cf_weighting
from .metrics import fwhm_lateral, CystROI, contrast_metrics
from .utils import make_grid


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nz", type=int, default=220)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    array = ArrayParams(num_elements=32, pitch_m=0.0003)
    acq = AcqParams(fs_hz=20e6, f0_hz=5e6, pulse_cycles=2, noise_std=0.01)
    grid = ImageGrid(nx=args.nx, nz=args.nz, x_min_m=-0.015, x_max_m=0.015, z_min_m=0.005, z_max_m=0.060)

    element_x = np.array(array.element_positions(), dtype=np.float32)

    # Phantom: speckle + cyst + one point target (for resolution)
    cyst = Cyst(center_x_m=0.0, center_z_m=0.035, radius_m=0.0045)
    xs, zs, amps = generate_speckle_scatterers(
        num_scatterers=2500,
        x_range_m=(grid.x_min_m, grid.x_max_m),
        z_range_m=(grid.z_min_m, grid.z_max_m),
        cyst=cyst,
        seed=args.seed,
    )
    xs, zs, amps = add_point_targets(xs, zs, amps, targets=((0.0, 0.030, 12.0),))

    rf, t = simulate_rf_channels(
        element_x_m=element_x,
        xs_m=xs,
        zs_m=zs,
        amps=amps,
        c0_m_s=acq.c0_m_s,
        fs_hz=acq.fs_hz,
        f0_hz=acq.f0_hz,
        pulse_cycles=acq.pulse_cycles,
        t_max_s=90e-6,
        noise_std=acq.noise_std,
        seed=args.seed,
    )

    grid_x, grid_z = make_grid(grid.x_min_m, grid.x_max_m, grid.nx, grid.z_min_m, grid.z_max_m, grid.nz)

    # --- DAS (via snapshots) ---
    t0 = time.perf_counter()
    snaps = delayed_snapshots_full_aperture(rf, t, element_x, grid_x, grid_z, acq.c0_m_s)  # (Nz,Nx,N)
    das_rf = das_from_snapshots(snaps, apodization="hanning")
    das_time = time.perf_counter() - t0

    das_analytic = hilbert(das_rf, axis=0)
    das_env = np.abs(das_analytic).astype(np.float32)
    das_env /= (np.max(das_env) + 1e-12)
    das_bmode = 20.0 * np.log10(das_env + 1e-12)
    das_bmode = np.maximum(das_bmode, -60.0)

    save_bmode_png(das_bmode, grid_x, grid_z, out_dir/"bmode_das.png", f"DAS (time={das_time:.2f}s)")

    # --- CF-weighted DAS (adaptive) ---
    t1 = time.perf_counter()
    flat_snaps = snaps.reshape(-1, snaps.shape[-1])  # (P,N)
    flat_das = das_rf.reshape(-1)                    # (P,)
    cf_rf = apply_cf_weighting(flat_snaps, flat_das).reshape(das_rf.shape)
    cf_time = time.perf_counter() - t1

    cf_analytic = hilbert(cf_rf, axis=0)
    cf_env = np.abs(cf_analytic).astype(np.float32)
    cf_env /= (np.max(cf_env) + 1e-12)
    cf_bmode = 20.0 * np.log10(cf_env + 1e-12)
    cf_bmode = np.maximum(cf_bmode, -60.0)

    save_bmode_png(cf_bmode, grid_x, grid_z, out_dir/"bmode_cf.png", f"CF-weighted DAS (time={cf_time:.2f}s)")

    # --- Metrics ---
    roi = CystROI(
        center_x_m=cyst.center_x_m,
        center_z_m=cyst.center_z_m,
        radius_m=cyst.radius_m,
        bg_inner_radius_m=cyst.radius_m * 1.3,
        bg_outer_radius_m=cyst.radius_m * 2.2,
    )

    das_fwhm = fwhm_lateral(das_bmode, grid_x, grid_z, target_x_m=0.0, target_z_m=0.030)
    cf_fwhm = fwhm_lateral(cf_bmode, grid_x, grid_z, target_x_m=0.0, target_z_m=0.030)

    das_cr, das_cnr = contrast_metrics(das_env, grid_x, grid_z, roi)
    cf_cr, cf_cnr = contrast_metrics(cf_env, grid_x, grid_z, roi)

    df = pd.DataFrame(
        [
            {"method": "DAS", "fwhm_mm": das_fwhm * 1e3, "cr_db": das_cr, "cnr": das_cnr, "time_s": das_time},
            {"method": "CF",  "fwhm_mm": cf_fwhm * 1e3, "cr_db": cf_cr,  "cnr": cf_cnr,  "time_s": cf_time},
        ]
    )
    df.to_csv(out_dir/"metrics.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
