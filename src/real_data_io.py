from __future__ import annotations

"""Helpers for loading *public* real ultrasound datasets.

This project ships without any real RF datasets to keep the repository small
and to avoid redistributing third-party data. However, course rubrics often
require a "real" example in addition to simulation.

To support that, this module provides a *best-effort* loader for HDF5-based
ultrasound channel datasets (including USTB's UFF files, which are HDF5 under
the hood).

Important:
    HDF5 file layouts vary across datasets. The functions below implement
    reasonable heuristics and expose CLI overrides so you can make the loader
    work with the dataset you choose.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RealDataLoaded:
    """Container for a loaded real-data acquisition."""

    rf_wnt: np.ndarray  # (W, N, T) float32
    t_s: np.ndarray  # (T,) float64
    element_x_m: np.ndarray  # (N,) float64
    angles_rad: np.ndarray  # (W,) float64
    meta: Dict[str, object]


def _require_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "h5py is required for loading HDF5/UFF files. Install it via `pip install h5py`."
        ) from e


def _iter_datasets(h5_group, prefix: str = "") -> Iterator[Tuple[str, object]]:
    """Yield (path, dataset) for all datasets under an HDF5 group."""
    for key in h5_group.keys():
        item = h5_group[key]
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        # h5py Dataset has .shape attribute; Group does not in the same way
        if hasattr(item, "shape"):
            yield path, item
        else:
            yield from _iter_datasets(item, prefix=path)


def summarize_hdf5(file_path: str, max_items: int = 200) -> List[str]:
    """Return a human-readable summary of datasets in an HDF5 file."""
    h5py = _require_h5py()
    out: List[str] = []
    with h5py.File(file_path, "r") as f:
        for i, (path, ds) in enumerate(_iter_datasets(f)):
            if i >= max_items:
                out.append(f"... (truncated at {max_items} datasets)")
                break
            shape = getattr(ds, "shape", None)
            dtype = getattr(ds, "dtype", None)
            out.append(f"{path}: shape={shape}, dtype={dtype}")
    return out


def _pick_rf_dataset(candidates: List[Tuple[str, object]]) -> Tuple[str, object]:
    """Pick the most plausible RF dataset among candidates by simple heuristics."""
    # Score prefers large arrays with 3-4 dims and a plausible channel dimension.
    best = None
    best_score = -1.0
    for path, ds in candidates:
        shape = tuple(int(s) for s in ds.shape)
        if len(shape) < 3 or len(shape) > 4:
            continue
        # numeric only
        if not np.issubdtype(ds.dtype, np.number):
            continue

        dims = np.array(shape, dtype=np.float64)
        total = float(np.prod(dims))
        # time axis should be large
        time_like = float(np.max(dims))
        # channel axis should be 16..512-ish
        chan_like = float(np.max([d for d in dims if 16 <= d <= 512] + [0.0]))
        score = np.log10(total + 1.0) + 0.5 * np.log10(time_like + 1.0) + 0.25 * np.log10(chan_like + 1.0)
        # small bonus if path contains suggestive keywords
        lower = path.lower()
        if "channel" in lower or "rf" in lower or "data" in lower:
            score += 0.25

        if score > best_score:
            best_score = score
            best = (path, ds)

    if best is None:
        raise ValueError("Could not locate a plausible RF/channel dataset inside the HDF5 file.")
    return best


def _guess_axes(shape: Tuple[int, ...]) -> Tuple[int, int, int, Optional[int]]:
    """Guess (time_axis, chan_axis, wave_axis, frame_axis) from a 3D/4D shape."""
    nd = len(shape)
    if nd not in (3, 4):
        raise ValueError(f"Expected 3D or 4D dataset. Got shape {shape}")

    # Time axis: largest dimension
    time_axis = int(np.argmax(shape))

    # Channel axis: dimension between 16 and 512 (prefer closest to 128)
    chan_candidates = [(i, s) for i, s in enumerate(shape) if i != time_axis and 16 <= s <= 512]
    if len(chan_candidates) == 0:
        # fallback: smallest non-time axis
        non_time = [(i, s) for i, s in enumerate(shape) if i != time_axis]
        chan_axis = int(sorted(non_time, key=lambda x: x[1])[0][0])
    else:
        chan_axis = int(sorted(chan_candidates, key=lambda x: abs(x[1] - 128))[0][0])

    remaining = [i for i in range(nd) if i not in (time_axis, chan_axis)]
    if nd == 3:
        wave_axis = int(remaining[0])
        frame_axis = None
    else:
        # For 4D: pick frame axis as the one with smallest size among remaining,
        # unless one remaining dimension looks like wave count (>=3 and <=256).
        rem_sizes = [(i, shape[i]) for i in remaining]
        # wave candidate: size 1..256
        wave_candidates = [(i, s) for i, s in rem_sizes if 1 <= s <= 256]
        if len(wave_candidates) == 0:
            # fallback: larger remaining is wave
            wave_axis = int(sorted(rem_sizes, key=lambda x: -x[1])[0][0])
        else:
            wave_axis = int(sorted(wave_candidates, key=lambda x: -x[1])[0][0])
        frame_axis = int([i for i in remaining if i != wave_axis][0])
    return time_axis, chan_axis, wave_axis, frame_axis


def _read_scalar_attr(obj, names: Iterable[str]) -> Optional[float]:
    for k in obj.attrs.keys():
        for name in names:
            if str(k).lower() == name.lower():
                v = obj.attrs[k]
                try:
                    return float(np.array(v).reshape(()))
                except Exception:
                    pass
    return None


def _search_scalar_attr_recursive(h5_group, names: Iterable[str], max_nodes: int = 5000) -> Optional[float]:
    """Search recursively for a scalar attribute by name."""
    stack = [h5_group]
    seen = 0
    while stack and seen < max_nodes:
        obj = stack.pop()
        seen += 1
        v = _read_scalar_attr(obj, names)
        if v is not None:
            return v
        # traverse
        if hasattr(obj, "keys"):
            for key in obj.keys():
                stack.append(obj[key])
    return None


def load_hdf5_channel_data_guess(
    file_path: str,
    fs_hz: Optional[float] = None,
    pitch_m: Optional[float] = None,
    angles_rad: Optional[np.ndarray] = None,
    rf_dataset_path: Optional[str] = None,
) -> RealDataLoaded:
    """Load channel data from an HDF5/UFF file using best-effort heuristics.

    Parameters
    ----------
    file_path:
        Path to the HDF5/UFF file.
    fs_hz:
        Override sampling frequency (Hz). If None, an attempt is made to find it
        in file attributes.
    pitch_m:
        If element positions cannot be found, assume a uniform linear array with
        this pitch (meters).
    angles_rad:
        Optional override for transmit angles (radians). If None, tries to find
        them; otherwise falls back to zeros.
    rf_dataset_path:
        If you know the dataset path for RF data inside the HDF5 file, provide
        it to avoid heuristics.

    Returns
    -------
    RealDataLoaded
        Contains RF data in (W,N,T) layout plus metadata.
    """

    h5py = _require_h5py()
    meta: Dict[str, object] = {"file_path": file_path}

    with h5py.File(file_path, "r") as f:
        # --- locate RF dataset ---
        if rf_dataset_path is not None:
            ds = f[rf_dataset_path]
            rf_path = rf_dataset_path
        else:
            all_ds = [(path, ds) for path, ds in _iter_datasets(f)]
            rf_path, ds = _pick_rf_dataset(all_ds)

        meta["rf_dataset_path"] = rf_path
        shape = tuple(int(s) for s in ds.shape)
        meta["rf_dataset_shape"] = shape

        time_axis, chan_axis, wave_axis, frame_axis = _guess_axes(shape)
        meta["time_axis"] = time_axis
        meta["chan_axis"] = chan_axis
        meta["wave_axis"] = wave_axis
        meta["frame_axis"] = frame_axis

        # Read one frame if present.
        if frame_axis is None:
            data = ds[...]
        else:
            slc = [slice(None)] * len(shape)
            slc[frame_axis] = 0
            data = ds[tuple(slc)]

        data = np.asarray(data)

        # Reorder to (W, N, T)
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D array after frame selection. Got {data.shape}")

        # Map original axes to desired
        # original axes indices: 0..2 correspond to those in `shape` excluding frame.
        # But our guess axes are in original shape; if frame removed, need to remap.
        # We'll compute a mapping by removing frame_axis.
        orig_axes = list(range(len(shape)))
        if frame_axis is not None:
            orig_axes.pop(frame_axis)
        # Find positions of time/chan/wave within the 3D `data` array
        t_pos = orig_axes.index(time_axis)
        c_pos = orig_axes.index(chan_axis)
        w_pos = orig_axes.index(wave_axis)

        rf_wnt = np.transpose(data, (w_pos, c_pos, t_pos)).astype(np.float32)

        W, N, T = rf_wnt.shape
        meta["W"] = W
        meta["N"] = N
        meta["T"] = T

        # --- sampling frequency ---
        if fs_hz is None:
            fs_hz = _read_scalar_attr(ds, ["sampling_frequency", "fs_hz", "fs", "samplingfrequency"])
        if fs_hz is None:
            fs_hz = _search_scalar_attr_recursive(f, ["sampling_frequency", "fs_hz", "fs", "samplingfrequency"])
        if fs_hz is None:
            raise ValueError(
                "Could not infer sampling frequency from the file. "
                "Pass --fs_hz explicitly (e.g., 20e6)."
            )
        fs_hz_f = float(fs_hz)
        meta["fs_hz"] = fs_hz_f
        t_s = (np.arange(T, dtype=np.float64) / fs_hz_f).astype(np.float64)

        # --- element positions ---
        element_x_m: Optional[np.ndarray] = None
        # Heuristic: look for 1D dataset of length N with 'x' and ('probe' or 'element') in path
        ds_list = list(_iter_datasets(f))
        for path, ds2 in ds_list:
            if not np.issubdtype(ds2.dtype, np.number):
                continue
            if ds2.ndim != 1:
                continue
            if int(ds2.shape[0]) != int(N):
                continue
            lower = path.lower()
            if ("probe" in lower or "element" in lower or "geometry" in lower) and ("x" in lower):
                try:
                    element_x_m = np.asarray(ds2[...], dtype=np.float64).reshape(N)
                    meta["element_x_path"] = path
                    break
                except Exception:
                    pass

        # Another heuristic: look for geometry dataset with shape (3,N) or (N,3)
        if element_x_m is None:
            for path, ds2 in ds_list:
                if not np.issubdtype(ds2.dtype, np.number):
                    continue
                if ds2.ndim != 2:
                    continue
                s0, s1 = int(ds2.shape[0]), int(ds2.shape[1])
                if (s0, s1) == (3, int(N)) or (s0, s1) == (int(N), 3):
                    lower = path.lower()
                    if "probe" in lower or "geometry" in lower:
                        arr = np.asarray(ds2[...], dtype=np.float64)
                        if arr.shape[0] == 3:
                            element_x_m = arr[0, :].reshape(N)
                        else:
                            element_x_m = arr[:, 0].reshape(N)
                        meta["geometry_path"] = path
                        break

        if element_x_m is None:
            if pitch_m is None:
                raise ValueError(
                    "Could not infer element positions from the file. "
                    "Provide --pitch_m to assume a uniform linear array."
                )
            pitch = float(pitch_m)
            # center the array at x=0
            element_x_m = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0) * pitch
            meta["element_x_path"] = "(synthetic_from_pitch)"
            meta["pitch_m"] = pitch

        # --- angles ---
        angles_out: Optional[np.ndarray] = None
        if angles_rad is not None:
            angles_out = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
            if angles_out.shape[0] != W:
                raise ValueError(f"Provided angles_rad has length {angles_out.shape[0]} but W={W}")
            meta["angles_source"] = "(override)"
        else:
            # Heuristic: find 1D dataset length W with 'angle' or 'theta'
            for path, ds2 in ds_list:
                if not np.issubdtype(ds2.dtype, np.number):
                    continue
                if ds2.ndim != 1:
                    continue
                if int(ds2.shape[0]) != int(W):
                    continue
                lower = path.lower()
                if "angle" in lower or "theta" in lower:
                    try:
                        angles_out = np.asarray(ds2[...], dtype=np.float64).reshape(W)
                        meta["angles_path"] = path
                        meta["angles_source"] = "(file)"
                        break
                    except Exception:
                        pass
        if angles_out is None:
            angles_out = np.zeros((W,), dtype=np.float64)
            meta["angles_source"] = "(default_zero)"

    return RealDataLoaded(
        rf_wnt=rf_wnt,
        t_s=t_s,
        element_x_m=element_x_m,
        angles_rad=angles_out,
        meta=meta,
    )
