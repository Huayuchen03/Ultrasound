"""Parameter containers for ultrasound array geometry and imaging grid.

These dataclasses provide light-weight validation and convenience helpers that
scripts can rely on without worrying about relative-import failures. The module
is importable either as ``src.params`` or via ``from src import ArrayParams``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ArrayParams:
    """Description of a linear array transducer."""

    num_elements: int
    pitch_m: float

    def __post_init__(self) -> None:
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")
        if self.pitch_m <= 0:
            raise ValueError("pitch_m must be positive")

    @property
    def aperture_m(self) -> float:
        """Total aperture length in meters."""

        return self.pitch_m * (self.num_elements - 1)


@dataclass(frozen=True)
class AcqParams:
    """Acquisition parameters for the simulation or replay."""

    fs_hz: float
    c_m_s: float = 1540.0
    center_freq_hz: float = 5e6

    def __post_init__(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError("fs_hz must be positive")
        if self.c_m_s <= 0:
            raise ValueError("c_m_s must be positive")
        if self.center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be positive")

    @property
    def dt(self) -> float:
        """Sampling period in seconds."""

        return 1.0 / self.fs_hz


@dataclass(frozen=True)
class ImageGrid:
    """Regular grid used for beamformed image export."""

    nx: int
    nz: int
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.nz <= 0:
            raise ValueError("nx and nz must be positive")
        if self.x_max_m <= self.x_min_m:
            raise ValueError("x_max_m must exceed x_min_m")
        if self.z_max_m <= self.z_min_m:
            raise ValueError("z_max_m must exceed z_min_m")

    @property
    def x_axis_m(self) -> Tuple[float, float]:
        """Return the lateral axis extent in meters."""

        return (self.x_min_m, self.x_max_m)

    @property
    def z_axis_m(self) -> Tuple[float, float]:
        """Return the axial axis extent in meters."""

        return (self.z_min_m, self.z_max_m)

    @property
    def dx(self) -> float:
        """Lateral grid spacing in meters."""

        return (self.x_max_m - self.x_min_m) / (self.nx - 1)

    @property
    def dz(self) -> float:
        """Axial grid spacing in meters."""

        return (self.z_max_m - self.z_min_m) / (self.nz - 1)


__all__ = ["ArrayParams", "AcqParams", "ImageGrid"]
