"""Ultrasound beamforming package skeleton."""

from __future__ import annotations

from pathlib import Path
import sys

__all__ = ["ArrayParams", "AcqParams", "ImageGrid"]


def _import_params():
    """Import parameter dataclasses regardless of how the module is invoked.

    When ``src`` is executed as ``python -m src`` or imported as a package,
    relative imports work normally. When someone runs ``python src/__init__.py``
    directly, ``__package__`` is ``None`` and relative imports fail; in that
    case we temporarily add the repository root to ``sys.path`` so that the
    absolute ``src.params`` import succeeds.
    """

    if __package__ is None:  # Direct script execution
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.params import ArrayParams, AcqParams, ImageGrid  # type: ignore
        return ArrayParams, AcqParams, ImageGrid

    # Normal package import
    from .params import ArrayParams, AcqParams, ImageGrid
    return ArrayParams, AcqParams, ImageGrid


ArrayParams, AcqParams, ImageGrid = _import_params()

