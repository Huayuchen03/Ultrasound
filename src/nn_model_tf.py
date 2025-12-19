"""TensorFlow model stubs used by ``train_tf`` without heavy dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NNConfig:
    """Lightweight configuration for the placeholder network."""

    num_channels: int = 64
    num_samples: int = 256
    hidden_units: Optional[List[int]] = None
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.hidden_units is None:
            self.hidden_units = [128, 64]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["hidden_units"] = list(self.hidden_units)
        return payload


class PlaceholderModel:
    """Minimal stand-in for a TensorFlow model."""

    def __init__(self, config: NNConfig):
        self.config = config

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "note": "TensorFlow model not bundled; this is a placeholder graph description.",
            "config": self.config.to_dict(),
        }
        (out_dir / "model_summary.json").write_text(
            json.dumps(summary, indent=2)
        )


def build_model(config: NNConfig) -> PlaceholderModel:
    """Return a placeholder model so imports succeed without TensorFlow."""

    return PlaceholderModel(config)
