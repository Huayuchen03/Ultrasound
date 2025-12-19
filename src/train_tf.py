"""Training stub that avoids relative-import errors and heavy TF dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Optional


def _ensure_repo_on_path() -> None:
    """Allow running as ``python src/train_tf.py`` from the repo root."""

    if __package__:
        return

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.nn_model_tf import NNConfig, build_model


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", type=Path, default=Path("data/train.npz"), help="Training dataset")
    parser.add_argument("--val_npz", type=Path, default=Path("data/val.npz"), help="Validation dataset")
    parser.add_argument("--out_dir", type=Path, default=Path("models/demo_tf"), help="Directory to save model + logs")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for the placeholder trainer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the stub config")
    parser.add_argument("--hidden_units", type=int, nargs="*", default=[128, 64], help="Hidden layer sizes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args(list(argv) if argv is not None else None)


def simulate_training(args: argparse.Namespace) -> dict:
    config = NNConfig(
        num_channels=64,
        num_samples=256,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
    )
    model = build_model(config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.out_dir)

    summary = {
        "train_npz": str(args.train_npz),
        "train_npz_exists": args.train_npz.exists(),
        "val_npz": str(args.val_npz),
        "val_npz_exists": args.val_npz.exists(),
        "epochs": args.epochs,
        "seed": args.seed,
        "note": "TensorFlow training not bundled; this is a placeholder run.",
    }
    (args.out_dir / "training_log.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    summary = simulate_training(args)
    print(f"Saved placeholder model and log to {args.out_dir}")
    if not summary["train_npz_exists"]:
        print("Warning: training dataset not found; no real training performed.")
    if not summary["val_npz_exists"]:
        print("Warning: validation dataset not found; no real validation performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
