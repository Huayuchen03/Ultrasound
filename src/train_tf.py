from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .nn_model_tf import build_model, NNConfig


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32).reshape(-1, 1)
    mean = data["mean"].astype(np.float32)
    var = data["var"].astype(np.float32)
    meta = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files if k not in {"X", "y", "mean", "var"}}
    return X, y, mean, var, meta


def make_ds(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--l2_reg", type=float, default=1e-6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, mean, var, meta_tr = load_npz(Path(args.train_npz))
    Xva, yva, mean2, var2, meta_va = load_npz(Path(args.val_npz))

    if Xtr.shape[1] != Xva.shape[1]:
        raise ValueError("Train/val element count mismatch")
    if mean.shape != mean2.shape:
        raise ValueError("Train/val mean shape mismatch")

    num_elements = int(Xtr.shape[1])

    cfg = NNConfig(hidden_units=args.hidden_units, hidden_layers=args.hidden_layers, l2_reg=args.l2_reg)
    model = build_model(num_elements=num_elements, mean=mean, var=var, cfg=cfg)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    ds_tr = make_ds(Xtr, ytr, batch_size=args.batch_size, shuffle=True)
    ds_va = make_ds(Xva, yva, batch_size=args.batch_size, shuffle=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "ckpt.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks)

    # Save trained model
    model.save(out_dir / "saved_model")

    # Save normalization (redundant but convenient for inspection)
    np.savez_compressed(out_dir / "normalization.npz", mean=mean, var=var, meta=meta_tr)

    # Save training curves
    np.savez_compressed(out_dir / "history.npz", **{k: np.array(v) for k, v in hist.history.items()})

    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
