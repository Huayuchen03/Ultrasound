from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class NNConfig:
    hidden_units: int = 128
    hidden_layers: int = 2
    l2_reg: float = 1e-6


class WeightPredictor(tf.keras.Model):
    '''
    Predicts data-dependent receive weights and forms a weighted sum y = w^T x_raw.

    Design:
      - Input: raw snapshot x_raw (shape [batch, N])
      - Internal: normalize x_raw -> x_norm (Normalization layer with fixed mean/var)
      - MLP(x_norm) -> w_raw
      - w = w_raw / (sum(|w_raw|) + eps)   # allows positive/negative weights but bounded
      - Output: y = sum(w * x_raw)

    This separates:
      - *feature scaling* (x_norm) used to estimate weights,
      - from the *physical beamforming sum* on raw channel samples.
    '''
    def __init__(self, num_elements: int, mean: np.ndarray, var: np.ndarray, cfg: NNConfig):
        super().__init__()
        self.num_elements = int(num_elements)

        mean = np.asarray(mean, dtype=np.float32).reshape((num_elements,))
        var = np.asarray(var, dtype=np.float32).reshape((num_elements,))

        self.norm = tf.keras.layers.Normalization(axis=-1, name="norm")
        # Build normalization layer weights explicitly:
        self.norm.build((None, num_elements))
        self.norm.set_weights([mean, var])

        reg = tf.keras.regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None

        layers = []
        for _ in range(int(cfg.hidden_layers)):
            layers.append(tf.keras.layers.Dense(cfg.hidden_units, activation="relu", kernel_regularizer=reg))
        self.mlp = tf.keras.Sequential(layers, name="mlp")
        self.to_w = tf.keras.layers.Dense(self.num_elements, activation=None, kernel_regularizer=reg, name="to_w")

    def call(self, x_raw: tf.Tensor, training: bool = False, return_weights: bool = False):
        x_norm = self.norm(x_raw)
        h = self.mlp(x_norm, training=training)
        w_raw = self.to_w(h, training=training)

        w = w_raw / (tf.reduce_sum(tf.abs(w_raw), axis=-1, keepdims=True) + 1e-6)
        y = tf.reduce_sum(w * x_raw, axis=-1, keepdims=True)

        if return_weights:
            return y, w
        return y


def build_model(num_elements: int, mean: np.ndarray, var: np.ndarray, cfg: NNConfig) -> tf.keras.Model:
    model = WeightPredictor(num_elements=num_elements, mean=mean, var=var, cfg=cfg)
    _ = model(tf.zeros((1, num_elements), dtype=tf.float32))
    return model
