# jax_logic_skeleton.py
# Pure JAX skeleton for a simple logical problem (e.g., AND) with a 2 -> 2 -> 1 network.
# Goal: fill in the TODOs (data, forward pass, loss, metrics, prediction, train_step).
# Constraints: no optax/flax; use jax.grad and manual SGD updates.
#
# Initialization instructions:
#   - Biases are initialized to zeros.
#   - Weights are sampled from a standard normal distribution and scaled with He scaling:
#         W = Normal(0, 1) * sqrt(2 / fan_in)
#     where fan_in is the number of input features to that layer.
#   - Init params should return a params dictionary of the form
#       params = {"W0": W0, "b0": b0,.....}
#
# Reproducibility / random seeds:
#   - Setting cfg.seed controls the PRNGKey used for parameter initialization.
#   - Different seeds => different initial parameters => training can converge differently
#     and final performance may differ (especially for small networks / small datasets).
#   - Try: for s in [0,1,2,3,4]: cfg = Config(seed=s); run training and compare metrics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Dict[str, Array]
Batch = Tuple[Array, Array]  # (X, y)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    lr: float = 0.5
    steps: int = 2000
    log_every: int = 200
    hidden_activation: Literal["linear", "relu"] = "linear"
    threshold: float = 0.5


# -------------------------
# Data: truth table (replace for AND/OR/XOR as needed)
# -------------------------
def truth_table_data() -> Batch:
    """
    Return:
      X: shape (4, 2) with rows [[0,0], [0,1], [1,0], [1,1]]
      y: shape (4,) with labels in {0,1} as float32
    """
    # TODO: fill in X and y
    raise NotImplementedError


# -------------------------
# Model: 2 -> 2 -> 1
# Convention: batched inputs X have shape (B,2) and use X @ W
# -------------------------
def init_params(key: Array, *, in_dim: int = 2, hidden: int = 2, out_dim: int = 1) -> Params:
    """
    Shapes:
      W0: (in_dim, hidden)   with fan_in = in_dim
      b0: (hidden,)          zeros
      W1: (hidden, out_dim)  with fan_in = hidden
      b1: (out_dim,)         zeros
    """
    # TODO: implement parameter initialization exactly as described above
    raise NotImplementedError


def forward_logits(params: Params, X: Array, *, hidden_activation: str) -> Array:
    """
    Forward pass returning logits (pre-sigmoid) with shape (B,).
    Steps:
      a0 = X @ W0 + b0
      z0 = a0 (linear) or relu(a0)
      a1 = z0 @ W1 + b1
      return a1.squeeze(-1)
    """
    # TODO: implement forward pass
    raise NotImplementedError


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))


# -------------------------
# Loss / metrics
# -------------------------
def loss_fn(params: Params, batch: Batch, *, cfg: Config) -> Array:
    """
    Binary cross-entropy with logits (stable form recommended).

    Recommended stable form (BCE-with-logits):
        logits = forward_logits(...)
        bce_per_example = softplus(logits) - y * logits
        loss = mean(bce_per_example)

    where softplus(t) = log(1 + exp(t)).
    """
    # TODO: implement BCE loss
    raise NotImplementedError


def accuracy(params: Params, batch: Batch, *, cfg: Config) -> Array:
    """
    Compute accuracy:
      - p = sigmoid(logits)
      - pred = p >= cfg.threshold
      - compare pred to y
    """
    # TODO: implement accuracy
    raise NotImplementedError


# -------------------------
# Prediction helpers
# -------------------------
def predict_proba(params: Params, X: Array, *, cfg: Config) -> Array:
    """
    Return P(y=1|x).
    Accept:
      X shape (2,)   -> return scalar ()
      X shape (B,2)  -> return (B,)
    """
    # TODO: implement predict_proba
    raise NotImplementedError


def predict(params: Params, X: Array, *, cfg: Config) -> Array:
    """
    Return class predictions in {0,1}.
    Same batching behavior as predict_proba.
    """
    # TODO: implement predict
    raise NotImplementedError


# -------------------------
# Manual SGD update (provided)
# -------------------------
def sgd_update(params: Params, grads: Params, lr: float) -> Params:
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def make_train_step(cfg: Config):
    @jax.jit
    def train_step(params: Params, batch: Batch) -> Params:
        """
        One training step:
          - compute gradients of loss_fn w.r.t. params using jax.grad
          - update params with sgd_update using cfg.lr
        """
        # TODO: compute grads with jax.grad and apply sgd_update
        raise NotImplementedError
    return train_step


# -------------------------
# Example usage (no main guard)
# -------------------------
cfg = Config()

# Initialize
key = jax.random.PRNGKey(cfg.seed)
params = init_params(key)

# Training step fn
train_step = make_train_step(cfg)

# Data
batch = truth_table_data()

# Training loop
for step in range(1, cfg.steps + 1):
    params = train_step(params, batch)

    if step % cfg.log_every == 0:
        l = loss_fn(params, batch, cfg=cfg)
        a = accuracy(params, batch, cfg=cfg)
        print(f"step {step:04d}  loss={float(l):.6f}  acc={float(a):.3f}")

# Predictions
X, y = batch
p = predict_proba(params, X, cfg=cfg)
yhat = predict(params, X, cfg=cfg)
print("\nPredictions:")
for i in range(X.shape[0]):
    print(f"x={X[i].tolist()}  y={int(y[i])}  proba={float(p[i]):.4f}  pred={int(yhat[i])}")
