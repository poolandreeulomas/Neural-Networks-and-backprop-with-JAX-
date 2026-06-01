from __future__ import annotations
import os

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp


from tqdm import tqdm

Array = jax.Array
Batch = Tuple[Array, Array, Array]  # x, y, mask
Metrics = Dict[str, Array]

#%%
# ----------------------------
# Configs
# ----------------------------
@dataclass(frozen=True)
class DataConfig:
    seed: int = 0
    batch_size: int = 16
    test_size: float = 0.3
    shuffle_train: bool = True

    # If True, train_test_split(..., stratify=y) is used when creating
    # train/test splits from a single dataset. This preserves the label
    # distribution across the splits, which is often useful for
    # classification tasks, especially with small or imbalanced datasets.
    # Usually not used for regression with continuous targets.
    stratify: bool = False


@dataclass(frozen=True)
class Config:
    # training
    seed: int = 0
    epochs: int = 50
    steps_per_epoch: Optional[int] = None  # Automatically computed in model init

    # optimizer
    lr: float = 1e-3
    optim: str = "adam"          # "adam" | "adamw" | "sgd"
    momentum: float = 0.9        # for SGD
    weight_decay: float = 0.0    # for AdamW
    clip_norm: Optional[float] = None  # optional grad clipping

    # model (placeholders)
    in_dim: int = 2
    hidden_dim: int = 32
    out_dim: int = 1
    dropout: float = 0.0


    # data
    data: DataConfig = dataclasses.field(default_factory=DataConfig)


# -----
# Data
# -----
def _make_batches_jax(
    X,
    y,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    repeat: bool,
) -> Iterator[Batch]:
    
    # We always pad the last batch to full size so JIT sees a fixed shape.
    # A mask marks which samples are real and which are padding.
    # Losses are computed only over real samples.

    n = int(X.shape[0])
    key = jax.random.PRNGKey(seed)

    while True:
        if shuffle:
            key, sub = jax.random.split(key)
            idx = np.array(jax.random.permutation(sub, n))
        else:
            idx = np.arange(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]

            xb = X[b]
            yb = y[b]

            actual = end - start
            pad = batch_size - actual

            if pad > 0:
                xb = np.pad(
                    xb,
                    pad_width=[(0, pad)] + [(0, 0)] * (xb.ndim - 1),
                    mode="constant",
                )
                yb = np.pad(
                    yb,
                    pad_width=[(0, pad)] + [(0, 0)] * (yb.ndim - 1),
                    mode="constant",
                )

            mask = np.zeros((batch_size,), dtype=np.float32)
            mask[:actual] = 1.0

            yield jnp.asarray(xb), jnp.asarray(yb), jnp.asarray(mask)

        if not repeat:
            break



class DataModule:
    """
    Minimal dataloader:
      - supports CSV / arrays / sklearn datasets
      - uses sklearn train_test_split
      - yields JAX batches (train: infinite, test: finite)
      - does NOT change labels (task-specific label transforms are up to you)
    """

    def __init__(self, cfg: Config, X_train: Any, y_train: Any, X_test: Any, y_test: Any):
        self.cfg = cfg
        self.X_train, self.y_train = np.asarray(X_train), np.asarray(y_train)
        self.X_test, self.y_test = np.asarray(X_test), np.asarray(y_test)

        # ======================================================
        # PLACEHOLDER A: label preprocessing (task-specific)
        # ======================================================
        # Examples:
        # - classification: map strings -> integers, optionally one-hot
        # - regression: cast to float32, optionally normalize target
        #
        # self.y_train = ...
        # self.y_test  = ...
        # ======================================================

        # ======================================================
        # PLACEHOLDER B: feature preprocessing
        # ======================================================
        #
        # Common options:
        # - Standardization: (X - mean) / std
        # - Normalization (e.g., [0, 1])
        # - Column-wise scaling
        #
        # self.X_train = ...
        # self.X_test  = ...
        # ======================================================

    @staticmethod
    def _read_csv_xy(csv_path: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        X_np = df.drop(columns=[label_col]).to_numpy()
        y_np = df[label_col].to_numpy()
        return X_np, y_np

    @staticmethod
    def _read_csv_X_y(X_csv_path: str, y_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        X_np = pd.read_csv(X_csv_path).to_numpy()
        y_df = pd.read_csv(y_csv_path)

        # If y has only one column, return a 1D array
        if y_df.shape[1] == 1:
            y_np = y_df.iloc[:, 0].to_numpy()
        else:
            y_np = y_df.to_numpy()

        return X_np, y_np

    @classmethod
    def from_arrays(cls, cfg: Config, *, X: Any, y: Any) -> "DataModule":
        """
        Split X,y into train/test using train_test_split
        and return a DataModule.
        """
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        stratify = y_np if cfg.data.stratify else None

        Xtr, Xte, ytr, yte = train_test_split(
            X_np,
            y_np,
            test_size=cfg.data.test_size,
            random_state=cfg.data.seed,
            shuffle=True,
            stratify=stratify,
        )

        return cls(cfg, Xtr, ytr, Xte, yte)

    @classmethod
    def from_splits(
        cls,
        cfg: Config,
        *,
        Xtr: Any,
        Xte: Any,
        ytr: Any,
        yte: Any,
    ) -> "DataModule":
        """
        Use already-split arrays (Xtr, Xte, ytr, yte)
        and return a DataModule.
        """
        return cls(cfg, Xtr, ytr, Xte, yte)

    @classmethod
    def from_csv(
        cls,
        cfg: Config,
        *,
        label_col: Optional[str] = None,
        csv_path: Optional[str] = None,
        train_csv_path: Optional[str] = None,
        test_csv_path: Optional[str] = None,
        X_csv_path: Optional[str] = None,
        y_csv_path: Optional[str] = None,
        X_train_csv_path: Optional[str] = None,
        y_train_csv_path: Optional[str] = None,
        X_test_csv_path: Optional[str] = None,
        y_test_csv_path: Optional[str] = None,
    ) -> "DataModule":
        """
        Supported CSV input modes:

        A) One CSV containing both X and y:
           csv_path + label_col

        B) Two pre-split CSVs containing both X and y:
           train_csv_path + test_csv_path + label_col

        C) Two separate CSVs for unsplit X and y:
           X_csv_path + y_csv_path

        D) Four separate pre-split CSVs:
           X_train_csv_path + y_train_csv_path + X_test_csv_path + y_test_csv_path
        """

        # --------------------------------------------------
        # D) Already split, X and y in separate files
        # --------------------------------------------------
        separate_split_given = any(
            p is not None
            for p in [X_train_csv_path, y_train_csv_path, X_test_csv_path, y_test_csv_path]
        )
        if separate_split_given:
            required = [X_train_csv_path, y_train_csv_path, X_test_csv_path, y_test_csv_path]
            if not all(p is not None for p in required):
                raise ValueError(
                    "If using separate split CSVs, provide all of "
                    "X_train_csv_path, y_train_csv_path, X_test_csv_path, and y_test_csv_path."
                )

            Xtr, ytr = cls._read_csv_X_y(X_train_csv_path, y_train_csv_path)
            Xte, yte = cls._read_csv_X_y(X_test_csv_path, y_test_csv_path)
            return cls.from_splits(cfg, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)

        # --------------------------------------------------
        # B) Already split, each CSV contains both X and y
        # --------------------------------------------------
        combined_split_given = (train_csv_path is not None) or (test_csv_path is not None)
        if combined_split_given:
            if train_csv_path is None or test_csv_path is None:
                raise ValueError(
                    "If using split CSVs with combined X/y, both train_csv_path and test_csv_path are required."
                )
            if label_col is None:
                raise ValueError(
                    "label_col is required when labels are stored in the same CSV as features."
                )

            Xtr, ytr = cls._read_csv_xy(train_csv_path, label_col)
            Xte, yte = cls._read_csv_xy(test_csv_path, label_col)
            return cls.from_splits(cfg, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)

        # --------------------------------------------------
        # C) Unsplit, X and y in separate files
        # --------------------------------------------------
        separate_unsplit_given = (X_csv_path is not None) or (y_csv_path is not None)
        if separate_unsplit_given:
            if X_csv_path is None or y_csv_path is None:
                raise ValueError("If using separate X/y CSVs, both X_csv_path and y_csv_path are required.")

            X_np, y_np = cls._read_csv_X_y(X_csv_path, y_csv_path)
            return cls.from_arrays(cfg, X=X_np, y=y_np)

        # --------------------------------------------------
        # A) Unsplit, one CSV containing both X and y
        # --------------------------------------------------
        if csv_path is not None:
            if label_col is None:
                raise ValueError(
                    "label_col is required when labels are stored in the same CSV as features."
                )

            X_np, y_np = cls._read_csv_xy(csv_path, label_col)
            return cls.from_arrays(cfg, X=X_np, y=y_np)

        raise ValueError(
            "Provide one of the supported CSV input modes:\n"
            "- csv_path + label_col\n"
            "- train_csv_path + test_csv_path + label_col\n"
            "- X_csv_path + y_csv_path\n"
            "- X_train_csv_path + y_train_csv_path + X_test_csv_path + y_test_csv_path"
        )

    @classmethod
    def from_sklearn(cls, cfg: Config, *, loader: Callable[..., Any], **loader_kwargs) -> "DataModule":
        """
        Example loaders:
        - sklearn.datasets.fetch_california_housing
        - sklearn.datasets.load_iris
        - sklearn.datasets.load_digits

        Must return either:
        - (X, y) tuple, OR
        - a Bunch/dict-like object with fields 'data' and 'target'
        """
        bunch = loader(**loader_kwargs)

        if isinstance(bunch, tuple) and len(bunch) == 2:
            X_np, y_np = bunch
        else:
            X_np = getattr(bunch, "data", None)
            y_np = getattr(bunch, "target", None)
            if X_np is None or y_np is None:
                X_np = bunch["data"]
                y_np = bunch["target"]

        return cls.from_arrays(cfg, X=X_np, y=y_np)

    def train_iter(self) -> Iterator[Batch]:
        return _make_batches_jax(
            self.X_train,
            self.y_train,
            batch_size=self.cfg.data.batch_size,
            seed=self.cfg.data.seed + 1,
            shuffle=self.cfg.data.shuffle_train,
            repeat=True,
        )

    def test_iter(self) -> Iterator[Batch]:
        return _make_batches_jax(
            self.X_test,
            self.y_test,
            batch_size=self.cfg.data.batch_size,
            seed=self.cfg.data.seed + 2,
            shuffle=False,
            repeat=False,
        )
    
# ============================================================
# Network (Flax NNX)
# ============================================================
class MyNetwork(nnx.Module):
    """
    Simple MLP example:
      Linear -> ReLU -> Dropout? -> Linear -> ReLU -> Dropout? -> Linear

    IMPORTANT:
      This network returns RAW OUTPUTS only.
      For classification these are logits.
      For regression these are predictions.
    """
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        self.cfg = cfg

        self.fc1 = nnx.Linear(cfg.in_dim, cfg.hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(cfg.hidden_dim, cfg.out_dim, rngs=rngs)

        self.drop = nnx.Dropout(cfg.dropout, rngs=rngs) if cfg.dropout > 0.0 else None

    def __call__(self, x: Array) -> Array:
        x = self.fc1(x)
        x = nnx.relu(x)
        if self.drop is not None:
            x = self.drop(x)

        x = self.fc2(x)
        x = nnx.relu(x)
        if self.drop is not None:
            x = self.drop(x)

        return self.fc_out(x)


NetworkFactory = Callable[[Config, nnx.Rngs], nnx.Module]


def make_my_network(cfg: Config, rngs: nnx.Rngs) -> nnx.Module:
    return MyNetwork(cfg, rngs=rngs)


# ============================================================
# Loss / metrics hook
# ============================================================
def loss_and_metrics(
    outputs: Array,
    y: Array,
    mask: Array,
) -> Tuple[Array, Metrics]:
    """
    TASK-SPECIFIC PLACEHOLDER.

    Implement a version appropriate for your task.
    'outputs' means raw network output:
      - regression: predictions, often shape (B, 1)
      - binary classification: logits, often shape (B, 1)
      - multiclass classification: logits, shape (B, C)
      - image-to-image regression: predictions, often shape (B, H, W, C)

    'mask' marks valid samples in the batch:
      - mask[i] = 1 for a real sample
      - mask[i] = 0 for a padded sample

    Must return:
      loss: scalar used for gradients
      metrics: dict of scalar arrays, must include "loss"

    Hints: 
        - For classification, use optax losses with logits (e.g., sigmoid_binary_cross_entropy or softmax_cross_entropy_with_integer_labels).
        - For regression, use MSE or MAE.
        - !! Make sure that the model outputs and labels have the same shape !!
        - Use the mask to compute losses/metrics only over real samples, not padding.
        - Use correct weighting when averaging over masked samples (e.g., divide by sum of mask).
    """
    raise NotImplementedError(
        "Implement loss_and_metrics(outputs, y, mask) for your task. "
        "See example implementation below."
    )


# ------------------------------------------------------------
# Example loss/metric implementation
#  ------------------------------------------------------------

## def loss_and_metrics(outputs: Array, y: Array, mask: Array) -> Tuple[Array, Metrics]:
#     """
#     Regression example: masked MSE + masked RMSE
#     Expects:
#       outputs: predictions with batch dimension first, e.g. (B,), (B, 1),
#                or more generally (B, ...)
#       y:       targets with the same shape as outputs (or reshapeable to it)
#       mask:    (B,) with 1 for valid samples and 0 for padded samples
#
#     Notes:
#       - The mask is reshaped to broadcast across all non-batch dimensions.
#       - Losses/metrics are computed only over valid (unmasked) samples.
#     """
#     preds = outputs.squeeze(-1) if outputs.ndim > 1 and outputs.shape[-1] == 1 else outputs
#     y = y.astype(jnp.float32).reshape(preds.shape)

#     if preds.ndim > 1:
#       mask = mask.astype(jnp.float32).reshape((preds.shape[0],) + (1,) * (preds.ndim - 1))
#     else:
#        mask = mask.astype(jnp.float32)
#
#     se = (preds - y) ** 2
#     valid_batch_size = jnp.maximum(jnp.sum(mask), 1.0)
#
#     mse = jnp.sum(se * mask) / valid_batch_size
#
#      # This is an example metric (e.g., for classification you might compute accuracy instead)
#     rmse = jnp.sqrt(mse) 
#     return mse, {"loss": mse, "rmse": rmse}




# ============================================================
# Optimizer factory
# ============================================================
def make_optax_tx(cfg: Config):
    """
    Create an Optax optimizer (tx = transformation) from cfg.optim.
    Optionally adds global-norm gradient clipping.
    """
    name = cfg.optim.lower()
    if name == "adam":
        tx = optax.adam(learning_rate=cfg.lr)
    elif name == "adamw":
        tx = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    elif name == "sgd":
        tx = optax.sgd(learning_rate=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optim}'. Use: adam | adamw | sgd")

    if cfg.clip_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(cfg.clip_norm), tx)

    return tx


# ============================================================
# Model (Keras-like wrapper)
# ============================================================
class Model:
    """
    Keras-like wrapper around an NNX network.

    - init(): creates network + optimizer once
    - fit(): training loop
    - evaluate(): compute mean metrics on test set
    - predict(): raw network outputs
    - save()/load(): checkpoint network (and optionally optimizer)
    """
    def __init__(self, cfg: Config, dm: DataModule, network_fn: NetworkFactory):
        self.cfg = cfg
        self.dm = dm
        self.network_fn = network_fn

        self.network: Optional[nnx.Module] = None
        self.optimizer: Optional[nnx.Optimizer] = None

        self._train_step = self.make_train_step()
        self._eval_step = self.make_eval_step()
        self._predict_step = self.make_predict_step()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def init(self) -> None:
        """
        Create the network and optimizer.
        Also computes steps_per_epoch if not set:
        steps_per_epoch = ceil(N_train / batch_size)
        """
        x0, _, _ = next(self.dm.train_iter())
        if x0.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"Config.in_dim={self.cfg.in_dim} but got batch with last dim {x0.shape[-1]}"
            )

        # Generate RNGs for network initialization and dropout
        rngs = nnx.Rngs(self.cfg.seed)

        self.network = self.network_fn(self.cfg, rngs)
        self.optimizer = nnx.Optimizer(
            self.network,
            make_optax_tx(self.cfg),
            wrt=nnx.Param,
        )

        if self.cfg.steps_per_epoch is None:
            n_train = int(self.dm.X_train.shape[0])
            bsz = int(self.cfg.data.batch_size)
            steps = (n_train + bsz - 1) // bsz
            self.cfg = dataclasses.replace(self.cfg, steps_per_epoch=steps)
            print(f"[init] steps_per_epoch = {steps} (N_train={n_train}, batch_size={bsz})")

    # --------------------------------------------------
    # Train step
    # --------------------------------------------------
    def make_train_step(self):
        """
        Returns a jitted function that runs ONE training step:
        forward -> loss -> grad -> optimizer update.
        """
        @nnx.jit
        def train_step(network: nnx.Module, optimizer: nnx.Optimizer, batch: Batch) -> Metrics:
            x, y, mask = batch

            def _loss_fn(net: nnx.Module):
                outputs = net(x)
                return loss_and_metrics(outputs, y, mask)

            (loss, metrics), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(network)
            optimizer.update(network, grads)
            return metrics

        return train_step

    # --------------------------------------------------
    # Eval step
    # --------------------------------------------------
    def make_eval_step(self):
        @nnx.jit
        def eval_step(network: nnx.Module, batch: Batch) -> Metrics:
            x, y, mask = batch
            outputs = network(x)
            _, metrics = loss_and_metrics(outputs, y, mask)
            return metrics

        return eval_step

    # --------------------------------------------------
    # Predict step
    # --------------------------------------------------
    def make_predict_step(self):
        @nnx.jit
        def predict_step(network: nnx.Module, xb: Array) -> Array:
            return network(xb)

        return predict_step

    
    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self) -> None:
        """
        Train for cfg.epochs epochs.
        Uses cfg.steps_per_epoch to define how many batches form one epoch.
        Reports epoch metrics averaged over valid (unmasked) samples.
        """
        if self.network is None or self.optimizer is None:
            self.init()

        assert self.network is not None
        assert self.optimizer is not None
        assert self.cfg.steps_per_epoch is not None

        steps = int(self.cfg.steps_per_epoch)

        for epoch in range(1, self.cfg.epochs + 1):
            self.network.train()
            train_it = self.dm.train_iter()

            metric_sums: Dict[str, float] = {}
            count_sum = 0.0

            pbar = tqdm(range(steps), desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for _ in pbar:
                batch = next(train_it)
                _, _, mask = batch

                valid_batch_size = float(jnp.sum(mask))
                count_sum += valid_batch_size

                metrics = self._train_step(self.network, self.optimizer, batch)
                metrics_host = jax.device_get(metrics)

                for k, v in metrics_host.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * valid_batch_size

            train_metrics = {k: v / count_sum for k, v in metric_sums.items()}

            val_metrics = self.evaluate()

            msg = f"Epoch {epoch}/{self.cfg.epochs}"
            for k, v in train_metrics.items():
                msg += f"  {k}: {v:.6f}"
            if val_metrics is not None:
                for k, v in val_metrics.items():
                    msg += f"  val_{k}: {v:.6f}"
            tqdm.write(msg)


    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    def evaluate(self) -> Optional[Dict[str, float]]:
        """
        Run the model in eval mode on the test set and return metrics
        averaged over valid (unmasked) samples.
        """
        assert self.network is not None

        self.network.eval()

        metric_sums: Dict[str, float] = {}
        count_sum = 0.0

        for batch in self.dm.test_iter():
            _, _, mask = batch

            valid_batch_size = float(jnp.sum(mask))
            count_sum += valid_batch_size

            metrics = self._eval_step(self.network, batch)
            metrics_host = jax.device_get(metrics)

            for k, v in metrics_host.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * valid_batch_size

        if count_sum == 0:
            return None

        return {k: v / count_sum for k, v in metric_sums.items()}
    

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, X: Array, *, batch_size: Optional[int] = None) -> Array:
        """
        Run the model in eval mode on input X and return raw network outputs.
        For classification these are logits.
        For regression these are predictions.

        Uses padded full-size batches so the jitted predict step always sees
        the same batch shape.
        """
        assert self.network is not None

        self.network.eval()

        X_np = np.asarray(X, dtype=np.float32)
        bs = batch_size or self.cfg.data.batch_size
        n = int(X_np.shape[0])

        # Do one dummy forward pass in case of n=0
        if n == 0:
            dummy_x = np.zeros((bs,) + X_np.shape[1:], dtype=np.float32)
            dummy_out = self._predict_step(self.network, jnp.asarray(dummy_x))
            return dummy_out[:0]

        outs = []
        for start in range(0, n, bs):
            end = min(start + bs, n)
            xb = X_np[start:end]

            actual = end - start
            pad = bs - actual

            if pad > 0:
                xb = np.pad(
                    xb,
                    pad_width=[(0, pad)] + [(0, 0)] * (xb.ndim - 1),
                    mode="constant",
                )

            xb = jnp.asarray(xb)
            out = self._predict_step(self.network, xb)

            # keep only real samples from the padded batch
            outs.append(out[:actual])

        return jnp.concatenate(outs, axis=0)

    # --------------------------------------------------
    # Optional convenience method for classification
    # --------------------------------------------------
    def predict_proba(self, X: Array, *, batch_size: Optional[int] = None) -> Array:
        """
        Optional helper for classification tasks.

        Binary classification:
          returns sigmoid(logits)

        Multiclass classification:
          returns softmax(logits)

        Regression:
          usually should not be used.
        """
        logits = self.predict(X, batch_size=batch_size)

        if logits.ndim == 1:
            return jax.nn.sigmoid(logits)

        if logits.ndim == 2 and logits.shape[-1] == 1:
            return jax.nn.sigmoid(logits)

        return jax.nn.softmax(logits, axis=-1)

    # --------------------------------------------------
    # Save / Load (NNX) Model Params
    # --------------------------------------------------
    def save(self, path: str, *, save_optimizer: bool = False) -> None:
        """
        Save model parameters (and optionally optimizer state) with Orbax.
        """
        assert self.network is not None

        path = os.path.abspath(path)

        payload = {"network": nnx.state(self.network)}
        if save_optimizer:
            assert self.optimizer is not None
            payload["optimizer"] = nnx.state(self.optimizer)

        ocp.PyTreeCheckpointer().save(path, payload, force=True)

    def load(self, path: str, *, load_optimizer: bool = False) -> None:
        """
        Restore model parameters (and optionally optimizer state) with Orbax.
        """
        path = os.path.abspath(path)

        if self.network is None or (load_optimizer and self.optimizer is None):
            self.init()

        assert self.network is not None

        payload = ocp.PyTreeCheckpointer().restore(path)
        nnx.update(self.network, payload["network"])

        if load_optimizer:
            assert self.optimizer is not None
            nnx.update(self.optimizer, payload["optimizer"])




# ============================================================
# Example usage
# ============================================================

# ------------------------------------------------------------
# 1) CSV
# ------------------------------------------------------------
# datacfg = DataConfig(seed=0, batch_size=64, test_size=0.3)
# cfg = Config(
#     seed=0,
#     epochs=50,
#     lr=1e-3,
#     steps_per_epoch=None,
#     in_dim=10,   # number of feature columns
#     out_dim=3,   # multiclass example
#     dropout=0.0,
#     data=datacfg,
# )
# dm = DataModule.from_csv(cfg, csv_path="classification.csv", label_col="y")
# model = Model(cfg, dm, make_my_network)
# model.fit()
# y_pred = model.predict(dm.X_test)
# y_prob = model.predict_proba(dm.X_test)

# ------------------------------------------------------------
# 2) scikit-learn dataset loader
# ------------------------------------------------------------
# from sklearn.datasets import fetch_california_housing
#
# datacfg = DataConfig(seed=0, batch_size=64, test_size=0.2)
# cfg = Config(
#     seed=0,
#     epochs=50,
#     lr=1e-3,
#     steps_per_epoch=None,
#     in_dim=8,
#     out_dim=1,   # regression example
#     dropout=0.0,
#     data=datacfg,
# )
# dm = DataModule.from_sklearn(cfg, loader=fetch_california_housing)
# model = Model(cfg, dm, make_my_network)
# model.fit()
# y_pred = model.predict(dm.X_test)

# ------------------------------------------------------------
# 3) Plain X and y arrays
# ------------------------------------------------------------
# X = ...  # shape (N, in_dim)
# y = ...  # shape depends on task
#
# datacfg = DataConfig(seed=0, batch_size=64, test_size=0.3)
# cfg = Config(
#     seed=0,
#     epochs=50,
#     lr=1e-3,
#     steps_per_epoch=None,
#     in_dim=X.shape[1],
#     out_dim=1,
#     dropout=0.0,
#     data=datacfg,
# )
# dm = DataModule.from_arrays(cfg, X=X, y=y)
# model = Model(cfg, dm, make_my_network)
# model.fit()
# y_pred = model.predict(dm.X_test)