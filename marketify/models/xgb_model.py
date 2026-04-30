from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from marketify.config import ModelConfig


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


class XGBModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        low_ram = _env_flag("LOCAL_LOW_RAM_MODE")
        force_gpu = _env_flag("LOCAL_FORCE_GPU")
        kwargs = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "objective": "reg:squarederror",
            "random_state": config.random_state,
            "n_jobs": 1 if low_ram else -1,
            "tree_method": "hist",
        }
        if force_gpu:
            kwargs["device"] = "cuda"
        self.model = XGBRegressor(**kwargs)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        return self.model.predict(x_pred)


def rolling_train_predict(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: ModelConfig,
) -> pd.Series:
    if len(frame) <= config.train_window + 10:
        raise ValueError("Not enough rows for rolling training. Use larger period or smaller train_window.")

    low_ram = _env_flag("LOCAL_LOW_RAM_MODE")
    dtype = np.float32 if low_ram else float
    x = frame[feature_cols].to_numpy(dtype=dtype, copy=False)
    y = frame[target_col].to_numpy(dtype=dtype, copy=False)
    preds = np.full(len(frame), np.nan, dtype=np.float32 if low_ram else float)

    start = config.train_window
    while start < len(frame):
        train_start = max(0, start - config.train_window)
        x_train = x[train_start:start]
        y_train = y[train_start:start]

        model = XGBModel(config)
        model.fit(x_train, y_train)

        end_pred = min(len(frame), start + config.retrain_every)
        preds[start:end_pred] = model.predict(x[start:end_pred])
        if low_ram:
            del model
            gc.collect()
        start = end_pred

    return pd.Series(preds, index=frame.index, name="pred_next_ret")
