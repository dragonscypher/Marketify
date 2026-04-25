from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from marketify.config import ModelConfig


class RidgeModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = Ridge(alpha=1.0)

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

    x = frame[feature_cols].to_numpy()
    y = frame[target_col].to_numpy()
    preds = np.full(len(frame), np.nan, dtype=float)

    start = config.train_window
    while start < len(frame):
        train_start = max(0, start - config.train_window)
        x_train = x[train_start:start]
        y_train = y[train_start:start]

        model = RidgeModel(config)
        model.fit(x_train, y_train)

        end_pred = min(len(frame), start + config.retrain_every)
        preds[start:end_pred] = model.predict(x[start:end_pred])
        start = end_pred

    return pd.Series(preds, index=frame.index, name="pred_next_ret")
