"""Train-if-missing pipeline.

Trains XGBoost (and optionally Ridge) on the configured ticker,
saves artifacts to disk, and returns a leaderboard dict.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from marketify.config import AppConfig

ARTIFACT_DIR = Path("artifacts")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_technical_feature_columns() -> list[str]:
    try:
        from marketify.features.technical import TECHNICAL_FEATURE_COLUMNS
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "ta":
            raise
        return [
            "ret_1",
            "ret_5",
            "ret_15",
            "ema_dist_10",
            "ema_dist_20",
            "ema_dist_50",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "vol_20",
            "sin_hour",
            "cos_hour",
            "sin_min",
            "cos_min",
            "is_morning",
            "is_afternoon",
            "is_late",
        ]
    return list(TECHNICAL_FEATURE_COLUMNS)


def train_xgb(feat: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    from marketify.models.xgb_model import XGBModel

    technical_feature_columns = _get_technical_feature_columns()
    x = feat[technical_feature_columns].to_numpy()
    y = feat["target_next_ret"].to_numpy()

    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = XGBModel(config.model)
    model.fit(x_train, y_train)

    preds_val = model.predict(x_val)
    mae = float(mean_absolute_error(y_val, preds_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))

    artifact_path = _ensure_dir(ARTIFACT_DIR) / f"xgb_{config.data.ticker}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    return {"model": "xgb", "ticker": config.data.ticker, "mae": mae, "rmse": rmse, "artifact": str(artifact_path)}


def train_ridge(feat: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    technical_feature_columns = _get_technical_feature_columns()
    x = feat[technical_feature_columns].to_numpy()
    y = feat["target_next_ret"].to_numpy()

    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)

    preds_val = model.predict(x_val)
    mae = float(mean_absolute_error(y_val, preds_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))

    artifact_path = _ensure_dir(ARTIFACT_DIR) / f"ridge_{config.data.ticker}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    return {"model": "ridge", "ticker": config.data.ticker, "mae": mae, "rmse": rmse, "artifact": str(artifact_path)}


def train_recurrent_baselines(
    feat: pd.DataFrame,
    config: AppConfig,
    hold_horizon_bars: int | None = None,
    label_mode: str = "return",
) -> list[dict[str, Any]]:
    """Train GRU/LSTM baselines with horizon-aligned labels.

    Lazy-import torch path so module import stays clean on local machines without
    recurrent dependencies installed.
    """

    from marketify.models.rnn_model import (RecurrentModelConfig,
                                            build_sequence_dataset,
                                            make_recurrent_model,
                                            prepare_recurrent_training_frame)
    from scripts.run_exit_rule_experiment import \
        split_for_validation_experiment

    if hold_horizon_bars is None:
        hold_horizon_bars = config.broker.time_stop_bars

    aligned, feature_cols, target_col = prepare_recurrent_training_frame(
        feat,
        hold_horizon_bars=hold_horizon_bars,
        mode=label_mode,
    )
    train_val, validation_index, split_meta = split_for_validation_experiment(aligned)
    train_index = train_val.index[:split_meta.train_end]

    leaderboard: list[dict[str, Any]] = []
    for architecture in ("gru", "lstm"):
        model_cfg = RecurrentModelConfig(architecture=architecture)
        model = make_recurrent_model(architecture, model_cfg)
        x_train, y_train, _train_ts = build_sequence_dataset(
            train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=model_cfg.sequence_length,
            selected_index=train_index,
        )
        model.fit(x_train, y_train)
        x_val, y_val, _val_ts = build_sequence_dataset(
            train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=model_cfg.sequence_length,
            selected_index=validation_index,
        )
        preds_val = model.predict(x_val)
        mae = float(mean_absolute_error(y_val, preds_val)) if len(y_val) > 0 else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y_val, preds_val))) if len(y_val) > 0 else float("nan")

        artifact_path = _ensure_dir(ARTIFACT_DIR) / f"{architecture}_{config.data.ticker}_h{hold_horizon_bars}_{label_mode}.pt"
        model.save(
            artifact_path,
            feature_cols=feature_cols,
            target_col=target_col,
            extra_meta={
                "ticker": config.data.ticker,
                "hold_horizon_bars": hold_horizon_bars,
                "label_mode": label_mode,
            },
        )
        leaderboard.append(
            {
                "model": architecture,
                "ticker": config.data.ticker,
                "mae": mae,
                "rmse": rmse,
                "artifact": str(artifact_path),
            }
        )

    leaderboard.sort(key=lambda item: item["mae"])
    return leaderboard


def train_if_missing(config: AppConfig | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Train all models if artifacts missing (or force=True). Return leaderboard."""
    if config is None:
        config = AppConfig()

    ticker = config.data.ticker
    xgb_path = ARTIFACT_DIR / f"xgb_{ticker}.pkl"
    ridge_path = ARTIFACT_DIR / f"ridge_{ticker}.pkl"

    if not force and xgb_path.exists() and ridge_path.exists():
        return _load_leaderboard(ticker)

    from marketify.data.market_data import fetch_market_data

    raw = fetch_market_data(
        ticker=ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    from marketify.features.technical import add_technical_features

    feat = add_technical_features(raw)

    leaderboard: list[dict[str, Any]] = []
    leaderboard.append(train_xgb(feat, config))
    leaderboard.append(train_ridge(feat, config))

    # Sort by MAE ascending (lower = better)
    leaderboard.sort(key=lambda x: x["mae"])

    # Save leaderboard
    lb_path = _ensure_dir(ARTIFACT_DIR) / f"leaderboard_{ticker}.json"
    with open(lb_path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    return leaderboard


def _load_leaderboard(ticker: str) -> list[dict[str, Any]]:
    lb_path = ARTIFACT_DIR / f"leaderboard_{ticker}.json"
    if lb_path.exists():
        with open(lb_path) as f:
            return json.load(f)
    return []


def load_model(model_name: str, ticker: str):
    """Load a trained model artifact from disk."""
    artifact_path = ARTIFACT_DIR / f"{model_name}_{ticker}.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No artifact at {artifact_path}. Run train_if_missing first.")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)
