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
from marketify.data.market_data import fetch_market_data
from marketify.features.technical import TECHNICAL_FEATURE_COLUMNS, add_technical_features
from marketify.models.xgb_model import XGBModel


ARTIFACT_DIR = Path("artifacts")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_xgb(feat: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    x = feat[TECHNICAL_FEATURE_COLUMNS].to_numpy()
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
    x = feat[TECHNICAL_FEATURE_COLUMNS].to_numpy()
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


def train_if_missing(config: AppConfig | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Train all models if artifacts missing (or force=True). Return leaderboard."""
    if config is None:
        config = AppConfig()

    ticker = config.data.ticker
    xgb_path = ARTIFACT_DIR / f"xgb_{ticker}.pkl"
    ridge_path = ARTIFACT_DIR / f"ridge_{ticker}.pkl"

    if not force and xgb_path.exists() and ridge_path.exists():
        return _load_leaderboard(ticker)

    raw = fetch_market_data(
        ticker=ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
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
