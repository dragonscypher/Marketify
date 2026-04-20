from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange


def _as_1d_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract *col* from *df* as a guaranteed 1-D numeric pandas Series.

    Handles three yfinance column layouts:
    1. Normal flat columns  – ``df['Close']`` is already a Series.
    2. MultiIndex columns   – ``df['Close']`` returns a single-column DataFrame
       (e.g. shape (N, 1)) because the real column is ``('Close', 'AAPL')``.
    3. Already-squeezed Series from a prior flatten step.

    After extraction the values are coerced to numeric and the result is
    renamed so downstream code always sees a clean ``pd.Series`` named *col*.
    """
    s = df[col]
    # MultiIndex selection can yield a DataFrame with one column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.name = col
    return s


TECHNICAL_FEATURE_COLUMNS = [
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


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Flatten MultiIndex columns if present (yfinance single-ticker quirk)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    close = _as_1d_series(out, "Close")
    high = _as_1d_series(out, "High")
    low = _as_1d_series(out, "Low")
    volume = _as_1d_series(out, "Volume")

    assert close.ndim == 1, f"close must be 1-D, got {close.ndim}"
    assert high.ndim == 1, f"high must be 1-D, got {high.ndim}"
    assert low.ndim == 1, f"low must be 1-D, got {low.ndim}"
    assert volume.ndim == 1, f"volume must be 1-D, got {volume.ndim}"

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)

    out["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
    out["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    out["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    out["ema_dist_10"] = (close - out["ema_10"]) / close
    out["ema_dist_20"] = (close - out["ema_20"]) / close
    out["ema_dist_50"] = (close - out["ema_50"]) / close

    out["rsi_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    out["vol_20"] = close.pct_change().rolling(20).std()

    out["minute"] = out.index.minute
    out["hour"] = out.index.hour
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_min"] = np.sin(2 * np.pi * out["minute"] / 60.0)
    out["cos_min"] = np.cos(2 * np.pi * out["minute"] / 60.0)
    out["is_morning"] = ((out["hour"] >= 6) & (out["hour"] <= 11)).astype(int)
    out["is_afternoon"] = ((out["hour"] >= 12) & (out["hour"] <= 15)).astype(int)
    out["is_late"] = ((out["hour"] >= 16) | (out["hour"] <= 5)).astype(int)

    out["target_next_ret"] = close.pct_change().shift(-1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out
