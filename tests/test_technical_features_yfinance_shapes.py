"""Test that add_technical_features handles yfinance MultiIndex columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketify.features.technical import add_technical_features

EXPECTED_COLS = [
    "ema_10",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "atr_pct",
    "vol_20",
    "ret_30",
    "volume_regime_ratio",
    "breakout_up_20",
    "breakout_down_20",
    "news_headline_count",
    "vix_proxy",
    "macro_regime_neutral",
]


def _base_ohlcv(rows: int = 200):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-02 09:30", periods=rows, freq="5min")
    base = 180.0 + rng.normal(0, 0.5, rows).cumsum()
    return idx, base, rng


def _assert_features(result):
    assert not result.empty
    for col in EXPECTED_COLS:
        assert col in result.columns, f"Missing expected column: {col}"


def test_flat_columns():
    """Normal flat-column DataFrame."""
    idx, base, rng = _base_ohlcv()
    df = pd.DataFrame(
        {
            "Close": base,
            "High": base + rng.uniform(0.1, 1.0, 200),
            "Low": base - rng.uniform(0.1, 1.0, 200),
            "Open": base + rng.normal(0, 0.3, 200),
            "Volume": rng.integers(1000, 50000, 200).astype(float),
        },
        index=idx,
    )
    _assert_features(add_technical_features(df))


def test_multiindex_field_first():
    """yfinance MultiIndex: ('Close', 'AAPL') — field first."""
    idx, base, rng = _base_ohlcv()
    data = {
        ("Close", "AAPL"): base,
        ("High", "AAPL"): base + rng.uniform(0.1, 1.0, 200),
        ("Low", "AAPL"): base - rng.uniform(0.1, 1.0, 200),
        ("Open", "AAPL"): base + rng.normal(0, 0.3, 200),
        ("Volume", "AAPL"): rng.integers(1000, 50000, 200).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    _assert_features(add_technical_features(df))


def test_multiindex_ticker_first():
    """yfinance MultiIndex: ('AAPL', 'Close') — ticker first."""
    idx, base, rng = _base_ohlcv()
    data = {
        ("AAPL", "Close"): base,
        ("AAPL", "High"): base + rng.uniform(0.1, 1.0, 200),
        ("AAPL", "Low"): base - rng.uniform(0.1, 1.0, 200),
        ("AAPL", "Open"): base + rng.normal(0, 0.3, 200),
        ("AAPL", "Volume"): rng.integers(1000, 50000, 200).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    _assert_features(add_technical_features(df))


def test_single_col_dataframe_close():
    """Selecting 'Close' from MultiIndex gives single-column DataFrame."""
    idx, base, rng = _base_ohlcv()
    data = {
        ("Close", "AAPL"): base,
        ("High", "AAPL"): base + rng.uniform(0.1, 1.0, 200),
        ("Low", "AAPL"): base - rng.uniform(0.1, 1.0, 200),
        ("Open", "AAPL"): base + rng.normal(0, 0.3, 200),
        ("Volume", "AAPL"): rng.integers(1000, 50000, 200).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Verify the problem scenario: df['Close'] is a DataFrame
    close = df["Close"]
    assert isinstance(close, pd.DataFrame), "Expected DataFrame for MultiIndex selection"
    assert close.ndim == 2
    # But add_technical_features handles it
    _assert_features(add_technical_features(df))
