"""Test that add_technical_features handles yfinance MultiIndex columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketify.features.technical import add_technical_features


def _make_yfinance_multiindex_df(rows: int = 200) -> pd.DataFrame:
    """Build a DataFrame that mimics yfinance single-ticker MultiIndex output."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-02 09:30", periods=rows, freq="5min")

    base = 180.0 + rng.normal(0, 0.5, rows).cumsum()
    data = {
        ("Close", "AAPL"): base,
        ("High", "AAPL"): base + rng.uniform(0.1, 1.0, rows),
        ("Low", "AAPL"): base - rng.uniform(0.1, 1.0, rows),
        ("Open", "AAPL"): base + rng.normal(0, 0.3, rows),
        ("Volume", "AAPL"): rng.integers(1000, 50000, rows).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_multiindex_produces_features():
    df = _make_yfinance_multiindex_df(200)
    result = add_technical_features(df)

    assert not result.empty
    for col in ["ema_10", "rsi_14", "macd", "atr_14"]:
        assert col in result.columns, f"Missing expected column: {col}"


def test_flat_columns_still_work():
    """Normal flat-column DataFrame must still work."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2025-03-01 09:30", periods=200, freq="5min")
    base = 100.0 + rng.normal(0, 0.3, 200).cumsum()
    df = pd.DataFrame(
        {
            "Close": base,
            "High": base + rng.uniform(0.05, 0.5, 200),
            "Low": base - rng.uniform(0.05, 0.5, 200),
            "Open": base + rng.normal(0, 0.2, 200),
            "Volume": rng.integers(500, 30000, 200).astype(float),
        },
        index=idx,
    )
    result = add_technical_features(df)
    assert not result.empty
    assert "ema_10" in result.columns
