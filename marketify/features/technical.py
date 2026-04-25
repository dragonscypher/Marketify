from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

from marketify.data.macro_data import MacroProvider
from marketify.data.news_data import NewsProvider


def _as_1d_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract *col* from *df* as a guaranteed 1-D numeric pandas Series.

    Handles every yfinance column layout:
    1. Normal flat columns  – ``df['Close']`` is already a Series.
    2. MultiIndex field-first – ``('Close', 'AAPL')``
    3. MultiIndex ticker-first – ``('AAPL', 'Close')``
    4. Already-squeezed Series from a prior flatten step.
    """
    if isinstance(df.columns, pd.MultiIndex):
        candidates = [c for c in df.columns
                      if isinstance(c, tuple) and col in c]
        if not candidates:
            raise KeyError(f"Missing {col!r}. Columns={list(df.columns)[:10]}")
        s = df.loc[:, candidates[0]]
    else:
        s = df[col]

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.squeeze()
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)
    s = pd.to_numeric(s, errors="coerce")
    s.index = df.index
    s.name = col

    if getattr(s, "ndim", 1) != 1:
        raise ValueError(f"{col} not 1-D after normalization. shape={getattr(s, 'shape', None)}")
    return s


TECHNICAL_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_30",
    "ema_dist_10",
    "ema_dist_20",
    "ema_dist_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "atr_pct",
    "vol_20",
    "intrabar_range_pct",
    "close_location",
    "volume_z20",
    "dollar_volume_z20",
    "volume_regime_ratio",
    "vol_regime_ratio",
    "volatility_regime_code",
    "breakout_up_20",
    "breakout_down_20",
    "ret_vs_spy_5",
    "ret_vs_spy_20",
    "ret_vs_sector_5",
    "ret_vs_sector_20",
    "signal_quality_20",
    "signal_stability_20",
    "trend_alignment_score",
    "news_risk",
    "news_sentiment_score",
    "news_headline_count",
    "news_event_shock",
    "vix_proxy",
    "macro_event_risk",
    "macro_event_window",
    "macro_policy_bias",
    "macro_trend_signal",
    "macro_regime_bull",
    "macro_regime_bear",
    "macro_regime_neutral",
    "macro_vol_level_high",
    "sin_hour",
    "cos_hour",
    "sin_min",
    "cos_min",
    "is_morning",
    "is_afternoon",
    "is_late",
]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - mean) / std


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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
    out["ret_30"] = close.pct_change(30)

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
    out["atr_pct"] = out["atr_14"] / close.replace(0.0, np.nan)
    out["vol_20"] = close.pct_change().rolling(20).std()
    intrabar_range = (high - low).clip(lower=0.0)
    out["intrabar_range_pct"] = intrabar_range / close.replace(0.0, np.nan)
    out["close_location"] = ((close - low) / intrabar_range.replace(0.0, np.nan)).clip(0.0, 1.0) - 0.5
    out["volume_z20"] = _rolling_zscore(volume, 20)
    out["dollar_volume_z20"] = _rolling_zscore(close * volume, 20)
    out["volume_regime_ratio"] = volume / volume.rolling(60).median().replace(0.0, np.nan)
    out["vol_regime_ratio"] = out["vol_20"] / out["vol_20"].rolling(60).median().replace(0.0, np.nan)
    out["volatility_regime_code"] = np.select(
        [out["vol_regime_ratio"] > 1.25, out["vol_regime_ratio"] < 0.75],
        [1.0, -1.0],
        default=0.0,
    )
    out["breakout_up_20"] = close / close.rolling(20).max().replace(0.0, np.nan) - 1.0
    out["breakout_down_20"] = close / close.rolling(20).min().replace(0.0, np.nan) - 1.0

    if "SPY_Close" in out.columns:
        spy_close = _as_1d_series(out, "SPY_Close")
        out["ret_vs_spy_5"] = close.pct_change(5) - spy_close.pct_change(5)
        out["ret_vs_spy_20"] = close.pct_change(20) - spy_close.pct_change(20)
    else:
        out["ret_vs_spy_5"] = 0.0
        out["ret_vs_spy_20"] = 0.0

    if "XLK_Close" in out.columns:
        sector_close = _as_1d_series(out, "XLK_Close")
        out["ret_vs_sector_5"] = close.pct_change(5) - sector_close.pct_change(5)
        out["ret_vs_sector_20"] = close.pct_change(20) - sector_close.pct_change(20)
    else:
        out["ret_vs_sector_5"] = 0.0
        out["ret_vs_sector_20"] = 0.0

    out["signal_quality_20"] = close.pct_change(5).abs() / (out["vol_20"] + 1e-9)
    out["signal_stability_20"] = close.pct_change().rolling(20).mean().abs() / (
        close.pct_change().rolling(20).std() + 1e-9
    )

    out["minute"] = out.index.minute
    out["hour"] = out.index.hour
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_min"] = np.sin(2 * np.pi * out["minute"] / 60.0)
    out["cos_min"] = np.cos(2 * np.pi * out["minute"] / 60.0)
    out["is_morning"] = ((out["hour"] >= 6) & (out["hour"] <= 11)).astype(int)
    out["is_afternoon"] = ((out["hour"] >= 12) & (out["hour"] <= 15)).astype(int)
    out["is_late"] = ((out["hour"] >= 16) | (out["hour"] <= 5)).astype(int)
    out["trend_alignment_score"] = (
        np.sign(out["ema_dist_10"].fillna(0.0))
        + np.sign(out["ema_dist_20"].fillna(0.0))
        + np.sign(out["macd_hist"].fillna(0.0))
    ) / 3.0

    ticker = str(out.attrs.get("ticker", "AAPL"))
    news = NewsProvider().summarize_risk(ticker)
    macro = MacroProvider().regime_features(ticker)
    out["news_risk"] = float(news.get("news_risk", 0.0))
    out["news_sentiment_score"] = float(news.get("sentiment_score", 0.0))
    out["news_headline_count"] = float(news.get("headline_count", 0.0))
    out["news_event_shock"] = float(news.get("event_shock_flag", 0.0))
    out["macro_event_risk"] = float(macro.get("macro_event_risk", 0.0))
    out["macro_event_window"] = float(macro.get("macro_event_window", 0.0))
    out["macro_policy_bias"] = float(macro.get("macro_policy_bias", 0.0))
    out["macro_trend_signal"] = float(macro.get("trend_signal", 0.0))
    out["macro_regime_bull"] = float(macro.get("regime_bull", 0.0))
    out["macro_regime_bear"] = float(macro.get("regime_bear", 0.0))
    out["macro_regime_neutral"] = float(macro.get("regime_neutral", 0.0))
    out["macro_vol_level_high"] = float(macro.get("vol_level_high", 0.0))
    if "vix_proxy" not in out.columns:
        out["vix_proxy"] = float(macro.get("vix_proxy", 20.0))

    out["target_next_ret"] = close.pct_change().shift(-1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out
