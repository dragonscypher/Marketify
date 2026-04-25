from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf


def _normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No market data returned. Try different period/interval.")
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _extract_close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        candidates = [c for c in df.columns if isinstance(c, tuple) and "Close" in c]
        if not candidates:
            raise KeyError("Missing Close column in reference market data.")
        series = df.loc[:, candidates[0]]
    else:
        series = df["Close"]

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.to_numeric(series.squeeze(), errors="coerce")
    series.index = df.index
    series.name = "Close"
    return series


@lru_cache(maxsize=32)
def _download_cached(ticker: str, interval: str, period: str, prepost: bool) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        prepost=prepost,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return _normalize_download(df)


def _maybe_load_reference_close(
    symbol: str,
    interval: str,
    period: str,
    prepost: bool,
    target_index: pd.Index,
) -> pd.Series | None:
    try:
        ref = _download_cached(symbol, interval, period, prepost)
        close = _extract_close(ref).reindex(target_index).ffill().bfill()
    except Exception:  # pylint: disable=broad-except
        return None

    if close.isna().all():
        return None
    return close


def fetch_market_data(ticker: str, interval: str, period: str, prepost: bool) -> pd.DataFrame:
    """Research/backtest fallback market data source using yfinance."""
    df = _download_cached(ticker, interval, period, prepost).copy()

    spy_close = _maybe_load_reference_close("SPY", interval, period, prepost, df.index)
    if spy_close is not None and ticker.upper() != "SPY":
        df["SPY_Close"] = spy_close.astype(float)

    sector_close = _maybe_load_reference_close("XLK", interval, period, prepost, df.index)
    if sector_close is not None and ticker.upper() != "XLK":
        df["XLK_Close"] = sector_close.astype(float)

    df.attrs["ticker"] = ticker.upper()
    df.attrs["index_symbol"] = "SPY"
    df.attrs["sector_symbol"] = "XLK"
    return df
