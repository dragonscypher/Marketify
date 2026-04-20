from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_market_data(ticker: str, interval: str, period: str, prepost: bool) -> pd.DataFrame:
    """Research/backtest fallback market data source using yfinance."""
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        prepost=prepost,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise ValueError("No market data returned. Try different period/interval.")
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df
