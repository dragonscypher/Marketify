from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class NewsItem:
    title: str
    source: str
    published_at: str
    sentiment_hint: float = 0.0


class NewsProvider:
    """Provider stub for future Alpha Vantage/Finnhub integrations."""

    def __init__(self, alpha_vantage_key: str | None = None, finnhub_key: str | None = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")

    def fetch(self, symbol: str) -> list[NewsItem]:
        if not self.alpha_vantage_key and not self.finnhub_key:
            return []
        return []

    def summarize_risk(self, symbol: str) -> dict[str, Any]:
        items = self.fetch(symbol)
        if not items:
            return {
                "news_risk": 0.0,
                "headline_count": 0,
                "summary": "No API key/news feed configured; using neutral news risk.",
            }
        risk = min(1.0, 0.2 + len(items) * 0.05)
        return {
            "news_risk": risk,
            "headline_count": len(items),
            "summary": f"News feed active with {len(items)} recent headlines.",
        }
