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


# Neutral result returned when no feed is available.
NEUTRAL_SUMMARY: dict[str, Any] = {
    "news_risk": 0.0,
    "sentiment_score": 0.0,
    "event_shock_flag": 0.0,
    "headline_count": 0,
    "summary": "No API key/news feed configured; using neutral news risk.",
}


class NewsProvider:
    """Provider stub for future Alpha Vantage/Finnhub integrations.

    All methods return neutral/empty results when no API key is present.
    Never raises on missing keys — safe to use in paper-trading pipeline.
    """

    def __init__(self, alpha_vantage_key: str | None = None, finnhub_key: str | None = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")

    def fetch(self, symbol: str) -> list[NewsItem]:  # noqa: ARG002
        """Fetch recent news items for *symbol*.

        Returns empty list when no API key configured.
        A future implementation would call Alpha Vantage / Finnhub here.
        """
        if not self.alpha_vantage_key and not self.finnhub_key:
            return []
        # Placeholder: real integration would populate items here.
        return []

    def compute_aggregate_sentiment(self, items: list[NewsItem]) -> float:
        """Average sentiment_hint across all items. Returns 0.0 for empty list."""
        if not items:
            return 0.0
        total = sum(item.sentiment_hint for item in items)
        return float(total / len(items))

    def summarize_risk(self, symbol: str) -> dict[str, Any]:
        """Return news-risk summary dict.

        news_risk is 0.0 when no feed; never > 1.0.
        """
        try:
            items = self.fetch(symbol)
        except Exception:  # pylint: disable=broad-except
            return dict(NEUTRAL_SUMMARY)

        if not items:
            return dict(NEUTRAL_SUMMARY)

        sentiment = self.compute_aggregate_sentiment(items)
        risk = min(1.0, 0.2 + len(items) * 0.05 + min(abs(sentiment), 1.0) * 0.15)
        event_shock_flag = 1.0 if risk >= 0.75 or abs(sentiment) >= 0.6 else 0.0
        return {
            "news_risk": risk,
            "sentiment_score": sentiment,
            "event_shock_flag": event_shock_flag,
            "headline_count": len(items),
            "summary": f"News feed active with {len(items)} recent headlines.",
        }
