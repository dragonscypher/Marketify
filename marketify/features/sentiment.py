from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marketify.data.news_data import NewsItem


@dataclass
class SentimentResult:
    score: float
    label: str
    reason: str


class FinBERTSentiment:
    """Stub wrapper. Returns neutral when model/API unavailable."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def analyze(self, text: str) -> SentimentResult:  # noqa: ARG002
        if not self.enabled:
            return SentimentResult(score=0.0, label="neutral", reason="Sentiment model disabled; neutral fallback.")
        return SentimentResult(score=0.0, label="neutral", reason="FinBERT integration stub.")


# ---------------------------------------------------------------------------
# Feature-extraction helpers used by fusion model
# ---------------------------------------------------------------------------


def compute_sentiment_features(items: list[NewsItem]) -> dict[str, float]:
    """Aggregate news items into scalar features for the fusion model.

    Returns safe neutral defaults for empty / missing inputs.
    Never raises.
    """
    if not items:
        return {
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "sentiment_positive_frac": 0.0,
            "sentiment_negative_frac": 0.0,
            "headline_count": 0.0,
        }

    scores = [float(item.sentiment_hint) for item in items]
    n = len(scores)
    mean_s = sum(scores) / n
    variance = sum((s - mean_s) ** 2 for s in scores) / max(n - 1, 1)
    std_s = variance ** 0.5
    pos_frac = sum(1 for s in scores if s > 0.1) / n
    neg_frac = sum(1 for s in scores if s < -0.1) / n

    return {
        "sentiment_mean": float(mean_s),
        "sentiment_std": float(std_s),
        "sentiment_positive_frac": float(pos_frac),
        "sentiment_negative_frac": float(neg_frac),
        "headline_count": float(n),
    }


def news_risk_gate(news_risk: float, max_news_risk: float = 0.75) -> bool:
    """Return True when news risk exceeds threshold → should abstain from trading.

    Deterministic — no model weights, no self-modification.
    """
    return float(news_risk) > float(max_news_risk)
