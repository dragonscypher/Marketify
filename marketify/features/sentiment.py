from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SentimentResult:
    score: float
    label: str
    reason: str


class FinBERTSentiment:
    """Stub wrapper. Returns neutral when model/API unavailable."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def analyze(self, text: str) -> SentimentResult:
        if not self.enabled:
            return SentimentResult(score=0.0, label="neutral", reason="Sentiment model disabled; neutral fallback.")
        return SentimentResult(score=0.0, label="neutral", reason="FinBERT integration stub.")
