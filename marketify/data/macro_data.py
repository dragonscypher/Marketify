"""Macro / market-regime data stub.

Returns neutral regime signals when no API key is configured.
Never raises on missing keys or network errors — always safe to import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RegimeSignal:
    regime: str           # "bull" | "bear" | "neutral"
    volatility_level: str  # "low" | "medium" | "high"
    vix_proxy: float      # 0-100 estimate
    trend_signal: float   # -1.0 … 1.0


class MacroProvider:
    """Macro/event-calendar provider.

    Currently a safe stub that returns neutral regime at all times.
    A future implementation may fetch VIX, FRED economic calendar,
    or Alpha Vantage macro indicators when an API key is present.

    All public methods are guaranteed not to raise.
    """

    def __init__(self, alpha_vantage_key: str | None = None) -> None:
        self.key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_regime(self, symbol: str = "AAPL") -> RegimeSignal:  # noqa: ARG002
        """Return current market-regime signal.

        Returns a neutral signal when no live data is available so that
        downstream consumers can always proceed safely.
        """
        try:
            return self._fetch_regime(symbol)
        except Exception:  # pylint: disable=broad-except
            return self._neutral_regime()

    def regime_features(self, symbol: str = "AAPL") -> dict[str, float]:
        """Return scalar regime features for use in fusion model."""
        r = self.get_regime(symbol)
        return {
            "vix_proxy": float(r.vix_proxy),
            "regime_bull": 1.0 if r.regime == "bull" else 0.0,
            "regime_bear": 1.0 if r.regime == "bear" else 0.0,
            "regime_neutral": 1.0 if r.regime == "neutral" else 0.0,
            "vol_level_high": 1.0 if r.volatility_level == "high" else 0.0,
            "trend_signal": float(r.trend_signal),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_regime(self, symbol: str) -> RegimeSignal:  # noqa: ARG002
        """Attempt live fetch. Raises on failure so caller falls back."""
        if not self.key:
            return self._neutral_regime()
        # Placeholder: real integration would call Alpha Vantage / FRED here.
        return self._neutral_regime()

    @staticmethod
    def _neutral_regime() -> RegimeSignal:
        return RegimeSignal(
            regime="neutral",
            volatility_level="medium",
            vix_proxy=20.0,
            trend_signal=0.0,
        )
