from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketify.config import BrokerConfig, RiskConfig, TradeIdea


@dataclass
class TradeIdeaInput:
    symbol: str
    last_price: float
    prediction: float
    sentiment_score: float
    volatility: float
    news_risk: float


def compute_cvar_95(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    q = returns.quantile(0.05)
    tail = returns[returns <= q]
    if tail.empty:
        return 0.0
    return float(abs(tail.mean()))


def build_trade_idea(
    info: TradeIdeaInput,
    equity: float,
    broker_cfg: BrokerConfig,
    risk_cfg: RiskConfig,
    cvar_95: float,
) -> TradeIdea | None:
    adjusted_pred = info.prediction * (1.0 + 0.2 * info.sentiment_score)

    if adjusted_pred > 0:
        side = "buy"
    elif adjusted_pred < 0 and broker_cfg.allow_shorts:
        side = "sell"
    else:
        return None

    notional_cap = max(0.0, equity * broker_cfg.max_position_fraction)
    qty = int(notional_cap / info.last_price) if info.last_price > 0 else 0
    if qty <= 0:
        return None

    confidence = float(np.clip(abs(adjusted_pred) / max(risk_cfg.min_expected_return, 1e-6), 0.0, 1.0))
    reason = (
        f"Model prediction={adjusted_pred:.5f}, sentiment={info.sentiment_score:.2f}, "
        f"vol={info.volatility:.4f}, news_risk={info.news_risk:.2f}"
    )

    if side == "buy":
        stop_loss = info.last_price * (1.0 - broker_cfg.stop_loss_pct)
        take_profit = info.last_price * (1.0 + broker_cfg.take_profit_pct)
    else:
        stop_loss = info.last_price * (1.0 + broker_cfg.stop_loss_pct)
        take_profit = info.last_price * (1.0 - broker_cfg.take_profit_pct)

    return TradeIdea(
        symbol=info.symbol,
        side=side,
        qty=qty,
        confidence=confidence,
        expected_return=float(adjusted_pred),
        cvar_95=float(cvar_95),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        reason=reason,
        risk_warning="Experimental suggestion. Approval required before paper order.",
    )
