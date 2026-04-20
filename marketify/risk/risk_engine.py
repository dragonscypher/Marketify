from __future__ import annotations

from dataclasses import dataclass

from marketify.config import BrokerConfig, RiskConfig, TradeIdea


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]


class RiskEngine:
    def __init__(self, risk_config: RiskConfig, broker_config: BrokerConfig):
        self.risk_config = risk_config
        self.broker_config = broker_config

    def evaluate(
        self,
        trade: TradeIdea,
        account: dict,
        volatility: float,
        news_risk: float,
        mark_price: float,
    ) -> RiskDecision:
        reasons: list[str] = []

        equity = float(account.get("equity", 0.0))
        drawdown_pct = float(account.get("drawdown_pct", 0.0))
        daily_loss_pct = float(account.get("daily_loss_pct", 0.0))

        trade_notional = trade.qty * mark_price
        max_notional = equity * self.broker_config.max_position_fraction
        if trade_notional > max_notional:
            reasons.append("position_size_exceeds_limit")

        if trade.expected_return < self.risk_config.min_expected_return:
            reasons.append("expected_return_below_threshold")

        if trade.cvar_95 > self.risk_config.max_cvar_95:
            reasons.append("cvar_limit_breached")

        if drawdown_pct > self.risk_config.max_drawdown_pct:
            reasons.append("max_drawdown_breached")

        if daily_loss_pct > self.risk_config.max_daily_loss_pct:
            reasons.append("max_daily_loss_breached")

        if volatility > self.risk_config.max_volatility:
            reasons.append("volatility_too_high")

        if news_risk > self.risk_config.max_news_risk:
            reasons.append("news_risk_too_high")

        return RiskDecision(approved=len(reasons) == 0, reasons=reasons)
