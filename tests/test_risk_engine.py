from marketify.config import BrokerConfig, RiskConfig, TradeIdea
from marketify.risk.risk_engine import RiskEngine


def test_risk_engine_rejects_oversized_trade():
    broker_cfg = BrokerConfig(max_position_fraction=0.1)
    risk_cfg = RiskConfig(min_expected_return=0.0001)
    engine = RiskEngine(risk_cfg, broker_cfg)

    idea = TradeIdea(
        symbol="AAPL",
        side="buy",
        qty=50,
        confidence=0.8,
        expected_return=0.002,
        cvar_95=0.01,
        stop_loss=99.0,
        take_profit=102.0,
        reason="unit-test",
        risk_warning="unit-test",
    )
    account = {
        "equity": 10000.0,
        "drawdown_pct": 0.0,
        "daily_loss_pct": 0.0,
    }

    decision = engine.evaluate(
        trade=idea,
        account=account,
        volatility=0.01,
        news_risk=0.1,
        mark_price=250.0,
    )

    assert decision.approved is False
    assert "position_size_exceeds_limit" in decision.reasons
