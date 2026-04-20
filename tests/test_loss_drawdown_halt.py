"""Tests for daily loss and drawdown auto-halt."""
from marketify.broker.paper import PaperBroker
from marketify.config import AppConfig
from marketify.state.store import InMemoryStore


def _make_state(tmp_path):
    config = AppConfig()
    config.broker.db_path = str(tmp_path / "halt.db")
    config.risk.max_daily_loss_pct = 0.02
    config.risk.max_drawdown_pct = 0.10
    config.risk.auto_halt_on_daily_loss = True
    config.risk.auto_halt_on_drawdown = True
    broker = PaperBroker(config.broker)
    return config, broker


def test_daily_loss_halt(tmp_path):
    """Simulate daily loss exceeding limit → halt_reason set."""
    import app

    config, broker = _make_state(tmp_path)
    state = {
        "config": config,
        "broker": broker,
        "risk": app.RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": app.NewsProvider(),
        "sentiment": app.FinBERTSentiment(enabled=False),
        "last_price": None,
    }

    # Buy at 100, then price drops to simulate loss
    broker.update_market_price("AAPL", 100.0)
    broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 50, "price": 100.0})

    # Price drops significantly — 5% loss on half the portfolio
    broker.update_market_price("AAPL", 95.0)

    halt = app._check_halt(state)
    # With 50 shares bought at 100 and price at 95:
    # equity = cash(5000) + 50*95 = 9750, initial was 10000
    # daily_loss_pct should be > 2%
    assert halt is not None
    assert "loss" in halt or "drawdown" in halt


def test_drawdown_halt(tmp_path):
    """Simulate drawdown exceeding limit → halt."""
    import app

    config, broker = _make_state(tmp_path)
    config.risk.max_drawdown_pct = 0.03  # Set very tight
    state = {
        "config": config,
        "broker": broker,
        "risk": app.RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": app.NewsProvider(),
        "sentiment": app.FinBERTSentiment(enabled=False),
        "last_price": None,
    }

    broker.update_market_price("AAPL", 100.0)
    broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 50, "price": 100.0})
    broker.update_market_price("AAPL", 96.0)

    halt = app._check_halt(state)
    assert halt is not None
