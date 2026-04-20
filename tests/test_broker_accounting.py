from marketify.broker.paper import PaperBroker
from marketify.config import BrokerConfig


def _broker(tmp_path):
    cfg = BrokerConfig(
        initial_cash=10000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper.db"),
    )
    return PaperBroker(cfg)


def test_cash_avg_cost_realized_unrealized_equity(tmp_path):
    broker = _broker(tmp_path)
    broker.update_market_price("AAPL", 100.0)

    broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
    account = broker.get_account()
    pos = broker.get_positions()[0]

    assert account["cash"] == 9000.0
    assert pos["qty"] == 10.0
    assert pos["avg_cost"] == 100.0

    broker.update_market_price("AAPL", 110.0)
    account = broker.get_account()
    pos = broker.get_positions()[0]
    assert round(account["equity"], 2) == 10100.0
    assert round(pos["unrealized_pnl"], 2) == 100.0

    broker.submit_order({"symbol": "AAPL", "side": "sell", "qty": 5, "price": 110.0})
    account = broker.get_account()
    pos = broker.get_positions()[0]
    assert round(account["realized_pnl"], 2) == 50.0
    assert pos["qty"] == 5.0
    assert pos["avg_cost"] == 100.0

    broker.submit_order({"symbol": "AAPL", "side": "sell", "qty": 5, "price": 90.0})
    account = broker.get_account()
    assert round(account["cash"], 2) == 10000.0
    assert round(account["realized_pnl"], 2) == 0.0
    assert broker.get_positions() == []

    orders = broker.get_orders()
    fills = broker.get_fills()
    assert len(orders) == 3
    assert len(fills) == 3
