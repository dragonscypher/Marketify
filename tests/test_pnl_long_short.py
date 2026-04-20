from marketify.broker.paper import PaperBroker
from marketify.config import BrokerConfig


def _broker(tmp_path):
    cfg = BrokerConfig(
        initial_cash=10000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_short.db"),
    )
    return PaperBroker(cfg)


def test_short_pnl_accounting(tmp_path):
    broker = _broker(tmp_path)
    broker.update_market_price("TSLA", 100.0)

    broker.submit_order({"symbol": "TSLA", "side": "sell", "qty": 10, "price": 100.0})
    account = broker.get_account()
    pos = broker.get_positions()[0]

    assert account["cash"] == 11000.0
    assert pos["qty"] == -10.0
    assert pos["avg_cost"] == 100.0

    broker.update_market_price("TSLA", 90.0)
    account = broker.get_account()
    assert round(account["equity"], 2) == 10100.0

    broker.submit_order({"symbol": "TSLA", "side": "buy", "qty": 10, "price": 90.0})
    account = broker.get_account()

    assert round(account["cash"], 2) == 10100.0
    assert round(account["realized_pnl"], 2) == 100.0
    assert broker.get_positions() == []
