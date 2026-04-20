"""Tests for persistent audit log."""
from marketify.broker.paper import PaperBroker
from marketify.config import BrokerConfig


def test_audit_log_insert_and_query(tmp_path):
    cfg = BrokerConfig(db_path=str(tmp_path / "audit.db"))
    broker = PaperBroker(cfg)

    broker.append_audit("order_approved", "filled", symbol="AAPL", side="buy", qty=10.0)
    broker.append_audit("order_rejected", "rejected", symbol="TSLA", reason="user_rejected")
    broker.append_audit("kill_switch_toggle", "ENABLED")

    log = broker.get_audit_log(limit=10)
    assert len(log) == 3
    assert log[0]["action_type"] == "kill_switch_toggle"  # most recent first
    assert log[1]["action_type"] == "order_rejected"
    assert log[2]["action_type"] == "order_approved"
    assert log[2]["symbol"] == "AAPL"
    assert log[2]["qty"] == 10.0


def test_audit_with_metadata(tmp_path):
    cfg = BrokerConfig(db_path=str(tmp_path / "audit2.db"))
    broker = PaperBroker(cfg)

    broker.append_audit("order_approved", "filled", symbol="AAPL", metadata={"order_id": "abc123"})
    log = broker.get_audit_log(limit=1)
    assert len(log) == 1
    assert '"order_id"' in log[0]["metadata_json"]


def test_audit_persists_across_reconnect(tmp_path):
    db_path = str(tmp_path / "audit3.db")
    cfg = BrokerConfig(db_path=db_path)

    broker1 = PaperBroker(cfg)
    broker1.append_audit("test_action", "ok", reason="first")

    broker2 = PaperBroker(cfg)
    log = broker2.get_audit_log(limit=10)
    assert len(log) == 1
    assert log[0]["reason"] == "first"
