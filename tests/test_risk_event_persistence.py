"""Tests for persistent risk events."""
from marketify.broker.paper import PaperBroker
from marketify.config import BrokerConfig


def test_risk_event_insert_and_query(tmp_path):
    cfg = BrokerConfig(db_path=str(tmp_path / "risk.db"))
    broker = PaperBroker(cfg)

    broker.append_risk_event("daily_loss_halt", "critical", "daily loss limit breached", {"equity": 9500})
    broker.append_risk_event("risk_rejection", "warning", "cvar_limit_breached")

    events = broker.get_risk_events(limit=10)
    assert len(events) == 2
    assert events[0]["event_type"] == "risk_rejection"  # most recent first
    assert events[1]["severity"] == "critical"


def test_risk_event_persists_across_restart(tmp_path):
    db_path = str(tmp_path / "risk2.db")
    cfg = BrokerConfig(db_path=db_path)

    broker1 = PaperBroker(cfg)
    broker1.append_risk_event("test_event", "info", "smoke test")

    broker2 = PaperBroker(cfg)
    events = broker2.get_risk_events(limit=10)
    assert len(events) == 1
    assert events[0]["reason"] == "smoke test"
