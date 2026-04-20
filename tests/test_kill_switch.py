"""Tests for kill switch blocking suggestions and approvals."""
import numpy as np
import pandas as pd

import app
from marketify.broker.paper import PaperBroker
from marketify.config import AppConfig
from marketify.risk.risk_engine import RiskEngine
from marketify.state.store import InMemoryStore


def _make_state(tmp_path, kill_switch=False):
    config = AppConfig()
    config.broker.db_path = str(tmp_path / "ks.db")
    config.risk.kill_switch_enabled = kill_switch
    return {
        "config": config,
        "broker": PaperBroker(config.broker),
        "risk": RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": app.NewsProvider(),
        "sentiment": app.FinBERTSentiment(enabled=False),
        "last_price": None,
    }


def _patch_data(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=40, freq="5min")
    raw = pd.DataFrame({
        "Open": np.linspace(100, 104, 40),
        "High": np.linspace(101, 105, 40),
        "Low": np.linspace(99, 103, 40),
        "Close": np.linspace(100, 104, 40),
        "Volume": np.full(40, 1_000),
    }, index=idx)
    feat = raw.copy()
    for col in app.TECHNICAL_FEATURE_COLUMNS:
        feat[col] = 0.001
    feat["vol_20"] = 0.01
    feat["target_next_ret"] = 0.001
    preds = pd.Series(np.nan, index=idx)
    preds.iloc[-1] = 0.002
    monkeypatch.setattr(app, "fetch_market_data", lambda **_: raw)
    monkeypatch.setattr(app, "add_technical_features", lambda _: feat)
    monkeypatch.setattr(app, "rolling_train_predict", lambda **_: preds)


def test_kill_switch_blocks_suggestion(tmp_path, monkeypatch):
    state = _make_state(tmp_path, kill_switch=True)
    _patch_data(monkeypatch)

    state, status, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
    assert "HALTED" in status
    assert len(state["broker"].get_orders()) == 0


def test_kill_switch_blocks_approval(tmp_path, monkeypatch):
    state = _make_state(tmp_path, kill_switch=False)
    _patch_data(monkeypatch)

    # Generate suggestion first
    state, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
    assert state["store"].pending_trade_idea is not None

    # Enable kill switch then try approve
    state["config"].risk.kill_switch_enabled = True
    state, status, *_ = app.approve_trade(state)
    assert "HALTED" in status
    assert len(state["broker"].get_orders()) == 0


def test_kill_switch_toggle(tmp_path):
    state = _make_state(tmp_path, kill_switch=False)
    state, status, *_ = app.toggle_kill_switch(state)
    assert "ENABLED" in status
    assert state["config"].risk.kill_switch_enabled is True

    state, status, *_ = app.toggle_kill_switch(state)
    assert "DISABLED" in status
    assert state["config"].risk.kill_switch_enabled is False


def test_reset_halt(tmp_path):
    state = _make_state(tmp_path, kill_switch=True)
    state["config"].halt_reason = "daily_loss_limit_breached"

    state, status, *_ = app.reset_halt(state)
    assert "cleared" in status.lower()
    assert state["config"].halt_reason == ""
    assert state["config"].risk.kill_switch_enabled is False
