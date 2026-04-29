import numpy as np
import pandas as pd

import app
from marketify.broker.paper import PaperBroker
from marketify.config import AppConfig
from marketify.risk.risk_engine import RiskEngine
from marketify.state.store import InMemoryStore


def test_generate_suggestion_does_not_submit_trade_before_approval(tmp_path, monkeypatch):
    config = AppConfig()
    config.broker.db_path = str(tmp_path / "ui_gate.db")
    state = {
        "config": config,
        "broker": PaperBroker(config.broker),
        "risk": RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": app.NewsProvider(),
        "sentiment": app.FinBERTSentiment(enabled=False),
        "last_price": None,
    }

    idx = pd.date_range("2026-01-01", periods=40, freq="5min")
    raw = pd.DataFrame(
        {
            "Open": np.linspace(100, 104, 40),
            "High": np.linspace(101, 105, 40),
            "Low": np.linspace(99, 103, 40),
            "Close": np.linspace(100, 104, 40),
            "Volume": np.full(40, 1_000),
        },
        index=idx,
    )

    feat = raw.copy()
    for col in app.TECHNICAL_FEATURE_COLUMNS:
        feat[col] = 0.001
    feat["vol_20"] = 0.01
    feat["target_next_ret"] = 0.001

    class ScriptedModel:
        def predict(self, x_pred):
            return np.full(len(x_pred), 0.002)

    monkeypatch.setattr(app, "fetch_market_data", lambda **_: raw)
    monkeypatch.setattr(app, "add_technical_features", lambda _: feat)
    monkeypatch.setattr(
        app,
        "_load_local_model_artifact",
        lambda _: (
            ScriptedModel(),
            {
                "LOCAL_MODEL_LOAD": "YES",
                "loaded_artifact_path": "SCRIPTED_TEST",
                "loaded_model_type": "xgb",
                "inference_smoke": "PENDING",
                "message": "scripted test model",
            },
        ),
    )
    monkeypatch.setattr(app, "_predict_with_local_model", lambda model, frame: (float(model.predict(frame.tail(1))[0]), {"inference_smoke": "PASS"}))

    state, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)

    assert state["store"].pending_trade_idea is not None
    assert len(state["broker"].get_orders()) == 0

    state, *_ = app.approve_trade(state)
    assert len(state["broker"].get_orders()) == 1


def test_latest_scalar_accepts_duplicate_index_series():
    frame = pd.DataFrame({"Close": [101.0, 102.5], "vol_20": [0.01, 0.02]}, index=[0, 0])

    assert app._latest_scalar(frame, "Close") == 102.5
    assert app._latest_scalar(frame, "vol_20") == 0.02
