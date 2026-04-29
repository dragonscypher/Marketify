import json
from pathlib import Path


def test_app_local_model_missing_fallback(monkeypatch, tmp_path):
    import app

    monkeypatch.setattr(app, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path / "reports")
    model, status = app._load_local_model_artifact("AAPL")

    assert model is None
    assert status["LOCAL_MODEL_LOAD"] == "NO"
    assert status["loaded_artifact_path"] == "MISSING"
    assert status["loaded_model_type"] == "MISSING"
    assert status["inference_smoke"] == "FAIL"
    assert status["message"] == "model missing, run training or sync artifacts"


def test_app_latest_artifact_pointer_discovery(monkeypatch, tmp_path):
    import app

    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    run_dir = artifact_dir / "training_runs" / "20260426T000000Z_test"
    run_dir.mkdir(parents=True)
    report_dir.mkdir()
    xgb_path = run_dir / "xgb_AAPL.pkl"
    xgb_path.write_bytes(b"not-a-real-pickle")
    pointer = {
        "latest_xgb_artifact_path": "artifacts/training_runs/20260426T000000Z_test/xgb_AAPL.pkl",
        "models": {"xgb": {"artifact_path": "artifacts/training_runs/20260426T000000Z_test/xgb_AAPL.pkl"}},
    }
    (artifact_dir / "latest_AAPL.json").write_text(json.dumps(pointer), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "ARTIFACT_DIR", Path("artifacts"))
    monkeypatch.setattr(app, "REPORTS_DIR", Path("reports"))

    assert app._find_latest_artifact("AAPL", "xgb") == Path("artifacts/training_runs/20260426T000000Z_test/xgb_AAPL.pkl")


def test_app_latest_scalar_handles_multiindex_close():
    import pandas as pd
    import app

    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    frame = pd.DataFrame(
        {
            ("Close", "AAPL"): [100.0, 101.5],
            ("vol_20", ""): [0.01, 0.02],
        },
        index=idx,
    )

    assert app._latest_scalar(frame, "Close") == 101.5
    assert app._latest_scalar(frame, "vol_20") == 0.02


def test_app_latest_scalar_handles_ticker_first_multiindex_close():
    import pandas as pd
    import app

    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    frame = pd.DataFrame(
        {
            ("AAPL", "Close"): [100.0, 101.5],
            ("features", "vol_20"): [0.01, 0.02],
        },
        index=idx,
    )

    assert app._latest_scalar(frame, "Close") == 101.5
    assert app._latest_scalar(frame, "vol_20") == 0.02


def test_paper_broker_reload_restores_pnl_state(tmp_path: Path):
    from marketify.broker.paper import PaperBroker
    from marketify.config import BrokerConfig

    db_path = tmp_path / "paper.db"
    config = BrokerConfig(initial_cash=100_000.0, fee_bps=0.0, slippage_bps=0.0, db_path=str(db_path))
    broker = PaperBroker(config)
    broker.update_market_price("AAPL", 100.0)
    broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
    broker.update_market_price("AAPL", 105.0)
    before = broker.get_account()
    broker.conn.close()

    reloaded = PaperBroker(config)
    after = reloaded.get_account()
    positions = reloaded.get_positions()
    reloaded.conn.close()

    assert len(positions) == 1
    assert round(after["cash"], 2) == round(before["cash"], 2)
    assert round(after["equity"], 2) == round(before["equity"], 2)
    assert round(after["unrealized_pnl"], 2) == round(before["unrealized_pnl"], 2)
