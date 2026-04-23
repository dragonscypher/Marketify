from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest

from marketify.models.rnn_model import (TECHNICAL_FEATURE_COLUMNS,
                                        load_recurrent_artifact,
                                        build_aligned_labels,
                                        make_recurrent_model,
                                        prepare_recurrent_training_frame)
from scripts.compare_models import (build_recurrent_leaderboard_markdown,
                                    prepare_validation_only_data)
from scripts.run_exit_rule_experiment import ValidationSplitMeta


def _make_feature_frame(rows: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="5min")
    base = pd.DataFrame(index=idx)
    base["Open"] = 100.0 + (pd.Series(range(rows), index=idx) * 0.01)
    base["High"] = base["Open"] + 0.2
    base["Low"] = base["Open"] - 0.2
    base["Close"] = base["Open"] + 0.05
    base["Volume"] = 1_000 + pd.Series(range(rows), index=idx)
    base["vol_20"] = 0.01
    for pos, col in enumerate(TECHNICAL_FEATURE_COLUMNS):
        if col in base.columns:
            continue
        base[col] = (pos + 1) * 0.001
    return base


def _make_split_meta() -> ValidationSplitMeta:
    return ValidationSplitMeta(
        train_end=700,
        val_end=850,
        test_start=850,
        validation_start_ts="2026-01-01 00:00:00",
        validation_end_ts="2026-01-02 00:00:00",
        test_start_ts="2026-01-03 00:00:00",
        test_split_touched=False,
    )


def _make_close_index() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=5, freq="5min")


def _make_recurrent_multiindex_frame(order: str, rows: int = 96, engineered_as_tuples: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="5min")
    base = {
        "Open": 100.0 + np.arange(rows) * 0.05,
        "High": 100.2 + np.arange(rows) * 0.05,
        "Low": 99.8 + np.arange(rows) * 0.05,
        "Close": 100.1 + np.arange(rows) * 0.05,
        "Volume": 1000.0 + np.arange(rows),
    }

    engineered = {
        "ret_1": np.linspace(0.001, 0.002, rows),
        "ret_5": np.linspace(0.002, 0.003, rows),
        "ret_15": np.linspace(0.003, 0.004, rows),
        "ema_dist_10": np.linspace(0.01, 0.02, rows),
        "ema_dist_20": np.linspace(0.02, 0.03, rows),
        "ema_dist_50": np.linspace(0.03, 0.04, rows),
        "rsi_14": np.linspace(40.0, 60.0, rows),
        "macd": np.linspace(0.1, 0.3, rows),
        "macd_signal": np.linspace(0.05, 0.25, rows),
        "macd_hist": np.linspace(0.02, 0.04, rows),
        "atr_14": np.linspace(0.5, 0.8, rows),
        "vol_20": np.linspace(0.01, 0.03, rows),
        "sin_hour": np.linspace(-1.0, 1.0, rows),
        "cos_hour": np.linspace(1.0, -1.0, rows),
        "sin_min": np.linspace(-0.5, 0.5, rows),
        "cos_min": np.linspace(0.5, -0.5, rows),
        "is_morning": np.where(np.arange(rows) % 3 == 0, 1.0, 0.0),
        "is_afternoon": np.where(np.arange(rows) % 3 == 1, 1.0, 0.0),
        "is_late": np.where(np.arange(rows) % 3 == 2, 1.0, 0.0),
    }

    columns: dict[object, np.ndarray] = {}
    ticker = "AAPL"
    for name, values in base.items():
        key = (name, ticker) if order == "field_first" else (ticker, name)
        columns[key] = values
    for name, values in engineered.items():
        if engineered_as_tuples:
            key = (name, ticker) if order == "field_first" else (ticker, name)
            columns[key] = values
        else:
            columns[name] = values

    return pd.DataFrame(columns, index=idx)


def _assert_h2_return(labels: pd.Series) -> None:
    assert labels.name == "target_h2_return"
    assert labels.iloc[0] == pytest.approx(0.10)
    assert labels.iloc[1] == pytest.approx(115.0 / 105.0 - 1.0)


def test_label_horizon_matches_hold_horizon():
    idx = _make_close_index()
    frame = pd.DataFrame({"Close": [100.0, 105.0, 110.0, 115.0, 120.0]}, index=idx)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_field_first_multiindex_close():
    idx = _make_close_index()
    columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
    frame = pd.DataFrame([[100.0], [105.0], [110.0], [115.0], [120.0]], index=idx, columns=columns)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_ticker_first_multiindex_close():
    idx = _make_close_index()
    columns = pd.MultiIndex.from_tuples([("AAPL", "Close")])
    frame = pd.DataFrame([[100.0], [105.0], [110.0], [115.0], [120.0]], index=idx, columns=columns)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_duplicate_flat_close_columns():
    idx = _make_close_index()
    frame = pd.DataFrame(
        np.array(
            [
                [100.0, 100.0],
                [105.0, 105.0],
                [110.0, 110.0],
                [115.0, 115.0],
                [120.0, 120.0],
            ]
        ),
        index=idx,
        columns=["Close", "Close"],
    )

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_prepare_recurrent_training_frame_accepts_field_first_multiindex_ohlcv():
    frame = _make_recurrent_multiindex_frame("field_first", engineered_as_tuples=True)

    out, feature_cols, target_col = prepare_recurrent_training_frame(frame, hold_horizon_bars=48, mode="return")

    assert target_col == "target_h48_return"
    assert {"Open", "Close", "ret_1", "ret_5", "ema_dist_10", "volatility_regime_code", "target_h48_return"}.issubset(out.columns)
    assert {"Open", "Close", "ret_1", "ret_5", "ema_dist_10", "volatility_regime_code"}.issubset(feature_cols)
    assert not out.empty


def test_prepare_recurrent_training_frame_accepts_ticker_first_multiindex_ohlcv():
    frame = _make_recurrent_multiindex_frame("ticker_first", engineered_as_tuples=True)

    out, feature_cols, target_col = prepare_recurrent_training_frame(frame, hold_horizon_bars=48, mode="return")

    assert target_col == "target_h48_return"
    assert {"Open", "Close", "ret_1", "ret_5", "ema_dist_10", "volatility_regime_code", "target_h48_return"}.issubset(out.columns)
    assert {"Open", "Close", "ret_1", "ret_5", "ema_dist_10", "volatility_regime_code"}.issubset(feature_cols)
    assert not out.empty


def test_gru_and_lstm_train_pipeline_import_cleanly():
    from marketify.models import train_pipeline
    from scripts import train_recurrent_baseline

    gru = make_recurrent_model("gru")
    lstm = make_recurrent_model("lstm")

    assert hasattr(train_pipeline, "train_recurrent_baselines")
    assert hasattr(train_recurrent_baseline, "main")
    assert gru.architecture == "gru"
    assert lstm.architecture == "lstm"


def test_load_recurrent_artifact_sets_weights_only_false(monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    payload = {
        "architecture": "gru",
        "config": asdict(make_recurrent_model("gru").config),
        "input_size": 3,
        "state_dict": {},
        "standardizer_mean": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "standardizer_std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }

    class DummyTorch:
        def load(self, artifact_path, map_location=None, weights_only=None):
            calls["artifact_path"] = artifact_path
            calls["map_location"] = map_location
            calls["weights_only"] = weights_only
            return payload

    monkeypatch.setattr("marketify.models.rnn_model._require_torch", lambda: (DummyTorch(), None, None, None))

    model, loaded = load_recurrent_artifact(tmp_path / "recurrent.pt")

    assert calls["map_location"] == "cpu"
    assert calls["weights_only"] is False
    assert model.architecture == "gru"
    assert loaded is payload


def test_load_recurrent_artifact_falls_back_without_weights_only(monkeypatch, tmp_path):
    calls: dict[str, object] = {"count": 0}
    payload = {
        "architecture": "lstm",
        "config": asdict(make_recurrent_model("lstm").config),
        "input_size": 2,
        "state_dict": {},
        "standardizer_mean": np.array([0.1, 0.2], dtype=np.float32),
        "standardizer_std": np.array([1.0, 1.0], dtype=np.float32),
    }

    class DummyTorch:
        def load(self, artifact_path, map_location=None, **kwargs):
            calls["count"] = int(calls["count"]) + 1
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            calls["artifact_path"] = artifact_path
            calls["map_location"] = map_location
            return payload

    monkeypatch.setattr("marketify.models.rnn_model._require_torch", lambda: (DummyTorch(), None, None, None))

    model, loaded = load_recurrent_artifact(tmp_path / "legacy.pt")

    assert calls["count"] == 2
    assert calls["map_location"] == "cpu"
    assert model.architecture == "lstm"
    assert loaded is payload


def test_compare_models_does_not_touch_final_test_split():
    frame = _make_feature_frame()

    train_val, validation_index, meta, _feature_cols, target_col = prepare_validation_only_data(
        frame,
        hold_horizon_bars=48,
        label_mode="return",
    )

    test_start_ts = pd.Timestamp(meta.test_start_ts)
    assert target_col == "target_h48_return"
    assert meta.test_split_touched is False
    assert train_val.index.max() < test_start_ts
    assert validation_index.max() < test_start_ts


def test_recurrent_report_never_claims_guaranteed_profit():
    df = pd.DataFrame(
        [
            {
                "model_name": "gru",
                "hold_horizon_bars": 48,
                "label_mode": "return",
                "regime_filter": "skip_high_vol",
                "high_vol_confidence_threshold": 0.8,
                "weekly_return_pct": 0.42,
                "sharpe": 0.2,
                "max_drawdown_pct": 0.5,
                "trade_count": 20,
                "win_rate_pct": 45.0,
                "profit_factor": 0.95,
                "expectancy": -0.1,
                "baseline_pass": False,
                "weekly_target_pass": False,
                "test_split_touched": False,
                "rejected": False,
                "reject_reason": "",
            }
        ]
    )

    md = build_recurrent_leaderboard_markdown(df, _make_split_meta(), "skip_high_vol", "return")

    assert "guarantee" not in md.lower()
    assert "guaranteed profit" not in md.lower()
    assert "No profit promise. Outcomes remain uncertain." in md
    assert "baseline comparison: FAIL" in md
    assert "Paper mode only. No leverage increase used." in md