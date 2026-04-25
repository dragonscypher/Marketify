from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from marketify.models.rnn_model import (TECHNICAL_FEATURE_COLUMNS,
                                        build_aligned_labels,
                                        load_recurrent_artifact,
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


def test_build_aligned_labels_handles_normal_close_series():
    idx = _make_close_index()
    frame = pd.DataFrame({"Close": pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=idx)}, index=idx)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_field_first_multiindex_close_column():
    idx = _make_close_index()
    columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
    frame = pd.DataFrame([[100.0], [105.0], [110.0], [115.0], [120.0]], index=idx, columns=columns)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_ticker_first_multiindex_close_column():
    idx = _make_close_index()
    columns = pd.MultiIndex.from_tuples([("AAPL", "Close")])
    frame = pd.DataFrame([[100.0], [105.0], [110.0], [115.0], [120.0]], index=idx, columns=columns)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    _assert_h2_return(labels)


def test_build_aligned_labels_handles_duplicate_flat_close_frame():
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


def test_xgb_real_leaderboard_keeps_xgb_primary():
    import scripts.compare_models as cm

    rows = pd.DataFrame(
        [
            asdict(
                cm.RealBenchmarkRow(
                    model_name="xgb",
                    role="champion",
                    weekly_return_pct=0.6977,
                    sharpe=1.2268,
                    max_drawdown_pct=0.1933,
                    cvar_95_pct=0.0500,
                    trade_count=10,
                    win_rate_pct=60.0,
                    expectancy=1.0,
                    weekly_target_pass=False,
                    comparison_only=False,
                    run_status="COMPLETE",
                    gate_block_count=12,
                    gate_keep_ratio_pct=44.0,
                    top_blocker="below_cost_buffer",
                    notes="primary real benchmark lane",
                )
            ),
            asdict(
                cm.RealBenchmarkRow(
                    model_name="lstm",
                    role="comparison",
                    weekly_return_pct=float("nan"),
                    sharpe=float("nan"),
                    max_drawdown_pct=float("nan"),
                    cvar_95_pct=float("nan"),
                    trade_count=0,
                    win_rate_pct=float("nan"),
                    expectancy=float("nan"),
                    weekly_target_pass=False,
                    comparison_only=True,
                    run_status="FROZEN",
                    gate_block_count=0,
                    gate_keep_ratio_pct=0.0,
                    top_blocker="comparison_only",
                    notes="comparison only; frozen off primary path",
                )
            ),
        ]
    )

    champion = cm.RealBenchmarkRow(
        model_name="xgb",
        role="champion",
        weekly_return_pct=0.6977,
        sharpe=1.2268,
        max_drawdown_pct=0.1933,
        cvar_95_pct=0.0500,
        trade_count=10,
        win_rate_pct=60.0,
        expectancy=1.0,
        weekly_target_pass=False,
        comparison_only=False,
        run_status="COMPLETE",
        gate_block_count=12,
        gate_keep_ratio_pct=44.0,
        top_blocker="below_cost_buffer",
        notes="primary real benchmark lane",
    )

    md = cm.build_xgb_real_leaderboard_markdown(rows, champion)

    assert "current_champion: xgb" in md
    assert "GRU/LSTM comparison only. Not eligible as champion here." in md
    assert "comparison only; frozen off primary path" in md


def test_next_status_reports_exact_fail_gap_for_xgb_real_path():
    import scripts.compare_models as cm

    champion = cm.RealBenchmarkRow(
        model_name="xgb",
        role="champion",
        weekly_return_pct=0.6977,
        sharpe=1.2268,
        max_drawdown_pct=0.1933,
        cvar_95_pct=0.0500,
        trade_count=10,
        win_rate_pct=60.0,
        expectancy=1.0,
        weekly_target_pass=False,
        comparison_only=False,
        run_status="COMPLETE",
        gate_block_count=12,
        gate_keep_ratio_pct=44.0,
        top_blocker="below_cost_buffer",
        notes="primary real benchmark lane",
        daily_return_pct=0.1395,
        approval_count=10,
        keep_coverage_pct=7.2,
        cost_buffer_reject_rate=40.0,
        disagreement_reject_rate=12.5,
        keep_rate_by_gate="kept=44.00%; below_cost_buffer=40.00%",
        keep_rate_by_disagreement_bucket="kept=44.00%; disagreement_rejected=12.50%",
        approval_precision_top_half=80.0,
        approval_precision_bottom_half=20.0,
        expectancy_by_confidence_bucket="low=-2.0 (3); mid=-0.5 (3); high=1.5 (4)",
        pnl_by_confidence_bucket="low=-6.0 (3); mid=-1.5 (3); high=6.0 (4)",
        selected_max_disagreement=0.006,
        selected_approval_precision_threshold=0.4,
        applied_max_disagreement=0.006,
        applied_approval_precision_threshold=0.4,
        final_trade_count=10,
        final_approval_count=10,
        final_keep_coverage_pct=7.2,
        final_expectancy=1.0,
        gate_config_match="YES",
    )

    md = cm.build_next_status_markdown(champion)

    assert "current champion: xgb" in md
    assert "daily_return_pct: 0.1395" in md
    assert "cvar_95_pct: 0.05" in md
    assert "approval_count: 10" in md
    assert "rejected_by_disagreement_count: 0" in md
    assert "disagreement_reject_rate: 12.50" in md
    assert "keep_rate_by_disagreement_bucket: kept=44.00%; disagreement_rejected=12.50%" in md
    assert "kept_trade_count: 0" in md
    assert "approval_precision_top_half: 80.00" in md
    assert "approval_precision_bottom_half: 20.00" in md
    assert "expectancy_by_confidence_bucket: low=-2.0 (3); mid=-0.5 (3); high=1.5 (4)" in md
    assert "pnl_by_confidence_bucket: low=-6.0 (3); mid=-1.5 (3); high=6.0 (4)" in md
    assert "selected_max_disagreement: 0.006000" in md
    assert "selected_approval_precision_threshold: 0.400000" in md
    assert "applied_max_disagreement: 0.006000" in md
    assert "applied_approval_precision_threshold: 0.400000" in md
    assert "final_trade_count: 10" in md
    assert "final_approval_count: 10" in md
    assert "final_keep_coverage_pct: 7.20" in md
    assert "final_expectancy: 1.000000" in md
    assert "gate_config_match: YES" in md
    assert "gap to 1.0% target: 0.3023 percentage points" in md
    assert "result: FAIL" in md
    assert "dominant_gate=below_cost_buffer" in md


def _make_real_benchmark_row(model_name: str = "xgb", **overrides):
    import scripts.compare_models as cm

    defaults = dict(
        model_name=model_name,
        role="candidate",
        weekly_return_pct=0.6977,
        sharpe=1.2268,
        max_drawdown_pct=0.1933,
        cvar_95_pct=0.0500,
        trade_count=10,
        win_rate_pct=60.0,
        expectancy=1.0,
        weekly_target_pass=False,
        comparison_only=False,
        run_status="COMPLETE",
        gate_block_count=12,
        gate_keep_ratio_pct=44.0,
        top_blocker="below_cost_buffer",
        notes="same-path real benchmark lane",
        daily_return_pct=0.1395,
        approval_count=10,
        keep_coverage_pct=7.2,
        cost_buffer_reject_rate=40.0,
        disagreement_reject_rate=12.5,
        keep_rate_by_gate="kept=44.00%; below_cost_buffer=40.00%",
        keep_rate_by_disagreement_bucket="kept=44.00%; disagreement_rejected=12.50%",
        approval_precision_top_half=80.0,
        approval_precision_bottom_half=20.0,
        expectancy_by_confidence_bucket="low=-2.0 (3); mid=-0.5 (3); high=1.5 (4)",
        pnl_by_confidence_bucket="low=-6.0 (3); mid=-1.5 (3); high=6.0 (4)",
        selected_max_disagreement=0.006,
        selected_approval_precision_threshold=0.4,
        applied_max_disagreement=0.006,
        applied_approval_precision_threshold=0.4,
        final_trade_count=10,
        final_approval_count=10,
        final_keep_coverage_pct=7.2,
        final_expectancy=1.0,
        gate_config_match="YES",
    )
    defaults.update(overrides)
    return cm.RealBenchmarkRow(**defaults)


def test_real_benchmark_markdown_includes_required_same_path_fields():
    import scripts.compare_models as cm

    champion = _make_real_benchmark_row(model_name="xgb", role="champion")
    fusion_reference = _make_real_benchmark_row(
        model_name="fusion_reference_only",
        role="reference",
        comparison_only=True,
        notes="reference only; never champion until deployable same-path semantics proven",
    )

    md = cm.build_real_benchmark_markdown(champion, champion, [champion, fusion_reference])

    assert "source_of_truth_path: scripts.compare_models:_run_real_lane" in md
    assert "daily_return_pct: 0.1395" in md
    assert "cvar_95_pct: 0.05" in md
    assert "approval_count: 10" in md
    assert "rejected_by_disagreement_count: 0" in md
    assert "rejected_by_low_edge_count: 0" in md
    assert "disagreement_reject_rate: 12.50" in md
    assert "keep_rate_by_disagreement_bucket: kept=44.00%; disagreement_rejected=12.50%" in md
    assert "kept_trade_count: 0" in md
    assert "approval_precision_top_half: 80.00" in md
    assert "approval_precision_bottom_half: 20.00" in md
    assert "expectancy_by_confidence_bucket: low=-2.0 (3); mid=-0.5 (3); high=1.5 (4)" in md
    assert "pnl_by_confidence_bucket: low=-6.0 (3); mid=-1.5 (3); high=6.0 (4)" in md
    assert "selected_max_disagreement: 0.006000" in md
    assert "selected_approval_precision_threshold: 0.400000" in md
    assert "applied_max_disagreement: 0.006000" in md
    assert "applied_approval_precision_threshold: 0.400000" in md
    assert "final_trade_count: 10" in md
    assert "final_approval_count: 10" in md
    assert "final_keep_coverage_pct: 7.20" in md
    assert "final_expectancy: 1.000000" in md
    assert "gate_config_match: YES" in md
    assert "fusion_reference_only" in md
    assert "reference only; never champion until deployable same-path semantics proven" in md


def test_model_leaderboard_markdown_uses_canonical_name_and_gate_fields():
    import scripts.compare_models as cm

    champion = _make_real_benchmark_row(model_name="xgb", role="champion")
    comparison = _make_real_benchmark_row(
        model_name="ridge",
        role="candidate",
        weekly_return_pct=0.401,
        rejected_by_disagreement_count=3,
        rejected_by_low_edge_count=2,
        kept_trade_count=7,
        cost_buffer_reject_rate=25.0,
        disagreement_reject_rate=10.0,
        keep_rate_by_gate="kept=55.00%; below_cost_buffer=25.00%; model_disagreement=10.00%",
        keep_rate_by_disagreement_bucket="kept=55.00%; disagreement_rejected=10.00%",
        approval_precision_top_half=66.67,
        approval_precision_bottom_half=33.33,
        expectancy_by_confidence_bucket="low=-1.0 (2); mid=0.5 (2); high=1.0 (3)",
        pnl_by_confidence_bucket="low=-2.0 (2); mid=1.0 (2); high=3.0 (3)",
        average_edge_kept=0.011,
        average_edge_rejected=0.004,
    )

    md = cm.build_model_leaderboard_markdown(champion, [champion, comparison])

    assert "# Model Leaderboard" in md
    assert "current_champion: xgb" in md
    assert "selected_max_disagreement" in md
    assert "applied_max_disagreement" in md
    assert "selected_approval_precision_threshold" in md
    assert "applied_approval_precision_threshold" in md
    assert "gate_config_match" in md
    assert "final_expectancy" in md
    assert "rejected_by_disagreement_count" in md
    assert "disagreement_reject_rate" in md
    assert "keep_rate_by_disagreement_bucket" in md
    assert "approval_precision_top_half" in md
    assert "approval_precision_bottom_half" in md
    assert "expectancy_by_confidence_bucket" in md
    assert "pnl_by_confidence_bucket" in md


def test_signal_gate_tracks_disagreement_and_low_edge_counts():
    import scripts.compare_models as cm
    from marketify.config import AppConfig

    idx = pd.date_range("2026-01-01", periods=3, freq="5min")
    feat = pd.DataFrame(
        {
            "news_sentiment_score": [0.0, 0.0, 0.0],
            "news_risk": [0.0, 0.0, 0.0],
            "news_event_shock": [0.0, 0.0, 0.0],
            "vix_proxy": [10.0, 10.0, 10.0],
            "macro_regime_bear": [0.0, 0.0, 0.0],
            "macro_vol_level_high": [0.0, 0.0, 0.0],
            "macro_event_risk": [0.0, 0.0, 0.0],
            "vol_20": [0.01, 0.01, 0.01],
        },
        index=idx,
    )
    preds = pd.Series([0.00001, 0.01, 0.012], index=idx)
    support_preds = {
        "gru": pd.Series([0.00001, -0.01, 0.0115], index=idx),
    }

    gated, stats = cm._apply_signal_gate(feat, preds, AppConfig(), comparison_preds=support_preds)

    assert int(stats["rejected_by_low_edge_count"]) == 1
    assert int(stats["rejected_by_disagreement_count"]) == 1
    assert int(stats["kept_trade_count"]) == 1
    assert float(stats["average_edge_kept"]) > 0.0
    assert float(stats["average_edge_rejected"]) > 0.0
    assert "kept=" in str(stats["keep_rate_by_gate"])
    assert int(gated.notna().sum()) == 1


def test_signal_gate_thresholds_allow_cost_buffer_and_precision_overrides():
    import scripts.compare_models as cm
    from marketify.config import AppConfig

    config = AppConfig()
    fee_floor = (float(config.broker.fee_bps) + float(config.broker.slippage_bps)) / 10000.0

    thresholds = cm._signal_gate_thresholds(
        config,
        gate_overrides={
            "cost_buffer_floor": 0.0,
            "approval_precision_threshold": 0.42,
            "max_disagreement": 0.009,
            "abstain_margin": 0.0,
        },
    )

    assert abs(float(thresholds["edge_floor"]) - fee_floor) < 1e-12
    assert float(thresholds["min_confidence"]) == 0.42
    assert float(thresholds["max_disagreement"]) == 0.009


def test_ordered_weak_feature_candidates_prioritizes_requested_features():
    import scripts.compare_models as cm

    importance_df = pd.DataFrame(
        [
            {"feature": "ret_5", "mean_importance": 0.0001},
            {"feature": "is_afternoon", "mean_importance": 0.0002},
            {"feature": "news_sentiment_score", "mean_importance": 0.0003},
            {"feature": "volume_z20", "mean_importance": 0.0004},
            {"feature": "news_risk", "mean_importance": 0.0005},
            {"feature": "is_morning", "mean_importance": 0.0006},
        ]
    )

    ordered = cm._ordered_weak_feature_candidates(importance_df, 5)

    assert ordered[:4] == ["news_risk", "news_sentiment_score", "is_afternoon", "is_morning"]


def test_tune_real_path_gate_knobs_prioritizes_expectancy_then_coverage_then_trades(monkeypatch):
    import scripts.compare_models as cm
    from marketify.config import AppConfig

    monkeypatch.setattr(
        cm,
        "_signal_gate_thresholds",
        lambda config, gate_overrides=None: {
            "min_confidence": 0.35,
            "max_disagreement": 0.006,
        },
    )

    def fake_run_real_lane(model_name, feat, preds, config, role, comparison_preds=None, gate_overrides=None, notes=""):
        overrides = gate_overrides or {}
        key = (round(float(overrides["max_disagreement"]), 6), round(float(overrides["approval_precision_threshold"]), 6))
        row_map = {
            (0.006, 0.30): _make_real_benchmark_row(expectancy=-0.5, keep_coverage_pct=20.0, trade_count=12, approval_count=12, weekly_return_pct=2.0),
            (0.006, 0.35): _make_real_benchmark_row(expectancy=0.2, keep_coverage_pct=6.0, trade_count=0, approval_count=0, weekly_return_pct=3.0),
            (0.006, 0.40): _make_real_benchmark_row(expectancy=0.1, keep_coverage_pct=7.0, trade_count=2, approval_count=2, weekly_return_pct=0.8),
        }
        row = row_map.get(key, _make_real_benchmark_row(expectancy=-1.0, keep_coverage_pct=1.0, trade_count=0, approval_count=0, weekly_return_pct=0.1))
        return row, {}

    monkeypatch.setattr(cm, "_run_real_lane", fake_run_real_lane)

    best = cm._tune_real_path_gate_knobs(
        feat=pd.DataFrame(),
        xgb_preds=pd.Series(dtype=float),
        config=AppConfig(),
        support_preds={},
    )

    assert best == {"max_disagreement": 0.006, "approval_precision_threshold": 0.4}


def test_fusion_reference_only_is_not_champion_eligible():
    import scripts.compare_models as cm

    fusion_reference = _make_real_benchmark_row(
        model_name="fusion_reference_only",
        role="reference",
        comparison_only=True,
        notes="reference only",
    )

    assert cm._is_eligible_for_champion(fusion_reference) is False


def test_weekly_target_required_for_champion_eligibility():
    import scripts.compare_models as cm

    under_target = _make_real_benchmark_row(model_name="xgb", weekly_return_pct=0.9999)
    deployable = _make_real_benchmark_row(model_name="xgb", weekly_return_pct=1.05)

    assert cm._is_eligible_for_champion(under_target) is False
    assert cm._is_eligible_for_champion(deployable) is True


def test_reference_row_prefers_xgb_traded_lane_over_phantom_weekly():
    import scripts.compare_models as cm

    phantom = _make_real_benchmark_row(
        model_name="ridge",
        weekly_return_pct=8.5256,
        trade_count=0,
        approval_count=0,
        keep_coverage_pct=0.0,
        expectancy=0.0,
    )
    xgb_real = _make_real_benchmark_row(
        model_name="xgb",
        weekly_return_pct=7.1354,
        trade_count=9,
        approval_count=9,
        keep_coverage_pct=0.46,
        expectancy=-1.580444,
    )

    chosen = cm._select_reference_row([phantom, xgb_real])

    assert chosen.model_name == "xgb"


def test_approval_precision_diagnostics_bucket_confidence():
    import scripts.compare_models as cm

    diag = pd.DataFrame(
        [
            {"pnl": -2.0, "signal_confidence": 0.1},
            {"pnl": -1.0, "signal_confidence": 0.2},
            {"pnl": 0.5, "signal_confidence": 0.6},
            {"pnl": 1.0, "signal_confidence": 0.9},
        ]
    )

    result = cm._approval_precision_diagnostics(diag)

    assert result["approval_precision_top_half"] == 100.0
    assert result["approval_precision_bottom_half"] == 0.0
    assert "low=" in result["expectancy_by_confidence_bucket"]
    assert "high=" in result["expectancy_by_confidence_bucket"]
    assert "low=" in result["pnl_by_confidence_bucket"]
    assert "high=" in result["pnl_by_confidence_bucket"]


def test_xgb_report_writers_emit_fix_and_ablation_sections(tmp_path):
    import scripts.compare_models as cm

    xgb_row = _make_real_benchmark_row(model_name="xgb", role="champion")
    trade_diag = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T09:35:00Z",
                "side": "LONG",
                "pnl": -1.2,
                "return_pct": -0.4,
                "signal_confidence": 0.32,
                "expected_return": 0.1,
                "entry_news_risk": 0.9,
                "entry_news_event_shock": 1.0,
                "entry_vol_regime_ratio": 1.4,
                "entry_signal_quality_20": 0.2,
                "exit_reason": "stop",
            },
            {
                "timestamp": "2026-01-01T10:10:00Z",
                "side": "LONG",
                "pnl": 0.8,
                "return_pct": 0.3,
                "signal_confidence": 0.78,
                "expected_return": 0.2,
                "entry_news_risk": 0.1,
                "entry_news_event_shock": 0.0,
                "entry_vol_regime_ratio": 0.9,
                "entry_signal_quality_20": 0.8,
                "exit_reason": "target",
            },
        ]
    )
    importance_df = pd.DataFrame(
        [
            {"feature": "sin_hour", "mean_importance": 0.001, "window_count": 4},
            {"feature": "news_risk_score", "mean_importance": 0.002, "window_count": 4},
            {"feature": "ret_5", "mean_importance": 0.12, "window_count": 4},
        ]
    )
    ablation_rows = [
        {
            "variant": "drop_time_features",
            "feature_count": 18,
            "weekly_return_pct": 0.8123,
            "delta_weekly_return_pct": 0.1146,
            "daily_return_pct": 0.1625,
            "sharpe": 1.35,
            "max_drawdown_pct": 0.18,
            "cvar_95_pct": 0.04,
            "trade_count": 11,
            "approval_count": 11,
            "expectancy": 1.1,
            "keep_coverage_pct": 7.4,
            "eligible": "YES",
            "recommendation": "KEEP",
        }
    ]
    fix_lines = [
        "Keep drop_time_features: improved weekly by +0.1146 pct points without collapsing coverage.",
        "Tighten deterministic news veto only if same-path xgb improves.",
        "Raise xgb approval precision on low-confidence trades.",
    ]

    cm._write_false_positive_trade_review(xgb_row, trade_diag, fix_lines, tmp_path)
    cm._write_feature_ablation_md(xgb_row, importance_df, ablation_rows, tmp_path)
    next_status = cm.build_next_status_markdown(xgb_row, fix_lines=fix_lines)

    false_positive_md = (tmp_path / "false_positive_trade_review.md").read_text(encoding="utf-8")
    feature_ablation_md = (tmp_path / "feature_ablation.md").read_text(encoding="utf-8")

    assert "Top 3 Highest-Impact Fixes" in false_positive_md
    assert "News-Risk Buckets" in false_positive_md
    assert "Weak/Noisy Candidates" in feature_ablation_md
    assert "drop_time_features" in feature_ablation_md
    assert "Rolling XGB Feature Importance Audit" in feature_ablation_md
    assert "## Top 3 Highest-Impact Fixes" in next_status