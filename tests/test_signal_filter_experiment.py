from __future__ import annotations

import pandas as pd

from scripts.run_signal_filter_experiment import (
    BASELINE_WEEKLY_RETURN_PCT,
    MODEL_HORIZON_BARS,
    SignalFilterConfig,
    ValidationSplitMeta,
    build_horizon_alignment_audit_markdown,
    build_signal_filter_experiment_markdown,
    signal_passes_filters,
    split_for_validation_experiment,
)


def _make_best_exit_result(time_stop_bars: int = 48) -> pd.Series:
    return pd.Series(
        {
            "k_stop": 1.5,
            "k_take": 2.2,
            "time_stop_mode": "fixed",
            "time_stop_bars": time_stop_bars,
            "trailing_enabled": True,
            "trailing_activate_profit_pct": 0.004,
            "trailing_distance_pct": 0.003,
            "max_position_fraction": 0.1,
        }
    )


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


def test_high_vol_filter_removes_high_vol_trades():
    cfg = SignalFilterConfig(
        regime_filter_name="exclude_high_volatility",
        allowed_regimes=("low", "medium", "unknown"),
        predicted_return_threshold=0.0,
        confidence_threshold=0.0,
        min_signal_strength=0.0,
        no_trade_zone=0.0,
    )

    allowed, reason = signal_passes_filters(
        cfg,
        predicted_return=0.01,
        confidence=0.8,
        expected_return=0.01,
        volatility_regime="high",
    )

    assert allowed is False
    assert reason == "volatility_regime_blocked:high"


def test_horizon_audit_detects_mismatch():
    exit_df = pd.DataFrame(
        [
            {"time_stop_bars": 24, "horizon_mismatch": True},
            {"time_stop_bars": 48, "horizon_mismatch": True},
        ]
    )

    md = build_horizon_alignment_audit_markdown(exit_df, _make_best_exit_result(time_stop_bars=24))

    assert f"- model_prediction_horizon_bars: {MODEL_HORIZON_BARS}" in md
    assert "- mismatch: YES" in md
    assert "Keep current model horizon and set time_stop_bars=1" in md


def test_report_never_claims_guaranteed_profit():
    df = pd.DataFrame(
        [
            {
                "regime_filter_name": "exclude_high_volatility",
                "allowed_regimes": '["low", "medium"]',
                "predicted_return_threshold": 0.0005,
                "confidence_threshold": 0.35,
                "min_signal_strength": 0.00025,
                "no_trade_zone": 0.00025,
                "weekly_return_pct": BASELINE_WEEKLY_RETURN_PCT - 0.2,
                "sharpe": -0.8,
                "max_drawdown_pct": 0.49,
                "trade_count": 59,
                "win_rate_pct": 47.0,
                "profit_factor": 0.9,
                "expectancy": -0.2,
                "avg_predicted_return": 0.0007,
                "avg_realized_return_after_signal": -0.0003,
                "trades_by_volatility_regime": '{"low": 20, "medium": 39}',
                "horizon_mismatch": True,
                "weekly_target_pass": False,
                "baseline_pass": False,
                "test_split_touched": False,
                "rejected": False,
                "reject_reason": "",
            }
        ]
    )

    md = build_signal_filter_experiment_markdown(df, _make_split_meta(), _make_best_exit_result())

    assert "guaranteed profit" not in md.lower()
    assert "No profit promise." in md
    assert "baseline comparison: FAIL" in md
    assert "weekly target (1.0%): FAIL" in md


def test_final_test_split_untouched():
    idx = pd.date_range("2026-01-01", periods=1000, freq="5min")
    frame = pd.DataFrame({"x": range(1000)}, index=idx)

    train_val, validation_index, meta = split_for_validation_experiment(frame)

    test_start_ts = pd.Timestamp(meta.test_start_ts)
    assert meta.test_split_touched is False
    assert train_val.index.max() < test_start_ts
    assert validation_index.max() < test_start_ts