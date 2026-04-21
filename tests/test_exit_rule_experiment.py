from __future__ import annotations

import json

import pandas as pd
import pytest

from marketify.backtest.exit_rules import (ExitRuleConfig,
                                           atr_scaled_exit_pcts,
                                           evaluate_dynamic_exit)
from scripts.run_exit_rule_experiment import (
    ValidationSplitMeta, _ensure_optional_dependencies_for_exit_experiment,
    build_exit_rule_experiment_markdown, split_for_validation_experiment)


def test_atr_exit_math_deterministic():
    stop_pct, take_pct = atr_scaled_exit_pcts(
        atr=2.0,
        price=100.0,
        k_stop=1.5,
        k_take=2.5,
        min_stop_pct=0.003,
        max_stop_pct=0.03,
        min_take_pct=0.004,
        max_take_pct=0.06,
    )
    assert stop_pct == pytest.approx(0.03, abs=1e-12)
    assert take_pct == pytest.approx(0.05, abs=1e-12)


def test_trailing_stop_triggers_correctly():
    cfg = ExitRuleConfig(
        trailing_enabled=True,
        trailing_activate_profit_pct=0.01,
        trailing_distance_pct=0.002,
        time_stop_mode="fixed",
        time_stop_bars=48,
    )
    state = {
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 130.0,
        "peak_price": 100.0,
        "trough_price": 100.0,
        "trailing_active": False,
    }

    p1, r1, state = evaluate_dynamic_exit(
        side="buy",
        state=state,
        row_open=100.0,
        row_high=101.5,
        row_low=101.35,
        bars_held=1,
        volatility=0.01,
        volatility_ref=0.01,
        cfg=cfg,
    )
    assert p1 is None
    assert r1 is None
    assert state["trailing_active"] is True

    p2, r2, state = evaluate_dynamic_exit(
        side="buy",
        state=state,
        row_open=101.4,
        row_high=101.6,
        row_low=101.2,
        bars_held=2,
        volatility=0.01,
        volatility_ref=0.01,
        cfg=cfg,
    )
    assert r2 == "trailing_stop"
    assert p2 == pytest.approx(101.3968, rel=1e-6)


def test_experiment_split_does_not_touch_final_test_split():
    idx = pd.date_range("2026-01-01", periods=1000, freq="5min")
    frame = pd.DataFrame({"x": range(1000)}, index=idx)

    train_val, validation_index, meta = split_for_validation_experiment(frame)

    test_start_ts = pd.Timestamp(meta.test_start_ts)
    assert meta.test_split_touched is False
    assert train_val.index.max() < test_start_ts
    assert validation_index.max() < test_start_ts


def test_missing_ta_dependency_message_is_clear(monkeypatch, capsys):
    def _raise_missing_ta(_name: str):
        raise ModuleNotFoundError("No module named 'ta'", name="ta")

    monkeypatch.setattr("scripts.run_exit_rule_experiment.importlib.import_module", _raise_missing_ta)

    with pytest.raises(ModuleNotFoundError):
        _ensure_optional_dependencies_for_exit_experiment()

    captured = capsys.readouterr()
    assert "Missing dependency: ta. Run python -m pip install -r requirements-colab.txt" in captured.err


def test_report_never_claims_guaranteed_profit():
    meta = ValidationSplitMeta(
        train_end=700,
        val_end=850,
        test_start=850,
        validation_start_ts="2026-01-01 00:00:00",
        validation_end_ts="2026-01-02 00:00:00",
        test_start_ts="2026-01-03 00:00:00",
        test_split_touched=False,
    )

    df = pd.DataFrame(
        [
            {
                "k_stop": 1.5,
                "k_take": 2.2,
                "time_stop_bars": 48,
                "time_stop_mode": "fixed",
                "trailing_enabled": True,
                "trailing_activate_profit_pct": 0.004,
                "trailing_distance_pct": 0.003,
                "max_position_fraction": 0.1,
                "weekly_return_pct": 0.6,
                "sharpe": 0.4,
                "max_drawdown_pct": 0.4,
                "trade_count": 95,
                "win_rate_pct": 51.0,
                "profit_factor": 1.2,
                "avg_win": 8.0,
                "avg_loss": -5.0,
                "expectancy": 1.2,
                "exit_reason_counts": json.dumps({"stop_loss": 10, "take_profit": 8}),
                "horizon_mismatch": True,
                "test_split_touched": False,
                "rejected": False,
                "reject_reason": "",
            }
        ]
    )

    md = build_exit_rule_experiment_markdown(df, meta)
    assert "guaranteed profit" not in md.lower()
