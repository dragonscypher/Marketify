"""Tests for strategy failure deep-dive analyzer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_strategy_failure import generate_strategy_failure_deep_dive


def _write_required_reports(base: Path) -> None:
    diag = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T10:00:00",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 1,
                "entry_price": 100,
                "exit_price": 110,
                "pnl": 10,
                "return_pct": 10,
                "hold_bars": 4,
                "signal_confidence": 0.9,
                "cvar_95": 0.01,
                "exit_reason": "take_profit",
            },
            {
                "timestamp": "2026-01-01T11:00:00",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 1,
                "entry_price": 100,
                "exit_price": 95,
                "pnl": -5,
                "return_pct": -5,
                "hold_bars": 8,
                "signal_confidence": 0.6,
                "cvar_95": 0.02,
                "exit_reason": "stop_loss",
            },
            {
                "timestamp": "2026-01-01T12:00:00",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 1,
                "entry_price": 100,
                "exit_price": 110,
                "pnl": 10,
                "return_pct": 10,
                "hold_bars": 5,
                "signal_confidence": 0.7,
                "cvar_95": 0.03,
                "exit_reason": "take_profit",
            },
            {
                "timestamp": "2026-01-01T13:00:00",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 1,
                "entry_price": 100,
                "exit_price": 95,
                "pnl": -5,
                "return_pct": -5,
                "hold_bars": 9,
                "signal_confidence": 0.4,
                "cvar_95": 0.04,
                "exit_reason": "stop_loss",
            },
        ]
    )
    diag.to_csv(base / "trade_diagnostics.csv", index=False)

    pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "buy", "qty": 1, "price": 100, "timestamp": "2026-01-01T10:00:00"},
            {"symbol": "AAPL", "side": "sell", "qty": 1, "price": 110, "timestamp": "2026-01-01T10:30:00"},
        ]
    ).to_csv(base / "trades.csv", index=False)

    pd.DataFrame(
        [
            {"ts": "2026-01-01T10:00:00", "equity": 10000},
            {"ts": "2026-01-01T11:00:00", "equity": 10010},
            {"ts": "2026-01-01T12:00:00", "equity": 10005},
            {"ts": "2026-01-01T13:00:00", "equity": 10010},
        ]
    ).to_csv(base / "equity_curve.csv", index=False)

    (base / "real_benchmark.json").write_text(
        json.dumps(
            {
                "source": "real_backtest",
                "synthetic": False,
                "status": "FAIL",
                "weekly_return_pct": 0.5851,
                "sharpe_ratio": 0.397,
                "max_drawdown_pct": 0.4065,
                "trade_count": 95,
            }
        )
    )

    pd.DataFrame(
        [
            {
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.015,
                "max_position_fraction": 0.1,
                "time_stop_bars": 48,
                "trade_count": 95,
                "weekly_return_pct": 0.206,
                "max_drawdown_pct": 0.40,
                "sharpe_ratio": 0.20,
                "profit_factor": 0.95,
                "win_rate_pct": 48.0,
                "rejected": False,
                "reject_reason": "",
            }
        ]
    ).to_csv(base / "tuning_leaderboard.csv", index=False)


def test_analyzer_missing_files_returns_fail_missing_reports(tmp_path: Path):
    result = generate_strategy_failure_deep_dive(tmp_path)
    assert result.status == "FAIL_MISSING_REPORTS"
    assert "FAIL_MISSING_REPORTS" in result.message
    assert result.md_path is not None and result.md_path.exists()
    assert result.csv_path is not None and result.csv_path.exists()


def test_analyzer_computes_expectancy_correctly(tmp_path: Path):
    _write_required_reports(tmp_path)
    result = generate_strategy_failure_deep_dive(tmp_path)
    assert result.status == "OK"
    assert result.csv_path is not None and result.csv_path.exists()

    df = pd.read_csv(result.csv_path)
    row = df[(df["section"] == "headline") & (df["metric"] == "expectancy_per_trade")]
    assert not row.empty
    val = float(row.iloc[0]["value"])
    # pnl sequence [10, -5, 10, -5] => total 10 over 4 trades => 2.5
    assert abs(val - 2.5) < 1e-9


def test_analyzer_does_not_claim_benchmark_pass(tmp_path: Path):
    _write_required_reports(tmp_path)
    result = generate_strategy_failure_deep_dive(tmp_path)
    assert result.status == "OK"
    assert result.md_path is not None and result.md_path.exists()

    text = result.md_path.read_text()
    assert "Real Benchmark Status: FAIL" in text
    assert "Real Benchmark Status: PASS" not in text
    assert "target: 1.0000" in text
