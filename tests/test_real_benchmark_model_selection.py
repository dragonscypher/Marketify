from __future__ import annotations

from scripts.run_backtest_and_benchmark import _pick_best_model, _trade_stats


def test_pick_best_model_prefers_weekly_then_sharpe_then_drawdown():
    rows = [
        {
            "model": "xgb",
            "report": {
                "weekly_return_pct": 0.70,
                "sharpe_ratio": 0.50,
                "max_drawdown_pct": 0.33,
                "trade_count": 90,
            },
        },
        {
            "model": "ridge",
            "report": {
                "weekly_return_pct": 0.70,
                "sharpe_ratio": 0.52,
                "max_drawdown_pct": 0.34,
                "trade_count": 88,
            },
        },
    ]

    best = _pick_best_model(rows)

    assert best["model"] == "ridge"


def test_trade_stats_handles_empty_df():
    import pandas as pd

    stats = _trade_stats(pd.DataFrame())

    assert stats["trade_count"] == 0
    assert stats["win_rate_pct"] == 0.0
    assert stats["avg_win"] == 0.0
    assert stats["avg_loss"] == 0.0
