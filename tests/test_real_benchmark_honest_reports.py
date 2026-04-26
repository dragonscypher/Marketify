from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_backtest_and_benchmark import (
    _enrich_real_report, _write_false_positive_trade_review,
    _write_next_status)


def test_enrich_real_report_adds_eligibility_fields():
    equity = pd.Series(
        [10000.0, 10020.0, 10035.0, 10060.0],
        index=pd.to_datetime([
            "2026-01-01T00:00:00",
            "2026-01-02T00:00:00",
            "2026-01-03T00:00:00",
            "2026-01-04T00:00:00",
        ]),
    )
    trade_diag = pd.DataFrame({
        "pnl": [10.0, -4.0, 8.0],
    })
    report = {
        "weekly_return_pct": 0.6,
        "daily_return_pct": 0.1,
        "max_drawdown_pct": 0.2,
        "sharpe_ratio": 1.0,
        "cvar_95_pct": 0.05,
        "weekly_pass": False,
        "drawdown_pass": True,
        "sharpe_pass": True,
        "trade_count": 3,
        "target_weekly_return_pct": 1.0,
        "status": "FAIL",
    }

    enriched = _enrich_real_report(report, equity, trade_diag)

    assert enriched["approval_count"] == 3
    assert enriched["expectancy"] > 0.0
    assert enriched["keep_coverage_pct"] >= 5.0
    assert enriched["champion_eligible"] is True
    assert "gap_to_1pct_target=" in enriched["exact_blocker"]


def test_write_next_status_and_false_positive_review(tmp_path: Path):
    champion = {
        "weekly_return_pct": 0.6977,
        "daily_return_pct": 0.08,
        "sharpe_ratio": 1.2268,
        "max_drawdown_pct": 0.1933,
        "cvar_95_pct": 0.05,
        "trade_count": 10,
        "approval_count": 10,
        "expectancy": 0.7,
        "keep_coverage_pct": 12.5,
        "weekly_pass": False,
        "champion_eligible": True,
        "exact_blocker": "gap_to_1pct_target=0.3023 percentage points; precision_on_approved_trades_not_high_enough",
    }
    comparison_rows = [
        {"model": "xgb", "report": {**champion}},
        {
            "model": "ridge",
            "report": {
                "weekly_return_pct": 0.5851,
                "daily_return_pct": 0.04,
                "sharpe_ratio": 0.397,
                "max_drawdown_pct": 0.4065,
                "cvar_95_pct": 0.1,
                "trade_count": 95,
                "approval_count": 95,
                "expectancy": 0.4,
                "keep_coverage_pct": 8.0,
                "champion_eligible": True,
            },
        },
    ]
    fix_lines = [
        "Prune or tighten momentum_trend features.",
        "Tighten high-news-risk veto.",
        "Raise approval precision on low-confidence trades.",
    ]
    diag = pd.DataFrame({
        "timestamp": ["2026-01-02"],
        "side": ["buy"],
        "pnl": [-5.0],
        "return_pct": [-0.5],
        "signal_confidence": [0.2],
        "expected_return": [0.001],
        "entry_news_risk": [0.9],
        "entry_news_event_shock": [1.0],
        "entry_vol_regime_ratio": [1.4],
        "entry_volatility_regime_code": [1.0],
        "exit_reason": ["stop_loss"],
    })

    _write_next_status("xgb", champion, comparison_rows, fix_lines, tmp_path)
    _write_false_positive_trade_review(diag, fix_lines, tmp_path)

    next_status = (tmp_path / "NEXT_STATUS.md").read_text(encoding="utf-8")
    review = (tmp_path / "false_positive_trade_review.md").read_text(encoding="utf-8")

    assert "current champion: xgb" in next_status
    assert "weekly return: 0.6977%" in next_status
    assert "approval count: 10" in next_status
    assert "exact blocker to 1.0% target" in next_status
    assert "Top 3 Highest-Impact Fixes" in review
    assert "entry_news_risk" in review
    assert "losses_from_low_confidence_trades:" in review
