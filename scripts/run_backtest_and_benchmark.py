"""Run walk-forward backtest and compute real benchmark metrics.

Produces:
  reports/trades.csv
  reports/equity_curve.csv
  reports/model_leaderboard.csv
  reports/model_leaderboard.md
  reports/real_benchmark.md
  reports/real_benchmark.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_walk_forward_backtest
    from marketify.config import AppConfig
    from marketify.models.train_pipeline import train_if_missing

    config = AppConfig()
    _ensure_dir(REPORTS)

    # --- 1. Train / load models ---
    print("[1/5] Training models (if needed)...")
    leaderboard = train_if_missing(config)
    lb_df = pd.DataFrame(leaderboard)
    lb_df.to_csv(REPORTS / "model_leaderboard.csv", index=False)

    md_lines = [
        "# Model Leaderboard",
        "",
        "| Model | Ticker | MAE | RMSE |",
        "|-------|--------|-----|------|",
    ]
    for row in leaderboard:
        md_lines.append(
            f"| {row['model']} | {row['ticker']} | {row['mae']:.6f} | {row['rmse']:.6f} |"
        )
    md_lines.append("")
    (REPORTS / "model_leaderboard.md").write_text("\n".join(md_lines))
    print(f"  Leaderboard: {len(leaderboard)} models saved.")

    # --- 2. Walk-forward backtest ---
    print("[2/5] Running walk-forward backtest...")
    result = run_walk_forward_backtest(config)
    equity: pd.Series = result["equity"]
    orders: pd.DataFrame = result["orders"]
    fills: pd.DataFrame = result["fills"]

    # --- 3. Save trades + equity curve ---
    print("[3/5] Saving trades and equity curve...")
    if not fills.empty:
        fills.to_csv(REPORTS / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "side", "qty", "price", "timestamp"]).to_csv(
            REPORTS / "trades.csv", index=False
        )

    eq_df = pd.DataFrame({"ts": equity.index, "equity": equity.values})
    eq_df.to_csv(REPORTS / "equity_curve.csv", index=False)

    trade_count = len(fills) if not fills.empty else 0
    print(f"  Trades: {trade_count}, Equity points: {len(equity)}")

    # --- 4. Compute real benchmark ---
    print("[4/5] Computing real benchmark...")

    if trade_count == 0:
        # No trades = honest FAIL
        report = _no_trade_report()
        status = "FAIL_NO_TRADES"
    else:
        history = [
            {"ts": str(ts), "equity": float(eq)}
            for ts, eq in zip(equity.index, equity.values)
        ]
        bm = compute_benchmark(history, weekly_goal=config.benchmark_weekly_goal)
        report = {
            "source": "real_backtest",
            "synthetic": False,
            "status": "PASS" if bm.weekly_pass else "FAIL",
            "weekly_return_pct": round(bm.weekly_return * 100, 4),
            "daily_return_pct": round(bm.daily_return * 100, 4),
            "max_drawdown_pct": round(bm.max_drawdown * 100, 4),
            "sharpe_ratio": round(bm.sharpe_ratio, 4),
            "cvar_95_pct": round(bm.cvar_95 * 100, 4),
            "weekly_pass": bm.weekly_pass,
            "drawdown_pass": bm.drawdown_pass,
            "sharpe_pass": bm.sharpe_pass,
            "trade_count": trade_count,
            "equity_start": float(equity.iloc[0]),
            "equity_end": float(equity.iloc[-1]),
        }
        status = report["status"]

    with open(REPORTS / "real_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- 5. Write real_benchmark.md ---
    print("[5/5] Writing real_benchmark.md...")
    ts_now = datetime.now(timezone.utc).isoformat()
    md = [
        "# Real Backtest Benchmark",
        f"**Generated:** {ts_now}",
        f"**Source:** real walk-forward backtest (NOT synthetic)",
        f"**Ticker:** {config.data.ticker}",
        f"**Status:** {status}",
        "",
        "## Metrics",
        f"- Weekly return: {report['weekly_return_pct']}%",
        f"- Daily return: {report['daily_return_pct']}%",
        f"- Max drawdown: {report['max_drawdown_pct']}%",
        f"- Sharpe ratio: {report['sharpe_ratio']}",
        f"- CVaR 5%: {report['cvar_95_pct']}%",
        f"- Trade count: {report['trade_count']}",
        "",
        "## Equity",
        f"- Start: ${report.get('equity_start', 'N/A'):,.2f}" if isinstance(report.get('equity_start'), (int, float)) else "- Start: N/A",
        f"- End: ${report.get('equity_end', 'N/A'):,.2f}" if isinstance(report.get('equity_end'), (int, float)) else "- End: N/A",
        "",
        "## Disclaimer",
        "Paper backtest on historical data. Past performance does not predict future results.",
        "No financial advice. Trading involves risk including loss of principal.",
    ]
    (REPORTS / "real_benchmark.md").write_text("\n".join(md))

    print(f"\n{'='*50}")
    print(f"Real benchmark: {status}")
    if status != "FAIL_NO_TRADES":
        print(f"  Weekly return: {report['weekly_return_pct']}%")
        print(f"  Sharpe: {report['sharpe_ratio']}")
        print(f"  Max DD: {report['max_drawdown_pct']}%")
        print(f"  Trades: {report['trade_count']}")
    else:
        print("  No trades executed. Cannot evaluate profit.")
    print(f"{'='*50}")

    return 0 if status == "PASS" else 1


def _no_trade_report() -> dict:
    return {
        "source": "real_backtest",
        "synthetic": False,
        "status": "FAIL_NO_TRADES",
        "weekly_return_pct": 0.0,
        "daily_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "cvar_95_pct": 0.0,
        "weekly_pass": False,
        "drawdown_pass": True,
        "sharpe_pass": False,
        "trade_count": 0,
    }


if __name__ == "__main__":
    sys.exit(main())
