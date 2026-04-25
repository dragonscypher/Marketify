"""Run walk-forward backtest and compute real benchmark metrics.

Produces:
  reports/trades.csv
  reports/equity_curve.csv
  reports/trade_diagnostics.csv
  reports/model_leaderboard.csv
  reports/model_leaderboard.md
  reports/real_benchmark.md
  reports/real_benchmark.json
  reports/failure_analysis.md
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
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


def _equity_history_from_series(equity: pd.Series) -> list[dict[str, str | float]]:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) < 2:
        raise ValueError("Real benchmark requires at least 2 real equity points.")
    return [{"ts": str(ts), "equity": float(value)} for ts, value in clean.items()]


def _fail_report(reason: str, trade_count: int, status: str = "FAIL") -> dict:
    return {
        "source": "real_backtest",
        "synthetic": False,
        "status": status,
        "weekly_return_pct": 0.0,
        "daily_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "cvar_95_pct": 0.0,
        "weekly_pass": False,
        "drawdown_pass": True,
        "sharpe_pass": False,
        "trade_count": trade_count,
        "reason": reason,
    }


def _build_real_benchmark_report(
    equity: pd.Series,
    trade_count: int,
    weekly_goal: float,
    compute_benchmark,
) -> dict:
    if trade_count == 0:
        return _no_trade_report()

    try:
        history = _equity_history_from_series(equity)
    except ValueError as exc:
        return _fail_report(str(exc), trade_count=trade_count)

    bm = compute_benchmark(history, weekly_goal=weekly_goal)
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
        "target_weekly_return_pct": round(weekly_goal * 100, 4),
    }
    if report["status"] != "PASS":
        report["reason"] = (
            f"weekly return {report['weekly_return_pct']}% < "
            f"{report['target_weekly_return_pct']}% target"
        )
    return report


def _trade_stats(trade_diag: pd.DataFrame) -> dict[str, float | int]:
    if trade_diag.empty or "pnl" not in trade_diag.columns:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnl = pd.to_numeric(trade_diag["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    trade_count = int(len(pnl))
    win_rate = float(len(wins) / trade_count * 100.0) if trade_count else 0.0
    return {
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
    }


def _pick_best_model(rows: list[dict]) -> dict:
    def _score(item: dict) -> tuple[float, float, float, int]:
        report = item["report"]
        return (
            float(report.get("weekly_return_pct", 0.0)),
            float(report.get("sharpe_ratio", 0.0)),
            -float(report.get("max_drawdown_pct", 0.0)),
            int(report.get("trade_count", 0)),
        )

    return sorted(rows, key=_score, reverse=True)[0]


def main() -> int:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_walk_forward_backtest
    from marketify.config import AppConfig
    from marketify.models.train_pipeline import train_if_missing

    config = AppConfig()
    _ensure_dir(REPORTS)

    # --- 1. Train / load models ---
    print("[1/6] Training models (if needed)...")
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

    # --- 2. Compare xgb vs ridge on real backtest path ---
    print("[2/6] Running walk-forward backtest (xgb vs ridge)...")
    model_rows: list[dict] = []
    for model_name in ("xgb", "ridge"):
        model_config = deepcopy(config)
        model_config.broker.backtest_model = model_name
        one = run_walk_forward_backtest(model_config)
        one_equity: pd.Series = one["equity"]
        one_fills: pd.DataFrame = one["fills"]
        one_trade_diag: pd.DataFrame = one.get("trade_diagnostics", pd.DataFrame())
        one_trade_count = len(one_fills) if not one_fills.empty else 0
        one_report = _build_real_benchmark_report(
            equity=one_equity,
            trade_count=one_trade_count,
            weekly_goal=config.benchmark_weekly_goal,
            compute_benchmark=compute_benchmark,
        )
        model_rows.append(
            {
                "model": model_name,
                "result": one,
                "report": one_report,
                "trade_stats": _trade_stats(one_trade_diag),
            }
        )

    best = _pick_best_model(model_rows)
    selected_model = best["model"]
    result = best["result"]
    equity: pd.Series = result["equity"]
    orders: pd.DataFrame = result["orders"]
    fills: pd.DataFrame = result["fills"]
    trade_diag: pd.DataFrame = result.get("trade_diagnostics", pd.DataFrame())
    comparison_rows = []
    for row in model_rows:
        rep = row["report"]
        stats = row["trade_stats"]
        comparison_rows.append(
            {
                "model": row["model"],
                "weekly_return_pct": rep.get("weekly_return_pct", 0.0),
                "daily_return_pct": rep.get("daily_return_pct", 0.0),
                "sharpe_ratio": rep.get("sharpe_ratio", 0.0),
                "max_drawdown_pct": rep.get("max_drawdown_pct", 0.0),
                "cvar_95_pct": rep.get("cvar_95_pct", 0.0),
                "trade_count": stats.get("trade_count", rep.get("trade_count", 0)),
                "win_rate_pct": stats.get("win_rate_pct", 0.0),
                "avg_win": stats.get("avg_win", 0.0),
                "avg_loss": stats.get("avg_loss", 0.0),
                "status": rep.get("status", "FAIL"),
            }
        )
    pd.DataFrame(comparison_rows).sort_values("weekly_return_pct", ascending=False).to_csv(
        REPORTS / "model_signal_comparison.csv", index=False
    )

    # --- 3. Save trades + equity curve ---
    print("[3/6] Saving trades and equity curve...")
    if not fills.empty:
        fills.to_csv(REPORTS / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "side", "qty", "price", "timestamp"]).to_csv(
            REPORTS / "trades.csv", index=False
        )

    eq_df = pd.DataFrame({"ts": equity.index, "equity": equity.values})
    eq_df.to_csv(REPORTS / "equity_curve.csv", index=False)

    # Save trade diagnostics
    if not trade_diag.empty:
        trade_diag.to_csv(REPORTS / "trade_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(REPORTS / "trade_diagnostics.csv", index=False)

    trade_count = len(fills) if not fills.empty else 0
    print(f"  Trades: {trade_count}, Equity points: {len(equity)}")

    # --- 4. Compute real benchmark ---
    print("[4/6] Computing real benchmark...")

    report = dict(best["report"])
    report["selected_model"] = selected_model
    report["model_comparison"] = comparison_rows
    status = report["status"]

    with open(REPORTS / "real_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- 5. Failure analysis ---
    print("[5/6] Generating failure analysis...")
    from marketify.backtest.failure_analysis import generate_failure_analysis

    fa_path = generate_failure_analysis(trade_diag, equity, report, REPORTS)
    print(f"  Written: {fa_path}")

    # --- 6. Write real_benchmark.md ---
    print("[6/6] Writing real_benchmark.md...")
    ts_now = datetime.now(timezone.utc).isoformat()
    md = [
        "# Real Backtest Benchmark",
        f"**Generated:** {ts_now}",
        f"**Source:** real walk-forward backtest (NOT synthetic)",
        f"**Ticker:** {config.data.ticker}",
        f"**Selected model:** {selected_model}",
        f"**Status:** {status}",
        "",
        "## Metrics",
        f"- Weekly return: {report['weekly_return_pct']}%",
        f"- Daily return: {report['daily_return_pct']}%",
        f"- Max drawdown: {report['max_drawdown_pct']}%",
        f"- Sharpe ratio: {report['sharpe_ratio']}",
        f"- CVaR 5%: {report['cvar_95_pct']}%",
        f"- Trade count: {report['trade_count']}",
        f"- Weekly target: {report.get('target_weekly_return_pct', round(config.benchmark_weekly_goal * 100, 4))}%",
        "",
        "## Model Comparison (Real Backtest Path)",
    ]
    for row in sorted(comparison_rows, key=lambda x: float(x["weekly_return_pct"]), reverse=True):
        md.extend(
            [
                f"- {row['model']}: weekly {row['weekly_return_pct']}%, sharpe {row['sharpe_ratio']}, max DD {row['max_drawdown_pct']}%, trades {int(row['trade_count'])}, win rate {row['win_rate_pct']}%",
            ]
        )

    md.extend(
        [
            "",
        "## Equity",
        f"- Start: ${report.get('equity_start', 'N/A'):,.2f}" if isinstance(report.get('equity_start'), (int, float)) else "- Start: N/A",
        f"- End: ${report.get('equity_end', 'N/A'):,.2f}" if isinstance(report.get('equity_end'), (int, float)) else "- End: N/A",
        "",
        "## Result",
        f"- Reason: {report.get('reason', 'Weekly target met from real equity history.')}",
        "- Synthetic replacement used: no",
        "",
        "## Disclaimer",
        "Paper backtest on historical data. Past performance does not predict future results.",
        "No financial advice. Trading involves risk including loss of principal.",
        ]
    )
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
    return _fail_report(
        reason="No real trades executed. No synthetic replacement used.",
        trade_count=0,
        status="FAIL_NO_TRADES",
    )


if __name__ == "__main__":
    sys.exit(main())
