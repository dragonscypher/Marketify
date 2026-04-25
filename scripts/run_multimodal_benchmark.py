"""Multimodal walk-forward benchmark.

Compares four model lanes on the same untouched test split:
  1. ridge   - tabular Ridge regression
  2. xgb     - tabular XGBoost
  3. gru     - GRU sequence model
  4. fusion  - deterministic fusion (xgb + gru + neutral sentiment)

Saves to reports/:
  multimodal_leaderboard.csv
  multimodal_leaderboard.md
  multimodal_real_benchmark.md
  multimodal_trades.csv
  multimodal_equity_curve.csv

Reports 1% weekly PASS/FAIL honestly.  If weekly return < 1% -> FAIL with exact numbers.

Run only in Colab (GRU training requires GPU / adequate RAM):
  python scripts/colab_check.py
  python -m pip install -r requirements-colab.txt
  python -m pip install -e .
  python scripts/run_multimodal_benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"
WEEKLY_GOAL = 0.01
DAILY_GOAL = 0.01


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free
    downside = returns[returns < 0]
    if downside.empty or float(downside.std()) < 1e-10:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(len(returns)))


def _cvar_95(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    q5 = returns.quantile(0.05)
    tail = returns[returns <= q5]
    return float(abs(tail.mean())) if not tail.empty else 0.0


def _win_rate(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    wins = (pnl > 0).sum()
    return float(wins / len(pnl) * 100.0)


def _expectancy(pnl: pd.Series) -> float:
    return float(pnl.mean()) if not pnl.empty else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return 0.0
    running_max = eq.cummax()
    dd = (eq / running_max - 1.0).min()
    return float(abs(dd))


def _weekly_return(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return 0.0
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def _daily_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.iloc[-1])


def _compute_extended_metrics(
    equity: pd.Series,
    trade_diag: pd.DataFrame,
    weekly_goal: float = WEEKLY_GOAL,
    daily_goal: float = DAILY_GOAL,
) -> dict:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    returns = eq.pct_change().dropna()

    pnl = pd.Series(dtype=float)
    if not trade_diag.empty and "pnl" in trade_diag.columns:
        pnl = pd.to_numeric(trade_diag["pnl"], errors="coerce").dropna()

    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) > 1e-10:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(min(len(returns), 252)))

    weekly_ret = _weekly_return(eq)
    daily_ret = _daily_return(returns)
    trade_count = len(pnl)
    win_rate = _win_rate(pnl)
    expectancy = _expectancy(pnl)
    max_dd = _max_drawdown(eq)
    cvar = _cvar_95(returns)
    sortino = _sortino(returns)

    weekly_pass = weekly_ret >= weekly_goal
    daily_pass = daily_ret >= daily_goal

    return {
        "weekly_return_pct": round(weekly_ret * 100, 4),
        "daily_return_pct": round(daily_ret * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "cvar_95_pct": round(cvar * 100, 4),
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate, 2),
        "expectancy": round(expectancy, 4),
        "weekly_pass": weekly_pass,
        "daily_pass": daily_pass,
        "weekly_status": "PASS" if weekly_pass else "FAIL",
        "daily_status": "PASS" if daily_pass else "FAIL",
        "target_weekly_pct": round(weekly_goal * 100, 2),
        "target_daily_pct": round(daily_goal * 100, 2),
    }


def _pick_best_model(rows: list[dict]) -> dict:
    def _key(row: dict) -> tuple:
        m = row["metrics"]
        return (
            float(m.get("weekly_return_pct", 0.0)),
            float(m.get("sharpe_ratio", 0.0)),
            -float(m.get("max_drawdown_pct", 0.0)),
            int(m.get("trade_count", 0)),
        )
    return sorted(rows, key=_key, reverse=True)[0]


def _write_leaderboard(rows: list[dict], reports_dir: Path) -> None:
    cols = [
        "model",
        "weekly_return_pct", "daily_return_pct",
        "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "cvar_95_pct",
        "trade_count", "win_rate_pct", "expectancy",
        "weekly_status", "daily_status",
    ]
    records = []
    for row in rows:
        rec = {"model": row["model"]}
        rec.update({k: row["metrics"].get(k, "") for k in cols if k != "model"})
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(reports_dir / "multimodal_leaderboard.csv", index=False)

    md_lines = ["# Multimodal Model Leaderboard", ""]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    md_lines.extend([header, sep])
    for rec in records:
        md_lines.append("| " + " | ".join(str(rec.get(c, "")) for c in cols) + " |")
    md_lines.append("")
    (reports_dir / "multimodal_leaderboard.md").write_text("\n".join(md_lines), encoding="utf-8")


def _write_real_benchmark_md(best: dict, all_rows: list[dict], reports_dir: Path) -> None:
    m = best["metrics"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# Multimodal Real Benchmark",
        "",
        f"Generated: {ts}",
        "",
        f"Best model: {best['model']}",
        "",
        f"Weekly return: {m['weekly_return_pct']}%  ->  {m['weekly_status']} "
        f"(target: {m['target_weekly_pct']}%)",
        f"Daily return:  {m['daily_return_pct']}%  ->  {m['daily_status']} "
        f"(target: {m['target_daily_pct']}%)",
        f"Sharpe:        {m['sharpe_ratio']}",
        f"Sortino:       {m['sortino_ratio']}",
        f"Max drawdown:  {m['max_drawdown_pct']}%",
        f"CVaR 5%:       {m['cvar_95_pct']}%",
        f"Trades:        {m['trade_count']}",
        f"Win rate:      {m['win_rate_pct']}%",
        f"Expectancy:    {m['expectancy']}",
        "",
        "## All models",
        "",
        "| Model | Weekly% | Sharpe | Max DD% | Trades | Weekly |",
        "| ----- | ------- | ------ | ------- | ------ | ------ |",
    ]
    for row in all_rows:
        r = row["metrics"]
        lines.append(
            f"| {row['model']} | {r['weekly_return_pct']} | {r['sharpe_ratio']} | "
            f"{r['max_drawdown_pct']} | {r['trade_count']} | {r['weekly_status']} |"
        )

    if not m["weekly_pass"]:
        lines += [
            "",
            "WARNING: FAIL — weekly return below 1% target.",
            f"Actual: {m['weekly_return_pct']}%.  Target: {m['target_weekly_pct']}%.",
            "No profit claims made.  Continue improving signal quality.",
        ]

    lines.append("")
    (reports_dir / "multimodal_real_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _write_trades_and_equity(
    best_model: str,
    all_rows: list[dict],
    reports_dir: Path,
) -> None:
    for row in all_rows:
        if row["model"] == best_model:
            trade_diag: pd.DataFrame = row.get("trade_diagnostics", pd.DataFrame())
            trade_diag.to_csv(reports_dir / "multimodal_trades.csv", index=False)

            equity: pd.Series = row.get("equity", pd.Series(dtype=float))
            equity_df = equity.reset_index()
            equity_df.columns = ["timestamp", "equity"]
            equity_df.to_csv(reports_dir / "multimodal_equity_curve.csv", index=False)
            break


def main() -> int:
    from marketify.backtest.simulator import run_walk_forward_backtest
    from marketify.config import AppConfig

    _ensure_dir(REPORTS)
    config = AppConfig()

    model_names = ["ridge", "xgb", "gru", "fusion"]
    rows: list[dict] = []

    for model_name in model_names:
        print(f"\n[multimodal] Running lane: {model_name} ...")
        lane_config = deepcopy(config)
        lane_config.broker.backtest_model = model_name  # type: ignore[attr-defined]

        try:
            result = run_walk_forward_backtest(lane_config)
        except Exception as exc:
            print(f"  ERROR in {model_name}: {exc}")
            rows.append({
                "model": model_name,
                "metrics": {
                    "weekly_return_pct": 0.0,
                    "daily_return_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "cvar_95_pct": 0.0,
                    "trade_count": 0,
                    "win_rate_pct": 0.0,
                    "expectancy": 0.0,
                    "weekly_pass": False,
                    "daily_pass": False,
                    "weekly_status": "ERROR",
                    "daily_status": "ERROR",
                    "target_weekly_pct": round(WEEKLY_GOAL * 100, 2),
                    "target_daily_pct": round(DAILY_GOAL * 100, 2),
                    "error": str(exc),
                },
                "equity": pd.Series(dtype=float),
                "trade_diagnostics": pd.DataFrame(),
            })
            continue

        equity: pd.Series = result.get("equity", pd.Series(dtype=float))
        trade_diag: pd.DataFrame = result.get("trade_diagnostics", pd.DataFrame())
        metrics = _compute_extended_metrics(equity, trade_diag)
        rows.append({
            "model": model_name,
            "metrics": metrics,
            "equity": equity,
            "trade_diagnostics": trade_diag,
        })
        status = metrics["weekly_status"]
        print(
            f"  {model_name}: weekly={metrics['weekly_return_pct']}%  "
            f"sharpe={metrics['sharpe_ratio']}  dd={metrics['max_drawdown_pct']}%  "
            f"trades={metrics['trade_count']}  ->  {status}"
        )

    if not rows:
        print("ERROR: all lanes failed -- no report written.")
        return 1

    best = _pick_best_model(rows)
    print(f"\n[multimodal] Best model: {best['model']}")

    _write_leaderboard(rows, REPORTS)
    _write_real_benchmark_md(best, rows, REPORTS)
    _write_trades_and_equity(best["model"], rows, REPORTS)

    print(f"\nReports written to {REPORTS}/")
    for fname in [
        "multimodal_leaderboard.csv",
        "multimodal_leaderboard.md",
        "multimodal_real_benchmark.md",
        "multimodal_trades.csv",
        "multimodal_equity_curve.csv",
    ]:
        exists = (REPORTS / fname).exists()
        print(f"  {'OK' if exists else 'MISSING'}: {fname}")

    winner_m = best["metrics"]
    if not winner_m.get("weekly_pass"):
        print(
            f"\n[FAIL] Weekly {winner_m['weekly_return_pct']}% < {winner_m['target_weekly_pct']}% target. "
            "Signal quality improvement required."
        )
        return 0

    print(f"\n[PASS] Weekly {winner_m['weekly_return_pct']}% >= {winner_m['target_weekly_pct']}% target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
