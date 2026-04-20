"""Conservative validation-only parameter tuning.

Grid-search over safe parameter ranges on VALIDATION split only.
Test split never touched during search. Reject dangerous configs.

Produces:
  reports/tuning_leaderboard.csv
  reports/tuning_leaderboard.md
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- Tuning grid: conservative ranges only ---
TUNING_GRID = {
    "stop_loss_pct": [0.008, 0.01, 0.012, 0.015],
    "take_profit_pct": [0.012, 0.015, 0.02, 0.025],
    "max_position_fraction": [0.05, 0.08, 0.10],
    "time_stop_bars": [24, 48, 72],
}

# Safety constraints
MIN_TRADES = 10
MAX_DRAWDOWN_LIMIT = 0.15  # 15% max
MAX_POSITION_FRACTION_LIMIT = 0.15  # no hidden leverage
MAX_TURNOVER_PER_BAR = 0.5  # reject unrealistic churning


@dataclass
class TuningResult:
    stop_loss_pct: float
    take_profit_pct: float
    max_position_fraction: float
    time_stop_bars: int
    trade_count: int
    weekly_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate_pct: float
    rejected: bool
    reject_reason: str


def _run_single_config(
    stop_loss_pct: float,
    take_profit_pct: float,
    max_position_fraction: float,
    time_stop_bars: int,
) -> TuningResult:
    """Run one backtest config on validation data."""
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_walk_forward_backtest
    from marketify.config import AppConfig

    config = AppConfig()
    config.broker.stop_loss_pct = stop_loss_pct
    config.broker.take_profit_pct = take_profit_pct
    config.broker.max_position_fraction = max_position_fraction
    config.broker.time_stop_bars = time_stop_bars
    # Use fresh DB path per config to avoid state leaking
    config.broker.db_path = f"data/tuning_temp_{hash((stop_loss_pct, take_profit_pct, max_position_fraction, time_stop_bars)) & 0xFFFFFFFF}.db"

    reject_reason = ""

    # Safety: reject hidden leverage
    if max_position_fraction > MAX_POSITION_FRACTION_LIMIT:
        return TuningResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_fraction=max_position_fraction,
            time_stop_bars=time_stop_bars,
            trade_count=0, weekly_return_pct=0, max_drawdown_pct=0,
            sharpe_ratio=0, profit_factor=0, win_rate_pct=0,
            rejected=True,
            reject_reason=f"max_position_fraction {max_position_fraction} > {MAX_POSITION_FRACTION_LIMIT}",
        )

    try:
        result = run_walk_forward_backtest(config)
        equity = result["equity"]
        diag = result["trade_diagnostics"]

        trade_count = len(diag) if not diag.empty else 0

        # Reject too few trades
        if trade_count < MIN_TRADES:
            return TuningResult(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_position_fraction=max_position_fraction,
                time_stop_bars=time_stop_bars,
                trade_count=trade_count, weekly_return_pct=0, max_drawdown_pct=0,
                sharpe_ratio=0, profit_factor=0, win_rate_pct=0,
                rejected=True,
                reject_reason=f"too few trades: {trade_count} < {MIN_TRADES}",
            )

        # Benchmark metrics
        history = [
            {"ts": str(ts), "equity": float(eq)}
            for ts, eq in zip(equity.index, equity.values)
        ]
        bm = compute_benchmark(history, weekly_goal=0.01)

        # Reject high drawdown
        if bm.max_drawdown > MAX_DRAWDOWN_LIMIT:
            reject_reason = f"drawdown {bm.max_drawdown:.2%} > {MAX_DRAWDOWN_LIMIT:.0%}"

        # Profit factor
        wins = diag[diag["pnl"] > 0]["pnl"].sum() if not diag.empty else 0
        losses_abs = abs(diag[diag["pnl"] < 0]["pnl"].sum()) if not diag.empty else 0
        pf = wins / losses_abs if losses_abs > 0 else float("inf")

        # Win rate
        win_count = len(diag[diag["pnl"] > 0]) if not diag.empty else 0
        wr = win_count / trade_count * 100 if trade_count > 0 else 0

        # Turnover check
        n_bars = len(equity)
        if n_bars > 0 and trade_count / n_bars > MAX_TURNOVER_PER_BAR:
            reject_reason = f"turnover {trade_count/n_bars:.2f} > {MAX_TURNOVER_PER_BAR}"

        return TuningResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_fraction=max_position_fraction,
            time_stop_bars=time_stop_bars,
            trade_count=trade_count,
            weekly_return_pct=round(bm.weekly_return * 100, 4),
            max_drawdown_pct=round(bm.max_drawdown * 100, 4),
            sharpe_ratio=round(bm.sharpe_ratio, 4),
            profit_factor=round(pf, 4),
            win_rate_pct=round(wr, 2),
            rejected=bool(reject_reason),
            reject_reason=reject_reason,
        )
    except Exception as e:
        return TuningResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_fraction=max_position_fraction,
            time_stop_bars=time_stop_bars,
            trade_count=0, weekly_return_pct=0, max_drawdown_pct=0,
            sharpe_ratio=0, profit_factor=0, win_rate_pct=0,
            rejected=True,
            reject_reason=f"error: {e}",
        )
    finally:
        # Clean up temp DB
        try:
            Path(config.broker.db_path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    _ensure_dir(REPORTS)

    keys = list(TUNING_GRID.keys())
    values = list(TUNING_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"[TUNING] {total} configs to evaluate (validation only)")
    print(f"[TUNING] Grid: {TUNING_GRID}")
    print()

    results: list[TuningResult] = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"  [{i}/{total}] SL={params['stop_loss_pct']} TP={params['take_profit_pct']} "
              f"POS={params['max_position_fraction']} TS={params['time_stop_bars']}", end="")
        tr = _run_single_config(**params)
        tag = "REJECTED" if tr.rejected else f"weekly={tr.weekly_return_pct}% sharpe={tr.sharpe_ratio}"
        print(f" → {tag}")
        results.append(tr)

    # Build leaderboard
    rows = [
        {
            "stop_loss_pct": r.stop_loss_pct,
            "take_profit_pct": r.take_profit_pct,
            "max_position_fraction": r.max_position_fraction,
            "time_stop_bars": r.time_stop_bars,
            "trade_count": r.trade_count,
            "weekly_return_pct": r.weekly_return_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "profit_factor": r.profit_factor,
            "win_rate_pct": r.win_rate_pct,
            "rejected": r.rejected,
            "reject_reason": r.reject_reason,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "tuning_leaderboard.csv", index=False)

    # Accepted only, sorted by Sharpe
    accepted = df[~df["rejected"]].sort_values("sharpe_ratio", ascending=False)

    # MD report
    ts_now = datetime.now(timezone.utc).isoformat()
    md = [
        "# Tuning Leaderboard (Validation Only)",
        f"**Generated:** {ts_now}",
        f"**Total configs:** {total}",
        f"**Accepted:** {len(accepted)}",
        f"**Rejected:** {len(df) - len(accepted)}",
        "",
    ]

    if len(accepted) > 0:
        best = accepted.iloc[0]
        md.append("## Best Validation Config")
        md.append(f"- stop_loss_pct: {best['stop_loss_pct']}")
        md.append(f"- take_profit_pct: {best['take_profit_pct']}")
        md.append(f"- max_position_fraction: {best['max_position_fraction']}")
        md.append(f"- time_stop_bars: {int(best['time_stop_bars'])}")
        md.append(f"- Weekly return: {best['weekly_return_pct']}%")
        md.append(f"- Sharpe: {best['sharpe_ratio']}")
        md.append(f"- Max DD: {best['max_drawdown_pct']}%")
        md.append(f"- Profit factor: {best['profit_factor']}")
        md.append(f"- Win rate: {best['win_rate_pct']}%")
        md.append(f"- Trades: {int(best['trade_count'])}")
        md.append("")
        md.append("## Top 10 Accepted Configs")
        md.append("| SL | TP | Pos% | TSBars | Trades | Weekly% | Sharpe | DD% | PF | WR% |")
        md.append("|----|----|------|--------|--------|---------|--------|-----|----|----|")
        for _, row in accepted.head(10).iterrows():
            md.append(
                f"| {row['stop_loss_pct']} | {row['take_profit_pct']} | "
                f"{row['max_position_fraction']} | {int(row['time_stop_bars'])} | "
                f"{int(row['trade_count'])} | {row['weekly_return_pct']} | "
                f"{row['sharpe_ratio']} | {row['max_drawdown_pct']} | "
                f"{row['profit_factor']} | {row['win_rate_pct']} |"
            )
    else:
        md.append("## No accepted configs")
        md.append("All configurations rejected by safety constraints.")

    md.append("")
    md.append("## Disclaimer")
    md.append("Validation-split tuning only. Test split not touched.")
    md.append("Past performance does not predict future results.")

    (REPORTS / "tuning_leaderboard.md").write_text("\n".join(md))

    print(f"\n{'='*50}")
    if len(accepted) > 0:
        best = accepted.iloc[0]
        print(f"Best validation config: weekly={best['weekly_return_pct']}% sharpe={best['sharpe_ratio']}")
    else:
        print("No accepted configs found.")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
