"""Generate failure analysis report from trade diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def generate_failure_analysis(
    diag: pd.DataFrame,
    equity: pd.Series,
    benchmark_json: dict,
    output_dir: Path,
) -> Path:
    """Write reports/failure_analysis.md from trade diagnostics.

    Args:
        diag: DataFrame from simulator trade_diagnostics.
        equity: equity Series from backtest.
        benchmark_json: real_benchmark report dict.
        output_dir: directory to write to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ts_now = datetime.now(timezone.utc).isoformat()

    status = benchmark_json.get("status", "UNKNOWN")
    weekly_ret = benchmark_json.get("weekly_return_pct", 0.0)

    lines.append("# Failure Analysis — Real Backtest")
    lines.append(f"**Generated:** {ts_now}")
    lines.append(f"**Benchmark Status:** {status}")
    lines.append(f"**Weekly Return:** {weekly_ret}% (target: 1%)")
    lines.append("")

    if diag.empty:
        lines.append("## No trades executed")
        lines.append("Model generated zero entries. Cannot analyze.")
        path = output_dir / "failure_analysis.md"
        path.write_text("\n".join(lines))
        return path

    n = len(diag)
    wins = diag[diag["pnl"] > 0]
    losses = diag[diag["pnl"] < 0]
    flat = diag[diag["pnl"] == 0]

    total_pnl = diag["pnl"].sum()
    win_rate = len(wins) / n * 100 if n > 0 else 0.0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity
    if len(equity) > 0:
        running_max = equity.cummax()
        dd = (equity / running_max - 1.0)
        max_dd_pct = abs(float(dd.min())) * 100
    else:
        max_dd_pct = 0.0

    # --- Summary ---
    lines.append("## Summary")
    lines.append(f"- Total trades: {n}")
    lines.append(f"- Wins: {len(wins)}, Losses: {len(losses)}, Flat: {len(flat)}")
    lines.append(f"- Win rate: {win_rate:.1f}%")
    lines.append(f"- Avg win: ${avg_win:.2f}")
    lines.append(f"- Avg loss: ${avg_loss:.2f}")
    lines.append(f"- Profit factor: {profit_factor:.2f}")
    lines.append(f"- Total PnL: ${total_pnl:.2f}")
    lines.append(f"- Max drawdown: {max_dd_pct:.2f}%")
    lines.append("")

    # --- Biggest losers ---
    lines.append("## Biggest Losing Trades")
    worst = diag.nsmallest(min(5, n), "pnl")
    lines.append("| Timestamp | Side | Qty | Entry | Exit | PnL | Return% | Exit Reason |")
    lines.append("|-----------|------|-----|-------|------|-----|---------|-------------|")
    for _, r in worst.iterrows():
        lines.append(
            f"| {r['timestamp']} | {r['side']} | {r['qty']} | "
            f"{r['entry_price']:.2f} | {r['exit_price']:.2f} | "
            f"${r['pnl']:.2f} | {r['return_pct']:.2f}% | {r['exit_reason']} |"
        )
    lines.append("")

    # --- Best trades ---
    lines.append("## Best Trades")
    best = diag.nlargest(min(5, n), "pnl")
    lines.append("| Timestamp | Side | Qty | Entry | Exit | PnL | Return% | Exit Reason |")
    lines.append("|-----------|------|-----|-------|------|-----|---------|-------------|")
    for _, r in best.iterrows():
        lines.append(
            f"| {r['timestamp']} | {r['side']} | {r['qty']} | "
            f"{r['entry_price']:.2f} | {r['exit_price']:.2f} | "
            f"${r['pnl']:.2f} | {r['return_pct']:.2f}% | {r['exit_reason']} |"
        )
    lines.append("")

    # --- PnL by exit reason ---
    lines.append("## PnL by Exit Reason")
    lines.append("| Exit Reason | Count | Total PnL | Avg PnL | Win Rate |")
    lines.append("|-------------|-------|-----------|---------|----------|")
    for reason, grp in diag.groupby("exit_reason"):
        cnt = len(grp)
        tot = grp["pnl"].sum()
        avg = grp["pnl"].mean()
        wr = len(grp[grp["pnl"] > 0]) / cnt * 100 if cnt > 0 else 0
        lines.append(f"| {reason} | {cnt} | ${tot:.2f} | ${avg:.2f} | {wr:.1f}% |")
    lines.append("")

    # --- PnL by confidence bucket ---
    lines.append("## PnL by Confidence Bucket")
    if "signal_confidence" in diag.columns:
        bins = [0, 0.25, 0.5, 0.75, 1.0, float("inf")]
        labels = ["0-25%", "25-50%", "50-75%", "75-100%", ">100%"]
        diag_c = diag.copy()
        diag_c["conf_bucket"] = pd.cut(diag_c["signal_confidence"], bins=bins, labels=labels, right=True)
        lines.append("| Confidence | Count | Total PnL | Avg PnL |")
        lines.append("|------------|-------|-----------|---------|")
        for bucket, grp in diag_c.groupby("conf_bucket", observed=True):
            if len(grp) > 0:
                lines.append(f"| {bucket} | {len(grp)} | ${grp['pnl'].sum():.2f} | ${grp['pnl'].mean():.2f} |")
    lines.append("")

    # --- PnL by volatility bucket ---
    lines.append("## PnL by Volatility Bucket")
    if "cvar_95" in diag.columns and len(diag) > 0:
        try:
            diag_v = diag.copy()
            diag_v["vol_bucket"] = pd.qcut(diag_v["cvar_95"], q=min(4, len(diag)), duplicates="drop")
            lines.append("| Volatility Bucket | Count | Total PnL | Avg PnL |")
            lines.append("|-------------------|-------|-----------|---------|")
            for bucket, grp in diag_v.groupby("vol_bucket", observed=True):
                lines.append(f"| {bucket} | {len(grp)} | ${grp['pnl'].sum():.2f} | ${grp['pnl'].mean():.2f} |")
        except Exception:
            lines.append("(insufficient data for volatility buckets)")
    lines.append("")

    # --- Root cause diagnosis ---
    lines.append("## Root Cause Diagnosis")
    lines.append("")

    # Bad signal?
    if win_rate < 50:
        lines.append(f"- **Bad signal**: Win rate {win_rate:.1f}% < 50%. Model predictions not accurate enough.")
    else:
        lines.append(f"- Signal quality OK: Win rate {win_rate:.1f}%.")

    # Stop/TP asymmetry?
    sl_exits = diag[diag["exit_reason"] == "stop_loss"]
    tp_exits = diag[diag["exit_reason"] == "take_profit"]
    ts_exits = diag[diag["exit_reason"] == "time_stop"]
    if len(sl_exits) > len(tp_exits) * 1.5 and len(sl_exits) > 5:
        lines.append(
            f"- **Stop loss dominance**: {len(sl_exits)} stop-loss exits vs {len(tp_exits)} take-profit. "
            "SL/TP ratio may be too tight."
        )
    if len(ts_exits) > n * 0.3:
        lines.append(
            f"- **Time-stop excess**: {len(ts_exits)}/{n} trades hit time stop. "
            "Positions held too long without resolution."
        )

    # Risk sizing?
    if avg_win > 0 and abs(avg_loss) > avg_win * 1.5:
        lines.append(
            f"- **Asymmetric losses**: Avg loss (${avg_loss:.2f}) much larger than avg win (${avg_win:.2f}). "
            "Risk/reward sizing imbalanced."
        )

    # Profit factor
    if profit_factor < 1.0:
        lines.append(f"- **Unprofitable**: Profit factor {profit_factor:.2f} < 1.0. Losses exceed gains.")
    elif profit_factor < 1.5:
        lines.append(f"- **Marginal**: Profit factor {profit_factor:.2f}. Barely profitable after costs.")

    lines.append("")
    lines.append("## Disclaimer")
    lines.append("Analysis of paper backtest on historical data. Not financial advice.")

    path = output_dir / "failure_analysis.md"
    path.write_text("\n".join(lines))
    return path
