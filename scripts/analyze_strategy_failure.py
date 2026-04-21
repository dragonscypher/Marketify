"""Deep-dive analysis for strategy underperformance.

Reads report artifacts and writes:
  - reports/strategy_failure_deep_dive.md
  - reports/strategy_failure_deep_dive.csv
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"

REQUIRED_REPORT_FILES = [
    "trade_diagnostics.csv",
    "trades.csv",
    "equity_curve.csv",
    "real_benchmark.json",
    "tuning_leaderboard.csv",
]


@dataclass
class AnalyzerResult:
    status: str
    message: str
    md_path: Path | None = None
    csv_path: Path | None = None


def _required_paths(reports_dir: Path) -> dict[str, Path]:
    return {name: reports_dir / name for name in REQUIRED_REPORT_FILES}


def _missing_reports(reports_dir: Path) -> list[str]:
    files = _required_paths(reports_dir)
    return [name for name, path in files.items() if not path.exists()]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines


def _bucket_confidence(diag: pd.DataFrame) -> pd.DataFrame:
    if "signal_confidence" not in diag.columns:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    if diag["signal_confidence"].dropna().empty:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    tmp = diag.copy()
    tmp["confidence_bucket"] = pd.cut(
        tmp["signal_confidence"],
        bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf],
        labels=["<=0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", ">1.0"],
        include_lowest=True,
    )
    grp = (
        tmp.groupby("confidence_bucket", observed=True)["pnl"]
        .agg(count="size", total_pnl="sum", avg_pnl="mean")
        .reset_index()
        .rename(columns={"confidence_bucket": "bucket"})
    )
    return grp


def _bucket_hold_bars(diag: pd.DataFrame) -> pd.DataFrame:
    if "hold_bars" not in diag.columns:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    if diag["hold_bars"].dropna().empty:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    tmp = diag.copy()
    tmp["hold_bars_bucket"] = pd.cut(
        tmp["hold_bars"],
        bins=[-np.inf, 4, 12, 24, 48, 96, np.inf],
        labels=["<=4", "5-12", "13-24", "25-48", "49-96", ">96"],
        include_lowest=True,
    )
    grp = (
        tmp.groupby("hold_bars_bucket", observed=True)["pnl"]
        .agg(count="size", total_pnl="sum", avg_pnl="mean")
        .reset_index()
        .rename(columns={"hold_bars_bucket": "bucket"})
    )
    return grp


def _bucket_volatility(diag: pd.DataFrame) -> pd.DataFrame:
    if "cvar_95" not in diag.columns:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    cvar = diag["cvar_95"].dropna()
    if cvar.empty or cvar.nunique() < 2:
        return pd.DataFrame(columns=["bucket", "count", "total_pnl", "avg_pnl"])
    q = min(4, int(cvar.nunique()))
    tmp = diag.copy()
    tmp["volatility_bucket"] = pd.qcut(tmp["cvar_95"], q=q, duplicates="drop")
    grp = (
        tmp.groupby("volatility_bucket", observed=True)["pnl"]
        .agg(count="size", total_pnl="sum", avg_pnl="mean")
        .reset_index()
        .rename(columns={"volatility_bucket": "bucket"})
    )
    grp["bucket"] = grp["bucket"].astype(str)
    return grp


def _group_exit_reason(diag: pd.DataFrame) -> pd.DataFrame:
    if "exit_reason" not in diag.columns:
        return pd.DataFrame(columns=["exit_reason", "count", "total_pnl", "avg_pnl", "win_rate_pct"])
    tmp = diag.copy()
    grp = (
        tmp.groupby("exit_reason", dropna=False)["pnl"]
        .agg(count="size", total_pnl="sum", avg_pnl="mean")
        .reset_index()
    )
    win_rate = (
        tmp.assign(_win=(tmp["pnl"] > 0).astype(float))
        .groupby("exit_reason", dropna=False)["_win"]
        .mean()
        .reset_index(name="win_rate")
    )
    out = grp.merge(win_rate, on="exit_reason", how="left")
    out["win_rate_pct"] = out["win_rate"] * 100.0
    out = out.drop(columns=["win_rate"])
    return out.sort_values("total_pnl", ascending=True)


def _top_trades(diag: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        c
        for c in [
            "timestamp",
            "symbol",
            "side",
            "qty",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "hold_bars",
            "signal_confidence",
            "exit_reason",
        ]
        if c in diag.columns
    ]
    worst = diag.nsmallest(min(n, len(diag)), "pnl")[cols].copy()
    best = diag.nlargest(min(n, len(diag)), "pnl")[cols].copy()
    return worst, best


def _rank_root_causes(diag: pd.DataFrame, metrics: dict[str, float]) -> list[dict[str, Any]]:
    n = max(int(metrics.get("trade_count", 0)), 1)
    win_rate = metrics.get("win_rate_pct", 0.0)
    expectancy = metrics.get("expectancy_per_trade", 0.0)
    avg_win = metrics.get("avg_win", 0.0)
    avg_loss = metrics.get("avg_loss", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)

    signal_score = 0.0
    if win_rate < 50:
        signal_score += (50 - win_rate) / 10.0
    if expectancy <= 0:
        signal_score += 2.0
    if profit_factor < 1.0:
        signal_score += 1.0

    stop_count = int((diag.get("exit_reason", pd.Series(dtype=str)) == "stop_loss").sum())
    tp_count = int((diag.get("exit_reason", pd.Series(dtype=str)) == "take_profit").sum())
    ts_count = int((diag.get("exit_reason", pd.Series(dtype=str)) == "time_stop").sum())

    exit_score = 0.0
    if stop_count > max(1, int(tp_count * 1.3)):
        exit_score += 2.0
    if (ts_count / n) > 0.35:
        exit_score += 1.5

    sizing_score = 0.0
    if avg_win > 0 and abs(avg_loss) > avg_win * 1.5:
        sizing_score += 2.0
    if len(diag) > 0:
        max_loss = abs(float(diag["pnl"].min()))
        median_abs = float(diag["pnl"].abs().median()) if "pnl" in diag.columns else 0.0
        if median_abs > 0 and max_loss > median_abs * 3.0:
            sizing_score += 1.0

    causes = [
        {
            "cause": "model_signal",
            "score": round(signal_score, 4),
            "evidence": f"win_rate={win_rate:.2f}%, expectancy={expectancy:.4f}, profit_factor={profit_factor:.4f}",
        },
        {
            "cause": "exit_rules",
            "score": round(exit_score, 4),
            "evidence": f"stop_loss={stop_count}, take_profit={tp_count}, time_stop={ts_count}",
        },
        {
            "cause": "position_sizing",
            "score": round(sizing_score, 4),
            "evidence": f"avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}",
        },
    ]
    causes.sort(key=lambda x: x["score"], reverse=True)
    return causes


def _recommendations() -> list[dict[str, str]]:
    return [
        {
            "rank": "1",
            "category": "data_features",
            "recommendation": "Add regime + microstructure features (volatility regime, spread proxy, intraday seasonality) and retrain on rolling windows.",
        },
        {
            "rank": "2",
            "category": "labels_horizon",
            "recommendation": "Rework labels to match hold horizon (triple-barrier or fixed horizon tied to time_stop_bars) so model target aligns with exits.",
        },
        {
            "rank": "3",
            "category": "model_choice",
            "recommendation": "Use calibrated classification for direction plus regression for magnitude; gate entries by calibrated probability, not raw score only.",
        },
        {
            "rank": "4",
            "category": "risk_policy",
            "recommendation": "Use volatility-scaled stop/take and confidence-scaled position caps. Do not increase leverage to chase target.",
        },
        {
            "rank": "5",
            "category": "execution_cost_assumptions",
            "recommendation": "Add explicit slippage, spread, and fee assumptions in backtest. Re-evaluate expectancy after costs.",
        },
    ]


def generate_strategy_failure_deep_dive(reports_dir: Path = REPORTS) -> AnalyzerResult:
    reports_dir.mkdir(parents=True, exist_ok=True)
    missing = _missing_reports(reports_dir)
    if missing:
        message = f"FAIL_MISSING_REPORTS: missing {', '.join(missing)}"
        md_path = reports_dir / "strategy_failure_deep_dive.md"
        csv_path = reports_dir / "strategy_failure_deep_dive.csv"
        md_path.write_text(
            "\n".join(
                [
                    "# Strategy Failure Deep Dive",
                    "",
                    "- Status: FAIL_MISSING_REPORTS",
                    f"- Details: {message}",
                    "",
                    "Cannot run deep-dive without required report artifacts.",
                ]
            )
        )
        pd.DataFrame(
            [
                {
                    "section": "status",
                    "metric": "status",
                    "value": "FAIL_MISSING_REPORTS",
                    "notes": message,
                }
            ]
        ).to_csv(csv_path, index=False)
        return AnalyzerResult(status="FAIL_MISSING_REPORTS", message=message, md_path=md_path, csv_path=csv_path)

    diag = _safe_read_csv(reports_dir / "trade_diagnostics.csv")
    trades = _safe_read_csv(reports_dir / "trades.csv")
    equity = _safe_read_csv(reports_dir / "equity_curve.csv")
    tuning = _safe_read_csv(reports_dir / "tuning_leaderboard.csv")
    with open(reports_dir / "real_benchmark.json") as f:
        benchmark = json.load(f)

    diag = _coerce_numeric(
        diag,
        [
            "pnl",
            "return_pct",
            "hold_bars",
            "signal_confidence",
            "cvar_95",
            "qty",
            "entry_price",
            "exit_price",
        ],
    )

    if "pnl" not in diag.columns:
        return AnalyzerResult(status="FAIL_ANALYSIS_ERROR", message="trade_diagnostics.csv missing required pnl column")

    n = int(len(diag))
    wins = diag[diag["pnl"] > 0]
    losses = diag[diag["pnl"] < 0]
    total_pnl = float(diag["pnl"].sum()) if n else 0.0
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
    gross_profit = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = (total_pnl / n) if n else 0.0

    bench_status = str(benchmark.get("status", "UNKNOWN"))
    bench_weekly = _to_float(benchmark.get("weekly_return_pct"), default=0.0)
    bench_sharpe = _to_float(benchmark.get("sharpe_ratio"), default=0.0)
    bench_dd = _to_float(benchmark.get("max_drawdown_pct"), default=0.0)
    bench_trades = int(_to_float(benchmark.get("trade_count"), default=float(n)))

    best_validation_weekly = 0.0
    if not tuning.empty and "weekly_return_pct" in tuning.columns:
        tuning_num = tuning.copy()
        tuning_num["weekly_return_pct"] = pd.to_numeric(tuning_num["weekly_return_pct"], errors="coerce")
        if "rejected" in tuning_num.columns:
            accepted = tuning_num[~tuning_num["rejected"].astype(bool)]
            if not accepted.empty:
                best_validation_weekly = float(accepted["weekly_return_pct"].max())
            else:
                best_validation_weekly = float(tuning_num["weekly_return_pct"].max())
        else:
            best_validation_weekly = float(tuning_num["weekly_return_pct"].max())
        if math.isnan(best_validation_weekly):
            best_validation_weekly = 0.0

    exit_reason_df = _group_exit_reason(diag)
    conf_df = _bucket_confidence(diag)
    hold_df = _bucket_hold_bars(diag)
    vol_df = _bucket_volatility(diag)
    worst10, best10 = _top_trades(diag, n=10)

    core_metrics = {
        "trade_count": float(n),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "total_pnl": total_pnl,
    }
    root_causes = _rank_root_causes(diag, core_metrics)
    primary_cause = root_causes[0]["cause"] if root_causes else "mixed"

    recommendation_rows = _recommendations()

    # Build CSV detail output
    csv_rows: list[dict[str, Any]] = [
        {"section": "headline", "metric": "real_benchmark_status", "value": bench_status, "notes": "From real_benchmark.json"},
        {"section": "headline", "metric": "weekly_return_pct", "value": round(bench_weekly, 6), "notes": "Target 1.0"},
        {"section": "headline", "metric": "sharpe_ratio", "value": round(bench_sharpe, 6), "notes": "From real_benchmark.json"},
        {"section": "headline", "metric": "max_drawdown_pct", "value": round(bench_dd, 6), "notes": "From real_benchmark.json"},
        {"section": "headline", "metric": "trades", "value": bench_trades, "notes": "From real_benchmark.json"},
        {"section": "headline", "metric": "best_validation_weekly_pct", "value": round(best_validation_weekly, 6), "notes": "From tuning_leaderboard.csv"},
        {"section": "headline", "metric": "win_rate_pct", "value": round(win_rate, 6), "notes": "trade_diagnostics.csv"},
        {"section": "headline", "metric": "avg_win", "value": round(avg_win, 6), "notes": "trade_diagnostics.csv"},
        {"section": "headline", "metric": "avg_loss", "value": round(avg_loss, 6), "notes": "trade_diagnostics.csv"},
        {"section": "headline", "metric": "profit_factor", "value": round(profit_factor, 6), "notes": "gross_profit / gross_loss"},
        {"section": "headline", "metric": "expectancy_per_trade", "value": round(expectancy, 6), "notes": "total_pnl / trades"},
        {"section": "headline", "metric": "primary_cause", "value": primary_cause, "notes": "Highest root-cause score"},
    ]

    for _, row in exit_reason_df.iterrows():
        csv_rows.append(
            {
                "section": "pnl_by_exit_reason",
                "metric": str(row.get("exit_reason", "unknown")),
                "value": round(_to_float(row.get("total_pnl"), 0.0), 6),
                "notes": f"count={int(_to_float(row.get('count'), 0))}, avg={_to_float(row.get('avg_pnl'), 0.0):.6f}, win_rate={_to_float(row.get('win_rate_pct'), 0.0):.4f}%",
            }
        )

    for df_name, section_name in [
        (conf_df, "pnl_by_confidence_bucket"),
        (hold_df, "pnl_by_hold_bars_bucket"),
        (vol_df, "pnl_by_volatility_bucket"),
    ]:
        for _, row in df_name.iterrows():
            csv_rows.append(
                {
                    "section": section_name,
                    "metric": str(row.get("bucket", "unknown")),
                    "value": round(_to_float(row.get("total_pnl"), 0.0), 6),
                    "notes": f"count={int(_to_float(row.get('count'), 0))}, avg={_to_float(row.get('avg_pnl'), 0.0):.6f}",
                }
            )

    for idx, row in enumerate(root_causes, start=1):
        csv_rows.append(
            {
                "section": "root_cause_ranking",
                "metric": f"rank_{idx}_{row['cause']}",
                "value": row["score"],
                "notes": row["evidence"],
            }
        )

    for rec in recommendation_rows:
        csv_rows.append(
            {
                "section": "recommendations",
                "metric": f"rank_{rec['rank']}_{rec['category']}",
                "value": rec["recommendation"],
                "notes": "",
            }
        )

    csv_path = reports_dir / "strategy_failure_deep_dive.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    # Build markdown output
    generated = datetime.now(timezone.utc).isoformat()
    md_lines: list[str] = [
        "# Strategy Failure Deep Dive",
        f"- Generated: {generated}",
        f"- Real Benchmark Status: {bench_status}",
        f"- Weekly Return %: {bench_weekly:.4f} (target: 1.0000)",
        f"- Sharpe: {bench_sharpe:.4f}",
        f"- Max Drawdown %: {bench_dd:.4f}",
        f"- Trades: {bench_trades}",
        f"- Best Validation Weekly %: {best_validation_weekly:.4f}",
        "",
        "## Failure Summary",
        f"- Gap to target: {bench_weekly - 1.0:.4f} percentage points",
        f"- Win rate: {win_rate:.2f}%",
        f"- Avg win: ${avg_win:.4f}",
        f"- Avg loss: ${avg_loss:.4f}",
        f"- Profit factor: {profit_factor:.4f}",
        f"- Expectancy per trade: ${expectancy:.6f}",
        "",
        "Benchmark miss is strategy result, not runtime crash. No leverage increase recommended.",
        "",
        "## PnL by Exit Reason",
    ]

    md_lines.extend(
        _markdown_table(
            ["Exit Reason", "Count", "Total PnL", "Avg PnL", "Win Rate %"],
            [
                [
                    str(r.get("exit_reason", "unknown")),
                    str(int(_to_float(r.get("count"), 0))),
                    f"{_to_float(r.get('total_pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('avg_pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('win_rate_pct'), 0.0):.2f}",
                ]
                for _, r in exit_reason_df.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## PnL by Confidence Bucket")
    md_lines.extend(
        _markdown_table(
            ["Bucket", "Count", "Total PnL", "Avg PnL"],
            [
                [
                    str(r.get("bucket", "unknown")),
                    str(int(_to_float(r.get("count"), 0))),
                    f"{_to_float(r.get('total_pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('avg_pnl'), 0.0):.4f}",
                ]
                for _, r in conf_df.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## PnL by Hold Bars Bucket")
    md_lines.extend(
        _markdown_table(
            ["Bucket", "Count", "Total PnL", "Avg PnL"],
            [
                [
                    str(r.get("bucket", "unknown")),
                    str(int(_to_float(r.get("count"), 0))),
                    f"{_to_float(r.get('total_pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('avg_pnl'), 0.0):.4f}",
                ]
                for _, r in hold_df.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## PnL by Volatility Bucket")
    md_lines.extend(
        _markdown_table(
            ["Bucket", "Count", "Total PnL", "Avg PnL"],
            [
                [
                    str(r.get("bucket", "unknown")),
                    str(int(_to_float(r.get("count"), 0))),
                    f"{_to_float(r.get('total_pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('avg_pnl'), 0.0):.4f}",
                ]
                for _, r in vol_df.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## Top 10 Losses")
    md_lines.extend(
        _markdown_table(
            ["Timestamp", "Symbol", "Side", "PnL", "Return %", "Hold Bars", "Exit Reason"],
            [
                [
                    str(r.get("timestamp", "")),
                    str(r.get("symbol", "")),
                    str(r.get("side", "")),
                    f"{_to_float(r.get('pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('return_pct'), 0.0):.4f}",
                    str(int(_to_float(r.get("hold_bars"), 0))),
                    str(r.get("exit_reason", "")),
                ]
                for _, r in worst10.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## Top 10 Wins")
    md_lines.extend(
        _markdown_table(
            ["Timestamp", "Symbol", "Side", "PnL", "Return %", "Hold Bars", "Exit Reason"],
            [
                [
                    str(r.get("timestamp", "")),
                    str(r.get("symbol", "")),
                    str(r.get("side", "")),
                    f"{_to_float(r.get('pnl'), 0.0):.4f}",
                    f"{_to_float(r.get('return_pct'), 0.0):.4f}",
                    str(int(_to_float(r.get("hold_bars"), 0))),
                    str(r.get("exit_reason", "")),
                ]
                for _, r in best10.iterrows()
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## Root Cause Ranking")
    md_lines.extend(
        _markdown_table(
            ["Rank", "Cause", "Score", "Evidence"],
            [
                [str(i), str(c["cause"]), f"{_to_float(c.get('score'), 0.0):.4f}", str(c.get("evidence", ""))]
                for i, c in enumerate(root_causes, start=1)
            ],
        )
    )

    md_lines.append("")
    md_lines.append("## Ranked Recommendations")
    md_lines.extend(
        _markdown_table(
            ["Rank", "Category", "Recommendation"],
            [[r["rank"], r["category"], r["recommendation"]] for r in recommendation_rows],
        )
    )

    md_lines.append("")
    md_lines.append("## Disclaimer")
    md_lines.append("Backtest diagnostics only. No financial advice. No profit promises.")

    md_path = reports_dir / "strategy_failure_deep_dive.md"
    md_path.write_text("\n".join(md_lines))

    return AnalyzerResult(
        status="OK",
        message=(
            f"real_benchmark_status={bench_status}; "
            f"weekly_return_pct={bench_weekly:.4f}; "
            f"expectancy_per_trade={expectancy:.6f}; "
            f"primary_cause={primary_cause}"
        ),
        md_path=md_path,
        csv_path=csv_path,
    )


def main() -> int:
    result = generate_strategy_failure_deep_dive(REPORTS)
    print(f"[ANALYZE] status={result.status}")
    print(f"[ANALYZE] {result.message}")
    if result.md_path is not None:
        print(f"[ANALYZE] markdown={result.md_path}")
    if result.csv_path is not None:
        print(f"[ANALYZE] csv={result.csv_path}")

    if result.status == "OK":
        return 0
    if result.status == "FAIL_MISSING_REPORTS":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
