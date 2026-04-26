"""Run honest real benchmark path for deployable paper trader.

Outputs:
  reports/trades.csv
  reports/equity_curve.csv
  reports/trade_diagnostics.csv
  reports/model_leaderboard.csv
  reports/model_leaderboard.md
  reports/real_benchmark.md
  reports/real_benchmark.json
  reports/NEXT_STATUS.md
  reports/false_positive_trade_review.md
  reports/feature_ablation.md

Rules:
    - xgb + ridge only on real path in this script
  - recurrent/news/regime/risk = support or veto only; not champion here
  - no exit-rule experiment path
  - no synthetic benchmark used as success proof
  - no validation-only optics
  - paper mode only; user approval required; no live trading default
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
MIN_KEEP_COVERAGE_PCT = 5.0
LOW_EDGE_ABSTAIN_FLOOR = 0.00045

FEATURE_GROUPS: dict[str, list[str]] = {
    "momentum_trend": [
        "ret_1", "ret_5", "ret_15", "ret_30",
        "ema_dist_10", "ema_dist_20", "ema_dist_50",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "breakout_up_20", "breakout_down_20",
        "signal_quality_20", "signal_stability_20", "trend_alignment_score",
    ],
    "volatility_regime": [
        "atr_14", "atr_pct", "vol_20", "intrabar_range_pct",
        "vol_regime_ratio", "volatility_regime_code", "vix_proxy",
    ],
    "volume_flow": [
        "close_location", "volume_z20", "dollar_volume_z20", "volume_regime_ratio",
    ],
    "relative_strength": [
        "ret_vs_spy_5", "ret_vs_spy_20", "ret_vs_sector_5", "ret_vs_sector_20",
    ],
    "news_macro": [
        "news_risk", "news_sentiment_score", "news_headline_count", "news_event_shock",
        "macro_event_risk", "macro_event_window", "macro_policy_bias", "macro_trend_signal",
        "macro_regime_bull", "macro_regime_bear", "macro_regime_neutral", "macro_vol_level_high",
    ],
    "intraday_time": [
        "sin_hour", "cos_hour", "sin_min", "cos_min", "is_morning", "is_afternoon", "is_late",
    ],
}

NEWS_VETO_FEATURES = [
    "news_risk", "news_sentiment_score", "news_headline_count", "news_event_shock",
]

REGIME_FEATURES = [
    "atr_14", "atr_pct", "vol_20", "intrabar_range_pct",
    "vol_regime_ratio", "volatility_regime_code", "vix_proxy",
    "macro_event_risk", "macro_event_window", "macro_policy_bias", "macro_trend_signal",
    "macro_regime_bull", "macro_regime_bear", "macro_regime_neutral", "macro_vol_level_high",
]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _equity_history_from_series(equity: pd.Series) -> list[dict[str, str | float]]:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) < 2:
        raise ValueError("Real benchmark requires at least 2 real equity points.")
    return [{"ts": str(ts), "equity": float(value)} for ts, value in clean.items()]


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    return clean.pct_change().dropna()


def _sortino_from_equity(equity: pd.Series) -> float:
    returns = _returns_from_equity(equity)
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or float(downside.std()) <= 1e-10:
        return 0.0
    return round(float(returns.mean() / downside.std() * (len(returns) ** 0.5)), 4)


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


def _no_trade_report() -> dict:
    return _fail_report(
        reason="No real trades executed. No synthetic replacement used.",
        trade_count=0,
        status="FAIL_NO_TRADES",
    )


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
            "approval_count": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    pnl = pd.to_numeric(trade_diag["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return {
            "trade_count": 0,
            "approval_count": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    trade_count = int(len(pnl))
    win_rate = float(len(wins) / trade_count * 100.0) if trade_count else 0.0
    return {
        "trade_count": trade_count,
        "approval_count": trade_count,
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "expectancy": round(float(pnl.mean()), 4),
    }


def _keep_coverage_pct(equity: pd.Series, approval_count: int) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    total_bars = max(len(clean) - 1, 1)
    return round(float(approval_count) / float(total_bars) * 100.0, 2)


def _is_eligible_report(report: dict) -> bool:
    return bool(
        int(report.get("trade_count", 0)) > 0
        and int(report.get("approval_count", 0)) > 0
        and float(report.get("expectancy", 0.0)) > 0.0
        and float(report.get("keep_coverage_pct", 0.0)) >= MIN_KEEP_COVERAGE_PCT
    )


def _gap_to_target_pct(report: dict) -> float:
    return round(
        float(report.get("target_weekly_return_pct", 1.0)) - float(report.get("weekly_return_pct", 0.0)),
        4,
    )


def _build_exact_blocker(report: dict) -> str:
    gap = _gap_to_target_pct(report)
    parts = [f"gap_to_1pct_target={gap} percentage points"]
    if int(report.get("trade_count", 0)) <= 0:
        parts.append("trade_count=0")
    if int(report.get("approval_count", 0)) <= 0:
        parts.append("approval_count=0")
    if float(report.get("expectancy", 0.0)) <= 0.0:
        parts.append("expectancy<=0")
    if float(report.get("keep_coverage_pct", 0.0)) < MIN_KEEP_COVERAGE_PCT:
        parts.append(f"keep_coverage<{MIN_KEEP_COVERAGE_PCT}%")
    if gap > 0 and int(report.get("approval_count", 0)) > 0 and float(report.get("expectancy", 0.0)) > 0.0:
        parts.append("precision_on_approved_trades_not_high_enough")
    return "; ".join(parts)


def _enrich_real_report(report: dict, equity: pd.Series, trade_diag: pd.DataFrame) -> dict:
    out = dict(report)
    stats = _trade_stats(trade_diag)
    out.update(stats)
    out["sortino_ratio"] = _sortino_from_equity(equity)
    out["keep_coverage_pct"] = _keep_coverage_pct(equity, int(stats["approval_count"]))
    out["alignment_status"] = "FIXED"
    out["champion_eligible"] = _is_eligible_report(out)
    if out["weekly_pass"] and not out["champion_eligible"]:
        out["status"] = "INVALID_FOR_CHAMPION_SELECTION"
        out["weekly_pass"] = False
    out["exact_blocker"] = _build_exact_blocker(out)
    if not out.get("reason"):
        out["reason"] = (
            "Weekly target met from real equity history."
            if out["weekly_pass"] and out["champion_eligible"]
            else out["exact_blocker"]
        )
    return out


def _pick_best_model(rows: list[dict]) -> dict:
    def _score(item: dict) -> tuple[float, float, float, float, int]:
        report = item["report"]
        return (
            float(report.get("weekly_return_pct", 0.0)),
            float(report.get("expectancy", 0.0)),
            -float(report.get("max_drawdown_pct", 0.0)),
            float(report.get("sharpe_ratio", 0.0)),
            int(report.get("trade_count", 0)),
        )

    eligible = [row for row in rows if _is_eligible_report(row["report"])]
    pool = eligible or rows
    return sorted(pool, key=_score, reverse=True)[0]


def _feature_variant_sets(feature_cols: list[str]) -> dict[str, list[str]]:
    feature_set = set(feature_cols)
    news_cols = [col for col in NEWS_VETO_FEATURES if col in feature_set]
    regime_cols = [col for col in REGIME_FEATURES if col in feature_set]
    price_cols = [col for col in feature_cols if col not in set(news_cols + regime_cols)]

    variants = {
        "price_only": price_cols,
        "price_plus_regime": [col for col in price_cols + regime_cols if col in feature_set],
        "price_plus_news_veto": [col for col in price_cols + news_cols if col in feature_set],
        "price_plus_regime_plus_news_veto": [col for col in price_cols + regime_cols + news_cols if col in feature_set],
    }
    return {name: cols for name, cols in variants.items() if len(cols) >= 5}


def _apply_smallest_strategy_patch(config):
    patched = deepcopy(config)
    patched.risk.min_expected_return = max(float(patched.risk.min_expected_return), LOW_EDGE_ABSTAIN_FLOOR)
    return patched


def _run_real_lane(
    feat: pd.DataFrame,
    feature_cols: list[str],
    config,
    compute_benchmark,
    model_name: str,
    predict_fn,
) -> dict:
    from marketify.backtest.simulator import run_backtest_from_predictions

    lane_config = deepcopy(config)
    lane_config.broker.backtest_model = model_name  # type: ignore[attr-defined]
    preds = predict_fn(
        frame=feat,
        feature_cols=feature_cols,
        target_col="target_next_ret",
        config=lane_config.model,
    )
    result = run_backtest_from_predictions(feat, preds, lane_config, model_used=model_name)
    trade_diag = result.get("trade_diagnostics", pd.DataFrame())
    fills = result.get("fills", pd.DataFrame())
    report = _build_real_benchmark_report(
        equity=result["equity"],
        trade_count=len(fills) if not fills.empty else 0,
        weekly_goal=config.benchmark_weekly_goal,
        compute_benchmark=compute_benchmark,
    )
    report = _enrich_real_report(report, result["equity"], trade_diag)
    return {
        "model": model_name,
        "result": result,
        "report": report,
    }


def _bucket_summary(diag: pd.DataFrame, column: str, group_name: str, bucket_count: int = 3) -> list[dict]:
    if diag.empty or column not in diag.columns:
        return []

    work = diag[[column, "pnl"]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna()
    if work.empty or work[column].nunique() < 2:
        return []

    quantiles = min(bucket_count, int(work[column].nunique()))
    if quantiles < 2:
        return []

    try:
        work["bucket"] = pd.qcut(work[column], q=quantiles, duplicates="drop")
    except ValueError:
        return []

    rows: list[dict] = []
    for bucket, grp in work.groupby("bucket", observed=True):
        rows.append(
            {
                "group": group_name,
                "bucket": str(bucket),
                "trade_count": int(len(grp)),
                "loss_count": int((grp["pnl"] < 0).sum()),
                "expectancy": round(float(grp["pnl"].mean()), 4),
                "total_pnl": round(float(grp["pnl"].sum()), 4),
            }
        )
    return rows


def _rank_real_path_fixes(report: dict, trade_diag: pd.DataFrame, ablation_rows: list[dict]) -> list[str]:
    ranked: list[tuple[float, str, str]] = []

    positive_ablations = [row for row in ablation_rows if float(row.get("delta_weekly_return_pct", 0.0)) > 0.0]
    positive_ablations.sort(key=lambda row: float(row["delta_weekly_return_pct"]), reverse=True)
    for row in positive_ablations[:2]:
        ranked.append(
            (
                float(row["delta_weekly_return_pct"]),
                f"keep_{row['variant']}",
                f"Keep {row['variant']} feature set: improved weekly by +{row['delta_weekly_return_pct']} pct points on real path.",
            )
        )

    if not trade_diag.empty and "pnl" in trade_diag.columns:
        diag = trade_diag.copy()
        diag["pnl"] = pd.to_numeric(diag["pnl"], errors="coerce")
        diag = diag.dropna(subset=["pnl"])

        if "entry_news_risk" in diag.columns:
            high_news = diag[pd.to_numeric(diag["entry_news_risk"], errors="coerce") >= 0.75]
            if len(high_news) >= 2 and float(high_news["pnl"].mean()) < 0.0:
                ranked.append(
                    (
                        abs(float(high_news["pnl"].sum())),
                        "news_veto",
                        f"Tighten high-news-risk veto: entry_news_risk>=0.75 bucket expectancy {round(float(high_news['pnl'].mean()), 4)}.",
                    )
                )

        high_vol = pd.DataFrame()
        if "entry_vol_regime_ratio" in diag.columns:
            high_vol = diag[pd.to_numeric(diag["entry_vol_regime_ratio"], errors="coerce") >= 1.25]
        elif "cvar_95" in diag.columns:
            cvar_num = pd.to_numeric(diag["cvar_95"], errors="coerce")
            high_vol = diag[cvar_num >= cvar_num.median()]
        if len(high_vol) >= 2 and float(high_vol["pnl"].mean()) < 0.0:
            ranked.append(
                (
                    abs(float(high_vol["pnl"].sum())),
                    "volatility_veto",
                    f"Tighten high-volatility/regime veto: stressed-vol bucket expectancy {round(float(high_vol['pnl'].mean()), 4)}.",
                )
            )

        if "signal_confidence" in diag.columns:
            conf = pd.to_numeric(diag["signal_confidence"], errors="coerce")
            low_conf = diag[conf <= conf.median()]
            if len(low_conf) >= 2 and float(low_conf["pnl"].mean()) < 0.0:
                ranked.append(
                    (
                        abs(float(low_conf["pnl"].sum())),
                        "precision_gate",
                        f"Raise approval precision on low-confidence trades: bottom-half confidence bucket expectancy {round(float(low_conf['pnl'].mean()), 4)}.",
                    )
                )

    seen: set[str] = set()
    lines: list[str] = []
    for _, key, line in sorted(ranked, key=lambda item: item[0], reverse=True):
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) == 3:
            break

    while len(lines) < 3:
        lines.append(
            f"Improve approved-trade precision without collapsing coverage: current weekly gap = {_gap_to_target_pct(report)} percentage points."
        )

    return lines[:3]


def _write_model_leaderboard(rows: list[dict], reports_dir: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "model_leaderboard.csv", index=False)

    cols = list(df.columns)
    lines = [
        "# Model Leaderboard",
        "",
        "Real benchmark path only. One deployable champion only.",
        "Champion selection objective: weekly return, then expectancy, then drawdown penalty.",
        "- xgb + ridge use same walk-forward split, same execution rules, same approval logic",
        "- recurrent = support lane only; not champion here",
        "- paper mode only; user approval required; no live trading default",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, rec in df.iterrows():
        lines.append("| " + " | ".join(str(rec[c]) for c in cols) + " |")
    lines.append("")
    (reports_dir / "model_leaderboard.md").write_text("\n".join(lines), encoding="utf-8")


def _write_real_benchmark_md(champion_name: str, champion: dict, comparison_rows: list[dict], reports_dir: Path) -> None:
    runner_up = next((row for row in comparison_rows if row["model"] != champion_name), None)
    lines = [
        "# Real Benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "Source: honest real walk-forward benchmark path only.",
        "Paper mode only. User approval required. No live trading default.",
        "No synthetic benchmark proof. No validation-only optics.",
        f"Low-edge abstain filter: min_expected_return >= {LOW_EDGE_ABSTAIN_FLOOR}",
        "",
        f"Champion: {champion_name}",
        f"Alignment status: {champion['alignment_status']}",
        f"Champion eligible: {'YES' if champion['champion_eligible'] else 'NO'}",
        f"Status: {'PASS' if champion['weekly_pass'] and champion['champion_eligible'] else 'FAIL'}",
        "",
        "## Champion Metrics",
        f"- Weekly return: {champion['weekly_return_pct']}%",
        f"- Daily return: {champion['daily_return_pct']}%",
        f"- Sharpe: {champion['sharpe_ratio']}",
        f"- Sortino: {champion['sortino_ratio']}",
        f"- Max drawdown: {champion['max_drawdown_pct']}%",
        f"- CVaR 95: {champion['cvar_95_pct']}%",
        f"- Trade count: {champion['trade_count']}",
        f"- Approval count: {champion['approval_count']}",
        f"- Expectancy: {champion['expectancy']}",
        f"- Keep coverage: {champion['keep_coverage_pct']}%",
        f"- Weekly target: {champion['target_weekly_return_pct']}%",
        f"- Exact blocker: {champion['exact_blocker']}",
        "",
        "## Real-Path Comparison",
        "| model | weekly_return_pct | daily_return_pct | sharpe | max_drawdown_pct | cvar_95_pct | trade_count | approval_count | expectancy | keep_coverage_pct | eligible | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        report = row["report"]
        lines.append(
            f"| {row['model']} | {report['weekly_return_pct']} | {report['daily_return_pct']} | {report['sharpe_ratio']} | "
            f"{report['max_drawdown_pct']} | {report['cvar_95_pct']} | {report['trade_count']} | {report['approval_count']} | "
            f"{report['expectancy']} | {report['keep_coverage_pct']} | {report['champion_eligible']} | {report['status']} |"
        )
    lines += [
        "",
        "## Selection Rule",
        "- primary: real traded weekly return",
        "- secondary: expectancy",
        "- tertiary: drawdown penalty",
        "- MAE/RMSE: diagnostic only, never champion criterion",
        "",
        "## Result",
        f"- PASS/FAIL: {'PASS' if champion['weekly_pass'] and champion['champion_eligible'] else 'FAIL'}",
        f"- Reason: {champion['reason']}",
        "",
        "## Guardrails",
        "- xgb + ridge only on real path",
        "- recurrent/news/regime/risk act as support or veto only",
        "- no exit-rule experiment path",
        "- no live trading by default",
        "",
    ]
    if runner_up is not None:
        lines += [
            "## Runner-Up",
            f"- model: {runner_up['model']}",
            f"- weekly return: {runner_up['report']['weekly_return_pct']}%",
            f"- expectancy: {runner_up['report']['expectancy']}",
            f"- max drawdown: {runner_up['report']['max_drawdown_pct']}%",
            "",
        ]
    (reports_dir / "real_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _write_false_positive_trade_review(
    trade_diag: pd.DataFrame,
    fix_lines: list[str],
    reports_dir: Path,
) -> None:
    lines = [
        "# False Positive Trade Review",
        "",
        "Real xgb benchmark path only. Focus: losing approved trades and low-expectancy buckets.",
        "Paper mode only. User approval required.",
        "",
    ]

    if trade_diag.empty or "pnl" not in trade_diag.columns:
        lines += [
            "No real trades available. Cannot review false positives.",
            "",
        ]
        (reports_dir / "false_positive_trade_review.md").write_text("\n".join(lines), encoding="utf-8")
        return

    diag = trade_diag.copy()
    diag["pnl"] = pd.to_numeric(diag["pnl"], errors="coerce")
    diag = diag.dropna(subset=["pnl"])
    losses = diag[diag["pnl"] < 0].copy()
    wins = diag[diag["pnl"] > 0].copy()

    lines += [
        "## Summary",
        f"- Total trades: {len(diag)}",
        f"- Losing trades: {len(losses)}",
        f"- Winning trades: {len(wins)}",
        f"- Loss share: {round(len(losses) / max(len(diag), 1) * 100.0, 2)}%",
        f"- Loss expectancy: {round(float(losses['pnl'].mean()), 4) if len(losses) else 0.0}",
        "",
    ]

    if len(losses):
        worst = losses.nsmallest(min(8, len(losses)), "pnl")
        cols = [
            "timestamp", "side", "pnl", "return_pct", "signal_confidence", "expected_return",
            "entry_news_risk", "entry_news_event_shock", "entry_vol_regime_ratio",
            "entry_volatility_regime_code", "exit_reason",
        ]
        available_cols = [col for col in cols if col in worst.columns]
        lines += [
            "## Worst Losing Trades",
            "| " + " | ".join(available_cols) + " |",
            "| " + " | ".join(["---"] * len(available_cols)) + " |",
        ]
        for _, row in worst.iterrows():
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in available_cols) + " |")
        lines.append("")

    bucket_sections = [
        ("signal_confidence", "Confidence buckets"),
        ("expected_return", "Expected-return buckets"),
        ("entry_news_risk", "News-risk buckets"),
        ("entry_vol_regime_ratio", "Volatility-regime buckets"),
    ]
    for column, title in bucket_sections:
        summaries = _bucket_summary(diag, column, title)
        if not summaries:
            continue
        lines += [
            f"## {title}",
            "| bucket | trade_count | loss_count | expectancy | total_pnl |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in summaries:
            lines.append(
                f"| {row['bucket']} | {row['trade_count']} | {row['loss_count']} | {row['expectancy']} | {row['total_pnl']} |"
            )
        lines.append("")

    if "exit_reason" in diag.columns:
        lines += [
            "## Exit Reason Mix",
            "| exit_reason | trade_count | expectancy | total_pnl |",
            "| --- | --- | --- | --- |",
        ]
        for reason, grp in diag.groupby("exit_reason"):
            lines.append(
                f"| {reason} | {len(grp)} | {round(float(grp['pnl'].mean()), 4)} | {round(float(grp['pnl'].sum()), 4)} |"
            )
        lines.append("")

    cluster_rows: list[tuple[float, str, str]] = []
    for column, label, predicate in [
        ("signal_confidence", "low_confidence", lambda s: pd.to_numeric(s, errors="coerce") < 0.5),
        ("entry_news_risk", "bad_news_period", lambda s: pd.to_numeric(s, errors="coerce") >= 0.75),
        ("entry_vol_regime_ratio", "bad_regime_period", lambda s: pd.to_numeric(s, errors="coerce") >= 1.25),
    ]:
        if column not in diag.columns:
            continue
        mask = predicate(diag[column])
        grp = diag[mask].copy()
        if grp.empty:
            continue
        cluster_rows.append(
            (
                float(grp["pnl"].sum()),
                label,
                f"count={len(grp)} expectancy={round(float(grp['pnl'].mean()), 4)} total_pnl={round(float(grp['pnl'].sum()), 4)}",
            )
        )

    cluster_rows = sorted(cluster_rows, key=lambda item: item[0])[:3]
    lines += [
        "## Top Losing Trade Clusters",
        "| cluster | details |",
        "| --- | --- |",
    ]
    if cluster_rows:
        for _, label, details in cluster_rows:
            lines.append(f"| {label} | {details} |")
    else:
        lines.append("| none_detected | no cluster with required fields |")
    lines.append("")

    low_conf_grp = diag[pd.to_numeric(diag["signal_confidence"], errors="coerce") < 0.5] if "signal_confidence" in diag.columns else pd.DataFrame()
    bad_news_grp = diag[pd.to_numeric(diag["entry_news_risk"], errors="coerce") >= 0.75] if "entry_news_risk" in diag.columns else pd.DataFrame()
    bad_regime_grp = diag[pd.to_numeric(diag["entry_vol_regime_ratio"], errors="coerce") >= 1.25] if "entry_vol_regime_ratio" in diag.columns else pd.DataFrame()
    lines += [
        "## Cluster Verdicts",
        f"- losses_from_low_confidence_trades: {'YES' if len(low_conf_grp) and float(low_conf_grp['pnl'].mean()) < 0.0 else 'NO'}",
        f"- losses_cluster_in_bad_news_periods: {'YES' if len(bad_news_grp) and float(bad_news_grp['pnl'].mean()) < 0.0 else 'NO'}",
        f"- losses_cluster_in_bad_regime_periods: {'YES' if len(bad_regime_grp) and float(bad_regime_grp['pnl'].mean()) < 0.0 else 'NO'}",
        "",
    ]

    lines += [
        "## Top 3 Highest-Impact Fixes",
        "| rank | fix |",
        "| --- | --- |",
    ]
    for idx, line in enumerate(fix_lines, start=1):
        lines.append(f"| {idx} | {line} |")
    lines.append("")
    (reports_dir / "false_positive_trade_review.md").write_text("\n".join(lines), encoding="utf-8")


def _write_feature_ablation_md(
    baseline_report: dict,
    ablation_rows: list[dict],
    reports_dir: Path,
) -> None:
    lines = [
        "# Feature Ablation",
        "",
        "Real xgb benchmark path only. Each row retrains xgb on same walk-forward split and reruns same execution/approval logic once.",
        "No synthetic data. No validation-only optics.",
        "",
        "## Baseline",
        f"- weekly_return_pct: {baseline_report['weekly_return_pct']}",
        f"- sharpe_ratio: {baseline_report['sharpe_ratio']}",
        f"- max_drawdown_pct: {baseline_report['max_drawdown_pct']}",
        f"- trade_count: {baseline_report['trade_count']}",
        f"- expectancy: {baseline_report['expectancy']}",
        "",
    ]

    if not ablation_rows:
        lines += ["No feature families available for ablation.", ""]
        (reports_dir / "feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## Variant Results",
        "| variant | feature_count | weekly_return_pct | delta_weekly_return_pct | sharpe_ratio | max_drawdown_pct | trade_count | expectancy | keep_coverage_pct | eligible | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(ablation_rows, key=lambda item: float(item["delta_weekly_return_pct"]), reverse=True):
        lines.append(
            f"| {row['variant']} | {row['feature_count']} | {row['weekly_return_pct']} | {row['delta_weekly_return_pct']} | "
            f"{row['sharpe_ratio']} | {row['max_drawdown_pct']} | {row['trade_count']} | {row['expectancy']} | {row['keep_coverage_pct']} | {row['eligible']} | {row['recommendation']} |"
        )
    lines.append("")

    improve_rows = [row for row in ablation_rows if row["recommendation"] == "KEEP"]
    if improve_rows:
        lines += [
            "## Keep Variants",
            "Variants below improved real weekly return without collapsing coverage.",
        ]
        for row in improve_rows[:3]:
            lines.append(
                f"- {row['variant']}: +{row['delta_weekly_return_pct']} pct points weekly."
            )
        lines.append("")
    else:
        lines += [
            "## Keep Variants",
            "No tested variant improved weekly return without collapsing coverage.",
            "",
        ]

    (reports_dir / "feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def _write_next_status(
    champion_name: str,
    champion: dict,
    comparison_rows: list[dict],
    fix_lines: list[str],
    reports_dir: Path,
) -> None:
    pass_fail = "PASS" if champion["weekly_pass"] and champion["champion_eligible"] else "FAIL"
    runner_up = next((row for row in comparison_rows if row["model"] != champion_name), None)
    lines = [
        "# NEXT_STATUS",
        "",
        "Honest real benchmark path only. One champion only. Paper mode only. User approval required.",
        f"Low-edge abstain filter: min_expected_return >= {LOW_EDGE_ABSTAIN_FLOOR}",
        "",
        f"- current champion: {champion_name}",
        f"- weekly return: {champion['weekly_return_pct']}%",
        f"- daily return: {champion['daily_return_pct']}%",
        f"- sharpe: {champion['sharpe_ratio']}",
        f"- max drawdown: {champion['max_drawdown_pct']}%",
        f"- cvar_95: {champion['cvar_95_pct']}%",
        f"- trade count: {champion['trade_count']}",
        f"- approval count: {champion['approval_count']}",
        f"- expectancy: {champion['expectancy']}",
        f"- keep coverage: {champion['keep_coverage_pct']}%",
        f"- PASS/FAIL: {pass_fail}",
        f"- exact blocker to 1.0% target: {champion['exact_blocker']}",
        "",
        "## Real-Path Comparison",
    ]
    for row in comparison_rows:
        report = row["report"]
        lines.append(
            f"- {row['model']}: weekly={report['weekly_return_pct']}% expectancy={report['expectancy']} max_dd={report['max_drawdown_pct']}% eligible={report['champion_eligible']}"
        )
    if runner_up is not None:
        lines.append(f"- runner_up: {runner_up['model']}")
    lines += [
        "",
        "## Top 3 Highest-Impact Fixes",
    ]
    for idx, line in enumerate(fix_lines, start=1):
        lines.append(f"- {idx}. {line}")
    lines += [
        "",
        "## Product Scope",
        "- xgb + ridge only on real path",
        "- recurrent = support lane only",
        "- news/regime/risk = veto + confidence shaping only",
        "- no live trading default",
        "",
    ]
    (reports_dir / "NEXT_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def _build_leaderboard_row(model: str, role: str, report: dict, champion_selected: bool, notes: str) -> dict:
    return {
        "model": model,
        "role": role,
        "champion_selected": champion_selected,
        "weekly_return_pct": report.get("weekly_return_pct", 0.0),
        "daily_return_pct": report.get("daily_return_pct", 0.0),
        "sharpe_ratio": report.get("sharpe_ratio", 0.0),
        "sortino_ratio": report.get("sortino_ratio", 0.0),
        "max_drawdown_pct": report.get("max_drawdown_pct", 0.0),
        "cvar_95_pct": report.get("cvar_95_pct", 0.0),
        "trade_count": report.get("trade_count", 0),
        "approval_count": report.get("approval_count", 0),
        "expectancy": report.get("expectancy", 0.0),
        "keep_coverage_pct": report.get("keep_coverage_pct", 0.0),
        "champion_eligible": report.get("champion_eligible", False),
        "status": report.get("status", "FAIL"),
        "notes": notes,
    }


def main() -> int:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_backtest_from_predictions
    from marketify.config import AppConfig
    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                              add_technical_features)
    from marketify.models.ridge_model import \
        rolling_train_predict as rolling_train_predict_ridge
    from marketify.models.xgb_model import \
        rolling_train_predict as rolling_train_predict_xgb

    base_config = _apply_smallest_strategy_patch(AppConfig())
    _ensure_dir(REPORTS)

    print("[1/6] Preparing honest real benchmark path...")
    raw = fetch_market_data(
        ticker=base_config.data.ticker,
        interval=base_config.data.interval,
        period=base_config.data.period,
        prepost=base_config.data.prepost,
    )
    feat = add_technical_features(raw)
    feature_cols = [col for col in TECHNICAL_FEATURE_COLUMNS if col in feat.columns]
    if not feature_cols:
        print("[FAIL] No xgb feature columns available.")
        return 1

    print("[2/6] Running xgb primary real benchmark...")
    xgb_row = _run_real_lane(
        feat=feat,
        feature_cols=feature_cols,
        config=base_config,
        compute_benchmark=compute_benchmark,
        model_name="xgb",
        predict_fn=rolling_train_predict_xgb,
    )

    print("[3/6] Running ridge real-path comparison...")
    ridge_row = _run_real_lane(
        feat=feat,
        feature_cols=feature_cols,
        config=base_config,
        compute_benchmark=compute_benchmark,
        model_name="ridge",
        predict_fn=rolling_train_predict_ridge,
    )
    model_rows = [xgb_row, ridge_row]
    champion_row = _pick_best_model(model_rows)
    champion_name = champion_row["model"]
    champion_report = champion_row["report"]

    result = champion_row["result"]
    equity: pd.Series = result["equity"]
    fills: pd.DataFrame = result.get("fills", pd.DataFrame())
    trade_diag: pd.DataFrame = result.get("trade_diagnostics", pd.DataFrame())

    print("[4/6] Writing production-path artifacts...")
    if not fills.empty:
        fills.to_csv(REPORTS / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "side", "qty", "price", "timestamp"]).to_csv(
            REPORTS / "trades.csv", index=False
        )
    pd.DataFrame({"ts": equity.index, "equity": equity.values}).to_csv(REPORTS / "equity_curve.csv", index=False)
    if not trade_diag.empty:
        trade_diag.to_csv(REPORTS / "trade_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(REPORTS / "trade_diagnostics.csv", index=False)

    leaderboard_rows = [
        _build_leaderboard_row(
            model="xgb",
            role="real_path_candidate",
            report=xgb_row["report"],
            champion_selected=champion_name == "xgb",
            notes="same walk-forward split; same execution and approval logic",
        ),
        _build_leaderboard_row(
            model="ridge",
            role="real_path_candidate",
            report=ridge_row["report"],
            champion_selected=champion_name == "ridge",
            notes="same walk-forward split; same execution and approval logic",
        ),
    ]
    _write_model_leaderboard(leaderboard_rows, REPORTS)

    print("[5/6] Running xgb feature variants on real path...")
    ablation_rows: list[dict] = []
    for variant_name, variant_cols in _feature_variant_sets(feature_cols).items():
        preds = rolling_train_predict_xgb(
            frame=feat,
            feature_cols=variant_cols,
            target_col="target_next_ret",
            config=base_config.model,
        )
        variant_config = deepcopy(base_config)
        variant_config.broker.backtest_model = "xgb"  # type: ignore[attr-defined]
        ablated_result = run_backtest_from_predictions(feat, preds, variant_config, model_used="xgb")
        ablated_report = _build_real_benchmark_report(
            equity=ablated_result["equity"],
            trade_count=len(ablated_result.get("fills", pd.DataFrame())),
            weekly_goal=base_config.benchmark_weekly_goal,
            compute_benchmark=compute_benchmark,
        )
        ablated_report = _enrich_real_report(
            ablated_report,
            ablated_result["equity"],
            ablated_result.get("trade_diagnostics", pd.DataFrame()),
        )
        ablation_rows.append(
            {
                "variant": variant_name,
                "feature_count": len(variant_cols),
                "weekly_return_pct": ablated_report["weekly_return_pct"],
                "delta_weekly_return_pct": round(
                    float(ablated_report["weekly_return_pct"]) - float(champion_report["weekly_return_pct"]),
                    4,
                ),
                "sharpe_ratio": ablated_report["sharpe_ratio"],
                "max_drawdown_pct": ablated_report["max_drawdown_pct"],
                "trade_count": ablated_report["trade_count"],
                "expectancy": ablated_report["expectancy"],
                "keep_coverage_pct": ablated_report["keep_coverage_pct"],
                "eligible": ablated_report["champion_eligible"],
                "recommendation": "KEEP" if (
                    float(ablated_report["weekly_return_pct"]) > float(champion_report["weekly_return_pct"])
                    and float(ablated_report["keep_coverage_pct"]) >= MIN_KEEP_COVERAGE_PCT
                ) else "REJECT",
            }
        )
    _write_feature_ablation_md(champion_report, ablation_rows, REPORTS)

    fix_lines = _rank_real_path_fixes(champion_report, trade_diag, ablation_rows)

    print("[6/6] Writing honest benchmark reports...")
    champion_report["selected_model"] = champion_name
    champion_report["comparison_rows"] = {row["model"]: row["report"] for row in model_rows}
    champion_report["top_fix_candidates"] = fix_lines
    champion_report["low_edge_abstain_floor"] = LOW_EDGE_ABSTAIN_FLOOR
    with open(REPORTS / "real_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(champion_report, f, indent=2)

    _write_real_benchmark_md(champion_name, champion_report, model_rows, REPORTS)
    _write_false_positive_trade_review(trade_diag, fix_lines, REPORTS)
    _write_next_status(champion_name, champion_report, model_rows, fix_lines, REPORTS)

    print("\n==================================================")
    print(f"Champion: {champion_name}")
    print(f"Weekly return: {champion_report['weekly_return_pct']}%")
    print(f"xgb weekly: {xgb_row['report']['weekly_return_pct']}%")
    print(f"ridge weekly: {ridge_row['report']['weekly_return_pct']}%")
    print(f"Sharpe: {champion_report['sharpe_ratio']}")
    print(f"Max DD: {champion_report['max_drawdown_pct']}%")
    print(f"Trades: {champion_report['trade_count']}")
    print(f"Status: {'PASS' if champion_report['weekly_pass'] and champion_report['champion_eligible'] else 'FAIL'}")
    print("==================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
