"""Multimodal walk-forward benchmark with validation-only fusion policy tuning.

Architecture unchanged:
  1. ridge   - tabular Ridge regression
  2. xgb     - tabular XGBoost
  3. gru     - GRU sequence model
  4. fusion  - deterministic fusion (xgb + gru + sentiment/news/regime gates)

Policy tuning rules:
  - tune fusion policy on validation slice only
  - final benchmark report uses real walk-forward path once after selection
  - no leverage increase; position sizing unchanged
  - deterministic only; no self-modification

Outputs:
  reports/validation_policy_tuning.csv
  reports/validation_policy_tuning.md
  reports/xgb_real_leaderboard.csv
  reports/xgb_real_leaderboard.md
  reports/model_leaderboard.csv
  reports/model_leaderboard.md
  reports/real_trading_benchmark.md
  reports/NEXT_STATUS.md
  reports/multimodal_trades.csv
  reports/multimodal_equity_curve.csv

Champion eligibility rules:
  - trade_count > 0 (reject phantom-pass from over-abstained fusion)
  - expectancy > 0
  - keep_coverage >= MIN_KEEP_COVERAGE_PCT
"""
from __future__ import annotations

import itertools
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from marketify.models.fusion_model import (FusionConfig,
                                           fuse_prediction_components)

REPORTS = ROOT / "reports"
WEEKLY_GOAL = 0.01
DAILY_GOAL = 0.01
VALIDATION_FRACTION = 0.15
MIN_KEEP_COVERAGE_PCT = 5.0  # min % of total periods that must have an approved trade

_GRID_MIN_CONFIDENCE = [0.30, 0.35]
_GRID_EXPECTED_FLOOR = [0.00030, 0.00035]
_GRID_MAX_DISAGREEMENT = [0.004, 0.006]
_GRID_NEWS_CUTOFF = [0.70, 0.75]
_GRID_REGIME_CUTOFF = [28.0, 30.0]
_GRID_MIN_HOLDING_BARS = [0, 6]


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

    # approval_count: trades executed on real path (paper-mode approvals)
    approval_count = trade_count

    # keep_coverage: % of total equity bars that had an approved trade
    total_bars = max(len(eq) - 1, 1)
    keep_coverage_pct = round(trade_count / total_bars * 100.0, 2)

    return {
        "weekly_return_pct": round(weekly_ret * 100, 4),
        "daily_return_pct": round(daily_ret * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "cvar_95_pct": round(cvar * 100, 4),
        "trade_count": trade_count,
        "approval_count": approval_count,
        "keep_coverage_pct": keep_coverage_pct,
        "win_rate_pct": round(win_rate, 2),
        "expectancy": round(expectancy, 4),
        "weekly_pass": weekly_pass,
        "daily_pass": daily_pass,
        "weekly_status": "PASS" if weekly_pass else "FAIL",
        "daily_status": "PASS" if daily_pass else "FAIL",
        "target_weekly_pct": round(weekly_goal * 100, 2),
        "target_daily_pct": round(daily_goal * 100, 2),
    }


def _is_eligible_for_champion(row: dict, metrics_key: str = "metrics") -> bool:
    """Reject candidates with zero trades, non-positive expectancy, or insufficient coverage.

    A phantom PASS can occur when over-tightened abstain gates produce 0 trades
    and the equity curve never moves — giving inflated returns on a flat line.
    These must be labelled INVALID_FOR_CHAMPION_SELECTION and excluded.
    """
    m = row[metrics_key]
    if int(m.get("trade_count", 0)) <= 0:
        return False
    if float(m.get("expectancy", 0.0)) <= 0.0:
        return False
    if float(m.get("keep_coverage_pct", 0.0)) < MIN_KEEP_COVERAGE_PCT:
        return False
    return True


def _pick_best_model(rows: list[dict], metrics_key: str = "metrics") -> dict:
    def _key(row: dict) -> tuple:
        m = row[metrics_key]
        return (
            float(m.get("weekly_return_pct", 0.0)),
            float(m.get("sharpe_ratio", 0.0)),
            -float(m.get("max_drawdown_pct", 0.0)),
            -float(m.get("cvar_95_pct", 0.0)),
            float(m.get("win_rate_pct", 0.0)),
            float(m.get("expectancy", 0.0)),
            int(m.get("trade_count", 0)),
        )

    eligible = [r for r in rows if _is_eligible_for_champion(r, metrics_key)]
    if not eligible:
        # All candidates ineligible — fall back to row with most trades to avoid silent phantom win
        eligible = sorted(rows, key=lambda r: int(r[metrics_key].get("trade_count", 0)), reverse=True)[:1]

    return sorted(eligible, key=_key, reverse=True)[0]


def _validation_start_index(n_rows: int) -> int:
    if n_rows <= 2:
        return 0
    start = int(round(n_rows * (1.0 - VALIDATION_FRACTION)))
    return max(0, min(start, n_rows - 2))


def _validation_slice(
    feat: pd.DataFrame,
    tabular_preds: pd.Series,
    sequence_preds: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Timestamp | None]:
    start = _validation_start_index(len(feat))
    val_feat = feat.iloc[start:].copy()
    val_tab = tabular_preds.reindex(val_feat.index)
    val_seq = sequence_preds.reindex(val_feat.index)
    start_ts = val_feat.index[0] if len(val_feat) else None
    return val_feat, val_tab, val_seq, start_ts


def _default_policy(config) -> dict:
    return {
        "fusion_min_confidence": float(getattr(config.broker, "fusion_min_confidence", 0.30)),
        "fusion_expected_return_floor": float(getattr(config.broker, "fusion_expected_return_floor", 0.0003)),
        "fusion_abstain_margin": float(getattr(config.broker, "fusion_abstain_margin", 0.0)),
        "fusion_max_model_disagreement": float(getattr(config.broker, "fusion_max_model_disagreement", 0.006)),
        "fusion_news_risk_cutoff": float(getattr(config.broker, "fusion_news_risk_cutoff", 0.75)),
        "fusion_regime_vix_cutoff": float(getattr(config.broker, "fusion_regime_vix_cutoff", 30.0)),
        "fusion_min_holding_bars": int(getattr(config.broker, "fusion_min_holding_bars", 0)),
    }


def _policy_grid(base_policy: dict) -> list[dict]:
    rows: list[dict] = []
    for combo in itertools.product(
        _GRID_MIN_CONFIDENCE,
        _GRID_EXPECTED_FLOOR,
        _GRID_MAX_DISAGREEMENT,
        _GRID_NEWS_CUTOFF,
        _GRID_REGIME_CUTOFF,
        _GRID_MIN_HOLDING_BARS,
    ):
        rows.append(
            {
                "fusion_min_confidence": float(combo[0]),
                "fusion_expected_return_floor": float(combo[1]),
                "fusion_abstain_margin": float(base_policy["fusion_abstain_margin"]),
                "fusion_max_model_disagreement": float(combo[2]),
                "fusion_news_risk_cutoff": float(combo[3]),
                "fusion_regime_vix_cutoff": float(combo[4]),
                "fusion_min_holding_bars": int(combo[5]),
            }
        )
    return rows


def _fusion_config_from_policy(policy: dict) -> FusionConfig:
    return FusionConfig(
        min_confidence=float(policy["fusion_min_confidence"]),
        min_expected_return=float(policy["fusion_expected_return_floor"]),
        abstain_margin=float(policy["fusion_abstain_margin"]),
        max_news_risk=float(policy["fusion_news_risk_cutoff"]),
        high_vol_vix_threshold=float(policy["fusion_regime_vix_cutoff"]),
        max_model_disagreement=float(policy["fusion_max_model_disagreement"]),
    )


def _apply_policy(config, policy: dict):
    cfg = deepcopy(config)
    cfg.broker.backtest_model = "fusion"  # type: ignore[attr-defined]
    cfg.broker.fusion_min_confidence = policy["fusion_min_confidence"]
    cfg.broker.fusion_expected_return_floor = policy["fusion_expected_return_floor"]
    cfg.broker.fusion_abstain_margin = policy["fusion_abstain_margin"]
    cfg.broker.fusion_max_model_disagreement = policy["fusion_max_model_disagreement"]
    cfg.broker.fusion_news_risk_cutoff = policy["fusion_news_risk_cutoff"]
    cfg.broker.fusion_regime_vix_cutoff = policy["fusion_regime_vix_cutoff"]
    cfg.broker.fusion_min_holding_bars = policy["fusion_min_holding_bars"]
    return cfg


def _run_cached_fusion_backtest(
    feat: pd.DataFrame,
    tabular_preds: pd.Series,
    sequence_preds: pd.Series,
    base_config,
    policy: dict,
    full_path: bool,
) -> dict:
    from marketify.backtest.simulator import run_backtest_from_predictions

    work_feat = feat
    work_tab = tabular_preds
    work_seq = sequence_preds
    validation_start_ts = None
    if not full_path:
        work_feat, work_tab, work_seq, validation_start_ts = _validation_slice(feat, tabular_preds, sequence_preds)

    cfg = _apply_policy(base_config, policy)
    preds = fuse_prediction_components(
        frame=work_feat,
        tabular_preds=work_tab,
        sequence_preds=work_seq,
        fusion_config=_fusion_config_from_policy(policy),
    )
    result = run_backtest_from_predictions(work_feat, preds, cfg, model_used="fusion")
    metrics = _compute_extended_metrics(result["equity"], result["trade_diagnostics"])
    return {
        "model": "fusion",
        "policy": policy,
        "metrics": metrics,
        "equity": result["equity"],
        "trade_diagnostics": result["trade_diagnostics"],
        "validation_start_ts": str(validation_start_ts) if validation_start_ts is not None else "",
    }


def _write_leaderboard(rows: list[dict], reports_dir: Path) -> None:
    cols = [
        "model",
        "weekly_return_pct", "daily_return_pct",
        "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "cvar_95_pct",
        "trade_count", "approval_count", "keep_coverage_pct",
        "win_rate_pct", "expectancy",
        "weekly_status", "champion_eligible",
    ]
    records = []
    for row in rows:
        rec = {"model": row["model"]}
        rec.update({k: row["metrics"].get(k, "") for k in cols if k not in ("model", "champion_eligible")})
        rec["champion_eligible"] = _is_eligible_for_champion(row)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(reports_dir / "model_leaderboard.csv", index=False)

    md_lines = ["# Model Leaderboard", ""]
    md_lines.append("Champion eligibility: trade_count > 0, expectancy > 0, keep_coverage >= 5%. Fusion policy selected on validation slice only.")
    md_lines.append("")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    md_lines.extend([header, sep])
    for rec in records:
        md_lines.append("| " + " | ".join(str(rec.get(c, "")) for c in cols) + " |")
    md_lines.append("")
    (reports_dir / "model_leaderboard.md").write_text("\n".join(md_lines), encoding="utf-8")


def _write_real_benchmark_md(best: dict, all_rows: list[dict], reports_dir: Path) -> None:
    m = best["metrics"]
    gap = round(float(m["target_weekly_pct"]) - float(m["weekly_return_pct"]), 4)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    is_eligible = _is_eligible_for_champion(best)
    invalid_label = "" if is_eligible else "INVALID_FOR_CHAMPION_SELECTION (trade_count=0 or expectancy<=0)"

    lines = [
        "# Real Trading Benchmark",
        "",
        f"Generated: {ts}",
        "",
        "Fusion policy selected on validation slice only. Final numbers use real benchmark path once.",
        "Champion eligibility: trade_count > 0, expectancy > 0, keep_coverage >= 5%.",
        "",
        f"Best model: {best['model']}",
    ]
    if invalid_label:
        lines.append(f"WARNING: {invalid_label}")
    lines += [
        "",
        f"Weekly return:   {m['weekly_return_pct']}%  ->  {m['weekly_status']} (target: {m['target_weekly_pct']}%)",
        f"Daily return:    {m['daily_return_pct']}%",
        f"Sharpe:          {m['sharpe_ratio']}",
        f"Sortino:         {m['sortino_ratio']}",
        f"Max drawdown:    {m['max_drawdown_pct']}%",
        f"CVaR 5%:         {m['cvar_95_pct']}%",
        f"Trades:          {m['trade_count']}",
        f"Approval count:  {m.get('approval_count', m['trade_count'])}",
        f"Keep coverage:   {m.get('keep_coverage_pct', 0.0)}%",
        f"Win rate:        {m['win_rate_pct']}%",
        f"Expectancy:      {m['expectancy']}",
        "",
        "## All models",
        "",
        "| Model | Weekly% | Daily% | Sharpe | Sortino | Max DD% | CVaR 5% | Trades | Approval | Coverage% | Expectancy | Weekly | Eligible |",
        "| ----- | ------- | ------ | ------ | ------- | ------- | ------- | ------ | -------- | --------- | ---------- | ------ | -------- |",
    ]
    for row in all_rows:
        r = row["metrics"]
        eligible_flag = "YES" if _is_eligible_for_champion(row) else "NO"
        lines.append(
            f"| {row['model']} | {r['weekly_return_pct']} | {r['daily_return_pct']} | {r['sharpe_ratio']} | "
            f"{r['sortino_ratio']} | {r['max_drawdown_pct']} | {r['cvar_95_pct']} | {r['trade_count']} | "
            f"{r.get('approval_count', r['trade_count'])} | {r.get('keep_coverage_pct', 0.0)} | "
            f"{r['expectancy']} | {r['weekly_status']} | {eligible_flag} |"
        )

    if not m["weekly_pass"] or not is_eligible:
        lines += [
            "",
            "WARNING: FAIL - weekly return below 1% target or model ineligible.",
            f"Actual: {m['weekly_return_pct']}%.  Target: {m['target_weekly_pct']}%.",
            f"Gap: {gap} percentage points.",
            "No profit claims made. Continue improving decision quality only.",
        ]

    lines.append("")
    (reports_dir / "real_trading_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _write_fusion_policy_tuning(rows: list[dict], reports_dir: Path) -> None:
    records: list[dict] = []
    for row in rows:
        p = row["policy"]
        m = row["metrics"]
        eligible = _is_eligible_for_champion(row)
        records.append(
            {
                **p,
                "validation_weekly_return_pct": m["weekly_return_pct"],
                "validation_daily_return_pct": m["daily_return_pct"],
                "validation_sharpe_ratio": m["sharpe_ratio"],
                "validation_sortino_ratio": m["sortino_ratio"],
                "validation_max_drawdown_pct": m["max_drawdown_pct"],
                "validation_cvar_95_pct": m["cvar_95_pct"],
                "validation_trade_count": m["trade_count"],
                "validation_win_rate_pct": m["win_rate_pct"],
                "validation_keep_coverage_pct": m.get("keep_coverage_pct", 0.0),
                "champion_eligible": eligible,
                "validation_weekly_status": m["weekly_status"] if eligible else "INVALID_FOR_CHAMPION_SELECTION",
                "validation_start_ts": row.get("validation_start_ts", ""),
            }
        )

    if not records:
        pd.DataFrame().to_csv(reports_dir / "validation_policy_tuning.csv", index=False)
        (reports_dir / "validation_policy_tuning.md").write_text("# Validation Policy Tuning\n\nNo rows.\n", encoding="utf-8")
        return

    df = pd.DataFrame(records).sort_values(
        by=["champion_eligible", "validation_weekly_return_pct", "validation_sharpe_ratio", "validation_max_drawdown_pct", "validation_trade_count"],
        ascending=[False, False, False, True, False],
    )
    df.to_csv(reports_dir / "validation_policy_tuning.csv", index=False)

    cols = list(df.columns)
    md_lines = ["# Validation Policy Tuning", ""]
    md_lines.append("Validation slice only. Ineligible combos (0 trades / non-positive expectancy) labelled INVALID_FOR_CHAMPION_SELECTION.")
    md_lines.append("")
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, rec in df.iterrows():
        md_lines.append("| " + " | ".join(str(rec[c]) for c in cols) + " |")
    md_lines.append("")
    (reports_dir / "validation_policy_tuning.md").write_text("\n".join(md_lines), encoding="utf-8")


def _write_xgb_real_leaderboard(xgb_rows: list[object], reports_dir: Path) -> None:
    from scripts.compare_models import build_xgb_real_leaderboard_markdown

    records = [asdict(row) for row in xgb_rows]
    df = pd.DataFrame(records)
    df.to_csv(reports_dir / "xgb_real_leaderboard.csv", index=False)

    champion = next((row for row in xgb_rows if getattr(row, "role", "") == "champion"), xgb_rows[0])
    md = build_xgb_real_leaderboard_markdown(df, champion)
    (reports_dir / "xgb_real_leaderboard.md").write_text(md, encoding="utf-8")


def _build_exact_blocker(current_metrics: dict, xgb_primary: object | None = None) -> str:
    gap = round(float(current_metrics["target_weekly_pct"]) - float(current_metrics["weekly_return_pct"]), 4)
    parts = [f"gap_to_target={gap} pct_points"]

    if float(current_metrics.get("expectancy", 0.0)) <= 0.0:
        parts.append("expectancy_not_positive")
    else:
        parts.append("per_trade_edge_still_below_target_after_safe_gates")

    if int(current_metrics.get("trade_count", 0)) <= 0:
        parts.append("no_approved_trades")

    if xgb_primary is not None:
        xgb_weekly = float(getattr(xgb_primary, "weekly_return_pct", float("nan")))
        if np.isfinite(xgb_weekly):
            parts.append(f"xgb_primary_single_model={xgb_weekly}%")

    return "; ".join(parts)


def _write_next_status(
    xgb_primary,
    baseline_fusion: dict,
    best_validation_policy: dict,
    final_tuned_fusion: dict,
    winner: dict,
    reports_dir: Path,
) -> None:
    b = baseline_fusion["metrics"]
    v = best_validation_policy["metrics"]
    t = final_tuned_fusion["metrics"]
    w = winner["metrics"]
    delta = round(float(t["weekly_return_pct"]) - float(b["weekly_return_pct"]), 4)
    gap = round(float(t["target_weekly_pct"]) - float(t["weekly_return_pct"]), 4)
    current_gap = round(float(w["target_weekly_pct"]) - float(w["weekly_return_pct"]), 4)
    exact_blocker = _build_exact_blocker(w, xgb_primary)
    champion_eligible = _is_eligible_for_champion(winner)
    invalid_label = "" if champion_eligible else "INVALID_FOR_CHAMPION_SELECTION (trade_count=0 or expectancy<=0)"

    lines = [
        "# NEXT STATUS",
        "",
        "Scope: xgb primary single-model lane + deterministic fusion decision layer. No new models. No leverage increase.",
        "",
        "## Current Champion",
        f"- current champion: {winner['model']}",
        f"- weekly return: {w['weekly_return_pct']}%",
        f"- daily return: {w['daily_return_pct']}%",
        f"- sharpe: {w['sharpe_ratio']}",
        f"- sortino: {w['sortino_ratio']}",
        f"- max drawdown: {w['max_drawdown_pct']}%",
        f"- cvar_95: {w['cvar_95_pct']}%",
        f"- trade count: {w['trade_count']}",
        f"- approval count: {w.get('approval_count', w['trade_count'])}",
        f"- expectancy: {w['expectancy']}",
        f"- win rate: {w['win_rate_pct']}%",
        f"- gap to 1.0% target: {current_gap} percentage points",
        f"- result: {'INVALID_FOR_CHAMPION_SELECTION' if invalid_label else w['weekly_status']}",
        f"- exact blocker: {exact_blocker}",
    ]
    if invalid_label:
        lines.append(f"- WARNING: {invalid_label}")
    lines += [
        "",
        "## Champion Path",
        f"- xgb_primary_single_model_weekly_return_pct: {getattr(xgb_primary, 'weekly_return_pct', '')}",
        f"- xgb_primary_single_model_sharpe: {getattr(xgb_primary, 'sharpe', '')}",
        f"- xgb_primary_single_model_max_drawdown_pct: {getattr(xgb_primary, 'max_drawdown_pct', '')}",
        f"- xgb_primary_single_model_trade_count: {getattr(xgb_primary, 'trade_count', '')}",
        "- recurrent_lane_role: support_context_only",
        "",
        "## Baseline Fusion (real benchmark path)",
        f"- weekly_return_pct: {b['weekly_return_pct']}",
        f"- daily_return_pct: {b['daily_return_pct']}",
        f"- sharpe_ratio: {b['sharpe_ratio']}",
        f"- sortino_ratio: {b['sortino_ratio']}",
        f"- max_drawdown_pct: {b['max_drawdown_pct']}",
        f"- cvar_95_pct: {b['cvar_95_pct']}",
        f"- trade_count: {b['trade_count']}",
        f"- weekly_status: {b['weekly_status']}",
        "",
        "## Best Validation Policy",
        f"- min_confidence: {best_validation_policy['policy']['fusion_min_confidence']}",
        f"- expected_return_floor: {best_validation_policy['policy']['fusion_expected_return_floor']}",
        f"- abstain_margin: {best_validation_policy['policy']['fusion_abstain_margin']}",
        f"- max_model_disagreement: {best_validation_policy['policy']['fusion_max_model_disagreement']}",
        f"- news_risk_cutoff: {best_validation_policy['policy']['fusion_news_risk_cutoff']}",
        f"- regime_vix_cutoff: {best_validation_policy['policy']['fusion_regime_vix_cutoff']}",
        f"- min_holding_bars: {best_validation_policy['policy']['fusion_min_holding_bars']}",
        f"- validation_weekly_return_pct: {v['weekly_return_pct']}",
        f"- validation_sharpe_ratio: {v['sharpe_ratio']}",
        f"- validation_max_drawdown_pct: {v['max_drawdown_pct']}",
        f"- validation_trade_count: {v['trade_count']}",
        "",
        "## Final Tuned Fusion (real benchmark path)",
        f"- weekly_return_pct: {t['weekly_return_pct']}",
        f"- daily_return_pct: {t['daily_return_pct']}",
        f"- sharpe_ratio: {t['sharpe_ratio']}",
        f"- sortino_ratio: {t['sortino_ratio']}",
        f"- max_drawdown_pct: {t['max_drawdown_pct']}",
        f"- cvar_95_pct: {t['cvar_95_pct']}",
        f"- trade_count: {t['trade_count']}",
        f"- win_rate_pct: {t['win_rate_pct']}",
        f"- weekly_delta_vs_baseline_pct_points: {delta}",
        "",
    ]

    if t["weekly_pass"]:
        lines += [
            "Result: PASS",
            f"Weekly {t['weekly_return_pct']}% >= {t['target_weekly_pct']}% target.",
        ]
    else:
        lines += [
            "Result: FAIL",
            f"Weekly {t['weekly_return_pct']}% < {t['target_weekly_pct']}% target.",
            f"Missing by: {gap} percentage points.",
        ]

    lines += [
        "",
        "## Numerical Reason",
        f"- win_rate_pct: {t['win_rate_pct']}",
        f"- expectancy: {t['expectancy']}",
        f"- trade_count: {t['trade_count']}",
        f"- approval_count: {t.get('approval_count', t['trade_count'])}",
        f"- drawdown_pct: {t['max_drawdown_pct']}",
        f"- cvar_95_pct: {t['cvar_95_pct']}",
        f"- final_tuned_fusion_eligible: {_is_eligible_for_champion(final_tuned_fusion)}",
        f"- winner_lane_after_tuning: {winner['model']}",
        f"- winner_weekly_return_pct: {w['weekly_return_pct']}",
        f"- winner_weekly_status: {w['weekly_status']}",
        "",
    ]
    (reports_dir / "NEXT_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_trades_and_equity(best_model: str, all_rows: list[dict], reports_dir: Path) -> None:
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
    from scripts.compare_models import _frozen_comparison_row, _run_real_lane

    _ensure_dir(REPORTS)
    config = AppConfig()

    rows: list[dict] = []
    shared_feat: pd.DataFrame | None = None
    xgb_preds: pd.Series | None = None
    gru_preds: pd.Series | None = None

    for model_name in ["ridge", "xgb", "gru"]:
        print(f"\n[multimodal] Running lane: {model_name} ...")
        lane_config = deepcopy(config)
        lane_config.broker.backtest_model = model_name  # type: ignore[attr-defined]

        result = run_walk_forward_backtest(lane_config)
        equity: pd.Series = result.get("equity", pd.Series(dtype=float))
        trade_diag: pd.DataFrame = result.get("trade_diagnostics", pd.DataFrame())
        metrics = _compute_extended_metrics(equity, trade_diag)
        row = {
            "model": model_name,
            "metrics": metrics,
            "equity": equity,
            "trade_diagnostics": trade_diag,
            "predictions": result.get("predictions", pd.Series(dtype=float)),
            "features": result.get("features"),
        }
        rows.append(row)
        print(
            f"  {model_name}: weekly={metrics['weekly_return_pct']}% "
            f"sharpe={metrics['sharpe_ratio']} dd={metrics['max_drawdown_pct']}% trades={metrics['trade_count']}"
        )

        if model_name == "xgb":
            xgb_preds = row["predictions"]
            shared_feat = row["features"]
        elif model_name == "gru":
            gru_preds = row["predictions"]
            if shared_feat is None:
                shared_feat = row["features"]

    if shared_feat is None or xgb_preds is None or gru_preds is None:
        print("ERROR: missing cached predictions for fusion tuning.")
        return 1

    xgb_real_rows: list[object] = []
    xgb_primary, _xgb_real_result = _run_real_lane(
        model_name="xgb",
        feat=shared_feat,
        preds=xgb_preds,
        config=config,
        role="champion",
        comparison_preds={"gru": gru_preds},
        notes="primary single-model lane",
    )
    xgb_real_rows.append(xgb_primary)
    gru_support, _gru_real_result = _run_real_lane(
        model_name="gru",
        feat=shared_feat,
        preds=gru_preds,
        config=config,
        role="comparison",
        comparison_preds={"xgb": xgb_preds},
        notes="support/context only; not primary",
    )
    xgb_real_rows.append(gru_support)
    xgb_real_rows.append(_frozen_comparison_row("lstm", "support/context only; frozen off primary path"))
    _write_xgb_real_leaderboard(xgb_real_rows, REPORTS)

    baseline_fusion = _run_cached_fusion_backtest(
        feat=shared_feat,
        tabular_preds=xgb_preds,
        sequence_preds=gru_preds,
        base_config=config,
        policy=_default_policy(config),
        full_path=True,
    )
    print(
        f"\n[multimodal] fusion baseline: weekly={baseline_fusion['metrics']['weekly_return_pct']}% "
        f"sharpe={baseline_fusion['metrics']['sharpe_ratio']} dd={baseline_fusion['metrics']['max_drawdown_pct']}% "
        f"trades={baseline_fusion['metrics']['trade_count']}"
    )
    rows.append(baseline_fusion)

    policy_grid = _policy_grid(_default_policy(config))
    print(f"\n[fusion-tuning] Running {len(policy_grid)} policy combos on validation slice only ...")
    tuned_rows: list[dict] = []
    for idx, policy in enumerate(policy_grid, start=1):
        tuned = _run_cached_fusion_backtest(
            feat=shared_feat,
            tabular_preds=xgb_preds,
            sequence_preds=gru_preds,
            base_config=config,
            policy=policy,
            full_path=False,
        )
        tuned_rows.append(tuned)
        tm = tuned["metrics"]
        print(
            f"  combo {idx}/{len(policy_grid)}: weekly={tm['weekly_return_pct']}% "
            f"sharpe={tm['sharpe_ratio']} dd={tm['max_drawdown_pct']}% trades={tm['trade_count']}"
        )

    if not tuned_rows:
        print("ERROR: no fusion policy results generated.")
        return 1

    best_validation_policy = _pick_best_model(tuned_rows, metrics_key="metrics")
    _write_fusion_policy_tuning(tuned_rows, REPORTS)

    final_tuned_fusion = _run_cached_fusion_backtest(
        feat=shared_feat,
        tabular_preds=xgb_preds,
        sequence_preds=gru_preds,
        base_config=config,
        policy=best_validation_policy["policy"],
        full_path=True,
    )

    final_rows: list[dict] = []
    for row in rows:
        if row["model"] == "fusion":
            final_rows.append(final_tuned_fusion)
        else:
            final_rows.append(row)

    best = _pick_best_model(final_rows)
    print(f"\n[multimodal] Best model after fusion tuning: {best['model']}")

    _write_leaderboard(final_rows, REPORTS)
    _write_real_benchmark_md(best, final_rows, REPORTS)
    _write_trades_and_equity(best["model"], final_rows, REPORTS)
    _write_next_status(xgb_primary, baseline_fusion, best_validation_policy, final_tuned_fusion, best, REPORTS)

    print(f"\nReports written to {REPORTS}/")
    for fname in [
        "fusion_policy_tuning.csv",
        "fusion_policy_tuning.md",
        "xgb_real_leaderboard.csv",
        "xgb_real_leaderboard.md",
        "multimodal_leaderboard.csv",
        "multimodal_leaderboard.md",
        "multimodal_real_benchmark.md",
        "NEXT_STATUS.md",
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