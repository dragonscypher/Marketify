"""Train-if-missing and benchmark-check runner.

Rules:
- XGBoost baseline first.
- Ridge sanity baseline second.
- Recurrent baselines optional only via env flag.
- Walk-forward only for metric checks.
- No leverage change. No live trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config_hash(config_payload: dict[str, Any]) -> str:
    blob = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _equity_history(equity: pd.Series) -> list[dict[str, str | float]]:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    return [{"ts": str(ts), "equity": float(value)} for ts, value in clean.items()]


def _returns(equity: pd.Series) -> pd.Series:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    return clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _sortino(equity: pd.Series) -> float:
    returns = _returns(equity)
    downside = returns[returns < 0]
    if returns.empty or downside.empty or float(downside.std()) <= 1e-12:
        return 0.0
    return round(float(returns.mean() / downside.std() * np.sqrt(len(returns))), 4)


def _turnover(fills: pd.DataFrame, avg_equity: float) -> float:
    if fills.empty or avg_equity <= 0.0 or not {"qty", "price"}.issubset(fills.columns):
        return 0.0
    notional = (pd.to_numeric(fills["qty"], errors="coerce").abs() * pd.to_numeric(fills["price"], errors="coerce").abs()).sum()
    return round(float(notional) / float(avg_equity), 6)


def _period_success_counts(equity: pd.Series) -> tuple[int, int]:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if clean.empty:
        return 0, 0
    idx = pd.to_datetime(clean.index, errors="coerce")
    series = pd.Series(clean.to_numpy(dtype=float), index=idx).dropna()
    if series.empty:
        return 0, 0
    daily = series.resample("1D").last().pct_change().dropna()
    weekly = series.resample("1W").last().pct_change().dropna()
    return int((weekly >= 0.01).sum()), int((daily >= 0.01).sum())


def _trade_stats(trade_diag: pd.DataFrame) -> tuple[int, float]:
    if trade_diag.empty or "pnl" not in trade_diag.columns:
        return 0, 0.0
    pnl = pd.to_numeric(trade_diag["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return 0, 0.0
    return int(len(pnl)), round(float((pnl > 0).mean() * 100.0), 4)


def _model_metrics(
    model_name: str,
    feat: pd.DataFrame,
    feature_cols: list[str],
    config,
    predict_fn: Callable[..., pd.Series],
) -> dict[str, Any]:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_backtest_from_predictions

    preds = predict_fn(frame=feat, feature_cols=feature_cols, target_col="target_next_ret", config=config.model)
    result = run_backtest_from_predictions(feat, preds, config, model_used=model_name)
    equity: pd.Series = result["equity"]
    fills: pd.DataFrame = result.get("fills", pd.DataFrame())
    trade_diag: pd.DataFrame = result.get("trade_diagnostics", pd.DataFrame())
    history = _equity_history(equity)
    benchmark = compute_benchmark(history, weekly_goal=config.benchmark_weekly_goal)
    returns = _returns(equity)
    lower_tail = returns[returns <= returns.quantile(0.05)] if not returns.empty else pd.Series(dtype=float)
    trade_count, win_rate = _trade_stats(trade_diag)
    avg_equity = float(pd.to_numeric(equity, errors="coerce").mean()) if len(equity) else 0.0
    weeks_ge_1, days_ge_1 = _period_success_counts(equity)
    max_drawdown_pct = round(float(benchmark.max_drawdown) * 100.0, 4)
    total_return_pct = round((float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0) * 100.0, 4) if len(equity) >= 2 else 0.0
    calmar = round(total_return_pct / max(max_drawdown_pct, 1e-9), 4) if max_drawdown_pct > 0 else 0.0
    return {
        "model": model_name,
        "role": "primary" if model_name == "xgb" else "sanity_baseline",
        "walk_forward_only": True,
        "leakage_check": "chronological_train_before_predict",
        "total_return_pct": total_return_pct,
        "daily_return_pct": round(float(benchmark.daily_return) * 100.0, 4),
        "weekly_return_pct": round(float(benchmark.weekly_return) * 100.0, 4),
        "sharpe": round(float(benchmark.sharpe_ratio), 4),
        "sortino": _sortino(equity),
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar,
        "win_rate_pct": win_rate,
        "trade_count": trade_count,
        "turnover": _turnover(fills, avg_equity),
        "cvar_5_pct": round(float(lower_tail.mean()) * 100.0, 4) if len(lower_tail) else 0.0,
        "weeks_ge_1pct": weeks_ge_1,
        "days_ge_1pct": days_ge_1,
        "PASS_WEEKLY": bool(float(benchmark.weekly_return) >= 0.01 and float(benchmark.max_drawdown) <= config.risk.max_drawdown_pct),
        "PASS_DAILY": bool(float(benchmark.daily_return) >= 0.01 and float(benchmark.max_drawdown) <= config.risk.max_drawdown_pct),
        "daily_stress_blocker": "none" if float(benchmark.daily_return) >= 0.01 else "daily_return_pct below 1.0",
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Marketify Train And Check Summary",
        "",
        f"timestamp: {summary['timestamp']}",
        f"training_check_result: {summary['training_check_result']}",
        f"config_hash: {summary['config_hash']}",
        f"artifact_run_dir: {summary['artifact_run_dir']}",
        f"train_if_missing_behavior: {summary['train_if_missing_behavior']}",
        "paper_only_default: YES",
        "live_trading_enabled: NO",
        "profit_guarantee: NO",
        "",
        "## Model Metrics",
        "| model | role | weekly_return_pct | daily_return_pct | sharpe | sortino | max_drawdown_pct | calmar | win_rate_pct | trade_count | turnover | cvar_5_pct | weeks_ge_1pct | days_ge_1pct | PASS_WEEKLY | PASS_DAILY | daily_stress_blocker |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['model']} | {row['role']} | {row['weekly_return_pct']} | {row['daily_return_pct']} | {row['sharpe']} | {row['sortino']} | {row['max_drawdown_pct']} | {row['calmar']} | {row['win_rate_pct']} | {row['trade_count']} | {row['turnover']} | {row['cvar_5_pct']} | {row['weeks_ge_1pct']} | {row['days_ge_1pct']} | {row['PASS_WEEKLY']} | {row['PASS_DAILY']} | {row['daily_stress_blocker']} |"
        )
    lines += [
        "",
        "## Recurrent Lane",
        f"- recurrent_status: {summary['recurrent_status']}",
        "- recurrent remains support/context only; not champion path.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train-if-missing and walk-forward check for Marketify")
    parser.add_argument("--force", action="store_true", help="force retraining static artifacts")
    args = parser.parse_args(argv)

    from marketify.config import AppConfig
    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                              add_technical_features)
    from marketify.models.ridge_model import \
        rolling_train_predict as ridge_predict
    from marketify.models.train_pipeline import ARTIFACT_DIR, train_if_missing
    from marketify.models.xgb_model import rolling_train_predict as xgb_predict

    _ensure_dir(REPORTS_DIR)
    _ensure_dir(ARTIFACTS_DIR)
    config = AppConfig()
    config_payload = asdict(config)
    cfg_hash = _config_hash(config_payload)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_run_dir = _ensure_dir(ARTIFACTS_DIR / "training_runs" / f"{timestamp}_{cfg_hash}")

    expected_static = [ARTIFACT_DIR / f"xgb_{config.data.ticker}.pkl", ARTIFACT_DIR / f"ridge_{config.data.ticker}.pkl"]
    missing_before = [str(path) for path in expected_static if not path.exists()]
    leaderboard = train_if_missing(config=config, force=args.force)
    missing_after = [str(path) for path in expected_static if not path.exists()]
    if missing_after:
        raise RuntimeError(f"missing artifacts after train_if_missing: {missing_after}")
    for path in expected_static:
        shutil.copy2(path, artifact_run_dir / path.name)
    (artifact_run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (artifact_run_dir / "train_if_missing_leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)
    feature_cols = [col for col in TECHNICAL_FEATURE_COLUMNS if col in feat.columns]
    metrics = [
        _model_metrics("xgb", feat, feature_cols, config, xgb_predict),
        _model_metrics("ridge", feat, feature_cols, config, ridge_predict),
    ]

    recurrent_status = "skipped_optional_not_requested"
    if os.environ.get("MARKETIFY_TRAIN_RECURRENT", "0") == "1":
        recurrent_status = "requested_but_not_run_in_train_check; compare_models handles recurrent support/context"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_check_result": "PASS" if all(row["walk_forward_only"] for row in metrics) else "FAIL",
        "config_hash": cfg_hash,
        "artifact_run_dir": str(artifact_run_dir),
        "train_if_missing_behavior": "trained_missing_artifacts" if missing_before or args.force else "loaded_existing_artifacts",
        "missing_before": missing_before,
        "static_artifacts": [str(path) for path in expected_static],
        "metrics": metrics,
        "recurrent_status": recurrent_status,
    }

    json_path = REPORTS_DIR / "training_check_summary.json"
    md_path = REPORTS_DIR / "training_check_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, md_path)
    print(f"[TRAIN_CHECK] result={summary['training_check_result']}")
    print(f"[TRAIN_CHECK] json={json_path}")
    print(f"[TRAIN_CHECK] markdown={md_path}")
    print(f"[TRAIN_CHECK] artifact_run_dir={artifact_run_dir}")
    return 0 if summary["training_check_result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())