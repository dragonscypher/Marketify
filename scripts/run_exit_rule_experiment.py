"""Exit-rule improvement experiment (validation split only).

Implements:
- ATR/volatility-scaled stop and take-profit exits.
- Fixed and volatility-adjusted time-stop variants.
- Optional trailing stop activated after profit threshold.
- Horizon-alignment check (model horizon vs time_stop_bars).
- Prediction caching so rolling model predictions are computed once per split.
- Quick mode, resumable checkpoints, and incremental report writes.

Outputs:
    reports/exit_rule_experiment.csv
    reports/exit_rule_experiment.md
    reports/exit_rule_diagnostics.csv
    reports/exit_rule_diagnostics.md
    reports/exit_rule_predictions_cache.parquet
    reports/exit_rule_experiment_checkpoint.json
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"

# Validation-only experiment split: train 70%, validation 15%, final test 15% untouched.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MODEL_HORIZON_BARS = 1  # target_next_ret predicts one-step ahead return

MAX_DRAWDOWN_LIMIT = 0.15
MIN_TRADES = 10
MAX_POSITION_FRACTION_LIMIT = 0.10
MAX_TURNOVER_PER_BAR = 0.50

EXIT_GRID = {
    "k_stop": [1.2, 1.5, 1.8],
    "k_take": [1.8, 2.2, 2.8],
    "time_stop_bars": [24, 48, 72],
    "time_stop_mode": ["fixed", "volatility_adjusted"],
    "trailing_enabled": [False, True],
    "trailing_activate_profit_pct": [0.004],
    "trailing_distance_pct": [0.003],
    "max_position_fraction": [0.10],
}

PARAM_KEY_FIELDS = (
    "k_stop",
    "k_take",
    "time_stop_bars",
    "time_stop_mode",
    "trailing_enabled",
    "trailing_activate_profit_pct",
    "trailing_distance_pct",
    "max_position_fraction",
)

BASELINE_WEEKLY_RETURN_PCT = 0.5851

SIGNAL_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "predicted_return",
    "realized_return_after_signal",
    "signal_hit",
    "risk_approved",
    "volatility",
    "volatility_regime",
)

TRADE_DIAGNOSTIC_COLUMNS = (
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
    "expected_return",
    "cvar_95",
    "stop_loss",
    "take_profit",
    "effective_time_stop_bars",
    "exit_reason",
    "volatility",
    "volatility_regime",
)

DIAGNOSTIC_COLUMNS = (
    "section",
    "group",
    "metric",
    "value",
    "count",
    "notes",
)


@dataclass(frozen=True)
class ValidationSplitMeta:
    train_end: int
    val_end: int
    test_start: int
    validation_start_ts: str
    validation_end_ts: str
    test_start_ts: str
    test_split_touched: bool


@dataclass
class ExitExperimentResult:
    k_stop: float
    k_take: float
    time_stop_bars: int
    time_stop_mode: str
    trailing_enabled: bool
    trailing_activate_profit_pct: float
    trailing_distance_pct: float
    max_position_fraction: float
    weekly_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    exit_reason_counts: str
    horizon_mismatch: bool
    test_split_touched: bool
    rejected: bool
    reject_reason: str


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    csv_path: Path
    md_path: Path
    diagnostics_csv_path: Path
    diagnostics_md_path: Path
    predictions_cache_path: Path
    checkpoint_path: Path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exit-rule validation experiment.")
    parser.add_argument("--quick", action="store_true", help="Run small starter slice (max 8 configs).")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit selected config count.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint in output dir.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force regeneration of cached predictions.")
    parser.add_argument("--output-dir", default="reports", help="Directory for CSV/Markdown/checkpoint artifacts.")
    args = parser.parse_args(argv)
    if args.max_configs is not None and args.max_configs < 0:
        parser.error("--max-configs must be >= 0")
    return args


def _resolve_output_dir(raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir


def _build_experiment_paths(output_dir: Path) -> ExperimentPaths:
    return ExperimentPaths(
        output_dir=output_dir,
        csv_path=output_dir / "exit_rule_experiment.csv",
        md_path=output_dir / "exit_rule_experiment.md",
        diagnostics_csv_path=output_dir / "exit_rule_diagnostics.csv",
        diagnostics_md_path=output_dir / "exit_rule_diagnostics.md",
        predictions_cache_path=output_dir / "exit_rule_predictions_cache.parquet",
        checkpoint_path=output_dir / "exit_rule_experiment_checkpoint.json",
    )


def _config_key(params: dict[str, object]) -> str:
    return json.dumps({field: params[field] for field in PARAM_KEY_FIELDS}, sort_keys=True)


def _result_config_key(result: ExitExperimentResult | dict[str, object]) -> str:
    if isinstance(result, ExitExperimentResult):
        payload = {field: getattr(result, field) for field in PARAM_KEY_FIELDS}
    else:
        payload = {field: result[field] for field in PARAM_KEY_FIELDS}
    return json.dumps(payload, sort_keys=True)


def _empty_signal_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_DIAGNOSTIC_COLUMNS)


def _empty_trade_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_DIAGNOSTIC_COLUMNS)


def _empty_diagnostics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)


def _select_best_result(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None

    ranked = df.copy()
    if "rejected" not in ranked.columns:
        ranked["rejected"] = False
    return ranked.sort_values(["rejected", "weekly_return_pct", "sharpe"], ascending=[True, False, False]).iloc[0]


def _best_result_to_params(best_result: pd.Series | dict[str, object]) -> dict[str, object]:
    payload = best_result.to_dict() if isinstance(best_result, pd.Series) else best_result
    return {field: payload[field] for field in PARAM_KEY_FIELDS}


def _beats_baseline(best_result: pd.Series | dict[str, object] | None) -> bool:
    if best_result is None:
        return False
    payload = best_result.to_dict() if isinstance(best_result, pd.Series) else best_result
    return (not bool(payload.get("rejected", False))) and float(payload.get("weekly_return_pct", float("-inf"))) >= BASELINE_WEEKLY_RETURN_PCT


def _select_experiment_params(quick: bool = False, max_configs: int | None = None) -> list[dict[str, object]]:
    keys = list(EXIT_GRID.keys())
    params = [dict(zip(keys, combo)) for combo in itertools.product(*[EXIT_GRID[k] for k in keys])]
    if quick:
        params = params[:8]
    if max_configs is not None:
        params = params[:max_configs]
    return params


def _load_checkpoint(checkpoint_path: Path) -> dict[str, object] | None:
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[EXIT-EXPERIMENT] Warning: ignoring unreadable checkpoint {checkpoint_path}")
        return None


def _restore_results_from_checkpoint(
    checkpoint: dict[str, object] | None,
    selected_keys: set[str],
) -> tuple[list[ExitExperimentResult], set[str]]:
    if checkpoint is None:
        return [], set()

    restored_results: list[ExitExperimentResult] = []
    restored_keys: set[str] = set()
    for item in checkpoint.get("results", []):
        if not isinstance(item, dict):
            continue
        try:
            key = _result_config_key(item)
            result = ExitExperimentResult(**item)
        except (KeyError, TypeError):
            continue
        if key in selected_keys and key not in restored_keys:
            restored_results.append(result)
            restored_keys.add(key)

    completed_keys = {
        key for key in checkpoint.get("completed_config_keys", []) if isinstance(key, str) and key in selected_keys
    }
    completed_keys.update(restored_keys)
    return restored_results, completed_keys


def _write_diagnostics_reports(
    paths: ExperimentPaths,
    diagnostics_df: pd.DataFrame | None,
    best_result: pd.Series | None,
    run_status: str,
) -> None:
    df = _empty_diagnostics_df() if diagnostics_df is None else diagnostics_df.copy()
    df = df.reindex(columns=DIAGNOSTIC_COLUMNS)
    df.to_csv(paths.diagnostics_csv_path, index=False)

    md = build_exit_rule_diagnostics_markdown(df, best_result, run_status)
    paths.diagnostics_md_path.write_text(md, encoding="utf-8")


def _write_checkpoint_and_reports(
    paths: ExperimentPaths,
    results: list[ExitExperimentResult],
    split_meta: ValidationSplitMeta,
    run_status: str,
    total_configs: int,
    selected_keys: list[str],
    completed_keys: set[str],
) -> None:
    df = pd.DataFrame([asdict(result) for result in results])
    df.to_csv(paths.csv_path, index=False)

    md = build_exit_rule_experiment_markdown(
        df,
        split_meta,
        run_status=run_status,
        total_configs=total_configs,
        completed_configs=len(completed_keys),
    )
    paths.md_path.write_text(md, encoding="utf-8")

    _write_diagnostics_reports(paths, _empty_diagnostics_df(), _select_best_result(df), run_status)

    checkpoint = {
        "completed_config_keys": sorted(completed_keys),
        "completed_configs": len(completed_keys),
        "results": [asdict(result) for result in results],
        "selected_config_keys": selected_keys,
        "status": run_status,
        "total_configs": total_configs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")


def _scalar(row, col: str) -> float:
    value = row[col]
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _ensure_optional_dependencies_for_exit_experiment() -> None:
    """Fail fast with clear instruction when optional TA dependency is missing."""
    try:
        importlib.import_module("ta")
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "ta":
            print(
                "Missing dependency: ta. Run python -m pip install -r requirements-colab.txt",
                file=sys.stderr,
            )
        raise


def split_for_validation_experiment(
    frame: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.Index, ValidationSplitMeta]:
    """Return train+validation frame and validation index only. Test tail excluded."""
    n = len(frame)
    if n < 400:
        raise ValueError(f"Insufficient rows for split-aware experiment: {n}")

    train_end = max(1, int(n * train_fraction))
    val_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))
    val_end = min(val_end, n - 1)

    train_val = frame.iloc[:val_end].copy()
    validation_index = train_val.index[train_end:]
    if len(validation_index) == 0:
        raise ValueError("Validation split is empty; cannot run experiment.")

    test_index = frame.index[val_end:]
    touched = bool(train_val.index.intersection(test_index).size > 0)

    meta = ValidationSplitMeta(
        train_end=train_end,
        val_end=val_end,
        test_start=val_end,
        validation_start_ts=str(validation_index[0]),
        validation_end_ts=str(validation_index[-1]),
        test_start_ts=str(frame.index[val_end]),
        test_split_touched=touched,
    )
    return train_val, validation_index, meta


def _compute_rolling_predictions(train_val: pd.DataFrame) -> pd.Series:
    from marketify.config import AppConfig
    from marketify.features.technical import TECHNICAL_FEATURE_COLUMNS
    from marketify.models.xgb_model import rolling_train_predict

    config = AppConfig()
    if len(train_val) <= config.model.train_window + 10:
        config.model.train_window = max(200, min(config.model.train_window, len(train_val) - 20))
        config.model.retrain_every = max(50, min(config.model.retrain_every, max(60, len(train_val) // 8)))

    raw_preds = rolling_train_predict(
        frame=train_val,
        feature_cols=TECHNICAL_FEATURE_COLUMNS,
        target_col="target_next_ret",
        config=config.model,
    )
    if isinstance(raw_preds, pd.Series):
        preds = raw_preds.rename("prediction")
    else:
        preds = pd.Series(raw_preds, index=train_val.index[-len(raw_preds):], name="prediction")
    return pd.to_numeric(preds, errors="coerce").reindex(train_val.index).rename("prediction")


def _write_predictions_cache(cache_path: Path, predictions: pd.Series) -> None:
    try:
        predictions.rename("prediction").to_frame().to_parquet(cache_path)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise RuntimeError("Prediction cache requires parquet support. Install pyarrow or fastparquet.") from exc


def _read_predictions_cache(cache_path: Path) -> pd.Series | None:
    if not cache_path.exists():
        return None
    try:
        cache_df = pd.read_parquet(cache_path)
    except Exception as exc:
        print(f"[EXIT-EXPERIMENT] Warning: ignoring unreadable prediction cache {cache_path}: {exc}")
        return None

    if "prediction" in cache_df.columns:
        preds = cache_df["prediction"]
    elif cache_df.shape[1] == 1:
        preds = cache_df.iloc[:, 0].rename("prediction")
    else:
        print(f"[EXIT-EXPERIMENT] Warning: prediction cache missing 'prediction' column: {cache_path}")
        return None
    return pd.to_numeric(preds, errors="coerce").rename("prediction")


def _load_or_build_prediction_cache(
    train_val: pd.DataFrame,
    paths: ExperimentPaths,
    rebuild_cache: bool = False,
) -> pd.Series:
    if not rebuild_cache:
        cached = _read_predictions_cache(paths.predictions_cache_path)
        if cached is not None and cached.index.equals(train_val.index):
            print(f"[EXIT-EXPERIMENT] Using cached predictions {paths.predictions_cache_path}")
            return cached.reindex(train_val.index)
        if cached is not None:
            print("[EXIT-EXPERIMENT] Prediction cache index mismatch. Rebuilding cache.")
    else:
        print(f"[EXIT-EXPERIMENT] Rebuilding prediction cache {paths.predictions_cache_path}")

    predictions = _compute_rolling_predictions(train_val)
    _write_predictions_cache(paths.predictions_cache_path, predictions)
    print(f"[EXIT-EXPERIMENT] Saved prediction cache {paths.predictions_cache_path}")
    return predictions


def _compute_realized_signal_return(train_val: pd.DataFrame, ts: pd.Timestamp) -> float:
    row = train_val.loc[ts]
    if "target_next_ret" in train_val.columns:
        target_next_ret = _scalar(row, "target_next_ret")
        if np.isfinite(target_next_ret):
            return float(target_next_ret)

    ts_loc = int(train_val.index.get_indexer([ts])[0])
    if ts_loc < 0 or ts_loc + MODEL_HORIZON_BARS >= len(train_val):
        return float("nan")

    current_price = _scalar(row, "Close")
    next_price = _scalar(train_val.iloc[ts_loc + MODEL_HORIZON_BARS], "Close")
    if current_price <= 0:
        return float("nan")
    return float(next_price / current_price - 1.0)


def _volatility_regime_bounds(train_val: pd.DataFrame, validation_index: pd.Index) -> tuple[float, float]:
    validation_vol = train_val.loc[validation_index, "vol_20"].replace([np.inf, -np.inf], np.nan).dropna()
    if validation_vol.empty:
        return 0.0, 0.0
    low = float(validation_vol.quantile(0.33))
    high = float(validation_vol.quantile(0.67))
    return low, max(low, high)


def _label_volatility_regime(volatility: float, low_cutoff: float, high_cutoff: float) -> str:
    if not np.isfinite(volatility):
        return "unknown"
    if volatility <= low_cutoff:
        return "low"
    if volatility <= high_cutoff:
        return "medium"
    return "high"


def _compute_trade_metrics(diag: pd.DataFrame) -> dict[str, float | int | str]:
    trade_count = int(len(diag)) if not diag.empty else 0
    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "exit_reason_counts": json.dumps({}, sort_keys=True),
        }

    wins = diag[diag["pnl"] > 0]["pnl"]
    losses = diag[diag["pnl"] < 0]["pnl"]
    win_sum = float(wins.sum()) if not wins.empty else 0.0
    loss_sum_abs = abs(float(losses.sum())) if not losses.empty else 0.0

    if loss_sum_abs > 0:
        profit_factor = win_sum / loss_sum_abs
    else:
        profit_factor = float("inf") if win_sum > 0 else 0.0

    win_rate = float((wins.shape[0] / trade_count) * 100.0)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    expectancy = float(diag["pnl"].mean())
    exit_counts = diag["exit_reason"].value_counts(dropna=False).to_dict() if "exit_reason" in diag.columns else {}

    return {
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate, 4),
        "profit_factor": float(profit_factor),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "expectancy": round(expectancy, 6),
        "exit_reason_counts": json.dumps({str(k): int(v) for k, v in exit_counts.items()}, sort_keys=True),
    }


def _run_single_config(
    train_val: pd.DataFrame,
    validation_index: pd.Index,
    split_meta: ValidationSplitMeta,
    params: dict,
    preds: pd.Series,
) -> tuple[ExitExperimentResult, pd.DataFrame, pd.DataFrame]:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.exit_rules import (ExitRuleConfig,
                                               build_exit_levels,
                                               evaluate_dynamic_exit)
    from marketify.broker.paper import PaperBroker
    from marketify.config import AppConfig
    from marketify.features.sentiment import FinBERTSentiment
    from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                           compute_cvar_95)
    from marketify.risk.risk_engine import RiskEngine

    reject_reasons: list[str] = []

    max_position_fraction = float(params["max_position_fraction"])
    if max_position_fraction > MAX_POSITION_FRACTION_LIMIT:
        reject_reasons.append(
            f"hidden leverage: max_position_fraction {max_position_fraction:.4f} > {MAX_POSITION_FRACTION_LIMIT:.4f}"
        )

    config = AppConfig()
    config.broker.max_position_fraction = max_position_fraction
    config.broker.time_stop_bars = int(params["time_stop_bars"])
    db_key = (
        params["k_stop"],
        params["k_take"],
        params["time_stop_bars"],
        params["time_stop_mode"],
        params["trailing_enabled"],
        params["trailing_activate_profit_pct"],
        params["trailing_distance_pct"],
        params["max_position_fraction"],
    )
    config.broker.db_path = f"data/exit_exp_{hash(db_key) & 0xFFFFFFFF}.db"

    broker = PaperBroker(config.broker)
    risk_engine = RiskEngine(config.risk, config.broker)
    sentiment = FinBERTSentiment(enabled=False)
    open_state: dict[str, dict] = {}

    exit_cfg = ExitRuleConfig(
        k_stop=float(params["k_stop"]),
        k_take=float(params["k_take"]),
        time_stop_bars=int(params["time_stop_bars"]),
        time_stop_mode=str(params["time_stop_mode"]),
        trailing_enabled=bool(params["trailing_enabled"]),
        trailing_activate_profit_pct=float(params["trailing_activate_profit_pct"]),
        trailing_distance_pct=float(params["trailing_distance_pct"]),
    )

    vol_ref = float(train_val["vol_20"].replace([np.inf, -np.inf], np.nan).dropna().median())
    if not np.isfinite(vol_ref) or vol_ref <= 0:
        vol_ref = 0.01
    vol_low_cutoff, vol_high_cutoff = _volatility_regime_bounds(train_val, validation_index)

    equity_points: list[tuple[pd.Timestamp, float]] = []
    signal_diagnostics: list[dict[str, object]] = []
    trade_diagnostics: list[dict] = []

    ticker = config.data.ticker

    for ts in validation_index:
        row = train_val.loc[ts]
        price = _scalar(row, "Close")
        row_open = _scalar(row, "Open")
        row_high = _scalar(row, "High")
        row_low = _scalar(row, "Low")
        vol_20 = _scalar(row, "vol_20")
        atr_14 = _scalar(row, "atr_14")
        volatility_regime = _label_volatility_regime(vol_20, vol_low_cutoff, vol_high_cutoff)

        broker.update_market_price(ticker, price)

        if ticker in open_state:
            state = open_state[ticker]
            bars_held = int(state.get("bars_held", 0)) + 1
            state["bars_held"] = bars_held

            exit_price, exit_reason, state = evaluate_dynamic_exit(
                side=state["side"],
                state=state,
                row_open=row_open,
                row_high=row_high,
                row_low=row_low,
                bars_held=bars_held,
                volatility=vol_20,
                volatility_ref=vol_ref,
                cfg=exit_cfg,
            )
            open_state[ticker] = state

            if exit_price is not None:
                close_side = "sell" if state["side"] == "buy" else "buy"
                broker.submit_order(
                    {
                        "symbol": ticker,
                        "side": close_side,
                        "qty": state["qty"],
                        "price": float(exit_price),
                        "reason": f"exit_rule_experiment:{exit_reason}",
                        "risk_warning": "Paper backtest exit",
                    }
                )

                entry_p = float(state["entry_price"])
                if state["side"] == "buy":
                    pnl = (float(exit_price) - entry_p) * float(state["qty"])
                    ret_pct = (float(exit_price) / entry_p - 1.0) * 100.0 if entry_p > 0 else 0.0
                else:
                    pnl = (entry_p - float(exit_price)) * float(state["qty"])
                    ret_pct = (1.0 - float(exit_price) / entry_p) * 100.0 if entry_p > 0 else 0.0

                trade_diagnostics.append(
                    {
                        "timestamp": str(ts),
                        "symbol": ticker,
                        "side": state["side"],
                        "qty": int(state["qty"]),
                        "entry_price": entry_p,
                        "exit_price": float(exit_price),
                        "pnl": round(float(pnl), 6),
                        "return_pct": round(float(ret_pct), 6),
                        "hold_bars": int(bars_held),
                        "signal_confidence": float(state.get("confidence", 0.0)),
                        "expected_return": float(state.get("expected_return", 0.0)),
                        "cvar_95": float(state.get("cvar_95", 0.0)),
                        "stop_loss": float(state["stop_loss"]),
                        "take_profit": float(state["take_profit"]),
                        "effective_time_stop_bars": int(state.get("effective_time_stop_bars", exit_cfg.time_stop_bars)),
                        "exit_reason": str(exit_reason),
                        "volatility": float(state.get("volatility", np.nan)),
                        "volatility_regime": str(state.get("volatility_regime", "unknown")),
                    }
                )
                open_state.pop(ticker, None)

        expected = float(preds.loc[ts]) if ts in preds.index and np.isfinite(float(preds.loc[ts])) else np.nan
        if np.isfinite(expected) and ticker not in open_state:
            ts_loc = train_val.index.get_loc(ts)
            ret_slice = train_val["ret_1"].iloc[max(0, ts_loc - 250): ts_loc + 1]
            cvar_95 = compute_cvar_95(ret_slice)
            sentiment_score = sentiment.analyze(ticker).score
            info = TradeIdeaInput(
                symbol=ticker,
                last_price=price,
                prediction=expected,
                sentiment_score=sentiment_score,
                volatility=vol_20,
                news_risk=0.0,
            )
            account = broker.get_account()
            idea = build_trade_idea(info, account["equity"], config.broker, config.risk, cvar_95)
            if idea is not None:
                decision = risk_engine.evaluate(
                    trade=idea,
                    account=account,
                    volatility=info.volatility,
                    news_risk=info.news_risk,
                    mark_price=price,
                )
                realized_after_signal = _compute_realized_signal_return(train_val, ts)
                signal_hit = float("nan")
                if np.isfinite(realized_after_signal):
                    signal_hit = bool(realized_after_signal > 0.0) if idea.side == "buy" else bool(realized_after_signal < 0.0)
                signal_diagnostics.append(
                    {
                        "timestamp": str(ts),
                        "symbol": ticker,
                        "side": idea.side,
                        "predicted_return": float(expected),
                        "realized_return_after_signal": float(realized_after_signal),
                        "signal_hit": signal_hit,
                        "risk_approved": bool(decision.approved),
                        "volatility": float(vol_20),
                        "volatility_regime": volatility_regime,
                    }
                )
                if decision.approved:
                    broker.submit_order({**idea.to_dict(), "price": price})
                    dyn_stop, dyn_take = build_exit_levels(idea.side, price, atr_14, exit_cfg)
                    open_state[ticker] = {
                        "side": idea.side,
                        "qty": int(idea.qty),
                        "entry_price": float(price),
                        "bars_held": 0,
                        "stop_loss": float(dyn_stop),
                        "take_profit": float(dyn_take),
                        "confidence": float(idea.confidence),
                        "expected_return": float(idea.expected_return),
                        "cvar_95": float(idea.cvar_95),
                        "trailing_active": False,
                        "peak_price": float(price),
                        "trough_price": float(price),
                        "volatility": float(vol_20),
                        "volatility_regime": volatility_regime,
                    }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], name="equity")
    history = [{"ts": str(ts), "equity": float(eq)} for ts, eq in zip(equity.index, equity.values)]
    bm = compute_benchmark(history, weekly_goal=0.01)

    signal_df = pd.DataFrame(signal_diagnostics, columns=SIGNAL_DIAGNOSTIC_COLUMNS)
    diag_df = pd.DataFrame(trade_diagnostics, columns=TRADE_DIAGNOSTIC_COLUMNS)
    trade_metrics = _compute_trade_metrics(diag_df)

    turnover = (trade_metrics["trade_count"] / max(len(equity), 1)) if len(equity) > 0 else 0.0
    if trade_metrics["trade_count"] < MIN_TRADES:
        reject_reasons.append(f"too few trades: {trade_metrics['trade_count']} < {MIN_TRADES}")
    if bm.max_drawdown > MAX_DRAWDOWN_LIMIT:
        reject_reasons.append(f"max drawdown unsafe: {bm.max_drawdown:.4f} > {MAX_DRAWDOWN_LIMIT:.4f}")
    if turnover > MAX_TURNOVER_PER_BAR:
        reject_reasons.append(f"unrealistic turnover: {turnover:.4f} > {MAX_TURNOVER_PER_BAR:.4f}")

    horizon_mismatch = MODEL_HORIZON_BARS != int(params["time_stop_bars"])
    rejected = len(reject_reasons) > 0

    try:
        Path(config.broker.db_path).unlink(missing_ok=True)
    except Exception:
        pass

    result = ExitExperimentResult(
        k_stop=float(params["k_stop"]),
        k_take=float(params["k_take"]),
        time_stop_bars=int(params["time_stop_bars"]),
        time_stop_mode=str(params["time_stop_mode"]),
        trailing_enabled=bool(params["trailing_enabled"]),
        trailing_activate_profit_pct=float(params["trailing_activate_profit_pct"]),
        trailing_distance_pct=float(params["trailing_distance_pct"]),
        max_position_fraction=max_position_fraction,
        weekly_return_pct=round(float(bm.weekly_return) * 100.0, 4),
        sharpe=round(float(bm.sharpe_ratio), 4),
        max_drawdown_pct=round(float(bm.max_drawdown) * 100.0, 4),
        trade_count=int(trade_metrics["trade_count"]),
        win_rate_pct=float(trade_metrics["win_rate_pct"]),
        profit_factor=float(trade_metrics["profit_factor"]),
        avg_win=float(trade_metrics["avg_win"]),
        avg_loss=float(trade_metrics["avg_loss"]),
        expectancy=float(trade_metrics["expectancy"]),
        exit_reason_counts=str(trade_metrics["exit_reason_counts"]),
        horizon_mismatch=bool(horizon_mismatch),
        test_split_touched=bool(split_meta.test_split_touched),
        rejected=bool(rejected),
        reject_reason="; ".join(reject_reasons),
    )
    return result, signal_df, diag_df


def _build_diagnostics_summary(
    best_result: ExitExperimentResult,
    signal_df: pd.DataFrame,
    trade_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _add_row(section: str, group: str, metric: str, value: object, count: int = 0, notes: str = "") -> None:
        rows.append(
            {
                "section": section,
                "group": group,
                "metric": metric,
                "value": value,
                "count": int(count),
                "notes": notes,
            }
        )

    signal_count = int(len(signal_df))
    hit_rate = float(signal_df["signal_hit"].dropna().mean() * 100.0) if signal_df["signal_hit"].dropna().size > 0 else float("nan")
    avg_predicted = float(signal_df["predicted_return"].mean()) if signal_count > 0 else float("nan")
    avg_realized = float(signal_df["realized_return_after_signal"].mean()) if signal_count > 0 else float("nan")

    _add_row("best_config", "overall", "weekly_return_pct", round(float(best_result.weekly_return_pct), 4), 1)
    _add_row("best_config", "overall", "baseline_weekly_return_pct", BASELINE_WEEKLY_RETURN_PCT, 1)
    _add_row("best_config", "overall", "beats_baseline", int(_beats_baseline(asdict(best_result))), 1, "1=yes, 0=no")

    _add_row("signal_summary", "overall", "signal_count", signal_count, signal_count)
    _add_row("signal_summary", "overall", "signal_hit_rate_pct", round(hit_rate, 4) if np.isfinite(hit_rate) else float("nan"), signal_count)
    _add_row("signal_summary", "overall", "avg_predicted_return", round(avg_predicted, 6) if np.isfinite(avg_predicted) else float("nan"), signal_count)
    _add_row(
        "signal_summary",
        "overall",
        "avg_realized_return_after_signal",
        round(avg_realized, 6) if np.isfinite(avg_realized) else float("nan"),
        signal_count,
    )

    for section_name, group_col in (("exit_reason", "exit_reason"), ("volatility_regime", "volatility_regime"), ("side_performance", "side")):
        if trade_df.empty or group_col not in trade_df.columns:
            continue
        for group_name, group in trade_df.groupby(group_col, dropna=False):
            trade_count = int(len(group))
            expectancy = float(group["pnl"].mean()) if trade_count > 0 else float("nan")
            avg_return_pct = float(group["return_pct"].mean()) if trade_count > 0 else float("nan")
            win_rate_pct = float((group["pnl"] > 0).mean() * 100.0) if trade_count > 0 else float("nan")
            group_label = str(group_name)
            _add_row(section_name, group_label, "trade_expectancy", round(expectancy, 6), trade_count)
            _add_row(section_name, group_label, "avg_return_pct", round(avg_return_pct, 6), trade_count)
            _add_row(section_name, group_label, "win_rate_pct", round(win_rate_pct, 4), trade_count)

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def _format_metric(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer, bool)):
        return str(int(value))
    return str(value)


def _diagnostic_metric(diagnostics_df: pd.DataFrame, section: str, group: str, metric: str) -> object | None:
    subset = diagnostics_df.loc[
        (diagnostics_df["section"] == section)
        & (diagnostics_df["group"] == group)
        & (diagnostics_df["metric"] == metric),
        "value",
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _append_group_diagnostics_table(lines: list[str], diagnostics_df: pd.DataFrame, section: str, title: str, group_label: str) -> None:
    section_df = diagnostics_df.loc[diagnostics_df["section"] == section].copy()
    lines.append(f"## {title}")
    if section_df.empty:
        lines.extend(["No data.", ""])
        return

    counts = section_df.groupby("group")["count"].max()
    pivot = section_df.pivot(index="group", columns="metric", values="value")
    lines.append(f"| {group_label} | Trades | Expectancy | Avg Return % | Win Rate % |")
    lines.append("|---|---:|---:|---:|---:|")
    for group_name, row in pivot.sort_index().iterrows():
        lines.append(
            f"| {group_name} | {int(counts.get(group_name, 0))} | { _format_metric(row.get('trade_expectancy')) } "
            f"| { _format_metric(row.get('avg_return_pct')) } | { _format_metric(row.get('win_rate_pct'), digits=4) } |"
        )
    lines.append("")


def build_exit_rule_diagnostics_markdown(
    diagnostics_df: pd.DataFrame,
    best_result: pd.Series | None,
    run_status: str = "COMPLETE",
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    beats_baseline = _beats_baseline(best_result)

    lines = [
        "# Exit Rule Diagnostics",
        f"**Generated:** {generated}",
        f"**Run Status:** {run_status}",
        "",
    ]

    if best_result is not None:
        lines.extend(
            [
                "## Best Config",
                f"- status: {'ACCEPTED' if not bool(best_result.get('rejected', False)) else 'REJECTED'}",
                f"- k_stop: {best_result['k_stop']}",
                f"- k_take: {best_result['k_take']}",
                f"- time_stop_mode: {best_result['time_stop_mode']}",
                f"- time_stop_bars: {int(best_result['time_stop_bars'])}",
                f"- trailing_enabled: {bool(best_result['trailing_enabled'])}",
                f"- weekly_return_pct: {best_result['weekly_return_pct']}",
                f"- sharpe: {best_result['sharpe']}",
                f"- baseline_weekly_return_pct: {BASELINE_WEEKLY_RETURN_PCT}",
                f"- baseline comparison: {'PASS' if beats_baseline else 'FAIL'}",
                "",
            ]
        )
    else:
        lines.extend(["## Best Config", "No completed config yet.", ""])

    if diagnostics_df.empty:
        lines.extend(
            [
                "## Summary",
                "Detailed diagnostics pending final best-config replay.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Signal Summary",
            f"- signal_count: {_format_metric(_diagnostic_metric(diagnostics_df, 'signal_summary', 'overall', 'signal_count'), digits=0)}",
            f"- signal_hit_rate_pct: {_format_metric(_diagnostic_metric(diagnostics_df, 'signal_summary', 'overall', 'signal_hit_rate_pct'), digits=4)}",
            f"- avg_predicted_return: {_format_metric(_diagnostic_metric(diagnostics_df, 'signal_summary', 'overall', 'avg_predicted_return'))}",
            f"- avg_realized_return_after_signal: {_format_metric(_diagnostic_metric(diagnostics_df, 'signal_summary', 'overall', 'avg_realized_return_after_signal'))}",
            "",
        ]
    )

    _append_group_diagnostics_table(lines, diagnostics_df, "exit_reason", "Trade Expectancy by Exit Reason", "Exit Reason")
    _append_group_diagnostics_table(lines, diagnostics_df, "volatility_regime", "Trade Expectancy by Volatility Regime", "Volatility Regime")
    _append_group_diagnostics_table(lines, diagnostics_df, "side_performance", "Long vs Short Performance", "Side")

    side_groups = diagnostics_df.loc[diagnostics_df["section"] == "side_performance", "group"].dropna().astype(str).unique().tolist()
    if len(side_groups) == 1:
        missing_side = "sell" if side_groups[0] == "buy" else "buy"
        lines.extend([f"Only {side_groups[0]} trades observed. No {missing_side} trades in best config.", ""])

    return "\n".join(lines)


def build_exit_rule_experiment_markdown(
    df: pd.DataFrame,
    split_meta: ValidationSplitMeta,
    run_status: str = "COMPLETE",
    total_configs: int | None = None,
    completed_configs: int | None = None,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    accepted = df.loc[~df["rejected"]].copy() if "rejected" in df.columns else pd.DataFrame(columns=df.columns)
    best = _select_best_result(df)
    beats_baseline = _beats_baseline(best)

    lines = [
        "# Exit Rule Experiment (Validation Only)",
        f"**Generated:** {generated}",
        f"**Run Status:** {run_status}",
        "",
        "## Run Progress",
        f"- Completed configs: {completed_configs if completed_configs is not None else len(df)}",
        f"- Selected configs: {total_configs if total_configs is not None else len(df)}",
        "",
        "## Scope",
        "- Validation split only experiments.",
        "- Final test split untouched until one config selected.",
        f"- Validation window start: {split_meta.validation_start_ts}",
        f"- Validation window end: {split_meta.validation_end_ts}",
        f"- Final test start: {split_meta.test_start_ts}",
        f"- Final test touched: {'YES' if split_meta.test_split_touched else 'NO'}",
        f"- Model horizon bars: {MODEL_HORIZON_BARS}",
        "",
    ]

    if run_status == "PARTIAL":
        lines.extend(
            [
                "## Partial Run",
                "Run interrupted or still in progress. Artifacts below reflect completed configs only.",
                "Resume with --resume to continue same selected config set.",
                "",
            ]
        )

    if bool(df.get("horizon_mismatch", pd.Series(dtype=bool)).any()):
        lines.extend(
            [
                "## Horizon Alignment Warning",
                "WARNING: Model predicts 1-bar target but time_stop_bars differs in tested configs.",
                "This mismatch can degrade exit quality; benchmark interpretation must stay conservative.",
                "",
            ]
        )

    lines.extend(
        [
            "## Aggregate",
            f"- Total configs with results: {len(df)}",
            f"- Accepted configs: {len(accepted)}",
            f"- Rejected configs: {len(df) - len(accepted)}",
            "",
        ]
    )

    if best is not None:
        lines.extend(
            [
                "## Best Validation Config",
                f"- status: {'ACCEPTED' if not bool(best['rejected']) else 'REJECTED'}",
                f"- k_stop: {best['k_stop']}",
                f"- k_take: {best['k_take']}",
                f"- time_stop_mode: {best['time_stop_mode']}",
                f"- time_stop_bars: {int(best['time_stop_bars'])}",
                f"- trailing_enabled: {bool(best['trailing_enabled'])}",
                f"- trailing_activate_profit_pct: {best['trailing_activate_profit_pct']}",
                f"- trailing_distance_pct: {best['trailing_distance_pct']}",
                f"- max_position_fraction: {best['max_position_fraction']}",
                f"- weekly_return_pct: {best['weekly_return_pct']}",
                f"- sharpe: {best['sharpe']}",
                f"- max_drawdown_pct: {best['max_drawdown_pct']}",
                f"- trade_count: {int(best['trade_count'])}",
                f"- win_rate_pct: {best['win_rate_pct']}",
                f"- profit_factor: {best['profit_factor']}",
                f"- avg_win: {best['avg_win']}",
                f"- avg_loss: {best['avg_loss']}",
                f"- expectancy: {best['expectancy']}",
                f"- baseline_weekly_return_pct: {BASELINE_WEEKLY_RETURN_PCT}",
                f"- beats baseline: {'YES' if beats_baseline else 'NO'}",
                f"- Baseline comparison: {'PASS' if beats_baseline else 'FAIL'}",
                f"- reject_reason: {best['reject_reason'] if str(best['reject_reason']).strip() else 'none'}",
                "",
            ]
        )

        lines.append("### Best Exit Reason Counts")
        lines.append("| Exit Reason | Count |")
        lines.append("|---|---:|")
        try:
            counts = json.loads(str(best["exit_reason_counts"]))
        except Exception:
            counts = {}
        for reason, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"| {reason} | {int(count)} |")
        lines.append("")

        lines.append("## Top 10 Accepted")
        lines.append(
            "| k_stop | k_take | mode | bars | trail | weekly% | sharpe | dd% | trades | pf | expectancy |"
        )
        lines.append("|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
        if len(accepted) > 0:
            for _, row in accepted.sort_values(["weekly_return_pct", "sharpe"], ascending=[False, False]).head(10).iterrows():
                lines.append(
                    f"| {row['k_stop']} | {row['k_take']} | {row['time_stop_mode']} | {int(row['time_stop_bars'])} "
                    f"| {bool(row['trailing_enabled'])} | {row['weekly_return_pct']} | {row['sharpe']} "
                    f"| {row['max_drawdown_pct']} | {int(row['trade_count'])} | {row['profit_factor']} | {row['expectancy']} |"
                )
        else:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
    else:
        lines.extend(
            [
                "## Best Validation Config",
                "No completed config yet.",
                "",
            ]
        )

    lines.extend(
        [
            "## Safety Rules",
            f"- Reject if max drawdown > {MAX_DRAWDOWN_LIMIT:.0%}",
            f"- Reject if trade_count < {MIN_TRADES}",
            f"- Reject if max_position_fraction > {MAX_POSITION_FRACTION_LIMIT}",
            f"- Reject if turnover per bar > {MAX_TURNOVER_PER_BAR}",
            "",
            "## Disclaimer",
            "Validation experiment only. Real benchmark remains honest PASS/FAIL.",
            "No synthetic benchmark counted as real. No leverage increase used.",
            "No profit promise.",
            "Past performance does not predict future results. Trading has risk.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        _ensure_optional_dependencies_for_exit_experiment()
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "ta":
            return 1
        raise

    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import add_technical_features

    output_dir = _ensure_dir(_resolve_output_dir(args.output_dir))
    paths = _build_experiment_paths(output_dir)

    print("[EXIT-EXPERIMENT] Preparing market features...")
    raw = fetch_market_data(ticker="AAPL", interval="5m", period="60d", prepost=False)
    feat = add_technical_features(raw)

    train_val, validation_index, split_meta = split_for_validation_experiment(feat)
    predictions = _load_or_build_prediction_cache(train_val, paths, rebuild_cache=args.rebuild_cache)

    selected_params = _select_experiment_params(quick=args.quick, max_configs=args.max_configs)
    selected_keys = [_config_key(params) for params in selected_params]
    total = len(selected_params)
    print(f"[EXIT-EXPERIMENT] {total} selected configs on validation split only")

    checkpoint = _load_checkpoint(paths.checkpoint_path) if args.resume else None
    results, completed_keys = _restore_results_from_checkpoint(checkpoint, set(selected_keys))
    if args.resume and checkpoint is not None:
        print(f"[EXIT-EXPERIMENT] Resume restored {len(completed_keys)} completed configs")

    initial_status = "COMPLETE" if total == 0 or len(completed_keys) >= total else "PARTIAL"
    _write_checkpoint_and_reports(
        paths,
        results,
        split_meta,
        initial_status,
        total,
        selected_keys,
        completed_keys,
    )

    try:
        for i, params in enumerate(selected_params, 1):
            config_key = selected_keys[i - 1]
            if config_key in completed_keys:
                print(
                    f"  [{i}/{total}] k_stop={params['k_stop']} k_take={params['k_take']} "
                    f"mode={params['time_stop_mode']} bars={params['time_stop_bars']} "
                    f"trail={params['trailing_enabled']} -> SKIP (resume)"
                )
                continue

            print(
                f"  [{i}/{total}] k_stop={params['k_stop']} k_take={params['k_take']} "
                f"mode={params['time_stop_mode']} bars={params['time_stop_bars']} "
                f"trail={params['trailing_enabled']}",
                end="",
            )
            one, _, _ = _run_single_config(train_val, validation_index, split_meta, params, predictions)
            tag = "REJECTED" if one.rejected else f"weekly={one.weekly_return_pct}% sharpe={one.sharpe}"
            print(f" -> {tag}")
            results.append(one)
            completed_keys.add(config_key)
            run_status = "COMPLETE" if len(completed_keys) >= total else "PARTIAL"
            _write_checkpoint_and_reports(
                paths,
                results,
                split_meta,
                run_status,
                total,
                selected_keys,
                completed_keys,
            )
    except KeyboardInterrupt:
        print("\n[EXIT-EXPERIMENT] Interrupted. Partial artifacts saved.")
        _write_checkpoint_and_reports(
            paths,
            results,
            split_meta,
            "PARTIAL",
            total,
            selected_keys,
            completed_keys,
        )
        return 130

    _write_checkpoint_and_reports(
        paths,
        results,
        split_meta,
        "COMPLETE",
        total,
        selected_keys,
        completed_keys,
    )

    df = pd.DataFrame([asdict(result) for result in results])
    best = _select_best_result(df)
    if best is not None:
        best_result, signal_df, trade_df = _run_single_config(
            train_val,
            validation_index,
            split_meta,
            _best_result_to_params(best),
            predictions,
        )
        diagnostics_df = _build_diagnostics_summary(best_result, signal_df, trade_df)
        _write_diagnostics_reports(paths, diagnostics_df, pd.Series(asdict(best_result)), "COMPLETE")

    print(f"\n[EXIT-EXPERIMENT] csv={paths.csv_path}")
    print(f"[EXIT-EXPERIMENT] md={paths.md_path}")
    print(f"[EXIT-EXPERIMENT] diagnostics_csv={paths.diagnostics_csv_path}")
    print(f"[EXIT-EXPERIMENT] diagnostics_md={paths.diagnostics_md_path}")
    print(f"[EXIT-EXPERIMENT] prediction_cache={paths.predictions_cache_path}")
    print(f"[EXIT-EXPERIMENT] checkpoint={paths.checkpoint_path}")

    if best is None:
        print("[EXIT-EXPERIMENT] RESULT: FAIL (no completed config)")
        return 0

    verdict = "PASS" if _beats_baseline(best) else "FAIL"
    print(
        "[EXIT-EXPERIMENT] RESULT: "
        f"{verdict} weekly={best['weekly_return_pct']}% baseline={BASELINE_WEEKLY_RETURN_PCT}% "
        f"sharpe={best['sharpe']} dd={best['max_drawdown_pct']}% trades={int(best['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
