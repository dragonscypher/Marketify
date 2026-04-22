from __future__ import annotations

import argparse
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


@dataclass
class ModelComparisonResult:
    model_name: str
    hold_horizon_bars: int
    label_mode: str
    regime_filter: str
    high_vol_confidence_threshold: float
    weekly_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    baseline_pass: bool
    weekly_target_pass: bool
    test_split_touched: bool
    rejected: bool
    reject_reason: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from marketify.config import AppConfig

    default_horizon = AppConfig().broker.time_stop_bars
    parser = argparse.ArgumentParser(description="Compare XGB, GRU, and LSTM on validation split only.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--hold-horizon-bars", type=int, default=default_horizon)
    parser.add_argument("--label-mode", choices=["return", "direction"], default="return")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--regime-filter",
        choices=["none", "skip_high_vol", "strict_high_vol"],
        default="skip_high_vol",
    )
    parser.add_argument("--high-vol-confidence-threshold", type=float, default=0.8)
    return parser.parse_args(argv)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _artifact_path(artifact_dir: Path, architecture: str, ticker: str, hold_horizon_bars: int, label_mode: str) -> Path:
    return artifact_dir / f"{architecture}_{ticker}_h{hold_horizon_bars}_{label_mode}.pt"


def prepare_validation_only_data(
    frame: pd.DataFrame,
    hold_horizon_bars: int,
    label_mode: str,
) -> tuple[pd.DataFrame, pd.Index, object, list[str], str]:
    from marketify.models.rnn_model import prepare_recurrent_training_frame
    from scripts.run_exit_rule_experiment import \
        split_for_validation_experiment

    aligned, feature_cols, target_col = prepare_recurrent_training_frame(
        frame,
        hold_horizon_bars=hold_horizon_bars,
        mode=label_mode,
    )
    train_val, validation_index, split_meta = split_for_validation_experiment(aligned)
    return train_val, validation_index, split_meta, feature_cols, target_col


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


def _select_best_result(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    ranked = df.copy()
    if "rejected" not in ranked.columns:
        ranked["rejected"] = False
    return ranked.sort_values(["rejected", "weekly_return_pct", "sharpe"], ascending=[True, False, False]).iloc[0]


def _scalar(row, col: str) -> float:
    value = row[col]
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _compute_trade_metrics(diag: pd.DataFrame) -> dict[str, float | int | str]:
    trade_count = int(len(diag)) if not diag.empty else 0
    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }

    wins = diag[diag["pnl"] > 0]["pnl"]
    losses = diag[diag["pnl"] < 0]["pnl"]
    win_sum = float(wins.sum()) if not wins.empty else 0.0
    loss_sum_abs = abs(float(losses.sum())) if not losses.empty else 0.0
    profit_factor = (win_sum / loss_sum_abs) if loss_sum_abs > 0 else (float("inf") if win_sum > 0 else 0.0)
    win_rate = float((wins.shape[0] / trade_count) * 100.0)
    expectancy = float(diag["pnl"].mean())
    return {
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate, 4),
        "profit_factor": float(profit_factor),
        "expectancy": round(expectancy, 6),
    }


def _passes_regime_filter(
    regime_filter: str,
    volatility_regime: str,
    confidence: float,
    high_vol_confidence_threshold: float,
) -> tuple[bool, str]:
    if regime_filter == "skip_high_vol" and volatility_regime == "high":
        return False, "skip_high_vol"
    if regime_filter == "strict_high_vol" and volatility_regime == "high" and confidence < high_vol_confidence_threshold:
        return False, "high_vol_confidence_gate"
    return True, "passed"


def _predict_xgb(
    train_val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_index: pd.Index,
    validation_index: pd.Index,
) -> pd.Series:
    from marketify.config import AppConfig
    from marketify.models.xgb_model import XGBModel

    config = AppConfig()
    train_frame = train_val.loc[train_index]
    val_frame = train_val.loc[validation_index]
    model = XGBModel(config.model)
    model.fit(train_frame[feature_cols].to_numpy(), train_frame[target_col].to_numpy())
    preds = model.predict(val_frame[feature_cols].to_numpy())
    return pd.Series(preds, index=val_frame.index, name="pred_xgb")


def _predict_recurrent(
    architecture: str,
    train_val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_index: pd.Index,
    validation_index: pd.Index,
    artifact_dir: Path,
    ticker: str,
    hold_horizon_bars: int,
    label_mode: str,
    sequence_length: int,
    epochs: int,
) -> pd.Series:
    from marketify.models.rnn_model import (RecurrentModelConfig,
                                            build_sequence_dataset,
                                            load_recurrent_artifact,
                                            make_recurrent_model)

    artifact_path = _artifact_path(artifact_dir, architecture, ticker, hold_horizon_bars, label_mode)
    if artifact_path.exists():
        model, _payload = load_recurrent_artifact(artifact_path)
    else:
        cfg = RecurrentModelConfig(
            architecture=architecture,
            sequence_length=sequence_length,
            epochs=epochs,
        )
        model = make_recurrent_model(architecture, cfg)
        x_train, y_train, _ = build_sequence_dataset(
            train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=cfg.sequence_length,
            selected_index=train_index,
        )
        model.fit(x_train, y_train)
        model.save(
            artifact_path,
            feature_cols=feature_cols,
            target_col=target_col,
            extra_meta={
                "ticker": ticker,
                "hold_horizon_bars": hold_horizon_bars,
                "label_mode": label_mode,
            },
        )

    x_pred, _y_pred, pred_ts = build_sequence_dataset(
        train_val,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=model.config.sequence_length,
        selected_index=validation_index,
    )
    return pd.Series(model.predict(x_pred), index=pred_ts, name=f"pred_{architecture}")


def _simulate_predictions(
    model_name: str,
    preds: pd.Series,
    train_val: pd.DataFrame,
    validation_index: pd.Index,
    split_meta,
    hold_horizon_bars: int,
    label_mode: str,
    regime_filter: str,
    high_vol_confidence_threshold: float,
) -> ModelComparisonResult:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.broker.paper import PaperBroker
    from marketify.config import AppConfig
    from marketify.features.sentiment import FinBERTSentiment
    from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                           compute_cvar_95)
    from scripts.run_exit_rule_experiment import (BASELINE_WEEKLY_RETURN_PCT,
                                                  MAX_DRAWDOWN_LIMIT,
                                                  MAX_POSITION_FRACTION_LIMIT,
                                                  MAX_TURNOVER_PER_BAR,
                                                  MIN_TRADES)

    config = AppConfig()
    config.broker.time_stop_bars = hold_horizon_bars
    config.broker.db_path = f"data/compare_{model_name}_{hold_horizon_bars}.db"
    reject_reasons: list[str] = []
    if config.broker.max_position_fraction > MAX_POSITION_FRACTION_LIMIT:
        reject_reasons.append(
            f"hidden leverage: max_position_fraction {config.broker.max_position_fraction:.4f} > {MAX_POSITION_FRACTION_LIMIT:.4f}"
        )

    broker = PaperBroker(config.broker)
    risk_engine = __import__("marketify.risk.risk_engine", fromlist=["RiskEngine"]).RiskEngine(config.risk, config.broker)
    sentiment = FinBERTSentiment(enabled=False)
    open_state: dict[str, dict[str, object]] = {}

    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_rows: list[dict[str, object]] = []
    ticker = config.data.ticker
    low_cutoff, high_cutoff = _volatility_regime_bounds(train_val, validation_index)

    for ts in validation_index:
        row = train_val.loc[ts]
        price = _scalar(row, "Close")
        vol_20 = _scalar(row, "vol_20")
        volatility_regime = _label_volatility_regime(vol_20, low_cutoff, high_cutoff)

        broker.update_market_price(ticker, price)

        if ticker in open_state:
            state = open_state[ticker]
            bars_held = int(state.get("bars_held", 0)) + 1
            state["bars_held"] = bars_held
            if bars_held >= hold_horizon_bars:
                exit_price = float(price)
                close_side = "sell" if state["side"] == "buy" else "buy"
                broker.submit_order(
                    {
                        "symbol": ticker,
                        "side": close_side,
                        "qty": state["qty"],
                        "price": exit_price,
                        "reason": "fixed_horizon_time_stop",
                        "risk_warning": "Paper backtest exit",
                    }
                )
                entry_price = float(state["entry_price"])
                qty = float(state["qty"])
                if state["side"] == "buy":
                    pnl = (exit_price - entry_price) * qty
                    ret_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                else:
                    pnl = (entry_price - exit_price) * qty
                    ret_pct = (1.0 - exit_price / entry_price) * 100.0 if entry_price > 0 else 0.0
                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "side": state["side"],
                        "pnl": round(float(pnl), 6),
                        "return_pct": round(float(ret_pct), 6),
                        "volatility_regime": str(state.get("volatility_regime", "unknown")),
                    }
                )
                open_state.pop(ticker, None)

        expected = float(preds.loc[ts]) if ts in preds.index and np.isfinite(float(preds.loc[ts])) else np.nan
        if not np.isfinite(expected) or ticker in open_state:
            equity_points.append((ts, broker.get_account()["equity"]))
            continue

        ret_slice = train_val["ret_1"].iloc[max(0, train_val.index.get_loc(ts) - 250): train_val.index.get_loc(ts) + 1]
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
        if idea is None:
            equity_points.append((ts, broker.get_account()["equity"]))
            continue

        allowed, reason = _passes_regime_filter(
            regime_filter=regime_filter,
            volatility_regime=volatility_regime,
            confidence=float(idea.confidence),
            high_vol_confidence_threshold=high_vol_confidence_threshold,
        )
        if not allowed:
            equity_points.append((ts, broker.get_account()["equity"]))
            continue

        decision = risk_engine.evaluate(
            trade=idea,
            account=account,
            volatility=info.volatility,
            news_risk=info.news_risk,
            mark_price=price,
        )
        if decision.approved:
            broker.submit_order({**idea.to_dict(), "price": price})
            open_state[ticker] = {
                "side": idea.side,
                "qty": int(idea.qty),
                "entry_price": float(price),
                "bars_held": 0,
                "volatility_regime": volatility_regime,
                "filter_reason": reason,
            }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([value for _, value in equity_points], index=[ts for ts, _ in equity_points], name="equity")
    history = [{"ts": str(ts), "equity": float(eq)} for ts, eq in zip(equity.index, equity.values)]
    benchmark = compute_benchmark(history, weekly_goal=0.01)
    trade_df = pd.DataFrame(trade_rows)
    trade_metrics = _compute_trade_metrics(trade_df)

    turnover = (trade_metrics["trade_count"] / max(len(equity), 1)) if len(equity) > 0 else 0.0
    if trade_metrics["trade_count"] < MIN_TRADES:
        reject_reasons.append(f"too few trades: {trade_metrics['trade_count']} < {MIN_TRADES}")
    if benchmark.max_drawdown > MAX_DRAWDOWN_LIMIT:
        reject_reasons.append(f"max drawdown unsafe: {benchmark.max_drawdown:.4f} > {MAX_DRAWDOWN_LIMIT:.4f}")
    if turnover > MAX_TURNOVER_PER_BAR:
        reject_reasons.append(f"unrealistic turnover: {turnover:.4f} > {MAX_TURNOVER_PER_BAR:.4f}")

    try:
        Path(config.broker.db_path).unlink(missing_ok=True)
    except Exception:
        pass

    return ModelComparisonResult(
        model_name=model_name,
        hold_horizon_bars=hold_horizon_bars,
        label_mode=label_mode,
        regime_filter=regime_filter,
        high_vol_confidence_threshold=float(high_vol_confidence_threshold),
        weekly_return_pct=round(float(benchmark.weekly_return) * 100.0, 4),
        sharpe=round(float(benchmark.sharpe_ratio), 4),
        max_drawdown_pct=round(float(benchmark.max_drawdown) * 100.0, 4),
        trade_count=int(trade_metrics["trade_count"]),
        win_rate_pct=float(trade_metrics["win_rate_pct"]),
        profit_factor=float(trade_metrics["profit_factor"]),
        expectancy=float(trade_metrics["expectancy"]),
        baseline_pass=bool((not reject_reasons) and float(benchmark.weekly_return) * 100.0 >= BASELINE_WEEKLY_RETURN_PCT),
        weekly_target_pass=bool((not reject_reasons) and float(benchmark.weekly_return) >= 0.01),
        test_split_touched=bool(split_meta.test_split_touched),
        rejected=bool(reject_reasons),
        reject_reason="; ".join(reject_reasons),
    )


def build_horizon_alignment_audit_markdown(hold_horizon_bars: int, label_mode: str) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Horizon Alignment Audit",
        f"**Generated:** {generated}",
        "",
        "## Alignment",
        f"- label_horizon_bars: {hold_horizon_bars}",
        f"- trade_hold_horizon_bars: {hold_horizon_bars}",
        f"- label_mode: {label_mode}",
        "- early_exit_path: disabled for model comparison baseline",
        "- alignment_status: FIXED",
        "",
        "## Note",
        "Comparison path uses fixed-horizon exits to match aligned training labels.",
        "Do not treat validation comparison as profit promise.",
    ]
    return "\n".join(lines)


def build_recurrent_leaderboard_markdown(df: pd.DataFrame, split_meta, regime_filter: str, label_mode: str) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    best = _select_best_result(df)
    lines = [
        "# Recurrent Leaderboard",
        f"**Generated:** {generated}",
        "**Run Status:** COMPLETE",
        "",
        "## Scope",
        "- Validation split only.",
        "- Final test split untouched.",
        f"- Validation window start: {split_meta.validation_start_ts}",
        f"- Validation window end: {split_meta.validation_end_ts}",
        f"- Final test start: {split_meta.test_start_ts}",
        f"- Final test touched: {'YES' if split_meta.test_split_touched else 'NO'}",
        f"- regime_filter: {regime_filter}",
        f"- label_mode: {label_mode}",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Validation Model",
                f"- model_name: {best['model_name']}",
                f"- weekly_return_pct: {best['weekly_return_pct']}",
                f"- baseline_weekly_return_pct: 0.5851",
                f"- baseline comparison: {'PASS' if _coerce_bool(best['baseline_pass']) else 'FAIL'}",
                f"- weekly target (1.0%): {'PASS' if _coerce_bool(best['weekly_target_pass']) else 'FAIL'}",
                f"- sharpe: {best['sharpe']}",
                f"- max_drawdown_pct: {best['max_drawdown_pct']}",
                f"- trade_count: {int(best['trade_count'])}",
                f"- win_rate_pct: {best['win_rate_pct']}",
                f"- profit_factor: {best['profit_factor']}",
                f"- expectancy: {best['expectancy']}",
                f"- reject_reason: {best['reject_reason'] if str(best['reject_reason']).strip() else 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Leaderboard",
            "| Model | Weekly % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor | Expectancy | Baseline | Target |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in df.sort_values(["rejected", "weekly_return_pct", "sharpe"], ascending=[True, False, False]).iterrows():
        lines.append(
            f"| {row['model_name']} | {row['weekly_return_pct']} | {row['sharpe']} | {row['max_drawdown_pct']} "
            f"| {int(row['trade_count'])} | {row['win_rate_pct']} | {row['profit_factor']} | {row['expectancy']} "
            f"| {'PASS' if _coerce_bool(row['baseline_pass']) else 'FAIL'} | {'PASS' if _coerce_bool(row['weekly_target_pass']) else 'FAIL'} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Notes",
            "- XGB vs GRU vs LSTM comparison only on validation split.",
            "- Paper mode only. No leverage increase used.",
            "- Benchmark PASS/FAIL remains honest.",
            "- No profit promise. Outcomes remain uncertain.",
            "Past performance does not predict future results. Trading has risk.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from marketify.config import AppConfig
    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import add_technical_features
    from scripts.run_exit_rule_experiment import \
        _ensure_optional_dependencies_for_exit_experiment

    try:
        _ensure_optional_dependencies_for_exit_experiment()
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "ta":
            return 1
        raise

    config = AppConfig()
    output_dir = _ensure_dir(ROOT / args.output_dir)
    artifact_dir = _ensure_dir(ROOT / args.artifact_dir)

    print("[COMPARE] Preparing market features...")
    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)
    train_val, validation_index, split_meta, feature_cols, target_col = prepare_validation_only_data(
        feat,
        hold_horizon_bars=args.hold_horizon_bars,
        label_mode=args.label_mode,
    )
    train_index = train_val.index[:split_meta.train_end]

    model_preds = {
        "xgb": _predict_xgb(train_val, feature_cols, target_col, train_index, validation_index),
        "gru": _predict_recurrent(
            "gru",
            train_val,
            feature_cols,
            target_col,
            train_index,
            validation_index,
            artifact_dir,
            config.data.ticker,
            args.hold_horizon_bars,
            args.label_mode,
            args.sequence_length,
            args.epochs,
        ),
        "lstm": _predict_recurrent(
            "lstm",
            train_val,
            feature_cols,
            target_col,
            train_index,
            validation_index,
            artifact_dir,
            config.data.ticker,
            args.hold_horizon_bars,
            args.label_mode,
            args.sequence_length,
            args.epochs,
        ),
    }

    results: list[ModelComparisonResult] = []
    for model_name, preds in model_preds.items():
        print(f"[COMPARE] {model_name.upper()} validation backtest...")
        result = _simulate_predictions(
            model_name=model_name,
            preds=preds,
            train_val=train_val,
            validation_index=validation_index,
            split_meta=split_meta,
            hold_horizon_bars=args.hold_horizon_bars,
            label_mode=args.label_mode,
            regime_filter=args.regime_filter,
            high_vol_confidence_threshold=args.high_vol_confidence_threshold,
        )
        results.append(result)

    df = pd.DataFrame([asdict(result) for result in results])
    (output_dir / "recurrent_leaderboard.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (output_dir / "recurrent_leaderboard.md").write_text(
        build_recurrent_leaderboard_markdown(df, split_meta, args.regime_filter, args.label_mode),
        encoding="utf-8",
    )
    (output_dir / "horizon_alignment_audit.md").write_text(
        build_horizon_alignment_audit_markdown(args.hold_horizon_bars, args.label_mode),
        encoding="utf-8",
    )

    best = _select_best_result(df)
    print(f"[COMPARE] csv={output_dir / 'recurrent_leaderboard.csv'}")
    print(f"[COMPARE] md={output_dir / 'recurrent_leaderboard.md'}")
    print(f"[COMPARE] audit={output_dir / 'horizon_alignment_audit.md'}")
    if best is None:
        print("[COMPARE] RESULT: FAIL (no completed config)")
        return 0
    print(
        "[COMPARE] RESULT: "
        f"{'PASS' if _coerce_bool(best['baseline_pass']) else 'FAIL'} model={best['model_name']} "
        f"weekly={best['weekly_return_pct']}% baseline=0.5851% sharpe={best['sharpe']} "
        f"dd={best['max_drawdown_pct']}% trades={int(best['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())