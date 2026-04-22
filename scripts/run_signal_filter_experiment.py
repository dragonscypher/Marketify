"""Signal-filter validation experiment on top of best exit config.

Uses validation split only. Final test split stays untouched.
Reuses cached rolling predictions from exit-rule experiment when available.

Outputs:
    reports/signal_filter_experiment.csv
    reports/signal_filter_experiment.md
    reports/horizon_alignment_audit.md
"""
from __future__ import annotations

import argparse
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

from scripts.run_exit_rule_experiment import (  # noqa: E402
    BASELINE_WEEKLY_RETURN_PCT,
    MAX_DRAWDOWN_LIMIT,
    MAX_POSITION_FRACTION_LIMIT,
    MAX_TURNOVER_PER_BAR,
    MIN_TRADES,
    MODEL_HORIZON_BARS,
    ExperimentPaths,
    ValidationSplitMeta,
    _build_experiment_paths,
    _compute_realized_signal_return,
    _compute_trade_metrics,
    _ensure_dir,
    _ensure_optional_dependencies_for_exit_experiment,
    _format_metric,
    _label_volatility_regime,
    _load_or_build_prediction_cache,
    _resolve_output_dir,
    _scalar,
    _select_best_result,
    _volatility_regime_bounds,
    split_for_validation_experiment,
)

SIGNAL_FILTER_REPORT = "signal_filter_experiment.csv"
SIGNAL_FILTER_MARKDOWN = "signal_filter_experiment.md"
HORIZON_AUDIT_MARKDOWN = "horizon_alignment_audit.md"
EXIT_EXPERIMENT_CSV = "exit_rule_experiment.csv"

REGIME_OPTIONS: dict[str, tuple[str, ...]] = {
    "all_regimes": ("low", "medium", "high", "unknown"),
    "exclude_high_volatility": ("low", "medium", "unknown"),
    "low_medium_only": ("low", "medium"),
}

PREDICTED_RETURN_THRESHOLDS = (0.0, 0.0005, 0.001)
CONFIDENCE_THRESHOLDS = (0.0, 0.35, 0.6)
MIN_SIGNAL_STRENGTHS = (0.0, 0.00025, 0.0005)
NO_TRADE_ZONES = (0.0, 0.00025)


@dataclass(frozen=True)
class SignalFilterConfig:
    regime_filter_name: str
    allowed_regimes: tuple[str, ...]
    predicted_return_threshold: float
    confidence_threshold: float
    min_signal_strength: float
    no_trade_zone: float


@dataclass
class SignalFilterResult:
    regime_filter_name: str
    allowed_regimes: str
    predicted_return_threshold: float
    confidence_threshold: float
    min_signal_strength: float
    no_trade_zone: float
    weekly_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    avg_predicted_return: float
    avg_realized_return_after_signal: float
    trades_by_volatility_regime: str
    horizon_mismatch: bool
    weekly_target_pass: bool
    baseline_pass: bool
    test_split_touched: bool
    rejected: bool
    reject_reason: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signal-filter validation experiment.")
    parser.add_argument("--output-dir", default="reports", help="Directory for experiment artifacts.")
    return parser.parse_args(argv)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_exit_params(best_result: pd.Series | dict[str, object]) -> dict[str, object]:
    payload = best_result.to_dict() if isinstance(best_result, pd.Series) else dict(best_result)
    return {
        "k_stop": float(payload["k_stop"]),
        "k_take": float(payload["k_take"]),
        "time_stop_bars": int(payload["time_stop_bars"]),
        "time_stop_mode": str(payload["time_stop_mode"]),
        "trailing_enabled": _coerce_bool(payload["trailing_enabled"]),
        "trailing_activate_profit_pct": float(payload["trailing_activate_profit_pct"]),
        "trailing_distance_pct": float(payload["trailing_distance_pct"]),
        "max_position_fraction": float(payload["max_position_fraction"]),
    }


def _load_best_exit_result(output_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = output_dir / EXIT_EXPERIMENT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run python scripts/run_exit_rule_experiment.py --quick --rebuild-cache first."
        )

    df = pd.read_csv(csv_path)
    best = _select_best_result(df)
    if best is None:
        raise ValueError("Exit experiment CSV has no completed configs.")
    return df, best


def _select_filter_configs() -> list[SignalFilterConfig]:
    configs: list[SignalFilterConfig] = []
    for regime_name, regimes in REGIME_OPTIONS.items():
        for predicted_threshold, confidence_threshold, min_signal_strength, no_trade_zone in itertools.product(
            PREDICTED_RETURN_THRESHOLDS,
            CONFIDENCE_THRESHOLDS,
            MIN_SIGNAL_STRENGTHS,
            NO_TRADE_ZONES,
        ):
            configs.append(
                SignalFilterConfig(
                    regime_filter_name=regime_name,
                    allowed_regimes=regimes,
                    predicted_return_threshold=float(predicted_threshold),
                    confidence_threshold=float(confidence_threshold),
                    min_signal_strength=float(min_signal_strength),
                    no_trade_zone=float(no_trade_zone),
                )
            )
    return configs


def _signal_strength(expected_return: float, confidence: float) -> float:
    return float(abs(expected_return) * confidence)


def signal_passes_filters(
    cfg: SignalFilterConfig,
    predicted_return: float,
    confidence: float,
    expected_return: float,
    volatility_regime: str,
) -> tuple[bool, str]:
    if volatility_regime not in cfg.allowed_regimes:
        return False, f"volatility_regime_blocked:{volatility_regime}"
    if cfg.no_trade_zone > 0 and abs(predicted_return) <= cfg.no_trade_zone:
        return False, "no_trade_zone"
    if abs(predicted_return) < cfg.predicted_return_threshold:
        return False, "predicted_return_below_threshold"
    if confidence < cfg.confidence_threshold:
        return False, "confidence_below_threshold"
    if _signal_strength(expected_return, confidence) < cfg.min_signal_strength:
        return False, "signal_strength_below_threshold"
    return True, "passed"


def _trades_by_volatility_regime(trade_df: pd.DataFrame) -> str:
    if trade_df.empty or "volatility_regime" not in trade_df.columns:
        return json.dumps({}, sort_keys=True)
    counts = trade_df["volatility_regime"].value_counts(dropna=False).to_dict()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def _run_filter_config(
    train_val: pd.DataFrame,
    validation_index: pd.Index,
    split_meta: ValidationSplitMeta,
    exit_params: dict[str, object],
    preds: pd.Series,
    filter_cfg: SignalFilterConfig,
) -> tuple[SignalFilterResult, pd.DataFrame, pd.DataFrame]:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.exit_rules import ExitRuleConfig, build_exit_levels, evaluate_dynamic_exit
    from marketify.broker.paper import PaperBroker
    from marketify.config import AppConfig
    from marketify.features.sentiment import FinBERTSentiment
    from marketify.models.ensemble import TradeIdeaInput, build_trade_idea, compute_cvar_95
    from marketify.risk.risk_engine import RiskEngine

    reject_reasons: list[str] = []
    max_position_fraction = float(exit_params["max_position_fraction"])
    if max_position_fraction > MAX_POSITION_FRACTION_LIMIT:
        reject_reasons.append(
            f"hidden leverage: max_position_fraction {max_position_fraction:.4f} > {MAX_POSITION_FRACTION_LIMIT:.4f}"
        )

    config = AppConfig()
    config.broker.max_position_fraction = max_position_fraction
    config.broker.time_stop_bars = int(exit_params["time_stop_bars"])
    db_key = (
        exit_params["k_stop"],
        exit_params["k_take"],
        exit_params["time_stop_bars"],
        exit_params["time_stop_mode"],
        exit_params["trailing_enabled"],
        exit_params["trailing_activate_profit_pct"],
        exit_params["trailing_distance_pct"],
        exit_params["max_position_fraction"],
        filter_cfg.regime_filter_name,
        filter_cfg.predicted_return_threshold,
        filter_cfg.confidence_threshold,
        filter_cfg.min_signal_strength,
        filter_cfg.no_trade_zone,
    )
    config.broker.db_path = f"data/signal_filter_exp_{hash(db_key) & 0xFFFFFFFF}.db"

    broker = PaperBroker(config.broker)
    risk_engine = RiskEngine(config.risk, config.broker)
    sentiment = FinBERTSentiment(enabled=False)
    open_state: dict[str, dict[str, object]] = {}

    exit_cfg = ExitRuleConfig(
        k_stop=float(exit_params["k_stop"]),
        k_take=float(exit_params["k_take"]),
        time_stop_bars=int(exit_params["time_stop_bars"]),
        time_stop_mode=str(exit_params["time_stop_mode"]),
        trailing_enabled=bool(exit_params["trailing_enabled"]),
        trailing_activate_profit_pct=float(exit_params["trailing_activate_profit_pct"]),
        trailing_distance_pct=float(exit_params["trailing_distance_pct"]),
    )

    vol_ref = float(train_val["vol_20"].replace([np.inf, -np.inf], np.nan).dropna().median())
    if not np.isfinite(vol_ref) or vol_ref <= 0:
        vol_ref = 0.01
    vol_low_cutoff, vol_high_cutoff = _volatility_regime_bounds(train_val, validation_index)

    equity_points: list[tuple[pd.Timestamp, float]] = []
    signal_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
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
                side=str(state["side"]),
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
                        "reason": f"signal_filter_experiment:{exit_reason}",
                        "risk_warning": "Paper backtest exit",
                    }
                )

                entry_price = float(state["entry_price"])
                qty = float(state["qty"])
                if state["side"] == "buy":
                    pnl = (float(exit_price) - entry_price) * qty
                    ret_pct = (float(exit_price) / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                else:
                    pnl = (entry_price - float(exit_price)) * qty
                    ret_pct = (1.0 - float(exit_price) / entry_price) * 100.0 if entry_price > 0 else 0.0

                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "symbol": ticker,
                        "side": str(state["side"]),
                        "pnl": round(float(pnl), 6),
                        "return_pct": round(float(ret_pct), 6),
                        "hold_bars": int(bars_held),
                        "exit_reason": str(exit_reason),
                        "volatility_regime": str(state.get("volatility_regime", "unknown")),
                        "expected_return": float(state.get("expected_return", 0.0)),
                        "signal_confidence": float(state.get("confidence", 0.0)),
                    }
                )
                open_state.pop(ticker, None)

        expected = float(preds.loc[ts]) if ts in preds.index and np.isfinite(float(preds.loc[ts])) else np.nan
        if not np.isfinite(expected) or ticker in open_state:
            equity_points.append((ts, broker.get_account()["equity"]))
            continue

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
        if idea is None:
            equity_points.append((ts, broker.get_account()["equity"]))
            continue

        realized_after_signal = _compute_realized_signal_return(train_val, ts)
        signal_hit = float("nan")
        if np.isfinite(realized_after_signal):
            signal_hit = bool(realized_after_signal > 0.0) if idea.side == "buy" else bool(realized_after_signal < 0.0)

        filter_passed, filter_reason = signal_passes_filters(
            filter_cfg,
            predicted_return=float(expected),
            confidence=float(idea.confidence),
            expected_return=float(idea.expected_return),
            volatility_regime=volatility_regime,
        )

        signal_rows.append(
            {
                "timestamp": str(ts),
                "symbol": ticker,
                "side": idea.side,
                "predicted_return": float(expected),
                "expected_return": float(idea.expected_return),
                "confidence": float(idea.confidence),
                "signal_strength": _signal_strength(float(idea.expected_return), float(idea.confidence)),
                "realized_return_after_signal": float(realized_after_signal),
                "signal_hit": signal_hit,
                "filter_passed": bool(filter_passed),
                "filter_reason": filter_reason,
                "volatility_regime": volatility_regime,
            }
        )

        if not filter_passed:
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
                "volatility_regime": volatility_regime,
            }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([value for _, value in equity_points], index=[ts for ts, _ in equity_points], name="equity")
    history = [{"ts": str(ts), "equity": float(eq)} for ts, eq in zip(equity.index, equity.values)]
    benchmark = compute_benchmark(history, weekly_goal=0.01)

    signal_df = pd.DataFrame(signal_rows)
    trade_df = pd.DataFrame(trade_rows)
    passed_signal_df = signal_df.loc[signal_df.get("filter_passed", False)].copy() if not signal_df.empty else signal_df
    trade_metrics = _compute_trade_metrics(trade_df if "pnl" in trade_df.columns else pd.DataFrame(columns=["pnl", "exit_reason"]))

    turnover = (trade_metrics["trade_count"] / max(len(equity), 1)) if len(equity) > 0 else 0.0
    if trade_metrics["trade_count"] < MIN_TRADES:
        reject_reasons.append(f"too few trades: {trade_metrics['trade_count']} < {MIN_TRADES}")
    if benchmark.max_drawdown > MAX_DRAWDOWN_LIMIT:
        reject_reasons.append(f"max drawdown unsafe: {benchmark.max_drawdown:.4f} > {MAX_DRAWDOWN_LIMIT:.4f}")
    if turnover > MAX_TURNOVER_PER_BAR:
        reject_reasons.append(f"unrealistic turnover: {turnover:.4f} > {MAX_TURNOVER_PER_BAR:.4f}")

    avg_predicted_return = float(passed_signal_df["predicted_return"].mean()) if not passed_signal_df.empty else 0.0
    avg_realized_after_signal = (
        float(passed_signal_df["realized_return_after_signal"].mean()) if not passed_signal_df.empty else 0.0
    )
    horizon_mismatch = MODEL_HORIZON_BARS != int(exit_params["time_stop_bars"])
    rejected = len(reject_reasons) > 0

    try:
        Path(config.broker.db_path).unlink(missing_ok=True)
    except Exception:
        pass

    result = SignalFilterResult(
        regime_filter_name=filter_cfg.regime_filter_name,
        allowed_regimes=json.dumps(list(filter_cfg.allowed_regimes)),
        predicted_return_threshold=float(filter_cfg.predicted_return_threshold),
        confidence_threshold=float(filter_cfg.confidence_threshold),
        min_signal_strength=float(filter_cfg.min_signal_strength),
        no_trade_zone=float(filter_cfg.no_trade_zone),
        weekly_return_pct=round(float(benchmark.weekly_return) * 100.0, 4),
        sharpe=round(float(benchmark.sharpe_ratio), 4),
        max_drawdown_pct=round(float(benchmark.max_drawdown) * 100.0, 4),
        trade_count=int(trade_metrics["trade_count"]),
        win_rate_pct=float(trade_metrics["win_rate_pct"]),
        profit_factor=float(trade_metrics["profit_factor"]),
        expectancy=float(trade_metrics["expectancy"]),
        avg_predicted_return=round(avg_predicted_return, 6),
        avg_realized_return_after_signal=round(avg_realized_after_signal, 6),
        trades_by_volatility_regime=_trades_by_volatility_regime(trade_df),
        horizon_mismatch=bool(horizon_mismatch),
        weekly_target_pass=bool(float(benchmark.weekly_return) >= 0.01),
        baseline_pass=bool((not rejected) and float(benchmark.weekly_return) * 100.0 >= BASELINE_WEEKLY_RETURN_PCT),
        test_split_touched=bool(split_meta.test_split_touched),
        rejected=bool(rejected),
        reject_reason="; ".join(reject_reasons),
    )
    return result, signal_df, trade_df


def build_horizon_alignment_audit_markdown(exit_df: pd.DataFrame, best_exit_result: pd.Series) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    time_stop_bars = int(best_exit_result["time_stop_bars"])
    mismatch = MODEL_HORIZON_BARS != time_stop_bars
    horizon_flags = exit_df.get("horizon_mismatch", pd.Series(dtype=bool))
    if not horizon_flags.empty:
        all_mismatch = horizon_flags.map(_coerce_bool).all()
    else:
        all_mismatch = mismatch

    lines = [
        "# Horizon Alignment Audit",
        f"**Generated:** {generated}",
        "",
        "## Observed Settings",
        f"- model_prediction_horizon_bars: {MODEL_HORIZON_BARS}",
        f"- best_exit_time_stop_bars: {time_stop_bars}",
        f"- mismatch: {'YES' if mismatch else 'NO'}",
        f"- all_exit_configs_horizon_mismatch: {'YES' if all_mismatch else 'NO'}",
        "",
    ]

    if mismatch:
        lines.extend(
            [
                "## Warning",
                "Model predicts 1-bar return while exit holds longer. Mismatch can distort signal quality and exit timing.",
                "",
                "## Recommended Aligned Settings",
                f"- Keep current model horizon and set time_stop_bars={MODEL_HORIZON_BARS} for aligned validation.",
                f"- Or retrain target_next_ret for {time_stop_bars}-bar horizon before evaluating same exit hold length.",
                "- Do not count mismatched validation as proof of profit.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Status",
                "Prediction horizon and exit hold length aligned in best config.",
                "",
            ]
        )

    return "\n".join(lines)


def build_signal_filter_experiment_markdown(
    df: pd.DataFrame,
    split_meta: ValidationSplitMeta,
    best_exit_result: pd.Series,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    best = _select_best_result(df)
    accepted = df.loc[~df["rejected"]].copy() if "rejected" in df.columns else pd.DataFrame(columns=df.columns)

    lines = [
        "# Signal Filter Experiment (Validation Only)",
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
        "",
        "## Base Exit Config",
        f"- k_stop: {best_exit_result['k_stop']}",
        f"- k_take: {best_exit_result['k_take']}",
        f"- time_stop_mode: {best_exit_result['time_stop_mode']}",
        f"- time_stop_bars: {int(best_exit_result['time_stop_bars'])}",
        f"- trailing_enabled: {_coerce_bool(best_exit_result['trailing_enabled'])}",
        "",
    ]

    if best is None:
        lines.extend(
            [
                "## Result",
                "No completed filter config.",
                "",
            ]
        )
        return "\n".join(lines)

    baseline_verdict = "PASS" if _coerce_bool(best["baseline_pass"]) else "FAIL"
    target_verdict = "PASS" if _coerce_bool(best["weekly_target_pass"]) else "FAIL"
    lines.extend(
        [
            "## Best Validation Config",
            f"- status: {'ACCEPTED' if not _coerce_bool(best['rejected']) else 'REJECTED'}",
            f"- regime_filter_name: {best['regime_filter_name']}",
            f"- allowed_regimes: {best['allowed_regimes']}",
            f"- predicted_return_threshold: {best['predicted_return_threshold']}",
            f"- confidence_threshold: {best['confidence_threshold']}",
            f"- min_signal_strength: {best['min_signal_strength']}",
            f"- no_trade_zone: {best['no_trade_zone']}",
            f"- weekly_return_pct: {best['weekly_return_pct']}",
            f"- baseline_weekly_return_pct: {BASELINE_WEEKLY_RETURN_PCT}",
            f"- baseline comparison: {baseline_verdict}",
            f"- weekly target (1.0%): {target_verdict}",
            f"- sharpe: {best['sharpe']}",
            f"- max_drawdown_pct: {best['max_drawdown_pct']}",
            f"- trade_count: {int(best['trade_count'])}",
            f"- win_rate_pct: {best['win_rate_pct']}",
            f"- profit_factor: {best['profit_factor']}",
            f"- expectancy: {best['expectancy']}",
            f"- avg_predicted_return: {best['avg_predicted_return']}",
            f"- avg_realized_return_after_signal: {best['avg_realized_return_after_signal']}",
            f"- trades_by_volatility_regime: {best['trades_by_volatility_regime']}",
            f"- horizon_mismatch: {_coerce_bool(best['horizon_mismatch'])}",
            f"- reject_reason: {best['reject_reason'] if str(best['reject_reason']).strip() else 'none'}",
            "",
        ]
    )

    lines.extend(
        [
            "## Aggregate",
            f"- Total filter configs: {len(df)}",
            f"- Accepted configs: {len(accepted)}",
            f"- Rejected configs: {len(df) - len(accepted)}",
            "",
            "## Top 10 Configs",
            "| filter | pred_thr | conf_thr | strength | zone | weekly% | sharpe | dd% | trades | baseline | target |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for _, row in df.sort_values(["rejected", "weekly_return_pct", "sharpe"], ascending=[True, False, False]).head(10).iterrows():
        lines.append(
            f"| {row['regime_filter_name']} | {row['predicted_return_threshold']} | {row['confidence_threshold']} "
            f"| {row['min_signal_strength']} | {row['no_trade_zone']} | {row['weekly_return_pct']} | {row['sharpe']} "
            f"| {row['max_drawdown_pct']} | {int(row['trade_count'])} | {'PASS' if _coerce_bool(row['baseline_pass']) else 'FAIL'} "
            f"| {'PASS' if _coerce_bool(row['weekly_target_pass']) else 'FAIL'} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "- Filter run reuses cached rolling predictions when present.",
            "- No leverage increase used.",
            "- No synthetic result counted as real.",
            "- No profit promise.",
            "Past performance does not predict future results. Trading has risk.",
        ]
    )
    return "\n".join(lines)


def _write_signal_filter_outputs(
    output_dir: Path,
    df: pd.DataFrame,
    split_meta: ValidationSplitMeta,
    best_exit_result: pd.Series,
    exit_df: pd.DataFrame,
) -> None:
    csv_path = output_dir / SIGNAL_FILTER_REPORT
    md_path = output_dir / SIGNAL_FILTER_MARKDOWN
    audit_path = output_dir / HORIZON_AUDIT_MARKDOWN

    df.to_csv(csv_path, index=False)
    md_path.write_text(build_signal_filter_experiment_markdown(df, split_meta, best_exit_result), encoding="utf-8")
    audit_path.write_text(build_horizon_alignment_audit_markdown(exit_df, best_exit_result), encoding="utf-8")


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
    exit_paths: ExperimentPaths = _build_experiment_paths(output_dir)
    exit_df, best_exit_result = _load_best_exit_result(output_dir)
    exit_params = _coerce_exit_params(best_exit_result)

    print("[SIGNAL-FILTER] Preparing market features...")
    raw = fetch_market_data(ticker="AAPL", interval="5m", period="60d", prepost=False)
    feat = add_technical_features(raw)
    train_val, validation_index, split_meta = split_for_validation_experiment(feat)
    preds = _load_or_build_prediction_cache(train_val, exit_paths, rebuild_cache=False)

    configs = _select_filter_configs()
    results: list[SignalFilterResult] = []
    print(f"[SIGNAL-FILTER] {len(configs)} filter configs on validation split only")
    for index, filter_cfg in enumerate(configs, start=1):
        print(
            f"  [{index}/{len(configs)}] regime={filter_cfg.regime_filter_name} pred_thr={filter_cfg.predicted_return_threshold} "
            f"conf_thr={filter_cfg.confidence_threshold} strength={filter_cfg.min_signal_strength} zone={filter_cfg.no_trade_zone}",
            end="",
        )
        result, _, _ = _run_filter_config(train_val, validation_index, split_meta, exit_params, preds, filter_cfg)
        tag = "REJECTED" if result.rejected else f"weekly={result.weekly_return_pct}% sharpe={result.sharpe}"
        print(f" -> {tag}")
        results.append(result)

    df = pd.DataFrame([asdict(result) for result in results])
    _write_signal_filter_outputs(output_dir, df, split_meta, best_exit_result, exit_df)

    best = _select_best_result(df)
    print(f"\n[SIGNAL-FILTER] csv={output_dir / SIGNAL_FILTER_REPORT}")
    print(f"[SIGNAL-FILTER] md={output_dir / SIGNAL_FILTER_MARKDOWN}")
    print(f"[SIGNAL-FILTER] horizon_audit={output_dir / HORIZON_AUDIT_MARKDOWN}")
    if best is None:
        print("[SIGNAL-FILTER] RESULT: FAIL (no completed config)")
        return 0

    print(
        "[SIGNAL-FILTER] RESULT: "
        f"{'PASS' if _coerce_bool(best['baseline_pass']) else 'FAIL'} weekly={best['weekly_return_pct']}% "
        f"baseline={BASELINE_WEEKLY_RETURN_PCT}% sharpe={best['sharpe']} dd={best['max_drawdown_pct']}% trades={int(best['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())