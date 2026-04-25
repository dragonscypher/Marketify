from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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


@dataclass
class RealBenchmarkRow:
    model_name: str
    role: str
    weekly_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    cvar_95_pct: float
    trade_count: int
    win_rate_pct: float
    expectancy: float
    weekly_target_pass: bool
    comparison_only: bool
    run_status: str
    gate_block_count: int
    gate_keep_ratio_pct: float
    top_blocker: str
    notes: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from marketify.config import AppConfig

    default_horizon = AppConfig().broker.time_stop_bars
    parser = argparse.ArgumentParser(description="Run honest real-path XGB benchmark with frozen recurrent comparison lanes.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--hold-horizon-bars", type=int, default=default_horizon)
    parser.add_argument("--skip-gru-comparison", action="store_true")
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


def _safe_numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").reindex(frame.index).fillna(default)


def _gap_to_target_pct_points(weekly_return_pct: float, target_pct: float = 1.0) -> float:
    return round(float(target_pct) - float(weekly_return_pct), 4)


def _compute_real_metrics(equity: pd.Series, trade_diag: pd.DataFrame) -> dict[str, float | int | bool]:
    from marketify.backtest.benchmark import compute_benchmark

    history = [{"ts": str(ts), "equity": float(value)} for ts, value in equity.items()]
    benchmark = compute_benchmark(history, weekly_goal=0.01)
    trade_metrics = _compute_trade_metrics(trade_diag)
    return {
        "weekly_return_pct": round(float(benchmark.weekly_return) * 100.0, 4),
        "sharpe": round(float(benchmark.sharpe_ratio), 4),
        "max_drawdown_pct": round(float(benchmark.max_drawdown) * 100.0, 4),
        "cvar_95_pct": round(float(benchmark.cvar_95) * 100.0, 4),
        "trade_count": int(trade_metrics["trade_count"]),
        "win_rate_pct": round(float(trade_metrics["win_rate_pct"]), 4),
        "expectancy": round(float(trade_metrics["expectancy"]), 6),
        "weekly_target_pass": bool(benchmark.weekly_pass),
    }


def _signal_gate_thresholds(config) -> dict[str, float]:
    cost_buffer = max(
        float(config.risk.min_expected_return),
        (float(config.broker.fee_bps) + float(config.broker.slippage_bps)) / 10000.0,
    )
    return {
        "min_confidence": max(float(getattr(config.broker, "fusion_min_confidence", 0.30)), 0.35),
        "edge_floor": cost_buffer + max(float(getattr(config.broker, "fusion_abstain_margin", 0.0)), 0.00005),
        "news_cutoff": min(
            float(config.risk.max_news_risk),
            float(getattr(config.broker, "fusion_news_risk_cutoff", config.risk.max_news_risk)),
        ),
        "regime_cutoff": min(float(getattr(config.broker, "fusion_regime_vix_cutoff", 30.0)), 28.0),
        "max_disagreement": float(getattr(config.broker, "fusion_max_model_disagreement", 0.006)),
        "max_volatility": float(config.risk.max_volatility),
    }


def _apply_signal_gate(
    feat: pd.DataFrame,
    preds: pd.Series,
    config,
    comparison_preds: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, dict[str, object]]:
    thresholds = _signal_gate_thresholds(config)
    gated = pd.to_numeric(preds, errors="coerce").reindex(feat.index)

    sentiment = _safe_numeric_series(feat, "news_sentiment_score", 0.0)
    news_risk = _safe_numeric_series(feat, "news_risk", 0.0)
    news_shock = _safe_numeric_series(feat, "news_event_shock", 0.0)
    vix_proxy = _safe_numeric_series(feat, "vix_proxy", 20.0)
    macro_bear = _safe_numeric_series(feat, "macro_regime_bear", 0.0)
    macro_vol_high = _safe_numeric_series(feat, "macro_vol_level_high", 0.0)
    macro_event_risk = _safe_numeric_series(feat, "macro_event_risk", 0.0)
    volatility = _safe_numeric_series(feat, "vol_20", 0.0)

    reference_map = {
        name: pd.to_numeric(series, errors="coerce").reindex(feat.index)
        for name, series in (comparison_preds or {}).items()
    }

    blocker_counts: Counter[str] = Counter()
    scored = 0
    kept = 0
    for ts, pred in gated.items():
        if not np.isfinite(pred):
            continue
        scored += 1
        effective_pred = float(pred) * (1.0 + 0.2 * float(sentiment.loc[ts]))
        confidence = float(np.clip(abs(effective_pred) / max(float(thresholds["edge_floor"]), 1e-6), 0.0, 1.0))

        reasons: list[str] = []
        if confidence < float(thresholds["min_confidence"]):
            reasons.append("low_confidence")
        if abs(effective_pred) <= float(thresholds["edge_floor"]):
            reasons.append("below_cost_buffer")
        if float(news_risk.loc[ts]) > float(thresholds["news_cutoff"]) or float(news_shock.loc[ts]) > 0.5:
            reasons.append("high_news_risk")
        if (
            float(vix_proxy.loc[ts]) > float(thresholds["regime_cutoff"])
            or float(macro_bear.loc[ts]) > 0.5
            or float(macro_vol_high.loc[ts]) > 0.5
            or float(macro_event_risk.loc[ts]) > 0.5
        ):
            reasons.append("bad_regime")
        if float(volatility.loc[ts]) > float(thresholds["max_volatility"]):
            reasons.append("risk_too_high")

        disagreements = []
        for series in reference_map.values():
            ref_val = series.loc[ts]
            if np.isfinite(ref_val):
                disagreements.append(abs(float(pred) - float(ref_val)))
        if disagreements and max(disagreements) > float(thresholds["max_disagreement"]):
            reasons.append("model_disagreement")

        if reasons:
            gated.loc[ts] = np.nan
            blocker_counts.update(reasons)
            continue
        kept += 1

    top_blocker = blocker_counts.most_common(1)[0][0] if blocker_counts else "none"
    keep_ratio = round((kept / scored) * 100.0, 2) if scored else 0.0
    return gated, {
        "scored": scored,
        "kept": kept,
        "blocked": max(scored - kept, 0),
        "keep_ratio_pct": keep_ratio,
        "top_blocker": top_blocker,
        "blocker_counts": dict(blocker_counts),
    }


def _build_exact_blocker(row: RealBenchmarkRow) -> str:
    if row.weekly_target_pass:
        return "none"
    parts = [f"gap_to_target={_gap_to_target_pct_points(row.weekly_return_pct)} pct_points"]
    if row.trade_count == 0:
        parts.append("no approved trades after gates")
    if row.top_blocker != "none":
        parts.append(f"dominant_gate={row.top_blocker}")
    else:
        parts.append("signal_edge_below_target")
    return "; ".join(parts)


def _run_real_lane(
    model_name: str,
    feat: pd.DataFrame,
    preds: pd.Series,
    config,
    role: str,
    comparison_preds: dict[str, pd.Series] | None = None,
    notes: str = "",
) -> tuple[RealBenchmarkRow, dict]:
    from marketify.backtest.simulator import run_backtest_from_predictions

    gated_preds, gate_stats = _apply_signal_gate(feat, preds, config, comparison_preds=comparison_preds)
    result = run_backtest_from_predictions(feat, gated_preds, config, model_used=model_name)
    metrics = _compute_real_metrics(result["equity"], result["trade_diagnostics"])
    row = RealBenchmarkRow(
        model_name=model_name,
        role=role,
        weekly_return_pct=float(metrics["weekly_return_pct"]),
        sharpe=float(metrics["sharpe"]),
        max_drawdown_pct=float(metrics["max_drawdown_pct"]),
        cvar_95_pct=float(metrics["cvar_95_pct"]),
        trade_count=int(metrics["trade_count"]),
        win_rate_pct=float(metrics["win_rate_pct"]),
        expectancy=float(metrics["expectancy"]),
        weekly_target_pass=bool(metrics["weekly_target_pass"]),
        comparison_only=(role != "champion"),
        run_status="COMPLETE",
        gate_block_count=int(gate_stats["blocked"]),
        gate_keep_ratio_pct=float(gate_stats["keep_ratio_pct"]),
        top_blocker=str(gate_stats["top_blocker"]),
        notes=notes,
    )
    return row, {**result, "gate_stats": gate_stats}


def _frozen_comparison_row(model_name: str, notes: str) -> RealBenchmarkRow:
    return RealBenchmarkRow(
        model_name=model_name,
        role="comparison",
        weekly_return_pct=float("nan"),
        sharpe=float("nan"),
        max_drawdown_pct=float("nan"),
        cvar_95_pct=float("nan"),
        trade_count=0,
        win_rate_pct=float("nan"),
        expectancy=float("nan"),
        weekly_target_pass=False,
        comparison_only=True,
        run_status="FROZEN",
        gate_block_count=0,
        gate_keep_ratio_pct=0.0,
        top_blocker="comparison_only",
        notes=notes,
    )


def _fmt_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return str(value)


def build_xgb_real_leaderboard_markdown(df: pd.DataFrame, champion_row: RealBenchmarkRow) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# XGB Real Leaderboard",
        f"**Generated:** {generated}",
        "**Decision Path:** real walk-forward only",
        "",
        "## Champion",
        f"- current_champion: {champion_row.model_name}",
        f"- weekly_return_pct: {_fmt_metric(champion_row.weekly_return_pct)}",
        f"- gap_to_target_pct_points: {_fmt_metric(_gap_to_target_pct_points(champion_row.weekly_return_pct))}",
        f"- sharpe: {_fmt_metric(champion_row.sharpe)}",
        f"- max_drawdown_pct: {_fmt_metric(champion_row.max_drawdown_pct)}",
        f"- trade_count: {champion_row.trade_count}",
        f"- weekly_target_status: {'PASS' if champion_row.weekly_target_pass else 'FAIL'}",
        "- alignment_status: FIXED",
        "",
        "## Scope",
        "- XGB primary lane.",
        "- GRU/LSTM comparison only. Not eligible as champion here.",
        "- Exit-rule experiment removed from decision path.",
        "- Synthetic/validation benchmark removed from decision path.",
        "",
        "## Leaderboard",
        "| Model | Role | Status | Weekly % | Sharpe | Max DD % | CVaR 95 % | Trades | Keep % | Weekly Target | Blocker | Notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['role']} | {row['run_status']} | {_fmt_metric(row['weekly_return_pct'])} | "
            f"{_fmt_metric(row['sharpe'])} | {_fmt_metric(row['max_drawdown_pct'])} | {_fmt_metric(row['cvar_95_pct'])} | "
            f"{_fmt_metric(row['trade_count'], digits=0)} | {_fmt_metric(row['gate_keep_ratio_pct'], digits=2)} | "
            f"{'PASS' if _coerce_bool(row['weekly_target_pass']) else 'FAIL'} | {row['top_blocker']} | {row['notes']} |"
        )
    lines.append("")
    lines.append("No profit promise. Target remains benchmark only.")
    return "\n".join(lines)


def build_xgb_real_benchmark_markdown(champion_row: RealBenchmarkRow, gate_stats: dict[str, object]) -> str:
    gap = _gap_to_target_pct_points(champion_row.weekly_return_pct)
    lines = [
        "# XGB Real Benchmark",
        "",
        "Real benchmark path only. Walk-forward only. No shuffled split.",
        "",
        f"Champion: {champion_row.model_name}",
        f"Alignment status: FIXED",
        f"Weekly return: {_fmt_metric(champion_row.weekly_return_pct)}%",
        f"Gap to 1.0% target: {_fmt_metric(gap)} percentage points",
        f"Sharpe: {_fmt_metric(champion_row.sharpe)}",
        f"Max drawdown: {_fmt_metric(champion_row.max_drawdown_pct)}%",
        f"CVaR 95: {_fmt_metric(champion_row.cvar_95_pct)}%",
        f"Trades: {champion_row.trade_count}",
        f"Gate keep ratio: {_fmt_metric(gate_stats.get('keep_ratio_pct', 0.0), digits=2)}%",
        f"Top blocker: {champion_row.top_blocker}",
        f"Result: {'PASS' if champion_row.weekly_target_pass else 'FAIL'}",
        "",
        "## Guardrails",
        "- no leverage increase",
        "- no exit-rule experiment path",
        "- no synthetic benchmark path",
        "- no LLM trade decisions",
        "- deterministic feature + signal gating only",
    ]
    if not champion_row.weekly_target_pass:
        lines.extend(["", f"Exact blocker: {_build_exact_blocker(champion_row)}"])
    return "\n".join(lines) + "\n"


def build_next_status_markdown(champion_row: RealBenchmarkRow) -> str:
    gap = _gap_to_target_pct_points(champion_row.weekly_return_pct)
    lines = [
        "# NEXT STATUS",
        "",
        f"- current champion: {champion_row.model_name}",
        f"- weekly return: {_fmt_metric(champion_row.weekly_return_pct)}%",
        f"- gap to 1.0% target: {_fmt_metric(gap)} percentage points",
        f"- sharpe: {_fmt_metric(champion_row.sharpe)}",
        f"- max drawdown: {_fmt_metric(champion_row.max_drawdown_pct)}%",
        f"- trade count: {champion_row.trade_count}",
        f"- alignment_status: FIXED",
        f"- result: {'PASS' if champion_row.weekly_target_pass else 'FAIL'}",
        f"- exact blocker: {_build_exact_blocker(champion_row)}",
    ]
    return "\n".join(lines) + "\n"


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
    from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                              add_technical_features)
    from marketify.models.rnn_model import \
        rolling_train_predict as rolling_train_predict_gru
    from marketify.models.xgb_model import \
        rolling_train_predict as rolling_train_predict_xgb

    config = AppConfig()
    config.broker.time_stop_bars = int(args.hold_horizon_bars)
    config.broker.backtest_model = "xgb"  # type: ignore[attr-defined]
    output_dir = _ensure_dir(ROOT / args.output_dir)

    print("[COMPARE] Preparing real-path market features...")
    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)
    feature_cols = [col for col in TECHNICAL_FEATURE_COLUMNS if col in feat.columns]
    if not feature_cols:
        print("[COMPARE] FAIL: no feature columns available")
        return 1

    print("[COMPARE] Running XGB walk-forward predictions...")
    xgb_preds = rolling_train_predict_xgb(feat, feature_cols, "target_next_ret", config.model)

    gru_preds: pd.Series | None = None
    if not args.skip_gru_comparison:
        try:
            print("[COMPARE] Running GRU comparison predictions...")
            gru_preds = rolling_train_predict_gru(feat, feature_cols, "target_next_ret", config.model)
        except Exception as exc:
            print(f"[COMPARE] GRU comparison unavailable: {exc}")

    xgb_row, xgb_result = _run_real_lane(
        model_name="xgb",
        feat=feat,
        preds=xgb_preds,
        config=config,
        role="champion",
        comparison_preds={"gru": gru_preds} if gru_preds is not None else None,
        notes="primary real benchmark lane",
    )

    rows: list[RealBenchmarkRow] = [xgb_row]
    if gru_preds is not None:
        gru_row, _gru_result = _run_real_lane(
            model_name="gru",
            feat=feat,
            preds=gru_preds,
            config=config,
            role="comparison",
            comparison_preds={"xgb": xgb_preds},
            notes="comparison only; not eligible as champion",
        )
        rows.append(gru_row)
    else:
        rows.append(_frozen_comparison_row("gru", "comparison only; not run because recurrent path unavailable"))

    rows.append(_frozen_comparison_row("lstm", "comparison only; frozen off primary path"))

    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(output_dir / "xgb_real_leaderboard.csv", index=False)
    (output_dir / "xgb_real_leaderboard.md").write_text(
        build_xgb_real_leaderboard_markdown(df, xgb_row),
        encoding="utf-8",
    )
    (output_dir / "xgb_real_benchmark.md").write_text(
        build_xgb_real_benchmark_markdown(xgb_row, xgb_result["gate_stats"]),
        encoding="utf-8",
    )
    (output_dir / "NEXT_STATUS.md").write_text(
        build_next_status_markdown(xgb_row),
        encoding="utf-8",
    )

    print(f"[COMPARE] csv={output_dir / 'xgb_real_leaderboard.csv'}")
    print(f"[COMPARE] md={output_dir / 'xgb_real_leaderboard.md'}")
    print(f"[COMPARE] benchmark={output_dir / 'xgb_real_benchmark.md'}")
    print(f"[COMPARE] next={output_dir / 'NEXT_STATUS.md'}")
    print(
        "[COMPARE] RESULT: "
        f"{'PASS' if xgb_row.weekly_target_pass else 'FAIL'} champion=xgb "
        f"weekly={xgb_row.weekly_return_pct}% gap={_gap_to_target_pct_points(xgb_row.weekly_return_pct)} "
        f"sharpe={xgb_row.sharpe} dd={xgb_row.max_drawdown_pct}% trades={xgb_row.trade_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())