"""Exit-rule improvement experiment (validation split only).

Implements:
- ATR/volatility-scaled stop and take-profit exits.
- Fixed and volatility-adjusted time-stop variants.
- Optional trailing stop activated after profit threshold.
- Horizon-alignment check (model horizon vs time_stop_bars).

Outputs:
  reports/exit_rule_experiment.csv
  reports/exit_rule_experiment.md
"""
from __future__ import annotations

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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scalar(row, col: str) -> float:
    value = row[col]
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


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
) -> ExitExperimentResult:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.exit_rules import ExitRuleConfig, build_exit_levels, evaluate_dynamic_exit
    from marketify.broker.paper import PaperBroker
    from marketify.config import AppConfig
    from marketify.features.sentiment import FinBERTSentiment
    from marketify.features.technical import TECHNICAL_FEATURE_COLUMNS
    from marketify.models.ensemble import TradeIdeaInput, build_trade_idea, compute_cvar_95
    from marketify.models.xgb_model import rolling_train_predict
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

    # Keep train_window valid for current split, but never include final test slice.
    if len(train_val) <= config.model.train_window + 10:
        config.model.train_window = max(200, min(config.model.train_window, len(train_val) - 20))
        config.model.retrain_every = max(50, min(config.model.retrain_every, max(60, len(train_val) // 8)))

    preds = rolling_train_predict(
        frame=train_val,
        feature_cols=TECHNICAL_FEATURE_COLUMNS,
        target_col="target_next_ret",
        config=config.model,
    )

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

    equity_points: list[tuple[pd.Timestamp, float]] = []
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
                    }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], name="equity")
    history = [{"ts": str(ts), "equity": float(eq)} for ts, eq in zip(equity.index, equity.values)]
    bm = compute_benchmark(history, weekly_goal=0.01)

    diag_df = pd.DataFrame(trade_diagnostics)
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

    return ExitExperimentResult(
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


def build_exit_rule_experiment_markdown(df: pd.DataFrame, split_meta: ValidationSplitMeta) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    accepted = df[~df["rejected"]].copy() if "rejected" in df.columns else pd.DataFrame()

    lines = [
        "# Exit Rule Experiment (Validation Only)",
        f"**Generated:** {generated}",
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
            f"- Total configs: {len(df)}",
            f"- Accepted configs: {len(accepted)}",
            f"- Rejected configs: {len(df) - len(accepted)}",
            "",
        ]
    )

    if len(accepted) > 0:
        best = accepted.sort_values(["weekly_return_pct", "sharpe"], ascending=[False, False]).iloc[0]
        target_met = float(best["weekly_return_pct"]) >= 1.0

        lines.extend(
            [
                "## Best Validation Config",
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
                f"- 1% weekly target: {'PASS' if target_met else f'FAIL ({best['weekly_return_pct']}%)'}",
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
        for _, row in accepted.sort_values(["weekly_return_pct", "sharpe"], ascending=[False, False]).head(10).iterrows():
            lines.append(
                f"| {row['k_stop']} | {row['k_take']} | {row['time_stop_mode']} | {int(row['time_stop_bars'])} "
                f"| {bool(row['trailing_enabled'])} | {row['weekly_return_pct']} | {row['sharpe']} "
                f"| {row['max_drawdown_pct']} | {int(row['trade_count'])} | {row['profit_factor']} | {row['expectancy']} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Best Validation Config",
                "No accepted config. All candidates rejected by safety constraints.",
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
            "Past performance does not predict future results. Trading has risk.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import add_technical_features

    _ensure_dir(REPORTS)

    print("[EXIT-EXPERIMENT] Preparing market features...")
    raw = fetch_market_data(ticker="AAPL", interval="5m", period="60d", prepost=False)
    feat = add_technical_features(raw)

    train_val, validation_index, split_meta = split_for_validation_experiment(feat)

    keys = list(EXIT_GRID.keys())
    combos = list(itertools.product(*[EXIT_GRID[k] for k in keys]))
    total = len(combos)
    print(f"[EXIT-EXPERIMENT] {total} configs on validation split only")

    results: list[ExitExperimentResult] = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(
            f"  [{i}/{total}] k_stop={params['k_stop']} k_take={params['k_take']} "
            f"mode={params['time_stop_mode']} bars={params['time_stop_bars']} "
            f"trail={params['trailing_enabled']}",
            end="",
        )
        one = _run_single_config(train_val, validation_index, split_meta, params)
        tag = "REJECTED" if one.rejected else f"weekly={one.weekly_return_pct}% sharpe={one.sharpe}"
        print(f" -> {tag}")
        results.append(one)

    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = REPORTS / "exit_rule_experiment.csv"
    md_path = REPORTS / "exit_rule_experiment.md"
    df.to_csv(csv_path, index=False)

    md = build_exit_rule_experiment_markdown(df, split_meta)
    md_path.write_text(md)

    print(f"\n[EXIT-EXPERIMENT] csv={csv_path}")
    print(f"[EXIT-EXPERIMENT] md={md_path}")

    accepted = df[~df["rejected"]]
    if len(accepted) == 0:
        print("[EXIT-EXPERIMENT] RESULT: FAIL (no accepted config)")
        return 0

    best = accepted.sort_values(["weekly_return_pct", "sharpe"], ascending=[False, False]).iloc[0]
    meets_target = float(best["weekly_return_pct"]) >= 1.0
    verdict = "PASS" if meets_target else "FAIL"
    print(
        "[EXIT-EXPERIMENT] RESULT: "
        f"{verdict} weekly={best['weekly_return_pct']}% sharpe={best['sharpe']} "
        f"dd={best['max_drawdown_pct']}% trades={int(best['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
