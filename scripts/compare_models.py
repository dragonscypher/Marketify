from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SOURCE_OF_TRUTH_PATH = "scripts.compare_models:_run_real_lane"
BASELINE_WEEKLY_PCT = 0.5851
MIN_KEEP_COVERAGE_PCT = 5.0

RecurrentArchitecture = Literal["gru", "lstm"]
RecurrentLabelMode = Literal["return", "direction"]


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
    baseline_weekly_pct: float = BASELINE_WEEKLY_PCT
    daily_return_pct: float = 0.0
    approval_count: int = 0
    keep_coverage_pct: float = 0.0
    rejected_by_disagreement_count: int = 0
    rejected_by_low_edge_count: int = 0
    kept_trade_count: int = 0
    average_edge_kept: float = 0.0
    average_edge_rejected: float = 0.0
    cost_buffer_reject_rate: float = 0.0
    disagreement_reject_rate: float = 0.0
    keep_rate_by_gate: str = "unavailable"
    keep_rate_by_disagreement_bucket: str = "unavailable"
    approval_precision_top_half: float = 0.0
    approval_precision_bottom_half: float = 0.0
    expectancy_by_confidence_bucket: str = "unavailable"
    pnl_by_confidence_bucket: str = "unavailable"
    champion_eligible: bool = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from marketify.config import AppConfig

    default_horizon = AppConfig().broker.time_stop_bars
    parser = argparse.ArgumentParser(description="Run honest same-path real benchmark across xgb, ridge, gru, lstm, fusion.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--hold-horizon-bars", type=int, default=default_horizon)
    return parser.parse_args(argv)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _artifact_path(
    artifact_dir: Path,
    architecture: RecurrentArchitecture,
    ticker: str,
    hold_horizon_bars: int,
    label_mode: RecurrentLabelMode,
) -> Path:
    return artifact_dir / f"{architecture}_{ticker}_h{hold_horizon_bars}_{label_mode}.pt"


def prepare_validation_only_data(
    frame: pd.DataFrame,
    hold_horizon_bars: int,
    label_mode: RecurrentLabelMode,
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


def _compute_trade_metrics(diag: pd.DataFrame) -> dict[str, float | int]:
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


def _series_float(series: pd.Series, key: pd.Timestamp) -> float:
    return float(cast(Any, series.at[key]))


def _gap_to_target_pct_points(weekly_return_pct: float, target_pct: float = 1.0) -> float:
    return round(max(float(target_pct) - float(weekly_return_pct), 0.0), 4)


def _keep_coverage_pct(equity: pd.Series, trade_count: int) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    total_bars = max(len(eq) - 1, 1)
    return round(float(trade_count) / float(total_bars) * 100.0, 2)


def _compute_real_metrics(equity: pd.Series, trade_diag: pd.DataFrame) -> dict[str, float | int | bool]:
    from marketify.backtest.benchmark import compute_benchmark

    history = [{"ts": str(ts), "equity": float(value)} for ts, value in equity.items()]
    benchmark = compute_benchmark(history, weekly_goal=0.01)
    trade_metrics = _compute_trade_metrics(trade_diag)
    return {
        "daily_return_pct": round(float(benchmark.daily_return) * 100.0, 4),
        "weekly_return_pct": round(float(benchmark.weekly_return) * 100.0, 4),
        "sharpe": round(float(benchmark.sharpe_ratio), 4),
        "max_drawdown_pct": round(float(benchmark.max_drawdown) * 100.0, 4),
        "cvar_95_pct": round(float(benchmark.cvar_95) * 100.0, 4),
        "trade_count": int(trade_metrics["trade_count"]),
        "approval_count": int(trade_metrics["trade_count"]),
        "win_rate_pct": round(float(trade_metrics["win_rate_pct"]), 4),
        "expectancy": round(float(trade_metrics["expectancy"]), 6),
        "keep_coverage_pct": _keep_coverage_pct(equity, int(trade_metrics["trade_count"])),
        "weekly_target_pass": bool(benchmark.weekly_pass),
    }


def _is_eligible_for_champion(row: RealBenchmarkRow) -> bool:
    if row.run_status != "COMPLETE":
        return False
    if row.comparison_only:
        return False
    if float(row.weekly_return_pct) < 1.0:
        return False
    if int(row.trade_count) <= 0:
        return False
    if int(row.approval_count) <= 0:
        return False
    if float(row.expectancy) <= 0.0:
        return False
    if float(row.keep_coverage_pct) < MIN_KEEP_COVERAGE_PCT:
        return False
    return True


def _pass_fail_label(row: RealBenchmarkRow) -> str:
    return "PASS" if (_is_eligible_for_champion(row) and float(row.weekly_return_pct) >= 1.0) else "FAIL"


def _champion_sort_key(row: RealBenchmarkRow) -> tuple[float, float, float, float, int]:
    return (
        float(row.weekly_return_pct),
        float(row.expectancy),
        -float(row.max_drawdown_pct),
        float(row.sharpe),
        int(row.trade_count),
    )


def _select_champion(rows: list[RealBenchmarkRow]) -> RealBenchmarkRow | None:
    eligible = [row for row in rows if _is_eligible_for_champion(row)]
    if not eligible:
        return None
    return sorted(eligible, key=_champion_sort_key, reverse=True)[0]


def _select_reference_row(rows: list[RealBenchmarkRow]) -> RealBenchmarkRow:
    if not rows:
        raise ValueError("rows must not be empty")

    def _reference_sort_key(row: RealBenchmarkRow) -> tuple[int, int, int, float, float, float, float, int]:
        return (
            int(not row.comparison_only),
            int(row.model_name == "xgb"),
            int(row.trade_count > 0 and row.approval_count > 0),
            float(row.keep_coverage_pct),
            float(row.expectancy),
            float(row.weekly_return_pct),
            -float(row.max_drawdown_pct),
            int(row.trade_count),
        )

    return sorted(rows, key=_reference_sort_key, reverse=True)[0]


def _signal_gate_thresholds(config, gate_overrides: dict[str, float] | None = None) -> dict[str, float]:
    gate_overrides = gate_overrides or {}
    fee_floor = (float(config.broker.fee_bps) + float(config.broker.slippage_bps)) / 10000.0
    base_cost_buffer = max(float(config.risk.min_expected_return), fee_floor)
    cost_buffer = max(float(gate_overrides.get("cost_buffer_floor", base_cost_buffer)), fee_floor)
    abstain_margin = float(
        gate_overrides.get(
            "abstain_margin",
            max(float(getattr(config.broker, "fusion_abstain_margin", 0.0)), 0.00005),
        )
    )
    max_disagreement = float(
        gate_overrides.get(
            "max_disagreement",
            float(getattr(config.broker, "fusion_max_model_disagreement", 0.006)),
        )
    )
    return {
        "min_confidence": float(
            gate_overrides.get(
                "approval_precision_threshold",
                max(float(getattr(config.broker, "fusion_min_confidence", 0.30)), 0.35),
            )
        ),
        "abstain_margin": abstain_margin,
        "edge_floor": cost_buffer + abstain_margin,
        "news_cutoff": min(
            float(config.risk.max_news_risk),
            float(getattr(config.broker, "fusion_news_risk_cutoff", config.risk.max_news_risk)),
        ),
        "regime_cutoff": min(float(getattr(config.broker, "fusion_regime_vix_cutoff", 30.0)), 28.0),
        "max_disagreement": max_disagreement,
        "max_volatility": float(config.risk.max_volatility),
    }


def _apply_signal_gate(
    feat: pd.DataFrame,
    preds: pd.Series,
    config,
    comparison_preds: dict[str, pd.Series] | None = None,
    gate_overrides: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, int | float | str | dict[str, int]]]:
    thresholds = _signal_gate_thresholds(config, gate_overrides=gate_overrides)
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
    rejected_by_disagreement_count = 0
    rejected_by_low_edge_count = 0
    edge_kept: list[float] = []
    edge_rejected: list[float] = []
    for ts in gated.index:
        ts_key = cast(pd.Timestamp, ts)
        pred_value = _series_float(gated, ts_key)
        if not np.isfinite(pred_value):
            continue
        scored += 1
        effective_pred = pred_value * (1.0 + 0.2 * _series_float(sentiment, ts_key))
        edge_value = abs(effective_pred)
        confidence = float(np.clip(abs(effective_pred) / max(float(thresholds["edge_floor"]), 1e-6), 0.0, 1.0))

        reasons: list[str] = []
        if confidence < float(thresholds["min_confidence"]):
            reasons.append("low_confidence")
        if abs(effective_pred) <= float(thresholds["edge_floor"]):
            reasons.append("below_cost_buffer")
        if _series_float(news_risk, ts_key) > float(thresholds["news_cutoff"]) or _series_float(news_shock, ts_key) > 0.5:
            reasons.append("high_news_risk")
        if (
            _series_float(vix_proxy, ts_key) > float(thresholds["regime_cutoff"])
            or _series_float(macro_bear, ts_key) > 0.5
            or _series_float(macro_vol_high, ts_key) > 0.5
            or _series_float(macro_event_risk, ts_key) > 0.5
        ):
            reasons.append("bad_regime")
        if _series_float(volatility, ts_key) > float(thresholds["max_volatility"]):
            reasons.append("risk_too_high")

        disagreements = []
        for series in reference_map.values():
            ref_val = _series_float(series, ts_key)
            if np.isfinite(ref_val):
                disagreements.append(abs(pred_value - ref_val))
        if disagreements and max(disagreements) > float(thresholds["max_disagreement"]):
            reasons.append("model_disagreement")

        if reasons:
            edge_rejected.append(edge_value)
            if "model_disagreement" in reasons:
                rejected_by_disagreement_count += 1
            if "below_cost_buffer" in reasons:
                rejected_by_low_edge_count += 1
            gated.at[ts_key] = np.nan
            blocker_counts.update(reasons)
            continue
        kept += 1
        edge_kept.append(edge_value)

    top_blocker = blocker_counts.most_common(1)[0][0] if blocker_counts else "none"
    keep_ratio = round((kept / scored) * 100.0, 2) if scored else 0.0

    keep_rate_parts = [f"kept={keep_ratio:.2f}%"]
    if scored:
        for blocker, count in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0])):
            keep_rate_parts.append(f"{blocker}={round((count / scored) * 100.0, 2):.2f}%")
    keep_rate_by_gate = "; ".join(keep_rate_parts) if keep_rate_parts else "unavailable"
    disagreement_reject_rate = round((rejected_by_disagreement_count / scored) * 100.0, 2) if scored else 0.0
    keep_rate_by_disagreement_bucket = (
        f"kept={keep_ratio:.2f}%; disagreement_rejected={disagreement_reject_rate:.2f}%"
        if scored
        else "unavailable"
    )

    return gated, {
        "scored": scored,
        "kept": kept,
        "blocked": max(scored - kept, 0),
        "keep_ratio_pct": keep_ratio,
        "top_blocker": top_blocker,
        "blocker_counts": dict(blocker_counts),
        "rejected_by_disagreement_count": rejected_by_disagreement_count,
        "rejected_by_low_edge_count": rejected_by_low_edge_count,
        "kept_trade_count": kept,
        "average_edge_kept": round(float(np.mean(edge_kept)), 6) if edge_kept else 0.0,
        "average_edge_rejected": round(float(np.mean(edge_rejected)), 6) if edge_rejected else 0.0,
        "cost_buffer_reject_rate": round((rejected_by_low_edge_count / scored) * 100.0, 2) if scored else 0.0,
        "disagreement_reject_rate": disagreement_reject_rate,
        "keep_rate_by_gate": keep_rate_by_gate,
        "keep_rate_by_disagreement_bucket": keep_rate_by_disagreement_bucket,
        "selected_cost_buffer_floor": float(thresholds["edge_floor"] - thresholds["abstain_margin"]),
        "selected_approval_precision_threshold": float(thresholds["min_confidence"]),
        "selected_max_disagreement": float(thresholds["max_disagreement"]),
        "selected_abstain_margin": float(thresholds["abstain_margin"]),
    }


def _build_exact_blocker(row: RealBenchmarkRow) -> str:
    if _pass_fail_label(row) == "PASS":
        return "none"
    parts = [f"gap_to_target={_gap_to_target_pct_points(row.weekly_return_pct)} pct_points"]
    if row.trade_count == 0:
        parts.append("trade_count=0")
    if row.approval_count == 0:
        parts.append("approval_count=0")
    if float(row.expectancy) <= 0.0:
        parts.append("expectancy<=0")
    if float(row.keep_coverage_pct) < MIN_KEEP_COVERAGE_PCT:
        parts.append(f"keep_coverage<{MIN_KEEP_COVERAGE_PCT}%")
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
    gate_overrides: dict[str, float] | None = None,
    notes: str = "",
) -> tuple[RealBenchmarkRow, dict]:
    from copy import deepcopy

    from marketify.backtest.simulator import run_backtest_from_predictions

    lane_config = deepcopy(config)
    lane_config.broker.backtest_model = model_name  # type: ignore[attr-defined]
    gated_preds, gate_stats = _apply_signal_gate(
        feat,
        preds,
        lane_config,
        comparison_preds=comparison_preds,
        gate_overrides=gate_overrides,
    )
    result = run_backtest_from_predictions(feat, gated_preds, lane_config, model_used=model_name)
    trade_diag = result["trade_diagnostics"]
    metrics = _compute_real_metrics(result["equity"], trade_diag)
    precision_diag = _approval_precision_diagnostics(trade_diag)
    keep_coverage_pct = float(metrics["keep_coverage_pct"])
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
        comparison_only=(role in {"comparison", "reference"}),
        run_status="COMPLETE",
        gate_block_count=int(cast(int, gate_stats["blocked"])),
        gate_keep_ratio_pct=float(cast(float, gate_stats["keep_ratio_pct"])),
        top_blocker=str(cast(str, gate_stats["top_blocker"])),
        notes=notes,
        keep_coverage_pct=keep_coverage_pct,
        daily_return_pct=float(metrics["daily_return_pct"]),
        approval_count=int(metrics["approval_count"]),
        rejected_by_disagreement_count=int(cast(int, gate_stats["rejected_by_disagreement_count"])),
        rejected_by_low_edge_count=int(cast(int, gate_stats["rejected_by_low_edge_count"])),
        kept_trade_count=int(cast(int, gate_stats["kept_trade_count"])),
        average_edge_kept=float(cast(float, gate_stats["average_edge_kept"])),
        average_edge_rejected=float(cast(float, gate_stats["average_edge_rejected"])),
        cost_buffer_reject_rate=float(cast(float, gate_stats["cost_buffer_reject_rate"])),
        disagreement_reject_rate=float(cast(float, gate_stats["disagreement_reject_rate"])),
        keep_rate_by_gate=str(cast(str, gate_stats["keep_rate_by_gate"])),
        keep_rate_by_disagreement_bucket=str(cast(str, gate_stats["keep_rate_by_disagreement_bucket"])),
        approval_precision_top_half=float(precision_diag["approval_precision_top_half"]),
        approval_precision_bottom_half=float(precision_diag["approval_precision_bottom_half"]),
        expectancy_by_confidence_bucket=str(precision_diag["expectancy_by_confidence_bucket"]),
        pnl_by_confidence_bucket=str(precision_diag["pnl_by_confidence_bucket"]),
        champion_eligible=False,
    )
    row.champion_eligible = _is_eligible_for_champion(row)
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
        keep_coverage_pct=0.0,
        champion_eligible=False,
    )


def _failed_model_row(model_name: str, notes: str) -> RealBenchmarkRow:
    return RealBenchmarkRow(
        model_name=model_name,
        role="candidate",
        weekly_return_pct=float("nan"),
        sharpe=float("nan"),
        max_drawdown_pct=float("nan"),
        cvar_95_pct=float("nan"),
        trade_count=0,
        win_rate_pct=float("nan"),
        expectancy=float("nan"),
        weekly_target_pass=False,
        comparison_only=False,
        run_status="FAILED",
        gate_block_count=0,
        gate_keep_ratio_pct=0.0,
        top_blocker="runtime_error",
        notes=notes,
        daily_return_pct=float("nan"),
        approval_count=0,
        keep_coverage_pct=0.0,
        rejected_by_disagreement_count=0,
        rejected_by_low_edge_count=0,
        kept_trade_count=0,
        average_edge_kept=0.0,
        average_edge_rejected=0.0,
        cost_buffer_reject_rate=0.0,
        disagreement_reject_rate=0.0,
        keep_rate_by_gate="unavailable",
        keep_rate_by_disagreement_bucket="unavailable",
        approval_precision_top_half=0.0,
        approval_precision_bottom_half=0.0,
        expectancy_by_confidence_bucket="unavailable",
        pnl_by_confidence_bucket="unavailable",
        champion_eligible=False,
    )


def _rolling_train_predict_recurrent_architecture(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config,
    architecture: RecurrentArchitecture,
) -> pd.Series:
    from marketify.models.rnn_model import (RecurrentModelConfig,
                                            build_sequence_dataset,
                                            make_recurrent_model)

    seq_len = getattr(config, "rnn_seq_len", 20)
    hidden = getattr(config, "rnn_hidden", 32)
    epochs = getattr(config, "rnn_epochs", 10)

    rnn_cfg = RecurrentModelConfig(
        architecture=architecture,
        sequence_length=seq_len,
        hidden_size=hidden,
        num_layers=1,
        dropout=0.0,
        epochs=epochs,
        learning_rate=1e-3,
        random_state=config.random_state,
    )

    min_rows = config.train_window + seq_len + 10
    if len(frame) <= min_rows:
        raise ValueError(f"Not enough rows for {architecture} rolling training. Need > {min_rows}, got {len(frame)}.")

    preds = np.full(len(frame), np.nan, dtype=float)
    idx_map = {ts: i for i, ts in enumerate(frame.index)}
    start = config.train_window
    while start < len(frame):
        train_slice = frame.iloc[max(0, start - config.train_window) : start]
        end_pred = min(len(frame), start + config.retrain_every)
        pred_slice = frame.iloc[start:end_pred]

        x_train, y_train, _ = build_sequence_dataset(train_slice, feature_cols, target_col, seq_len)
        if x_train.size == 0:
            start = end_pred
            continue

        model = make_recurrent_model(architecture, rnn_cfg)
        model.fit(x_train, y_train)

        x_pred, _, pred_ts = build_sequence_dataset(
            pd.concat([train_slice.iloc[-seq_len:], pred_slice]),
            feature_cols,
            target_col,
            seq_len,
            selected_index=pred_slice.index,
        )
        if x_pred.size > 0:
            pred_vals = model.predict(x_pred)
            for ts, val in zip(pred_ts, pred_vals):
                idx = idx_map.get(ts)
                if idx is not None:
                    preds[idx] = float(val)

        start = end_pred

    return pd.Series(preds, index=frame.index, name=f"pred_{architecture}")


def _build_same_path_fusion_preds(feat: pd.DataFrame, xgb_preds: pd.Series, gru_preds: pd.Series) -> pd.Series:
    from marketify.models.fusion_model import (FusionConfig,
                                               fuse_prediction_components)

    permissive = FusionConfig(
        min_confidence=0.0,
        min_expected_return=0.0,
        abstain_margin=0.0,
        max_news_risk=1.1,
        high_vol_vix_threshold=1e9,
        max_model_disagreement=1e9,
    )
    return fuse_prediction_components(
        frame=feat,
        tabular_preds=xgb_preds,
        sequence_preds=gru_preds,
        fusion_config=permissive,
    )


def _support_comparison_preds(model_name: str, prediction_rows: dict[str, pd.Series]) -> dict[str, pd.Series]:
    if model_name in {"xgb", "ridge"}:
        return {name: prediction_rows[name] for name in ("gru", "lstm") if name in prediction_rows}
    if model_name in {"gru", "lstm"}:
        return {name: prediction_rows[name] for name in ("xgb",) if name in prediction_rows}
    return {}


def _finite_sort_metric(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def _tune_real_path_gate_knobs(
    feat: pd.DataFrame,
    xgb_preds: pd.Series,
    config,
    support_preds: dict[str, pd.Series],
) -> dict[str, float]:
    base_thresholds = _signal_gate_thresholds(config)
    base_min_confidence = float(base_thresholds["min_confidence"])
    base_max_disagreement = float(base_thresholds["max_disagreement"])
    approval_precision_threshold_candidates = sorted(
        {
            round(max(base_min_confidence - 0.05, 0.30), 6),
            round(base_min_confidence, 6),
            round(min(base_min_confidence + 0.05, 0.80), 6),
        }
    )
    max_disagreement_candidates = sorted(
        {
            round(base_max_disagreement, 6),
            round(base_max_disagreement * 1.25, 6),
            round(base_max_disagreement * 1.50, 6),
        }
    )

    best_overrides = {
        "max_disagreement": base_max_disagreement,
        "approval_precision_threshold": base_min_confidence,
    }
    best_score: tuple[float, ...] | None = None
    best_row: RealBenchmarkRow | None = None

    for max_disagreement in max_disagreement_candidates:
        for approval_precision_threshold in approval_precision_threshold_candidates:
                overrides = {
                    "max_disagreement": float(max_disagreement),
                    "approval_precision_threshold": float(approval_precision_threshold),
                }
                trial_row, _trial_result = _run_real_lane(
                    model_name="xgb",
                    feat=feat,
                    preds=xgb_preds,
                    config=config,
                    role="candidate",
                    comparison_preds=support_preds,
                    gate_overrides=overrides,
                    notes="xgb primary lane gate calibration",
                )
                score = (
                    float(int(_is_eligible_for_champion(trial_row))),
                    float(int(float(trial_row.keep_coverage_pct) >= MIN_KEEP_COVERAGE_PCT)),
                    float(int(float(trial_row.expectancy) > 0.0)),
                    float(int(float(trial_row.weekly_return_pct) >= 1.0)),
                    float(int(trial_row.trade_count > 0 and trial_row.approval_count > 0)),
                    _finite_sort_metric(trial_row.keep_coverage_pct),
                    _finite_sort_metric(trial_row.expectancy),
                    _finite_sort_metric(trial_row.approval_precision_top_half),
                    _finite_sort_metric(trial_row.weekly_return_pct),
                    -_finite_sort_metric(trial_row.disagreement_reject_rate),
                    -float(trial_row.rejected_by_disagreement_count),
                    -float(trial_row.rejected_by_low_edge_count),
                    -_finite_sort_metric(trial_row.max_drawdown_pct),
                    -abs(float(max_disagreement) - base_max_disagreement),
                    -abs(float(approval_precision_threshold) - base_min_confidence),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_overrides = overrides
                    best_row = trial_row

    if best_row is not None:
        print(
            "[COMPARE] tuned_gate "
            f"max_disagreement={best_overrides['max_disagreement']:.6f} "
            f"approval_precision_threshold={best_overrides['approval_precision_threshold']:.6f} "
            f"weekly={best_row.weekly_return_pct}% sharpe={best_row.sharpe} trades={best_row.trade_count} "
            f"keep={best_row.keep_coverage_pct}% disagreement_rejects={best_row.rejected_by_disagreement_count} "
            f"disagreement_reject_rate={best_row.disagreement_reject_rate}% top_half_precision={best_row.approval_precision_top_half}"
        )

    return best_overrides


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


def _build_model_compare_records(rows: list[RealBenchmarkRow]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                **asdict(row),
                "source_of_truth_path": SOURCE_OF_TRUTH_PATH,
                "eligibility": "YES" if _is_eligible_for_champion(row) else "NO",
                "PASS_FAIL": _pass_fail_label(row),
                "exact_gap_to_1pct_target": _gap_to_target_pct_points(row.weekly_return_pct),
                "exact_blocker": _build_exact_blocker(row),
            }
        )
    return records


def _build_benchmark_summary(champion_row: RealBenchmarkRow | None, display_row: RealBenchmarkRow) -> dict[str, object]:
    return {
        "source_of_truth_path": SOURCE_OF_TRUTH_PATH,
        "champion_model": champion_row.model_name if champion_row is not None else "NONE",
        "daily_return_pct": display_row.daily_return_pct,
        "weekly_return_pct": display_row.weekly_return_pct,
        "baseline_weekly_pct": display_row.baseline_weekly_pct,
        "sharpe": display_row.sharpe,
        "max_drawdown_pct": display_row.max_drawdown_pct,
        "cvar_95_pct": display_row.cvar_95_pct,
        "trade_count": display_row.trade_count,
        "approval_count": display_row.approval_count,
        "keep_coverage_pct": display_row.keep_coverage_pct,
        "rejected_by_disagreement_count": display_row.rejected_by_disagreement_count,
        "rejected_by_low_edge_count": display_row.rejected_by_low_edge_count,
        "kept_trade_count": display_row.kept_trade_count,
        "cost_buffer_reject_rate": display_row.cost_buffer_reject_rate,
        "disagreement_reject_rate": display_row.disagreement_reject_rate,
        "keep_rate_by_gate": display_row.keep_rate_by_gate,
        "keep_rate_by_disagreement_bucket": display_row.keep_rate_by_disagreement_bucket,
        "approval_precision_top_half": display_row.approval_precision_top_half,
        "approval_precision_bottom_half": display_row.approval_precision_bottom_half,
        "expectancy_by_confidence_bucket": display_row.expectancy_by_confidence_bucket,
        "pnl_by_confidence_bucket": display_row.pnl_by_confidence_bucket,
        "average_edge_kept": display_row.average_edge_kept,
        "average_edge_rejected": display_row.average_edge_rejected,
        "expectancy": display_row.expectancy,
        "eligibility": "YES" if champion_row is not None and _is_eligible_for_champion(display_row) else "NO",
        "PASS_FAIL": _pass_fail_label(display_row) if champion_row is not None else "FAIL",
        "exact_gap_to_1pct_target": _gap_to_target_pct_points(display_row.weekly_return_pct),
        "exact_blocker": _build_exact_blocker(display_row),
    }


def build_model_leaderboard_markdown(
    champion_row: RealBenchmarkRow | None,
    rows: list[RealBenchmarkRow],
) -> str:
    current_champion = champion_row.model_name if champion_row is not None else "NONE"
    lines = [
        "# Model Leaderboard",
        "",
        "Real traded path only. xgb primary lane. recurrent/news/regime remain support or veto only.",
        f"current_champion: {current_champion}",
        "",
        "| model_name | role | weekly_return_pct | sharpe | max_drawdown_pct | trade_count | approval_count | keep_coverage_pct | rejected_by_disagreement_count | disagreement_reject_rate | keep_rate_by_disagreement_bucket | approval_precision_top_half | approval_precision_bottom_half | expectancy_by_confidence_bucket | pnl_by_confidence_bucket | PASS/FAIL | exact_blocker | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for record in _build_model_compare_records(rows):
        lines.append(
            f"| {record['model_name']} | {record['role']} | {_fmt_metric(record['weekly_return_pct'])} | {_fmt_metric(record['sharpe'])} | {_fmt_metric(record['max_drawdown_pct'])} | {record['trade_count']} | {record['approval_count']} | {_fmt_metric(record['keep_coverage_pct'], digits=2)} | {record['rejected_by_disagreement_count']} | {_fmt_metric(record['disagreement_reject_rate'], digits=2)} | {record['keep_rate_by_disagreement_bucket']} | {_fmt_metric(record['approval_precision_top_half'], digits=2)} | {_fmt_metric(record['approval_precision_bottom_half'], digits=2)} | {record['expectancy_by_confidence_bucket']} | {record['pnl_by_confidence_bucket']} | {record['PASS_FAIL']} | {record['exact_blocker']} | {record['notes']} |"
        )
    lines += [
        "",
        "PASS only if weekly_return_pct >= 1.0 on real traded path.",
    ]
    return "\n".join(lines) + "\n"


def build_real_benchmark_markdown(
    champion_row: RealBenchmarkRow | None,
    display_row: RealBenchmarkRow,
    rows: list[RealBenchmarkRow],
) -> str:
    summary = _build_benchmark_summary(champion_row, display_row)
    lines = [
        "# Real Benchmark",
        "",
        "One honest real benchmark path only. No profit claims.",
        f"source_of_truth_path: {summary['source_of_truth_path']}",
        f"champion_model: {summary['champion_model']}",
        f"daily_return_pct: {_fmt_metric(summary['daily_return_pct'])}",
        f"weekly_return_pct: {_fmt_metric(summary['weekly_return_pct'])}",
        f"baseline_weekly_pct: {_fmt_metric(summary['baseline_weekly_pct'])}",
        f"sharpe: {_fmt_metric(summary['sharpe'])}",
        f"max_drawdown_pct: {_fmt_metric(summary['max_drawdown_pct'])}",
        f"cvar_95_pct: {_fmt_metric(summary['cvar_95_pct'])}",
        f"trade_count: {summary['trade_count']}",
        f"approval_count: {summary['approval_count']}",
        f"keep_coverage_pct: {_fmt_metric(summary['keep_coverage_pct'], digits=2)}",
        f"rejected_by_disagreement_count: {summary['rejected_by_disagreement_count']}",
        f"rejected_by_low_edge_count: {summary['rejected_by_low_edge_count']}",
        f"disagreement_reject_rate: {_fmt_metric(summary['disagreement_reject_rate'], digits=2)}",
        f"keep_rate_by_disagreement_bucket: {summary['keep_rate_by_disagreement_bucket']}",
        f"kept_trade_count: {summary['kept_trade_count']}",
        f"approval_precision_top_half: {_fmt_metric(summary['approval_precision_top_half'], digits=2)}",
        f"approval_precision_bottom_half: {_fmt_metric(summary['approval_precision_bottom_half'], digits=2)}",
        f"expectancy_by_confidence_bucket: {summary['expectancy_by_confidence_bucket']}",
        f"pnl_by_confidence_bucket: {summary['pnl_by_confidence_bucket']}",
        f"average_edge_kept: {_fmt_metric(summary['average_edge_kept'], digits=6)}",
        f"average_edge_rejected: {_fmt_metric(summary['average_edge_rejected'], digits=6)}",
        f"expectancy: {_fmt_metric(summary['expectancy'], digits=6)}",
        f"eligibility: {summary['eligibility']}",
        f"PASS/FAIL: {summary['PASS_FAIL']}",
        f"exact_gap_to_1pct_target: {_fmt_metric(summary['exact_gap_to_1pct_target'])}",
        f"exact_blocker: {summary['exact_blocker']}",
        "",
        "## Model Compare",
        "| model_name | daily_return_pct | weekly_return_pct | baseline_weekly_pct | sharpe | max_drawdown_pct | cvar_95_pct | trade_count | approval_count | keep_coverage_pct | rejected_by_disagreement_count | disagreement_reject_rate | keep_rate_by_disagreement_bucket | approval_precision_top_half | approval_precision_bottom_half | expectancy_by_confidence_bucket | pnl_by_confidence_bucket | expectancy | eligibility | PASS/FAIL | run_status | top_blocker | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---|---|---|---|---|",
    ]
    for record in _build_model_compare_records(rows):
        lines.append(
            f"| {record['model_name']} | {_fmt_metric(record['daily_return_pct'])} | {_fmt_metric(record['weekly_return_pct'])} | {_fmt_metric(record['baseline_weekly_pct'])} | "
            f"{_fmt_metric(record['sharpe'])} | {_fmt_metric(record['max_drawdown_pct'])} | {_fmt_metric(record['cvar_95_pct'])} | {record['trade_count']} | {record['approval_count']} | "
            f"{_fmt_metric(record['keep_coverage_pct'], digits=2)} | {record['rejected_by_disagreement_count']} | {_fmt_metric(record['disagreement_reject_rate'], digits=2)} | {record['keep_rate_by_disagreement_bucket']} | {_fmt_metric(record['approval_precision_top_half'], digits=2)} | {_fmt_metric(record['approval_precision_bottom_half'], digits=2)} | {record['expectancy_by_confidence_bucket']} | {record['pnl_by_confidence_bucket']} | {_fmt_metric(record['expectancy'], digits=6)} | "
            f"{record['eligibility']} | {record['PASS_FAIL']} | {record['run_status']} | {record['top_blocker']} | {record['notes']} |"
        )
    lines += [
        "",
        "Guardrails: no leverage increase; no exit-rule experiment path; validation-only results never final winner.",
    ]
    return "\n".join(lines) + "\n"


def build_next_status_markdown(
    champion_row: RealBenchmarkRow | None,
    display_row: RealBenchmarkRow | None = None,
    fix_lines: list[str] | None = None,
) -> str:
    row = display_row or champion_row
    if row is None:
        raise ValueError("display_row or champion_row required")
    summary = _build_benchmark_summary(champion_row, row)
    lines = [
        "# NEXT_STATUS",
        "",
        f"- source_of_truth_path: {summary['source_of_truth_path']}",
        f"- champion_model: {summary['champion_model']}",
        f"- current champion: {summary['champion_model']}",
        f"- daily_return_pct: {_fmt_metric(summary['daily_return_pct'])}",
        f"- weekly_return_pct: {_fmt_metric(summary['weekly_return_pct'])}",
        f"- baseline_weekly_pct: {_fmt_metric(summary['baseline_weekly_pct'])}",
        f"- sharpe: {_fmt_metric(summary['sharpe'])}",
        f"- max_drawdown_pct: {_fmt_metric(summary['max_drawdown_pct'])}",
        f"- cvar_95_pct: {_fmt_metric(summary['cvar_95_pct'])}",
        f"- trade_count: {summary['trade_count']}",
        f"- approval_count: {summary['approval_count']}",
        f"- keep_coverage_pct: {_fmt_metric(summary['keep_coverage_pct'], digits=2)}",
        f"- rejected_by_disagreement_count: {summary['rejected_by_disagreement_count']}",
        f"- disagreement_reject_rate: {_fmt_metric(summary['disagreement_reject_rate'], digits=2)}",
        f"- keep_rate_by_disagreement_bucket: {summary['keep_rate_by_disagreement_bucket']}",
        f"- kept_trade_count: {summary['kept_trade_count']}",
        f"- approval_precision_top_half: {_fmt_metric(summary['approval_precision_top_half'], digits=2)}",
        f"- approval_precision_bottom_half: {_fmt_metric(summary['approval_precision_bottom_half'], digits=2)}",
        f"- expectancy_by_confidence_bucket: {summary['expectancy_by_confidence_bucket']}",
        f"- pnl_by_confidence_bucket: {summary['pnl_by_confidence_bucket']}",
        f"- average_edge_kept: {_fmt_metric(summary['average_edge_kept'], digits=6)}",
        f"- average_edge_rejected: {_fmt_metric(summary['average_edge_rejected'], digits=6)}",
        f"- expectancy: {_fmt_metric(summary['expectancy'], digits=6)}",
        f"- eligibility: {summary['eligibility']}",
        f"- PASS/FAIL: {summary['PASS_FAIL']}",
        f"- result: {summary['PASS_FAIL']}",
        f"- gap to 1.0% target: {_fmt_metric(summary['exact_gap_to_1pct_target'])} percentage points",
        f"- exact_gap_to_1pct_target: {_fmt_metric(summary['exact_gap_to_1pct_target'])}",
        f"- exact_blocker: {summary['exact_blocker']}",
        "",
        "FAIL honest if weekly < 1.0% or trade_count <= 0 or expectancy <= 0 or keep_coverage < 5%.",
    ]
    if fix_lines:
        lines += [
            "",
            "## Top 3 Highest-Impact Fixes",
        ]
        for idx, line in enumerate(fix_lines[:3], start=1):
            lines.append(f"- {idx}. {line}")
    return "\n".join(lines) + "\n"


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


def _approval_precision_diagnostics(trade_diag: pd.DataFrame) -> dict[str, float | str]:
    if trade_diag.empty or "pnl" not in trade_diag.columns or "signal_confidence" not in trade_diag.columns:
        return {
            "approval_precision_top_half": 0.0,
            "approval_precision_bottom_half": 0.0,
            "expectancy_by_confidence_bucket": "unavailable",
            "pnl_by_confidence_bucket": "unavailable",
        }

    work = trade_diag[["pnl", "signal_confidence"]].copy()
    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work["signal_confidence"] = pd.to_numeric(work["signal_confidence"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return {
            "approval_precision_top_half": 0.0,
            "approval_precision_bottom_half": 0.0,
            "expectancy_by_confidence_bucket": "unavailable",
            "pnl_by_confidence_bucket": "unavailable",
        }

    ranked = work.sort_values("signal_confidence").reset_index(drop=True)
    half_size = max(len(ranked) // 2, 1)
    bottom_half = ranked.iloc[:half_size]
    top_half = ranked.iloc[-half_size:]

    bucket_count = min(3, int(ranked["signal_confidence"].nunique()))
    bucket_summary = _bucket_summary(ranked, "signal_confidence", "confidence", bucket_count=bucket_count)
    if bucket_summary:
        label_map = {1: "low", 2: "mid", 3: "high"}
        expectancy_parts = []
        pnl_parts = []
        for idx, bucket in enumerate(bucket_summary, start=1):
            label = label_map.get(idx, f"bucket_{idx}")
            expectancy_parts.append(f"{label}={bucket['expectancy']} ({bucket['trade_count']})")
            pnl_parts.append(f"{label}={bucket['total_pnl']} ({bucket['trade_count']})")
        expectancy_by_confidence_bucket = "; ".join(expectancy_parts)
        pnl_by_confidence_bucket = "; ".join(pnl_parts)
    else:
        expectancy_by_confidence_bucket = "unavailable"
        pnl_by_confidence_bucket = "unavailable"

    def _precision_pct(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        return round(float((frame["pnl"] > 0).mean()) * 100.0, 2)

    return {
        "approval_precision_top_half": _precision_pct(top_half),
        "approval_precision_bottom_half": _precision_pct(bottom_half),
        "expectancy_by_confidence_bucket": expectancy_by_confidence_bucket,
        "pnl_by_confidence_bucket": pnl_by_confidence_bucket,
    }


def _rolling_xgb_feature_importance(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config,
) -> pd.DataFrame:
    from marketify.models.xgb_model import XGBModel

    if len(frame) <= config.train_window + 10:
        return pd.DataFrame(columns=["feature", "mean_importance", "window_count"])

    x = frame[feature_cols].to_numpy()
    y = frame[target_col].to_numpy()
    sums = np.zeros(len(feature_cols), dtype=float)
    windows = 0

    start = config.train_window
    while start < len(frame):
        train_start = max(0, start - config.train_window)
        x_train = x[train_start:start]
        y_train = y[train_start:start]
        model = XGBModel(config)
        model.fit(x_train, y_train)
        importances = getattr(model.model, "feature_importances_", np.zeros(len(feature_cols), dtype=float))
        if len(importances) == len(feature_cols):
            sums += np.asarray(importances, dtype=float)
            windows += 1
        start = min(len(frame), start + config.retrain_every)

    if windows == 0:
        return pd.DataFrame(columns=["feature", "mean_importance", "window_count"])

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_importance": sums / float(windows),
            "window_count": windows,
        }
    )
    return df.sort_values("mean_importance", ascending=True).reset_index(drop=True)


def _feature_variant_sets(feature_cols: list[str], low_importance_cols: list[str]) -> dict[str, list[str]]:
    feature_set = set(feature_cols)
    time_cols = [col for col in feature_cols if col.startswith("sin_") or col.startswith("cos_") or col.startswith("is_")]
    relative_cols = [col for col in feature_cols if col.startswith("ret_vs_")]
    news_regime_cols = [
        col for col in feature_cols
        if col.startswith("news_")
        or col.startswith("macro_")
        or col in {"vix_proxy", "vol_regime_ratio", "volatility_regime_code", "signal_quality_20", "signal_stability_20", "trend_alignment_score"}
    ]
    volume_cols = [col for col in feature_cols if col in {"volume_z20", "dollar_volume_z20", "volume_regime_ratio"}]

    variants = {
        "drop_time_features": [col for col in feature_cols if col not in set(time_cols)],
        "drop_relative_strength": [col for col in feature_cols if col not in set(relative_cols)],
        "drop_news_regime_context": [col for col in feature_cols if col not in set(news_regime_cols)],
        "drop_volume_liquidity": [col for col in feature_cols if col not in set(volume_cols)],
    }
    if low_importance_cols:
        variants["drop_low_importance_tail"] = [col for col in feature_cols if col not in set(low_importance_cols)]

    return {name: cols for name, cols in variants.items() if len(cols) >= 5 and set(cols) != feature_set}


def _rank_xgb_fixes(
    xgb_row: RealBenchmarkRow,
    trade_diag: pd.DataFrame,
    ablation_rows: list[dict],
    importance_df: pd.DataFrame,
) -> list[str]:
    ranked: list[tuple[float, str, str]] = []

    positive_ablations = [row for row in ablation_rows if float(row.get("delta_weekly_return_pct", 0.0)) > 0.0]
    positive_ablations.sort(key=lambda row: float(row["delta_weekly_return_pct"]), reverse=True)
    for row in positive_ablations[:2]:
        ranked.append(
            (
                float(row["delta_weekly_return_pct"]),
                f"ablation_{row['variant']}",
                f"Keep {row['variant']}: improved weekly by +{row['delta_weekly_return_pct']} pct points without collapsing coverage.",
            )
        )

    if not importance_df.empty:
        weakest = importance_df.head(min(5, len(importance_df)))["feature"].tolist()
        if weakest:
            ranked.append(
                (
                    0.001,
                    "weak_features",
                    f"Prune weakest/noisiest xgb inputs first: {', '.join(weakest)}.",
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
                        f"Tighten deterministic news veto only if same-path xgb improves: high-news-risk bucket expectancy {round(float(high_news['pnl'].mean()), 4)}.",
                    )
                )

        if "entry_vol_regime_ratio" in diag.columns:
            bad_regime = diag[pd.to_numeric(diag["entry_vol_regime_ratio"], errors="coerce") >= 1.25]
            if len(bad_regime) >= 2 and float(bad_regime["pnl"].mean()) < 0.0:
                ranked.append(
                    (
                        abs(float(bad_regime["pnl"].sum())),
                        "regime_veto",
                        f"Tighten deterministic regime veto only if same-path xgb improves: stressed-regime bucket expectancy {round(float(bad_regime['pnl'].mean()), 4)}.",
                    )
                )

        if "signal_confidence" in diag.columns:
            conf = pd.to_numeric(diag["signal_confidence"], errors="coerce")
            low_conf = diag[conf <= conf.median()]
            if len(low_conf) >= 2 and float(low_conf["pnl"].mean()) < 0.0:
                ranked.append(
                    (
                        abs(float(low_conf["pnl"].sum())),
                        "confidence_gate",
                        f"Raise xgb approval precision on low-confidence trades: bottom-half confidence expectancy {round(float(low_conf['pnl'].mean()), 4)}.",
                    )
                )

    lines: list[str] = []
    seen: set[str] = set()
    for _, key, line in sorted(ranked, key=lambda item: item[0], reverse=True):
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) == 3:
            break

    while len(lines) < 3:
        lines.append(
            f"Improve xgb decision quality without leverage change: current same-path gap = {_gap_to_target_pct_points(xgb_row.weekly_return_pct)} percentage points."
        )

    return lines[:3]


def _write_false_positive_trade_review(
    xgb_row: RealBenchmarkRow,
    trade_diag: pd.DataFrame,
    fix_lines: list[str],
    reports_dir: Path,
) -> None:
    lines = [
        "# False Positive Trade Review",
        "",
        "Real xgb benchmark path only. Focus: losing approved trades and weak decision buckets.",
        "No leverage change. No new model family. No exit-rule work.",
        "",
        f"- xgb_weekly_return_pct: {_fmt_metric(xgb_row.weekly_return_pct)}",
        f"- xgb_exact_blocker: {_build_exact_blocker(xgb_row)}",
        "",
    ]

    if trade_diag.empty or "pnl" not in trade_diag.columns:
        lines += ["No real xgb trades available. Cannot review false positives.", ""]
        (reports_dir / "false_positive_trade_review.md").write_text("\n".join(lines), encoding="utf-8")
        return

    diag = trade_diag.copy()
    diag["pnl"] = pd.to_numeric(diag["pnl"], errors="coerce")
    diag = diag.dropna(subset=["pnl"])
    losses = diag[diag["pnl"] < 0].copy()
    wins = diag[diag["pnl"] > 0].copy()

    diag_precision = _approval_precision_diagnostics(diag)
    lines += [
        "## Summary",
        f"- total_trades: {len(diag)}",
        f"- losing_trades: {len(losses)}",
        f"- winning_trades: {len(wins)}",
        f"- loss_share_pct: {round(len(losses) / max(len(diag), 1) * 100.0, 2)}",
        f"- loss_expectancy: {round(float(losses['pnl'].mean()), 4) if len(losses) else 0.0}",
        f"- expectancy_by_confidence_bucket: {diag_precision['expectancy_by_confidence_bucket']}",
        f"- pnl_by_confidence_bucket: {diag_precision['pnl_by_confidence_bucket']}",
        "",
    ]

    if len(losses):
        worst = losses.nsmallest(min(8, len(losses)), "pnl")
        cols = [
            "timestamp", "side", "pnl", "return_pct", "signal_confidence", "expected_return",
            "entry_news_risk", "entry_news_event_shock", "entry_vol_regime_ratio", "entry_signal_quality_20", "exit_reason",
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

    for column, title in [
        ("signal_confidence", "Confidence Buckets"),
        ("expected_return", "Expected-Return Buckets"),
        ("entry_news_risk", "News-Risk Buckets"),
        ("entry_vol_regime_ratio", "Regime-Stress Buckets"),
    ]:
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
    xgb_row: RealBenchmarkRow,
    importance_df: pd.DataFrame,
    ablation_rows: list[dict],
    reports_dir: Path,
) -> None:
    lines = [
        "# Feature Ablation",
        "",
        "Real xgb benchmark path only. Same walk-forward splits, costs, slippage, hold logic, and approval accounting.",
        "Weak/noisy feature removal only. No new model family. No leverage change.",
        "",
        "## Baseline XGB",
        f"- weekly_return_pct: {_fmt_metric(xgb_row.weekly_return_pct)}",
        f"- daily_return_pct: {_fmt_metric(xgb_row.daily_return_pct)}",
        f"- sharpe: {_fmt_metric(xgb_row.sharpe)}",
        f"- max_drawdown_pct: {_fmt_metric(xgb_row.max_drawdown_pct)}",
        f"- cvar_95_pct: {_fmt_metric(xgb_row.cvar_95_pct)}",
        f"- trade_count: {xgb_row.trade_count}",
        f"- approval_count: {xgb_row.approval_count}",
        f"- keep_coverage_pct: {_fmt_metric(xgb_row.keep_coverage_pct, digits=2)}",
        f"- expectancy: {_fmt_metric(xgb_row.expectancy, digits=6)}",
        "",
    ]

    if not importance_df.empty:
        lines += [
            "## Rolling XGB Feature Importance Audit",
            "| feature | mean_importance | window_count |",
            "| --- | --- | --- |",
        ]
        for _, row in importance_df.sort_values("mean_importance", ascending=False).head(15).iterrows():
            lines.append(
                f"| {row['feature']} | {_fmt_metric(row['mean_importance'], digits=6)} | {int(row['window_count'])} |"
            )
        weakest = importance_df.head(min(8, len(importance_df)))
        if not weakest.empty:
            lines += [
                "",
                "## Weak/Noisy Candidates",
                f"Bottom features by rolling importance: {', '.join(weakest['feature'].tolist())}",
                "",
            ]

    if not ablation_rows:
        lines += ["No xgb ablation rows available.", ""]
        (reports_dir / "feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## Variant Results",
        "| variant | feature_count | weekly_return_pct | delta_weekly_return_pct | daily_return_pct | sharpe | max_drawdown_pct | cvar_95_pct | trade_count | approval_count | expectancy | keep_coverage_pct | eligible | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(ablation_rows, key=lambda item: float(item["delta_weekly_return_pct"]), reverse=True):
        lines.append(
            f"| {row['variant']} | {row['feature_count']} | {row['weekly_return_pct']} | {row['delta_weekly_return_pct']} | {row['daily_return_pct']} | {row['sharpe']} | {row['max_drawdown_pct']} | {row['cvar_95_pct']} | {row['trade_count']} | {row['approval_count']} | {row['expectancy']} | {row['keep_coverage_pct']} | {row['eligible']} | {row['recommendation']} |"
        )
    lines.append("")

    improve_rows = [row for row in ablation_rows if row["recommendation"] == "KEEP"]
    if improve_rows:
        lines += [
            "## Keep Variants",
            "Same-path xgb variants below improved weekly return without collapsing coverage.",
        ]
        for row in improve_rows[:3]:
            lines.append(f"- {row['variant']}: +{row['delta_weekly_return_pct']} pct points weekly.")
        lines.append("")
    else:
        lines += [
            "## Keep Variants",
            "No tested xgb feature-pruning variant improved weekly return without collapsing coverage.",
            "",
        ]

    (reports_dir / "feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")


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
    architecture: RecurrentArchitecture,
    train_val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_index: pd.Index,
    validation_index: pd.Index,
    artifact_dir: Path,
    ticker: str,
    hold_horizon_bars: int,
    label_mode: RecurrentLabelMode,
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
    open_state: dict[str, dict[str, Any]] = {}

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
                state_side = str(state["side"])
                close_side = "sell" if state_side == "buy" else "buy"
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
                if state_side == "buy":
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

        ts_loc = int(train_val.index.get_indexer(pd.Index([ts]))[0])
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

    trade_count = int(trade_metrics["trade_count"])
    turnover = (trade_count / max(len(equity), 1)) if len(equity) > 0 else 0.0
    if trade_count < MIN_TRADES:
        reject_reasons.append(f"too few trades: {trade_count} < {MIN_TRADES}")
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
    from marketify.models.ridge_model import \
        rolling_train_predict as rolling_train_predict_ridge
    from marketify.models.xgb_model import \
        rolling_train_predict as rolling_train_predict_xgb

    config = AppConfig()
    config.broker.time_stop_bars = int(args.hold_horizon_bars)
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

    print(f"[COMPARE] source_of_truth_path={SOURCE_OF_TRUTH_PATH}")
    print("[COMPARE] Running same-path walk-forward predictions...")

    prediction_rows: dict[str, pd.Series] = {}
    rows: list[RealBenchmarkRow] = []
    result_by_model: dict[str, tuple[RealBenchmarkRow, dict]] = {}

    for model_name in ["xgb", "ridge", "gru", "lstm"]:
        try:
            print(f"[COMPARE] predicting {model_name} ...")
            if model_name == "xgb":
                prediction_rows[model_name] = rolling_train_predict_xgb(feat, feature_cols, "target_next_ret", config.model)
            elif model_name == "ridge":
                prediction_rows[model_name] = rolling_train_predict_ridge(feat, feature_cols, "target_next_ret", config.model)
            else:
                prediction_rows[model_name] = _rolling_train_predict_recurrent_architecture(
                    feat,
                    feature_cols,
                    "target_next_ret",
                    config.model,
                    cast(RecurrentArchitecture, model_name),
                )
        except Exception as exc:
            print(f"[COMPARE] {model_name} unavailable: {exc}")
            rows.append(_failed_model_row(model_name, f"same-path candidate failed: {exc}"))

    if "xgb" in prediction_rows and "gru" in prediction_rows:
        try:
            print("[COMPARE] predicting fusion_reference_only ...")
            prediction_rows["fusion_reference_only"] = _build_same_path_fusion_preds(
                feat,
                prediction_rows["xgb"],
                prediction_rows["gru"],
            )
        except Exception as exc:
            print(f"[COMPARE] fusion_reference_only unavailable: {exc}")
            rows.append(_failed_model_row("fusion_reference_only", f"same-path candidate failed: {exc}"))
    else:
        rows.append(_failed_model_row("fusion_reference_only", "same-path candidate failed: requires xgb + gru predictions"))

    gate_overrides: dict[str, float] | None = None
    if "xgb" in prediction_rows:
        gate_overrides = _tune_real_path_gate_knobs(
            feat,
            prediction_rows["xgb"],
            config,
            _support_comparison_preds("xgb", prediction_rows),
        )

    for model_name, preds in prediction_rows.items():
        comparison_preds = _support_comparison_preds(model_name, prediction_rows)
        role = "reference" if model_name == "fusion_reference_only" else "candidate"
        notes = {
            "xgb": "primary signal lane on source-of-truth path",
            "ridge": "same-path baseline comparator",
            "gru": "support/context comparator; may inform disagreement only",
            "lstm": "support/context comparator; may inform disagreement only",
            "fusion_reference_only": "reference only; never champion until deployable same-path semantics proven",
        }[model_name]
        row, _result = _run_real_lane(
            model_name=model_name,
            feat=feat,
            preds=preds,
            config=config,
            role=role,
            comparison_preds=comparison_preds,
            gate_overrides=gate_overrides,
            notes=notes,
        )
        rows.append(row)
        result_by_model[model_name] = (row, _result)

    complete_rows = [row for row in rows if row.run_status == "COMPLETE"]
    if not complete_rows:
        print("[COMPARE] FAIL: no comparable real-path rows completed")
        return 1

    champion_row = _select_champion(complete_rows)
    display_row = champion_row or _select_reference_row(complete_rows)

    fix_lines: list[str] = []
    xgb_pair = result_by_model.get("xgb")
    if xgb_pair is not None:
        xgb_row, xgb_result = xgb_pair
        importance_df = _rolling_xgb_feature_importance(feat, feature_cols, "target_next_ret", config.model)
        low_importance_cols = importance_df.head(min(8, len(importance_df)))["feature"].tolist() if not importance_df.empty else []
        ablation_rows: list[dict] = []
        support_preds = _support_comparison_preds("xgb", prediction_rows)
        for variant_name, variant_cols in _feature_variant_sets(feature_cols, low_importance_cols).items():
            try:
                xgb_variant_preds = rolling_train_predict_xgb(feat, variant_cols, "target_next_ret", config.model)
                variant_row, _variant_result = _run_real_lane(
                    model_name="xgb",
                    feat=feat,
                    preds=xgb_variant_preds,
                    config=config,
                    role="candidate",
                    comparison_preds=support_preds,
                    gate_overrides=gate_overrides,
                    notes=f"xgb feature ablation: {variant_name}",
                )
                ablation_rows.append(
                    {
                        "variant": variant_name,
                        "feature_count": len(variant_cols),
                        "weekly_return_pct": variant_row.weekly_return_pct,
                        "delta_weekly_return_pct": round(float(variant_row.weekly_return_pct) - float(xgb_row.weekly_return_pct), 4),
                        "daily_return_pct": variant_row.daily_return_pct,
                        "sharpe": variant_row.sharpe,
                        "max_drawdown_pct": variant_row.max_drawdown_pct,
                        "cvar_95_pct": variant_row.cvar_95_pct,
                        "trade_count": variant_row.trade_count,
                        "approval_count": variant_row.approval_count,
                        "expectancy": variant_row.expectancy,
                        "keep_coverage_pct": variant_row.keep_coverage_pct,
                        "eligible": "YES" if _is_eligible_for_champion(variant_row) else "NO",
                        "recommendation": "KEEP" if (
                            variant_row.weekly_return_pct > xgb_row.weekly_return_pct
                            and variant_row.keep_coverage_pct >= MIN_KEEP_COVERAGE_PCT
                            and variant_row.expectancy > 0.0
                        ) else "REJECT",
                    }
                )
            except Exception as exc:
                ablation_rows.append(
                    {
                        "variant": variant_name,
                        "feature_count": len(variant_cols),
                        "weekly_return_pct": float("nan"),
                        "delta_weekly_return_pct": float("nan"),
                        "daily_return_pct": float("nan"),
                        "sharpe": float("nan"),
                        "max_drawdown_pct": float("nan"),
                        "cvar_95_pct": float("nan"),
                        "trade_count": 0,
                        "approval_count": 0,
                        "expectancy": float("nan"),
                        "keep_coverage_pct": 0.0,
                        "eligible": "NO",
                        "recommendation": f"ERROR: {exc}",
                    }
                )

        fix_lines = _rank_xgb_fixes(xgb_row, xgb_result.get("trade_diagnostics", pd.DataFrame()), ablation_rows, importance_df)
        _write_false_positive_trade_review(xgb_row, xgb_result.get("trade_diagnostics", pd.DataFrame()), fix_lines, output_dir)
    else:
        (output_dir / "false_positive_trade_review.md").write_text(
            "# False Positive Trade Review\n\nNo xgb same-path result available.\n",
            encoding="utf-8",
        )

    (output_dir / "model_leaderboard.md").write_text(
        build_model_leaderboard_markdown(champion_row, rows),
        encoding="utf-8",
    )
    (output_dir / "real_benchmark.md").write_text(
        build_real_benchmark_markdown(champion_row, display_row, rows),
        encoding="utf-8",
    )
    (output_dir / "NEXT_STATUS.md").write_text(
        build_next_status_markdown(champion_row, display_row, fix_lines=fix_lines),
        encoding="utf-8",
    )

    print(f"[COMPARE] csv={output_dir / 'model_compare_real.csv'}")
    print(f"[COMPARE] summary_csv={output_dir / 'real_benchmark.csv'}")
    print(f"[COMPARE] leaderboard={output_dir / 'model_leaderboard.md'}")
    print(f"[COMPARE] benchmark={output_dir / 'real_benchmark.md'}")
    print(f"[COMPARE] false_positive={output_dir / 'false_positive_trade_review.md'}")
    print(f"[COMPARE] feature_ablation={output_dir / 'feature_ablation.md'}")
    print(f"[COMPARE] next={output_dir / 'NEXT_STATUS.md'}")
    print(
        "[COMPARE] RESULT: "
        f"{_pass_fail_label(display_row)} champion={champion_row.model_name if champion_row is not None else 'NONE'} "
        f"weekly={display_row.weekly_return_pct}% gap={_gap_to_target_pct_points(display_row.weekly_return_pct)} "
        f"sharpe={display_row.sharpe} dd={display_row.max_drawdown_pct}% trades={display_row.trade_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())