from __future__ import annotations

from dataclasses import replace

import pandas as pd

from marketify.backtest.exit_rules import (ExitRuleConfig, build_exit_levels,
                                           evaluate_dynamic_exit)
from marketify.broker.paper import PaperBroker
from marketify.config import AppConfig
from marketify.data.market_data import fetch_market_data
from marketify.features.sentiment import FinBERTSentiment
from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                          add_technical_features)
from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                       compute_cvar_95)
from marketify.models.fusion_model import FusionConfig
from marketify.models.fusion_model import \
    rolling_train_predict as rolling_train_predict_fusion
from marketify.models.ridge_model import \
    rolling_train_predict as rolling_train_predict_ridge
from marketify.models.rnn_model import \
    rolling_train_predict as rolling_train_predict_gru
from marketify.models.xgb_model import \
    rolling_train_predict as rolling_train_predict_xgb
from marketify.risk.risk_engine import RiskEngine


def _scalar(row, col: str) -> float:
    """Extract scalar float from row[col], handling both scalar and 1-element Series."""
    value = row[col]
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def evaluate_exit_price(
    side: str,
    stop_loss: float,
    take_profit: float,
    row_open: float,
    row_high: float,
    row_low: float,
    bars_held: int,
    time_stop_bars: int,
) -> tuple[float | None, str | None]:
    if side == "buy":
        if row_low <= stop_loss:
            return float(stop_loss), "stop_loss"
        if row_high >= take_profit:
            return float(take_profit), "take_profit"
    else:
        if row_high >= stop_loss:
            return float(stop_loss), "stop_loss"
        if row_low <= take_profit:
            return float(take_profit), "take_profit"

    if bars_held >= time_stop_bars:
        return float(row_open), "time_stop"
    return None, None


def _build_exit_rule_config(config: AppConfig, variant: str) -> ExitRuleConfig:
    cfg = ExitRuleConfig(
        k_stop=float(getattr(config.broker, "exit_k_stop", 1.5)),
        k_take=float(getattr(config.broker, "exit_k_take", 2.5)),
        min_stop_pct=float(getattr(config.broker, "exit_min_stop_pct", 0.003)),
        max_stop_pct=float(getattr(config.broker, "exit_max_stop_pct", 0.03)),
        min_take_pct=float(getattr(config.broker, "exit_min_take_pct", 0.004)),
        max_take_pct=float(getattr(config.broker, "exit_max_take_pct", 0.06)),
        time_stop_bars=int(getattr(config.broker, "time_stop_bars", 48)),
        time_stop_mode=str(getattr(config.broker, "exit_time_stop_mode", "fixed")),
        vol_time_min_mult=float(getattr(config.broker, "exit_vol_time_min_mult", 0.5)),
        vol_time_max_mult=float(getattr(config.broker, "exit_vol_time_max_mult", 1.5)),
        trailing_enabled=bool(getattr(config.broker, "exit_trailing_enabled", False)),
        trailing_activate_profit_pct=float(getattr(config.broker, "exit_trailing_activate_profit_pct", 0.004)),
        trailing_distance_pct=float(getattr(config.broker, "exit_trailing_distance_pct", 0.003)),
    )

    if variant == "atr":
        return replace(cfg, trailing_enabled=False, time_stop_mode="fixed")
    if variant == "trailing":
        return replace(cfg, trailing_enabled=True, time_stop_mode="fixed")
    if variant == "hybrid":
        return replace(cfg, trailing_enabled=True, time_stop_mode="volatility_adjusted")
    return cfg


def _compute_predictions(feat: pd.DataFrame, config: AppConfig) -> tuple[pd.Series, str]:
    model_name = str(getattr(config.broker, "backtest_model", "xgb")).lower()
    if model_name == "ridge":
        preds = rolling_train_predict_ridge(
            frame=feat,
            feature_cols=TECHNICAL_FEATURE_COLUMNS,
            target_col="target_next_ret",
            config=config.model,
        )
        return preds, model_name

    if model_name == "gru":
        preds = rolling_train_predict_gru(
            frame=feat,
            feature_cols=TECHNICAL_FEATURE_COLUMNS,
            target_col="target_next_ret",
            config=config.model,
        )
        return preds, model_name

    if model_name == "fusion":
        fusion_cfg = FusionConfig(
            min_confidence=float(getattr(config.broker, "fusion_min_confidence", 0.30)),
            min_expected_return=float(getattr(config.broker, "fusion_expected_return_floor", 0.0003)),
            abstain_margin=float(getattr(config.broker, "fusion_abstain_margin", 0.0)),
            max_news_risk=float(getattr(config.broker, "fusion_news_risk_cutoff", 0.75)),
            high_vol_vix_threshold=float(getattr(config.broker, "fusion_regime_vix_cutoff", 30.0)),
            max_model_disagreement=float(getattr(config.broker, "fusion_max_model_disagreement", 0.006)),
        )
        preds = rolling_train_predict_fusion(
            frame=feat,
            feature_cols=TECHNICAL_FEATURE_COLUMNS,
            target_col="target_next_ret",
            config=config.model,
            fusion_config=fusion_cfg,
        )
        return preds, model_name

    preds = rolling_train_predict_xgb(
        frame=feat,
        feature_cols=TECHNICAL_FEATURE_COLUMNS,
        target_col="target_next_ret",
        config=config.model,
    )
    return preds, "xgb"


def run_backtest_from_predictions(
    feat: pd.DataFrame,
    preds: pd.Series,
    config: AppConfig,
    model_used: str,
) -> dict:
    broker = PaperBroker(config.broker)
    risk_engine = RiskEngine(config.risk, config.broker)
    sentiment = FinBERTSentiment(enabled=False)
    open_state: dict[str, dict] = {}
    exit_variant = str(getattr(config.broker, "exit_rule_variant", "fixed"))
    exit_cfg = _build_exit_rule_config(config, exit_variant)
    min_hold_bars = int(getattr(config.broker, "fusion_min_holding_bars", 0)) if model_used == "fusion" else 0

    # Shifted expanding median avoids lookahead in volatility reference.
    if "vol_20" in feat.columns:
        vol_ref = (
            pd.to_numeric(feat["vol_20"], errors="coerce")
            .expanding(min_periods=20)
            .median()
            .shift(1)
        )
    else:
        vol_ref = pd.Series(0.0, index=feat.index)

    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_diagnostics: list[dict] = []

    for ts in preds.dropna().index:
        row = feat.loc[ts]
        price = _scalar(row, "Close")
        broker.update_market_price(config.data.ticker, price)

        if config.data.ticker in open_state:
            state = open_state[config.data.ticker]
            bars_held = state["bars_held"] + 1
            state["bars_held"] = bars_held
            if exit_variant == "fixed":
                exit_price, exit_reason = evaluate_exit_price(
                    side=state["side"],
                    stop_loss=state["stop_loss"],
                    take_profit=state["take_profit"],
                    row_open=_scalar(row, "Open"),
                    row_high=_scalar(row, "High"),
                    row_low=_scalar(row, "Low"),
                    bars_held=bars_held,
                    time_stop_bars=config.broker.time_stop_bars,
                )
            else:
                ts_vol = _scalar(row, "vol_20") if "vol_20" in feat.columns else 0.0
                ts_vol_ref = float(vol_ref.get(ts, ts_vol)) if len(vol_ref) else ts_vol
                exit_price, exit_reason, state = evaluate_dynamic_exit(
                    side=state["side"],
                    state=state,
                    row_open=_scalar(row, "Open"),
                    row_high=_scalar(row, "High"),
                    row_low=_scalar(row, "Low"),
                    bars_held=bars_held,
                    volatility=ts_vol,
                    volatility_ref=ts_vol_ref,
                    cfg=exit_cfg,
                )
                open_state[config.data.ticker] = state

            if (
                exit_price is not None
                and bars_held < int(state.get("min_hold_bars", 0))
                and exit_reason in {"take_profit", "time_stop", "trailing_stop"}
            ):
                exit_price, exit_reason = None, None

            if exit_price is not None:
                close_side = "sell" if state["side"] == "buy" else "buy"
                broker.submit_order(
                    {
                        "symbol": config.data.ticker,
                        "side": close_side,
                        "qty": state["qty"],
                        "price": float(exit_price),
                        "reason": f"exit_rule_triggered:{exit_reason}",
                        "risk_warning": "Paper backtest exit",
                    }
                )
                # Record trade diagnostic
                entry_p = state["entry_price"]
                if state["side"] == "buy":
                    pnl = (exit_price - entry_p) * state["qty"]
                    ret_pct = (exit_price / entry_p - 1.0) * 100 if entry_p > 0 else 0.0
                else:
                    pnl = (entry_p - exit_price) * state["qty"]
                    ret_pct = (1.0 - exit_price / entry_p) * 100 if entry_p > 0 else 0.0

                trade_diagnostics.append({
                    "timestamp": str(ts),
                    "symbol": config.data.ticker,
                    "side": state["side"],
                    "qty": state["qty"],
                    "entry_price": entry_p,
                    "exit_price": float(exit_price),
                    "pnl": round(pnl, 4),
                    "return_pct": round(ret_pct, 4),
                    "hold_bars": bars_held,
                    "signal_confidence": state.get("confidence", 0.0),
                    "expected_return": state.get("expected_return", 0.0),
                    "cvar_95": state.get("cvar_95", 0.0),
                    "stop_loss": state["stop_loss"],
                    "take_profit": state["take_profit"],
                    "effective_time_stop_bars": state.get("effective_time_stop_bars", config.broker.time_stop_bars),
                    "min_holding_bars": state.get("min_hold_bars", 0),
                    "exit_reason": exit_reason,
                })
                open_state.pop(config.data.ticker, None)

        expected = float(preds.loc[ts])
        cvar_95 = compute_cvar_95(feat["ret_1"].iloc[max(0, feat.index.get_loc(ts) - 250): feat.index.get_loc(ts) + 1])
        sentiment_score = (
            _scalar(row, "news_sentiment_score")
            if "news_sentiment_score" in feat.columns
            else sentiment.analyze(config.data.ticker).score
        )
        news_risk = _scalar(row, "news_risk") if "news_risk" in feat.columns else 0.0
        vol_20 = _scalar(row, "vol_20")
        info = TradeIdeaInput(
            symbol=config.data.ticker,
            last_price=price,
            prediction=expected,
            sentiment_score=sentiment_score,
            volatility=vol_20,
            news_risk=news_risk,
        )
        account = broker.get_account()
        idea = build_trade_idea(info, account["equity"], config.broker, config.risk, cvar_95)

        if idea is not None and config.data.ticker not in open_state:
            decision = risk_engine.evaluate(
                trade=idea,
                account=account,
                volatility=info.volatility,
                news_risk=info.news_risk,
                mark_price=price,
            )
            if decision.approved:
                broker.submit_order(
                    {
                        **idea.to_dict(),
                        "price": price,
                    }
                )

                stop_loss = float(idea.stop_loss)
                take_profit = float(idea.take_profit)
                if exit_variant != "fixed":
                    atr = _scalar(row, "atr_14") if "atr_14" in feat.columns else 0.0
                    stop_loss, take_profit = build_exit_levels(
                        side=idea.side,
                        entry_price=price,
                        atr=atr,
                        cfg=exit_cfg,
                    )

                open_state[config.data.ticker] = {
                    "side": idea.side,
                    "qty": idea.qty,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "bars_held": 0,
                    "entry_price": price,
                    "confidence": idea.confidence,
                    "expected_return": idea.expected_return,
                    "cvar_95": idea.cvar_95,
                    "peak_price": price,
                    "trough_price": price,
                    "min_hold_bars": min_hold_bars,
                    "trailing_active": False,
                    "trailing_stop": None,
                }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], name="equity")
    return {
        "model_used": model_used,
        "equity": equity,
        "orders": pd.DataFrame(broker.get_orders()),
        "fills": pd.DataFrame(broker.get_fills()),
        "positions": pd.DataFrame(broker.get_positions()),
        "trade_diagnostics": pd.DataFrame(trade_diagnostics),
        "predictions": preds,
        "features": feat,
    }


def run_walk_forward_backtest(config: AppConfig) -> dict:
    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)

    preds, model_used = _compute_predictions(feat, config)
    return run_backtest_from_predictions(feat, preds, config, model_used)
