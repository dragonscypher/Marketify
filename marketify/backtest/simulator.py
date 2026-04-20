from __future__ import annotations

import pandas as pd

from marketify.broker.paper import PaperBroker
from marketify.config import AppConfig
from marketify.data.market_data import fetch_market_data
from marketify.features.sentiment import FinBERTSentiment
from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                          add_technical_features)
from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                       compute_cvar_95)
from marketify.models.xgb_model import rolling_train_predict
from marketify.risk.risk_engine import RiskEngine


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


def run_walk_forward_backtest(config: AppConfig) -> dict:
    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)

    preds = rolling_train_predict(
        frame=feat,
        feature_cols=TECHNICAL_FEATURE_COLUMNS,
        target_col="target_next_ret",
        config=config.model,
    )

    broker = PaperBroker(config.broker)
    risk_engine = RiskEngine(config.risk, config.broker)
    sentiment = FinBERTSentiment(enabled=False)
    open_state: dict[str, dict] = {}

    equity_points: list[tuple[pd.Timestamp, float]] = []

    for ts in preds.dropna().index:
        row = feat.loc[ts]
        price = float(row["Close"])
        broker.update_market_price(config.data.ticker, price)

        if config.data.ticker in open_state:
            state = open_state[config.data.ticker]
            bars_held = state["bars_held"] + 1
            state["bars_held"] = bars_held
            exit_price, exit_reason = evaluate_exit_price(
                side=state["side"],
                stop_loss=state["stop_loss"],
                take_profit=state["take_profit"],
                row_open=float(row["Open"]),
                row_high=float(row["High"]),
                row_low=float(row["Low"]),
                bars_held=bars_held,
                # Time-stop bug fix: use broker.time_stop_bars (not signal.time_stop_bars)
                time_stop_bars=config.broker.time_stop_bars,
            )

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
                open_state.pop(config.data.ticker, None)

        expected = float(preds.loc[ts])
        cvar_95 = compute_cvar_95(feat["ret_1"].iloc[max(0, feat.index.get_loc(ts) - 250): feat.index.get_loc(ts) + 1])
        sentiment_score = sentiment.analyze(config.data.ticker).score
        info = TradeIdeaInput(
            symbol=config.data.ticker,
            last_price=price,
            prediction=expected,
            sentiment_score=sentiment_score,
            volatility=float(row["vol_20"]),
            news_risk=0.0,
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
                open_state[config.data.ticker] = {
                    "side": idea.side,
                    "qty": idea.qty,
                    "stop_loss": idea.stop_loss,
                    "take_profit": idea.take_profit,
                    "bars_held": 0,
                }

        equity_points.append((ts, broker.get_account()["equity"]))

    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], name="equity")
    return {
        "equity": equity,
        "orders": pd.DataFrame(broker.get_orders()),
        "fills": pd.DataFrame(broker.get_fills()),
        "positions": pd.DataFrame(broker.get_positions()),
    }
