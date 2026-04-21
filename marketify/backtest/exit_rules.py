from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


TimeStopMode = Literal["fixed", "volatility_adjusted"]


@dataclass(frozen=True)
class ExitRuleConfig:
    k_stop: float = 1.5
    k_take: float = 2.5
    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.03
    min_take_pct: float = 0.004
    max_take_pct: float = 0.06
    time_stop_bars: int = 48
    time_stop_mode: TimeStopMode = "fixed"
    vol_time_min_mult: float = 0.5
    vol_time_max_mult: float = 1.5
    trailing_enabled: bool = False
    trailing_activate_profit_pct: float = 0.004
    trailing_distance_pct: float = 0.003


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def atr_scaled_exit_pcts(
    atr: float,
    price: float,
    k_stop: float,
    k_take: float,
    min_stop_pct: float,
    max_stop_pct: float,
    min_take_pct: float,
    max_take_pct: float,
) -> tuple[float, float]:
    """Return ATR-scaled stop/take percentages with sane clamping."""
    safe_price = float(price) if isfinite(float(price)) and float(price) > 0 else 0.0
    safe_atr = float(atr) if isfinite(float(atr)) and float(atr) > 0 else 0.0
    atr_ratio = (safe_atr / safe_price) if safe_price > 0 else 0.0

    stop_pct = _clip(float(k_stop) * atr_ratio, float(min_stop_pct), float(max_stop_pct))
    take_pct = _clip(float(k_take) * atr_ratio, float(min_take_pct), float(max_take_pct))
    return stop_pct, take_pct


def build_exit_levels(side: str, entry_price: float, atr: float, cfg: ExitRuleConfig) -> tuple[float, float]:
    stop_pct, take_pct = atr_scaled_exit_pcts(
        atr=atr,
        price=entry_price,
        k_stop=cfg.k_stop,
        k_take=cfg.k_take,
        min_stop_pct=cfg.min_stop_pct,
        max_stop_pct=cfg.max_stop_pct,
        min_take_pct=cfg.min_take_pct,
        max_take_pct=cfg.max_take_pct,
    )

    if side == "buy":
        stop_loss = float(entry_price) * (1.0 - stop_pct)
        take_profit = float(entry_price) * (1.0 + take_pct)
    else:
        stop_loss = float(entry_price) * (1.0 + stop_pct)
        take_profit = float(entry_price) * (1.0 - take_pct)
    return float(stop_loss), float(take_profit)


def effective_time_stop_bars(
    base_bars: int,
    mode: TimeStopMode,
    volatility: float,
    volatility_ref: float,
    min_mult: float,
    max_mult: float,
) -> int:
    safe_base = max(1, int(base_bars))
    if mode != "volatility_adjusted":
        return safe_base

    safe_vol = float(volatility) if isfinite(float(volatility)) and float(volatility) > 0 else 0.0
    safe_ref = float(volatility_ref) if isfinite(float(volatility_ref)) and float(volatility_ref) > 0 else 0.0
    if safe_vol <= 0 or safe_ref <= 0:
        return safe_base

    mult = _clip(safe_ref / safe_vol, float(min_mult), float(max_mult))
    return max(1, int(round(safe_base * mult)))


def evaluate_dynamic_exit(
    side: str,
    state: dict,
    row_open: float,
    row_high: float,
    row_low: float,
    bars_held: int,
    volatility: float,
    volatility_ref: float,
    cfg: ExitRuleConfig,
) -> tuple[float | None, str | None, dict]:
    """Evaluate stop-loss/take-profit/trailing/time-stop in order."""
    stop_loss = float(state["stop_loss"])
    take_profit = float(state["take_profit"])
    entry_price = float(state["entry_price"])

    if side == "buy":
        if float(row_low) <= stop_loss:
            return stop_loss, "stop_loss", state
        if float(row_high) >= take_profit:
            return take_profit, "take_profit", state
    else:
        if float(row_high) >= stop_loss:
            return stop_loss, "stop_loss", state
        if float(row_low) <= take_profit:
            return take_profit, "take_profit", state

    if cfg.trailing_enabled:
        if side == "buy":
            peak_price = max(float(state.get("peak_price", entry_price)), float(row_high))
            state["peak_price"] = peak_price
            profit_pct = (peak_price / entry_price - 1.0) if entry_price > 0 else 0.0
            if profit_pct >= cfg.trailing_activate_profit_pct:
                state["trailing_active"] = True

            if state.get("trailing_active", False):
                trailing_stop = peak_price * (1.0 - cfg.trailing_distance_pct)
                state["trailing_stop"] = float(trailing_stop)
                if float(row_low) <= trailing_stop:
                    return float(trailing_stop), "trailing_stop", state
        else:
            trough_price = min(float(state.get("trough_price", entry_price)), float(row_low))
            state["trough_price"] = trough_price
            profit_pct = (1.0 - trough_price / entry_price) if entry_price > 0 else 0.0
            if profit_pct >= cfg.trailing_activate_profit_pct:
                state["trailing_active"] = True

            if state.get("trailing_active", False):
                trailing_stop = trough_price * (1.0 + cfg.trailing_distance_pct)
                state["trailing_stop"] = float(trailing_stop)
                if float(row_high) >= trailing_stop:
                    return float(trailing_stop), "trailing_stop", state

    stop_bars = effective_time_stop_bars(
        base_bars=cfg.time_stop_bars,
        mode=cfg.time_stop_mode,
        volatility=volatility,
        volatility_ref=volatility_ref,
        min_mult=cfg.vol_time_min_mult,
        max_mult=cfg.vol_time_max_mult,
    )
    state["effective_time_stop_bars"] = int(stop_bars)
    if int(bars_held) >= stop_bars:
        return float(row_open), "time_stop", state

    return None, None, state
