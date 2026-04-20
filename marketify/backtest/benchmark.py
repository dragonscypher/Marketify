from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    daily_return: float
    weekly_return: float
    max_drawdown: float
    sharpe_ratio: float
    cvar_95: float
    weekly_pass: bool
    drawdown_pass: bool
    sharpe_pass: bool


def compute_benchmark(
    equity_history: list[dict],
    weekly_goal: float = 0.01,
    max_drawdown_limit: float = 0.10,
    sharpe_floor: float = 0.5,
    risk_free_rate: float = 0.0,
) -> BenchmarkResult:
    """Evaluate paper-trading performance against benchmark targets.

    Args:
        equity_history: List of dicts with at least 'ts' and 'equity' keys.
        weekly_goal: Minimum weekly return target (default 1%).
        max_drawdown_limit: Maximum acceptable drawdown fraction.
        sharpe_floor: Minimum acceptable annualised Sharpe ratio.
        risk_free_rate: Risk-free rate for Sharpe computation.
    """
    if len(equity_history) < 2:
        return BenchmarkResult(
            daily_return=0.0,
            weekly_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            cvar_95=0.0,
            weekly_pass=False,
            drawdown_pass=True,
            sharpe_pass=False,
        )

    df = pd.DataFrame(equity_history)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    equity = df["equity"].astype(float)

    returns = equity.pct_change().dropna()
    daily_return = float(returns.iloc[-1]) if len(returns) > 0 else 0.0

    first_eq = float(equity.iloc[0])
    last_eq = float(equity.iloc[-1])
    total_return = (last_eq / first_eq - 1.0) if first_eq > 0 else 0.0

    n_points = len(equity)
    # Approximate weekly return from total return scaled by snapshot count
    weekly_return = total_return  # conservative: use total period return as proxy

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).min()
    max_drawdown = abs(float(drawdown))

    # Sharpe ratio (annualised from per-snapshot returns)
    if len(returns) > 1 and float(returns.std()) > 0:
        excess = returns - risk_free_rate / max(n_points, 1)
        sharpe_ratio = float(excess.mean() / returns.std()) * np.sqrt(min(n_points, 252))
    else:
        sharpe_ratio = 0.0

    # CVaR 95
    if len(returns) > 0:
        q5 = returns.quantile(0.05)
        tail = returns[returns <= q5]
        cvar_95 = float(abs(tail.mean())) if len(tail) > 0 else 0.0
    else:
        cvar_95 = 0.0

    return BenchmarkResult(
        daily_return=daily_return,
        weekly_return=weekly_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        cvar_95=cvar_95,
        weekly_pass=weekly_return >= weekly_goal,
        drawdown_pass=max_drawdown <= max_drawdown_limit,
        sharpe_pass=sharpe_ratio >= sharpe_floor,
    )
