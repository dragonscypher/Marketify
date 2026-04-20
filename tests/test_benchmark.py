"""Tests for benchmark performance checker."""
from marketify.backtest.benchmark import BenchmarkResult, compute_benchmark


def test_rising_equity_passes():
    history = [
        {"ts": f"2026-01-0{i+1}T00:00:00", "equity": 10000 + i * 50}
        for i in range(7)
    ]
    result = compute_benchmark(history, weekly_goal=0.01)
    assert isinstance(result, BenchmarkResult)
    assert result.weekly_return > 0
    assert result.weekly_pass is True


def test_flat_equity_fails_weekly():
    history = [
        {"ts": f"2026-01-0{i+1}T00:00:00", "equity": 10000}
        for i in range(7)
    ]
    result = compute_benchmark(history, weekly_goal=0.01)
    assert result.weekly_pass is False


def test_drawdown_detection():
    equity = [10000, 10100, 10200, 9800, 9700, 9600, 9500]
    history = [
        {"ts": f"2026-01-0{i+1}T00:00:00", "equity": equity[i]}
        for i in range(7)
    ]
    result = compute_benchmark(history, max_drawdown_limit=0.05)
    # Drawdown from 10200 to 9500 = ~6.86%
    assert result.max_drawdown > 0.05
    assert result.drawdown_pass is False


def test_sharpe_computation():
    history = [
        {"ts": f"2026-01-0{i+1}T00:00:00", "equity": 10000 + i * 10}
        for i in range(7)
    ]
    result = compute_benchmark(history)
    assert result.sharpe_ratio is not None
