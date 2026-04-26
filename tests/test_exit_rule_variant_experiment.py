from __future__ import annotations

import pandas as pd

from scripts.run_exit_rule_variant_experiment import (_select_best,
                                                      _test_start_index,
                                                      _variant_overrides)


class _Broker:
    def __init__(self):
        self.max_position_fraction = 0.2
        self.time_stop_bars = 48


class _Config:
    def __init__(self):
        self.broker = _Broker()


def test_test_start_index_keeps_two_points_minimum():
    assert _test_start_index(0) == 0
    assert _test_start_index(1) == 0
    assert _test_start_index(2) == 0
    assert _test_start_index(10) <= 8


def test_variant_overrides_never_increase_position_fraction():
    config = _Config()

    _variant_overrides(config, "fixed")
    assert config.broker.max_position_fraction <= 0.10
    assert config.broker.exit_rule_variant == "fixed"

    _variant_overrides(config, "hybrid")
    assert config.broker.max_position_fraction <= 0.10
    assert config.broker.exit_rule_variant == "hybrid"
    assert config.broker.exit_time_stop_mode == "volatility_adjusted"
    assert config.broker.exit_trailing_enabled is True


def test_select_best_uses_test_metrics_priority():
    df = pd.DataFrame(
        [
            {
                "variant": "fixed",
                "weekly_return_pct": 0.91,
                "sharpe_ratio": 0.8,
                "max_drawdown_pct": 0.30,
                "trade_count": 20,
            },
            {
                "variant": "hybrid",
                "weekly_return_pct": 0.95,
                "sharpe_ratio": 0.5,
                "max_drawdown_pct": 0.40,
                "trade_count": 18,
            },
            {
                "variant": "atr",
                "weekly_return_pct": 0.95,
                "sharpe_ratio": 0.7,
                "max_drawdown_pct": 0.35,
                "trade_count": 12,
            },
        ]
    )

    best = _select_best(df)

    assert best is not None
    assert best["variant"] == "atr"
