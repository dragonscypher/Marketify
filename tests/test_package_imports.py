"""Smoke test: verify every marketify subpackage imports cleanly."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "marketify",
    "marketify.config",
    "marketify.data",
    "marketify.data.market_data",
    "marketify.data.news_data",
    "marketify.broker.paper",
    "marketify.broker.ibkr",
    "marketify.features.technical",
    "marketify.models.xgb_model",
    "marketify.models.ensemble",
    "marketify.models.drift",
    "marketify.models.train_pipeline",
    "marketify.risk.risk_engine",
    "marketify.state.store",
    "marketify.backtest.benchmark",
    "marketify.backtest.simulator",
    "marketify.onboarding",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod: str) -> None:
    importlib.import_module(mod)
