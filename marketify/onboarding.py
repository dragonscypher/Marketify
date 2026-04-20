"""Onboarding wizard: load/save user_config.yaml and apply to AppConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from marketify.config import AppConfig

DEFAULT_CONFIG_PATH = Path("user_config.yaml")

DEFAULT_USER_CONFIG: dict[str, Any] = {
    "broker_backend": "paper",
    "initial_cash": 10000.0,
    "tickers": ["AAPL"],
    "risk_limits": {
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.10,
        "max_position_fraction": 0.10,
        "max_cvar_95": 0.02,
    },
    "allow_shorts": True,
    "benchmark_weekly_goal": 0.01,
    "approval_mode": "manual",
}


def load_user_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Merge with defaults for missing keys
        merged = {**DEFAULT_USER_CONFIG, **data}
        merged["risk_limits"] = {**DEFAULT_USER_CONFIG["risk_limits"], **data.get("risk_limits", {})}
        return merged
    return dict(DEFAULT_USER_CONFIG)


def save_user_config(config: dict[str, Any], path: Path | str | None = None) -> Path:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path


def apply_user_config(app_config: AppConfig, user_config: dict[str, Any]) -> AppConfig:
    """Apply user_config dict onto AppConfig fields."""
    app_config.broker.initial_cash = float(user_config.get("initial_cash", app_config.broker.initial_cash))
    app_config.broker.allow_shorts = bool(user_config.get("allow_shorts", app_config.broker.allow_shorts))
    app_config.broker.max_position_fraction = float(
        user_config.get("risk_limits", {}).get("max_position_fraction", app_config.broker.max_position_fraction)
    )
    app_config.benchmark_weekly_goal = float(
        user_config.get("benchmark_weekly_goal", app_config.benchmark_weekly_goal)
    )

    risk = user_config.get("risk_limits", {})
    app_config.risk.max_daily_loss_pct = float(risk.get("max_daily_loss_pct", app_config.risk.max_daily_loss_pct))
    app_config.risk.max_drawdown_pct = float(risk.get("max_drawdown_pct", app_config.risk.max_drawdown_pct))
    app_config.risk.max_cvar_95 = float(risk.get("max_cvar_95", app_config.risk.max_cvar_95))

    tickers = user_config.get("tickers", [])
    if tickers:
        app_config.data.ticker = str(tickers[0])

    return app_config
