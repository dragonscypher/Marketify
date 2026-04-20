"""Tests for onboarding wizard (load/save/apply user config)."""
from pathlib import Path

from marketify.config import AppConfig
from marketify.onboarding import (
    DEFAULT_USER_CONFIG,
    apply_user_config,
    load_user_config,
    save_user_config,
)


def test_default_config_keys():
    assert "initial_cash" in DEFAULT_USER_CONFIG
    assert "risk_limits" in DEFAULT_USER_CONFIG
    assert "max_position_fraction" in DEFAULT_USER_CONFIG["risk_limits"]


def test_save_and_load(tmp_path):
    cfg_path = tmp_path / "user_config.yaml"
    user_cfg = {**DEFAULT_USER_CONFIG, "initial_cash": 50000}
    save_user_config(user_cfg, path=cfg_path)
    loaded = load_user_config(path=cfg_path)
    assert loaded["initial_cash"] == 50000


def test_apply_user_config(tmp_path):
    app_config = AppConfig()
    user_cfg = {
        "initial_cash": 25000,
        "risk_limits": {
            "max_position_fraction": 0.15,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.12,
        },
    }
    apply_user_config(app_config, user_cfg)
    assert app_config.broker.initial_cash == 25000
    assert app_config.broker.max_position_fraction == 0.15
    assert app_config.risk.max_daily_loss_pct == 0.03
    assert app_config.risk.max_drawdown_pct == 0.12


def test_load_missing_file(tmp_path):
    cfg_path = tmp_path / "nonexistent.yaml"
    loaded = load_user_config(path=cfg_path)
    assert loaded == DEFAULT_USER_CONFIG
