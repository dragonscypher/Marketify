from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RISK_DISCLAIMER = (
    "Experimental paper-trading engine. No financial advice. Trading involves risk, "
    "including loss of principal. Target metrics are benchmarks, not guarantees."
)


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    interval: str = "5m"
    period: str = "60d"
    prepost: bool = False


@dataclass
class ModelConfig:
    train_window: int = 2500
    retrain_every: int = 500
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42


@dataclass
class BrokerConfig:
    initial_cash: float = 10000.0
    fee_bps: float = 2.0
    slippage_bps: float = 1.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.015
    time_stop_bars: int = 48
    allow_shorts: bool = True
    max_position_fraction: float = 0.1
    db_path: str = "data/marketify_paper.db"


@dataclass
class RiskConfig:
    min_expected_return: float = 0.0003
    max_cvar_95: float = 0.02
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.1
    max_news_risk: float = 0.75
    max_volatility: float = 0.04
    kill_switch_enabled: bool = False
    auto_halt_on_daily_loss: bool = True
    auto_halt_on_drawdown: bool = True
    drift_warning_threshold: float = 0.5
    drift_critical_threshold: float = 1.0


@dataclass
class IBKRConfig:
    account_id: str = ""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    paper: bool = True
    read_only: bool = True


@dataclass
class AppConfig:
    trading_mode: Literal["paper", "live"] = "paper"
    benchmark_weekly_goal: float = 0.01
    halt_reason: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)

    @property
    def is_halted(self) -> bool:
        return bool(self.halt_reason) or self.risk.kill_switch_enabled


@dataclass
class TradeIdea:
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    confidence: float
    expected_return: float
    cvar_95: float
    stop_loss: float
    take_profit: float
    reason: str
    risk_warning: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "cvar_95": self.cvar_95,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": self.reason,
            "risk_warning": self.risk_warning,
        }
