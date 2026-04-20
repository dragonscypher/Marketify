"""IBKR paper-trading adapter skeleton.

Read-only connection test + paper-only order guard.
Requires `ib_insync` package and TWS/Gateway running on configured host:port.
No live orders ever submitted — paper account only.
"""
from __future__ import annotations

import os
from typing import Any

from marketify.broker.base import BrokerBase
from marketify.config import IBKRConfig


class IBKRBroker(BrokerBase):
    """Skeleton IBKR adapter. Paper-only, read-only by default."""

    def __init__(self, config: IBKRConfig | None = None):
        self.config = config or IBKRConfig()
        self._ib = None  # lazy import ib_insync.IB

    def _connect(self) -> Any:
        if self._ib is not None:
            return self._ib
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError("ib_insync not installed. pip install ib_insync")
        ib = IB()
        ib.connect(
            host=self.config.host,
            port=self.config.port,
            clientId=self.config.client_id,
            readonly=self.config.read_only,
        )
        self._ib = ib
        return ib

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None

    def connection_test(self) -> dict[str, Any]:
        """Test connection to TWS/Gateway. Returns account summary."""
        ib = self._connect()
        accounts = ib.managedAccounts()
        if not accounts:
            return {"connected": False, "error": "No managed accounts found"}
        summary = ib.accountSummary(account=accounts[0])
        result = {"connected": True, "account": accounts[0], "summary_count": len(summary)}
        return result

    def get_account(self) -> dict[str, Any]:
        ib = self._connect()
        accounts = ib.managedAccounts()
        if not accounts:
            return {"cash": 0, "equity": 0}
        summary = ib.accountSummary(account=accounts[0])
        data: dict[str, Any] = {}
        for item in summary:
            if item.tag == "TotalCashValue":
                data["cash"] = float(item.value)
            elif item.tag == "NetLiquidation":
                data["equity"] = float(item.value)
        return data

    def get_positions(self) -> list[dict[str, Any]]:
        ib = self._connect()
        positions = ib.positions()
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.contract.symbol,
                "qty": float(pos.position),
                "avg_cost": float(pos.avgCost),
                "market_value": float(pos.position) * float(pos.avgCost),
            })
        return result

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        if not self.config.paper:
            raise RuntimeError("IBKR adapter: live trading disabled. Set paper=True in IBKRConfig.")
        if self.config.read_only:
            raise RuntimeError("IBKR adapter: read_only mode. Disable read_only to submit paper orders.")
        raise NotImplementedError("IBKR paper order submission not yet implemented.")

    def get_orders(self) -> list[dict[str, Any]]:
        ib = self._connect()
        trades = ib.trades()
        return [
            {
                "order_id": str(t.order.orderId),
                "symbol": t.contract.symbol,
                "side": t.order.action.lower(),
                "qty": float(t.order.totalQuantity),
                "status": t.orderStatus.status,
            }
            for t in trades
        ]

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError("IBKR order cancellation not yet implemented.")
