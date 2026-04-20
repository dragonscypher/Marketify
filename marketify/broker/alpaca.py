from __future__ import annotations

from marketify.broker.base import BrokerBase


class AlpacaBroker(BrokerBase):
    def get_account(self) -> dict:
        raise NotImplementedError("Alpaca broker stub only. Live trading disabled by default.")

    def get_positions(self) -> list[dict]:
        raise NotImplementedError("Alpaca broker stub only. Live trading disabled by default.")

    def submit_order(self, order: dict) -> dict:
        raise NotImplementedError("Alpaca broker stub only. Live trading disabled by default.")

    def get_orders(self) -> list[dict]:
        raise NotImplementedError("Alpaca broker stub only. Live trading disabled by default.")

    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("Alpaca broker stub only. Live trading disabled by default.")
