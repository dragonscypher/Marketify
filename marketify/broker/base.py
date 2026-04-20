from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BrokerBase(ABC):
    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError
