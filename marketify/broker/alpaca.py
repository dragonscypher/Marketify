from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marketify.broker.base import BrokerBase


class AlpacaBroker(BrokerBase):
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(self, base_url: str | None = None, key_id: str | None = None, secret_key: str | None = None):
        _load_dotenv_if_present()
        self.base_url = (base_url or os.environ.get("ALPACA_BASE_URL") or self.PAPER_BASE_URL).rstrip("/")
        self.key_id = key_id or os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET_KEY", "")

    @property
    def paper_only(self) -> bool:
        return self.base_url.startswith(self.PAPER_BASE_URL) and os.environ.get("LIVE_TRADING", "false").lower() != "true"

    def _request(self, path: str) -> dict[str, Any]:
        if not self.key_id or not self.secret_key:
            raise RuntimeError("Alpaca env unavailable. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY for paper read-only test.")
        if not self.paper_only:
            raise RuntimeError("Alpaca live endpoint blocked. Paper endpoint only by default.")
        req = Request(
            f"{self.base_url}{path}",
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            method="GET",
        )
        with urlopen(req, timeout=10) as resp:  # nosec B310 - user-configured broker readiness endpoint
            return json.loads(resp.read().decode("utf-8"))

    def connection_test(self) -> dict[str, Any]:
        if not self.key_id or not self.secret_key:
            return {"connected": False, "skipped": True, "reason": "Alpaca paper env unavailable"}
        if not self.paper_only:
            return {"connected": False, "skipped": True, "reason": "Alpaca live endpoint blocked"}
        try:
            account = self._request("/v2/account")
            return {
                "connected": True,
                "paper_only": True,
                "account_status": account.get("status", "unknown"),
                "trading_blocked": account.get("trading_blocked", "unknown"),
            }
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            return {"connected": False, "skipped": False, "reason": str(exc)}

    def get_account(self) -> dict:
        return self._request("/v2/account")

    def get_positions(self) -> list[dict]:
        data = self._request("/v2/positions")
        return data if isinstance(data, list) else []

    def submit_order(self, order: dict) -> dict:
        raise RuntimeError("Alpaca order submission disabled. Paper readiness is read-only unless explicit user approval path is implemented.")

    def get_orders(self) -> list[dict]:
        data = self._request("/v2/orders?status=open")
        return data if isinstance(data, list) else []

    def cancel_order(self, order_id: str) -> dict:
        raise RuntimeError("Alpaca cancel disabled in read-only readiness adapter.")


def _load_dotenv_if_present(path: str | os.PathLike[str] = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
