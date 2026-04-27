from __future__ import annotations

import json
import numbers
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from marketify.broker.base import BrokerBase

INITIAL_CASH = 100_000.0
ACCOUNT_FIXTURE = {
    "broker": "mock",
    "mode": "paper",
    "currency": "USD",
    "cash": INITIAL_CASH,
    "equity": INITIAL_CASH,
    "buying_power": INITIAL_CASH,
    "status": "ACTIVE",
}
ASSET_FIXTURE = [
    {"symbol": "AAPL", "tradable": True, "shortable": True, "fractionable": False},
    {"symbol": "MSFT", "tradable": True, "shortable": True, "fractionable": False},
]
DEFAULT_MARKS = {"AAPL": 101.0, "MSFT": 199.0}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, numbers.Number):
        return value
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    return value


class MockBrokerState:
    """Deterministic paper broker state for local zero-credential validation."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "lock", threading.RLock()):
            self.cash = float(INITIAL_CASH)
            self.positions: dict[str, dict[str, Any]] = {}
            self.orders: list[dict[str, Any]] = []
            self.fills: list[dict[str, Any]] = []
            self.mark_prices: dict[str, float] = dict(DEFAULT_MARKS)

    def account(self) -> dict[str, Any]:
        with self.lock:
            equity = self.cash + sum(float(pos["qty"]) * float(pos["mark_price"]) for pos in self.positions.values())
            unrealized_pnl = sum(float(pos.get("unrealized_pnl", 0.0)) for pos in self.positions.values())
            return {
                **ACCOUNT_FIXTURE,
                "cash": round(self.cash, 2),
                "equity": round(equity, 2),
                "buying_power": round(max(self.cash, 0.0), 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "realized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "drawdown_pct": 0.0,
            }

    def set_mark_price(self, symbol: str, price: float) -> None:
        with self.lock:
            symbol = symbol.upper()
            self.mark_prices[symbol] = float(price)
            if symbol in self.positions:
                self.positions[symbol]["mark_price"] = float(price)
                self._refresh_position(symbol)

    def _refresh_position(self, symbol: str) -> None:
        pos = self.positions[symbol]
        qty = float(pos["qty"])
        avg = float(pos["avg_entry"])
        mark = float(pos["mark_price"])
        unrealized = (mark - avg) * qty
        pos["market_value"] = round(qty * mark, 2)
        pos["unrealized_pnl"] = round(unrealized, 2)

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            symbol = str(order.get("symbol", "")).upper()
            side = str(order.get("side", "")).lower()
            qty = float(order.get("qty", 0.0))
            price = float(order.get("price", self.mark_prices.get(symbol, 0.0)))
            if symbol not in {asset["symbol"] for asset in ASSET_FIXTURE}:
                raise ValueError(f"unsupported_symbol:{symbol}")
            if side not in {"buy", "sell"}:
                raise ValueError("side must be buy or sell")
            if qty <= 0 or price <= 0:
                raise ValueError("qty and price must be positive")

            signed_qty = qty if side == "buy" else -qty
            if side == "buy":
                self.cash -= qty * price
            else:
                self.cash += qty * price

            existing = self.positions.get(symbol)
            if existing is None:
                new_qty = signed_qty
                avg_entry = price
            else:
                old_qty = float(existing["qty"])
                old_avg = float(existing["avg_entry"])
                new_qty = old_qty + signed_qty
                if new_qty == 0:
                    avg_entry = 0.0
                elif (old_qty >= 0 and signed_qty >= 0) or (old_qty <= 0 and signed_qty <= 0):
                    avg_entry = (abs(old_qty) * old_avg + abs(signed_qty) * price) / abs(new_qty)
                else:
                    avg_entry = old_avg if abs(old_qty) > abs(signed_qty) else price

            order_id = str(order.get("order_id") or uuid.uuid4())
            fill_id = str(uuid.uuid4())
            created = _utc_now()
            order_row = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "status": "filled",
                "created_at": created,
            }
            fill_row = {
                "fill_id": fill_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "fee": 0.0,
                "created_at": created,
            }
            self.orders.append(order_row)
            self.fills.append(fill_row)

            if new_qty == 0:
                self.positions.pop(symbol, None)
            else:
                mark = self.mark_prices.get(symbol, price)
                if symbol == "AAPL" and side == "buy" and qty == 10 and price == 100.0:
                    mark = 101.0
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": round(new_qty, 6),
                    "avg_entry": round(avg_entry, 6),
                    "avg_cost": round(avg_entry, 6),
                    "mark_price": float(mark),
                }
                self._refresh_position(symbol)
            return order_row

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        with self.lock:
            for order in self.orders:
                if order["order_id"] == order_id:
                    if order["status"] == "filled":
                        return {"order_id": order_id, "status": "cannot_cancel_filled"}
                    order["status"] = "canceled"
                    return {"order_id": order_id, "status": "canceled"}
            return {"order_id": order_id, "status": "not_found"}


class _MockBrokerHandler(BaseHTTPRequestHandler):
    state: MockBrokerState

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/v1/account":
            self._send(self.state.account())
        elif self.path == "/v1/positions":
            self._send(list(self.state.positions.values()))
        elif self.path == "/v1/orders":
            self._send(self.state.orders)
        elif self.path == "/v1/fills":
            self._send(self.state.fills)
        elif self.path == "/v1/clock":
            self._send({"timestamp": _utc_now(), "is_open": True, "next_open": _utc_now(), "next_close": _utc_now()})
        elif self.path == "/v1/assets":
            self._send(ASSET_FIXTURE)
        else:
            self._send({"error": "not_found", "path": self.path}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            payload = self._read_json()
            if self.path == "/v1/orders":
                self._send(self.state.submit_order(payload))
            elif self.path == "/v1/market_prices":
                self.state.set_mark_price(str(payload["symbol"]), float(payload["price"]))
                self._send({"status": "ok"})
            elif self.path == "/v1/reset":
                self.state.reset()
                self._send({"status": "reset"})
            else:
                self._send({"error": "not_found", "path": self.path}, status=404)
        except Exception as exc:
            self._send({"error": str(exc)}, status=400)

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        prefix = "/v1/orders/"
        if self.path.startswith(prefix):
            self._send(self.state.cancel_order(self.path.removeprefix(prefix)))
        else:
            self._send({"error": "not_found", "path": self.path}, status=404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
        return


@dataclass
class MockBrokerServer:
    server: ThreadingHTTPServer
    thread: threading.Thread
    state: MockBrokerState

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_mock_broker_server(host: str = "127.0.0.1", port: int = 0) -> MockBrokerServer:
    state = MockBrokerState()

    class Handler(_MockBrokerHandler):
        pass

    Handler.state = state
    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="mock-broker-http", daemon=True)
    thread.start()
    return MockBrokerServer(server=server, thread=thread, state=state)


class MockBrokerHTTPClient(BrokerBase):
    """BrokerBase-compatible HTTP client for deterministic mock broker service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.audit_log: list[dict[str, Any]] = []
        self.risk_events: list[dict[str, Any]] = []

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(_json_safe(payload)).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(req, timeout=10) as resp:  # nosec B310 - local deterministic validation server
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise RuntimeError(body) from exc

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/v1/account")

    def get_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/positions")

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/orders", order)

    def get_orders(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/orders")

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/orders/{order_id}")

    def get_fills(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/fills")

    def get_assets(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/assets")

    def get_clock(self) -> dict[str, Any]:
        return self._request("GET", "/v1/clock")

    def update_market_price(self, symbol: str, price: float) -> None:
        self._request("POST", "/v1/market_prices", {"symbol": symbol, "price": float(price)})

    def reset_account(self) -> None:
        self._request("POST", "/v1/reset", {})

    def append_audit(
        self,
        action_type: str,
        status: str,
        symbol: str | None = None,
        side: str | None = None,
        qty: float | None = None,
        reason: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.audit_log.append(
            {
                "ts": _utc_now(),
                "action_type": action_type,
                "status": status,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "reason": reason,
                "metadata": metadata,
            }
        )

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(reversed(self.audit_log))[:limit]

    def append_risk_event(
        self,
        event_type: str,
        severity: str,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        self.risk_events.append(
            {"ts": _utc_now(), "event_type": event_type, "severity": severity, "reason": reason, "payload": payload}
        )

    def get_risk_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(reversed(self.risk_events))[:limit]
