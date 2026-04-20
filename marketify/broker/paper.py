from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marketify.broker.base import BrokerBase
from marketify.config import BrokerConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperBroker(BrokerBase):
    def __init__(self, config: BrokerConfig):
        self.config = config
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.market_prices: dict[str, float] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    cash REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    max_equity REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    expected_return REAL,
                    cvar_95 REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    reason TEXT,
                    risk_warning TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_history (
                    ts TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    metadata_json TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT
                )
                """
            )
            row = self.conn.execute("SELECT id FROM account WHERE id=1").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO account(id, cash, realized_pnl, max_equity, updated_at) VALUES (1, ?, 0.0, ?, ?)",
                    (self.config.initial_cash, self.config.initial_cash, _utc_now()),
                )

    def update_market_price(self, symbol: str, price: float) -> None:
        self.market_prices[symbol] = float(price)
        self.record_equity_snapshot()

    def _position_row(self, symbol: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT symbol, qty, avg_cost FROM positions WHERE symbol=?", (symbol,)).fetchone()

    def _account_row(self) -> sqlite3.Row:
        row = self.conn.execute("SELECT cash, realized_pnl, max_equity FROM account WHERE id=1").fetchone()
        if row is None:
            raise RuntimeError("Paper account not initialized.")
        return row

    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.config.fee_bps / 10000.0)

    def _slipped_price(self, price: float, side: str) -> float:
        side_sign = 1 if side.lower() == "buy" else -1
        return float(price * (1.0 + side_sign * self.config.slippage_bps / 10000.0))

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        symbol = str(order["symbol"]).upper()
        side = str(order["side"]).lower()
        qty = float(order["qty"])
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if qty <= 0:
            raise ValueError("qty must be positive.")

        price = float(order.get("price") or self.market_prices.get(symbol) or 0.0)
        if price <= 0:
            raise ValueError("Valid market price required for paper fill.")

        with self.lock:
            fill_price = self._slipped_price(price, side)
            notional = qty * fill_price
            fee = self._fee(notional)

            order_id = str(order.get("order_id") or uuid.uuid4())
            fill_id = str(uuid.uuid4())
            created = _utc_now()

            self.conn.execute(
                """
                INSERT INTO orders(order_id, symbol, side, qty, price, status, expected_return, cvar_95, stop_loss,
                                   take_profit, reason, risk_warning, created_at)
                VALUES (?, ?, ?, ?, ?, 'filled', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    symbol,
                    side,
                    qty,
                    fill_price,
                    order.get("expected_return"),
                    order.get("cvar_95"),
                    order.get("stop_loss"),
                    order.get("take_profit"),
                    order.get("reason"),
                    order.get("risk_warning"),
                    created,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO fills(fill_id, order_id, symbol, side, qty, price, fee, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fill_id, order_id, symbol, side, qty, fill_price, fee, created),
            )

            account = self._account_row()
            cash = float(account["cash"])
            realized = float(account["realized_pnl"])
            max_equity = float(account["max_equity"])

            signed_fill = qty if side == "buy" else -qty
            if side == "buy":
                cash -= notional + fee
            else:
                cash += notional - fee

            pos = self._position_row(symbol)
            pos_qty = float(pos["qty"]) if pos is not None else 0.0
            pos_avg = float(pos["avg_cost"]) if pos is not None else 0.0

            if pos_qty == 0 or (pos_qty > 0 and signed_fill > 0) or (pos_qty < 0 and signed_fill < 0):
                new_qty = pos_qty + signed_fill
                new_avg = (
                    (abs(pos_qty) * pos_avg + abs(signed_fill) * fill_price) / abs(new_qty)
                    if new_qty != 0
                    else 0.0
                )
            else:
                closing_qty = min(abs(pos_qty), abs(signed_fill))
                realized += closing_qty * (fill_price - pos_avg) * (1 if pos_qty > 0 else -1)
                new_qty = pos_qty + signed_fill
                if new_qty == 0:
                    new_avg = 0.0
                elif (pos_qty > 0 and new_qty > 0) or (pos_qty < 0 and new_qty < 0):
                    new_avg = pos_avg
                else:
                    new_avg = fill_price

            if new_qty == 0:
                self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            else:
                self.conn.execute(
                    """
                    INSERT INTO positions(symbol, qty, avg_cost, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_cost=excluded.avg_cost, updated_at=excluded.updated_at
                    """,
                    (symbol, new_qty, new_avg, created),
                )

            equity = self._compute_equity(cash=cash)
            max_equity = max(max_equity, equity)
            self.conn.execute(
                "UPDATE account SET cash=?, realized_pnl=?, max_equity=?, updated_at=? WHERE id=1",
                (cash, realized, max_equity, created),
            )
            self.conn.commit()
            self.record_equity_snapshot()

        return {
            "order_id": order_id,
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": fill_price,
            "fee": fee,
        }

    def _compute_equity(self, cash: float | None = None) -> float:
        account = self._account_row()
        cash_val = float(account["cash"] if cash is None else cash)
        rows = self.conn.execute("SELECT symbol, qty, avg_cost FROM positions").fetchall()
        mtm = 0.0
        for row in rows:
            symbol = row["symbol"]
            qty = float(row["qty"])
            mark = float(self.market_prices.get(symbol, row["avg_cost"]))
            mtm += qty * mark
        return cash_val + mtm

    def get_account(self) -> dict[str, Any]:
        with self.lock:
            account = self._account_row()
            equity = self._compute_equity()
            cash = float(account["cash"])
            realized = float(account["realized_pnl"])
            max_equity = float(account["max_equity"])
            drawdown_pct = (equity / max_equity - 1.0) if max_equity > 0 else 0.0
            daily_loss_pct = (equity - max_equity) / max_equity if max_equity > 0 else 0.0
            unrealized = 0.0
            for pos in self.get_positions():
                unrealized += pos["unrealized_pnl"]
            return {
                "cash": cash,
                "equity": equity,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "max_equity": max_equity,
                "drawdown_pct": abs(min(0.0, drawdown_pct)),
                "daily_loss_pct": abs(min(0.0, daily_loss_pct)),
            }

    def get_positions(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT symbol, qty, avg_cost FROM positions ORDER BY symbol").fetchall()
            positions: list[dict[str, Any]] = []
            for row in rows:
                symbol = row["symbol"]
                qty = float(row["qty"])
                avg_cost = float(row["avg_cost"])
                mark = float(self.market_prices.get(symbol, avg_cost))
                unrealized = (mark - avg_cost) * qty
                positions.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "avg_cost": avg_cost,
                        "mark_price": mark,
                        "market_value": qty * mark,
                        "unrealized_pnl": unrealized,
                    }
                )
            return positions

    def get_orders(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM orders ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_fills(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM fills ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute("SELECT status FROM orders WHERE order_id=?", (order_id,)).fetchone()
            if row is None:
                return {"order_id": order_id, "status": "not_found"}
            if row["status"] == "filled":
                return {"order_id": order_id, "status": "cannot_cancel_filled"}
            self.conn.execute("UPDATE orders SET status='canceled' WHERE order_id=?", (order_id,))
            self.conn.commit()
            return {"order_id": order_id, "status": "canceled"}

    def record_equity_snapshot(self) -> None:
        with self.lock:
            account = self.get_account()
            ts = _utc_now()
            self.conn.execute(
                """
                INSERT OR REPLACE INTO equity_history(ts, equity, cash, unrealized_pnl, realized_pnl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    account["equity"],
                    account["cash"],
                    account["unrealized_pnl"],
                    account["realized_pnl"],
                ),
            )
            self.conn.commit()

    # ── Audit log ──────────────────────────────────────────────

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
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO audit_log(ts, action_type, symbol, side, qty, status, reason, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    action_type,
                    symbol,
                    side,
                    qty,
                    status,
                    reason,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            self.conn.commit()

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Persistent risk events ─────────────────────────────────

    def append_risk_event(
        self,
        event_type: str,
        severity: str,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO risk_events(ts, event_type, severity, reason, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    event_type,
                    severity,
                    reason,
                    json.dumps(payload) if payload else None,
                ),
            )
            self.conn.commit()

    def get_risk_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM risk_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_equity_history(self) -> list[dict[str, Any]]:
        """Return equity snapshots as list of dicts with 'ts' and 'equity' keys."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT ts, equity, cash, unrealized_pnl, realized_pnl "
                "FROM equity_history ORDER BY ts"
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
            # Fallback: single entry from current account
            account = self.get_account()
            return [{"ts": _utc_now(), "equity": account["equity"]}]

    def reset_account(self) -> None:
        """Reset paper account to initial state."""
        with self.lock:
            self.conn.execute(
                "UPDATE account SET cash=?, realized_pnl=0.0, max_equity=?, updated_at=? WHERE id=1",
                (self.config.initial_cash, self.config.initial_cash, _utc_now()),
            )
            self.conn.execute("DELETE FROM positions")
            self.conn.execute("DELETE FROM orders")
            self.conn.execute("DELETE FROM fills")
            self.conn.execute("DELETE FROM equity_history")
            self.conn.commit()
            self.market_prices.clear()
