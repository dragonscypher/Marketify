from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketify.config import TradeIdea


@dataclass
class InMemoryStore:
    pending_trade_idea: TradeIdea | None = None
    approval_history: list[dict[str, Any]] = field(default_factory=list)
    rejection_history: list[dict[str, Any]] = field(default_factory=list)
    risk_events: list[dict[str, Any]] = field(default_factory=list)

    def set_pending(self, idea: TradeIdea) -> None:
        self.pending_trade_idea = idea

    def clear_pending(self) -> None:
        self.pending_trade_idea = None

    def approve_pending(self) -> TradeIdea | None:
        idea = self.pending_trade_idea
        if idea is not None:
            self.approval_history.append(idea.to_dict())
        self.pending_trade_idea = None
        return idea

    def reject_pending(self, reason: str) -> TradeIdea | None:
        idea = self.pending_trade_idea
        if idea is not None:
            payload = idea.to_dict()
            payload["reject_reason"] = reason
            self.rejection_history.append(payload)
        self.pending_trade_idea = None
        return idea

    def add_risk_event(self, payload: dict[str, Any]) -> None:
        self.risk_events.append(payload)
