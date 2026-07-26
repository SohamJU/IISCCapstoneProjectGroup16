"""Router agent configuration constants."""

from __future__ import annotations

from src.agents.router.schemas import ROUTE_LABELS

__all__ = ["ROUTE_LABELS", "MAX_ROUTES_PER_TURN", "HISTORY_TURNS_FOR_ROUTING"]

#: Hard cap on how many specialists a single user turn may fan out to.
#: The old substring parser regularly produced 3-4 routes from one question;
#: two is the realistic maximum for a genuine compound request.
MAX_ROUTES_PER_TURN = 2

#: How many recent turns the router sees. It needs *some* history to resolve
#: references like "cancel it" or "the second one", but that history must never
#: be keyword-scanned — see RouterAgent.classify_multi.
HISTORY_TURNS_FOR_ROUTING = 4
