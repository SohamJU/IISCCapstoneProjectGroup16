"""Typed state for the support supervisor graph.

This is the piece the previous ``for route in routes:`` loop could not
provide. When the router selected two specialists, the second had no access
to anything the first had resolved — so a "return this order and send me a
replacement" request made the return agent look up ``ORD-000123`` and then
the order agent immediately ask the customer for their order ID again.

``facts`` is the shared scratchpad that closes that gap, and ``messages`` is
the single canonical conversation replacing the six per-agent checkpointers.

Reducers
--------
Fields written by more than one node need a reducer, otherwise the second
writer silently replaces the first (which is exactly how ``executed_routes``
lost the first of two routes). The accumulators below share one convention:

* an update of ``None`` **resets** the field — used once per turn by
  :func:`new_turn_state`,
* any other update is **combined** with what is already there.

That distinction matters: an agent node legitimately returns ``facts={}`` when
it found no identifiers, and that must not wipe facts the previous agent
established in the same turn.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def append_list(current: list | None, update: list | None) -> list:
    """Accumulate list updates within a turn; ``None`` resets."""
    if update is None:
        return []
    if not isinstance(update, list):
        return list(current or [])
    return list(current or []) + update


def merge_dict(current: dict | None, update: dict | None) -> dict:
    """Merge dict updates within a turn; ``None`` resets."""
    if update is None:
        return {}
    if not isinstance(update, dict):
        return dict(current or {})
    return {**(current or {}), **update}


class SupportState(TypedDict, total=False):
    """State threaded through every node of the support graph."""

    # ── Conversation ──────────────────────────────────────────────────────
    #: Canonical message history. ``add_messages`` appends rather than
    #: replaces, and is what the checkpointer persists.
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Identity ──────────────────────────────────────────────────────────
    customer_id: str | None
    session_id: str

    # ── Routing ───────────────────────────────────────────────────────────
    #: Routes chosen for the current turn, in execution order.
    routes: list[str]
    #: Routes not yet executed this turn. The supervisor pops from the front.
    #: Deliberately has NO reducer — each write must replace the queue.
    pending_routes: list[str]
    #: Routes actually executed (may differ from ``routes`` after fallbacks).
    executed_routes: Annotated[list[str], append_list]
    confidence: float
    route_reason: str

    # ── Working data ──────────────────────────────────────────────────────
    #: Facts resolved by specialists this turn, e.g. ``{"order_id": "ORD-000123"}``.
    #: Passed forward so a later agent does not re-ask for what is already known.
    facts: Annotated[dict[str, Any], merge_dict]
    #: ``(route, answer)`` pairs collected from each specialist that ran.
    agent_outputs: Annotated[list[tuple[str, str]], append_list]
    #: Tool names invoked this turn, for the UI trace.
    tools_used: Annotated[list[str], append_list]

    # ── Control ───────────────────────────────────────────────────────────
    #: Set when the guardrail or router short-circuits with a direct reply.
    direct_response: str
    #: Non-fatal problems worth surfacing in the UI (degraded mode, tool errors).
    warnings: Annotated[list[str], append_list]
    #: Final customer-facing answer for this turn.
    final_response: str


def new_turn_state(
    user_message: str,
    session_id: str,
    customer_id: str | None,
) -> dict[str, Any]:
    """Build the per-turn state delta fed into ``graph.invoke``.

    Accumulator fields are passed ``None`` to reset them for the new turn —
    see the reducer convention documented at module level. ``messages`` is
    intentionally *not* reset: it accumulates across turns via
    ``add_messages`` and the checkpointer, which is what gives the assistant
    conversational memory.
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "customer_id": customer_id,
        "routes": [],
        "pending_routes": [],
        "executed_routes": None,
        "confidence": 0.0,
        "route_reason": "",
        "facts": None,
        "agent_outputs": None,
        "tools_used": None,
        "direct_response": "",
        "warnings": None,
        "final_response": "",
    }
