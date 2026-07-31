"""Construction of the support supervisor ``StateGraph``.

Topology::

              ┌───────────┐
    START ───▶│ guardrail │──(invalid)──────────────┐
              └─────┬─────┘                         │
                    │ (valid)                       │
                    ▼                               │
              ┌────────────┐                        │
        ┌────▶│ supervisor │──(nothing pending)────▶├──▶ synthesize ──▶ END
        │     └─────┬──────┘                        │
        │           │ (next pending route)          │
        │           ▼                               │
        │   product │ order │ return │              │
        │   recommendation │ escalation │ fallback  │
        │           │                               │
        └───────────┘

The supervisor is re-entered after each specialist, which is what allows a
two-route turn to share ``facts`` between agents.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.escalation_agent import EscalationAgent
from src.agents.fallback_agent import FallbackAgent
from src.agents.graph.nodes import (
    guardrail_node,
    make_agent_node,
    make_supervisor_node,
    make_synthesis_node,
    route_after_guardrail,
    route_after_supervisor,
)
from src.agents.graph.state import SupportState
from src.agents.order_agent import OrderAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.return_agent import ReturnAgent
from src.agents.router import RouterAgent
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

#: Order matters only for readability; dispatch is data-driven.
SPECIALIST_ROUTES: tuple[str, ...] = (
    "product",
    "order",
    "return",
    "recommendation",
    "escalation",
    "fallback",
)


def build_specialists(debug: bool = False) -> dict[str, Any]:
    """Instantiate every specialist agent once, for the whole process.

    Unlike the previous implementation these are **not** per-session. The
    agents are stateless, so one instance serves every conversation — which
    also removes the unbounded ``_sessions`` dict that leaked six agents,
    six LLM clients and six checkpointers per session.

    Raises
    ------
    Exception
        Propagated from the first agent that cannot be constructed. Callers
        decide whether to degrade; failures are no longer swallowed here.
    """
    return {
        "product": ProductAgent(debug=debug),
        "order": OrderAgent(debug=debug),
        "return": ReturnAgent(debug=debug),
        "recommendation": RecommendationAgent(debug=debug),
        "escalation": EscalationAgent(debug=debug),
        "fallback": FallbackAgent(debug=debug),
    }


def build_support_graph(
    specialists: dict[str, Any] | None = None,
    router: RouterAgent | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    debug: bool = False,
):
    """Compile and return the support supervisor graph.

    Parameters
    ----------
    specialists
        Route name to agent mapping. Built via :func:`build_specialists` when
        omitted. Injectable so tests can pass stubs.
    router
        Router instance. Constructed when omitted.
    checkpointer
        Conversation persistence. Defaults to an in-process
        :class:`~langgraph.checkpoint.memory.MemorySaver`. Swap for
        ``PostgresSaver`` to survive restarts.
    debug
        Enables verbose agent tracing.

    Returns
    -------
    CompiledStateGraph
        Invoke with ``{"messages": [...], ...}`` and a
        ``{"configurable": {"thread_id": session_id}}`` config.
    """
    agents = specialists if specialists is not None else build_specialists(debug=debug)
    router = router or RouterAgent()

    builder = StateGraph(SupportState)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("supervisor", make_supervisor_node(router))
    builder.add_node("synthesize", make_synthesis_node())

    available_routes: list[str] = []
    for route in SPECIALIST_ROUTES:
        agent = agents.get(route)
        if agent is None:
            _LOGGER.warning("No agent registered for route '%s'; skipping node", route)
            continue
        builder.add_node(route, make_agent_node(route, agent))
        available_routes.append(route)

    builder.add_edge(START, "guardrail")

    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"supervisor": "supervisor", "synthesize": "synthesize"},
    )

    supervisor_targets: dict[str, str] = {route: route for route in available_routes}
    supervisor_targets["synthesize"] = "synthesize"
    builder.add_conditional_edges(
        "supervisor", route_after_supervisor, supervisor_targets
    )

    # Every specialist returns to the supervisor, which either dispatches the
    # next pending route or moves on to synthesis.
    for route in available_routes:
        builder.add_edge(route, "supervisor")

    builder.add_edge("synthesize", END)

    graph = builder.compile(checkpointer=checkpointer or MemorySaver())
    _LOGGER.info("Support graph compiled with routes: %s", ", ".join(available_routes))
    return graph
