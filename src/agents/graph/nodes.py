"""Node implementations for the support supervisor graph.

Flow::

    START -> guardrail -> supervisor -> <specialist> -> supervisor -> ... -> synthesize -> END

The supervisor is re-entered after every specialist. That loop is what lets a
two-route turn share state: the first agent's findings land in ``facts`` and
are injected into the second agent's scope instruction.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.common import validate_user_input
from src.agents.graph.state import SupportState
from src.agents.llm import get_synthesis_llm
from src.agents.router import RouterAgent
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Identifiers worth carrying between agents within a turn.
_FACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "order_id": re.compile(r"\bORD-\d{6}\b", re.IGNORECASE),
    "return_id": re.compile(r"\bRET-\d{6}\b", re.IGNORECASE),
    "product_id": re.compile(r"\bB0[A-Z0-9]{8}\b"),
}

#: Routes below this confidence are treated as guesses. Unlike the old
#: constant this is now reachable, because the router reports genuine
#: confidence instead of a hardcoded 0.7.
LOW_CONFIDENCE_THRESHOLD = 0.45


# ══════════════════════════════════════════════════════════════════════════
# Scope instructions
# ══════════════════════════════════════════════════════════════════════════

_SCOPE_INSTRUCTIONS: dict[str, str] = {
    "order": (
        "SCOPE: Handle only the order-related part of this request — status, "
        "tracking, delivery estimates, placement or cancellation. Do not "
        "attempt returns, refunds or product research."
    ),
    "return": (
        "SCOPE: Handle only the return/refund part of this request — policy, "
        "eligibility, creating or checking a return. Do not place new orders."
    ),
    "product": (
        "SCOPE: Handle only the product-information part of this request — "
        "specs, prices, comparisons, availability. Do not run order or return "
        "workflows."
    ),
    "recommendation": (
        "SCOPE: Provide personalised product recommendations grounded in the "
        "customer's profile and purchase history."
    ),
    "escalation": (
        "SCOPE: Assess whether this needs a human, and either produce a handoff "
        "summary or give calm next-step guidance."
    ),
    "fallback": (
        "SCOPE: The request is outside e-commerce support. Give brief, friendly "
        "guidance on what you can help with."
    ),
}


def build_scope_instruction(
    route: str,
    customer_id: str | None = None,
    facts: dict[str, Any] | None = None,
) -> str:
    """Assemble the per-turn ``SystemMessage`` for a specialist.

    This replaces ``RouterAgent.build_subtask_message``, which concatenated
    the same guidance onto the *user's* message — so the agent's stored
    history showed the customer saying "Subtask for Product Agent: Focus only
    on product facts...". As a system message it steers the turn without ever
    entering the conversation record.
    """
    parts: list[str] = [_SCOPE_INSTRUCTIONS.get(route, _SCOPE_INSTRUCTIONS["fallback"])]

    if customer_id:
        parts.append(
            "CUSTOMER IDENTITY: The customer is signed in. Your tools already "
            "know who they are and operate only on this customer's own records "
            "— you do not pass a customer ID to them, and there is no way to "
            "look up anyone else's orders or returns. Never ask the customer "
            "for their customer ID or account details. If a tool says a record "
            "is not on this account, relay that plainly; do not retry it, do "
            "not guess other IDs, and never claim it belongs to someone else."
        )
    else:
        parts.append(
            "CUSTOMER IDENTITY: This session is NOT signed in. Order, return "
            "and account tools are unavailable and will refuse. You can still "
            "help with product search and general policy questions."
        )

    if facts:
        known = ", ".join(f"{key}={value}" for key, value in facts.items())
        parts.append(
            f"ALREADY ESTABLISHED THIS TURN: {known}. "
            f"Use these values directly — do not ask the customer to repeat them."
        )

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════════════════


def guardrail_node(state: SupportState) -> dict[str, Any]:
    """Validate the incoming message before any model call."""
    user_message = _latest_user_message(state)
    ok, error = validate_user_input(user_message)
    if not ok:
        return {"direct_response": error}
    return {"direct_response": ""}


def make_supervisor_node(router: RouterAgent):
    """Build the supervisor node bound to a router instance.

    On first entry for a turn it classifies the message and seeds
    ``pending_routes``. On every re-entry it simply reports what is left, so
    the conditional edge can dispatch the next specialist or finish.
    """

    def supervisor_node(state: SupportState) -> dict[str, Any]:
        # Re-entry after a specialist ran — routing already decided.
        if state.get("routes"):
            return {}

        user_message = _latest_user_message(state)
        history = _render_history(state, limit=6)

        decision = router.classify_multi(user_message, history=history)

        clarifying = str(decision.get("clarifying_question", "")).strip()
        if clarifying:
            # The router itself decided the request is unroutable as stated.
            # Ask once rather than guessing and producing a wrong answer.
            return {
                "routes": [],
                "pending_routes": [],
                "direct_response": clarifying,
                "route_reason": str(decision.get("reason", "")),
                "confidence": 0.0,
            }

        reason = str(decision.get("reason", ""))

        # When the router itself failed on provider quota, routing to the
        # fallback agent would emit a cheerful "I can help with products,
        # orders..." reply that hides an operational failure and invites the
        # customer to retry into the same wall.
        if _is_rate_limit(reason):
            message = (
                "I'm temporarily over capacity and can't complete that request "
                "right now. Please try again in a few minutes."
            )
            _LOGGER.error("Routing unavailable due to provider quota: %s", reason)
            return {
                "routes": [],
                "pending_routes": [],
                "direct_response": message,
                "route_reason": reason,
                "confidence": 0.0,
                "warnings": [reason],
            }

        routes = [str(route) for route in (decision.get("routes") or ["fallback"])]
        confidences = decision.get("confidences") or {}
        confidence = 0.0
        if isinstance(confidences, dict) and confidences:
            confidence = float(min(confidences.values()))

        warnings: list[str] = []
        if confidence < LOW_CONFIDENCE_THRESHOLD and routes != ["fallback"]:
            warnings.append(f"low routing confidence ({confidence:.2f})")

        _LOGGER.info(
            "route=%s confidence=%.2f reason=%s",
            routes,
            confidence,
            str(decision.get("reason", ""))[:120],
        )

        return {
            "routes": routes,
            "pending_routes": list(routes),
            "confidence": confidence,
            "route_reason": str(decision.get("reason", "")),
            "warnings": warnings,
        }

    return supervisor_node


def make_agent_node(route: str, agent: Any):
    """Build a graph node that runs one specialist agent."""

    def agent_node(state: SupportState) -> dict[str, Any]:
        pending = list(state.get("pending_routes") or [])
        if route in pending:
            pending.remove(route)

        scope = build_scope_instruction(
            route,
            customer_id=state.get("customer_id"),
            facts=state.get("facts") or {},
        )

        # customer_id goes in as a real argument, not just as prose inside
        # `scope`. The prose tells the model who it is talking to; this is what
        # the tools actually enforce against.
        result = agent.run(
            state.get("messages") or [],
            scope_instruction=scope,
            customer_id=state.get("customer_id"),
        )

        if not result.ok:
            _LOGGER.warning("[%s] produced no usable answer: %s", route, result.error)
            return {
                "pending_routes": pending,
                "executed_routes": [route],
                "warnings": [f"{route} agent failed: {result.error}"[:200]],
            }

        return {
            "pending_routes": pending,
            "executed_routes": [route],
            "agent_outputs": [(route, result.text)],
            "tools_used": result.tool_calls,
            "facts": _extract_facts(result.text),
        }

    return agent_node


def make_synthesis_node():
    """Build the node that turns agent outputs into one customer-facing reply.

    Replaces ``SupportOrchestrator._merge_responses``, which emitted::

        Handled your request in 2 steps: return, order.

        Step 1 - return (confidence 0.70)
        <900 chars, truncated mid-sentence>

        Step 2 - order (confidence 0.70)
        ...

    Single-agent turns pass straight through untouched — no extra LLM call.
    """
    llm = None

    def synthesis_node(state: SupportState) -> dict[str, Any]:
        nonlocal llm

        direct = str(state.get("direct_response") or "").strip()
        if direct:
            return {"final_response": direct, "messages": [AIMessage(content=direct)]}

        outputs = list(state.get("agent_outputs") or [])

        if not outputs:
            warnings = [str(w) for w in (state.get("warnings") or [])]

            # A quota/rate-limit failure is not the customer's fault and is not
            # a rephrasing problem. Saying "could you rephrase?" sends them into
            # a loop that can never succeed, and hides an operational issue.
            if any(_is_rate_limit(w) for w in warnings):
                message = (
                    "I'm temporarily over capacity and can't complete that "
                    "request right now. Please try again in a few minutes."
                )
            else:
                detail = f" ({warnings[0]})" if warnings else ""
                message = (
                    "I wasn't able to put together an answer for that"
                    f"{detail}. Could you rephrase, or tell me a bit more "
                    "about what you need?"
                )
            return {"final_response": message, "messages": [AIMessage(content=message)]}

        # Single specialist — its answer is already the whole reply.
        if len(outputs) == 1:
            text = outputs[0][1].strip()
            return {"final_response": text, "messages": [AIMessage(content=text)]}

        # Multiple specialists — merge into one voice.
        if llm is None:
            llm = get_synthesis_llm()

        user_message = _latest_user_message(state)
        sections = "\n\n".join(
            f"### {route} specialist\n{text}" for route, text in outputs
        )

        prompt = [
            SystemMessage(
                content=(
                    "You are the single customer-facing voice of an e-commerce "
                    "support team. Below are answers from two internal "
                    "specialists who each handled part of the customer's "
                    "request.\n\n"
                    "Merge them into ONE reply that reads as if a single person "
                    "wrote it.\n"
                    "- Address every part of what the customer asked.\n"
                    "- Preserve all concrete facts exactly: order IDs, prices, "
                    "dates, product names, policy terms. Never alter a number.\n"
                    "- Do not mention agents, routing, steps, confidence scores "
                    "or internal structure.\n"
                    "- Drop duplicated pleasantries and repeated information.\n"
                    "- If both specialists asked for the same missing detail, "
                    "ask for it once.\n"
                    "- Keep it tight and scannable. Use short paragraphs or "
                    "bullets where it helps."
                )
            ),
            HumanMessage(
                content=(
                    f"Customer asked:\n{user_message}\n\n"
                    f"Specialist answers:\n{sections}\n\n"
                    "Write the unified reply."
                )
            ),
        ]

        try:
            merged = str(llm.invoke(prompt).content).strip()
        except Exception as exc:
            _LOGGER.error("Synthesis failed, falling back to concatenation: %s", exc)
            merged = "\n\n".join(text for _, text in outputs)

        if not merged:
            merged = "\n\n".join(text for _, text in outputs)

        return {"final_response": merged, "messages": [AIMessage(content=merged)]}

    return synthesis_node


# ══════════════════════════════════════════════════════════════════════════
# Routing helpers used by conditional edges
# ══════════════════════════════════════════════════════════════════════════


def route_after_guardrail(state: SupportState) -> str:
    """Skip straight to synthesis when the guardrail produced a reply."""
    return "synthesize" if state.get("direct_response") else "supervisor"


def route_after_supervisor(state: SupportState) -> str:
    """Dispatch the next pending specialist, or synthesise when done."""
    if state.get("direct_response"):
        return "synthesize"

    pending = state.get("pending_routes") or []
    if not pending:
        return "synthesize"
    return pending[0]


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════


def _latest_user_message(state: SupportState) -> str:
    """Return the most recent human message text."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _render_history(state: SupportState, limit: int = 6) -> str:
    """Render recent turns as a compact transcript for the router."""
    messages = state.get("messages") or []
    # Exclude the current user turn — the router receives that separately.
    prior = messages[:-1][-limit:]

    lines: list[str] = []
    for message in prior:
        if isinstance(message, HumanMessage):
            lines.append(f"user: {message.content}")
        elif isinstance(message, AIMessage) and str(message.content).strip():
            lines.append(f"assistant: {str(message.content)[:400]}")
    return "\n".join(lines)


def _is_rate_limit(message: str) -> bool:
    """Detect provider quota/rate-limit failures in a warning string."""
    lowered = message.lower()
    return (
        "ratelimit" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
        or "429" in lowered
        or "quota" in lowered
    )


def _extract_facts(text: str) -> dict[str, Any]:
    """Pull shareable identifiers out of an agent's answer.

    Cheap and deliberately conservative — only well-formed IDs are carried
    forward, so a later specialist in the same turn does not re-ask for an
    order number the previous one already resolved.
    """
    facts: dict[str, Any] = {}
    for key, pattern in _FACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            facts[key] = match.group(0).upper()
    return facts
