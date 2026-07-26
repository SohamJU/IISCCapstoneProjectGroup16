"""Intent Router Agent for routing user requests to specialist agents.

Design notes
------------
Two bugs in the previous implementation dominated answer quality:

1. **Keyword scanning over conversation history.** The orchestrator passed a
   history-enriched blob to ``classify_multi``, which ran the escalation
   regex over the whole thing. Once any turn contained "human", "escalate",
   "angry" etc. — including the assistant's own reply — *every* subsequent
   turn in that session hard-routed to escalation at 0.98 confidence. A
   question about product pricing would return a human-handoff summary.

   Fix: keyword safety checks run on the **current user message only**.
   History is supplied separately and used solely for reference resolution.

2. **Substring parsing of the LLM reply.** ``if label in text`` over raw model
   output matched 3-4 labels from a single prose sentence.

   Fix: ``with_structured_output(RouteDecision)`` — no text parsing at all,
   and the model reports its own confidence instead of a hardcoded 0.7.
"""

from __future__ import annotations

import re

from src.agents.common import validate_user_input
from src.agents.llm import get_router_llm
from src.agents.router.config import HISTORY_TURNS_FOR_ROUTING, MAX_ROUTES_PER_TURN
from src.agents.router.schemas import ROUTE_LABELS, RouteDecision
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)


# ── Fast-path keyword shortcuts ───────────────────────────────────────────
# These exist purely to skip an LLM round-trip on unambiguous requests. They
# are applied ONLY to the current user message, never to history.

# Hard safety override — explicit human/legal/safety requests must not wait on
# a model decision. Deliberately narrower than the old regex: "complaint" and
# "angry" alone are not escalation triggers, they are ordinary support tone.
_ESCALATION_RE = re.compile(
    r"\b(speak|talk|connect)\s+(to|with)\s+(a\s+)?(human|person|agent|representative|manager)\b"
    r"|\b(real|live)\s+(human|person|agent)\b"
    r"|\b(lawyer|attorney|legal action|sue you|lawsuit|fraud|scam|chargeback)\b"
    r"|\b(caught fire|electric shock|injured|unsafe|hazard)\b"
    r"|\bescalate\b",
    re.IGNORECASE,
)

# An explicit order ID plus a tracking verb is unambiguous.
_ORDER_FAST_RE = re.compile(
    r"\bORD-\d{6}\b.*\b(track|status|where|deliver|cancel)\b"
    r"|\b(track|status|where|deliver|cancel)\b.*\bORD-\d{6}\b",
    re.IGNORECASE,
)
_RETURN_FAST_RE = re.compile(r"\bRET-\d{6}\b", re.IGNORECASE)


_ROUTER_SYSTEM_PROMPT = """\
You are the intent router for an e-commerce customer support system. Choose \
which specialist agent(s) must handle the user's latest message.

Available routes:
- product         Product facts, specs, prices, comparisons, availability, reviews.
- order           Placing, tracking, status, delivery dates, cancelling orders.
- return          Return/refund eligibility, policy, creating or tracking returns.
- recommendation  Personalised suggestions based on the customer's history/profile.
- escalation      Customer wants a human, or there is a legal/safety/fraud concern.
- fallback        Greetings, small talk, or anything outside e-commerce support.

Rules:
- Pick exactly ONE route unless the user clearly asked for two separate things
  (e.g. "return my headphones and order the newer model" -> return, order).
- Never return more than two routes.
- Route on the user's LATEST message. Earlier turns are context for resolving
  references like "it", "that one", "the second option" — they do NOT decide
  the route. If the previous turn was about escalation but the new message is
  a normal product question, route to product.
- "Where is my order" is order, not product.
- Asking about the return *policy* is return. Asking whether an item is still
  cancellable is order.

Product vs recommendation — the most common confusion:
- **product**: any catalog search or lookup driven by stated criteria.
  "Show me wireless headphones under $250", "do you have 4K monitors",
  "what's the price of the Sony WH-1000XM4", "compare these two laptops".
  The customer told you what they want; you just have to find it.
- **recommendation**: only when the answer depends on WHO the customer is —
  their purchase history, profile or past behaviour. "What should I buy next",
  "recommend something for me", "what goes with my camera",
  "based on my orders, what would I like".
- When in doubt between the two, choose product. A stated budget or product
  category is a search, not a personalisation request.
- Only set needs_clarification when you genuinely cannot tell what the user
  wants. A slightly vague but answerable question should still be routed.
- Report honest confidence. Below 0.6 means you guessed.
"""


class RouterAgent:
    """Router agent that maps user queries to downstream specialist agents."""

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._llm = None
        self.init_error: str = ""

        if use_llm_fallback:
            try:
                self._llm = get_router_llm().with_structured_output(RouteDecision)
            except Exception as exc:  # pragma: no cover - config/network dependent
                # Surface loudly instead of silently degrading to keyword-only
                # routing, which was previously invisible in the UI.
                self.init_error = f"{type(exc).__name__}: {exc}"
                _LOGGER.error(
                    "Router LLM unavailable, falling back to keyword routing: %s",
                    self.init_error,
                )
                self.use_llm_fallback = False

    # ── Public API ────────────────────────────────────────────────────────

    def classify_multi(
        self,
        user_message: str,
        history: str = "",
    ) -> dict[str, object]:
        """Classify the current user message into one or more routes.

        Parameters
        ----------
        user_message
            The **raw** current user message. Must not include conversation
            history or injected context blocks — keyword safety checks run
            against this string and history would poison them.
        history
            Optional recent-turn transcript, used only to help the LLM resolve
            pronouns and references. Never keyword-scanned.

        Returns
        -------
        dict
            ``routes``, ``confidences``, ``reason``, and when the router wants
            a clarification first, ``clarifying_question``.
        """
        ok, _ = validate_user_input(user_message)
        if not ok:
            return self._decision(["fallback"], 0.0, "invalid_input")

        # Safety override — current message only.
        if _ESCALATION_RE.search(user_message):
            return self._decision(["escalation"], 0.98, "keyword_safety")

        # Cheap deterministic fast paths that skip the LLM round-trip.
        if _RETURN_FAST_RE.search(user_message):
            return self._decision(["return"], 0.95, "keyword_fastpath")
        if _ORDER_FAST_RE.search(user_message):
            return self._decision(["order"], 0.95, "keyword_fastpath")

        if not self.use_llm_fallback or self._llm is None:
            return self._decision(
                ["fallback"],
                0.30,
                f"router_llm_unavailable ({self.init_error})" if self.init_error else "router_llm_disabled",
            )

        prompt: list[tuple[str, str]] = [("system", _ROUTER_SYSTEM_PROMPT)]
        if history.strip():
            trimmed = self._trim_history(history)
            prompt.append(
                (
                    "human",
                    f"Recent conversation (context only, does not decide the route):\n{trimmed}",
                )
            )
        prompt.append(("human", f"Latest user message to route:\n{user_message}"))

        try:
            decision = self._llm.invoke(prompt)
        except Exception as exc:
            _LOGGER.error("Router LLM call failed: %s", exc)
            return self._decision(["fallback"], 0.0, f"router_error: {type(exc).__name__}")

        if not isinstance(decision, RouteDecision):  # defensive
            return self._decision(["fallback"], 0.0, "router_bad_payload")

        if decision.needs_clarification and decision.clarifying_question.strip():
            return {
                "routes": [],
                "confidences": {},
                "reason": f"needs_clarification: {decision.reasoning}",
                "clarifying_question": decision.clarifying_question.strip(),
            }

        routes = self._normalise_routes(decision.routes)
        if not routes:
            return self._decision(["fallback"], max(decision.confidence, 0.5), "llm_no_valid_route")

        return {
            "routes": routes,
            "confidences": {route: decision.confidence for route in routes},
            "reason": f"llm: {decision.reasoning}",
            "clarifying_question": "",
        }

    def classify(self, user_message: str, history: str = "") -> dict[str, object]:
        """Return single-route classification with confidence metadata."""
        multi = self.classify_multi(user_message, history=history)
        routes = multi.get("routes") or ["fallback"]
        confidences = multi.get("confidences") or {}
        first_route = str(routes[0]) if isinstance(routes, list) and routes else "fallback"
        first_conf = 0.0
        if isinstance(confidences, dict):
            first_conf = float(confidences.get(first_route, 0.0))
        return {
            "route": first_route,
            "confidence": first_conf,
            "reason": str(multi.get("reason", "")),
        }

    def route(self, user_message: str, history: str = "") -> str:
        """Return one route label for the given user request."""
        return str(self.classify(user_message, history=history).get("route", "fallback"))

    # ── Subtask scoping ───────────────────────────────────────────────────

    def build_subtask_message(self, route: str, user_message: str) -> str:
        """Deprecated. Kept for backwards compatibility only.

        Subtask scoping now happens via a ``SystemMessage`` built by
        :func:`src.agents.graph.nodes.build_scope_instruction`. The old
        approach concatenated the instruction into the *user* turn, so each
        agent's stored history showed the customer saying
        "Subtask for Product Agent: Focus only on ..." — which polluted every
        later turn in the conversation.
        """
        from src.agents.graph.nodes import build_scope_instruction

        return f"{user_message.strip()}\n\n{build_scope_instruction(route)}"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _decision(routes: list[str], confidence: float, reason: str) -> dict[str, object]:
        return {
            "routes": routes,
            "confidences": {route: confidence for route in routes},
            "reason": reason,
            "clarifying_question": "",
        }

    @staticmethod
    def _normalise_routes(raw_routes: list[str]) -> list[str]:
        """Validate, de-duplicate, and cap the model's route list."""
        routes: list[str] = []
        for item in raw_routes:
            name = str(item).strip().lower()
            if name in ROUTE_LABELS and name not in routes:
                routes.append(name)

        # A compound request that includes escalation should escalate first.
        if "escalation" in routes and routes[0] != "escalation":
            routes.remove("escalation")
            routes.insert(0, "escalation")

        # fallback is only meaningful on its own — drop it when a real
        # specialist was also selected.
        if len(routes) > 1 and "fallback" in routes:
            routes.remove("fallback")

        return routes[:MAX_ROUTES_PER_TURN]

    @staticmethod
    def _trim_history(history: str) -> str:
        """Keep only the most recent turns so routing stays cheap."""
        lines = [line for line in history.strip().splitlines() if line.strip()]
        return "\n".join(lines[-HISTORY_TURNS_FOR_ROUTING:])
