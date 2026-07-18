"""Intent Router Agent for routing user requests to specialist agents."""

from __future__ import annotations

import re

from src.agents.common import validate_user_input
from src.agents.llm import get_llm
from src.agents.router.config import ROUTE_LABELS


_ORDER_RE = re.compile(
    r"\b(order|track|tracking|shipment|deliver|delivery|cancel order|where is my order)\b",
    re.IGNORECASE,
)
_RETURN_RE = re.compile(
    r"\b(return|refund|replace|exchange|defective|wrong item|return policy)\b",
    re.IGNORECASE,
)
_RECOMMEND_RE = re.compile(
    r"\b(recommend|suggest|similar|cheaper|best for|goes with|also buy|alternative|instead of)\b",
    re.IGNORECASE,
)
_PRODUCT_RE = re.compile(
    r"\b(spec|specs|feature|compare|warranty|battery|product|price|rating)\b",
    re.IGNORECASE,
)
_ESCALATION_RE = re.compile(
    r"\b(human|manager|legal|lawyer|fraud|unsafe|angry|complaint|escalate)\b",
    re.IGNORECASE,
)
_CUSTOMER_ID_RE = re.compile(
    r"\b[A-Z0-9]{16,}\b",
    re.IGNORECASE,
)


class RouterAgent:
    """Router agent that maps user queries to a single downstream agent."""

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._llm = None
        if use_llm_fallback:
            try:
                self._llm = get_llm()
            except Exception:
                # Fall back to deterministic keyword routing when LLM is unavailable.
                self.use_llm_fallback = False

    def classify_multi(self, user_message: str) -> dict[str, object]:
        """Return multi-intent classification with ordered routes and confidences."""
        ok, _ = validate_user_input(user_message)
        if not ok:
            return {
                "routes": ["fallback"],
                "confidences": {"fallback": 0.0},
                "reason": "invalid_input",
            }

        # Safety override: escalation takes full precedence.
        if _ESCALATION_RE.search(user_message):
            return {
                "routes": ["escalation"],
                "confidences": {"escalation": 0.98},
                "reason": "keyword",
            }

        keyword_routes: list[tuple[str, float]] = []
        if _RETURN_RE.search(user_message):
            keyword_routes.append(("return", 0.95))
        if _ORDER_RE.search(user_message):
            keyword_routes.append(("order", 0.95))
        if _RECOMMEND_RE.search(user_message) or _CUSTOMER_ID_RE.search(user_message):
            keyword_routes.append(("recommendation", 0.92))
        if _PRODUCT_RE.search(user_message):
            keyword_routes.append(("product", 0.90))

        if keyword_routes:
            routes = [route for route, _ in keyword_routes]
            confidences = {route: score for route, score in keyword_routes}
            return {
                "routes": routes,
                "confidences": confidences,
                "reason": "keyword",
            }

        if not self.use_llm_fallback or self._llm is None:
            return {
                "routes": ["fallback"],
                "confidences": {"fallback": 0.45},
                "reason": "no_match",
            }

        prompt = (
            "You are an intent router for an e-commerce support system. "
            "Return one or more comma-separated labels from this set: "
            "product, order, return, recommendation, escalation, fallback. "
            "If query has multiple intents, return multiple labels in priority order. "
            f"User message: {user_message}"
        )
        response = self._llm.invoke(prompt)
        text = str(getattr(response, "content", response)).strip().lower()

        routes: list[str] = []
        for label in ["escalation", "return", "order", "recommendation", "product", "fallback"]:
            if label in text and label not in routes:
                routes.append(label)

        if not routes:
            routes = ["fallback"]
            return {
                "routes": routes,
                "confidences": {"fallback": 0.5},
                "reason": "llm_unknown",
            }

        confidences = {route: 0.7 for route in routes}
        return {
            "routes": routes,
            "confidences": confidences,
            "reason": "llm",
        }

    def classify(self, user_message: str) -> dict[str, object]:
        """Return route classification with confidence metadata."""
        multi = self.classify_multi(user_message)
        routes = multi.get("routes", ["fallback"])
        confidences = multi.get("confidences", {"fallback": 0.0})
        first_route = str(routes[0]) if isinstance(routes, list) and routes else "fallback"
        first_conf = 0.0
        if isinstance(confidences, dict):
            first_conf = float(confidences.get(first_route, 0.0))
        return {
            "route": first_route,
            "confidence": first_conf,
            "reason": str(multi.get("reason", "")),
        }

    def route(self, user_message: str) -> str:
        """Return one route label for the given user request."""
        return str(self.classify(user_message).get("route", "fallback"))

    def build_subtask_message(self, route: str, user_message: str) -> str:
        """Build a route-scoped subtask instruction from the full user query."""
        route = route.strip().lower()
        base = f"User request: {user_message.strip()}\n\n"

        if route == "order":
            return (
                base
                + "Subtask for Order Agent: Focus only on order actions/status/tracking/cancellation. "
                "Ask for missing order identifiers if needed."
            )
        if route == "return":
            return (
                base
                + "Subtask for Return Agent: Focus only on return/refund policy, eligibility, "
                "or creating/checking return requests. Ask for order_id/order_item_id if missing."
            )
        if route == "recommendation":
            return (
                base
                + "Subtask for Recommendation Agent: Provide personalized recommendations. "
                "Ask for customer_id or constraints if missing."
            )
        if route == "product":
            return (
                base
                + "Subtask for Product Agent: Focus only on product facts/specs/comparison/reviews. "
                "Do not handle order or return workflow in this step."
            )
        if route == "escalation":
            return (
                base
                + "Subtask for Escalation Agent: Assess escalation need and produce handoff summary "
                "or clear next-step guidance."
            )

        return base + "Subtask for Fallback Agent: Provide standard out-of-scope support guidance."
