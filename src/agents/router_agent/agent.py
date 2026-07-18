"""Prompt-based router agent for product, order, and return conversations.

This router does one thing well: inspect a user message with a prompt and return
an explicit routing decision in a standard structure that downstream agents can
consume.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.agents.llm import get_llm


@dataclass(frozen=True)
class RouteDecision:
    """Structured routing result for a user message."""

    target_agent: str
    confidence: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRouteDefinition:
    """Definition for a routable specialist agent."""

    name: str
    description: str


class RouterAgent:
    """Route user intent to the most appropriate specialist agent."""

    def __init__(
        self,
        session_id: str = "default",
        default_agent: str = "escalation",
        llm: Optional[Any] = None,
        use_llm_fallback: bool = True,
        debug: bool = False,
    ) -> None:
        self.session_id = session_id
        self.default_agent = default_agent
        self.debug = debug
        self.use_llm_fallback = use_llm_fallback
        self._registry: dict[str, AgentRouteDefinition] = {}
        self._register_builtin_agents()
        self._llm = llm if llm is not None else self._create_default_llm()

    def route(self, user_message: str) -> RouteDecision:
        """Return a structured routing decision for the given user message."""
        if not isinstance(user_message, str) or not user_message.strip():
            fallback_agent = "escalation" if "escalation" in self._registry else self.default_agent
            return RouteDecision(
                target_agent=fallback_agent,
                confidence=0.2,
                reason="No usable user message was provided.",
                metadata={"strategy": "empty"},
            )

        if self.use_llm_fallback and self._llm is not None:
            try:
                decision = self._route_with_llm(user_message)
                if decision is not None:
                    return decision
            except Exception as exc:  # pragma: no cover - defensive fallback
                if self.debug:
                    print(f"LLM routing failed: {exc}")

        return self._route_with_rules(user_message)

    def chat(self, user_message: str) -> RouteDecision:
        """Compatibility wrapper around route()."""
        return self.route(user_message)

    def register_agent(self, name: str, description: str) -> None:
        """Register a routing target."""
        self._registry[name] = AgentRouteDefinition(name=name, description=description)

    def list_agents(self) -> list[str]:
        """Return the registered agent names."""
        return sorted(self._registry)

    def _route_with_llm(self, user_message: str) -> Optional[RouteDecision]:
        """Use a prompt to decide which agent should handle the request."""
        if self._llm is None:
            return None

        prompt = self._build_routing_prompt(user_message)
        response = self._llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        if not isinstance(content, str):
            content = str(content)

        parsed = self._extract_json(content)
        if not parsed:
            return None

        agent_name = str(parsed.get("agent", "")).strip().lower()
        if agent_name not in self._registry:
            return None

        confidence = parsed.get("confidence", 0.75)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.75

        return RouteDecision(
            target_agent=agent_name,
            confidence=min(0.99, max(0.25, confidence_value)),
            reason=str(parsed.get("reason", "LLM-based routing")),
            metadata={"strategy": "llm"},
        )

    def _route_with_rules(self, user_message: str) -> RouteDecision:
        """Fallback routing when the LLM is unavailable."""
        normalized = self._normalize(user_message)
        cues = {
            "product": ["recommend", "catalog", "price", "compare", "review", "laptop", "phone", "tablet", "buy"],
            "order": ["order", "ship", "shipment", "delivery", "track", "status", "where is", "arrived"],
            "return": ["return", "refund", "replacement", "exchange", "broken", "damaged", "defective", "cancel"],
        }

        scores = {name: 0 for name in self._registry}
        for agent_name, terms in cues.items():
            for term in terms:
                if term in normalized:
                    scores[agent_name] += 1

        best_agent = self.default_agent
        best_score = -1
        for agent_name, score in scores.items():
            if score > best_score:
                best_agent = agent_name
                best_score = score

        if best_score < 1:
            fallback_agent = "escalation" if "escalation" in self._registry else self.default_agent
            return RouteDecision(
                target_agent=fallback_agent,
                confidence=0.2,
                reason="No strong intent signals were found; routing to escalation.",
                metadata={"strategy": "default"},
            )

        return RouteDecision(
            target_agent=best_agent,
            confidence=min(0.9, 0.45 + best_score * 0.1),
            reason=f"Fallback heuristic matched {best_agent} from the message.",
            metadata={"strategy": "fallback"},
        )

    def _register_builtin_agents(self) -> None:
        self.register_agent(
            name="product",
            description="Product discovery, recommendations, comparisons, pricing, and availability.",
        )
        self.register_agent(
            name="order",
            description="Order status, tracking, shipment, delivery, and ETA questions.",
        )
        self.register_agent(
            name="return",
            description="Returns, refunds, replacements, exchanges, cancellations, and damaged items.",
        )
        self.register_agent(
            name="escalation",
            description="Unsupported, ambiguous, or complex issues that need human or specialist review.",
        )

    def _create_default_llm(self) -> Optional[Any]:
        """Create the shared LLM client if it is available."""
        try:
            return get_llm()
        except Exception:
            return None

    def _build_routing_prompt(self, user_message: str) -> str:
        """Create a clear prompt that asks the LLM to pick one specialist agent."""
        agents = "\n".join(
            f"- {name}: {definition.description}" for name, definition in self._registry.items()
        )
        return (
            "You are a routing classifier for a customer support assistant. "
            "Your job is to understand the user's request and identify which specialist agent should handle it.\n\n"
            "Choose exactly one agent from the list below.\n"
            f"{agents}\n\n"
            "Return valid JSON only with these exact keys: 'agent', 'confidence', 'reason'.\n"
            "The confidence should be a number between 0.0 and 1.0.\n\n"
            f"User message: {user_message}"
        )

    @staticmethod
    def _normalize(message: str) -> str:
        lowered = message.lower()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            return json.loads(content[start : end + 1])
        except Exception:
            return {}
