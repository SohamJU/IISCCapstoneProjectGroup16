"""Router-centric orchestrator for all support agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

from src.agents.deterministic_agent import DeterministicSupportAgent
from src.agents.escalation_agent import EscalationAgent
from src.agents.fallback_agent import FallbackAgent
from src.agents.orchestrator.config import LOW_CONFIDENCE_THRESHOLD
from src.agents.order_agent import OrderAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.return_agent import ReturnAgent
from src.agents.router import RouterAgent
from src.config import settings
from src.memory.session_manager import SessionManager

_CLOSE_CHAT_RE = re.compile(
    r"\b(close|end|stop|exit|quit|bye|goodbye|thanks,? bye|chat over)\b",
    re.IGNORECASE,
)


@dataclass
class OrchestratorResponse:
    """Unified orchestration response payload."""

    route: str
    confidence: float
    response: str
    routes: list[str] = field(default_factory=list)


class SupportOrchestrator:
    """Single-entry orchestrator that routes all requests through RouterAgent."""

    def __init__(
        self,
        use_llm_router_fallback: bool = True,
        deterministic_mode: bool | None = None,
        auto_fallback_on_agent_init_error: bool = True,
    ) -> None:
        self.router = RouterAgent(use_llm_fallback=use_llm_router_fallback)
        if deterministic_mode is None:
            deterministic_mode = (
                os.getenv("SUPPORT_DETERMINISTIC_MODE", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        self.deterministic_mode = deterministic_mode
        self.auto_fallback_on_agent_init_error = auto_fallback_on_agent_init_error
        self._agent_init_reason = ""
        self._sessions: dict[str, dict[str, object]] = {}
        self._session_manager = SessionManager()
        self.debug = getattr(settings, "DEBUG", False)

    def _get_or_create_session_agents(self, session_id: str) -> dict[str, object]:
        agents = self._sessions.get(session_id)
        if agents is not None:
            return agents

        if self.deterministic_mode:
            agents = {
                "product": DeterministicSupportAgent("product", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "order": DeterministicSupportAgent("order", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "return": DeterministicSupportAgent("return", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "recommendation": DeterministicSupportAgent("recommendation", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "escalation": DeterministicSupportAgent("escalation", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "fallback": DeterministicSupportAgent("fallback", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
            }
            self._sessions[session_id] = agents
            return agents

        try:
            agents = {
                "product": ProductAgent(session_id=session_id),
                "order": OrderAgent(session_id=session_id),
                "return": ReturnAgent(session_id=session_id),
                "recommendation": RecommendationAgent(session_id=session_id),
                "escalation": EscalationAgent(session_id=session_id),
                "fallback": FallbackAgent(session_id=session_id),
            }
        except Exception as exc:
            if not self.auto_fallback_on_agent_init_error:
                raise
            self.deterministic_mode = True
            self._agent_init_reason = f"agent init fallback: {type(exc).__name__}"
            agents = {
                "product": DeterministicSupportAgent("product", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "order": DeterministicSupportAgent("order", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "return": DeterministicSupportAgent("return", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "recommendation": DeterministicSupportAgent("recommendation", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "escalation": DeterministicSupportAgent("escalation", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
                "fallback": DeterministicSupportAgent("fallback", session_id=session_id, reason=self._agent_init_reason, debug=self.debug),
            }

        self._sessions[session_id] = agents
        return agents

    @staticmethod
    def _truncate(text: str, max_chars: int = 900) -> str:
        """Keep merged responses concise for POC UX."""
        compact = text.strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _merge_responses(
        self,
        route_payloads: list[tuple[str, float, str]],
    ) -> str:
        """Merge per-route outputs into one deterministic, readable response."""
        if not route_payloads:
            return "I wasn't able to generate a response. Please try again."

        if len(route_payloads) == 1:
            return self._truncate(route_payloads[0][2])

        routes = [route for route, _, _ in route_payloads]
        lines = [
            f"Handled your request in {len(route_payloads)} steps: {', '.join(routes)}.",
            "",
        ]

        for index, (route, confidence, response) in enumerate(route_payloads, start=1):
            section = self._truncate(response)
            lines.append(f"Step {index} - {route} (confidence {confidence:.2f})")
            lines.append(section)
            lines.append("")

        return "\n".join(lines).strip()

    def handle(self, user_message: str, session_id: str = "default") -> OrchestratorResponse:
        """Route a request and invoke one or more specialist agents sequentially."""
        if _CLOSE_CHAT_RE.search(user_message):
            self._session_manager.clear_session(session_id)
            self._sessions.pop(session_id, None)
            return OrchestratorResponse(
                route="closed",
                confidence=1.0,
                response="Chat session closed. You can start a new session with a new session ID.",
                routes=["closed"],
            )

        session = self._session_manager.get_or_create_session(session_id)
        history_context = session.build_context_window(limit=6)
        if history_context:
            enriched_message = f"Conversation history:\n{history_context}\n\nLatest user message: {user_message}"
        else:
            enriched_message = user_message

        if self.debug:
            print(f"\n=== DEBUG WORKFLOW START ===")
            print(f"session_id={session_id}")
            print(f"user_message={user_message}")

        multi = self.router.classify_multi(enriched_message)
        raw_routes = multi.get("routes", ["fallback"])
        route_confidences = multi.get("confidences", {})

        routes: list[str] = []
        if isinstance(raw_routes, list):
            for item in raw_routes:
                route_name = str(item)
                if route_name in {
                    "product",
                    "order",
                    "return",
                    "recommendation",
                    "escalation",
                    "fallback",
                } and route_name not in routes:
                    routes.append(route_name)

        if not routes:
            routes = ["fallback"]

        if self.debug:
            print(f"route_decision={routes}")
            print(f"route_confidences={route_confidences}")

        agents = self._get_or_create_session_agents(session_id)
        executed_routes: list[str] = []
        executed_confidences: list[float] = []
        route_payloads: list[tuple[str, float, str]] = []

        for route in routes:
            confidence = 0.0
            if isinstance(route_confidences, dict):
                confidence = float(route_confidences.get(route, 0.0))

            effective_route = route
            if (
                route in {"product", "order", "return", "recommendation"}
                and confidence < LOW_CONFIDENCE_THRESHOLD
            ):
                effective_route = "escalation"

            agent = agents.get(effective_route)
            if agent is None:
                effective_route = "fallback"
                agent = agents["fallback"]

            scoped_message = self.router.build_subtask_message(effective_route, enriched_message)
            if self.debug:
                print(f"calling_agent={effective_route}")
                print(f"agent_input={scoped_message}")

            agent_response = agent.chat(scoped_message)  # type: ignore[attr-defined]
            if self.debug:
                print(f"agent_output[{effective_route}]={agent_response}")
            executed_routes.append(effective_route)
            executed_confidences.append(confidence)
            route_payloads.append((effective_route, confidence, agent_response))

        final_route = executed_routes[0] if executed_routes else "fallback"
        final_confidence = min(executed_confidences) if executed_confidences else 0.0
        final_response = self._merge_responses(route_payloads)
        if not final_response:
            final_response = "I wasn't able to generate a response. Please try again."

        self._session_manager.append_turn(
            session_id,
            role="user",
            text=user_message,
        )
        self._session_manager.append_turn(
            session_id,
            role="assistant",
            text=final_response,
        )

        if self.debug:
            print(f"final_response={final_response}")
            print("=== DEBUG WORKFLOW END ===\n")

        return OrchestratorResponse(
            route=final_route,
            confidence=final_confidence,
            response=final_response,
            routes=executed_routes,
        )
