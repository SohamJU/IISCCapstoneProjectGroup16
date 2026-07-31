"""Router-centric orchestrator for all support agents.

This is now a thin adapter over the LangGraph supervisor in
:mod:`src.agents.graph`. It owns three things the graph does not:

* the public ``handle()`` API used by the CLI, Gradio app and tests,
* session close/reset semantics,
* mirroring turns into the ``customer_sessions`` table so history survives a
  process restart and can be shown per customer.

Everything else — routing, dispatch, shared state, synthesis — lives in the
graph.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.deterministic_agent import DeterministicSupportAgent
from src.agents.graph import build_specialists, build_support_graph, new_turn_state
from src.agents.router import RouterAgent
from src.config import settings
from src.data.session_persistence import initialize_sessions_table
from src.memory.persistent_session_manager import PersistentSessionManager
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Only treat a message as "close the chat" when that is the *whole* message.
# The previous pattern matched close|end|stop|exit|quit|bye anywhere in the
# text, so "when does the return window close?", "I want to stop the
# subscription" and "it should arrive by the end of the week" all silently
# terminated the customer's session mid-conversation.
_CLOSE_CHAT_RE = re.compile(
    r"^\s*(?:"
    r"bye|goodbye|good\s?bye|exit|quit|"
    r"(?:ok(?:ay)?[,\s]*)?(?:thanks?|thank\s+you)[,\s]*(?:bye|goodbye)|"
    r"that'?s\s+all(?:\s+thanks?)?|"
    r"close\s+(?:the\s+)?(?:chat|session)|"
    r"end\s+(?:the\s+)?(?:chat|session)"
    r")\s*[.!]*\s*$",
    re.IGNORECASE,
)


@dataclass
class OrchestratorResponse:
    """Unified orchestration response payload."""

    route: str
    confidence: float
    response: str
    routes: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False


class SupportOrchestrator:
    """Single-entry orchestrator backed by the LangGraph supervisor."""

    def __init__(
        self,
        use_llm_router_fallback: bool = True,
        deterministic_mode: bool | None = None,
        auto_fallback_on_agent_init_error: bool = True,
    ) -> None:
        try:
            initialize_sessions_table()
        except Exception as exc:
            # Still non-fatal — chat works without persistence — but no longer
            # silent. A bare `except: pass` here hid a broken upsert for the
            # entire life of the feature.
            _LOGGER.warning("Session table init failed, persistence disabled: %s", exc)

        if deterministic_mode is None:
            deterministic_mode = os.getenv(
                "SUPPORT_DETERMINISTIC_MODE", "false"
            ).strip().lower() in {"1", "true", "yes", "on"}

        self.deterministic_mode = bool(deterministic_mode)
        self.auto_fallback_on_agent_init_error = auto_fallback_on_agent_init_error
        self.degraded_reason = ""
        self.debug = getattr(settings, "DEBUG", False)

        self.router = RouterAgent(use_llm_fallback=use_llm_router_fallback)
        if self.router.init_error:
            self.degraded_reason = f"router LLM unavailable ({self.router.init_error})"

        self._session_manager = PersistentSessionManager(persist_to_db=True)
        self._graph = None
        self._deterministic_agents: dict[str, Any] = {}

        if self.deterministic_mode:
            self._build_deterministic_agents("SUPPORT_DETERMINISTIC_MODE enabled")
        else:
            self._build_graph()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        """Build the supervisor graph, degrading loudly on failure."""
        try:
            specialists = build_specialists(debug=self.debug)
            self._graph = build_support_graph(
                specialists=specialists,
                router=self.router,
                debug=self.debug,
            )
        except Exception as exc:
            if not self.auto_fallback_on_agent_init_error:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            # This path used to be completely invisible: a bad GROQ_API_KEY
            # swapped all six agents for canned deterministic strings and the
            # UI reported nothing, so "bad answers" looked like a model
            # quality problem rather than a config problem.
            _LOGGER.error(
                "Agent initialisation failed — falling back to DETERMINISTIC MODE. "
                "Answers will be canned until this is fixed. Cause: %s",
                reason,
            )
            self.deterministic_mode = True
            self._build_deterministic_agents(f"agent init failed: {reason}")

    def _build_deterministic_agents(self, reason: str) -> None:
        """Populate the LLM-free fallback agents."""
        self.degraded_reason = reason
        self._deterministic_agents = {
            route: DeterministicSupportAgent(route, reason=reason, debug=self.debug)
            for route in (
                "product",
                "order",
                "return",
                "recommendation",
                "escalation",
                "fallback",
            )
        }

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_degraded(self) -> bool:
        """True when the system is not running full LLM agents."""
        return self.deterministic_mode or bool(self.degraded_reason)

    @staticmethod
    def _thread_id(session_id: str, customer_id: str | None) -> str:
        """Namespace the graph thread by customer.

        The checkpointer was previously keyed on ``session_id`` alone. Because
        the UI holds the session ID constant while the customer dropdown
        changes, switching customer mid-session resumed the *previous*
        customer's message history — one customer could see another's
        conversation, and agents would act on the wrong person's orders.

        Scoping the thread key by customer makes that structurally impossible,
        regardless of what the caller passes as ``session_id``. Database
        persistence is already keyed on ``(customer_id, session_id)``, so this
        brings the in-memory graph state in line with it.
        """
        return f"{customer_id or 'anon'}::{session_id}"

    def handle(
        self,
        user_message: str,
        session_id: str = "default",
        customer_id: str | None = None,
    ) -> OrchestratorResponse:
        """Route a request through the graph and return the unified reply."""
        if _CLOSE_CHAT_RE.match(user_message.strip()):
            self.reset_session(session_id, customer_id=customer_id)
            return OrchestratorResponse(
                route="closed",
                confidence=1.0,
                response="Chat session closed. Start a new session any time.",
                routes=["closed"],
            )

        if self.deterministic_mode:
            return self._handle_deterministic(user_message, session_id, customer_id)

        assert self._graph is not None  # guaranteed by __init__

        self._seed_history_if_new(session_id, customer_id)

        config = {
            "configurable": {"thread_id": self._thread_id(session_id, customer_id)},
            "recursion_limit": 50,
        }
        state_delta = new_turn_state(user_message, session_id, customer_id)

        try:
            final_state = self._graph.invoke(state_delta, config=config)
        except Exception as exc:
            _LOGGER.error("Graph invocation failed: %s", exc, exc_info=True)
            return OrchestratorResponse(
                route="error",
                confidence=0.0,
                response=(
                    "Something went wrong while handling that request. "
                    "Please try again in a moment."
                ),
                routes=["error"],
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        response_text = str(final_state.get("final_response") or "").strip()
        if not response_text:
            response_text = "I wasn't able to generate a response. Please try again."

        executed = [str(route) for route in (final_state.get("executed_routes") or [])]
        routes = executed or [str(route) for route in (final_state.get("routes") or [])]
        warnings = [str(w) for w in (final_state.get("warnings") or [])]

        if not routes:
            if any("ratelimit" in w.lower() or "429" in w for w in warnings):
                routes = ["unavailable"]
            elif final_state.get("direct_response"):
                routes = ["clarify"]
            else:
                routes = ["fallback"]

        self._persist_turn(session_id, customer_id, user_message, response_text)

        return OrchestratorResponse(
            route=routes[0],
            confidence=float(final_state.get("confidence") or 0.0),
            response=response_text,
            routes=routes,
            tools_used=[str(t) for t in (final_state.get("tools_used") or [])],
            warnings=warnings,
            degraded=self.is_degraded,
        )

    def reset_session(self, session_id: str, customer_id: str | None = None) -> None:
        """Clear both graph state and stored history for a session."""
        self._session_manager.clear_session(session_id)
        if self._graph is not None:
            try:
                # Wipe the checkpointed message list for this thread.
                self._graph.update_state(
                    {
                        "configurable": {
                            "thread_id": self._thread_id(session_id, customer_id)
                        }
                    },
                    {"messages": [], "facts": {}, "agent_outputs": []},
                )
            except Exception as exc:
                _LOGGER.debug("Could not reset graph state for %s: %s", session_id, exc)

    # ── Deterministic mode ────────────────────────────────────────────────

    def _handle_deterministic(
        self,
        user_message: str,
        session_id: str,
        customer_id: str | None,
    ) -> OrchestratorResponse:
        """Serve a turn without any LLM calls."""
        route = self.router.route(user_message)
        agent = (
            self._deterministic_agents.get(route)
            or self._deterministic_agents["fallback"]
        )
        response_text = agent.chat(user_message)

        self._persist_turn(session_id, customer_id, user_message, response_text)

        return OrchestratorResponse(
            route=route,
            confidence=0.0,
            response=response_text,
            routes=[route],
            warnings=[self.degraded_reason] if self.degraded_reason else [],
            degraded=True,
        )

    # ── Persistence helpers ───────────────────────────────────────────────

    def _seed_history_if_new(self, session_id: str, customer_id: str | None) -> None:
        """Load stored turns into graph state the first time a session is seen.

        This replaces the old approach of prepending a ``CONVERSATION_HISTORY:``
        block to *every* user message — which duplicated context the agents
        already had and polluted the conversation record.
        """
        if self._graph is None:
            return

        config = {
            "configurable": {"thread_id": self._thread_id(session_id, customer_id)}
        }
        try:
            existing = self._graph.get_state(config)
            if existing.values.get("messages"):
                return  # graph already has this thread
        except Exception:
            pass

        session = self._session_manager.get_or_create_session(
            session_id, customer_id=customer_id
        )
        if not session.turns:
            return

        seeded = []
        for turn in session.get_recent_turns(limit=10):
            if turn.role == "user":
                seeded.append(HumanMessage(content=turn.text))
            elif turn.role == "assistant":
                seeded.append(AIMessage(content=turn.text))

        if seeded:
            try:
                self._graph.update_state(config, {"messages": seeded})
                _LOGGER.info(
                    "Seeded %d stored turns into session %s", len(seeded), session_id
                )
            except Exception as exc:
                _LOGGER.warning("Could not seed history for %s: %s", session_id, exc)

    def _persist_turn(
        self,
        session_id: str,
        customer_id: str | None,
        user_message: str,
        response_text: str,
    ) -> None:
        """Mirror the completed turn into the session store."""
        try:
            self._session_manager.append_turn(
                session_id, role="user", text=user_message, customer_id=customer_id
            )
            self._session_manager.append_turn(
                session_id,
                role="assistant",
                text=response_text,
                customer_id=customer_id,
            )
        except Exception as exc:
            _LOGGER.warning("Could not persist turn for %s: %s", session_id, exc)
