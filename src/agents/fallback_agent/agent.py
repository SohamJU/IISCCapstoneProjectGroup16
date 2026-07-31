"""Fallback agent for greetings, small talk and out-of-scope queries.

Deliberately LLM-free: these replies must be instant, free and identical
every time. It implements the same ``run``/``chat`` surface as
:class:`src.agents.base_agent.SpecialistAgent` so the supervisor graph can
dispatch to it without special-casing.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage

from src.agents.base_agent import AgentResult
from src.agents.common import standard_out_of_scope_message, validate_user_input


class FallbackAgent:
    """Simple deterministic agent for unsupported intents."""

    name = "fallback"

    def __init__(self, session_id: str = "default", debug: bool = False) -> None:
        self.session_id = session_id
        self.debug = debug

    def run(
        self,
        messages: list[AnyMessage],
        scope_instruction: str = "",
        history_window: int = 0,
        customer_id: str | None = None,
    ) -> AgentResult:
        """Return a canned in-scope guidance reply for the latest user turn.

        ``customer_id`` is accepted and ignored: this agent touches no
        customer data, but the supervisor passes the same arguments to every
        specialist. This class duck-types :class:`SpecialistAgent` rather than
        inheriting from it, so signature changes there must be mirrored here —
        omitting this parameter made every fallback-routed turn (including a
        bare "Hi") raise TypeError and surface as a generic error.
        """
        latest = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                latest = str(message.content)
                break

        return AgentResult(name=self.name, text=standard_out_of_scope_message(latest))

    def chat(self, user_message: str, customer_id: str | None = None) -> str:
        """Return a warmer fallback response for unsupported requests."""
        ok, error = validate_user_input(user_message)
        if not ok:
            return error
        return standard_out_of_scope_message(user_message)

    def reset_memory(self) -> None:
        """No-op for API compatibility."""
        return None
