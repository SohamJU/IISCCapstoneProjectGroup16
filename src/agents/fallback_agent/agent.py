"""Fallback agent for out-of-scope queries."""

from __future__ import annotations

from src.agents.common import standard_out_of_scope_message, validate_user_input


class FallbackAgent:
    """Simple fallback agent for unsupported intents."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id

    def chat(self, user_message: str) -> str:
        """Return a warmer fallback response for unsupported requests."""
        ok, error = validate_user_input(user_message)
        if not ok:
            return error
        return standard_out_of_scope_message(user_message)

    def reset_memory(self) -> None:
        """No-op for API compatibility."""
        return None
