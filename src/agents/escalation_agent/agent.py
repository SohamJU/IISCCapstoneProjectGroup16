"""Handles escalation scenarios when specialist answers are insufficient."""

from __future__ import annotations


class EscalationAgent:
    """Escalates to a human-level support or broader fallback response."""

    name = "escalation"

    def chat(self, user_message: str) -> str:
        """Return a general escalation-style response."""
        return (
            "It looks like your request needs a specialist or human support team. "
            "I can connect you with a human agent or provide next-step guidance."
        )
