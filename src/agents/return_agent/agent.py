"""Specialist agent for returns, refunds, and replacements."""

from __future__ import annotations


class ReturnAgent:
    """Handles return, refund, replacement, and cancellation requests."""

    name = "return"

    def chat(self, user_message: str) -> str:
        """Return a simple response for return-related queries."""
        return "I can help with returns, refunds, replacements, and cancellations."
