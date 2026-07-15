"""Specialist agent for order-related conversations."""

from __future__ import annotations


class OrderAgent:
    """Handles order status, tracking, shipment, and delivery questions."""

    name = "order"

    def chat(self, user_message: str) -> str:
        """Return a simple response for order-related queries."""
        return (
            "I can help with order status, shipment tracking, and delivery updates."
        )
