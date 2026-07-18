"""Deterministic fallback agents for core POC execution without LLM dependencies."""

from __future__ import annotations

import re


_ORDER_ID_RE = re.compile(r"ORD-\d{6}", re.IGNORECASE)
_RETURN_ID_RE = re.compile(r"RET-\d{6}", re.IGNORECASE)


class DeterministicSupportAgent:
    """Simple route-scoped deterministic agent used as runtime fallback."""

    def __init__(self, route: str, session_id: str = "default", reason: str = "", debug: bool = False) -> None:
        self.route = route
        self.session_id = session_id
        self.reason = reason.strip()
        self.debug = debug

    def chat(self, user_message: str) -> str:
        """Return a deterministic route-specific response."""
        prefix = "[deterministic-mode] "
        if self.reason:
            prefix += f"({self.reason}) "

        if self.debug:
            print(f"[debug] deterministic_agent route={self.route} input={user_message}")

        message = user_message.strip()
        if self.route == "order":
            order_match = _ORDER_ID_RE.search(message)
            if order_match:
                return (
                    f"{prefix}I detected order {order_match.group(0).upper()}. "
                    "I can help with tracking/status/cancellation once full LLM mode is available."
                )
            return (
                f"{prefix}For order support, share an order ID like ORD-000123 and whether you want "
                "status, tracking, or cancellation."
            )

        if self.route == "return":
            return_match = _RETURN_ID_RE.search(message)
            order_match = _ORDER_ID_RE.search(message)
            if return_match:
                return (
                    f"{prefix}I detected return request {return_match.group(0).upper()}. "
                    "I can help check status once full LLM mode is available."
                )
            if order_match:
                return (
                    f"{prefix}I detected order {order_match.group(0).upper()}. "
                    "For returns, also provide order_item_id (OI-xxxxxxx) and reason."
                )
            return (
                f"{prefix}For return/refund help, provide order_id, order_item_id, and reason."
            )

        if self.route == "recommendation":
            return (
                f"{prefix}I can provide personalized recommendations after you share customer_id, "
                "budget, and preferred category."
            )

        if self.route == "product":
            return (
                f"{prefix}I can help with product facts, specs, comparisons, and review questions. "
                "Please specify product name or product_id."
            )

        if self.route == "escalation":
            return (
                f"{prefix}I am escalating this case to a human support specialist due to "
                "low confidence or high-risk indicators."
            )

        return (
            f"{prefix}I can only help with products, orders, returns, recommendations, and escalation."
        )

    def reset_memory(self) -> None:
        """No-op for API compatibility."""
        return None
