"""Shared guardrail and SQL safety helpers for agent packages."""

from __future__ import annotations

import re
from typing import Any


_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_INJECTION_PATTERN = re.compile(
    r"(?i)(ignore previous instructions|system prompt|developer message|jailbreak|bypass)",
)


def standard_out_of_scope_message(user_message: str | None = None) -> str:
    """Return a warmer fallback response for unsupported or non-support requests."""
    message = (user_message or "").strip()
    lowered = message.lower()

    if lowered in {"hi", "hello", "hey", "hey there", "hi there"}:
        return (
            "Hi! I’m here to help with e-commerce support questions about products, "
            "orders, returns, recommendations, and escalation. What would you like help with?"
        )

    if any(token in lowered for token in ["thanks", "thank you", "good morning", "good afternoon", "good evening"]):
        return (
            "You’re welcome! I can help with product info, orders, returns, recommendations, "
            "or escalation support. What do you need today?"
        )

    return (
        "I can help with e-commerce support topics such as products, orders, returns, "
        "recommendations, and escalation. If you have a specific question, feel free to share it."
    )


def validate_user_input(user_message: str) -> tuple[bool, str]:
    """Apply lightweight input guardrails before sending prompts to the model."""
    trimmed = user_message.strip()
    if not trimmed:
        return False, "Please share your request so I can help."

    if len(trimmed) > 4000:
        return False, "Your message is too long. Please shorten it and try again."

    if _INJECTION_PATTERN.search(trimmed):
        return False, (
            "I can't follow instructions that try to override system rules. "
            "Please ask a normal product or support question."
        )

    return True, ""


def validate_agent_output(output_text: str) -> tuple[bool, str]:
    """Apply lightweight output guardrails before returning a response."""
    trimmed = output_text.strip()
    if not trimmed:
        return False, "I couldn't generate a response. Please try again."
    return True, ""


def reject_write_sql(sql_query: str) -> tuple[bool, str]:
    """Block write operations for read-only query tools."""
    if _WRITE_PATTERN.search(sql_query):
        return (
            False,
            "ERROR: Only SELECT queries are allowed. "
            "Write operations are blocked in this tool.",
        )
    return True, ""


def limit_rows(rows: list[dict[str, Any]], max_rows: int = 5) -> tuple[list[dict[str, Any]], str]:
    """Trim large result sets to keep token usage bounded."""
    total = len(rows)
    if total <= max_rows:
        return rows, ""

    footer = f"\n... (showing {max_rows} of {total} results)"
    return rows[:max_rows], footer