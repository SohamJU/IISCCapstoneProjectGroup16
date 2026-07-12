"""Tool definitions for the Escalation Agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config.data import KNOWLEDGE_BASE_DIR


@tool
def retrieve_knowledge(query: str) -> str:
    """Retrieve related policy or knowledge base content for escalation decisions."""
    # Placeholder implementation. Replace this with a real RAG retriever.
    return (
        f"No retrieval backend is configured yet."
        f" Received query: {query}"
    )


@tool
def classify_escalation(issue_description: str) -> str:
    """Classify the escalation destination based on issue type."""
    normalized = issue_description.lower()
    if any(keyword in normalized for keyword in ["refund", "billing", "charge", "payment"]):
        return "Billing"
    if any(keyword in normalized for keyword in ["technical", "bug", "error", "not working", "connect", "login"]):
        return "Technical Support"
    if any(keyword in normalized for keyword in ["complaint", "manager", "supervisor", "escalate", "policy exception"]):
        return "Management"
    return "General Support"
