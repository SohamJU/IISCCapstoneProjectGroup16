"""Tool definitions for the Escalation Agent."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.tools import tool

try:
    from src.config.data import KNOWLEDGE_BASE_DIR
except Exception:
    KNOWLEDGE_BASE_DIR = ROOT_DIR / "data" / "knowledge_base"


@tool
def retrieve_knowledge(query: str) -> str:
    """Retrieve related policy or knowledge base content for escalation decisions."""
    # Placeholder implementation. Replace this with a real RAG retriever.
    if not KNOWLEDGE_BASE_DIR.exists():
        return (
            "No retrieval backend is configured yet. "
            "Also could not find the local knowledge base directory."
        )

    return (
        "No retrieval backend is configured yet. "
        f"Received query: {query}. "
        f"Knowledge base is available at {KNOWLEDGE_BASE_DIR}."
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
