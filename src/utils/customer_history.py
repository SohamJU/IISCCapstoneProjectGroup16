"""Utilities to retrieve and format customer conversation history for LLM context."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.data.session_persistence import get_customer_session_history
from src.memory.conversation_memory import ConversationTurn


def format_session_summary(
    session_data: dict[str, object],
    max_turns: int = 3,
) -> str:
    """
    Format a single session into a readable summary for LLM context.
    
    Args:
        session_data: Session dictionary from get_customer_session_history
        max_turns: Maximum number of turns to include in summary
    
    Returns:
        Formatted session summary string
    """
    session_id = session_data.get("session_id", "unknown")
    created_at = session_data.get("created_at", "unknown")
    turns = session_data.get("conversation_turns", [])

    # Limit to most recent turns
    recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns

    summary_lines = [
        f"[Session: {session_id}]",
        f"Date: {created_at}",
        "---",
    ]

    for turn in recent_turns:
        role = turn.get("role", "unknown").upper()
        text = turn.get("text", "")
        summary_lines.append(f"{role}: {text}")

    if len(turns) > max_turns:
        summary_lines.append(f"... ({len(turns) - max_turns} more turns)")

    return "\n".join(summary_lines)


def format_customer_history_context(
    customer_id: str,
    num_sessions: int = 3,
    max_turns_per_session: int = 3,
) -> Optional[str]:
    """
    Retrieve and format a customer's conversation history for LLM context.
    
    Args:
        customer_id: Customer identifier
        num_sessions: Number of recent sessions to retrieve (default: 3)
        max_turns_per_session: Max turns to show per session (default: 3)
    
    Returns:
        Formatted history context string or None if no history exists
    """
    sessions = get_customer_session_history(customer_id, limit=num_sessions)

    if not sessions:
        return None

    history_lines = [
        f"=== Customer {customer_id} - Recent Conversation History ===",
        "",
    ]

    for i, session_data in enumerate(sessions, 1):
        session_summary = format_session_summary(
            session_data,
            max_turns=max_turns_per_session,
        )
        history_lines.append(f"--- Previous Session {i} ---")
        history_lines.append(session_summary)
        history_lines.append("")

    return "\n".join(history_lines)


def enrich_message_with_history(
    user_message: str,
    customer_id: str,
    include_history: bool = True,
    num_sessions: int = 2,
    max_turns_per_session: int = 3,
) -> str:
    """
    Enrich a user message with customer history context for better LLM responses.
    
    Args:
        user_message: The current user message
        customer_id: Customer identifier
        include_history: Whether to include history (default: True)
        num_sessions: Number of previous sessions to include
        max_turns_per_session: Max turns per session in context
    
    Returns:
        Enriched message with history context prepended
    """
    if not include_history:
        return user_message

    history_context = format_customer_history_context(
        customer_id,
        num_sessions=num_sessions,
        max_turns_per_session=max_turns_per_session,
    )

    if not history_context:
        return user_message

    return f"{history_context}\n\n--- Current Interaction ---\nUser: {user_message}"


def get_customer_topics_from_history(customer_id: str) -> list[str]:
    """
    Extract topics/categories a customer has discussed from their history.
    
    Args:
        customer_id: Customer identifier
    
    Returns:
        List of topics mentioned in customer's conversations
    """
    sessions = get_customer_session_history(customer_id, limit=5)
    topics = set()

    keywords = {
        "product": ["product", "item", "thing", "model", "brand", "quality", "feature"],
        "order": ["order", "purchase", "bought", "ordered", "delivery", "shipped", "tracking"],
        "return": ["return", "refund", "exchange", "back", "money back", "issue"],
        "shipping": ["ship", "delivery", "fast", "slow", "arrive", "package", "tracking"],
        "price": ["price", "cost", "expensive", "cheap", "discount", "deal"],
        "complaint": ["broken", "damage", "issue", "problem", "wrong", "not working"],
        "recommendation": ["suggest", "recommend", "similar", "alternative", "other options"],
    }

    for session in sessions:
        turns = session.get("conversation_turns", [])
        for turn in turns:
            text = turn.get("text", "").lower()
            for topic, keywords_list in keywords.items():
                if any(keyword in text for keyword in keywords_list):
                    topics.add(topic)

    return list(topics)


def get_customer_sentiment_summary(customer_id: str) -> dict[str, object]:
    """
    Generate a summary of customer sentiment based on recent interactions.
    
    Args:
        customer_id: Customer identifier
    
    Returns:
        Dictionary with sentiment info and key patterns
    """
    sessions = get_customer_session_history(customer_id, limit=5)

    total_messages = 0
    issue_count = 0
    positive_keywords = ["thank", "great", "love", "perfect", "excellent", "amazing"]
    negative_keywords = ["broken", "issue", "problem", "bad", "terrible", "upset"]

    for session in sessions:
        turns = session.get("conversation_turns", [])
        for turn in turns:
            text = turn.get("text", "").lower()
            if turn.get("role") == "user":
                total_messages += 1

            if any(kw in text for kw in negative_keywords):
                issue_count += 1

    return {
        "customer_id": customer_id,
        "total_sessions": len(sessions),
        "total_interactions": total_messages,
        "issue_count": issue_count,
        "has_issues": issue_count > 0,
        "topics": get_customer_topics_from_history(customer_id),
    }
