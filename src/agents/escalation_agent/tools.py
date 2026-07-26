"""Tool functions available to the Escalation Agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def assess_escalation_risk(message: str, unresolved_attempts: int = 0) -> str:
    """Assess whether a message should be escalated to a human agent."""
    text = message.lower()

    high_risk_terms = [
        "lawyer",
        "legal",
        "fraud",
        "scam",
        "chargeback",
        "caught fire",
        "unsafe",
        "dangerous",
        "human",
        "manager",
    ]
    frustration_terms = ["angry", "terrible", "worst", "ridiculous", "unacceptable"]

    risk_score = 0
    matched_terms: list[str] = []

    for term in high_risk_terms:
        if term in text:
            risk_score += 3
            matched_terms.append(term)

    for term in frustration_terms:
        if term in text:
            risk_score += 1
            matched_terms.append(term)

    risk_score += max(0, unresolved_attempts)
    should_escalate = risk_score >= 4

    return json.dumps(
        {
            "should_escalate": should_escalate,
            "risk_score": risk_score,
            "matched_terms": matched_terms,
        },
        indent=2,
    )


@tool
def generate_handoff_summary(customer_message: str, context_summary: str = "") -> str:
    """Generate a concise handoff summary for a human support agent."""
    payload = {
        "handoff_required": True,
        "customer_message": customer_message.strip(),
        "context_summary": context_summary.strip(),
        "recommended_queue": "human_escalations",
    }
    return json.dumps(payload, indent=2)
