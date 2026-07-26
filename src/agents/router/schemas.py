"""Structured-output schemas for the intent router.

The router used to ask the LLM for comma-separated labels and then scan the
raw reply with ``if label in text``. Any prose preamble ("this is a product
question, not an order or return issue") matched three labels at once, which
fanned the request out to three agents and produced a stitched-together,
incoherent answer.

Forcing the model into this Pydantic schema removes the parsing step
entirely — the model either returns valid routes or langchain retries.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RouteName = Literal[
    "product",
    "order",
    "return",
    "recommendation",
    "escalation",
    "fallback",
]

#: Canonical set of route labels, derived from the Literal above so the type
#: and the runtime validation set can never drift apart.
ROUTE_LABELS: frozenset[str] = frozenset(RouteName.__args__)


class RouteDecision(BaseModel):
    """A single routing decision produced by the router LLM."""

    routes: list[RouteName] = Field(
        description=(
            "The agents needed to fully answer the user, in execution order. "
            "Use exactly ONE route unless the user genuinely asked for two "
            "distinct things (e.g. 'return this and order a replacement'). "
            "Never return more than two."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your genuine confidence in this routing decision. Use a value "
            "below 0.6 when the request is ambiguous or you had to guess."
        ),
    )
    reasoning: str = Field(
        description="One short sentence explaining the choice. For logs, not the user.",
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True when the request is too vague to route at all and the user "
            "must be asked a clarifying question first."
        ),
    )
    clarifying_question: str = Field(
        default="",
        description=(
            "The single question to ask the user. Only set when "
            "needs_clarification is True."
        ),
    )
