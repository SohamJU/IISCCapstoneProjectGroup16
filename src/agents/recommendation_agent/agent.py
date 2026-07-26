"""Recommendation Agent — stateless LangGraph ReAct agent.

Conversation state is owned by the supervisor graph
(:mod:`src.agents.graph`), not by this class.
"""

from __future__ import annotations

from src.agents.base_agent import SpecialistAgent
from src.agents.product_agent.tools import search_products
from src.agents.recommendation_agent.config import MAX_REACT_ITERATIONS
from src.agents.recommendation_agent.prompts import build_system_prompt
from src.agents.recommendation_agent.tools import (
    get_customer_order_history,
    get_customer_profile,
    recommend_for_customer,
)


class RecommendationAgent(SpecialistAgent):
    """ReAct agent producing personalised product recommendations."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        super().__init__(
            name="recommendation",
            system_prompt=build_system_prompt(),
            tools=[
                get_customer_profile,
                get_customer_order_history,
                recommend_for_customer,
                # Lets the agent widen beyond a customer's prior categories
                # instead of dead-ending when history is thin or absent.
                search_products,
            ],
            max_iterations=max_iterations,
            session_id=session_id,
            debug=debug,
        )
