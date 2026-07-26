"""Order Agent — stateless LangGraph ReAct agent.

Conversation state is owned by the supervisor graph
(:mod:`src.agents.graph`), not by this class.
"""

from __future__ import annotations

from src.agents.base_agent import SpecialistAgent
from src.agents.order_agent.config import MAX_REACT_ITERATIONS
from src.agents.order_agent.prompts import build_system_prompt
from src.agents.order_agent.tools import (
    cancel_order,
    get_order_status,
    list_customer_orders,
    place_order,
    track_order,
)
from src.agents.shared_tools import lookup_support_policy


class OrderAgent(SpecialistAgent):
    """ReAct agent for order placement, tracking, status and cancellation."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        super().__init__(
            name="order",
            system_prompt=build_system_prompt(),
            tools=[
                get_order_status,
                track_order,
                list_customer_orders,
                cancel_order,
                place_order,
                # Order agents field "can I still cancel?" and "when will it
                # arrive?" constantly. Without this they had no grounded source
                # for shipping/cancellation rules despite being told never to
                # invent policy terms.
                lookup_support_policy,
            ],
            max_iterations=max_iterations,
            session_id=session_id,
            debug=debug,
        )
