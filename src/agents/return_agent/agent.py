"""Return Agent — stateless LangGraph ReAct agent.

Conversation state is owned by the supervisor graph
(:mod:`src.agents.graph`), not by this class.
"""

from __future__ import annotations

from src.agents.base_agent import SpecialistAgent
from src.agents.return_agent.config import MAX_REACT_ITERATIONS
from src.agents.return_agent.prompts import build_system_prompt
from src.agents.return_agent.tools import (
    check_return_eligibility,
    create_return_request,
    get_return_status,
    list_order_items,
    lookup_return_policy,
)
from src.agents.shared_tools import lookup_support_policy


class ReturnAgent(SpecialistAgent):
    """ReAct agent for returns, refunds and exchange eligibility."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        super().__init__(
            name="return",
            system_prompt=build_system_prompt(),
            tools=[
                list_order_items,
                check_return_eligibility,
                create_return_request,
                get_return_status,
                lookup_return_policy,
                lookup_support_policy,
            ],
            max_iterations=max_iterations,
            session_id=session_id,
            debug=debug,
        )
