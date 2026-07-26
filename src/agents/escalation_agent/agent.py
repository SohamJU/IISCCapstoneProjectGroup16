"""Escalation Agent — stateless LangGraph ReAct agent.

Conversation state is owned by the supervisor graph
(:mod:`src.agents.graph`), not by this class.
"""

from __future__ import annotations

from src.agents.base_agent import SpecialistAgent
from src.agents.escalation_agent.config import MAX_REACT_ITERATIONS
from src.agents.escalation_agent.prompts import build_system_prompt
from src.agents.escalation_agent.tools import (
    assess_escalation_risk,
    generate_handoff_summary,
)


class EscalationAgent(SpecialistAgent):
    """ReAct agent that assesses escalation risk and prepares handoffs."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        super().__init__(
            name="escalation",
            system_prompt=build_system_prompt(),
            tools=[assess_escalation_risk, generate_handoff_summary],
            max_iterations=max_iterations,
            session_id=session_id,
            debug=debug,
        )
