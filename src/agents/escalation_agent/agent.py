"""Escalation Agent — monitors messages and summarizes tickets for human handoff."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agents.llm import get_llm
from src.agents.escalation_agent.config import MAX_REACT_ITERATIONS
from src.agents.escalation_agent.prompts import build_system_prompt
from src.agents.escalation_agent.tools import classify_escalation, retrieve_knowledge


class EscalationAgent:
    """LangGraph-based ReAct agent that detects issues requiring human support."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.debug = debug

        self._system_prompt = build_system_prompt()
        self._tools = [retrieve_knowledge, classify_escalation]
        self._checkpointer = MemorySaver()
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's assessment or handoff summary."""
        config = {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": self.max_iterations * 2,
        }

        if self.debug:
            print(f"\n{'='*20} DEBUG: Escalation Agent Started {'='*20}")
            final_content = "I wasn't able to generate a response. Please try again."
            for event in self._agent.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="values",
            ):
                msgs = event.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    last_msg.pretty_print()
                    final_content = last_msg.content
            print(f"{'='*20} DEBUG: Escalation Agent Finished {'='*20}\n")
            return final_content

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "I wasn't able to generate a response. Please try again."

    def reset_memory(self) -> None:
        """Clear conversation history for this session."""
        self._checkpointer = MemorySaver()
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

    @staticmethod
    def render_handoff_summary(
        customer_intent: str,
        previous_actions: str,
        pain_points: str,
        rag_evidence: str,
        destination: str,
        reason: str,
    ) -> str:
        """Helper to create the escalation summary template."""
        from src.agents.escalation_agent.prompts import render_handoff_summary

        return render_handoff_summary(
            customer_intent=customer_intent,
            previous_actions=previous_actions,
            pain_points=pain_points,
            rag_evidence=rag_evidence,
            destination=destination,
            reason=reason,
        )
