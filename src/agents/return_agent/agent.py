"""Return Agent — LangGraph ReAct agent with memory."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agents.common import validate_agent_output, validate_user_input
from src.agents.llm import get_llm
from src.agents.return_agent.config import MAX_REACT_ITERATIONS
from src.agents.return_agent.prompts import build_system_prompt
from src.agents.return_agent.tools import (
    check_return_eligibility,
    create_return_request,
    get_return_status,
    lookup_return_policy,
)


class ReturnAgent:
    """LangGraph ReAct return/refund support agent."""

    def __init__(
        self,
        session_id: str = "default",
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.debug = debug

        self._tools = [
            lookup_return_policy,
            check_return_eligibility,
            create_return_request,
            get_return_status,
        ]
        self._system_prompt = build_system_prompt()
        self._checkpointer = MemorySaver()
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

    def chat(self, user_message: str) -> str:
        """Process a user message and return the Return Agent response."""
        ok, error = validate_user_input(user_message)
        if not ok:
            return error

        config: RunnableConfig = {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": self.max_iterations * 2,
        }

        if self.debug:
            final_content = "I wasn't able to generate a response. Please try again."
            for event in self._agent.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="values",
            ):
                messages = event.get("messages", [])
                if messages:
                    final_content = str(messages[-1].content)
            valid, guardrail_msg = validate_agent_output(final_content)
            return final_content if valid else guardrail_msg

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )

        messages = result.get("messages", [])
        if not messages:
            return "I wasn't able to generate a response. Please try again."

        final = str(messages[-1].content)
        valid, guardrail_msg = validate_agent_output(final)
        return final if valid else guardrail_msg

    def reset_memory(self) -> None:
        """Clear conversation history for this session."""
        self._checkpointer = MemorySaver()
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )
